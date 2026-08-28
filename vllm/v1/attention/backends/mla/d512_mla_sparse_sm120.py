# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 rope-free (NoPE) sparse-MLA backend driven by the ``d512`` Triton
kernels.

GLM-5.3-Flash (``glm5_next``) uses rope-free sparse MLA (``kv_lora_rank=512``,
``qk_rope_head_dim=0``), i.e. query head dim 512 with nothing to rotate. On
family-120 (SM120 / SM121) the FlashInfer lanes only carry a DeepSeek-shaped
(rope-64 / query-576) sparse-MLA kernel, so a rope-free 512-wide query has no
path there (``vllm-project/vllm``#53963, #53969, #54031). The ``d512`` Triton
kernel family -- vendored from the ``jasl/vllm`` SM12x branch (PR #41834) --
is a plain indexed inner product over a 512-wide latent and was validated at
GLM-5.3-Flash shapes on RTX PRO 6000 (SM120) with rel ~1e-3 against an fp32
reference (kernel-level correctness only; full-model boot-verification is
tracked in the sm120-enablement notes).

Wiring follows the API facts measured on ``vllm/vllm-openai:glm53-flash`` in
#53963: GLM reaches attention through the generic MLA wrapper
(``MultiHeadLatentAttentionWrapper``), so this backend only needs an
``AttentionBackend`` + metadata reuse + a thin ``MLAAttentionImpl``; no model
file, no C++, no image rebuild. ``forward_mqa`` receives ``q = (q_nope
[T,H,512], q_rope [T,H,0])``; ``--block-size 128`` is mandatory (the GLM kpool
indexer asserts ``block_size % (index_kpool * 32) == 0``); and the kpool can
widen the top-k buffer past ``index_topk`` (always-selected tail, padded to a
multiple of 128), so the effective width -- not the raw ``index_topk`` -- is
validated here and passed through to the kernels.
"""

from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionLayer,
    AttentionType,
    MLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    FlashInferMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.mla.sparse_mla_d512_kernels import (
    accumulate_indexed_d512_chunked_sparse_mla_attention,
    accumulate_indexed_d512_split_sparse_mla_attention,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

# Candidates per chunk processed by the d512 chunked kernels (0 < C <= 1152).
_CHUNK = 1024
_HEAD_BLOCK = 32
_CANDIDATE_BLOCK = 64
_VALUE_BLOCK = 128
# The chunked path was validated up to this many candidates.
_MAX_CANDIDATES = 4096


def _effective_topk_width(index_topk: int, index_kpool: int) -> int:
    """Top-k BUFFER width: index_topk + kpool tail, padded to a 128 multiple."""
    eff = index_topk + (index_kpool - 1 if index_kpool > 1 else 0)
    return ((eff + 127) // 128) * 128


class D512MLASparseBackend(AttentionBackend):
    """SM120 NoPE sparse-MLA backend using the indexed ``d512`` Triton lane."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "bf16"]

    @staticmethod
    def get_name() -> str:
        return "D512_SM120"

    @classmethod
    def get_impl_cls(cls) -> type[MLAAttentionImpl]:
        return D512MLASparseImpl

    @classmethod
    def get_builder_cls(cls) -> type[FlashInferMLASparseMetadataBuilder]:
        return FlashInferMLASparseMetadataBuilder

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [512]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # The GLM kpool indexer requires block_size % (index_kpool * 32) == 0;
        # index_kpool=4 -> 128 (see #53963: --block-size 128 is mandatory).
        return [128]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 12

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: str | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if not use_mla or not use_sparse:
            return "sparse MLA only"
        if head_size != 512:
            return "512-wide NoPE query heads only"
        if dtype != torch.bfloat16:
            return "dtype not supported (bf16 required)"
        if kv_cache_dtype not in (None, "auto", "bf16"):
            return "kv_cache_dtype not supported"
        if block_size is not None and block_size != 128:
            return "requires --block-size 128"
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is None:
            return None
        hf = vllm_config.model_config.hf_text_config
        if getattr(hf, "qk_rope_head_dim", None) != 0:
            return "rope-free (NoPE) MLA only"
        index_topk = getattr(hf, "index_topk", None)
        if index_topk is None:
            return "requires a model with an index_topk config"
        kpool = getattr(hf, "index_kpool", 1) or 1
        if _effective_topk_width(int(index_topk), int(kpool)) > _MAX_CANDIDATES:
            return (
                "D512_SM120 supports effective topk buffer widths up to "
                f"{_MAX_CANDIDATES}; got "
                f"{_effective_topk_width(int(index_topk), int(kpool))}"
            )
        return None


class D512MLASparseImpl(MLAAttentionImpl[FlashInferMLASparseMetadata]):
    """Thin SM120 NoPE decode impl driving the ``d512`` Triton kernels."""

    is_sparse = True
    supports_dense_mha_prefill = False
    masked_mha_available = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "D512_SM120 does not support alibi_slopes / sliding_window / "
                "logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "D512_SM120 only supports decoder self-attention"
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim: int = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]
        if self.qk_rope_head_dim != 0:
            raise NotImplementedError(
                "D512_SM120 only serves rope-free MLA (qk_rope_head_dim == 0); "
                f"got {self.qk_rope_head_dim}"
            )

        # Skip-topk layers share the indexer's buffer via mla_args (cf.
        # FLASHMLA_SPARSE / FLASHINFER_MLA_SPARSE_SM120).
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer
            if indexer is not None
            else mla_args.get("topk_indices_buffer")
        )
        assert self.topk_indices_buffer is not None
        self.supports_quant_query_input = False
        # Rope-pad width of the fixed DS-shaped cache tile for the KV write.
        self._nope_rope_pad = 64

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        # Rope-free model: k_pe is [T, H, 0]. Pad to the DS-shaped tile width
        # so the shared concat_and_cache_mla write path accepts the shape.
        # A zero rope vector contributes 0 to q_pe.k_pe, so this is exact.
        if k_pe.size(-1) == 0:
            k_pe = k_pe.new_zeros((*k_pe.shape[:-1], self._nope_rope_pad))
        super().do_kv_cache_update(
            kv_c_normed,
            k_pe,
            kv_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        num_tokens = q.shape[0]
        num_heads = q.shape[1]
        if q.shape[2] != 512:
            raise RuntimeError(
                "D512_SM120 expects a 512-wide NoPE query head; got "
                f"q.shape={tuple(q.shape)}"
            )

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_tokens]

        # Compact valid top-k entries to a contiguous prefix and return the
        # per-token valid count. Load-bearing for the d512 kernels: the
        # one-shot entry point asserts indices.shape[1] <= 1152, and -1 padding
        # must never fall inside the scored window.
        indices, valid_counts = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_tokens],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )
        num_candidates = indices.shape[1]

        # Flat row view of the compressed MLA cache: [num_slots, latent+pe].
        # Rope-free => latent == head size, so kv_flat is the paged cache
        # viewed as rows of width kv_lora_rank (see #53963 overlay notes).
        latent_dim = self.kv_lora_rank + self.qk_rope_head_dim
        kv_flat = kv_c_and_k_pe_cache.view(-1, latent_dim)

        max_score = torch.empty(
            (num_tokens, num_heads), device=q.device, dtype=torch.float32
        )
        denom = torch.zeros(
            (num_tokens, num_heads), device=q.device, dtype=torch.float32
        )
        acc = torch.zeros(
            (num_tokens, num_heads, self.kv_lora_rank),
            device=q.device,
            dtype=torch.float32,
        )
        chunk_max_score = torch.empty_like(max_score)
        chunk_denom = torch.empty_like(denom)
        chunk_acc = torch.empty_like(acc)

        kwargs = dict(
            q=q,
            kv_flat=kv_flat,
            indices=indices,
            lens=valid_counts,
            scale=self.scale,
            scores=torch.empty(
                (num_tokens, num_heads, num_candidates),
                device=q.device,
                dtype=torch.float32,
            ),
            max_score=max_score,
            denom=denom,
            acc=acc,
            head_block_size=_HEAD_BLOCK,
            candidate_block_size=_CANDIDATE_BLOCK,
            value_block_size=_VALUE_BLOCK,
        )
        if num_candidates <= _CHUNK:
            accumulate_indexed_d512_split_sparse_mla_attention(**kwargs)
        else:
            kwargs["scores"] = torch.empty(
                (num_tokens, num_heads, _CHUNK),
                device=q.device,
                dtype=torch.float32,
            )
            accumulate_indexed_d512_chunked_sparse_mla_attention(
                chunk_max_score=chunk_max_score,
                chunk_denom=chunk_denom,
                chunk_acc=chunk_acc,
                **kwargs,
            )

        # Online-softmax merge: out = acc / denom (the kernels accumulate
        # exp(w) and exp(w)*v across chunks with rescaling; see the #53963
        # repro. The score kernel masks stores to candidate positions < the
        # per-token valid count, so the -1 tail never contributes -- the
        # return_valid_counts=True compaction above is load-bearing).
        denom_safe = denom.clamp(min=1e-9).unsqueeze(-1)
        # Match the other MLA lanes' output dtype (q.dtype, bf16).
        out = (acc / denom_safe).to(q.dtype)
        return out, None
