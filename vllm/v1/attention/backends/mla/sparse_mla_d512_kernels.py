# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rope-free (``qk_rope_head_dim == 0``) indexed sparse-MLA Triton kernels.

Vendored from the ``jasl/vllm`` SM12x DeepSeek-V4 branch (PR #41834) —
the ``_indexed_d512_*`` family validated at GLM-5.3-Flash shapes
(``kv_lora_rank=512``, ``qk_rope_head_dim=0``, ``index_topk`` up to 4096
chunked) on SM120 by shing100 and the SM120 RTX-PRO-6000 issue #53963
participants (rel ~1e-3 vs fp32 reference). These are the SM120 no-rope
sparse-MLA attention kernels used by the ``D512_SM120`` backend.
"""

import torch

from vllm.triton_utils import tl, triton
@triton.jit
def _indexed_d512_split_score_kernel(
    q_ptr,
    kv_flat_ptr,
    indices_ptr,
    lens_ptr,
    scores_ptr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_kv_t,
    stride_kv_d: tl.constexpr,
    stride_indices_t: tl.constexpr,
    stride_indices_c: tl.constexpr,
    stride_scores_t: tl.constexpr,
    stride_scores_h: tl.constexpr,
    stride_scores_c: tl.constexpr,
    num_heads: tl.constexpr,
    num_candidates: tl.constexpr,
    scale: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    candidate_block = tl.program_id(2)
    head_offsets = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    candidate_offsets = candidate_block * BLOCK_C + tl.arange(0, BLOCK_C)
    dim_offsets = tl.arange(0, HEAD_DIM)
    head_mask = head_offsets < num_heads
    valid_len = tl.load(lens_ptr + token_idx)
    if candidate_block * BLOCK_C >= tl.minimum(valid_len, num_candidates):
        return
    candidate_mask = candidate_offsets < tl.minimum(valid_len, num_candidates)

    q = tl.load(
        q_ptr
        + token_idx * stride_q_t
        + head_offsets[:, None] * stride_q_h
        + dim_offsets[None, :] * stride_q_d,
        mask=head_mask[:, None],
        other=0.0,
    )
    kv_indices = tl.load(
        indices_ptr
        + token_idx * stride_indices_t
        + candidate_offsets * stride_indices_c,
        mask=candidate_mask,
        other=-1,
    )
    valid_kv = kv_indices >= 0
    kv = tl.load(
        kv_flat_ptr
        + kv_indices[None, :].to(tl.int64) * stride_kv_t
        + dim_offsets[:, None] * stride_kv_d,
        mask=valid_kv[None, :],
        other=0.0,
    )
    scores = tl.dot(q, kv) * scale
    tl.store(
        scores_ptr
        + token_idx * stride_scores_t
        + head_offsets[:, None] * stride_scores_h
        + candidate_offsets[None, :] * stride_scores_c,
        scores,
        mask=head_mask[:, None] & candidate_mask[None, :],
    )


@triton.jit
def _indexed_d512_split_stats_kernel(
    scores_ptr,
    lens_ptr,
    max_score_ptr,
    denom_ptr,
    stride_scores_t: tl.constexpr,
    stride_scores_h: tl.constexpr,
    stride_scores_c: tl.constexpr,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    num_candidates: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    candidate_offsets = tl.arange(0, BLOCK_C)
    valid_len = tl.load(lens_ptr + token_idx)
    candidate_mask = candidate_offsets < tl.minimum(valid_len, num_candidates)
    scores = tl.load(
        scores_ptr
        + token_idx * stride_scores_t
        + head_idx * stride_scores_h
        + candidate_offsets * stride_scores_c,
        mask=candidate_mask,
        other=-float("inf"),
    ).to(tl.float32)
    running_max = tl.max(scores, axis=0)
    safe_max = tl.where(valid_len > 0, running_max, 0.0)
    weights = tl.where(candidate_mask, tl.exp(scores - safe_max), 0.0)
    running_denom = tl.sum(weights, axis=0)

    tl.store(
        max_score_ptr + token_idx * stride_state_t + head_idx * stride_state_h,
        running_max,
    )
    tl.store(
        denom_ptr + token_idx * stride_state_t + head_idx * stride_state_h,
        running_denom,
    )


@triton.jit
def _indexed_d512_split_value_kernel(
    scores_ptr,
    kv_flat_ptr,
    indices_ptr,
    lens_ptr,
    max_score_ptr,
    acc_ptr,
    stride_scores_t: tl.constexpr,
    stride_scores_h: tl.constexpr,
    stride_scores_c: tl.constexpr,
    stride_kv_t,
    stride_kv_d: tl.constexpr,
    stride_indices_t: tl.constexpr,
    stride_indices_c: tl.constexpr,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_acc_t: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_acc_d: tl.constexpr,
    num_heads: tl.constexpr,
    num_candidates: tl.constexpr,
    head_dim: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    dim_block = tl.program_id(2)
    head_offsets = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    candidate_offsets = tl.arange(0, BLOCK_C)
    dim_offsets = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim
    valid_len = tl.load(lens_ptr + token_idx)
    max_score = tl.load(
        max_score_ptr + token_idx * stride_state_t + head_offsets * stride_state_h,
        mask=head_mask,
        other=0.0,
    ).to(tl.float32)
    safe_max = tl.where(valid_len > 0, max_score, 0.0)
    acc = tl.zeros((HEAD_BLOCK, BLOCK_D), tl.float32)

    for candidate_start in range(0, num_candidates, BLOCK_C):
        if candidate_start < tl.minimum(valid_len, num_candidates):
            candidates = candidate_start + candidate_offsets
            candidate_mask = candidates < tl.minimum(valid_len, num_candidates)
            kv_indices = tl.load(
                indices_ptr
                + token_idx * stride_indices_t
                + candidates * stride_indices_c,
                mask=candidate_mask,
                other=-1,
            )
            valid_kv = kv_indices >= 0
            scores = tl.load(
                scores_ptr
                + token_idx * stride_scores_t
                + head_offsets[:, None] * stride_scores_h
                + candidates[None, :] * stride_scores_c,
                mask=head_mask[:, None] & candidate_mask[None, :],
                other=-float("inf"),
            ).to(tl.float32)
            weights = tl.where(
                candidate_mask[None, :],
                tl.exp(scores - safe_max[:, None]),
                0.0,
            )
            values = tl.load(
                kv_flat_ptr
                + kv_indices[:, None].to(tl.int64) * stride_kv_t
                + dim_offsets[None, :] * stride_kv_d,
                mask=valid_kv[:, None] & dim_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(weights.to(tl.bfloat16), values)

    tl.store(
        acc_ptr
        + token_idx * stride_acc_t
        + head_offsets[:, None] * stride_acc_h
        + dim_offsets[None, :] * stride_acc_d,
        acc,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


def accumulate_indexed_d512_split_sparse_mla_attention(
    q: torch.Tensor,
    kv_flat: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    scale: float,
    scores: torch.Tensor,
    max_score: torch.Tensor,
    denom: torch.Tensor,
    acc: torch.Tensor,
    head_block_size: int = 32,
    candidate_block_size: int = 64,
    value_block_size: int = 128,
) -> None:
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]

    assert q.dim() == 3, f"Expected q shape [T, H, D], got {q.shape}"
    assert kv_flat.dim() == 2
    assert indices.dim() == 2
    assert indices.shape[0] == q.shape[0]
    assert lens.shape[0] == q.shape[0]
    assert kv_flat.shape[-1] == q.shape[-1]
    assert q.shape[-1] == 512
    assert scores.shape == (q.shape[0], max_score.shape[1], indices.shape[1])
    assert denom.shape == max_score.shape
    assert acc.shape == (*max_score.shape, q.shape[-1])
    assert max_score.dtype == torch.float32
    assert denom.dtype == torch.float32
    assert acc.dtype == torch.float32
    assert scores.dtype == torch.float32
    assert q.is_cuda and kv_flat.is_cuda and indices.is_cuda and lens.is_cuda
    assert scores.is_cuda and max_score.is_cuda and denom.is_cuda and acc.is_cuda
    assert head_block_size in (8, 16, 32)
    assert candidate_block_size in (32, 64, 128)
    assert value_block_size in (32, 64, 128)
    assert indices.shape[1] <= 1152

    num_tokens, _, head_dim = q.shape
    num_heads = max_score.shape[1]
    num_candidates = indices.shape[1]
    score_grid = (
        num_tokens,
        triton.cdiv(num_heads, head_block_size),
        triton.cdiv(num_candidates, candidate_block_size),
    )
    _indexed_d512_split_score_kernel[score_grid](
        q,
        kv_flat,
        indices,
        lens,
        scores,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_flat.stride(0),
        kv_flat.stride(1),
        indices.stride(0),
        indices.stride(1),
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        num_heads,
        num_candidates,
        scale,
        HEAD_BLOCK=head_block_size,
        BLOCK_C=candidate_block_size,
        HEAD_DIM=head_dim,
        num_warps=8,
        num_stages=3,
    )

    stats_grid = (num_tokens, num_heads)
    stats_block_c = next_power_of_2(num_candidates)
    _indexed_d512_split_stats_kernel[stats_grid](
        scores,
        lens,
        max_score,
        denom,
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        max_score.stride(0),
        max_score.stride(1),
        num_candidates,
        BLOCK_C=stats_block_c,
        num_warps=4,
        num_stages=3,
    )

    value_grid = (
        num_tokens,
        triton.cdiv(num_heads, head_block_size),
        triton.cdiv(head_dim, value_block_size),
    )
    _indexed_d512_split_value_kernel[value_grid](
        scores,
        kv_flat,
        indices,
        lens,
        max_score,
        acc,
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        kv_flat.stride(0),
        kv_flat.stride(1),
        indices.stride(0),
        indices.stride(1),
        max_score.stride(0),
        max_score.stride(1),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        num_heads,
        num_candidates,
        head_dim,
        HEAD_BLOCK=head_block_size,
        BLOCK_C=candidate_block_size,
        BLOCK_D=value_block_size,
        num_warps=4,
        num_stages=3,
    )


@triton.jit
def _indexed_d512_chunked_merge_acc_kernel(
    max_score_ptr,
    acc_ptr,
    chunk_max_score_ptr,
    chunk_acc_ptr,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_acc_t: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_acc_d: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    dim_block = tl.program_id(2)
    head_offsets = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    dim_offsets = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim

    running_max = tl.load(
        max_score_ptr + token_idx * stride_state_t + head_offsets * stride_state_h,
        mask=head_mask,
        other=-float("inf"),
    ).to(tl.float32)
    chunk_max = tl.load(
        chunk_max_score_ptr
        + token_idx * stride_state_t
        + head_offsets * stride_state_h,
        mask=head_mask,
        other=-float("inf"),
    ).to(tl.float32)
    next_max = tl.maximum(running_max, chunk_max)
    running_valid = running_max != -float("inf")
    chunk_valid = chunk_max != -float("inf")
    running_scale = tl.where(running_valid, tl.exp(running_max - next_max), 0.0)
    chunk_scale = tl.where(chunk_valid, tl.exp(chunk_max - next_max), 0.0)

    running_acc = tl.load(
        acc_ptr
        + token_idx * stride_acc_t
        + head_offsets[:, None] * stride_acc_h
        + dim_offsets[None, :] * stride_acc_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    chunk_acc = tl.load(
        chunk_acc_ptr
        + token_idx * stride_acc_t
        + head_offsets[:, None] * stride_acc_h
        + dim_offsets[None, :] * stride_acc_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    merged_acc = running_acc * running_scale[:, None] + chunk_acc * chunk_scale[:, None]
    tl.store(
        acc_ptr
        + token_idx * stride_acc_t
        + head_offsets[:, None] * stride_acc_h
        + dim_offsets[None, :] * stride_acc_d,
        merged_acc,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


@triton.jit
def _indexed_d512_chunked_merge_state_kernel(
    max_score_ptr,
    denom_ptr,
    chunk_max_score_ptr,
    chunk_denom_ptr,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    num_heads: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    head_mask = head_idx < num_heads

    running_max = tl.load(
        max_score_ptr + token_idx * stride_state_t + head_idx * stride_state_h,
        mask=head_mask,
        other=-float("inf"),
    ).to(tl.float32)
    running_denom = tl.load(
        denom_ptr + token_idx * stride_state_t + head_idx * stride_state_h,
        mask=head_mask,
        other=0.0,
    ).to(tl.float32)
    chunk_max = tl.load(
        chunk_max_score_ptr + token_idx * stride_state_t + head_idx * stride_state_h,
        mask=head_mask,
        other=-float("inf"),
    ).to(tl.float32)
    chunk_denom = tl.load(
        chunk_denom_ptr + token_idx * stride_state_t + head_idx * stride_state_h,
        mask=head_mask,
        other=0.0,
    ).to(tl.float32)
    next_max = tl.maximum(running_max, chunk_max)
    running_valid = running_max != -float("inf")
    chunk_valid = chunk_max != -float("inf")
    running_scale = tl.where(running_valid, tl.exp(running_max - next_max), 0.0)
    chunk_scale = tl.where(chunk_valid, tl.exp(chunk_max - next_max), 0.0)
    next_denom = running_denom * running_scale + chunk_denom * chunk_scale

    tl.store(
        max_score_ptr + token_idx * stride_state_t + head_idx * stride_state_h,
        next_max,
        mask=head_mask,
    )
    tl.store(
        denom_ptr + token_idx * stride_state_t + head_idx * stride_state_h,
        next_denom,
        mask=head_mask,
    )


def accumulate_indexed_d512_chunked_sparse_mla_attention(
    q: torch.Tensor,
    kv_flat: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    scale: float,
    scores: torch.Tensor,
    max_score: torch.Tensor,
    denom: torch.Tensor,
    acc: torch.Tensor,
    chunk_max_score: torch.Tensor,
    chunk_denom: torch.Tensor,
    chunk_acc: torch.Tensor,
    head_block_size: int = 32,
    candidate_block_size: int = 64,
    value_block_size: int = 128,
) -> None:
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]

    assert q.dim() == 3, f"Expected q shape [T, H, D], got {q.shape}"
    assert kv_flat.dim() == 2
    assert indices.dim() == 2
    assert indices.shape[0] == q.shape[0]
    assert lens.shape[0] == q.shape[0]
    assert kv_flat.shape[-1] == q.shape[-1]
    assert q.shape[-1] == 512
    assert scores.shape[0] == q.shape[0]
    assert scores.shape[1] == max_score.shape[1]
    assert 0 < scores.shape[2] <= 1152
    assert max_score.shape == denom.shape == chunk_max_score.shape == chunk_denom.shape
    assert acc.shape == chunk_acc.shape == (*max_score.shape, q.shape[-1])
    assert max_score.dtype == torch.float32
    assert denom.dtype == torch.float32
    assert acc.dtype == torch.float32
    assert chunk_max_score.dtype == torch.float32
    assert chunk_denom.dtype == torch.float32
    assert chunk_acc.dtype == torch.float32
    assert scores.dtype == torch.float32
    assert q.is_cuda and kv_flat.is_cuda and indices.is_cuda and lens.is_cuda
    assert scores.is_cuda and max_score.is_cuda and denom.is_cuda and acc.is_cuda
    assert chunk_max_score.is_cuda and chunk_denom.is_cuda and chunk_acc.is_cuda
    assert head_block_size in (8, 16, 32)
    assert candidate_block_size in (32, 64, 128)
    assert value_block_size in (32, 64, 128)

    num_tokens, _, head_dim = q.shape
    num_heads = max_score.shape[1]
    chunk_size = scores.shape[2]
    max_score.fill_(float("-inf"))
    denom.zero_()
    acc.zero_()

    merge_acc_grid = (
        num_tokens,
        triton.cdiv(num_heads, head_block_size),
        triton.cdiv(head_dim, value_block_size),
    )
    merge_state_grid = (num_tokens, num_heads)
    for candidate_start in range(0, indices.shape[1], chunk_size):
        candidate_end = min(candidate_start + chunk_size, indices.shape[1])
        chunk_candidates = candidate_end - candidate_start
        chunk_lens = torch.clamp(
            lens - candidate_start,
            min=0,
            max=chunk_candidates,
        )
        accumulate_indexed_d512_split_sparse_mla_attention(
            q=q,
            kv_flat=kv_flat,
            indices=indices[:, candidate_start:candidate_end],
            lens=chunk_lens,
            scale=scale,
            scores=scores[:, :, :chunk_candidates],
            max_score=chunk_max_score,
            denom=chunk_denom,
            acc=chunk_acc,
            head_block_size=head_block_size,
            candidate_block_size=candidate_block_size,
            value_block_size=value_block_size,
        )
        _indexed_d512_chunked_merge_acc_kernel[merge_acc_grid](
            max_score,
            acc,
            chunk_max_score,
            chunk_acc,
            max_score.stride(0),
            max_score.stride(1),
            acc.stride(0),
            acc.stride(1),
            acc.stride(2),
            num_heads,
            head_dim,
            HEAD_BLOCK=head_block_size,
            BLOCK_D=value_block_size,
            num_warps=4,
            num_stages=3,
        )
        _indexed_d512_chunked_merge_state_kernel[merge_state_grid](
            max_score,
            denom,
            chunk_max_score,
            chunk_denom,
            max_score.stride(0),
            max_score.stride(1),
            num_heads,
            num_warps=4,
            num_stages=3,
        )


