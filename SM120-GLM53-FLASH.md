# GLM-5.3-Flash on SM120 (ocnr "0.29" branch)

Serve `zai-org/GLM-5.3-Flash` (320B/18B-active multimodal MoE, hybrid KDA
linear-attention + rope-free sparse MLA, mHC, MTP) on the ocnr SM120 box
(4x RTX PRO 6000 Blackwell Max-Q, SM120, 96 GB each).

**This branch now ports the hardware-verified SM120 path** from
`chriswritescode-dev/glm-5.3-flash-sm120` (itself an overlay on the stock
`vllm/vllm-openai:glm53-flash` image, validated end-to-end on 4x RTX PRO 6000:
needle-in-haystack passes at up to 527k prompt tokens, engine init ~185 s). The
format differs from that overlay because this is a source build, but the
semantics are identical.

## Why fp8 + FLASHINFER_MLA_SPARSE_SM120 (not the d512/bfloat16 lane)

- On SM120 the sparse-MLA decoder kernel only takes the packed **`fp8_ds_mla`**
  KV layout; **bf16 KV has no kernel on this arch** (#53963). So `--kv-cache-dtype
  fp8` is mandatory.
- The checkpoint is NoPE (`qk_rope_head_dim=0`, 512-wide query) but every SM120
  kernel is the 576-wide GLM_NSA/DSv3.2 geometry (or DSv4 512, which carries
  448/64 and topk<=1024). We keep `FLASHINFER_MLA_SPARSE_SM120` and **zero-pad
  the latent 512 -> 576** (exact: a zero RoPE adds nothing to QK; the value comes
  from the 512 NoPE region; ~656 B/token DSA KV instead of ~528).
- The `d512` Triton lane on this branch (vendored from jasl/vllm #41834) is a
  bf16-capable fallback only, reachable via
  `--attention-config '{"backend_per_kind":{"mla_attention":"D512_SM120"}}'`
  (or `--kv-cache-dtype bf16`, which auto-routes to it). It is **not the default**
  and is end-to-end unproven; prefer fp8.

## What the branch changes (vs glm-release base)

`git log --oneline 0.29 ^glm5/glm-release`

| Commit | What |
|---|---|
| FlashInfer pinned 0.6.17 | max version on https://flashinfer.ai/whl (the recipe's ">=0.6.18" is not published); same as the reference image |
| masked_mha_available=False | #54057: SM120 startup AttributeError |
| kpool persistent slot mapping + positions | ZJY0516/vllm#7: fp8_fp4_mqa_logits ILLEGAL_ADDRESS past topk + CUDA-graph fault |
| **NoPE on FLASHINFER_MLA_SPARSE_SM120** | the overlay's four changes: rope zero-pad (576 GLM_NSA); `return_valid_counts` + `seq_lens=topk_lengths` + empty-row handling; `buffer_width = topk_tokens` (kernel template is exactly 2048) in glm5next model/mtp; drop the lowest-ranked pool (`select_k-1`) in the kpool indexer |
| **SM120 page-alignment fix** | `_get_indexer_block_alignment` restricts PAGED_MQA_PAGE_SIZES to 64 on major 12 -> block_size 1792, kpool storage 448 (=7x64), satisfying DeepGEMM `block_kv==64`; SM120 kernel block sizes pinned `[64]` |
| D512_SM120 backend (vendored d512 kernels) | experimental bf16 fallback; cuda.py prefers FlashInfer for fp8, d512 only at the tail/explicit |
| SM120 build config | VERSION 0.29.0+sm120.cu130, GitLab CI, Dockerfile (arch 12.0, cuda-nvrtc-dev, py-spy) |
| ocnr runtime backports | FlashInfer autotune TP sync, KV-offload barrier, C128A JIT fix |

## Build

```bash
git switch 0.29
git push origin 0.29   # GitLab CI builds infra/vllm:0.29.0-sm120-cu130
```

## Launch (verified config; deviating at your own risk)

```bash
vllm serve /models/zai-org/GLM-5.3-Flash \
  --served-model-name glm-5.3-flash \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --max-num-seqs 10 \
  --max-model-len 524288 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
```
Env: `HF_HUB_OFFLINE=1`, `NCCL_P2P_LEVEL=NODE`, `VLLM_ENGINE_READY_TIMEOUT_S=3600`,
plus JIT caps (`RAYON_NUM_THREADS`/`OMP_NUM_THREADS`, `MAX_JOBS`).

Notes (from the verified overlay / serve.sh):
- `--block-size` is NOT set: SM120 alignment computes the 1792-token manager
  block (so the kpool storage block 448 tiles by 64). Forcing 128 reproduces
  mode 4 (`fp8_fp4_paged_mqa_logits` assert).
- `max-num-seqs 10` pairs with 5 MTP tokens (10 x 6 = 60 <= 64 decode-batch
  ceiling for FlashInfer's split-K decode kernel). For full 1M context use
  `num_speculative_tokens: 1` or 0 (KV is ~8.7 KiB/token).
- `gpu-memory-utilization 0.95`; 0.93 failed to start in the overlay's testing.
- Verified numbers: 609,172 tokens KV (1.16x concurrency at full context),
  MTP acceptance 29-82%.

## Boot-verify checklist (still required for THIS build)

- [ ] `Engine: ready`; confirm the log selects FLASHINFER_MLA_SPARSE_SM120
      (N.B. the module-name report may show `TRITON_MLA` first in the list but
      FlashInfer wins for fp8+sparse; the fp8_ds_mla / GLM_NSA kernel is the 
      one actually used).
- [ ] No reuse of the old `pe_dim must be 64 for fp8_ds_mla` / DeepGEMM
      `block_kv==64` asserts on first decode (overlay's modes 1 & 4 are patched).
- [ ] Greedy temp-0: two identical prompts -> identical, sane output.
- [ ] Prefix-cache hit on a repeated prefix.
- [ ] Needle-style long-context retrieval at ≥100k prompt tokens (the overlay
      passes at 527k; do at least one 100k+ run).
- [ ] Vision + tool-call smoke (multimodal processor; `glm47` parser).
- [ ] MTP smoke (acceptance printed in logs; expect ~2.5-5 avg with 5 tokens).

If it jams: `dump-jam-state.sh` is in the image; capture before touching anything.

## Caveats learned upstream (don't chase these as branch bugs)

- **Checkpoint choice matters** (#54150): ModelOpt NVFP4 conversions emit
  invalid UTF-8 tokens; `RedHatAI/GLM-5.3-Flash-NVFP4` (compressed-tensors) is
  clean. NVFP4 MoE on SM120 only served correctly by `marlin` (flashinfer_trtllm
  / cutedsl reject the device; flashinfer_cutlass / cutlass / emulation collapse
  to single-token loops). The staged native-FP8 checkpoint uses the fp8 path
  above, so these matter only if you switch to NVFP4.
- Upstream integration is still landing (PR #53906 open, unmerged); re-base and
  drop the vendored pieces once a native family-120 NoPE lane exists.

## Runtime-overlay fallback (no rebuild)

For the fastest boot path against the stock image, the `chriswritescode-dev/
glm-5.3-flash-sm120` overlay (single-layer Dockerfile + serve.sh, image
`cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1`) is the proven
shortcut — this branch implements the same changes in source so you can build
your own SM120 image instead of trusting a third-party one.
