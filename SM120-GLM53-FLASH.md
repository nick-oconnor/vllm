# GLM-5.3-Flash on SM120 (ocnr "0.30" branch)

Serve `zai-org/GLM-5.3-Flash` (320B/18B-active multimodal MoE, hybrid KDA
linear-attention + rope-free sparse MLA, mHC, MTP) on the ocnr SM120 box
(4x RTX PRO 6000 Blackwell Max-Q, SM120, 96 GB each).

**Base (2026-09-18 re-cut as the `0.30` branch): upstream `vllm-project/vllm`
main, past the v0.30.0rc1 fork.** GLM-5.3-Flash model support merged upstream
as #53906 (2026-09-03, after the v0.29.0 release branch forked, so the
v0.29.0 release tag does NOT contain it). The previous ocnr 0.29 branch
carried the ZJY0516/vllm `glm-release` fork for model support; that fork
is retired — upstream main now carries the model plus follow-ups (#55119 EPLB,
#55214 packaging). The branch is upstream main + the ocnr commits listed below.

## Why fp8 + FLASHINFER_MLA_SPARSE_SM120 (not the d512/bfloat16 lane)

- On SM120 the sparse-MLA decoder kernel only takes the packed **`fp8_ds_mla`**
  KV layout; **bf16 KV has no kernel on this arch** (#53963). So `--kv-cache-dtype
  fp8` is mandatory.
- The checkpoint is NoPE (`qk_rope_head_dim=0`, 512-wide query) but every SM120
  kernel is the 576-wide GLM_NSA/DSv3.2 geometry (or DSv4 512, which carries
  448/64 and topk<=1024). We keep `FLASHINFER_MLA_SPARSE_SM120` and **zero-pad
  the latent 512 -> 576** (exact: a zero RoPE adds nothing to QK; the value comes
  from the 512 NoPE region; ~656 B/token DSA KV instead of ~528).
- The experimental `D512_SM120` bf16 fallback lane (vendored d512 kernels from
  jasl/vllm #41834) was **dropped** in this rebuild: it was never end-to-end
  verified, is not reachable from the production config (fp8 is mandatory on
  SM120), and the upstream PR is still open — re-vendor it from
  `backup/pre-029-upstream-rebase` if a bf16 lane is ever needed.

## What the branch changes (vs upstream main)

`git log --oneline upstream/main..0.30`

| Commit | What |
|---|---|
| FlashInfer autotune TP sync | `VLLM_FLASHINFER_AUTOTUNE_PROCESS_GROUP=1`: all-reduce measured autotune timings across the TP group so every rank picks the same tactic (diverging picks surfaced as illegal-kernel-op on the next collective) |
| b12x PCIe oneshot allreduce | `VLLM_ENABLE_PCIE_ALLREDUCE=1`: the only custom-AR path supporting TP>2 on PCIe-only topologies (4x RTX PRO 6000 have no NVLink); installs `b12x==1.3.0` |
| SM120 kernel block size [64] | the SM120 GLM_NSA/DSv3.2 kernels are instantiated at PAGE_BLOCK_SIZE=64 only |
| NoPE backend priority | for NoPE+sparse models on major==12, FLASHINFER_MLA_SPARSE_SM120 is tried before TRITON_MLA (upstream's default order is TRITON_MLA first; rope-64 DeepSeek-shaped models keep the default) |
| **NoPE on FLASHINFER_MLA_SPARSE_SM120** | zero-pad q and k_pe into the 576 geometry (`do_kv_cache_update` + `forward_mqa`); `return_valid_counts` + `seq_lens=valid counts` + empty-row handling; kpool indexer drops the lowest-ranked pool on family-120 and glm5next model/mtp pin the buffer width to `index_topk` (the fp8_ds_mla trtllm-gen kernel is instantiated for exactly 2048). The SM100 native-nope lane (nope_mla_dimensions, generic topk) is untouched |
| SM120 build config | VERSION 0.30.0+sm120.cu130, GitLab CI, Dockerfile (arch 12.0, DeepEP 12.0a + NCCL LIBRARY_PATH fix, MAX_JOBS=32/NVCC_THREADS=1, g++ + cuda-nvrtc-dev for runtime JIT, py-spy + dump-jam-state.sh) |

Notes on pieces deliberately NOT carried over from the old branch:

- **kpool tail-slot persistence fix**: upstream via #53906 (verified identical
  code on main; the fork's PR #7 was the same fix).
- **SM120 page-alignment fix** (`_get_indexer_block_alignment` family-120 →
  pool page 64): already upstream, equivalent implementation.
- **C128A topk metadata JIT-recompile fix** (do_not_specialize + warmup in
  `deepseek_v4/sparse_mla.py`): DSv4-only — glm5next does not touch
  `deepseek_v4/`, and this box serves GLM-5.3-Flash exclusively. Upstream's
  #53574 (merged in v0.29.0) fixed the SM120 C128A decode-view contiguity
  half. Restore from the backup branch if DSv4 returns to this box.
- **FlashInfer 0.6.17 pin**: obsolete — 0.6.18.post1 is now on the
  flashinfer.ai cu130 index (upstream main pins it); main's pin is used.

## Build

```bash
git switch 0.30
git push origin 0.30   # GitLab CI builds infra/vllm:0.30.0-sm120-cu130 (build is gated to the default branch)
```

## Launch (production config, verified 2026-09-11; deviating at your own risk)

```bash
vllm serve /models/zai-org/GLM-5.3-Flash \
  --served-model-name GLM-5.3-Flash \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-model-len auto \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.97 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --kv-offloading-size 100 \
  --kv-offloading-backend native \
  --enable-chunked-prefill \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --default-chat-template-kwargs '{"thinking": true}'
```

Env: `HF_HUB_OFFLINE=1`, `NCCL_P2P_LEVEL=NODE`, `RAYON_NUM_THREADS=4`,
`OMP_NUM_THREADS=4`, `MAX_JOBS=32`, `VLLM_FLASHINFER_AUTOTUNE_PROCESS_GROUP=1`
(autotune stays enabled; the env syncs tactic choice across TP ranks),
`VLLM_ENABLE_PCIE_ALLREDUCE=1` (b12x oneshot all-reduce).

Notes:
- `--block-size` is NOT set: SM120 alignment computes the 1792-token manager
  block (so the kpool storage block 448 tiles by 64). Forcing 128 reproduces
  mode 4 (`fp8_fp4_paged_mqa_logits` assert).
- `gpu-memory-utilization 0.97`: auto-fit holds the full 1M context with the
  vision stack resident (7.68 GiB/GPU KV, 1,064,361 tokens, 1.02x
  concurrency, 2026-09-20 boot on the upstream-main re-cut; 7.91 GiB /
  1,095,931 tokens on the 09-09/09-11/09-17 pre-0.30 boots). The overlay-era
  floor was 0.95; 0.93 failed to start there.
- KV offloading IS enabled in production (`--kv-offloading-backend native`,
  `--kv-offloading-size 100`): upstream's native `CPUOffloadingSpec` mmaps a
  100 GiB pool in `/dev/shm` (`vllm_offload_*.mmap`) — the pod's dshm must
  be **120Gi**. Upstream scopes offload configs to prefix-cacheable groups
  itself (`get_offloading_group_ids` → `prefix_cacheable_group_ids`), so
  the kpool-tail scratch group is excluded and the old #54743 carry is not
  needed on this branch. **0.30 context regression — root-caused and fixed
  (2026-09-20):** the 09-18 boot profiled only **3.81 GiB** for KV and
  auto-fit cut max_model_len to **516,096**. The cause was *not* the
  CUDAGraph reserve: the indexer prefill gather workspace
  (`Indexer.max_total_seq_len = get_max_prefill_buffer_size()`) is sized in
  **tokens** while this indexer's KV is **pool-granular**
  (`compress_ratio == index_kpool == 4`), so it reserved
  `max_model_len * 40 * 132 B` = **5.16 GiB/GPU** instead of 1.29 GiB.
  Upstream PR **#55222** (issue #55221) divides by `index_kpool` at that call
  site, exactly as `deepseek_v4/attention.py` already did; it is carried on
  this branch. Measured effect: consumed memory 82.27 → **78.40 GiB/GPU**,
  available KV 3.81 → **7.68 GiB**, `Auto-fit max_model_len: full model
  context length 1048576 fits`. Verified by A/B: booting the *unfixed* image
  at `--max-model-len 262144` (which scales the same workspace by 1/4)
  reproduced consumed 78.4 GiB / KV 7.77 GiB. Ignore the
  `CUDA graph pool memory: 0.07 GiB (actual), 4.19 GiB (estimated)` line —
  the estimator's profiling capture is the first forward with a KV cache, so
  it charges the one-time persistent workspace allocations to CUDA graphs;
  the reserve is real memory and **must not** be disabled with
  `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` (that double-spends it and
  OOMs). The 2026-09-10 corruption was NOT caused by offloading — root cause
  #55600 (state-seed units on any prefix-cache hit), fixed by #55601
  (carried here); see the sm120-enablement incident note.
- MTP variant (overlay-era, not in production): `--max-num-seqs 10` pairs with
  5 MTP tokens (10 x 6 = 60 <= 64 decode-batch ceiling for FlashInfer's
  split-K decode kernel); for full 1M context use
  `num_speculative_tokens: 1` or 0 (KV is ~8.7 KiB/token); MTP acceptance
  measured 29-82% there.
- c=1 serving profile (2026-09-11 bench): 89.0-89.7 tok/s decode on the
  2K/8K/32K cells, 82.1 at 128K, P50 ITL 10.3-10.5 ms.

## Silent KV-cache poisoning: the kpool tail seed kernel (fixed 2026-09-20)

`get_kv_cache_config_from_groups` aliases each layer's **tail** tensor onto
its **indexer** tensor at the same offset, so the tail view carries the
indexer's block stride. Measured on this box (probe print at first prefill):
`shape=(N, 2, 4, 128) stride=(38016, 512, 128, 1)` — `stride(0)` is 38016
elements, not the dense `2 * kpool * head_dim = 1024`.

The NVIDIA `_kpool_tail_seed_kernel` addressed blocks densely
(`base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM`). For every tail block
`blk > 0` that means the seed write lands **inside an unrelated indexer
block** (`blk * 1024` falls in indexer block `blk // 37`), overwriting pooled
indexer keys with raw K / gate-score values, while the request's own tail
block is never seeded and is read back as stale bytes. It fires on every
prefill, is silent (writes stay inside the shared allocation, so no OOB and
no crash), and the damage is persistent and prefix-cached — which is why the
symptom is progressive degeneration over a long session that only a restart
clears. The AMD kernel never had the bug.

Fixed upstream by **#57477** (`TAIL_BLOCK_ELEMS` / `KPOOL_HEAD` from
`tail.stride(0)` / `tail.stride(1)`), in this branch's base since the
2026-09-20 re-cut. Regression test:
`tests/kernels/test_kpool_decode_update_batched.py::test_prefill_seed_honors_padded_tail_block_stride`
(runs on every platform) — it FAILS on the 09-18 image and PASSES here.

## Boot-verify checklist (REQUIRED after this rebuild)

The 0.30 re-cut (2026-09-20, base `4868312128`) keeps the flashinfer
0.6.18.post1 pin (unchanged in requirements/cuda.txt since the 09-09
rebuild) and the same SM120 semantics, but upstream moved underneath: the
`masked_mha_available` declare was dropped (upstream computes it in
`SparseMLACommonImpl.__init__` since #48770; #54057 closed 2026-09-18),
#57102 now shares the token-to-request mapping across KV cache groups in
`build_attn_metadata`, and #57546 retired the local `_use_cooperative_topk`
gate in favour of the shared `SparseIndexerTopk` dispatcher (which already
carries the SM12x exclusion, so the ocnr override was dropped in the
rebase). Treat the first boot as unverified:

- [ ] `Engine: ready`; confirm the log selects FLASHINFER_MLA_SPARSE_SM120
      (N.B. the module-name report may show `TRITON_MLA` first in the list but
      FlashInfer wins for fp8+sparse; the fp8_ds_mla / GLM_NSA kernel is the
      one actually used).
- [ ] No `pe_dim must be 64 for fp8_ds_mla` / DeepGEMM `block_kv==64` asserts
      on first decode, and no shape/stride assert from the trtllm-gen kernel
      (flashinfer 0.6.18.post1 changed kernels vs the verified 0.6.17 — if the
      SM120 fp8_ds_mla path broke, pin `flashinfer-python==0.6.17` +
      `flashinfer-jit-cache==0.6.17` in requirements/cuda.txt and
      docker/versions.json as the old branch did).
- [ ] Greedy temp-0 sanity: use a **wide-margin prompt** (e.g. an exact-copy
      needle or arithmetic Q) and compare across modes (eager vs CUDA graphs vs
      MTP) — NOT two separate runs: MoE+TP4 temp-0 is not bit-reproducible
      across runs on open-ended prompts (soft argmax ties resolved by TP
      reduction noise; #53963 determinism note). Wide-margin prompts are
      bit-identical across modes.
- [ ] Run with CUDA graphs + torch.compile (the production regime). Eager-only
      numbers understate decode ~5-6x on SM120 (#53963 tmttodd), so don't judge
      throughput from an eager boot.
- [ ] Prefix-cache hit on a repeated prefix (exercises the kpool-tail group
      path).
- [ ] Needle-style long-context retrieval at ≥100k prompt tokens (the overlay
      passes at 527k; do at least one 100k+ run).
- [ ] Vision + tool-call smoke (multimodal processor; `glm47` parser).
- [ ] MTP smoke (acceptance printed in logs; expect ~2.5-5 avg with 5 tokens).

If it jams: `dump-jam-state.sh` is in the image; capture before touching anything.

## Caveats learned upstream (don't chase these as branch bugs)

- **Checkpoint choice matters** (#54150): ModelOpt NVFP4 conversions emit
  invalid UTF-8 tokens; `RedHatAI/GLM-5.3-Flash-NVFP4` (compressed-tensors) is
  clean. NVFP4 MoE on SM120 only served correctly by `marlin` and only with it
  passed explicitly (`--moe-backend marlin`; `auto`/others spin or collapse to
  single-token loops - #53963/@53906 field reports). Marlin repack can OOM at
  TP=2 (fixed by host-staged repack; relevant only if you go NVFP4 at <TP4).
  The staged native-FP8 checkpoint uses the fp8 path above, so these matter
  only if you switch to NVFP4.
- **Stay at TP4 for this box.** A TP=2 report on the same overlay hung in the
  MHC prenorm GEMM (both DeepGEMM and TileLang) at small warmup batches /
  32-heads-per-rank geometry - every verified config here is TP4 (16
  heads/rank), which doesn't hit it (#53963).
- FlashInfer's natively rope-free SM120 path would replace the 512->576
  zero-pad. **Decision (2026-08-28, re-checked 2026-09-18): still don't
  switch.** The kernel side is now ready — flashinfer #4802 (refactor) and
  #5075 (runtime KV row stride + canonical 528B GLM53_NOPE rows + NaN-safe
  masked gathers) are merged, and a field report on RTX PRO 6000 confirms
  GLM-5.3-Flash serving with #5075 + a temporary NoPE KV-cache writer
  (#53963, 2026-09-18). The blocker is vLLM-side writer support: #55277
  (native `pe_dim=0` in `concat_and_cache`) is open but conflicted since
  09-12, and #53969's width check rejects every faithful checkpoint — all
  released conversions carry `index_topk=2048` -> effective 2176, only
  `--hf-overrides '{"text_config":{"index_topk":2044}}'` gets past it
  (#53969 omid-a, 2026-09-18). Revisit when #55277 lands; expect ~24% DSA-KV
  capacity win (528 vs 656 B/token).
- Upstream merged b12x **MoE/GEMM kernels** for SM12x (`--kernel-backend
  b12x` / `flashinfer_b12x`, FP4 + FP8 linear/MoE) — name-share with our
  PCIe allreduce patch, but the upstream kernels are self-contained (no
  b12x pip import), so the `--no-deps` install is unaffected. Relevant only
  if you revisit NVFP4 (currently marlin-only per above).
- Upstream main moves fast; rebase this branch per release and re-run the
  boot-verify checklist each time.
