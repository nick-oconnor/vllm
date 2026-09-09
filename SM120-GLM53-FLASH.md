# GLM-5.3-Flash on SM120 (ocnr "0.29" branch)

Serve `zai-org/GLM-5.3-Flash` (320B/18B-active multimodal MoE, hybrid KDA
linear-attention + rope-free sparse MLA, mHC, MTP) on the ocnr SM120 box
(4x RTX PRO 6000 Blackwell Max-Q, SM120, 96 GB each).

**Base (2026-09-09 rebuild): upstream `vllm-project/vllm` main.** GLM-5.3-Flash
model support merged upstream as #53906 (2026-09-03, after the v0.29.0 release
branch forked, so the release tag does NOT contain it). The previous ocnr 0.29
branch carried the ZJY0516/vllm `glm-release` fork for model support; that fork
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

`git log --oneline upstream/main..0.29`

| Commit | What |
|---|---|
| FlashInfer autotune TP sync | `VLLM_FLASHINFER_AUTOTUNE_PROCESS_GROUP=1`: all-reduce measured autotune timings across the TP group so every rank picks the same tactic (diverging picks surfaced as illegal-kernel-op on the next collective) |
| KV-offload collective barrier | `VLLM_KV_OFFLOAD_COLLECTIVE_BARRIER=1`: after `start_load_kv`, wait for this rank's CPU->GPU loads then barrier the TP group (async loads landing at different times across ranks deadlocked NCCL). Distinct from upstream's merged #52596, which fixes the shm region setup race |
| KV-offload GPU-resident groups | groups outside the prefix-cache hash chain (GLM-5.3's kpool tail, block 4) are never offloaded — their block size cannot align to `tokens_per_hash`; they stay GPU-resident (upstream renamed the spec flag to `prefix_cacheable`) |
| b12x PCIe oneshot allreduce | `VLLM_ENABLE_PCIE_ALLREDUCE=1`: the only custom-AR path supporting TP>2 on PCIe-only topologies (4x RTX PRO 6000 have no NVLink); installs `b12x==1.3.0` |
| masked_mha_available=False | #54057 (still open): SM120 startup AttributeError in the prefill dispatcher |
| SM120 kernel block size [64] | the SM120 GLM_NSA/DSv3.2 kernels are instantiated at PAGE_BLOCK_SIZE=64 only |
| NoPE backend priority | for NoPE+sparse models on major==12, FLASHINFER_MLA_SPARSE_SM120 is tried before TRITON_MLA (upstream's default order is TRITON_MLA first; rope-64 DeepSeek-shaped models keep the default) |
| **NoPE on FLASHINFER_MLA_SPARSE_SM120** | zero-pad q and k_pe into the 576 geometry (`do_kv_cache_update` + `forward_mqa`); `return_valid_counts` + `seq_lens=valid counts` + empty-row handling; kpool indexer drops the lowest-ranked pool on family-120 and glm5next model/mtp pin the buffer width to `index_topk` (the fp8_ds_mla trtllm-gen kernel is instantiated for exactly 2048). The SM100 native-nope lane (nope_mla_dimensions, generic topk) is untouched |
| SM120 build config | VERSION 0.29.0+sm120.cu130, GitLab CI, Dockerfile (arch 12.0, DeepEP 12.0a + NCCL LIBRARY_PATH fix, MAX_JOBS=32/NVCC_THREADS=1, g++ + cuda-nvrtc-dev for runtime JIT, py-spy + dump-jam-state.sh) |

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

## Boot-verify checklist (REQUIRED after this rebuild)

The base moved from the 08-27 fork snapshot to upstream main (09-09) and
flashinfer moves 0.6.17 -> 0.6.18.post1, so treat the first boot as
unverified even though the SM120 semantics ported 1:1:

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
- [ ] Prefix-cache hit on a repeated prefix (exercises the GPU-resident
      kpool-tail group path).
- [ ] Needle-style long-context retrieval at ≥100k prompt tokens (the overlay
      passes at 527k; do at least one 100k+ run).
- [ ] Vision + tool-call smoke (multimodal processor; `glm47` parser).
- [ ] MTP smoke (acceptance printed in logs; expect ~2.5-5 avg with 5 tokens).
- [ ] KV offload smoke with `VLLM_KV_OFFLOAD_COLLECTIVE_BARRIER=1`
      (`--kv-offloading-size 100 --kv-offloading-backend native`), since the
      offloading connector/scheduler code drifted upstream.

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
- FlashInfer's natively rope-free SM120 path (flashinfer PR #4802 / #4791,
  D_CKV=512 / D_ROPE=0) would replace the 512->576 zero-pad. **Decision
  (2026-08-28, re-checked 2026-09-09): still don't switch.** It fixes mode 1
  only (cleaner + ~24% DSA-KV capacity) and leaves mode 4, the indexer
  changes and the SM120 build untouched; #53969 (vLLM-side NoPE support) is
  still open and its validate formula rejects the real checkpoint config
  (2048 + 3 kpool tail -> 2176 != 2048). Revisit when #53969 lands with a
  formula that matches `zai-org/GLM-5.3-Flash` AND a released flashinfer
  ships the rope-free SM120 kernel that vLLM's SM100 lane uses.
- Upstream main moves fast; rebase this branch per release and re-run the
  boot-verify checklist each time.
