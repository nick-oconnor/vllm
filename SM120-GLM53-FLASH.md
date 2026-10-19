# GLM-5.3-Flash on SM120 (ocnr "0.32" branch)

Serve `zai-org/GLM-5.3-Flash` (320B/18B-active multimodal MoE, hybrid KDA
linear-attention + rope-free sparse MLA, mHC, MTP) on the ocnr SM120 box
(4x RTX PRO 6000 Blackwell Max-Q, SM120, 96 GB each).

**Base: the upstream `v0.32.0` release tag (`5df63a7203`, 2026-10-17) —
not main.** Earlier branches tracked upstream main past the release fork;
0.31 and 0.32 are cut on the tag itself. Note `v0.32.0` is *not* an ancestor
of upstream main: the release branch carries its own cherry-picks, so
presence of a fix must be checked by PR number, not by commit SHA. Previous
cuts: 0.31 on `v0.31.0` (`db9527a468`, 2026-10-02); 0.30 re-cuts 2026-09-27
`924707f1bf`, 09-23 `9f07d023d0`, 09-22 `d90f0eade5`, 09-18/09-20
`4868312128`. GLM-5.3-Flash model support merged upstream as #53906
(2026-09-03, after the v0.29.0 release branch forked, so the v0.29.0 release
tag does NOT contain it). The previous ocnr 0.29 branch carried the
ZJY0516/vllm `glm-release` fork for model support; that fork is retired —
upstream now carries the model plus follow-ups (#55119 EPLB, #55214
packaging).

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
  SM120), and the upstream PR is still open — re-vendor it from the
  `jasl/ds4-sm120` fork (the #41834 source) if a bf16 lane is ever needed.

## What the branch changes (vs upstream v0.32.0)

`git log --oneline v0.32.0..0.32`

| Commit | What |
|---|---|
| ocnr scheduler aligned-split fix | `_mamba_block_aligned_split` derives cacheable positions and chunk ends from the resolved LCM alignment (`self.block_size`) instead of `cache_config.block_size` (the smallest group block); reported against #55600, no upstream PR yet |
| b12x PCIe oneshot allreduce | `VLLM_ENABLE_PCIE_ALLREDUCE=1`: the only custom-AR path supporting TP>2 on PCIe-only topologies (4x RTX PRO 6000 have no NVLink); installs `b12x==1.3.0` |
| NoPE backend priority | for NoPE+sparse models on major==12, FLASHINFER_MLA_SPARSE_SM120 is tried before TRITON_MLA (upstream's default order is TRITON_MLA first; rope-64 DeepSeek-shaped models keep the default) |
| **NoPE on FLASHINFER_MLA_SPARSE_SM120** | zero-pad q and k_pe into the 576 geometry (`do_kv_cache_update` + `forward_mqa`); `return_valid_counts` + `seq_lens=valid counts` + empty-row handling; kpool indexer drops the lowest-ranked pool on family-120 and glm5next model/mtp pin the buffer width to `index_topk` (the fp8_ds_mla trtllm-gen kernel is instantiated for exactly 2048). The SM100 native-nope lane (nope_mla_dimensions, generic topk) is untouched |
| SM120 leader-only warmup autotune dropped | the rank-0 sparse-MLA decode autotune + cache-file broadcast could not equalise the ranks (the in-process profiling cache is consulted before file configs, and FlashInfer keys MoE entries by tp/ep rank), deadlocking the synchronized pass — every rank now tunes in the synchronized pass instead, which also drops the `VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS` workaround |
| SM120 build config | VERSION 0.32.0+sm120.cu130, GitLab CI, Dockerfile (arch 12.0, DeepEP 12.0a + NCCL LIBRARY_PATH fix, MAX_JOBS/NVCC_THREADS sized for the CI host, g++ + cuda-nvrtc-dev for runtime JIT, py-spy + dump-jam-state.sh) |

### 2026-10-18 cut onto v0.32.0

**Four of the six carries retired into upstream.** `v0.32.0` contains, by
PR number (checked against the tag's own history, since release branches
cherry-pick under different SHAs):

| Carry | Status on `v0.32.0` |
|---|---|
| #55222 indexer prefill workspace | upstream `c37f86e572` — dropped |
| #55601 mamba state-index seed | upstream `3d6853b32c` — dropped |
| #56531 speculation-capable rows | upstream `0eac152707` — dropped |
| #57635 per-rank autotune cache | upstream `3403e0f176` — dropped, see below |
| #57477 kpool tail seed stride | upstream `db1bfdd4fb` — already uncarried |
| #59504 async KV-load zeroing | upstream `ca0df1f122` — **the 0.31 known gap is closed** |

Upstream's #57635 is a superset of the ocnr carry: it keeps the per-rank
`_dp{dp}_rank{r}` cache path but replaces the "every rank has a file" check
with `_all_ranks_have_matching_cache`, which all-gathers a fingerprint so
ranks also reject *mutually inconsistent* files, and rescopes the gather and
barrier from the world group to the tuning group for PP. Ours was dropped in
favour of it.

**Upstream now ships its own `flashinfer_mla_sparse_sm120.py`** (for
rope-carrying DSv3.2/GLM_NSA), so the ocnr NoPE port shrank from a new file
to a 150-line delta on upstream's. Two divergences were resolved in
upstream's favour:

- `get_supported_kernel_block_sizes` goes back to `MultipleOf(64)`. The ocnr
  hard `[64]` existed because the 256 page selected a kernel shape that does
  not exist on this arch; upstream now reinterprets the cache as 64-row pages
  explicitly (`kv_rows.view(-1, 64, ...)`), which generalises the same
  constraint.
- `flat_kv_row_view` / `BLOCK_STRIDE_ROWS` is taken: it addresses the cache in
  flat-row space so interleaved pages from other layers are not misaddressed.
  The ocnr `return_valid_counts` + `seq_lens` + empty-row handling layers on
  top of it (upstream's `triton_convert_req_index_to_global_index` still
  accepts `return_valid_counts`).

The two remaining one-liner-ish carries were re-checked by hand rather than
trusted to a clean apply:

- **scheduler aligned-split**: still applies, but upstream has moved toward it
  — `_mamba_block_aligned_split` now derives `ckpt_block_size` from
  `self.block_size`, and `core.py` narrows `cache_config.block_size` to the
  min over *prefix-cacheable* groups precisely to stop it desyncing from
  mamba. Whether the remaining `cache_config.block_size` at the mamba
  alignment site is still the wrong value is model-dependent (it differs from
  the LCM only with multiple KV groups); verified at boot, see below.
- **leader-only warmup autotune**: `_drop_leader_only_tuning` is absent
  upstream, and the leader-only path it removes is still present, so it is
  still required.

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
- **FlashInfer 0.6.17 pin**: obsolete — upstream main has moved on twice
  since (0.6.18.post1 at the 09-23 re-cut, **0.7.0 since #58069**, in this
  branch's base); main's pin is used.
- **FlashInfer autotune TP sync** (`VLLM_FLASHINFER_AUTOTUNE_PROCESS_GROUP`):
  dropped 2026-09-22 — upstream now syncs autotune tactics natively: the
  generic warmup sets the autotune process group to the world group and
  broadcasts the leader's cached results (`kernel_warmup.py`). The SM120
  sparse-MLA decode warmup originally ran leader-only autotune plus a
  cache-file broadcast; that leader path is dropped on this branch
  (2026-09-27) — broadcasting cannot equalise the ranks, so every rank
  re-tunes in the synchronized pass instead (see the commit table). The
  residual ocnr value was only guarding lazy runtime re-tuning of shapes the
  warmup did not cover; if a diverging-tactic illegal-kernel-op ever
  reappears mid-session, restore the commit from the pre-rebase 0.30 history.

## Build

```bash
git switch main
git push origin main   # GitLab CI builds infra/vllm:0.32.0-sm120-cu130 (the build is gated to the default branch: push main, not a snapshot branch)
```

## 0.32 boot-verify results (2026-10-18, image `vllm:0.32.0+sm120.cu130`)

Production config + `cudagraph_mode: FULL`, NVFP4 checkpoint, TP4+EP, MTP k=3,
on the CI-equivalent image (default last stage `vllm-openai-nonroot`, run as
`--user 2000:0`), at the deployed **gmu 0.644**:

| check | result |
|---|---|
| Engine ready, `FLASHINFER_MLA_SPARSE_SM120` selected | PASS |
| `pe_dim` / `block_kv==64` / trtllm-gen shape asserts | 0 occurrences |
| auto-fit | full 1,048,576 (8.73 GiB KV, 1,068,443 tokens, 1.02x) |
| greedy temp-0 (primes, spelling, 7919x311, capitals, reversal) | all correct |
| vision (split-colour image) | "Red, blue." PASS |
| tool call (`glm47`) | `get_weather(city=Reykjavik, unit=c)` PASS |
| prefix-cache cached/fresh/cached @ 122k | ok / ok / ok |
| needle @ **1,045,745 tokens** (2,831 short of the ceiling) | FOUND cold (135.1 s) and warm (2.9 s) |
| eviction: 5 x 502,902-token sessions, revisited after turnover | **10/10**, 2,499,840 external hit tokens, 0 preemptions |
| accumulated decode, 10 rounds | no looping (worst *consecutive* 6-gram repeat 1). `soak.py` scores `max(consecutive, total)` and false-positives on benign prompt echo -- judge consecutive only |
| MTP acceptance | 12812/21720 = 59.0% |
| restart with populated per-rank autotune cache | PASS -- all 4 ranks `0 new, 126 from previous`, no deadlock; init 274 s cold -> 115 s warm, confirming the `/home/vllm` cache PVC works as mounted |

The two `shm_broadcast` "No available shared memory broadcast block" notices
on the warm boot are emitted during JIT kernel warmup and clear on their own;
the #57635 deadlock signature is rank 0 finished with ranks 1-3 stuck at 0%
repeating that message.

**Build the default (last) stage, not `--target vllm-openai`.** CI passes no
`--target`, so buildkit builds the final stage `vllm-openai-nonroot`: user
`vllm` (UID 2000, GID 0), `HOME=/home/vllm`, entrypoint
`vllm-nonroot-entrypoint.sh`. Every cache dir (HF, Triton, Inductor, vLLM,
Numba, Outlines) defaults to `$HOME/.cache/...`, so the cache PVC mounted at
`/home/vllm` in `vllm.yaml` is correct. Building `--target vllm-openai`
gives the root variant (`HOME=/root`, bare `vllm serve` entrypoint), which
silently writes its caches somewhere the PVC is not mounted -- a local-test
trap, not a production bug.

## REQUIRED on 0.32: `cudagraph_mode: FULL`

**Without `--compilation-config '{"cudagraph_mode": "FULL"}'` the 0.32 build
produces token salad** (`locklocklock...`) on even a trivial prompt. This is
a 0.32 regression: the 0.31 image is correct with CUDA graphs on the same
host, model and mount, and both images ship an identical kernel stack
(flashinfer 0.7.0.post1, torch 2.13.0+cu130, triton 3.7.1, tilelang 0.1.12)
and an identical compilation config, so the cause is vLLM Python-side.

Bisected to the **PIECEWISE capture path**, not any decode kernel:

| configuration | result |
|---|---|
| `--enforce-eager` | correct |
| eager + MTP | correct |
| `cudagraph_mode: NONE` | correct |
| `cudagraph_mode: PIECEWISE` | **garbage** |
| `cudagraph_mode: FULL_AND_PIECEWISE` (upstream default) | **garbage** |
| `cudagraph_mode: FULL` | correct |

Because `CompilationMode` is `NONE` for this model (no torch.compile),
`splitting_ops` is empty, so "PIECEWISE" does not split around attention --
it captures the whole model for mixed prefill+decode batches, including the
sparse-MLA prefill path, which is not graph-safe. `FULL` captures only the
uniform decode batches and falls back to eager for mixed ones. 0.31 captured
piecewise graphs on the same settings harmlessly.

Ruled out by experiment, so do not re-test these: the 256-token block page
(`--block-size 64` still garbage); upstream's `flat_kv_row_view` /
`BLOCK_STRIDE_ROWS` flat-row addressing (it is *required* -- reverting to
0.31's direct addressing raises `SM120 sparse-MLA has no decode kernel ...
page_block_size=2304 ... instantiated only for page_block_size=64`);
`index_group.convert_logical_to_physical_topk` persistent buffers; the
`logical_topk_ready` event wait; b12x PCIe allreduce; MTP; torch.compile;
the ocnr `return_valid_counts` / empty-row zeroing; the ocnr scheduler
aligned-split carry; `cudagraph_decoder_replay`; the new indexer TP
row-sharding gate (`tp_prefill_row_sharding_supported`, whose
`mixed_mode() != FULL` condition looks like an exact fit but is a
coincidence -- forcing sharding off under FULL_AND_PIECEWISE is still
garbage); and swapping 0.31's KDA kernels / `fused_recurrent` /
`mla_attention` into the 0.32 image.

`FULL` is a verified workaround, not a root-cause fix: upstream should
either populate `splitting_ops` or refuse piecewise capture for this
backend. **It costs no measurable throughput.** Decode batches use FULL
graphs under both settings -- piecewise only covers mixed prefill+decode
batches, which are a small fraction of steps at `max-num-seqs 4` and fall
back to eager. Measured on this box, same benchmark, 18 runs each:

| | 0.31 FULL_AND_PIECEWISE | 0.32 FULL | delta |
|---|---|---|---|
| decode 200 tok | 145.3 tok/s (124-164) | 146.9 (140-165) | +1.1% |
| decode 400 tok | 156.0 tok/s (142-169) | 159.1 (146-168) | +2.0% |
| decode 800 tok | 170.2 tok/s (149-194) | 159.6 (152-170) | -6.2% |
| all decode | **156.0 tok/s** | **157.7 tok/s** | **+1.1%** |
| cold prefill 40k | 6,162 tok/s | 7,380-9,985 tok/s | faster |

The `decode_800` bucket is the only one favouring 0.31 and sits inside that
arm's own 149-194 spread. Do not quote cross-session throughput numbers as a
baseline; re-measure both images with one harness.

**It also fixes the memory regression, and buys a lower gmu.** Under
FULL_AND_PIECEWISE the profiler charges a "peak activation 6.13 GiB" term
and auto-fit cuts `max_model_len` to 732,672 at gmu 0.655. Under FULL that
term drops to ~2.5 GiB, so the full 1M fits on far less and the deploy moves
to **gmu 0.644** (8.75 GiB KV / 1,068,443 tokens, 1.02x), returning ~1.05
GiB/GPU to the co-resident image-generation workload.

| gmu | KV | tokens | full 1M? |
|---|---|---|---|
| 0.642 | 8.56 GiB | 1,045,923 | no (-4,864) |
| 0.643 | 8.65 GiB | 1,057,406 | yes, 0.8% margin (floor) |
| **0.644** | **8.75 GiB** | **1,068,443** | **yes, 1.02x (deployed)** |
| 0.655 | 9.79 GiB | 1,196,480 | yes, 14% spare |

KV scales ~11,600 tokens per 0.001 gmu and the accounting is deterministic
(two independent boots at 0.655 both gave exactly 1,196,480 tokens).

## Launch (production config — mirrors k8s-gitops stage3/apps/vllm.yaml)

```bash
vllm serve /models/nvidia/GLM-5.3-Flash-NVFP4 \
  --served-model-name GLM-5.3-Flash \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-model-len auto \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.644 \
  --compilation-config '{"cudagraph_mode": "FULL"}' \
  --moe-backend flashinfer_cutlass \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --kv-offloading-size 100 \
  --kv-offloading-backend native \
  --enable-chunked-prefill \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --limit-mm-per-prompt '{"image": 999, "video": 0}' \
  --default-chat-template-kwargs '{"thinking": true}'
```

Env: `VLLM_ENABLE_PCIE_ALLREDUCE=1`, `NCCL_P2P_LEVEL=NODE`, `HF_HUB_OFFLINE=1`,
`RAYON_NUM_THREADS=4`, `OMP_NUM_THREADS=4`.

The served checkpoint is NVIDIA's NVFP4 export. NVIDIA ships the MTP layer
(45) in bf16 but omits it from the quantisation ignore list, so the on-disk
`config.json` / `hf_quant_config.json` are patched to add **both**
`model.language_model.layers.45*` and `model.layers.45*` — the MTP head is
built as a text-only model, so `apply_vllm_mapper` does not rewrite the
second spelling. `--moe-backend marlin` is therefore unusable (it rejects
the unquantised bf16 MTP experts); `flashinfer_cutlass` is required.


Env: `HF_HUB_OFFLINE=1`, `NCCL_P2P_LEVEL=NODE`, `RAYON_NUM_THREADS=4`,
`OMP_NUM_THREADS=4`, `MAX_JOBS=32`, `VLLM_ENABLE_PCIE_ALLREDUCE=1` (b12x
oneshot all-reduce). FlashInfer autotune stays enabled; tactic consistency
across TP ranks is upstream-native now (the generic warmup broadcasts the
leader's cached results; the SM120 sparse-MLA decode warmup tunes on every
rank — the leader-only shortcut is dropped).

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
  site, exactly as `deepseek_v4/attention.py` already did; both halves of the
  PR are carried on this branch (the call-site fix plus the chunker budget
  sized in compressed rows, which otherwise admits up to `compress_ratio`x
  more rows than the workspace holds at ≥64 requests near max-model-len).
  Measured effect: consumed memory 82.27 → **78.40 GiB/GPU**,
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

The 0.32 cut keeps the same SM120 semantics but retires four carries into
upstream (#55222, #55601, #56531, #57635) and rebuilds the NoPE port on top
of upstream's own `flashinfer_mla_sparse_sm120.py`, so upstream code now runs
under every path this branch depends on — including the
FlashInfer **0.7.0** bump (#58069), GLM-5.3-Flash corruption fixes from
the #56868/#56605 hunt (#58454 kpool pool-selection corruption with
speculative decoding, #58368 prompt-tail prefix-cache hits with MTP),
the profiling-fragmentation fix (#58430), the GLM5.3 metadata-op
optimization (#58450), #58061 (dense MLP on the sequence-parallel shard),
#55442 (deferred disposable MTP head) and the #55277 merge. Treat the
first boot as unverified:

- [ ] `Engine: ready`; confirm the log selects FLASHINFER_MLA_SPARSE_SM120
      (N.B. the module-name report may show `TRITON_MLA` first in the list but
      FlashInfer wins for fp8+sparse; the fp8_ds_mla / GLM_NSA kernel is the
      one actually used).
- [ ] No `pe_dim must be 64 for fp8_ds_mla` / DeepGEMM `block_kv==64` asserts
      on first decode, and no shape/stride assert from the trtllm-gen kernel
      (flashinfer 0.7.0 changed kernels vs the verified 0.6.18.post1 — if the
      SM120 fp8_ds_mla path broke, pin `flashinfer-python==0.6.18.post1` +
      `flashinfer-jit-cache==0.6.18.post1` (or 0.6.17) in requirements/cuda.txt
      and docker/versions.json as the old branch did).
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
- [ ] Restart once and confirm the **second** boot completes the FlashInfer
      autotune warmup (upstream #57635 writes per-rank cache files under the
      persistent cache root; before it, a rank-0-only file deadlocked ranks
      1-3 at the tuning reduce — the symptom was rank 0 done, ranks 1-3 at 0%
      with repeated `shm_broadcast` "No available shared memory broadcast
      block").

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
  zero-pad. **Decision (2026-08-28, re-checked 2026-09-23): still don't
  switch.** The kernel side is ready — flashinfer #4802 (refactor) and
  #5075 (runtime KV row stride + canonical 528B GLM53_NOPE rows + NaN-safe
  masked gathers) are merged, and a field report on RTX PRO 6000 confirms
  GLM-5.3-Flash serving with #5075 + a temporary NoPE KV-cache writer
  (#53963, 2026-09-18). vLLM-side writer support landed: **#55277 merged
  2026-09-23** (native `pe_dim=0` rows in `fp8_ds_mla` `concat_and_cache`,
  in this branch's base since the 2026-09-23 re-cut). The remaining blocker
  is #53969's width check, which rejects every faithful checkpoint — all
  released conversions carry `index_topk=2048` -> effective 2176, only
  `--hf-overrides '{"text_config":{"index_topk":2044}}'` gets past it
  (#53969 omid-a, 2026-09-18). Revisit once that lifts; expect ~24% DSA-KV
  capacity win (528 vs 656 B/token).
- Upstream merged b12x **MoE/GEMM kernels** for SM12x (`--kernel-backend
  b12x` / `flashinfer_b12x`, FP4 + FP8 linear/MoE) — name-share with our
  PCIe allreduce patch, but the upstream kernels are self-contained (no
  b12x pip import), so the `--no-deps` install is unaffected. Relevant only
  if you revisit NVFP4 (currently marlin-only per above).
- Upstream main moves fast; rebase this branch per release and re-run the
  boot-verify checklist each time.
