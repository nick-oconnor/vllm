# GLM-5.3-Flash on SM120 (ocnr "0.29" branch)

Serve `zai-org/GLM-5.3-Flash` (320B/18B-active multimodal MoE, hybrid KDA
linear-attention + rope-free sparse MLA, mHC, MTP) on the ocnr SM120 box
(4x RTX PRO 6000 Blackwell Max-Q, SM120, 96 GB each).

## What this branch adds (over the `glm-release` base)

The branch is based on the GLM-5.3-Flash enablement line
(`zai-org`-adjacent fork `ZJY0516/vllm` `glm-release`, i.e. PR #53906), which
ships the `glm5next` model (NVIDIA), the sparse kpool indexer with
`index_kpool_compress`, MTP, and the multimodal processor. On top of that base
this branch adds the ocnr/SM120 deltas below. See `git log --oneline 0.29 ^glm5/glm-release`.

| Commit | What | Why |
|---|---|---|
| FlashInfer **pinned at 0.6.17** | requirements/cuda.txt, docker/Dockerfile, versions.json | 0.6.17 is the latest resolvable version on https://flashinfer.ai/whl/ (the recipe's ">= 0.6.18" is not yet published there). Matches the `glm53-flash` reference image and the #53963 SM120 test env; the d512 lane is pure Triton and the fp8 no-rope lane works on 0.6.17 |
| `masked_mha_available=False` on SM120 impl | #54057 backport | SM120 startup-profiling AttributeError |
| persistent kpool slot mapping + `positions` | ZJY0516/vllm#7 backport | fp8_fp4_mqa_logits CUDA_ERROR_ILLEGAL_ADDRESS past topk; CUDA-graph kpool fault |
| NoPE on FLASHINFER_MLA_SPARSE_SM120 | #53969 (surgical) | zero-pad rope write/query (bit-exact); effective-topk-width check |
| **D512_SM120 backend** | vendored d512 Triton kernels (jasl/vllm #41834) + backend/impl | the only family-120 lane for a 512-wide NoPE query; serves the staged checkpoint (effective topk 2176 > 2048) |
| SM120 build config | VERSION / .gitlab-ci.yml / Dockerfile / jam diag (ported from ocnr 0.28) | builds `registry.ocnr.org/infra/vllm:0.29.0-sm120-cu130` |
| FlashInfer autotune TP sync, kv-offload barrier, C128A JIT fix | ocnr 0.28 backports | keeps DeepSeek-V4 stable; GLM sparse-MLA warmup parity |

## Build

```bash
# from infra/vllm (submodule), pushes to GitLab which builds the image
git switch 0.29
git push origin 0.29        # GitLab CI builds infra/vllm:0.29.0-sm120-cu130
```
The Dockerfile builds SM120-only (`torch_cuda_arch_list='12.0'`), CUDA 13.0.x,
FlashInfer 0.6.18, `cuda-nvrtc-dev` + `g++` in the runtime image for JIT.

## Launch (first boot-verify)

Weights are staged at `/models/zai-org/GLM-5.3-Flash` (native FP8, 308 GB).
Capacity: FP8 weights ~328 GB on 384 GB VRAM is marginal, so the first boot
uses **bf16 KV** (the D512_SM120 lane; fp8_ds_mla cannot serve the staged
effective topk width of 2176 anyway) and short context. MTP is left off for
the first boot. See `deploy/vllm-glm53.yaml` (cutover-ready).

```bash
vllm serve /models/zai-org/GLM-5.3-Flash \
  --served-model-name glm-5.3-flash \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --kv-cache-dtype bf16 \
  --block-size 128 \
  --max-model-len 131072 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --no-disable-hybrid-kv-cache-manager \
  --attention-config '{"backend_per_kind":{"mla_attention":"D512_SM120"}}'
```
Required env: `HF_HUB_OFFLINE=1`, `NCCL_P2P_LEVEL=NODE`, `VLLM_SSM_CONV_STATE_LAYOUT=DS`,
`VLLM_KV_CACHE_LAYOUT=HND`, plus JIT caps (`RAYON_NUM_THREADS=4`, `MAX_JOBS=32`).

## Boot-verify checklist (outstanding; we cannot run CUDA here)

- [ ] sparse-MLA init + `Engine: ready` (expect the D512_SM120 lane selected; watch
      for the old `pe_dim must be 64 for fp8_ds_mla` == the FlashInfer lane leaking in)
- [ ] bf16 rope-free KV write through `concat_and_cache_mla` (the #53963 open item)
- [ ] greedy temp-0 single turn: identical prompt twice -> identical, sane output
- [ ] `prompt_tokens`/prefill sanity, prefix-cache hit on a repeated prompt (#53906 APC report)
- [ ] eager vs CUDA-graph output equality (graph needs PR #7 fix on the branch)
- [ ] vision + tool-call smoke (multimodal processor; tool parser `glm47`)
- [ ] decode throughput (Triton d512 lane is slower than the SM100 TRTLLM lane; expected)

Log post-mortem if it jams: `dump-jam-state.sh` (py-spy built into the image).

## Runtime-overlay fallback (no rebuild)

The same D512 wiring was proven on the pinned `vllm/vllm-openai:glm53-flash`
image (digest `sha256:2e771fa...`) as a `PYTHONPATH` overlay registering
`AttentionBackendEnum.CUSTOM` from a `sitecustomize` (see #53963, tmttodd's
wiring report). If the source build is not ready yet, that overlay is the
fastest boot path against the stock image; the backend here is the in-tree
equivalent.

## Production path (later)

- **NVFP4**: the staged FP8 won't leave room for KV/concurrency at 1M context.
  Quantize to NVFP4 (M3-NVFP4 pattern) and switch to `--kv-cache-dtype fp8`
  (flashinfer_cutlass MoE); revisit MTP + `--kv-offloading-size` then.
- Watch upstream: the GLM-5.3-Flash integration (PR #53906) and the SM120 NoPE
  kernel work are still landing; re-base when they merge and drop the vendored
  d512 kernels / backends once a native family-120 NoPE lane exists.
