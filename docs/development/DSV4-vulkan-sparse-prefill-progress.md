# DeepSeek V4 Vulkan sparse prefill progress

This file is a self-contained handoff for the DeepSeek V4 sparse-attention prompt-processing optimization on AMD Strix Halo. Read the repository `AGENTS.md` and `CONTRIBUTING.md` before continuing.

## Repository state

- Repository: `https://github.com/Nathanw1014/llama.cpp`
- Branch: `strix-halo-vulkan`
- Starting commit: `baf0025de861c6f6ea3720fa81c52ae1b2e6c078`
- Target GPU: AMD Radeon 8060S / gfx1151, RADV, Vulkan, wave64
- The device reports `GL_KHR_cooperative_matrix`, f16 inputs with f32 accumulation, 64 KiB shared memory, and a maximum 512-thread workgroup used by this path.

The implementation is not upstream `ggml-org/llama.cpp`. It builds on this branch's DeepSeek V4 graph, Lightning Indexer, sparse top-K hint, decode gather path, fused HC kernels, and Vulkan profiler changes.

## Important execution constraints

The system is an APU. CPU compilation and GPU benchmarking share power and memory bandwidth. Never build and benchmark at the same time. Serialize all builds, correctness tests, and performance tests.

GPU commands must run with host GPU access. In an agent sandbox, request elevated/out-of-sandbox execution. A sandboxed benchmark showed only CPU activity and is invalid.

Redirect the full model benchmark to a log. Inspect only the last profiler block with `tail`; do not load the full log into agent context.

## Build and ccache

The build directory is `build`, configured as Release with Vulkan enabled. The default ccache directory was read-only in the agent environment, so use a writable directory:

```bash
cmake -S . -B build \
  -DGGML_VULKAN=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

CCACHE_DIR=/tmp/llama-cpp-ccache cmake --build build --config Release \
  --target llama-bench test-backend-ops -j "$(nproc)"

CCACHE_DIR=/tmp/llama-cpp-ccache ccache -s
```

ccache was verified active. The final build reported direct hits. Note that an incremental change to `ggml-vulkan.cpp` is one large C++ translation unit and therefore uses one compiler core even with `-j`. Shader object regeneration can run in parallel.

## Canonical benchmark command

Do not change or omit switches for the 32k acceptance run:

```bash
GGML_VK_PERF_LOGGER=1 ./build/bin/llama-bench \
  -m ~/Projects/docker/localLLaMA/models/models--unsloth--DeepSeek-V4-Flash-0731-GGUF/snapshots/109848da2469efe1f1aab9e11acea08a065ccd4f/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf \
  -r 1 -d 32768 -p 2048 -ub 2048 -fa 1 -n 0 \
  > /tmp/dsv4-vulkan-32k.log 2>&1

tail -n 180 /tmp/dsv4-vulkan-32k.log
```

The known model path exists on the target system.

## Graph and dispatch findings

DeepSeek V4 builds the Lightning Indexer and sparse attention in `src/models/deepseek4.cpp`:

1. `build_lid_top_k()` creates indexer Q/K/weights and calls `ggml_lightning_indexer()`.
2. `ggml_top_k()` selects up to `hparams.indexer_top_k` compressed-cache indices for every query token.
3. `build_csa_lid_attention()` concatenates the raw SWA K prefix with compressed CSA K, builds a dense mask carrying the same sparse selection, and calls `build_attn_mha(..., top_k, raw_k->ne[2])`.
4. `build_attn_mha()` attaches `top_k` and `n_kv_raw` to `GGML_OP_FLASH_ATTN_EXT` through `ggml_flash_attn_ext_add_top_k()`.

At the final 32k PP2048 batch:

- total K rows: 11,008
- `n_kv_raw`: 2,304 raw SWA rows, always attended subject to the mask
- `n_top_k`: 512 selected compressed rows per query token
- active rows: 2,816
- selectable compressed region: 8,704

The custom Vulkan path is selected by `ggml_vk_flash_attn_top_k()` before ordinary FA. Its gate requires the DeepSeek V4 shape and `total_k >= 3 * (n_kv_raw + n_top_k)`. The final shape satisfies `11008 >= 3 * 2816`.

The old `flash_attn_top_k.comp` shader is scalar/subgroup code. One 512-thread workgroup covers eight heads for one query token. It stages 16 selected 512-wide K/V rows, computes QK with scalar FMAs and `subgroupAdd`, updates online softmax one key at a time, and accumulates PV manually. It does not use cooperative matrices.

The top-K set differs by query token but is shared by all 64 query heads for that token. This makes the attention for one token a regular matrix problem across heads and selected keys despite sparse per-token indexing.

## Root cause evidence

The existing Vulkan timestamp infrastructure was extended with `ggml_vk_perf_mark_subop()` after the sparse dispatch. This reports the sparse kernel separately as `FA_TOP_K_SPARSE (sub-op)` or `FA_TOP_K_CM (sub-op)`.

Focused test shape:

```bash
GGML_VK_PERF_LOGGER=1 ./build/bin/test-backend-ops perf \
  -b Vulkan0 -o FLASH_ATTN_EXT \
  -p 'kv=32768,nb=512,n_kv_raw=1024,n_top_k=512,sinks=0'
```

Results:

- old scalar sparse kernel: 61.42 ms, 1.68 TFLOPS of useful active-set work
- ordinary dense FA diagnostic (`GGML_VK_FA_TOPK=0`): 255.97 ms, about 8.7 TFLOPS over the full dense work
- final cooperative sparse kernel: 32.55 ms, 3.17 TFLOPS of useful active-set work

The residual `FLASH_ATTN_EXT` interval after the sparse timestamp is only about 4-7 us. The cost is inside the shader, not dispatch or surrounding synchronization.

A temporary uniform stage-profiling mode was used and removed. For the final 32-head tile:

- selected K gather + cooperative QK: 14.30 ms
- gather + QK + serial softmax: 26.34 ms
- gather + QK + parallel softmax: 15.53 ms
- full kernel: 32.55 ms
- the remaining cooperative PV/output portion is about 17.0 ms

The old scalar shader was compute/issue inefficient. Dense FA proved matrix hardware is much faster but was still too expensive because it processes all K rows. The final implementation preserves sparsity and uses the matrix hardware for both QK and PV.

## Implementation

Files changed:

- `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_top_k_cm.comp`
  - New cooperative-matrix sparse prefill shader.
  - One 512-thread workgroup covers 32 query heads for one token.
  - Eight wave64 subgroups cover two 16-head tiles by four 16-key or 16-output-dimension tiles.
  - Processes 64 selected keys per online-softmax block.
  - Stages only indexed selected K/V tiles, never the full K range.
  - Uses f16 cooperative-matrix inputs and f32 accumulation for QK and PV.
  - Uses a 16-lane segmented softmax per head. XOR subgroup shuffles reduce max and sum for four independent heads per wave without workgroup barriers.
  - Keeps f32 output accumulators and normalizes after all active blocks.
  - Preserves the raw prefix, top-K index validation, mask, sinks, stream strides, and K == V latent behavior.
- `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp`
  - Embeds the new shader when cooperative-matrix shader support is available.
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
  - Adds the cooperative sparse pipeline when the device supports the required 16x16x16 f16/f32 cooperative matrix shape.
  - Selects it by capability and keeps the scalar shader as fallback.
  - Adds `GGML_VK_FA_TOPK=0` to force ordinary dense FA for diagnostics.
  - Adds `GGML_VK_FA_TOPK_CM=0` to force the old scalar sparse shader for A/B tests.
  - Adds sparse sub-operation timestamps through the existing profiler.

The path is capability-based, not hardcoded to Strix Halo. The current sparse shape gate remains DeepSeek V4-specific. Devices without the required cooperative matrix support keep the correct scalar sparse or dense fallback.

Two discarded prototypes are useful context:

- A 16-head, 64-key streaming cooperative tile was correct and reduced the focused test from 61.4 to 44.6 ms.
- Keeping 32 complete 512-wide K/V rows in LDS grew shared memory to about 41 KiB, reduced residency, doubled block/barrier count, and regressed to 73.8 ms. Do not retry full-row LDS staging without solving occupancy.
- A 32-head, 64-key streaming tile halved irregular row loads but initially stayed near 44.7 ms because serial softmax cost about 11.5 ms. Parallel segmented softmax produced the final 32.55 ms result.

## Correctness validation

Run:

```bash
./build/bin/test-backend-ops test -b Vulkan0 -o FLASH_ATTN_EXT -p 'n_top_k='
```

Final result: 8/8 sparse top-K FA cases passed against the CPU reference. Cases include:

- decode and short batches that use dense/gather fallback
- sparse prefill batch sizes 64 and 128
- `n_kv_raw` plus top-K selection
- invalid top-K index handling from the test fixture
- sinks enabled and disabled
- sparse threshold transitions
- an active-key count of 193, which exercises a partial final 64-key block

The test uses the existing FA tolerance of NMSE <= `5e-4`. No NaN or Inf failure occurred. The implementation changes Q and probability inputs to f16 cooperative-matrix operands with f32 accumulation, matching the precision strategy of ordinary Vulkan cooperative FA.

Still desirable before broader submission:

- compare model logits on controlled prompts between `GGML_VK_FA_TOPK_CM=0` and the default cooperative path

## Canonical 32k results

Exact clean runs, same command and machine, no concurrent build:

```text
32k context, PP 2048, ub 2048

Before (commit baf0025de):
112.29 tok/s
Total Vulkan: 18.1957 s
Sparse FA: 8.84496 s, 421.189 ms/layer
Lightning Indexer: 1.19438 s
TOP_K: 0.076948 s

After:
152.32 tok/s
Total Vulkan: 13.4032 s
Sparse FA: 4.44701 s, 211.762 ms/layer
Lightning Indexer: 1.13156 s
TOP_K: 0.073945 s

Change:
Throughput: +35.65%
Total Vulkan time: -26.34%
Sparse FA time: -49.72%
Sparse FA saved: 4.398 s
Total GPU time saved: 4.793 s
```

The profiler now lists the optimized dispatch as `FA_TOP_K_CM (sub-op)`. The following residual `FLASH_ATTN_EXT` line is only the post-mark interval and must not be interpreted as the kernel time.

## Context-depth measurements

All points use PP2048, ub2048, FA enabled, one repetition, and no token generation. They were run sequentially with no compiler active:

```text
Existing depth    tok/s    Total Vulkan    Final large FA    Lightning Indexer    TOP_K
0                 253.44    8.041 s          0.294 s           0.095 s               0.001 s
8192              211.01    9.666 s          1.625 s           0.379 s               0.022 s
16384             177.19   11.518 s          3.031 s           0.678 s               0.044 s
32768             152.32   13.403 s          4.447 s           1.132 s               0.074 s
```

At 0, 8k, and 16k, total K is below the existing sparse-path gate `total_k >= 3 * (n_kv_raw + n_top_k)`. These points use the unchanged ordinary dense FA implementation, so the cooperative sparse change does not affect or regress them. At 32k, total K is 11,008 and the cooperative sparse path engages. The 32k `Final large FA` value is the `FA_TOP_K_CM (sub-op)` total; the lower-depth values are the large ordinary `FLASH_ATTN_EXT` totals.

Logs:

- `/tmp/dsv4-vulkan-cm-0k.log`
- `/tmp/dsv4-vulkan-cm-8k.log`
- `/tmp/dsv4-vulkan-cm-16k.log`
- `/tmp/dsv4-vulkan-cm-32k.log`
- `/tmp/dsv4-vulkan-baseline-clean.log`
- `/tmp/dsv4-fa-cm-correctness-final.log`

## Next optimization target

The cooperative sparse FA remains the largest context-dependent cost at about 4.45 s total. Stage profiling indicates approximately 14.3 ms of focused-test time in gather/QK, about 1.2 ms in parallel softmax, and about 17 ms in PV/output.

The next useful work is PV and output accumulation, not TOP_K. Investigate:

- reducing repeated selected V staging across the two 32-head workgroups per token without increasing LDS enough to lose occupancy
- reducing the eight output-dimension passes or retaining more PV state in cooperative fragments/registers
- checking register count and spills for the 32 f32 output accumulators per invocation using RADV shader statistics
- alternate 32-head layouts that keep the same eight-wave occupancy but improve PV scheduling
- query-tile overlap/union gathering only if measured top-K overlap is high enough; a whole-2048-query union is unlikely to help

Do not optimize TOP_K first. At 32k it is only about 74 ms total. Lightning Indexer is about 1.13 s and is the next context-dependent target only after sparse FA improves further.

## Useful diagnostics

Force old scalar sparse path:

```bash
GGML_VK_FA_TOPK_CM=0 GGML_VK_PERF_LOGGER=1 ./build/bin/test-backend-ops perf \
  -b Vulkan0 -o FLASH_ATTN_EXT \
  -p 'kv=32768,nb=512,n_kv_raw=1024,n_top_k=512,sinks=0'
```

Force ordinary dense FA:

```bash
GGML_VK_FA_TOPK=0 GGML_VK_PERF_LOGGER=1 ./build/bin/test-backend-ops perf \
  -b Vulkan0 -o FLASH_ATTN_EXT \
  -p 'kv=32768,nb=512,n_kv_raw=1024,n_top_k=512,sinks=0'
```

Default cooperative sparse path:

```bash
GGML_VK_PERF_LOGGER=1 ./build/bin/test-backend-ops perf \
  -b Vulkan0 -o FLASH_ATTN_EXT \
  -p 'kv=32768,nb=512,n_kv_raw=1024,n_top_k=512,sinks=0'
```

Always run these sequentially. Do not run a compiler concurrently on this APU.
