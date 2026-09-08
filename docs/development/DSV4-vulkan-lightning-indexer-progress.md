# DeepSeek V4 Vulkan Lightning Indexer progress

This is a restart note for the Strix Halo Lightning Indexer optimization. It is a development scratch pad and can be removed before the final PR.

## Repository state

- Main repository: `/home/jaap/Projects/git/llama.cpp`
- Optimization worktree: `/tmp/llama-strix-beta-bench`
- Branch: `strix-halo-vulkan-lightning-indexer`
- Base commit: `316c72ee9eab590f5891089d3b6bfc0d01d00d19`
- Base branch: Nathan's `strix-halo-vulkan-beta`
- Decode microbench work is stored in the main worktree as `stash@{0}: On strix-halo-vulkan: wip: DSV4 decode microbench depth matrix`.
- The Indexer changes are uncommitted. Do not commit without explicit user approval. An assisted commit needs an `Assisted-by:` trailer.
- Do not run builds and GPU benchmarks together. The APU shares its power and memory-bandwidth budget.
- GPU commands need sandbox escalation.

## Objective and result

After sparse prefill attention was flattened, the context-dependent Lightning Indexer became the next prefill bottleneck. The old cooperative-matrix shader used one wave64 subgroup per workgroup, processed one 16-key tile, and loaded one query head at a time.

The new wide pipeline uses eight wave64 subgroups per workgroup. Each subgroup processes a separate 16-key tile, so one workgroup covers 128 keys. It stages four query heads and their weights together, reuses them across all eight subgroups, and uses subgroup-scoped synchronization between cooperative-matrix result stores. A workgroup barrier remains between four-head groups because all subgroups reuse the shared query storage.

The optimized shader requires 512 workgroup invocations and 64 KiB shared memory. Pipeline creation is capability-based. Devices without those limits use a one-wave, one-head cooperative-matrix specialization. The scalar implementation remains the fallback when cooperative matrices are unavailable. The decode-specific cooperative-matrix pipeline is unchanged.

At the 32k-equivalent prefill microbench shape:

| Version | Time per layer | Throughput |
| --- | ---: | ---: |
| Baseline | 50.51 ms | 5.83 TFLOPS |
| Optimized | 31.81 ms | 9.25 TFLOPS |

This is a 37.0% reduction in Lightning Indexer kernel time.

The canonical 32k llama-bench improved from 209.45 to 216.32 tokens/s. Total Vulkan time fell from 9.73729 to 9.42448 seconds. Total Lightning Indexer time fell from 1.11860 to 0.697276 seconds. Sparse attention and top-K were effectively unchanged.

## Changed files

- `ggml/src/ggml-vulkan/vulkan-shaders/lightning_indexer_cm.comp`: parameterizes the shader, adds the eight-wave four-head implementation, and remains usable for the small fallback.
- `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp`: generates wide `N_WAVES=8`, `HEADS_PER_TILE=4` and small `N_WAVES=1`, `HEADS_PER_TILE=1` variants.
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`: creates and selects the capability-gated wide pipeline and the small cooperative-matrix fallback.
- `tests/test-backend-ops.cpp`: adds 127, 128, and 129-key correctness boundaries and PP2048 performance shapes through 512k simulated source context.

## Performance data

The performance rows model the actual PP2048 Indexer shapes after the source context is filled in 2048-token batches. `kv=8704` is the measured shape near 32k source context. It differs from 32768 because the Indexer compresses source tokens into rows.

| Source depth | `kv` | Baseline | Optimized | Reduction |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 512 | 4.23 ms | 2.14 ms | 49.5% |
| 8k | 2560 | 17.10 ms | 10.09 ms | 41.0% |
| 16k | 4608 | 29.97 ms | 17.62 ms | 41.2% |
| 32k | 8704 | 50.51 ms | 32.94 ms | 34.8% |
| 64k | 16896 | 94.36 ms | 64.87 ms | 31.3% |
| 128k | 33280 | 174.96 ms | 128.76 ms | 26.4% |
| 256k | 66048 | 352.73 ms | 250.88 ms | 28.9% |
| 512k | 131584 | 696.79 ms | 495.90 ms | 28.8% |

The final isolated 32k run after cleanup measured 31.81159 ms. Small matrix differences are normal laptop GPU clock variation.

| Canonical 32k metric | Baseline | Optimized |
| --- | ---: | ---: |
| PP2048 | 209.45 tokens/s | 216.32 tokens/s |
| Total Vulkan | 9.73729 s | 9.42448 s |
| Lightning Indexer | 1.11860 s | 0.697276 s |
| Sparse FA raw | 0.198438 s | 0.195367 s |
| Sparse FA selected | 0.835110 s | 0.843385 s |
| Sparse FA reduce | 0.088463 s | 0.086678 s |
| TOP_K | 0.075473 s | 0.075350 s |

Logs:

- Baseline matrix: `/tmp/dsv4-lightning-prefill-baseline.log`
- Optimized matrix: `/tmp/dsv4-lightning-four-head-matrix.log`
- Final selected-pipeline 32k microbench: `/tmp/dsv4-lightning-final-selected-32k.log`
- Baseline canonical 32k llama-bench: `/tmp/dsv4-nathan-beta-32k-rerun-new-first.log`
- Optimized canonical 32k llama-bench: `/tmp/dsv4-lightning-final-32k-llama-bench.log`

## Correctness and resources

- Wide pipeline: all 20 focused F16 cases passed, including 127, 128, and 129-key boundaries.
- Small cooperative-matrix fallback: temporarily forced and all the same 20 cases passed.
- Wide shader on gfx1151: 168 VGPRs, 63,488 bytes LDS, no spills, eight subgroups per SIMD.
- The final pipeline-statistics run confirmed `lightning_indexer_cm_f16` was selected.
- No NaN, Inf, or comparison failures were reported.

Correctness logs:

- Wide: `/tmp/dsv4-lightning-consolidated-wide-correctness.log`
- Small fallback: `/tmp/dsv4-lightning-consolidated-small-correctness.log`

## Experiments and decisions

- Four waves improved the 32k shape about 3% and became worse at deep simulated contexts.
- Eight waves improved it about 8% before the other changes.
- Subgroup-scoped synchronization after cooperative-matrix stores improved the eight-wave version.
- Staging four query heads and weights produced the large gain by reducing redundant loads and barriers.
- Using only a subgroup barrier between head groups failed four boundary tests. One subgroup could overwrite shared query data while another still read it. A workgroup barrier is required there.
- A separate fallback shader source was avoided. Generator definitions create both variants from one file.

## Commands

Build only, with no GPU benchmark running:

```sh
cd /tmp/llama-strix-beta-bench
git diff --check
cmake --build build --config Release --target test-backend-ops llama-bench -j "$(nproc)"
```

Focused correctness:

```sh
cd /tmp/llama-strix-beta-bench
./build/bin/test-backend-ops test -b Vulkan0 -o LIGHTNING_INDEXER -p 'type_K=f16' > /tmp/dsv4-lightning-correctness.log 2>&1
tail -n 30 /tmp/dsv4-lightning-correctness.log
```

Final 32k-equivalent microbench and pipeline selection:

```sh
cd /tmp/llama-strix-beta-bench
GGML_VK_PIPELINE_STATS=lightning_indexer_cm_f16 ./build/bin/test-backend-ops perf -b Vulkan0 -o LIGHTNING_INDEXER -p 'kv=8704' > /tmp/dsv4-lightning-final-selected-32k.log 2>&1
tail -n 16 /tmp/dsv4-lightning-final-selected-32k.log
```

Full Indexer depth matrix:

```sh
cd /tmp/llama-strix-beta-bench
./build/bin/test-backend-ops perf -b Vulkan0 -o LIGHTNING_INDEXER -p 'nb=2048,nh=64,ns=1,nm=1,type_K=f16' > /tmp/dsv4-lightning-matrix.log 2>&1
rg 'kv=(512|2560|4608|8704|16896|33280|66048|131584),nb=2048' /tmp/dsv4-lightning-matrix.log
```

Canonical 32k llama-bench. Run it only when needed, never while compiling, and inspect only the final block:

```sh
cd /tmp/llama-strix-beta-bench
GGML_VK_PERF_LOGGER=1 ./build/bin/llama-bench -m /home/jaap/Projects/docker/localLLaMA/models/models--unsloth--DeepSeek-V4-Flash-0731-GGUF/snapshots/109848da2469efe1f1aab9e11acea08a065ccd4f/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf -r 1 -d 32768 -p 2048 -ub 2048 -fa 1 -n 0 > /tmp/dsv4-lightning-32k-llama-bench.log 2>&1
last=$(grep -n 'Vulkan Timings:' /tmp/dsv4-lightning-32k-llama-bench.log | tail -n 1 | cut -d: -f1)
sed -n "${last},\$p" /tmp/dsv4-lightning-32k-llama-bench.log | tail -n 180
```

Patch inspection:

```sh
cd /tmp/llama-strix-beta-bench
git diff --check
git diff --stat
git diff
git status --short
```

## Next actions

1. The user reviews and understands the four-file implementation and result summary.
2. Commit only after explicit user approval for that commit action.
3. Remove this scratch pad before a PR if it is not useful as permanent documentation.
4. Restore the separate decode microbench stash from the main worktree only if that work resumes.
