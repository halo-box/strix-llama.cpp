# ROCm MMB context ownership

The gfx1151 MMB kernels used process-global activation scratch, BF16 marks, graph fingerprints, and converted-weight caches. Target and MTP draft contexts have independent asynchronous streams. Draft catch-up could overwrite storage still used by the target, including storage referenced by captured HIP graphs. Freeing either backend also released the other backend's cached storage.

Keep this state in its backend context. Scratch allocations and temporary conversions belong to the active stream's pool. Producer slots can be read across streams through graph dependencies. Converted weights and BF16 marks belong to the context. At context destruction, wait for its queued work, destroy its captured graphs, then release its MMB allocations before destroying the pools. No target/draft synchronization is added to inference.

Sharing immutable token embeddings or output projection weights is not itself the cause. The unsafe sharing was in the backend's mutable temporary storage and its lifetime management.

## Regression coverage

Run `test-backend-ops test -b ROCm0 -o MMB_CONTEXT` on gfx1151. The case uses two independent schedulers and the existing HC_CHAIN graph, with 512 tokens each. It preserves graph inputs across replay, checks stable serialized references, warms captured graphs, overlaps target and draft 16 times, and checks the surviving target after draft destruction. Every output must be finite and byte-identical to its serialized reference.

The final test fails against unchanged backend revision `8c1c282ecb194e8f02613defcc4a07c22b6d1c08` and passes against the fix, including with `GGML_CUDA_DISABLE_GRAPHS=1`. During test development, an initial version let the allocator overwrite inputs between replays; that version was not a valid race oracle. The final test protects those inputs and was rerun against the unchanged libraries.

## Model validation

Hardware: Ryzen AI Max+ PRO 395, Radeon 8060S/gfx1151, 128 GiB unified memory, balanced power profile. ROCm 10.0.0-2, HIP Clang 23, GCC 16.2.1, Release build, native CPU and gfx1151 GPU targets, HIP graphs enabled.

Target: AesSedai Qwen3.8-Flash-Next Q4_K_M, snapshot `b90616d610f47a124fe246b70bb136aeeb25a2ad`, five GGUF shards. Draft: `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`. Model hashes and corpus hashes are in `/tmp/strix-mtp-investigation/{model,corpus}-sha256.txt` on the test machine.

A diagnostic probe uses the production draft catch-up hook, fixed token IDs, FA enabled, full ROCm0 offload, batch/ubatch 2048, context 32768, lazy-mode on-direct, and load-mode dio. It saves complete final-prefill logits and 32 fixed-token continuation logit vectors, each with 248320 vocabulary entries. The reference is the unchanged fork with target-only MTP hidden extraction enabled. This isolates target/draft interference without sampling differences.

| Corpus | Prefill tokens | Token IDs, all logits, and top-1 |
| --- | ---: | --- |
| Code | 4096 | Byte-identical |
| Prose | 4096 | Byte-identical |
| Structured | 4096 | Byte-identical |
| Numeric | 4096 | Byte-identical |
| Code | 26776 | Byte-identical |

The HC_CHAIN and GATED_DELTA_NET focused suite passed 56/56 cases. The complete ROCm backend suite passed 29695/29695 cases, followed by the new MMB_CONTEXT regression. Unsupported operations were skipped by the existing test harness.

## Memory and performance

Context ownership can increase memory use because target and draft retain separate scratch and converted-weight caches. It does not duplicate the entire model or KV cache. Converted-weight initialization can also be repeated per context. The default inference path remains asynchronous; the added wait is at context destruction.

For the same 4096-token code probe with synchronized draft catch-up in both builds, intercepting hipMalloc/hipFree measured peak live allocations of 87,896,241,152 bytes before and 87,978,816,512 bytes after: an increase of 78.75 MiB. Both ended with zero tracked live bytes. This is one workload and measures HIP allocation requests, not total physical memory, registered host mappings, or allocator-internal overhead. It is not a maximum for other batch sizes or contexts. No throughput or latency conclusion is established by this allocation measurement.

## Upstream applicability

Checked upstream ggml-org/llama.cpp revision `53ed051ce5e8193652e449f43216ca3859454f49` on 2026-09-24. Its complete source tree has neither MMB nor hyperconnection kernel files. Its CUDA/HIP allocation pools already belong to each backend context and stream. Upstream also supports asynchronous MTP draft catch-up, but does not contain the process-global MMB state responsible for this failure.

The affected file entered this fork in commit `0636c9aee4a2ab1d38bb7e8c038227c6634c3857` (PR #63). This specific bug and fix are fork-specific. This source inspection does not certify upstream MTP against unrelated defects; no upstream runtime comparison was performed.

Sources: [upstream backend](https://github.com/ggml-org/llama.cpp/tree/53ed051ce5e8193652e449f43216ca3859454f49/ggml/src/ggml-cuda), [context-owned pools](https://github.com/ggml-org/llama.cpp/blob/53ed051ce5e8193652e449f43216ca3859454f49/ggml/src/ggml-cuda/common.cuh#L1559), [fork introduction](https://github.com/halo-box/strix-llama.cpp/commit/0636c9aee4a2ab1d38bb7e8c038227c6634c3857).

## Limits and artifacts

Status: TARGET PASS for the ROCm scratch-race regression and model interference checks; INCOMPLETE for the broader project certification matrix described below.

Raw logs, probe outputs, commands, source diff, allocation tracer, and upstream source snapshots are under `/tmp/strix-mmb-fix` on the test machine. The earlier reproduction and probe source are under `/tmp/strix-mtp-investigation`.

The model probe exercises catch-up and deterministic target decode, not the complete server sampling/acceptance loop or the original browser conversation. Multi-device configurations, NVIDIA CUDA, the optional graph-fork optimization, and a complete throughput/acceptance benchmark matrix are not certified. The separate rollback-validity and Vulkan attention issues are not changed by this fix.
