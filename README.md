# strix-llama.cpp

<img src="halo-box.png" alt="Halo Box" width="260">

<b>llama.cpp for AMD Strix Halo</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

[upstream llama.cpp](https://github.com/ggml-org/llama.cpp) / [ggml](https://github.com/ggml-org/ggml) / [halo-box/llama.cpp](https://github.com/halo-box/llama.cpp)

## Halo Box

The goal is simple: more functionality, and the fastest llama.cpp around. And help the community with a single fast
llama.cpp fork instead of many competing ones.

Halo Box keeps two forks, and which one you want depends on your hardware:

| Fork | What it is |
| --- | --- |
| [halo-box/llama.cpp](https://github.com/halo-box/llama.cpp) | Stays close to mainline. Tracks upstream `master` and adds features and speedups on top, without diverging from how upstream works. |
| [halo-box/strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp) (this repo) | Purely optimised for AMD Strix Halo machines (Ryzen AI Max+, RDNA 3.5 / gfx1151). Free to diverge from upstream wherever that buys speed. |

Use `halo-box/llama.cpp` if you want upstream behaviour plus extras. Use this repo if you run a Strix Halo box and
want every last token/s out of it. Everything in `halo-box/llama.cpp` is merged in here regularly, so this repo is a
superset of it.

## What this is

A community fork of [`llama.cpp`](https://github.com/ggml-org/llama.cpp) for **AMD Strix Halo** - the Ryzen AI Max / Max+ 300 series
APUs (`gfx1151`, RDNA 3.5 integrated GPU, up to 128 GB of unified LPDDR5X shared between CPU and GPU).

Strix Halo is an unusual target. It has more addressable memory than almost any consumer discrete GPU, and far less
bandwidth; the iGPU shares its memory controller with the CPU; and both the Vulkan (RADV) and ROCm/HIP paths have
RDNA 3.5 specific behaviour that upstream has no hardware to reproduce. Changes that only make sense on this one
device - or that need a lot of measurement on it before they are ready to propose anywhere else - live here.

Practically, that means this repo has its own rules - most visibly, **AI coding agents may open pull requests here**.
See [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md).

## Quick start

Build from source. Two backends are worth using on Strix Halo:

**Vulkan** (RADV on Mesa; the easiest path, and the best one for most models)

```sh
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

**ROCm / HIP** (needs ROCm installed; build for `gfx1151` explicitly)

```sh
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

Then:

```sh
# chat, pulling the model straight from Hugging Face
./build/bin/llama-cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# OpenAI-compatible API server + web UI on http://localhost:8080
./build/bin/llama-server -hf ggml-org/Qwen3.5-0.8B-GGUF
```

Full build documentation, including Windows and Docker, is in [docs/build.md](docs/build.md).

## Running on Strix Halo

**Give the iGPU enough memory.** The APU's memory is shared, and the GPU can only use what the firmware and kernel let
it map. Two things control this: the UMA / dedicated-VRAM split in your BIOS, and the `amdgpu` GTT limit on Linux
(`amdgpu.gttsize`, in MB, and `ttm.pages_limit`, in 4 KB pages, as kernel command-line parameters). Which of those you
need depends on your kernel version - newer kernels size GTT more generously on their own. If a model that clearly
fits in RAM fails to allocate, this is almost always why.

**ROCm and batched inference.** On `gfx1151` there is an async-execution correctness bug in the HIP path: batched
inference can return badly wrong output (perplexity ~88 against ~9.4 for the same model). Setting
`HIP_LAUNCH_BLOCKING=1` serializes kernel launches and restores correctness, at a performance cost. Our ROCm CI runs
with it set. It is a workaround for a ROCm/HIP issue, not a fix, and it should go away when that is fixed upstream.

**Vulkan mat-vec chunking.** Batched mat-vec at 3, 5 and 6 columns is several times slower than at 1, 2 and 4 on RADV
here, which hits speculative decoding hard (it verifies at exactly those batch sizes). This fork splits such batches
into column counts that are measured to scale. Set `GGML_VK_MMV_NO_SPLIT=1` to restore the single upstream dispatch,
e.g. to compare against it.

**Measure things.** `GGML_VK_PERF_LOGGER=1` (any value) gives per-op timings on the Vulkan backend and is how most of the findings
above were made. `llama-bench` and `llama-perplexity` are the tools for before/after numbers, and PRs here are
expected to carry them - see [Benchmarking requirements](CONTRIBUTING.md#benchmarking-requirements).

## What differs from upstream

Everything else is upstream `llama.cpp`. The additions currently carried here:

**Inherited from [halo-box/llama.cpp](https://github.com/halo-box/llama.cpp)** (general features, sent upstream from there)

| Change | Flag / switch | What it does |
| --- | --- | --- |
| Speculative prefill | `--spec-prefill` | A small draft model scores prompt tokens by attention importance so the target model only prefills the ones that matter, cutting time-to-first-token on long prompts |
| N-gram table on disk | `--ngram-on-disk`, `--ngram-cache`, `--ngram-io-threads` | Keeps a model's n-gram hash-embedding table (28.8 GB on Qwen3.8-Flash-Next) off the memory budget entirely, reading only the rows each batch actually gathers |
| Adaptive speculative draft length | `--spec-draft-adaptive` | Sizes each draft from a measured per-sequence acceptance EMA rather than always drafting `--spec-draft-n-max`; speeds up MTP and DFlash |
| Vulkan fixes and tuning for RDNA 3.5 | | Driver-gated coopmat LDS stride padding, UMA bulk readback gated on host-cached mappings, IQ3_S mat-vec at batch sizes > 4, and a radix top-k kernel for large k |
| Hidden server presets | `hidden` in the models `.ini` | Keep a model loadable by name while omitting it from `GET /models` |

**Strix Halo only** (lives here, measured on `gfx1151`)

| Change | Flag / switch | What it does |
| --- | --- | --- |
| Multi-point reasoning budget | `--reasoning-budget-*` (upstream has the hard budget only) | Intro message, two soft warnings, a grace period to finish a paragraph after the budget runs out, and reasoning-token usage telemetry |
| Vulkan batched mat-vec chunking | `GGML_VK_MMV_NO_SPLIT=1` to disable | Dispatches batched mat-vec at the column counts that actually scale on RDNA 3.5, instead of the slow NUM_COLS shader variants |
| ROCm/HIP quantized matmul on RDNA 3.5 | | MMQ/MMVQ tile configurations and register prefetching, compact `MUL_MAT_ID` with quant- and shape-specific tiles for 256-expert MoE prefill, fused activation quantization for Q8_0 and Q6_K decode |
| ROCm/HIP MoE decode fusion | `GGML_CUDA_DISABLE_WEIGHTED_DOWN=1`, `GGML_CUDA_DISABLE_MMID_512=1` | Fused routing, weighted expert reduction and shared-expert gate for Qwen3.5/3.6 and Ling style MoE |
| ROCm/HIP Gated DeltaNet | `GGML_CUDA_DISABLE_GDN_GATE=1` | DPP reductions and a tiled multi-column kernel for prefill; the whole conv -> norm -> gate -> recurrence -> state copy decode chain as one kernel |
| ROCm/HIP grouped decode matvecs | `GGML_CUDA_DISABLE_MMV_GROUP=1` | Consecutive single-column matvecs that read the same activation (gate/up pairs, hyper-connection projections) launch as one kernel |
| ROCm/HIP flash attention on RDNA 3.5 | | WMMA path for D=256 prefill at depth, Q8_0 KV tile kernel for decode |
| Speculative checkpoints on device | | `llama-server` keeps speculative-decoding checkpoints in device memory instead of copying them to the host |
| ROCmFPx quant types | `llama-quantize` types `Q4_0_ROCMFP4`, `Q4_0_ROCMFP4_FAST`, `Q2/Q3/Q6/Q8_0_ROCMFPX` and the `_LEAN`/`_COHERENT`/`_STRIX` recipes | Loads the ROCmFP4 GGUFs published for Strix Halo. CPU codecs plus Vulkan dequant, mat-vec, matmul and integer-dot kernels; the coopmat2 flash-attention path does not decode them yet |
| Repeatable output at depth | | Freed KV cells are zeroed so masked-out rows never carry stale K/V, and the Vulkan radix top-k assigns output slots deterministically |

Every ROCm/HIP change above is guarded on architecture, shape and layout, so other devices see upstream behaviour.
Run `--help`, or see [tools/server/README.md](tools/server/README.md), for the full options.

## Supported backends

The ones that matter on this hardware:

| Backend | Notes |
| --- | --- |
| [Vulkan](docs/build.md#vulkan) | RADV on the RDNA 3.5 iGPU - the default recommendation |
| [HIP](docs/build.md#hip) | ROCm on `gfx1151` - see the `HIP_LAUNCH_BLOCKING` note above |
| [CPU](docs/build.md) | Zen 5 cores with AVX-512 - useful for offloading part of a model, though it shares bandwidth with the iGPU |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU acceleration |
| [RPC](tools/rpc) | Distribute a model across several machines |

`llama.cpp` supports many more (CUDA, Metal, SYCL, CANN, OpenCL, WebGPU, ...); they are all still present in the tree
and unmodified. See the [upstream README](https://github.com/ggml-org/llama.cpp) for that list.

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [Completions](docs/completions.md)
- [Models](docs/models.md)

## Contributing

This is a small community project. Strix Halo owners with a benchmark, a bug report, or a patch are exactly who it is
for.

- Anyone can open a PR, **including AI coding agents acting on their own** - this repo permits automated PR submission,
  unlike upstream. The rules that replace the upstream ban are in [AGENTS.md](AGENTS.md).
- Device-specific claims need numbers from the device, against a baseline you built and ran yourself. What exactly to
  report is in [Benchmarking requirements](CONTRIBUTING.md#benchmarking-requirements); read it before you benchmark.
- If your change is not Strix Halo specific, send it to [halo-box/llama.cpp](https://github.com/halo-box/llama.cpp)
  instead, so it can reach upstream.
- Read [CONTRIBUTING.md](CONTRIBUTING.md) before your first PR.

CI runs the standard llama.cpp suite plus a self-hosted `gfx1151` ROCm job on real hardware.

## Acknowledgements

This project is a fork and owes everything to the people who built what it forks:

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and [ggml](https://github.com/ggml-org/ggml) - Georgi Gerganov and the llama.cpp contributors - MIT license
- [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX) - the origin of the ROCmFPx project and the creator of the ROCmFP4 format - MIT license
- [ciru-ai/ROCmFPX](https://github.com/ciru-ai/ROCmFPX) - a fork of the above, and the tree the ROCmFPx quant formats and reference codecs here were hand-ported from (it shares no git history with llama.cpp, so it cannot be merged) - MIT license
- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
