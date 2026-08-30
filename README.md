# strix-llama.cpp

<div align="center">

<b>llama.cpp for AMD Strix Halo</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

[upstream llama.cpp](https://github.com/ggml-org/llama.cpp) / [ggml](https://github.com/ggml-org/ggml) / [halo-box/llama.cpp](https://github.com/halo-box/llama.cpp)

</div>

## What this is

A community fork of [`llama.cpp`](https://github.com/ggml-org/llama.cpp) for **AMD Strix Halo** - the Ryzen AI Max / Max+ 300 series
APUs (`gfx1151`, RDNA 3.5 integrated GPU, up to 128 GB of unified LPDDR5X shared between CPU and GPU).

Strix Halo is an unusual target. It has more addressable memory than almost any consumer discrete GPU, and far less
bandwidth; the iGPU shares its memory controller with the CPU; and both the Vulkan (RADV) and ROCm/HIP paths have
RDNA 3.5 specific behaviour that upstream has no hardware to reproduce. Changes that only make sense on this one
device - or that need a lot of measurement on it before they are ready to propose anywhere else - live here.

This fork tracks upstream `master` and stays mergeable with it. It is not a rewrite.

## Relationship to upstream

```
ggml-org/llama.cpp            upstream, the real project
  |
  +-- halo-box/llama.cpp      staging fork for changes intended to go upstream
        |
        +-- halo-box/strix-llama.cpp   <-- you are here: the Strix Halo community fork
```

**Nothing in this repository goes upstream from here.** If a change is general enough for upstream, it belongs in
[halo-box/llama.cpp](https://github.com/halo-box/llama.cpp) and is submitted from there, under the upstream project's
contribution and AI-usage rules. That separation is the whole point: it keeps upstream's review queue free of
device-specific work, and it lets this repo move fast on the things only Strix Halo owners care about.

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

| Change | Flag / switch | What it does |
| --- | --- | --- |
| Vulkan batched mat-vec chunking | `GGML_VK_MMV_NO_SPLIT=1` to disable | Dispatches batched mat-vec at the column counts that actually scale on RDNA 3.5, instead of the slow NUM_COLS shader variants |
| Adaptive speculative draft length | `--spec-draft-adaptive` | Sizes each draft from a measured per-sequence acceptance EMA rather than always drafting `--spec-draft-n-max` |
| Multi-point reasoning budget | `--reasoning-budget-*` | Intro message, two soft warnings, a grace period to finish a paragraph after the budget runs out, and reasoning-token usage telemetry |

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
- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
