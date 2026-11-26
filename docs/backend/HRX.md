# Build and run HRX on Strix Halo

The Linux x86-64 HRX backend uses the HRX runtime and Loom compiler to run ggml kernels on the GPU. The release build includes `llama-server`, `llama-bench`, `llama-perplexity`, the SDK runtime libraries and the hardware tests. On gfx1151, the Inkling 2 Mini loop controller uses the optimized mixed-precision kernel for eligible prefill batches.

## Build from source

Use a Linux x86-64 machine with Git, Bash, Python 3.12 with venv, CMake, Ninja, Clang, a C++ compiler and development headers. The build needs network access to install the SDK's pinned Python tools and fetch its source dependencies. You can build on a machine without a GPU. The CI build uses Ubuntu 24.04.

From the strix-llama.cpp repository root, fetch the SDK used for this backend and build the package:

```bash
git clone --depth 1 --branch v0.2.0 https://github.com/ROCm/hrx-system.git ../hrx-system-0.2.0
git -C ../hrx-system-0.2.0 rev-parse HEAD
# Expected: 0cd370f2baaa0cfef47cc23007f1a109906f8a52
CMAKE_BUILD_PARALLEL_LEVEL=4 scripts/hrx/build-release.sh ../hrx-system-0.2.0 build-hrx-release
```

The script builds the stock SDK from source, installs its `HrxPublicDist` component in the build directory and produces `build-hrx-release/llama-hrx-linux-x86_64.tar.gz`. It builds ggml in Release mode with `GGML_HRX=ON` and `GGML_NATIVE=OFF` so the host compiler does not specialize CPU code for the build machine. Loom compiles GPU kernels for the selected device at run time.

If you already installed that SDK, you can build the server directly. Set `HRX_SDK` to the install prefix containing `lib/cmake/hrx` and `lib/cmake/loomc`.

```bash
cmake -S . -B build-hrx -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HRX=ON -DCMAKE_PREFIX_PATH="$HRX_SDK" -DLLAMA_BUILD_TESTS=ON
cmake --build build-hrx --target llama-server test-backend-ops test-flash-attn-indexed-bias test-llama-archs
scripts/hrx/verify.sh build-hrx/bin build-hrx/check-logs
```

## Verify the package on the GPU

Use a Strix Halo host with a working amdgpu/KFD driver, permission to open its GPU device nodes, and system libraries compatible with the build machine. Keep the packaged libraries beside the executables when moving the build. From the directory containing the archive:

```bash
tar -xzf llama-hrx-linux-x86_64.tar.gz
cd llama-hrx
bin/llama-server --list-devices
scripts/verify.sh bin check-logs
```

The check verifies automatic Inkling placement and explicit device selection, requires `HRX0`, runs a strided Inkling controller case against the CPU reference, and checks that HRX dispatched `ggml_inkling_ctrl_apply_native`. It also runs indexed attention at 1,048,576 keys with 4,096 query rows and checks the output across the 32-bit offset boundary. Allow at least 30 GiB of free GPU-accessible memory for the wide attention test. The checks run without model weights.

The **HRX backend** workflow builds the archive on a hosted Linux runner. A Strix Halo DevLab runner downloads and tests that same archive on pushes, manual runs, and pull requests from branches in the repository. Fork pull requests run the hosted build. A maintainer can run the hardware job on a reviewed branch in the repository. The workflow saves the device list, dispatch trace and numerical test logs.

## Serve Inkling 2 Mini with DSpark

From the unpacked package, set `models` to the directory containing the two Mainline_Support GGUF files. For the tested one-slot, million-token configuration:

```bash
models=/path/to/Mainline_Support
scripts/serve-inkling2-dspark-hrx.sh \
  "$models/Inkling-2-Mini-UD-Q4_K_XL.gguf" \
  "$models/dspark-Inkling-2-Mini-Q8_0.gguf" \
  -c 1048576 -b 4096 -ub 4096 --fit off
```

Llama prefers an available HRX GPU for the `inkling2_loop` target. DSpark follows a single HRX target device. Use `--device` or `--device-draft` for explicit placement. The launcher proposes up to three tokens per step. The mixed controller runs on gfx1151 at multiples of 128 from 128 through 8,192 prefill tokens. Other shapes use the regular controller.

Add `--hrx-npu` to select the HRX GPU paired with XDNA2. It offloads eligible Q8_0 matrix multiplications when the GPU has independent work before the result is needed. The option requires an available NPU. GPU-only runs do not open the NPU. Use `--server FILE` before the two model paths to launch a binary from another build directory.

## Regenerate the XDNA2 GEMMs

Building `ggml-hrx` runs the Python 3 standard-library generator in `ggml/src/ggml-hrx/npu/`. It combines the signed INT8 core program with the DMA and compute schedule for each supported K dimension (2304 or 3072) and group count (1 or 2). CMake embeds the resulting four Loom files in the library. To inspect one independently:

```sh
python3 ggml/src/ggml-hrx/npu/generate-gemm.py 2304 1 gemm_i8_k2304_g1.loom
```

## Reproduce the 32k server comparison

All four rows used llama.cpp commit `f24a6872bbc8f6475d93dd2fb2c5c215aa2615bd` on `halo-box/strix-llama.cpp` base `1da4f8e50cd679aba69682990a9abe9bacec7f5c`. The Release build enabled HRX, Vulkan and HIP for gfx1151. The host ran Ubuntu 26.04.1, kernel 7.0.0-43-generic, on a Ryzen AI Max+ 395 with Radeon 8060S. The host compiler was GCC 15.2.0. HIP used AMD Clang 24.0.0git at d6f6cb691863e9258034ab48c1dfce21689d5606. HRX used the SDK revision above.

| Device | Context | Prefill tokens/s | Decode tokens/s |
|---|---:|---:|---:|
| HRX0 | 40,960 | 1,494.61 +/- 2.85 | 47.96 +/- 0.57 |
| Vulkan0 | 40,960 | 120.50 +/- 4.92 | 30.83 +/- 1.29 |
| ROCm0 | 40,960 | 101.69 +/- 0.78 | 9.68 +/- 1.07 |
| HRX0 | 1,048,576 | 1,487.37 +/- 9.85 | 47.68 +/- 0.35 |

Use `Inkling-2-Mini-UD-Q4_K_XL.gguf` (SHA-256 `e64276656f072e9b2197a61723667cbdb71da66a716ffd966e572c21088f3cd3`) and `dspark-Inkling-2-Mini-Q8_0.gguf` (`3da00cf7c31bbe5662b3c0d373ecf782d74d32ef43990acc158ce60c2add72e0`) from the Mainline_Support directory. Set both `--device` and `--device-draft` to the device in the row. Use `-c` as shown, `-b 4096 -ub 4096 -np 1 -ngl 99 -ngld 99 -fa on -ctk f16 -ctv f16 --fit off --backend-sampling --spec-type draft-dspark --spec-draft-n-max 3`, and leave the NPU off. Each request processed the same 32,768 input tokens with prompt caching disabled and generated 128 tokens at temperature 0. The table gives the mean and sample standard deviation across three requests after one warm-up.

### Compare perplexity and KL divergence

Build `llama-perplexity` at the base and candidate revisions in separate build directories. Use the same backend, model, corpus and cache settings in each pair. For example, from the repository root with `target` set to the target GGUF:

```bash
mkdir -p quality
args=(-m "$target" -dev Vulkan0 -ngl 99 -c 256 -b 256 -ub 128 -fa on -ctk f16 -ctv f16 -t 16 --chunks 2)
for corpus in prose code structured numeric; do
    ref="quality/$corpus.kld"
    "$base/bin/llama-perplexity" "${args[@]}" -f "tests/corpus/correctness-$corpus.txt" --kl-divergence-base "$ref"
    "$candidate/bin/llama-perplexity" "${args[@]}" --kl-divergence --kl-divergence-base "$ref"
done
```

The second command scores the tokens saved by the first and reports both perplexities and mean KL divergence in nats per scored token. Regenerate the reference for each FA setting, ubatch size and backend. For the CPU pair use `-dev none -ngl 0`. The target-only comparison does not need a draft model.
