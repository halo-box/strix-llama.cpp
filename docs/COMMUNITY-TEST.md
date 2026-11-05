# Community test branch: hb/community-test

A test branch for Strix Halo owners (Ryzen AI Max / Max+ 300, Radeon 8060S / 8050S, gfx1151) on Linux, 64 GB and 128 GB.
It carries the Vulkan prefill and decode work for Qwen3.8-Flash-Next that has no merged PR yet, plus the MTP (multi-token
prediction) speedups, on top of halo-box master. It is not a release. Every change below is on by default unless marked
opt-in, and every one has a switch to turn it off, so you can compare.

## What is in it

Vulkan, Flash-Next prefill and decode (set the variable to 0 to turn a change off; `GGML_VK_EXPDOWN_V` is a kernel
variant number, default 9, and `GGML_VK_RAGGED_TAIL` a tile size, default 128):

| change | switch |
|---|---|
| hc combine kernels: load-first order, deferred attention-side store | `GGML_VK_HC_CMB`, `GGML_VK_HC_DEFER` |
| hc MoE epilogue folded into the post norm (MoE sum + shared expert) | `GGML_VK_HC_MOE_EPI` |
| hc inject, post-norm add and mix copy folded into the norm kernels | `GGML_VK_HC_INJECT_FOLD`, `GGML_VK_HC_POST_NORM_ADD`, `GGML_VK_HC_MIX_CPY` |
| hc down GEMM on an int8 (W8A8) route | `GGML_VK_HC_W8A8` |
| bf16 storage for the hc residual between combines | `GGML_VK_HC_BF16RES` |
| hc up mix on the dense q8_0 GEMM, 256 x 128 tile | `GGML_VK_HC_DQ8_TILE` |
| GDN: q/k norm and conv folded into the chunked scan, branch-free conv window | `GGML_VK_GDN_QKNORM`, `GGML_VK_GDN_CONVFOLD`, `GGML_VK_GDN_CONV_DIRECT` |
| GDN: gated output norm in the scan store, head-major z | `GGML_VK_GDN_OUTNORM`, `GGML_VK_GDN_ZHM` |
| MoE: expert repack at load, streaming down kernel, select-free A fragments | `GGML_VK_EXPERT_REPACK`, `GGML_VK_EXPDOWN_V`, `GGML_VK_EXPDOWN_NOSEL` |
| MoE: q8_0 experts on the streaming down kernel, f16 router | `GGML_VK_MMID_DOWN_Q8`, `GGML_VK_ROUTER_F16H` |
| dense q8_0 GEMM: long-K and f32-B variants, two-step prefetch | `GGML_VK_DENSE_DQ8_LONGK`, `GGML_VK_DENSE_DQ8_F32B`, `GGML_VK_DQ8_PF2` |
| attention: output gate folded into flash attention, cheaper selected-key pre-pass | `GGML_VK_FA_OUT_GATE`, `GGML_VK_FA_SEL_PREP2` |
| ragged GEMM tails, hc boundary GEMM | `GGML_VK_RAGGED_TAIL`, `GGML_VK_HC_BOUNDARY_GEMM` |
| decode weight prefetch from a second queue (gfx1151 only) | `GGML_VK_DECODE_PREFETCH_Q` |
| TQ2_T routed experts on the register-prefetch tile (#252) | `GGML_VK_MMID_TQ2T_PF` |

Model loading and MTP:

| change | switch |
|---|---|
| loads the merged-tensor Flash-Next GGUFs from the new upstream converter (qkv + gate, hc down + inject) | `LLAMA_HC_SPLIT=0` runs the merged hc matmul unsplit |
| MTP eh_proj as one GEMM over all streams and tokens | `LLAMA_MTP_EH_GEMM` |
| MTP keeps only the carry row of a prompt batch | `LLAMA_MTP_VERIFY_H_ALL=1` keeps every row |
| GDN conv split at a recurrent state snapshot | `LLAMA_GDN_CONV_SPLIT` |
| MTP draft over a vocabulary subset (works with the head-less "shared" MTP files) | opt-in: `--spec-draft-mtp-vocab 94208` |
| split of the rollback-slot GDN scan on wide ubatches | opt-in: `LLAMA_GDN_KEEP_SPLIT=1` |
| cache-only MTP prompt catch-up | opt-in: `LLAMA_MTP_CACHE_ONLY=1` |
| resident MTP hand-over on Vulkan (faster prefill, slower decode) | opt-in: `LLAMA_MTP_RESIDENT=1` |
| one-token-ahead decode chaining for greedy decode | `LLAMA_DECODE_CHAIN` |

## Requirements

- Linux, Strix Halo. 128 GB boxes run every Flash-Next quant listed below. See "Which quant" for 64 GB.
- Mesa 26.2.4 or newer, stock RADV (check with `vulkaninfo --summary`, driverInfo). No special driver build or env.
- A recent shaderc `glslc` (the one that ships with the Vulkan SDK or a shaderc build). Older distro `glslc` builds
  (2023.x) lack `GL_EXT_integer_dot_product`, so the int8 kernels are not built and you get slower, different numbers.
  The configure log must say `GL_EXT_integer_dot_product supported by glslc`.
- This guide is for the Vulkan backend. The branch also builds with ROCm / HIP, but HIP is not tested here.
- SPIRV-Headers (the `spirv-headers` package, or a checkout). If the build stops with `'spv' has not been declared`,
  point the compiler at the headers as shown below.

## Build

```sh
git clone https://github.com/halo-box/strix-llama.cpp
cd strix-llama.cpp
git checkout hb/community-test
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF \
    -DVulkan_GLSLC_EXECUTABLE=/path/to/shaderc/glslc \
    -DSPIRV-Headers_DIR=/path/to/SPIRV-Headers/lib/cmake/SPIRV-Headers
# only if you see "'spv' has not been declared":
#   add  "-DCMAKE_CXX_FLAGS=-isystem /path/to/SPIRV-Headers/include"
cmake --build build --config Release -j
```

## Best prefill

```sh
./build/bin/llama-bench -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
    -ngl 99 -fa 1 -ctk f16 -ctv f16 -lm none -lzm on-direct \
    -p 2048 -n 0 -b 2048 -ub 2048 -d 0,32768
./build/bin/llama-bench -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
    -ngl 99 -fa 1 -ctk f16 -ctv f16 -lm none -lzm on-direct \
    -p 8192 -n 0 -b 8192 -ub 8192
```

- `-lm none -lzm on-direct` loads without mmap and streams the large per-layer embedding table from disk. Use it for
  every Flash-Next run; the default load mode keeps a second copy and can run out of memory.
- `-ub 2048` is the default we measure and the safe choice for MoE models. `-ub 8192` runs on 128 GB (needs about
  4 GB more GPU memory); we have not compared it against `-ub 2048` at the same prompt length on this branch, so try both
  on your long prompts. On 64 GB stay at `-ub 2048` (or 4096).
- `-fa 1` always.

## Best decode

```sh
./build/bin/llama-bench -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
    -ngl 99 -fa 1 -ctk f16 -ctv f16 -lm none -lzm on-direct -p 0 -n 128 -b 2048 -ub 2048
```

## llama-server with MTP

The MTP head is public: `unsloth/Qwen3.8-Flash-Next-GGUF`, file `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2.8 GB,
head-less: it borrows the embeddings and LM head of the main model). The numbers below use that file. The repo-root
`mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (4.1 GB, own LM head) also exists, but `--spec-draft-mtp-vocab` does not apply to it
(it is ignored with a warning) and we have not measured it on this branch.

```sh
LLAMA_GDN_KEEP_SPLIT=1 LLAMA_MTP_CACHE_ONLY=1 ./build/bin/llama-server \
    -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
    -ngl 99 -fa on -ctk f16 -ctv f16 -lm none -lzm on-direct -np 1 -c 32768 -b 2048 -ub 2048 \
    -md mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf -ngld 99 --spec-type draft-mtp \
    --spec-draft-n-max 3 --spec-draft-mtp-vocab 94208
```

- Draft length: `--spec-draft-n-max 3` for code and reasoning, `2` for prose.
- Greedy output with MTP is not bit-identical to plain decoding (the verify step runs batched kernels), but it is
  stable from run to run.

## Which quant (Flash-Next, unsloth/Qwen3.8-Flash-Next-GGUF)

Measured on a 128 GB box (balanced power mode) with the same flags as above, on an earlier build of this stack (same
kernels for these files). KLD and top-1 are against the BF16 model on WikiText-2 (24 x 2048 tokens); lower KLD is better.

| quant | file GB | GPU memory peak (pp8192 ub8192) | pp2048 | pp2048 at depth 32000 | tg128 | KLD | same top-1 |
|---|---|---|---|---|---|---|---|
| UD-Q4_K_XL | 111.3 | 84.2 GB | 2002.98 | 1667.74 | 31.43 | 0.0377 | 93.5 % |
| UD-IQ4_XS | 93.7 | 67.4 GB | 1384.94 | 1183.14 | 35.84 | 0.0662 | 91.1 % |
| UD-Q3_K_XL | 89.5 | 63.4 GB | 1379.05 | 1179.25 | 36.70 | 0.0761 | 90.6 % |
| UD-IQ3_XXS | 83.6 | 57.8 GB | 1387.57 | 1193.17 | 37.87 | 0.0940 | 89.4 % |

- 128 GB: UD-Q4_K_XL is the best quality and the fastest prefill (its expert and dense types have the most tuned
  kernels here). The smaller quants decode 15-21 % faster but prefill about 30 % slower.
- 64 GB: the model we run on 64 GB is Gyro-M (`agentionai/Qwen3.8-Flash-Next-Gyro-GGUF`, file
  `Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf`, a pruned TQ2_0 build; its routed experts use the TQ2_T kernel of #252). It
  loads and runs on this branch on a 64 GB box with `-lm none -lzm on-direct -ub 2048` (about 37 GB of GPU memory at
  pp2048); we did not re-measure its speed for this guide. The unsloth quants above peak at 57.8 GB of GPU memory or
  more, which is likely too tight for 64 GB; the smaller unsloth quants (UD-IQ2 / UD-IQ1, 66-80 GB files) are untested
  here. If you try one, use `-ub 2048`, `-lm none -lzm on-direct`, a small `-c`, and please report the result,
  including a failure to load.
- The UD-Q4_K_XL files from before and after the unsloth v3.1 re-upload have the same tensor types and bytes: same KLD, speed within run spread.

## Reference numbers (128 GB box, BIOS power mode balanced)

Our numbers on this branch, 128 GB box, BIOS power mode balanced, Mesa 26.2.4, UD-Q4_K_XL, the commands above.
llama-bench: mean of 2 runs of `-r 3`, spread = max - min of the two.

| test | t/s |
|---|---|
| pp2048, depth 0 (`-ub 2048`) | 2011.41 +- 10.70 |
| pp8192 (`-b 8192 -ub 8192`) | 1913.04 +- 0.37 |
| pp2048 at depth 32768 | 1685.35 +- 6.73 |
| tg128 | 31.25 +- 0.40 |

llama-server, greedy, 512 generated tokens, mean of 2 runs per prompt (the MTP command above; `--spec-draft-n-max` varied):

| prompt | no MTP | n1 | n2 | n3 | n3 acceptance |
|---|---|---|---|---|---|
| code | 31.67 | 44.50 | 54.87 | 60.56 | 88.1 % |
| reasoning | 31.14 | 43.56 | 53.42 | 59.61 | 89.2 % |
| prose | 31.15 | 38.67 | 42.21 | 41.11 | 48.0 % |

Without `LLAMA_GDN_KEEP_SPLIT=1 LLAMA_MTP_CACHE_ONLY=1 --spec-draft-mtp-vocab 94208`, n3 gives 53.59 / 53.10 / 35.56
(code / reasoning / prose). Server prompt processing with MTP on: 1420 t/s at 2k, 1506 at 8k, 1475 at 32k tokens.

## How to report

Open an issue or a discussion on halo-box/strix-llama.cpp with:

- the exact command line and the full `llama-bench` output (or the server log lines with the timings);
- `git rev-parse --short HEAD` of your checkout;
- Mesa version (`vulkaninfo --summary`, driverInfo), kernel (`uname -r`), RAM size and BIOS power mode;
- for a suspected regression, the same command with the matching switch from the tables above set to 0.
