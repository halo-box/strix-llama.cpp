# ROCmFP4

`Q4_0_ROCMFP4` and `Q4_0_ROCMFP4_FAST` are the 4-bit ROCmFPx tensor types.
The format originates in [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX)
(MIT); the codec here was hand-ported via [ciru-ai/ROCmFPX](https://github.com/ciru-ai/ROCmFPX).

This directory owns the CPU codec (`rocmfp4.c`): quantize, dequantize, imatrix
weighting and row validation. The rest of ggml registers and dispatches the type.

## Layout

32 weights per block. Each weight is a 4-bit code: sign bit plus a 3-bit index
into the magnitude table `0, 1, 2, 3, 4, 6, 8, 10`. The low nibbles of the 16
payload bytes are weights 0-15, the high nibbles weights 16-31. Scales are
unsigned E4M3 bytes (`e <= 0x7E`, values above are treated as 0) decoded as
`mant * 2^-10` for exponent 0 and `(8 + mant) * 2^(exp - 11)` otherwise.

| Type | GGUF type id | Payload | Scales | Block bytes | BPW |
|---|---:|---|---|---:|---:|
| `Q4_0_ROCMFP4` | 100 | 16 bytes, 32 nibbles | 2, one per 16-weight half | 18 | 4.50 |
| `Q4_0_ROCMFP4_FAST` | 101 | 16 bytes, 32 nibbles | 1 per block | 17 | 4.25 |

Type ids match charlie12345/ROCmFPX, so GGUFs quantized there (the published
`*-ROCmFP4-FAST*`, `*-ROCmFP4-STRIX*` files) load here.

## What is implemented in this tree

- CPU: quantize / dequantize / vec_dot through the ggml type traits.
- Vulkan: dequant, mat-vec, matmul, and q8_1 integer-dot (MMQ / MMVQ) kernels
  under `ggml/src/ggml-vulkan/vulkan-shaders/`.
- Not implemented: ROCm/HIP kernels for these types. The HIP backend does not
  know them, so on ROCm they run through the CPU codec. Flash attention does
  not accept them as KV-cache types.

## Recipes in `llama-quantize`

`Q4_0_ROCMFP4`, `_LEAN` (Q5_K token embeddings), `_COHERENT` (Q6_K token
embeddings), `_FAST`, `_FAST_COHERENT`, `_STRIX` and `_STRIX_LEAN` (Strix Halo
attention K/V recipes). See `tools/quantize/quantize.cpp` and
`src/llama-quant.cpp` for the exact per-tensor routing.

## Tests

- `test-quantize-fns` covers both types with the 4-bit error budget.
- `test-backend-ops` includes both in its type lists (MUL_MAT, MUL_MAT_ID,
  GET_ROWS, CPY, SET_ROWS); run with `-b Vulkan0` for the Vulkan kernels.
- `gguf-py/tests/test_quants.py --type Q4_0_ROCMFP4` (and `_FAST`) checks the
  Python dequantizer against the C one bit for bit.
