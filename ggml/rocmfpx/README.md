# ROCmFPx

The 2/3/6/8-bit ROCmFPx tensor types. The formats originate in
[charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX) (MIT); the codecs
here were hand-ported via [ciru-ai/ROCmFPX](https://github.com/ciru-ai/ROCmFPX).
The 4-bit types live in `../rocmfp4/`.

This directory owns the CPU codecs (`rocmfpx.c`) and the standalone ROCmFP2
reference encoder (`rocmfp2_reference.c`) that the optimized FP2 encoder is
checked against. The rest of ggml registers and dispatches the types.

## Layouts

All formats use 32-weight blocks and the same unsigned E4M3 scale byte as
ROCmFP4 (see `../rocmfp4/README.md`).

| Type | GGUF type id | Payload | Scales | Block bytes | BPW |
|---|---:|---|---|---:|---:|
| `Q2_0_ROCMFPX` | 107 | 32 packed 2-bit codes, LSB first | 2, one per 16-weight half | 10 | 2.50 |
| `Q3_0_ROCMFPX` | 104 | 32 packed 3-bit codes, LSB first | 2, one per 16-weight half | 14 | 3.50 |
| `Q6_0_ROCMFPX` | 102 | 32 packed 6-bit codes, LSB first | 2, one per 16-weight half | 26 | 6.50 |
| `Q8_0_ROCMFPX` | 103 | 32 signed int8 | 1 per block | 33 | 8.25 |

- ROCmFP2 codes map to `-4, -1, 1, 4`.
- ROCmFP3 codes are sign + 2-bit magnitude index into `0, 1, 2, 4`.
- ROCmFP6 codes are sign + 5-bit magnitude; code 32 (sign set, magnitude 0)
  decodes to `-32`.
- ROCmFP8 is signed int8 clamped to `[-127, 127]`.

### Type ids and the two upstreams

Ids 100-104 are the same in every ROCmFPX tree. For `Q2_0_ROCMFPX` the
upstreams disagree: charlie12345/ROCmFPX (and the kingjones30 fork the ROCm
builds come from) use tensor type 107 and file type 119; ciru-ai/ROCmFPX moved
it to 108 / 122 and put `Q7_0_ROCMFPX` at 107 / 119. This tree follows
charlie12345, which is what the published GGUFs carry. Ids 105, 106 (TURBO3_0,
TURBO4_0) and 108 (ROCMI4 or ciru-ai's Q2_0) are deliberately left unassigned
so files using them are rejected instead of misread.

## What is implemented in this tree

- CPU: quantize / dequantize / vec_dot for all four types, imatrix weighting.
- Vulkan: dequant, mat-vec and matmul for Q3/Q6/Q8; q8_1 integer-dot mat-vec
  for Q8. Q2 runs on CPU.
- Not implemented: ROCm/HIP kernels. On ROCm these types run through the CPU
  codecs.

## Tests

- `ctest -R rocmfp` runs `test-rocmfpx` (codecs and imatrix weighting) and
  `test-rocmfp2-reference` (the FP2 reference encoder: layout, scale encoding,
  golden vectors, non-finite input).
- `test-quantize-fns` covers all four types with per-width error budgets.
- `test-backend-ops` includes them in its type lists.
- `gguf-py/tests/test_quants.py --type Q6_0_ROCMFPX` (etc.) checks the Python
  dequantizers against C.
