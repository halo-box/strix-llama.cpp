# EXL3

CPU reference decoder for the EXL3 weights of [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3)
(MIT), read at `a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a`. EXL3 is exllamav3's own format, not a GGUF quantization
type. This tree does not add a ggml type for it: the payload is carried in GGUF as plain `I16` / `F16` tensors (see
"Container" below). Fork-local, built only with `-DGGML_EXL3=ON`.

No code from QTIP (GPL-3.0) is used. The format was implemented from the exllamav3 sources and is checked against
its independent CPU decoder `tests/testlib/trellis.py`.

## Format (one quantized linear, `y = x W`, `W` is k x n)

| Tensor | Type | Shape (numpy order) | Content |
|---|---|---|---|
| `.trellis` | int16 | `[k/16, n/16, 16*K]` | one 16x16 tile per (k-tile, n-tile), `256*K` bits each |
| `.suh` | fp16 | `[k]` | input-side sign and scale |
| `.svh` | fp16 | `[n]` | output-side sign and scale |
| `.mcg` / `.mul1` | int32 scalar | `[]` | codebook marker, value `0xCBAC1FED` / `0x83DCD12D`; no marker means 3inst |

- Rates: K = 1..8 for every codebook; K = 1.5, 2.5, 3.5 with mul1 only. K comes from the last trellis dim.
- Tile: `16*K` uint16 read as `8*K` little-endian uint32 words; ring bit p is bit `31 - p%32` of word `p/32`.
  Position i (0..255) adds D(i) = K bits (half rates: `floor(K)` plus bit `i%16` of `0xAAAA`). Its 16-bit state is
  the ring bits `[S(i) - 16, S(i))`, MSB first, wrapping (`S(i)` = sum of D up to i).
- Position `i = 8t + j` is tile element (r, c), `r = (t%4)*2 + (j&1) + 8*((j>>1)&1)`, `c = t/4 + 8*((j>>2)&1)`.
- Codebooks (uint32 arithmetic wraps, one fp16 rounding):
  3inst `x = s*89226354 + 64248484`, mcg `x = s*0xCBAC1FED`, then `x = (x & 0x8FFF8FFF) ^ 0x3B603B60` and
  `v = half(x & 0xFFFF) + half(x >> 16)`; mul1 `x = s*0x83DCD12D`, `v = fma(1024 + bytesum(x), half(0x1EEE), half(0xC931))`.
- `W = diag(suh) * H * W_hat * H * diag(svh)`, H the 128-point Sylvester Hadamard divided by sqrt(128), per 128 block.

`ggml_exl3_decode_rotated` returns `W_hat` (fp16). `ggml_exl3_dequantize` returns `W` (f32) in the order of
exllamav3 (H, suh, H, svh). It computes in double: the butterflies and suh are exact there, svh rounds once, so the
result is the f32 rounding of the exact `W`.

## Container (GGUF)

`convert_hf_to_gguf.py` detects `quantization_config.quant_method == "exl3"` and writes, per linear with GGUF base
name `<b>` (from the usual tensor name map):

| GGUF tensor | Type | ne | Content |
|---|---|---|---|
| `<b>.exl3_trellis` | I16 | `[16*K, n/16, k/16]` | the trellis bytes |
| `<b>.exl3_suh` | F16 | `[k]` | `.suh` bytes |
| `<b>.exl3_svh` | F16 | `[n]` | `.svh` bytes |

There is no `<b>.weight`. A `.bias` goes through the normal path. Other tensors keep their stored type unless the
arch converter transforms them. The markers are checked and replaced by one key:

| Key | Type | Value |
|---|---|---|
| `exl3.version` | u32 | 1 |
| `exl3.codebook` | string | `3inst`, `mcg` or `mul1` (files that mix codebooks are refused) |
| `exl3.hadamard_size` | u32 | 128 |
| `exl3.permutation` | string | name of the block reorder of the arch converter, if any |
| `exl3.permuted_tensors` | string array | base names whose 128-blocks were reordered |
| `exl3.source.quantization_config` | string | the source `quantization_config` as JSON |

The qwen35 / qwen3next converter reorders linear-attention V heads from grouped to tiled order. For EXL3 linears
that is done on whole 128-blocks (8 tiles plus the matching scale entries), which is exact because H is block
diagonal; the tiles themselves are unchanged. Not supported yet, refused by the converter: legacy packed `.su`/`.sv`,
2-D row-ring trellises (quantized token embeddings, n-gram tables), mixed codebooks.

## Tests

- `test-exl3`: the three codebook tables, and all 27 rate x codebook cases on synthetic tiles, against hashes made
  with exllamav3's `trellis.py`; type and shape checks.
- `test-exl3 --gguf FILE`: decodes every EXL3 linear of a container and prints sha256 of `W_hat` and `W` per tensor.
