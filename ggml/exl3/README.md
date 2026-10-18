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

## Op

`ggml_exl3_mul_mat(ctx, trellis, x, suh, svh, codebook)` (`GGML_OP_EXL3_MUL_MAT`) computes `y = x W` without
materializing `W`: `y = H(H(x * suh) W_hat) * svh / 128`, H the unnormalized 128-point Hadamard. The CPU backend
(`ggml/src/ggml-cpu/exl3-ops.cpp`, built with `GGML_EXL3=ON`) decodes `W_hat` tile by tile through a 65536-entry
table of `ggml_exl3_codebook_value`, so every weight is exact, and accumulates in f32: per 128 block, then across
blocks.

The HIP backend (`ggml/src/ggml-cuda/exl3.cu`, built with `GGML_EXL3=ON`) runs the op on RDNA3 and RDNA4. It is
adapted from the RDNA GEMV kernels of exllamav3 (MIT, see the file header). One kernel computes `H(x * suh)` in f32;
the dot kernel decodes the trellis in registers to the exact codebook values, multiplies them with the f32
activations (`v_fma_mix_f32`), and the last block to finish a 128-column segment applies `H` and `svh / 128` to it.
One pass over the weights serves up to 8 rows; up to 32 rows run in groups of 8, one pass per group. On RDNA3, more
rows (prompt processing) take a GEMM: `H(x * suh)` is split into f16 hi + lo parts with a power-of-two scale per row,
and a WMMA kernel decodes 128-column slabs of `W_hat` into LDS once per block and k step and multiplies both parts
(f32 accumulation, each k step added in f32), so the op stays at f32 accuracy. RDNA4 keeps the groups of 8 for every
row count. Other backends (CUDA, Vulkan, ...) do not support the op.

## Model loading (llama)

`exl3.*` keys set `hparams.exl3_codebook`. `llama_model_loader::create_tensor` then takes `<name>.exl3_trellis`,
`.exl3_suh` and `.exl3_svh` for a requested `<name>.weight` and returns the trellis as the weight. This happens
before the caller sees a missing tensor, so an optional head is not replaced by the tied embedding. The buffer type
is chosen with an `EXL3_MUL_MAT` probe. `build_lora_mm` makes `ggml_exl3_mul_mat` for such a weight (table
`llama_model::exl3`), and a check on the reserved graph refuses any other op that reads a trellis. Supported archs:
qwen35. EXL3 weights cannot be combined with LoRA adapters, prism.hadamard, the MTP draft vocabulary or a virtual
model of `llama_model_init_from_user` (metadata without a complete tensor set).

## Direct loading of an exllamav3 directory

With `GGML_EXL3=ON`, `-m DIR` of the tools built on `common` (llama-cli, llama-server, llama-perplexity, ...) loads an
exllamav3 model directory as shipped. `common/exl3-dir.cpp` reads `config.json`, the safetensors files and the
tokenizer files, builds the GGUF metadata and tensor set that `convert_hf_to_gguf.py` writes for the directory, and
gives them to `llama_model_init_from_user`. The data are copied from the safetensors files while the model loads
(no mmap). Only qwen35 is supported. The importer:

- reads the header of every `*.safetensors` file of the directory, with or without `model.safetensors.index.json`,
  and refuses a tensor name found in two files; `quantization_config.json` (`tensor_storage`) is not read
- does not load the vision tower (`model.visual.*`) or MTP tensors, and does not take an MTP layer count from
  `config.json`
- loads a stored `lm_head` also when `tie_word_embeddings` is set
- names the pre-tokenizer by the sha256 of `json.dumps({"normalizer": ..., "pre_tokenizer": ...}, sort_keys=True)`
  of `tokenizer.json`, and refuses an unknown one

## Tests

- `test-exl3`: the three codebook tables, and all 27 rate x codebook cases on synthetic tiles, against hashes made
  with exllamav3's `trellis.py`; type and shape checks.
- `test-exl3 --gguf FILE`: decodes every EXL3 linear of a container and prints sha256 of `W_hat` and `W` per tensor.
- The op: `test-exl3` also runs `ggml_exl3_mul_mat` against `ggml_mul_mat` on the f32 `W` of `ggml_exl3_dequantize`
  for every rate x codebook case; `test-exl3 --op-gguf FILE` does it on the first linear of each (k, n, K) of a
  container at 1 to 513 rows (both sides of the 32-row GEMM threshold). Both require NMSE <= 1e-10. With `--backend NAME` first (e.g. `ROCm0`) the op runs on
  that backend while `ggml_mul_mat` stays on the CPU. `test-backend-ops -o EXL3_MUL_MAT` has the op cases.
- `test-exl3 --dequantize IN OUT`: writes a plain GGUF of a container, for model-level comparisons. Each EXL3 linear
  becomes `<base>.weight` F32 `[k, n]` from `ggml_exl3_dequantize`; the `exl3.*` keys are dropped and every other key
  and tensor is copied unchanged. Any build can load the result.
- `test-exl3-dir`: writes a small synthetic qwen35 EXL3 directory (two shards with an index, a side file the index
  does not list, a vision tower, an MTP layer only in `config.json`) and checks the import, the transforms, the
  refusal of a repeated tensor name, and that the model loads and runs. `test-exl3-dir --compare DIR CONTAINER.gguf`
  compares the import of a directory with the converter's container of it (tensors, bytes and keys);
  `test-exl3-dir --dump DIR OUT.gguf` writes the import as a GGUF file.
