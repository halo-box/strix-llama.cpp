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

## Row rings (quantized token embeddings and n-gram / PLE tables)

exllamav3 stores a quantized token embedding (`-eb`) and its hashed n-gram tables in a second format
(`modules/quant/exl3_lib/ngram_codec.py`): rows that decode one by one. A ring of `dim` positions is `1 + dim*K/16`
int16 words: word 0 is an fp16 scale, then stream bits `[i*K, (i+1)*K)` (LSB first in little-endian uint16 words) are
the low K bits of the 16-bit state of position i, whose higher bits are the symbols of positions i-1, i-2, ...
(wrapping). Always the mul1 codebook, no marker, K = 1..8.

- Token embedding `<key>.trellis` int16 `[rows, groups*(1 + 16*K)]` and `<key>.signs` fp16 `[groups, 256]`: each row is
  `groups` rings of 256, group g = `signs[g] * H256(decode * scale)`, H256 the orthonormal Walsh-Hadamard transform,
  cropped to the hidden size. K = `(row_words/groups - 1)/16`.
- n-gram table: rings of 160 plus the fp16 bias of the row's hash head (`head_bias [heads, 160]`, the head is the last
  `head_offsets[h] <= row`), with I64 `head_offsets`, `head_vocab_sizes`, `layer_multipliers` for the hashing; in its
  own file with `format = exl3_ngram_trellis`, `version = 1`, `K`, optionally split into `.shard_N.trellis`.

`ggml_exl3_decode_embd_row` and `ggml_exl3_decode_ngram_row` follow `dequant_rows`: the f32 product of the codebook
value and the scale (exact), then the Hadamard butterflies in f32 in the order of `hadamard_rotate` and the signs, or
the bias. Their outputs equal `dequant_rows` bit for bit (the CPU path of exllamav3 rounds the embedding to fp16 after
that; this decoder does not).

## Container (GGUF)

`convert_hf_to_gguf.py` detects `quantization_config.quant_method == "exl3"` and writes, per linear with GGUF base
name `<b>` (from the usual tensor name map):

| GGUF tensor | Type | ne | Content |
|---|---|---|---|
| `<b>.exl3_trellis` | I16 | `[16*K, n/16, k/16]` | the trellis bytes |
| `<b>.exl3_suh` | F16 | `[k]` | `.suh` bytes |
| `<b>.exl3_svh` | F16 | `[n]` | `.svh` bytes |

There is no `<b>.weight`. A `.bias` goes through the normal path. The experts of a MoE layer (exllamav3 writes one
module per expert, `<prefix>.experts.<e>.<projection>`) are stacked into the arch's merged expert tensor:
`<b>.exl3_trellis` I16 `[16*K, n/16, k/16, E]`, `.exl3_suh` F16 `[k, E]`, `.exl3_svh` F16 `[n, E]`, experts 0..E-1 in
order, every tile unchanged; the experts of a stack must have the same K. Other tensors keep their stored type unless
the arch converter transforms them. A quantized token embedding becomes `token_embd.exl3r_trellis` (I16
`[row_words, rows]`) and `token_embd.exl3r_signs` (F16 `[256, groups]`), bytes as stored. The markers are checked and
replaced by one key:

| Key | Type | Value |
|---|---|---|
| `exl3.version` | u32 | 1 |
| `exl3.codebook` | string | `3inst`, `mcg` or `mul1` (files that mix codebooks are refused) |
| `exl3.hadamard_size` | u32 | 128 |
| `exl3.permutation` | string | name of the block reorder of the arch converter, if any |
| `exl3.permuted_tensors` | string array | base names whose 128-blocks were reordered |
| `exl3.source.quantization_config` | string | the source `quantization_config` as JSON |
| `exl3.rows.version` | u32 | 1, when the file has a row table |

The qwen35 / qwen3next converter reorders linear-attention V heads from grouped to tiled order. For EXL3 linears
that is done on whole 128-blocks (8 tiles plus the matching scale entries), which is exact because H is block
diagonal; the tiles themselves are unchanged. Not supported yet, refused by the converter: legacy packed `.su`/`.sv`,
row tables other than the token embedding (n-gram / PLE tables), mixed codebooks.

The qwen35 converter takes the number of MTP layers of an EXL3 checkpoint from its `mtp.layers.<i>` tensors, not
from `mtp_num_hidden_layers` of `config.json` (a checkpoint can declare a layer it does not store).

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
rows (prompt processing) take a GEMM: `H(x * suh)` is written in f16 with a power-of-two scale per row, and a WMMA
kernel decodes 128-column slabs of `W_hat` into LDS once per block and k step (`W_hat` is exact in f16) and applies
`H` and `svh / 128` to the result in registers. The rotated activations are rounded to f16 once (one product, NMSE
~4e-8 against a double sum) or split into hi + lo parts (two products, f32 accumulation per k step, NMSE ~1e-13).
The default takes the precision of the GEMM the same backend context runs for an F32 weight of the same shape: one
product where that GEMM is mmb's `mmb_f32split_kernel` (one f16 product with both operands rounded, NMSE ~8e-8:
from 512 rows with the mmb opt-in that llama sets, for contiguous x), two products where it is f32 (MMVF, MMF,
rocBLAS: up to 511 rows, strided x, no mmb). `GGML_EXL3_GEMM_PRODUCTS=1` or `2` forces one choice. RDNA4 keeps the
groups of 8 for every row count. Other backends (CUDA, Vulkan, ...) do not support the op.

`ggml_exl3_mul_mat_id(ctx, trellis, x, ids, suh, svh, codebook)` (`GGML_OP_EXL3_MUL_MAT_ID`) is the EXL3
counterpart of `ggml_mul_mat_id` for a stack of experts: x F32 `[k, ne11, n_tokens]` (ne11 = 1 broadcasts x over the
slots), ids I32 `[n_used, n_tokens]`, result F32 `[n, n_used, n_tokens]`. The CPU backend groups the (slot, token) rows
by expert and runs each expert's rows through the routines of `EXL3_MUL_MAT` in the same order, so a row equals
`EXL3_MUL_MAT` of that expert on the same rows bit for bit.

`ggml_exl3_get_rows(ctx, table, ids, signs, n_out)` (`GGML_OP_EXL3_GET_ROWS`) decodes the rows `ids` of a token
embedding row table to F32 `[n_out, n]`; the CPU backend runs `ggml_exl3_decode_embd_row` on each row.

exllamav3's wide-row path (reconstruct `W` in fp16 with the Hadamards and scales folded in, then hgemm) was measured
here as a prototype and not kept: on the Qwen3.5-4B shapes at 512 rows the reconstruction pass alone took 356..480 us
per 2560 x 9216 linear (~70 ms per 512-token ubatch, its memory floor ~41 ms), the fp16-weight GEMM with the same
tile loop took 2.0..2.5 ms against 0.85 ms for the decoding GEMM (four times the weight bytes per row block), and
rocBLAS chose a 64x32x8 tile (5.3..7.3 ms). Sum over the 200 linears: 390..400 ms (own GEMM) and 944..968 ms
(rocBLAS) against 131..139 ms for the one-product decoding GEMM. Its accuracy equals that of the F32-weight mmb GEMM.

## Model loading (llama)

`exl3.*` keys set `hparams.exl3_codebook`. `llama_model_loader::create_tensor` then takes `<name>.exl3_trellis`,
`.exl3_suh` and `.exl3_svh` for a requested `<name>.weight` and returns the trellis as the weight. This happens
before the caller sees a missing tensor, so an optional head is not replaced by the tied embedding. The buffer type
is chosen with an `EXL3_MUL_MAT` probe. `build_lora_mm` makes `ggml_exl3_mul_mat` for such a weight (table
`llama_model::exl3`), and a check on the reserved graph refuses any other op that reads a trellis. An MTP layer is
a normal EXL3 block of linears (the NextN probe of the loader sees an EXL3 `eh_proj`). A token embedding row table
stands in the same way for `token_embd.weight` (probe `EXL3_GET_ROWS`, not as a tied output head): `build_get_rows`
decodes its rows with `ggml_exl3_get_rows` in `build_inp_embd` and in the qwen35 MTP graph, and the graph check also
refuses any other reader of a row table. A stack of experts stands for the `{k, n, E}` weight of a `MUL_MAT_ID`
(probe `EXL3_MUL_MAT_ID`) and `build_lora_mm_id` makes `ggml_exl3_mul_mat_id` for it; the MoE expert cache is refused.
Supported archs: qwen35, lfm2moe. EXL3 weights cannot be combined with LoRA adapters,
prism.hadamard, the MTP draft vocabulary or a virtual model of `llama_model_init_from_user` (metadata without a
complete tensor set).

## Direct loading of an exllamav3 directory

With `GGML_EXL3=ON`, `-m DIR` of the tools built on `common` (llama-cli, llama-server, llama-perplexity, ...) loads an
exllamav3 model directory as shipped. `common/exl3-dir.cpp` reads `config.json`, the safetensors files and the
tokenizer files, builds the GGUF metadata and tensor set that `convert_hf_to_gguf.py` writes for the directory, and
gives them to `llama_model_init_from_user`. The data are copied from the safetensors files while the model loads
(no mmap). Only qwen35 is supported. The importer:

- reads the header of every `*.safetensors` file of the directory, with or without `model.safetensors.index.json`,
  and refuses a tensor name found in two files; `quantization_config.json` (`tensor_storage`) is not read
- does not load the vision tower (`model.visual.*`, also an EXL3 one)
- loads the MTP layers under the names the converter gives them (`mtp.layers.<i>.*` as layer `n_layer + i`,
  `mtp.fc` as `nextn.eh_proj`, `mtp.pre_fc_norm_*` as `nextn.enorm` / `nextn.hnorm`, `mtp.norm` as
  `nextn.shared_head_norm`); their number comes from the tensors, not from `mtp_num_hidden_layers` of `config.json`
- loads a quantized token embedding as `token_embd.exl3r_trellis` / `_signs`
- loads a stored `lm_head` also when `tie_word_embeddings` is set
- names the pre-tokenizer by the sha256 of `json.dumps({"normalizer": ..., "pre_tokenizer": ...}, sort_keys=True)`
  of `tokenizer.json`, and refuses an unknown one

## Tests

- `test-exl3`: the three codebook tables, and all 27 rate x codebook cases on synthetic tiles, against hashes made
  with exllamav3's `trellis.py`; type and shape checks; row rings for K = 1..8 (n-gram rows with a bias, token
  embedding rows with signs, cropped) against hashes made with exllamav3's `ngram_codec.dequant_rows`, and
  `EXL3_GET_ROWS` on the op backend against that decoder, bit for bit; `EXL3_MUL_MAT_ID` on the op backend against
  `EXL3_MUL_MAT` per expert on the op backend, bit for bit, in the path the op took (the expert's rows in one call,
  or on a GPU backend chunks of up to 511 rows padded to 33, which run the GEMM), and against `ggml_mul_mat_id` on
  the f32 W of the experts: NMSE 1e-10 against the CPU, or against a double sum on 8 pairs at most twice that of the
  op backend's own `ggml_mul_mat_id` on the same W (`nmse_be_mul_mat_id_vs_f64`).
- `test-exl3 --gguf FILE`: decodes every EXL3 linear of a container and prints sha256 of `W_hat` and `W` per tensor
  (a stack of experts: all of them, one after the other).
- The op: `test-exl3` also runs `ggml_exl3_mul_mat` against `ggml_mul_mat` on the f32 `W` of `ggml_exl3_dequantize`
  for every rate x codebook case; `test-exl3 --op-gguf FILE` does it on the first linear of each (k, n, K) of a
  container at 1 to 513 rows (both sides of the 32-row GEMM threshold and of the 512-row mmb threshold). With
  `--backend NAME` first (e.g. `ROCm0`) the op runs on that backend while `ggml_mul_mat` stays on the CPU; the same
  `ggml_mul_mat` on the f32 `W` also runs on that backend (opted in to mmb as llama does). A case passes when the op
  is within NMSE 1e-10 of the CPU `ggml_mul_mat` (f32 grade), or when its NMSE against a double sum on the first rows
  is at most twice that of the backend's own `ggml_mul_mat` (the `nmse_be_mul_mat_vs_f64` field). The fp16 values
  of `W_hat` are checked bit for bit by the decode tests above. `test-backend-ops -o EXL3_MUL_MAT` and
  `-o EXL3_GET_ROWS` and `-o EXL3_MUL_MAT_ID` have the op cases. For a stack of experts `--op-gguf` runs
  `EXL3_MUL_MAT_ID` at 1 to 512 tokens, x broadcast and one x per slot.
- `test-exl3 [--backend NAME] --exact-gguf FILE`: the op of every linear and stack of a container on one-hot rows
  (`suh` = 1, x row r = a row of the 128-point Hadamard, so `H(x * suh)` = 128 e_r and every product is exact)
  against the CPU op. Rows run in chunks of 32 (the HIP dot kernel) must be equal bit for bit, which holds iff the
  backend decodes every value of `W_hat` as the CPU does. The WMMA paths (all rows in one call; chunks of 511) are
  checked to 2^-18 of each row's largest value: the gfx1151 f16 WMMA returns about half of the negative single
  products one f32 ulp toward -inf.
- `test-exl3 [--backend NAME] --rows-gguf FILE`: `EXL3_GET_ROWS` on every row of each row table of a container
  against `ggml_exl3_decode_embd_row`, bit for bit.
- `test-exl3 --dequantize IN OUT`: writes a plain GGUF of a container, for model-level comparisons. Each EXL3 linear
  becomes `<base>.weight` F32 `[k, n]` (a stack of experts F32 `[k, n, E]`) from `ggml_exl3_dequantize`; the `exl3.*`
  keys are dropped and every other key and tensor is copied unchanged. Any build can load the result.
- `test-exl3-dir`: writes a small synthetic qwen35 EXL3 directory (two shards with an index, a side file the index
  does not list, a vision tower, an MTP layer only in `config.json`) and checks the import, the transforms, the
  refusal of a repeated tensor name, and that the model loads and runs; then a second one with a stored MTP layer and a
  row-ring token embedding, which loads with its MTP layer and runs. `test-exl3-dir --compare DIR CONTAINER.gguf`
  compares the import of a directory with the converter's container of it (tensors, bytes and keys);
  `test-exl3-dir --dump DIR OUT.gguf` writes the import as a GGUF file.
