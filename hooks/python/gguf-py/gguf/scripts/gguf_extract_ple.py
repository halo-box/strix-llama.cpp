#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import BinaryIO

import numpy as np
from tqdm import tqdm

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gguf

logger = logging.getLogger("gguf-extract-ple")

DEFAULT_TENSORS = ["per_layer_token_embd.weight"]

OUT_TYPES = {
    "f32":  gguf.GGMLQuantizationType.F32,
    "f16":  gguf.GGMLQuantizationType.F16,
    "bf16": gguf.GGMLQuantizationType.BF16,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "q5_1": gguf.GGMLQuantizationType.Q5_1,
    "q5_0": gguf.GGMLQuantizationType.Q5_0,
    "q4_1": gguf.GGMLQuantizationType.Q4_1,
    "q4_0": gguf.GGMLQuantizationType.Q4_0,
}

LOSSLESS_SOURCES = (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16)


def model_files(path: Path) -> list[Path]:
    # every shard of a split model, else just the file
    m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", path.name)
    if m is None:
        return [path]
    n = int(m.group(3))
    return [path.with_name(f"{m.group(1)}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]


def write_rows(fout: BinaryIO, rows: np.ndarray, src_type: gguf.GGMLQuantizationType, out_type: gguf.GGMLQuantizationType, out_row_bytes: int, chunk_rows: int) -> None:
    # converted and written a chunk at a time, so a table larger than RAM is never resident
    n_rows = rows.shape[0]
    bar = tqdm(desc="Writing", total=n_rows * out_row_bytes, unit="byte", unit_scale=True)
    for r0 in range(0, n_rows, chunk_rows):
        n = min(chunk_rows, n_rows - r0)
        chunk = np.asarray(rows[r0:r0 + n])
        if out_type != src_type:
            f32 = gguf.quants.dequantize(chunk, src_type).astype(np.float32, copy=False)
            chunk = gguf.quants.quantize(f32, out_type)
        chunk = np.ascontiguousarray(chunk).view(np.uint8)
        assert chunk.nbytes == n * out_row_bytes
        chunk.tofile(fout)
        bar.update(chunk.nbytes)
    bar.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the per-layer embedding (PLE) tables of a model to their own GGUF, for llama.cpp --ple")
    parser.add_argument("model",  type=Path, help="GGUF model, or any shard of a split model (all shards are searched)")
    parser.add_argument("output", type=Path, help="GGUF file to write")
    parser.add_argument("--tensor", action="append", help=f"tensor to extract, can be repeated (default: {', '.join(DEFAULT_TENSORS)})")
    parser.add_argument("--type", choices=["keep", *OUT_TYPES], default="keep", help="type of the written tables (default: keep, a byte copy)")
    parser.add_argument("--metadata-from", type=Path, help="take the architecture keys from this GGUF, for a source that has none (e.g. a lone shard)")
    parser.add_argument("--chunk-rows", type=int, default=1 << 20, help="rows converted and written at a time (default: 1048576)")
    parser.add_argument("--force", action="store_true", help="overwrite the output file")
    parser.add_argument("--verbose", action="store_true", help="increase output verbosity")
    args = parser.parse_args(None if len(sys.argv) > 1 else ["--help"])

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.output.exists() and not args.force:
        logger.error(f"{args.output} exists, use --force to overwrite")
        sys.exit(1)

    names = args.tensor or DEFAULT_TENSORS

    files = model_files(args.model)
    readers = [gguf.GGUFReader(f, "r") for f in files]

    meta = gguf.GGUFReader(args.metadata_from, "r") if args.metadata_from else readers[0]
    arch_field = meta.get_field(gguf.Keys.General.ARCHITECTURE)
    if arch_field is None:
        logger.error(f"{args.metadata_from or files[0]} has no {gguf.Keys.General.ARCHITECTURE}, pass --metadata-from with a GGUF of the model")
        sys.exit(1)
    arch = arch_field.contents()

    # the keys that describe the layout of the tables; llama.cpp checks them against the model
    keys = [f for f in meta.fields.values() if f.name.startswith(f"{arch}.ple.") or f.name == f"{arch}.embedding_length_per_layer_input"]

    found: dict[str, gguf.ReaderTensor] = {}
    for rd in readers:
        for t in rd.tensors:
            if t.name in names:
                found[t.name] = t
    missing = [n for n in names if n not in found]
    if missing:
        logger.error(f"not found in {args.model}: {', '.join(missing)}")
        sys.exit(1)

    writer = gguf.GGUFWriter(args.output, arch=arch, endianess=readers[0].endianess)

    name_field = meta.get_field(gguf.Keys.General.NAME)
    if name_field is not None:
        writer.add_name(name_field.contents())
    for f in keys:
        val_type = f.types[0]
        sub_type = f.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(f.name, f.contents(), val_type, sub_type=sub_type)
        logger.info(f"key {f.name}")
    if not keys:
        logger.warning(f"no {arch}.ple.* keys found, llama.cpp cannot check the tables against the model")

    jobs: list[tuple[np.ndarray, gguf.GGMLQuantizationType, gguf.GGMLQuantizationType, int]] = []
    for name in names:
        t = found[name]
        src_type = t.tensor_type
        out_type = src_type if args.type == "keep" else OUT_TYPES[args.type]
        if out_type != src_type and src_type not in LOSSLESS_SOURCES:
            logger.warning(f"{name}: converting {src_type.name} -> {out_type.name} adds a second rounding, prefer an F32/F16/BF16 source")

        ne = [int(x) for x in t.shape]
        n_rows = int(np.prod(ne[1:]))
        rows = t.data.reshape(n_rows, -1)

        block_size, type_size = gguf.GGML_QUANT_SIZES[out_type]
        if ne[0] % block_size != 0:
            logger.error(f"{name}: row size {ne[0]} is not a multiple of the {out_type.name} block size {block_size}")
            sys.exit(1)
        out_row_bytes = ne[0] // block_size * type_size

        byte_shape = (*reversed(ne[1:]), out_row_bytes)
        writer.add_tensor_info(name, byte_shape, np.dtype(np.uint8), n_rows * out_row_bytes, raw_dtype=out_type)
        jobs.append((rows, src_type, out_type, out_row_bytes))
        logger.info(f"tensor {name}: {src_type.name} -> {out_type.name}, shape {ne}, {n_rows * out_row_bytes / 1e9:.2f} GB")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    writer.close()

    # the data section follows the tensor infos; each tensor starts on an alignment boundary, as their offsets assume
    align = writer.data_alignment
    with open(args.output, "r+b") as fout:
        fout.seek(0, os.SEEK_END)
        for rows, src_type, out_type, out_row_bytes in jobs:
            fout.write(bytes(-fout.tell() % align))
            write_rows(fout, rows, src_type, out_type, out_row_bytes, args.chunk_rows)
        fout.write(bytes(-fout.tell() % align))

    logger.info(f"wrote {args.output}")


if __name__ == "__main__":
    main()
