# EXL3 (exllamav3) checkpoints: the trellis payloads are copied into the GGUF container, see ggml/exl3/README.md

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

import gguf

if TYPE_CHECKING:
    from .base import ModelBase

logger = logging.getLogger("hf-to-gguf")

EXL3_CONTAINER_VERSION = 1
EXL3_HADAMARD_SIZE     = 128
EXL3_MARKERS           = {"mcg": 0xCBAC1FED, "mul1": 0x83DCD12D}
EXL3_PARTS             = ("trellis", "suh", "svh", "mcg", "mul1", "su", "sv")


def exl3_rate_supported(tile_words: int, codebook: str) -> bool:
    if tile_words % 16 == 0:
        return 16 <= tile_words <= 128
    return codebook == "mul1" and tile_words in (24, 40, 56)


def exl3_block_rows(block_perm: np.ndarray, size: int) -> np.ndarray:
    # expand a permutation of 128-blocks to a permutation of elements of `size` per block
    return (block_perm[:, None] * size + np.arange(size)[None, :]).reshape(-1)


def write_exl3_tensors(model: ModelBase) -> None:
    from .base import LazyTorchTensor

    modules: dict[str, dict[str, str]] = {}
    for name in model.model_tensors:
        base, _, part = name.rpartition(".")
        if part in EXL3_PARTS:
            modules.setdefault(base, {})[part] = name
    modules = {base: parts for base, parts in modules.items() if "trellis" in parts}
    if not modules:
        raise ValueError("quant_method is exl3 but the checkpoint has no .trellis tensors")

    codebooks: set[str] = set()
    permuted: list[str] = []
    for base, parts in sorted(modules.items()):
        if "su" in parts or "sv" in parts:
            raise NotImplementedError(f"{base}: legacy packed su/sv EXL3 tensors are not supported")
        if "mcg" in parts and "mul1" in parts:
            raise ValueError(f"{base}: both .mcg and .mul1 codebook markers are present")
        if "suh" not in parts or "svh" not in parts:
            raise ValueError(f"{base}: EXL3 linear without .suh/.svh")

        codebook = "mcg" if "mcg" in parts else "mul1" if "mul1" in parts else "3inst"
        if codebook != "3inst":
            marker = LazyTorchTensor.to_eager(model.model_tensors[parts[codebook]]())
            if marker.dtype != torch.int32 or marker.numel() != 1 or marker.dim() != 0:
                raise ValueError(f"{base}: malformed .{codebook} marker {marker.dtype} {tuple(marker.shape)}")
            if int(marker.view(-1)[0].item()) & 0xFFFFFFFF != EXL3_MARKERS[codebook]:
                raise ValueError(f"{base}: .{codebook} marker holds {int(marker.item()) & 0xFFFFFFFF:#x}, "
                                 f"expected {EXL3_MARKERS[codebook]:#x}")
        codebooks.add(codebook)

        trellis = model.model_tensors[parts["trellis"]]()
        suh     = model.model_tensors[parts["suh"]]()
        svh     = model.model_tensors[parts["svh"]]()
        if trellis.dim() != 3:
            raise NotImplementedError(f"{base}: {trellis.dim()}-D trellis (row-ring embedding or n-gram table) is not supported yet")
        if trellis.dtype != torch.int16 or suh.dtype != torch.float16 or svh.dtype != torch.float16:
            raise ValueError(f"{base}: unexpected dtypes {trellis.dtype} {suh.dtype} {svh.dtype}")
        tk, tn, tile_words = trellis.shape
        k, n = tk * 16, tn * 16
        if not exl3_rate_supported(tile_words, codebook):
            raise ValueError(f"{base}: unsupported EXL3 rate K = {tile_words / 16} with codebook {codebook}")
        if k % EXL3_HADAMARD_SIZE or n % EXL3_HADAMARD_SIZE or tuple(suh.shape) != (k,) or tuple(svh.shape) != (n,):
            raise ValueError(f"{base}: inconsistent EXL3 shapes {tuple(trellis.shape)} {tuple(suh.shape)} {tuple(svh.shape)}")

        new_base = model.map_tensor_name(base + ".weight").removesuffix(".weight")

        # the arch converter may reorder whole 128-blocks; H is block diagonal, so that is a lossless tile reorder
        k_perm, n_perm = model.exl3_block_permutation(base, k, n)
        if k_perm is not None:
            trellis = trellis[torch.from_numpy(exl3_block_rows(k_perm, EXL3_HADAMARD_SIZE // 16))]
            suh     = suh[torch.from_numpy(exl3_block_rows(k_perm, EXL3_HADAMARD_SIZE))]
        if n_perm is not None:
            trellis = trellis[:, torch.from_numpy(exl3_block_rows(n_perm, EXL3_HADAMARD_SIZE // 16))]
            svh     = svh[torch.from_numpy(exl3_block_rows(n_perm, EXL3_HADAMARD_SIZE))]
        if k_perm is not None or n_perm is not None:
            permuted.append(new_base)

        logger.info(f"{new_base + '.exl3_trellis,':<40} K = {tile_words / 16:g} {codebook}, k = {k}, n = {n}"
                    f"{' (block-permuted)' if new_base in permuted else ''}")
        model.gguf_writer.add_tensor(new_base + ".exl3_trellis", trellis.numpy(), raw_dtype=gguf.GGMLQuantizationType.I16)
        model.gguf_writer.add_tensor(new_base + ".exl3_suh",     suh.numpy(),     raw_dtype=gguf.GGMLQuantizationType.F16)
        model.gguf_writer.add_tensor(new_base + ".exl3_svh",     svh.numpy(),     raw_dtype=gguf.GGMLQuantizationType.F16)

        for part in ("trellis", "suh", "svh", codebook):
            if part in parts:
                del model.model_tensors[parts[part]]

    if len(codebooks) != 1:
        raise NotImplementedError(f"EXL3 checkpoints that mix codebooks are not supported: {sorted(codebooks)}")
    model.exl3_codebook = codebooks.pop()
    model.exl3_permuted = permuted


def add_exl3_metadata(model: ModelBase, quant_config: dict[str, Any]) -> None:
    w = model.gguf_writer
    w.add_uint32(gguf.Keys.Exl3.VERSION, EXL3_CONTAINER_VERSION)
    w.add_string(gguf.Keys.Exl3.CODEBOOK, model.exl3_codebook)
    w.add_uint32(gguf.Keys.Exl3.HADAMARD_SIZE, EXL3_HADAMARD_SIZE)
    if model.exl3_permuted:
        w.add_string(gguf.Keys.Exl3.PERMUTATION, model.exl3_permutation)
        w.add_array(gguf.Keys.Exl3.PERMUTED_TENSORS, model.exl3_permuted)
    w.add_string(gguf.Keys.Exl3.SOURCE_QUANT_CONFIG, json.dumps(quant_config, sort_keys=True))
