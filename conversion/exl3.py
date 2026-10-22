# EXL3 (exllamav3) checkpoints: the trellis payloads are copied into the GGUF container, see ggml/exl3/README.md

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

import gguf

if TYPE_CHECKING:
    from .base import ModelBase

logger = logging.getLogger("hf-to-gguf")

EXL3_CONTAINER_VERSION = 1
EXL3_ROWS_VERSION      = 1
EXL3_HADAMARD_SIZE     = 128
EXL3_GROUP_DIM         = 256
EXL3_ROW_DIM           = 160
EXL3_MARKERS           = {"mcg": 0xCBAC1FED, "mul1": 0x83DCD12D}
EXL3_PARTS             = ("trellis", "suh", "svh", "mcg", "mul1", "su", "sv", "signs")
# a per-expert module of exllamav3's BlockSparseMLP: <prefix>.experts.<e>.<projection>
EXL3_EXPERT_RE         = re.compile(r"^(?P<prefix>.+)\.experts\.(?P<e>\d+)\.(?P<w>[^.]+)$")
# one shard of an n-gram row table (exllamav3 row_table.py): <prefix>.shard_<s>.trellis
EXL3_SHARD_RE          = re.compile(r"^(?P<prefix>.+)\.shard_(?P<s>\d+)$")


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
    rows: list[str] = []

    # MoE: the experts of one projection are stacked into the arch's merged expert tensor
    stacks: dict[str, dict[int, str]] = {}
    for base in modules:
        if m := EXL3_EXPERT_RE.match(base):
            stacks.setdefault(f"{m['prefix']}.experts.{m['w']}", {})[int(m["e"])] = base
    for merged, experts in sorted(stacks.items()):
        codebooks.add(write_exl3_experts(model, merged, [modules.pop(experts[e]) for e in sorted(experts)], sorted(experts)))

    # n-gram (PLE) row tables: a 2-D trellis, unsharded or in shards, next to the .head_bias of its hash heads
    tables: dict[str, dict[int, str]] = {}
    for base, parts in list(modules.items()):
        m = EXL3_SHARD_RE.match(base)
        prefix, shard = (m["prefix"], int(m["s"])) if m else (base, -1)
        if set(parts) == {"trellis"} and prefix + ".head_bias" in model.model_tensors:
            tables.setdefault(prefix, {})[shard] = modules.pop(base)["trellis"]
    for prefix, shards in sorted(tables.items()):
        write_exl3_ngram_rows(model, prefix, shards)
        rows.append(prefix)

    for base, parts in sorted(modules.items()):
        # a 2-D trellis with signs is a quantized token embedding (exllamav3 ngram_codec.py): rows of rings of 256
        if "signs" in parts:
            write_exl3_rows(model, base, parts)
            rows.append(base)
            continue
        codebook = exl3_codebook_of(model, base, parts)
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

        # the arch converter may split the output columns into several GGUF linears, or name the linear itself
        targets = model.exl3_linear_names(base, n)
        if targets is None:
            targets = [(model.map_tensor_name(base + ".weight").removesuffix(".weight"), 0, n)]

        # the arch converter may reorder whole 128-blocks; H is block diagonal, so that is a lossless tile reorder
        k_perm, n_perm = model.exl3_block_permutation(base, k, n)
        if (k_perm is not None or n_perm is not None) and len(targets) != 1:
            raise NotImplementedError(f"{base}: an EXL3 linear cannot be both block-permuted and split")
        if k_perm is not None:
            trellis = trellis[torch.from_numpy(exl3_block_rows(k_perm, EXL3_HADAMARD_SIZE // 16))]
            suh     = suh[torch.from_numpy(exl3_block_rows(k_perm, EXL3_HADAMARD_SIZE))]
        if n_perm is not None:
            trellis = trellis[:, torch.from_numpy(exl3_block_rows(n_perm, EXL3_HADAMARD_SIZE // 16))]
            svh     = svh[torch.from_numpy(exl3_block_rows(n_perm, EXL3_HADAMARD_SIZE))]
        if k_perm is not None or n_perm is not None:
            permuted.append(targets[0][0])

        # a split keeps whole 128-column blocks: the output H is block diagonal, so each part is an EXL3 linear of its own
        # with the same suh, its columns of svh and its n-tiles of the trellis
        bounds = [0] + [n1 for _, _, n1 in targets]
        if [n0 for _, n0, _ in targets] != bounds[:-1] or bounds[-1] != n or \
                any(b <= a or b % EXL3_HADAMARD_SIZE for a, b in zip(bounds, bounds[1:])):
            raise ValueError(f"{base}: the split {[t[1:] for t in targets]} does not cut {n} columns at 128-blocks")
        for new_base, n0, n1 in targets:
            part_trellis = trellis if len(targets) == 1 else trellis[:, n0 // 16:n1 // 16].contiguous()
            part_svh     = svh if len(targets) == 1 else svh[n0:n1].contiguous()
            logger.info(f"{new_base + '.exl3_trellis,':<40} K = {tile_words / 16:g} {codebook}, k = {k}, n = {n1 - n0}"
                        f"{' (block-permuted)' if new_base in permuted else ''}"
                        f"{f' (columns {n0}..{n1} of {n})' if len(targets) > 1 else ''}")
            model.gguf_writer.add_tensor(new_base + ".exl3_trellis", part_trellis.numpy(), raw_dtype=gguf.GGMLQuantizationType.I16)
            model.gguf_writer.add_tensor(new_base + ".exl3_suh",     suh.numpy(),          raw_dtype=gguf.GGMLQuantizationType.F16)
            model.gguf_writer.add_tensor(new_base + ".exl3_svh",     part_svh.numpy(),     raw_dtype=gguf.GGMLQuantizationType.F16)

        for part in ("trellis", "suh", "svh", codebook):
            if part in parts:
                del model.model_tensors[parts[part]]

    if len(codebooks) != 1:
        raise NotImplementedError(f"EXL3 checkpoints that mix codebooks are not supported: {sorted(codebooks)}")
    model.exl3_codebook = codebooks.pop()
    model.exl3_permuted = permuted
    model.exl3_rows = rows


def exl3_codebook_of(model: ModelBase, base: str, parts: dict[str, str]) -> str:
    """the codebook of one EXL3 linear from its marker (none: 3inst), every marker checked against its fixed value"""
    from .base import LazyTorchTensor

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
    return codebook


def write_exl3_experts(model: ModelBase, merged: str, experts: list[dict[str, str]], ids: list[int]) -> str:
    """stack the EXL3 linears of experts 0..E-1 of one projection: trellis [E, k/16, n/16, 16*K], suh [E, k], svh [E, n]"""
    from .base import LazyTorchTensor

    if ids != list(range(len(ids))):
        raise ValueError(f"{merged}: experts {ids[:4]}... are not numbered 0..{len(ids) - 1}")
    n_expert = model.find_hparam(["num_local_experts", "num_experts", "n_routed_experts"], optional=True)
    if n_expert is not None and n_expert != len(ids):
        raise ValueError(f"{merged}: {len(ids)} experts stored, config.json has {n_expert}")

    codebooks = {exl3_codebook_of(model, merged, parts) for parts in experts}
    if len(codebooks) != 1:
        raise NotImplementedError(f"{merged}: experts with different codebooks {sorted(codebooks)}")
    codebook = codebooks.pop()

    data: dict[str, list[Any]] = {"trellis": [], "suh": [], "svh": []}
    for parts in experts:
        for part in data:
            data[part].append(LazyTorchTensor.to_eager(model.model_tensors[parts[part]]()))
    shapes = {tuple(t.shape) for t in data["trellis"]}
    if len(shapes) != 1 or len({tuple(t.shape) for t in data["suh"]}) != 1 or len({tuple(t.shape) for t in data["svh"]}) != 1:
        raise NotImplementedError(f"{merged}: experts of different shapes or rates {sorted(shapes)}, they cannot be stacked")
    trellis = torch.stack(data["trellis"])
    suh     = torch.stack(data["suh"])
    svh     = torch.stack(data["svh"])
    _, tk, tn, tile_words = trellis.shape
    k, n = tk * 16, tn * 16
    if trellis.dim() != 4 or trellis.dtype != torch.int16 or suh.dtype != torch.float16 or svh.dtype != torch.float16:
        raise ValueError(f"{merged}: unexpected expert dtypes or shapes {trellis.dtype} {tuple(trellis.shape)}")
    if not exl3_rate_supported(tile_words, codebook) or k % EXL3_HADAMARD_SIZE or n % EXL3_HADAMARD_SIZE or \
            tuple(suh.shape) != (len(ids), k) or tuple(svh.shape) != (len(ids), n):
        raise ValueError(f"{merged}: unsupported EXL3 rate or inconsistent shapes {tuple(trellis.shape)}")

    # the name the arch converter gives the merged experts, with or without the leading "model."
    new_base = None
    for name in (merged, merged.removeprefix("model.")):
        try:
            new_base = model.map_tensor_name(name + ".weight").removesuffix(".weight")
            break
        except ValueError:
            continue
    if new_base is None or "_exps" not in new_base:
        raise NotImplementedError(f"{merged}: no merged expert tensor name for {len(ids)} EXL3 experts")

    logger.info(f"{new_base + '.exl3_trellis,':<40} K = {tile_words / 16:g} {codebook}, k = {k}, n = {n}, {len(ids)} experts")
    model.gguf_writer.add_tensor(new_base + ".exl3_trellis", trellis.numpy(), raw_dtype=gguf.GGMLQuantizationType.I16)
    model.gguf_writer.add_tensor(new_base + ".exl3_suh",     suh.numpy(),     raw_dtype=gguf.GGMLQuantizationType.F16)
    model.gguf_writer.add_tensor(new_base + ".exl3_svh",     svh.numpy(),     raw_dtype=gguf.GGMLQuantizationType.F16)
    for parts in experts:
        for part in ("trellis", "suh", "svh", codebook):
            if part in parts:
                del model.model_tensors[parts[part]]
    return codebook


def write_exl3_rows(model: ModelBase, base: str, parts: dict[str, str]) -> None:
    from .base import LazyTorchTensor

    if set(parts) != {"trellis", "signs"}:
        raise ValueError(f"{base}: row-ring table with unexpected parts {sorted(parts)}")
    trellis = model.model_tensors[parts["trellis"]]()
    signs   = model.model_tensors[parts["signs"]]()
    if trellis.dim() != 2 or trellis.dtype != torch.int16 or signs.dim() != 2 or signs.dtype != torch.float16:
        raise ValueError(f"{base}: unexpected row table {trellis.dtype} {tuple(trellis.shape)}, signs {signs.dtype} {tuple(signs.shape)}")
    n_rows, row_words = trellis.shape
    groups, group_dim = signs.shape
    # exllamav3 Embedding: K = (row_words / groups - 1) / 16, rows padded to whole groups of 256
    if group_dim != EXL3_GROUP_DIM or row_words % groups or (row_words // groups - 1) % 16 or not 1 <= (row_words // groups - 1) // 16 <= 8:
        raise ValueError(f"{base}: row width {row_words} does not hold {groups} rings of {EXL3_GROUP_DIM}")
    K = (row_words // groups - 1) // 16

    new_base = model.map_tensor_name(base + ".weight").removesuffix(".weight")
    if new_base != gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.TOKEN_EMBD]:
        raise NotImplementedError(f"{base}: only the token embedding can be an EXL3 row table, not {new_base}")
    hidden = model.find_hparam(["hidden_size", "n_embd"])
    if groups * EXL3_GROUP_DIM < hidden or (groups - 1) * EXL3_GROUP_DIM >= hidden:
        raise ValueError(f"{base}: {groups} groups of {EXL3_GROUP_DIM} do not cover hidden size {hidden}")

    logger.info(f"{new_base + '.exl3r_trellis,':<40} K = {K} mul1 rows, {n_rows} x {groups} groups of {EXL3_GROUP_DIM}")
    model.gguf_writer.add_tensor(new_base + ".exl3r_trellis", LazyTorchTensor.to_eager(trellis).numpy(), raw_dtype=gguf.GGMLQuantizationType.I16)
    model.gguf_writer.add_tensor(new_base + ".exl3r_signs",   LazyTorchTensor.to_eager(signs).numpy(),   raw_dtype=gguf.GGMLQuantizationType.F16)
    for part in ("trellis", "signs"):
        del model.model_tensors[parts[part]]


def write_exl3_ngram_rows(model: ModelBase, prefix: str, shards: dict[int, str]) -> None:
    """an n-gram row table (exllamav3 ngram_codec.py): rows of one ring of 160, plus the F16 bias of the row's hash head;
    the shards are written back to back as one table, a shard at a time"""
    from .base import LazyTorchTensor

    ids = sorted(shards)
    if ids != [-1] and ids != list(range(len(ids))):
        raise ValueError(f"{prefix}: row table shards {ids[:4]}... are not numbered 0..{len(ids) - 1}")
    names  = [shards[i] for i in ids]
    metas  = [model.model_tensors[name]() for name in names]
    shapes = [tuple(t.shape) for t in metas]
    if any(t.dtype != torch.int16 or t.dim() != 2 for t in metas) or len({s[1] for s in shapes}) != 1:
        raise ValueError(f"{prefix}: unexpected row table shards {sorted({(str(t.dtype), tuple(t.shape)) for t in metas})}")
    # exllamav3 RowTable: every shard but the last has the same row count
    if len({s[0] for s in shapes[:-1]}) > 1 or shapes[-1][0] > shapes[0][0]:
        raise ValueError(f"{prefix}: row table shards of unequal sizes {sorted({s[0] for s in shapes})}")
    row_words = shapes[0][1]
    K = (row_words - 1) * 16 // EXL3_ROW_DIM
    if not 1 <= K <= 8 or 1 + EXL3_ROW_DIM * K // 16 != row_words:
        raise ValueError(f"{prefix}: row width {row_words} does not hold one ring of {EXL3_ROW_DIM}")

    bias = model.model_tensors[prefix + ".head_bias"]()
    if bias.dtype != torch.float16 or bias.dim() != 2 or bias.shape[1] != EXL3_ROW_DIM:
        raise ValueError(f"{prefix}: unexpected head bias {bias.dtype} {tuple(bias.shape)}")

    new_base = model.exl3_row_table_name(prefix, EXL3_ROW_DIM)
    n_rows = sum(s[0] for s in shapes)

    def load(gen: Callable[[], Any]):
        # a fresh lazy tensor every call, or to_eager() memoizes every shard
        return lambda: LazyTorchTensor.to_eager(gen()).numpy()

    table = gguf.LazyChunkedTensor([load(model.model_tensors[name]) for name in names], shape=(n_rows, row_words), dtype=np.int16)
    logger.info(f"{new_base + '.exl3r_trellis,':<40} K = {K} mul1 rows, {n_rows} x {EXL3_ROW_DIM} in {len(names)} shard(s), "
                f"{bias.shape[0]} hash heads")
    model.gguf_writer.add_tensor(new_base + ".exl3r_trellis", table, raw_dtype=gguf.GGMLQuantizationType.I16)
    model.gguf_writer.add_tensor(new_base + ".exl3r_bias",    LazyTorchTensor.to_eager(bias).numpy(), raw_dtype=gguf.GGMLQuantizationType.F16)
    for name in names + [prefix + ".head_bias"]:
        del model.model_tensors[name]


def add_exl3_metadata(model: ModelBase, quant_config: dict[str, Any]) -> None:
    w = model.gguf_writer
    w.add_uint32(gguf.Keys.Exl3.VERSION, EXL3_CONTAINER_VERSION)
    w.add_string(gguf.Keys.Exl3.CODEBOOK, model.exl3_codebook)
    w.add_uint32(gguf.Keys.Exl3.HADAMARD_SIZE, EXL3_HADAMARD_SIZE)
    if model.exl3_rows:
        w.add_uint32(gguf.Keys.Exl3.ROWS_VERSION, EXL3_ROWS_VERSION)
    if model.exl3_permuted:
        w.add_string(gguf.Keys.Exl3.PERMUTATION, model.exl3_permutation)
        w.add_array(gguf.Keys.Exl3.PERMUTED_TENSORS, model.exl3_permuted)
    w.add_string(gguf.Keys.Exl3.SOURCE_QUANT_CONFIG, json.dumps(quant_config, sort_keys=True))
