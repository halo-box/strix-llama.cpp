from __future__ import annotations

import json
import re
from typing import Callable, Iterable, cast

import torch
from torch import Tensor

import gguf
import numpy as np

from .base import LazyTorchTensor, ModelBase, logger
from .qwen import _LinearAttentionVReorderBase, _Qwen35MRopeMixin
from .qwen3vl import Qwen3VLVisionModel


@ModelBase.register("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")
@ModelBase.example("Qwen/Qwen3.8-Flash-Next")
class Qwen4ExpTextModel(_Qwen35MRopeMixin, _LinearAttentionVReorderBase):
    """Qwen3.8-Flash-Next.

    Shares the Qwen3.5 gated delta net and interleaved mrope, and adds three things:
    hyper-connections in place of every layer norm, QSA sparse attention on the full
    attention layers, and PLE n-gram hash embeddings on a single layer.
    """

    model_arch = gguf.MODEL_ARCH.QWEN4EXP

    # the MTP head: one full-attention QSA block after the trunk, fed by the trunk's hc-wide residual
    supports_mtp_export = True

    # MTP tensors the shared Qwen remapper does not know
    _MTP_EXTRA = {
        "fc_embedding":           "nextn_fc_embedding",
        "fc_hidden":              "nextn_fc_hidden",
        "hyper_connection_mixer": "nextn_hc_head",
    }

    # the int64 PLE hash constants: name in a dense checkpoint, name in exllamav3's quantized n-gram table
    _PLE_CONSTANTS = {
        "layer_multipliers": ("ple_embedding.layer_multipliers",        "ple_embedding.ngram_embedding.layer_multipliers"),
        "head_offsets":      ("ple_embedding.ngram_heads_offsets",      "ple_embedding.ngram_embedding.head_offsets"),
        "head_vocab_sizes":  ("ple_embedding.ngram_heads_vocab_sizes",  "ple_embedding.ngram_embedding.head_vocab_sizes"),
    }

    # exllamav3 writes the quantized n-gram table to this file, which model.safetensors.index.json does not list
    _EXL3_NGRAM_FILE = "ngram_embedding.safetensors"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # only the shard names, so the table itself is never held
        self._ple_shards: dict[int, str] = {}
        self._ple_row_dim: int | None = None
        self._mtp_fc: dict[str, Tensor] = {}

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        part = name.split(".")[1] if name.startswith("mtp.") else None
        if part in cls._MTP_EXTRA:
            if cls.no_mtp:
                return None
            assert cls._original_block_count is not None
            rest = name.split(".", 2)[2]
            return f"model.layers.{cls._original_block_count}.{cls._MTP_EXTRA[part]}.{rest}", gen
        return super().filter_tensors(item)

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        tensors = super().index_tensors(remote_hf_model_id=remote_hf_model_id)

        # the quantized n-gram table of an EXL3 checkpoint is found by its file, not through the index
        sidecar = self.dir_model / self._EXL3_NGRAM_FILE
        if remote_hf_model_id is not None or not sidecar.is_file() or any(".ngram_embedding." in name for name in tensors):
            return tensors
        with open(sidecar, "rb") as f:
            header = json.loads(f.read(int.from_bytes(f.read(8), byteorder="little")))
        meta = header.get("__metadata__", {})
        if meta.get("format") != "exl3_ngram_trellis" or meta.get("version") != "1":
            raise ValueError(f"{sidecar}: not an exllamav3 n-gram table of version 1 ({meta.get('format')!r}, {meta.get('version')!r})")
        logger.info(f"gguf: indexing EXL3 n-gram table '{self._EXL3_NGRAM_FILE}' (K = {meta.get('K')}, {meta.get('rows')} rows)")
        for name, data in gguf.utility.SafetensorsLocal(sidecar).tensors.items():
            if self.lazy:
                data_gen = lambda data=data: LazyTorchTensor.from_local_tensor(data)  # noqa: E731
            else:
                dtype = LazyTorchTensor._dtype_str_map[data.dtype]
                data_gen = lambda data=data, dtype=dtype: torch.from_numpy(data.mmap_bytes()).view(dtype).reshape(data.shape)  # noqa: E731
            if titem := self.filter_tensors((name, data_gen)):
                tensors[titem[0]] = titem[1]
        return tensors

    def _read_hash_constants(self, key: str) -> list[int]:
        """Read an int64 PLE constant straight from the checkpoint.

        prepare_tensors() casts every non-float dtype to float32 before
        modify_tensors() sees it (base.py), which would silently round these
        45-bit multipliers. Reading the lazy tensor here bypasses that.
        """
        for suffix in self._PLE_CONSTANTS[key]:
            for name, gen in self.model_tensors.items():
                if name.endswith(suffix):
                    t = gen()
                    if t.dtype != torch.int64:
                        t = t.to(torch.int64)
                    return [int(x) for x in t.tolist()]
        raise ValueError(f"PLE constant {' or '.join(map(repr, self._PLE_CONSTANTS[key]))} missing from the checkpoint")

    def exl3_linear_names(self, name: str, n: int) -> list[tuple[str, int, int]] | None:
        m = re.match(r"^model\.layers\.(\d+)\.", name)
        bid = int(m[1]) if m else None
        # the fused indexer projection, split at a 128-column block like the dense path
        if name.endswith(".indexer.index_qk_proj"):
            n_q = self.hparams["indexer_n_heads"] * self.hparams["indexer_head_dim"]
            if n != n_q + self.hparams["indexer_head_dim"]:
                raise ValueError(f"{name}: {n} output columns, expected {n_q} + {self.hparams['indexer_head_dim']}")
            return [
                (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_Q_PROJ, bid, ""), 0, n_q),
                (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_K_PROJ, bid, ""), n_q, n),
            ]
        # the two MTP input projections have their own scales, so they cannot become one EXL3 eh_proj
        if name.endswith(".nextn_fc_embedding"):
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.NEXTN_E_PROJ, bid, ""), 0, n)]
        if name.endswith(".nextn_fc_hidden"):
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.NEXTN_H_PROJ, bid, ""), 0, n)]
        return None

    def exl3_row_table_name(self, name: str, row_dim: int) -> str:
        if not name.endswith(".ple_embedding.ngram_embedding"):
            return super().exl3_row_table_name(name, row_dim)
        self._ple_row_dim = row_dim
        return gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams

        self.gguf_writer.add_hyper_connection_count(hp["hc_count"])
        self.gguf_writer.add_hyper_connection_low_rank(hp["hc_lowrank"])

        n_layer = hp["num_hidden_layers"]
        self.gguf_writer.add_indexer_head_count(hp["indexer_n_heads"])
        self.gguf_writer.add_indexer_key_length(hp["indexer_head_dim"])
        self.gguf_writer.add_indexer_top_k(hp["indexer_budget"])
        ratio = hp["indexer_compress_ratio"]
        layer_types = hp["layer_types"]
        # the MTP block is a full-attention QSA layer too
        self.gguf_writer.add_attention_compress_ratios(
            [ratio if layer_types[i] == "full_attention" else 0 for i in range(n_layer)]
            + [ratio] * (self.block_count - n_layer)
        )

        # ple_layer_ids is 1-based in the HF config; empty means no n-gram table,
        # so emit no PLE keys rather than optional ones
        # the MTP head never reads PLE, so an MTP-only file carries none of it
        ple_layers = [i - 1 for i in hp["ple_layer_ids"]]
        if not ple_layers or self.mtp_only:
            return
        self.gguf_writer.add_ple_layers(ple_layers)
        self.gguf_writer.add_ple_ngram_size(hp["ngram_size"])
        self.gguf_writer.add_ple_heads_per_ngram(hp["heads_per_ngram"])
        self.gguf_writer.add_ple_conv_kernel(hp["ple_conv_kernel_size"])
        self.gguf_writer.add_ple_eos_token_id(self._eos_token_id())
        # an image is decoded as an embeddings-only batch, so the graph has no placeholder
        # ids to hash; carry the id and let it stand in for those positions
        _img = self._image_token_id()
        if _img is not None:
            self.gguf_writer.add_ple_image_token_id(int(_img))
        if self._ple_row_dim is not None:
            self.gguf_writer.add_embedding_length_per_layer_input(self._ple_row_dim)

        self.gguf_writer.add_ple_layer_multipliers(self._read_hash_constants("layer_multipliers"))
        self.gguf_writer.add_ple_head_offsets(self._read_hash_constants("head_offsets"))
        self.gguf_writer.add_ple_head_vocab_sizes(self._read_hash_constants("head_vocab_sizes"))

    def _image_token_id(self) -> int | None:
        img = self.hparams.get("image_token_id")
        return None if img is None else int(img)

    def _eos_token_id(self) -> int:
        eos = self.hparams.get("eos_token_id")
        if isinstance(eos, list):
            # the PLE hash resets n-grams on the primary EOS
            return int(eos[-1])
        if eos is None:
            raise ValueError("eos_token_id is required: the PLE hash resets its n-grams on it")
        return int(eos)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # int64 hash constants must stay exact; 1-D tensors force F32, so use KV
        if name.endswith("ple_embedding.layer_multipliers"):
            self._ple_multipliers = [int(x) for x in data_torch.tolist()]
            return []
        if name.endswith("ple_embedding.ngram_heads_offsets"):
            self._ple_head_offsets = [int(x) for x in data_torch.tolist()]
            return []
        if name.endswith("ple_embedding.ngram_heads_vocab_sizes"):
            self._ple_head_vocab_sizes = [int(x) for x in data_torch.tolist()]
            return []
        # the same constants of a quantized n-gram table, read by set_gguf_parameters
        if name.endswith(tuple(names[1] for names in self._PLE_CONSTANTS.values())):
            return []

        if ".ngram_embedding.shard_" in name:
            return self._place_ple_shard(data_torch, name)

        # eh_proj([e ; h_s]) = fc_embedding(e) + fc_hidden(h_s) for every hc stream s
        if name.endswith((".nextn_fc_embedding.weight", ".nextn_fc_hidden.weight")):
            self._mtp_fc[name.rsplit(".", 2)[1]] = data_torch
            if len(self._mtp_fc) < 2:
                return []
            eh = torch.cat([self._mtp_fc.pop("nextn_fc_embedding"), self._mtp_fc.pop("nextn_fc_hidden")], dim=1)
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.NEXTN_EH_PROJ, bid, ".weight"), eh)]

        # one projection feeds indexer q and k; split it, as minimax-m3 does
        if ".indexer.index_qk_proj.weight" in name:
            n_q = self.hparams["indexer_n_heads"] * self.hparams["indexer_head_dim"]
            q = data_torch[:n_q]
            k = data_torch[n_q:]
            return [
                (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_Q_PROJ, bid, ".weight"), q),
                (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_K_PROJ, bid, ".weight"), k),
            ]

        # Gemma zero-centred gammas the inherited norm.weight rule misses
        if name.endswith((".ple.norm_key.weight", ".ple.norm_query.weight", ".ple.norm_conv.weight",
                          ".indexer.q_layernorm.weight", ".indexer.k_layernorm.weight")):
            return [(self.map_tensor_name(name), data_torch + 1)]

        if name.endswith(".ple.conv1d.weight"):
            return [(self.map_tensor_name(name), data_torch.squeeze())]

        return super().modify_tensors(data_torch, name, bid)

    # the shards concatenate into a tensor of well over 100 GB
    # use LazyChunkedTensor here, a single shard resident at a time
    def _place_ple_shard(self, data_torch: Tensor, name: str) -> Iterable[tuple[str, Tensor]]:

        idx = int(name.rpartition(".shard_")[2].partition(".")[0])
        n_parts = self.hparams["split_ngram_parts"]

        self._ple_shards[idx] = name
        self._ple_row_dim = int(data_torch.shape[-1])

        if len(self._ple_shards) < n_parts:
            return []

        # the checkpoint may yield the shards in any order, the row order is by index
        shards = [self._ple_shards[i] for i in sorted(self._ple_shards)]
        rows = 0
        for shard in shards:
            shape = self.model_tensors[shard]().shape
            if int(shape[-1]) != self._ple_row_dim:
                raise ValueError(
                    f"PLE shard {shard} has row dim {int(shape[-1])}, expected {self._ple_row_dim}")
            rows += int(shape[0])

        table = gguf.LazyChunkedTensor(
            [self._load_ple_shard(shard) for shard in shards],
            shape=(rows, self._ple_row_dim),
            dtype=np.float32,
        )
        gguf_name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD]
        return [(gguf_name + ".weight", cast(Tensor, table))]

    def _load_ple_shard(self, name: str):
        def load() -> np.ndarray:
            from .base import LazyTorchTensor

            # a fresh lazy tensor every call, or to_eager() memoizes every shard
            eager = LazyTorchTensor.to_eager(self.model_tensors[name]())
            return eager.to(torch.float32).contiguous().numpy()
        return load

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._mtp_fc:
            raise ValueError(f"MTP projection missing its other half: {sorted(self._mtp_fc)}")
        n_parts = self.hparams.get("split_ngram_parts", 0)
        if self._ple_shards and len(self._ple_shards) != n_parts:
            raise ValueError(
                f"got {len(self._ple_shards)} PLE embedding shards, expected {n_parts}"
            )


@ModelBase.register("Qwen4ExpForConditionalGeneration")
@ModelBase.example("Qwen/Qwen3.8-Flash-Next")
class Qwen4ExpVisionModel(Qwen3VLVisionModel):
    """The vision tower is an unmodified Qwen3-VL ViT."""
