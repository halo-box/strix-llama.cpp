# Deferred: Vulkan sparse Flash Attention (Strix Halo)

Written during the 2026-09-15 sync of `halo-box/llama.cpp` into this fork (PR #64).
Upstream's sparse Flash Attention could not be merged mechanically and was deferred.
This file says why, and what integrating it actually requires, so the next person does
not have to rediscover it.

Nothing below is a measurement.

---

## What is not in this fork

`fc82583e6 vulkan: support sparse Flash Attention (ggml-org#28105)` is **not** here.
`ggml/src/ggml-vulkan/` was reverted wholesale to the strix version during the sync.

Deferred with it: `f1e44dcc1 vulkan: workaround NV queuesubmit driver bug
(ggml-org#28830)`.

## Why it could not be merged mechanically

Upstream and strix collide on three things at once:

| | strix | upstream #28105 |
|---|---|---|
| Spec-constant bit | `DYNAMIC_KV = (Flags & 16)` | `USE_SPARSE = (Flags & 16)` |
| FA push constants | `m_row_len, gqa_ratio, split_k_num, output_k_num, partial_output` | `sparse_base` |
| Sparse mechanism | gather/compaction family + DeepSeek-V4 split keyed on bit 31 of `gqa_ratio` | binding-7 index gather, `fa_kv_index()` |

They are rival implementations of overlapping functionality on the same bit.

Taking either side alone breaks the build. Upstream's sparse code merged *cleanly* into
`flash_attn.comp`, `flash_attn_cm1.comp` and `flash_attn_cm2.comp` — strix had not
touched those regions — so keeping strix's `flash_attn_base.glsl` leaves those three
files calling an undefined `fa_kv_index()` / `USE_SPARSE` / `data_sparse`. Reverting the
whole directory is the only self-consistent option without a GPU.

## What integration actually requires

1. Renumber one feature's `Flags` bit. Audit every `string_to_spv` call in
   `vulkan-shaders-gen.cpp` and every `Flags` write in `ggml-vulkan.cpp` — the bit is set
   on the C++ side and read in GLSL, with no shared header to keep the two honest.
2. Merge the two push-constant structs and update the single C++ writer.
   **A mismatched layout does not crash.** The shader reads whatever bytes sit at those
   offsets and emits plausible-looking wrong tensors. Validate numerically, not by
   "it ran".
3. Decide whether `DYNAMIC_KV` and `USE_SPARSE` coexist or one supersedes. strix's gather
   family (`flash_attn_gather{,_dq,_union,_union_dq}.comp`) may already cover upstream's
   case, in which case the answer is to drop one.

## How to settle it

```sh
cmake -B build-vk -DGGML_VULKAN=ON && cmake --build build-vk -j
build-vk/bin/test-backend-ops -b Vulkan0 -o FLASH_ATTN_EXT        # correctness vs CPU ref
build-vk/bin/test-backend-ops -b Vulkan0 perf -o FLASH_ATTN_EXT   # includes the strix probes
```

The strix probes already in `test-backend-ops.cpp` are the ones that matter here:
MALL-spill (32 KV heads, K/V footprint 8x > 32MB MALL), L2-residence (1 KV head x GQA 32,
K/V 5.2MB fits L2), and the KV-cache-layout probe (`{0, 2, 1, 3}`). They were written to
distinguish cache-bandwidth-bound from issue-bound on this part. If upstream's sparse path
is integrated, it must not regress them.

Baseline, measured on a Radeon 8060S (RADV STRIX_HALO) at `dffbb7888`: FLASH_ATTN_EXT
5339/5339 pass on Vulkan0.

---

## Provenance

| Ref | What |
|---|---|
| `fc82583e6` | upstream sparse FA (ggml-org#28105) — deferred |
| `f1e44dcc1` | upstream NV queuesubmit workaround (ggml-org#28830) — deferred |
| `dced43c58` | the sync merge (PR #64) |
| `82aed6017` | #63 merged into that sync |

The PLE n-gram table's disk reader survived the sync as the backend of
`--lazy-mode on-direct`; see PR #64's description for what changed and what was dropped
(O_DIRECT reads and the `--ngram-*` tuning knobs).
