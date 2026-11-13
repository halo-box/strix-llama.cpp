> **Branches and sources.** This PR branch (`qsa-rewrite`, head e2bed0b4f, 97 commits on master 1dd4c926c) is `qsa-rewrite-evidence`'s code with `scripts/qsa-validation/` removed from history (02ad86516, tree-identical outside that directory), plus the 15 trim commits and the rebase described under "Update: trimmed, rebased and re-proven". Commit SHAs cited below (4d40d7f6a, 1dbee4d9a, …) are those of [`qsa-rewrite-evidence`](https://github.com/halo-box/strix-llama.cpp/tree/qsa-rewrite-evidence), which also carries the lab harnesses and the cited sources in [`scripts/qsa-validation/evidence/final219/`](https://github.com/halo-box/strix-llama.cpp/tree/qsa-rewrite-evidence/scripts/qsa-validation/evidence/final219) (PLAN-v3.md, LEDGER.md, MEASUREMENTS.md, IMPLEMENTATION.md, PROOF-*.md, TODO-LATER.md). Code comments that cite `scripts/qsa-validation/...` refer to that branch.

## Status: ready for review

- **Head** e2bed0b4f, **base** 1dd4c926c (origin/master on 2026-11-13), 97 commits. Master has since gained #258 (grammar / chat / jinja recursion limits); `git merge-tree` against 58e69e504 is clean.
- **Verified on this head** (strix-002, gfx1151 / RADV; REBASE4.md on `qsa-rewrite-evidence`): see "Update 2" below. No regression against master found; the remaining gate misses are listed with their causes under "Open failures" and "Known issues".
- **The open MTP gates are tracked in #286.**
- **Owner decisions (2026-11-12):**
  - AT-O: the f16 V-pack stays **capped** (shipped default); always-pack showed no end-to-end gain and identical KLD.
  - Vulkan `vpack` stays **test-only**; a capped variant is a follow-up in #287.
  - Accepted as known issues (no regression vs master): the P3 cell where master's own q8/f16 is above 1 (gate amended to skip such cells), Vulkan f16 d32768 ub512 pp512 at 0.995, depth 266240 q8/f16 decode 0.9415 (passes the 0.90 floor; follow-up in #287), Q1 direct form for q4_0 / bf16 (the difference form passes).
  - Performance follow-ups: #287.

## Update 2: rebased onto 1dd4c926c, M5 multi-slot reserve fixed, coverage gates (head e2bed0b4f)

### Rebase (REBASE3.md, REBASE4.md)
- Onto b620b0d9a, then 1dd4c926c. Messages, authors and dates byte-identical; Iggy's three commits first; the tree differs from the previous head by master's delta only (minus #271, below).
- **#272** (indexer cells' image 2D position on restore) conflicted with the indexer restore guard; both kept. The rewrite *needs* #272: its compact predicate and block positions read the indexer cells' x/y. New regression test in `test-qsa-idx-type` (save/restore of a slot with an 8×8 image); with #272's line removed it fails 6 checks.
- **#271** (qwen4exp maskless block selection with a quantized K/V cache) is deleted with the maskless path, as #270 was; block selection already runs for any K/V type here.
- **#225** (DFlash layer inputs), **#249** (Vulkan radix top-k subgroup counts), **#254** (MTP hidden-row carry reset) kept unchanged. DFlash not tested end to end (no drafter model on the test machine).

### M5 fix (f54eb9f9a)
Since the TE rows read their visibility from the KQ mask (so the Vulkan TOPK_QSA fusion fires), every QSA layer built its own view of the mask. The mask is a host input, and the scheduler gives each distinct view a device copy: 12 × n_ctx × 1 KiB on the main context, one on MTP. The view is now built once per strip and graph and shared, the cast stays per layer (the fusion needs it single-use).

| default `llama-server` (4 slots, c262144, f16) | main compute | MTP compute |
|---|---|---|
| before | 4720.02 MiB | 1889.02 MiB |
| after | **1648.02 MiB** | **1633.02 MiB** (= before the regression) |

M5 38/40 (was 8/40; the 2 misses are MTP c131072 ub8192 +102.65 MiB, pre-existing). Vulkan TOPK_QSA census 6/6, `test-qsa-te-rows` passes on HIP / Vulkan / CPU. M1 and M2 unchanged.

### Verification on e2bed0b4f
- Builds HIP / Vulkan / CPU: warnings identical to master. VGPR gate in the pinned image: PASS (6,824 kernels, 0 VGPR spills).
- ROCm0 SPARSE_ATTN 1439/1439 on 9 tiers, PLAN 185/185, QSA_EXPAND 56/56, QSA_SCORE, QSA_STRIP, QSA_TE_STRIP, TOPK_QSA pass. Vulkan0 SPARSE_ATTN 772/772 on default / scalar / portable / vstaged, PLAN 92/92, QSA ops pass.
- FLASH_ATTN_EXT identical to master case by case on HIP and Vulkan. ctest 21/21 on HIP, Vulkan, CPU (the full CPU suite fails the same 3 environment tests as master).
- HIP PPL identical to the previous head (f16 1.9419 / 1.9485, q8_0 1.9530, q4_0 1.9523). #272 scenario end to end: same answers as master. Lab image + `-np 2` strict reserve: 0 violations.
- Speed vs the previous head: 0.999–1.005 (the change only touches multi-slot graphs).

### Coverage gates (PROOF-final3.md, head 7e7c75b08; M1/M2/M5 re-run on e2bed0b4f)

| check | result |
|---|---|
| Memory M1 (HIP, 9 KV types, c32k/c131k, ub4096/512, main + MTP) | 36/36: every type ≤ f16 + 0.27 MiB |
| Memory M2 (device high-water vs #219, server filling the context) | 4/4, diff 0.00 MiB |
| Memory M5 (multi-slot reserve) | 38/40 after the fix |
| Memory M6 (f16 c524288) / AT-O always-pack at ub512/1024 up to c393216 | 1660 ≤ 1932 MiB / no reserve difference: V pack stays capped |
| Pinned KLD: Q1-C ub64 (incl. forced G1), Q3 c32k, Q4, Q8 planted defects, Q11a/b/c | 47/47 pass, bit-identical to the pre-trim proof |
| HIP prefill: P1 (q8_0 vs base) / P4 (q4_0, bf16, mixed vs f16, d0) | 1.041× / 32/32 ≥ 0.988× f16 |
| Vulkan decode vs master (f16, q8_0, d0–32k) | 0.997–1.068×; q8_0/f16 ≥ 0.989 |

Known misses: Vulkan compute buffer q8_0/q4_0 +184 MiB over f16 at c131072 ub4096, new f16 +335 MiB over master at c32768 ub4096, MTP +104 / +12 MiB at c131072 (cause not identified). Not re-measured: Q12 (`-nkvo`, `-ngl 0`), P4 at depth and for the generic types, M6 at c1048576.

## Update 1: trimmed, rebased onto 50c2c8cb5 and re-proven end to end (head 49a6d75ab)

### Trim (15 commits, no production behaviour change)
- **Size:** 109 → 96 files; +17,780 / −2,509 → +15,821 / −2,437. Comments −1,173 (internal plan ids and lab references replaced by short explanations), measurement-only code −759 (now a lab patch on `qsa-rewrite-evidence`), dead code −88. `tools/server`, `tools/llama-bench` and `tools/perplexity` are back to master.
- **New environment variables: 18 → 2**, one test switch per backend:
  - `GGML_CUDA_SA_FORCE` = `g1` | `g4` | `vdirect` | `vpacked` | `generic` | `portable` | `fixup` | `reject` | `plan_global` | `rot_dump`
  - `GGML_VK_SA_FORCE` = `scalar` | `portable` | `portable_sg32` | `vpack` | `vdirect` | `vstaged` | `no_cm` | `no_scalar` | `no_portable` | `no_plan` | `max_q=N` | `rot_dump`
  - The kernel dispatch census is a backend debug proc, off until reset; no env var.
- **Coverage kept:** census coverage gate, plan and gate-fusion engagement checks, NMSE band chosen from the kernel that ran, H_q bit-exact on HIP and Vulkan, every forced tier reachable. VGPRs identical kernel by kernel; PPL identical in every chunk.

### Vulkan binding-limit fix (58cae010e)
`test-sparse-attn-vk` failed 2 checks: its hook reported the 1 GiB suballocation block as the binding limit while `supports_op` (correctly) allows up to `maxStorageBufferRange` (~4 GiB on RADV; four op cases past 1 GiB already pass). The hook now reports the limit `supports_op` applies. 0 failed checks; no op-test case changed status.

### Rebase onto 50c2c8cb5 (#263–#270, #277)
- 7 conflicts, all in tests, the VGPR baseline and the deleted `flash_attn_sel*` path; a line-by-line comparison of the rebased change against the original finds no other difference. Iggy's three commits are kept as the first three.
- **#270** (Vulkan QSA quantized K/V) extends the deleted maskless path and is deleted with it. The rewrite covers more types (bf16, f32, quantized × quantized) and keeps no f16 copy of K/V (#270's is 256 MiB at 131k, 2 GiB at 1M). At op level #270 is 1.3–1.8× faster for quantized prefill at 32k; end to end the rewrite is the faster one (see Performance).
- **#264** (Vulkan QSA top-k) is kept: compact ubatches use its select; TOP_K with #264's engagement checks matches master. #263, #265–#269, #277 apply unchanged.
- VGPR baseline regenerated (49a6d75ab): 6,824 kernels, 0 VGPR spills.

### Checked on the rebased head (strix-002, gfx1151 / RADV)
- HIP, Vulkan and CPU builds: no new warning vs master.
- ROCm0 SPARSE_ATTN 1439/1439, also with graphs off and under `vdirect`, `vpacked`, `generic`, `portable`, `fixup`; SPARSE_ATTN_PLAN 185/185; QSA ops and TOPK_QSA pass.
- Vulkan0 SPARSE_ATTN 772/772, also under `scalar`, `portable`, `vstaged`; PLAN 92/92; QSA ops 56/56.
- FLASH_ATTN_EXT identical case by case to master 50c2c8cb5 on both backends (the 250 deleted QSA cases aside).
- ctest 21/21 on HIP, Vulkan and CPU.
- `llama-perplexity` (HIP, c8192): identical in every chunk to the pre-rebase head (f16 1.9419 / 1.9485, q8_0 1.9530, q4_0 1.9523).

### End-to-end validation (head 49a6d75ab; reference master 50c2c8cb5 / #219 W0-lab)

118 jobs, all rc 0; no amdgpu kernel error; peak GTT 99.57 GB (master's f16 image server).

| check | result |
|---|---|
| Pinned KLD vs #219 (Q1-A ub4096, 6 KV types; Q1-B ub512) | 9/9 within band (f16 +1.37e-4, the others −7.3e-5 to −2.7e-6; band 2.0e-4); bit-identical to pre-trim head 1dbee4d9a |
| Images A/B/C × 6 KV types; image understanding short and at 42k | 18/18 correct, 0 strict-reserve violations, 0 CPU, 0 masked; understanding correct |
| `-np 2` with images, split (f16, q8_0, q8_0K/f16V) and unified (q8_0) | 0 strict-reserve violations (was +21.55 MiB) |
| MTP route census, gated share | 96/96 for draft default / q8_0 / q4_0 (was 56/96) |
| HIP prefill vs master, f16 / q8_0 | 1.00–1.08× (12/12 PASS) / 1.02–2.77× (8/8 PASS); new q8_0 / new f16 ≥ 0.923 |
| HIP decode vs master | f16 0.998–1.000×, q8_0 0.99–1.13× |
| Depth 266240 | f16 prefill 1.013× master; q8_0/f16 prefill 0.958; decode unresolved (q8_0/f16 0.9447 vs 0.94 floor) |
| MTP decode vs master (default draft) | f16 d32k 1.040× [1.002, 1.080] (was 0.898: resolved); q8_0 d32k 1.168× |
| Vulkan prefill vs master | f16 0.998–1.003×; q8_0 1.02–1.12× at d4k, **1.66–2.15× at d32k** (master runs q8_0 at 0.42–0.56 of its own f16 at d32k) |
| AT-O, f16 V-pack always vs capped | prefill 0.9955 [0.9866, 1.0044], decode 1.0013 at d266240; same compute buffer and peak at c131072–c393216 (ub4096); identical KLD → **keep capped** |
| Vulkan `vpack` (test-only arm) | +6–10 % at d4k, +6.7 % at d32k ub2048, −1.8 to −2.5 % at d32k ub512 → **keep test-only** |

Not covered by this run: Q1 at ub64, direct form for q8_0 and the mixed pairs, Q3–Q6, Q8–Q12, P1 vs base, P4 (bf16 / mixed / generic), M1/M2/M5/M6, `-ctkd q8_0` MTP cells, Vulkan memory and decode, depth 300k/400k.

## Problem

QSA (the sparse attention of qwen4exp / Qwen3.8-Flash-Next) was only fast with an f16 KV cache. #219 made q8_0 usable on HIP. The rest stayed open:

- **Quantized KV on master falls off a cliff.**
  - The compute buffer: q8_0 needs 21578.66 MiB ROCm0 at `-c 131072 -b/-ub 4096`, against 4133.44 MiB for f16 (MEASUREMENTS C16).
  - Prefill on real text at 131k: q8_0 runs at 0.126-0.234× f16 (C10, "q8_0 ÷ f16" table, base rows).
- **#219 (q8_0 on HIP) closes most of the q8_0 gap but leaves:**
  - a q8_0 regression at ub 512, depth 0: −3.70 % [−4.02, −2.96] at pp4096 on text (C10). The cause is the n_q ≤ 512 → `qsa_decode` routing (C13);
  - a residual q8_0/f16 gap of 0.896 / 0.916 at d131k pp512 (ub512 / ub4096, C10);
  - no other type: q4_0, bf16 and the mixed pairs stay on the masked route. A q4_0 MTP draft reserves 12568.67 MiB ROCm0 (C2). At ~300k depth, #219's q4_0 prefill is 116.82 t/s (PROOF-initial §6.6);
  - Vulkan q8_0 / bf16 on the masked route: 0.575× / 0.323× of f16 on text at d32k pp2048 ub4096 (C11).
- **Correctness and accounting issues on the existing path** (measured on the #219 tree):
  - **Silent CPU fallback is possible by construction:** `ggml_cuda_flash_attn_ext_supported` declines a maskless op that neither QSA kernel takes (LEDGER L29).
  - **`qsa_decode` deduplicates with an `atomicCAS` race.** The output is not run-to-run deterministic on rows with duplicate keys (R2-BOP).
  - **The masked route is not run-to-run deterministic on HIP:** KLD 0.010303 / 0.010408 between identical f16 / q8_0 runs (R1-NKVO).
  - **Image (non-compact) ubatches overrun the reserve** by +1937.02 MiB ROCm0 on a default server with images in two slots (R1-C). #219 reserves no non-compact variant. With one forced by a lab switch, the `-fit` probe sits below the real reserve by −940.54 MiB ROCm0 at unified c131072 ub1024 and −480.88 MiB at `-np 2` c262144 ub1024 (mode 2). The −2539.06 MiB maximum is the mode-1 variant, which reserves a graph larger than any real ubatch (R2-RES, MEASUREMENTS.md:4229-4234, :4425).
  - **Kernel scratch comes from `ctx.pool()`,** outside the reserve (LEDGER L32-L34).

## Design

Plan: PLAN-v3 §1 (decisions D1-D22, each with its evidence); work log: IMPLEMENTATION W0-W13.

**Ops** (`ggml.h`, D1):
- `GGML_OP_SPARSE_ATTN_PLAN` is an integer plan. Its dst is bit-identical on CPU, HIP and Vulkan.
- `GGML_OP_SPARSE_ATTN` is the attend.
- Every scratch area lives in the op's own `get_alloc_size` over-allocation, or in the CPU work buffer, never in `ctx.pool()` (D1, D12). At 4d40d7f6a, `grep -rn "ctx.pool()" ggml/src/ggml-cuda/sparse-attn-*` matches only a comment (`sparse-attn-gate-scratch.h:8`, added by 7a96532ad); no code line calls `ctx.pool()`. The empty result in IMPLEMENTATION W13 test 6 was on 1dbee4d9a (IMPLEMENTATION.md:1729, :1733).

**Producers** (qwen4exp, D5-D7):
- Compact ubatches keep complete-block + tail rows.
- Non-compact ubatches keep #219's token-expanded set `top_k(expanded, 2051)` as a *row producer*, with hidden cells set to −1.
- The masked attention route is deleted past the dense shortcut, for every type, backend, `-fa` and `-nkvo` setting.
- Compact visibility is extended to 2-D ubatches of single-sequence streams (D7).
- Non-compact ubatches longer than 512 tokens are re-split before placement, in the main and MTP contexts (D17).

**Dispatch policy** (backend procs, D3-D4):
- The work split `G ∈ {1, 4}` follows a measured per-backend policy. The HIP table encodes the AT-A fit (PROOF-initial §10.1; 4552df2a4).
- In the dense shortcut (n_kv ≤ 2051), f16/f16 keeps the dense FLASH_ATTN_EXT. Other pairs run the sparse op with all-visible rows, unless the pre-registered 0.98 rule flips a (pair, n_q band) back to dense (AT-W7PERF, AT-B4).
- A dense choice is taken only where the device runs that FLASH_ATTN_EXT (233cfd10d, b7394c04f).

**HIP kernels** (`ggml/src/ggml-cuda/sparse-attn-*`, D8-D10):

| kernel | design |
|---|---|
| G4 attend | #219's `qsa3_attn_kernel` body without the mask path. Split-register K for q8_0 / q4_0 / bf16. V packed when n_kv < 64·n_q, direct split-gather otherwise |
| G1 attend | #219's decode body with LDS-staged V, and #219's decode q8 K loader. Deterministic fixed-order split merge |
| generic tier | f32, q4_1, q5_0, q5_1, iq4_nl |
| portable tier | every non-RDNA3.5 HIP device |
| bf16 | converted to f16 at load |

- The Hadamard rotation of quantized caches is kept and moves into the op (rot_q prologue, rot_o epilogue). That lets the gate fusion match for quantized caches too (D13, W10).

**Vulkan** (D20):
- `SPARSE_ATTN_PLAN` and `SPARSE_ATTN` with three tiers:
  - a KHR-coopmat tier, with K/V types through `flash_attn_dequant.glsl` and bf16 converted at staging;
  - a scalar tier;
  - a portable tier with no subgroup-size or 46 KB shared-memory requirement.
- G = 1.
- supports_op is true only where a pipeline was created.

**Placement** (D15):
- supports_op is static.
- `-ngl 0` and partial `-ngl` pin plan and attend to the CPU.
- Under `-nkvo`, f16 / q8_0 / q4_0 / bf16 take the named host-KV copy route on the GPU that holds the layer's weights: the scheduler copies K/V to it every graph, as #219 does (R1-NKVO, MEASUREMENTS.md:3115: 300/300 FA on ROCm0 with K/V copies). In the rewrite the route is chosen by `qwen4exp_host_kv_copy` (head 4d40d7f6a, src/models/qwen4exp.cpp:1126-1131, :1763). The route is kept on purpose, logged as `route=host-kv-copy` and accepted by the guard (IMPLEMENTATION.md:857-858; PROOF-initial §4.6 D4: every trace line `route=host-kv-copy backend=ROCm0`). Other types are pinned to the CPU.
- A guard after every reserve and every `sched_alloc_graph` turns any QSA node on the wrong device into an init or decode error, never a silent fallback.

**Indexer cache** (D14): `--cache-type-k-idx auto` maps a sub-8-bit `-ctk` to q8_0 on qwen4exp. The flag stays as an override.

**MTP draft** (D16): same op, policies, pins and guard as the trunk. The draft KV type stays independent of `-ctk` (default f16).

**`-fa off`** (D21): QSA layers keep the sparse op. The hybrid-idx cache is created with `v_trans = false`.

## What is deleted

(IMPLEMENTATION W13; `git ls-tree HEAD ggml/src/ggml-cuda/` lists no `qsa-*` file.)

- **HIP:**
  - `qsa-prefill.cu/.cuh`, `qsa-decode.cu/.cuh` and `qsa-decode-wmma.cuh`;
  - the QSA dispatch, the maskless abort and the maskless `_supported` branch in `fattn.cu`;
  - `ggml_cuda_qsa_gate_ok`, and the QSA branch of the FA gate fusion;
  - the env switches `QSA_DIRECT` and `GGML_QSA_RANKSORT` (#219 qsa-prefill.cu:682, :710), and the compile-time constants `GGML_QSA_MAX_KEYS` (common.cuh:109) and `QSA_DECODE_MAX_QUERIES` (qsa-decode.cuh:5).
- **Vulkan:**
  - `flash_attn_sel.comp`, `flash_attn_sel_cm.comp` and `flash_attn_sel_prep.comp`;
  - `ggml_vk_flash_attn_sel*`, and the maskless FA branches of the dispatch and of supports_op;
  - `GGML_VK_SEL_CM`.
  - DSA's `flash_attn_top_k` is kept.
- **llama:**
  - the selected-key probe, `selected_key_attn`, `LLAMA_QSA_SELECTED_KEY` and `QSA_MASKLESS_MAX_KV`;
  - the masked half of `build_attn_qsa` and `llm_graph_input_qsa::maskless`;
  - `LLAMA_QSA_DENSE_SHORTCUT`, and the q/out rotation nodes.
- **Tests:** `QSA_PREFILL`, `QSA_DECODE` and both `_MASKLESS` cases (`QSA_SCORE`, `QSA_STRIP` and `QSA_EXPAND` stay).
- **`ggml.h`:** `ggml_flash_attn_ext_add_top_k` now asserts a mask (74fc732f3). Its callers were checked: deepseek4 passes a mask, and glm5-next passes no top_k.
- **Measurement-only env hooks** moved out of the branch: the hooks were removed in 0da383ddd (`git show --stat`: 4 files, −44 lines), and the patch file `scripts/qsa-validation/patches/06-w13-measurement-hooks.patch` was added in 953b794a0 (`git log --diff-filter=A`).

## Taken from #219

The first three commits are #219's 68da4191d, b60447506 and 94e086847, cherry-picked onto ea5285ddf with unchanged diffs (`git patch-id --stable` equal for each pair); the commit messages were amended with a W0 note, a Claude-Session trailer and a "(cherry picked from ...)" line. The dry-run applied cleanly, and Iggy Jackson stays the author: 509e8ee3b, 21fd1b73d and 0e1121792 (IMPLEMENTATION W0). Twelve later commits carry `Co-authored-by: Iggy Jackson <iggy@theiggy.com>`, among them the ports ed8d3d70e (q8 helper), c0ef4a848 (G4), bb5490b5e (G1) and 4e6328f61 (tests) (`git log --format='%h|%(trailers:key=Co-authored-by,valueonly)|%s' ea5285ddf..HEAD`).

| #219 piece | here | evidence |
|---|---|---|
| `strip_views` + its (input, first, strip size) key | kept as-is | without it, +1120.43 MiB ROCm0 for one deep image ubatch (HIP q8_0, c65536 ub4096); +9014.36 MiB on a default server with two image slots (C19, R1-C) |
| exact q8 → half helpers | one copy, `sparse-attn-q8.cuh` | 4905 + 4320 bitwise checks (C18) |
| decode q8 K loader | kept in G1 | 3.770 vs 3.774 ms against split K (C18) |
| block-w addressing of the direct q8 K read | adapted (split register layout) | −6.7 % kernel time at wiki d131k (C18) |
| `n_q·64 < n_kv` direct-V threshold | default for every type's V | C18 d1; owner decision (always-pack deferred, AT-O) |
| q8_0 / mixed test shapes | ported, with census assertions | LEDGER L11, R2-BOP |

Replaced:
- the decode V gather: staged V is −41 % at ub512 d131k (C18);
- the spilling prefill DIRECT_V q8 loader: 120 B/lane scratch (C18);
- the bool template flags;
- the probe and its override;
- pool scratch;
- the n_q ≤ 512 → decode routing (C13).

## Behaviour changes (user-visible)

- **f16 output is no longer bit-identical to master.**
  - Unpinned KLD against base f16 logits is 0.0096 for the new f16 path, against 0 for #219 (PROOF-initial §5.5).
  - Pinned against #219: +1.369e-4 ± 2.9e-5, inside the band of 2.0e-4 (§5.1).
  - Short image prompts in the dense shortcut give the same opening text as #219 and as master (§7.3).
- **Backends without a SPARSE_ATTN kernel get an init error** (owner decision D22, PLAN-v3 §20.6):
  - This covers CUDA, MUSA, Metal, SYCL, OpenCL, CANN and WebGPU, and Vulkan devices that cannot create any tier. The message names the remedies (`-ngl 0` / `--no-kv-offload`, or a HIP / Vulkan build).
  - The fake-device path is tested (PROOF-initial §4.5). On real non-HIP / non-Vulkan devices it is by code only (UNPROVEN). No such device or compiler exists on strix-002.
- **HIP archs other than RDNA3.5 run the portable tier.** Their speed is UNPROVEN (PLAN-v3 U15).
- **Other init errors:**
  - `-sm tensor` with QSA;
  - `n_ctx_seq` above 2097156 cells (PROOF-initial §4.5).
- **RPC:** a client of this branch rejects an rpc-server older than protocol 8.1, even for models without QSA (a82bdb924; PROOF-review-r2 §3).
- **Indexer cache:** a sub-8-bit `-ctk` gets a q8_0 indexer (D14).
  - KLD smoke against the old mapping: q4_0 0.0282, iq4_nl 0.0254 (PROOF-initial §5.4).
  - Old qwen4exp session files saved with a sub-8-bit `-ctk`, or with `-fa off`, must be re-saved (PLAN-v3 §17).
- **Multi-sequence contexts** reserve a non-compact variant, and long non-compact ubatches are re-split to ≤ 512 tokens (D17). Measured on 1dbee4d9a, the main ROCm0 reserve at ub512 is +138 to +337 MiB above #219 (AT-C12, new − #219); the default server (c262144 ub512) reserves 1648.29 MiB main, within the M5 bound of 1745.36 MiB, against #219's unforced 1323.34 MiB (PROOF-initial §10.3; MEASUREMENTS.md:4349).

## How it was proven

- **Arms.** "new" is the branch. "#219" is W0-lab, i.e. #219's three commits on ea5285ddf plus measurement patches that do nothing unless their env var is set. W0-lab equals W0 in op tests, reserve and PPL (IMPLEMENTATION W0). "base" is ea5285ddf.
- **Statistic** (PLAN-v3 §16): per-process mean of reps 1-4, hierarchical bootstrap with 4000 draws, seed 1. PASS if the CI lower bound ≥ θ, FAIL if the CI upper bound < θ, unresolved otherwise. No "2× processes" reruns were run, so unresolved cells count as not passing (PROOF-initial §1 conventions).
- **Quality.** KLD with QSA rows and MoE expert choices pinned to #219's selections (QPIN), against bands measured on #219's own route changes and frozen before scoring (PLAN-v3 §15.0, W12). Unpinned checks are added on top.
- **Reach.** Every op test run ends with a census of the compiled kernel keys it launched. A route census on the model checks the expected rows, 0 masked and 0 CPU attention nodes (PROOF-initial §3.1, §4.1).
- **Machine.** strix-002 (Radeon 8060S, gfx1151, RADV STRIX_HALO with KHR_coopmat), model Qwen3.8-Flash-Next UD-Q4_K_XL for the bench, op, census and KLD runs; Signal-3.8-Flash-Next-AP-Q4_K_XL with its mmproj and the MTP sidecar for the MTP cells and the image runs (PROOF-initial §6.5, §7). GPU jobs were serialized, and test-backend-ops ran one op per process.

### Correctness

| check | head | result | source |
|---|---|---|---|
| test-backend-ops SPARSE_ATTN, ROCm0, default / graphs off / each `GGML_CUDA_SA_FORCE` | 1dbee4d9a | 1390/1390 each. Census 492/492 compiled keys (default) | PROOF-initial §3.1 |
| SPARSE_ATTN_PLAN ROCm0 / Vulkan0 | 1dbee4d9a | 185/185 / 92/92 | §3.1 |
| SPARSE_ATTN Vulkan0 (RADV), default and 6 forcings, `--fail-unsupported` | 1dbee4d9a | 744/744 each, every "not supported" on the expected list | §3.1 |
| FLASH_ATTN_EXT ROCm0 / Vulkan0 | 1dbee4d9a | 4051/4051 / 5383/5383 | §3.1 |
| model-level determinism, logit hash run twice | 1dbee4d9a | HIP 12/12 identical. Vulkan 6/6 (f16, q8_0, q4_0); bf16 and mixed NOT RUN | §3.3 |
| HIP route census, 6 pairs × ub {4096, 512, 16} + bench | 1dbee4d9a | 24/24 EXPECT PASS, masked 0, CPU 0 | §4.1 |
| CPU oracle suite, run twice, identical outcome hashes | 4d40d7f6a | SPARSE_ATTN 1439/1439, PLAN 185/185 | PROOF-review-r4 §2.3 |
| lavapipe vs CPU, run twice, `--fail-unsupported` | 4d40d7f6a | SPARSE_ATTN 763/763, PLAN 92/92. Census 81/81 + 1/1 | r4 §2.4 |
| ctest QSA subsets | 4d40d7f6a | CPU 21/21, CPU + RPC 24/24, lavapipe 21/21. **te-rows on lavapipe aborts 7/15** | r4 §2.2, §2.5 |
| HIP VGPR gate (`-Werror` build + `gcn-cdna-vgpr-check.py`) | 953b794a0 | `PASS: zero VGPR spills with complete baseline coverage`, 8868 kernels, 162 TUs. Not re-run on later heads | IMPLEMENTATION W13 |

### Performance (head 1dbee4d9a)

| cell | reference | ratio [CI] or t/s | verdict | source |
|---|---|---|---|---|
| P1 text q8_0 ub512 pp4096 d0 | base | 1.0310 [1.0255, 1.0371], θ 0.995 | PASS | PROOF-initial §6.1 |
| P2 text f16, 24 cells (d 0 / 32k / 131k × ub × pp) | #219 | 23 PASS. d0 ub512 pp2048 0.9979 [0.9879, 1.0082] unresolved | 23/24 | §6.2 |
| P2 f16 d32768 ub512 pp512 | #219 | 1095.8 → 1174.8 t/s (1.0721) | PASS | §6.2 |
| P2 f16 d131072 ub4096 pp4096 | #219 | 1846.1 → 1850.3 t/s (1.0023) | PASS | §6.2 |
| P3 q8_0 vs f16, ratio of ratios | #219 | 24/24 PASS (1.012-1.038). Floor new_q8 / new_f16 ≥ 0.90 in 24/24; worst d131072 pp512 0.9226 / 0.9217 / 0.9241 (ub 512 / 1024 / 4096) | PASS | §6.2 |
| text q4_0 | #219 | 24/24 PASS; new_q4_0 / new_f16 ≥ 0.90 in 24/24 | PASS | §6.2 |
| random tokens, q4_0, d131k | #219 | 4.45-8.07×; pp4096 ub4096 221.8 → 1790.8 t/s | PASS 4/4 at d131k (12/12 over d0 / d32k / d131k) | §6.3 |
| random tokens, q8_0 | base (d0, d32k) | 1.022-2.70 | PASS 8/8 | §6.3 |
| f16 d300k pp4096 | #219 / base | 1.0101 [1.0016, 1.0185] / 1.0075 [0.9982, 1.0171] | PASS | §6.6 |
| q4_0 d300k pp4096 / tg32 | #219 | 116.82 → 1565.51 t/s (13.40×) / 7.80 → 20.04 (2.57×) | PASS | §6.6 |
| P4 bf16 vs f16 (θ 0.90) | new f16 | 11 PASS; d131072 ub512 pp512 0.9188 [0.8872, 0.9494] unresolved | 11/12 | §6.4 |
| P4 q8_0K/f16V, f16K/q8_0V | new f16 | 12/12, 12/12 | PASS | §6.4 |
| P5 tg (shortcut, depth, cross-type) | #219 / new f16 | 47/47 | PASS | §6.5 |
| P5 MTP f16, default draft, d32768 | #219 | 0.898 [0.859, 0.949] | **FAIL** | §6.5 |
| P9 image server A / B / C prompt t/s (θ 0.97) | #219 | f16 0.9945 [0.973, 1.019], 1.0089, 1.0165; q8_0 1.0091, 1.0123, 1.0251 | PASS 6/6 | §7.1 |
| AT-W7PERF shortcut, e2e | dense | 10/10 ≥ 0.98 | PASS | §10.7 |
| stall (no sample > 1.5 × its median) | — | 1 of 3232 (1.61×) | **FAIL** | §6.7 |
| AT-D2 `-ngl 0` (CPU) | #219 | f16 pp512 0.700 [0.690, 0.712], tg32 0.804; q8_0 0.706 / 0.923 | **FAIL (deferred)** | §12 #2 |
| Vulkan AT-B1, q8_0 / q4_0 vs new f16 (θ 0.95). **W7 measurement on an earlier head, not 1dbee4d9a; AT-B1 is NOT RUN on 1dbee4d9a (PROOF-initial §8, §13)** | new f16 | worst 0.921 [0.919, 0.924] / 0.916 (pp2048 ub4096 d4096); bf16 12/12 PASS (0.978-0.991) | **FAIL (deferred)** | IMPLEMENTATION.md:900 (W7) |

MTP cells (llama-server, new / #219, §6.5):

| main KV | draft | d16 | d1024 | d32768 |
|---|---|---|---|---|
| f16 | default | 0.996 unresolved | 1.001 PASS | **0.898 [0.859, 0.949] FAIL** |
| f16 | q8_0 | 0.997 unresolved | 1.000 unresolved | 0.954 unresolved |
| q8_0 | default | 1.033 PASS | 1.482 PASS | 1.097 PASS |
| q8_0 | q8_0 | 1.000 unresolved | 1.354 PASS | 1.019 unresolved |

### Quality (head 1dbee4d9a; new − #219 under pins unless stated)

| gate | result | verdict | source |
|---|---|---|---|
| Q1-A ub4096 (band 2.0e-4) | f16 +1.369e-4 ± 2.9e-5, q8_0 −2.7e-6, q4_0 −1.5e-5, bf16 −4.1e-5, q8_0K/f16V −5.0e-6, f16K/q8_0V −2.3e-5 | PASS 6/6 | PROOF-initial §5.1 |
| Q1-B ub512 / Q1-C ub64 | all inside the bands (2.03e-4 / 2.75e-4) | PASS | §5.1 |
| Q1 direct form, KLD(new ‖ #219) | f16, q8_0 and both mixed pass. q4_0 2.690e-3 > 1.441e-3, bf16 1.802e-4 > 1.032e-4; stage 2: 2.082e-3, 1.250e-4 | **FAIL (deferred)** | §5.1 |
| Q3 depth, c32768 (band 1.801e-4 / 3.34e-4) | f16 +1.442e-4, q8_0 +4.5e-6, q4_0 −1.07e-4; ub256 q4_0 −1.95e-4 | PASS 6/6 | §5.2 |
| Q4 selection agreement | every type ≥ its threshold; f16 margin +6.3e-6, f16K + q8_0 idx +2.0e-6 | PASS (thin) | §5.3 |
| Q11c np2 split ≡ np1 | per-chunk KLD lines identical | PASS | §5.3 |
| Q12 pinned `-nkvo` and `-ngl 0` | inside the bands | PASS | §5.3 |
| Q12 coarse unpinned, `-nkvo` vs GPU route | f16 0.014491 ≤ 0.01587; q8_0 0.015717 ± 0.00054 > 0.01563 | **FAIL (q8_0)** | §5.3 |
| unpinned, vs base f16 logits (KLD new / #219) | f16 0.0096 / 0; q8_0 0.0104 / 0.0102; q4_0 0.0242 / 0.0258; bf16 0.0101 / 0.0108 | no catastrophe | §5.5 |
| Q1-vk (Vulkan, B_vk 3.214e-4) | f16 +2.391e-4 ± 2.6e-5, q8_0 +7.5e-7, q4_0 −1.0e-4, bf16 −5.1e-6, q8_0K/f16V −2.1e-5, f16K/q8_0V −1.3e-5 | PASS 6/6 | §5.6 |
| Q5 op-local (NMSE vs the CPU contract, ≤ 2 × #219 p100) | 1432/1432 SA_REPORT lines inside their band | PASS | §5.9 |
| Q6 image prompts | 4/4 ≤ τ_T; every answer names "MEN WALK ON MOON", The New York Times, July 21, 1969 | PASS | §5.8 |
| Q10 non-compact multi-sequence | control 8.485e-3 / 2.569e-2 > floor 3.302e-3 / 1.050e-2; image p3 seq1 3.408e-3 > 1.553e-3; cells 6/6 | **FAIL** | §5.8 |
| Q7 MTP draft acceptance (LCB ≥ −1 pp) | stage 1 mean −1.41 / −1.88 / −0.28 / −1.44 pp (default / q8_0 / q4_0 / bf16), stage 2 +1.28 / +0.93 / +1.17 / +1.12 pp; one-sided 95 % LCB −1.74 to −3.83 | **FAIL (deferred)** | §2.5 |
| image answers A / B / C, 6 types, MTP on | 2749 / 12749 / 26999 in 18/18; strict 0, CPU 0, masked 0 | PASS | §7.1 |

### Memory (head 1dbee4d9a)

| check | result | source |
|---|---|---|
| M1, HIP: f16 / q8_0 / q4_0 / bf16 / iq4_nl × c {32768, 131072} × ub {4096, 512}, main + MTP | 20/20; every type ≤ f16 + 0.27 MiB | PROOF-initial §11 |
| image server, c261888 ub8192, ROCm0 compute | 3170.80 MiB for every type (q8_0 / q4_0 +0.27 MiB) | §6.8 |
| AT-C17, f16 ub4096 reserve, new vs #219 | c524288 1660.42 vs 1932.42 MiB; c1048576 2032.42 vs 2224.42 MiB | §10.3 |
| AT-C12 cells with an M5 bound | all pass, e.g. c262144 ub512 main 1648.29 ≤ 1745.36 MiB | §10.3 |
| AT-C12 cells without a bound, new − #219 main | ub512 +138 to +337 MiB; ub1024 c261888 +220 / +244; ub4096 c261888 −239 / −335; ub8192 0; np2 −80 / −96 | §10.3 |
| M2 device high-water (server filling the context, MTP on) | new = #219 in 4/4 cells | §11 |
| M3 hipMalloc after warm-up | 0 in 16/16 bench runs | §11 |
| AT-C16b strip-view sharing at c524288 | no-share − share = 10547.29 − 2099.29 = 8448 MiB (criterion ≥ 4224) | §10.3 |
| before (master), q8_0 c131072 b/ub 4096 | 21578.66 MiB ROCm0 | MEASUREMENTS C16 |

## Open failures (non-deferred), as of the 49a6d75ab run

| # | failure (first seen) | state on 49a6d75ab | source |
|---|---|---|---|
| 1 | strict reserve, HIP `-np 2` split ub8192: +21.55 MiB (1dbee4d9a) | **PASS**: 0 violations, f16 / q8_0 / q8_0K-f16V split and q8_0 unified | PROOF-final2 §3.2 |
| 2 | strict reserve, Vulkan ub16, f16 / bf16 / f16K-q8_0V: +44,800 B | not fixed; not re-run (no Vulkan memory runs). Likely allocator fragmentation (UNPROVEN) | PROOF-initial §2.2 |
| 3 | MTP gated share 56/96 < 0.9 | **PASS**: 96/96 for all three draft types | PROOF-final2 §3.4 |
| 4 | AT-C18 guard-on-probe 4/8 | fixed (f12eb329d), re-scored PASS on old logs; not re-run | PROOF-after-fix §2.3 |
| 5 | Q10 control and image p3 seq1 | open; floor degenerate (SD 0), owner ruling needed; not re-run | PROOF-initial §5.8 |
| 6 | Q12 coarse q8_0 | open; not re-run | PROOF-initial §5.3 |
| 7 | AT-A, one policy cell (V q8_0, class 135168) | table re-fitted (4552df2a4), 16/16 cells match; e2e cell not run | PROOF-after-fix §2.4 |
| 8 | P5 MTP f16 d32k 0.898 | **PASS**: 1.040× [1.002, 1.080] | PROOF-final2 §3.5 |
| 9 | stall; unresolved perf cells | stall does not recur (0/288 PR samples). After 2×/4× reruns: 6 of 8 HIP cells PASS; MTP f16 d16 1.0026 [0.9611, 1.0457] (token-identical work in both builds) and depth 266240 new q8/f16 decode 0.9415 [0.9281, 0.9541] vs 0.94 (passes PROOF-initial's 0.90 floor) FAIL by the stopping rule; Vulkan f16 7/8 PASS, d32768 ub512 pp512 0.9951 [0.9871, 1.0030] | PROOF-final2 §11 |
| 10 | AT-B2 compares against the deleted `sel_cm` | not applicable as written; owner ruling | PROOF-initial §2.12 |
| 11 | AT-C17 #219 pool high-water | not measured (the fill needs `--rope-scale 2`) | PROOF-initial §2.13 |
| 12 | Vulkan model runs (AT-B4, AT-B5, depth, bf16 / mixed determinism) | q8_0 and f16 prefill to d32k measured (see table); the rest not run (no Vulkan bf16 model run: a Vulkan bf16 perplexity hung the kernel on 2026-11-10) | PROOF-final2 §6 |
| 13 | lavapipe `test-qsa-te-rows` abort 7/15 | same signature on upstream TOPK_QSA; not re-run | PROOF-review-r4 §2.5-§2.6 |
| 14 | U30 masks on Vulkan | not run | PROOF-initial §2 |
| 15 | Q10-style KLD of the AT-C12 image-slot layout | not run | PROOF-initial §2 |
| 16 | MTP decode q8_0 d1024 0.69× master | 4+4 rerun 0.6878 [0.6736, 0.7024]: **not a regression.** The PR's q8_0 takes the sparse route for >127-query batches where master is dense; that flips a near-tie greedy token and master's text then falls into an easy-to-draft repetition loop. At `-ub 64` (dense in both) PR and master are bit-identical (texts, accept counts, top-5 logprobs). On 7 other d1024 prompts: 1.050 [0.950, 1.148], 590 vs 579 accepted. KLD of the sparse route vs dense at c1024 +0.00062 ± 0.00074 | PROOF-final2 §11 |
| 17 | MTP decode new_q8_0 / new_f16 at d16 0.747 | rerun 0.7401 [0.7064, 0.7758], master 0.7465: ratio artifact (f16 and q8_0 produce different greedy texts in both builds; per-verify-step time q8/f16 0.998; 1.009 on 7 other prompts) | PROOF-final2 §11 |
| 18 | P3 d0 ub512 pp4096, q8_0-vs-f16 relative to master | 6+6 rerun 0.9862 [0.9841, 0.9887]: master's own q8/f16 is 1.0162 in this cell (its f16 lags); the PR gains +5.0 % on f16 and +3.5 % on q8_0, absolute q8/f16 1.0022 | PROOF-final2 §11 |
| 19 | Vulkan binding limit in `test-sparse-attn-vk` (27cbb3bc8) | **fixed** (58cae010e), 0 failed checks | "Update" above |

## Known issues (deferred by the owner; TODO-LATER.md)

| # | item | last measurement | source |
|---|---|---|---|
| 1 | Q1 direct form, q4_0 / bf16 | 49a6d75ab: q4_0 2.690e-3 > 1.441e-3; bf16 1.802e-4 > 1.032e-4 (f16 1.375e-4 ≤ 1.403e-4 PASS). Earlier stage 2: q4_0 2.082e-3, bf16 1.250e-4. Against f16 both arms sit at the same distance (q4_0 new 4.999e-3 vs #219 4.995e-3) | PROOF-initial §12; TODO-LATER |
| 2 | `-ngl 0` CPU speed (AT-D2) | f16 pp512 0.700 [0.690, 0.712], tg32 0.804; q8_0 0.706 / 0.923 vs #219. The CPU SPARSE_ATTN op is scalar and converts K/V row by row | PROOF-initial §12 (ratios); TODO-LATER #2 (cause, `ggml-cpu/sparse-attn.cpp:217-270`) |
| 3 | Vulkan q8_0 / q4_0 < 0.95 × f16 (AT-B1) | 49a6d75ab: q8_0 0.886–0.931 × f16, 0/8 cells (master's q8_0 is 0.42–0.56 × its f16 at d32k). Earlier: 0.921 / 0.916 (W7). Op level: q8_0 attend 15.99 ms vs f16 7.19 ms (2.22×), q4_0 15.43 (2.14×); the V-pack arm reaches 9.06 / 8.80 ms but costs 68-264 MiB | IMPLEMENTATION.md:900; PROOF-initial §8 |
| 4 | Vulkan MTP f16 cell +12 MiB over #219 (AT-B5) | not re-measured | TODO-LATER |
| 5 | AT-D1 kind-4 count | new 12 per full graph, #219 0; the criterion is not met as worded; U16 hashes identical | PROOF-initial §10.4 |
| 6 | M1 iq4_nl MTP | +0.26 MiB, cleared | §11 |
| 7 | `-nkvo` copy route | bf16 10 + 2 PASS, 2 unresolved; d0 f16 / q8_0 2 + 1 unresolved | §10.7 |
| 8 | Q4 thin f16-KV margins | +6.3e-6 / +2.0e-6 | §5.3 |
| 9 | stall; unresolved perf cells | stall does not recur (0/288 PR samples). After 2×/4× reruns: 6 of 8 HIP cells PASS; MTP f16 d16 1.0026 [0.9611, 1.0457] (token-identical work in both builds) and depth 266240 new q8/f16 decode 0.9415 [0.9281, 0.9541] vs 0.94 (passes PROOF-initial's 0.90 floor) FAIL by the stopping rule; Vulkan f16 7/8 PASS, d32768 ub512 pp512 0.9951 [0.9871, 1.0030] | PROOF-final2 §11 |

## Test plan

**Done on the rebased head (strix-002, gfx1151 / RADV; every GPU job serialized under a lock):**
- [x] Builds HIP, Vulkan, CPU: no new warning vs master. HIP VGPR gate in the pinned image: PASS, 0 VGPR spills.
- [x] ROCm0 SPARSE_ATTN 1439/1439 (default, graphs off, `vdirect`, `vpacked`, `generic`, `portable`, `fixup`), PLAN 185/185, QSA ops, TOPK_QSA.
- [x] Vulkan0 SPARSE_ATTN 772/772 (default, `scalar`, `portable`, `vstaged`), PLAN 92/92, QSA ops 56/56; FLASH_ATTN_EXT identical to master case by case.
- [x] ctest HIP / Vulkan / CPU 21/21 (Vulkan includes `test-sparse-attn-vk`, 0 failed checks).
- [x] PPL smoke identical to the pre-rebase head; lab (instrumented) and plain builds give identical PPL.
- [x] End-to-end: pinned KLD (6 KV types), images, `-np 2`, MTP census and perf, HIP perf vs master, depth 266240, AT-O both settings, Vulkan prefill vs master (see "Update").

**Done earlier on 4d40d7f6a without an AMD GPU** (PROOF-review-r4 §2): CPU oracle per op twice with identical hashes, lavapipe vs CPU twice, RPC builds and ctest 24/24, guard on a fake device.

**Still open:**
- [x] Reruns with saved texts and 4 + 4 processes for open failures 16–18 and the unresolved cells: no PR defect found (PROOF-final2 §11).
- [x] Owner ruling on the gates that fail as written without a regression (accepted as known issues; MTP ones tracked in #286) (16–18, MTP f16 d16, depth q8/f16 decode vs 0.94, Vulkan d32768 ub512 pp512).
- [ ] `test-qsa-te-rows` on lavapipe (7/15 abort, same signature upstream).
- [ ] Not covered by the end-to-end run: Q1 ub64, Q3–Q6, Q8–Q12, P1 vs base, P4, M1/M2/M5/M6, `-ctkd q8_0` MTP cells, Vulkan memory and decode, Vulkan ub16 strict reserve, depth 300k/400k.

## Risks

- **Coverage gaps.** The end-to-end run covers the owner's reduced matrix only (see Test plan, "Not covered"). Q1 direct form for q4_0 / bf16 and Q7 for q8_0 / q4_0 still fail their gates (deferred).
- **The kernel hang.** It happened in a Vulkan bf16 determinism run, after `Couldn't update BO_VA (-12)`. W7 saw the same kernel error under host memory pressure. That the bf16 Vulkan path caused it is UNPROVEN (PROOF-initial §13).
- **Host OOM.** Two `-d 1040000` bench runs reached 111.5 GB VRAM + GTT, and the OOM killer also killed desktop processes. 400k is the deepest depth measured (§6.6, §13).
- **Numerics.** f16 output changes against master (behaviour changes). Non-compact ubatches keep GPU `top_k` for the token-expanded rows and are not claimed deterministic (PLAN-v3 §2.2, contract item 4, PLAN-v3.md:150).
- **Speed on untested hardware.**
  - CPU `-ngl 0` is about 0.70× of #219 at pp512.
  - Vulkan q8_0 is 0.886–0.931 × f16 on this head (criterion 0.95), though 1.02–2.15× faster than master's q8_0 end to end; q4_0 was not re-run.
  - Non-RDNA3.5 HIP and non-RADV Vulkan devices are UNPROVEN (U15, U28, U41).
  - Backends without a kernel lose QSA (D22).
- **Memory.** Some ub512 reserves are +138 to +337 MiB above #219 (AT-C12, unbounded cells). The Vulkan ub16 strict reserve still fails by 44,800 B.
- **Conflicts.** Rebased on 50c2c8cb5; clean against current master 093c552d7. Open #271 targets the deleted maskless path.

## Rollback

(PLAN-v3 §17)
- **Inside the branch:** W10 (in-op rotation) and W11 (indexer mapping) can each be reverted on their own.
- **Policy changes:** `prefer_dense` and the G policy are backend procs, so a pair or cell can be flipped back to dense with a one-line table change.
- **After merge:** revert the merge commit. There is no runtime switch back to the masked route. qwen4exp session files saved with a sub-8-bit `-ctk` must be re-saved after a revert as well (PLAN-v3.md:833). That `-fa off` files also need it after a revert is UNPROVEN (PLAN-v3 §17 requires it only after W7, D21).

## Coordination

- #219 (iggy) was closed without merging by its author on 2026-11-11. Its three commits stay the first three here, authored by Iggy Jackson, so this PR carries them.
- Rebased onto 50c2c8cb5, so #234 and #263–#270, #277 are in the base (see "Update"). #270's Vulkan quantized K/V path is deleted together with the maskless path it extends.
- #254 (open) resets the MTP hidden-row carry at position 0. The reruns found that `pending_h` in `common/speculative.cpp` (unchanged here) carries the previous request's hidden state into the next request's first MTP row, so draft acceptance depends on request history; #254 addresses it.
- Open PRs touching the deleted path: #271 (qwen4exp maskless block selection with a quantized K/V cache) extends what this PR replaces.
- The Vulkan commit rcfa offered on #219 (93e3aff93) extends `flash_attn_sel*`, which this branch deletes.

Reviewed with Claude (Opus 5.5).

https://claude.ai/code/session_01LzX5gGXu1HutDf5C55PuKJ



