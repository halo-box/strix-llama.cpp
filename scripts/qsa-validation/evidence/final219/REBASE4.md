# REBASE4: PR #276 (QSA sparse attention rewrite) onto master 1dd4c926c, the M5 fix, verification

Status: complete (2026-11-13 06:05). Rebase, M5 fix and verification done; the VGPR baseline is regenerated (`e2bed0b4f`). strix-002 hung hard once at 04:28 and was rebooted by the user (§7.1); the job running then was re-run alone and finished normally. The 8080 server is restored (`/health` ok).

Nothing was pushed and nothing was written to GitHub (GitHub was only read: `git fetch`, `gh issue view 286`). strix-001 was not used. Every build and GPU command on strix-002 ran under `flock ~/gpu.lock`, one at a time; test-backend-ops ran one op per process (census mode: one label filter per process), never the full suite, never `-j`.

## 1. Result

- Laptop branch `qsa-rebased4` (`laptop:~/dev/qsa-rebased4`), head **`e2bed0b4f`**, **97 commits** over master 1dd4c926c. Master moved during the job: `origin/master` is now **58e69e504** (#258, grammar / chat / jinja recursion-depth limits: 25 files in common/, src/llama-grammar.*, src/llama-vocab.cpp, tests/, tools/server/). The branch is 5 commits behind it; `git merge-tree --write-tree origin/master qsa-rebased4` is clean. The only files both sides touch are `common/common.{h,cpp}`, where the PR adds one line each (`cache_type_k_idx`), far from #258's hunks. Per the REBASE3 precedent no second rebase was done. Not pushed.
- 95 commits = qsa-rebased3 (7e7c75b08) minus its VGPR commit, rebased; then 2 new commits:
  - `f54eb9f9a` qwen4exp : TE rows share one device copy of the strip mask across the QSA layers (the M5 fix, §3);
  - `e2bed0b4f` hip : regenerate the gfx1151 VGPR baseline on the rebased tree (§5; `source_revision` = f54eb9f9a).
- Iggy's 3 commits are first; every rebased commit keeps its author, e-mail, date and message byte for byte (checked pairwise, §2.4). The two new commits: author `Gaetan Puleo <12990773+gaetan-puleo@users.noreply.github.com>`, message ends with `Assisted-by: Claude Opus 5.5` and the `Claude-Session` line. No amend, no reset.
- M5 is fixed: default `llama-server` (4 slots, c262144, f16) main 4720.02 -> **1648.02 MiB**, MTP 1889.02 -> **1633.02 MiB**, exactly the values before 3eb54536f; M5 38/40 (the 2 misses are the MTP c131072 ub8192 cells, +102.65 MiB over PR219, unchanged since PROOF-initial).
- Verification (§4): every check passes and matches REBASE3 / master (ops, FA case by case, ctest 21/21 in each of the 3 builds, PPL identical to 7e7c75b08 on all 6 cells, #272 e2e, strict reserve 0 violations, M1 36/36, M2 4/4); the speed check is 3 PASS + 1 RERUN (f16 d32768, ratio 1.005, a graph the fix does not touch). VGPR re-check on the final head: PASS.

## 2. Rebase

### 2.1 Base and master's new commits
- Old head: qsa-rebased3 `7e7c75b08` (= `origin/qsa-rewrite`), 96 commits on b620b0d9a. Its VGPR commit 7e7c75b08 was dropped and regenerated at the end; the 95 others were rebased with `git rebase --onto origin/master b620b0d9a` (c456e6e5e).
- New base: `origin/master` **1dd4c926c**. b620b0d9a..1dd4c926c: 8 files, +68 / -45.

| master commit | what | touches | decision |
|---|---|---|---|
| #282 (9801f8ce2, 7f7883126), #288 (76d2896a3) | CI / docker lanes | `.github/` only | applies unchanged |
| **#271** 1d5bb5a1c | qwen4exp: maskless block selection with a quantized K/V cache (probe with the cache K/V types, drop the f16 gate in `qwen4exp_use_block_selection`, a three-way init log line) | `src/llama-memory-hybrid-idx.cpp`, `src/models/qwen4exp.cpp` | **deleted with the maskless path** (as #270 in REBASE2): 2 conflicts in Iggy's first two commits, merged there (§2.2); fa1e24cc0's rebased commit deletes the probe, `selected_key_attn` and the gate together with #271's lines. Net: none of #271's lines is in the final tree |
| **#225** ddccb72ce (merge 6b46f7df3) | model: qwen4exp layer inputs for DFlash drafters (`hc_stream_mean`, `t_layer_inp` only for requested layers) | `src/models/qwen4exp.cpp` | **kept**: applies without conflict; the rewrite does not touch that loop. Present in the final tree (`hc_stream_mean` at qwen4exp.cpp:644, its use in the layer loop at :942) |
| **#249** 1451efa00 | Vulkan radix top-k `sg_cnt[BLOCK_SIZE]` | `topk_radix_select.comp` | **kept**, unchanged. The PR does not touch the file; the same shader also backs the TOPK_QSA fusion (`pipeline_topk_radix_qsa`), so the fusion gets the fix too (TOPK_QSA tests and census pass with it, §4.1) |
| **#254** 44e7d2f17 | speculative: draft-mtp resets the hidden-row carry at position 0 | `common/speculative.cpp` | **kept**, unchanged (the PR does not touch common/speculative.cpp) |

What #271 brings and whether the rewrite has it:
- #271's point is that a quantized K/V cache takes block selection (no masked per-cell top-k) on a backend that runs the selected-key op for that type. The rewrite's `qwen4exp_use_block_selection` has no backend or K/V-type gate at all: compact ubatches take block selection on GGML_OP_SPARSE_ATTN for every K/V type on every backend (HIP, Vulkan, CPU), non-compact ones the TE rows. So the behaviour #271 adds is a subset of the rewrite's.
- #271's log line names the cache types: the rewrite logs one `QSA ctx=... K=.. V=.. idx=..` line per context and graph shape (llama-context.cpp), which carries the same information.
- Nothing of #271 is missing. Its probe op (FLASH_ATTN_EXT + `add_top_k` with the cache types) has no meaning after the rewrite, which deletes the maskless FLASH_ATTN_EXT route.

### 2.2 Conflict resolutions (all in the commit that introduces them)
| # | commit (original -> rebased) | file | resolution |
|---|---|---|---|
| 1 | Iggy c5695f302 -> `0a350acd1` (q8_0 K/V for the selected-key kernels) | `llama-memory-hybrid-idx.cpp` constructor | master's #271 structure (probe with f16 K/V = "the kernels exist", probe with the cache types, `kernels && all_typed`, three-way log) plus Iggy's `rocm` check and `selected_key_q8 = selected_key_attn && rocm`; the f16 probe call takes `GGML_TYPE_F16, GGML_TYPE_F16` (#271's signature) |
| | | `qwen4exp.cpp` (comment, `qwen4exp_use_block_selection` and its 2 call sites) | Iggy's side: the `type_k, type_v, selected_key_q8` parameters and the `F16 or (Q8_0 && selected_key_q8)` gate come back (#271 had removed the type parameters); the comment merges both (F16 or Q8_0, probe with the cache types) |
| 2 | Iggy 848ce3bd1 -> `986750c73` (probe with the cache types) | `llama-memory-hybrid-idx.cpp` probe | Iggy's two-size probe (n_q 1 and 513) with #271's signature |
| | | same, constructor | #271's structure (f16 probe + typed probe, three-way log) without `rocm` / `selected_key_q8`, which this commit removes |
| | | `qwen4exp.cpp` comment | merged text (head 256; HIP F16 or Q8_0, Vulkan F16; the probe with the cache types) |
| 3 | fa1e24cc0 -> `b182a5b1f` (QSA compact and shortcut rows on SPARSE_ATTN; deletes the probe and `selected_key_attn`) | `llama-memory-hybrid-idx.cpp`, `qwen4exp.cpp` | the commit's side: the whole merged constructor block, the probe and the old `qwen4exp_use_block_selection` body are deleted, #271's lines with them |

No other commit conflicted (92 of 95 applied cleanly).

### 2.3 Old commit dropped / regenerated
- 7e7c75b08 (VGPR baseline of the b620b0d9a tree) dropped; regenerated as the last commit (§5).

### 2.4 Checks that the rebase changed nothing else
- Pairwise, the 95 rebased commits against the 95 originals (`b620b0d9a..c456e6e5e`): author name, e-mail, date and full message identical for all 95; the sorted multiset of added/removed lines identical for 92. The 3 that differ are exactly the 3 conflicted commits of §2.2.
- Tree level: `git diff c456e6e5e 03d92e062` (old code head -> rebased code head, before the fix) = master's delta b620b0d9a..1dd4c926c **minus #271**: the 4 `.github/` files, `common/speculative.cpp` (+9, #254), `topk_radix_select.comp` (#249), `src/models/qwen4exp.cpp` +19 / -1 (#225, byte-identical hunks). No other file differs; `src/llama-memory-hybrid-idx.cpp` is identical to the old head (all of #271 deleted with the path, the deliberate resolution of §2.2).

## 3. M5: the multi-slot reserve growth

### 3.1 Root cause
- PROOF-final3 bisected the growth to 3eb54536f (rebased `c9bff3e0d` here): `qwen4exp_te_select` reads the visibility of the picked cells from the strip's KQ mask instead of from `expanded`, so the Vulkan TOPK_QSA fusion fires again.
- It did so through a `ggml_view_4d(vis, 1, n_kv, n_query, n_stream, ...)` built **inside `qwen4exp_te_select`, i.e. once per QSA layer**. The KQ mask is a host input (it lives in the CPU buffer). In `ggml_backend_sched_split_graph` every *distinct tensor* that a device node reads from another backend gets its own copy in the split's backend (`tensor_id_copy(hash_id(src), ...)`, `ggml_dup_tensor_layout(src)`, the full extent of the view), and the split's inputs are added to the copied graph at the split's start ("so that they are allocated by ggml-alloc at the start of the split"). So the 12 per-layer views were 12 mask-sized device copies, all live at the start of the one big ROCm0 split, next to the copy of `attn_inp_kq_mask` that the mask cast reads.
- Proof (debug build of the head with a log line in the scheduler at each copy ≥ 16 MiB, `F/m5/srv_dbg_head.log`): in the non-compact reserve graph of the main context, `attn_inp_kq_mask (view) op=VIEW nbytes=256.00 MiB` is copied once per QSA layer (12 GET_ROWS consumers), plus `attn_inp_kq_mask` itself (256 MiB, read by the cast); on the MTP context one view copy + the mask. f16 mask at ub512 = n_ctx × 512 × 2 B = n_ctx × 1 KiB: **12 × 256 MiB = 3072 MiB main, 256 MiB MTP at c262144**, exactly the measured growth (4720.02 − 1648.02, 1889.02 − 1633.02), and 1536 / 128 MiB at c131072.
- Why only multi-slot below ub8192: a single-sequence context never builds the non-compact pass (the compact predicate holds), and at ub8192 the pp graph's peak is larger than the non-compact one.
- The same thing is what Iggy's strip_views comment (models.h) already describes for the strip views of the mask and the bias; the TE gather added a view that bypassed it.

### 3.2 Fix (`f54eb9f9a`)
- `qwen4exp_te_vis_cells` (qwen4exp-te.h) builds the one-element-row view; `build_qsa_top_k` builds it **once per strip and graph** (`te_vis`, keyed by the strip's visibility source, like `strip_views`), and **the mask cast of expanded's ADD reads the same view**, so a single device copy serves the cast and the gather of every layer. The cast itself stays per layer (the fusion requires the CPY single-use). `qwen4exp_te_select` takes the view.
- The view node is expanded into the graph ahead of the expanded chain: otherwise, in the first layer that reads it, the VIEW node sits between the chain's CONT and the cast and the TOPK_QSA pattern (consecutive nodes) does not match. Found by the census: with the shared view but without the early expand, `census TOPK_QSA ...,strip=te` was 0/3 (`F/pre_vk/census_TOPK_QSA.log`); with it 3/3.
- f32 mask (`-fa off` route): the ADD's reshape of the view is shared too (that route made a new reshape per layer, the same per-layer copy). Bias-only route: the bias was already a shared strip view; its cells view is now shared as well.
- test-backend-ops `test_qsa_te_strip` and `test-qsa-te-rows` build the strip the same way (cells view, cast through it, early expand in the test-backend-ops graph).

### 3.3 Before / after (init-only default server, f16, mmproj + MTP, `sched_reserve`)
| build | main ROCm0 / host | MTP ROCm0 / host | non-compact pass main | view copies of the mask (main graph) |
|---|---|---|---|---|
| head before the fix (03d92e062 + debug log) | 4720.02 / 412.03 | 1889.02 / 412.02 | 4720.02 | 12 views + the mask = 13 × 256 MiB |
| shared view only (first fix attempt) | 1648.02 / 412.03 | 1633.02 / 412.02 | 1648.02 | 1 (the cells view, read by cast and gather) |
| final fix f54eb9f9a (+ early expand) | 1648.02 / 412.03 | 1633.02 / 412.02 | 1648.02 | 1 |
| PROOF-initial / e31548e7d (before 3eb54536f) | 1648.02 / 412.03 | 1633.02 / 412.02 | | |

No remaining delta: the reserve is byte-equal to the pre-3eb54536f value. The full M5 rerun is in §4.3.

### 3.4 What 3eb54536f bought is kept
- Vulkan TOPK_QSA fusion: `test-backend-ops census -b Vulkan0 -p TOPK_QSA` 6/6 OK on the final head (3 `strip=te` production-strip cases + 3 base), same as 7e7c75b08 (6/6).
- Production graph on Vulkan: `llama-batched-bench` on Vulkan (`-npl 2,4`, with and without `-kvu`, GGML_VK_DISPATCH_CENSUS_LOG) gives identical dispatch censuses on 7e7c75b08 code and the final head; it did not reach the TE rows on either build (no TOPK_QSA, QSA_EXPAND only: split streams are one sequence each, and with `-kvu` the QSA re-split makes the multi-sequence prefill ubatches compact), so the production TE fusion is shown by the census cases above, not by an e2e run. On HIP the TE rows did run in production: the lab #272 run traced 93 `producer=te` ubatches (2 slots decoding together; 93 in REBASE3's run too), with correct answers and 0 strict-reserve violations
- TE-rows correctness: `test-qsa-te-rows` passes on ROCm0, Vulkan0 and CPU (inside ctest, §4.1), QSA_TE_STRIP 2/2 on ROCm0 and Vulkan0 with `--fail-unsupported`.

## 4. Verification of the final head
Logs on strix-002 under `F` = `~/dev/wt_lab/final5` (`chain5.out` = job log). Worktrees: `~/dev/wt_rebase4` (qsa-rebased4), `~/dev/wt_master_1dd` (master 1dd4c926c), `F/lab` (lab tree). Every job: watchdog `kerr_new=0`; `journalctl -k -b` holds only the 3 boot-time lines.

### 4.1 Builds, ops, ctest, PPL, e2e
| check | expected | result | verdict | log |
|---|---|---|---|---|
| Builds HIP / Vulkan / CPU, fresh dirs, REBASE3 flags, at f54eb9f9a | rc 0, warnings = master | rc 0; HIP 87 lines / 17 unique, Vulkan 77 / 19, CPU 74 / 16 on both PR and master 1dd4c926c; nothing only in one (`h/warncmp.sh`) | PASS | `build/warncmp.txt` |
| ROCm0 SPARSE_ATTN `--fail-unsupported` + census: default, graphs off, forced g1 / g4 / vdirect / vpacked / generic / portable / fixup | 1439/1439 | 1439/1439 in all 9; census 492/492 (forced: 492, 492, 388, 393, 24, 2, 492) | PASS | `verify/ops_hip/summary.txt` |
| ROCm0 SPARSE_ATTN_PLAN, default and plan_global | 185/185 | 185/185 both, census 2/2 | PASS | same |
| ROCm0 QSA ops `--fail-unsupported` | all | QSA_EXPAND 56/56, QSA_SCORE 13/13, QSA_STRIP 6/6, QSA_TE_STRIP 2/2, TOPK_QSA 4/4 | PASS | same |
| Vulkan0 SPARSE_ATTN `--fail-unsupported` + census: default / scalar / portable / vstaged | 772/772 | 772/772 in all 4; census 85/85, 84/84, 81/81, 85/85 | PASS | `verify/ops_vk/summary.txt` |
| Vulkan0 SPARSE_ATTN_PLAN | 92/92 | 92/92, census 1/1 | PASS | same |
| Vulkan0 QSA ops `--fail-unsupported` | all | QSA_EXPAND 56/56, QSA_SCORE 13/13, QSA_STRIP 6/6, QSA_TE_STRIP 2/2, TOPK_QSA 4/4 | PASS | same |
| Vulkan0 fusion census (engagement), one label per process | OK | TOPK_QSA 6/6, QSA_EXPAND 2/2; QSA_SCORE 2/3: the `test_qsa_strip(32256, ...)` case also fires a QSA_EXPAND its expectation does not list. **Same on master 1dd4c926c and on 7e7c75b08** (master's census case, not the PR's) | PASS (TOPK_QSA); QSA_SCORE mismatch pre-existing on master | `verify/census/` |
| FLASH_ATTN_EXT case by case vs master 1dd4c926c (`h/facmp.py`) | identical | HIP 5393 cases both, OK 4050 / not supported 1343, 0 only-in-one, 0 status differences; Vulkan 5393 both, OK 5382 / 11, 0 / 0 / 0; Vulkan TOPK_QSA 4/4 both | PASS | `verify/fa/facmp.txt` |
| ctest `-R 'test-qsa\|test-sparse-attn'` HIP / Vulkan / CPU | 21/21 | 21/21 in each build | PASS | `verify/ctest_summary.txt` |
| ctest whole CPU suite (`-E test-backend-ops`), PR vs master | no new failure | PR 90/93, master 72/75; the same 3 fail on both (test-tokenizers-ggml-vocabs, test-grammar-parser, test-jinja-py), as in REBASE3 | PASS | same |
| HIP PPL c8192 × 2, f16 / q8_0 / q4_0 × ub512 / ub4096, final head vs 7e7c75b08 code (`wt_rebase3/build`) | identical | every pair identical per chunk: f16 ub512 `[1]2.0978 [2]1.9419`, f16 ub4096 `[1]2.1082 [2]1.9485`, q8_0 both `[1]2.1139 [2]1.9530`, q4_0 both `[1]2.1240 [2]1.9523` | PASS | `verify/ppl/summary.txt` |
| #272 e2e (Signal-3.8 + mmproj + MTP, 2 slots unified, prompt cache, NYT image to slot 0 ×4, slot save/restore), PR vs master, f16 and q8_0 | correct, no decode error | all 4 arms: every answer names The New York Times / 1969 / MEN WALK ON MOON; 0 decode errors; save 1235 / restore 1235; the file restore reuses 1072 cached tokens and decodes 4. Restored-text identity with r1 varies on both builds (as REBASE3) | PASS (PR = master) | `t272/`, `verify/t272_summary.txt` |
| lab image + `-np 2` strict reserve (`LLAMA_STRICT_RESERVE=1`, trace, census) | 0 violations | main 0 (34 reallocs), MTP 0 (623 reallocs); 357 trace lines, all `backend=ROCm0`; `LLAMA_QSA_RESPLIT` main 2 / MTP 2, `decode_errors=0`; answers correct | PASS | `t272/srv_lab_f16.log` |

### 4.2 DFlash / #225
- No DFlash drafter model is on strix-002 (`~/models`: Signal-3.8, Qwen3.8 Flash Next Q4_K_XL / IQ4_XS + MTP + mmproj, gemma-4, gpt-oss-20b, north-mini; no DFlash / draft model), so the #225 path (`embeddings_layer_inp` requested by a DFlash drafter) was **not exercised end to end**.
- What was checked: #225's hunks are in the final tree byte for byte (§2.4: the tree diff over the old head contains exactly #225's +19 / -1 in qwen4exp.cpp); the rewrite touches neither `hc_stream_mean` nor the layer loop; the tree builds in all three configurations with master's warnings. Every non-DFlash run (PPL, #272 e2e, M1 / M2 / M5) goes through the changed loop with `embeddings_layer_inp` empty, i.e. no `t_layer_inp` is published, as on master.

### 4.3 Memory: M5 (all 40 rows), M1 HIP, M2
- **M5, all 40 rows** (`F/mem/c12/m5.md`, scorer `h/an4.py m5`, PR219 logs reused from `prove/c12` as in PROOF-final3): **38/40 PASS** (PROOF-final3: 8/40). Every main and MTP value now equals PROOF-initial's new value exactly (q8_0 MTP rows carry the same +0.27 MiB as in PROOF-final3). Default server f16: 1648.02 / 1633.02 MiB (bound 1745.36 / 1674.03); c131072 ub512 and ub1024: 846.52 / 831.52; c261888 ub512 / 1024: 1646.46 / 1631.45, ub4096 1658.52 / 1631.45; `-np 2` split c131072 ub1024: 667.19 / 669.18.
  - The 2 rows that do not pass are **MTP c131072 ub8192** (f16 and q8_0): 2912.12 MiB vs PR219 2809.47 (+102.65). Unchanged since PROOF-initial (same value there, a cell without an instantiated plan bound); the main context of that cell passes; not related to the multi-slot growth (ub8192 never showed it).
- **M1 HIP** (`F/mem/m1/m1.md`, `mem_an.py`): **36/36 PASS**, the per-cell table is byte-identical to PROOF-final3's (every type f16 + 0.00 / + 0.27 MiB). The report rows change only where the fix acts: default server main 4720.02 -> 1648.02, MTP 1889.02 -> 1633.02 (f16 / bf16; q8_0 / q4_0 +0.27); `--parallel 2` c131072 ub1024 main 1230.52 -> 667.19, MTP 669.18 unchanged.
- **M2** (`F/mem/m2/m2.md`, C7 shim, server filling the context, W0-lab12 rerun in the same campaign): **4/4 PASS, diff 0.00 MiB** in every cell: f16 c65536 79721.15, c131072 81704.65; q8_0 c65536 78911.43, c131072 80084.93 MiB; pool high-water equal on both arms. Identical to PROOF-final3 and PROOF-initial.
- So the fix changes the reserve of multi-slot contexts below ub8192 only, back to the pre-3eb54536f values, and nothing else.

### 4.4 Speed of the fix (HIP)
Text llama-bench (C10 corpus binary), `-ngl 99 -fa 1 -lzm on-direct -b 4096 -ub 512`, old = 7e7c75b08 code (`final4/bench/bin_new3`) vs new = f54eb9f9a (`final5/bench/bin_new4`), O N interleaved, 20 s pause between processes, §16 statistic (`F/h/perf5_an.py` over `prove/h/w7/perf_an.py`), θ 0.99. Logs `F/extra/perf/`, `F/perf5_an.out`, `F/perf5_an2.out`.

| KV | cell | old t/s (procs) | new t/s (procs) | new / old [95 % CI] | verdict |
|---|---|---|---|---|---|
| q8_0 | d0 pp4096 | 1273.43 (3) | 1272.34 (3) | 0.9991 [0.9939, 1.0045] | PASS |
| q8_0 | d32768 pp512 | 1138.78 (3) | 1138.76 (3) | 1.0000 [0.9908, 1.0091] | PASS |
| f16 | d0 pp4096 | 1269.25 (3) | 1269.78 (3) | 1.0004 [0.9951, 1.0060] | PASS |
| f16 | d32768 pp512 | 1153.45 (6) | 1159.21 (6) | 1.0050 [0.9873, 1.0292] | RERUN after the 2× step (3 -> 6 processes per arm); not taken to 4× |

- llama-bench is one sequence, so these graphs are compact and do not contain the changed code at all (the fix is inside the non-compact TE branch); the f16 d32768 cell is process-to-process spread around a point estimate of 1.005, and the 4× step was not run. 0 eviction samples, 0 P6(a) samples.
- Multi-sequence prefill, the TE side (`llama-batched-bench -c 65536 -b 4096 -ub 512 -npp 4096,8192 -ntg 16 -npl 2,4`, f16 and q8_0, 2 processes per arm, with and without `-kvu`; `F/extra/bb_hang/`, `F/extra_kvu/bb/`): PP and TG t/s equal within ~1-2 % in 7 of the 8 cells of each pair. The PP 8192 × 4 cell (32832 cells) has outliers on **both** builds (f16 `-kvu`: old 868 and 1441 t/s, new 1447 and 1097; q8_0: old 1449 / 1441, new 1267 / 1440), not a build effect. As noted in §3.4, the QSA re-split makes these ubatches compact, so this does not time the TE rows either; no e2e speed measurement of the TE path exists, which matches the fix's nature (fewer input copies, the same kernels).

## 5. VGPR baseline regeneration
`F/h/vgpr4.sh` (REBASE3's script, worktree changed to `~/dev/wt_rebase4`): git archive of **f54eb9f9a** (the code-final head), pinned image `docker.io/rocm/dev-ubuntu-26.04@sha256:4f5ed1bf6532a4b9920400401b0ad0f706af356dae61e40afefbfd6c8073f0ce`, `harness/w13/vgpr_in_image.sh` unchanged, under `flock ~/gpu.lock`.
- In the image: apt rc 0; `-Werror` configure / build rc 0 with 18 warnings (the same 18 as REBASE2 / REBASE3); `test_vgpr_check.py` 21 tests OK; EXPORT_METRICS build rc 0, 167 `*.lto.s`.
- Checker vs the baseline in the tree (6805faa86's, as committed before 7e7c75b08): PASS, 6824 kernels / 162 TUs, 0 VGPR-spilling, 11 SGPR-spilling all bounded (3 `sgpr_lane_spill`, 8 `sgpr_lane_spill_slower`).
- Regeneration (README python, verbatim): `minimum_kernel_count` 6824 -> 6824, TUs 162 -> 162, copies 7164 -> 7164, no TU count changed; `source_revision` -> f54eb9f9a; `conv2d`, `fattn`, `norm` move in key order with unchanged values. Checker vs the regenerated baseline: PASS. Laptop `test_vgpr_check.py`: OK.
- The regenerated file equals 7e7c75b08's except `source_revision` (c456e6e5e -> f54eb9f9a). Moved by base64, md5 `80c32412715d894d228306e5d4498b73` on both sides; committed as **`e2bed0b4f`** (message modelled on 7e7c75b08's) and fast-forwarded into `~/dev/wt_rebase4`. Logs: `F/vgpr4.out`, `F/vgpr/w/`.

## 6. Optional: MTP gates of #286 on top of #254
**Not run.** It was optional; after the unexplained hard hang of strix-002 at 04:28 (§7) I kept the remaining GPU time to the required checks and did not start the ~75 min MTP campaign (16 server processes). A script is ready: `F/h/mtp5.sh` (final3's `run_mtp.sh` E16/E17 design, `gate+extra`, 4 master 1dd4c926c + 4 new processes per main KV, interleaved, logs to `F/mtp/`) and `F/h/mtp_an5.py` (final3's `mtp_an.py`, arm `m50` renamed `mb`). Both builds now contain #254, so its carry reset applies to both arms. The Q7 acceptance-vs-#219 item and the `-ctkd q8_0` cells of #286 would need their own scripts.

## 7. Not run / open
### 7.1 Incident: hard hang of strix-002 at 04:28
- At 04:27:47 the chain started `bb_f16_1_old` (`llama-batched-bench`, 7e7c75b08 code from `wt_rebase3`, f16, `-c 65536 -npl 2,4`, no `-kvu`), right after `bb_f16_0_old` and `bb_f16_0_new` had finished normally (peak GTT 86.4 GB, MemAvailable ≥ 43 GB). Its log ends during model load (`model has unused tensor per_layer_token_embd.weight ... ignoring`, 0.36 s). The machine stopped answering ping and SSH on both the LAN and tailscale addresses; the user hard-rebooted it at ~04:46.
- `journalctl -b -1` ends at 04:27:58 (an SSH session close): no amdgpu error, no GPU reset, no hung-task / soft-lockup / RCU line, no OOM, no watchdog line; the watchdog had written no STOP. `last -x` shows no shutdown. **Cause unknown.** The hanging process was a PR-free reference build (7e7c75b08 code) in the same load as two runs that had just passed.
- After the reboot: `journalctl -k -b` holds only the 3 boot-time lines; the chain was resumed (`F/h/chain5c.sh`) with a 20 s pause between GPU jobs. Before anything else on the GPU, the same job was re-run alone under `timeout 900` and the watchdog (`F/hang_rerun/bb_f16_1_old.log`): rc 0, 101 s, peak GTT 86.4 GB, MemAvailable ≥ 43.7 GB, numbers equal to round 0. It did not hang again; everything after it (8 batched-bench runs, 2 Vulkan runs, 19 llama-bench processes, the VGPR re-check) finished normally. Archived: `F/stopped_hang/chain5.out`, `F/extra/bb_hang/`.

### 7.2 Not run / open
- DFlash / #225 end to end: no drafter model on strix-002 (§4.2).
- The production TE fusion on Vulkan is proven by the census cases and unit tests, not by an e2e Vulkan run (no workload here produces non-compact ubatches on Vulkan, §3.4).
- Perf: the f16 d32768 llama-bench cell stopped at RERUN after the 2× step (§4.4); the TE-path speed has no e2e measurement.
- Not re-run on this head (unchanged since PROOF-final3 and not affected by the fix or by master's delta): M1-Vulkan / AT-B5 (the Vulkan compute-buffer differences listed there), M6, AT-O, Vulkan decode, the quality gates Q1-Q11, P4.
- M5's remaining 2 rows (MTP c131072 ub8192, +102.65 MiB over PR219) and the QSA_SCORE census expectation that misses a QSA_EXPAND (also on master) are pre-existing.
- The optional #286 MTP rerun (§6).
- Master moved to 58e69e504 during the job (merge clean, not rebased, §1).
- `~/dev/wt_bis` was restored to its bisect state (detached e31548e7d, clean); its `build/` holds the debug llama-server (scheduler copy log) from this job.

## 8. Paths
- strix-002, `F` = `~/dev/wt_lab/final5`: `chain5.out`, `h/` (lib5.sh, build5.sh, lab5.sh, verify5.sh, mem5.sh, extra5.sh, chain5.sh, m5probe.sh, vkq.sh, t272*, facmp.py, warncmp.sh, an4.py, vgpr4.sh), `build/`, `verify/`, `t272/`, `mem/` (c12 = M5, m1, m2), `m5/` (debug probes), `pre_vk/` (pre-commit Vulkan checks), `extra/`, `vgpr/`; `m5dbg.patch` (the scheduler debug log, never committed), `fix1.patch` / `fix2.patch`.
- Worktrees: `~/dev/wt_rebase4` (qsa-rebased4), `~/dev/wt_master_1dd`, `F/lab`, `~/dev/wt_bis` (bisect worktree, back at e31548e7d, clean).
- Laptop: `laptop:~/dev/qsa-rebased4`; bundles in the session scratchpad `r4/`.
