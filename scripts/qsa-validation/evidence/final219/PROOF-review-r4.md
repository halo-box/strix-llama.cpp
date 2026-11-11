# PROOF (review-r4): qsa-rewrite head 4d40d7f6a on strix-002

**Verdict: pass = false.**
- The AMD GPU on strix-002 is still unusable, so no ROCm0 or RADV check could run on head `4d40d7f6a`.
- The lavapipe `test-qsa-te-rows` abort is still open: 7/15 runs abort.

Why nothing could run on the GPU:
- `~/gpu.lock` is still held by the hung Vulkan `llama-perplexity` that the initial Prove started:
  - pid 4041225 (child of `timeout` pid 4041224);
  - state D, wchan `drm_suballoc_insert`, SIGKILL pending;
  - 91,631,312,896 B of GTT held;
  - elapsed 11:32:13 at 23:26:40.
- The same process blocked review-r1, review-r2, review-r3, F3, F4 and F5.
- A bounded wait on the lock (lock only, no GPU action) never got it, from 22:58:27 to 23:26:40 (§1.1).
- No sudo is available, and the rules forbid signalling a process this agent did not start.

What this step proved without the GPU, on the new head (F5, 6 commits on top of review-r3's `e31548e7d`):
- All six build trees build at HEAD. Five have 0 warnings. `build-hip-nofa` has 72, all on lines that blame to the merge-base (§2.1).
- ctest QSA subsets pass: CPU 21/21, CPU + RPC 24/24, lavapipe 21/21. The lavapipe ctest's te-rows run passed by chance (§2.5).
- The CPU op suites pass, twice, with identical outcome hashes: SPARSE_ATTN 1439/1439, PLAN 185/185. All other QSA ops pass on the CPU too.
- On lavapipe, run twice with `--fail-unsupported` and identical hashes: SPARSE_ATTN 763/763 and PLAN 92/92. The census covers 81/81 and 1/1 compiled keys.
- F5's new tests pass on the CPU and lavapipe:
  - `test-qsa-dup-key`;
  - `test-sparse-attn-neginf-v`;
  - `test-qsa-prefer-dense` (lavapipe: dense 0 of 504 cells);
  - the `fake-idx-w` guard for 4 types, and `causal-sync`;
  - the TOPK_QSA census, 6/6 OK, one process per case.
- New finding, §2.6: the lavapipe glibc heap corruption also happens on upstream `ea5285ddf` and on W0.
  - Multi-case TOPK_QSA aborts 4/6 on upstream `ea5285ddf` and 5/6 on W0, with the same `corrupted size vs. prev_size` / `realloc(): invalid next size` signatures.
  - With `GGML_VK_DISABLE_FUSION=1` it still aborts 1/6 on base.
  - So that defect predates the rewrite, and it is not limited to the fused kernel.
  - Whether te-rows' abort comes from the same defect is UNPROVEN.

**Conventions.**
- `R4` = `strix-002:~/dev/wt_lab/prove_r4`, 2.4 MB of logs and harnesses (`nogpu.sh`, `nogpu.out`, `lvp.sh`, `lockwait.log`).
- `R3` = `~/dev/wt_lab/prove_r3`. `F5` = `~/dev/wt_rewrite_logs/F5`.
- ssh login-banner lines were filtered from every output.

## 1. Blocking condition (evidence)

| fact | command | output |
|---|---|---|
| HEAD | `git -C ~/dev/wt_rewrite rev-parse HEAD` | `4d40d7f6af58ef596c4c058d998ab08c5c8c18c2` |
| worktree clean | `git status --short \| wc -l` | `0` at the start and at the end |
| new commits since review-r3 | `git log --oneline e31548e7d..HEAD` | 6: 3eb54536f, 4fc0dd204, 10d67d49c, 6a713c0f3, 233cfd10d, 4d40d7f6a (IMPLEMENTATION.md "F5", :2303-2399) |
| merge-base | `git merge-base HEAD origin/master` | `ea5285ddf`; `git rev-list --count ea5285ddf..HEAD` = 96 |
| remote master | `git ls-remote origin refs/heads/master` | `0c1021e1f…`; local `origin/master` = `62a3f4144` (not fetched: a fetch writes refs in the shared repository). The rewrite is still based on `ea5285ddf`, so no current-master rebuild is needed. A merge-tree check against `0c1021e1f` is **NOT RUN** (objects not local). |
| base / PR arms | review-r3 §1 | base arm `~/dev/wt_lab/prove/base_ea5` (`13a279bfa` of `wt_pr219v2_base` is an ancestor of `ea5285ddf`); PR arm `~/dev/wt_pr219v2` (94e086847). Unchanged. |
| lock at 22:57 | `flock -n ~/gpu.lock true && echo LOCK_FREE \|\| echo LOCK_HELD` | `LOCK_HELD` |
| holder | `lslocks \| grep gpu.lock` | `timeout 4041224 FLOCK WRITE 0 0 0 /home/gaetan/gpu.lock` |
| holder process | `ps -eo pid,user,stat,etime,cmd` | `4041225 gaetan D 11:03:31 …/wt_rewrite/build-vk/bin/llama-perplexity … -ctk bf16 -ctv bf16 --kl-divergence-base …/prove/c1/det_model/x_2.kld` |
| signals | `grep -E "State\|SigPnd\|ShdPnd" /proc/4041225/status; cat /proc/4041225/wchan` | `State: D (disk sleep)`, `SigPnd: 0000000000000100` (SIGKILL), `ShdPnd: 0000000000004000`; wchan `drm_suballoc_insert` |
| at 23:26:40 | `ps -o pid,stat,etime,wchan:25 -p 4041225,4041224` | `4041225 D 11:32:13 drm_suballoc_insert`; `4041224 S 11:32:13 sigsuspend.isra.0`; `LOCK_HELD` |
| GTT | `cat /sys/class/drm/card*/device/mem_info_gtt_used` | `91631312896` at 22:57 and 23:26 (unchanged since PROOF-initial §13) |
| kernel log | `journalctl -k --since -12h \| tail -5` | last amdgpu lines 11:30 and 11:55 (`Couldn't update BO_VA (-12)`) and a test segfault at 14:56; no GPU reset or recovery logged since |
| root | `sudo -n true` | `sudo: a password is required` |
| host memory | `free -g` | 124 total, 88 used, 35 available |

### 1.1 Lock wait
- Command: `timeout 7200 flock ~/gpu.lock -c 'date "+GOT_LOCK %F %T"'`, under `nohup`, started 22:58:27 (`R4/lockwait.log`).
- No `GOT_LOCK` line was written. This agent stopped its own waiter (pids 1553188 / 1553225) at 23:26:40, once the no-GPU work was done. The holder's state did not change during the step.

**Required action, for the owner.** Reboot strix-002, or reset the amdgpu device (this needs root). Then run, in order:
1. `bash ~/dev/wt_rewrite_logs/F5/gpu_rerun.sh`. The criteria are in its comments: b6 nofa before FAIL / after OK; b6 HIP dense count equal to the table's; b5 on ROCm0 default / portable / fixup / generic and on RADV default / scalar / portable; b1 RADV census one case per process, plus te-strip / topk / te-rows on RADV and HIP.
2. `bash ~/dev/wt_rewrite_logs/F4/gpu_rerun.sh`, then the `F3/gpu_rerun.sh`, `final_fix/rerun_gpu.sh` and `prove_fix/rerun.sh` chains (review-r3 §1), retargeted to `4d40d7f6a`.
3. The full Prove chains of PROOF-initial §15, on the owner's reduced matrix.

## 2. Checks that ran (no AMD GPU)
- Source: `R4/nogpu.sh`, output in `R4/nogpu.out`, run 22:58:56 → 23:19:41. The §2.6 diagnostics ran afterwards.
- Every command in it uses the CPU, CPU + RPC or lavapipe. None opens `/dev/kfd` or an AMD DRM node.
- `R4/lvp.sh` is review-r3's harness. It exports:
  - `VK_ICD_FILENAMES=~/dev/wt_lab/w6/lvp43/icd.json` (Mesa 25.3.6 lavapipe);
  - `LD_LIBRARY_PATH=…/lvp43/usr/lib64`;
  - `GGML_VK_VISIBLE_DEVICES=0`;
  - `VK_LOADER_LAYERS_DISABLE=~implicit~`;
  - `NODEVICE_SELECT=1`.
- It then runs `exec timeout 2400 "$@"`. The logs print `ggml_vulkan: 0 = llvmpipe (LLVM 21.1.8, 256 bits) … warp size: 8 | shared memory: 32768 | matrix cores: none`.

### 2.1 Builds at HEAD (incremental, `cmake --build <tree> -j 16`, `R4/rebuild_<tree>.log`)
| tree | rc | `warning:` | `error:` | verdict |
|---|---|---|---|---|
| build (HIP, rule flags) | 0 | 0 | 0 | PASS |
| build-vk | 0 | 0 | 0 | PASS |
| build-cpu | 0 | 0 | 0 | PASS |
| build-rpc | 0 | 0 | 0 | PASS |
| build-rpc-hip | 0 | 0 | 0 | PASS |
| build-hip-nofa (F5's tree: HIP rule flags + `-DGGML_CUDA_FA=OFF`) | 0 | 72 | 0 | PASS (no rewrite warning) |

About `build-hip-nofa`:
- F5 built only `test-qsa-prefer-dense` there, so this step built every target for the first time. That is why the host-gcc warnings show up.
- Classification (`grep warning: | sort | uniq -c`):
  - 54 × `-Wdeprecated-enum-enum-conversion` (`clip-graph.h`, `test-backend-ops.cpp:464,467`);
  - 16 × `u8path` deprecated (`common/*`, `tools/server/*`);
  - 1 × `ggml_flash_attn_ext_set_prec` deprecated;
  - 1 × `-Woverloaded-virtual` (`test_top_k::max_err` at `test-backend-ops.cpp:9438`, hidden by `test_top_k_stable_ties` at `:9639`).
- Each of the 27 distinct warning lines, and the hider at `:9639` (commit `0fb53c038`), was `git blame`d. Each blames to a commit that is an ancestor of `ea5285ddf` (`git merge-base --is-ancestor`). So the rewrite introduces 0 of them.

### 2.2 ctest QSA subsets
| build | command | result | log |
|---|---|---|---|
| build-cpu | `ctest -R "test-qsa\|test-sparse-attn" -j1 --output-on-failure` | **21/21**, rc 0 (review-r3: 18; + test-qsa-dup-key, test-sparse-attn-neginf-v, test-qsa-prefer-dense) | `R4/ctest_cpu.log` |
| build-rpc | `ctest -R "test-qsa\|test-sparse-attn\|test-rpc" -j1 --output-on-failure` | **24/24**, rc 0 (includes test-rpc-sparse-attn, test-rpc-multi-server) | `R4/ctest_rpc.log` |
| build-vk on lavapipe | the build-cpu command under `lvp.sh` | **21/21**, rc 0; test-qsa-te-rows was in the run and passed (5.98 s), which is chance, see §2.5 | `R4/lvp_ctest_vk.log` |
| `build-cpu/bin/test-qsa-prefer-dense` | | `CPU (CPU): no sparse-attention dense policy, skipped` / `OK (0 devices with the policy, 0 bad dense picks)` | `R4/cpu_prefer_dense.log` |

### 2.3 CPU oracle op suite (`build-cpu/bin/test-backend-ops test -b CPU -o <OP>`, one op per process)
| op | r1 | r2 | outcome hash (colour-stripped case lines), r1 = r2 | log |
|---|---|---|---|---|
| SPARSE_ATTN | 1439/1439, rc 0, 0 FAIL | 1439/1439, rc 0 | `4d7628ad49018326` (1411 lines), equal, and equal to review-r3 | `R4/cpu_ops_SPARSE_ATTN_r{1,2}.log` |
| SPARSE_ATTN_PLAN | 185/185 | 185/185 | `81bcbb8c39d72337` (185 lines), equal, and equal to review-r3 | `R4/cpu_ops_SPARSE_ATTN_PLAN_r{1,2}.log` |
| GET_ROWS / SET_ROWS | 655/655 / 871/871 | | | `R4/cpu_ops_{GET,SET}_ROWS.log` |
| TOPK_QSA / QSA_TE_STRIP / TOP_K | 4/4 / 2/2 / 587/587 | | | `R4/cpu_ops_*.log` |
| QSA_EXPAND / QSA_SCORE / QSA_STRIP | 56/56 / 13/13 / 6/6 | | | |
| SPARSE_ATTN `-p 'leadneginf\|leadnan\|nanscore'` | 48/48 | | | `R4/cpu_ops_lead.log` |

- Every CPU case is checked against O_CON (`tests/sparse-attn-ref.h`), with sentinels (f805cf9a5).
- `grep -c "n_kv=2097156\|n_kv=4194308"` = 3: the ≥ 2 GiB cases.
- The identical SPARSE_ATTN hash is expected: F5 changes no SPARSE_ATTN case or CPU kernel. The `ggml.h` change in 4d40d7f6a is contract text only (`git show 4d40d7f6a` touches `ggml.h` and the new test only).

### 2.4 Guard and lavapipe runs
Guard on the fake device (`build-cpu/bin/test-qsa-guard <model> <mode> [T]`):

| mode | types | result |
|---|---|---|
| `fake-idx` (F4 bug 5) | iq4_nl, q8_0, f16, bf16, q4_0, q5_1, f32 | 7/7 PASS |
| `fake-idx-w` (F5 bugs 4 + 7: writer check before the scheduler) | q8_0, f16, q4_0, iq4_nl | 4/4 PASS |
| `fake` | | PASS |
| `causal-sync` (F5 bug 3) | | PASS |

- Logs: `R4/cpu_guard_*.log`.
- The real HIP device, the default `--cache-type-k-idx auto` load on HIP and RADV, and MXFP4 on HIP are **NOT RUN**.

Other F5 unit tests on the CPU:
- `test-sparse-attn-neginf-v`: `OK (0 failed checks)`.
- `test-qsa-dup-key`: `OK (0 failed checks)`.

lavapipe vs the CPU oracle, one op per process (`R4/lvp_ops_*.log`):

| op | flags | result | "not supported" lines | census / hash |
|---|---|---|---|---|
| SPARSE_ATTN r1 / r2 | `--fail-unsupported` | **763/763** each, rc 0, 0 FAIL | 676 each | `Vulkan0: 81/81 compiled keys of {sa_attend/,sa_fixup/,sa_generic/,sa_merge,sa_pack/,sa_portable/} launched (336 excluded by a forcing switch): OK`; hash `edc7ff22b662fcae` (1425 lines), r1 = r2 = review-r3 |
| SPARSE_ATTN_PLAN r1 / r2 | `--fail-unsupported` | **92/92** each | 93 each | `1/1 compiled keys of {sa_plan/} launched: OK`; hash `38b1295714c50169`, equal |
| SPARSE_ATTN lead / nanscore | | 27/27 | | |
| QSA_EXPAND / QSA_SCORE / QSA_STRIP | | 56/56, 13/13, 6/6 | | |
| GET_ROWS / SET_ROWS / CPY | | 643/643, 439/439, 307/307 | | |
| TOPK_QSA census, one process per case (the 6 F5 `gpu_rerun.sh` patterns) | `census -b Vulkan0 -p <p>` | **6/6 OK**: `width=1500`, `1500,strip=te`, `1499,strip=te` (n_tps 1), `1999,strip=te` (n_stream 2), `width=1499`, `width=1999` | | `R4/lvp_census_topk_{1..6}.log`. F5 bug 1 fix: the te strips now fuse. |
| TOPK_QSA, all cases in one process, 3 runs | | 1 × 4/4; 2 × abort rc 134 (`corrupted size vs. prev_size …`) | | `R4/lvp_ops_TOPK_QSA_multi_*.log`; pre-existing upstream, §2.6 |
| QSA_TE_STRIP, all cases in one process, 3 runs | | 3 × abort rc 134 (`realloc(): invalid next size` / `corrupted size …`), after the first case prints OK | | `R4/lvp_ops_TE_STRIP_multi_*.log`; the same defect class, §2.6 |

lavapipe unit tests:

| test | result |
|---|---|
| `test-sparse-attn-vk` | `OK: 0 failed checks` |
| `test-sparse-attn-rpc-view` | `27 rows, 0 failed checks: OK` |
| `test-qsa-allvis-rows` | `RESULT PASS (failures 0)` |
| `test-sparse-attn-shape-guard` | `78 malformed nodes checked, 0 failed: OK` |
| `test-sparse-attn-gate-scratch` | `0 failed checks: OK` |
| `test-qsa-reserve-view` | `test-qsa-reserve-view: OK` |
| `test-sparse-attn-neginf-v` (F5) | `OK (0 failed checks)` (default and portable tiers; scalar is not supported on lavapipe, F5 b5) |
| `test-qsa-prefer-dense` (F5) | `Vulkan0 (Vulkan): dense 0 of 504 cells, 0 not runnable; FLASH_ATTN_EXT f16/f16 supported, q8_0/q8_0 supported` / `OK (1 devices with the policy, 0 bad dense picks)` |

**Determinism and kernel paths.**
- Within a process, the suite memcmps two runs of each case (review-r1 §2.3).
- Across processes, the per-case outcome lines are identical (the hashes above).
- A cross-process output hash is not valid, because inputs are drawn fresh in each process (PROOF-initial §3.3).
- On lavapipe, the census proves the portable tier and its siblings ran.
- The cm and scalar tiers, and every HIP tier (G1, G4, fixup, portable, generic, prefer-dense with FA on and off), are **NOT RUN**.

### 2.5 `test-qsa-te-rows` on lavapipe (open, FAIL)
| run set | aborts (rc 134) | logs |
|---|---|---|
| build-vk, both backends (default), 15 runs | **7/15** (runs 1, 2, 4, 6, 13, 14, 15); review-r3 10/15 | `R4/lvp_terows_{1..15}.log` |
| build-vk, filter `Vulkan`, 10 runs | **6/10** (runs 3, 4, 5, 8, 9, 10); review-r3 4/10 | `R4/lvp_terows_onlyvk_*.log` |
| build-cpu, 10 runs | 0/10, all `RESULT PASS (failures 0)` | `R4/cpu_terows_*.log` |

- Signature: `corrupted size vs. prev_size while consolidating`, then `timeout: the monitored command dumped core`.
- Every non-aborting run prints `RESULT PASS (failures 0)`.
- F5 rewrote the te_select graph (3eb54536f) but did not change the abort rate in any material way.
- This is not a TODO-LATER item, so it counts as **FAIL** (see §2.6 for the upstream evidence).

### 2.6 New: the lavapipe heap corruption predates the rewrite
- Method:
  - `test-backend-ops` was built from `git archive ea5285ddf` (the merge-base, upstream master): `cmake -B build-vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-I/home/gaetan/.local/spirv-headers/include -DLLAMA_CURL=OFF`, target `test-backend-ops`, rc 0, 0 `error:` (`R4/base_ea5_vk/{cfg,build}.log`). The source and build tree, 833 MB, were deleted afterwards.
  - The W0 lab build `~/dev/wt_lab/w0lab/build-vk` (HEAD `0e1121792`, `hip : q8_0 direct K/V reads in qsa_prefill`, a #219 commit) was used as is.
- The TOPK_QSA pattern is upstream: `git merge-base --is-ancestor daef7b687 ea5285ddf` is true.

| build | `test-backend-ops test -b Vulkan0 -o TOPK_QSA` (4 cases in one process) under `lvp.sh` | aborts | logs |
|---|---|---|---|
| upstream `ea5285ddf` | 6 runs | **4/6** (`corrupted size vs. prev_size` × 3, `realloc(): invalid next size` × 1) | `R4/base_lvp_TOPK_QSA_*.log` |
| upstream `ea5285ddf`, `GGML_VK_DISABLE_FUSION=1` | 6 runs | **1/6** | `R4/base_lvp_TOPK_QSA_nofuse_*.log` |
| upstream `ea5285ddf`, `-o TOP_K` | 3 runs | 0/3 (586/586) | `R4/base_lvp_TOP_K_*.log` |
| W0 (`w0lab`) | 6 runs | **5/6** | `R4/w0_lvp_TOPK_QSA_*.log` |
| rewrite HEAD | 6 runs | **6/6** | `R4/new_lvp_TOPK_QSA_*.log` |
| rewrite HEAD, `GGML_VK_DISABLE_FUSION=1` | 6 runs | **2/6** | `R4/new_lvp_TOPK_QSA_nofuse_*.log` |

- **Proven.** Upstream master, before any QSA rewrite or #219 code, corrupts the glibc heap on lavapipe in the same multi-case TOPK_QSA suite, with the same signatures. It does so with the fusion disabled as well. The defect is not introduced by the rewrite.
- **Not proven.**
  - That the te-rows abort has the same writer: UNPROVEN, since no Vulkan validation layer is installed (review-r3 §2.5).
  - Whether the rewrite raises the rate (6/6 vs 4/6 and 5/6): UNPROVEN, because 6 draws per arm cannot separate these rates.
- The te-rows abort stays **FAIL** in this proof, because the test is the rewrite's. The upstream reproduction is evidence that the owner may use to re-classify it.

## 3. Code reading of the 6 F5 commits (`git show e31548e7d..HEAD`)
- Diffstat: 20 files, +983 / −63.
- Every commit is by Gaetan Puleo `<12990773+gaetan-puleo@users.noreply.github.com>` and carries the `Claude-Session` trailer (`git log --format="%h %an <%ae> | %s%n%(trailers:only)"`).
- None carries `Co-authored-by: Iggy Jackson <iggy@theiggy.com>` (`git log -1 --format="%an <%ae>" 94e086847`). The touched code is not #219's: `qwen4exp-te.h` dates from 79308ac04, TOPK_QSA is upstream (daef7b687), and the HIP table is the rewrite's AT-W7PERF table (IMPLEMENTATION.md:2389). So the absence of the trailer is correct.

| commit | change (verified in the diff) | observation | GPU proof still needed (NOT RUN) |
|---|---|---|---|
| 3eb54536f | `qwen4exp_te_select(ctx, expanded, vis, width)` gathers visibility from the mask / bias through an element view (`qwen4exp-te.h:46-73`); `qwen4exp.cpp:1708` passes `blk_bias ? query_mask : bias`; Vulkan `graph_optimize` adds `add_pattern_alloc_deps` for `topk_qsa_pattern` (`ggml-vulkan.cpp:21225`) | `nb_stream` = `vis->ne[3] == n_stream ? nb[3] : nb[2]`. That is correct for the 4-D mask (ne[3] = n_stream) and the 3-D bias (ne[3] = 1); with n_stream 1 the stride is unused. The view's nb[1] = the element size, so each GET_ROWS "row" is one element, and an F16 mask gives F32 with −inf kept. The rows set is PR219's `top_k(expanded) ∩ {mask == 0}` by construction. The lavapipe census, 6/6 OK, shows the te strips fuse. Peak-memory cost of the alloc deps: UNPROVEN (F5 note 1). | RADV census and te-rows (`b1_radv_*`), HIP te-rows / te-strip (`b1_hip_*`); a model-level reserve = runtime after the alloc-deps change |
| 4fc0dd204 | `seq_pos_ndup[s]`, kept by `seq_pos_inc` / `seq_pos_dec` with a neighbour test (`llama-kv-cells.h`); `qsa_seq_has_dup_key` returns from it (`prefix.h:62-88`) | Invariant: the count is the sum over positions of (count − 1). Insert adds 1 iff the new entry has a twin; erase removes 1 iff the erased entry had one. `reset` zeroes it, and copies carry the array. `test-qsa-dup-key` (40000 random edits, count = recount, predicate = old walk) passes on the CPU. | the decode-step cost inside a real server: UNPROVEN (F5 note 2) |
| 10d67d49c | `qwen4exp_block_selection_eligible` shared by the graph and the memory; NC0 ratios filtered at construction; `qsa_nc0` / `resplit` need `qsa_nc_causal` and a ratio > 1 dividing top_k; the context forwards causal attention at init and in `set_causal_attn` | Consistent with a22777440's graph-side terms. The mask-shape terms are assumed true, an over-approximation that keeps graph-NC ⇒ NC0. `test-qsa-resplit` (lavapipe and CPU ctest) and `causal-sync` pass. | `-np 2` image server run with strict reserve on HIP |
| 6a713c0f3 | `qsa_idx_types_taken` probes `ggml_set_rows(ctx, a, t->src[0], t->src[1])`; `qsa_guard_indexer_writer` runs before the scheduler in `graph_reserve` (throw) and `process_ubatch` (decode error); `build_qsa_store_k`'s writer is registered | The probe now matches the `ggml_set_rows(ctx, a, b, c)` signature (rows, indices). The writer check uses the scheduler's own rule (`supports_buft && supports_op` over `backend_ptrs`). Its cost is one loop over the registered nodes per ubatch. `fake-idx-w` passes 4/4. | real HIP + default `auto` mapping must not false-positive (a refusal of a working model); MXFP4 on HIP |
| 233cfd10d | `ggml_backend_cuda_sparse_attn_prefer_dense` also needs `ggml_cuda_sa_dense_fa_supported` (a FLASH_ATTN_EXT probe with D 256, 24 / 2 heads, a causal f16 mask) | The probe hard-codes the qwen4exp head counts, as the Vulkan probe does. A model with other head counts would be probed on the wrong shape; today only qwen4exp builds the shortcut. One no_alloc ggml context per dense query, once per layer per graph build. | `b6_nofa_before` FAIL / `_after` OK; `b6_hip_after` dense count = the gfx1151 table's |
| 4d40d7f6a | `ggml.h:2669-2676` contract text only | Text matches the CPU op at `sparse-attn.cpp:414-420` (p = 0, 0 · v still added). `test-sparse-attn-neginf-v` passes on the CPU and lavapipe (default, portable). | b5 on G1, G4 WMMA, fixup, generic, Vulkan cm / scalar |

## 4. Scorecard on head 4d40d7f6a (task items 1-8)

| # | check | verdict | source |
|---|---|---|---|
| 0 | builds HIP / Vulkan / CPU / RPC / RPC + HIP / HIP no-FA at HEAD; 0 rewrite-introduced warnings | PASS | §2.1 |
| 1 | test-backend-ops per op on ROCm0 (graphs on / off, each `GGML_CUDA_SA_FORCE`) and RADV (each `GGML_VK_SA_FORCE`), HIP census coverage per kernel key | **NOT RUN** (GPU lock) | §1 |
| 1 | CPU oracle suite: SPARSE_ATTN 1439/1439, PLAN 185/185, GET 655, SET 871, TOPK_QSA 4, QSA_TE_STRIP 2, TOP_K 587, QSA_* 75, lead 48; twice, identical hashes | PASS | §2.3 |
| 1 | lavapipe vs the CPU oracle: SPARSE_ATTN 763/763, PLAN 92/92 (`--fail-unsupported`), census 81/81 + 1/1; TOPK_QSA census 6/6 | PASS (portable tier only) | §2.4 |
| 1 | unsupported shapes asserted unsupported | lavapipe / CPU PASS; ROCm0 / RADV **NOT RUN** | §2.4 |
| 1 | determinism | CPU / lavapipe PASS; HIP / RADV and the model-level logit hash **NOT RUN** | §2.3, §2.4 |
| 1 | ctest QSA | CPU 21/21, RPC 24/24, lavapipe ctest 21/21 PASS; **lavapipe te-rows 7/15 aborts: FAIL** (the same defect class reproduces on upstream `ea5285ddf`, §2.6); HIP / HIP + RPC / RADV **NOT RUN** | §2.2, §2.5, §2.6 |
| 1 | guard | fake device PASS (fake-idx 7/7, fake-idx-w 4/4, fake, causal-sync); real HIP **NOT RUN** | §2.4 |
| 2 | HIP route census (0 masked, 0 CPU FA / sparse nodes for offloaded layers); Vulkan placement census | **NOT RUN** | |
| 2 | reserve = runtime (strict) on every route, including the mid-session reserve (e31548e7d) and the new TOPK_QSA alloc deps (3eb54536f) | **NOT RUN** | |
| 2 | MTP census, gated share ≥ 0.9 | **NOT RUN** | |
| 3 | pinned KLD (QPIN) f16 / q8_0 / q4_0 / bf16 / q8_0K-f16V / f16K-q8_0V vs the floor; f16 bit-identity; unpinned catastrophe; depth; Q1-vk | **NOT RUN**. 3eb54536f changes the TE-route graph (same rows by construction) and must be covered by the bit-identity rerun. | |
| 3 | Q12 coarse q8_0, Q10 control, image p3 seq1 (FAIL in PROOF-initial; no fix since) | **NOT RUN**, open | PROOF-initial §1 |
| 4 | perf: text + random, base vs new vs PR219, f16 / q8_0 / q4_0 / bf16 / mixed, ub 512 / 1024 / 4096, d 0 / 32k / 131k / ~262k (~300k text) / deepest; tg at depth; §16 statistic; stall; compute buffer; peak VRAM | **NOT RUN**. On Vulkan, 3eb54536f turns the TOPK_QSA fusion back on for the TE route. That changes Vulkan perf and memory and must be measured. | |
| 5 | images A / B / C (f16, q8_0), `-np 2` split with images, `-ctk q8_0 -ctv f16` with images, image understanding f16 vs q8_0, MTP draft | **NOT RUN** | |
| 6 | Vulkan build | PASS | §2.1 |
| 6 | Vulkan tests on RADV; q8_0 depth run on RADV | **NOT RUN** (lavapipe tests: §2.4) | |
| 7 | every deferred acceptance test | **NOT RUN** on this head (§5) | |
| 8 | every Moved-to-Prove check, W8 and later | **NOT RUN** on this head (§6) | |

**Op-level-only types (owner scope note).**
- q4_1, q5_0, q5_1, iq4_nl and f32 are covered on this head by the CPU oracle (1439/1439) and lavapipe (763/763 `--fail-unsupported`). `test-qsa-supported-types` is in the ctest.
- iq4_nl, q5_1 and f32 pass the indexer guard (`fake-idx`), and iq4_nl passes `fake-idx-w`.
- ROCm0 / RADV op runs and op time are **NOT RUN**. These types are "op-level proven only, no end-to-end run".

## 5. Deferred acceptance tests (item 7) on head 4d40d7f6a
Every test needs ROCm0 or RADV runs, and none could run. No reduced-matrix reduction was applied, because nothing ran. The criteria are those of the task text.

| test | criterion (short) | state |
|---|---|---|
| AT-A (Lab A, round 1 #0) | G1 where it beats min(G4 packed + pack, G4 direct) by > 3 × p90 noise; the HIP proc encodes the fitted rule; e2e f16 ub512 pp512 d131k neutral or better under §16 (half-width ≤ 1 %); the trimmed-layer row is a W4 gate row | **NOT RUN**. The G1 / G4 inner loops changed in c16d2360c (review-r3), so the timings must be retaken. |
| Lab B VB1 / AT-B1 | f16 new / W0 neutral or better under §16; q8_0, bf16, S_fast ≥ 0.95 × new f16 | **NOT RUN** (the q8_0 / q4_0 miss is TODO-LATER #3) |
| VB2 | new f16 op ≤ 1.10 × the in-build sel_cm median | **NOT APPLICABLE as written**: sel_cm was deleted in W13 (740d48097). It needs an owner ruling, and counts as not passing. |
| VB3 | the faster of direct and staged V per type; ≤ 1.15 × f16 or ≤ the HIP C18 ratio | **NOT RUN** |
| VB4 / AT-B4 | f16 sparse ≥ 0.98 × dense per cell; 0 CPU FA / sparse nodes in the q8_0 / q4_0 shortcut; decode at depth ≥ W0 | **NOT RUN**. HIP prefer_dense changed (233cfd10d; unchanged where FA takes the pair, by code reading only). |
| VB5 / AT-B5 | every T ≤ new f16 + 1 MiB; new f16 ≤ W0 Vulkan f16 | **NOT RUN** (the MTP +12 MiB cell is TODO-LATER #4). The TOPK_QSA alloc deps (3eb54536f) may change Vulkan reserve sizes. |
| VB6 | mean paired KLD difference ≤ B_vk 6.4e-4 | **NOT RUN** |
| C-5 | 0 strict-reserve violations in every cell; reserved partial ≥ the §7 formula | **NOT RUN** |
| C-17 | new f16 ≤ PR219 reserve + pool high-water (or the stated delta); partials scale with n_stream | **NOT RUN** |
| C-12 | new reserve ≤ PR219 + the bound; runtime ≤ reserve; answers 2749 / 12749 | **NOT RUN**. 10d67d49c changes which ubatches re-split. |
| C-16 | strip-view census > 0 for both srcs; no-share − share ≥ ½ the PR219 delta | **NOT RUN** |
| D1 | kind-4 count = W0; U16 memcmp identical; OVL ratio-of-ratios CI contains 0 or lies above it | **NOT RUN** (the criterion wording is TODO-LATER #5) |
| D2 / D3 | plan and attend on their K buffer's device in 100 % of prints; guard 0; KLD ≤ R1; CPU perf ≥ 1 − 2 × the W0 spread | **NOT RUN** (CPU perf is TODO-LATER #2) |
| D4 | 0 GPU-placed plan / attend / shortcut-FA nodes under `-nkvo`, probe included; PPL within 3 × SE of C15 | **NOT RUN** |
| D5 | `-fa auto` resolution logged; under `-fa off` / `-nkvo`: census > 0, 0 masked, guard 0; PPL within 3 × SE of PR219 | **NOT RUN** |
| AT-W12-BANDS | bands frozen in §15.0 before scoring; self-replay KLD ≤ 3.7e-7 | The bands are frozen (PROOF-initial §10.5, head-independent). The self-replay on this head is **NOT RUN**. |
| AT-Q10Q6 | calibration valid; Q10 control; Q10 cells ≤ τ; Q6 ≤ τ_T; answers | The PR219-side calibration stands as in PROOF-initial. The gates on this head are **NOT RUN**. The Q10 control was FAIL in PROOF-initial and has had no fix. |
| AT-W7PERF | sparse ≥ 0.98 × dense per (pair, n_q band) or prefer_dense flips; MTP tg P5 ≥ 0.94; `-nkvo` new / PR219 ≥ 0.98 per cell; q8_0 / f16 ≥ 0.94 under `-nkvo`; census | **NOT RUN** |
| AT-B1 ext / AT-B3 vpack / AT-B7 | S_fast ≥ 0.95 × f16 per cell; vpack rule; portable op runs on lavapipe and RADV sg32; supports_op false without a pipeline | **lavapipe part PASS**: PLAN 92/92 and SPARSE_ATTN 763/763 `--fail-unsupported`, 0 failures; `test-sparse-attn-vk` OK. RADV sg32, the AT-B1 cells and vpack are **NOT RUN**. |

## 6. Moved-to-Prove checks, W8 and later (item 8)
All of these need the GPU, so all are **NOT RUN** on `4d40d7f6a`. The criteria are verbatim in IMPLEMENTATION.md at the lines given (`grep -n "Moved to Prove"`: :923, :1052, :1241, plus the W10 / W11 / W13 sections).

| W item | checks | criterion source | state |
|---|---|---|---|
| W8 | AT-Q10Q6 (1)(2)(3), Q10, Q6 | IMPLEMENTATION.md:1052-1066 | NOT RUN |
| W9 | AT-C12 grid and image-slot layout, M5, AT-C17 / M6, AT-C16b, M4 (b) (c), C15 ub1024 + A/B/C, np2 shared-prefix trim on/off, U27 | IMPLEMENTATION.md:1241-1266 | NOT RUN |
| W10 | U6 OP_PROF A/B, rotated op time, Q5, Q1-A / Q1-B, P1-P6, Vulkan in-op H_q quality | IMPLEMENTATION.md:1461-1474 | NOT RUN |
| W11 | Q4 selection agreement, M1 iq4_nl MTP, Vulkan q8_0 indexer for q4_0 KV, GLM5_NEXT (no GGUF: not applicable) | IMPLEMENTATION.md:1586-1601 | NOT RUN (fake-device guard PASS, §2.4) |
| W13 | the full §14-§16.5 campaign, AT-C16b, VGPR after rebase, merge-tree | IMPLEMENTATION.md:1754-1771 | NOT RUN. No rebase was made, so VGPR-after-rebase does not apply. The merge-tree against `0c1021e1f` is NOT RUN. |
| W7 remainder | Q3-B q4_0 ub256, Q11c, Q12 `-ngl 0` and coarse, P2 / P3, NDB, NPIN, P5 + MTP, P4, P6N, D1V, M2 / M3, AT-B4 e2e, AT-B1 cells, U30 on Vulkan | IMPLEMENTATION.md:923-945 | NOT RUN |
| Prove-fix 1 / Final-fix 2 / F3 / F4 reruns | per review-r3 §6 | IMPLEMENTATION.md:1907, :2059, :2164, :2286-2301 | NOT RUN |
| F5 reruns | `b6_nofa_before` FAIL / `b6_nofa_after` OK / `b6_hip_after` OK with the table's dense count; `b5_hip{,_portable,_fixup,_generic}` and `b5_radv{,_scalar,_portable}` OK; RADV TOPK_QSA census 6/6 OK; `b1_radv_te_strip`, `b1_radv_topk`, `b1_radv_te_rows`, `b1_hip_te_rows`, `b1_hip_te_strip` | IMPLEMENTATION.md:2396-2399, `F5/gpu_rerun.sh` | NOT RUN |

## 7. Known issues (deferred: TODO-LATER.md; reported, not counted in pass)
None could be re-measured on `4d40d7f6a`, because of the GPU lock. The last measurements are those of TODO-LATER.md, PROOF-initial §12, review-r1 §6 and review-r2 §7.

| # | item | last measurement | status on 4d40d7f6a |
|---|---|---|---|
| 1 | Q1 direct form, q4_0 / bf16 | q4_0 2.082e-3 > band 1.441e-3; bf16 1.250e-4 > 1.032e-4 | NOT RUN; last FAIL |
| 2 | AT-D2 `-ngl 0` CPU perf | pp512 0.69-0.71 × PR219, tg32 0.89 × | NOT RUN; last FAIL |
| 3 | AT-B1 Vulkan q8_0 / q4_0 < 0.95 × f16 | 0.921 / 0.916 | NOT RUN; last FAIL. The TE-route TOPK_QSA fusion now fires (3eb54536f), which may move these ratios (UNPROVEN). |
| 4 | AT-B5 Vulkan MTP f16 +12 MiB | not re-measured since W7 | NOT RUN |
| 5 | AT-D1 kind-4 wording | new 12, PR219 / W0 0 | NOT RUN; not met as worded |
| 6 | M1 iq4_nl MTP | +0.26 MiB after W11 (PROOF-initial) | NOT RUN; last PASS |
| 7 | `-nkvo` copy route: four d0 cells, bf16 | d0 cells unresolved | NOT RUN; last open |
| 8 | Q4 thin f16-KV margins | +6.3e-6 / +2.0e-6 | NOT RUN; last PASS (thin) |
| 9 | Q7 MTP draft acceptance | LCB −1.87 to −3.83 pp against the gate's ≥ −1 pp (review-r1 §6) | NOT RUN; last FAIL (deferred) |

## 8. Commands run in this step (all on strix-002)
1. State:
   - `uptime; flock -n ~/gpu.lock true && echo LOCK_FREE || echo LOCK_HELD; lslocks | grep gpu.lock; ps -eo pid,user,stat,etime,cmd | grep -E "llama|test-|flock"`;
   - `cat /sys/class/drm/card*/device/mem_info_{gtt,vram}_used; free -g`;
   - `git rev-parse HEAD; git status --short | wc -l; git log --oneline e31548e7d..HEAD; journalctl -k --since -12h | tail -5`;
   - `grep -E "State|SigPnd|ShdPnd" /proc/4041225/status; cat /proc/4041225/wchan; sudo -n true`.
2. Lock waiter, in the background: `timeout 7200 flock ~/gpu.lock -c 'date "+GOT_LOCK %F %T"'` → `R4/lockwait.log`. This agent stopped it at 23:26:40.
3. `R4/nogpu.sh` → `R4/nogpu.out`. It ran:
   - the 6 incremental builds;
   - the CPU and RPC ctest subsets;
   - CPU `test-qsa-prefer-dense`;
   - the CPU SPARSE_ATTN / PLAN suites, twice each;
   - the CPU GET_ROWS, SET_ROWS, TOPK_QSA, QSA_TE_STRIP, TOP_K and QSA_* suites, and the lead cases;
   - `test-qsa-guard` fake-idx × 7, fake-idx-w × 4, fake and causal-sync;
   - `test-sparse-attn-neginf-v` and `test-qsa-dup-key`;
   - the lavapipe SPARSE_ATTN / PLAN suites, twice each, with `--fail-unsupported`;
   - lavapipe lead, QSA_*, GET_ROWS / SET_ROWS / CPY;
   - lavapipe TOPK_QSA / QSA_TE_STRIP multi-case × 3, and the TOPK_QSA census × 6 (one process each);
   - 8 lavapipe unit tests;
   - te-rows on lavapipe 15 times, with the `Vulkan` filter 10 times, and on the CPU 10 times;
   - the lavapipe ctest.
4. The warnings audit: `grep warning: rebuild_build-hip-nofa.log`, `git blame -L l,l --porcelain HEAD -- f` per warning line, and `git merge-base --is-ancestor <c> ea5285ddf`.
5. Hashes: `grep -E "^\s+(SPARSE_ATTN|QSA)" <log> | sed 's/\x1b\[[0-9;]*m//g' | sha256sum`.
6. Base Vulkan build:
   - `git archive ea5285ddf | tar -x -C R4/base_ea5_vk/src`;
   - `cmake -B build-vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-I/home/gaetan/.local/spirv-headers/include -DLLAMA_CURL=OFF`;
   - `cmake --build build-vk -j 16 --target test-backend-ops`.
7. §2.6 runs, each under `R4/lvp.sh`:
   - `test-backend-ops test -b Vulkan0 -o TOPK_QSA` × 6 for base, W0 (`~/dev/wt_lab/w0lab/build-vk`) and HEAD;
   - × 6 with `GGML_VK_DISABLE_FUSION=1` for base and HEAD;
   - `-o TOP_K` × 3 for base.
8. Code reading:
   - `git diff --stat e31548e7d..HEAD`;
   - `git log --format="%h %an <%ae> | %s%n%(trailers:only)" e31548e7d..HEAD`;
   - `git show` of the 6 commits (the source files above).

## 9. Side effects and cleanup
- No AMD GPU command was run, and no process this agent did not start was signalled. pids 4041225 and 4041224 were not touched.
- The only processes stopped were this agent's own lock waiter (1553188, 1553225).
- No systemd unit was touched, strix-001 was not used, and nothing was pushed or posted.
- `~/dev/wt_rewrite`: only incremental builds ran there (plus the first full build of `build-hip-nofa`). There are no source changes; `git status --short` is 0.
- `R4/base_ea5_vk/src` (833 MB: the archive and build) was deleted; its configure and build logs are kept. `R4` is 2.4 MB, and no file over 1 GB remains.
