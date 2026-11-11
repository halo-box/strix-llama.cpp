# PROOF (review-r3): qsa-rewrite head e31548e7d on strix-002

**Verdict: pass = false.** The AMD GPU on strix-002 is still unusable, so no GPU check could run on head `e31548e7d`.

Why nothing could run on the GPU:
- `~/gpu.lock` is still held by the hung Vulkan `llama-perplexity` that the initial Prove started:
  - pid 4041225 (child of `timeout` pid 4041224);
  - state D, wchan `drm_suballoc_insert`, SIGKILL pending;
  - 91.6 GB GTT held;
  - elapsed 09:18:45 at 21:13.
- This is the same process that blocked review-r1, review-r2, F3 and F4.
- Every GPU command must take `flock ~/gpu.lock`, so each one would block forever. A bounded wait on the lock (no GPU action) never got it (§1.1).
- No sudo is available, and the rules forbid signalling a process this agent did not start. A signal would not end a D-state process with SIGKILL already pending anyway.
- 32 GiB of host memory is available (`free -g`). The 104 GB model cannot run on the CPU either.

What this step proved without the GPU:
- All five build trees are at HEAD and build with 0 warnings.
- The CPU QSA ctest passes 18/18, and CPU + RPC passes 21/21.
- The CPU op suite passes SPARSE_ATTN 1439/1439 (it was 1402 on review-r2; F4 added 37 cases) and SPARSE_ATTN_PLAN 185/185. Both were run twice, with identical per-case outcome hashes.
- The F4 cases pass: `leadneginf|leadnan|nanscore` 48/48 on the CPU, and 27/27 on lavapipe (21 G4 cases are expected unsupported there).
- The F4 indexer guard (`test-qsa-guard fake-idx`) passes for all 7 types. The `fake` mode passes too.
- On lavapipe:
  - every changed op passes against the CPU oracle, twice, with identical outcome hashes: SPARSE_ATTN 763/763 with `--fail-unsupported`, and PLAN 92/92;
  - the census coverage is 81/81 and 1/1;
  - every unit test passes, including the F4 `test-qsa-reserve-view` part.
- The 7 F4 commits (`574689270..e31548e7d`) were read (§3).

What still fails or did not run:
- The lavapipe `test-qsa-te-rows` heap-corruption abort is still open, and it got worse: **10/15 direct runs aborted** (review-r2: 6/15).
  - This step narrowed it down (§2.5). It aborts only when the Vulkan backend computes: 4/10 with backend filter `Vulkan`, 0/10 with filter `CPU`.
  - It fires during the first case (`text_L8192`).
  - An AddressSanitizer build of the same test ran 10/10 with no report, rc 0.
  - The fusion switches do not remove it: `GGML_VK_QSA_EXPAND=0` gave 8/12 aborts, and `GGML_VK_DISABLE_FUSION=1` gave 8/12.
  - The writer is UNPROVEN.
- Every check that items 1-8 need on ROCm0 or RADV is **NOT RUN**, so none of them passes.

**Conventions.**
- `R3` = `strix-002:~/dev/wt_lab/prove_r3`. It holds this step's logs and harnesses (`nogpu.sh`, `lvp.sh`, `lockwait.log`), 2.3 MB in all.
- `R2` = `~/dev/wt_lab/prove_r2`.
- `F4` = `~/dev/wt_rewrite_logs/F4`.
- All ssh output had its login banner filtered out.

## 1. Blocking condition (evidence)

| fact | command | output |
|---|---|---|
| HEAD | `git -C ~/dev/wt_rewrite rev-parse HEAD` | `e31548e7d03545dcfa67bc9d4498f0153ef8911b` |
| worktree clean | `git status --short \| wc -l` | `0` at the start and at the end |
| new commits since review-r2 | `git log --oneline 574689270..HEAD` | 7 commits: df88ab1bb, c16d2360c, 21f346a6e, f805cf9a5, a5b72e078, 4e08e5b91, e31548e7d (IMPLEMENTATION.md "F4", lines 2174-2301) |
| merge-base | `git merge-base HEAD origin/master` | `ea5285ddf`. `git rev-list --count ea5285ddf..HEAD` = 90. |
| remote master | `git ls-remote origin refs/heads/master` | `0c1021e1f…`. That is newer than the local `origin/master` `62a3f4144`, which was not fetched (a fetch writes refs in the shared repository). The rewrite is still based on `ea5285ddf`, not on a newer master, so no current-master rebuild is needed. A merge-tree check against `0c1021e1f` is **NOT RUN**. |
| base arm | `git merge-base --is-ancestor 13a279bfa ea5285ddf` | true. `~/dev/wt_pr219v2_base` (13a279bfa) is older than the rewrite's base, so the base arm stays `~/dev/wt_lab/prove/base_ea5`. `build/bin/llama-bench` exists in it. The PR219 arm `~/dev/wt_pr219v2` (94e086847) `build/bin/llama-bench` also exists. |
| lock at 20:42 | `flock -n ~/gpu.lock true && echo LOCK_FREE \|\| echo LOCK_HELD` | `LOCK_HELD` |
| holder | `lslocks \| grep gpu.lock` | `timeout 4041224 FLOCK WRITE 0 0 0 /home/gaetan/gpu.lock` |
| holder process | `ps -eo pid,user,stat,etime,cmd` | `4041225 gaetan D 08:47:51 …/wt_rewrite/build-vk/bin/llama-perplexity … -ctk bf16 -ctv bf16 --kl-divergence-base …/prove/c1/det_model/x_2.kld` |
| signals | `grep -E "State\|SigPnd\|ShdPnd" /proc/4041225/status`; `cat /proc/4041225/wchan` | `State: D (disk sleep)`, `SigPnd: 0000000000000100` (SIGKILL), `ShdPnd: 0000000000004000` (SIGTERM); wchan `drm_suballoc_insert` |
| at 21:13 | `ps -o pid,stat,etime,wchan:25 -p 4041225,4041224` | `4041225 D 09:18:45 drm_suballoc_insert`; `4041224 S 09:18:45 sigsuspend.isra.0` |
| GTT held | `cat /sys/class/drm/card*/device/mem_info_gtt_used` | `91631312896` at 20:42 and 21:13, unchanged since PROOF-initial §13 |
| host memory | `free -g` | 124 total, 91 used, 32 available |
| kernel log | `journalctl -k --since "2026-11-10 14:57"` | `-- No entries --` (no GPU reset or recovery is logged) |
| root | `sudo -n true` | `sudo: a password is required` |

### 1.1 Lock wait
- Command: `timeout 5400 flock ~/gpu.lock -c 'date "+GOT_LOCK %F %T"'`. It waits on the lock only and takes no GPU action. It started at `2026-11-10 20:42:49` (`R3/lockwait.log`).
- No `GOT_LOCK` line was written. This agent stopped its own waiter at 21:15:51 (`R3/lockwait.log`: `Terminated … rc=143`, `end 2026-11-10 21:15:51`), after 33 minutes. At that moment `ps` showed `4041225 D 09:21:23 drm_suballoc_insert`, so the holder's state did not change during the step.
- Counting review-r1 (40 min), review-r2 (60 min), F3 and F4, the lock has not been free at any check since the initial Prove started pid 4041225.

**Required action, for the owner.** Reboot strix-002, or reset the amdgpu device (this needs root). Then run, in order:
1. `bash ~/dev/wt_rewrite_logs/F4/gpu_rerun.sh`: F4's before / after GPU checks. The criteria are in its header and in IMPLEMENTATION.md "F4 / Left for the orchestrator".
2. `bash ~/dev/wt_rewrite_logs/F3/gpu_rerun.sh`.
3. `bash ~/dev/wt_lab/final_fix/rerun_gpu.sh`.
4. `bash ~/dev/wt_lab/prove_fix/rerun.sh setup ctest ops np2 vkc mtp c18 census smoke`, retargeted to `e31548e7d`.
5. The full Prove chains of PROOF-initial §15, on the reduced matrix of the owner's scope note.

## 2. Checks that ran (no AMD GPU)

They come from `R3/nogpu.sh` (output `R3/nogpu.out`), which ran from 20:45:33 to 21:05:03. The §2.5 diagnostics ran after it.
- No command in it opens `/dev/kfd` or the AMD DRM node. It uses the CPU, CPU + RPC and lavapipe only.
- The lavapipe harness `R3/lvp.sh` exports:
  - `VK_ICD_FILENAMES=~/dev/wt_lab/w6/lvp43/icd.json` (Mesa 25.3.6 lavapipe);
  - `LD_LIBRARY_PATH=…/lvp43/usr/lib64`;
  - `GGML_VK_VISIBLE_DEVICES=0`;
  - `VK_LOADER_LAYERS_DISABLE=~implicit~`;
  - `NODEVICE_SELECT=1`.
- It then runs `timeout 2400 "$@"`.
- Each log prints `ggml_vulkan: 0 = llvmpipe (LLVM 21.1.8, 256 bits) … warp size: 8 | shared memory: 32768 | matrix cores: none`.

### 2.1 Builds at HEAD (incremental)
- Command: `cmake --build <tree> -j 16`, logged to `R3/rebuild_<tree>.log`.
- The configurations are those of review-r2 §2.1: `build` uses the HIP rule flags; `build-vk` has `GGML_VULKAN=ON`; `build-cpu`, `build-rpc` and `build-rpc-hip` complete the set.

| tree | rc | `warning:` | `error:` | verdict |
|---|---|---|---|---|
| build (HIP) | 0 | 0 | 0 | PASS |
| build-vk | 0 | 0 | 0 | PASS |
| build-cpu | 0 | 0 | 0 | PASS |
| build-rpc | 0 | 0 | 0 | PASS |
| build-rpc-hip | 0 | 0 | 0 | PASS |

The trees were already current from F4's final build (`F4/final/builds.txt`).

### 2.2 ctest QSA subsets
| build | command | result | log |
|---|---|---|---|
| build-cpu | `ctest -R "test-qsa\|test-sparse-attn" -j1 --output-on-failure` | **18/18**, rc 0 | `R3/ctest_cpu.log` |
| build-rpc | `ctest -R "test-qsa\|test-sparse-attn\|test-rpc" -j1 --output-on-failure` | **21/21**, rc 0 | `R3/ctest_rpc.log` |
| build-vk on lavapipe | same as build-cpu, under `lvp.sh` | 18/18 (te-rows happened to pass in this single run) | `R3/lvp_ctest_vk.log` |

### 2.3 CPU oracle op suite (`test-backend-ops -b CPU`, one op per process)
| op | r1 | r2 | outcome hash (colour-stripped case lines), r1 = r2 | log |
|---|---|---|---|---|
| SPARSE_ATTN | 1439/1439, rc 0, 0 FAIL | 1439/1439, rc 0 | `4d7628ad49018326` (1411 lines), equal | `R3/cpu_ops_SPARSE_ATTN_r{1,2}.log` |
| SPARSE_ATTN_PLAN | 185/185 | 185/185 | `81bcbb8c39d72337` (185 lines, the same as review-r2), equal | `R3/cpu_ops_SPARSE_ATTN_PLAN_r{1,2}.log` |
| GET_ROWS,SET_ROWS | 1526/1526, rc 0 | | | `R3/cpu_ops_GET_SET_ROWS.log` |
| SPARSE_ATTN `-p 'leadneginf\|leadnan\|nanscore'` (F4 bugs 1/9 and 8) | 48/48, rc 0 (30 lead lines) | | | `R3/cpu_ops_lead.log` |

- The 1439 cases are review-r2's 1402, plus 30 lead cases, 6 nanscore cases at n_q 64, and 1 q8_0 2^22+4 case (IMPLEMENTATION.md F4 "Regression runs").
- Each case's CPU result is checked against O_CON (`tests/sparse-attn-ref.h`). From f805cf9a5 on, the sentinels are checked too (`test-backend-ops.cpp:1374,1723`).
- 3 cases name `n_kv=2097156` or `n_kv=4194308` (`grep -c`): the ≥ 2 GiB cases, which F4's `check_reach` requires to select past byte INT32_MAX.
- The sentinel and INT32_MAX mutation demonstrations (F4 `before/B6_*`, `before/B7_*`) were not rerun here. Running them means editing the CPU source in `wt_rewrite`. Their result stands as F4 recorded it.

### 2.4 Guard and lavapipe runs
F4 bug 5 guard:
- Command: `build-cpu/bin/test-qsa-guard <model> fake-idx <type>`, for iq4_nl, q8_0, f16, bf16, q4_0, q5_1 and f32.
- All 7 PASS, rc 0.
- iq4_nl gives the expected init error: `the guard fired on the iq4_nl indexer cache`, `QSA layer 3: the indexer K cache (iq4_nl, on FAKE_GPU) is read by GET_ROWS 'node_537', which device FAKE0 does not take …`.
- The other types create the context.
- The `fake` mode also PASSes (`R3/cpu_guard_*.log`).
- The real HIP device (`hip_guard_pass`) is **NOT RUN**.

lavapipe vs the CPU oracle, one op per process (`R3/lvp_ops_*.log`):

| op | flags | result | "not supported" lines | census / hash |
|---|---|---|---|---|
| SPARSE_ATTN r1 | `--fail-unsupported` | **763/763**, rc 0, 0 FAIL | 676 | `Vulkan0: 81/81 compiled keys of {sa_attend/,sa_fixup/,sa_generic/,sa_merge,sa_pack/,sa_portable/} launched (336 excluded by a forcing switch): OK`; hash `edc7ff22b662fcae` (1425 lines) |
| SPARSE_ATTN r2 | same | **763/763**, rc 0 | 676 | hash `edc7ff22b662fcae`, equal |
| SPARSE_ATTN_PLAN r1 / r2 | `--fail-unsupported` | **92/92** each | 93 each | `1/1 compiled keys of {sa_plan/} launched: OK`; hash `38b1295714c50169`, equal (the same as review-r2) |
| SPARSE_ATTN `-p 'leadneginf\|leadnan\|nanscore'` | | **27/27**, rc 0 | 21 (G4 cases, not supported on lavapipe, as F4 states) | |
| QSA_EXPAND / QSA_SCORE / QSA_STRIP | | 56/56, 13/13, 6/6 | | |
| GET_ROWS / SET_ROWS / CPY | | 643/643, 439/439, 307/307 | | |
| TOP_K (diagnostic for §2.5), 3 processes | | 586/586 each, rc 0, no abort | | `R3/lvp_ops_TOP_K_{1,2,3}.log` |

lavapipe unit tests:

| test | result |
|---|---|
| `test-sparse-attn-vk` | `OK: 0 failed checks` |
| `test-sparse-attn-rpc-view` | `27 rows, 0 failed checks: OK` |
| `test-qsa-allvis-rows` | `RESULT PASS (failures 0)` |
| `test-sparse-attn-shape-guard` | `78 malformed nodes checked, 0 failed: OK` |
| `test-sparse-attn-gate-scratch` | `0 failed checks: OK` |
| `test-qsa-reserve-view` (includes F4 bug 4's new part) | `test-qsa-reserve-view: OK`, 0 `FAIL` lines |

- **Counts.** SPARSE_ATTN went from 742 cases (review-r2) to 763: F4 added 18 lead cases and 3 nanscore G1 cases at n_q 64. The "not supported" lines went from 660 to 676. `--fail-unsupported` passed, so each unsupported case is on the expected list.
- **Determinism.**
  - Within one process, the suite runs each case twice and memcmps the two outputs (review-r1 §2.3).
  - Across processes, the per-case outcome lines are identical (the hashes above).
  - A cross-process output hash is not a valid test, because the inputs are drawn fresh in each process (PROOF-initial §3.3).
- **Kernel paths reached.**
  - On lavapipe, the census shows the portable tier and its siblings ran (`sa_portable/`, `sa_generic/`, `sa_pack/`, `sa_fixup/`, `sa_merge`).
  - The cm and scalar tiers cannot run there: F4 records that `GGML_VK_SA_FORCE=scalar` gives `not supported [Vulkan0]`.
  - So F4's cm / scalar / split-K reduce fixes (21f346a6e) and the HIP G4 / G1 / fixup / portable fixes (df88ab1bb, c16d2360c) are proven by code reading only (§3). Their GPU runs are **NOT RUN**.

### 2.5 `test-qsa-te-rows` abort (open, FAIL)

| run set | aborts (rc 134, `corrupted size vs. prev_size while consolidating`) | logs |
|---|---|---|
| lavapipe build-vk, both backends (default), 15 runs | **10/15** (runs 1-3, 5-8, 11, 14, 15) | `R3/lvp_terows_{1..15}.log` |
| build-cpu, 10 runs | 0/10 | `R3/cpu_terows_*.log` |
| lavapipe build-vk, filter arg `CPU` (Vulkan backend loaded, CPU only computes), 10 runs | **0/10** | `R3/lvp_terows_onlycpu_*.log` |
| lavapipe build-vk, filter arg `Vulkan`, 10 runs | **4/10** | `R3/lvp_terows_onlyvk_*.log` |
| `Vulkan`, `GGML_VK_QSA_EXPAND=0`, 12 runs | 8/12 | `R3/lvp_terows_GGML_VK_QSA_EXPAND_*.log` |
| `Vulkan`, `GGML_VK_DISABLE_FUSION=1` (no `topk_radix_qsa`, `ggml-vulkan.cpp:19239`), 12 runs | 8/12 | `R3/lvp_terows_GGML_VK_DISABLE_FUSION_*.log` |
| `Vulkan`, `stdbuf -oL`, 8 runs | 4/8. Every abort printed `backend Vulkan0` and 0 case lines, so it fires inside the first case `text_L8192` | `R3/lvp_terows_lb_*.log` |
| AddressSanitizer build of the test (ROCm clang 24, `-DLLAMA_SANITIZE_ADDRESS=ON -DGGML_SANITIZE_ADDRESS=ON -DBUILD_SHARED_LIBS=OFF`, Vulkan on lavapipe, both backends ran: `backend Vulkan0`, `backend CPU`, `RESULT PASS`), 10 runs | **0/10**, and no `ERROR: AddressSanitizer` in any log | `R3/asan_terows_{1..10}.log`, `R3/asan_cfg.log`, `R3/asan_build.log` |

What this shows:
- The corruption needs the Vulkan backend to compute the te-rows graph on lavapipe.
- Neither QSA fusion path is the cause.
- Instrumented ggml / llama code never wrote out of bounds in 10 ASAN runs.
- The remaining candidates are writes that ASAN does not instrument: Mesa / lavapipe code, or a JIT-compiled shader that writes past a buffer in lavapipe's host-memory device allocation. That second candidate is plausible because the base `top_k` pipelines are created with robustness disabled (`ggml-vulkan.cpp:4522-4534`, the `true` after the alignment). Which writer it is stays UNPROVEN, since no validation layer is installed (`/usr/share/vulkan/explicit_layer.d/` is empty).
- The `te-rows` graph uses only `get_rows`, `cont`, `add`, `top_k`, `step`, `scale_bias`, `mul` and `cast` (`src/models/qwen4exp-te.h:25-60`), plus the PR219 mask route of the test.
- The standalone TOP_K suite did not abort in 3 processes, but its shapes are not te-rows' (width 2051 over n_kv 8448 × 80 rows).
- On RADV, the test is **NOT RUN**.
- This is not a TODO-LATER item, so it counts as **FAIL**.
- The first ASAN configure used gcc. Its link failed, because `/usr/lib64/libasan.so.8.0.0` is missing (`R3/asan_build.log`, first attempt), so the build used ROCm clang instead. The ASAN build directory (`build-vk-asan`, 1.4 GB) was deleted.

## 3. Code reading of the 7 F4 commits (`git show 574689270..e31548e7d`)
- Diffstat: 21 files, +488 / −35.
- Every commit is by Gaetan Puleo `<12990773+gaetan-puleo@users.noreply.github.com>` and carries the `Claude-Session` trailer (`git log --format="%h %an <%ae> | %s%n%(trailers:only)"`).
- df88ab1bb, c16d2360c and 21f346a6e carry `Co-authored-by: Iggy Jackson <iggy@theiggy.com>`. That matches `git log -1 --format="%an <%ae>" 94e086847` = `Iggy Jackson <iggy@theiggy.com>`. The trailer is required, because those commits change kernels adapted from #219 (the file headers they cite).
- The other four commits change only tests, the llama guard and the reserve. They port no #219 code, so the absence of the trailer is correct.

| commit | change (verified in the diff) | observation | GPU proof still needed (NOT RUN) |
|---|---|---|---|
| df88ab1bb | p = 0 for a −inf score in the CPU op (`sparse-attn.cpp:414`), O_CON (`sparse-attn-ref.h:209`), HIP portable (`:175`), G4 fixup (`sparse-attn-attend-g4.cuh:765`), Vulkan portable and scalar, and the cm `m_ref`; contract text `ggml.h:2670-2672` | Each select differs from the old expression only when the running max is −inf, so every row with a finite running max is bit-identical. The CPU SPARSE_ATTN suite's O_CON bit check (1439/1439) is consistent with that. | `hip_lead_before` FAIL / `_after` OK (default, portable, fixup, generic); `radv_lead_after` (cm, scalar) |
| c16d2360c | G4 / G1 take exp2f against `mref = 0` while the step max is −inf (`sparse-attn-attend-g4.cuh:458-462`, `-g1.cuh:318-322`) | With mnew = −inf, every non-NaN score of the step is −inf (fmaxf drops only NaN), so exp2f(−inf) = 0 and a NaN stays NaN. A later finite step gives alpha = exp2f(−inf) = 0, and NaN · 0 keeps l NaN, so the row is 0 (item 5). Reading the code, one select per step replaces the per-element select. Throughput is unmeasured. | `hip_lead_before` FAIL / `_after` OK |
| 21f346a6e | `sa_l_pos` (an integer bit test) in the cm / scalar epilogues and the split-K reduce | The NaN normaliser now gives 0 by a select, not by `acc * 0`. A ±inf **score** is still tested with a float compare (`s == NEG_INF` in the scalar tier, `m_new == NEG_INF` in cm). The commit's own rationale (no `SignedZeroInfNanPreserve` mode) applies to those compares too. Whether RADV folds them is UNPROVEN, and it cannot be settled without RADV. | `radv_nan_before` / `_scalar` FAIL, `radv_lead_after` / `_scalar` OK |
| f805cf9a5 | the sentinel check is opt-in for test_sparse_attn and its subclasses | The other whole-graph cases (test_topk_qsa, test_qsa_score, test_qsa_strip) still do not check sentinels (F4 note 6). | ROCm0 / RADV suites with sentinels |
| a5b72e078 | `SA_ROWS_TAIL`; a q8_0 2^22+4 case; `check_reach` | Present in the CPU suite (3 large cases). On Vulkan they are expected unsupported (binding limit). | `hip_2g_after` |
| 4e08e5b91 | `qsa_guard_indexer` (`llama-context.cpp:1031`, called at `:1190`); qwen4exp registers the indexer SET_ROWS and GET_ROWS | Host caches are skipped, which is correct for `-nkvo` / `-ngl 0`. The guard turns a silent CPU fallback into an init error, so on a real HIP device a false positive would refuse a model that works today: the default `--cache-type-k-idx auto` mapping must not trip it. | **`hip_guard_pass` (q8_0) on ROCm0 is NOT RUN, and neither is the default-type model load on HIP / RADV.** |
| e31548e7d | `LLM_GRAPH_QSA_RESERVE_GRAPH` on every `graph_reserve` graph; `qsa_graph_compact` takes the empty-cache verdict for reserves; `can_reuse` decides through it | Runtime graphs (bit unset) keep `qsa_scalar_visibility`. At init the cache is empty, so the first reserve is unchanged (F4, and `test-qsa-reserve-view` OK here). | reserve = runtime (strict) after a mid-session reserve on a model, with images / seq_cp |

## 4. Scorecard on head e31548e7d (task items 1-8)

| # | check | verdict | source |
|---|---|---|---|
| 0 | builds HIP / Vulkan / CPU / RPC / RPC + HIP at HEAD, 0 warnings | PASS | §2.1 |
| 1 | test-backend-ops per op on ROCm0 (graphs on / off, each `GGML_CUDA_SA_FORCE`) and on RADV (each `GGML_VK_SA_FORCE`), with HIP census coverage per kernel key | **NOT RUN** (GPU lock) | §1 |
| 1 | CPU oracle suite: SPARSE_ATTN 1439/1439, PLAN 185/185, GET/SET_ROWS 1526/1526, lead 48/48, run twice with identical hashes | PASS | §2.3 |
| 1 | lavapipe vs the CPU oracle: SPARSE_ATTN 763/763, PLAN 92/92, lead 27/27, QSA_* and GET/SET_ROWS/CPY; census 81/81 and 1/1 | PASS (portable tier only) | §2.4 |
| 1 | unsupported shapes asserted unsupported (`--fail-unsupported`; shape-guard 78/78) | lavapipe / CPU PASS; ROCm0 / RADV **NOT RUN** | §2.4 |
| 1 | determinism | CPU / lavapipe PASS; HIP / RADV and the model-level logit hash **NOT RUN** | §2.3, §2.4 |
| 1 | ctest QSA | CPU 18/18 PASS; RPC 21/21 PASS; lavapipe **FAIL** (te-rows 10/15 aborts); HIP / HIP + RPC / RADV **NOT RUN** | §2.2, §2.5 |
| 1 | guard (F4 bug 5) | fake device PASS 7/7; real HIP **NOT RUN** | §2.4 |
| 2 | HIP route census (EXPECT rows, 0 masked, 0 CPU FA / sparse nodes for offloaded layers); Vulkan placement census | **NOT RUN** | |
| 2 | reserve = runtime (strict) on every route: 1s-passB (195dadc10), HIP `-np 2` split ub8192, Vulkan ub16 (open since PROOF-initial §4.3), the mid-session reserve (e31548e7d) | **NOT RUN** | |
| 2 | MTP census, gated share ≥ 0.9 | **NOT RUN** | |
| 3 | pinned KLD (QPIN) for f16, q8_0, q4_0, bf16, q8_0K/f16V, f16K/q8_0V vs the floor; f16 bit-identity; unpinned catastrophe check; depth check; Q1-vk | **NOT RUN**. F4's −inf / NaN selects claim bit-identity for every row without a −inf running max (§3), so the f16 bit-identity check must be rerun on this head. | |
| 3 | Q12 coarse q8_0, the Q10 control, image p3 seq1 (FAIL in PROOF-initial; no fix since) | **NOT RUN**, open | PROOF-initial §1 |
| 4 | perf: text + random, base (`prove/base_ea5`) vs new vs PR219, f16 / q8_0 / q4_0 / bf16 / mixed, ub 512 / 1024 / 4096, d 0 / 32k / 131k / ~262k (~300k text) / deepest; tg at depth; §16 statistic; stall; compute buffer; peak VRAM | **NOT RUN**. c16d2360c changes the G4 / G1 inner loop (VGPR sums −48 … +15, F4 `final/spill_after.txt`), so perf needs a rerun. | |
| 5 | images A / B / C (f16, q8_0); `-np 2` split with images; `-ctk q8_0 -ctv f16` with images; image understanding f16 vs q8_0; MTP draft | **NOT RUN** | |
| 6 | Vulkan build | PASS | §2.1 |
| 6 | Vulkan tests on RADV; q8_0 depth run on RADV | **NOT RUN** (lavapipe tests: §2.4) | |
| 7 | every deferred acceptance test | **NOT RUN** on this head (§5) | |
| 8 | every Moved-to-Prove check, W8 and later | **NOT RUN** on this head (§6) | |

**Op-level-only types (owner scope note).**
- q4_1, q5_0, q5_1, iq4_nl and f32 are covered on this head by the CPU oracle suite (1439/1439) and by lavapipe (763/763 `--fail-unsupported`). `test-qsa-supported-types` is in the 18/18 ctest.
- iq4_nl, q5_1 and f32 also pass the indexer guard (`fake-idx`).
- ROCm0 / RADV op runs and op time are **NOT RUN**.
- By scope these types are "op-level proven only, no end-to-end run".

## 5. Deferred acceptance tests (item 7) on head e31548e7d
Every test needs ROCm0 or RADV runs, and none could run. Because none ran, no reduced-matrix reduction was used. The criteria are those of the task text.

| test | criterion (short) | state |
|---|---|---|
| AT-A (Lab A, round 1 #0) | G1 where it beats min(G4 packed + pack, G4 direct) by more than 3 × p90 noise; the HIP proc encodes the fitted rule; e2e f16 ub512 pp512 d131k neutral or better under §16 (CI contains 0 or lies above it, half-width ≤ 1 %); the trimmed-layer row is a W4 gate row | **NOT RUN**. c16d2360c changes the G1 and G4 inner loops, so the G1 / G4 timings that fitted the rule must be retaken on this head. |
| Lab B VB1 / AT-B1 | f16 new / W0 neutral or better under §16; q8_0, bf16, S_fast ≥ 0.95 × new f16 | **NOT RUN** (the q8_0 / q4_0 miss is TODO-LATER #3) |
| VB2 | new f16 op ≤ 1.10 × the in-build sel_cm median | **NOT APPLICABLE as written**: sel_cm was deleted in W13 (740d48097). It needs an owner ruling, and counts as not passing. |
| VB3 | the faster of direct and staged V per type; ≤ 1.15 × f16 or ≤ the HIP C18 ratio | **NOT RUN** |
| VB4 / AT-B4 | f16 sparse ≥ 0.98 × dense in each cell; 0 CPU FA / sparse nodes for the q8_0 / q4_0 shortcut; decode at depth ≥ W0 | **NOT RUN** |
| VB5 / AT-B5 | every T ≤ new f16 + 1 MiB; new f16 ≤ W0 Vulkan f16 | **NOT RUN** (the MTP +12 MiB cell is TODO-LATER #4) |
| VB6 | mean paired KLD difference ≤ B_vk 6.4e-4 | **NOT RUN** |
| C-5 | 0 strict-reserve violations in every cell; reserved partial ≥ the §7 formula | **NOT RUN** |
| C-17 | new f16 ≤ PR219 reserve + pool high-water, or the stated delta; partials scale with n_stream | **NOT RUN** |
| C-12 | each cell's new reserve ≤ PR219 + the bound; runtime ≤ reserve; answers 2749 / 12749 | **NOT RUN**. e31548e7d changes which route a later reserve builds, so this grid's mid-session cells are new risk. |
| C-16 | strip-view census > 0 for both srcs; no-share − share ≥ ½ the PR219 delta | **NOT RUN** |
| D1 | kind-4 count = W0; U16 memcmp identical; OVL ratio-of-ratios CI contains 0 or lies above it | **NOT RUN** (the criterion wording is TODO-LATER #5) |
| D2 / D3 | plan and attend on their K buffer's device in 100 % of prints; guard 0; KLD ≤ R1; CPU perf ≥ 1 − 2 × the W0 spread | **NOT RUN** (CPU perf is TODO-LATER #2) |
| D4 | 0 GPU-placed plan / attend / shortcut-FA nodes under `-nkvo`, probe included; PPL within 3 × SE of C15 | **NOT RUN** |
| D5 | `-fa auto` resolution logged; under `-fa off` / `-nkvo`: census > 0, 0 masked, guard 0; PPL within 3 × SE of PR219 | **NOT RUN** |
| AT-W12-BANDS | bands frozen in §15.0 before scoring; self-replay KLD 0 (≤ 3.7e-7) | The bands are frozen (PROOF-initial §10.5, head-independent). The self-replay on this head is **NOT RUN**. |
| AT-Q10Q6 | calibration valid; Q10 control (new rerun = 0, new vs PR219 ≤ floor); Q10 cells ≤ τ; Q6 ≤ τ_T; answers | The calibration (PR219 side) stands as in PROOF-initial. The gates on this head are **NOT RUN**; the Q10 control was FAIL in PROOF-initial and has had no fix. |
| AT-W7PERF | sparse ≥ 0.98 × dense per (pair, n_q band), or prefer_dense flips; MTP tg P5 ≥ 0.94; `-nkvo` new / PR219 ≥ 0.98 per cell; q8_0 / f16 ≥ 0.94 under `-nkvo`; census | **NOT RUN** |
| AT-B1 ext / AT-B3 vpack / AT-B7 | S_fast ≥ 0.95 × f16 per AT-B1 cell; vpack rule; portable op runs pass on lavapipe and RADV sg32; supports_op false without a pipeline | **lavapipe part PASS**: PLAN 92/92 and SPARSE_ATTN 763/763 `--fail-unsupported`, 0 failures; `test-sparse-attn-vk` OK. RADV sg32, the AT-B1 cells and vpack are **NOT RUN**. |

## 6. Moved-to-Prove checks, W8 and later (item 8)
All of these need the GPU, so all are **NOT RUN** on `e31548e7d`. The criteria are verbatim in IMPLEMENTATION.md at the lines given.

| W item | checks | criterion source | state |
|---|---|---|---|
| W8 | AT-Q10Q6 (1)(2)(3), Q10, Q6 on the head | IMPLEMENTATION.md:1052-1066 | NOT RUN |
| W9 | AT-C12 grid and image-slot layout, M5, AT-C17 / M6, AT-C16b, M4 (b) (c), C15 ub1024 + A/B/C, np2 shared-prefix trim on/off, U27 | IMPLEMENTATION.md:1241-1266 | NOT RUN |
| W10 | U6 OP_PROF A/B, rotated op time, Q5, Q1-A / Q1-B, P1-P6, Vulkan in-op H_q quality | IMPLEMENTATION.md:1461-1474 | NOT RUN |
| W11 | Q4 selection agreement, M1 iq4_nl MTP, Vulkan q8_0 indexer for q4_0 KV, GLM5_NEXT session (no GGUF exists: not applicable) | IMPLEMENTATION.md:1586-1601 | NOT RUN (indexer guard on a fake device: PASS, §2.4) |
| W13 | the full §14-§16.5 campaign, AT-C16b, VGPR after rebase, merge-tree | IMPLEMENTATION.md:1754-1771 | NOT RUN. No rebase was made, so VGPR-after-rebase does not apply. Remote master is now `0c1021e1f`; the merge-tree against it is NOT RUN (§1). |
| W7 remainder | Q3-B q4_0 ub256, Q11c, Q12 `-ngl 0` and coarse, P2 / P3, NDB, NPIN, P5 + MTP, P4, P6N, D1V, M2 / M3, AT-B4 e2e, AT-B1 cells, U30 on Vulkan | IMPLEMENTATION.md:923-945 | NOT RUN |
| Prove-fix 1 reruns | ctest, ops, np2, vkc, mtp, c18, census, smoke | IMPLEMENTATION.md:1907 | NOT RUN |
| Final-fix 2 reruns | RADV `test-sparse-attn-vk dense reason vpack race`; HIP / RADV rpc-view; HIP / RADV op tests; ctest build / build-vk | IMPLEMENTATION.md:2059 | NOT RUN |
| F3 reruns | `hip_nanscore_*`, `hip_shape_*`, `hiprpc_*`, the ROCm0 / RADV suites, the HIP / HIP + RPC / RADV ctest | IMPLEMENTATION.md:2164 | NOT RUN |
| F4 reruns | `hip_lead_before` FAIL / `_after` OK (default, portable, fixup, generic); `radv_nan_before{,_scalar}` FAIL; `radv_lead_after{,_scalar}` OK; `hip_2g_after`; the ROCm0 / RADV SPARSE_ATTN / PLAN suites; the HIP / RADV ctest; `hip_guard_pass` | IMPLEMENTATION.md:2286-2301, `F4/gpu_rerun.sh` header | NOT RUN |

## 7. Known issues (deferred: TODO-LATER.md; reported, not counted in pass)
None could be re-measured on `e31548e7d`, because of the GPU lock. The last measurements are those of TODO-LATER.md, PROOF-initial §12 and review-r2 §7.

| # | item | last measurement | status on e31548e7d |
|---|---|---|---|
| 1 | Q1 direct form, q4_0 / bf16 | q4_0 2.082e-3 > band 1.441e-3; bf16 1.250e-4 > 1.032e-4 | NOT RUN; last FAIL |
| 2 | AT-D2 `-ngl 0` CPU perf | pp512 0.69-0.71 × PR219, tg32 0.89 × | NOT RUN; last FAIL. df88ab1bb adds one compare per cell to the CPU op (`sparse-attn.cpp:414`); its cost is UNPROVEN. |
| 3 | AT-B1 Vulkan q8_0 / q4_0 < 0.95 × f16 | 0.921 / 0.916 | NOT RUN; last FAIL |
| 4 | AT-B5 Vulkan MTP f16 +12 MiB | not re-measured since W7 | NOT RUN |
| 5 | AT-D1 kind-4 wording | new 12, PR219 / W0 0 | NOT RUN; not met as worded |
| 6 | M1 iq4_nl MTP | +0.26 MiB after W11 (PROOF-initial) | NOT RUN; last PASS |
| 7 | `-nkvo` copy route: four d0 cells, bf16 | the d0 cells unresolved | NOT RUN; last open |
| 8 | Q4 thin f16-KV margins | +6.3e-6 / +2.0e-6 | NOT RUN; last PASS (thin) |
| 9 | Q7 MTP draft acceptance | stage 1, new − PR219: default −1.41, q8_0 −1.88, q4_0 −0.28, bf16 −1.44 pp. Stage 2: +1.28, +0.93, +1.17, +1.12 pp. LCB −1.87 to −3.83 against the gate's ≥ −1 pp (review-r1 §6). | NOT RUN; last FAIL (deferred) |

## 8. Commands run in this step (all on strix-002)
1. `uptime; flock -n ~/gpu.lock true && echo LOCK_FREE || echo LOCK_HELD; lslocks | grep gpu.lock; ps -eo pid,user,stat,etime,cmd | grep -E "llama|test-|flock"; cat /sys/class/drm/card*/device/mem_info_{gtt,vram}_used; free -g; git rev-parse HEAD; git status --short | wc -l; git log --oneline -12; git merge-base HEAD origin/master`
2. `grep -E "State|SigPnd|ShdPnd" /proc/4041225/status; cat /proc/4041225/wchan; journalctl -k --since "2026-11-10 14:57"; sudo -n true`
3. In the background: `timeout 5400 flock ~/gpu.lock -c 'date "+GOT_LOCK %F %T"'`, logged to `R3/lockwait.log`.
4. `R3/nogpu.sh`, logged to `R3/nogpu.out`. It ran:
   - the 5 incremental builds;
   - the CPU and RPC ctest subsets;
   - the CPU SPARSE_ATTN / PLAN suites, twice each;
   - CPU GET_ROWS,SET_ROWS and the lead cases;
   - `test-qsa-guard fake-idx` for 7 types, plus `fake`;
   - the lavapipe SPARSE_ATTN / PLAN suites, twice each, with `--fail-unsupported`;
   - the lavapipe lead cases, QSA_EXPAND / SCORE / STRIP and GET_ROWS / SET_ROWS / CPY;
   - 6 lavapipe unit tests;
   - te-rows 15 times on lavapipe and 10 times on the CPU;
   - the lavapipe ctest.
5. The ASAN build:
   - `CC=/opt/rocm/core-10.1/lib/llvm/bin/clang CXX=…/clang++ cmake -B build-vk-asan -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLAMA_SANITIZE_ADDRESS=ON -DGGML_SANITIZE_ADDRESS=ON -DCMAKE_CXX_FLAGS=-I/home/gaetan/.local/spirv-headers/include -DLLAMA_CURL=OFF`;
   - `cmake --build build-vk-asan -j 8 --target test-qsa-te-rows`;
   - 10 runs of `ASAN_OPTIONS=detect_leaks=0 R3/lvp.sh build-vk-asan/bin/test-qsa-te-rows`.
   The directory was then deleted.
6. The te-rows localization: the `CPU` and `Vulkan` filter args (10 runs each); `GGML_VK_QSA_EXPAND=0` and `GGML_VK_DISABLE_FUSION=1` (12 runs each); `stdbuf -oL` (8 runs); `test-backend-ops -b Vulkan0 -o TOP_K` (3 runs).
7. Outcome hashes: `grep -E "^\s+(SPARSE_ATTN|QSA)" <log> | sed 's/\x1b\[[0-9;]*m//g' | sha256sum`, plus `grep "compiled keys"`.
8. Code reading:
   - `git diff --stat 574689270..HEAD`;
   - `git log --format="%h %an <%ae> | %s%n%(trailers:only)" 574689270..HEAD`;
   - `git log -1 --format="%an <%ae>" 94e086847`;
   - `git show` of df88ab1bb, c16d2360c, 21f346a6e, 4e08e5b91 and e31548e7d;
   - `sparse_attn.comp:180-225`, `sparse_attn_cm.comp:276-363`, `qwen4exp-te.h:20-60`, `ggml-vulkan.cpp:4522-4540, 19101-19105, 19239`;
   - `git ls-remote origin refs/heads/master`; `git merge-base --is-ancestor 13a279bfa ea5285ddf`.

## 9. Side effects and cleanup
- No AMD GPU command was run, and no process this agent did not start was signalled. pids 4041225 and 4041224 were not touched.
- The only process this agent stopped is its own lock waiter (§1.1).
- No systemd unit was touched, strix-001 was not used, and nothing was pushed or posted.
- `~/dev/wt_rewrite`: only incremental no-op builds and the temporary `build-vk-asan` (deleted) ran there. There are no source changes; `git status --short` is empty.
- `R3` holds logs only (2.3 MB). No file over 1 GB remains.
