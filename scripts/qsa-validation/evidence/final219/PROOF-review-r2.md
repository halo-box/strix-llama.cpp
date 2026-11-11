# PROOF (review-r2): qsa-rewrite head 574689270 on strix-002

**Verdict: pass = false.** The AMD GPU on strix-002 is still unusable, so no GPU check could run on head `574689270`.
- `~/gpu.lock` is still held by the hung Vulkan `llama-perplexity` that the initial Prove started: pid 4041225, state D, SIGKILL pending, 91.6 GB GTT held. This is the same process that blocked review-r1 and F3.
- Every GPU command must take `flock ~/gpu.lock`, so each one would block forever. A bounded wait on the lock (no GPU action) timed out (§1).
- No sudo is available, and the rules forbid signalling a process this agent did not start. The process is in D state with SIGKILL already pending, so a signal would not end it anyway.
- The model does not fit in host memory next to the stuck process: 32 GiB available (`free -g`), and the model is 104 GB (F3, IMPLEMENTATION.md:2142). So no model run is possible on the CPU either.

The step proved what it could without the GPU:
- all five build trees are at HEAD and build with 0 warnings;
- the CPU QSA ctest subset passes 18/18, and the CPU + RPC subset passes 21/21;
- the CPU op suite passes SPARSE_ATTN 1402/1402 and SPARSE_ATTN_PLAN 185/185, run twice with identical per-case outcomes;
- lavapipe (a CPU Vulkan device) passes every changed op against the CPU oracle, run twice with identical per-case outcomes, including the 6 new G1 `nanscore` cases;
- the lavapipe unit tests pass, including the new shape-guard test (78 malformed nodes, 0 failed);
- the 7 F3 commits (`ed4bd3ea5..574689270`) were read.

**What still fails or is not run:**
- The lavapipe `test-qsa-te-rows` heap-corruption abort is still open. It aborted in 6 of 15 direct runs. It never aborts on the CPU-only build (0 of 10 runs).
- Every check that items 1-8 need on ROCm0 or RADV is **NOT RUN**, so none of them passes. That covers perf, pinned KLD, images, MTP, the GPU census, reserve = runtime on the GPU, the Vulkan depth run, every deferred acceptance test, and every Moved-to-Prove check.

**Conventions.**
- `R2` = `strix-002:~/dev/wt_lab/prove_r2`, which holds this step's logs and harnesses (`nogpu.sh`, `lvp.sh`), 1.7 MB in all.
- `R1` = `~/dev/wt_lab/prove_r1`, the review-r1 logs.
- `F3` = `~/dev/wt_rewrite_logs/F3`.
- All ssh output had its login banner filtered out.

## 1. Blocking condition (evidence)

| fact | command | output |
|---|---|---|
| HEAD | `git -C ~/dev/wt_rewrite rev-parse HEAD` | `57468927092708d52ff76a942fb231a06d3131b6` |
| worktree clean | `git status --short \| wc -l` | `0` |
| new commits since review-r1 | `git log --oneline ed4bd3ea5..HEAD` | 7 commits: d7e4656d1, 108d6ce00, 195dadc10, a22777440, 458acecf6, a82bdb924, 574689270 (IMPLEMENTATION.md "F3") |
| merge-base with origin/master | `git merge-base HEAD origin/master` | `ea5285ddf`. origin/master is `62a3f4144`, 35 commits past it (`git rev-list --count ea5285ddf..origin/master` = 35). The rewrite is not on a newer master, so no current-master rebuild is needed. |
| base arm | `git merge-base --is-ancestor 13a279bfa ea5285ddf` | true. `~/dev/wt_pr219v2_base` (13a279bfa) is older than the rewrite's base, so the base arm stays `~/dev/wt_lab/prove/base_ea5`, whose `build/bin/llama-bench` exists (`ls`), as in PROOF-initial. The PR219 arm is `~/dev/wt_pr219v2/build/bin`, whose `llama-bench` exists. |
| lock at 17:56 | `flock -n ~/gpu.lock true && echo LOCK_FREE \|\| echo LOCK_HELD` | `LOCK_HELD` |
| holder | `lslocks \| grep gpu.lock` | `timeout 4041224 FLOCK WRITE 0 0 0 /home/gaetan/gpu.lock` |
| holder process | `ps -eo pid,user,stat,etime,cmd` | `4041225 gaetan D 06:02:32 …/wt_rewrite/build-vk/bin/llama-perplexity … -ctk bf16 -ctv bf16 --kl-divergence-base …/prove/c1/det_model/x_2.kld`, the child of `4041224 S timeout 3600 …` |
| signals | `grep -E "State\|SigPnd\|ShdPnd" /proc/4041225/status`; `cat /proc/4041225/wchan` | `State: D (disk sleep)`, `SigPnd: 0000000000000100` (SIGKILL pending), `ShdPnd: 0000000000004000` (SIGTERM from `timeout`); wchan `drm_suballoc_insert` |
| at 18:16 | `ps -o pid,stat,etime,wchan:25 -p 4041225,4041224` | `4041225 D 06:22:15 drm_suballoc_insert`; `4041224 S 06:22:15 sigsuspend.isra.0` |
| GTT held | `cat /sys/class/drm/card*/device/mem_info_gtt_used` | `91631312896` at 17:56 and at 18:16. This is unchanged since PROOF-initial §13 and review-r1. |
| host memory | `free -g` | 124 total, 91 used, 32 available |
| kernel log | `journalctl -k --since "2026-11-10 14:57"` | `-- No entries --`. No GPU reset or recovery is logged. The last kernel lines (14:56:36) are the expected bug-8 segfault demonstration of Final-fix 2's "before" tree. |
| root | `sudo -n true` | `sudo: a password is required` |
| lock wait | `timeout 3600 flock ~/gpu.lock -c 'date "+GOT_LOCK %F %T"'`, a wait on the lock only (no GPU action), started 17:57:21 (`R2/lockwait.log`) | see §1.1 |

### 1.1 Lock wait result
- `R2/lockwait.log`: `start 2026-11-10 17:57:21`, `rc=124` (timeout), `end 2026-11-10 18:57:21`. The lock never freed in 60 minutes.
- At 18:57:26:
  - `ps` showed `4041225 D 07:02:59 drm_suballoc_insert` and `4041224 S 07:02:59 sigsuspend.isra.0`;
  - GTT was still `91631312896`;
  - `journalctl -k --since "2026-11-10 14:57"` still printed `-- No entries --`.
- Together with review-r1's 40-minute wait (`R1/lockwait.log`) and F3's 13-minute wait, the lock has not been free at any check since the initial Prove started pid 4041225.

**Required action, for the owner.** Reboot strix-002, or reset the amdgpu device (this needs root). Then run, in order:
1. `bash ~/dev/wt_rewrite_logs/F3/gpu_rerun.sh`: the F3 before / after GPU checks (criteria in IMPLEMENTATION.md "F3 / Left for the orchestrator", line 2164).
2. `bash ~/dev/wt_lab/final_fix/rerun_gpu.sh`: the Final-fix 2 RADV and HIP checks (IMPLEMENTATION.md:2059).
3. `bash ~/dev/wt_lab/prove_fix/rerun.sh setup ctest ops np2 vkc mtp c18 census smoke`, retargeted to 574689270 (IMPLEMENTATION.md:1907).
4. The full Prove chains of PROOF-initial §15, on the reduced matrix of the owner's scope note.

## 2. Checks that ran (no AMD GPU)

All of them come from `R2/nogpu.sh`, whose output is `R2/nogpu.out`. The run started at 17:57:45 and ended at 18:14:47.

### 2.1 Builds at HEAD (compile only, incremental)
- Command: `cmake --build <tree> -j 16`. Logs: `R2/rebuild_<tree>.log`.
- Configs, from `CMakeCache.txt`:
  - `build`: `GGML_HIP=ON`, `GPU_TARGETS=gfx1151`, Release, `CMAKE_HIP_COMPILER=/opt/rocm/core-10.1/lib/llvm/bin/clang` (the rule flags);
  - `build-vk`: `GGML_VULKAN=ON`;
  - `build-cpu`: HIP, Vulkan and RPC off;
  - `build-rpc`: CPU + `GGML_RPC=ON`;
  - `build-rpc-hip`: HIP rule flags + `GGML_RPC=ON`.

| tree | rc | `warning:` | `error:` | verdict |
|---|---|---|---|---|
| build | 0 | 0 | 0 | PASS |
| build-vk | 0 | 0 | 0 | PASS |
| build-cpu | 0 | 0 | 0 | PASS |
| build-rpc | 0 | 0 | 0 | PASS |
| build-rpc-hip | 0 | 0 | 0 | PASS |

The trees were already current from F3's final build (`F3/final_*.log`), so these runs only confirm that nothing was stale.

### 2.2 ctest QSA subsets (CPU)
| build | command | result | log |
|---|---|---|---|
| build-cpu | `ctest -R "test-qsa\|test-sparse-attn" -j1 --output-on-failure` | **18/18 passed**, rc 0 | `R2/ctest_cpu.log` |
| build-rpc | `ctest -R "test-qsa\|test-sparse-attn\|test-rpc" -j1 --output-on-failure` | **21/21 passed**, rc 0 | `R2/ctest_rpc.log` |

The new F3 tests pass in both builds:
- `test-qsa-reserve-passes` (#33);
- `test-qsa-noncompact-check` (#34);
- `test-qsa-supported-types` (#36, whole-entry match);
- `test-sparse-attn-shape-guard` (#86);
- `test-rpc-sparse-attn` (#87 in build-rpc);
- `test-rpc-hello` (#88).

As F3 states, `test-rpc-sparse-attn` part (1) is vacuous on a CPU server, because the CPU alloc size is `nbytes`. Its HIP-server run (`hiprpc_before` / `hiprpc_after`) is **NOT RUN**.

### 2.3 CPU op suite (`test-backend-ops -b CPU`)
Each op ran in its own process, twice. Logs: `R2/cpu_ops_<OP>_r{1,2}.log`.

| op | r1 | r2 | per-case outcome hash (r1 = r2) |
|---|---|---|---|
| SPARSE_ATTN | 1402/1402, rc 0 | 1402/1402, rc 0 | `69b383a2bc85dec8` (1374 lines), equal |
| SPARSE_ATTN_PLAN | 185/185, rc 0 | 185/185, rc 0 | `81bcbb8c39d72337` (185 lines), equal |

### 2.4 lavapipe runs (Vulkan on a CPU device), compared against the CPU oracle
- Harness: `R2/lvp.sh`. It exports the following, then runs under `timeout 2400`:
  - `VK_ICD_FILENAMES=~/dev/wt_lab/w6/lvp43/icd.json` (Fedora 43 Mesa 25.3.6 lavapipe);
  - `LD_LIBRARY_PATH=…/lvp43/usr/lib64`;
  - `GGML_VK_VISIBLE_DEVICES=0`;
  - `VK_LOADER_LAYERS_DISABLE=~implicit~`;
  - `NODEVICE_SELECT=1`.
- Every op log prints `ggml_vulkan: 0 = llvmpipe (LLVM 21.1.8, 256 bits) … warp size: 8 | shared memory: 32768 | matrix cores: none` (`grep -c llvmpipe` = 2 per log).
- This device is not the AMD GPU, and these runs did not take `flock`.

test-backend-ops, one op per process:

| op | flags | result | "not supported" | census coverage (kernel path) |
|---|---|---|---|---|
| SPARSE_ATTN r1 | `--fail-unsupported` | **742/742, rc 0** | 660 | `Vulkan0: 81/81 compiled keys of {sa_attend/,sa_fixup/,sa_generic/,sa_merge,sa_pack/,sa_portable/} launched (336 excluded by a forcing switch): OK` |
| SPARSE_ATTN r2 | same | **742/742, rc 0** | 660 | 81/81 OK |
| SPARSE_ATTN `-p nanscore` (F3 bug 1) | | **6/6, rc 0** (the G1 cases; G4 is unsupported on lavapipe, as F3 states) | | |
| SPARSE_ATTN_PLAN r1 / r2 | `--fail-unsupported` | **92/92, rc 0** each | 93 each | `1/1 compiled keys of {sa_plan/} launched: OK` |
| QSA_EXPAND / QSA_SCORE / QSA_STRIP | | 56/56, 13/13, 6/6, rc 0 | | |
| GET_ROWS / SET_ROWS / CPY | | 643/643, 439/439, 307/307, rc 0 | | |

Notes on these runs:
- **Counts.** The counts rose from 736 / 654 in review-r1 to 742 / 660. That is the 6 nanscore G1 cases and their 6 G4 counterparts, which are expected unsupported. `--fail-unsupported` passed, so each unsupported case is on the expected list.
- **Determinism.**
  - Within one process, the suite runs SPARSE_ATTN and SPARSE_ATTN_PLAN twice and memcmps the outputs (`tests/test-backend-ops.cpp`, as cited in review-r1 §2.3), so every pass above includes that check.
  - Across processes, the per-case outcome lines are identical. The `sha256` prefix of the colour-stripped lines is `ee5789869b411698` for both SPARSE_ATTN runs (1388 lines) and `38b1295714c50169` for both PLAN runs (185 lines; the same hash as review-r1).
  - A cross-process output hash is not a valid test here, because the inputs are drawn fresh in each process (PROOF-initial §3.3).
- **Kernel-path statement.**
  - On lavapipe, the census shows that the portable tier and its siblings ran (`sa_portable/`, `sa_generic/`, `sa_pack/`, `sa_fixup/`, `sa_merge`).
  - The cm-tier keys are excluded by the forcing switch / tier rule.
  - The RADV cm and scalar tiers, and every HIP kernel key (G1 and G4 attend, merge, pack, fixup, plan), are **NOT RUN**.
  - So the F3 HIP merge fix (d7e4656d1, `sparse-attn-merge.cu:37`) is proven by code reading only (§3), and the HIP `nanscore` cases are NOT RUN.

Unit tests on lavapipe:

| test | result | log |
|---|---|---|
| `test-sparse-attn-vk` (all parts) | `OK: 0 failed checks`, rc 0 | `R2/lvp_test-sparse-attn-vk.log` |
| `test-sparse-attn-rpc-view` | `27 rows, 0 failed checks: OK` | `R2/lvp_test-sparse-attn-rpc-view.log` |
| `test-qsa-allvis-rows` | `RESULT PASS (failures 0)` | `R2/lvp_test-qsa-allvis-rows.log` |
| `test-sparse-attn-shape-guard` (F3 bug 2) | `78 malformed nodes checked, 0 failed: OK` (the same count as F3's after run) | `R2/lvp_test-sparse-attn-shape-guard.log` |
| `test-sparse-attn-gate-scratch` | `0 failed checks: OK` | `R2/lvp_test-sparse-attn-gate-scratch.log` |
| `test-qsa-te-rows`, 15 direct runs | **6 rc 134** (`corrupted size vs. prev_size while consolidating`): runs 1, 2, 5 and 3 of runs 6-15 | `R2/lvp_terows_{1..15}.log` |
| `build-vk` ctest subset, `-j1` | 18/18 passed (te-rows passed in this one run) | `R2/lvp_ctest_vk.log` |
| `test-qsa-te-rows` on build-cpu, 10 runs | 10/10 rc 0 | `R2/cpu_terows_{1..10}.log` |

**te-rows abort (open, FAIL).**
- It is the pre-existing lavapipe heap corruption: 9/20 on `ed4bd3ea5`, 11/20 on `97865cacb` (IMPLEMENTATION.md Final-fix 2), and 2/6 in review-r1. In this step it is 6/15.
- None of the 7 F3 commits touches a `qwen4exp-te` file (`git diff --stat ed4bd3ea5..HEAD`, §3).
- New evidence, from gdb under lavapipe (`R2/lvp_terows_gdb_1.log`):
  - the abort fires in thread 35, one of the `test-qsa-te-row` worker threads, not an `llvmpipe-*` thread;
  - the frames are `abort ← malloc_printerr ← _int_free_merge_chunk ← _int_free_chunk ← ??`;
  - it happens after `ggml_vulkan: MUL_MAT_ID f16-B route: …` is printed.
- The CPU-only build never aborts (0/10). So the corruption needs the Vulkan backend (or lavapipe) in the process. Which writer corrupts the heap is UNPROVEN: there is no valgrind on strix-002 (`which valgrind`: not found), and there was no ASAN build.
- On RADV it is **NOT RUN**.
- It is not a TODO-LATER item, so it counts as FAIL.

## 3. Code reading of the 7 F3 commits (`git show ed4bd3ea5..574689270`)
- Diffstat: 22 files, +1216 / −74.
- Every commit is by Gaetan Puleo `<12990773+gaetan-puleo@users.noreply.github.com>` and carries the `Claude-Session` trailer (`git log --format=… %(trailers:only)`).
- d7e4656d1 carries `Co-authored-by: Iggy Jackson <iggy@theiggy.com>`. This matches `git log -1 --format="%an <%ae>" 94e086847` = `Iggy Jackson <iggy@theiggy.com>`, and is required because `sparse-attn-merge.cu` adapts #219's decode merge, as its header comment states.
- The other six commits port no #219 code, so the absence of the trailer is correct.

| commit | change (verified in the diff) | observation | GPU-side proof still needed (NOT RUN) |
|---|---|---|---|
| d7e4656d1 | `sparse-attn-merge.cu:37`: `count > 0.0f` becomes `count != 0.0f`, so a NaN normaliser is kept and the row becomes 0 | This matches the CPU op `l > 0 ? a / l : 0` with l NaN. An empty split still has l = 0 and is skipped. | HIP `nanscore` before FAIL / after PASS (`gpu_rerun.sh hip_nanscore_*`); HIP SPARSE_ATTN suite |
| 108d6ce00 | `ggml_sparse_attn_shape_why` / `_plan_shape_why` (`ggml.c`, after `ggml_sparse_attn`); used in HIP `sa_select` and plan supports_op, Vulkan `ggml_vk_sa_select`, plan supports_op and the unsupported reason, and CPU supports_op | The constructors already asserted every K/V, head, stream and row-layout invariant before this commit (`ggml.c` `ggml_sparse_attn`: `k->ne[1] >= n_kv4`, etc.). The new function adds only the plan-size and op_params = sel-shape checks, which `ggml_sparse_attn_plan` itself writes. So a node built by the model graph cannot newly fail the added `GGML_ASSERT(… == NULL)`. That is a code-reading argument; a model run on this head is NOT RUN. | `hip_shape_before` FAIL / `_after` OK; ROCm0 / RADV suites |
| 195dadc10 | the pass list moves to `llama_qsa_reserve_passes`, and gains `1s-passB` and `1s-passB-0out` (one stream at n_kv_b, `min(ub, n_kv_b)` rows) when `!kv_unified && n_seqs > 1` and a shortcut exists | The existing passes are unchanged in the refactor (diff read; `test-qsa-reserve-passes` passes). The two new passes can only grow the reserve of split multi-sequence contexts. | reserve = runtime on `-np 2/4/8` split; the M-cells (memory) of split contexts; the size of the growth is UNPROVEN |
| a22777440 | `qwen4exp_qsa_noncompact_error`: false unless `blocks && !swa`; `set_causal_attn` sets `sched_need_reserve` when QSA layers exist | `swa = hparams.n_swa != 0`, the same predicate `scalar` uses (`qwen4exp.cpp:1459`) | a non-causal QSA context run (NOT RUN, no model) |
| 458acecf6 | RPC alloc-size cache key gains every src's type / op / op_params / ne / nb | | `hiprpc_before` FAIL / `_after` OK on a HIP server |
| a82bdb924 | `RPC_CMD_SUPPORTS_OP` for the two QSA ops; `RPC_PROTO_MIN_SERVER_MINOR_VERSION 1` | **Behaviour change, recorded for the owner:** base `ea5285ddf` accepts any server with `minor <= client minor` (`git show ea5285ddf:ggml/src/ggml-rpc/ggml-rpc.cpp:432`). This head rejects an 8.0 server. So a client of this branch no longer talks to a base-version rpc-server, even for a model without QSA. This is intended (commit text, `test-rpc-hello`). It is not counted as a failure, but it belongs in the PR text. | |
| 574689270 | `test-qsa-supported-types` matches whole list entries | | none |

## 4. Scorecard on head 574689270 (task items 1-8)

| # | check | verdict | source |
|---|---|---|---|
| 0 | builds HIP / Vulkan / CPU / RPC / RPC + HIP at HEAD, 0 warnings | PASS | §2.1 |
| 1 | test-backend-ops per op on ROCm0 (graphs on and off, each `GGML_CUDA_SA_FORCE`) and on RADV Vulkan0 (each `GGML_VK_SA_FORCE`), with HIP census coverage per kernel key | **NOT RUN** (GPU lock) | §1 |
| 1 | CPU oracle op suite: SPARSE_ATTN 1402/1402, PLAN 185/185, run twice, identical outcome hashes | PASS | §2.3 |
| 1 | lavapipe vs the CPU oracle: SPARSE_ATTN 742/742, PLAN 92/92, nanscore 6/6, QSA_EXPAND/SCORE/STRIP, GET_ROWS/SET_ROWS/CPY; census 81/81 and 1/1 | PASS (portable tier only) | §2.4 |
| 1 | unsupported shapes asserted unsupported (`--fail-unsupported` with the expected list; shape-guard 78/78) | lavapipe and CPU PASS. ROCm0 and RADV **NOT RUN** | §2.2, §2.4 |
| 1 | determinism | CPU and lavapipe PASS. HIP and RADV, and the model-level logit hash run twice: **NOT RUN** | §2.3, §2.4 |
| 1 | ctest QSA: CPU 18/18, CPU + RPC 21/21, lavapipe build-vk 18/18 in ctest but te-rows aborts in 6/15 direct runs; HIP, HIP + RPC and RADV | CPU PASS; RPC PASS; lavapipe **FAIL** (te-rows); HIP / HIP + RPC / RADV **NOT RUN** | §2.2, §2.4 |
| 2 | HIP route census (EXPECT rows, 0 masked, 0 CPU FA / sparse nodes for offloaded layers); Vulkan placement census | **NOT RUN** | |
| 2 | reserve = runtime (strict) on every route, including the new 1s-passB passes (195dadc10), HIP `-np 2` split ub8192, and Vulkan ub16 (still unfixed: IMPLEMENTATION.md Prove-fix 1 "Not fixed" 2.2) | **NOT RUN** | |
| 2 | MTP census, gated share ≥ 0.9 (7a96532ad) | **NOT RUN** | |
| 2 | guard / AT-C18 on a real run | **NOT RUN** | |
| 3 | pinned KLD (QSA + MoE replay, `~/dev/wt_measQ` QPIN) for f16, q8_0, q4_0, bf16, q8_0K/f16V, f16K/q8_0V vs the measured floor; f16 bit-identity; unpinned catastrophe check; depth check; Q1-vk | **NOT RUN** | |
| 3 | Q12 coarse q8_0, the Q10 control, image p3 seq1 (FAIL in PROOF-initial, no fix since) | **NOT RUN**, open | PROOF-initial §2 |
| 4 | perf: text + random, base (`prove/base_ea5`) vs new vs PR219, f16 / q8_0 / q4_0 / bf16 / mixed, ub 512 / 1024 / 4096, d 0 / 32k / 131k / ~262k (~300k text) / deepest; tg at depth; §16 statistic; stall (no sample > 1.5 × median); compute buffer; peak VRAM | **NOT RUN** | |
| 5 | images: `~/repro_img/repro_kv.sh` A / B / C for f16 and q8_0; `-np 2` split with images; `-ctk q8_0 -ctv f16` with images; image understanding f16 vs q8_0; MTP draft | **NOT RUN** | |
| 6 | Vulkan build | PASS | §2.1 |
| 6 | Vulkan tests on RADV; q8_0 depth run on the RADV device | **NOT RUN** (lavapipe tests: §2.4) | |
| 7 | every deferred acceptance test | **NOT RUN** on this head (§5) | |
| 8 | every Moved-to-Prove check, W8 and later | **NOT RUN** on this head (§6) | |

**Op-level-only types (owner scope note).**
- q4_1, q5_0, q5_1, iq4_nl and f32 are the 9-entry list of `test-qsa-supported-types`, which passes with the whole-entry match in this step.
- On this head their op-level proof is the CPU oracle suite (1402/1402) and lavapipe (742/742 with `--fail-unsupported`).
- ROCm0 / RADV op runs and op time are **NOT RUN**.
- By scope they are "op-level proven only, no end-to-end run". On this head even the GPU op level is missing.

## 5. Deferred acceptance tests (item 7) on head 574689270

Each test needs ROCm0 or RADV runs, and none could run. The reduced matrix of the owner's scope note would apply to each run. Because none ran, no reduction was used. Criteria are those of the task text.

| test | criterion (short) | state |
|---|---|---|
| AT-A (Lab A, round 1 #0) | G1 is picked where it beats min(G4 packed + pack, G4 direct) by more than 3 × p90 noise; the HIP proc encodes the fitted rule; e2e f16 ub512 pp512 d131k is neutral or better under §16 (CI contains 0 or lies above it, half-width ≤ 1 %); the trimmed-layer row is a W4 gate row | **NOT RUN**. The F3 commits do not touch the G policy (`sparse-attn-attend-g1.cu` is not in the diffstat). The code reading of PROOF-after-fix §2.4 (the fitted CPP line, 16/16 cells) still applies, but a code reading is not a pass. |
| Lab B VB1 / AT-B1 | f16 new / W0 is neutral or better under §16; q8_0, bf16 and the S_fast pairs ≥ 0.95 × new f16 | **NOT RUN**. The q8_0 / q4_0 miss is TODO-LATER #3 (§7). |
| VB2 | new f16 op ≤ 1.10 × the in-build sel_cm median | **NOT APPLICABLE as written**: sel_cm was deleted in W13 (740d48097). It needs an owner ruling, and until then it is counted as not passing. |
| VB3 | ship the faster of direct and staged V per type; diagnostic op gate ≤ 1.15 × f16 or ≤ the HIP C18 ratio | **NOT RUN** |
| VB4 / AT-B4 | f16 sparse ≥ 0.98 × dense in every cell; the q8_0 / q4_0 shortcut has 0 CPU FA / sparse nodes; decode at depth ≥ W0 | **NOT RUN** |
| VB5 / AT-B5 | M1: every T ≤ new f16 + 1 MiB; new f16 ≤ W0 Vulkan f16 | **NOT RUN**. The MTP +12 MiB cell is TODO-LATER #4. |
| VB6 | mean paired KLD difference ≤ B_vk 6.4e-4 | **NOT RUN** |
| C-5 | 0 strict-reserve violations in every cell; reserved partial ≥ the §7 formula | **NOT RUN** |
| C-17 | new f16 ≤ PR219 reserve + pool high-water (M2), or the stated delta; partials scale with n_stream (`-np 4`: 195dadc10 adds passes here) | **NOT RUN** |
| C-12 | each cell's new reserve ≤ PR219 + the bound; runtime ≤ reserve; answers 2749 / 12749 | **NOT RUN** |
| C-16 | strip-view census > 0 for both srcs; no-share − share ≥ ½ the PR219 delta | **NOT RUN** |
| D1 | W7 kind-4 count = W0; U16 memcmp identical; the OVL ratio-of-ratios CI contains 0 or lies above it | **NOT RUN**. The criterion wording is TODO-LATER #5. |
| D2 / D3 | sparse plan and attend on the device of their K buffer in 100 % of the prints; guard 0; KLD ≤ the R1 value; CPU perf ≥ 1 − 2 × the W0 spread | **NOT RUN**. CPU perf is TODO-LATER #2. |
| D4 | 0 GPU-placed plan, attend or shortcut-FA nodes under `-nkvo`, the probe included; PPL within 3 × chunk SE of C15 | **NOT RUN** |
| D5 | `-fa auto` resolution logged; under `-fa off` / `-nkvo`, census > 0, 0 masked nodes, guard 0; PPL within 3 × SE of PR219 | **NOT RUN** |
| AT-W12-BANDS | bands computed by their formulas and frozen in §15.0 before scoring; self-replay KLD 0 (≤ 3.7e-7) | the bands are frozen in PLAN-v3 §15.0 (PROOF-initial §10.5, head-independent). The self-replay on this head is **NOT RUN**. |
| AT-Q10Q6 | calibration valid; Q10 control: new rerun vs new = 0 and new vs PR219 ≤ floor; Q10 cells ≤ τ; Q6 ≤ τ_T; answers | the calibration (PR219 side, head-independent) stays as in PROOF-initial. The gates on this head are **NOT RUN**. The Q10 control was FAIL in PROOF-initial and has had no fix since. |
| AT-W7PERF | per (pair, n_q band) sparse ≥ 0.98 × dense, or prefer_dense flips; MTP tg P5 ≥ 0.94; `-nkvo` new / PR219 ≥ 0.98 in each cell; q8_0 / f16 ≥ 0.94 under `-nkvo`; census | **NOT RUN** |
| AT-B1 ext / AT-B3 vpack / AT-B7 | S_fast ≥ 0.95 × f16 in each AT-B1 cell; the vpack rule; portable op runs pass on lavapipe and on RADV sg32; supports_op false without a pipeline | **lavapipe part PASS**: PLAN 92/92 and SPARSE_ATTN 742/742 `--fail-unsupported`, 0 failures; `test-sparse-attn-vk` "OK: 0 failed checks" (§2.4). RADV sg32, the AT-B1 cells and the AT-B3 vpack arm are **NOT RUN**. The q8_0 / q4_0 AT-B1 miss is TODO-LATER #3. |

## 6. Moved-to-Prove checks, W8 and later (item 8)

All of these need the GPU, so all are **NOT RUN** on 574689270. Criteria are verbatim in IMPLEMENTATION.md at the lines given.

| W item | checks | criterion source | state |
|---|---|---|---|
| W8 | AT-Q10Q6 (1)(2)(3), Q10 on the head, Q6 on the head | IMPLEMENTATION.md:1052-1066 | NOT RUN |
| W9 | AT-C12 grid, AT-C12 image-slot layout, M5 cells, AT-C17 / M6, AT-C16b, M4 (b), M4 (c), C15 ub1024 + A/B/C, np2 shared-prefix trim on/off, U27 | IMPLEMENTATION.md:1241-1266 | NOT RUN |
| W10 | U6 OP_PROF A/B, rotated op time, Q5, Q1-A / Q1-B, P1-P6, Vulkan in-op H_q quality | IMPLEMENTATION.md:1461-1474 | NOT RUN |
| W11 | Q4 selection agreement, M1 iq4_nl MTP, Vulkan q8_0 indexer for q4_0 KV, GLM5_NEXT session (only if a GGUF exists: none was found, so not applicable) | IMPLEMENTATION.md:1586-1601 | NOT RUN |
| W13 | the full §14-§16.5 campaign, AT-C16b, VGPR after rebase, merge-tree | IMPLEMENTATION.md:1754-1771 | NOT RUN. VGPR-after-rebase and merge-tree are not applicable yet: no rebase was made (the merge-base is still ea5285ddf, §1). |
| W7 remainder (listed for completeness; it predates W8) | Q3-B q4_0 ub256, Q11c, Q12 `-ngl 0` and coarse, P2 / P3, NDB, NPIN, P5 + MTP, P4, P6N, D1V, M2 / M3, AT-B4 e2e, AT-B1 unresolved cells, U30 on Vulkan | IMPLEMENTATION.md:923-945 | NOT RUN |
| Prove-fix 1 reruns | ctest, ops, np2, vkc, mtp, c18, census, smoke | IMPLEMENTATION.md:1907 | NOT RUN |
| Final-fix 2 reruns | RADV `test-sparse-attn-vk dense reason vpack race`; HIP / RADV `test-sparse-attn-rpc-view`; HIP / RADV op tests (the at_end tail poison on ROCm0); ctest in build and build-vk | IMPLEMENTATION.md:2059 | NOT RUN |
| F3 reruns | `hip_nanscore_before` FAIL / `_after` pass; `hip_shape_before` FAIL / `_after` OK; `hiprpc_before` FAIL part (1) / `_after` OK; the ROCm0 and RADV SPARSE_ATTN / PLAN suites; the HIP, HIP + RPC and RADV ctest subsets (`F3/gpu_rerun.sh`) | IMPLEMENTATION.md:2164 | NOT RUN |

## 7. Known issues (deferred: TODO-LATER.md; reported, not counted in pass)

None of them could be re-measured on 574689270, because of the GPU lock. Each one's last measurement is the one cited in TODO-LATER.md and PROOF-initial §12. Review-r1 §6 records the same states.

| # | item | last measurement | status on 574689270 |
|---|---|---|---|
| 1 | Q1 direct form, q4_0 / bf16 | stage 2: q4_0 2.082e-3 > band 1.441e-3; bf16 1.250e-4 > 1.032e-4 | NOT RUN; last FAIL |
| 2 | AT-D2 `-ngl 0` CPU perf | new / PR219 at pp512 0.69-0.71, tg32 0.89 (TODO-LATER) | NOT RUN; last FAIL. 108d6ce00 adds one `ggml_sparse_attn_shape_why` call to the CPU op (`ggml-cpu/sparse-attn.cpp`, +5 lines). Its cost is UNPROVEN, but it runs once per node, not per row. |
| 3 | AT-B1 Vulkan q8_0 / q4_0 < 0.95 × f16 | 0.921 / 0.916 | NOT RUN; last FAIL |
| 4 | AT-B5 Vulkan MTP f16 +12 MiB | not re-measured since W7 | NOT RUN |
| 5 | AT-D1 kind-4 wording | new 12, PR219 / W0 0 | NOT RUN; the criterion is not met as worded |
| 6 | M1 iq4_nl MTP | +0.26 MiB after W11 (cleared in PROOF-initial) | NOT RUN; last PASS |
| 7 | `-nkvo` copy route: four d0 cells, bf16 | bf16 partially measured; the d0 cells unresolved | NOT RUN; last open |
| 8 | Q4 thin f16-KV margins | +6.3e-6 / +2.0e-6 | NOT RUN; last PASS (thin) |
| 9 | Q7 MTP draft acceptance | stage 1, new − PR219 for default / q8_0 / q4_0 / bf16: −1.41 / −1.88 / −0.28 / −1.44 pp; stage 2: +1.28 / +0.93 / +1.17 / +1.12 pp; LCB −1.87 to −3.83 against the gate LCB ≥ −1 pp (review-r1 §6) | NOT RUN; last FAIL (deferred) |

## 8. Commands run in this step (all on strix-002)
1. `uptime; flock -n ~/gpu.lock true && echo LOCK_FREE || echo LOCK_HELD; ps -eo pid,user,stat,etime,cmd | grep -E "llama|test-|flock"; cat /sys/class/drm/card*/device/mem_info_{gtt,vram}_used; free -g; git rev-parse HEAD; git status --short | wc -l; git log --oneline -15; git merge-base HEAD origin/master`
2. `grep -E "State|SigPnd|ShdPnd" /proc/4041225/status; cat /proc/4041225/wchan; lslocks | grep gpu.lock; journalctl -k --since "2026-11-10 14:50"; sudo -n true`
3. Lock wait in the background: `timeout 3600 flock ~/gpu.lock -c 'date "+GOT_LOCK %F %T"'` → `R2/lockwait.log`
4. `R2/nogpu.sh` → `R2/nogpu.out`. It runs:
   - the 5 incremental builds;
   - the CPU and RPC ctest subsets;
   - the CPU op suite × 2;
   - the lavapipe op suites (SPARSE_ATTN / PLAN × 2, nanscore, QSA_*, GET_ROWS / SET_ROWS / CPY);
   - the lavapipe unit tests;
   - te-rows × 5;
   - the lavapipe build-vk ctest.
5. Outcome hashes: `grep -E "^\s+(SPARSE_ATTN|QSA)" <log> | sed 's/\x1b\[[0-9;]*m//g' | sha256sum`, plus `grep "compiled keys"`.
6. te-rows abort rate and backtrace:
   - 10 × `build-cpu/bin/test-qsa-te-rows`;
   - 10 × the lavapipe run (runs 6-15);
   - `R2/lvp.sh gdb -q -batch -ex run -ex bt -ex "info threads" build-vk/bin/test-qsa-te-rows` → `R2/lvp_terows_gdb_1.log`.
7. Code reading:
   - `git diff --stat ed4bd3ea5..HEAD`;
   - `git log --format="%h %an <%ae> | %s%n%(trailers:only)" ed4bd3ea5..HEAD`;
   - `git log -1 --format="%an <%ae>" 94e086847`;
   - `git show` of d7e4656d1, 108d6ce00, 195dadc10, a22777440 and a82bdb924 (stat);
   - `ggml.c` lines 6640-6770;
   - `grep -n "n_swa|scalar" src/models/qwen4exp.cpp`;
   - `git show ea5285ddf:ggml/src/ggml-rpc/ggml-rpc.cpp | grep -A6 "version mismatch"`.
8. `git merge-base --is-ancestor 13a279bfa ea5285ddf`; `git rev-list --count` for `ea5285ddf..HEAD` (83) and `ea5285ddf..origin/master` (35).

## 9. Side effects and cleanup
- No AMD GPU command was run, and no process was signalled. pids 4041225 and 4041224 were not touched.
- No systemd unit was touched, strix-001 was not used, and nothing was pushed or posted.
- `~/dev/wt_rewrite`: only incremental no-op builds ran. There are no source changes, and `git status` is clean.
- `R2` holds logs only (1.7 MB). The gdb run left no core file in the tree: `timeout` reported "dumped core", which goes to the system core handler, not to `R2`. No file over 1 GB was created.
