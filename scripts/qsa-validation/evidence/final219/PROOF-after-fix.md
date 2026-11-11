# PROOF (after fix): qsa-rewrite head 97865cacb on strix-002

**Verdict: pass = false.** The GPU on strix-002 is still unusable. Not one GPU check could run on the fixed head 97865cacb. The GPU lock `~/gpu.lock` is still held by the Vulkan `llama-perplexity` process that hung during the initial Prove (PROOF-initial §13). That process is in state D, has SIGKILL pending and holds 91.6 GB GTT. No sudo is available, and the rules forbid killing processes this agent did not start. Every GPU command must take `flock ~/gpu.lock`, so each one would block forever.

What this step did prove, without the GPU:
- the three build trees are at HEAD, with 0 new warnings;
- the QSA ctest subset passes on the CPU-only build;
- the AT-C18 checker re-scores the old logs as PASS;
- a code reading of the 4 fix commits.

Every check that items 1-8 require on the GPU is **NOT RUN**, and so it is not passing. The results of PROOF-initial were measured on head `1dbee4d9a`. They are listed in §5 as the previous head's evidence only, each marked with whether the fix commits can affect it.

**Conventions.**
- `PA` = `strix-002:~/dev/wt_lab/prove_after` holds this step's logs.
- `PV` = `strix-002:~/dev/wt_lab/prove` holds the initial Prove's logs.
- `PF` = `strix-002:~/dev/wt_lab/prove_fix` holds the rerun harness that Prove-fix 1 prepared.
- All ssh output had its login banner filtered out.

## 1. Blocking condition (evidence)

| fact | command | output |
|---|---|---|
| HEAD | `git -C ~/dev/wt_rewrite rev-parse HEAD` | `97865cacbad9076389de73016c04e9bb97e27fdc` |
| worktree clean | `git status --short \| wc -l` | `0` |
| merge-base with origin/master | `git merge-base HEAD origin/master` | `ea5285ddf…`. The rewrite is not based on a newer master, so no current-master rebuild is needed. The base arm is PV/base_ea5, as in PROOF-initial. |
| lock state, 13:33 | `flock -n ~/gpu.lock true && echo LOCK_FREE \|\| echo LOCK_HELD` | `LOCK_HELD` |
| holder | `ps -eo pid,stat,etime,wchan:30,cmd` | `4041225 D 01:39:06 drm_suballoc_insert …/wt_rewrite/build-vk/bin/llama-perplexity … -ctk bf16 -ctv bf16 --kl-divergence-base …/prove/c1/det_model/x_2.kld`. Its parent is `4041224 S timeout 3600 …`. |
| signals | `grep -E "State\|SigPnd\|ShdPnd" /proc/4041225/status` | `State: D (disk sleep)`, `SigPnd 0000000000000100` (SIGKILL pending), `ShdPnd 0000000000004000` (SIGTERM) |
| GTT held | `cat /sys/class/drm/card*/device/mem_info_gtt_used` | `91631312896` (unchanged since PROOF-initial §13) |
| host memory | `free -g` | 124 total, 92 used, 31 available |
| kernel log since 11:50 | `journalctl -k --since "2026-11-10 11:50"` | one line: `11:55:28 … [drm:amdgpu_gem_va_update_vm [amdgpu]] *ERROR* Couldn't update BO_VA (-12)`. No GPU reset or recovery was logged after it. |
| root | `sudo -n true` | `sudo: a password is required` |
| wait | a poll loop with `flock -n` every 10 s for 170 iterations (≈ 28 min, from 13:34) | see §1.1 |

### 1.1 Lock wait result
The lock never freed:
- The poll loop ran from 13:34 and ended `STILL_HELD 14:02:25`, `State: D (disk sleep)`.
- Three blocking waits followed, each `ssh strix-002 'timeout N flock ~/gpu.lock true'`, a lock-only wait with no GPU action. They ended `STILL_HELD 13:49:06`, `STILL_HELD 13:58:50` and `STILL_HELD 14:05:37`.
- At 14:05:37, process 4041225 was still `D` with etime `02:11:10`, and `mem_info_gtt_used` was still `91631312896`.
- Total wait: about 32 minutes. The hang has lasted 2 h 10 min since the kernel error at 11:55:28.
- Recovery without a reset is UNPROVEN either way. Nothing in the kernel log shows progress.

**Required action, for the owner:** reboot strix-002 or reset the amdgpu device. This needs root. Then rerun with `bash ~/dev/wt_lab/prove_fix/rerun.sh setup ctest ops np2 vkc mtp c18 census smoke`, followed by the full Prove chains (PROOF-initial §15) on head 97865cacb. The previous Prove agent sent the process a SIGKILL (PROOF-initial §13). This agent sent no signal.

## 2. Checks that ran (no GPU)

### 2.1 Builds at HEAD (compile only; no device opened)
Command: `cd ~/dev/wt_rewrite && cmake --build {build,build-vk,build-cpu} -j 16`. Logs: `PA/rebuild_*.log`.

| tree | rc | `warning:` lines | recompiled | verdict |
|---|---|---|---|---|
| build (HIP, gfx1151) | 0 | 0 | 6 objects: `ggml.c`, `llama.cpp`, `llama-context.cpp`, build-info, ui, license. The commits were written after the last build, so the mtimes moved. | PASS |
| build-vk | 0 | 0 | the same 6 | PASS |
| build-cpu (`GGML_HIP/VULKAN/CUDA=OFF`, `CMakeCache.txt`) | 0 | 63 | 195 (every target) | PASS. All 63 warnings are pre-existing. The 8 in `tests/test-backend-ops.cpp` (lines 463, 466, 9386, 11455, 17332, 17339…) blame to commits that are ancestors of the base ea5285ddf (`git blame` + `merge-base --is-ancestor`, "in-base" for each). The rest are in `tools/mtmd/clip-graph.h` (45, `ggml_scale_mode \| ggml_scale_flag`), `tools/server/*` and `tests/gguf-model-data.cpp` (`u8path`), which the rewrite does not touch. |

So `build/bin` and `build-vk/bin` now hold the binaries of 97865cacb.

### 2.2 QSA ctest subset, CPU-only build
Command: `cd build-cpu && ctest -R "test-qsa|test-sparse-attn" -j1 --output-on-failure`. Log: `PA/ctest_cpu.log`. **rc 0, 12/12 passed.**
- 10 tests ran for real: `test-qsa-prefix`, `-fast`, `-recover`, `-allvis-rows`, `-te-rows`, `-visibility`, `-resplit` (32.6 s), `-reserve-view`, `-idx-type`, and `test-sparse-attn-plan-oracle`.
- 2 skipped on this build:
  - `test-sparse-attn-vk` printed `no Vulkan backend: skipped`;
  - `test-sparse-attn-rot` printed `no ROCm backend: skipped`.
  - These two pass vacuously here. They need the GPU builds: **NOT RUN on HIP / Vulkan**.
- `test-qsa-reserve-view` (new in a16582997):
  - `n_seq_max=2 unified=0`: full 2 streams, pass B 2, one-stream 1.
  - `n_seq_max=4 unified=0`: full 4, pass B 4, one-stream 1.
  - `n_seq_max=2 unified=1`: full 1, pass B 1, one-stream 1.
  - `n_seq_max=1`: 1 / 1 / 1.
  - Result: `test-qsa-reserve-view: OK`.

### 2.3 AT-C18 re-score on the old logs (no GPU)
- `python3 scripts/qsa-validation/harness/w7/at_c18_an.py ~/dev/wt_lab/prove/c18` with the HEAD checker gives **`AT-C18 PASS`**. Every run prints `inspected >= required: True`.
- The 1dbee4d9a checker (`git show 1dbee4d9a:…/at_c18_an.py`) on the same logs gives `AT-C18 FAIL (4)`.
- Limit: these logs come from the old head. The new guard line `route=cache-only … guard=0/0` (f12eb329d) has never been printed by a real run. **The AT-C18 rerun is NOT RUN.** This re-score is partial evidence, and it does not on its own pass the item.

### 2.4 Code reading of the 4 fix commits (`git show 1dbee4d9a..HEAD`)

| commit | change (source) | what it can affect |
|---|---|---|
| f12eb329d | guard log line for a cache-only graph (`src/llama-context.cpp:1107-1117`) and the checker | log text only; no compute |
| a16582997 | four `1s-*` reserve passes when `!kv_unified && n_seq_max > 1` (`llama-context.cpp` sched_reserve), plus the `n_stream` argument of `llama_kv_cache_context` / the hybrid(-idx) contexts | reserve size and fit probe of split multi-sequence contexts only (`-np ≥ 2` without `--kv-unified`). Unified and single-sequence contexts get no new pass (per the commit text and the test above). |
| 4552df2a4 | `nq_max[q8_0][135168 class]` 96 → 64 (`ggml/src/ggml-cuda/sparse-attn-attend-g1.cu:68-72`) | HIP layers with V q8_0, n_kv ≥ 94118 and 65-96 queries per stream now take G4 (`:76-77`). The table now equals the CPP line of `scripts/qsa-validation/summaries/prove_ata.md:282` in all 16 cells (read in this step). |
| 97865cacb | the gated attend writes to `ggml_cuda_pool_alloc` scratch plus one `cudaMemcpyAsync` when its output overlaps a read, instead of refusing the fusion (`ggml/src/ggml-cuda/ggml-cuda.cu` ggml_cuda_try_fuse, the hunk at 6804-6841) | every HIP graph where gallocr overlaps the MUL output with a read: fusion rate, pool high-water (M2), perf. The kernel and its arguments are unchanged for the non-overlap case. That the fused output equals the unfused SIGMOID/MUL output within the old tolerance is a claim of W3. Under HIP graph capture the pool alloc is new on this path: **UNPROVEN on the GPU.** |

## 3. Scorecard on head 97865cacb

| # | check | verdict |
|---|---|---|
| 0 | builds HIP / Vulkan / CPU at HEAD | PASS (§2.1) |
| 1 | test-backend-ops per op (SPARSE_ATTN_PLAN, SPARSE_ATTN, FLASH_ATTN_EXT, QSA_EXPAND/SCORE/STRIP, GET_ROWS/SET_ROWS/CPY) on ROCm0 (graphs on / off, each forcing), Vulkan0 (each forcing), lavapipe; census coverage per kernel key | **NOT RUN** (GPU lock) |
| 1 | unsupported shapes asserted unsupported (`--fail-unsupported`, expected list) | **NOT RUN** |
| 1 | CPU oracle: test-backend-ops vs CPU | **NOT RUN**. `test-sparse-attn-plan-oracle` (CPU) PASS (§2.2). |
| 1 | determinism: in-process memcmp, model-level logit hash run twice (HIP 12 cells, Vulkan incl. bf16 / mixed) | **NOT RUN** |
| 1 | ctest QSA on HIP and Vulkan builds (12 per tree) | **NOT RUN**. CPU build 12/12 (2 vacuous skips) PASS (§2.2). |
| 2 | HIP route census, EXPECT rows, 0 masked, 0 CPU; Vulkan placement census | **NOT RUN** |
| 2 | reserve = runtime (strict) incl. the PROOF-initial failures: HIP `-np 2` split ub8192 (fixed in code by a16582997) and Vulkan ub16 (not fixed, IMPLEMENTATION Prove-fix 1 "Not fixed" 2.2) | **NOT RUN**. The Vulkan ub16 failure is unfixed in code, so it is expected to stay FAIL (UNPROVEN until rerun). |
| 2 | MTP census, gated share ≥ 0.9 (fixed in code by 97865cacb) | **NOT RUN** |
| 2 | guard / AT-C18 | checker re-score PASS on the old logs (§2.3). New-head run **NOT RUN**. |
| 3 | pinned KLD Q1/Q3/Q4/Q8/Q11/Q12 (f16, q8_0, q4_0, bf16, q8_0K/f16V, f16K/q8_0V), f16 bit-identity, unpinned catastrophe, depth, Q1-vk | **NOT RUN** |
| 3 | Q12 coarse q8_0 (FAIL in PROOF-initial, not fixed) | **NOT RUN**; open |
| 3 | Q10 control and image p3 seq1 (FAIL in PROOF-initial, not fixed) | **NOT RUN**; open |
| 3 | Q5 op-local bands | **NOT RUN** |
| 4 | perf P1-P6, text + random, ub 512/1024/4096, d0/32k/131k, ~300k, deepest; tg at depth; MTP cells; stall; compute buffer; peak VRAM | **NOT RUN** |
| 5 | images A/B/C × types, `-np 2` split with images, mixed q8_0/f16, image understanding, MTP | **NOT RUN** |
| 6 | Vulkan build | PASS (§2.1) |
| 6 | Vulkan op tests, q8_0 depth run | **NOT RUN** |
| 7 | deferred acceptance tests AT-A, Lab B (VB1-VB6), Lab C (C-5, C-17, C-12, C-16), Lab D (D1-D5), AT-W12-BANDS, AT-Q10Q6, AT-W7PERF, AT-B1/B3/B7 | **NOT RUN** on this head (§4) |
| 8 | Moved-to-Prove checks W7-W13 | **NOT RUN** on this head (§4) |

## 4. Deferred acceptance tests (item 7) and Moved-to-Prove (item 8) on head 97865cacb

Every entry needs GPU runs, so every entry is **NOT RUN**. For each one, the table gives its criterion (from the task text and IMPLEMENTATION.md) and its state.

| test | criterion (short) | state on 97865cacb |
|---|---|---|
| AT-A (Lab A) | the HIP G proc encodes the fitted (n_q, n_kv, T_v) rule; e2e f16 ub512 pp512 d131k neutral or better (§16, half-width ≤ 1 %) | table = fitted CPP line in 16/16 cells (code reading, §2.4). Kernel re-fit on this head and the e2e cell **NOT RUN** |
| Lab B VB1 (AT-B1) | f16 new/W0 neutral or better; q8_0, bf16 and S_fast pairs ≥ 0.95 × f16 | **NOT RUN** (q8_0 / q4_0 miss is TODO-LATER #3, deferred) |
| VB2 (AT-B2) | new f16 op ≤ 1.10 × the in-build sel_cm | **NOT APPLICABLE as written**: sel_cm was deleted in W13 (740d48097). Needs an owner ruling. Counted as not passing. |
| VB3 | ship the faster of direct / staged; diagnostic op gate | **NOT RUN** |
| VB4 | sparse ≥ 0.98 × dense per cell; 0 CPU nodes; decode at depth ≥ W0 | **NOT RUN** |
| VB5 | M1: T ≤ f16 + 1 MiB; f16 ≤ W0 | **NOT RUN** (MTP +12 MiB cell is TODO-LATER #4) |
| VB6 | Q1-vk ≤ B_vk | **NOT RUN** |
| C-5 | 0 strict violations; reserved partial ≥ §7 formula | **NOT RUN** |
| C-17 | new f16 ≤ PR219 reserve + pool high-water; partials scale with n_stream | **NOT RUN** (PR219 fill needs `--rope-scale 2`, Prove-fix 1 2.13) |
| C-12 | new reserve ≤ PR219 + bound; runtime ≤ reserve; answers 2749 / 12749 | **NOT RUN**. a16582997 grows the split `-np 2` reserve, so the np2 rows must be re-measured. |
| C-16 | strip-view census > 0; no-share − share ≥ ½ the PR219 delta | **NOT RUN** |
| D1 | kind-4 count = W0; U16 memcmp; OVL ratio-of-ratios ≥ 0 | **NOT RUN** (criterion wording is TODO-LATER #5) |
| D2 / D3 | placement 100 % on the K device; guard 0; KLD ≤ R1 value | **NOT RUN** (CPU perf is TODO-LATER #2) |
| D4 / D5 | 0 GPU-placed nodes under -nkvo; PPL within 3 × SE; -fa auto logged | **NOT RUN** |
| AT-W12-BANDS | bands frozen in §15.0 before scoring; self-replay 0 | the bands are a W12 artifact, already frozen in PLAN-v3 §15.0 (PROOF-initial §10.5). The self-replay on this head is **NOT RUN**. |
| AT-Q10Q6 | calibration valid; Q10 control, Q10 cells, Q6 | calibration from PROOF-initial (PR219-side, independent of the head) stays valid. The gates on this head are **NOT RUN**. The Q10 control was FAIL and is unfixed. |
| AT-W7PERF | shortcut ≥ 0.98 dense per band; -nkvo ≥ 0.98 PR219; MTP tg P5 | **NOT RUN** |
| AT-B1 ext / AT-B3 pack / AT-B7 | S_fast ≥ 0.95 × f16 every cell; vpack rule; lavapipe + RADV sg32 0 failures | **NOT RUN** |
| Moved-to-Prove W7 (Q3-B q4_0 ub256, Q11c, Q12 -ngl 0, P2/P3, NDB, NPIN, P5 + MTP, P4, P6N, D1 value, M2/M3, AT-B4 e2e, AT-B1 cells, U30, phase K) | as in IMPLEMENTATION.md:923-943 | **NOT RUN** |
| Moved-to-Prove W8 (AT-Q10Q6, Q10, Q6) | IMPLEMENTATION.md:1052-1061 | **NOT RUN** |
| Moved-to-Prove W9 (AT-C12 grid + image-slot KLD, M5, AT-C17 / M6, AT-C16b, M4 b/c, C15 ub1024, np2 shared prefix, U27) | IMPLEMENTATION.md:1241-1255 | **NOT RUN** |
| Moved-to-Prove W10 (U6 OP_PROF A/B, rotated-op time, Q5, Q1-A/B, P1-P6, Vulkan in-op H_q quality) | IMPLEMENTATION.md:1461-1470 | **NOT RUN** |
| Moved-to-Prove W11 (Q4, M1 iq4_nl MTP, Vulkan q8_0 indexer, MTP smoke) | IMPLEMENTATION.md:1586-1593 | **NOT RUN** |
| Moved-to-Prove W13 (full campaign, AT-C16b, VGPR after rebase, merge-tree) | IMPLEMENTATION.md:1754-1762 | merge-tree / VGPR: no rebase was made (merge-base still ea5285ddf, §1). The rest is **NOT RUN**. |

## 5. Previous head's evidence (1dbee4d9a, PROOF-initial.md), for context only

These results were measured on the head before the 4 fix commits. They are **not counted** as passes for 97865cacb. The table marks, from the code reading in §2.4, whether a fix commit can change each one.

| PROOF-initial result on 1dbee4d9a | can a fix commit change it? |
|---|---|
| op tests 100 % (HIP 1390/1390 per forcing, Vulkan 744/744, lavapipe) | 97865cacb: no. test-backend-ops runs SPARSE_ATTN alone, without the gate-fusion pattern (UNPROVEN that no case builds that pattern). 4552df2a4: only if a case has V q8_0 at n_kv ≥ 94118 with 65-96 queries (UNPROVEN). |
| HIP census 24/24 EXPECT PASS, Vulkan placement 18/18 | 4552df2a4 changes the expected rows only at n_kv ≥ 94118 (the census runs are c8192: no) |
| strict FAILs: HIP np2 split ub8192, Vulkan ub16 | a16582997 targets the first; the second is unfixed |
| MTP gated share 56/96 FAIL | 97865cacb targets it |
| Q1/Q3/Q4/Q8/Q11/Q12 pinned PASS, Q1-vk 6/6 | 97865cacb (fusion of the previously refused attends; claimed bit-identical to the fused kernel, UNPROVEN versus the unfused path within the KLD band); 4552df2a4 for q8_0 V at depth (Q3 is c32768: no) |
| perf P1-P5 (23/24, 24/24, 36/36 …), images 18/18, M1 20/20 | 97865cacb (an extra device copy where the fusion was refused before, and the fusion where it was not); a16582997 (np2 split reserve) |
| FAILs not targeted by any fix: Q10 control and p3 seq1, Q12 coarse q8_0, P5 MTP f16 d32k 0.898, stall 1/3232, unresolved §16 cells, AT-B2, AT-C17 high-water, Vulkan depth / AT-B1 / AT-B4 / AT-B5 | still open |

## 6. Known issues (deferred: TODO-LATER.md; reported, not counted in pass)

None of these could be re-measured on 97865cacb (GPU lock). The last measurement of each, on 1dbee4d9a, is from PROOF-initial §12 and TODO-LATER.md.

| # | item | last measurement (1dbee4d9a) | status on 97865cacb |
|---|---|---|---|
| 1 | Q1 direct form q4_0 / bf16 | stage 2 q4_0 2.082e-3 > 1.441e-3; bf16 1.250e-4 > 1.032e-4 | NOT RUN; last FAIL |
| 2 | AT-D2 `-ngl 0` CPU perf | f16 pp512 0.700, tg32 0.804; q8_0 0.706 / 0.923 vs PR219 | NOT RUN; last FAIL |
| 3 | AT-B1 Vulkan q8_0 / q4_0 < 0.95 × f16 | W7: 0.921 / 0.916; op level 2.22× / 2.14× f16 | NOT RUN; last FAIL |
| 4 | AT-B5 Vulkan MTP f16 +12 MiB | not re-measured in PROOF-initial | NOT RUN |
| 5 | AT-D1 kind-4 wording | new 12, PR219 0 | NOT RUN; criterion not met as worded |
| 6 | M1 iq4_nl MTP | +0.26 MiB (cleared) | NOT RUN; last PASS |
| 7 | -nkvo copy route open cells, bf16 | bf16 10 + 2 PASS, 2 unresolved; d0 cells unresolved | NOT RUN; last open |
| 8 | Q4 thin f16-KV margins | +6.3e-6 / +2.0e-6 | NOT RUN; last PASS (thin) |
| 9 | Q7 MTP draft acceptance | stage 1 mean new − PR219 default / q8_0 / q4_0 / bf16 = −1.41 / −1.88 / −0.28 / −1.44 pp (LCB −2.97 / −3.83 / −1.74 / −3.25); stage 2 +1.28 / +0.93 / +1.17 / +1.12 pp (LCB −1.91 / −2.45 / −1.87 / −2.15); gate LCB ≥ −1 pp | NOT RUN; last FAIL (deferred) |

## 7. Commands run in this step (all on strix-002; no GPU command; no lock taken)
1. `uptime; git log --oneline -8; git status --short; flock -n ~/gpu.lock true …; ps -eo …; cat …/mem_info_gtt_used; free -g`
2. `journalctl -k --since "2026-11-10 11:50"`; `grep State|SigPnd|ShdPnd /proc/4041225/status`; `sudo -n true`
3. Lock poll: `for i in $(seq 1 170); do flock -n ~/gpu.lock true && { echo LOCK_FREE; exit; }; sleep 10; done`. `flock -n` with the command `true` takes no GPU action, and it would have released the lock at once.
4. `cmake --build build -j 16`, `build-vk`, `build-cpu` → `PA/rebuild_*.log`
5. `git blame -L <line> --porcelain HEAD -- tests/test-backend-ops.cpp` + `git merge-base --is-ancestor <c> ea5285ddf` for the warning lines
6. `ctest -R "test-qsa|test-sparse-attn" -j1 --output-on-failure` in build-cpu → `PA/ctest_cpu.log`; direct runs of `test-sparse-attn-vk`, `test-sparse-attn-rot`, `test-qsa-reserve-view`, `test-qsa-idx-type`
7. `python3 scripts/qsa-validation/harness/w7/at_c18_an.py ~/dev/wt_lab/prove/c18`, and the 1dbee4d9a version of the same script
8. `git show 1dbee4d9a..HEAD`, `sed -n 60,80p ggml/src/ggml-cuda/sparse-attn-attend-g1.cu`, `grep CPP scripts/qsa-validation/summaries/prove_ata.md`

## 8. Side effects and cleanup
- No GPU command was run, and no process was signalled. The hung process 4041225 and its parent 4041224 were not touched.
- No systemd unit was touched, strix-001 was not used, and nothing was pushed or posted.
- `~/dev/wt_rewrite`: only the build trees changed (an incremental rebuild). There are no source changes, and `git status` is clean.
- No file over 1 GB was created. `PA` holds only logs.
