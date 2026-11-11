# PROOF (review-r1): qsa-rewrite head ed4bd3ea5 on strix-002

**Verdict: pass = false.** The AMD GPU on strix-002 is still unusable, so no GPU check could run on head `ed4bd3ea5`.
- The GPU lock `~/gpu.lock` is still held by the hung Vulkan `llama-perplexity` that the initial Prove started: pid 4041225, state D, SIGKILL pending, 91.6 GB GTT held.
- Every GPU command must take `flock ~/gpu.lock`, so each one would block forever.
- No sudo is available, and the rules forbid signalling a process this agent did not start. The process is in D state with SIGKILL already pending, so a signal would not end it anyway.

The step proved what it could without the GPU:
- all four build trees are at HEAD;
- the CPU-only QSA ctest subset passes 15/15;
- lavapipe (a CPU Vulkan device) passes every changed op against the CPU oracle, run twice with identical per-case outcomes;
- the lavapipe Vulkan unit tests and the RPC-view test pass;
- the 9 commits since the last proof (`97865cacb..ed4bd3ea5`) were read.

Every check that items 1-8 need on ROCm0 or RADV is **NOT RUN**, so none of them passes. That covers perf, pinned KLD, images, MTP, census on the GPU, reserve = runtime on the GPU, the Vulkan depth run, and every deferred acceptance test.

**Conventions.**
- `R1` = `strix-002:~/dev/wt_lab/prove_r1`, which holds this step's logs.
- `PV` = `~/dev/wt_lab/prove`, the initial Prove (head 1dbee4d9a).
- `F2` = `~/dev/wt_rewrite_logs/F2`, the Final-fix 2 logs.
- All ssh output had its login banner filtered out.

## 1. Blocking condition (evidence)

| fact | command | output |
|---|---|---|
| HEAD | `git -C ~/dev/wt_rewrite rev-parse HEAD` | `ed4bd3ea5cf62ea9d43e452303b33fcc380d8d2e` |
| worktree clean | `git status --short \| wc -l` | `0` |
| new commits since the last proof | `git log --oneline 97865cacb..HEAD` | 9 commits, 9e84db9e7 … ed4bd3ea5 (IMPLEMENTATION.md "Final-fix 2") |
| merge-base with origin/master | `git merge-base HEAD origin/master` | `ea5285ddf`. origin/master is `62a3f4144` (35 commits past ea5285ddf). The rewrite is not on a newer master, so no current-master rebuild is needed. |
| base arm | `git merge-base --is-ancestor 13a279bfa ea5285ddf` | true, 8 commits apart. `~/dev/wt_pr219v2_base` (13a279bfa) is older than the rewrite's base, so the base arm is `PV/base_ea5` (built: `PV/base_ea5/build/bin/llama-bench` exists), as in PROOF-initial. |
| lock at 15:24 | `flock -n ~/gpu.lock true && echo LOCK_FREE \|\| echo LOCK_HELD` | `LOCK_HELD` |
| holder | `ps -eo pid,stat,etime,cmd` | `4041225 D 03:30:17 …/wt_rewrite/build-vk/bin/llama-perplexity … -ctk bf16 -ctv bf16 --kl-divergence-base …/prove/c1/det_model/x_2.kld`, child of `4041224 S timeout 3600 …` |
| at 15:37 | `ps -o pid,stat,etime,wchan:25 -p 4041225,4041224` | `4041225 D 03:43:09 drm_suballoc_insert`; `4041224 S 03:43:09 sigsuspend.isra.0` |
| signals | `grep -E "State\|SigPnd" /proc/4041225/status` | `State: D (disk sleep)`, `SigPnd: 0000000000000100` (SIGKILL pending) |
| GTT held | `cat /sys/class/drm/card*/device/mem_info_gtt_used` | `91631312896`, unchanged since PROOF-initial §13 |
| host memory | `free -g` | 124 total, 93 used, 31 available |
| kernel log | `journalctl -k --since "2026-11-10 13:00"` | last lines are the 14:56:36 segfault of the Final-fix 2 "before" tree's `test-sparse-attn-rpc-view` on lavapipe (the expected bug-8 demonstration, F2 `before_lvp_rpcview_gdb.log`); since 14:57: `-- No entries --`. No GPU reset or recovery is logged. |
| root | `sudo -n true` | `sudo: a password is required` |
| lock wait | `timeout 2400 flock ~/gpu.lock -c 'date "+GOT_LOCK …"'`, a wait on the lock only (no GPU action), from 15:25:37 | `rc=124` (timeout) at `end 2026-11-10 16:05:37`: the lock never freed in 40 min. At 16:05:41 pid 4041225 was still `D 04:11:14 drm_suballoc_insert`, GTT was still `91631312896`, and the kernel log since 14:57 was still `-- No entries --` (`R1/lockwait.log`). |

**Required action, for the owner.** Reboot strix-002 or reset the amdgpu device (this needs root). Then run:
1. `bash ~/dev/wt_lab/final_fix/rerun_gpu.sh`: the RADV and HIP checks of the 9 new commits (criteria in IMPLEMENTATION.md "Final-fix 2 / GPU reruns").
2. `bash ~/dev/wt_lab/prove_fix/rerun.sh setup ctest ops np2 vkc mtp c18 census smoke`, retargeted to ed4bd3ea5.
3. The full Prove chains of PROOF-initial §15.

## 2. Checks that ran (no AMD GPU)

### 2.1 Builds at HEAD (compile only)
Command: `cd ~/dev/wt_rewrite && cmake --build <tree> -j 16`. Logs: `R1/rebuild_<tree>.log`.

| tree | config (CMakeCache) | rc | `warning:` lines | verdict |
|---|---|---|---|---|
| build | HIP gfx1151, rule flags | 0 | 0 (incremental: 322 lines) | PASS |
| build-vk | `GGML_VULKAN=ON`, HIP/CUDA/RPC OFF | 0 | 0 | PASS |
| build-cpu | HIP/VULKAN/CUDA/RPC OFF | 0 | 0 | PASS |
| build-rpc | `GGML_RPC=ON`, CPU only | 0 | 72 (321 targets rebuilt: the tree was stale) | PASS. Warnings per file: `tools/mtmd/clip-graph.h` 45, `tests/test-backend-ops.cpp` 8, `common/hf-cache.cpp` 4, `tests/gguf-model-data.cpp` 3, `common/download.cpp` 3, `tools/server/*` 4, `tools/mtmd/models/{qwen3vl,ling3vl}.cpp` 2. These are the same pre-existing set as PROOF-after-fix §2.1. The 9 new commits touch none of these warning lines (the 8 test-backend-ops lines are the in-base set of PROOF-after-fix §2.1). |

### 2.2 QSA ctest subset, CPU-only build
- Command: `cd build-cpu && ctest -R "test-qsa|test-sparse-attn" -j1 --output-on-failure`. Log: `R1/ctest_cpu.log`.
- Result: **rc 0, 15/15 passed.** The new tests pass: `test-qsa-supported-types` (bug 5), `test-sparse-attn-rpc-view` (bug 8) and `test-sparse-attn-gate-scratch` (bug 9). `test-qsa-resplit` (bug 6 decode phase) took 32.92 s.
- `test-sparse-attn-vk` and `test-sparse-attn-rot` pass vacuously on this build (no Vulkan or ROCm backend). Their device runs are §2.3 for lavapipe; on HIP and RADV they are **NOT RUN**.

### 2.3 lavapipe runs (Vulkan on a CPU device)
- Harness: `R1/lvp.sh` and `R1/lvp_batch.sh`, logs `R1/lvp_*.log`.
- Environment:
  - `VK_ICD_FILENAMES=~/dev/wt_lab/w6/lvp43/icd.json`, `LD_LIBRARY_PATH=…/lvp43/usr/lib64`: Fedora 43 Mesa 25.3.6, the lavapipe of PROOF-initial §3.1;
  - `GGML_VK_VISIBLE_DEVICES=0`, `VK_LOADER_LAYERS_DISABLE=~implicit~`, `NODEVICE_SELECT=1`;
  - `timeout 2400`.
- Every log prints `ggml_vulkan: 0 = llvmpipe (LLVM 21.1.8, 256 bits) … warp size: 8 | shared memory: 32768 | matrix cores: none` (2 lines per log, `grep -c llvmpipe` = 2).
- **Not a GPU job.** 20 s into the first SPARSE_ATTN run, `/proc/<pid>/fd` of test-backend-ops held **0** `/dev/dri` or `/dev/kfd` descriptors. These runs could not take `flock` (the lock is held), and none of them opens the AMD GPU.

test-backend-ops, one op per process, against the CPU backend (the CPU oracle):

| op | flags | result | "not supported" | census coverage (kernel path) |
|---|---|---|---|---|
| SPARSE_ATTN run 1 | `--fail-unsupported` | **736/736, rc 0** | 654 (= PROOF-initial §3.1 and F2; with `--fail-unsupported` each one is on the expected list, or the run would fail) | `81/81 compiled keys of {sa_attend/,sa_fixup/,sa_generic/,sa_merge,sa_pack/,sa_portable/} launched (336 excluded by a forcing switch): OK` |
| SPARSE_ATTN run 2 | same | **736/736, rc 0** | 654 | 81/81 OK |
| SPARSE_ATTN_PLAN run 1 / run 2 | `--fail-unsupported` | **92/92, rc 0** each | 93 each | `1/1 compiled keys of {sa_plan/} launched: OK` |
| QSA_EXPAND / QSA_SCORE / QSA_STRIP | | 56/56, 13/13, 6/6, rc 0 | 0 | |
| GET_ROWS / SET_ROWS / CPY | | 643/643, 439/439, 307/307, rc 0 | 12 / 600 / 514 (no `--fail-unsupported`; lavapipe type coverage) | |
| QSA_PREFILL / QSA_DECODE (deleted in W13) | | 0/0 cases each | | |

- **Determinism.** Within one process, the suite runs SPARSE_ATTN and SPARSE_ATTN_PLAN twice and memcmps the outputs (`tests/test-backend-ops.cpp:1375-1377, 1757-1772`); every pass above includes that check. Across the two processes, the per-case outcome lines are identical: `sha256` of the colour-stripped case lines is `ba9b5a33d353c188` for both SPARSE_ATTN runs (1376 lines) and `38b1295714c50169` for both PLAN runs (185 lines). As PROOF-initial §3.3 established, a cross-process output hash is not a valid test here, because the inputs are drawn fresh in each process.
- **Kernel-path statement.** On lavapipe, the census shows which keys ran: the portable tier and its siblings (`sa_portable/`, `sa_generic/`, `sa_pack/`, `sa_fixup/`, `sa_merge`). The cm-tier keys are excluded, with the reason `another tier runs on this device, or in this process` (e.g. `sa_attend/vk/cm/direct/f16/f16`). The RADV cm and scalar tiers and every HIP kernel key are **NOT RUN**.

Unit tests on lavapipe:

| test | result | log |
|---|---|---|
| `test-sparse-attn-vk` (all parts: limits, capability, rot, stress, dense, reason, vpack, race) | `OK: 0 failed checks`, rc 0. `race`: `2621653 reads on the other thread, 0 differ`. `vpack` needs the cm tier, so it is skipped on lavapipe (F2 scorecard). | `R1/lvp_test-sparse-attn-vk.log` |
| `test-sparse-attn-rpc-view` | `27 rows, 0 failed checks: OK`. For example, `q4_0 G 4 n_q 512: attend nbytes 12582912, alloc linked 12582912, server view 12582912`. | `R1/lvp_rpcview.log` |
| `test-qsa-allvis-rows` | `RESULT PASS (failures 0)` | `R1/lvp_allvis.log` |
| `test-qsa-te-rows`, 5 direct runs | 4 rc 0, **1 rc 134** (`corrupted size vs. prev_size while consolidating`) | `R1/lvp_terows_{1..5}.log` |
| `build-vk` ctest subset, `-j1` | **14/15**. `test-qsa-te-rows` was `Subprocess aborted` | `R1/lvp_ctest_vk.log` |

**te-rows abort.** This is the pre-existing lavapipe heap-corruption abort that IMPLEMENTATION.md Final-fix 2 documents:
- its rate was 9/20 on head and 11/20 on a clean `97865cacb` build, so it predates the 9 commits;
- the test uses only TE ops, which the 9 commits do not touch (`git diff --stat 97865cacb..HEAD` has no `qwen4exp-te` file);
- here it aborted in 2 of 6 runs;
- its cause is UNPROVEN, and on RADV it is NOT RUN;
- it is counted as **FAIL / open** for the ctest-on-Vulkan check. It is not a known-deferred item (it is not in TODO-LATER.md).

### 2.4 Code reading of the 9 commits (`git show 97865cacb..ed4bd3ea5`)
Diffstat: 18 files, +871 / −70. All commits are by Gaetan Puleo `<12990773+gaetan-puleo@users.noreply.github.com>` with the `Claude-Session` trailer (`git log --format=… %(trailers:only)`). None of them ports #219 code, so the lack of a Co-authored-by trailer is correct.

| commit | change | GPU-side effect still to prove (NOT RUN) |
|---|---|---|
| 9e84db9e7 | the probe ctx of `qsa_supported_types` holds 2 objects per type | none (host init only). CPU test PASS. |
| 7da229bf8 | NC0 is evaluated only on ubatches with `n_tokens > U` | the tg host cost before and after is UNPROVEN (no timing). The re-split counts are unchanged (F2). |
| b7394c04f | Vulkan `prefer_dense` needs tier cm and FLASH_ATTN_EXT support | RADV cm: the dense cells and the shortcut perf (VB4 / AT-B4) |
| ae956e69a | Vulkan unsupported reasons | none on a supported shape |
| e46cd6c94 | Vulkan V-pack reserve is monotone in n_kv | the vpack-arm reserve grows. Size UNPROVEN (AT-B3, M1 / AT-B5). |
| 482b5f0f8 | `sa_tier_ok` is an atomic, stored once per walk | none on value. TSan PASS on lavapipe (F2). |
| d665efc53 | `ggml_sparse_attn_plan` op_params 2-4 = n_sel, n_q, n_stream; HIP `sa_select` (`sparse-attn-attend-g4.cu:225`) and Vulkan (`ggml-vulkan.cpp:10950, 11035`) read n_sel from them | every HIP and Vulkan SPARSE_ATTN alloc size now comes from op_params. Checked: index 2 of the plan is written only at `ggml.c:6699-6703` and read only at the three sites above. The CPU kernel reads only index 1 (`ggml-cpu/sparse-attn.cpp:320`). HIP rows of `test-sparse-attn-rpc-view` and every HIP reserve: NOT RUN. |
| 7a96532ad | the gated attend's hazardous output goes to graph memory (attend dst, then CONT, then SIGMOID), never the pool; otherwise the fusion is refused (`ggml-cuda.cu:6806-6845`, `sparse-attn-gate-scratch.h`) | on the GPU: the MTP gated share (≥ 0.9; it was 56/96 in PROOF-initial), fused output bitwise equal to the unfused path within the W3 tolerance, the pool high-water (M2) and perf. Code reading: a candidate is rejected when it overlaps any read, the V-pack workspace included (`reads[n_reads++] = { node->data + ws_off, ws_bytes }`). Whether every candidate is dead memory at the attend's point in the graph is argued in the commit text only: UNPROVEN on the GPU. |
| ed4bd3ea5 | at_end cases poison the 272 B past the last K/V storage with 0xFF and check it after compute | ROCm0 SPARSE_ATTN at_end cases: NOT RUN. On lavapipe the at_end cases pass (in the 736/736). |

## 3. Scorecard on head ed4bd3ea5 (task items 1-8)

| # | check | verdict | source |
|---|---|---|---|
| 0 | builds HIP / Vulkan / CPU / RPC at HEAD | PASS | §2.1 |
| 1 | test-backend-ops per op on ROCm0 (graphs on and off, each `GGML_CUDA_SA_FORCE`), RADV Vulkan0 (each `GGML_VK_SA_FORCE`), with HIP census coverage per kernel key | **NOT RUN** (GPU lock) | §1 |
| 1 | test-backend-ops on lavapipe vs the CPU oracle: SPARSE_ATTN, SPARSE_ATTN_PLAN, QSA_EXPAND/SCORE/STRIP, GET_ROWS/SET_ROWS/CPY; census 81/81 and 1/1 | PASS (lavapipe portable tier only) | §2.3 |
| 1 | unsupported shapes asserted unsupported (`--fail-unsupported`, expected list) | lavapipe PASS (654 / 93). ROCm0 and RADV: **NOT RUN** | §2.3 |
| 1 | determinism: in-process memcmp, cross-process identical outcomes | lavapipe PASS. HIP and RADV, and the model-level logit hash run twice: **NOT RUN** | §2.3 |
| 1 | ctest QSA: CPU 15/15; lavapipe build-vk 14/15 (te-rows abort, pre-existing, open); HIP and RADV | CPU PASS; lavapipe **FAIL** (te-rows); HIP / RADV **NOT RUN** | §2.2, §2.3 |
| 2 | HIP route census (EXPECT rows, 0 masked, 0 CPU FA / sparse nodes); Vulkan placement census | **NOT RUN** | |
| 2 | reserve = runtime (strict) on every route, including HIP `-np 2` split ub8192 (a16582997) and Vulkan ub16 (still unfixed: Prove-fix 1 "Not fixed" 2.2) | **NOT RUN**. The Vulkan ub16 case is expected to stay FAIL (UNPROVEN until rerun). | IMPLEMENTATION.md:1861-1883 |
| 2 | MTP census, gated share ≥ 0.9 (changed again by 7a96532ad) | **NOT RUN** | |
| 2 | guard / AT-C18 on a real run (`route=cache-only … guard=0/0`) | **NOT RUN** | |
| 3 | pinned KLD (QSA + MoE replay, `~/dev/wt_measQ` QPIN) for f16, q8_0, q4_0, bf16, q8_0K/f16V, f16K/q8_0V vs the measured floor; f16 bit-identity; unpinned catastrophe check; depth check; Q1-vk | **NOT RUN** | |
| 3 | Q12 coarse q8_0, Q10 control and image p3 seq1 (FAIL in PROOF-initial, no fix since) | **NOT RUN**, open | PROOF-initial §2 |
| 4 | perf: text + random, base (`PV/base_ea5`) vs new vs PR219, f16 / q8_0 / q4_0 / bf16 / mixed, ub 512 / 1024 / 4096, d 0 / 32k / 131k / ~262k (~300k text) / deepest; tg at depth; §16 statistic; stall (no sample > 1.5 × median); compute buffer; peak VRAM | **NOT RUN** | |
| 5 | images: `~/repro_img/repro_kv.sh` A / B / C for f16 and q8_0 (and the reduced type set); `-np 2` split with images; `-ctk q8_0 -ctv f16` with images; image understanding f16 vs q8_0; MTP draft | **NOT RUN** | |
| 6 | Vulkan build | PASS | §2.1 |
| 6 | Vulkan tests on RADV, q8_0 depth run on the RADV device | **NOT RUN** | |
| 7 | every deferred acceptance test | **NOT RUN** on this head (§4) | |
| 8 | every Moved-to-Prove check, W8 and later | **NOT RUN** on this head (§5) | |

Op-level-only types (owner scope note): q4_1, q5_0, q5_1, iq4_nl and f32 are listed by `test-qsa-supported-types` (`'f16, bf16, q8_0, q4_0, f32, q4_1, q5_0, q5_1, iq4_nl'`, F2). On this head their op-level proof is lavapipe only: they fall inside the SPARSE_ATTN 736/736 run and the 654 expected-unsupported. **The ROCm0 and RADV op runs are NOT RUN.** They are "op-level proven only, no end-to-end run" by scope, and on this head even the GPU op level is missing.

## 4. Deferred acceptance tests (item 7) on head ed4bd3ea5

Each test needs ROCm0 or RADV runs. The table gives the criterion (task text) and the state. The reduced matrix of the owner's scope note would apply to each run. None ran, so no reduction was used.

| test | criterion (short) | state |
|---|---|---|
| AT-A (Lab A, round 1 #0) | G1 is picked where it beats min(G4 packed + pack, G4 direct) by more than 3 × p90 noise; the HIP proc encodes the fitted rule; e2e f16 ub512 pp512 d131k is neutral or better under §16 (CI contains 0 or lies above it, half-width ≤ 1 %); the trimmed-layer row is a W4 gate row | **NOT RUN.** Code: the q8_0 135168-class cell is 64 (`sparse-attn-attend-g1.cu:68-72`, 4552df2a4), equal to the fitted CPP line of `prove_ata.md:282` in 16/16 cells (PROOF-after-fix §2.4). This is a code reading, not a pass. |
| Lab B VB1 / AT-B1 | f16 new / W0 is neutral or better under §16; q8_0, bf16 and S_fast pairs ≥ 0.95 × new f16 | **NOT RUN**. The q8_0 / q4_0 miss is TODO-LATER #3 (§6). |
| VB2 | new f16 op ≤ 1.10 × the in-build sel_cm median | **NOT APPLICABLE as written**: sel_cm was deleted in W13 (740d48097). It needs an owner ruling, and it is counted as not passing. |
| VB3 | ship the faster of direct and staged V per type; diagnostic op gate ≤ 1.15 × f16 or ≤ the HIP C18 ratio | **NOT RUN** |
| VB4 / AT-B4 | f16 sparse ≥ 0.98 × dense in every cell; q8_0 / q4_0 shortcut has 0 CPU FA / sparse nodes; decode at depth ≥ W0. b7394c04f changes this route. | **NOT RUN** |
| VB5 / AT-B5 | M1: every T ≤ new f16 + 1 MiB; new f16 ≤ W0 Vulkan f16. e46cd6c94 grows the vpack reserve. | **NOT RUN**. The MTP +12 MiB cell is TODO-LATER #4. |
| VB6 | mean paired KLD difference ≤ B_vk 6.4e-4 | **NOT RUN** |
| C-5 | 0 strict-reserve violations in every cell; reserved partial ≥ the §7 formula | **NOT RUN** |
| C-17 | new f16 ≤ PR219 reserve + pool high-water (M2), or the stated delta; partials scale with n_stream | **NOT RUN** |
| C-12 | each cell's new reserve ≤ PR219 + bound; runtime ≤ reserve; answers 2749 / 12749 | **NOT RUN** |
| C-16 | strip-view census > 0 for both srcs; no-share − share ≥ ½ the PR219 delta | **NOT RUN** |
| D1 | W7 kind-4 count = W0; U16 memcmp identical; OVL ratio-of-ratios ≥ 0 | **NOT RUN**. The criterion wording is TODO-LATER #5. |
| D2 / D3 | sparse plan and attend on the device of their K buffer in 100 % of the prints; guard 0; KLD ≤ the R1 value; CPU perf ≥ 1 − 2 × the W0 spread | **NOT RUN**. CPU perf is TODO-LATER #2. |
| D4 | 0 GPU-placed plan, attend or shortcut-FA nodes under `-nkvo`, the probe included; PPL within 3 × chunk SE of C15 | **NOT RUN** |
| D5 | `-fa auto` resolution logged; under `-fa off` / `-nkvo` the census > 0, 0 masked, guard 0; PPL within 3 × SE of PR219 | **NOT RUN** |
| AT-W12-BANDS | bands computed by their formulas and frozen in §15.0 before scoring; self-replay KLD 0 (≤ 3.7e-7) | bands frozen in PLAN-v3 §15.0 (PROOF-initial §10.5, head-independent). The self-replay on this head is **NOT RUN**. |
| AT-Q10Q6 | calibration valid (127 vs 127 = 0, ≥ 6 draws, LOO false-fail ≤ 1/6, τ_p1,q8_0 filled); Q10 control: new rerun vs new = 0 and new vs PR219 ≤ floor; Q10 cells ≤ τ; Q6 ≤ τ_T; answers | calibration (PR219 side, head-independent) stays as in PROOF-initial. The gates on this head are **NOT RUN**. The Q10 control was FAIL and has no fix since. |
| AT-W7PERF | per (pair, n_q band) sparse ≥ 0.98 × dense or prefer_dense flips; MTP tg P5 ≥ 0.94; `-nkvo` new / PR219 ≥ 0.98 in each cell; q8_0 / f16 ≥ 0.94 under `-nkvo`; census | **NOT RUN** |
| AT-B1 ext / AT-B3 vpack / AT-B7 | S_fast ≥ 0.95 × f16 in each AT-B1 cell; vpack rule; portable op runs pass on lavapipe and on RADV sg32; supports_op false without a pipeline | **lavapipe part PASS** (§2.3: PLAN 92/92 and SPARSE_ATTN 736/736 `--fail-unsupported`, 0 failures; `test-sparse-attn-vk capability` within "OK: 0 failed checks"). RADV sg32, AT-B1 cells and the AT-B3 vpack arm: **NOT RUN**. The q8_0 / q4_0 AT-B1 miss is TODO-LATER #3. |

## 5. Moved-to-Prove checks, W8 and later (item 8)

All of these need the GPU, so all are **NOT RUN** on ed4bd3ea5. Criteria are verbatim in IMPLEMENTATION.md at the lines given.

| W item | checks | criterion source | state |
|---|---|---|---|
| W8 | AT-Q10Q6 (1)(2)(3), Q10 on the head, Q6 on the head | IMPLEMENTATION.md:1052-1061 | NOT RUN |
| W9 | AT-C12 grid, AT-C12 image-slot layout, M5 cells, AT-C17 / M6, AT-C16b, M4 (b), M4 (c), C15 ub1024 + A/B/C, np2 shared prefix trim on/off, U27 | IMPLEMENTATION.md:1241-1255 | NOT RUN |
| W10 | U6 OP_PROF A/B, rotated op time, Q5, Q1-A / Q1-B, P1-P6, Vulkan in-op H_q quality | IMPLEMENTATION.md:1461-1470 | NOT RUN |
| W11 | Q4 selection agreement, M1 iq4_nl MTP, Vulkan q8_0 indexer for q4_0 KV, GLM5_NEXT session (only if a GGUF exists: none found, not applicable) | IMPLEMENTATION.md:1586-1593 | NOT RUN |
| W13 | full §14-§16.5 campaign, AT-C16b, VGPR after rebase, merge-tree | IMPLEMENTATION.md:1754-1762 | NOT RUN. VGPR-after-rebase and merge-tree are not applicable yet: no rebase was made (merge-base still ea5285ddf, §1). |
| Prove-fix 1 reruns | ctest, ops, np2, vkc, mtp, c18, census, smoke | IMPLEMENTATION.md:1906-1921 | NOT RUN |
| Final-fix 2 reruns | RADV `test-sparse-attn-vk dense reason vpack race`; HIP / RADV `test-sparse-attn-rpc-view`; HIP / RADV op tests (at_end tail poison on ROCm0); ctest in build and build-vk | IMPLEMENTATION.md:2059-2068 | NOT RUN |

## 6. Known issues (deferred: TODO-LATER.md; reported, not counted in pass)

None of them could be re-measured on ed4bd3ea5, because of the GPU lock. Each one's last measurement is on 1dbee4d9a (PROOF-initial §12, TODO-LATER.md).

| # | item | last measurement | status on ed4bd3ea5 |
|---|---|---|---|
| 1 | Q1 direct form, q4_0 / bf16 | stage 2: q4_0 2.082e-3 > band 1.441e-3; bf16 1.250e-4 > 1.032e-4 | NOT RUN; last FAIL |
| 2 | AT-D2 `-ngl 0` CPU perf | new / PR219 at pp512 0.69-0.71, tg32 0.80-0.92 | NOT RUN; last FAIL. The CPU op (`ggml-cpu/sparse-attn.cpp`) is unchanged by the 9 commits. |
| 3 | AT-B1 Vulkan q8_0 / q4_0 < 0.95 × f16 | 0.921 / 0.916 | NOT RUN; last FAIL |
| 4 | AT-B5 Vulkan MTP f16 +12 MiB | not re-measured in PROOF-initial | NOT RUN. e46cd6c94 can change Vulkan reserves. |
| 5 | AT-D1 kind-4 wording | new 12, PR219 0 | NOT RUN; criterion not met as worded |
| 6 | M1 iq4_nl MTP | +0.26 MiB (cleared) | NOT RUN; last PASS |
| 7 | `-nkvo` copy route: four d0 cells, bf16 | bf16 10 + 2 PASS, 2 unresolved; the d0 cells unresolved | NOT RUN; last open |
| 8 | Q4 thin f16-KV margins | +6.3e-6 / +2.0e-6 | NOT RUN; last PASS (thin) |
| 9 | Q7 MTP draft acceptance | stage 1, new − PR219 for default / q8_0 / q4_0 / bf16: −1.41 / −1.88 / −0.28 / −1.44 pp; stage 2: +1.28 / +0.93 / +1.17 / +1.12 pp; LCB −1.87 to −3.83 against the gate LCB ≥ −1 pp | NOT RUN; last FAIL (deferred) |

## 7. Commands run in this step (all on strix-002)
1. `uptime; flock -n ~/gpu.lock true && echo LOCK_FREE || echo LOCK_HELD; ps -eo pid,stat,etime,cmd | grep -E "llama|test-backend"; cat /sys/class/drm/card*/device/mem_info_gtt_used; free -g; git rev-parse HEAD; git merge-base HEAD origin/master; git status --short | wc -l; git log --oneline 97865cacb..HEAD`
2. `journalctl -k --since "2026-11-10 13:00"`; `grep -E "State|SigPnd" /proc/4041225/status`; `cat /proc/4041225/wchan`; `sudo -n true`
3. Lock wait: `timeout 2400 flock ~/gpu.lock -c 'date "+GOT_LOCK %F %T"'` → `R1/lockwait.log`
4. `cmake --build {build,build-vk,build-cpu,build-rpc} -j 16` → `R1/rebuild_*.log`
5. `cd build-cpu && ctest -R "test-qsa|test-sparse-attn" -j1 --output-on-failure` → `R1/ctest_cpu.log`
6. `R1/lvp.sh build-vk/bin/test-backend-ops test -b Vulkan0 -o SPARSE_ATTN --fail-unsupported`, with an fd check of the running pid (`ls -l /proc/<pid>/fd | grep -cE "dri|kfd"` → 0)
7. `R1/lvp_batch.sh` (SPARSE_ATTN run 2; PLAN × 2; QSA_EXPAND/SCORE/STRIP, GET_ROWS/SET_ROWS/CPY, QSA_PREFILL/DECODE; test-sparse-attn-vk; rpc-view; allvis; te-rows × 5; build-vk ctest) → `R1/lvp_batch.out`
8. The per-case outcome hash of runs 1 and 2 (`grep -E "^\s+(SPARSE_ATTN|QSA)" | sed 's/\x1b\[[0-9;]*m//g' | sha256sum`)
9. `git diff --stat 97865cacb..HEAD`, `git show 7a96532ad`, `git show d665efc53`, and a grep of the plan op_params readers
10. `git merge-base --is-ancestor 13a279bfa ea5285ddf`; `git rev-list --count` for 13a279bfa..ea5285ddf, ea5285ddf..origin/master and ea5285ddf..HEAD

## 8. Side effects and cleanup
- No AMD GPU command was run, and no process was signalled. pids 4041225 and 4041224 were not touched.
- No systemd unit was touched, strix-001 was not used, and nothing was pushed or posted.
- `~/dev/wt_rewrite`: only the build trees changed (incremental rebuilds). There are no source changes, and `git status` is clean.
- `R1` holds logs only. No file over 1 GB was created.
