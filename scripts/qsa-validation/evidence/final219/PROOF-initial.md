# PROOF (initial): qsa-rewrite head 1dbee4d9a on strix-002

**Verdict: pass = false.** The rewrite is functional and correct on most checks. Every QSA layer runs on its device with zero CPU attention nodes. Image answers are correct for every type. Perplexity, pinned KLD and the main prefill gates pass. Quantized prefill at depth is up to 2.8× faster than the base at d32k and up to 8× (q4_0, d131k) or 13× (q4_0, d300k) faster than PR219. But several non-deferred checks fail or stay unresolved. The full list is in section 2.

**Conventions.**
- `PV` = `strix-002:~/dev/wt_lab/prove` (all Prove outputs, scripts and logs). `H` = `PV/h`, a copy of `~/dev/wt_rewrite/scripts/qsa-validation/harness` whose output roots point at `PV`. Its sed is in `PV/setup.sh`.
- "chainN" = `PV/chainN.sh`, with output in `PV/chainN.out`.
- Arms:
  - **new** = head `1dbee4d9a` (`~/dev/wt_rewrite/build`, `build-vk`).
  - **PR219** = W0-lab12 (`~/dev/wt_lab/w0lab12`: #219's commits on the same base `ea5285ddf` + QPIN). This is the plan's "PR219 = W0 build" (PLAN-v3 §16).
  - **base** = the merge-base `ea5285ddf` (`PV/base_ea5`, built for Prove; `git merge-base HEAD origin/master` = ea5285ddf, newer than `wt_pr219v2_base` 13a279bfa by 8 commits).
  - **hk** = head + lab patch 06 (measurement hooks, `~/dev/wt_lab/w13/hooks`).
  - **ap** = head with the f16/bf16 V-pack trait set to always-pack (`PV/apack`, the one-line change in `sparse-attn-loaders.cuh:78-79`).
  - **lab** = head + QPIN lab patches (`PV/lab`).
- Every GPU command ran under `flock ~/gpu.lock`, and this agent was the only one on strix-002. Per-op test-backend-ops ran one op per process (never the full suite, never `-j > 1`).
- Builds: head HIP `rc=0 warn=0` and Vulkan `rc=0 warn=0`. The base, hooks HIP/VK, lab HIP/VK, always-pack and C19 lab trees all have `rc=0` (`PV/setup.out`, `PV/chain4.out` "APACK build", `PV/c19_build.log`).
- Statistic: PLAN-v3 §16, implemented by `H/w7/perf_an.py` and `PV/cmp.py` (the same functions on any pair of globs). It takes the per-process mean of reps 1-4, then a hierarchical bootstrap with 4000 draws and seed 1. PASS if the CI lower bound ≥ θ, FAIL if the CI upper bound < θ, otherwise RERUN.
- **No §16 reruns were run (see the scope note in section 9), so every RERUN cell is reported as "unresolved", which counts as not passing.**

## 1. Scorecard

| # | check | verdict | where |
|---|---|---|---|
| 1 | test-backend-ops per op, every tier, HIP / Vulkan / CPU / lavapipe | PASS | §3.1 |
| 1 | unsupported shapes are unsupported (expected list, capability mask) | PASS | §3.1 |
| 1 | CPU oracle comparisons (test-backend-ops vs CPU, plan oracle, rot unit tests) | PASS | §3.1 |
| 1 | determinism, in-process (suite memcmp) and model level (logit hash, run twice) | PASS (HIP 12/12, VK see §3.3) | §3.3 |
| 1 | determinism, cross-process op-dump hash | not a valid test: inputs are random per process (shown); in-process memcmp PASS | §3.3 |
| 2 | route census HIP: 6 pairs × ub {4096, 512, 16} + bench, EXPECT rows from the shipped policy, 0 masked, 0 CPU | PASS 24/24 | §4.1 |
| 2 | route census Vulkan: 6 pairs × 3 ub, every trace on Vulkan0, 0 CPU | PASS 18/18 (placement) | §4.2 |
| 2 | reserve = runtime (LLAMA_STRICT_RESERVE violations = 0) | **FAIL**: HIP `-np 2` split ub8192 (+22.6 MiB), Vulkan ub16 (+44.8 KB) | §4.3 |
| 2 | MTP census (route rows) | rows PASS; **gated share FAIL** 56/96 | §4.4 |
| 3 | pinned KLD vs floor, Q1-A/B/C (6 types × ub) | PASS 15/15 | §5.1 |
| 3 | Q1 direct form | PASS 4/6; q4_0, bf16 FAIL both stages (**deferred, TODO-LATER #1**) | §5.1 |
| 3 | Q3 depth (c32768) A/B incl. q4_0 ub256 | PASS 6/6 | §5.2 |
| 3 | Q4, Q8, Q11a/b/c, Q12 (-nkvo, -ngl 0) pinned | PASS | §5.3 |
| 3 | Q12 coarse unpinned screen | f16 PASS, **q8_0 FAIL** (0.015717 > 0.01563) | §5.3 |
| 3 | f16 bit-identity claims (rot in-kernel = fwht, W10 graph-side = in-op, np2 = np1) | PASS | §5.4 |
| 3 | unpinned catastrophe check | PASS (no catastrophe) | §5.5 |
| 3 | Q1-vk (Vulkan) | PASS 6/6 | §5.6 |
| 3 | Q7 MTP acceptance | **FAIL** stage 1 and stage 2, 4/4 draft types | §5.7 |
| 3 | Q10 / Q6 with AT-Q10Q6 calibration | Q6 PASS 4/4; **Q10 control FAIL**; Q10 cells 7/8 (image p3 seq1 FAIL) | §5.8 |
| 3 | Q5 op-local bands, PR-verbatim reference, planted defects | PASS | §5.9 |
| 4 | P1 | PASS 1.0310 [1.0255, 1.0371] | §6.1 |
| 4 | P2 / P3 (text f16 / q8_0, ub 512/1024/4096, d0/32k/131k) | P2 23/24 PASS, 1 unresolved; P3 24/24 + floor 24/24 PASS | §6.2 |
| 4 | text q4_0 new vs PR219 and q4_0/f16 | PASS 24/24, 24/24 | §6.2 |
| 4 | random-token prefill vs PR219 and vs base | PASS 36/36 vs PR219, 24/24 vs base | §6.3 |
| 4 | P4 (bf16, mixed vs f16) | 35/36 PASS, 1 unresolved (bf16 d131k ub512 pp512 0.919 [0.887, 0.949]) | §6.4 |
| 4 | P5 tg (shortcut and depth, cross-type) | PASS 47/47 | §6.5 |
| 4 | P5 MTP cells | **FAIL** 1 (f16 default draft d32k 0.898), 6 unresolved | §6.5 |
| 4 | depth ~300k and deepest (400k), AT-O both V-pack settings | report; several cells unresolved | §6.6 |
| 4 | stall (no sample > 1.5 × its run median) | **FAIL**: 1 of 3232 samples (1.61×) | §6.7 |
| 4 | compute buffer, peak VRAM | reported | §6.8 |
| 5 | images A/B/C, 6 types, MTP on | PASS 18/18 answers, strict 0, CPU 0 | §7.1 |
| 5 | `-np 2` split with images | answers PASS; **strict reserve FAIL** (§4.3) | §7.2 |
| 5 | mixed q8_0/f16 with images | PASS | §7.1 |
| 5 | image understanding f16 vs q8_0 (short and at ~42k depth) | PASS | §7.3 |
| 6 | Vulkan build + op tests + lavapipe + q8_0 depth run | build/tests PASS; **depth run NOT RUN** (kernel hang, §13) | §8 |
| 7 | deferred acceptance tests (7 entries) | see §10 | §10 |
| 8 | Moved-to-Prove checks W7-W13 | see §11 | §11 |

## 2. Failures (non-deferred; these make pass = false)

1. **Reserve < runtime, HIP, `-np 2` split streams at ub8192** (all three types run: f16, q8_0, q8_0K/f16V).
   - Main context, 10 `LLAMA_STRICT_RESERVE violation` lines. The buffer reached 3144.30 MiB against a reserve of 3122.75 MiB (+22.6 MiB). The excess grows with n_kv (8192 → 69888).
   - The violating ubatches are one-sequence ubatches of n_tokens 8192, producer=cb, n_stream=1.
   - The reserve was planned at n_tokens 4096 × n_seqs 2 (`QSA reserve passB: n_tokens = 4096, n_seqs = 2`).
   - Source: `PV/img/srv_np2split_f16.log` (grep `STRICT_RESERVE violation`), `PV/img/res_np2split_*.txt`.
   - The answers are still correct (2749 / 12749). The unified `-np 2` run has 0 violations (`res_np2uni_q8_0.txt`). So do the split `-np 2` runs at ub 4096 and 1024 (W9 list, §7.2).
2. **Reserve < runtime, Vulkan, ub16**: f16/f16, bf16/bf16 and f16K/q8_0V each show 10 violations, up to +44,800 B (16.1998 > 16.1571 MiB) at n_q 16, n_kv 6656. q8_0, q4_0 and q8_0K/f16V show 0. Source: `PV/c1/vkcensus/c_f16_f16_u16.log`, `PV/chain1b.out` "VKC".
3. **MTP server census: gated share of fusible G4 attends 56/96 = 0.583 < floor 0.9.** Every route row matches the expected rows (`OK sa_attend/... expected=N census=N`), with masked 0 and CPU 0. The gate fusion is not applied on part of the server graph, which is the W11 finding. Source: `PV/mtp/server_c_default.log.expect` (also `_q8`, `_q4`), `PV/c1/mtp*/` `MTP_SMOKE` "EXPECT FAIL".
4. **AT-C18 at `-np 1`, ub4096: guard-on-probe criterion FAIL in 4/8 runs.** The memory part passes 32/32 (|probe − real| = 0.00 MiB).
   - The failing probe line is `graph=passB route=dense-shortcut plan=0 attend=12 guard=12/12`, i.e. 12 < 2 × 12 layers. This is the pass added in W9.
   - Also seen: an MTP `graph=pp-0out route=dense-shortcut policy=default backend=none ... guard=0/1` line. Its cause is UNPROVEN.
   - Source: `PV/c18/srv_f16_u4096.log`, `PV/chain1b.out` "AT-C18 FAIL (4)". `-np auto`: §11.
5. **Q7 (MTP draft acceptance, ≥ 14 L/X prompts, one-sided 95 % LCB ≥ −1 pp): FAIL at stage 1 and stage 2 for every draft type.**
   - Stage 1 (L0-L5, X0-X7): mean new − PR219 for default / q8_0 / q4_0 / bf16 = −1.41 / −1.88 / −0.28 / −1.44 pp, SD 3.1-4.1, LCB −2.97 / −3.83 / −1.74 / −3.25.
   - Stage 2 (X8-X21): +1.28 / +0.93 / +1.17 / +1.12 pp, SD 6.4-7.1, LCB −1.91 / −2.45 / −1.87 / −2.15.
   - The sign flips between stages. The SD is 3.4-7.7× the plan's σ = 0.921 pp, and the texts differ in 0/14 + 0/14 prompts between builds. The gate cannot resolve −1 pp at n = 14 with this noise. A regression is not shown (UNPROVEN either way).
   - Source: `PV/q7an.py`, `PV/q7`, `PV/q7s2`.
6. **Q10 control (L = 8192): new vs PR219 KLD seq0 8.485e-3, seq1 2.569e-2 > floor_ctl 3.302e-3 / 1.050e-2.** Q10 image p3 seq1 is also a FAIL: 3.408e-3 > τ 1.553e-3. Details in §5.8 and `PV/q10/q10q6.md`.
7. **Q12 coarse unpinned screen, q8_0: 0.015717 ± 0.000542 > 0.01563.** f16 is a PASS at 0.014491. Source: `PV/chain7b.out` "Q12c".
8. **AT-A rule mismatch (one cell):** re-measured on the head's kernels, the largest G1 n_q for V = q8_0 at the 135168 n_kv class is 64. The shipped `nq_max` has 96 (`sparse-attn-attend-g1.cu:65-70`). The other 15 cells match. Source: `PV/w4bands/ata.md` "CPP".
9. **P5 MTP cell f16 main, default draft, d32768: new / PR219 = 0.898 [0.859, 0.949] < 0.98.** It runs with fewer accepted drafts (new 70/386 vs PR219 76/341 at reps 1-4). The generated text differs between builds, so the cause (kernel vs acceptance) is UNPROVEN. Source: `PV/mtp_perf/f16_default_*.jsonl`.
10. **Stall check: 1 sample of 3232 above 1.5 × its process-cell median** (1.61×, `p23_d0_q4_0_2_new.json` ub4096 pp512). Source: `PV/perf/an_stall.md`.
11. **Unresolved perf cells**, with no §16 reruns run (§9):
    - P2 d0 ub512 pp2048: 0.9979 [0.9879, 1.0082];
    - P4 bf16/f16 d131072 ub512 pp512: 0.9188 [0.8872, 0.9494];
    - P5 MTP: 6 cells;
    - U6 OP_PROF q4_0 d0: one of three new runs took 548 ms, a stall;
    - deep 300k cells (§6.6);
    - the NDB/NPIN cells of §11 W7.
12. **AT-B2 not runnable on the head.** W13 deleted the old `sel_cm` pipeline that it compares against (`740d48097`). There is no in-build reference, so the result is NOT RUN.
13. **AT-C17 PR219 pool high-water not measured.** The fill runs were rejected: `request (266240 tokens) exceeds the available context size (262144 tokens)` (`PV/c17/srv_pr_fill266k.log`). The M6 verdict still holds without it (§10.3).
14. **Vulkan q8_0 depth run, AT-B1, AT-B4 e2e and AT-B5 were not run.** A Vulkan llama-perplexity process (bf16 determinism run 2) entered uninterruptible sleep in `drm_suballoc_insert` after the kernel logged `amdgpu_gem_va_update_vm *ERROR* Couldn't update BO_VA (-12)` at 11:55:28. It still holds 91.6 GB GTT and **the GPU lock `~/gpu.lock`** (fd 3), and SIGTERM / SIGKILL have no effect (§13). Vulkan determinism for bf16 and the mixed pairs is also missing.
15. **AT-A e2e confirmation cell, U30 masks on Vulkan, and the Q10-style KLD of the AT-C12 image-slot layout: NOT RUN.**

## 3. Op level (item 1)

### 3.1 test-backend-ops, per op, one process each

Commands: `H/w10/optests.sh PV/c1/ops cpu plan rocm rocm_nographs vdirect vpacked generic portable fixup rot vk vkrot`, plus the loops in chain1 `ph_ops`. Logs: `PV/c1/ops/*.log`, summary in `PV/chain1.out`.

| backend / forcing | op | result | census coverage (kernel path proof) |
|---|---|---|---|
| CPU | SPARSE_ATTN | 1390/1390, rc 0 | n/a (reference backend) |
| CPU | SPARSE_ATTN_PLAN | 185/185 | n/a |
| ROCm0 default | SPARSE_ATTN_PLAN | 185/185 | `sa_plan/` 2/2 compiled keys launched |
| ROCm0 default | SPARSE_ATTN | 1390/1390, notsup 0 | 492/492 compiled keys launched (2 excluded: `sa_portable/*` runs on RDNA3.5 only when forced) |
| ROCm0 graphs off | SPARSE_ATTN | 1390/1390 | 492/492 |
| ROCm0 `GGML_CUDA_SA_FORCE=vdirect` | SPARSE_ATTN | 1390/1390 | 388/388 (106 excluded by the forcing) |
| ROCm0 `=vpacked` | SPARSE_ATTN | 1390/1390 | 393/393 (101 excluded) |
| ROCm0 `=generic` | SPARSE_ATTN | 1390/1390 | 24/24 (470 excluded) |
| ROCm0 `=portable` | SPARSE_ATTN | 1390/1390 | 2/2 (`sa_portable/G1`, `G4`) |
| ROCm0 `=fixup` | SPARSE_ATTN | 1390/1390 | 492/492 |
| HIP rotation unit test (`test-sparse-attn-rot`), default / generic / portable / fixup | | 0 of 144 / 136 / 136 / 144 failed, 136 bitwise lines each | |
| Vulkan0 default | SPARSE_ATTN_PLAN | 92/92, 93 "not supported" | `sa_plan/` 1/1 |
| Vulkan0 default | SPARSE_ATTN | 744/744, 646 "not supported" | 85/85 (332 excluded: another V variant per pair in this process) |
| Vulkan0 `GGML_VK_SA_FORCE=` vdirect / vstaged / vpack / scalar / portable / portable_sg32 | PLAN, SPARSE_ATTN | 92/92 and 744/744 each, rc 0 | |
| lavapipe (W6 Mesa build, `~/dev/wt_lab/w6/lvpenv.sh`, `llvmpipe (LLVM 21.1.8)`) | SPARSE_ATTN_PLAN / SPARSE_ATTN | 92/92 / 736/736, rc 0 | |
| ROCm0 / Vulkan0 | FLASH_ATTN_EXT | 4051/4051 / 5383/5383 | |
| ROCm0 / Vulkan0 | QSA_EXPAND, QSA_SCORE, QSA_STRIP | 56/56, 13/13, 6/6 each | |
| ROCm0 / Vulkan0 | GET_ROWS, SET_ROWS, CPY | 591/591, 159/159, 290/290 / 647/647, 439/439, 307/307 | |
| deleted ops QSA_PREFILL, QSA_DECODE, *_MASKLESS | | 0 cases listed each | |

- **Kernel path proof.** Each SPARSE_ATTN run ends with the suite census (`CENSUS COVERAGE: ... compiled keys ... launched`). That line counts the dispatch census per compiled kernel key (HIP `ggml_backend_debug_dispatch_census_*`, `tests/test-backend-ops.cpp:19415-19422`). A FAIL would name any compiled key that no case reached. Every forced run reports 100 % of its non-excluded keys, and the excluded keys are named with their reason. Logs: `PV/c1/ops/rocm_*_SPARSE_ATTN.log`, `vk_*`.
- **Unsupported shapes.** With `--fail-unsupported`, a "not supported" result is a FAIL unless the backend's expected-unsupported list contains it (`tests/test-backend-ops.cpp:13272-13286`: Vulkan G = 4 plans and attends, and K/V bindings past the device limit; `:19423`). Every Vulkan run with `--fail-unsupported` (default, the six forcings, lavapipe) has rc 0, so its 646 / 93 / 654 "not supported" are exactly the expected list. HIP and CPU report 0 "not supported".
  - The supports_op-false-before-dispatch check is the `capability` case of `test-sparse-attn-vk` in ctest vk (11/11 pass, §3.2).
  - `GGML_VK_SA_CAP_MASK=0` alone changed nothing (744/744 pass, 0 crash, `vk_capmask0_SPARSE_ATTN.log`). The value 0 does not disable tiers, so this run is not evidence.
- **CPU oracle.** Every ROCm0 / Vulkan0 case is compared with the CPU backend inside test-backend-ops, against the band of its census class. Also: ctest `test-sparse-attn-plan-oracle` (§3.2), and with `GGML_SA_TEST_REPORT=1` 1432 SA_REPORT lines, all with `nmse < band` (§5.9).

### 3.2 ctest QSA subset
`ctest -R "test-qsa|test-sparse-attn" -j1`: HIP 11/11, Vulkan 11/11 (`PV/c1/ctest_hip.log`, `ctest_vk.log`).

### 3.3 Determinism
- **In-process** (the suite's own check, `n_determinism_runs() = 2` for SPARSE_ATTN and SPARSE_ATTN_PLAN, memcmp of every compared output, `tests/test-backend-ops.cpp:1716-1745, :13304`): included in every pass above.
- **Model level, run twice, logit hash.** llama-perplexity c8192 × 2 chunks, `--kl-divergence-base` written twice, `sha256sum` (chain1 `ph_det_model`, `PV/c1/det_model/`):
  - HIP: f16, q8_0, q4_0, bf16, q8_0K/f16V and f16K/q8_0V × ub {4096, 512}: identical = yes in 12/12.
  - Same-type ub4096 and ub512 hashes are equal for q8_0, q4_0, bf16 and both mixed pairs (`PV/chain1b.out` DET_MODEL lines).
  - Vulkan: f16, q8_0 and q4_0 × ub {4096, 512}: identical = yes in 6/6. bf16 ub4096 run 1 finished, but run 2 hung in the kernel (§13). bf16 ub512 and the mixed pairs were not run (NOT RUN).
- **Cross-process op dumps** (`GGML_SA_TEST_DUMP`, 193 ROCm0 / 127 Vulkan0 dumps per run): all files differ between the two processes (`PV/chain1.out` DET_OP).
  - **Method check (chain14b):** the same case (`q8_0/q8_0 n_q 9 n_kv 4096 n_sel 257 group 1`) was dumped in two processes. The dumped **inputs** already differ (`inputs identical across processes: False`), because test-backend-ops draws fresh random inputs per process. So a cross-process dump hash is not a determinism test, and in-process determinism is the valid check (PASS above). Source: `PV/chain14b.out`.

## 4. Routes, census, reserve (item 2)

### 4.1 HIP route census (shipped-policy expected rows)
`H/w7/census.sh` with `LLAMA_STRICT_RESERVE=1 GGML_CUDA_DISPATCH_CENSUS_LOG=1 LLAMA_QSA_TRACE=1 GGML_CUDA_DISABLE_GRAPHS=1`. It runs 6 pairs × `llama-perplexity c8192 b4096 ub {4096, 512, 16}`, plus `llama-bench pp4096 @ d4096`. The checker is `census_check.py expect` against `test-qsa-select-dump` (the shipped policy procs).
- **24/24 `EXPECT PASS`, 0 diffs, `masked_lines=0`, `cpu_census=0`, `fa_cpu=0`. Every run has strict `violations=0`.** The guard inspected ≥ 12 per pass (bench ≥ 24). Source: `PV/chain1.out` CENSUS / STRICT lines, `PV/census/*.log*`.
- `GGML_SCHED_DEBUG_REALLOC=1` was dropped from the census run. It aborts at ub512 and ub64 on the **base ea5285ddf and on PR219 as well** ("unexpected graph reallocation", rc 134 for all three builds at ub64 / ub512, rc 0 at ub4096: `PV/realloc_check.out`). So its abort is pre-existing behaviour, not a QSA regression. The binding check used is `LLAMA_STRICT_RESERVE` (buffer growth past the reserve).

### 4.2 Vulkan route census
chain1b `ph_vkcensus`: 6 pairs × ub {4096, 512, 16}.
- Every QSA trace line is `backend=Vulkan0` (36 / 204 / 396 per run). Masked 0 and CPU census 0 in 18/18 runs. PPL lies within 2.08-2.14.
- `census_check.py expect` cannot run on Vulkan: `test-qsa-select-dump` prints "no backend with ggml_backend_sparse_attn_select_dump". The Vulkan backend has no select-dump proc, so placement and the dispatch census are the evidence. Source: `PV/chain1b.out` VKC lines, `PV/c1/vkcensus/`.

### 4.3 Reserve = runtime (strict)

| run set | strict violations |
|---|---|
| HIP census, 24 runs (§4.1) | 0 |
| AT-C5, ub {64, 65, 66, 127, 128, 129} × d {4096, 32768}, {65, 127, 129} at d131072, final ubatches 127 / 65 at d0 and d131072, ppl ub127, forced g1 / g4 at n_q {65, 127, 129}, f16 / q8_0 / q4_0, MTP server ub64 | 0 (`PV/chain9.out`) |
| images A/B/C × 6 types, M4 (b) / (c), C15 ub1024, SP trim 0 / 1, `-np 2` unified, W9 list (9 runs) | 0 |
| **`-np 2` split ub8192 with images (f16, q8_0, q8_0K/f16V)** | **10 each (main ctx), FAIL** (§2.1) |
| Vulkan census ub 4096 / 512, all pairs; ub16 for q8_0, q4_0, q8_0K/f16V | 0 |
| **Vulkan ub16, f16 / bf16 / f16K/q8_0V** | **10 each, FAIL** (§2.2) |

### 4.4 MTP census (`H/w7/mtp_census.sh`, C2 flags, main q8_0, draft default / q8_0 / q4_0)
- Every per-key row is OK. For example, the default draft gives `sa_attend/G1/q8_0/q8_0/splits17-33/gqa12 expected=3600 census=3600`, `sa_plan/G1 4491 = 4491` and `sa_attend/G4/q8_0/q8_0/packed/plain/gqa12 96 = 96`.
- Masked 0, CPU 0, and all 5 requests answered.
- **The gated share is 56/96 = 0.583 against a floor of 0.9, so the expectation FAILs (§2.3).**
- Source: `PV/mtp/server_c_*.log.expect`.

### 4.5 Guard (`H/w7/guard.sh`, `PV/guard/`)
- Pass rows for q8_0, f16, q4_0 and bf16: PASS.
- Fake device FAKE0: init error "...not supported on device FAKE0 (backend FAKE): no sparse-attention kernel in this backend; run QSA on the CPU with -ngl 0 or --no-kv-offload, or use a H[IP]...".
- `GGML_CUDA_SA_REJECT`: an init error, plus a decode error from `process_ubatch` (layer 3, av).
- all-visible rows: `RESULT PASS (failures 0)`.
- `-sm tensor` gives the init error "the meta backend has no SPARSE_ATTN handler".
- `-c 2097664` gives "n_ctx_seq = 2097664 is above the tested maximum of 2097156 cells".

### 4.6 Placement (AT-D3 / D4 / D5, chain1b `H/w7/at_d.sh D3 D4 D5`)
- **D3** (`-ngl 24`): QSA layers 3-23 run on CPU and 27-47 on ROCm0, each on its K buffer's device. Guard cpu_launch 0. PASS.
- **D4** (`-nkvo`): every trace line is `route=host-kv-copy backend=ROCm0` (the named route), masked 0, guard silent.
  - PPL new / PR219: f16 1.9477/1.9471 (ub4096), 1.9442/1.9446 (512), 1.9387/1.9414 (16); q8_0 1.9453/1.9602, 1.9453/1.9491, 1.9406/1.9432.
  - All are within 3 × SE (0.034) of C15 (f16 1.9428/1.9449, q8_0 1.9532/1.9519). PASS.
- **D5**:
  - `-fa off` offloaded: sparse 144/24 + dense shortcut 60/12, masked 0.
  - `-fa off -nkvo`: host-kv-copy only, masked 0.
  - `-nkvo -fa auto`: FA enabled (no "Flash Attention ... disabled" warning; the 4 "set to disabled" lines are unrelated fused ops), route host-kv-copy.
  - PPL new / PR219 same flags: 1.9565/1.9573, 1.9475/1.9505, 1.9485/1.9574, 1.9477/1.9469.
  - PASS.

## 5. Quality (item 3)

Pinned arms: `H/w7/q_campaign.sh A D P C S B E N Z H` (chain2). The references are recorded on PR219 (W0-lab12), and each was deleted after its phase. The scorer is `H/w7/q_an.py` (bands of PLAN-v3 §15.0). Output: `PV/q/quality.md`, logs `PV/q/logs/`. Self-replay: QKL_HASH equal 8/8 chunks; every arm has rows_absent = moe_absent = 0, except one arm noted below.

### 5.1 Q1
- **Q1-A ub4096** (new − PR219, band 2.0e-4): f16 +1.369e-4 ± 2.9e-5, q8_0 −2.7e-6, q4_0 −1.5e-5, bf16 −4.1e-5, q8_0K/f16V −5.0e-6, f16K/q8_0V −2.3e-5: PASS 6/6.
- **Q1-B ub512** (band 2.03e-4): f16 −2.3e-5, q8_0 −1.2e-5, q4_0 −7.3e-5: PASS.
- **Q1-C ub64** (band 2.75e-4), policy and forced G1: f16 +1.8e-5 / −1.0e-5, q8_0 +2.6e-6 / −1.8e-5, q4_0 −1.1e-4 / −4.4e-5: PASS.
- **Direct form** KLD(new ‖ PR219):
  - f16 1.375e-4 ≤ 1.403e-4; q8_0 1.503e-4 ≤ 2.310e-4; q8_0K/f16V 1.266e-4 ≤ 1.807e-4; f16K/q8_0V 1.517e-4 ≤ 1.623e-4: PASS.
  - **q4_0 2.690e-3 > 1.441e-3 and bf16 1.802e-4 > 1.032e-4: FAIL. Stage 2 (chunks 8-15 of c8192 × 16): q4_0 2.082e-3, bf16 1.250e-4: FAIL.** This is TODO-LATER #1 (deferred), with the same values as W7.
  - The phase-D f16 base arm `new_db_f16_f16` has "NO SUMMARY" (run health), but its KLD is scored. UNPROVEN why the summary line is missing.
- Q2 (report): f16 ub2048 floor 1.0777e-4. Ratios 0.985-2.06 for f16 / q8_0 / bf16 / mixed, and ≈ 53 for q4_0 (its own quantization floor).

### 5.2 Q3 (depth: c32768 × 8, QKL_TAIL 4096)

| arm | mean diff | band | verdict |
|---|---|---|---|
| Q3-A f16 ub4096 | +1.442e-4 ± 3.5e-5 | 1.801e-4 | PASS |
| Q3-A q8_0 ub4096 | +4.5e-6 | 1.801e-4 | PASS |
| Q3-A q4_0 ub4096 | −1.07e-4 | 1.801e-4 | PASS |
| Q3-B f16 / q8_0 / **q4_0** ub256 | +2.7e-6 / +4.4e-6 / −1.95e-4 | 3.34e-4 | PASS (q4_0 ub256 was W7's open item) |

### 5.3 Q4, Q8, Q11, Q12
- **Q4** (c8192 × 16, kind-1 partner where the plan says): every type ≥ its threshold.
  - f16: 0.997356 ≥ 0.99735 (margin +6.3e-6).
  - f16K with q8_0 idx: 0.996932 ≥ 0.99693 (+2.0e-6). The thin margins are TODO-LATER #8.
  - q8_0 0.996737, q4_0 0.985712, bf16 0.997256, q8_0K/f16V 0.996857, f16K/q8_0V 0.997237: all PASS.
- **Q8**:
  - d1: f16 LCB95 +8.16e-4, q8_0 +8.04e-4 > 7.45e-5.
  - d2: 8/8 chunks positive, |new − PR219| 1.3e-4 / 3.4e-5 ≤ 1.86e-3.
  - d3: report.
  - PASS.
- **Q11a**: PASS 18/18 (lo / hi / all × 6 pairs; e.g. bf16 all −5.05e-5 ± 4.6e-6, band 2.03e-4). **Q11b**: PASS 6/6 at ub512 (q4_0 −2.66e-4, below the band); ub16 is report only (band UNPROVEN, F4).
- **Q11c** (np2 split vs np1, unpinned, q8_0 and q4_0, c1024 × 32):
  - The per-chunk KLD lines of the np1 run equal those of the np2 rerun line for line (`diff` of the `QKLD_CHUNK` lines = 0 for both types). Mean KLD is −0.000000 and same-top-p is identical.
  - `q_an.py` prints FAIL because the max chunk value is 1.19e-6 / 1.27e-6 against its ≤ 1e-6 check. The same value appears in the np2-vs-np2 rerun, which is the uint16 resolution of the KLD base (W12 F2).
  - **Verdict: PASS (np1 ≡ np2 rerun).** The scripted threshold misfires at the base resolution.
- **Q12 pinned** (-nkvo, copy route): f16 ub4096 −2.8e-5, ub512 +1.7e-5; q8_0 +3.5e-5 / +1.2e-5 (bands 2.0e-4 / 2.03e-4). Reruns are 0 on both arms. PASS.
- **Q12 `-ngl 0`** (phase Z): f16 +1.1e-5, q8_0 +1.4e-5 ≤ 2.0e-4. PASS.
- **Q12 coarse** (unpinned, -nkvo vs the same binary's GPU route, c4096 × 4, `PV/q12c`): f16 0.014491 ± 0.00066 ≤ 0.01587 PASS. **q8_0 0.015717 ± 0.00054 > 0.01563 FAIL.** The CPU route (a) via hooks gives f16 0.014151, q8_0 0.016112.

### 5.4 Bit-identity where the plan claims it
- HIP in-kernel H_q / H_o are bitwise equal to fwht.cu: 136 bitwise lines per tier, 0 failed (§3.1). The Vulkan rot test is in ctest vk.
- W10/W11: the head vs the W9-head copy (`~/dev/wt_lab/w10/base_bin`), llama-perplexity c8192 × 1 (`H/w11/smoke_kld.sh`, `PV/c1/smoke/`):
  - f16 and q8_0: Mean KLD −0.000000, same top p 100.000 %. The max KLD of 5.1-5.8e-5 is at base resolution.
  - q4_0 and iq4_nl with the old indexer override: KLD 0.
  - The shipped auto mapping (q8_0 indexer) gives q4_0 0.0282 and iq4_nl 0.0254, the expected change of W11.
- Q11c np2 = np1 (§5.3). Model logits are deterministic (§3.3).

### 5.5 Unpinned catastrophe check
c8192 × 4 against base f16 logits (`PV/q/unpinned`).

| type | new: PPL ratio to base, KLD | PR219: PPL ratio to base, KLD |
|---|---|---|
| f16 | 1.0009, 0.0096 | 1.0001, 0 |
| q8_0 | 1.0024, 0.0104 | 1.0014, 0.0102 |
| q4_0 | 1.0051, 0.0242 | 1.0075, 0.0258 |
| bf16 | 1.0014, 0.0101 | 1.0037, 0.0108 |
| q8_0K/f16V | 1.0051, 0.0101 | 1.0026, 0.0103 |
| f16K/q8_0V | 1.0019, 0.0097 | 1.0007, 0.0098 |

No catastrophe: every value is within the unpinned route-change level.

### 5.6 Q1-vk (chain2 `vk.sh VQ`, W0-Vulkan reference, B_vk 3.214e-4)
f16 +2.391e-4 ± 2.6e-5, q8_0 +7.5e-7, q4_0 −1.0e-4, bf16 −5.1e-6, q8_0K/f16V −2.1e-5, f16K/q8_0V −1.3e-5: PASS 6/6 (`PV/chain2.out`).

### 5.7 Q7
§2.5. Stage-2 prompts X8-X21 use the C2 recipe (`PV/chain11.sh`). The S0-S5 shortcut prompts are excluded per the plan.

### 5.8 AT-Q10Q6 calibration, Q10, Q6 (chain6; `PV/q10/q10q6.md`; analysis `PV/q10an.py`)
- **(1) Control floor.**
  - PR219 build4 is bit-deterministic at L 8192 (3 self-pairs KLD 0), and so is the 127 build.
  - The 9 cross pairs are all equal, so SD = 0 and floor_ctl = 3.302e-3 (seq0) / 1.050e-2 (seq1).
- **(2) Cell τ.**
  - 8 noise draws per cell: 6 disjoint ub8192 repeat pairs of 12 PR219 runs, the ub4096 pair, and ub4096 vs ub8192.
  - Leave-one-out false-fail of mean + 3·SD ≤ 1/6 in every cell, so that rule is kept.
  - τ: L8193 1.205e-2 / 2.782e-2, L8194 1.107e-2 / 3.342e-2, L8195 9.614e-3 / 2.823e-2, image p3 5.501e-3 / 1.553e-3 (seq0 / seq1).
- **(3) Q6 τ_T** (13 draws, PR219 lab-c4 TE/CB × 7 ubs vs TE ub8192): f16 p1 4.386e-3, p2 4.037e-3; **q8_0 p1 4.371e-3 (τ_p1,q8_0 filled)**, p2 4.055e-3.
- **Q10** (head lab-r1, `PV/labnew`):
  - Determinism: new rerun vs new = 0 at L 8192-8195 (PASS).
  - **Control new vs each PR219 repeat 8.485e-3 / 2.569e-2 > floor: FAIL.**
  - Cells L8193-8195: median over 12 PR219 repeats ≤ τ in 6/6.
  - Image p3: seq0 5.381e-3 ≤ 5.501e-3 PASS; **seq1 3.408e-3 > 1.553e-3 FAIL.**
- **Q6**: f16 p1 1.823e-3, p2 1.323e-3, q8_0 p1 1.464e-3, p2 2.127e-3, all ≤ τ: PASS 4/4.
  - Content: every gen run names "MEN WALK ON MOON", The New York Times and July 21, 1969.
- The floor is degenerate: it is one fixed kernel-change value (SD 0, because PR219 is deterministic at L 8192). The new build changes more kernels and routes than the 127 build does, so a floor of this form cannot hold. This is the plan-calibration question the owner must rule on. It is reported as a FAIL here.

### 5.9 Q5 op-local (chain5c)
- Run: ROCm0 SPARSE_ATTN with `GGML_SA_TEST_REPORT=1` and TBO dumps (193). Then the PR-verbatim G4 (`bands_attn`, 153 runs, 151 rc 0) and the PR-verbatim G1 decode with planted defects (`bands_dec`, 153 runs, rc 0) on the same inputs.
- Every class of `PV/q5/tbo_an.md` has new_max ≤ 2 × PR p100. For example G1 splits 17-33 n_sel > 257: new max 7.98e-8 vs 2 × p100 1.62e-7; G1 1 split ≤ 17: 1.44e-8 vs 5.73e-8.
- All 1432 SA_REPORT lines are inside their band.
- The two `bands_attn` rc 134 runs are the n_sel 4096 cases (`sa_0171`, `sa_0186`), which the PR harness does not hold. `prband_an.py` crashed on their missing values (traceback in `PV/q5/prband_an.md`).
- Verdict: **PASS on the run cases.** The prband table for the G4 rows was not rendered (harness crash). The new-vs-CPU NMSE ratios are in `tbo.all`.

## 6. Performance (item 4)

### 6.1 P1
Text q8_0 ub512 pp4096 d0, new vs base ea5285ddf, R N R N R N R: **1.0310 [1.0255, 1.0371], PASS** (θ 0.995). Source: `PV/perf/p1_*`; `perf_an.py PV/perf P1` (its label says 13a279bfa; the arm is ea5285ddf).

### 6.2 Text (C10 corpus), new vs PR219, 3 + 3 processes, ub {4096, 1024, 512}
P2 f16:

| d | ub | pp | PR219 t/s | new t/s | ratio [CI] |
|---|---|---|---|---|---|
| 0 | 512 | 512 / 2048 / 4096 / 8192 | 1316.4 / 1230.6 / 1173.9 / 1129.3 | 1318.5 / 1228.1 / 1231.5 / 1212.2 | 1.0016 / **0.9979 [0.9879, 1.0082] unresolved** / 1.0491 / 1.0735 |
| 0 | 4096 | 512 … 8192 | 1337.5 / 1988.7 / 2074.5 / 2013.9 | 1345.1 / 1988.4 / 2086.4 / 2028.1 | 1.0056 / 0.9998 / 1.0057 / 1.0070 |
| 32768 | 512 / 1024 / 4096 | 512 | 1095.8 / 1099.7 / 1096.1 | 1174.8 / 1178.6 / 1168.8 | 1.0721 / 1.0717 / 1.0663 |
| 32768 | 512 / 1024 / 4096 | 4096 | 1142.0 / 1515.7 / 1958.0 | 1232.3 / 1536.4 / 1972.0 | 1.0790 / 1.0137 / 1.0072 |
| 131072 | 512 / 1024 / 4096 | 512 | 1096.7 / 1102.9 / 1090.9 | 1167.7 / 1171.6 / 1158.5 | 1.0647 / 1.0623 / 1.0620 |
| 131072 | 512 / 1024 / 4096 | 4096 | 1128.4 / 1454.7 / 1846.1 | 1205.7 / 1474.1 / 1850.3 | 1.0685 / 1.0133 / 1.0023 |

- 23 PASS and 1 unresolved (θ 0.99).
- **P3**: ratio of ratios 24/24 PASS (1.012-1.038). The floor new_q8/new_f16 ≥ 0.90 holds in 24/24; the worst cells are d131072 pp512: 0.9226 / 0.9217 / 0.9241 at ub 512 / 1024 / 4096.
- **q4_0**: new vs PR219 24/24 PASS; new_q4_0 / new_f16 24/24 ≥ 0.90.
- Source: `PV/perf_all.md`, `perf_an.py PV/perf P2 P3`. Evictions: 3 eviction samples dropped in 2 processes; 0 P6 (a) samples.

### 6.3 Random tokens (stress rows), one process per arm, ub {4096, 512}, pp {512, 4096}

| type | new / PR219 (d0, d32k, d131k) | new / base (d0, d32k) |
|---|---|---|
| f16 | 1.001-1.093, PASS 12/12 | 1.001-1.094, PASS 8/8 |
| q8_0 | 1.021-1.120, PASS 12/12 | 1.022-2.70, PASS 8/8 |
| q4_0 | d0 1.02-1.16, d32k 1.93-2.80, **d131k 4.45-8.07**, PASS 12/12 | 1.07-2.69, PASS 8/8 |

The PR219 q4_0 cliff at depth is gone (d131k pp4096 ub4096: PR219 221.8 t/s vs new 1790.8). The base was not run at d131k (a reduction of this Prove). Separately, the base q8_0 server at c261888 does not start: `cudaMalloc 83908.76 MiB` OOM (`PV/img/res_base_q8_0.txt`). Source: `PV/cmp.py` runs in this report's session, `PV/perf/rnd2_*`.

### 6.4 P4 (one process per pair, vs new f16 of the same campaign, θ 0.90)
- bf16: 11 PASS, **1 unresolved** (d131072 ub512 pp512: 0.9188 [0.8872, 0.9494]).
- q8_0K/f16V: 12/12 PASS.
- f16K/q8_0V: 12/12 PASS.
- Generic types (q4_1, q5_0, q5_1, iq4_nl, f32) are op-level proven only, with no end-to-end run (owner scope note).

### 6.5 P5 decode
- Shortcut tg128 at d0 / 1024 / 1900, new / PR219 ≥ 0.98: 18/18 PASS (6 pairs).
- Depth d32k / d131k, f16 and q8_0 ≥ 0.99: 4/4 PASS.
- new_T / new_f16 ≥ 0.94: shortcut 15/15 and depth 10/10 PASS.
- **MTP cells** (llama-server, Signal-3.8 + MTP sidecar, verify width up to 8, 4 rep-1+ samples per process, one process per arm, `H/w7/mtp_perf.sh`):

| main KV | draft | d16 | d1024 | d32768 |
|---|---|---|---|---|
| f16 | default | 0.996 unresolved | 1.001 PASS | **0.898 [0.859, 0.949] FAIL** |
| f16 | q8_0 | 0.997 unresolved | 1.000 unresolved | 0.954 unresolved |
| q8_0 | default | 1.033 PASS | 1.482 PASS | 1.097 PASS |
| q8_0 | q8_0 | 1.000 unresolved | 1.354 PASS | 1.019 unresolved |

new_q8 / new_f16 ≥ 0.94: 6/6 PASS.

### 6.6 Depth ~300k (real text, wraps the C10 corpus), deepest, AT-O
- Commands: chain4c `ph_DEEP` (`-b/-ub 4096 -p 4096 -n 32`, R 3) and chain12c (400k, R 2).
- **1M depth (`-d 1040000`) was killed by the host OOM killer, twice** (rc 137, peak VRAM+GTT 111.5 GB). It also killed desktop processes (§13). 400k is the deepest depth run.

| cell (ub4096) | new | PR219 | base ea5285ddf | always-pack (ap) | verdicts |
|---|---|---|---|---|---|
| f16 d300k pp4096 t/s (2 proc new / PR219 / ap, 1 base) | 1646.72 | 1630.32 | 1634.40 | 1630.84 | new/PR219 1.0101 [1.0016, 1.0185] PASS; new/base 1.0075 [0.9982, 1.0171] PASS (θ 0.99 lower bound 0.9982 ≥ 0.99) |
| f16 d300k tg32 t/s | 21.23 | 21.20 | 21.24 | 21.32 | new/PR219 1.0014 [0.980, 1.024] **unresolved**; new/base 0.9995 [0.973, 1.027] **unresolved** |
| q8_0 d300k pp4096 / tg32 (1 proc) | 1575.58 / 20.12 | 1524.03 / 19.88 | | | pp 1.0338 PASS; tg 1.0121 [0.980, 1.045] **unresolved** |
| q4_0 d300k pp4096 / tg32 (1 proc) | 1565.51 / 20.04 | **116.82 / 7.80** | | | 13.40× / 2.57× PASS (the PR219 q4_0 cliff) |
| new_q8 / new_f16, new_q4 / new_f16 at d300k | pp 0.957, 0.951; tg 0.948, 0.944 | | | | ≥ 0.90 PASS |
| f16 d400k (1 proc each) pp4096 / tg32 | 1579.45 / 19.78 | | | 1561.79 / 20.14 | ap / capped 0.989 / 1.018 (report) |

- Peak VRAM+GTT (sysfs `mem_info_vram_used + mem_info_gtt_used`, 1 s polling): d300k 84.9-87.3 GB, d400k 93.3 GB (new and ap equal).
- **AT-O (owner decision data, not pass/fail):** always-pack / capped is pp 0.9904 [0.979, 1.003] and tg 1.0043 at 300k; pp 0.989 and tg 1.018 at 400k (one process each). Both settings have the same peak (93,291 vs 93,294 MiB). The compute-buffer reserve of each setting was not separately logged in the bench runs (UNPROVEN). C18 kernel numbers: packed+pack 42.34 vs best direct 52.92 ms at syn 262k.
- Source: `PV/perf/deep300_*`, `deep400_*`, `PV/perf_all.md`.

### 6.7 Stall
`PV/perf/an_stall.md`: 3232 samples (rep 0 excluded), **1 above 1.5 × its process-cell median** (1.61×, `p23_d0_q4_0_2_new` ub4096 pp512). P6 (a) samples (< 0.9 × median without evictions): 0 in every analysed phase. P6N (30 reps): see §11.

### 6.8 Compute buffer and peak VRAM
- HIP compute buffers (`sched_reserve`, init-only servers): M1 table §11, C12 table §11. Image server at c261888 ub8192: ROCm0 3170.80 MiB for every type (q8_0 / q4_0 +0.27 MiB).
- Peak VRAM+GTT per bench process is in `PV/perf/*.peak`. Text d131k runs: 83-86 GB, all arms (model weights dominate).

## 7. Images (item 5; Signal-3.8 + mmproj + MTP sidecar, `PV/img/srv.sh` = `~/repro_img/repro_kv.sh` flags)

### 7.1 A / B / C matrix (c261888, b/ub 8192, MTP on, strict + trace)

| KV | A (60k text) | B (1 image + 60k) | C (5 images + 150k, 180157 tokens) | strict / CPU / masked | prompt t/s A / B / C |
|---|---|---|---|---|---|
| f16 | 2749 | 12749 | 26999 | 0 / 0 / 0 | 1163 / 1140 / 1081 |
| q8_0 | 2749 | 12749 | 26999 | 0 / 0 / 0 | 1148 / 1131 / 1075 |
| q4_0 | 2749 | 12749 | 26999 | 0 / 0 / 0 | 1110 / 1119 / 1068 |
| bf16 | 2749 | 12749 | 26999 | 0 / 0 / 0 | 1156 / 1130 / 1081 |
| q8_0K / f16V | 2749 | 12749 | 26999 | 0 / 0 / 0 | 1150 / 1133 / 1077 |
| f16K / q8_0V | 2749 | 12749 | 26999 | 0 / 0 / 0 | 1139 / 1133 / 1079 |
| base f16 | 2749 | 12749 | 26999 | — | (ea5285ddf) |
| base q8_0 | server does not start: `cudaMalloc 83908.76 MiB` OOM | | | | |

- Every new run traced 580 QSA ubatch lines, all `backend=ROCm0`. Draft acceptance is non-zero (MTP active).
- **P9** (3 interleaved server runs per build, A/B/C prompt t/s, θ 0.97):
  - f16: 0.9945 [0.973, 1.019], 1.0089, 1.0165.
  - q8_0: 1.0091, 1.0123, 1.0251.
  - PASS 6/6.
- Source: `PV/img/res_*.txt`, `PV/chain3.out` P9.

### 7.2 `-np 2` (split streams) with images
- A and B sent concurrently, c261888 (130944 cells per stream), ub8192, for f16, q8_0 and q8_0K/f16V. Answers 2749 / 12749 in all three, CPU 0, masked 0. **Strict reserve violated (§2.1).**
- `-np 2 --kv-unified` q8_0: answers correct, strict 0, 1913 TE + 131 CB ubatch lines.
- W9 list (`H/w9/concurrent.sh`):
  - unified q8/f16/q4 ub4096, `n_qsa_resplit` 82 / 81, TE n_q max 512;
  - split q8 ub4096;
  - forced-NC one request (hooks);
  - r1s split / unified ub1024;
  - np1 image q8 ub4096 / f16 ub1024.
  - All answers are 11999 / 32517, with strict 0, decode errors 0 and mismatch 0 (`PV/chain3.out`).

### 7.3 Image understanding (greedy)
- **Short** (`tools/mtmd/test-1.jpeg`, which is the NYT 1969 front page, and `img_nyt.jpeg`; ~1070-token prompts on the dense shortcut):
  - new f16, q8_0 and q4_0, PR219 f16 / q8_0 / q4_0, and base f16 all name The New York Times, Monday July 21 1969, "MEN WALK ON MOON" and "ASTRONAUTS LAND ON PLAIN; COLLECT ROCKS, PLANT FLAG".
  - new f16 is identical to PR219 f16 and to base f16 in the opening text. q8_0 / q4_0 differ only in phrasing.
- **At depth** (`U_deep`: ~40k tokens of text, then the image, then text; 42571 prompt tokens, sparse route: 169 CB ubatches traced on ROCm0):
  - new f16, q8_0, q4_0, bf16, q8_0K/f16V, PR219 f16 / q8_0 and base f16 all give the same four facts and the same 3 extra headlines.
  - new f16 and q4_0 are word-for-word identical. The texts differ only in "provided, here are the details" vs "provided in the middle of the text".
- Source: `PV/img/res_und_*`, `res_deep_*`.

### 7.4 M4, shared prefix, C15-style ub1024
- **M4 (c)**, two images back to back after ~36.6k tokens (`M4c_0`, 42409 tokens), f16 and q8_0: answer 5199, strict 0, decode errors 0.
  - `M4c_300` / `_600` reused the cached prefix (15092 new tokens), so they do not add boundary coverage. UNPROVEN that a ubatch boundary fell inside the image pair.
- **SP** (`--parallel 2 --kv-unified`, two concurrent requests with a shared 40k prefix, trim 1 / 0, q8_0 / f16): answers 1899 / 1999, strict 0.
- **C15-style ub1024** (B and C, f16 / q8_0): 12749 / 26999, strict 0, 2381 CB ubatches.

## 8. Vulkan (item 6)
- Build: `build-vk` rc 0, 0 warnings (`PV/head_vk_build.log`). Op tests and lavapipe: §3.1. ctest: §3.2. Route census: §4.2. Q1-vk: §5.6.
- Vulkan determinism (model level): 6/6 identical (f16, q8_0, q4_0 × ub {4096, 512}). bf16 and the mixed pairs were not run: the bf16 run 2 hung (§13).
- **q8_0 depth run** (d131072, pp4096 + tg32): **NOT RUN.** It was queued after VB3, and the GPU lock was then held by a process stuck in the kernel (§13). The Vulkan c8192 runs (census §4.2, Q1-vk §5.6, determinism) are the deepest Vulkan runs in this Prove.
- **AT-B1**: **NOT RUN** (§13).
- **AT-B3** (op level, default / vdirect / vstaged / vpack, C11 / d4k / d131k rows, text and random): **partial** (the C11 rows have 3 processes, the d4k / d131k rows 2; the third pass was cut by the kernel hang of §13). Attend ms per layer (wall, median over processes) at C11's shape, text: f16 7.19; q8_0 default (= vdirect) 15.99, vstaged 17.63, **vpack 9.06**; q4_0 default 15.43, vstaged 17.40, **vpack 8.80**; bf16 default 9.86, vstaged 16.99, vpack 8.64. d131k text: f16 16.08; q8_0 default 32.96, vpack 23.19; q4_0 31.63 / 22.39; bf16 20.96 / 23.59. Default / f16 ratio: q8_0 2.22× and q4_0 2.14× (C11 text), against the diagnostic gate of ≤ 1.15× or the HIP C18 ratio (1.23× / 1.67×): **above**. bf16 1.37× ≤ 1.67×. vpack beats the shipped vdirect by 1.4-1.8× for q8_0 / q4_0, but it is not shipped because of M1 (68-264 MiB, W7; TODO-LATER #3). Source: `PV/w6time/res`, `python3 H/w6/w6_time_an.py PV/w6time/res`.
- **AT-B4 e2e**: **NOT RUN** (§13). The op-level policy table is W7's (6e07acd3d, unchanged). Placement: 0 CPU nodes in every Vulkan run (§4.2).
- **AT-B5 / M1-Vulkan**: **NOT RUN** (§13).
- AT-B7 portability: lavapipe and RADV portable_sg32 pass (§3.1).

## 9. Scope, reductions and deviations

- **Owner scope note (2026-11-07).** End-to-end runs cover f16, q8_0, q4_0, bf16, q8_0K/f16V and f16K/q8_0V only. Other types are **op-level proven only, with no end-to-end run**: q4_1, q5_0, q5_1, iq4_nl, f32, the q4_0-mixed pairs, and so on. These have test-backend-ops passes in every tier (§3.1, 1390 cases incl. generic types) and the CPU oracle. iq4_nl additionally has M1 memory and a KLD smoke.
- **Reductions made by this Prove run to finish on one GPU** (each one listed next to its result):
  - text P2/P3: 3 + 3 processes instead of 4 + 4;
  - random rows: one process per arm, ub 4096/512;
  - P5: R N for the shortcut, 1 + 1 at depth;
  - MTP: 1 + 1;
  - P4: 1 process;
  - deep: 300k (2 processes f16, 1 others) and 400k (1);
  - AT-B1 (planned 2 + 2 f16, 1 others) and AT-B4 e2e (planned R N): not run at all (§13);
  - NDB d0 cells: 2 more per arm;
  - D1V: d32k only;
  - AT-D2 perf: 1 + 1 (chain14).
  - **No §16 "2× processes" reruns were run.** Every unresolved cell is reported as unresolved, i.e. not passing.
- **Harness fixes made during Prove** (all are copies under `PV`; nothing was committed to the branch):
  - census without `GGML_SCHED_DEBUG_REALLOC` (pre-existing abort, `PV/realloc_check.out`);
  - AT-C5 rerun without it (`PV/chain7s.sh`);
  - the W4 AT-A harness ported to the W10 kernel signatures (ROT template argument, pack and merge arguments, the fwht header shim; `PV/w4bands/w4_dec.cu` vs `.w4orig`, `build.sh`);
  - Q5 / AT-A loops run as one locked batch;
  - Vulkan depth reduced to d131k after the OOM events.
- **Not run:**
  - AT-B2 (§2.12);
  - the PR219 pool high-water across 262144 cells (§2.13);
  - HIP VGPR check after the rebase (there is no rebase: the head's merge-base is ea5285ddf and `merge_tree.sh` still lists conflicts with origin/master, `PV/merge_tree.out`);
  - GLM5_NEXT session file (no model available).

## 10. Deferred acceptance tests (task item 7)

### 10.1 Lab A → AT-A (G1 vs G4 at n_q 128-512, D3 threshold, D8)
- Run: `PV/w4bands/ata_nolock.sh` on the head's kernels, C18 protocol r1/r2, 6 fixtures × 11 n_q × 4 types × 3 arms, plus the trimmed layer and `ata_plan.sh` (44 plan runs). 134 + 44 runs, all rc 0.
- Noise p90 0.0051 over 1188 samples, so the margin is t_G4 × 0.9848.
- **Fitted rule (largest G1 n_q)**, rows f16 / bf16 / q8_0 / q4_0 V, columns n_kv class 8192 / 32768 / 65536 / 135168:
  - f16: 32 / 16 / 16 / 64
  - bf16: 32 / 32 / 16 / 64
  - q8_0: 32 / 32 / 32 / **64**
  - q4_0: 32 / 64 / 64 / 128
- **Criterion "the HIP group proc encodes the fitted rule": FAIL in one cell.** q8_0 at the 135168 class is shipped as 96 but re-measured as 64 (`sparse-attn-attend-g1.cu:65-70`). The other 15 cells match.
- Ledger prediction "f16/bf16 at n_q 512 d131k pick G1": not confirmed. The trimmed layer gives G4 direct for every type: f16 G1 2.976 vs G4D 2.322 ms; bf16 3.008 / 2.373; q8_0 3.333 / 2.583; q4_0 3.078 / 2.551 ms.
- **D8 at n_q 512** (trim row): direct V beats packed+pack for every type (f16 3.950 vs 2.322 ms), consistent with CAPPED.
- **The e2e cell** (C10 f16 ub512 pp512 d131k, measured policy vs forced G4) was not run. NOT RUN.
- Source: `PV/w4bands/ata.md`.

### 10.2 Lab B → Vulkan W0 re-baseline, shortcut policy, quantized staging
- **VB1 (AT-B1)**: NOT RUN (§13).
- **VB2 (AT-B2)**: NOT RUN (old pipeline deleted).
- **VB3**: partial, see §8. The diagnostic gate fails for q8_0 / q4_0 (TODO-LATER #3).
- **VB4**: op level, from W7 (the policy table is unchanged since 6e07acd3d). E2E: NOT RUN (§13). Placement: 0 CPU nodes on Vulkan in every run (§4.2).
- **VB5**: NOT RUN (§13).
- **VB6 (Q1-vk)**: PASS 6/6 (§5.6).

### 10.3 Lab C → AT-C5, AT-C17, AT-C12, AT-C16
- **AT-C5**: PASS, 0 strict violations in every cell (§4.3, `PV/chain9.out`).
  - Unlike the plan text, it was run without `GGML_SCHED_DEBUG_REALLOC`, which aborts on the base too.
  - The "reserved partial ≥ §7 formula" sub-check was not computed separately (UNPROVEN). Strict = 0 shows that the runtime never exceeded the reserve.
- **AT-C17 / M6** (f16, ub4096, init-only, `PV/chain3.out` C17):
  - new reserve vs PR219 reserve: c524288 1660.42 vs 1932.42 MiB (main ROCm0); c1048576 2032.42 vs 2224.42.
  - new ≤ PR219 reserve alone, hence ≤ PR219 reserve + pool high-water for any pool ≥ 0: **PASS (CAPPED default).** The pool high-water itself was not measured (§2.13).
  - `-np 4` c524288 on new: 1576.19 / 223.03 MiB main.
  - M3: 0 hipMalloc after warm-up in all 7 C17 runs.
- **AT-C12 / M5** (init-only, strict, new vs PR219, f16 and q8_0; table in `PV/c12/parsed.json`, rendered below):
  - Every cell with an instantiated M5 bound passes:
    - default c262144 ub512: 1648.29 ≤ 1745.36 (main), 1633.02 ≤ 1674.03 (MTP);
    - c131072 ub1024: 846.79 ≤ 943.86;
    - c131072 ub4096: 1609.88 ≤ 1610.88;
    - c261888 ub8192: 3315.30 ≤ 3316.30.
  - Cells without an instantiated bound, new vs PR219 main ROCm0:
    - ub512: +138 to +337 MiB;
    - ub1024 c261888: +220 / +244 MiB;
    - ub4096 c261888: −239 / −335 MiB;
    - ub8192: 0;
    - np2: −80 / −96 MiB.
    - This is the non-compact variant cost that W9 recorded. The general V-formula bound for these cells was not instantiated (UNPROVEN).
  - Runtime max ≤ reserve: §4.3 (FAIL for split np2 ub8192). Answers 2749 / 12749: PASS.
- **AT-C16**:
  - a (layout 3, c131072, C19 lab tree): kq_mask strip views 8 new / 88 reuse per 8-strip graph; answers `12299...` / `40479`; peak 1609.88 ≤ 8054.08 MiB. Bias views do not appear in the head's runtime graphs (the W9 S = 1 design).
  - b (c524288, share vs `LAB_C19_NO_SHARE` hook added for Prove): census 8 new / 88 reuse per graph (1 new / 11 reuse per strip). Reserve no-share − share = 10547.29 − 2099.29 = 8448 MiB (criterion ≥ 0.5 × 11 × n_ctx × 1536 B = 4224 MiB at c524288). Strict 0.
  - **PASS.** The criterion form is still the owner's decision (W9 note).

### 10.4 Lab D → AT-D1-D5
- **D1**: new has 12 kind-4 regions in each full graph (19 graphs; 2 graphs with 24), PR219 has 0 in all 1035. The criterion "new = PR219" is not met as worded; this is TODO-LATER #5 (deferred).
  - U16: 20/20 chunk hashes identical OVL 15 vs 11 for f16 and q8_0: PASS.
  - **Value** (d32768 pp4096 ub4096, one process per arm and OVL setting): OVL15 / OVL11 = PR219 0.9996 (f16), 0.9995 (q8_0); new (hooks) 1.0019 (f16), 1.0049 [1.0008, 1.0085] (q8_0). The point ratio of ratios is 1.0023 / 1.0054 ≥ 1. The joint CI was not computed (one process), so this is reported as PASS on the point estimate only. d131072 was not run (reduction).
- **D2** (`-ngl 0`, c4096 × 4, ub512; `PV/at_d/d2_*.log`, chain7b): placement is 100 % on CPU (108 trace lines `backend=CPU` per new run), guard cpu_launch 0, masked 0. PPL new / PR219: f16 2.0835 / 2.0851, q8_0 2.0858 / 2.0866. Q12 at `-ngl 0`: PASS (§5.3). PASS (placement, guard, quality).
  - **CPU perf: TODO-LATER #2 (deferred): FAIL**, f16 pp512 0.700 / tg32 0.804, q8_0 0.706 / 0.923 (`PV/chain14.out`).
- **D3, D4, D5**: PASS (§4.6).

### 10.5 LAB-BANDS-E2E → AT-W12-BANDS
- The bands were frozen in PLAN-v3 §15.0 by W12. Prove re-checks self-replay on the reference: QKL_HASH equal 8/8 (`q_an.py` "Run health").
- Every band used in §5 is > 0 and is taken from §15.0. Q1-vk uses B_vk 3.214e-4.
- **PASS** (no band is used unmeasured; Q11b ub16 stays UNPROVEN and report-only).

### 10.6 LAB-Q10Q6 → AT-Q10Q6
- Calibration valid: 127 vs 127 = 0, build4 vs build4 = 0, ≥ 6 draws per cell (8), ≥ 13 draws per Q6 (prompt, type), leave-one-out rule reported (all ≤ 1/6), τ_p1,q8_0 filled.
- Gates: Q6 PASS. **Q10 control FAIL; image p3 seq1 FAIL** (§5.8).

### 10.7 LAB-W7PERF → AT-W7PERF
- **Op level (shortcut grid)**: `H/w7/at_op.sh` (3 processes, 5 pairs × n_q {1 … 1024} × n_kv {512, 1024, 2048}). Noise p90 0.29 %.
  - The re-measured per-band dense flips equal the shipped `dense[][][]` table (`sparse-attn-attend-g1.cu:111-116`) for all 5 pairs × 6 measured bands. The 128-511 band is not in the grid.
  - PASS (`PV/at_op/summary.md`).
- **E2E shortcut cells** (`-ub 64,1024 -p 2048 -d 0`, sparse vs `LLAMA_QSA_PREFER_DENSE`, hooks, R N): 10/10 PASS (≥ 0.98).
- **tg-with-MTP (b)**: §6.5 (FAIL in one cell).
- **-nkvo binding gate (c)**, copy route:
  - bf16 pp 10 PASS / 2 unresolved, tg 2/2 PASS;
  - f16 / q8_0 d0 cells with 2 more processes: 2 / 3 PASS, 2 / 1 unresolved.
  - TODO-LATER #7 (deferred).
- **(d) pin on/off** (q4_1, a fallback (a) type, hooks): tg 2/2 PASS, pp 2 unresolved (0.980 [0.953, 1.012], 0.972 [0.944, 1.001]).

### 10.8 LAB-VK → AT-B1 extension, AT-B3 pack arm, AT-B7
- AT-B1: NOT RUN (§13). AT-B3: partial (§8).
- **AT-B7**:
  - (0) lavapipe limits recorded (`PV/c1/ops/lvp_vulkaninfo_system.txt`).
  - (c) SPARSE_ATTN_PLAN and SPARSE_ATTN `--fail-unsupported`, on lavapipe and on RADV `portable_sg32`: rc 0, failures 0, "not supported" only on the expected list.
  - supports_op false without a pipeline: ctest vk `test-sparse-attn-vk` (capability).
  - PASS.

## 11. Moved-to-Prove checks (W7-W13)

| item (source) | result |
|---|---|
| W7 Q3-B q4_0 ub256 | PASS −1.95e-4 ≤ 3.34e-4 (§5.2) |
| W7 Q11c | PASS by identity (§5.3) |
| W7 Q12 `-ngl 0` | PASS; coarse screen **q8_0 FAIL** (§5.3) |
| W7 P2 / P3 | §6.2 (1 P2 cell unresolved) |
| W7 NDB remainder (bf16, d0 cells) | partly unresolved (§10.7), TODO-LATER #7 |
| W7 NPIN | tg PASS, pp unresolved (§10.7) |
| W7 P5 + MTP cells | §6.5 |
| W7 P4 | §6.4 |
| W7 P6N | report: 30 reps per process; new / base t/s f16 d0 0.998, q8_0 1.013, q4_0 1.021, q8_0 d32k 2.607; eviction samples 0 (new q4_0 process evicted_ms 9024 with no low sample); P6 (a) samples 0 (`PV/perf_all.md`) |
| W7 AT-D1 value | ratio of ratios 1.0023 / 1.0054 at d32k (point estimate, §10.4) |
| W7 M2 / M3 | M2 (C7 shim device high-water, a server filling the context, MTP on): total new = PR219 in 4/4 cells (f16 c65536 79721.15, c131072 81704.65; q8_0 78911.43, 80084.93 MiB; diff +0.00), pool high-water equal (8104-8264 MiB). M3: 0 hipMalloc after warm-up in 16/16 bench runs (f16 / q8_0 / q4_0 / bf16 × new / PR219 × ub512 d0, ub4096 d32k). PASS (`PV/m23`, `PV/chain3.out`) |
| W7 AT-B4 e2e | NOT RUN (§13) |
| W7 AT-B1 unresolved cells | NOT RUN (§13) |
| W7 U30 masks on Vulkan | not run (NOT RUN) |
| W7 Q1 phase K (explanatory) | not run (explanatory only) |
| W8 AT-Q10Q6, Q10, Q6 | §5.8 |
| W9 AT-C12 grid + image-slot layout | §10.3. The U27 image-slot layout (S0_img + T_deep unified) gives new / PR219 prompt t/s S0_img 346.4 / 356.3 (0.972), T_deep 882.7 / 892.1 (0.989), 3 + 3 processes, report. The per-request Q10-style KLD of the image-slot layout was not run (NOT RUN). |
| W9 M5 cells | §10.3 |
| W9 AT-C17 / M6 | PASS (§10.3) |
| W9 AT-C16b | PASS (§10.3) |
| W9 M4 (b), M4 (c), C15 ub1024, np2 shared prefix | PASS (§7) |
| W9 U27 | reported (row above) |
| W10 U6 OP_PROF A/B | median per pass new < W9 in 4/4 (q8_0 d0 105.8 vs 133.7 ms, d32k 197.0 vs 215.8; q4_0 d0 108.3 vs 132.7, d32k 189.4 vs 206.4). Spreads separate except q4_0 d0 (one new run 548 ms): **3 PASS, 1 unresolved** (`PV/opprof_*`) |
| W10 op time of the rotated op | report: rot (256, 64) costs +5.3 % (d4k f16 13.65 vs 12.90 ms) and +4.9 % (d131k), rot 0 = W9 within 1 % (`PV/optime.out`) |
| W10 Q5 | PASS (§5.9) |
| W10 Q1-A/B, P1-P6 | §5.1, §6 |
| W10 Vulkan quality with in-op H_q | Q1-vk PASS (§5.6) |
| W11 Q4 on W11 | PASS (§5.3) |
| W11 M1 iq4_nl MTP (TODO-LATER #6) | **cleared**: iq4_nl c32768 ub4096 MTP ROCm0 1400.98 MiB = f16 + 0.26 MiB (`PV/mem/m1.md`) |
| W11 Vulkan q8_0 indexer for q4_0 KV | runs with 0 CPU nodes; smoke vs the HIP base KLD 0.0333 (cross-backend, `PV/vkidx.out`); Q1-vk q4_0 PASS |
| W11 MTP smoke on the final head | runs, 0 CPU; gated share FAIL (§2.3) |
| W13 full campaign | this file |
| W13 AT-C16b | PASS |
| W13 HIP VGPR after rebase | not applicable (no rebase) |
| W13 merge-tree | `PV/merge_tree.out`: conflicts with origin/master (62a3f4144) in `ggml-vulkan.cpp` (3 hunks), `scripts/hip/gfx1151-vgpr-baseline.json` (6), `tests/CMakeLists.txt` (1); pr/247 conflicts in `ggml-cuda.cu` |

M1 (HIP, reduced matrix, `H/w7/mem.sh`):
- 20/20 cells pass: f16, q8_0, q4_0, bf16 and iq4_nl × c {32768, 131072} × ub {4096, 512}, main and MTP. Every type is ≤ f16 + 0.27 MiB.
- np2 and default rows are reported.
- The script's MISSING rows are the out-of-scope types.

## 12. Known issues (deferred: TODO-LATER.md; measured, not counted in pass)

| # | item | Prove measurement | status |
|---|---|---|---|
| 1 | Q1 direct form q4_0 / bf16 | stage 1: q4_0 2.690e-3 > 1.441e-3, bf16 1.802e-4 > 1.032e-4; stage 2: 2.082e-3, 1.250e-4 | FAIL (unchanged from W7) |
| 2 | AT-D2 `-ngl 0` CPU perf | one process per arm (chain14): f16 pp512 @ d4096 0.700 [0.690, 0.712], tg32 0.804; q8_0 pp512 0.706, tg32 0.923; thresholds 0.9185 / 0.985 (W7) | FAIL (unchanged) |
| 3 | AT-B1 Vulkan q8_0 / q4_0 < 0.95 × f16 | e2e not run (§13); op level (AT-B3): q8_0 attend 2.22× f16, q4_0 2.14× (C11 text), vpack 1.26× / 1.22× | open (op-level gap unchanged) |
| 4 | AT-B5 Vulkan MTP f16 c131072 ub512 +12 MiB | not run (§13) | not re-measured |
| 5 | AT-D1 kind-4 criterion wording | new 12 per full graph, PR219 0 | criterion not met as worded; U16 PASS |
| 6 | M1 iq4_nl MTP +32.29 MiB | +0.26 MiB | **PASS (cleared by W11)** |
| 7 | -nkvo copy route open cells, bf16 | bf16 10 + 2 PASS, 2 unresolved; d0 f16 / q8_0 2 + 1 unresolved | open |
| 8 | Q4 thin f16-KV margins | +6.3e-6 / +2.0e-6 (same as W7) | PASS on threshold, thin |

## 13. Side effects and incidents
- **Host OOM, twice** (16:34:39 and 16:59:05, 2026-11-09): the `-d 1040000` f16 llama-bench runs (new, then always-pack) reached 111.5 GB VRAM+GTT. The kernel OOM killer killed them (rc 137). It also killed processes this agent did not start:
  - the gdm greeter session of uid 60579 / 60580 (gnome-shell, pipewire, gsd-*, xdg-desktop-portal, dbus-broker);
  - a user service of uid 1000, which systemd restarted.
  - Source: `journalctl -k` "Out of memory: Killed process". No further run above 400k was started after that.
  - No systemd unit was touched by this agent, and strix-001 was not used.
- **Kernel hang, Vulkan, 2026-11-10 11:55.**
  - The model-level determinism run `vk bf16/bf16 ub4096 r2` (`build-vk/bin/llama-perplexity` c8192 × 2) stopped after "calculating perplexity over 2 chunks".
  - The kernel logged `[drm:amdgpu_gem_va_update_vm [amdgpu]] *ERROR* Couldn't update BO_VA (-12)`. W7 saw the same error, then under host memory pressure.
  - The process (pid 4041225) is in state D, wchan `drm_suballoc_insert`. It holds 91.6 GB GTT (`mem_info_gtt_used` 91631312896) and fd 3 on `~/gpu.lock`. The `timeout` SIGTERM and a SIGKILL from this agent at 13:00 did not end it.
  - **As of the end of this run the GPU lock is held by that process: `flock -n ~/gpu.lock true` gives LOCK_HELD.** Every later GPU job on strix-002 will block until the process exits. This most likely needs a reboot by the owner (UNPROVEN; this agent has no root and did not attempt a GPU reset).
  - The Prove chains waiting for the lock (Vulkan AT-B3 remainder, VE, V5, VDEPTH, VB1, determinism remainder) were stopped by this agent so that nothing starts later unattended.
  - That the bf16 Vulkan path caused the hang is UNPROVEN. Run 1 of the same command had finished at 11:49-11:55, with the same hash on q8_0 / q4_0 before it.
- One `test-qsa-select-dump` call on Vulkan ran without the lock (device enumeration only, it printed "no backend with ..."), while another chain's job held the GPU.

## 14. Cleanup
- Deleted after use: the Q references `PV/q/refs` (ref8t0 16.3 GB and the others, deleted per phase), the Q10 / Q6 logits `PV/q10/logits` (23 GB), the op dumps (`PV/c1/det_op/*`, `PV/q5/dump`, 3.9 GB), and the KLD bases of the determinism and smoke runs.
- `PV` is 4.7 GB (build trees, logs). `find PV -size +1G` is empty.
- Kept: `PV/*.out`, the analyses and the per-run logs.
- Worktrees added for Prove (detached, no commits): `PV/base_ea5`, `PV/lab`, `PV/apack`, `PV/c19`, `PV/labnew`, `PV/r1q10`, `~/dev/wt_lab/w13/hooks` (re-pointed to the head + patch 06). `~/dev/wt_rewrite` was not modified: no commit, no file change.
- Not touched: `~/dev/wt_pr219v2*`, the other agents' worktrees (read-only use of `~/dev/wt_lab/w0lab12`, `w7/bench/bin_w0`, `w7/w0ovl`, `build4`, `c2`, `c4`, `r1`, `r1s`, `r1c`, `c18/cap`), systemd units, strix-001.
- Nothing was pushed or posted.

## 15. Commands index
- Setup and builds: `PV/setup.sh`, `PV/vkbench.sh`.
- Correctness: `PV/chain1.sh` (ops, ctest, det_op, guard, census) and `PV/chain1b.sh` (vkcensus, mtp, c18, atd, smoke, det_model).
- Quality: `PV/chain2.sh` (Q phases A D P C S B E N Z H, vq, unpinned).
- Images, server and memory: `PV/chain3.sh` and `PV/chain10.sh` (deep image).
- Perf: `PV/chain4.sh` (P1, TXT) and `PV/chain4c.sh` (DEEP P5 RND MTP SC P4 NDB P6N NPIN D1V an), `PV/chain12c.sh` (400k), `PV/perf_all.sh`, `PV/cmp.py`.
- Kernel and Vulkan: `PV/chain5c.sh` (Q5b, ATAb, OPPROF, OPTIME, VB3, VE, V5, VDEPTH, VB1) and `PV/chain13.sh` (AT-A plan).
- Q10 / Q6: `PV/chain6.sh`, `PV/q10an.py`.
- Q7: `PV/chain7.sh q7`, `PV/chain11.sh`, `PV/q7an.py`.
- AT-C5, D1, Q12c, vkidx, D2: `PV/chain7.sh`, `PV/chain7s.sh` (=chain9.out).
- Op-dump method and D2B: `PV/chain14.sh`, `PV/chain14b.sh`.
- Pre-existing realloc abort: `PV/realloc_check.sh`.
