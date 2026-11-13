# REBASE3: PR #276 (QSA sparse attention rewrite) onto master b620b0d9a

Status: complete (2026-11-13 01:50). Verification passed on the rebased head after one test fix (c456e6e5e); the VGPR baseline is regenerated (7e7c75b08). The 8080 server is restored (`/health` ok). The coverage gates are in PROOF-final3.md; one of them (M5) fails on a pre-existing PR commit and needs an owner decision.

Nothing was pushed and nothing was written to GitHub; GitHub was only read (`gh pr view`, `git fetch`). strix-001 was not used. Every build and GPU command on strix-002 ran under `flock ~/gpu.lock`, one at a time. test-backend-ops ran one op per process, never the full suite, never `-j`.

## 1. Result

- Laptop branch `qsa-rebased3` (`laptop:~/dev/qsa-rebased3`), head **`7e7c75b08`**, **96 commits** over master b620b0d9a (= over `origin/master` 8a44a73b3 by `rev-list`; the branch is 3 CI-only commits behind it, merge clean). Not pushed.
- 93 commits = the old PR head 49a6d75ab minus its VGPR commit, rebased; then 3 new commits:
  - `9a2ee52bd` tests : test-qsa-idx-type checks the indexer cells' image 2D position after a restore (previous agent; kept);
  - `c456e6e5e` tests : QSA_EXPAND engagement check names its census label (this job; §4.1);
  - `7e7c75b08` hip : regenerate the gfx1151 VGPR baseline on the rebased tree (this job; §5; `source_revision` = c456e6e5e).
- Every commit message ends with the `Assisted-by` / `Claude-Session` lines; author `Gaetan Puleo <12990773+gaetan-puleo@users.noreply.github.com>`. No amend, no reset.
- Verification (§4): every check passes; FLASH_ATTN_EXT is case-by-case identical to master on HIP and Vulkan; PPL is identical to 49a6d75ab on all 6 cells; ctest 21/21 × 3 builds.

## 2. Rebase decisions

### 2.1 Base
- Old PR head: `49a6d75ab` (`origin/qsa-rewrite`, laptop `qsa-final`), 94 commits on master 50c2c8cb5. Its last commit is the VGPR baseline of that base, which was dropped and regenerated (§5).
- New base: master `b620b0d9a`. Master commits brought in (50c2c8cb5..b620b0d9a, 3 merges, 29 files, +85 / -194):

| master commit | what | touches | decision |
|---|---|---|---|
| 093c552d7 (#280) | CI: build and test only the CPU, ROCm and Vulkan lanes | `.github/` only | (a) applies unchanged |
| 296034563 / a7b067e89 (#272) | `llama-memory-hybrid-idx`: the indexer cells take their image 2D position from the attention cache on restore (`llama_kv_cache::ext_2d_copy`) | `src/llama-kv-cache.{h,cpp}`, `src/llama-memory-hybrid-idx.cpp` | (c) intent kept: one textual conflict (§2.2); a new test proves it on the rewrite (§2.3) |
| dcbf22855 / b620b0d9a (#284) | grammar: `MAX_REPETITION_THRESHOLD` 100000 | `src/llama-grammar.cpp` (1 line; the rewrite does not touch it) | (a) applies unchanged |

- Master moved again during this job: `origin/master` is now `8a44a73b3` (#282, 9801f8ce2 + 7f7883126: Docker CI lanes, checkout token). It touches only `.github/workflows/` (4 files, +9 / -26), not src/, ggml/ or tests/, and `git merge-tree --write-tree origin/master qsa-rebased3` is clean. Per the brief, **no second rebase** was done; the PR is 0 behind on code, 3 commits behind on CI only.

### 2.2 The one conflict (#272)
- Rebased commit `86a1ab8a5` (original `e811ed7fa`, "qwen4exp : indexer key cache type auto ..., clearer restore error"), file `src/llama-memory-hybrid-idx.cpp`, `llama_memory_hybrid_idx::state_read`.
- Master inserted `mem_idx->ext_2d_copy(*get_mem_attn(), sinfos_attn);` right after the indexer's `state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn)`. The rewrite's commit wraps that read in `reading_idx = true; ... reading_idx = false;` (the clearer restore error).
- Resolution: keep the rewrite's wrapper and put master's `ext_2d_copy` after `reading_idx = false;`, before the pool-layout stale mark. Net effect of the rebased tree over the old one: exactly master's +1 line at that place.

### 2.3 New commits on the rebased branch
| commit | why |
|---|---|
| `9a2ee52bd` tests : test-qsa-idx-type checks the indexer cells' image 2D position after a restore | The rewrite's compact predicate (`qsa_key_2d`, the duplicate (pos, y, x) key) and the scan's ranking read the indexer cells' x/y, so #272 matters more on the rewrite than on master. The test saves/restores slot 0 (text + 8x8 image + text) on a weightless qwen4exp hybrid-idx memory, 1 and 2 slots, and checks every restored indexer cell's x/y and the next ubatch's compact verdict. Negative control (run by the previous agent, per the commit message; not re-run here): without the `ext_2d_copy` line it fails (image cells at x = y = 0, next ubatch non-compact). The test passes in all three builds (§4). |
| `c456e6e5e` tests : QSA_EXPAND engagement check names its census label | `-o QSA_EXPAND -b ROCm0 --fail-unsupported` failed 0/56 (no compare error): the check required a `QSA_EXPAND` census count, which the HIP census names `qsa_expand`. One-line fix, tests only (§4.1). |
| `7e7c75b08` hip : regenerate the gfx1151 VGPR baseline on the rebased tree | Last commit, as in REBASE2; `source_revision` = c456e6e5e, the commit before it (§5). |

### 2.4 Checks that the rebase changed nothing else
- The 93 rebased commits (before 9a2ee52bd) pair one to one with the 93 originals (`50c2c8cb5..49a6d75ab~1`): author name, e-mail, date and message are byte-identical for all 93, and every commit's set of added/removed lines is identical to its original's (only context lines differ, at the #272 hunk).
- Tree level: the `+`/`-` lines of `git diff 49a6d75ab~1 88c3a10e3` equal those of `git diff 50c2c8cb5 b620b0d9a` (master's delta, 29 files, +85 / -194): the rebased PR differs from the old PR by master's changes only.

## 3. Builds
- strix-002 worktrees: `~/dev/wt_rebase3` (branch `qsa-rebased3`, fast-forwarded to `c456e6e5e` and then `7e7c75b08`) and `~/dev/wt_master_b62` (master `b620b0d9a`). Fresh build directories, same flags as REBASE2 (`F/build.sh`, `F` = `strix-002:~/dev/wt_lab/final4`): HIP (`build`, gfx1151, ROCm core-10.1 clang), Vulkan (`build-vk`), CPU (`build-cpu`). All rc 0 at 9a2ee52bd.
- **Warnings identical to master**: HIP 87 lines / 17 unique (file, message) pairs on both, Vulkan 77 / 19, CPU 74 / 16; `h/warncmp.sh`: nothing only in the PR, nothing only in master.
- After `c456e6e5e` (tests only) `test-backend-ops` was rebuilt in the three build dirs (rc 0; its 10 warnings are the same 10 the full build had for that file: `verify/rebuild_*.log`).
- Lab tree `F/lab` = 9a2ee52bd + `lab-instrumentation.patch` + QPIN 01/02/03 (rc 0). It was not rebuilt for c456e6e5e, which changes only `tests/test-backend-ops.cpp`; src/ and ggml/ are identical.
- C10 llama-bench binaries: `F/bench/bin_{new3,mb}` (HIP), `bin_{vknew3,vkmb}` (Vulkan).

## 4. Verification of the rebased head
All under `F/verify/` (`F/verify.out` is the job log). Every job: rc as stated, watchdog `kerr_new=0`, no amdgpu kernel line.

| check | expected | result | verdict | log |
|---|---|---|---|---|
| ROCm0 SPARSE_ATTN `--fail-unsupported` + census, default, graphs off, forced g1 / g4 / vdirect / vpacked / generic / portable / fixup | 1439/1439 each | 1439/1439 in all 9 runs; census 492/492 compiled keys launched (forced runs: 388/388, 393/393, 24/24, 2/2 with the rest excluded by the switch) | PASS | `ops_hip/summary.txt` |
| ROCm0 SPARSE_ATTN_PLAN, default and `plan_global` | 185/185 | 185/185 both, census 2/2 | PASS | same |
| ROCm0 QSA_SCORE / QSA_STRIP / QSA_TE_STRIP / TOPK_QSA / QSA_EXPAND, `--fail-unsupported` | all pass | at 9a2ee52bd: 13/13, 6/6, 2/2, 4/4 and **QSA_EXPAND 0/56 (rc 1, no compare error)**; without the flag 56/56. After the fix `c456e6e5e`: 56/56, 13/13, 6/6, 2/2, 4/4 with the flag | PASS after fix (§4.1) | `ops_hip/hip_*_c456.log`, `hip_QSA_EXPAND.log` |
| Vulkan0 SPARSE_ATTN `--fail-unsupported` + census: default / scalar / portable / vstaged | 772/772 | 772/772 in all 4; census 85/85, 84/84, 81/81, 85/85 compiled keys launched | PASS | `ops_vk/summary.txt` |
| Vulkan0 SPARSE_ATTN_PLAN | 92/92 | 92/92, census 1/1 | PASS | same |
| Vulkan0 QSA ops (`--fail-unsupported`) | QSA_EXPAND 56/56 etc. | QSA_EXPAND 56/56, QSA_SCORE 13/13, QSA_STRIP 6/6, QSA_TE_STRIP 2/2, TOPK_QSA 4/4 | PASS | same |
| FLASH_ATTN_EXT case by case vs master b620b0d9a (`h/facmp.py`) | identical apart from deleted QSA cases | HIP: 5393 cases on both, OK 4050 / not supported 1343 on both, **0 cases only in one, 0 status differences** (summary line 4051/4051 both). Vulkan: 5393 on both, OK 5382 / not supported 11, 0 / 0 / 0 (5383/5383 both). Master has no QSA FA case left, so nothing is missing on either side. Vulkan TOPK_QSA 4/4 on both | PASS | `fa/` |
| ctest `-R 'test-qsa\|test-sparse-attn'`, HIP / Vulkan / CPU builds | 21/21 | 21/21 on each (the new check is inside test-qsa-idx-type, so the count did not change) | PASS | `ctest_{hip,vk,cpu_qsa}.log` |
| ctest, whole CPU suite (`-E test-backend-ops`), PR vs master | no new failure | PR 90/93, master 72/75. The same 3 fail on both: test-tokenizers-ggml-vocabs (vocab files missing), test-grammar-parser (abort), test-jinja-py; every other master test passes on both; the PR adds 18 tests, all pass | PASS (no regression) | `ctest_cpu_all{,_mb}.log` |
| HIP PPL c8192 × 2 chunks, f16 / q8_0 / q4_0 × ub512 / ub4096, PR vs 49a6d75ab tree (`wt_rebase2`) | identical; f16 1.9419 / 1.9485, q8_0 1.9530, q4_0 1.9523 | every pair identical per chunk: f16 ub512 `[1]2.0978 [2]1.9419`, f16 ub4096 `[1]2.1082 [2]1.9485`, q8_0 both ub `[1]2.1139 [2]1.9530`, q4_0 both ub `[1]2.1240 [2]1.9523` | PASS | `ppl/summary.txt` |
| #272 scenario e2e (Signal-3.8 + mmproj + MTP, 2 slots unified, prompt cache, NYT image to slot 0 four times, slot save/restore API), PR vs master, f16 and q8_0 | correct answers, no decode error | 4 arms: every image answer names The New York Times, 21 July 1969, "MEN WALK ON MOON"; 0 decode errors; save 1235 / restore 1235 tokens; after the file restore the request reuses 1072 cached tokens and decodes 4 (the restored-image path #272 fixes). Text identity with r1 varies run to run in **both** builds (MTP + 2-slot batching), restored answers identical to r1: PR f16 2/3, q8_0 1/3; master f16 1/3, q8_0 2/3; the r3/r4 prompt-cache requests reprocess the full 1076-token prompt (cache_n 0) on both builds | PASS (PR = master) | `t272/`, `t272_summary.txt` |
| lab image + `-np 2` strict-reserve smoke (lab tree, `LLAMA_STRICT_RESERVE=1`, trace, census; same t272 scenario) | 0 violations | main 0 (34 reallocs), MTP 0 (623 reallocs); 368 QSA trace lines, all `backend=ROCm0`; `LLAMA_QSA_RESPLIT` main 2 / MTP 2 with `decode_errors=0`; answers correct | PASS | `t272/srv_lab_f16.log` |

### 4.1 The QSA_EXPAND `--fail-unsupported` failure and its fix (`c456e6e5e`)
- `test_qsa_expand::verify_engagement` called `engagement_required(backend, "GGML_VK_TEST_REQUIRE_QSA_EXPAND")` with no census label. Under `--fail-unsupported`, every backend with a census proc then had to report a `QSA_EXPAND` count. The PR's HIP census names that dispatch `qsa_expand`, so on ROCm0 all 56 cases failed with no compare error (they pass without the flag). The flag and the HIP census are both the PR's, so the PR introduced this.
- Fix: pass `"QSA_EXPAND"` as the label, as QSA_SCORE does. The HIP skip proc reports "not a HIP dispatch census key" and the check is skipped there; on Vulkan the skip proc returns nothing for QSA_EXPAND (pipelines present), so Vulkan still requires one QSA_EXPAND dispatch per strip (56/56). One line, tests only.
- Not changed: on the CPU backend, `--fail-unsupported` still fails the QSA ops because CPU has no census proc ("a missing census proc FAILs", by design, REBASE.md).

### 4.2 Incidents during verification (not code)
- The first verify run (previous agent, 17:40) was stopped by the watchdog at GTT 108.2 GB during master's FLASH_ATTN_EXT. Cause: the 8080 server had been restarted at 18:00:07 while that job ran (`ps` start time), and its model load is what filled GTT. Rerun: master HIP FLASH_ATTN_EXT peaks at 0.45 GB GTT. Archived in `F/stopped1/`.
- The second stop (19:41) was MemAvailable 3.94 GB < 4 GiB during PPL f16 ub4096, which `verify.sh` ran without the load flags used everywhere else (`-lm mmap -lzm on-direct`, final2 `COMMON`). With them min MemAvailable is 33 GB. The flags change loading, not numerics: the PPL values equal 49a6d75ab's and PROOF-final2's. Archived in `F/stopped2/`.

## 5. VGPR baseline regeneration
`F/h/vgpr4.sh` (`F/vgpr/w/`): git archive of `c456e6e5e` (the code-final head), pinned image `docker.io/rocm/dev-ubuntu-26.04@sha256:4f5ed1bf6532a4b9920400401b0ad0f706af356dae61e40afefbfd6c8073f0ce`, `harness/w13/vgpr_in_image.sh` unchanged, under `flock ~/gpu.lock`.
- In the image: apt rc 0; `-Werror` configure/build rc 0 with 18 warnings, none fatal (13 `ld.lld` loop-not-unrolled, 4 host C++ in test-backend-ops, 1 libstdc++ `-Wnonnull`: the same 18 as REBASE2); `test_vgpr_check.py` 21 tests OK; EXPORT_METRICS build rc 0, 167 `*.lto.s`.
- Checker vs the committed baseline: PASS, 6824 kernels / 162 TUs, 0 VGPR-spilling, 11 SGPR-spilling all bounded (3 `sgpr_lane_spill`, 8 `sgpr_lane_spill_slower`).
- Regeneration (README python, verbatim): `minimum_kernel_count` 6824 -> 6824, TUs 162 -> 162, copies 7164 -> 7164, no TU count changed; `source_revision` 6805faa86 -> c456e6e5e; three keys (`conv2d`, `fattn`, `norm`) move in key order with unchanged values. Checker vs the regenerated baseline: PASS. Laptop `test_vgpr_check.py`: OK.
- **Re-check on the final head.** A chain restart re-ran `vgpr4.sh` on a git archive of `7e7c75b08` (the head including the committed baseline): apt / `-Werror` build / `test_vgpr_check.py` (21 OK) / metrics build all rc 0, 18 warnings, 167 `*.lto.s`; checker vs the **committed** baseline PASS (6824 kernels, 162 TUs, 0 VGPR spills, 11 bounded SGPR spills); a regeneration from that build differs from the committed file only in `source_revision` (c456e6e5e -> 7e7c75b08), which was discarded (`git checkout`). Logs: `F/vgpr/vgpr4_7e7c.out`; the first run's output is `F/vgpr/vgpr4_c456.out` (its `out/` build logs were overwritten by the re-run; its regenerated JSON is kept as `F/vgpr/baseline_c456.json`, md5 equal to the commit's).
- Committed as `7e7c75b08` on the laptop branch (the JSON was moved by base64, md5 `43b75218ba29ae11081686af4a2aae87` on both sides) and fast-forwarded into `~/dev/wt_rebase3`.

## 6. Not run / UNPROVEN
- Not run on the rebased head: lavapipe / other Vulkan devices (RADV only), the full test-backend-ops suite (never run, by rule: one op per process for the ops listed), CPU-backend QSA ops with `--fail-unsupported` (fail by design: no census proc).
- The #272 e2e test reaches the restored-image path only through the slot-file restore (r7: 1072 cached tokens reused, 4 decoded). The prompt-cache restores (r3, r4) reprocessed the whole prompt (cache_n 0) on PR and master alike, so they do not exercise the restored indexer 2D positions. The unit test 9a2ee52bd covers that path directly (with a negative control).
- Restored-answer text identity with the fresh answer is not stable run to run on either build (MTP + 2 slots), so it is not used as a criterion; the content check is.
- Master moved to 8a44a73b3 during the job (CI only); the branch was not rebased onto it (§2.1).
- The coverage gates and their findings (notably M5, the non-compact reserve growth bisected to 3eb54536f / rebased 4f9296cca) are in PROOF-final3.md.

## 7. Paths
- strix-002 (`F` = `~/dev/wt_lab/final4`): `verify.out` (job log), `verify/` (ops_hip, ops_vk, fa, ppl, ctest_*, t272_summary.txt), `t272/` (servers, results, slot files), `build/` (build logs, warnings), `vgpr/` (`vgpr4_c456.out`, `vgpr4_7e7c.out`, `baseline_c456.json`, `w/`), `stopped1/` (previous agent's run, stopped by the 8080 restart), `stopped2/` (the PPL MemAvailable stop), `h/` (scripts: lib4.sh, verify.sh, vgpr4.sh, t272.sh/.py, t272sum.py, facmp.py, warncmp.sh).
- Worktrees: `~/dev/wt_rebase3` (qsa-rebased3 at 7e7c75b08, clean), `~/dev/wt_master_b62` (b620b0d9a), `F/lab` (lab tree), `~/dev/wt_bis` (bisect worktree of wt_rewrite, PROOF-final3 §2.4).
- Laptop: `laptop:~/dev/qsa-rebased3` (branch `qsa-rebased3`, head 7e7c75b08, 96 commits over origin/master and over b620b0d9a); bundles in the session scratchpad (`r4.bundle`, `r5.bundle`).
