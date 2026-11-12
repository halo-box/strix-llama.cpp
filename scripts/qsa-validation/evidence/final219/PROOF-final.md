# PROOF-final: qsa-rewrite head 4d40d7f6a on strix-002 — PARTIAL (paused)

**Status: PARTIAL, paused by the owner at 2026-11-11 08:45 (CET, strix-002 clock). Verdict: pass = false for now.** The verdict is not final. Most checks are NOT RUN because the campaign was paused about 12 minutes after the first GPU job started. Nothing that ran has failed.

- Code under test: `~/dev/wt_rewrite`, branch `qsa-rewrite`, HEAD `4d40d7f6af58ef596c4c058d998ab08c5c8c18c2`, `git status --short` = 0. PR #276 head `02ad86516` is this tree minus `scripts/qsa-validation/`.
- strix-002 after the reboot: up 5 min at 08:29, `~/gpu.lock` free, GTT 20.8 MB, 120 GiB host memory free.
- Output root: `PFN = strix-002:~/dev/wt_lab/prove_final`.
  - Chain scripts: `lib.sh`, `chainA.sh` … `chainZ.sh`, `queue.sh`, `watchdog.sh`.
  - Logs: `chain*.out`, `ops/`, `img/`, `watchdog.log`.
- Every GPU job ran under `flock ~/gpu.lock`, one at a time.

## Pause actions (2026-11-11 ~08:45)
- Stopped by exact PID, all started by this agent:
  - queue 105673, chainA 8167, chainB 42912, watchdog 70115, and the remote monitor tail 107404;
  - the running job `GGML_CUDA_SA_FORCE=vdirect test-backend-ops -o SPARSE_ATTN -b ROCm0` (timeout pid 110313 / tbo 110320), 12 s into the run;
  - the waiting `img/srv.sh np2split_q8_0` (110314) and its `flock` (110321), which had not yet got the lock.
- After the stop: none of this agent's processes remain, `flock -n ~/gpu.lock` = LOCK_FREE, and `lslocks` shows no holder. GTT is 27 MB.
- Kernel log: `journalctl -k -b` has only the 3 boot-time lines that match the error filter: 2 × amdxdna "Running without IOMMU", plus the amdgpu "TMZ feature disabled". No amdgpu error appeared during the runs, and the watchdog logged no TRIGGER.
- No source change, commit, push or GitHub post was made. No systemd unit was touched, and strix-001 was not used.

## Scorecard (what ran)

| # | item | head 4d40d7f6a | 1dbee4d9a (PROOF-initial) | PR219 | verdict | log |
|---|---|---|---|---|---|---|
| 1 | builds `build`, `build-vk`, `build-hip-nofa`, `build-rpc-hip`, `build-cpu`, `build-rpc` (incremental, under flock) | all rc 0, "up to date", 0 warnings, 0 errors. The full rebuilds at this HEAD are in PROOF-review-r4 §2.1. | rc 0 / 0 warn | — | PASS | `PFN/build.out`, `PFN/build_<tree>.log` |
| 2 | ROCm0 SPARSE_ATTN_PLAN `--fail-unsupported` | 185/185, rc 0, notsup 0; census `sa_plan/` 2/2 OK (8 s) | 185/185, 2/2 | — | PASS | `PFN/ops/rocm_SPARSE_ATTN_PLAN.log` |
| 2 | ROCm0 SPARSE_ATTN `--fail-unsupported` (default, graphs on) | **1439/1439**, rc 0, notsup 0, fail 0. Census: 492/492 compiled keys launched (2 excluded: `sa_portable/*` runs on RDNA3.5 only when forced): OK (157 s) | 1390/1390, 492/492 | — | PASS | `PFN/ops/rocm_SPARSE_ATTN.log` |
| 2 | ROCm0 SPARSE_ATTN, `GGML_CUDA_DISABLE_GRAPHS=1` | **1439/1439**, rc 0, 492/492 OK (327 s) | 1390/1390, 492/492 | — | PASS | `PFN/ops/rocm_nographs_SPARSE_ATTN.log` |
| 2 | ROCm0 SPARSE_ATTN_PLAN, graphs off | 185/185, 2/2 OK | — | — | PASS | `PFN/ops/rocm_nographs_SPARSE_ATTN_PLAN.log` |
| 3 | **strict reserve, HIP `-np 2` split ub8192 with images, f16** (open failure 1, fix a16582997 / 195dadc10) | **violations 0** in main and MTP (`ctx=main violations=0 reallocs=26`, `ctx=mtp violations=0 reallocs=29`). Answers A `2749`, B `12749`. decode_err 0, masked 0, CPU 0, 301 trace lines all `backend=ROCm0`, guard cpu_launch 0. Reserve ROCm0 3170.22 MiB. The new one-stream passes are present: `QSA reserve 1s-pp: n_tokens = 8192, n_seqs = 1`, plus `1s-0out`, `1s-fast`, `1s-trim`. Peak GTT 96.3 GB (HIP server; below 100 GB). | **10 violations, +22,594,688 B** (`buffer=3144.2990 MiB > reserve=3122.7510 MiB`, ubatch n_tokens 8184, n_seqs 1, producer=cb) | — | **PASS (f16)**. q8_0 and q8_0K/f16V NOT RUN (cut by the pause). | `PFN/img/srv_np2split_f16.log`, `PFN/img/res_np2split_f16.txt`; old: `~/dev/wt_lab/prove/img/srv_np2split_f16.log` |

Side observation (report only): the concurrent prompt rates in this run were A 501.7 t/s and B 828.4 t/s, and B's `predicted_per_second` was 0.33. The two requests share the server, so these are not comparable with the one-request rates of P9.

## Not run (cut by the pause): every remaining item
All of these are prepared as scripts under `PFN`, and none has run.

- **Order on resume:** chain A from `rocm_vdirect`, chain B from `np2split_q8_0`, then `queue.sh`.
- **queue.sh order:** B2 → C → D → E → F → G → A(vkx) → Z.
  - On resume, `queue.sh` has to be edited: it waits on the old chainA PID 8167, which is gone. The order itself stays.
  - Each chain skips nothing automatically except finished perf jsons. Phases must be re-listed so that the jobs that already passed do not run again.

**Priority 2 (chain A, `chainA.sh hip vk ut ctest`):**
- ROCm0 SPARSE_ATTN with `GGML_CUDA_SA_FORCE` = vdirect (cut 12 s in), vpacked, generic, portable, fixup.
- ROCm0 per op: FLASH_ATTN_EXT, QSA_EXPAND, QSA_SCORE, QSA_STRIP, QSA_TE_STRIP, TOPK_QSA, TOP_K, GET_ROWS, SET_ROWS, CPY.
- `test-sparse-attn-rot`: default, generic, portable, fixup.
- Vulkan0 (RADV) `--fail-unsupported`:
  - SPARSE_ATTN_PLAN and SPARSE_ATTN;
  - the same 10 ops as on ROCm0;
  - scalar and portable tiers;
  - the TOPK_QSA census, one process per case (6).
- F5 checks:
  - prefer-dense b6, nofa before / after, HIP, RADV;
  - -inf/NaN-V b5 on HIP default / portable / fixup / generic and RADV default / scalar / portable;
  - te-rows on HIP, and on RADV 15 runs plus 5 Vulkan-only runs. Whether te-rows is clean on real hardware is the open question.
- F4 checks:
  - lead cases after on HIP (4 tiers) and RADV (cm, scalar);
  - lead / NaN cases before (F4_before builds);
  - the ≥ 2 GiB cases.
- F3 checks:
  - nanscore before;
  - shape guard before / after on HIP and RADV;
  - HIP-server RPC before / after.
- Final-fix checks:
  - `test-sparse-attn-vk dense reason vpack race` and the full test;
  - rpc-view, gate-scratch, reserve-view and allvis on HIP and RADV.
- `test-qsa-guard pass` on the real HIP device with the default `--cache-type-k-idx auto`, for q8_0, f16, q4_0 and bf16.
- ctest QSA subsets: HIP, HIP + RPC, RADV.
- Vulkan V-loader and sg32 forced tiers (phase `vkx`).

**Priority 3:**
- chain B:
  - `-np 2` split q8_0 and q8_0K/f16V;
  - `-np 2 --kv-unified`;
  - W9 concurrent list (mid-session reserve coverage);
  - Vulkan ub16 strict for f16/f16, f16/q8_0, q8_0, q4_0, q8_0/f16 (bf16 is in chain Z, last);
  - MTP census gated share;
  - AT-C18 at `-np 1` and `-np auto`;
  - Q12 coarse f16 / q8_0;
  - P5 MTP f16 default-draft cell, 3 + 3 processes.
- chain B2:
  - Q10 control, cells and image p3, with PR219 repeats reduced to 3 per cell and the analysis in `PFN/q10red.py`;
  - AT-A rerun on the head kernels (w4_dec rebuilt from the head sources).

**Priority 4 (chain C):**
- HIP route census: 6 pairs × ub {4096, 512, 16} + bench, strict, expected rows, gated share.
- Vulkan placement census at ub 4096 and 512.
- Determinism: HIP 6 pairs × 2 ub × 2 runs; Vulkan 5 pairs (bf16 in Z).
- Guard harness.
- Smoke: head vs 1dbee4d9a (the w13 hooks tree), f16 / q8_0 / q4_0 / bf16.
- AT-D3.

**Priority 5 (chain D):**
- QPIN lab tree at HEAD. Patches 01-03 apply; patch 06 needs `git apply -3`, which applies cleanly, and `lab_tree.sh` in `PFN/h` is adjusted for it.
- Pinned KLD:
  - Q1-A on 6 pairs;
  - Q1-B on f16 / q8_0 / q4_0;
  - direct form for q4_0 / bf16 (TODO-LATER #1).
- The PR219 reference `ref8t0` must be re-recorded: `~/dev/wt_lab/prove/q/refs` no longer exists. PROOF-initial §14 deleted it.

**Priority 6-9:**
- chain E: images A/B/C × 6 pairs; image understanding, short and at depth.
- chain F:
  - text perf: d0 and d131k, f16 / q8_0, ub 512 / 4096, 3 + 3 processes;
  - tg at depth;
  - MTP cells;
  - AT-O always-pack tree;
  - ~300k f16 / q8_0;
  - AT-O reserve.
- chain G: Vulkan AT-B1 (reduced to d4096 / d32768), AT-B4 e2e, AT-B5, the q8_0 depth run, Q1-vk.
- chain Z, last: Vulkan bf16 strict census, bf16 determinism, bf16 Q1-vk. Every job goes through the GTT / amdgpu-error watchdog, and a trigger writes `VK_STOP`.

**Deferred items (TODO-LATER #1-#9):** none re-measured yet.

## Notes for the resume
- HIP servers on this APU use GTT: the `-np 2` image server peaked at 96.3 GB GTT. `watchdog.sh` enforces the 100 GB stop only for Vulkan processes. For HIP it only logs.
- The bench binary `PFN/bench/bin_w7/llama-bench` was rebuilt against the HEAD libraries. `bin_w0` and `bin_base_ea5` are symlinks to the earlier builds.
- `PFN/labnew` (`lab-r1` / `lab-c4` at HEAD) is built. The worktrees `PFN/lab` and `PFN/labnew` are detached, with no commits. `PFN/lab` has the QPIN patches applied in its working tree only.
