# PROOF-final3: coverage gates on the rebased PR #276 head (strix-002, gfx1151)

**Status: complete.** All GPU work ended at 2026-11-13 01:47 (chain `F/chain4.out` ends in `CHAIN_DONE`); the 8080 server is back (`systemctl --user is-active llama-master.service` = active, `/health` = ok). Every job rc 0 except the 4 watchdog stops listed in §6 (archived, never used as results); `journalctl -k -b` holds only the 3 boot-time lines. Coverage GPU time: 20:09-01:47 (5 h 38 min), plus the M5 bisect interleaved with it.

**Overall verdict under the plan's rules: pass = false, because of one memory gate.**
- **M5 FAIL (32/40 rows): a multi-slot server reserves far more device memory than PROOF-initial's head** (default `llama-server`, 4 slots, c262144: main 4720 MiB vs 1648, MTP 1889 vs 1633). Bisected to the PR's own commit 3eb54536f (rebased `4f9296cca`, "TE rows read the mask for the visibility, so the Vulkan TOPK_QSA fusion fires again"). It is already in the current PR head 49a6d75ab; it is not a rebase effect.
- Vulkan memory (optional arm): M1-Vulkan 10/12 and AT-B5 1/4.
- Everything else that ran passes: M1 HIP 36/36, M2 4/4, M6 c524288, AT-O equal reserve at ub512 / 1024, Vulkan decode 6/6, Q1-C 6/6, Q3 6/6, Q4 7/7, Q8 4/4, Q11 26/26, P1, P4 (d0) 32/32. All 47 quality arms are bit-identical to PROOF-initial's.
- Cut for time: Q12, P4 at d32768 and the generic types. Not run under the memory rules: M6 c1048576.

These are the gates PROOF-final2 §8 listed as "not run": memory (M1, M1-Vulkan / AT-B5, M2, M5, M6), AT-O memory at small ub, Vulkan decode, quality (Q1-C, Q3, Q4, Q8, Q11, Q12) and perf (P1, P4). They ran on the rebased branch `qsa-rebased3` (REBASE3.md); verification of the head itself is in REBASE3 §4.

Conventions:
- `F` = `strix-002:~/dev/wt_lab/final4`.
- **new** = the plain PR build, `~/dev/wt_rebase3/build` (HIP) / `build-vk`, built at 9a2ee52bd; the later commits c456e6e5e (tests only) and 7e7c75b08 (VGPR JSON) change no src/ or ggml/ file, so these binaries are the final head's code.
- **lab** = 9a2ee52bd + lab-instrumentation.patch + QPIN 01/02/03 (`F/lab`), used for the pinned KLD arms.
- **master / mb** = b620b0d9a (`~/dev/wt_master_b62`). **PR219** = W0-lab12 (`~/dev/wt_lab/w0lab12`); the PR219 partner logs of the quality gates are PROOF-initial's (`~/dev/wt_lab/prove/q/logs`, same binary), each reference checked first by a PR219 replay that must reproduce the reused log's per-chunk KLD and QKL_HASH lines. **base** = ea5285ddf (`prove/bench/bin_base_ea5`, P1 as in PROOF-initial).
- Every GPU job ran under `flock ~/gpu.lock`, one at a time, under the `lib4.sh` watchdog (stop on GTT > 100e9, MemAvailable < 4 GiB, or a new amdgpu / OOM kernel line).
- Perf statistic (PLAN-v3 §16, `~/dev/wt_lab/prove/h/w7/perf_an.py`): per process the mean of reps 1-4; ratio CI from the hierarchical bootstrap (4000 draws, seed 1); PASS if CI LB ≥ θ, FAIL if CI UB < θ, else RERUN (2× processes, then 4×; still unresolved = FAIL).

## 0. Scorecard

| # | gate | criterion | measured | verdict | log (`F/…`) |
|---|---|---|---|---|---|
| 1 | M1 HIP, 9 types × c{32k,131k} × ub{4096,512}, main + MTP | T ≤ f16 + 1 MiB | every type f16 + 0.00 or + 0.27 MiB | **PASS 36/36** | `mem/m1/m1.md` |
| 2 | M1 Vulkan (f16, q8_0, q4_0) | T ≤ f16 + 1 MiB | q8_0 / q4_0 c131072 ub4096 main **+183.68 MiB**; other cells ≤ +0.27 | **10/12 PASS, 2 FAIL** | `mem/m1vk/m1vk.md` |
| 3 | AT-B5 Vulkan new f16 vs master f16 | ≤ + 1 MiB | c32768 ub4096 main **+334.87**; c131072 ub4096 MTP **+104.00**; c131072 ub512 MTP **+12.00** (TODO-LATER #4); c32768 ub512 −14.97 / −37.00 | **1/4 PASS, 3 FAIL** | same |
| 4 | M2 total device high-water, new vs PR219 (shim, server fill) | ≤ + 1 MiB | diff 0.00 in all 4 cells (f16 / q8_0 × c65536 / c131072) | **PASS 4/4** | `mem/m2/m2.md` |
| 5 | M5 default / multi-slot server reserve | new ≤ max(PR219, V) + 1 MiB | non-compact reserve +12 × n_ctx × 1 KiB main, +1 × n_ctx × 1 KiB MTP vs PROOF-initial; default c262144: 4720.02 / 1889.02 MiB vs bounds 1745.36 / 1674.03 | **8/40 PASS, 32 FAIL**; bisected to 3eb54536f | `mem/c12/m5.md`, `mem/bis/` |
| 6 | M6 f16 ub4096, new vs PR219 reserve (+ pool) | ≤ | c524288 1660.42 ≤ 1932.42 MiB; c1048576 not run (GTT > 100 GB) | **PASS 1/1**, 1 NOT RUN | `mem/c17/` |
| 7 | AT-O memory at ub512 / ub1024, capped vs always-pack, c131072 / 270336 / 393216 | report | identical reserve in all 6 cells | report: **always-pack costs no reserve; keep CAPPED** (no speed gain, PROOF-final2 §5) | `ato/ato_small_ub.md` |
| 8 | Vulkan decode tg128 vs master, f16 / q8_0, d0 / 4096 / 32768 | ≥ 0.98 (d0), ≥ 0.99 | 0.9972 … 1.0679 | **PASS 6/6**; new q8_0 / f16 ≥ 0.94: **PASS 3/3** (0.9893 … 0.9947) | `vkdec/vkdec.md` |
| 9 | Q1-C ub64, policy and forced G1, f16 / q8_0 / q4_0 | ≤ 2.75e-4 | −1.075e-4 … +1.778e-5 | **PASS 6/6** | `q/quality.md` |
| 10 | Q3 c32768 × 8, ub4096 / ub256 | ≤ 1.801e-4 / 3.34e-4 | −1.954e-4 … +1.442e-4 | **PASS 6/6** | same |
| 11 | Q4 selection agreement, 7 types | thresholds | all above; f16 margins +6.3e-6 / +2.0e-6 (TODO-LATER #8) | **PASS 7/7** | same |
| 12 | Q8 planted defects D1 / D2 (f16, q8_0) | LCB95 > 7.45e-5; sign 8/8 | LCB +8.158e-4 / +8.044e-4; 8/8 both | **PASS 4/4** (+ D3 report) | same |
| 13 | Q11a / Q11b / Q11c | bands; np1 = np2 | 18/18; 6/6 (+ ub16 report); np1 ≡ np2 line for line | **PASS 26/26** | same |
| 14 | Q12 `-nkvo` / `-ngl 0` | bands | — | **NOT RUN (time cut)** | — |
| 15 | P1 q8_0 ub512 pp4096 d0, new vs base ea5285ddf | CI LB ≥ 0.995 | 1.0413 [1.0372, 1.0458] | **PASS** | `perf_P1.out` |
| 16 | P4 d0, q4_0 / bf16 / q8_0K-f16V / f16K-q8_0V vs new f16 | ≥ 0.90 | 0.9875 … 1.0121 | **PASS 32/32**; d32768 and generic types NOT RUN | `perf_P4.out` |

## 1. Setup
- Branch `qsa-rebased3`, final head 7e7c75b08 (REBASE3.md). Plain arm built at 9a2ee52bd; c456e6e5e (tests only) and 7e7c75b08 (VGPR JSON) change no src/ or ggml/ file.
- Scripts: `F/h/chain4.sh` ran `vgpr4.sh`, then `mem4.sh` (m1 m1vk m2 m5 m6 ato vkdec), `q4.sh` (A N C P B S E Z, with the cuts of §6), `perf4.sh` (P1 P4a P4b an); `F/h/an4.py` scores M1-Vulkan, M5, M6, AT-O and the Vulkan decode; the plan's own scorers do M1 (`mem_an.py`), M2 (`m2_an.py`), quality (`q_an.py`) and perf (`perf_an.py`).

## 2. Memory gates (init-only servers, `sched_reserve -lv 4`; PLAN-v3 §16 memory gates)

### 2.1 M1, HIP (`F/mem/m1/m1.md`, scorer `prove/h/w7/mem_an.py`)
- Rule: every T ≤ new f16 + 1 MiB on ROCm0, main and MTP, at -c {32768, 131072} × b/ub {4096, 512}, `--parallel 1`, Signal-3.8 + MTP sidecar.
- Types: f16, q8_0, q4_0, bf16, iq4_nl (PROOF-initial's matrix) **plus f32, q4_1, q5_0, q5_1** (the rest of S; PROOF-initial left them MISSING).
- **36/36 PASS.** Every type is f16 + 0.00 (bf16, f32) or f16 + 0.26 / 0.27 MiB (all quantized types), main and MTP, host the same. Values: c32768 ub4096 main 1563.67 / MTP 1400.72; c131072 ub4096 1573.42 / 1402.22; c32768 ub512 197.02 / 184.52; c131072 ub512 655.55 / 639.15 MiB (f16).
- Reported rows: `--parallel 2` c131072 ub1024 (split): main 1230.52, MTP 669.18 for f16 / bf16, +0.27 for q8_0 / q4_0. Default server (4 slots, c262144, ub512): main **4720.02**, MTP 1889.02 for f16 / bf16, +0.27 for q8_0 / q4_0. Both carry the non-compact reserve growth of §2.4.

### 2.2 M1 on Vulkan / AT-B5 (`F/mem/m1vk/m1vk.md`)
- Rule: every T ≤ new f16 + 1 MiB (main and MTP); new f16 ≤ the reference f16 (master b620b0d9a stands in for W0). bf16 never run on Vulkan.

| cell | new f16 main / MTP | new q8_0 (= q4_0) main / MTP | T vs f16 | master f16 main / MTP | new f16 − master f16 |
|---|---|---|---|---|---|
| c32768 ub4096 | 2373.33 / 1321.72 | 2133.60 / 1321.98 | −239.73 / +0.26 PASS | 2038.46 / 1321.72 | **+334.87** / 0.00 FAIL |
| c32768 ub512 | 286.24 / 184.43 | 286.50 / 184.70 | +0.26 / +0.27 PASS | 301.21 / 221.43 | −14.97 / −37.00 PASS |
| c131072 ub4096 | 2144.90 / 1347.22 | 2328.58 / 1347.48 | **+183.68** / +0.26 FAIL | 2181.05 / 1243.22 | −36.15 / **+104.00** FAIL |
| c131072 ub512 | 647.09 / 629.28 | 647.35 / 629.54 | +0.26 / +0.26 PASS | 688.09 / 617.28 | −41.00 / **+12.00** FAIL |

- **M1-Vulkan: 10/12 PASS, 2 FAIL** (q8_0 and q4_0 at c131072 ub4096, main, +183.68 MiB). **AT-B5 vs master: 1/4 PASS, 3 FAIL.** The c131072 ub512 MTP +12.00 MiB is TODO-LATER #4 (same value as W7); the other two misses are new information (AT-B5 was never run in Prove).
- Where it comes from (partial): the differences are already in the plain pp / tg reserve ("reserve after the pp and tg graphs": f16 2373.33 vs q8_0 2133.60 at c32768, 2144.90 vs 2328.58 at c131072), not in the QSA reserve passes; new f16 is larger at c32768 than at c131072. The SPARSE_ATTN workspace (`ggml_vk_sa_select`) does not grow with n_kv for the shipped direct-V q8_0 path. Cause **UNPROVEN** (allocator layout of the pp graph is the likely suspect, not shown).
- Master q8_0 c131072 ub4096 was not run: its masked route reserved 22,933.80 MiB on Vulkan0 + 1,737.99 MiB host and took GTT to 103.6 GB; the watchdog stopped it (`F/stopped3/`). Master q8_0 for context: c32768 ub4096 7204.07 MiB, c131072 ub512 1244.40 MiB (new q8_0: 2133.60 / 647.35).

### 2.3 M2 (`F/mem/m2/m2.md`, C7 hipMalloc shim, server filling the context, MTP on, ub4096)
- Rule: new total device high-water ≤ PR219 (W0-lab12) + 1 MiB, same campaign.
- **4/4 PASS**, diff 0.00 MiB in every cell: f16 c65536 79721.15, c131072 81704.65; q8_0 c65536 78911.43, c131072 80084.93 MiB. Pool high-water equal on both arms (8104.36 / 8263.86 / 8104.63 / 8264.13 MiB). The same values as PROOF-initial.

### 2.4 M5 / AT-C12 default server (`F/mem/c12/m5.md`; PR219 logs reused from `prove/c12`, same W0-lab12 binary)
- Rule: new reserve ≤ max(PR219 reserve, V) + moved pool items + 1 MiB per backend, main and MTP. V is the §7 non-compact variant formula (K = n_ctx / s, T = min(ub, 512)); "moved pool items" taken as 0. The plan's instantiated bounds are checked too.
- **M5: 8/40 rows PASS, 32 FAIL.** Every FAIL is a multi-slot cell below ub8192 (and the default server):

| cell (f16; q8_0 is +0.27) | new main / MTP ROCm0 | PROOF-initial new | PR219 | plan bound |
|---|---|---|---|---|
| default (4 slots, c262144, ub512) | **4720.02 / 1889.02** | 1648.02 / 1633.02 | 1311.08 / 1285.03 | 1745.36 / 1674.03 |
| c131072 ub512 or ub1024 | **2382.52 / 959.52** | 846.52 / 831.52 | 696.08–759.15 / 675.53–742.06 | 943.86 / 872.53 (ub1024) |
| c131072 ub4096 | **2382.52** / 1465.23 | 1609.61 / 1465.23 | 1609.61 / 1465.23 | 1610.88 / 1466.23 |
| c261888 ub512 / 1024 / 4096 | **4715.46 / 1887.20** | 1646.46–1658.52 / 1631.45 | 1309.88–1897.39 / 1283.84–1705.12 | — |
| c261888 ub8192 | **4715.46** / 2938.45 | 3315.03 / 2938.45 | 3315.03 / 2938.45 | 3316.30 / 2939.45 |
| `-np 2` split c131072 ub1024 | **1230.52** / 669.18 | 667.19 / 669.18 | 747.19 / 741.18 | — |
| c131072 ub8192 | 3218.22 / 2912.12 | 3218.22 / 2912.12 | 3218.22 / 2809.47 | — |

- The growth is in the non-compact reserve pass alone (the `QSA reserve noncompact` line: main 1275.08 -> 4720.02 MiB here, -> 1648.02 at PROOF-initial's head), main = 12 QSA layers × n_ctx × 1 KiB (+3072 MiB at c262144, +1536 at c131072), MTP = 1 layer × n_ctx × 1 KiB (+256 / +128). Host is unchanged. ub8192 cells are unaffected (their pp graph is larger). The MTP c131072 ub8192 excess over PR219 (+102.65 MiB) is unchanged since PROOF-initial (cell without an instantiated bound there).
- **Bisected** (`F/mem/bis/`, `F/bisb.out`; init-only default server, f16, mmproj + MTP, each build under the lock): 49a6d75ab (the pre-rebase PR head) 4720.02 / 1889.02; the pre-trim tree 4d40d7f6a and the trim tree 27cbb3bc8 the same; **e31548e7d 1648.02 / 1633.02; its child 3eb54536f 4720.02 / 1889.02.** 3eb54536f is "qwen4exp : TE rows read the mask for the visibility, so the Vulkan TOPK_QSA fusion fires again" (rebased: `4f9296cca`): `qwen4exp_te_select` gathers the visibility of the picked cells from a view of the KQ mask instead of from `expanded`. The mechanism inside ggml-alloc / the scheduler is **not identified** (a `GGML_SCHED_DEBUG` run printed nothing at the server's log level).
- It is not a rebase effect (49a6d75ab has it) and it predates PROOF-final2, which did not measure M5. Single-sequence contexts (M1, `--parallel 1`) do not build the non-compact pass and are unaffected. It costs device memory on every multi-slot server below ub8192, including the default `llama-server` (4 slots): +3 GiB main + 0.25 GiB MTP at c262144.

### 2.5 M6 / AT-C17 (`F/mem/c17/`)
- Rule (CAPPED default): new f16 reserve ≤ PR219 reserve + PR219 pool high-water, ub4096.
- c524288: new 1660.42 / 204.80 MiB ≤ PR219 1932.42 / 204.80 (reserve alone): **PASS**; equal to PROOF-initial's new value.
- c1048576: **not run.** The init-only server took GTT to 104.2 GB (f16 KV of 1M cells next to the weights) and the watchdog stopped it (`F/stopped4/`), the brief's GTT < 100 GB rule. `-np 4` c524288 not run either: it adds the §2.4 non-compact reserve (12 × 512 MiB) to the 97.1 GB GTT of `-np 1`.

### 2.6 AT-O memory at small ub (`F/ato/ato_small_ub.md`; f16, `--parallel 1`, init-only, capped vs `GGML_CUDA_SA_FORCE=vpacked`)

| ub | c131072 | c270336 | c393216 |
|---|---|---|---|
| 512 | 643.57 = 643.57 | 733.32 = 733.32 | 1037.07 = 1037.07 |
| 1024 | 710.14 = 710.14 | 738.57 = 738.57 | 1111.14 = 1111.14 |

(main ROCm0 MiB, capped = always-pack; host equal too; the force line is present in every always-pack log.)
- The always-pack V bound would be n_kv × 1 KiB (128 / 264 / 384 MiB) against 32 MiB (ub512) / 64 MiB (ub1024) capped, but the reserve does not move: the attend workspace sits under a larger peak at small ub as well. This closes PROOF-final2's open AT-O question: **no reserve cost of always-pack at ub512 / ub1024 up to c393216**; with no e2e speed gain (PROOF-final2 §5) the recommendation stays CAPPED.

## 3. Vulkan decode vs master (`F/vkdec/vkdec.md`; C10 text llama-bench, `-ngl 99 -fa 1 -b 4096 -ub 512 -p 0 -n 128 -d 0,4096,32768 -r 5`, f16 and q8_0, 3 master + 3 new processes interleaved after a warm-up, §16 statistic; never bf16)

| type | depth | master t/s | new t/s | new / master [95 % CI] | θ | verdict |
|---|---|---|---|---|---|---|
| f16 | 0 | 28.67 | 28.72 | 1.0020 [0.9969, 1.0069] | 0.98 | PASS |
| f16 | 4096 | 27.08 | 27.01 | 0.9972 [0.9950, 0.9993] | 0.99 | PASS |
| f16 | 32768 | 26.77 | 26.72 | 0.9980 [0.9957, 1.0005] | 0.99 | PASS |
| q8_0 | 0 | 28.55 | 28.57 | 1.0007 [0.9954, 1.0056] | 0.98 | PASS |
| q8_0 | 4096 | 26.76 | 26.86 | 1.0039 [1.0016, 1.0060] | 0.99 | PASS |
| q8_0 | 32768 | 24.75 | 26.43 | 1.0679 [1.0639, 1.0719] | 0.99 | PASS |

- **6/6 PASS.** new q8_0 / new f16 (P5 floor 0.94): 0.9947, 0.9946, 0.9893, **3/3 PASS** (master: 0.9960, 0.9880, 0.9244 at d32768). No P6(a) sample. θ: 0.98 at d0 (the P5 shortcut threshold) and 0.99 at depth; d4096 is not a P5 cell and takes the depth threshold.

## 4. Quality (PLAN-v3 §15; `F/q/quality.md`, scorer `~/dev/wt_lab/prove_final/h/w7/q_an.py`; runs `F/q/runs.txt`, logs `F/q/logs/`)

### 4.1 References and reuse
- References recorded fresh on the PR219 arm with `LLAMA_FN_LAST_ATTN_TRIM=0`: c8192 × 8 (`ref8t0`), c8192 × 16 (`ref16t0`), c32768 × 8 with `QKL_TAIL=4096` (`ref32x8`), c8192 × 8 with `QKL_TAIL=8191` (`ref8L`), c2048 × 16 dense f16 b/ub 2048 (`ref2k`) and Q11c's unpinned per-type bases; each was deleted after its phase.
- Reuse checks (QKLD_CHUNK and QKL_HASH lines, md5 of the sorted set, `F/q/ident.txt`): the new f16 ub4096 replay on the fresh `ref8t0` equals final2's (`ea71876db0e4`); the PR219 replays on `ref16t0` (f16), `ref32x8` (f16 ub4096), `ref8L` (f16 ub512, split 2048) and `ref2k` (f16 ub512) equal PROOF-initial's logs. So the reused PR219 partner logs are valid partners, and every QPIN arm had rows_absent = moe_absent = 0.
- **Bit identity with PROOF-initial.** Every new arm run here was compared with PROOF-initial's log of the same tag (`prove/q/logs`): **47 of 47 new arms identical** (Q1-C 3, Q1-C forced G1 3, Q4 7, Q3 6, Q8 8, Q11a 6, Q11b 8, Q11c 6; plus the f16 ub4096 control). The rewrite's pinned outputs have not moved since PROOF-initial's head on any of these arms, so the verdicts equal PROOF-initial's to the last digit.

### 4.2 Results
| gate | arms | rule | values | verdict |
|---|---|---|---|---|
| Q1-C ub64 (policy) | f16, q8_0, q4_0 | mean(new − PR219) ≤ 2.75e-4 | +1.778e-5 ± 1.67e-5, +2.621e-6 ± 9.44e-6, −1.075e-4 ± 8.90e-5 | **PASS 3/3** |
| Q4 selection agreement, c8192 × 16 | f16, f16K + q8_0 idx, q8_0, q4_0 (q8_0 idx), bf16, q8_0K/f16V, f16K/q8_0V | mean_c(new − PR219) ≥ −B_T and the absolute thresholds | f16 0.997356 (≥ 0.99735, margin +6.3e-6), f16K q8 idx 0.996932 (≥ 0.99693, +2.0e-6), q8_0 0.996737 (+1.1e-4), q4_0 0.985712 (+1.8e-4), bf16 0.997256, q8_0K/f16V 0.996857, f16K/q8_0V 0.997237 (+1.2e-4 each) | **PASS 7/7** (thin f16 margins = TODO-LATER #8, unchanged) |
| Q3 depth, c32768 × 8, `QKL_TAIL=4096` | Q3-A ub4096 and Q3-B ub256 × f16, q8_0, q4_0 | ≤ 1.801e-4 (A), ≤ 3.34e-4 (B) | A: f16 +1.442e-4 ± 3.52e-5, q8_0 +4.456e-6, q4_0 −1.073e-4; B: f16 +2.689e-6, q8_0 +4.376e-6, q4_0 −1.954e-4 | **PASS 6/6** |
| Q11a shortcut route, c8192 × 8, b/ub 512, `QKL_TAIL=8191`, split 2048 | 6 pairs × lo / hi / all | ≤ 2.073e-4 (lo), 2.03e-4 | worst q4_0 lo +1.561e-4 ± 1.51e-4; bf16 all −5.054e-5 ± 4.6e-6 | **PASS 18/18** |
| Q11b c2048 × 16, dense f16 b/ub 2048 reference | 6 pairs at ub512; f16 and q8_0 at ub16 | ≤ 2.03e-4 | ub512: f16 0, q8_0 −2.157e-5, q4_0 −2.656e-4, bf16 −3.900e-5, q8_0K/f16V −3.403e-5, f16K/q8_0V −3.569e-5; ub16: 0 and 0 (report only, band UNPROVEN) | **PASS 6/6** (+ 2 report) |
| Q11c `-c 1024 -b 2048` (np2 split) vs `-b 1024` (np1), c1024 × 32, unpinned | q8_0, q4_0 | np1 = np2 to 0 per sequence | the np1 run's QKLD_CHUNK lines equal the np2 rerun's line for line for both types; the scorer's "≤ 1e-6" check prints FAIL on all four arms (max 1.191e-6 / 1.273e-6), the uint16 resolution of the KLD base, the same in the np2-vs-np2 rerun | **PASS 2/2** (as PROOF-initial) |
| Q1-C ub64, forced G1 (`GGML_CUDA_SA_FORCE=g1`) | f16, q8_0, q4_0 | ≤ 2.75e-4 | −1.002e-5 ± 9.39e-6, −1.794e-5 ± 1.41e-5, −4.388e-5 ± 1.08e-4 | **PASS 3/3** |
| Q8 planted defects (on − off, same pins, c8192 × 8) | D1, D2, D3 × f16, q8_0 | D1 LCB95 > 7.45e-5; D2 sign ≥ 7/8 and \|new − PR219\| ≤ 1.86e-3; D3 report | D1 f16 +1.187e-3 (LCB +8.158e-4), q8_0 +1.160e-3 (LCB +8.044e-4); D2 f16 +5.706e-4 8/8 (PR219 +7.037e-4), q8_0 +5.744e-4 8/8 (PR219 +6.080e-4); D3 −2.9e-8 / −1.4e-6 | **PASS 4/4** (+ 2 report) |

- Q8 q8_0 D1 / D2 score against the clean q8_0 ub4096 arm (Q1-A), which was not re-run here: its log is final2's (49a6d75ab lab tree; md5 of its chunk lines equals PROOF-initial's, `7f81fd1448b3`), `F/q/logs/REUSED_Q8OFF.txt`. f16 uses this run's control arm.
- **Not run (cut for the time budget):** Q12 (`-nkvo` pinned, 8 arms, and `-ngl 0`, 2 arms; PROOF-initial: PASS, with the coarse unpinned q8_0 screen FAIL), the Q1 direct form and its stage 2 (TODO-LATER #1, final2 re-measured it), Q1-A / Q1-B (final2), Q5, Q6 / Q10, Q7 (deferred, final2), Q1-vk. Given that all 47 arms run here are bit-identical to PROOF-initial's, the PROOF-initial Q12 values are expected to hold, but they were not re-measured on this head (UNPROVEN).

## 5. Performance (C10 text llama-bench, HIP, `-ngl 99 -fa 1 -lzm on-direct`, 5 reps, rep 0 dropped, C7 KFD sampler; `F/perf/`, `F/perf_P1.out`, `F/perf_P4.out`)

### 5.1 P1: new vs base, text q8_0 ub512 pp4096 d0 (θ 0.995)
- R N R N R N R in one campaign after a base warm-up (4 base + 3 new processes).
- base 1221.82 t/s, new 1272.28 t/s: **1.0413 [1.0372, 1.0458], PASS** (PROOF-initial: 1.0310 [1.0255, 1.0371]). 0 eviction samples, 0 P6(a) samples.

### 5.2 P4: new_T / new_f16, d0, pp {512, 2048, 4096, 8192} × ub {512, 4096} (θ 0.90)
- 2 processes per type, types interleaved (f16, q4_0, bf16, q8_0K/f16V, f16K/q8_0V, then a second round), after an f16 warm-up.

| pair | worst cell (ratio [95 % CI]) | best cell | verdict |
|---|---|---|---|
| q4_0 | ub4096 pp2048 0.9875 [0.9828, 0.9919] | ub512 pp2048 1.0095 | PASS 8/8 |
| bf16 | ub4096 pp2048 0.9903 [0.9856, 0.9948] | ub512 pp2048 1.0121 | PASS 8/8 |
| q8_0K/f16V | ub4096 pp2048 0.9876 [0.9829, 0.9920] | ub512 pp2048 1.0117 | PASS 8/8 |
| f16K/q8_0V | ub4096 pp2048 0.9879 [0.9826, 0.9930] | ub512 pp2048 1.0095 | PASS 8/8 |

- **P4 at d0: 32/32 PASS**, every ratio 0.9875-1.0121 (floor 0.90). New f16 itself: 1340.64 / 1247.83 / 1253.26 / 1234.30 t/s at ub512 and 1355.36 / 2004.85 / 2090.82 / 2033.03 at ub4096 (pp 512 / 2048 / 4096 / 8192). 0 eviction samples dropped (2 of 10 processes saw KFD evictions, none in a timed sample), 0 P6(a) samples.
- **Not run:** P4 at d32768 (the PROOF-initial unresolved cell was bf16 d131k ub512 pp512 0.919, not re-measured) and the S_generic types (q4_1, q5_0, q5_1, iq4_nl, f32).

## 6. Cuts, deviations, incidents

**Cut for the time budget (priority order of the brief; nothing else was cut):**
- Q12: `-nkvo` pinned (8 arms) and `-ngl 0` (2 arms, up to 8 h of CPU each). Not re-measured on this head.
- P4 at d32768 (P4b) and the S_generic types of P4 (q4_1, q5_0, q5_1, iq4_nl, f32). The f16 reference and the 4 first-class pairs ran with 2 processes each (the plan asks 4 + 4 for a campaign; every cell resolved, so no rerun was needed).
- No §16 rerun was needed: no perf cell here is RERUN.

**Not run because of the memory rules (watchdog stops, archived):**
- M6 c1048576 (GTT 104.2 GB at init, `F/stopped4/`) and `-np 4` c524288 (would add 6 GiB of §2.4 reserve to 97.1 GB).
- Master Vulkan q8_0 c131072 ub4096 (masked route, GTT 103.6 GB, `F/stopped3/`).

**Deviations:**
- Reference for Vulkan memory and decode: master b620b0d9a, not W0 (as in PROOF-final2).
- P1 base is ea5285ddf (`prove/bench/bin_base_ea5`, as PROOF-initial); the scorer's title still names 13a279bfa.
- M1 adds f32, q4_1, q5_0, q5_1 to PROOF-initial's type matrix.
- Q8 q8_0 D1 / D2 reuse the clean q8_0 ub4096 arm from final2 (bit-identical to PROOF-initial's).
- M5's bound is instantiated with "moved pool items" = 0 (stricter than the plan).

**Incidents (no code effect):** the watchdog stopped jobs four times: the 8080 server restarted mid-run by the previous agent (`F/stopped1/`), PPL without the `-lm mmap -lzm on-direct` load flags (`F/stopped2/`), master Vulkan q8_0 (`F/stopped3/`), M6 c1048576 (`F/stopped4/`). Each was archived and the remaining jobs resumed; no job result was taken from a stopped run. `journalctl -k -b` shows only the 3 boot-time lines after all GPU work (no amdgpu error, no OOM).

## 7. Logs (`F` = `strix-002:~/dev/wt_lab/final4`)
- Chain: `chain4.out`, `h/chain4.sh`; scripts `h/{lib4,mem4,q4,qrun,perf4,an4,bis,bisb,bisdbg}.sh|.py` (`q4.sh.orig`, `perf4.sh.orig` are the scripts before the time-budget edits).
- Memory: `mem/m1/` (`m1.md`), `mem/m1vk/` (`m1vk.md`), `mem/m2/` (`m2.md`, `*.shim`), `mem/c12/` (`m5.md`, `parsed.json`), `mem/c17/`, `mem/bis/` + `bis.out`, `bisb.out` (the M5 bisect), `ato/` (`ato_small_ub.md`).
- Vulkan decode: `vkdec/` (`vkdec.md`, `*.json`).
- Quality: `q/quality.md`, `q/runs.txt`, `q/ident.txt`, `q/logs/` (incl. `REUSED_Q8OFF.txt`); the references were deleted.
- Perf: `perf/`, `perf_P1.out`, `perf_P4.out`.
- Stopped runs: `stopped1/` … `stopped4/`.

## 8. PR body summary

```markdown
### Coverage gates (strix-002, gfx1151, rebased head 7e7c75b08 on master b620b0d9a)

| check | result |
|---|---|
| Memory M1 (HIP, 9 KV types, c32k/c131k, ub4096/512, main + MTP) | 36/36: every type ≤ f16 + 0.27 MiB |
| Memory M2 (total device high-water vs PR219, server filling the context) | 4/4, identical (diff 0.00 MiB) |
| Memory M6 (f16 c524288) / AT-O (always-pack at ub512/1024 up to c393216) | 1660 ≤ 1932 MiB / no reserve difference; V pack stays CAPPED |
| Pinned KLD: Q1-C ub64 (incl. forced G1), Q3 c32k, Q4 agreement, Q8 planted defects, Q11a/b/c | 47/47 arms pass and are bit-identical to the pre-trim proof |
| HIP prefill: P1 (q8_0 vs base) / P4 (q4_0, bf16, mixed pairs vs f16, d0) | 1.041× / 32/32 cells ≥ 0.988× f16 |
| Vulkan decode vs master (f16, q8_0, d0-32k) | 0.997-1.068×; q8_0/f16 ≥ 0.989 |

**Known issues / follow-ups**
- Multi-slot servers (default `llama-server`, 4 slots) reserve 12 × n_ctx × 1 KiB more on the main context (+3 GiB at c262144, +1.5 GiB at c131072) and n_ctx × 1 KiB on MTP below ub8192, since the TE rows read their visibility from the KQ mask (Vulkan TOPK_QSA fusion); single-slot servers are unaffected.
- Vulkan compute buffer: q8_0/q4_0 +184 MiB over f16 at c131072 ub4096; new f16 +335 MiB over master at c32768 ub4096 and MTP +104 / +12 MiB at c131072 (cause not identified).
- Not re-measured on this head: Q12 (`-nkvo`, `-ngl 0`), P4 at depth and for the generic types, M6 at c1048576.
```

