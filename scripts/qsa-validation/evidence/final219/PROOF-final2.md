# PROOF-final2: PR #276 head 49a6d75ab on strix-002 (FINAL)

**Status: complete.** Both chains finished (`chain1.out` and `chain2.out` end in `CHAIN_DONE`). 118 jobs, all rc 0. No watchdog STOP file. Every job reports `kerr_new=0`, and `journalctl -k -b` still shows only the 3 boot-time lines. Peak GTT was 99.57 GB (the master f16 image server) and the lowest MemAvailable 8.98 GB (AT-O quality run). No new GPU job was started for this report. Every verdict below was recomputed from the raw logs (per-chunk KLD, bench JSON, MTP jsonl), not copied from the chain summary lines.

**Overall verdict: pass = false under the plan's rules, with no kernel-level regression found.**
- **Fixed since PROOF-initial:** the two blocking failures this head was meant to fix now PASS:
  - `-np 2` split-stream strict reserve: 0 violations (was +22.6 MiB);
  - MTP gated share: 96/96 (was 56/96).
- **Same as before:** every pinned Q1-A/B arm passes. Its per-chunk KLD is bit-identical to PROOF-initial's head 1dbee4d9a.
- **Prefill vs master:** f16 is 1.000–1.082× master, q8_0 1.017–2.768×.
- **Non-deferred FAILs (5 cells/samples)**, each with an identified cause that is not a kernel slowdown (part of that is UNPROVEN, see §4):
  - P3 ratio-of-ratios, d0 ub512 pp4096: 0.9856;
  - P5 MTP q8_0 d1024: 0.690;
  - P5 MTP new_q8_0/new_f16 at d16: 0.747 (master: 0.745);
  - P6(a): 1 slow sample.
- **Unresolved (RERUN) cells:** 8 in the HIP gates, plus 4 Vulkan f16 cells on the optional arm. No §16 reruns were done.
- **Deferred FAILs (TODO-LATER):** Q1 direct form for q4_0/bf16 (#1), Vulkan AT-B1 q8_0 (#3), Q7 q8_0/q4_0 (#9).

Conventions:
- `F` = `strix-002:~/dev/wt_lab/final2`.
- **new** = the plain PR build (`~/dev/wt_rebase2/build`, same tree as 49a6d75ab except the VGPR json line).
- **lab** = 49a6d75ab + lab patch + QPIN.
- **master / m50** = 50c2c8cb5 (`~/dev/wt_master50`).
- **PR219** = W0-lab12.
- **ap** = always-pack, via `GGML_CUDA_SA_FORCE=vpacked`.
- Perf statistic (PLAN-v3 §16, `~/dev/wt_lab/prove/h/w7/perf_an.py` via a cmp.py-style driver):
  - per process, the mean of reps 1-4;
  - hierarchical bootstrap, 4000 draws, seed 1;
  - PASS if CI LB ≥ θ, FAIL if CI UB < θ, otherwise RERUN.
- The plan's perf gates use PR219 (W0) as reference. This run uses master 50c2c8cb5 (which contains the #219 QSA path and #262-#277) as the stand-in, as briefed.


## 0. Scorecard

| # | gate (plan id) | criterion | measured | verdict | log (`F/…`) |
|---|---|---|---|---|---|
| 1 | lab = plain | identical output | PPL f16 c8192 × 2: plain and lab both `[1]2.1082 [2]1.9485`, 1.9485 ± 0.03445. Unpinned KLD per-chunk table lab = plain (md5 equal), 0.008892 both | PASS | `ident/ppl_{plain,lab}.log`, `q/logs/{lab,plain}_u_f16_f16_u4096.log` |
| 2 | M4 strict reserve, `-np 2` split with images, ub8192 (PROOF-initial failure #1) | 0 violations; answers 2749 / 12749 | f16, q8_0, q8_0K/f16V: main 0 / MTP 0, answers correct, decode_err 0, CPU 0, masked 0; `1s-pp/0out/fast` reserve passes present | **PASS 3/3** (was FAIL) | `img/srv_np2split_*.log`, `img/res_np2split_*.txt` |
| 3 | same, `-np 2 --kv-unified` q8_0 | same | 0 / 0, answers correct, 1913 TE + 131 CB ubatches | PASS | `img/*np2uni_q8_0*` |
| 4 | images A/B/C × 6 types (c261888, ub8192, MTP on, strict + trace) | 2749 / 12749 / 26999; strict 0; CPU 0; masked 0 | 18/18 answers. strict 0 (main and MTP), CPU 0, masked 0, 580 QSA trace lines all `backend=ROCm0` per type | **PASS 18/18** | `img/res_img_*.txt`, `img/srv_img_*.log` |
| 5 | image understanding, short (f16 / q8_0 / q4_0) and at depth (42571 tokens: f16 / q8_0 / master f16) | headline, newspaper, date | all name The New York Times, 21 July 1969, "MEN WALK ON MOON", "ASTRONAUTS LAND ON PLAIN…"; deep answers give the same 4 facts and 3 headlines as master | PASS | `img/res_und_new_*.txt`, `img/res_deep_*.txt` |
| 6 | MTP census rows + gated share (PROOF-initial failure #3) | rows = expected; gated share ≥ 0.9 | `EXPECT PASS`, 0 diffs, masked 0, CPU 0 for draft default / q8_0 / q4_0; gated share **96/96 = 1.000** in all 3 | **PASS 3/3** (was 56/96 FAIL) | `mtp/server_c_*.log.expect`, `mtp_census.out` |
| 7 | QPIN self-replay (§15.0) | QKL_HASH equal | 8/8 chunks; PR219 f16 replay per-chunk KLD + hashes identical to PROOF-initial | PASS | `q/logs/w0_a_f16_f16_u4096.log`, `chain2.out` |
| 8 | Q1-A ub4096, 6 types | mean(new − PR219) ≤ 2.0e-4 | f16 +1.369e-4, q8_0 −2.65e-6, q4_0 −1.50e-5, bf16 −4.12e-5, q8_0K/f16V −4.98e-6, f16K/q8_0V −2.31e-5 | **PASS 6/6** | `q/quality.md`, `q/logs/new_a_*`, `w0_a_*` |
| 9 | Q1-B ub512, f16 / q8_0 / q4_0 | ≤ 2.03e-4 | −2.27e-5, −1.20e-5, −7.26e-5 | **PASS 3/3** | same |
| 10 | Q1 direct form (f16 thin margin; q4_0 / bf16 = TODO-LATER #1) | KLD(new ‖ PR219) ≤ band | f16 1.375e-4 ≤ 1.403e-4 PASS; **q4_0 2.690e-3 > 1.441e-3; bf16 1.802e-4 > 1.032e-4** | f16 PASS; q4_0, bf16 **FAIL (deferred #1)** | `q/logs/new_db_*.log` |
| 11 | unpinned catastrophe screen (report) | no catastrophe | plain vs PR219 f16 logits c8192 × 8: KLD 0.0085–0.0089 (q4_0 0.0196); master f16 0.0088, q8_0 0.0094 | PASS (report) | `q/logs/{plain,m50}_u_*.log` |
| 12 | P2 text f16 prefill, new vs master, d0 / d32k / d131k × ub 512/4096 × pp 512/4096 | CI LB ≥ 0.99 | 0.9995 … 1.0820, every LB ≥ 0.9984 | **PASS 12/12** | `perf/p_*_f16_*` |
| 13 | text q8_0 prefill, new vs master (d0, d32k; master q8_0 not run at d131k) | ≥ 0.99 | 1.0173 … 2.7680 | PASS 8/8 | `perf/p_*_q8_0_*` |
| 14 | P3 (new_q8/new_f16)/(m50_q8/m50_f16) | ≥ 0.99 | 10 PASS (up to 2.736); **d0 ub512 pp4096 0.9856 [0.9828, 0.9884] FAIL**; d0 ub4096 tg128 0.9899 [0.9575, 1.0028] RERUN | **10 PASS, 1 FAIL, 1 RERUN** | `perf/` |
| 15 | P3/P4 floor new_q8/new_f16 prefill | ≥ 0.90 | 0.9230 … 1.0022 (worst d131k ub4096 pp512 0.9230, ub512 0.9238) | PASS 12/12 | `perf/` |
| 16 | P5 tg128 vs master, d0 (θ 0.98) and d32k / d131k (θ 0.99) | as stated | f16: 5 PASS, d131k ub4096 0.9998 [0.9840, 1.0151] RERUN; q8_0: 3 PASS (d32k 1.122–1.126), d0 ub4096 0.9889 [0.9566, 1.0010] RERUN (stall sample, row 18) | **8 PASS, 2 RERUN** | `perf/` |
| 17 | P5 new_q8/new_f16 tg | ≥ 0.94 | 0.9839 … 0.9950 | PASS 6/6 | `perf/` |
| 18 | P6(a) stall | no eviction-free sample < 0.9 × its cell median | **1 of 748 samples**: `p_d0_q8_0_0_new` tg128 ub4096 rep 1 = 22.94 t/s vs median 26.30 (0.872×). Rep 0 of that row (22.26) was also slow, so this looks like a warm-up ramp (UNPROVEN) | **FAIL (1 sample)** | `perf/p_d0_q8_0_0_new.json` |
| 19 | P5 MTP cells, default draft, new vs master (θ 0.98) | ≥ 0.98 | f16 d16 0.9991 RERUN, d1024 0.9986 RERUN, **d32768 1.0396 [1.0021, 1.0799] PASS** (was 0.898 FAIL); q8_0 d16 1.0018 RERUN, **d1024 0.6902 [0.6713, 0.7097] FAIL**, d32768 1.1677 PASS | **2 PASS, 1 FAIL, 3 RERUN** | `mtp_perf/*.jsonl` |
| 20 | P5 MTP new_q8_0/new_f16 | ≥ 0.94 | **d16 0.7467 [0.7006, 0.7976] FAIL** (master 0.7447, same tokens drafted / accepted); d1024 1.042 PASS; d32768 1.011 [0.971, 1.049] PASS (LB 0.971 ≥ 0.94) | **2 PASS, 1 FAIL** | same |
| 21 | depth d266240 (n_kv 270336 > 262144), ub4096 | f16 new/master ≥ 0.99; q8/f16 ≥ 0.90 pp, ≥ 0.94 tg | f16 pp4096 1.0134 [1.0050, 1.0213] PASS; tg32 1.0095 [0.9834, 1.0362] RERUN; new q8_0/f16 pp 0.9581 PASS, tg 0.9447 [0.9169, 0.9730] RERUN at θ 0.94 | 2 PASS, 2 RERUN | `perf/deep_*` |
| 22 | AT-O (owner decision data, not pass/fail) | report | §5. Speed ap/capped pp 0.9955, tg 1.0013 (both CIs contain 1). Memory: equal reserve at c131072 / 270336 / 393216. Quality: KLD identical to a capped rerun | **recommend keeping CAPPED** | `ato/`, `perf/deep_*` |
| 23 | Q7 MTP acceptance (deferred, TODO-LATER #9) | LCB95 ≥ −1 pp, n = 14 | default +0.842 (LCB −0.764) PASS; **q8_0 +0.852 (LCB −1.005) FAIL**; **q4_0 +0.021 (LCB −1.940) FAIL**; bf16 +1.176 (LCB −0.756) PASS | 2/4 PASS (**deferred**) | `q7/q7an.out`, `q7/res_q7_*.jsonl` |
| 24 | Vulkan (optional): f16 new vs master (AT-B1 θ 0.99) | ≥ 0.99 | 0.9981 … 1.0034: 4 PASS, 4 RERUN, 0 FAIL | 4 PASS, 4 RERUN | `vk/vk_f16_*` |
| 25 | Vulkan q8_0 new vs master | report | 1.025–1.122 at d4096, **1.660–2.147 at d32768** | report | `vk/vk_q8_0_*` |
| 26 | Vulkan AT-B1 new_q8_0 / new_f16 (TODO-LATER #3) | ≥ 0.95 | 0.8862 … 0.9310 | **FAIL 0/8 (deferred #3)** | `vk/` |
| 27 | Vulkan vpack arm (test-only) | report | q8_0 vpack/default: +6.0 to +10.1 % at d4096, +6.65 % at d32k ub4096 pp2048, **−1.8 to −2.5 % in the 3 other d32k cells**; vpack q8_0/f16: 5/8 ≥ 0.95 | report: do not enable as-is | `vk/vk_*_vkpack*` |

## 1. Setup (unchanged from the partial report)
- **Head and branch.** Head `49a6d75ab` (laptop `qsa-final`), moved to strix-002 as a bundle and fetched as the local branch `qsa-final-proof` in `~/dev/wt_rebase2`.
  - `git diff 51c903292 qsa-final-proof` touches only `scripts/hip/gfx1151-vgpr-baseline.json` (1 line, `source_revision`).
  - The plain arm is therefore the existing `wt_rebase2` `build` (HIP) and `build-vk`, incrementally relinked under flock, rc 0.
- **Lab arm** (`F/lab`):
  - 49a6d75ab + `lab-instrumentation.patch`, which applied cleanly;
  - QPIN 01/02/03, applied as in `prove_final/h/w7/lab_tree.sh`;
  - HIP build rc 0, no warning in the patched QSA code.
- **Other arms.** Master 50c2c8cb5 and PR219 W0-lab12 were used read-only.
- **llama-bench binaries.** The C10 text llama-bench binaries are `F/bench/bin_{new,m50}` (HIP) and `bin_{vknew,vkm50}` (Vulkan, built in phase `vkbuild`, rc 0).
- **Locking and watchdog.** Every GPU job ran under `flock ~/gpu.lock`, one at a time, under the `lib.sh` watchdog:
  - stop on GTT > 100e9, MemAvailable < 4 GiB, or a new amdgpu or OOM kernel line;
  - it never fired.

## 2. Quality: KLD / Q1 (`F/q/quality.md`, scorer `~/dev/wt_lab/prove_final/h/w7/q_an.py`)

### 2.1 Reference and reuse
- `ref8t0` was re-recorded on W0-lab12, as `qrun.sh record w0 f16 f16 4096 8192 8 LLAMA_FN_LAST_ATTN_TRIM=0`. QPIN rows_rec = 786432 and moe_rec = 3112960. The reference was deleted afterwards (22 GB, phase `kldclean`).
- **Bit-identity precondition for reuse: held.**
  - The PR219 f16 ub4096 replay has per-chunk KLD and QKL_HASH lines identical to PROOF-initial's (`KLD w0 f16 u4096 replay identical …: 1`).
  - Self-replay hashes are 8/8, and every arm has rows_absent = moe_absent = 0.
- **Reused logs.** Per `q/logs/REUSED`, 8 PR219 arm logs came from `~/dev/wt_lab/prove/q/logs`:
  - Q1-A partners: q8_0, q4_0, bf16, q8_0K/f16V, f16K/q8_0V (ub4096);
  - Q1-B partners: f16, q8_0, q4_0 (ub512).
- **Fresh runs:** all new arms (9 Q1 arms and 3 direct-form arms), plus the direct-form PR219 bases.
- **Further finding (checked here): all 12 new-arm logs are bit-identical to PROOF-initial's head 1dbee4d9a.** The sorted `QKLD_CHUNK` + `QKL_HASH` lines have the same md5 for:
  - `new_a_{f16,q8_0,q4_0,bf16,q8_0_f16,f16_q8_0}_u4096`;
  - `new_a_{f16,q8_0,q4_0}_u512`;
  - `new_db_{f16,q4_0,bf16}`.
  The trim and the rebase did not change the pinned main-model numerics.
- **Spot check.** I recomputed the paired means and SEs from the raw `QKLD_CHUNK n=… kld=` lines. They equal `q_an.py` to 4 digits (for example f16 Q1-A +1.3687e-4 ± 2.884e-5).

### 2.2 Q1 values, compared with earlier rounds

| arm | final2 (49a6d75ab) | PROOF-initial (1dbee4d9a) | band | verdict |
|---|---|---|---|---|
| Q1-A f16 | +1.369e-4 ± 2.88e-5 | +1.369e-4 | 2.0e-4 | PASS |
| Q1-A q8_0 | −2.65e-6 ± 3.17e-6 | −2.7e-6 | 2.0e-4 | PASS |
| Q1-A q4_0 | −1.50e-5 ± 4.95e-5 | −1.5e-5 | 2.0e-4 | PASS |
| Q1-A bf16 | −4.12e-5 ± 2.00e-5 | −4.1e-5 | 2.0e-4 | PASS |
| Q1-A q8_0K/f16V | −4.98e-6 ± 5.81e-6 | −5.0e-6 | 2.0e-4 | PASS |
| Q1-A f16K/q8_0V | −2.31e-5 ± 2.70e-5 | −2.3e-5 | 2.0e-4 | PASS |
| Q1-B f16 ub512 | −2.27e-5 ± 2.52e-5 | −2.3e-5 | 2.03e-4 | PASS |
| Q1-B q8_0 ub512 | −1.20e-5 ± 9.57e-6 | −1.2e-5 | 2.03e-4 | PASS |
| Q1-B q4_0 ub512 | −7.26e-5 ± 8.24e-5 | −7.3e-5 | 2.03e-4 | PASS |
| direct f16 | 1.375e-4 ± 2.89e-5 | 1.375e-4 | 1.403e-4 | PASS (margin 2.8e-6) |
| direct q4_0 | **2.690e-3 ± 3.46e-4** | 2.690e-3 (stage 2: 2.082e-3) | 1.441e-3 | **FAIL, deferred #1** |
| direct bf16 | **1.802e-4 ± 1.94e-5** | 1.802e-4 (stage 2: 1.250e-4) | 1.032e-4 | **FAIL, deferred #1** |

Absolute pinned KLD vs the PR219 f16 reference (new arm; PR219 arm in brackets) at ub4096:
- f16 1.368e-4 [−3.9e-8, the self-replay];
- q8_0 1.345e-4 [1.372e-4];
- q4_0 5.771e-3 [5.786e-3];
- bf16 1.312e-4 [1.724e-4];
- q8_0K/f16V 1.213e-4 [1.263e-4];
- f16K/q8_0V 1.216e-4 [1.447e-4].

Every quantized new arm is at or below its PR219 partner. f16 is the only arm that moves away from PR219 (+1.37e-4, 68 % of the band). This is the known f16 kernel-change floor (Q2: f16 ub2048 floor 1.08e-4).

Not run in this campaign: Q1-C (ub64), the direct form for q8_0 and the mixed pairs (PROOF-initial: PASS), stage 2 of the direct form, Q2-Q12. Their PROOF-initial values stand only to the extent that the head is unchanged for them, which is shown for the 12 arms above.

### 2.3 Unpinned KLD vs the PR219 f16 logits (c8192 × 8, ub4096, catastrophe screen, no gate)

| arm | Mean KLD | PPL(Q)/PPL(base) | same top p |
|---|---|---|---|
| new f16 | 0.008892 ± 0.000229 | 0.99933 | 97.08 % |
| new q8_0 | 0.008775 ± 0.000170 | 1.00072 | 97.07 % |
| new q4_0 | 0.019601 ± 0.000333 | 1.00246 | 95.31 % |
| new bf16 | 0.008545 ± 0.000175 | 0.99910 | 97.09 % |
| new q8_0K/f16V | 0.008772 ± 0.000181 | 1.00090 | 97.23 % |
| new f16K/q8_0V | 0.008797 ± 0.000179 | 1.00042 | 97.03 % |
| master f16 | 0.008817 ± 0.000198 | 0.99882 | 96.99 % |
| master q8_0 | 0.009417 ± 0.000236 | 1.00108 | 96.87 % |

- new and master sit at the same unpinned route-change level for f16. new q8_0 is lower than master q8_0 (0.00878 vs 0.00942).
- PROOF-initial's §5.5 numbers (new f16 0.0096, q8_0 0.0104, q4_0 0.0242) used a different reference (base f16 logits, c8192 × 4), so they are not directly comparable. They are at the same level.

## 3. Images and MTP

### 3.1 Images, A/B/C matrix (lab build, strict + trace, c261888, b/ub 8192, MTP on)

| KV | A (61057 tok) | B (69803 tok) | C (180157 tok) | prompt t/s A / B / C | peak GTT |
|---|---|---|---|---|---|
| f16 | 2749 | 12749 | 26999 | 1171.6 / 1153.7 / 1094.1 | 96.3 GB |
| q8_0 | 2749 | 12749 | 26999 | 1161.9 / 1143.3 / 1085.4 | 92.9 GB |
| q4_0 | 2749 | 12749 | 26999 | 1156.9 / 1143.9 / 1085.7 | 91.3 GB |
| bf16 | 2749 | 12749 | 26999 | 1168.8 / 1148.5 / 1088.3 | 96.3 GB |
| q8_0K/f16V | 2749 | 12749 | 26999 | 1165.5 / 1142.8 / 1084.9 | 94.4 GB |
| f16K/q8_0V | 2749 | 12749 | 26999 | 1167.6 / 1148.3 / 1089.8 | 94.8 GB |
| master f16 (plain) | 2749 | 12749 | 26999 | 1155.3 / 1135.4 / 1063.9 | 99.6 GB |

- Every lab run: strict `violations=0` in main and MTP, decode_err 0, CPU 0, masked 0, 543 CB ubatches, 580 trace lines all on ROCm0. Spot-checked in the raw `srv_img_q4_0_q4_0.log` and `srv_img_bf16_bf16.log`.
- new / master f16 prompt t/s: A 1.014, B 1.016, C 1.028. This is one server run per arm, so it is report-only. P9 needs 3 interleaved runs and was not scored.
- PROOF-initial (1dbee4d9a): same 18/18 answers. f16 prompt t/s was 1163 / 1140 / 1081.

### 3.2 `-np 2` with images (the PROOF-initial failure #1)
- Split f16 / q8_0 / q8_0K/f16V, and unified q8_0: `LLAMA_STRICT_RESERVE ctx=main violations=0`, `ctx=mtp violations=0`. Answers 2749 / 12749.
- The new one-stream reserve passes `1s-pp` (8192 tokens), `1s-0out`, `1s-fast` and `1s-trim` are present.
- PROOF-initial: 10 violations, +22,594,688 B.

### 3.3 Image understanding (plain build)
- Short (`U_test1`, `U_nyt`), f16 / q8_0 / q4_0: all correct. f16 and q8_0 are identical in the first ~350 characters. q4_0 adds the volume / price line.
- Deep (`U_deep`, 42571 prompt tokens, image in the middle):
  - new f16, new q8_0 and master f16 give the same 4 facts and the same 3 extra headlines;
  - new q8_0 and master f16 are word-identical in the shown prefix, and new f16 differs by "in the middle of the text";
  - prompt t/s: new f16 1074.0, q8_0 1091.2, master f16 1080.7 (single runs).

### 3.4 MTP census (lab, main q8_0, draft default / q8_0 / q4_0; PROOF-initial failure #3)
- `EXPECT PASS` with 0 diffs in all three.
- masked 0, CPU 0. The guard inspected ≥ 12 layers per main pass with cpu_launch 0.
- **Gated share of the fusible G4 attends: 96/96 = 1.000** in all three (floor 0.9). PROOF-initial had 56/96 = 0.583.

### 3.5 MTP perf (P5 MTP cells; llama-server + MTP sidecar, `--spec-draft-n-max 7`, default draft, 2 master + 2 new processes interleaved, 4 timed reps each)

| main KV | depth | master t/s | new t/s | new/master [CI] | verdict | accepted / drafted (reps 1-4) master → new | ms per verify step master → new |
|---|---|---|---|---|---|---|---|
| f16 | 16 | 23.84 | 23.81 | 0.9991 [0.918, 1.084] | RERUN | 620/2698 → 620/2698 | 107.10 → 107.21 |
| f16 | 1024 | 25.37 | 25.34 | 0.9986 [0.978, 1.019] | RERUN | 650/2554 → 650/2554 | 107.94 → 108.10 |
| f16 | 32768 | 19.18 | 19.94 | **1.0396 [1.0021, 1.0799]** | PASS | 552/3160 → 560/3024 | 113.27 → 110.80 |
| q8_0 | 16 | 17.75 | 17.78 | 1.0018 [0.966, 1.040] | RERUN | 480/3544 → 480/3544 | 106.23 → 106.03 |
| q8_0 | 1024 | 38.27 | 26.41 | **0.6902 [0.6713, 0.7097]** | **FAIL** | **768/1624 → 656/2360** | 104.64 → 105.45 |
| q8_0 | 32768 | 17.26 | 20.16 | 1.1677 [1.1135, 1.2226] | PASS | 562/3072 → 568/2944 | 128.68 → 111.67 |

- "ms per verify step" = Σ(n/tg) / (n − accepted).
- **The q8_0 d1024 FAIL comes from acceptance, not from kernel time:**
  - master's greedy continuation is far more draftable: per request, 96 of 203 drafted tokens accepted on master vs 82 of 295 on new;
  - per verify step the two builds are within 0.8 % (104.64 vs 105.45 ms);
  - master's own q8_0 d1024 cell is 1.51× its f16 cell, which shows how text-dependent this cell is.
  The texts were not saved, so "different generated text" is inferred from the different draft counts (UNPROVEN in the strict sense).
- **new_q8_0 / new_f16 ≥ 0.94:**
  - d16 0.7467 [0.7006, 0.7976] FAIL, but master's own ratio is 0.7447 with token-identical draft counts. The d16 q8_0 continuation drafts 3544 and accepts 480, against 2698 / 620 for f16, and the per-step time is equal (106.0 vs 107.2 ms). Acceptance-driven, and not a PR regression.
  - d1024 1.042 PASS; d32768 1.011 PASS.
- **The known P5 MTP f16 d32k failure** (PROOF-initial 0.898 [0.859, 0.949] vs PR219) is **resolved**: 1.0396 [1.0021, 1.0799] vs master, with acceptance 0.185 vs 0.175 and a 2.2 % shorter verify step.
- **Comparison with PROOF-initial (reference was PR219 there):**
  - f16: d16 0.996 → 0.999, d1024 1.001 → 0.999, d32k 0.898 → 1.040;
  - q8_0: d16 1.033 → 1.002, d1024 1.482 → 0.690, d32k 1.097 → 1.168.
  The swing at q8_0 d1024 in both directions shows that this cell measures the draftability of the text more than kernel speed.
- Not run in this campaign: the `-ctkd q8_0` draft cells.

## 4. Performance vs master 50c2c8cb5 (C10 text llama-bench, HIP; 3 + 3 processes at d0 / d32k, 2 + 2 at d131k; 5 reps, rep 0 dropped)

### 4.1 Prefill (pp t/s, new/master ratio [95 % CI])

| depth | ub | pp | f16 master → new | f16 ratio | q8_0 master → new | q8_0 ratio | new q8_0 / new f16 |
|---|---|---|---|---|---|---|---|
| 0 | 512 | 512 | 1355.9 → 1356.5 | 1.0004 [0.9996, 1.0013] | 1328.6 → 1351.6 | 1.0173 | 0.9964 |
| 0 | 512 | 4096 | 1200.3 → 1259.7 | 1.0495 | 1220.6 → 1262.5 | 1.0344 | 1.0022 |
| 0 | 4096 | 512 | 1356.5 → 1355.7 | 0.9995 [0.9984, 1.0004] | 1328.2 → 1352.2 | 1.0181 | 0.9974 |
| 0 | 4096 | 4096 | 2108.3 → 2124.7 | 1.0078 [0.9995, 1.0160] | 1797.6 → 2110.9 | 1.1743 | 0.9935 |
| 32768 | 512 | 512 | 1110.7 → 1189.6 | 1.0711 | 629.8 → 1161.1 | 1.8436 | 0.9760 |
| 32768 | 512 | 4096 | 1149.4 → 1243.6 | 1.0820 | 609.0 → 1224.4 | 2.0105 | 0.9845 |
| 32768 | 4096 | 512 | 1101.3 → 1183.0 | 1.0742 | 626.6 → 1150.1 | 1.8355 | 0.9722 |
| 32768 | 4096 | 4096 | 1948.1 → 1971.0 | 1.0117 | 704.4 → 1949.7 | 2.7680 | 0.9892 |
| 131072 | 512 | 512 | 1108.8 → 1182.6 | 1.0666 | not run | — | **0.9238** |
| 131072 | 512 | 4096 | 1134.9 → 1220.4 | 1.0754 | not run | — | 0.9776 |
| 131072 | 4096 | 512 | 1086.8 → 1163.8 | 1.0709 | not run | — | **0.9230** |
| 131072 | 4096 | 4096 | 1817.6 → 1834.4 | 1.0093 | not run | — | 0.9814 |

- **P2 f16:** 12/12 PASS (θ 0.99). Every ratio not printed with a CI has CI LB > 1.004.
- **q8_0 vs master:** 8/8 PASS. Master's q8_0 runs at 0.36-0.57 of its own f16 at d32k; the PR removes that cliff.
- **P3 floor** (new_q8 / new_f16 ≥ 0.90): 12/12 PASS.
  - The worst cells are d131k pp512 at 0.9230 / 0.9238. They are below 0.94 but above the 0.90 floor. PROOF-initial had 0.9217-0.9241 for the same cells.
- **P3 ratio of ratios:**
  - 10 PASS;
  - **d0 ub512 pp4096: 0.9856 [0.9828, 0.9884] FAIL.** In that cell new f16 gains +4.95 % over master and new q8_0 +3.44 %, and new_q8/new_f16 = 1.0022. The rule fails because f16 improved more than q8_0, not because q8_0 slowed down. PROOF-initial (vs PR219) had this cell at PASS.
  - d0 ub4096 tg128: RERUN (the stall sample of §4.3).
- These ratios match PROOF-initial's f16-vs-PR219 table to about 0.002, for example d32k ub512 pp512 1.0711 vs 1.0721 and d131k ub512 pp512 1.0666 vs 1.0647.

### 4.2 Decode (tg128 t/s)

| depth | ub | f16 master → new | ratio [CI] | q8_0 master → new | ratio [CI] | new q8/f16 |
|---|---|---|---|---|---|---|
| 0 | 512 | 26.92 → 26.91 | 0.9995 [0.9991, 0.9998] PASS | 26.76 → 26.77 | 1.0003 PASS | 0.9950 |
| 0 | 4096 | 26.47 → 26.44 | 0.9990 [0.9970, 1.0011] PASS | 26.31 → 26.02 | 0.9889 [0.9566, 1.0010] RERUN | 0.9839 |
| 32768 | 512 | 24.82 → 24.78 | 0.9985 [0.9950, 1.0019] PASS | 21.82 → 24.57 | 1.1263 PASS | 0.9917 |
| 32768 | 4096 | 24.44 → 24.39 | 0.9982 [0.9947, 1.0016] PASS | 21.54 → 24.17 | 1.1221 PASS | 0.9909 |
| 131072 | 512 | 23.94 → 23.91 | 0.9988 [0.9941, 1.0033] PASS | not run | — | 0.9861 |
| 131072 | 4096 | 23.06 → 23.05 | 0.9998 [0.9840, 1.0151] RERUN | not run | — | 0.9861 |

### 4.3 Stall (P6)
- 748 timed samples were checked, and 1 sample was dropped as an eviction sample (`p_d131072_f16_1_m50`).
- **P6(a): 1 eviction-free sample below 0.9 × its cell median.** It is `p_d0_q8_0_0_new`, tg128 ub4096, rep 1: 22.94 vs median 26.30 (0.872×).
  - The row's samples are `[22.26, 22.94, 26.30, 26.23, 26.30]`. Rep 0 is slow too, so this looks like a warm-up ramp in the first new q8_0 process rather than a mid-run stall. UNPROVEN.
  - It is a FAIL under P6(a) as written.
- PROOF-initial had 0 P6(a) samples and 1 sample > 1.5 × its median.

### 4.4 Depth 266240 (n_kv 270336, just past 262144; ub4096, pp4096 + tg32, RR 3)

| arm | pp4096 t/s (procs) | tg32 t/s | peak VRAM+GTT |
|---|---|---|---|
| master f16 | 1669.73 (2) | 21.43 | 90142 MiB |
| new f16 (capped) | 1692.18 (2) | 21.63 | 89541 MiB |
| new f16 always-pack | 1684.50 (2) | 21.66 | 89540 MiB |
| new q8_0 | 1621.22 (1) | 20.43 | 86189 MiB |

- **new / master f16:**
  - pp 1.0134 [1.0050, 1.0213] PASS;
  - tg 1.0095 [0.9834, 1.0362] RERUN.
- **new q8_0 / new f16:**
  - pp 0.9581 [0.9481, 0.9678] PASS (≥ 0.90);
  - tg 0.9447 [0.9169, 0.9730]: RERUN against the P5 0.94 floor, with only 1 q8_0 process.
- Master q8_0 was not run at depth (memory safety).
- PROOF-initial at d300k (vs PR219): f16 pp 1.0101 PASS, tg 1.0014 unresolved; q8_0/f16 pp 0.957, tg 0.948.

## 5. AT-O: the f16 V-pack cap (owner decision)

**What the setting does** (`sparse-attn-loaders.cuh`; the comment in `sparse-attn-attend-g4.cu`):
- **CAPPED (shipped):** the G4 attend packs V only when n_kv < 64·n_q. At n_q 4096 the switch is at 262144 cells; at n_q 512 it is at 32768.
  - The reserve bound is min(n_kv, 64·n_q) cells. A packed cell costs ~1 KiB here (2 KiB per quad × n_head_kv).
  - The trait is CAPPED for every type, not only f16/bf16.
- **ALWAYS:** packs at every n_kv.
- **How the arm was built:** "ap" is `GGML_CUDA_SA_FORCE=vpacked`, whose get_alloc_size follows the force. In the init-only servers the force is parsed inside `sched_reserve`, before the QSA reserve passes. The llama-bench and llama-perplexity logs filter INFO lines, so the force line is not visible there. Those runs passed it with the same `env` wrapper.

| aspect | measurement | capped | always-pack | ratio / delta |
|---|---|---|---|---|
| speed, prefill | d266240 pp4096 ub4096 (n_kv 270336: capped = direct V, ap = packed), 2 + 2 processes | 1692.18 t/s | 1684.50 t/s | 0.9955 [0.9866, 1.0044] (no difference shown) |
| speed, decode | d266240 tg32 | 21.63 | 21.66 | 1.0013 [0.9733, 1.0301] |
| memory, reserve | init-only server `-lv 4`, f16, b/ub 4096, ROCm0 / host compute buffer | c131072 1573.42 / 201.43; c270336 1611.98 / 202.63; **c393216 1635.42 / 203.68** MiB | **identical at all three** | 0 MiB (arithmetic predicted +8 MiB at 270336 and +128 MiB at 393216 for the attend workspace; at ub4096 the compute-buffer peak is set elsewhere) |
| memory, runtime | peak VRAM+GTT, deep bench | 89541 MiB | 89540 MiB | 0 |
| quality | c32768 ub256 (cap at 16384 cells: capped = direct, ap = packed past it), KLD vs a capped base, 2 chunks | capped rerun: Mean KLD −0.000000, max 6.7e-5, same top p 99.997 %, PPL(Q)/PPL(base) 1.000091 | **identical per-chunk and summary lines** | 0 at the KLD base resolution |

- **Prior rounds (PROOF-initial §6.6, code-trait always-pack tree):**
  - 300k: pp 0.9904 [0.979, 1.003], tg 1.0043;
  - 400k (1 process): pp 0.989, tg 1.018;
  - equal peak (93,291 vs 93,294 MiB).
- **Kernel level (C18):** packed + pack 42.34 vs best direct 52.92 ms at syn 262k, and 62.97 vs 63.65 at syn 1M. That op-level advantage does not show end-to-end in any of three e2e measurements (point estimates 0.990, 0.989, 0.9955).

**Recommendation: keep CAPPED (the shipped default) and close AT-O.**
- **No e2e speed benefit.** At the decision point (n_kv just past 262144 at ub4096) the always-pack prefill point estimate is lower in all three measurements (0.9955, 0.990, 0.989), and every CI contains 1. Decode is flat.
- **No quality difference.** The KLD of always-pack vs capped equals the capped-vs-capped rerun line for line.
- **The memory risk is one-sided.** At ub4096 the reserve is equal up to c393216, because the attend workspace sits under the MoE activation peak. But the always-pack bound grows with n_kv at every ub, by 1 KiB per cell: at ub512, 384 MiB at c393216 against 32 MiB capped. **Whether that raises the reserve at ub512 / ub1024 is UNPROVEN.** The atomem runs were ub4096 only, and the default server uses small ubatches (M5). CAPPED keeps the bound at 64·n_q and cannot exceed it.
- What would change this: an e2e gain at ≥ 1M context (C18's 1M kernel numbers show none), or a ub512 reserve measurement that shows 0 cost and a speed gain. Neither is in hand.

## 6. Vulkan (optional; RADV, `build-vk`, C10 text llama-bench, `-ngl 99 -fa 1 -b 4096 -ub 4096,512 -p 512,2048 -n 0 -d 4096,32768 -r 5`)

**Run design:**
- Arms: master `vkm50` (50c2c8cb5), PR `vknew`, and PR `vkpack` (`GGML_VK_SA_FORCE=vpack`, test-only).
- Processes: 2 per arm for q8_0; 2 / 2 / 1 for f16.
- bf16 was refused by the script. Depth ≤ 32768.
- **Decode was not measured** (`-n 0` in the phase), so Vulkan decode stays UNPROVEN.
- No device loss. GTT returned to 27 MB after every job, and the kernel error count stayed at 3 (boot lines).

| depth | ub | pp | f16 master → new (ratio) | q8_0 master → new (ratio) | q8_0 vpack (vs new default) | new q8_0 / new f16 | vpack q8_0 / new f16 |
|---|---|---|---|---|---|---|---|
| 4096 | 512 | 512 | 1007.2 → 1005.3 (0.998, RERUN) | 901.1 → 930.3 (1.032) | 987.1 (1.061) | 0.925 | 0.982 |
| 4096 | 512 | 2048 | 1010.2 → 1010.4 (1.000) | 882.7 → 933.4 (1.057) | 989.8 (1.060) | 0.924 | 0.980 |
| 4096 | 4096 | 512 | 1032.8 → 1031.6 (0.999, RERUN) | 934.0 → 957.3 (1.025) | 1016.9 (1.062) | 0.928 | 0.986 |
| 4096 | 4096 | 2048 | 1595.2 → 1595.2 (1.000) | 1260.4 → 1413.7 (1.122) | 1556.8 (1.101) | **0.886** | 0.976 |
| 32768 | 512 | 512 | 959.5 → 958.4 (0.999, RERUN) | 527.9 → 882.0 (1.671) | 866.2 (**0.982**) | 0.920 | 0.904 |
| 32768 | 512 | 2048 | 994.8 → 997.0 (1.002) | 546.1 → 918.2 (1.681) | 896.2 (**0.976**) | 0.921 | 0.899 |
| 32768 | 4096 | 512 | 966.1 → 966.7 (1.001, RERUN) | 542.2 → 899.9 (1.660) | 877.9 (**0.976**) | 0.931 | 0.908 |
| 32768 | 4096 | 2048 | 1521.4 → 1526.5 (1.003) | 638.1 → 1370.2 (2.147) | 1461.3 (1.067) | 0.898 | 0.957 |

Every CI for the q8_0 ratios excludes 1. The f16 vpack arm (1 process) is 1.000-1.014 of default.

**Does the op-level gap vs #270 show end to end? No. The PR is the faster one end to end.**
- REBASE2 (op bench, Vulkan0) found the rewrite's quantized prefill attend 1.30-1.65× slower than #270 at 32k with 512 queries (q8_0 4.78 vs 2.89 ms), and 1.69-1.84× slower with 4096 queries (35.1 vs 19.1 ms).
- End to end on the real model, master 50c2c8cb5 (which carries #270) runs q8_0 at **0.42-0.56 of its own f16** at d32k. The PR runs it at 0.90-0.93.
- So the PR's q8_0 prefill is **1.66-2.15× master at d32768** and 1.02-1.12× at d4096. f16 is at parity (0.998-1.003, no FAIL).
- Master's q8_0 runs also peak 5.8 GB higher in GTT (91.1 vs 85.3 GB).
- Why master's e2e q8_0 is so much slower than its op-level number is not established here. A candidate is the f16 copy of the whole K/V that #270 keeps, plus its dequant/gather work, which the op bench does not time. UNPROVEN.

**AT-B1 (new q8_0 ≥ 0.95 × new f16; TODO-LATER #3): FAIL in 8/8 cells, 0.886-0.931.**
- The worst cell (d4096 ub4096 pp2048, 0.886) is below the W7 figure in TODO-LATER #3 (worst 0.921), on a different grid.
- Deferred, as in TODO-LATER #3.

**What vpack recovers:**
- **+6.0 to +10.1 % at d4096**, which lifts q8_0/f16 to 0.976-0.986 (AT-B1 passes in those 4 cells).
- **+6.65 % at d32k ub4096 pp2048** (0.957, passes).
- **−1.8 to −2.5 % in the three d32k cells with 512-query ubatches** (0.899-0.908, still failing).
- This matches REBASE2: vpack closes the 4096×32k case but not the 512-query one. The pack cost scales with n_kv per ubatch, and the gain scales with n_q.
- **Memory cost was not measured in this run.** llama-bench prints no reserve, and the 2 s GTT sampling shows no difference: 85.30 GB both, within 4 MB.
- Prior numbers: +64 MiB at 4096 q × 32k (REBASE2, op level); 68-264 MiB (W7, TODO-LATER #3, against M1).

**Recommendation on vpack: do not enable it by default as it is.**
- It is a net loss at d32k with 512-query ubatches, which is the common server shape.
- It costs 64-264 MiB against M1.
- The PR already beats master on Vulkan q8_0 end to end everywhere measured.

**Follow-up worth one experiment:** the HIP CAPPED rule (pack only when n_kv < 64·n_q) picks the faster arm in **all 8 measured cells**:
- pack at d4096 (n_kv ≤ 6144 < 32768) and at d32k with n_q 2048 (n_kv ≈ 35k < 131072);
- direct at d32k with n_q 512 (n_kv > 32768).
A capped vpack would also bound the workspace at min(n_kv, 64·n_q). It is not implemented and not measured.

## 7. Q7 (MTP draft acceptance, deferred: TODO-LATER #9)

C2 harness, greedy, 14 L/X prompts (L0-L5, X0-X7); S0-S5 are excluded as the plan says.
- **PR219 arms:** the PR219 default arm was rerun and is identical to PROOF-initial (req, draft_n, draft_n_accepted and text), so the PR219 q8_0 / q4_0 / bf16 arms were reused (`q7/REUSED`).
- **New arms:** all 4 were fresh, 20 results each, 0 errors.
- I recomputed the paired statistic from the jsonl files: one-sided t, t_0.95,13 = 1.771. It matches `q7an.out`.

| draft type | W7 pooled (~34 pairs) | PROOF-initial stage 1 (same 14 prompts) | PROOF-initial stage 2 (X8-X21) | **final2 stage 1** | verdict |
|---|---|---|---|---|---|
| default f16 | −0.13 | −1.41 (LCB −2.97) | +1.28 (−1.91) | **+0.842, SD 3.392, LCB −0.764** | PASS |
| q8_0 | −0.53 | −1.88 (−3.83) | +0.93 (−2.45) | **+0.852, SD 3.923, LCB −1.005** | **FAIL** (by 0.005 pp) |
| q4_0 | +0.16 | −0.28 (−1.74) | +1.17 (−1.87) | **+0.021, SD 4.143, LCB −1.940** | **FAIL** |
| bf16 | +0.05 | −1.44 (−3.25) | +1.12 (−2.15) | **+1.176, SD 4.082, LCB −0.756** | PASS |

- Mean acceptance per prompt (PR219 → new): default 62.64 → 63.49 %, q8_0 62.56 → 63.41 %, q4_0 59.51 → 59.53 %, bf16 62.08 → 63.25 %. Identical text: 0/14 for every type.
- Every mean difference is now positive. The q8_0 and q4_0 failures come from the per-prompt SD (3.4-4.1 pp, against the plan's σ = 0.921 pp), not from a drop in the mean.
- Compared with PROOF-initial stage 1 on the same prompts, the new arm moved by +1.7 to +2.7 pp. Its texts differ from PROOF-initial's new arm in 18/20 prompts, which fits the MTP gate-fusion change (gated share 56 → 96/96).
- Stage 2 (14 fresh prompts) was not run in this campaign. Status: **deferred known issue**, not blocking per the owner's decision. Report the four LCBs.

## 8. Deviations, reruns, and what remains unproven

**Deviations from the plan:**
- **Reference.** The perf reference is master 50c2c8cb5, not PR219 W0 (as briefed).
- **Fewer processes.** 3 + 3 processes at d0/d32k and 2 + 2 at d131k, against the plan's 4 + 4. MTP perf used 2 + 2. Vulkan used 2 + 2, and 1 process for f16 vpack.
- **No §16 reruns.** No "2× processes" reruns were run, so every RERUN cell counts as not passing.
- **Depth.** The depth run is 266240 (not ~300k / deepest). AT-O memory is init-only up to c393216 (the ≤ 400k rule), at ub4096 only.
- **Master q8_0** was not run at d131k or at depth (memory safety on the masked route).
- **Reused PR219 logs.** The PR219 KLD and Q7 partner logs were reused from PROOF-initial after the bit-identity checks held (§2.1, §7).
- **Builds.** The plain arm is the `wt_rebase2` build, which differs from 49a6d75ab only in the VGPR json line.
- **The ap arm** is the env force, not a code-trait build as in PROOF-initial.
- **Vulkan decode** was not part of the vk phase (`-n 0`).

**Reruns:** none. Nothing was rerun, and no new GPU job was launched for this report.

**Not run or still unproven** (PROOF-initial results stand only where the head is shown unchanged):
- **Quality gates:** Q1-C, the direct form for q8_0 and the mixed pairs, direct-form stage 2, Q2, Q3, Q4 (TODO #8 margins), Q5, Q6/Q10, Q8, Q11, Q12, Q1-vk.
- **Op tests and placement:** the op tests are covered by REBASE2 on the rebased head a820fa4f0 (1439/1439 ROCm0, 772/772 Vulkan0), not re-run on 49a6d75ab. AT-C and AT-D are not run.
- **Perf:** P1 vs base, P4 (bf16, mixed, generic types), P6(b), the random-token rows, and P9 with 3 runs. Image prompt t/s here is 1 run (new/master f16 1.014-1.028).
- **MTP:** the `-ctkd q8_0` cells.
- **Memory gates:** M1, M2, M5, M6.
- **Vulkan:** decode, memory (AT-B5), AT-B4 e2e, and the vpack memory cost.
- **AT-O at small ub:** its memory at ub ≤ 1024.
- **Two causes left inferred:** the cause of master's slow Vulkan q8_0 e2e, and the text divergence behind the P5 MTP q8_0 d1024 FAIL (texts were not saved).

**TODO-LATER state after this run:**

| # | item | this run | state |
|---|---|---|---|
| 1 | Q1 direct q4_0 / bf16 | q4_0 2.690e-3 > 1.441e-3, bf16 1.802e-4 > 1.032e-4 (bit-identical to PROOF-initial stage 1) | FAIL, deferred, unchanged |
| 2 | AT-D2 `-ngl 0` CPU perf | not re-measured | open (PROOF-initial 0.70× pp512) |
| 3 | AT-B1 Vulkan q8_0 < 0.95 × f16 | first e2e measurement: 0.886-0.931 (8/8 below); vpack 0.899-0.986 | FAIL, deferred; e2e now 1.02-2.15× master |
| 4 | AT-B5 Vulkan MTP +12 MiB | not re-measured | open |
| 5 | AT-D1 kind-4 wording | not re-measured | open |
| 6 | M1 iq4_nl MTP | not re-measured (PROOF-initial: cleared) | closed per PROOF-initial |
| 7 | `-nkvo` open cells | not re-measured | open |
| 8 | Q4 thin margins | not re-measured | open |
| 9 | Q7 | default +0.842 (LCB −0.764), q8_0 +0.852 (−1.005), q4_0 +0.021 (−1.940), bf16 +1.176 (−0.756) | 2/4 pass at stage 1; deferred |

**New open items from this run (not in TODO-LATER):**
- P5 MTP q8_0 d1024: 0.690 vs master. Acceptance-driven; the per-step time is within 0.8 %.
- P5 MTP q8_0/f16 at d16: 0.747. Master has the same ratio, 0.745.
- P3 ratio of ratios, d0 ub512 pp4096: 0.9856. new f16 gained more than new q8_0.
- P6(a): 1 sample at 0.872×, in the first rep after warm-up.
- 8 RERUN cells in the HIP gates, plus 4 Vulkan f16 cells on the optional arm:
  - perf: P5 f16 tg d131k ub4096, q8_0 tg d0 ub4096, the P3 tg d0 cell;
  - MTP perf: 3 cells;
  - depth 266240: 2 cells.

**Owner decisions needed:**
1. AT-O: keep CAPPED (recommended, §5), or ask for a ub512 reserve measurement first.
2. Whether the new P5 MTP and P3 single-cell FAILs need a rerun with saved texts and 4 + 4 processes before merge, or can be listed as known issues. Each has evidence that it is not a kernel slowdown.
3. Vulkan vpack: leave it test-only (recommended). Optionally try a 64·n_q-capped vpack as a follow-up.

## 9. Logs (all under `F` = `strix-002:~/dev/wt_lab/final2`)
- **Chains and helpers:** `chain1.out`, `chain2.out`, `chain.sh`, `chain2.sh`, `lib.sh`; harness copies in `h/`.
- **Identity:** `ident/`.
- **Images and `-np 2`:** `img/` (`srv_*.log`, `res_*.txt`, `watch_*.txt`).
- **MTP census:** `mtp/`, `mtp_census.out`.
- **AT-O:** `ato/` (`mem_{capped,vpacked}_c{131072,270336,393216}.log`, `q_{capped,capped2,vpacked}.log`).
- **KLD:** `q/logs/` (incl. `REUSED`) and `q/quality.md`. The references were deleted.
- **Perf and depth:** `perf/` (`p_*`, `deep_*`: `.json`, `.samp`, `.peak`, `.watch`).
- **MTP perf:** `mtp_perf/`, `mtp_perf.out`.
- **Q7:** `q7/` (`q7an.out`, `REUSED`, `res_q7_*.jsonl`).
- **Vulkan:** `vk/`, `bench/bin_vk*`, `bb_vk*.log`, `m50_vk_incr.log`.
- **Analysis drivers used for this report:** `/tmp/f2an.py` and `/tmp/f2vk.py` on strix-002 (they import `~/dev/wt_lab/prove/h/w7/perf_an.py`). The MTP-perf and Q7 recomputations were inline Python and are reproduced in §3.5 / §7.

## 10. PR body summary

```markdown
### End-to-end validation (strix-002, gfx1151, head 49a6d75ab; reference master 50c2c8cb5 / PR219 W0-lab12)

| check | result |
|---|---|
| Pinned KLD vs PR219 (Q1-A ub4096, 6 KV types; Q1-B ub512) | 9/9 within band (f16 +1.37e-4, the others −7.3e-5 to −2.7e-6; band 2.0e-4); bit-identical to the pre-trim head |
| Images A/B/C × 6 KV types, `-np 2` split/unified | 18/18 + 4/4 correct; strict reserve 0 violations (split-stream fix verified) |
| MTP route census, gated share | 96/96 (floor 0.9) for draft default / q8_0 / q4_0 |
| HIP prefill vs master, f16 / q8_0 | 1.00-1.08× / 1.02-2.77× (d0-d131k); new q8_0/f16 ≥ 0.923 |
| HIP decode vs master | f16 0.998-1.000×, q8_0 0.99-1.13× |
| Depth 266240 f16 | pp 1.013× master; q8_0/f16 pp 0.958 |
| MTP tg vs master (default draft) | f16 d32k 1.040× (old 0.898 fail resolved), q8_0 d32k 1.168× |
| Vulkan prefill vs master | f16 0.998-1.003×; q8_0 1.03-1.12× (d4k), 1.66-2.15× (d32k) |

**Known issues / follow-ups**
- Q1 direct form q4_0 2.69e-3 (band 1.44e-3), bf16 1.80e-4 (band 1.03e-4); difference form passes.
- Q7 MTP acceptance: mean new − PR219 +0.02 to +1.18 pp, but LCB95 q8_0 −1.005 / q4_0 −1.94 pp < −1 pp (per-prompt SD 3.4-4.1 pp).
- Vulkan q8_0 prefill 0.886-0.931 × f16 (criterion 0.95); test-only vpack helps at d4k (+6-10 %) but loses 2 % at d32k with 512-query ubatches.
- MTP q8_0 d1024 0.69× master is acceptance-driven (per-verify-step time within 0.8 %); one P3 cell 0.9856 (f16 gained more than q8_0); one warm-up sample at 0.87×.
- `-ngl 0` CPU sparse op is scalar (0.70× PR219 at pp512); f16 V-pack kept CAPPED (always-pack: no e2e gain, identical KLD).
```
