
## C1

**Question.** Dense mode yes or no (Track P4): does sparse attention lose to dense FA at small n_kv / n_q, and does G=4 beat base dense q8_0 at ub512 d0?
**Answer.** For **q8_0, no dense path is needed.** Sparse attention matches or beats dense FA in every prefill cell. For decode (n_q = 1) at short depth it gives up 1.6–2.0% end-to-end. For **f16/f16, keep a dense FLASH_ATTN_EXT shortcut** at n_kv ≤ 2051: there, sparse loses 6% at n_q = n_kv = 1024 and 27% at n_q = n_kv = 2048. Delete FLASH_ATTN_EXT past the shortcut: sparse G=4 beats masked dense FA in every cell with n_kv ≥ 2560 and n_q ≥ 128. **G=4 beats base dense q8_0 at ub512 d0**: +1.9% end-to-end, where the PR loses 4.2%. The G1/G4 crossover is **below n_q = 128**.

### Setup (all on strix-002, run on 2026-11-04 between 00:39 and 04:33 CET; every GPU run under `flock ~/gpu.lock`; Q-track jobs interleaved through the same lock)
- Worktree `~/dev/wt_lab`, detached at 94e086847 (PR #219 head). Lab patch: `~/dev/wt_lab/c1/lab.diff`, copy at `SP/final219/c1/lab.diff`. Every switch is env-gated and defaults off. With no env set, the lab binary matches the PR binary: ub512 pp4096 q8_0 gives lab 1400.9 vs PR 1400.7 t/s, and f16 gives 1434.7 vs 1435.3 (rows E below).
  - `GGML_QSA_TRACE=1` (fattn.cu) prints the kernel family once per (n_q, n_kv, ids width, mask, K, V). That is how each route below was checked.
  - `GGML_QSA_DECODE_MAXQ=0` (qsa-decode.cu supported check) sends 128 ≤ n_q ≤ 512 to qsa_prefill. qsa_prefill is the existing G=4 union kernel: `#define QSA3_G 4`, PR qsa-prefill.cu:14.
  - `LLAMA_QSA_ALLVIS=2` (qwen4exp.cpp, dense-shortcut branch, n_kv ≤ 2051). It builds **maskless all-visible rows**: cell j where kq_mask = 0, else −1, made by step/mul/scale_bias/cast ops in the graph. It calls the selected-key op with no mask, so it is the "all-visible rows" prototype of plans W and Z. `=1` is an earlier variant that passes ids 0..n_kv−1 plus the causal mask. It is slower and is kept only as a reference.
  - `GGML_QSA_MASKED_DENSE=1` (build2), with `LLAMA_QSA_SELECTED_KEY=0`. It forces the masked route onto the dense FA kernels. This is the "dense" reference past the shortcut. It was needed because on the PR binary the masked route (mask + src[5]) is **taken by qsa_decode** for n_q ≤ 512 (trace `qsa_decode nq 512 … ids 2051 mask 1` in `logs/ppl_sk0_q8_0_q512.log`). The dense FA in md runs on `dense_tile_vec_mma` (trace in `logs/ppl_md_*`). md and base agree for q8_0 n_q 512: 37.16 vs 37.15 ms at n_kv 4096, and 198.75 vs 197.77 at 16384.
  - Builds: `build` (lab), `build2` (+ MASKED_DENSE), `build3` (+ ALLVIS=2). All use the mandated HIP flags with `-j 16 --target llama-bench llama-perplexity`. BASE = `~/dev/wt_pr219v2_base/build/bin` (13a279bfa). PR = `~/dev/wt_pr219v2/build/bin` (94e086847).
- **Real text, op-profiled.** One run gives the full n_kv sweep, because ubatch j of a chunk has n_q = ub and n_kv = (j+1)·ub. Command (`c1/ppl.sh TAG BIN TYPE NQ ENV…`):
  `flock ~/gpu.lock env <ENV> GGML_CUDA_OP_PROF=1 GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_OVL=0 GGML_QSA_TRACE=1 LLAMA_FN_LAST_ATTN_TRIM=0 <BIN>/llama-perplexity -m ~/models/qwen3.8-flash-next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf -f ~/data/wikitext-2-raw/wiki.test.raw -c 16384 -b NQ -ub NQ --chunks 6 -ngl 99 -fa on -lm mmap -lzm on-direct -ctk T -ctv T`
  - The trim is off so that all 12 QSA layers run the full n_q. The trim is measured separately in row T.
  - Metric: the sum of the `FLASH_ATTN_EXT` OPPROF lines per graph (12 ops = the 12 QSA layers, checked as n_fa = 12 on every graph). This is **FA ms per ubatch**: the median over chunks 1..5, with chunk 0 dropped as warm-up (n = 5), and ± is the sd over those chunks.
  - `d_sum` is the paired per-chunk delta of the whole-graph op sum (same text). It captures the mask-build, ids-build and indexer costs outside FA.
  - Parsers: `c1/parse.py`, `c1/cmp.py`, `c1/summary.py`.
- **Random tokens, op-profiled** (`bench.sh … sweep`): `llama-bench -p 16384 -n 0 -d 0 -b NQ -ub NQ -r 5` with the same env. It uses the last 5 reps and drops rep 0 (n = 4).
- **End-to-end, no profiler** (`bench.sh … e2e`): `llama-bench -ngl 99 -fa 1 -lzm on-direct -ctk T -ctv T -p 4096 -n 0 -d 0 -b 4096 -ub 512 -r 5 -o json`, run in the order base, lab, g4, avg4, pr, base2 (av2g4 was run later). Decode: `-p 0 -n 128 -d 0,1024,1900 -r 5`.
- Raw logs and CSVs: `strix-002:~/dev/wt_lab/c1/logs/` (219 MB, all < 1 GB). Drivers: `c1/campaign.sh` (first 4 q8 n_q512 runs), `campaign_b.sh`, `campaign_c.sh`, `campaign_d.sh`, `campaign_e.sh`. Copies of the scripts and the table are in `SP/final219/c1/`. No run had a non-zero exit, an abort or a load failure (`grep "EXIT [1-9]|abort|failed to load"` on all logs is empty).

### Noise
- Within a run, FA sd over 5 chunks is 0.0–0.9 ms. Most cells are below 1% of the median, as the ± columns show.
- The whole-graph sum varies more on real text (sd 5–15 ms on ~400–600 ms graphs, because MoE routing depends on the text). For that reason d_sum is paired by chunk.
- Run-to-run repeats:
  - md vs base, q8_0 n_q512: within 0.7% at the 8 listed n_kv.
  - baseR vs base, q8_0 n_q4096: medians within 1.2%.
  - labR vs lab, q8_0 n_q4096: medians within 1% at n_kv 4096–12288. At 16384 the first run's median (319.21) is stall-affected; labR gives 204.44.
  - e2e base vs base2: −0.3% (q8_0), −0.2% (f16).
  - decode lab vs lab2: −1.1% to +0.5%.
- **Unexplained stall samples (cause UNPROVEN).** The first q8_0 n_q4096 lab run had 8 of 24 graphs with FA about 2× (111.8 → 244.8 ms at n_kv 4096; 200 → 457 ms at 16384; `logs/ppl_lab_q8_0_q4096.csv`). The repeat `labR` had none (`logs/ppl_labR_q8_0_q4096.csv`, sd ≤ 3.3 ms). Isolated ~2× samples also occur in f16 n_q1024 n_kv4096 (sd 11.55). The q8_0 n_q4096 rows below use labR.

### Results: FA ms per ubatch (12 QSA layers), real text, median ± sd (n = 5)
In the "allvis" rows, n_kv ≤ 2048 is the dense-shortcut region. There, dense = today's plain causal FA and sparse = maskless all-visible rows (ALLVIS=2). In the "select" rows, dense = masked dense FA (md for f16, base for q8_0) and sparse = the PR's 2051-key selection. G1 = qsa_decode and G4 = qsa_prefill. Ratio = dense / sparse; ≥ 0.98 means sparse is not slower than the W rule allows. Full table: `SP/final219/c1/summary.txt`.

| type | n_q | n_kv | dense | G1 (PR route ≤512) | G4 | dense/G4 |
|---|---|---|---|---|---|---|
| f16 | 128 | 512 / 1024 / 2048 (allvis) | 2.07 / 3.14 / 5.29 | 5.53 / 10.75 / 18.69 (masked allvis) | 1.45 / 2.54 / 4.66 | 1.43 / 1.24 / 1.135 |
| f16 | 128 | 2560 / 4096 / 8192 / 16384 | 6.28 / 9.58 / 18.31 / 39.54 | 14.17 / 14.29 / 14.63 / 15.01 | 5.36 / 6.82 / 8.79 / 9.48 | 1.17 / 1.41 / 2.08 / 4.17 |
| f16 | 512 | 512 / 1024 / 2048 (allvis) | 4.19 / 11.03 / 21.15 | 17.52 / 34.57 / 67.64 (masked allvis) | 3.94 / 7.59 / 15.85 | **1.063** / 1.45 / 1.33 |
| f16 | 512 | 2560 / 4096 / 8192 / 16384 | 25.15 / 40.00 / 80.86 / 202.25 | 53.66 / 54.09 / 55.15 / 56.58 | 19.98 / 22.94 / 26.61 / 31.59 | 1.26 / 1.74 / 3.04 / 6.40 |
| f16 | 1024 | 1024 / 2048 (allvis) | 10.44 / 31.53 | – | 11.12 / 24.62 | **0.938** / 1.28 |
| f16 | 1024 | 4096 / 16384 | 62.00 / 384.05 | – | 40.05 / 53.45 | 1.55 / 7.19 |
| f16 | 2048 | 2048 (allvis) | 25.64 ± 0.08 | – | 35.30 ± 0.14 | **0.726** |
| f16 | 2048 | 4096 / 16384 | 112.06 / 766.90 | – | 71.40 / 99.71 | 1.57 / 7.69 |
| f16 | 4096 | 4096 / 8192 / 16384 | 158.96 / 432.99 / 1225.02 | – | 108.62 / 168.80 / 192.71 | 1.46 / 2.57 / 6.36 |
| q8_0 | 128 | 512 / 1024 / 2048 (allvis) | 2.13 / 3.21 / 5.41 | 7.44 / 14.98 / 25.96 (masked allvis) | 1.57 / 2.74 / 5.08 | 1.36 / 1.17 / **1.063** |
| q8_0 | 128 | 2560 / 4096 / 8192 / 16384 | 6.51 / 10.14 / 19.12 / 42.41 | 17.18 / 17.76 / 18.67 / 19.49 | 5.89 / 7.22 / 8.65 / 11.05 | 1.105 / 1.40 / 2.21 / 3.84 |
| q8_0 | 512 | 512 / 1024 / 2048 (allvis) | 6.31 / 11.22 / 22.00 | 18.89 / 43.36 / 91.79 (masked allvis) | 3.16 / 7.26 / 16.10 | 2.00 / 1.55 / 1.37 |
| q8_0 | 512 | 2560 / 3072 / 4096 / 8192 / 16384 | 23.98 / 28.42 / 37.15 / 80.41 / 197.77 | **63.14 / 64.41 / 65.41** / 67.35 / 72.29 | 20.05 / 21.71 / 23.61 / 27.11 / 31.16 | 1.20 / 1.31 / 1.57 / 2.97 / 6.35 |
| q8_0 | 1024 | 1024 / 2048 (allvis) | 13.93 / 31.01 | – | 9.89 / 25.01 | 1.41 / 1.24 |
| q8_0 | 1024 | 4096 / 16384 | 62.48 / 388.80 | – | 42.15 / 55.63 | 1.48 / 6.99 |
| q8_0 | 2048 | 2048 (allvis) / 4096 / 16384 | 45.14 / 110.63 / 722.43 | – | 35.30 / 76.27 / 106.05 | 1.28 / 1.45 / 6.81 |
| q8_0 | 4096 | 4096 / 8192 / 16384 | 158.65 / 428.91 / 1218.99 | – | 113.44 / 179.91 / 204.44 (labR) | 1.40 / 2.38 / 5.96 |

Notes on the table:
- **Cells without a cell.** n_kv < n_q cannot occur in single-sequence prefill (n_kv = (j+1)·n_q). So for n_q 4096 the sweep starts at n_kv 4096. n_q 2048 was added because ub2048 is the only remaining d0 shape whose first ubatch falls in the shortcut region.
- **The G1 column is the PR's ub512 regression.** At n_q 512, qsa_decode costs 63–72 ms against 24–80 ms for dense q8_0. G1 loses to dense up to n_kv ≈ 7.2k (67.46 vs 67.18 ms at 7168, `ppl_lab_q8_0_q512.csv` vs `ppl_base_…`). The pp4096 ubatches (n_kv 2560–4096) cost +23.6 to +37.0 ms d_sum each. This settles C13: the cause is the n_q ≤ 512 → qsa_decode routing (PR fattn.cu dispatch order, qsa-decode.cuh:5 `QSA_DECODE_MAX_QUERIES 512`), not the qsa_prefill repack.
- **Whole-graph deltas** (d_sum, paired). They change no conclusion:
  - f16 n_q2048 n_kv2048: allvis +7.48 ± 0.36 ms vs dense.
  - f16 n_q1024 n_kv1024: +8.92 ± 13.25 (within noise).
  - q8_0 n_q512 n_kv 512–2048: allvis G4 −2.98 to −8.91 ms.
  - q8_0 n_q128 n_kv2048: 0.00 ± 0.17.
- **Random tokens** (`logs/bench_sweep_*`, n = 4) give the same ordering. G4 costs more than on text (union inflation): q8_0 n_q512 at 4096 / 8192 / 16384 is 27.69 / 35.82 / 42.62 on random vs 23.61 / 27.11 / 31.16 on text. It still beats base dense (39.37 / 86.19 / 209.75), and base dense is 6–7% above its text value.
  - q8_0 n_q4096 at 16384: PR 278.15 vs base 1230.99.
  - f16 n_q512 n_kv512 (masked allvis, ALLVIS=1): 4.50 vs dense 3.78. The maskless allvis variant was not run on random tokens.

### Results: end-to-end and the other cells
- **E: ub512 pp4096 d0, random tokens, no profiler** (`logs/bench_e2e_*.json`, avg ± sd of 5 samples, t/s):
  - q8_0:
    - base 1465.0 ± 12.8 and base2 1460.2 ± 17.5 (mean 1462.6)
    - PR 1400.7 ± 14.9 (**−4.2%**); lab 1400.9 ± 13.2
    - **g4 (DECODE_MAXQ=0) 1490.2 ± 14.8 (+1.9%)**
    - avg4 1494.9 ± 15.7 (+2.2%)
    - av2g4 1498.7 ± 19.0 (+2.5%)
  - f16:
    - base 1435.8 and base2 1433.0 (mean 1434.4)
    - PR 1435.3; lab 1434.7
    - g4 1509.8 ± 18.8 (+5.3%)
    - avg4 1515.6 (+5.7%)
    - av2g4 1518.6 ± 23.3 (+5.9%)
  - The sample sd (~1.2%) mostly reflects a per-sample pattern shared by every run (samples 4–5 are the lowest in every run), so run-to-run differences are tighter. The base vs base2 drift is ≤ 0.3%.
- **Decode, n_q = 1, at short depth** (`logs/bench_tg_*.json`, tg128 t/s at d 0 / 1024 / 1900, all in the shortcut region):
  - f16:
    - dense lab 26.19 / 26.41 / 26.35 and lab2 26.16 / 26.13 / 26.29
    - av2 (maskless all-visible → qsa_decode) 25.49 / 26.06 / 25.98, i.e. 0.974 / 0.992 / 0.987 of the dense mean. The d0 value carries a 23.7 first sample; the medians are 25.95 / 26.11 / 26.00, i.e. 0.990 / 0.994 / 0.988.
  - q8_0:
    - dense lab 26.10 / 26.19 / 26.11 and lab2 26.04 / 26.33 / 26.22
    - av2 25.62 / 25.74 / 25.74, i.e. **0.983 / 0.980 / 0.984**
  - In all 6 cells the sparse value sits below the dense mean, but by less than 2.6%.
- **T: trimmed last layer** (ub4096 pp4096 d12288, trim on, `logs/bench_trim_*`).
  - Layer 47 then runs 512 queries. The f16 per-layer label `FLASH_ATTN_EXT <Qcur-47 … +5` gives G1 (PR) 4.5–5.0 ms vs G4 1.8–3.2 ms per ubatch.
  - q8_0 (aggregate FA): 116.7 / 210.8 / 231.1 vs 113.5 / 207.3 / 229.5 ms per ubatch, so 1.6–4.1 ms is saved per ubatch.
  - e2e (profiled run): q8_0 1896.1 → 1899.8 and f16 1944.7 → 1953.3 t/s. These are within noise.
- **PR masked route (`LLAMA_QSA_SELECTED_KEY=0`, sk0) and `LLAMA_QSA_DENSE_SHORTCUT=0` (sc0)**, q8_0 n_q512.
  - Both run on qsa_decode with a mask, according to the trace.
  - sk0: 92.09–115.15 ms. sc0 in the shortcut region: 18.63 / 39.49 / 82.96 ms at n_kv 512 / 1024 / 2048, against dense 6.31 / 11.23 / 22.08.
  - So disabling the shortcut on the PR binary is never a speedup. The PR's masked route (images, non-scalar ubatches) is slower than dense FA below n_kv ≈ 10k (`logs/ppl_sk0_q8_0_q512.csv`, `ppl_sc0_*`).
- **Sanity check only, not a quality claim.** The 6-chunk c16384 PPL of every n_q 512 configuration lies in 2.9215–2.9258. The G4 paths give 2.9215 (q8_0) and 2.9220 (f16) in every run. Base gives 2.9233 / 2.9230 (`grep "PPL =" logs/ppl_*.log`). Quality gating belongs to Track Q (KLD).

### Decision (rule: sparse ≥ 0.98 × dense ⇒ delete dense FLASH_ATTN_EXT; plan W E7, rewrite219/plan-v2-W.md:264)
1. **Past the shortcut (n_kv > 2051), delete FLASH_ATTN_EXT, i.e. the masked route, on QSA layers.**
   - G4 sparse beats masked dense FA in every measured cell with n_q ≥ 128, by 1.105× to 7.7×, on text and on random tokens.
   - This supports W, X, Y and Z (one sparse op past the shortcut), not the critique/A "dense mode for small n_kv".
   - The ub512 regression is the G1 routing: the "dense/G1" column is < 1 up to n_kv ≈ 7k.
2. **Dense shortcut (n_kv ≤ 2051).**
   - **f16/f16 fails the rule**: dense/G4 is 0.938 at n_q = n_kv = 1024 and 0.726 at n_q = n_kv = 2048 (first ubatch at ub ≥ 1024, d0).
   - So FLASH_ATTN_EXT stays for f16/f16 in the shortcut. This is X's, Y's and Z's position; the W §2.7 "replace the dense shortcut for every type" is refuted for f16.
   - **q8_0 passes**: prefill ≥ 1.063 in every cell; decode e2e 0.980–0.984.
   - So **non-f16 pairs can take the sparse op with all-visible rows** (Z §7.8 / L78). This also removes the need to intersect dense-FA type matrices (the Vulkan bf16-mix issue, L40).
   - The q8_0 decode margin is thin (0.980 at d1024, n = 5, run-to-run noise ≈ 1%). The rewrite should re-check it on its own kernel.
   - The threshold stays `n_kv ≤ indexer_top_k + r − 1` (2051). No data supports moving it: sparse already wins from n_kv 2176 (f16 n_q128: 4.88 vs 5.76).
3. **G1/G4 crossover: G4 wins at every measured n_q ≥ 128.**
   - G1/G4 time is 1.6–2.9× at n_q 128 and 1.8–3.1× at n_q 512, past the shortcut (n_kv 2560–16384).
   - So the per-query kernel should take only n_q < 128. qsa_prefill's floor is n_q ≥ 128 (PR qsa-prefill.cu:616). Equivalently, `QSA_DECODE_MAX_QUERIES` goes from 512 to 127.
   - The crossover between n_q 1 and 127 is **UNPROVEN**: there is no G4-capable kernel below 128 queries to time. The trimmed last layer (512 queries) moves to G4 by the same rule (row T).
4. **G=4 vs base dense q8_0 at ub512 d0: yes.**
   - FA per ubatch at n_kv 2560–4096 is 20.05–23.61 ms vs 23.98–37.15 ms.
   - End-to-end, pp4096 is 1490.2 vs 1462.6 t/s (+1.9%), and 1498.7 (+2.5%) with all-visible rows in the shortcut, where the PR is at 1400.7 (−4.2%).
   - f16 gains +5.3% to +5.9% over base for the same reason.

## C2

**Question.** MTP sparse or dense (Track P6). Do the draft's FA nodes reach the QSA kernels? How does the draft behave with `-ctkd q8_0`? (LEDGER.md §C2)

### Setup (strix-002, 2026-11-04 04:39–05:50 CET; every build and run under `flock ~/gpu.lock`; I was the only GPU user)
- **Binary:** `~/dev/wt_lab` at 94e086847 (the PR head).
  - The tree carries the C1 lab diff. It is env-gated and inert here: none of `GGML_QSA_TRACE`, `GGML_QSA_MASKED_DENSE`, `GGML_QSA_DECODE_MAXQ`, `LLAMA_QSA_ALLVIS` was set. The diff is saved before C2 as `~/dev/wt_lab/c1/c1_lab_state_before_c2.diff`.
  - On top of it is a C2 launch counter, saved as `~/dev/wt_lab/c2/c2_lab.diff`, patch scripts `c1/c2_patch.py` and `c1/c2_patch2.py`. With `GGML_QSA_COUNT=1` it does three things:
    - It counts every HIP flash-attention launch per (backend context pointer, kernel family, node, n_q class, n_kv class, mask?, src[5] ids?, K type, V type). It hooks each branch of `ggml_cuda_flash_attn_ext`, PR fattn.cu:1151-1197, and the two fused gated paths, `qsa_prefill_gated` and `fa7_gated`, PR ggml-cuda.cu:6787-6795.
    - It prints the table at exit (`C2COUNT exit`).
    - It prints every CPU `flash_attn_ext` launch (`C2CPU_FA`, ggml-cpu ops.cpp), which detects a silent CPU fallback.
  - The draft context is identified as the context pointer first seen after the log line `create_memory: MTP context uses a hybrid-idx memory`.
  - Build: `cmake --build build -j 16 --target llama-server`, logs `c1/c2_build{,2,3}.log`.
- **Server:** `~/dev/wt_lab/c2/run_c2.sh TAG DRAFT_KV SPEC_TYPE`. It uses the `~/repro_img/repro_kv.sh` flags verbatim, with these changes:
  - main KV is fixed at `-ctk q8_0 -ctv q8_0`;
  - `-ctkd X -ctvd X` is added unless DRAFT_KV=default;
  - `--spec-type` and `--parallel` are parametrised;
  - the binary is the wt_lab one;
  - `GGML_QSA_COUNT=1` is set.

  The other flags are unchanged: Signal-3.8 + `mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, mmproj, `-c 261888 -b 8192 -ub 8192 --spec-draft-n-max 3 -fa on -fit off`, `-dev ROCm0 --spec-draft-device ROCm0`.
- **Workload:** real text only, 14 chat requests (`c2/mkreq_c2.py`, `c2/mkreq_c2x.py`).
  - Each request is a wikitext-2 test article excerpt plus "Summarize ...".
  - Settings: `temperature 0, top_k 1, seed 1, max_tokens 320, enable_thinking false`.
  - S0–S5: 910–1062 prompt tokens. The draft stays in the dense shortcut, n_kv ≤ 2051.
  - L0–L5: 5083–6675 prompt tokens.
  - X0/X1: 28658 / 27062 prompt tokens.
  - Requests are sequential. Acceptance is the response `timings.draft_n_accepted / draft_n`.
- **Runs:** logs `~/dev/wt_lab/c2/server_<TAG>.log`, results `res_<TAG>.jsonl`, summaries via `python3 summ_c2.py TAG` and `python3 paired_c2.py A B`.

  | TAG | draft KV | spec type | notes |
  |---|---|---|---|
  | repro_q8d | q8_0 | draft-mtp,ngram-mod | HIP graphs on; S0–L5 only |
  | g0_repro_f16d / g1_repro_f16d | default (= f16) | draft-mtp,ngram-mod | two repeats |
  | g0_repro_q8d / g1_repro_q8d | q8_0 | draft-mtp,ngram-mod | two repeats |
  | g1_mtp_f16d / g1_mtp_q8d | f16 / q8_0 | draft-mtp only | isolates MTP acceptance from ngram-mod drafts |
  | g1_repro_q4d | q4_0 | draft-mtp,ngram-mod | type-coverage probe |
  | g1_par2_q8d / g1_par2_f16d | q8_0 / f16 | draft-mtp | `--parallel 2`, S0/L0/X0. With n_seq_max > 1 the draft is not cache-only (PR qwen4exp.cpp:491), so its prefill runs attention |

  The g0 runs set `GGML_CUDA_DISABLE_GRAPHS=1` and the g1 runs set it to `=0`. The variable's mere presence disables HIP graphs (PR ggml-cuda/common.cuh:1408), so both g0 and g1 are graphs-off repeats. Only `repro_q8d` ran with graphs on.

### Results: route census (draft context only; launches per run, from `C2COUNT exit`)
| run | dense FA, masked, n_kv ≤ 2051 (shortcut) | qsa_decode, maskless + ids, n_kv > 2051 | qsa_prefill(_gated), n_kv > 2051 | dense FA with n_kv > 2051 | CPU FA lines | draft KV |
|---|---|---|---|---|---|---|
| g1_repro_f16d | 1833 (nq 1) + 1 (nq 2, reserve) | **2415** (nq 1) | 0 | **0** | 0 | f16 |
| g1_repro_q8d | 1830 + 1 | **2424** (nq 1), K=V=q8_0 | 0 | **0** | 0 | q8_0 |
| g1_mtp_f16d | 1833 + 1 | 2430 | 0 | 0 | 0 | f16 |
| g1_mtp_q8d | 1827 + 1 | 2427 | 0 | 0 | 0 | q8_0 |
| g1_par2_f16d | 298 + 101 (nq 2-8) + 1 (nq ≥ 128) | 619 (nq 1) + 208 (nq 2-8) | **5 `qsa_prefill_gated`** (nq ≥ 128) | 0 | 0 | f16 |
| g1_par2_q8d | 298 + 101 + 1 | 616 + 207 | **5 `qsa_prefill`** (nq ≥ 128), q8_0 | 0 | 0 | q8_0 |
| g1_repro_q4d | 1866 + 1 | **0** | 0 | **2502 (nq 1, mask + ids, q4_0)** | 0 | q4_0 |

- **Startup logs.**
  - Draft f16/q8_0: `QSA attention: selected-key kernels (maskless block selection) (K f16, V f16)`, or `(K q8_0, V q8_0)`. See `server_g1_repro_{f16d,q8d}.log`, the line after `MTP context uses a hybrid-idx memory`.
  - Draft q4_0: `QSA attention: masked top-k (no selected-key kernels for this backend and cache type) (K q4_0, V q4_0)` (`server_g1_repro_q4d.log:7241`).
  - The selection comes from the probe at PR src/llama-memory-hybrid-idx.cpp:50-58 and :120-129. qsa_decode accepts only K = V ∈ {f16, q8_0}, plus a GQA-12 alignment condition for q8_0 (PR ggml-cuda/qsa-decode.cu:92-102).
- **Cache-only draft.** With `--parallel 1` (cache-only draft, PR qwen4exp.cpp:491-497 and :1954-1965) the draft runs no prefill attention: there are 0 draft launches with nq ≥ 128.
- **Main context.** It is identical in every run: qsa_decode nq 2-8 for MTP verify, and qsa_prefill or dense in the shortcut, all q8_0. The `qsa_prefill` counts of the parallel-1 runs omit the fused gated path, which was instrumented only for the par2 runs.
- **Default draft KV.** The draft KV defaults to f16 whatever `-ctk` is: `common_params_speculative_draft::cache_type_k = GGML_TYPE_F16`, PR common/common.h:348/373, wired at common/speculative.cpp:2783. This re-confirms L49.

### Results: draft memory (draft context, `sched_reserve` lines after the MTP context line)
| draft KV | draft KV buffer (attn + idx) | draft compute ROCm0 | draft compute ROCm_Host |
|---|---|---|---|
| f16 (`server_g1_repro_f16d.log`) | 511.50 + 63.94 MiB | 2806.43 MiB | 402.75 MiB |
| q8_0 (`server_g1_repro_q8d.log`) | 271.73 + 33.97 MiB | 2806.70 MiB | 403.01 MiB |
| q4_0 (`server_g1_repro_q4d.log:7134-7519`) | 143.86 + 17.98 MiB | **12568.67 MiB** | **6541.61 MiB** |

The q4_0 draft hits the same masked-route memory cliff as the trunk: +9762 MiB device and +6139 MiB host for a single layer. The q8_0 draft costs +0.27 MiB.

### Results: acceptance (per request; `paired_c2.py`, 14 requests)
- **Run-to-run noise is zero.**
  - g0 vs g1 gave identical draft_n, draft_n_accepted and output text on 14/14 requests, for both f16 and q8_0.
  - `repro_q8d` (HIP graphs on) is identical to g0/g1_repro_q8d on S0–L5, so 12/12 requests match.
  - The uncertainty below is therefore across prompts (paired t, df = 13).
- **Trunk output.** The trunk output text is identical between f16 and q8_0 draft KV on 14/14 requests in both spec modes, as expected for greedy speculative decoding.

| comparison | total accepted / drafted | paired Δ acceptance (pp), mean ± sd | 95% CI | min / max |
|---|---|---|---|---|
| MTP only: q8_0 − f16 draft KV | 2959/4239 = 0.6980 vs 2955/4250 = 0.6953 | +0.29 ± 0.69 | [−0.10, +0.69] | −0.89 / +1.23 |
| repro (mtp + ngram-mod): q8_0 − f16 | 2949/4557 = 0.6471 vs 2950/4553 = 0.6479 | +0.05 ± 0.77 | [−0.39, +0.49] | −1.30 / +1.23 |
| repro: q4_0 (masked route) − f16 | 2910/4672 = 0.6229 vs 0.6479 | **−2.37 ± 2.54** | **[−3.84, −0.91]** | −5.89 / +3.77 |
| par2 (S0/L0/X0): q8_0 vs f16 | 651/911 = 0.7146 vs 650/914 = 0.7112 | (n = 3, not tested) | | |

- Per-request rates for S, L and X are in `res_*.jsonl`; for example, g1_mtp X0 is 0.7458 for f16 and 0.7568 for q8_0.
- There is no depth trend: X0/X1 (27–29k tokens) differ by +1.10 / −0.89 pp, which is within the per-prompt spread.
- **Not separated (UNPROVEN):** the q4_0 loss may come from q4_0 quantization of the draft KV or from the masked route (masked top-k vs complete-block selection). Isolating the cause would need a q4_0 draft on a maskless kernel, which does not exist on the PR.
- The tg t/s values in `res_*.jsonl` are not perf claims: the counter was on, and graphs were off except in `repro_q8d`.

### Decision
1. **The MTP draft is sparse today. Reject the critique/Y "keep the draft dense".**
   - Every draft attention launch past n_kv 2051 ran on `qsa_decode`: 2415–2430 launches per run, and 0 dense launches past 2051 for f16 and q8_0.
   - With a non-cache-only draft (`--parallel 2`) the draft prefill also ran `qsa_prefill` (q8_0) or `qsa_prefill_gated` (f16).
   - Below 2051 the draft takes the dense shortcut, exactly like the trunk.
   - So the new op must cover the draft context (W/X/Z position; L47–L49, L79). Making the draft dense would be the behaviour change.
2. **Guard and reserve must include the draft context and all of its routes.** These are:
   - nq = 1 decode;
   - nq 2–8 (multi-slot);
   - nq ≥ 128 prefill, when the draft is not cache-only, i.e. n_seq_max > 1 or ub < 4096 (qwen4exp.cpp:491);
   - the fused gated prefill path;
   - the dense shortcut.

   No CPU fallback was observed in any run (0 `C2CPU_FA` lines).
3. **Type coverage.**
   - `-ctkd q8_0` works on the PR, on the same maskless route, at +0.27 MiB compute.
   - Its acceptance is indistinguishable from the f16 draft (95% CI [−0.10, +0.69] pp MTP-only), so it passes both W's "≥ −1 pp lower bound" and Z's M5 "CI ∋ 0 or > 0".
   - `-ctkd q4_0` falls to the masked top-k route. The draft compute buffer becomes 12568.67 + 6541.61 MiB (4.5× device), and acceptance drops by 2.37 pp (CI [−3.84, −0.91]).
   - **Therefore the draft needs exactly the trunk's quantized-type coverage.** The draft's probe/init error must use the same type set S as the trunk: the probe at llama-memory-hybrid-idx.cpp:120-129 runs separately for the MTP context and silently picks the masked route.
4. **Acceptance-rate gate.** Use the paired per-prompt design above: greedy decoding, the 14 fixed real-text prompts, f16 draft KV as the reference.
   - Gate: 95% CI lower bound ≥ −1 pp per supported draft type, new vs PR.
   - The run-to-run repeat noise of this harness is 0, so a single run per configuration suffices.
   - The draft KV type defaults to f16 independently of `-ctk` (common.h:373). The rewrite should state this in the PR, and the gate must cover both the default and `-ctkd` = each trunk type.

Large files: none created; `~/dev/wt_lab/c2` totals 52 MB (server logs).

## C3

**Question.** One op with a private plan (X), or a plan op plus an attend op (critique, W, Y, Z)? The plan-cost budgets disagree (ms per layer at ub4096 d131k):
- B ≤ 0.3 (plan-B.md:254)
- W ≤ 1.0 (plan-v2-W.md:153, :263)
- X ≤ 1.0 (plan-v2-X.md:266)
- Y ≤ 1.3, or ≤ 0.5 with bucketing (plan-v2-Y.md:291, :293)
- Z ≤ 1.3 (plan-v2-Z.md:227)

**Answer.** Two ops. Both parts of the LEDGER §C3 test pass.
- **(a) Cost.** I ran the plan stage as standalone kernels (validate, sort, dedup, G=4 merge, optional bucket) on real rows captured from all 12 QSA layers at ub4096, n_kv 135168.
  - With a 512-lane ASC merge it costs **0.733 ms/layer on text and 0.883 ms/layer on random tokens**. With a 256-lane BUCKET merge it costs 0.942 / 1.162.
  - That is 4–5% of the measured in-model attention (17.97 / 24.90 ms per layer).
  - Every output matches a CPU reference bit for bit: 1,056 of 1,056 (layer × capture × variant) checks.
  - So no fusion into attend is needed to meet the W/X (≤ 1.0) or Y/Z (≤ 1.3) budgets. No variant meets B's ≤ 0.3 or Y's bucketed ≤ 0.5.
- **(b) Allocator, splits and HIP graphs.** I tested a single-consumer integer plan tensor through gallocr reuse of its bytes, 2-split and 3-split schedules (in the 3-split case the plan stays live across a CPU split), and HIP graph capture and replay. All 6 configurations gave 20 of 20 bit-identical iterations.

### Setup (strix-002, 2026-11-04 05:25–05:52 CET; every GPU run under `flock ~/gpu.lock`; I was the only GPU user)
- **Where the code lives.** Worktree `~/dev/wt_lab`, detached at 94e086847. The C1/C2 lab diffs are untouched and inert: none of their env variables is set.
- **Build.** The C3 tools are in `~/dev/wt_lab/c3/` and are built outside the CMake tree. They link against the unmodified PR libraries `~/dev/wt_pr219v2/build/bin`, used read-only, or against `~/dev/wt_planA/build/bin` for the in-situ hook. Copies of the sources, scripts and raw outputs are in `SP/final219/c3/` (`c3_capture.cpp`, `c3_plan.cu`, `c3_alloc.cpp`, `scripts/`, `res/`, `alloc/`, `*_insitu.log`). The md5 of each source matches the strix-002 copy.
- **Row capture (`c3_capture.cpp`, `scripts/cap.sh`).**
  - The eval callback asks for every `FLASH_ATTN_EXT` node that has `src[5]` and dumps that I32 `[2051, 4096]` tensor for the ubatches at pos0 32768 (n_kv 36864) and 131072 (n_kv 135168).
  - Text input is the first 135168 wikitext-2 test tokens.
  - Random input uses llama-bench's generator: `srand(1)`, `rand() % n_vocab`, BOS first (tools/llama-bench/llama-bench.cpp:2219-2228).
  - Command: `flock ~/gpu.lock env LLAMA_FN_LAST_ATTN_TRIM=0 ./c3_capture $M <wiki.test.raw|RANDOM> 135168 4096 cap/<corpus>_<type> <f16|q8_0> 32768,131072`
  - Captures: text f16, random f16, and text q8_0 (KV and indexer cache q8_0).
  - Every dump logged `calls=12 n_q=4096 ns=2051 nk=135168|36864 mask=0`, i.e. the maskless selected-key route (`~/dev/wt_lab/c3/cap/*.log`, `C3CAP` lines).
  - In every capture, all 4096/4096 rows had ascending valid cells, with 2049.5 valid cells per query on average.
- **Timing harness (`c3_plan.cu`).**
  - Build: `hipcc --offload-arch=gfx1151 -O3`.
  - Command: `flock ~/gpu.lock ./c3_plan cap/<file>.rows 20 [7]` (`scripts/plan_campaign.sh`).
  - For each layer and variant: 3 warm-up launches, then **n = 20** timed reps (10 for the shuffled runs). A `hipEventRecord` brackets each kernel. After the reps it copies the output back and compares it with a CPU reference built from `std::map` per group (order-invariant, with dedup by construction).
  - Variants:
    - **T0**: the PR's `qsa3_rows_kernel` + `qsa3_merge_kernel`, copied verbatim (PR qsa-prefill.cu:28-240 into `pr_union_kernels.inc`) and launched with the PR's parameters (qsa-prefill.cu:701-718).
    - **P_asc / P_bucket**: T0's rows kernel followed by a new `sa_merge<ASC|BUCKET, lanes>` with 128, 256 or 512 lanes.
    - **P_staged_bucket**: a new `sa_rows` that always writes sorted, deduped per-query rows, i.e. Y's "per-query sorted rows" region (plan-v2-Y.md:121), followed by a BUCKET merge.
  - Duplicate cells collapse in the 4-bit cell mask, so every variant outputs the deduped union.
  - BUCKET order (W :117, Y :119, Z :96): stable by the 4-bit query-presence pattern, quad ascending within a bucket, each bucket padded to 4 entries.
  - **Shuffled control (seed 7).** Each row is randomly permuted, which forces the bitonic sort ("rows in any order", Y :162).
- **In-situ cross-check.** The wt_planA build is the PR plus a `PLANA_STATS` hook that times pack, union (= rows + merge) and attention per qsa_prefill call (`~/dev/wt_planA` `git diff`).
  - Text: `flock ~/gpu.lock env PLANA_STATS=1 GGML_CUDA_DISABLE_GRAPHS=1 LLAMA_FN_LAST_ATTN_TRIM=0 ./c3_capture_planA $M wiki.test.raw 135168 4096 insitu/none <f16|q8_0> -1` (`scripts/text_insitu.sh`).
  - Random: `… wt_planA/build/bin/llama-bench -p 4096 -n 0 -d 131072 -b 4096 -ub 4096 -r 1 -fa 1 -lzm on-direct -ctk f16 -ctv f16` (`scripts/insitu.sh`).
  - Parser: `SP/final219/c3/parse_insitu.py`.

### Results (a): plan-stage ms per layer (mean of the 12 per-layer medians, [min–max layer]; `res/*_r1.out`, parser `summ_plan.py`)
| variant | text f16 d131k | text q8_0 d131k | random f16 d131k | text f16 d32k | random f16 d32k |
|---|---|---|---|---|---|
| T0 (PR, 128 lanes) | 1.187 [0.977–1.336] | 1.192 [0.986–1.374] | 1.488 [1.030–1.938] | 1.165 | 1.249 |
| P_asc 128 | 1.197 | 1.202 | 1.495 | 1.175 | 1.255 |
| P_bucket 128 | 1.245 [1.020–1.410] | 1.248 | 1.566 [1.066–2.062] | 1.223 | 1.302 |
| P_asc 256 | 0.896 [0.725–1.022] | 0.903 | 1.102 [0.758–1.459] | 0.898 | 0.920 |
| P_bucket 256 | 0.942 [0.756–1.080] | 0.948 | 1.162 [0.789–1.551] | 0.945 | 0.963 |
| **P_asc 512** | **0.733 [0.584–0.847]** | **0.739** | **0.883 [0.606–1.169]** | 0.737 | 0.731 |
| P_bucket 512 | 1.058 | 1.062 | 1.279 | 1.045 | 1.043 |
| P_staged_bucket 128 | 1.522 | 1.524 | 1.816 | 1.501 | 1.562 |

**Kernel split.**
- Rows kernel: 0.133–0.160 ms in every non-staged variant. It writes nothing for rows that are already sorted.
- Staged rows: 0.36 ms.
- Merge on text d131k: 1.054 ms (128 lanes), 0.755 (256), 0.591 (512).
- Bucketing adds 0.046–0.070 ms with 128/256 lanes, but 0.33–0.40 ms with 512 lanes. The likely cause is occupancy from the 16 × (lanes + 2) u16 counter array in LDS (UNPROVEN, not profiled).

**Union size** (quads per group of 4 queries, padded).
- Text d131k: 947.1 ASC, 968.1 BUCKET (+2.2%).
- Random d131k: 1408.5 ASC, 1429.5 BUCKET (+1.5%).
- Largest bucketed group: 1940 (random d131k). This is within W's cap, round4(min(4·513, n_kv/4)) + 4·15 = 2112 (plan-v2-W.md:119).

**Shuffled rows** (`res/*_shuf7.out`, d131k).
- The rows kernel takes 2.665–2.671 ms (bitonic).
- Totals per layer, all bit-exact: text 3.29 (P_asc 512) to 3.83 (P_bucket 128); random 3.42 to 4.12.
- P_staged_bucket: 5.51 / 5.81.

**In-situ, inside the model** (PR kernels, graphs off, TRIM=0, n_kv 135168, 12 calls):
- text f16: union 1.409 ms (sd 0.109, 1.207–1.563); attention 17.97 ms; union/attention 7.8%.
- text q8_0: union 1.384 ms (sd 0.106); attention 18.98 ms; 7.3%.
- random f16: union 1.463 ms (sd 0.126); attention 24.90 ms; 5.9%. llama-bench's measured ubatch restarts its token stream, so these rows are not the captured random rows.
- The in-situ text union size, `g4_mean_blk` 947.6, equals the harness's 947.1, so these are the same rows.
- In-situ T0 runs 0.22 ms above the harness T0 on the same rows (1.409 vs 1.187). The cause is UNPROVEN; candidates are a cache that is cold after the attention kernel, and the hook's pool allocations between its events.
- This **verifies plan A M2's "1.3 ms/layer today"**: in situ the union costs 1.38–1.41 ms at d131k on text, and 0.95 ms at d4k (`text_f16_insitu.log`).

### Noise
- **Within a run:** the per-layer rep sd is 0.004–0.012 ms, i.e. ≤ 1%.
- **Run to run:** r1 vs r2 are separate processes on the same rows (d131k, all three captures, `res/*_r2.out`). Every variant agrees within 0.009 ms (≤ 0.8%). Example, T0 on text f16: 1.187 vs 1.196.
- **Layer to layer:** this is the dominant spread, e.g. 0.606–1.169 ms for P_asc 512 on random. So the budget check below uses both the mean and the worst layer.

### Budget check (mean over layers, worst layer in brackets)
| budget | met by |
|---|---|
| B ≤ 0.3 | **none.** The rows pass alone reads 33.6 MB of rows (4096 × 2051 × 4 B) in 0.13–0.16 ms, and the merge reads them again, so a two-pass plan cannot reach 0.3 |
| W/X ≤ 1.0 | text: P_asc 256 (0.896 [1.022]), P_bucket 256 (0.942 [1.080]), P_asc 512 (0.733 [0.847]). Random: **P_asc 512 only** (0.883 [1.169]). T0 fails on both |
| Y/Z ≤ 1.3 | every 256/512 variant on both corpora. T0 fails on random (1.488) |
| Y bucketed ≤ 0.5 | **none** (best: 0.942 text, 1.162 random) |

### Results (b): single-consumer plan tensor through gallocr, splits and HIP graphs (`c3_alloc.cpp`, `alloc/*.out`)
- **Surrogate graph.** No plan op exists yet, so existing ops stand in for it:
  - plan = `GGML_OP_ARGSORT` (I32 `[1024, 256]`, 1 MiB), whose sole consumer, attend, is `GGML_OP_GET_ROWS`;
  - 6 pressure tensors are live while the plan is live;
  - 12 plan-sized "late" tensors are all alive at once after attend;
  - all data are small integers, so every op is exact.
- **Backends.** ROCm0 and CPU in `ggml_backend_sched`. A `scale` op is pinned to CPU with `ggml_backend_sched_set_tensor_backend`.
- **Topologies.** A has 2 splits: CPU, then ROCm with plan, pressure, attend and tail. B has 3 splits: ROCm with the plan, then CPU, then ROCm with attend, so the plan stays live across the CPU split.
- **Modes.** "reuse" keeps one cgraph and recomputes it. "rebuild" makes a new context and resets and re-allocates the scheduler every iteration.
- **Checks.** 20 iterations with fresh random inputs each time. Both outputs are memcmp'd against a host reference.
- **Command:** `flock ~/gpu.lock env GGML_CUDA_GRAPH_DEBUG=1 [GGML_CUDA_DISABLE_GRAPHS=1] ./c3_alloc A|B reuse|rebuild 20`
- **Capture count.** Graph captures are counted from `CUDA graph warmup complete` (ggml-cuda.cu:7486). No `warmup reset` occurred.

| topo | mode | HIP graphs | splits | tensors placed in the plan's freed bytes | captures | memcmp equal |
|---|---|---|---|---|---|---|
| A | reuse | on | 2 | late1, late2 | 1 (at iteration 1; iterations 2–19 replay it) | 20/20 |
| A | rebuild | on | 2 | late1, late2 | 4 | 20/20 |
| B | reuse | on | 3 | late1 | 2 (one per ROCm split) | 20/20 |
| B | rebuild | on | 3 | late1 | 8 | 20/20 |
| A | reuse | off | 2 | late1, late2 | 0 | 20/20 |
| B | reuse | off | 3 | late1 | 0 | 20/20 |

**What (b) does not cover:**
- **A written input**, i.e. attend writing into a workspace region of the plan (W plan-v2-W.md:107, Z plan-v2-Z.md:84). No ggml op writes its source outside a view, so the surrogate cannot express one. It remains **UNPROVEN**.
- **This tree's OVL side-stream regions.** They trigger only on named prefill GEMMs (ggml-cuda.cu:6930-6962), and the test graph has none.
  - From the code: the kind-4 region runs from `Qcur_full`+1 to the `FLASH_ATTN_EXT` node (:6958-6962), and every tensor it touches stays allocated through that end node (`add_alloc_dep`, :7024-7033).
  - A plan node placed before attend would fall inside that region. A region is rejected if a node reads both sides (:6973).
  - Runtime behaviour is **UNPROVEN** until the real op exists. The real-model test of the rewrite must run with OVL on (default mask 15, :6907) and with graphs on.

### Code verification (supports the decision's conditions)
- **ggml-cuda already over-allocates the FA dst for private scratch.** `ggml_backend_cuda_buffer_type_get_alloc_size` calls `ggml_cuda_flash_attn_ext_get_alloc_size` for `FLASH_ATTN_EXT` (ggml-cuda.cu:1020-1024 → fattn.cu:751-791, used for the f16 K/V copies). This is X's workspace mechanism. Vulkan's `get_alloc_size` returns `ggml_nbytes` (ggml-vulkan.cpp:17125-17128), so Vulkan needs the same hook.
- **gallocr computes only the ops in `ggml_op_can_inplace` in place over a parent** (ggml-alloc.c:22-50: elementwise, norm, rope, softmax). FLASH_ATTN_EXT, and any new op that is not added to that list, never overwrites its plan source.
- **Plan tensor size at ub4096** (G = 4, ns = 2051) [arithmetic]:
  - union region: W cap of 2112 entries × 6 B × 1024 groups = 13.0 MB;
  - sorted-row scratch: 4096 × 2051 × 4 B = 33.6 MB;
  - total about 46.6 MB, with one plan live at a time (one consumer; (b) shows the bytes are reused afterwards).
  - Today's in-op pool use, with cap 8204 (qsa-prefill.cu:702), is 50.4 + 33.6 = 84 MB.

### Decision (LEDGER §C3 rule: if the plan cost is under budget and (b) passes, two ops)
1. **Freeze two ops: `ggml_sparse_attn_plan` (integer, bit-exact vs CPU) and `ggml_sparse_attn` (attend).**
   - As standalone kernels, the plan meets ≤ 1.0 ms/layer on both text and random rows (P_asc 512: 0.733 / 0.883) and costs 4–5% of attention.
   - Fusing the plan into attend is not needed for cost, so the one-op fallback (fuse, plus a debug plan dump) is not triggered.
   - The plan kernels are identical in either structure, so the measured cost does not favour X's single op.
   - X's remaining arguments (the layout frozen in ggml.h, the G hint plumbing) are design arguments, not measurements. The W/Z superset layout answers the freeze argument by making G and order plan parameters (L77).
2. **No written input.** Put attend's split-K workspace in attend's own dst via `get_alloc_size` over-allocation (verified on HIP; needs a Vulkan hook), or in Y's fixed buffer, not inside the plan as W and Z propose. (b) proves the read-only, single-consumer plan; a written input stays unproven.
3. **Plan-cost gate:** ≤ 1.0 ms/layer, as the mean over the 12 layers at ub4096 d131k, on text and on random rows, timed with hipEvents at n ≥ 10. `c3_plan.cu` is the reference harness, and the CPU reference memcmp is part of the gate.
   - Drop B's ≤ 0.3 and Y's bucketed ≤ 0.5: no variant met them, and the rows pass alone takes 0.13–0.16 ms.
   - HIP default: the 512-lane ASC merge.
   - If BUCKET is adopted by the bucketing claim, its plan costs 0.942 (text) and 1.162 (random) with 256 lanes. That passes ≤ 1.3 but not ≤ 1.0 on random, so the bucketing decision must charge the extra +0.21 (text) / +0.28 (random) ms/layer over P_asc 512.
4. **Row order.** The plan must accept rows in any order; the shuffled runs are bit-exact. But unsorted rows cost 3.3–4.1 ms/layer, because the bitonic sort alone takes 2.67 ms.
   - The complete-block producer already emits ascending rows (4096/4096 in every capture), so it pays nothing.
   - Any score-ordered producer (token-expanded top_k for image or non-scalar ubatches) pays about +2.5 ms/layer unless it emits sorted rows.
5. **Cap bound.** Per group, cap = round4(min(4·n_sel_quads, ceil(n_kv/4))) + 4·15 = 2112, against a measured maximum of 1940.

### Incident and large files
- **Failed first in-situ attempt.** I first ran it with `llama-perplexity -c 135168`.
  - The f16 run was SIGKILLed (`EXIT text f16 137`) at nk 102400.
  - The q8_0 run that followed failed at model load: `cudaMalloc … 78056.46 MiB … out of memory` (`~/dev/wt_lab/c3/insitu/campaign.out`; `insitu/text_q8_0.log` was overwritten by the later successful run).
  - Cause, inferred and UNPROVEN (I could not read the kernel log): llama-perplexity keeps logits for every token of the chunk, n_ctx × n_vocab floats.
  - Both processes were mine. `ps` showed no other process affected.
  - I replaced it with the `c3_capture` driver, which keeps last-token logits only.
- **Large files.** The 6 row dumps (403 MB each, 2.3 GB total) are deleted. `~/dev/wt_lab/c3` now holds 3.6 MB. To recapture, use `scripts/cap.sh`.

## C4

**Question.** Selection rule for images and non-scalar ubatches (Track P7). On real image prompts, does the token-expanded rule (TE) differ from complete-block (CB)? Is complete-block the reference?
**Answer.** **They differ only by the 3 − tail "fill" cells, and the difference is not image-specific.** The width is 2051 = 512·4 + 3. CB takes 512 whole blocks plus the query's incomplete tail (0–3 cells). TE fills the remaining 3 − tail slots with cells from the 513th-best block.
- In every image row, CB ⊆ TE.
- Neither rule ever drops the query's own cell on the image ubatch.
- The extra cells carry about 1e-4 of the attention mass.
- End to end, the pinned KLD between the two rules is below the noise from changing the ubatch size.
- **Complete-block is the reference.** The in-graph CB construction reproduces the offline reference rule on 346/346 image rows. The code names CB as the reference (L22).
- **Decision: one rule everywhere (X/Y), complete-block, including image and non-scalar ubatches.**

### Setup (strix-002, `~/dev/wt_lab/c4`; copies in `SP/final219/c4/`)
- **Binary.** `~/dev/wt_lab/build4/bin/lab-c4`, built from wt_lab at 94e086847 (PR head) with the default-off lab patches.
  - The C4 patch is `c4/c4_only.diff`, applied by `c4/patch_c4.py`. `LAB_QSA_C4=1` builds the CB rows next to the TE rows on the masked route and dumps both, while TE still feeds attention. `=2` feeds the CB rows to the masked FA instead (c4_only.diff, hunk 1 comment and hunk 3).
  - The patch touches only the `blk_bias` (masked, non-compact) branch, `if (lab_qsa_c4() && blk_bias)`. The compact text ubatches and the scoring decode are therefore complete-block in every mode, and the mode switch changes **only the image ubatch's rule**.
- **Run command.** `c4/run.sh`:
  `flock ~/gpu.lock env LAB_QSA_C4=<0|1|2> … lab-c4 -m Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf --mmproj mmproj-Signal-3.8-Flash-Next-f16.gguf --image img_nyt.jpeg -c 40960 -b UB -ub UB -ngl 99 -fa on -ctk f16 -ctv f16 --image-min-tokens 1024 -n 160`
- **Prompts** (`c4/mkprompt.py`): chat prompt = wikitext + image + wikitext + an image-understanding question (headline, newspaper, date, one-sentence photo description).
  - **p2:** 28930 text tokens, then a 1036-token image, then 1328 text tokens (`logs/p2_c4dump.log:730-747`).
  - **p1:** 5427, then 1036, then 1993 (`logs/p1_c4dump.log:505-522`).
  - **Deviation from the ledger experiment.** Request B was specified as 1 image + 60k. p2 reaches about 30k (n_kv 31488), so no run past 32k is measured.
- **Dumps** (`p2.sh`, last line): `LAB_QSA_C4=1`, layers 3, 23 and 43, for the image ubatch (img1) and the first text ubatch after it (txt2).
  - Dumped tensors: `lab_te_rows`, `lab_cb_rows`, `lab_blk_score`, `lab_blk_cells`, `indexer_top_k` and `indexer_score_tokens`, plus the FA q/k/v/mask/out.
- **Offline analysis.** `c4/c4_analyze.py DUMP LOG 3`. It samples every 3rd query row, which gives 346 image rows and 443 txt2 rows.
  - **REF** = consecutive groups of 4 causally visible cells, the top-512 groups by the dumped block score, plus the incomplete tail.
  - **TE** = the dumped `lab_te_rows` on img1. On txt2, which runs the compact route, TE is rebuilt offline from the block scores; ties inside the 513th block take its lowest cells (script comment).
  - **Mass and output.** Mass is the dense softmax over all visible cells, averaged over the 24 heads. The output comparison is the renormalised attention output over each set.

### Soundness checks (logs/an_p2_s3.txt, md5 8e949d61373b6cc6ff2427fb165c168f)
- **The offline REF equals the in-graph CB rows.**
  - img1: 346/346 at all three layers (lines 6, 17, 28).
  - txt2: 442/443, 442/443 and 443/443 (lines 37, 45, 53).
  - The single txt2 mismatch at L3 and at L23 lines up with the single row where REF − TE > 0 (lines 38, 46). That these are the same row, and that the cause is a tie or rounding in the offline top-512, is UNPROVEN: the script does not print row ids.
- **The image ubatch's mask is plain token-order causal** in 346/346 rows, and the FA mask set equals the TE rows in 346/346 (lines 3-4, 14-15, 25-26).
  - The ubatch is non-scalar only because of its 2-D positions (L20), not because of non-causal visibility.
  - The budget binds in every sampled row (visible > 2051), so no TE row contains an invisible or −1 cell (line 4: "TE rows all visible 346/346").
- **Offline recompute floor.** Offline o_TE vs the dumped FA output gives rel-L2 means of 9.16e-5, 6.57e-5 and 4.85e-5, with max ≤ 1.29e-4 (lines 11, 22, 33). This floor is the noise for the output-diff numbers below.
- **The earlier stride-4 run is invalid.** `logs/an_p2.txt` shows a perfect 1.000000 because every sampled row had a full 3-cell tail (forced histogram {3: 259}), so it had no fill cells.
  - With stride 3 the tail sizes cycle: {0: 86, 1: 86, 2: 87, 3: 87} (line 3).
  - Mean TE − REF = (86·3 + 86·2 + 87·1)/346 = 1.494, matching line 7 exactly. Rows > 0 = 86 + 86 + 87 = 259, also matching.
- **Run completion.** The first stride-3 process (PID 3227584) died before completing. Its output stopped after the txt2 layer 3 block (41 lines, no timing line), and is kept as `logs/an_p2_s3.partial_dead.txt`.
  - I reran the same command once, on CPU only, as PID 3232719. It completed with `real 27m14.213s` (`logs/an_p2_s3.time`).
  - The rerun's first 41 lines are identical to the partial output (diff empty), so the analysis is deterministic.

### Results, p2 (logs/an_p2_s3.txt)
| ubatch / layer | TE−REF mean (max, rows>0) | REF−TE mean (max) | Jaccard mean (min) | own cell kept TE / REF | TE-only mass mean (max) | out rel-L2 mean (max) | lines |
|---|---|---|---|---|---|---|---|
| img1 L3 | 1.494 (3, 259/346) | 0 (0) | 0.999271 (0.998537) | 346/346 / 346/346 | 8.05e-5 (6.94e-4) | 2.64e-4 (3.02e-3) | 1-11 |
| img1 L23 | 1.494 (3, 259/346) | 0 (0) | 0.999271 (0.998537) | 346/346 / 346/346 | 6.23e-5 (1.05e-3) | 6.87e-5 (1.51e-3) | 12-22 |
| img1 L43 | 1.494 (3, 259/346) | 0 (0) | 0.999271 (0.998537) | 346/346 / 346/346 | 7.05e-5 (2.22e-3) | 7.29e-5 (1.96e-3) | 23-33 |
| txt2 L3 | 1.501 (4, 332/443) | 0.005 (2) | 0.999266 (0.997077) | 443/443 / 443/443 | 9.35e-5 (1.87e-3) | 2.36e-4 (4.01e-3) | 34-41 |
| txt2 L23 | 1.506 (4, 333/443) | 0.009 (4) | 0.999262 (0.996107) | 443/443 / 443/443 | 1.03e-4 (1.53e-3) | 1.07e-4 (1.26e-3) | 42-49 |
| txt2 L43 | 1.497 (3, 332/443) | 0 (0) | 0.999270 (0.998537) | 442/443 / 442/443 | 9.73e-5 (3.18e-3) | 1.02e-4 (4.10e-3) | 50-57 |
- **p1** (n_kv 6656, every row, `logs/an_p1.txt`) has the same structure: img1 TE − REF mean 1.500, max 3, REF − TE 0, own cell 1036/1036 for both rules (lines 7-8, 18-19, 29-30). txt2 gives 1.501 (lines 38, 46, 54). So the difference does not depend on depth.
- **Rows missing their own cell.** The count is 0 on every image row under both rules. On txt2 L43 it is 1/443 under both rules (p1: 5/1993), and it is the same count for TE and REF.
  - The script reports only counts. That it is the same row for both rules is inferred, not shown: when the tail is empty, the own cell sits in the last whole block, and a rule can drop it only if that block loses on score.
- **Size of the difference.** The mean output difference (6.9e-5 to 2.6e-4) is 1–3× the recompute floor (4.8e-5 to 9.2e-5). The max (up to 4.1e-3) is about 10–30× the floor. The fill cells hold at most 3.2e-3 of the softmax mass.

### End to end: pinned answer, image ubatch rule only (logs/kld_p2.txt, logs/kld_p1.txt, logs/p2_campaign.out)
- **Fixed answer.** The pinned answer is TE's own greedy output (`p2.sh` writes answer_p2.txt from p2_te_gen). TE's log-prob is therefore favoured by construction.
- **p2** (97 tokens), te_s1 vs cb_s1: KLD mean 9.397e-4, max 1.932e-2, median 1.907e-5, same top1 100%, log-prob −14.2705 → −14.3729 (−0.10).
  - Noise, same rule, ub 8192 → 4096: te 2.348e-3 (max 3.691e-2, top1 97.9%); cb 1.103e-3 (max 1.123e-2).
  - Repeats (s1 vs s2) are bit-identical (0).
  - te vs te1 (mode 1, CB built but not used) gives 0, so the instrumentation does not perturb the run.
  - LAB_SCORE argmax_match is 95/97 in every p2 mode.
- **p1** (92 tokens), te vs cb: KLD mean 1.266e-3, max 1.749e-2, top1 98.9%, −0.25. ub noise: te 1.680e-3, cb 1.532e-3.
- **The rule difference is below the ubatch-size noise in both prompts.**
- **Free generation.** Both rules answer correctly on both prompts: "MEN WALK ON MOON", The New York Times, Monday, July 21, 1969, astronaut on the lunar surface. The CB answers differ only in the wording of the photo sentence (`logs/p{1,2}_cb_gen.log` vs `answer_p{1,2}.txt`).

### Decision
1. **Complete-block is the reference.**
   - Code comments say so: "the reference returns indexer_top_k + compress_ratio - 1: whole blocks plus the tail" (L22). The C4 patch comment attributes the rule to transformers `Qwen4ExpTextQSAIndexer`. I did not read the transformers source, so that attribution is UNPROVEN beyond the comment.
   - Measured: the CB construction equals the offline reference on 346/346 image rows.
   - TE's only deviation is the fill cells, which are not part of the reference rule.
2. **One rule everywhere (X/Y): complete-block for image and non-scalar ubatches too.**
   - The measured fidelity change from switching the image ubatch to CB is below noise: KLD ≤ ub-size noise, and Δmass about 1e-4. This removes the "model fidelity, out of scope" objection W/Z raise (L80).
   - It also drops the score-ordered producer, which costs about +2.5 ms/layer unless it emits sorted rows (C3 decision 4).
   - Scope limits, all UNPROVEN because C4 did not exercise them:
     - rows with ≤ 2051 visible cells on a non-scalar ubatch, where −1 entries appear;
     - non-causal or 2-D-dependent visibility; on this model the image mask was causal in 346/346 rows;
     - several images or sequences;
     - depth beyond 31.5k.
3. **Image test set.** Adopt p2 and p1 with img_nyt.jpeg (`c4/mkprompt.py`, `run.sh`, `p2.sh`) as the image-understanding gate.
   - Gate: pinned-answer KLD of new vs current, compared against the same-rule ub 8192/4096 noise (mean about 1.1e-3 to 2.3e-3), plus an answer content check.
   - Add L64's request C (5 images + 150k) to cover multiple images and depth. It is not measured here.
4. **strip_views.** In this PR it serves only the masked-route (TE) views of `bias_all` and `kq_mask` (L6; `wt_lab src/models/qwen4exp.cpp:1612-1625`).
   - The C4 CB-on-image prototype still reads per-row visibility from the `kq_mask` strip view: `vis = step(query_mask + 1)` in c4_only.diff hunk 3.
   - So whether strip_views stays load-bearing depends on how the generalized CB gets non-scalar visibility. If it gets it from the mask, the kq_mask view stays. If it gets it from cell positions, the view can go. Neither option was measured: UNPROVEN.

### Large files
- `strix-002:~/dev/wt_lab/c4/dump/` (p1 824 MB + p2 1.6 GB) and `~/dev/wt_lab/c4/logits/` (1.3 GB) are **not deleted**. My `rm -rf` was refused by the permission classifier, so the owner needs to remove them or allow the removal.
- The small summaries are copied to `SP/final219/c4/`: an_p2_s3.txt, an_p2_s3.time, an_p2_s3.partial_dead.txt, an_p2.txt, p2_campaign.out, plus the earlier an_p1*, kld_p*, answers and scripts.

## C5

**Question.** Union/bucketing payoff on real text vs random tokens (Track P2), all 12 QSA layers. What are the per-query union for G ∈ {1,2,4,8}, the tile-chunks per query in ascending (ASC) vs bucketed (BUCKET) order, and the permutation rate? How do G4-ASC, G4-BUCKET and G8 prototypes time on replayed rows? Decides: whether bucketing enters the plan layout (W/X/Z: ≥ 10 % at d32k on text; Y: ≥ 15 %), the surviving G set, and the bench weighting (LEDGER §C5).

**Answer.**
- **Bucketing does not enter the plan layout.** A correct G4-BUCKET prototype saves **9.0–9.3 %** of attend time at d32k on text (wiki, code, chat) against the same kernel in ASC order, or 10.3–12.8 % against the PR kernel. Once the +0.21 ms/layer bucketed plan cost measured in C3 is charged, the saving falls to **7.3–7.8 %**.
- **It loses on random tokens at depth.** At d131k bucketing is **+10.1 %** slower on continuous random tokens and **+13.4 %** on llama-bench tokens (worst layers +33.8 % / +38.9 %). On wiki d131k its worst layer is +3.9 %.
- So it fails Y (15 %), X ("≤ 3 % loss elsewhere"), Z ("no worse on rand") and, against the like-for-like baseline, W (10 %).
- **G set: G4.** G8 is 1.20–1.49× slower than G4 in every cell. G2 was not timed. G1 is kept for small n_q, per C1.
- **Bench weighting:** decide on real text only. llama-bench random tokens inflate the G4 union by +17 % (d32k) to +27 % (d131k) against wikitext. They inflate the attend time by +19 % (d32k) and +44 % (d131k), and they reverse the sign of the bucketing result at d131k.

### Setup (strix-002, 2026-11-04 08:19–09:27 CET; every GPU run under `flock ~/gpu.lock`; I was the only GPU user)
- **Where the code lives.** Worktree `~/dev/wt_lab`, detached at 94e086847; the C1/C2/C4 lab diffs are untouched, and none of their env switches is set. C5 tools are in `~/dev/wt_lab/c5/`, built outside CMake and linked against the unmodified PR libraries `~/dev/wt_pr219v2/build/bin`, used read-only. Copies of every source, script, log and result are in `SP/final219/c5/remote/`. md5 of local vs strix-002 copies match: `c5_attn.cu` fa0242c2…, `c5_variant.inc` ba8a3ee2…, `c5_union.cpp` 301e54bb…, `c5_capture.cpp` 16083bf4….
- **Corpora** (`mkcorpus.py`):
  - wiki: `~/data/wikitext-2-raw/wiki.test.raw`;
  - code: the PR tree's `src/` then `ggml/src/` (*.c, *.cpp, *.h, *.cu, *.cuh, sorted), 3.09 MB;
  - chat: the ho3b holdout chat items (oasst2/Dolly) in `sources.jsonl` order, rendered as `<|im_start|>role\n…<|im_end|>\n` and tokenized with `parse_special=1`. The first token is 248045, so the tags were parsed as special tokens (`logs/cap_chat_f16.log`);
  - randc: llama-bench's generator as one continuous stream (`srand(1)`, `rand() % n_vocab`);
  - bench: exactly what `llama-bench -d D -p 4096` feeds. `test_prompt` calls `srand(1)` on every call (tools/llama-bench/llama-bench.cpp:2218, :2509, :2536), so the measured ubatch repeats the first 4096 depth tokens: tokens = R[0:D] ++ R[0:4096].
- **Row capture** (`c5_capture.cpp`, an extension of C3's `c3_capture.cpp`; script `cap.sh`).
  - Command: `flock ~/gpu.lock env LLAMA_FN_LAST_ATTN_TRIM=0 ./c5_capture $M <corpus|RANDOM|BENCH> <135168 | D+4096> 4096 cap/<tag> <f16|q8_0> 8192,32768,65536,131072 <special> <qkv_pos0>`, with M = `~/models/qwen3.8-flash-next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`, ub = b = 4096, flash attention on, KV f16 (plus wiki q8_0 at 32k/131k).
  - The eval callback dumps FLASH_ATTN_EXT `src[5]` (I32 [2051, 4096]) for the ubatch at pos0 = depth (n_kv = depth + 4096), plus the Q/K/V bytes for wiki d32k (f16 and q8_0).
  - The trim is off, so all 12 layers carry 4096 queries. Every one of the 22 captured ubatches logged 12 calls with `n_q=4096 ns=2051 … mask=0` (the maskless selected-key route); no `mask=1` line appears (`logs/cap_*.log`).
- **Offline statistics** (`c5_union.cpp`, CPU only; `./c5_union cap/X.rows tag`, tables by `summ_union.py`).
  - Per query: valid cells, distinct physical quads (cell>>2), qpb = quads/(cells/4), adjacent-query overlap.
  - Per group of G consecutive queries: union of quads, ASC chunks of 4 quads, and WMMA 16-row tiles for today's packing (12 heads per query, ceil(12G/16) tiles per chunk).
  - BUCKET: quads stably sorted by G-bit query-presence pattern, each bucket padded to 4. b12 = tiles touched by the pattern in the 12-row packing; b16 = one tile per present query (plan-B.md:10 "tS/q").
- **Attend prototypes** (`c5_attn.cu` + `c5_variant.inc`; `hipcc --offload-arch=gfx1151 -O3 -std=c++17`; `attn_campaign.sh`).
  - Command per cell: `flock ~/gpu.lock ./c5_attn cap/<rows> <f16|q8_0> 20 <cap/X.qkv|SYN> all 0,1,2,3,4,5,6,7,13,14`.
  - Per layer: 3 warm-up rounds, then n = 20 rounds. Each round launches every variant once (interleaved), timed by a hipEvent pair.
  - Reported: the mean over the 12 layers of the per-layer medians.
  - Data: K direct + V packed, the in-situ route at ub4096 (PR qsa-prefill.cu:684-686, n_q·64 > n_kv). The f16 rows of each cell are used. Real Q/K/V for wiki d32k f16 and q8_0; otherwise SYN (N(0,1) Q/K/V, q8_0 quantized on the host).
  - **V0** = the PR's `qsa3_attn_kernel<true,8,false,Q8>`, verbatim (pr_attn.inc = PR qsa-prefill.cu:10-26, 241-601, 630-670), on the plan built by the PR's own verbatim rows+merge kernels (pr_union_kernels.inc, identical to C3's; diff against PR :28-240 is empty).
  - **V1–V7** = `c5_attn_kernel<NH, RPQ, SKIP, Q8>`, derived line by line from V0, on CPU-built plans:

| id | name | plan | kernel |
|---|---|---|---|
| V1 | c5-G4-ASC | G4 ASC | NH1, RPQ12, no skip (V0's arithmetic) |
| V2 | G4-ASC-skip | G4 ASC | per-chunk scalar-branch skip of inactive row tiles (QK/PV WMMA, LDS partials, softmax) |
| V3 | PR kernel, BUCKET order | G4 BUCKET (+4·15 cap) | V0 kernel unchanged |
| **V4** | **G4-BUCKET** | G4 BUCKET | V2's kernel: the bucket order makes the skip effective |
| V5 | G4-BUCKET-r16 | G4 BUCKET | one 16-row tile per query (B's layout), 235 VGPRs, occupancy 6 |
| V6 | G8-ASC | G8 ASC (2 u16 masks per group) | 512 threads: two G4 halves walk one G8 union |
| V7 | G8-ASC-skip | G8 ASC | V6 plus the scalar-branch skip |
| V13/V14 | ORACLE | G4 ASC | fixed 2 or 1 active tiles per chunk: wrong output, a timing bound for zero-overhead skipping |

- **Resource usage** (`logs/build_attn.log`, `-Rpass-analysis=kernel-resource-usage`):
  - V0, V1, V2/V4, V7: 192 VGPRs, scratch 0, occupancy 8.
  - V5: 235 VGPRs, scratch 0, occupancy 6.
  - V6: 192 VGPRs, **36 B/lane scratch**.
- **Variants that failed to compile cleanly.** Three tile-set-specialised designs spilled at the 192-VGPR cap: a switch over constexpr tile sets (328–364 B), a full body plus a branch body (232 B), and two sequential loops (48–68 B). One quick test of those variants on layer 0 was 19.7–33.6 ms against 17.3 for V0 (`res/quick2_L0.out`), so they were dropped.
- **An implementation trap, found in the ISA** (`--save-temps`, kernel sequence dump). An early V2/V4 also made the V gather of zero-mask quads conditional. That puts `s_waitcnt vmcnt(0)` after every V load and made V2 36 % slower than V0 (`res/quick_L0.out`: 23.95 vs 17.55 ms; V4 23.49). The final kernels gather V unconditionally, as the PR does (`res/quick3_L0.out`).

### Correctness of the harness (`res/attn_*.out`, `res/attn_summary.txt`)
- **Plan:** the PR merge output equals the CPU G4-ASC plan bit for bit in 16/16 files × 12 layers (0 of 1024 groups differ each time).
- **V0 against a CPU f64 reference** (8 queries × 24 heads per layer): max |err|/max|ref| per head 1.3e-4 (real data) to 3.9e-4 (SYN).
- **V1, V2 against V0:** bit-identical in every layer and cell.
- **Reordered variants against V0** (fp order only): V3/V4/V5 max_abs ≤ 1.1e-3, rel_l2 ≤ 2.2e-4; V6/V7 ≤ 5.2e-4, ≤ 2.1e-5. No NaN anywhere.
- **Fidelity to the model.** Harness V0 is within 6.2 % of the C3 in-situ attention at d131k: 16.85 vs 17.97 ms on wiki, and 24.22 vs 24.90 ms on llama-bench random.

### Results (a): union statistics, 12-layer means (`res/union_summary.md`; per-layer rows in `res/union_*.txt`)
| corpus | pos0 (n_kv) | adj ovl | U/q G2 | U/q G4 [layer min–max] | U/q G8 | G4 ch/q ASC | G4 ch/q BKT | G4 tiles/q ASC | G4 tiles/q BKT (12-row) | BKT/ASC tiles | G8 ch/q | G8 tiles/q ASC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wiki | 8192 (12288) | 0.771 | 1.23 | 1.51 [1.43–1.68] | 1.85 | 48.4 | 49.7 | 145.1 | 121.4 | 0.837 | 29.7 | 178.4 |
| wiki | 32768 (36864) | 0.660 | 1.34 | 1.81 [1.69–2.01] | 2.46 | 58.1 | 59.4 | 174.2 | 132.2 | 0.759 | 39.5 | 236.9 |
| wiki | 65536 (69632) | 0.666 | 1.33 | 1.80 [1.69–2.06] | 2.46 | 57.8 | 59.1 | 173.5 | 131.7 | 0.759 | 39.5 | 237.0 |
| wiki | 131072 (135168) | 0.653 | 1.35 | 1.84 [1.66–2.19] | 2.57 | 59.2 | 60.5 | 177.5 | 132.8 | 0.748 | 41.2 | 247.0 |
| code | 8192 | 0.735 | 1.26 | 1.58 [1.51–1.68] | 1.96 | 50.8 | 52.1 | 152.3 | 124.9 | 0.820 | 31.5 | 189.0 |
| code | 32768 | 0.587 | 1.41 | 2.00 [1.71–2.25] | 2.81 | 64.3 | 65.6 | 192.8 | 139.3 | 0.723 | 45.1 | 270.6 |
| code | 65536 | 0.715 | 1.29 | 1.65 [1.53–1.75] | 2.14 | 52.9 | 54.2 | 158.6 | 127.0 | 0.801 | 34.3 | 205.9 |
| code | 131072 | 0.563 | 1.44 | 2.08 [1.88–2.26] | 3.03 | 66.8 | 68.1 | 200.5 | 141.6 | 0.706 | 48.6 | 291.5 |
| chat | 8192 | 0.748 | 1.25 | 1.55 [1.42–1.66] | 1.92 | 49.8 | 51.1 | 149.5 | 123.7 | 0.827 | 30.9 | 185.1 |
| chat | 32768 | 0.667 | 1.33 | 1.77 [1.53–1.95] | 2.37 | 56.9 | 58.2 | 170.6 | 131.5 | 0.771 | 38.0 | 228.3 |
| chat | 65536 | 0.659 | 1.34 | 1.81 [1.54–2.09] | 2.47 | 58.0 | 59.3 | 174.0 | 132.4 | 0.761 | 39.6 | 237.6 |
| chat | 131072 | 0.631 | 1.37 | 1.89 [1.58–2.26] | 2.63 | 60.5 | 61.9 | 181.6 | 135.0 | 0.743 | 42.3 | 253.6 |
| randc | 8192 | 0.536 | 1.46 | 2.06 [1.48–2.38] | 2.75 | 66.0 | 67.3 | 198.1 | 144.2 | 0.728 | 44.2 | 265.1 |
| randc | 32768 | 0.412 | 1.59 | 2.46 [1.68–2.98] | 3.73 | 79.1 | 80.4 | 237.2 | 156.0 | 0.658 | 59.8 | 358.8 |
| randc | 65536 | 0.363 | 1.64 | 2.63 [1.79–3.17] | 4.16 | 84.5 | 85.8 | 253.5 | 160.7 | 0.634 | 66.6 | 399.8 |
| randc | 131072 | 0.331 | 1.67 | 2.74 [1.92–3.34] | 4.44 | 88.0 | 89.3 | 264.1 | 163.9 | 0.621 | 71.2 | 427.1 |
| bench | 8192 | 0.659 | 1.34 | 1.82 [1.62–2.26] | 2.46 | 58.5 | 59.8 | 175.6 | 132.3 | 0.753 | 39.5 | 237.0 |
| bench | 32768 | 0.564 | 1.44 | 2.12 [1.80–2.78] | 3.15 | 67.9 | 69.2 | 203.6 | 141.4 | 0.694 | 50.6 | 303.6 |
| bench | 65536 | 0.535 | 1.47 | 2.21 [1.91–2.98] | 3.38 | 70.8 | 72.1 | 212.4 | 144.1 | 0.678 | 54.2 | 325.3 |
| bench | 131072 | 0.494 | 1.50 | 2.33 [1.88–3.22] | 3.70 | 74.8 | 76.1 | 224.5 | 147.8 | 0.658 | 59.4 | 356.2 |

- **Units.** U/q is the group union over the mean per-query quad count (512.8 quads = 2049.5 cells per query in every capture). ch/q is the number of chunks of 4 quads walked per query (the K/V gather count). tiles/q is the number of WMMA 16-row tiles per query.
- **The q8_0 captures** (wiki at 32k/131k) give the same statistics within 0.3 %.
- **b16 (one 16-row tile per present query)** is 131.2 tiles/q in every cell, which reproduces plan-B.md:12 "tS/q 131". The ideal is 128.2.
- **G1** walks 128.8 chunks/q, 2.2× G4. **G2** walks 1.48× G4's chunks (wiki d32k: 86.0 vs 58.1) with about the same tile count (171.9 vs 174.2).
- **Permutation rate.** qpb is 1.001 and the fraction of partly selected quads is 0.001 (the tail quad) in all 22 captures, i.e. no permutation in a fresh single-sequence cache. Multi-sequence or reused-cache layouts were not captured: UNPROVEN.
- **Bucket tile histogram** (G4, `res/union_summary.md` second table).
  - In ASC order, 86–94 % of text chunks need all 3 row tiles, and 2–5 % need only 1. That is why the ASC tile skip is worth ≤ 3 %.
  - In BUCKET order on wiki d32k, the chunks need 1/2/3 tiles in 23/31/46 % of cases.
- **Cross-check with earlier evidence.** L63 put the random-token G4 union at 2.48–2.66× (28k–61k). That matches randc here (2.46–2.63), not llama-bench (2.12–2.21): llama-bench's repeated prompt tokens cut the union.

### Results (b): attend time per layer, ms (mean over the 12 layers of the per-layer medians, n = 20 per layer; `table.py` → `SP/final219/c5/attn_table.md`)
| cell | V0 PR G4-ASC | V1 G4-ASC | V2 ASC-skip | **V4 G4-BUCKET** | V4/V0 | **V4/V2** | V4/V2 worst layer | V3 (PR kernel, bucket order)/V0 | V5 r16/V0 | V6 G8/V0 | V7 G8-skip/V0 | oracle 2 tiles/V0 | net (V4+P_b)/(V2+P_a) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wiki d8k SYN | 13.498 | 13.260 | 13.325 | 12.467 | 0.924 | 0.936 | 0.950 | 1.029 | 1.052 | 1.225 | 1.199 | 0.705 | n/m |
| **wiki d32k real r1** | 15.976 | 15.702 | 15.704 | 14.266 | 0.893 | **0.908** | 0.929 | 1.031 | 0.972 | 1.349 | 1.282 | 0.718 | **0.925** |
| wiki d32k real r2 | 16.152 | 15.903 | 15.883 | 14.406 | 0.892 | 0.907 | 0.929 | 1.031 | 0.970 | 1.346 | 1.278 | 0.715 | 0.924 |
| wiki d32k SYN | 16.217 | 16.030 | 15.959 | 14.442 | 0.891 | 0.905 | 0.926 | 1.030 | 0.968 | 1.347 | 1.277 | 0.716 | 0.922 |
| **code d32k SYN** | 17.877 | 17.663 | 17.188 | 15.587 | 0.872 | **0.907** | 0.916 | 1.029 | 0.932 | 1.386 | 1.267 | 0.717 | **0.922** |
| **chat d32k SYN** | 15.942 | 15.747 | 15.722 | 14.306 | 0.897 | **0.910** | 0.926 | 1.032 | 0.982 | 1.327 | 1.272 | 0.714 | **0.927** |
| wiki d64k SYN | 16.307 | 16.110 | 16.039 | 14.566 | 0.893 | 0.908 | 0.921 | 1.030 | 0.976 | 1.347 | 1.274 | 0.722 | n/m |
| wiki d131k SYN | 16.848 | 16.632 | 16.579 | 15.503 | 0.920 | 0.935 | **1.039** | 1.040 | 0.990 | 1.364 | 1.285 | 0.748 | 0.950 |
| wiki q8_0 d32k real | 17.844 | 17.111 | 17.484 | 15.536 | 0.871 | 0.889 (0.908 vs V1) | 0.916 | 1.015 | 0.941 | 1.346 | 1.288 | 0.701 | 0.905¹ |
| wiki q8_0 d131k SYN | 18.275 | 17.606 | 17.882 | 15.968 | 0.874 | 0.893 (0.907 vs V1) | 0.916 | 1.019 | 0.946 | 1.368 | 1.296 | 0.710 | 0.908 |
| randc d8k SYN | 18.176 | 17.940 | 17.993 | 15.802 | 0.869 | 0.878 | 0.933 | 1.028 | 0.917 | 1.330 | 1.289 | 0.708 | n/m |
| randc d32k SYN | 22.109 | 21.857 | 21.859 | 18.990 | 0.859 | 0.869 | 0.922 | 1.033 | 0.894 | 1.485 | 1.388 | 0.729 | 0.883 |
| **randc d131k SYN** | 28.823 | 28.640 | 28.559 | 31.436 | 1.091 | **1.101** | **1.338** | 1.133 | 1.091 | 1.381 | 1.264 | 0.892 | **1.107** |
| bench d8k SYN | 16.183 | 15.986 | 16.067 | 14.318 | 0.885 | 0.891 | 0.913 | 1.031 | 0.965 | 1.342 | 1.301 | 0.707 | n/m |
| bench d32k SYN | 19.044 | 18.864 | 18.994 | 16.732 | 0.879 | 0.881 | 0.918 | 1.036 | 0.938 | 1.470 | 1.393 | 0.726 | 0.897² |
| **bench d131k SYN** | 24.224 | 24.038 | 24.078 | 27.304 | 1.127 | **1.134** | **1.389** | 1.168 | 1.126 | 1.385 | 1.287 | 0.912 | **1.140**² |

- **Net column.** P_a / P_b are C3's measured plan-stage costs (P_asc 512 / P_bucket 256, this file §C3 table): text d32k 0.737/0.945, text d131k 0.733/0.942, random d32k 0.731/0.963, random d131k 0.883/1.162, q8_0 text d131k 0.739/0.948.
  - ¹ The q8_0 d131k plan cost is used for q8_0 d32k.
  - ² C3's random rows (continuous stream) are used for bench.
  - n/m: plan cost not measured at that depth.
- **Why bucketing loses at depth.** BUCKET order alone, with no skip and the PR kernel (V3), costs +3 % at ≤ 64k but +13–17 % on random at d131k. Bucketing re-walks the n_kv range once per bucket instead of once in ascending order. The likely cause is lost cache locality at large n_kv (UNPROVEN: not profiled).
- **The oracle bound.** Zero-overhead 2-tile chunks cost 0.72× V0 at d32k. Weighting the bucket histogram (23/31/46 % at 1/2/3 tiles, +2.2 % chunks) gives about 0.85× on wiki d32k layer 0 (arithmetic from `res/quick_L0.out` V13/V14 and the histogram). That is the ceiling a perfect bucketed kernel could reach there. It is not a measured kernel.

### Noise
- **Within a run:** per-layer rep sd is 0.05–0.2 ms (≤ 1.3 %), e.g. wiki d32k L0: V0 17.438 ± 0.151, V4 15.290 ± 0.076 (`res/attn_wiki_f16_p32768_real_r1.out`).
- **Run to run** (r1 vs r2, separate processes, same real data): V0 15.976 vs 16.152 (+1.1 %), V4/V2 0.908 vs 0.907.
- **Data sensitivity** (real vs SYN Q/K/V on the same rows): V0 within 1.5 %, V4/V2 0.908 vs 0.905. The verdicts below are therefore insensitive to the data and depend on the rows.
- **Layer to layer:** this is the dominant spread for V4/V2 (0.86–0.93 on text d32k; up to 1.34–1.39 on random d131k), hence the worst-layer column.

### Decision (LEDGER §C5 rules, plans W:260, X:264, Y:291, Z:224)
1. **Bucketing does not enter the plan layout. Freeze ASC as the only order.**
   - The best correct bucketed kernel (V4, 0 spills) gains 9.0–9.3 % attend time at d32k on text f16 against the same kernel in ASC order (V2), and 11.1 / 9.2 % on q8_0 (vs V2 / V1). After the +0.21 ms/layer bucketed plan (C3), the gain is 7.3–7.8 % (f16) and 9.5 % (q8_0).
   - Against each plan's rule:

| plan | rule | result |
|---|---|---|
| W | ≥ 10 % at d32k on T, never slower at d4k | **fails** against the same-family ASC (9.0–9.3 %). It passes only against the PR kernel, attend-only |
| X | ≥ 10 % and ≤ 3 % loss elsewhere | **fails**: random d131k +10.1 %, wiki d131k worst layer +3.9 % |
| Y | ≥ 15 % for f16 and q8 | **fails**: best 12.9 % (wiki q8_0 vs V0); f16 text best 12.8 % (code vs V0) |
| Z | ≥ 10 % at d32k on text and no worse on rand | **fails**: random d131k +10.1 %, bench +13.4 % |

   - Consequences for the contract:
     - drop the `order` parameter and the BUCKET cap term `4·(2^G−1)` from W/Y/Z's layout (plan-v2-W.md:83, :119; plan-v2-Y.md:90, :119; plan-v2-Z.md:62, :96);
     - keep the per-entry query masks, which G4-ASC needs anyway;
     - B's "131 tiles/q" is reproduced as a count (b16 = 131.2), but a 16-row-per-query kernel (V5) is slower than the bucketed 12-row one (V4) in 14 of 16 cells and within 0.1 % in the other two (random/bench d131k).
   - The oracle ceiling (about 0.85× on wiki d32k) and the depth-locality loss suggest a future depth-gated or window-local bucketing could pay. That is a later, separately measured change: UNPROVEN.
2. **Surviving G set: G4 for prefill groups. G1 (row kernel) below the C1 crossover, n_q < 128.**
   - G8 is slower than V0 in all 16 cells: V7 (0 spills) 1.20–1.39×, V6 1.23–1.49×. It walks 32 % fewer chunks per query but does 36–62 % more tile work, and the kernel is compute/LDS-bound: the oracle with 1/3 fewer tiles is 28 % faster.
   - W's "G8 ≥ 8 % faster at d32k on T" and Y/Z's "wins some text cell by ≥ 5 %" both fail.
   - G2 was not timed. Its statistics show 1.48× G4's chunk walks for equal tile work, so there is no measured case for it (UNPROVEN as a timing).
   - Tile skipping in ASC order (V2) changes the time by −2.7 % to +2.2 % against V1. Not worth a variant.
3. **Bench weighting.**
   - **Gate and tune (G, order, thresholds) on real text** (wiki + code + chat at d32k/d131k).
   - Report llama-bench random tokens side by side as a stress row, not as an input to the policy. Measured in this campaign:
     - llama-bench's union is +17 % (d32k) to +27 % (d131k) above wiki;
     - its V0 attend time is +19 % (d32k) and +44 % (d131k) above wiki;
     - continuous random is worse still (+38 % / +71 %);
     - the bucketing verdict flips sign between text and random at d131k.
   - llama-bench's `srand(1)` per `test_prompt` call makes the measured ubatch repeat its first 4096 depth tokens. That gives smaller unions than continuous random tokens (2.12 vs 2.46 at d32k).
4. **P3 note.** These captures ran with `LLAMA_FN_LAST_ATTN_TRIM=0`, and 12/12 layers carried n_q = 4096 (capture logs). With the default trim the last QSA layer runs ≤ 512 queries (L24), so at ub4096 the G4 prefill kernel serves 11 of the 12 layers.

### Large files
- The row and Q/K/V dumps in `strix-002:~/dev/wt_lab/c5/cap/` are deleted: 22 × 403 MB rows, plus 2.1 GB and 1.7 GB of QKV. So is the ISA dump `/tmp/c5isa`. `~/dev/wt_lab/c5` now holds 17 MB. To recapture, use `cap.sh`.

## C6

**Question.** What causes the depth-dependent FA cost: union inflation on random tokens (critique), or MALL/memory-bound behaviour (plan A)? Decides: grid order (L2/MALL-friendly), V prefetch depth, loader priority (LEDGER §C6).

**Answer.**
- **Both, and they apply in different regimes.**
  - The PR's G4 attention kernel is **issue-bound while the K/V cells it touches stay cache-resident**: time is proportional to chunks walked, and q8_0 dequant costs +12 % against packed f16.
  - It becomes **memory-bound once the selected cells spread over a large footprint**. At a fixed 513 quads per query, time per chunk grows **+83 % (f16 V0) at a 131k-cell spread**, and at 1M every loader variant is 1.4–3.6× slower (f16 2.2–3.6×, q8_0 1.4–2.1×). That clears the > 20 % rule.
  - What matters is the **spread, not n_kv**. With n_kv = 1M but cells drawn from a 4k window, the kernel runs at its 4k speed.
- **Real workloads at n_kv ≤ 135k sit mostly in the issue-bound regime.** Measured in the model (PR binary, ub4096), per-layer FA growth from n_kv 12k to 135k breaks down as follows:

| workload | total growth | union inflation | V-pack | per-chunk memory cost |
|---|---|---|---|---|
| wikitext f16 | +5.80 ms | 3.02 (52 %) | 1.66 (29 %) | 1.09 (19 %) |
| code f16 | +7.32 ms | 4.54 (62 %) | 1.66 (23 %) | 1.11 (15 %) |
| random tokens f16 | +13.76 ms | 6.09 (44 %) | 1.66 (12 %) | **5.77 (42 %)** |
| random tokens q8_0 | +9.35 ms | 6.64 (71 %) | 0.98 (10 %) | 1.49 (16 %) |

- The random-token growth (+13.76 / +9.35 ms/layer for f16 / q8_0) reproduces FACTS' "13.7 / 8.9 ms/layer at 131k". On random tokens about half of it is memory cost, which supports the critique. On text, union inflation plus the O(n_kv) V-pack explain 81–85 %, which supports A's "not cache misses" for text. A's attribution of FACTS' growth to qsa_decode remains wrong (critique §2c; here every graph is a 4096-query qsa_prefill graph).
- **Decisions in short:**
  - Loader work comes first. The PR's direct-V loader is 19–26 % slower even when everything is cache-resident, and it must be replaced before the O(n_kv) V-pack can be dropped. Keep native q8_0 K reads.
  - A kv-head-interleaved grid is not the default. It pays −18 % to −30 % only for f16 at a uniform spread of ≥ 262k cells, and costs 0–9 % elsewhere.
  - V prefetch depth is a later, gated item. It can recover at most the per-chunk memory term: ≤ 5 % of text FA at 135k, ≤ 17 % on random f16.

### Setup (strix-002, 2026-11-04 09:31–10:21 CET; every GPU run under `flock ~/gpu.lock`; I was the only GPU user; no build in any shared tree)
- **Where the code lives.** Worktree `~/dev/wt_lab`, detached at 94e086847. The C1–C5 lab diffs are untouched and none of their env switches is set. C6 tools are in `~/dev/wt_lab/c6/`. Copies of every source, script, result and log are in `SP/final219/c6/remote/` (md5 of `c6_attn.cu` local = remote = 11c70168…).
- **(a) Synthetic fixed-work harness `c6_attn.cu`.**
  - Build: `flock ~/gpu.lock /opt/rocm/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -Rpass-analysis=kernel-resource-usage c6_attn.cu -o c6_attn` (`logs/build.log`, BUILD 0).
  - Kernels: the PR's `qsa3_attn_kernel` and its rows+merge plan kernels, included verbatim from C5's `pr_attn.inc` / `pr_union_kernels.inc` (= PR qsa-prefill.cu, see §C5). The plan is built by the PR's own kernels with the PR launch parameters (qsa-prefill.cu:701-718).
  - Workload: n_q 4096, 24 q heads / 2 kv heads, head dim 256, scale 1/16. **Fixed work per query: ns = 2051 cells = 512 full quads + 3 cells of a 513th quad** (the captured rows have 512.8 quads per query, §C5).
  - Union per G4 group fixed at round(U·513) quads, U ∈ {1.0, 1.8}. U = 1.8 gives 57.75 chunks/query, the wiki d32k value (58.1, §C5). Query 0 takes S[0:513], query 1 takes S[u−513:u], and queries 2 and 3 take random 513-subsets. The union is exact (entries/group min = max = 516 / 924 in every cell, `res/a_*.out`).
  - `uni`: each group draws its union uniformly over all n_kv/4 quads. `win W`: the union is drawn from the first W cells while K/V hold n_kv cells (control).
  - Data: hash-uniform K/V/Q of amplitude 1.7 (q8_0: random qs, d = 1.7/127). C5 showed the timing is insensitive to data (real vs SYN within 1.5 %).
  - Variants, all 192 VGPRs and 0 scratch except VD q8_0 at 120 B/lane (`logs/build.log`):
    - **V0** = `<DIRECT=true, 8, DIRECT_V=false, Q8>`: K read from the cache rows, V packed. This is the PR route at ub4096 for n_kv ≤ 64·n_q (qsa-prefill.cu:684-686).
    - **IL** = V0 with a kv-head-interleaved grid (ng4·2, 1), `g = bx>>1, kvh = bx&1`. `pr_attn_il.inc` differs from `pr_attn.inc` only in line 41.
    - **VD** = `<true, 8, true, Q8>`: K and V read from the cache rows. This is the PR route for n_kv > 262144 at ub4096.
    - **KP** = `<false, 8, false, Q8>`: K and V both from the PR's packed per-head quad layout.
    - **PACKV** = the PR's `qsa_pack_v` alone.
  - Timing: 3 warm-up rounds, then n = 20 interleaved rounds, one hipEvent pair per launch; median reported.
  - Commands (`campaign_a.sh`), each for seed s ∈ {1, 2} and T ∈ {f16, q8_0}:
    - `flock ~/gpu.lock ./c6_attn T U uni 4096,8192,16384,32768,65536,131072,262144,524288,1048576 20 s` for U ∈ {1.0, 1.8};
    - `… T 1.8 win 65536,1048576 20 s 4096`;
    - `… T 1.8 win 1048576 20 s 131072`.
  - Tables: `summ_a.py` → `res/summary_a.md`.
- **(b) In-model text vs random.**
  - Driver: C5's `c5_capture` (links the unmodified PR `libllama`, `ldd` → `~/dev/wt_pr219v2/build/bin/libllama.so.0`). It is run with a pos0 that never matches, so no eval callback is installed. It keeps last-token logits only.
  - Same tool, same token streams as the C5 captures: the `tokens … distinct … first …` lines match `c5/logs/cap_{wiki,code,randc}_f16.log`. So the C5 per-layer union statistics apply to these exact rows.
  - Command (`campaign_b2.sh`, runs r ∈ {1, 2}, T ∈ {f16, q8_0}, src ∈ {wiki.test.raw, c5/corp/code.txt, RANDOM}):
    `flock ~/gpu.lock env GGML_CUDA_OP_PROF=1 GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_OVL=0 LLAMA_FN_LAST_ATTN_TRIM=0 ../c5/c5_capture $M <src> 135168 4096 /dev/null/x T 999999999 0 -1`
    with M = `~/models/qwen3.8-flash-next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`. ub = b = 4096 and the trim is off, so all 12 QSA layers run 4096 queries.
  - Each run gives 33 graphs, n_kv = 4096·(j+1), each with 12 FLASH_ATTN_EXT ops (`n_fa [12]` in all 12 logs).
  - Metric: FA ms per ubatch = the sum of the 12 FA OPPROF lines. Parser: `summ_b2.py` → `res/summary_b2.md`.
  - Cross-check driver for random tokens: `flock ~/gpu.lock env <same env> ~/dev/wt_pr219v2/build/bin/llama-bench -m $M -p 131072 -n 0 -d 0 -b 4096 -ub 4096 -r 1 -ngl 99 -fa 1 -lzm on-direct -ctk f16 -ctv f16 -o json` (`logs/b_rand_f16.log`, warm-up + 1 rep = 2 × 32 graphs).
- **Incident.**
  - My first (b) attempt used `llama-perplexity -c 131072 --chunks 2`. The f16 run was **OOM-killed by the kernel**: `journalctl -k`: `Out of memory: Killed process 3267074 (llama-perplexit) … anon-rss:28525432kB`, `EXIT 137`. This is the failure C3 already recorded (all-token host logits).
  - I killed my own following q8_0 perplexity process (pid 3267612) and its script (pid 3267069) before it reached the same point.
  - No other process was affected. Free memory afterwards was 122 GiB available. `logs/b_text_*` were deleted.

### Correctness of the harness
- IL, VD and KP give output **bit-identical to V0** in all 252 comparisons (16 files × every n_kv × 3 variants: `words_differ 0 nan 0`).
- V0 against a CPU f64 reference (6 queries × 24 heads per cell): max |err|/max|ref| per head 2.6e-4 to 3.9e-4.
- Fidelity: synthetic U = 1.8 at n_kv 4k gives V0 15.25 ms at 57.75 ch/q. The C5 replay of real wiki d32k rows gives V0 15.98 ms at 58.1 ch/q (§C5). Per chunk that is 32.2 vs 33.6 ns.

### Results (a): fixed work, spread over n_kv (U = 1.8, median ms, mean of seeds 1 and 2; ratio to the 4k cell in parentheses; `res/summary_a.md`)
| type | variant | 4k | 32k | 64k | 131k | 262k | 524k | 1M |
|---|---|---|---|---|---|---|---|---|
| f16 | V0 (PR) | 15.25 | 16.13 (1.057) | 16.91 (1.109) | 27.98 (**1.834**) | 39.01 (2.558) | 45.47 (2.981) | 49.06 (**3.216**) |
| f16 | KP | 15.14 | 15.82 (1.045) | 16.13 (1.065) | 17.34 (1.145) | 23.69 (1.564) | 29.65 (1.958) | 32.69 (2.159) |
| f16 | IL | 15.30 | 16.48 (1.077) | 17.46 (1.141) | 27.94 (1.825) | 31.78 (2.077) | 33.66 (2.200) | 34.68 (2.266) |
| f16 | VD | 18.19 | 18.58 (1.021) | 21.45 (1.179) | 52.29 (2.874) | 60.27 (3.312) | 64.02 (3.519) | 65.90 (3.622) |
| f16 | PACKV | 0.060 | 0.386 | 0.847 | 1.771 | 3.532 | 7.010 | 13.951 |
| q8_0 | V0 (PR) | 16.97 | 17.51 (1.032) | 17.75 (1.046) | 18.14 (1.069) | 21.62 (**1.274**) | 25.66 (1.512) | 28.44 (**1.676**) |
| q8_0 | KP | 15.20 | 15.79 (1.039) | 16.23 (1.068) | 17.37 (1.143) | 23.36 (1.537) | 29.24 (1.924) | 32.54 (2.141) |
| q8_0 | IL | 17.04 | 17.75 (1.042) | 18.07 (1.061) | 19.41 (1.139) | 23.54 (1.382) | 26.23 (1.539) | 27.17 (1.595) |
| q8_0 | VD | 21.45 | 22.92 (1.069) | 23.98 (1.118) | 24.89 (1.161) | 26.59 (1.240) | 28.65 (1.336) | 30.00 (1.399) |
| q8_0 | PACKV | 0.024 | 0.170 | 0.475 | 1.048 | 2.177 | 4.448 | 8.966 |

- **U = 1.0** (32.25 ch/q, `res/summary_a.md`), ratio to 4k at 131k / 1M:
  - f16: V0 2.24 / 3.23, KP 1.48 / 2.22, IL 1.91 / 2.23, VD 3.03 / 3.59;
  - q8_0: V0 1.21 / 1.74, KP 1.45 / 2.17, IL 1.31 / 1.60, VD 1.19 / 1.40.
  - At 1M, f16 V0 is 28.58 ms against IL 19.99 (−30 %).
- **Spread, not n_kv** (`win` controls, U = 1.8):
  - n_kv 1M with cells from the first 4k: f16 V0 15.15, q8_0 16.77 ms. That equals the n_kv = 4k cells (15.25 / 16.97).
  - n_kv 1M with a 131k window: f16 V0 28.44, q8_0 18.04. That equals n_kv = 131k (27.98 / 18.14).
  - Only PACKV scales with n_kv itself (13.99 ms at 1M in every window).
- **Requested bandwidth** (bytes the kernel asks for / time, `RES … GB/s`, s1, U = 1.8):
  - At 4k, served from caches: f16 V0 508, KP 512; q8_0 V0 349, KP 510 GB/s.
  - At 1M: f16 V0 158, IL 224, KP 238, VD 118; q8_0 V0 209, IL 219, KP 237, VD 137 GB/s.
  - KP reaches about 237 GB/s at 1M in both types. It is the only variant near the LPDDR5X figure of ~200–256 GB/s in FACTS.md:6, which is unverified here. V0 and VD stay below it, so their large-spread cost is access pattern and latency, not pure bandwidth. The mechanism (DRAM page/TLB locality of 512-B head halves at a 1-KB row stride) is UNPROVEN: no hardware counters were read.

### Results (b): in-model FA vs depth, text vs random (PR binary, ub4096, 12 QSA layers; `res/summary_b2.md`)
FA ms per ubatch, r1 / r2:

| corpus | type | n_kv 12288 | 36864 | 69632 | 98304 | 131072 | 135168 | 135168 / 12288 |
|---|---|---|---|---|---|---|---|---|
| wiki | f16 | 178.5 / 178.7 | 225.0 / 222.9 | 233.1 / 230.3 | 239.0 / 236.9 | 251.4 / 247.8 | 254.4 / 248.0 | 1.407 |
| code | f16 | 189.2 / 188.8 | 250.2 / 247.1 | 215.7 / 212.6 | 267.5 / 262.4 | 280.1 / 273.1 | 346.3 / 276.7 | 1.648 |
| random | f16 | 236.4 / 236.9 | 298.4 / 298.0 | 344.1 / 341.7 | 366.8 / 362.0 | 402.8 / 399.0 | 406.1 / 401.5 | 1.706 |
| wiki | q8_0 | 191.9 / 192.3 | 237.1 / 237.0 | 241.0 / 241.0 | 244.9 / 244.6 | 252.9 / 252.7 | 265.6 / 253.3 | 1.351 |
| code | q8_0 | 202.7 / 202.5 | 262.9 / 263.1 | 222.5 / 222.8 | 272.5 / 273.0 | 281.1 / 282.2 | 284.2 / 296.6 | 1.433 |
| random | q8_0 | 255.1 / 256.2 | 315.5 / 315.5 | 343.4 / 344.6 | 351.7 / 351.5 | 363.6 / 364.2 | 442.5 / 367.4 | 1.584 |

- The random-token driver agrees with llama-bench on the same tokens (same `srand(1)` stream) within 0.02–0.81 % of the two-sample means at every n_kv ≥ 8192 (median 0.41 %, 31 graphs; `logs/b_rand_f16.csv`): e.g. llama-bench gives 234.6 / 242.5 at 12288 and 399.6 / 401.2 at 131072.
- **Decomposition per layer** (min of r1/r2; pack, plan and union are derived, see below):
  - attend = FA/12 − pack − plan;
  - ns/chunk = attend / (G4 ch/q × 4096 queries × 2 kv heads);
  - pack = `qsa_pack_v` from (a): 13.5 ns/cell f16, 8.0 ns/cell q8_0, linear (1.771 ms / 131072 and 1.048 / 131072);
  - plan = the PR's rows+merge (C3 T0: text 1.165 at d32k / 1.187 at d131k, random 1.249 / 1.488, §C3 results table; the d32k value is used for n_kv < 100k);
  - ch/q = C5's 12-layer G4 mean for the same token stream (`c5/remote/res/union_*.txt`; q8_0 rows are within 0.3 % of f16, §C5).

| corpus | type | ns/chunk at 12288 / 36864 / 69632 / 135168 | ns/chunk growth 12k→135k | dFA/12 12k→135k (ms) | of which pack | plan | union (ch/q growth) | per-chunk |
|---|---|---|---|---|---|---|---|---|
| wiki | f16 | 34.16 / 35.54 / 36.08 / 36.41 | +6.6 % | 5.80 | 1.66 | 0.02 | 3.02 | 1.09 |
| code | f16 | 34.61 / 35.94 / 36.02 / 36.63 | +5.8 % | 7.32 | 1.66 | 0.02 | 4.54 | 1.11 |
| random | f16 | 33.82 / 35.63 / 37.98 / 41.82 | **+23.7 %** | **13.76** | 1.66 | 0.24 | 6.09 | **5.77** |
| wiki | q8_0 | 37.14 / 38.42 / 38.77 / 38.85 | +4.6 % | 5.12 | 0.98 | 0.02 | 3.29 | 0.83 |
| code | q8_0 | 37.51 / 38.82 / 38.82 / 39.13 | +4.3 % | 6.81 | 0.98 | 0.02 | 4.92 | 0.89 |
| random | q8_0 | 36.83 / 38.19 / 38.73 / 38.90 | +5.6 % | **9.35** | 0.98 | 0.24 | 6.64 | 1.49 |

- **The same rows replayed warm** (C5 `SP/final219/c5/remote/res/attn_*.out` V0 over `union_*.txt` ch/q, per layer, script `SP/final219/c6/c5_nschunk.py`), ns/chunk at n_kv 12288 → 135168:
  - wiki 34.06 → 34.73 (+2.0 %);
  - llama-bench tokens 33.77 → 39.00;
  - continuous random 33.63 → 39.35 (+17 %), with the worst layer at 50.17 (+49 %).
  - In-model the text term is larger (+6.6 % vs +2.0 %). The harness repeats one layer with its K/V warm, so it understates the memory term. The cause (MALL state between layers) is UNPROVEN.
- **Shallow anchor.** The fit-regime ns/chunk (33.8–34.6 f16, 36.8–37.5 q8_0 at 12k) equals the synthetic window-resident value (32.0–32.3 f16 V0, 35.4–35.9 q8_0). So the 12k cells are not memory-limited.

### Noise
- **(a) Seed to seed** (separate processes, different rows and data): s2/s1 median 1.000. Every attention cell is within 0.984–1.018, except f16 V0 at 131k (0.960, per-run sd 1.3–1.6 ms: the f16 transition point). PACKV cells under 0.1 ms are within 0.971–1.029.
- **(a) Within a run:** sd ≤ 0.3 ms below the transition, up to 1.9 ms at 131k (f16 V0 / win131k).
- **(b) Run to run,** |r1 − r2| / mean over the 32 graphs with n_kv ≥ 8192: median 0.12–1.81 % and p90 0.38–2.50 % for every corpus/type.
  - Isolated stall samples, > 5 %: random q8_0 at 20480 / 90112 / 135168 (318.1 vs 285.5, 385.9 vs 348.7, 442.5 vs 367.4); code f16 at 90112 / 135168 (346.3 vs 276.7); wiki q8_0 at 90112 (308.7 vs 237.9). Cause UNPROVEN; this is the C7 question.
  - The first ubatch (n_kv 4096) is a first-call outlier (113–1293 ms) and is excluded everywhere. The decomposition uses the min of the two runs for that reason.

### Decision (rules: W E4 plan-v2-W.md:261, X PF-4 plan-v2-X.md:265, Y X4 plan-v2-Y.md:292, Z E-P4 plan-v2-Z.md:226)
1. **Bound regime: the > 20 % / > 1.3× trigger fires, but only for spread selections.**
   - Every variant exceeds 1.3× by 1M on uniform spread: f16 V0 3.22, q8_0 V0 1.68.
   - On real text to n_kv 135k the per-chunk growth is +4.3 % to +6.6 %. In the fit regime the kernel is issue-bound: q8_0 K-direct is +12 % slower than packed-f16 K at 4k (16.97 vs 15.20) while moving fewer K bytes.
   - So Y's "otherwise prioritise dequant ALU" holds for text, and the memory-bound branch holds for random tokens and for spread beyond 131k.
   - The cause of FACTS' 13.7 / 8.9 ms/layer is now measured: union inflation + V-pack + (random only) per-chunk memory cost, per the table above.
2. **Grid order: keep the PR order (kv head on blockIdx.y) as the default. A kv-head-interleaved order is an opt-in for f16 at very large spread only.**
   - IL measured:
     - **−18 % to −30 %** for f16 at a uniform spread of ≥ 262k cells (U = 1.8: 31.78 vs 39.01 at 262k, 34.68 vs 49.06 at 1M);
     - **−0.2 % to +3.6 %** for f16 at ≤ 64k (U 1.8 and U 1.0; the regime real text occupies);
     - **+0.4 % to +8.9 %** for q8_0 up to 262k;
     - only −4.5 % (U 1.8) / −7.7 % (U 1.0) for q8_0 at 1M.
   - So W/X/Z's "default grid becomes L2-friendly / kv-heads interleaved" is **not** adopted as a default.
   - Gating IL on f16 with n_kv ≥ 262144 is supported by synthetic data only. Real rows beyond 135k were not captured, so a real-text benefit there is UNPROVEN. It must be re-measured on the rewrite's own kernel before it is enabled.
3. **Loader priority: the direct V loader is the first loader work package. It gates dropping the O(n_kv) V-pack.**
   - The pack costs 1.77 ms/layer per ub4096 call at 131k f16 (1.05 q8_0) and 13.95 ms at 1M (8.97 q8_0). That is 8.8 % of wiki f16 FA per layer at 135k (1.825 / 20.67) and 5.1 % for q8_0.
   - The PR's DIRECT_V loader (pr_attn.inc:209-232: per-dim 2-byte column loads for f16, q8_0 with 120 B/lane scratch) is **+19 % (f16) / +26 % (q8_0) slower than packed V at 4k, where everything is cache-resident**. At 131k spread it is +87 % / +37 % slower.
   - Rule for the rewrite: **a native-type direct V loader may replace the pack only if it is ≤ 2 % slower than V0 at the 4k fit cell and not slower than V0 + PACKV at 131k** (f16: 27.98 + 1.77; q8_0: 18.14 + 1.05), on this harness. PR #219's DIRECT_V kernels are not reused as-is.
   - **Keep native q8_0 K reads** (PR 94e086847). Against packed-f16 K they cost +12 % at 4k but win at spread (1M: 28.44 vs 32.54, −13 %). The dequant ALU (issue) cost is the item to optimise in the fit regime.
   - The KP result (f16 at 131k: 1.145 vs V0 1.834) shows that the per-head-contiguous quad layout is what tolerates spread. A direct f16 K loader that reads 512-B head rows at a 1-KB stride inherits V0's spread penalty unless the grid pairs the two heads (point 2).
4. **V prefetch depth: not mandatory at design freeze; a gated optimisation.**
   - It was not prototyped, so any gain is UNPROVEN. The most latency hiding can recover on measured workloads is the per-chunk memory term:
     - ≤ 1.11 ms/layer on text at 135k (≤ 5.3 % of FA/layer);
     - 1.49 ms on random q8_0;
     - 5.77 ms on random f16 (17 %).
   - Y's "LDS prefetch depth 2 mandatory" is therefore not adopted. X's "add V prefetch" becomes a measured follow-up with this harness (`c6_attn … uni …` at 131k–1M) as its gate.
5. **Bench weighting (consistent with C5).** llama-bench random tokens double the depth cost on f16 (13.76 vs 5.80 ms/layer on wiki) and give it a memory-bound component that text does not have. Perf gates on depth must be stated on text, with random tokens as a stress row.

### Large files
- None created. Device buffers (up to about 4 GB at 1M) live only inside each `c6_attn` process. `~/dev/wt_lab/c6` holds 15 MB (logs 14 MB).

## C7

**Question.** What causes the ~421 t/s stall sample (PR q8_0 ub512 pp512 d0) and the 585 ms FA call (plan-A sweep, f16, n_kv 45056)? Pool growth, graph recapture, lazy loading, or the kernel? Decides: does removing `ctx.pool()` fix it by design, or is a root-cause work package needed before landing; and what is the stall gate (LEDGER §C7).

**Answer.**
- **None of the four.** Both kinds of outlier are **KFD process-queue evictions**. They happen while the kernel is migrating pages system-wide, and they hit **base and PR alike**.
- Over 486 traced llama-bench reps, every stall sample (< 0.9× its run median, 5 of 5) had a KFD `evicted_ms` increase inside its window. The eviction covers 92–100 % of the excess wall time. No rep without a stall had an eviction.
- There were **0 hipMalloc/hipFree calls in any rep window** (0 of 486).
- At pp512 d0 no QSA kernel runs at all. All 12 attention ops take the dense `dense_tile_vec_mma` path through the n_kv ≤ 2051 shortcut. So `QSA_DIRECT` and QSA pool scratch cannot matter in that cell.
- **Reproduced in the A sweep:** one f16 attention call took **1255.8 ms** against a 16.4 ms median at n_kv 36864. Its 1411 ms stream sync contained **+1238 ms of KFD eviction** and 1.03 M migrated pages, with no hipMalloc in the sync. The nearest earlier hipMalloc was 11.2 s before.
- **Decision.**
  - Removing `ctx.pool()` does **not** fix the stall, and that was never its cause. Keep the removal for its memory-accounting reasons (C3/C4), not as a stall fix.
  - **No QSA root-cause work package is needed before landing.**
  - The perf protocol and the stall gate must record KFD `evicted_ms` for each sample and exclude or rerun samples that contain an eviction (gate wording below). Drop rep 0: it carries the only HIP graph capture and is 5 % slow on every build.
  - The kernel path that triggers the eviction is **UNPROVEN**: proving it needs root (see Limits).

### Setup (strix-002, 2026-11-04 10:29–11:15 CET; every GPU run under `flock ~/gpu.lock`; I was the only GPU user; no build in any shared tree; nothing was rebuilt)
- **Binaries (unmodified).**
  - PR = `~/dev/wt_pr219v2/build/bin` (94e086847).
  - BASE = `~/dev/wt_pr219v2_base/build/bin` (13a279bfa).
  - Plan-A PLANA build = `~/dev/wt_planA/build/bin` (94e086847 + per-call hipEvents in qsa_prefill: `PLANA_STATS=1`, which syncs the stream after each call; diff in `~/dev/wt_planA`, unchanged).
  - Lab build `~/dev/wt_lab/build/bin`, used only for the route trace (`GGML_QSA_TRACE=1 GGML_QSA_COUNT=1`, C1/C2 env switches).
  - All builds have `GGML_HIP_NO_VMM=ON` (legacy pool) and `GGML_HIP_GRAPHS=ON` (`~/dev/wt_pr219v2/build/CMakeCache.txt:506,512`).
- **Tools** in `strix-002:~/dev/wt_lab/c7/`, copied to `SP/final219/c7/remote/` (md5 of `c7shim.c` 324b3c00…, `c7samp.py` 64272f6e…, local = remote):
  - `c7shim.so`: an LD_PRELOAD shim (`gcc -O2 -Wall -shared -fPIC -o c7shim.so c7shim.c -ldl -lpthread`). It logs CLOCK_MONOTONIC start and duration for:
    - hipMalloc, hipMallocManaged, hipExtMallocWithFlags, hipHostMalloc, hipFree and hipHostRegister;
    - hipStreamBeginCapture/EndCapture, hipGraphInstantiate, hipGraphExecUpdate and hipGraphLaunch;
    - hipStream/Event/DeviceSynchronize longer than `C7SHIM_SYNC_MIN_US`;
    - `llama_process`, `llama_synchronize` and `llama_memory_clear`, which mark the sample boundaries.
    - Check: the 81 848 134 528-byte weight hipMalloc and the pool allocations all appear, so interposition works.
  - `c7samp.py PID OUT 0.01`: samples about every 10–30 ms. It records:
    - `/sys/class/kfd/kfd/proc/<pid>/stats_<gpu>/evicted_ms` (the cumulative time the KFD kept this process's queues evicted);
    - `/proc/meminfo` (MemFree, Cached, SwapFree);
    - `/proc/vmstat` (pgmigrate_success, pgscan_*, compact_stall, allocstall, pswpout);
    - the process's minflt/majflt.
  - `c7run.sh TAG BIN LZM REPS ENV…` runs `llama-bench -m ~/models/qwen3.8-flash-next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf -ngl 99 -fa 1 -lzm LZM -ctk q8_0 -ctv q8_0 -b 4096 -ub 512 -p ${C7P:-512} -n 0 -d 0 -r REPS -o json` under the shim and the sampler.
  - `c7cmd.sh TAG cmd…` does the same for any command.
  - Analysis scripts: `c7ana.py` (per-sample windows), `c7corr.py` (stall × eviction × alloc), `c7vm.py` and `c7base.py` (migration × eviction), `c7noise.py`, `c7A.py` (A sweep).
- **Campaigns.**
  - `pr_auto_od`: PR, no env, on-direct, -r 30. It ran first, after the box had been idle.
  - `seq1` (`c7seq1.sh`): a replica of the `conf2.sh` run that produced the L59 samples. Back to back: base1, pr_auto, pr_d0 (`QSA_DIRECT=0`), pr_d1, pr_d2, base2, with `C7P=512,2048,4096,8192 -r 4`, on-direct.
  - `seq2` (`c7seq2.sh`): `-p 512 -r 30`, interleaved pr_auto, base, pr_d0, pr_d1, pr_d2, base2. First the whole set with `-lzm on-direct`, then the whole set with `-lzm off`.
  - `seqA` / `seqB` (`c7seqA.sh`, `c7seqB.sh`): the plan-A real-text sweep (`~/dev/wt_planA/m/runA.sh`) twice per type, with the shim and `C7SHIM_SYNC_MIN_US=2000`:
    `env PLANA_STATS=1 GGML_CUDA_DISABLE_GRAPHS=1 ~/dev/wt_planA/build/bin/llama-perplexity -m <M> -f ~/data/wikitext-2-raw/wiki.test.raw -c 65536 -b 4096 -ub 4096 --chunks 1 -ngl 99 -fa on -lm mmap -lzm on-direct -ctk T -ctv T`, with T ∈ {f16, q8_0}.
    - seqA wrapped the binary in `timeout 900`, so the sampler watched `timeout`'s pid and no KFD counter was recorded (`nan`). seqB drops `timeout` and has the counter.
  - Workload: llama-bench random tokens (fixed `srand(1)` per prompt, llama-bench.cpp:2220, so every rep is the same workload); perplexity wikitext.
- Raw logs: `strix-002:~/dev/wt_lab/c7/out/` (41 MB) and `SP/final219/c7/remote/out/` (json, err, shim and A/B logs; the 10 ms sampler files stay remote).
- The shim does not change results:
  - the A/B perplexities equal plan A's (f16 3.2688 ± 0.03692, q8_0 3.2743 ± 0.03707, in all 4 + 4 runs);
  - pp512 medians are 1583–1588 t/s, against 1578–1585 in r3 (L59).

### Results 1: pp512 d0 ub512 q8_0 (random tokens), per run (`c7noise.py`)

| run | lzm | n | median t/s | rep 0 | stall samples (< 0.9× median) |
|---|---|---|---|---|---|
| pr_auto_od | on-direct | 30 | 1585.1 | 1484 | none |
| s1_base1 / s1_base2 | on-direct | 4 / 4 | 1584.6 / 1580.3 | 689 / 1495 | **[689]** / none |
| s1_pr_auto / d0 / d1 / d2 | on-direct | 4 each | 1579.8 / 1579.7 / 1578.4 / 1578.2 | 1496 / 1500 / 1495 / 1478 | none |
| s2_base / s2_base2 | on-direct | 30 / 30 | 1584.2 / 1585.9 | 1484 / 1507 | none |
| s2_pr_auto / d0 / d1 / d2 | on-direct | 30 each | 1583.3 / 1583.7 / 1583.4 / 1583.9 | 1495 / **987** / **918** / 1495 | – / [987] / **[918, 1306]** / – |
| s2_base / s2_base2 | off | 30 / 30 | 1583.5 / 1586.8 | 1496 / 1498 | none |
| s2_pr_auto / d0 / d1 / d2 | off | 30 each | 1584.6 / 1588.1 / 1586.4 / 1586.9 | 1505 / 1485 / **512** / 1503 | – / – / **[512]** / – |

- **Noise.**
  - Excluding rep 0 and stalls, the run sd is 2.0–3.7 t/s (0.13–0.23 %; e.g. `s2_pr_d1_on-direct` sd 2.0, `s1_base1` sd 3.7).
  - Between runs, the medians span 1578.2–1588.1 (0.6 %).
  - PR and base are indistinguishable at pp512 d0: PR medians 1578.2–1588.1, base 1580.3–1586.8. They run the same dense kernel (route trace below).
- **Rep 0 is always about 5 % slow (1478–1507 t/s) on both builds.** The shim shows the run's only `hipStreamBeginCapture` / `hipGraphInstantiate` (5–6 ms) / `hipGraphExecUpdate` (3–4 ms) inside the rep-0 window, and none in later reps. For p2048–8192 the capture also happens only in rep 0 (`c7ana.py -r4 s1_*`). This is the HIP-graph (re)capture cost. It is deterministic, and it is not the stall.

**Every stall sample, with what happened inside its window** (`c7corr.py`, `c7vm.py`; pgmigrate = vmstat `pgmigrate_success` delta over the 600 ms ending at the eviction):

| run, rep | t/s | wall ms (median) | excess ms | KFD evicted_ms in window | pages migrated | hipMalloc/Free in window | majflt |
|---|---|---|---|---|---|---|---|
| s1_base1, rep 0 (**BASE**) | 689 | 742.7 (323.5) | 419.2 | **+402** | 503 351 | 0 | 2 |
| s2_pr_d0_on-direct, rep 0 | 987 | 518.4 (323.2) | 195.3 | **+179** | 199 567 | 0 | 0 |
| s2_pr_d1_off, rep 0 | 512 | 1000.8 (322.7) | 678.1 | **+662** | 523 297 | 0 | 0 |
| s2_pr_d1_on-direct, rep 0 | 918 | 557.3 (323.3) | 234.0 | **+218** | 235 284 | 0 | 0 |
| s2_pr_d1_on-direct, rep 1 | 1306 | 392.0 (323.3) | 68.7 | **+69** | 47 252 | 0 | 0 |

- **Contingency over all reps (seq1 + seq2 + pr_auto_od, n = 486):**
  - stall & eviction 5; stall without eviction **0**; eviction without stall **0**.
  - Reps with any hipMalloc/hipFree in the window: **0**. All pool growth happens in the warm-up window: 179 hipMalloc calls in the first warm-up, for example the 32 MiB / 10.5 MiB / 8 MiB pool blocks in `pr_auto_od.shim`.
- **Migration vs eviction**, in 600 ms bins after the first `llama_process` across seq1 and seq2 (`c7base.py`):
  - bins with more than 10 k migrated pages: 15 of 18 had an eviction of ≥ 50 ms;
  - bins with 10 k pages or fewer: 4 of 761 had one.
  - Every eviction of ≥ 50 ms during benchmarking followed 47 k–1.03 M migrated pages (0.18–3.9 GiB).
  - System state at those moments: MemFree 1.6–12.8 GiB and Cached 13–44 GiB, against 111 GB of weights (`c7vm.py`). There were no allocation stalls, no direct compaction (compact_stall 0 in stall windows), no swap-out and no major faults.
- **Timing.** The stalls fall 0.8–3.6 s after the first `llama_process`, i.e. in rep 0 or 1 of the first test. Eviction bursts cluster during and just after model load (the `evicted_ms` increment lists in `c7corr.py` output). This is why the earlier 4–5-rep runs showed "one stall per run, early" (L59).
- **PR vs base.** Stall reps: PR 4 of 286 (3 of 13 runs), base 1 of 128 (1 of 6 runs). Two-sided Fisher exact p = 1.0. "Never on base" (L59) is refuted: s1_base1 rep 0 = 689 t/s with +402 ms of eviction.
- **QSA_DIRECT.** Stalls appeared under d0 and d1, not under d2 or auto in this campaign. The L59 samples were under auto and d2, with none under d0. The mode does not matter, because **no QSA kernel runs in this cell**:
  - `s2_lab_trace.err`: `C1TRACE dense_tile_vec_mma nq 512 nkv 512 ids -1 mask 1 K q8_0 V q8_0`, and `C2COUNT exit` lists only dense_tile_vec_mma for all 12 FA nodes, with no qsa_decode or qsa_prefill.
  - Code: PR qwen4exp.cpp:1871-1878 takes the dense shortcut while idx n_kv ≤ indexer_top_k + r − 1 (= 2051).
- **Lazy load.** Stalls occur with `-lzm off` (s2_pr_d1_off, 512 t/s) as well as `on-direct`. The lazy mode is confirmed per run in the json `"lazy_mode"` field. The stall windows show no major faults.

### Results 2: plan-A 64k wikitext sweep (real text, ub4096, per-call hipEvents, graphs disabled) (`c7A.py`)

| run | attention outlier calls (> 2× n_kv median and > 5 ms excess) | syncs > 400 ms: duration, KFD evicted, pages migrated, hipMalloc inside, last hipMalloc before | pool growth after the first llama_process |
|---|---|---|---|
| A1_f16 | call 0: 61.8, call 5: 49.9 (n_kv 4096, median 7.84) | 675 / 1132 ms; KFD n/a (timeout pid); 297 k / 707 k pages; 0 inside; 22.6 / 23.4 s before | 26 mallocs, high-water 650.7 MiB, longest 5.8 ms |
| A2_f16 | call 98: 123.4 (16.55), call 116: 128.5 (15.15), **call 125: 626.2 ms** (n_kv 49152, median 15.67) | 420 / 663 / 782 ms; KFD n/a; 195 k / 221 k / 413 k pages; 0 inside; 0.59 / 25.0 / 25.7 s before | 26, 650.7 MiB, 13.5 ms |
| A1_q8_0 | none | none | 25, 617.0 MiB, 4.6 ms |
| A2_q8_0 | none | 523 ms; KFD n/a; 356 k pages; 0 inside; 0.50 s before | 25, 617.0 MiB, 32.6 ms |
| B1_f16 | **call 89: 1255.8 ms** (n_kv 36864, median 16.37) | **1411 ms, evicted +1238 ms**, 1.03 M pages, 0 inside, 11.2 s before | 26, 650.7 MiB, 7.8 ms |
| B1_q8_0 | call 125: 83.4 (16.80) | none | 25, 617.0 MiB, 2.1 ms |
| B2_f16 | call 4: 23.0 (7.86), call 107: 123.8 (15.76) | 466 ms, evicted +298 ms, 93 k pages, 0 inside, 11.5 s before | 26, 650.7 MiB, 6.2 ms |
| B2_q8_0 | none | none | 25, 617.0 MiB, 8.3 ms |

- **The 585 ms class reproduces:** A2_f16 call 125 = 626 ms and B1_f16 call 89 = 1256 ms. As in L62, union and pack were normal (B1 call 89: pack 0.575 ms, union 1.341 ms).
- In B1 the KFD counter accounts for 1238 of the 1411 ms sync, with no hipMalloc inside it and the last hipMalloc 11.2 s earlier.
- Pool growth stops early:
  - all but one of the 25–26 hipMalloc calls are in the first ubatch (t + 1.3–1.9 s; B2 + 3.3–4.2 s);
  - one more 33.6 MiB (f16) or 5.2 MiB (q8_0) block arrives at about +11 s.
  - Pool high-water: **650.7 MiB f16, 617.0 MiB q8_0** at c64k, ub4096. That is a slice of what is in flight, as each block is reused.
- No long stall overlaps a hipMalloc.
- The sweep runs with `GGML_CUDA_DISABLE_GRAPHS=1`, so graph recapture cannot explain it.
- **Evictions also hit non-attention work.** KFD evicted time during evaluation was +3095 / +1705 / +2449 / +5745 ms in B1_f16 / B1_q8_0 / B2_f16 / B2_q8_0. B2_q8_0 had no attention outlier, so its evictions landed in other ops. Per-op FA timing alone cannot be a stall detector.

### Verdict per hypothesis
- **Pool growth: rejected.**
  - 0 of 486 bench reps contained a hipMalloc or hipFree.
  - At pp512 d0 the QSA kernels and their `ctx.pool()` scratch (PR qsa-prefill.cu:678-707, qsa-decode.cu:130,145; L32-L34) do not run.
  - In the A sweep, no long sync overlaps a hipMalloc; the last one was 0.5–25.7 s before.
  - The longest pool hipMalloc during evaluation was 32.6 ms, against 400–1400 ms stalls.
- **Recapture: rejected for the stalls.** Capture happens only in rep 0 and costs about 5 % (deterministic, both builds). The A-sweep stalls happen with graphs disabled.
- **Lazy load: rejected.** Stalls occur under `-lzm off` and `on-direct`, with no major faults in the stall windows.
- **Kernel: rejected.**
  - The pp512 d0 cell runs the same dense kernel on PR and base (trace), and base stalls too.
  - In the A sweep, the stalled calls have normal union and pack, and the KFD eviction covers the excess.
- **Cause: KFD queue eviction during system page-migration bursts** (measured correlation: 5/5 stall samples, 0 false positives, 15/18 vs 4/761 bins). It is machine-level: GTT-backed UMA with `amdgpu.gttsize=126976`, `/proc/sys/vm/compaction_proactiveness` = 20, and 37–44 GiB of page cache next to 111 GB of weights.

### Limits (UNPROVEN)
- **Which kernel path evicts the queues is UNPROVEN.** Candidates: MMU-notifier invalidation of a userptr or SVM range moved by kcompactd proactive compaction, or TTM eviction. The shim saw no `hipHostRegister` calls (0 HOSTREG in every log), so any userptr would be created inside the runtime.
  - Settling this needs root, which this account does not have (`sudo -n true` → password required; `/sys/kernel/tracing` permission denied). Either of these would do:
    - the `amdgpu`/`kfd` tracepoints (e.g. eviction and restore events);
    - an A/B with `vm.compaction_proactiveness=0` or a dropped page cache.
  - This is an owner action. I did not change any system setting.
- **The historical samples are attributed by analogy only.** The specific r3/conf ~421 t/s samples (L59) had no eviction counter. The closest reproduction here is 512 t/s (+678 ms, evicted +662 ms).
- seqA has no KFD counter (the sampler watched the `timeout` pid). seqB fixes this.

### Decision and stall gate
- Removing `ctx.pool()` from the sparse kernels is still right for memory accounting (C3/C4, G3), but it is **not a stall fix**. **No QSA root-cause work package is required before landing.**
- **Stall gate** (replaces "no sample < 0.8× / 0.9× median" in plans X/Y/A):
  - Every perf cell records the per-sample KFD `evicted_ms` delta (the `c7samp.py` method) and `pgmigrate_success`.
  - A sample < 0.9× its cell median **with no KFD eviction in its window** fails the gate.
  - Samples with an eviction are excluded and rerun.
  - Their rate is reported for the new build and for base, interleaved over ≥ 30 reps per stall cell (ub512 pp512 d0, ub4096 pp2048 d32k). The gate passes if the new build's rate is not significantly above base's (Fisher, α = 0.05).
  - For per-call FA timing (no FA call > 5× its cell median): calls whose sync overlaps an eviction are excluded the same way.
  - Rep 0 (graph capture, −5 % on every build) is dropped, or reported separately.
- Large files: none created (the remote `out/` is 41 MB). Nothing to delete.

## C8

**Question.** Is QSA+MoE-pinned KLD the primary e2e quality gate, and with which statistic? The pinned floor (1.08e-4) lies between the X/Y line (≤ 1e-4) and W's line (≤ 1e-3). Pinned q8_0 is paired t ≈ 3.6 and unpaired 1.6σ. This decides: the primary e2e gate (paired per-chunk, or floor + kσ), the stated MDE, and whether layer-local replay must be primary (LEDGER §C8).

**Answer.**
- **Yes. QSA+MoE-pinned KLD is the primary e2e gate.**
- **Statistic:** the paired per-chunk difference, new build minus the PR219 build of the same KV type, under the same pins and on the same chunks. The gate is one-sided at mean diff ≤ 3·SE_paired. "Floor + kσ" on unpaired chunk SEs is not the statistic.
- **Layer-local replay does not have to be primary.** Every plan's switch rule (W, X, Y) is met on the side of the pinned gate. It stays a mandatory component gate, because the pinned gate's detection power on real defects is not measured (parts (a) and (b) were not run).
- **New finding (this pass, CPU only, from Q's logs):** the per-chunk values behind the paired t ≈ 3.6 and the MDE 1.6e-5 come from the 5-decimal cumulative table of llama-perplexity. At pinned scale, rounding alone can produce the observed scatter. So **the paired SE, the t and the MDE are not resolved yet**. The mean difference is robust.
- **Precondition for the gate:** llama-perplexity must emit per-chunk KLD at full precision. This is a CPU-side output change to tools/perplexity.

### Sources
This pass ran nothing on the GPU and built nothing. It used reads, `parse.py` and CPU arithmetic only.
- **Q's write-up:** `SP/rewrite219/results-Q.md` §Q1, §Q2, §Q4. Ledger: L65, L67–L72.
- **Q's logs** (`strix-002:~/dev/wt_measQ/q/logs/`, called QL):
  - `f16ub2048_qsamoe.log` (floor), `q8_qsamoe.log`, `q8kf16v_qsamoe.log`, `q4_qsamoe.log`;
  - `*_moe.log`, `*_none.log`, `self_qm.log`, `plain_f16.log`;
  - `d32_*`.
- **Run command:** `~/dev/wt_measQ/q/runQ.sh`. It is `llama-perplexity -c 8192 -b 4096 -ub UB --chunks 8 -fa on -lm mmap -lzm on-direct`, with KLD scored over positions 4096–8191 of each chunk. The binary is PR head 94e086847 plus `q/qpin.patch` (`git -C ~/dev/wt_measQ log -1`, `status`: 4 modified src files plus `q/`).
- **Paired statistics** were recomputed with QL `parse.py` (per chunk = n·k_n − (n−1)·k_{n−1} from the cumulative table). Output: `SP/final219/c8/paired.out`.
- **Rounding simulation:** `SP/final219/c8/round_sim.py`, output in `SP/final219/c8/round_sim.out`.

### Numbers (wikitext c8192 × 8 chunks; reference = hooked f16 ub4096; KLD ×1e-4; chunk SE in parentheses)

| arm | unpinned | QSA pinned | MoE pinned | QSA+MoE pinned | source |
|---|---|---|---|---|---|
| f16 ub2048 (reorder floor) | 65.8 (11.8) | 66.3 (12.4) | 1.19 (0.15) | **1.08 (0.12)** | QL f16ub2048_{none,qsa,moe,qsamoe}.log; L69 |
| q8_0/q8_0 ub4096 (complete-block route) | 75.3 (14.1) | 72.9 (14.3) | 1.42 (0.19) | **1.37 (0.14)** | QL q8_*.log; L70 |
| q8_0 K / f16 V (masked route, L72) | 69.3 (11.9) | 70.1 (12.5) | – | **1.26 (0.17)** | QL q8kf16v_*.log; L70 |
| q4_0/q4_0 (masked route) | 200.0 (30.0) | 194.1 (28.4) | 58.2 (5.9) | **57.9 (6.3)** | QL q4_*.log; L70 |
| f16 ub4096 replayed into itself | | | | 0.000000 | QL self_qm.log, self_none.log; L67 |
| plain (unhooked) f16 ub4096 vs the hooked reference | 88.2 (17.9) | | | | QL plain_f16.log; L68 |
| c32768 × 2, positions 28672–32767: floor / q8_0 | 55.1 / 57.8 | | | **1.01 / 1.35** | QL d32_*.log; results-Q §Q4 |

**Paired per-chunk differences** (`SP/final219/c8/paired.out`; values are multiples of 1e-5 because of the table resolution):

| comparison | per-chunk diff ×1e-5 | mean | SD | SE | t |
|---|---|---|---|---|---|
| q8_0 − floor, QSA+MoE pinned | 3, 5, −2, 6, 3, 3, 3, 3 | 3.0e-5 | 2.33e-5 | 0.82e-5 | 3.64 |
| q8K/f16V − floor, QSA+MoE pinned | 3, 1, 5, 3, −2, 2, 2, 2 | 2.0e-5 | 2.00e-5 | 0.71e-5 | 2.83 |
| q8_0 − q8K/f16V, QSA+MoE pinned | 0, 4, −7, 3, 5, 1, 1, 1 | 1.0e-5 | 3.67e-5 | 1.30e-5 | 0.77 |
| q8_0 − floor, MoE pinned | 5, 1, 0, 6, −2, 2, 2, 2 | 2.0e-5 | 2.56e-5 | 0.91e-5 | 2.21 |
| q8_0 − floor, **unpinned** | 273, 75, 162, 90, 80, 40, 22, 18 | 9.5e-4 | 8.5e-4 | 3.0e-4 | **3.14** |
| q4_0 − floor, QSA+MoE pinned | 899, 589, 534, 722, 441, 403, 584, 372 | 5.68e-3 | 1.76e-3 | 0.62e-3 | 9.13 |

- **The robust mean.** Pinned q8_0 − floor from the 6-decimal final "Mean KLD" lines is 0.000137 − 0.000108 = **2.9e-5** (QL q8_qsamoe.log, f16ub2048_qsamoe.log). The per-chunk mean of 3.0e-5 equals 8·Δk_8 by telescoping, so it is equally robust.
- **The "1.6σ" figure is the pinned unpaired one:** (1.37 − 1.08)/√(0.14² + 0.12²) = 1.57 (L71). The unpinned unpaired figure is (75.3 − 65.8)/√(14.1² + 11.8²) = 0.52σ.
  - Unpinned *paired* is t = 3.14 (table above). The task's phrase "unpinned 1.6σ" mixes these two.
  - Unpaired is the wrong statistic here: the arms share the text, and their per-chunk values are strongly correlated (chunk 1 is highest in every arm, as the per-chunk lists in `paired.out` show).
- **Pinning did not raise q8_0's measured t** (pinned 3.6 vs unpinned 3.1, 8 chunks). What it changes:
  - **scale.** The floor goes from 6.6e-3 to 1.08e-4 (61×, L69). The number becomes a numeric cost (q8_0 = 1.27× the floor; q4_0 = 54×) instead of a cascade.
  - **chunk dominance.** Unpinned, chunk 1 alone holds 36% of the q8 − floor sum (273 of 760 ×1e-5).

### Noise: the per-chunk resolution problem (new, CPU arithmetic)
- **The table resolution.** llama-perplexity prints cumulative KLD to 5 decimals (QL f16ub2048_qsamoe.log lines 17–25: 0.00016, 0.00013, … 0.00011).
  - Reconstructing chunk n from that table carries an error of up to (2n − 1)·5e-6, which is up to 7.5e-5 at n = 8. The pinned signal is about 3e-5.
- **The simulation.** I took true per-chunk floor values ~N(1.08e-4, 2.5e-5) and set q8 = floor + 2.9e-5 exactly, so the true paired noise is **zero**. Reconstructing as parse.py does gives a paired-difference SD of **median 1.07e-5, 90th percentile 3.74e-5** from rounding alone (`round_sim.out`).
  - Q's observed SD of 2.33e-5 lies inside that band. **The true paired SE of pinned KLD is therefore not measured**: it is anywhere from about 0 up to the observed 0.82e-5.
- **Consequences.**
  - q8_0 > floor stays supported: mean +2.9e-5, and the cumulative q8 value exceeds the floor's at every n = 1..8 (QL tables).
  - **t ≈ 3.6 and the MDE "1.6e-5 (2σ, 15% of floor)" (results-Q §Q2) are UNPROVEN as noise figures.** They are upper-bound-style estimates distorted by quantization, not measured noise.
  - The unpinned paired t (3.14) is unaffected: its per-chunk values are about 1e-3, against a rounding error of at most 7.5e-5.
  - The floor's own chunk SE (1.2e-5) is also partly rounding. In the simulation, about 30% of its per-chunk variance is expected to be rounding (same arithmetic).

### Which parts of the proposed experiment Q did NOT run, and whether that matters
| part | run by Q? | covered otherwise? | matters for the decision? |
|---|---|---|---|
| **(a) Z's T5**: attention-local ×(1 + 2^-12·ε) at ub4096, pinned, 8 chunks | **No.** No T5 hook in `q/qpin.patch` (grep: only the QPIN/QPIN_IDX_TYPE/QKL_TAIL switches) | No. The only floor measured is the ub2048 reorder floor, which also reorders GDN chunking and MoE batching (plan-v2-Z.md:350) | **Only for Z's adoption rule.** Z's rule is "KLD(T5) ≤ 0.1 × KLD(q8) and planted defects at ≥ 5σ" (plan-v2-Z.md:219). If the ub floor stood in for T5, it would be 0.79× q8 and **fail** Z's rule. T5 itself is UNPROVEN. It does not change the choice of statistic: a paired new-vs-old gate under the same pins needs no floor as its tolerance. The floor is only context |
| **(b) planted defects** (drop the tail cell in L23, drop one block in L3, V ×(1 + 1e-3)), each as a paired t vs the floor | **No** | No | **Yes, for the stated MDE and for how much weight the gate can carry.** The detection power of pinned KLD for kernel-class bugs is UNPROVEN. Until (b) is run, the pinned gate cannot be claimed to catch a single-cell selection bug, so layer-local replay (rel-L2) and plan bit-exactness stay mandatory co-gates. (b) must be run on a build with full-precision per-chunk output, or its t values hit the same resolution limit |
| **(c) second floor run** on other chunks | **No** (c8192 only on chunks 1–8) | Partly. The c32768 floor on different text positions (28672–32767, 2 chunks) is 1.01e-4 vs 1.08e-4 (QL d32_f16ub2048_qsamoe.log; results-Q §Q4). The level looks stable; its σ across chunk sets is not measured | **Little.** A paired gate takes its tolerance from the paired SE of the comparison itself, not from the floor's level |
| **(d) plain vs plain rerun** on the wt_measQ binary | **No** | By inference, yes: hooked vs hooked replay = 0.000000 (QL self_none.log, self_qm.log; L67). Plain vs plain on the PR binary = 0.000000 (`~/dev/critique219/rerun4096.log`, `kld.sh`, 2 chunks). With QPIN unset no cb_eval is installed (`q/qpin.patch` qpin_init, `if (!m …) return;`, and `if (g_qpin.mode) cparams.cb_eval = qpin_cb`). So L68's 0.0088 is the hook's graph split, not nondeterminism. A direct plain-vs-plain run on this binary is missing | **No.** Whatever its cause, every gate arm must run hooked: reference and candidates under the same QPIN record/replay configuration. The pinned gate then measures numerics in the hooked fusion configuration (results-Q §Q1 caveat). That limitation is stated, not removed |

### Decision
1. **Primary e2e gate: QSA+MoE-pinned KLD, paired per chunk.**
   - Record the f16 ub4096 reference with QPIN once per gate run set. Replay every arm hooked (same binary configuration) with `QPIN_PIN=qsa,moe`.
   - Compare new-type-X against PR219-type-X on the same chunks. Gate: mean paired diff ≤ 3·SE_paired (one-sided).
   - Also report the floor (f16 ub2048 pinned) and each arm's ratio to it.
   - W's rule is met (floor ≤ 1e-3 and q8_0 resolvable above it; plan-v2-W.md:258). X/Y's "≤ 1e-4" (plan-v2-X.md:262, plan-v2-Y.md:289) is met within 1 chunk-SE: 1.08 ± 0.12 at 8k, 1.01 at 32k. The 1e-4 line is not resolvable at this noise, so it should not be treated as a hard threshold.
   - **QSA-only pinning is rejected** (6.63e-3, L69). MoE-only pinning (1.19e-4) is an acceptable cheaper fallback.
2. **Required before the gate is used:** per-chunk KLD at full precision (for example `%.3e` per-chunk means, or the per-chunk sums from `kld_values`) in the measurement perplexity build. Without it, the paired SE at pinned scale is resolution-limited (above).
3. **Stated MDE:** provisional ≤ 2.5e-5 (3 × the observed paired SE 0.82e-5, an upper-bound-style estimate) at c8192 × 8. The figure "1.6e-5" is UNPROVEN until full-precision per-chunk values exist. The MDE in defect terms is UNPROVEN until (b) is run.
4. **Layer-local replay is not primary.** None of the plans' switch conditions is triggered (W: floor > 5e-3; X/Y: no pin level < 1e-3). It remains a required component gate next to plan bit-exactness, because (a) and (b) are unmeasured. If (b) shows a planted single-cell defect below 5σ_paired on the pinned gate, Z's rule applies and replay rel-L2 becomes primary for kernel-local changes.
5. **Route caveat (L72).** q8K/f16V and q4_0 ran on the masked route, f16 and q8_0 on complete-block. Pinned, the two routes agree within noise (q8_0 − q8K/f16V = +1.0e-5 ± 1.3e-5, t 0.77). Cross-route pinned comparisons are still valid, but same-type, same-route pairing stays the gate rule.
6. **Large files:** none created; the bases and pins were already deleted by Q (results-Q §Cleanup). New small files: `SP/final219/c8/{round_sim.py,round_sim.out,paired.out}`.

## C9

**Question.** Should the indexer key cache keep `auto` = -ctk (PR behaviour: the indexer cache is created with the attention `type_k`, L16, PR src/llama-memory-hybrid-idx.cpp:115), or should `auto` map sub-8-bit -ctk types to q8_0 or f16? This decides the `--cache-type-k-idx` default mapping and whether the flag ships. Rule from LEDGER §C9 (W E12): auto → q8_0 for <8-bit types if q8_0's selection agreement is ≥ 0.99× f16's and its paired KLD is within 1σ of f16's; otherwise f16.

**Answer.**
- **Change `auto`: sub-8-bit -ctk → q8_0 indexer. Keep -ctk for f16 and q8_0.** Both parts of the rule are met (tables below).
- The reason is **selection fidelity, not output KLD**. A q4_0 indexer loses 1.2 to 2.1 points of selection agreement. q8_0 recovers f16-level agreement (0.99982× f16 on q4 KV). Unpinned KLD cannot tell any of the three indexer types apart at 8 chunks.
- **The flag does not need to ship.** No measured configuration beats the new `auto`. A public override is optional, and no measurement here supports or opposes it.

### Sources
This pass ran nothing on the GPU and built nothing (reads plus CPU arithmetic only).
- **Q's write-up:** `/tmp/claude-1000/-home-gaetan-puleo/78d4f4eb-9ba7-448e-8841-dce6638c30f6/scratchpad/rewrite219/results-Q.md` §Q1 and §Q3. Ledger: L16, L72, L73, L74, L75.
- **Q's logs:** `strix-002:~/dev/wt_measQ/q/logs/` (called QL). A read-only copy, fetched with `tar` over ssh, is at `SP/final219/c9/logs/`, where SP = `/tmp/claude-1000/-home-gaetan-puleo/78d4f4eb-9ba7-448e-8841-dce6638c30f6/scratchpad`.
- **Run commands:**
  - `q/runQ.sh`, run as `llama-perplexity -c 8192 -b 4096 -ub 4096 --chunks 8 -fa on -lm mmap -lzm on-direct -ctk K -ctv V` against the hooked f16 ub4096 reference `ref8`. KLD is scored over positions 4096–8191 of each chunk.
  - The indexer type is set through `QPIN_IDX_TYPE` (`q/qpin.patch`:271-279, a getenv override of the type used at PR llama-memory-hybrid-idx.cpp:115).
  - The arm list is `q/campaign.sh` (qsamoe and none arms) and `q/campaign3.sh` (moe arms). Every arm listed there has a log with rc=0 (`campaign.out`, `campaign3.out`, `CAMPAIGN_DONE`).
- **Per-chunk values:** Q's `parse.py`, which reconstructs each chunk from the cumulative table. The paired statistics were computed with the new script `SP/final219/c9/paired_c9.py`. Its output is `SP/final219/c9/paired_c9.out`.

### Numbers

**Selection agreement under QSA+MoE pins** (teacher-forced; computed vs reference rows; mean |C∩R|/|R| over query positions ≥ 2052).
- Masked-route (tk) runs are corrected as 2·ov − 1 for the double-hooked view (results-Q §Q1 tool defect). All q4_0-KV runs are tk: `tk_tensors=384, cb_tensors=0` in the QPIN_SUMMARY of every q4 log (L72).
- The indexer override did take effect. The overlaps differ across the arms while the KV type stays the same.

| KV | indexer | overlap (raw → corrected) | ratio to the f16 indexer, same KV | exact-row fraction | source |
|---|---|---|---|---|---|
| f16 | f16 (self) | 1.000000 | 1 | 1.000 | QL self_qm.log; L67 |
| f16 ub2048 (reorder floor) | f16 | 0.997386 | 0.9974 | 0.256 | QL f16ub2048_qsamoe.log |
| f16 | **q8_0** | 0.998509 | **0.9985** | 0.386 | QL f16_idxq8_qsamoe.log |
| f16 | q4_0 | 0.979373 | 0.9794 | 0.005 | QL f16_idxq4_qsamoe.log |
| q8_0 | q8_0 (PR auto) | 0.996726 | 0.99959 | 0.152 | QL q8_qsamoe.log |
| q8_0 | f16 | 0.997135 | 1 | 0.234 | QL q8_idxf16_qsamoe.log |
| q4_0 | q4_0 (PR auto) | 0.987205 → **0.97441** | 0.9879 | – | QL q4_qsamoe.log |
| q4_0 | **q8_0** | 0.993076 → **0.98615** | **0.99982** | – | QL q4_idxq8_qsamoe.log (was pending, L75) |
| q4_0 | f16 | 0.993163 → **0.98633** | 1 | – | QL q4_idxf16_qsamoe.log (was pending, L75) |
| q8_0 / q8_0, c32768 ×2, positions ≥ 2052 | q8_0 | 0.995316 | 0.9990× the c32k floor (0.996327) | 0.065 | `q/campaign2.out`, lines d32_q8_qsamoe and d32_f16ub2048_qsamoe |

- **Agreement rule (≥ 0.99× f16):** q8_0 passes on both KV types (0.9985 on f16 KV, 0.99982 on q4 KV). q4_0 fails on both (0.9794 and 0.9879).
- Under MoE-only pins (no QSA teacher forcing; no tk correction needed), the q4 KV runs give idx q4 0.974451 and idx f16 0.986420 (`q/campaign3.out`, lines q4_moe and q4_idxf16_moe). That is the same gap. No q4_idxq8_moe run exists (`q/campaign3.sh`).
- **Pinned KLD is identical across indexer types,** as it must be (L74). The q4 KV arms with idx q4, q8 and f16 under QSA+MoE pins all give 0.005786 with the same per-chunk lists (QL q4_qsamoe.log, q4_idxq8_qsamoe.log, q4_idxf16_qsamoe.log; parse.py).

**Unpinned KLD** (KLD ×1e-4; chunk SE in parentheses; the paired diff is per chunk, n = 8; `paired_c9.out`):

| comparison | A | B | per-chunk A − B (×1e-4) | mean ± SE_paired | t | \|mean\| / chunk-SE(B) |
|---|---|---|---|---|---|---|
| q4 KV: idx q8 − idx q4 | 202.9 (31.5) | 200.0 (30.0) | +19.4, −6.4, −4.9, −2.9, +10.3, +1.3, +5.6, +0.8 | +2.9 ± 3.05 | +0.95 | 0.10 |
| q4 KV: idx f16 − idx q4 | 201.6 (31.2) | 200.0 (30.0) | +6.6, −1.8, −15.9, +6.7, +0.4, +11.2, +10.3, −4.7 | +1.6 ± 3.21 | +0.50 | 0.05 |
| **q4 KV: idx q8 − idx f16** | 202.9 (31.5) | 201.6 (31.2) | +12.8, −4.6, +11.0, −9.6, +9.9, −9.9, −4.7, +5.5 | **+1.3 ± 3.36** | **+0.39** | 0.04 |
| q8 KV: idx f16 − idx q8 | 75.5 (14.7) | 75.3 (14.1) | +3.9, +1.1, −9.2, +1.8, −3.6, +4.8, +1.9, +0.9 | +0.2 ± 1.61 | +0.12 | 0.01 |
| f16 KV idx q8 − f16 ub2048 floor | 63.8 (11.6) | 65.8 (11.8) | −1.3, −4.9, +1.4, −6.0, −2.7, −3.3, +0.7, +0.1 | −2.0 ± 0.95 | −2.11 | 0.17 |

Sources: QL q4_none.log, q4_idxq8_none.log, q4_idxf16_none.log, q8_none.log, q8_idxf16_none.log, f16_idxq8_none.log, f16ub2048_none.log. The last four of the indexer logs were pending at L75; all are now done with rc=0 (`q/campaign.out`).

- The other columns do not separate the indexer types either. For q4 KV with idx q4 / q8 / f16:
  - same top p: 95.48 / 95.20 / 95.33 %, each ± 0.12 per-token;
  - mean ln(PPL ratio): 0.00469 / 0.00427 / 0.00325, each ± 0.0013 per-token.
  - Source: the "Same top p" and "Mean ln(PPL(Q)/PPL(base))" lines of the same logs.
- **f16 KV + idx q8 unpinned (63.8e-4) is itself a pure selection-perturbation cascade.** The f16 idx f16 arm replayed into itself is 0.000000 (QL self_none.log). Its size is reorder-floor class (65.8e-4).

**MoE-pinned KLD** (selection free, routing pinned; values at 1e-4 scale are limited by the 5-decimal table, see C8 "Noise"). Robust means come from the 6-decimal final lines:
- f16 KV: idx q8 1.20e-4, idx q4 1.26e-4. The floor is 1.19e-4, and f16 idx f16 is 0 by construction (QL f16_idxq8_moe.log, f16_idxq4_moe.log, f16ub2048_moe.log).
- q8 KV: idx q8 1.42e-4, idx f16 1.35e-4, so the f16 indexer is −0.7e-5 (QL q8_moe.log, q8_idxf16_moe.log). The paired t = −3.06 in `paired_c9.out` comes from per-chunk values that are identical at the table resolution. Like the C8 t values, it is **not a measured noise figure (UNPROVEN)**.
- q4 KV: idx q4 58.19e-4, idx f16 56.96e-4. The paired difference is −1.2e-4 ± 0.70e-4 (t −1.71), which is 2% of q4's own cost (QL q4_moe.log, q4_idxf16_moe.log; `paired_c9.out`).

### Noise
- **Unpinned:**
  - The paired SE is 3.0–3.4e-4 for q4 KV and 1.6e-4 for q8 KV (8 chunks). The per-token "±" printed by llama-perplexity (≈ 0.0004) is too small for a paired test (results-Q "Noise").
  - The table-resolution error is at most 7.5e-5 per reconstructed chunk (C8). That is well below these SEs, so the unpinned paired figures are resolved.
- **The KLD rule has low power at n = 8.** "Within 1σ" accepts any q8 − f16 difference up to about 3.4e-4, which is 1.7% of q4 KV's unpinned cost.
  - The measured +1.3e-4 ± 3.4e-4 does not rule out a true q8-vs-f16 indexer difference of a few 1e-4. It does rule out one large compared with the q4 KV cost itself (0.020).
- **Agreement:** the q4 KV agreements of idx q8 and idx f16 differ by 0.00018, or 0.0009 raw. That is below the gap between the floor (ub2048) and f16 self (0.0026). No per-chunk SE of agreement exists in the logs: `.stats` hold per-layer and per-bucket means only, so a σ for agreement is UNPROVEN. The q4 → q8 gap (0.0117) is 45× the q8 → f16 gap.
- **Correction caveat:** the 2·ov − 1 correction assumes that the second (view) pass sees the pinned rows (overlap exactly 1), as results-Q §Q1 states. The MoE-only q4 runs need no correction and show the same q4 → f16 indexer gap (0.9745 → 0.9864).

### Which parts of the proposed experiment Q did NOT run, and whether that matters
| part | run by Q? | matters for the decision? |
|---|---|---|
| q4_idxf16_qsamoe, q4_idxq8_qsamoe | **Yes.** QL q4_idxf16_qsamoe.log, q4_idxq8_qsamoe.log, rc=0 (`q/campaign.out`) | These are the q4 KV agreement inputs (0.98633 / 0.98615) |
| unpinned f16_idxq8, q8_idxf16, q4_idxf16, q4_idxq8 | **Yes.** QL `*_none.log`, rc=0 (`q/campaign.out`, `campaign.sh` last block) | These are the 8-chunk KLD inputs above |
| **16-chunk paired unpinned KLD at c8192, q4 KV with q4/q8/f16 indexer** | **No.** Every arm is `--chunks 8` (`q/campaign.sh`, CH = 8). No 16-chunk log is in QL | **No, not for the mapping.** The KLD half of the rule is met at 8 chunks (q8 − f16 = +0.13σ_chunk, paired t 0.39). Unpinned KLD is also blind to the indexer in the q4 → q8 direction (t 0.95), which is the change actually being made. With more chunks the SE would shrink (expected about 1/√2, UNPROVEN until run). A flip to "f16" would need a true q8 − f16 gap at or above the new SE, and nothing in the agreement data points that way (0.99982×). It would only firm up the low-power caveat above |
| **passkey at 32k** | **No.** There is no passkey log in QL and no passkey arm in `campaign*.sh` | **Little.** Indexer-type agreement was measured only at c8192. At c32768 the only indexer evidence is q8 KV + q8 indexer agreement 0.9953 = 0.999× the 32k floor (`campaign2.out`), which meets the ≥ 0.99 rule at depth for a q8 indexer. The new mapping replaces a q4 indexer with q8, and q8 agreement ≥ q4 agreement on every KV type measured, so a passkey result cannot argue for keeping q4. The unknowns are a q4 KV + q8 indexer at depth and any retrieval-level effect: both **UNPROVEN**. They bear on the size of the benefit, not on its sign |
| the rule itself (agreement ≥ 0.99× f16, KLD within 1σ → q8_0) | **Evaluated here** from Q's data | q8_0 passes both parts on f16 KV and q4 KV, so the mapping is **q8_0**, not f16 |

**Not reproduced: Plan A's "q4 + f16 idx = 0.0287 (1.6σ unpaired)" (LEDGER §C9 Evidence).** In Q's setup (hooked f16 ub4096 reference, same 8 chunks), q4 + f16 idx is 0.020162 against 0.020001 for q4 + q4 idx: paired +1.6e-4 ± 3.2e-4, t 0.50 (`paired_c9.out`). A's reference and setup were not re-checked here. That figure remains as A reported it and is not used.

### Decision
1. **`--cache-type-k-idx auto` maps as follows:**
   - -ctk q4_0 → q8_0 indexer. The rule is met: agreement 0.98615 vs 0.98633 for f16 (0.99982×), KLD q8 − f16 = +1.3e-4 ± 3.4e-4 paired (|t| 0.39 < 1).
   - -ctk f16 → f16 and q8_0 → q8_0, unchanged (= -ctk).
     - A separate f16 indexer on q8 KV gains 0.0004 agreement (0.997135 vs 0.996726) and nothing measurable in unpinned KLD (+0.2e-4 ± 1.6e-4).
     - Under MoE pins it lowers KLD by 0.7e-5 (from the final lines; its σ is resolution-limited, UNPROVEN).
     - That does not justify doubling the indexer bytes for q8 users.
2. **Other sub-8-bit -ctk types** (q4_1, q5_0, q5_1, iq4_nl) were **not measured: UNPROVEN.** Mapping them to q8_0 too is the conservative extension, because q8_0 is the measured floor-class indexer. This build's FA_QUANTS line lists only q4_0-q4_0, q8_0-q8_0, f16-f16 and bf16-bf16 (QL q4_idxq8_none.log line 14), so q4_0 is the only sub-8-bit KV type exercised. bf16 -ctk → bf16 indexer (= -ctk) is likewise not measured: UNPROVEN.
3. **Cost of the change:** for q4_0 KV users, the indexer cache goes from 4.5 to 8.5 bits per element. The block sizes are 18 B and 34 B per 32 elements (PR ggml/src/ggml-common.h:266 and :323), a factor of 1.89×. The absolute size in MiB is not in Q's logs (`-lv` quiet): UNPROVEN.
4. **The flag does not have to ship.** No measured setting beats the new `auto` on output, and the one measurable defect (q4 indexer selection) is removed by `auto`. Ship it only as an explicit override with `auto` as the default. Measurement keeps `QPIN_IDX_TYPE` in the wt_measQ patch for future type checks.
5. **The gate for this option is selection agreement under QSA+MoE pins** (results-Q DECISION Q3.1), not KLD.
   - Pinned KLD erases the indexer effect (L74). Unpinned KLD cannot resolve it at 8 chunks.
   - A rewrite that changes the indexer path must reproduce, on the same `ref8`:
     - f16 KV + q8 idx agreement ≥ 0.9985;
     - q4 KV + q8 idx (corrected) ≥ 0.9861.
6. **Large files:** none created. New small files: `SP/final219/c9/{paired_c9.py,paired_c9.out}` and the read-only log copy `SP/final219/c9/logs/` (plus Q's `parse.py`, `layers.py` and `campaign*` copies in `SP/final219/c9/`).

## C10

**Question.** What is the real-text prefill performance of base and PR (Track P1)? The grid is base/PR × f16/q8_0 × ub {512, 4096} × depth {0, 32k, 131k}, with text and random tokens side by side. It decides the baselines for every perf gate, and whether the PR's q8_0 ub512 d0 regression and the q8-vs-f16 gaps hold on text (LEDGER §C10).

**Answer.**
- **Use the text column below as the perf-gate baseline.** On text, the PR-vs-base deltas are the same as on random tokens to within 1 point at d0. At depth they are smaller (32k) or about the same (131k). The absolute throughput, however, differs:
  - At d0, text runs at 0.82–0.86× the random-token t/s at ub512 (every pp) and at ub4096 pp512. It runs at 0.93× at ub4096 pp2048.
  - At ub4096 pp ≥ 4096, text and random are within 2 %.
  - At depth, text runs at 0.87–0.99× random for pp512 and for ub512, and at 1.00–1.03× for ub4096 pp4096.
  - So random-token absolutes are not valid baselines.
- **The PR's q8_0 ub512 d0 regression holds on text:**
  - pp4096: −3.70 % [−4.02, −2.96] on text, against −4.66 % [−5.12, −4.24] on random.
  - pp8192: −2.05 % [−2.30, −0.88] on text, against −2.16 % on random.
  - pp512 and pp2048 show no regression on either workload.
- **f16 is unchanged by the PR on both workloads.** Every f16 cell's 95 % CI contains 0 (median deltas −0.9 % to +1.2 %).
- **The q8-vs-f16 gap holds on text.**
  - **Base:** q8_0 is 0.37–0.57× f16 at d32k and 0.13–0.23× at d131k on text, and 0.84–0.85× at d0 ub4096 pp ≥ 4096.
  - **PR:** q8_0 runs at 0.96–0.99× f16 at d0. It runs at 0.94–0.98× at d32k and 0.90–0.97× at d131k, on text.
  - **The remaining PR gap is largest for pp512 at depth:** 0.944 / 0.951 at d32k and 0.896 / 0.916 at d131k (ub512 / ub4096). This cell is the target the rewrite has to close.
- **The random-token PR gain over base on q8_0 is overstated at pp512 depth.** On text it is +65 % / +66 % at d32k (random +73 % / +74 %) and +290 % / +291 % at d131k (random +305 % / +306 %).
- **Not PR-specific: the "~421 t/s stall sample".** In this campaign, isolated low samples (< 0.6× the row median) occur on **base as well as PR**: base q8_0 584 and 597, base f16 810, PR q8_0 702 (`logs/d0_q8_0_{0,6}_base_rand.json`, `d0_f16_0_base_rand.json`, `d0_q8_0_5_pr_rand.json`). Four of the six occur in the first benchmark instance of a process. The other two are rep 0 of the second instance, one on base (d32k f16 995) and one on PR (d131k f16 1062) (details under Noise). The r3 observation "never on base" is therefore not reproduced. Its cause is UNPROVEN, and it is a C-stall (P5) input, not a PR defect.

### Setup (strix-002, 2026-11-04 11:32 → 16:03 CET, every GPU process under `flock ~/gpu.lock`, no other GPU job ran)
- **Harness:** a C10 patch of llama-bench. The patch is `strix-002:~/dev/wt_lab/c10/c10.diff` (129 lines), generated by `c10/src/patch_c10.py`. Copies are in `SP/final219/c10/`, where SP = `/tmp/claude-1000/-home-gaetan-puleo/78d4f4eb-9ba7-448e-8841-dce6638c30f6/scratchpad`.
  - It is env-gated by `C10_TEXT=file1[,file2]`.
  - `test_prompt` reads tokens from a pre-tokenized corpus instead of `rand()` (PR tools/llama-bench/llama-bench.cpp:2220-2229).
  - The depth fill uses corpus[0:D]. The timed prompt uses corpus[D:D+P], which is the continuation of the depth text. The warm-up uses corpus[0:P]. Offsets are set at the warm-up, depth and prompt call sites (llama-bench.cpp:2467, :2509, :2536).
  - Without `C10_TEXT`, the code path is upstream.
- **Builds:** `c10/build.sh` compiles the patched llama-bench.cpp/main.cpp with the exact flags of each tree's CMake (`build/tools/llama-bench/CMakeFiles/*/flags.make`). It links against that tree's own built libraries, which are only read, never modified.
  - The outputs are `~/dev/wt_lab/c10/bin_base` (→ `~/dev/wt_pr219v2_base/build/bin`, 13a279bfa) and `bin_pr` (→ `~/dev/wt_pr219v2/build/bin`, 94e086847). `ldd` resolves to those directories (build output).
  - `tools/llama-bench`, `common/` and `include/` are identical between 13a279bfa and 94e086847 (`git diff 13a279bfa 94e086847 --stat -- tools/llama-bench common include ggml/include` is empty). One patched source therefore serves both builds.
- **Text corpus:** wikitext-2 `wiki.test.raw` (1,290,590 B → 297,193 tokens) and `code.txt` (the llama.cpp src/ + ggml/src sources from C5's `mkcorpus.py`, first 1,999,986 B → 580,562 tokens).
  - The two are interleaved in 16,384-token segments: wiki, code, wiki, code, …
  - Total 877,755 tokens, add_bos 0, FNV-1a 48eaad26c3165780. That line is identical in all 42 text runs (`grep "C10: corpus" logs/*text.err`).
  - So the timed prompt is wikitext at d0 ("= Robert Boulter = …"), at d32k (segment 2) and at d131k (segment 8). The depth fill is half prose, half code.
- **Random workload:** the unpatched llama-bench generator: `srand(1)` on every call, `rand() % n_vocab` (llama-bench.cpp:2220-2228). These are exactly the tokens of every earlier bench gate (r3 / L54-L60).
- **Command** (`c10/campaign.sh`, R = 5), where M = `~/models/qwen3.8-flash-next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`:
  `flock ~/gpu.lock env [C10_TEXT=~/data/wikitext-2-raw/wiki.test.raw,~/dev/wt_lab/c10/corp/code.txt] ~/dev/wt_lab/c10/bin_{base,pr}/llama-bench -m $M -ngl 99 -fa 1 -lzm on-direct -ctk T -ctv T -b 4096 -ub 4096,512 -p P -n 0 -d D -r 5 --progress -o json`
  - P = 512,2048,4096,8192 at d0 and 512,4096 at d32768 and d131072.
  - One process covers both ub values. The depth is filled once, in the first instance (ub4096 pp512), and restored for the other three instances through llama-bench's cstate. Every depth .err shows `(cached)`, and each d131k process has exactly one non-cached depth run (`grep`).
  - The indexer cache is part of the restored state (PR src/llama-memory-hybrid-idx.cpp:367-397).
- **Order per (depth, type):** 7 processes, B P B P B P B. In each slot the text run comes first, then random. That gives base 4 processes × 5 reps = 20 samples and PR 3 × 5 = 15 samples per cell. Totals: 84 processes plus 4 equivalence runs, 88 rc=0 (`c10/campaign.out`). No `error|fail|abort` appears in any .err.
- **Equivalence check** (prelude, PR q8_0 d0 ub {4096, 512} p {512, 4096}, upstream binary vs patched in random mode, interleaved ×2; `logs/eq_*.json`). The averages are:

  | row | upstream | patched |
  |---|---|---|
  | ub4096 pp4096 | 2092.1 / 2087.1 | 2088.6 / 2088.3 |
  | ub512 pp4096 | 1376.8 / 1380.6 | 1380.2 / 1380.9 |
  | ub512 pp512 | 1556.6 / 1546.2 | 1539.8 / 1547.5 |

  - The patch is neutral within 0.3 %, except for one stall sample (966) in the upstream ub4096 pp512 row.
  - **Cross-check with r3** (random tokens, upstream binaries): PR/base q8_0 ub512 d0 is pp2048 −0.02 % / pp4096 −4.66 % / pp8192 −2.16 % here, against −0.2 / −4.9 / −2.4 % in r3. d32k pp512 is +74.3 % (ub4096) and +73.1 % (ub512), against +76 % and +73 %. Consistent.
- **Statistics** (`c10/analyze.py`):
  - The ratio is PR/base of the medians of all samples.
  - The 95 % CI comes from a hierarchical bootstrap: resample processes within each build, then reps within each process. It uses 4,000 draws for medians and 20,000 for means, seed 1.
  - The means, the CV, base drift (last ÷ first base process) and outliers are in `c10/summary.txt`. The same analysis with rep 0 dropped is in `c10/summary_drop0.txt`.
  - Raw JSON/err: `strix-002:~/dev/wt_lab/c10/logs/` (1.2 MB), with a copy in `SP/final219/c10/logs/`.

### Results: median t/s, PR/base on text vs random (n = 20 base / 15 PR samples per cell; CI from the hierarchical bootstrap)
| depth | type | ub | pp | text base / PR t/s | text PR/base [95% CI] | random base / PR t/s | random PR/base [95% CI] | text/random base, PR |
|---|---|---|---|---|---|---|---|---|
| 0 | f16 | 512 | 512 | 1324.3 / 1322.2 | -0.16% [-0.70, +0.12] | 1584.0 / 1584.8 | +0.05% [-0.54, +0.70] | 0.836, 0.834 |
| 0 | f16 | 512 | 2048 | 1237.8 / 1237.6 | -0.01% [-0.13, +0.13] | 1512.1 / 1513.2 | +0.07% [-0.29, +0.41] | 0.819, 0.818 |
| 0 | f16 | 512 | 4096 | 1187.3 / 1188.1 | +0.07% [-0.13, +0.44] | 1410.0 / 1411.1 | +0.08% [-0.37, +0.29] | 0.842, 0.842 |
| 0 | f16 | 512 | 8192 | 1144.8 / 1134.4 | -0.91% [-1.63, +0.73] | 1337.4 / 1340.3 | +0.22% [-0.93, +1.29] | 0.856, 0.846 |
| 0 | f16 | 4096 | 512 | 1344.3 / 1342.3 | -0.15% [-14.43, +5.01] | 1611.7 / 1614.2 | +0.15% [-5.21, +6.09] | 0.834, 0.832 |
| 0 | f16 | 4096 | 2048 | 1987.7 / 1984.6 | -0.16% [-0.38, +0.10] | 2138.1 / 2141.7 | +0.17% [-0.33, +0.40] | 0.930, 0.927 |
| 0 | f16 | 4096 | 4096 | 2068.5 / 2065.8 | -0.13% [-0.42, +0.29] | 2102.2 / 2101.6 | -0.03% [-0.21, +0.85] | 0.984, 0.983 |
| 0 | f16 | 4096 | 8192 | 2011.8 / 2012.2 | +0.02% [-0.13, +0.11] | 2027.5 / 2026.3 | -0.06% [-0.16, +0.18] | 0.992, 0.993 |
| 0 | q8_0 | 512 | 512 | 1299.7 / 1300.8 | +0.09% [-0.31, +0.34] | 1542.7 / 1549.5 | +0.44% [-0.44, +0.98] | 0.842, 0.839 |
| 0 | q8_0 | 512 | 2048 | 1223.6 / 1224.2 | +0.05% [-0.30, +0.23] | 1494.3 / 1494.1 | -0.02% [-0.61, +0.46] | 0.819, 0.819 |
| 0 | q8_0 | 512 | 4096 | 1207.1 / 1162.5 | -3.70% [-4.02, -2.96] | 1446.2 / 1378.8 | -4.66% [-5.12, -4.24] | 0.835, 0.843 |
| 0 | q8_0 | 512 | 8192 | 1124.3 / 1101.3 | -2.05% [-2.30, -0.88] | 1328.1 / 1299.5 | -2.16% [-3.80, -0.57] | 0.847, 0.847 |
| 0 | q8_0 | 4096 | 512 | 1317.5 / 1317.3 | -0.01% [-4.03, +0.15] | 1576.8 / 1579.4 | +0.16% [-0.33, +0.67] | 0.836, 0.834 |
| 0 | q8_0 | 4096 | 2048 | 1910.1 / 1911.7 | +0.08% [-0.11, +0.28] | 2053.3 / 2055.9 | +0.13% [-0.18, +0.47] | 0.930, 0.930 |
| 0 | q8_0 | 4096 | 4096 | 1766.6 / 2021.8 | +14.45% [+14.21, +14.69] | 1799.3 / 2057.9 | +14.37% [+14.11, +14.79] | 0.982, 0.982 |
| 0 | q8_0 | 4096 | 8192 | 1690.3 / 1965.6 | +16.29% [+15.54, +16.52] | 1720.7 / 1980.7 | +15.11% [+14.45, +15.45] | 0.982, 0.992 |
| 32k | f16 | 512 | 512 | 1099.7 / 1100.8 | +0.10% [-2.14, +2.22] | 1261.0 / 1268.9 | +0.63% [-3.42, +3.74] | 0.872, 0.868 |
| 32k | f16 | 512 | 4096 | 1144.3 / 1146.0 | +0.14% [-0.08, +0.40] | 1242.7 / 1241.1 | -0.13% [-0.71, +0.20] | 0.921, 0.923 |
| 32k | f16 | 4096 | 512 | 1085.3 / 1088.4 | +0.28% [-2.31, +3.63] | 1236.6 / 1236.1 | -0.04% [-3.06, +3.80] | 0.878, 0.881 |
| 32k | f16 | 4096 | 4096 | 1948.0 / 1952.0 | +0.20% [-0.97, +0.99] | 1932.9 / 1936.8 | +0.20% [-0.82, +0.87] | 1.008, 1.008 |
| 32k | q8_0 | 512 | 512 | 629.7 / 1039.0 | +65.00% [+61.47, +68.44] | 678.9 / 1175.4 | +73.13% [+69.65, +77.87] | 0.928, 0.884 |
| 32k | q8_0 | 512 | 4096 | 609.9 / 1093.8 | +79.33% [+78.60, +80.02] | 638.2 / 1173.4 | +83.87% [+83.44, +84.33] | 0.956, 0.932 |
| 32k | q8_0 | 4096 | 512 | 622.7 / 1035.4 | +66.29% [+62.24, +69.61] | 671.3 / 1170.2 | +74.31% [+71.07, +76.02] | 0.928, 0.885 |
| 32k | q8_0 | 4096 | 4096 | 711.0 / 1902.1 | +167.51% [+164.92, +169.64] | 711.0 / 1888.5 | +165.59% [+162.48, +166.86] | 1.000, 1.007 |
| 131k | f16 | 512 | 512 | 1101.4 / 1100.2 | -0.11% [-3.29, +2.97] | 1172.8 / 1177.4 | +0.39% [-3.16, +3.96] | 0.939, 0.934 |
| 131k | f16 | 512 | 4096 | 1129.7 / 1129.9 | +0.02% [-1.01, +1.00] | 1168.2 / 1181.3 | +1.11% [-0.65, +1.56] | 0.967, 0.956 |
| 131k | f16 | 4096 | 512 | 1075.9 / 1076.7 | +0.08% [-3.32, +3.27] | 1151.9 / 1165.4 | +1.17% [-1.88, +3.19] | 0.934, 0.924 |
| 131k | f16 | 4096 | 4096 | 1825.1 / 1826.2 | +0.06% [-0.97, +0.97] | 1765.0 / 1769.6 | +0.26% [-0.78, +1.04] | 1.034, 1.032 |
| 131k | q8_0 | 512 | 512 | 252.7 / 986.0 | +290.19% [+278.71, +292.49] | 256.2 / 1038.2 | +305.30% [+291.60, +307.61] | 0.986, 0.950 |
| 131k | q8_0 | 512 | 4096 | 241.2 / 1049.4 | +335.01% [+333.13, +337.48] | 244.1 / 1086.7 | +345.23% [+342.39, +350.15] | 0.988, 0.966 |
| 131k | q8_0 | 4096 | 512 | 251.9 / 986.0 | +291.44% [+281.69, +293.53] | 256.1 / 1040.3 | +306.27% [+296.41, +310.06] | 0.984, 0.948 |
| 131k | q8_0 | 4096 | 4096 | 229.7 / 1764.5 | +668.06% [+659.93, +670.81] | 229.7 / 1728.9 | +652.66% [+643.56, +656.99] | 1.000, 1.021 |

**q8_0 ÷ f16, text, medians** (`summary.txt`, "q8_0 / f16" block):

| depth | ub512 pp512 | ub512 pp4096 | ub4096 pp512 | ub4096 pp4096 |
|---|---|---|---|---|
| 0 base / PR | 0.981 / 0.984 | 1.017 / 0.978 | 0.980 / 0.981 | 0.854 / 0.979 |
| 32k base / PR | 0.573 / 0.944 | 0.533 / 0.955 | 0.574 / 0.951 | 0.365 / 0.975 |
| 131k base / PR | 0.229 / 0.896 | 0.214 / 0.929 | 0.234 / 0.916 | 0.126 / 0.966 |

On random tokens the PR's q8/f16 ratio is lower at depth, or within 0.001 of the text value: 0.926 / 0.946 / 0.947 / 0.975 at d32k and 0.882 / 0.920 / 0.893 / 0.977 at d131k.

### Noise
- **Within a process,** steady-state rows have a CV of 0.1–1 % (pp ≥ 2048) and 1–3 % (pp512) (`summary.txt` cv columns).
- **The first instance of every process (ub4096 pp512) carries warm-up samples.** Reps 0–2 run 0.63–0.96× the steady value on text, for example `d0_f16_1_pr_text` [902, 1191, 1196, 1344, 1346]. On random it is mostly rep 0 only, for example `d0_q8_0_0_base_rand` [584, 1378, 1570, 1572, 1580].
  - This hits base and PR alike. It is why the means of the d0 ub4096 pp512 cells have CV 11–21 %, and why medians are the reported statistic.
  - The ub512 pp512 cells (second half of the process) run the same 512-token graph at the same t/s: for example q8_0 d0 1299.7 vs 1317.5 text, and 1542.7 vs 1576.8 random.
  - The cause of the warm-up is UNPROVEN.
- **Low samples < 0.6× the row median:** 6 of 2,240 samples (scan over all `logs/d*.json`). They are d0 base f16 810, base q8_0 584 and 597, and PR q8_0 702 (all in instance 0, i.e. ub4096 pp512, random). The other two are rep 0 of instance 1 (ub4096 pp4096): d32k base f16 995 (`d32768_f16_4_base_text`) and d131k PR f16 1062 (`d131072_f16_3_pr_text`). So 3 of the 6 are base and 3 are PR.
- **Base drift** (mean of the last ÷ the first base process):
  - Within ±1 % in 53 of 64 cells.
  - The worst cases are 0.958 (d131k q8_0 ub4096 pp512 random), 0.970 (d0 f16 ub4096 pp2048 random), 0.971 (d131k q8_0 ub4096 pp4096 random), 0.976 (d131k f16 ub512 pp512 random) and 0.978 (d32k q8_0 ub4096 pp512 text), plus 1.178 for the first-instance cell d0 f16 ub4096 pp512 random.
  - Text cells drift by ≤ 0.9 % except four: d0 f16 ub4096 pp512 (+10.1 %, the first-instance cell), d0 f16 ub4096 pp2048 (+1.0 %), d32k q8_0 ub4096 pp512 (−2.3 %) and d32k q8_0 ub4096 pp4096 (+1.2 %). In none of the four is the drift comparable to that cell's decision effect: 0 for f16 and +66 % / +168 % for q8_0.
- **Minimum resolvable PR/base effect on text with this design:** about ±0.3 % at pp ≥ 2048 (CI half-widths 0.1–0.5 %) and about ±2–3 % for pp512 at depth.
  - At d0 ub4096 pp512 the CI is −14 % / +5 % because of the warm-up.
  - A gate on that cell needs a warm-up instance or rep-0 dropping. With rep 0 dropped, q8_0 text is −0.05 % [−0.11, +0.12] (`summary_drop0.txt`).

### Caveats
- **Restored depth at ub512.** At depth, the ub512 instances start from a depth that was filled at ub4096 and then restored. The numerics differ slightly from an ub512 fill (UNPROVEN size, not perf-relevant). r3 filled at ub512, and the r3 random d32k numbers are reproduced within 3 points.
- **Workload definition.** "Text" means wikitext prose for the timed prompt, after a depth that is half prose, half code. A chat corpus was not run. Whether chat or code in the timed prompt changes the deltas is UNPROVEN.
- **Text is slower than random in absolute terms.** At d0 ub512 the ratio is 0.82–0.86 on both builds and both types, and the PR does not change it. The cause is UNPROVEN (a plausible one, not measured: MoE expert-routing concentration on real text). This is why gate absolutes must be text.
- **Decode** (tg) was not part of C10.

### Decision
1. **Perf-gate baselines:**
   - Use the text medians in the table above (`SP/final219/c10/summary.txt`) together with the harness (`c10.diff`, `build.sh`, `campaign.sh`, `analyze.py`).
   - Report random tokens as a stress row, which is consistent with C5/C6.
   - The gate is the hierarchical-bootstrap CI of the PR/base ratio from interleaved B P B P B P B processes, using medians or rep 0 dropped.
2. **The q8_0 ub512 d0 regression is real on text:** −3.7 % at pp4096 and −2.1 % at pp8192, with CIs excluding 0. The rewrite must remove it, consistent with C1: G=4 at n_q 512 gives +1.9 % over base.
   - The gate is text q8_0 ub512 pp4096 d0 ≥ the base 1207.1 t/s with the CI lower bound ≥ −0.5 %.
   - Neutrality must also hold at pp512 and pp2048 (now within ±0.3 %).
3. **f16 must stay neutral.** Every cell's CI contains 0 for the PR today, so the rewrite's f16 gate is "CI contains 0 or is above it" on text at each depth.
4. **The q8-vs-f16 gap on text, with the PR as the starting point, is the target for "no performance cliff".**
   - The gap is 1–4 % at d0. At pp512 it is 5–6 % at d32k and 8–10 % at d131k; at pp4096 it is 2.5–7 %.
   - Base's cliff (0.13–0.57×) is gone in the PR on text as on random. The residual gap is largest at pp512 / depth.
   - The rewrite's q8_0 gate should be stated as a q8/f16 ratio on text per (depth, pp), with these PR values as the floor it may not go below.
5. **The "stall only on PR" fact is downgraded.** Low samples appear on base too and sit in the first instance of a process. P5 should time them under a fixed warm-up before attributing anything to the PR's pool behaviour.
6. **Large files:** none (logs 1.2 MB). The scripts and results are in `SP/final219/c10/` (`patch_c10.py`, `c10.diff`, `build.sh`, `campaign.sh`, `analyze.py`, `table.py`, `summary.txt`, `summary_drop0.txt`, `table.md`, `campaign.out`, `logs/`).

## C11

**Question.** What Vulkan capabilities does strix-002 have today (Mesa 26.0.3): coopmat1/2, bf16, subgroup size, shared memory? Does a Vulkan build run QSA? It decides the Vulkan kernel inventory (coopmat1 vs scalar), bf16 on Vulkan, register trick vs shared-memory coopMatLoad, the Vulkan gates, and whether all Vulkan validation can run on strix-002 instead of strix-001 (LEDGER §C11).

**Answer.**
- **Device** (`c11/list_devices.log`, `c11/tbo_QSA_PREFILL_MASKLESS.log` line 1, `c11/vkprobe.out`). The ggml line is:
  `ggml_vulkan: 0 = Radeon 8060S Graphics (RADV STRIX_HALO) (radv) | uma: 1 | fp16: 1 | bf16: 0 | fp4: 0 | warp size: 64 | shared memory: 65536 | int dot: 1 | matrix cores: KHR_coopmat`
  - The driver is `radv Mesa 26.0.3`, API 1.4.335, vendor 0x1002 (`vkprobe.out`). Its 128000 MiB device memory has 127816 MiB free.
  - **coopmat1 (VK_KHR_cooperative_matrix): YES.** There are 14 configurations, all M16 N16 K16, subgroup scope (`vkprobe.out`):
    - f16×f16 with f16 or f32 accumulate;
    - u8/s8 × u8/s8 with s32/u32 accumulate, saturating and non-saturating.
    - **No bf16 configuration and no fp8 configuration.** ggml uses the f32 accumulator ("KHR coopmat matmuls use the f32 accumulator (GGML_VK_F32ACC)", tbo log).
  - **coopmat2 (VK_NV_cooperative_matrix2): NO.** It is not in `vkEnumerateDeviceExtensionProperties`, and neither is VK_NV_cooperative_matrix_decode_vector (`vkprobe.out`). Every `coopmat2` path in ggml-vulkan is dead code on this machine.
  - **bf16: NO as a shader type.**
    - `VK_KHR_shader_bfloat16` is not advertised, so ggml reports `bf16: 0` (ggml-vulkan.cpp:6306 @94e086847).
    - The coopmat list has no bf16 entry, so `coopmat_bf16_support` = 0.
    - When the KHR bf16 feature struct is chained anyway, RADV fills shaderBFloat16Type/DotProduct/CooperativeMatrix = 1. That is not usable without the extension, and why RADV fills it is UNPROVEN.
    - `strings libvulkan_radeon.so` shows that the ISA knows `v_wmma_f32_16x16x16_bf16`. No RADV_PERFTEST option names bf16 (listing of the option table near `cswave32`).
  - **Subgroups:**
    - The default subgroup size is **64**, and size control goes from 32 to 64.
    - `requiredSubgroupSizeStages` = 0xf0 includes COMPUTE, and computeFullSubgroups = 1.
    - Ballot, shuffle, arithmetic and clustered are all supported (supportedOperations 0x6ff).
  - **Shared memory:** `maxComputeSharedMemorySize` is **65536 B** and `maxComputeWorkGroupInvocations` is 1024.
  - **Integer dot:** int dot = 1, and int8 4×8 packed mixed-signedness is accelerated.
  - **Architecture:** VK_AMD_shader_core_properties is present with wavefrontsPerSimd = 16. ggml therefore classifies the device as **AMD_RDNA3** (get_device_architecture, ggml-vulkan.cpp:56 @94e086847). That makes the RADV/RDNA3-only `flash_attn_sel_cm` gate true (ggml-vulkan.cpp:4760-4763).
- **The Vulkan backend at the PR base does not compile.**
  - At 13a279bfa and 94e086847 (#219 does not touch ggml-vulkan: `git diff 13a279bfa 94e086847 --stat -- ggml/src/ggml-vulkan` is empty), `ggml_vk_mmid_pair_glu_enabled` is truncated at ggml-vulkan.cpp:10156-10161.
  - The first build fails with `error: a function-definition is not allowed here before '{' token` at :10161 (`c11/build.log` of the first attempt, quoted here).
  - The truncation was introduced with #223 (13a279bfa) and is fixed on origin/master by **6411b4315** ("vulkan : restore the truncated ends of ggml_vk_mmid_down_stream and ggml_vk_mmid_pair_glu_enabled (#240)").
  - **So the rewrite branch must sit on origin/master, or carry #240, to build Vulkan at all.** Whether #219's author ever built Vulkan on that base is UNPROVEN.
- **Yes, a Vulkan build runs QSA** (94e086847 + 6411b4315):
  - **f16** takes the selected-key maskless route. The log says `QSA attention: selected-key kernels (maskless block selection) (K f16, V f16)` (`logs/A_*.err`; src/llama-memory-hybrid-idx.cpp:128-130 @94e086847).
  - **q8_0 and bf16** take the masked route. The log says `masked top-k (no selected-key kernels for this backend and cache type)` (`logs/C_*.err`, `logs/E_*.err`). The reason is that `ggml_vk_flash_attn_sel_pipeline` requires `k->type == v->type == F16`, D = 256 and GQA ∈ {4, 8, 12} (ggml-vulkan.cpp:10577-10616 @94e086847).
  - The **correctness tests** (`test-backend-ops test -o <op> -b Vulkan0`, -j 1) find **no FAIL anywhere**. Each count below excludes the summary line; the log's "OK=" grep is that count plus one.

    | op | passed | not supported |
    |---|---|---|
    | QSA_PREFILL | 26/26 | 0 |
    | QSA_DECODE | 92/92 | 0 |
    | QSA_PREFILL_MASKLESS | 14/14 | 12 |
    | QSA_DECODE_MASKLESS | 34/34 | 19 |

    - **Every** "not supported" case is a q8_0 K or V case (`grep "not supported" | grep -vc q8_0` = 0).
    - With `GGML_VK_SEL_CM=0` (the scalar kernel) the maskless tests give the same 14/14 and 34/34 (`tbo_*_selcm0.log`).
    - FLASH_ATTN_EXT with hsk = 256 and type_K ∈ {f16, q8_0, q4_0} passes 162/162 (`tbo_FA_hsk256.log`). There is no hsk = 256 bf16 case in the suite.
    - FLASH_ATTN_EXT with bf16 K passes 337/337 (hsk 64/72/128 only; `tbo_FA_bf16.log`). The bf16 path is FA_SCALAR, because `FA_COOPMAT1 && BF16 && !coopmat_bf16_support → FA_SCALAR` (ggml-vulkan.cpp:1473).
  - **Quality** is a consistency check, not a KLD proof (`ppl_*.log`). The command was `llama-perplexity -f wiki.test.raw -c 8192 -b 4096 -ub 4096 --chunks 4 -fa on -ngl 99`:

    | backend | path | cache | final PPL |
    |---|---|---|---|
    | Vulkan | sel_cm | f16 | 2.9613 |
    | Vulkan | sel scalar | f16 | 2.9684 |
    | Vulkan | masked | q8_0 | 2.9631 |
    | Vulkan | masked | f16 | 2.9591 |
    | HIP PR (`wt_pr219v2/build`) | | f16 | 2.9660 |
    | HIP PR | | q8_0 | 2.9701 |

    - Every value is ± 0.0445, so all are within 0.011 of each other.
    - The routes were confirmed by a `-lv 4 --chunks 1` rerun of vkA and vkC (`pplv_*.log`, chunk 1 = 2.0915 / 2.0868, identical to the 4-chunk runs' [1]).
- **Performance at d32k pp2048, Vulkan, 94e086847 + #240** (median t/s; text = the C10 corpus, random = llama-bench `rand()`):

  | config | text ub4096 | text ub512 | rand ub4096 | rand ub512 | ×A (text 4096 / 512) |
  |---|---|---|---|---|---|
  | A f16, flash_attn_sel_cm (register trick, default) | **1259.9** | **919.6** | 1277.1 | 983.2 | 1 / 1 |
  | B f16, flash_attn_sel scalar (`GGML_VK_SEL_CM=0`) | 918.5 | 731.2 | 912.2 | 756.6 | 0.729 / 0.795 |
  | C q8_0, masked route (dense FA + mask) | 724.0 | 629.7 | 695.6 | 628.2 | 0.575 / 0.685 |
  | D f16, masked route (`LLAMA_QSA_SELECTED_KEY=0`) | 718.1 | 631.5 | 690.3 | 629.1 | 0.570 / 0.687 |
  | E bf16, masked route | 406.8 | 370.7 | 322.4 | 319.6 | 0.323 / 0.403 |

  - **Op level** (GGML_VK_PERF_LOGGER, one process per config, random tokens, ub4096, last graph = pp2048 @ d32768, n_kv 34816, GQA 12 = 24 Q heads / 2 KV heads; `prof_fa_lines.txt`). The selected-key FLASH_ATTN_EXT per layer (11 layers × 2048 q, plus 1 × 512 q) takes:
    - A sel_cm: **8.35 ms**. That is 94.2 ms in all, 5.9 % of the 1591 ms graph.
    - B scalar: **65.28 ms**, 7.8× A. That is 734 ms, 32.5 % of the graph.
    - The masked dense FA (12 × 2048 q, mask 34816×2048) takes **103.3 ms** for q8_0 (C) and **103.6 ms** for f16 (D), 12.4× A. That is 42 % of the graph.
    - These op times are single samples (-r 1). The end-to-end t/s of the same runs is 1261.9 / 895.6 / 676.9 / 671.6 (`prof_?.out`), consistent with the table.
  - **Compute buffer** (Vulkan0, sched_reserve, d32k):
    - maskless (A/B): **1494.23 MiB** (ub4096) and **226.22 MiB** (ub512);
    - masked q8_0: **2635.77 / 401.01 MiB**;
    - masked f16 / bf16: **2635.50 / 392.99 MiB** (`logs/*_0_rand.err`, `logs/E_*`).

    The masked route grows with n_kv × n_ubatch (qwen4exp.cpp:1113-1119 @94e086847). Beyond d32k on Vulkan this was not measured (UNPROVEN magnitude).

### Noise
- **Within a process** (5 reps): CV 0.16–0.91 % for A–D, and 0.26–0.71 % for E (`summary.txt`).
- **Between processes** (2 per config and workload, interleaved A B C D A B C D): the max/min of the process means is ≤ 1.0016 for every text cell except B text ub4096 (1.0113). Over all cells it is ≤ 1.016 (B rand ub4096 1.0159, A rand ub4096 1.0120).
  - So the A/B (27 %), A/C (42 %) and A/E (68 %) gaps are more than 15× the noise.
  - C vs D (q8_0 vs f16 on the masked route) is +0.8 % / −0.3 % on text. That is inside the noise: on Vulkan the masked route costs the same for both types.
  - E (bf16) has one process per workload, so its between-process noise is not measured. Its text/random gap (406.8 vs 322.4) is unexplained (UNPROVEN cause).
- The first rep of every ub4096 instance is 1–2 % low (e.g. A text ub4096: 1247.1 then 1259.9–1263.3). Medians are reported.
- **text / random:** A runs at 0.987× random at ub4096 and 0.935× at ub512. So random-token absolutes overstate the ub512 Vulkan baseline by about 7 %, as on HIP (C10).

### Setup (strix-002, 2026-11-04 16:00–17:40 CET; every GPU or build command under `flock ~/gpu.lock`; no other GPU job ran)
- **Tree:** `~/dev/wt_lab` (detached 94e086847, which carries other C-labs' env-gated diffs in cuda/cpu/qwen4exp, all inert without their env vars).
  - Plus `git apply c11/fix_6411b4315.diff` (= `git show 6411b4315 -- ggml/src/ggml-vulkan/ggml-vulkan.cpp`, 55+/19−). **`wt_lab/ggml/src/ggml-vulkan/ggml-vulkan.cpp` is now modified by this patch.** The HIP `build/` directories do not compile that file.
- **Configure** (`c11/cfg2.log`):
  `cmake -B build-vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DCMAKE_CXX_FLAGS=-I$HOME/.local/spirv-headers/include -DSPIRV-Headers_DIR=$HOME/.local/spirv-headers/share/cmake/SPIRV-Headers`
  - The spirv-headers flag is needed: `spirv/unified1/spirv.hpp` is not installed system-wide (first-attempt error, ggml-vulkan-types.h:47). It is the same flag the other build-vk trees on strix-002 use (`wt_vkskip/build-vk/CMakeCache.txt`).
  - glslc is `~/.local/bin/glslc`, a wrapper over glslang 16.6.0. CMake reports `GL_KHR_cooperative_matrix`, `GL_NV_cooperative_matrix2`, `GL_EXT_integer_dot_product` and `GL_EXT_bfloat16` supported by glslc. These are compiler capabilities, not device capabilities.
- **Build:** `cmake --build build-vk -j 16 --target llama-cli llama-bench test-backend-ops llama-perplexity llama-completion`, rc 0.
- **Text bench:** the C10-patched llama-bench (`c10/src/llama-bench.cpp`) linked against build-vk (`c11/build_bench_vk.sh`, flags from `build-vk/tools/llama-bench/CMakeFiles/llama-bench-impl.dir/flags.make`; ldd resolves to `wt_lab/build-vk/bin`).
  - The corpus line is identical to C10: `C10: corpus 877755 tokens (seg 16384, add_bos 0), fnv1a 48eaad26c3165780`.
  - The timed prompt is corpus[32768:34816] ("2 to rudder damage , the IJN decided to").
- **Capability probe:** `c11/vkprobe.c` (vkGetPhysicalDeviceProperties2/Features2, vkEnumerateDeviceExtensionProperties, vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR; no device is created). Built with `gcc -O1 -o vkprobe vkprobe.c -lvulkan`, run as `flock ~/gpu.lock ./vkprobe`.
- **Campaign** (`c11/campaign.sh`): `flock ~/gpu.lock env [C10_TEXT=$HOME/data/wikitext-2-raw/wiki.test.raw,$HOME/dev/wt_lab/c10/corp/code.txt] <cfg env> c11/bin_vk/llama-bench -m ~/models/qwen3.8-flash-next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf -ngl 99 -fa 1 -ctk T -ctv T -b 4096 -ub 4096,512 -p 2048 -n 0 -d 32768 -r 5 -v -o json`
  - The configs are A `GGML_VK_SEL_CM=1` f16, B `GGML_VK_SEL_CM=0` f16, C q8_0, and D `LLAMA_QSA_SELECTED_KEY=0` f16. There are 2 rounds, and each slot runs text then random. That gives 16 processes, all rc 0 (`c11/campaign.out`). E (bf16) was run afterwards with the same command, 1 process per workload.
  - A first launch was aborted by me. `~` after the comma in C10_TEXT was not expanded, and the C10 patch stops with `C10: cannot open ~/dev/...`. Its logs were deleted before the rerun.
- **Profiles:** `flock ~/gpu.lock env GGML_VK_PERF_LOGGER=1 <cfg env> build-vk/bin/llama-bench … -ub 4096 -p 2048 -n 0 -d 32768 -r 1`.
- **Correctness:** `flock ~/gpu.lock build-vk/bin/test-backend-ops test -o {QSA_PREFILL_MASKLESS,QSA_DECODE_MASKLESS,QSA_PREFILL,QSA_DECODE} -b Vulkan0`. FA subsets use `-o FLASH_ATTN_EXT -j 1 -p "hsk=256.*type_K=(bf16|q8_0|q4_0|f16),"` and `-p "type_K=bf16,"`. The full suite was never run.
- **Raw files:** `strix-002:~/dev/wt_lab/c11/`. A copy is in `SP/final219/c11/` (396 KB): `vkprobe.{c,out}`, `list_devices.log`, `tbo_*.log`, `logs/*.json`, `campaign.{sh,out}`, `build_bench_vk.sh`, `analyze.py`, `summary.txt`, `prof_?.out`, `prof_fa_lines.txt`, `ppl*_*.log`, `fix_6411b4315.diff` and `cfg2.log`. No file over 1 GB was created.

### Decision
1. **Kernel inventory: coopmat1 plus scalar, no coopmat2.** Build the Vulkan sparse-attention shader for KHR coopmat 16×16×16 f16 (f32 accumulate), with a scalar fallback. Do not write or rely on NV coopmat2 variants for strix-class hardware: they cannot be tested on strix-002. The wave size is a free choice, because compute accepts a required subgroup size of 32 or 64. Shared-memory staging has a 64 KiB budget per workgroup (the current scalar sel kernel uses 46,720 B at GQA 12, ggml-vulkan.cpp:4747-4749 @94e086847).
2. **bf16 on Vulkan is not a WMMA target.** RADV 26.0.3 exposes neither VK_KHR_shader_bfloat16 nor any bf16 coopmat config. bf16 K/V must be converted to f16 when staged (or read by a scalar path) inside the sparse kernel.
   - Today bf16 QSA on Vulkan runs the masked route at **0.32× (ub4096) / 0.40× (ub512)** of f16 on text at d32k pp2048. That is the worst cliff measured here, and the rewrite has to remove it.
   - Its correctness at hsk = 256 is untested: no test case exists. Add one.
3. **Register trick vs shared-memory coopMatLoad: there is no shared-memory D256 kernel yet, so the comparison cannot be run today.**
   - The only coopMatLoad selected-key shader, `flash_attn_top_k_cm`, is D512/MQA only (`ggml_vk_flash_attn_top_k` gate: `q->ne[0] != 512 … k->ne[2] != 1`, ggml-vulkan.cpp:10736 @94e086847).
   - The measured reference the new shader must meet is sel_cm (register trick, flash_attn_sel_cm.comp:22): **8.35 ms per layer** at d32k pp2048 ub4096 (11 × 2048 q, GQA 12, 2051 selected). It must also meet **1259.9 / 919.6 t/s** end to end on text (ub4096 / ub512).
   - The scalar kernel is 7.8× slower on the op and 0.73× / 0.80× end to end. So the "scalar fallback" is a correctness tier, not a performance tier, on this device.
   - Keep sel_cm as the RADV/RDNA3 fast path until the shared-memory kernel measures within 10 % of these numbers. The plans' ≥ 10 % rule (plan-v2-W E10, plan-v2-X PF-9) then decides whether to drop it.
4. **Vulkan gates (baselines, text, d32k pp2048, 94e086847 + #240):**
   - **f16** 1259.9 (ub4096) / 919.6 (ub512) t/s, with noise ≤ 1.1 % between processes.
   - **q8_0** today is 724.0 / 629.7 (0.575× / 0.685×), the q8_0 cliff on Vulkan; bf16 is 406.8 / 370.7.
   - **Memory:** maskless 1494 / 226 MiB against masked 2636 / 393–401 MiB of Vulkan0 compute buffer.
   - **Proposed gate:** with the rewrite, q8_0 and bf16 reach f16's maskless route. Their t/s must be ≥ 0.95× the f16 maskless t/s of the same build. Their compute buffer must equal f16's (as the HIP PR already achieves for q8_0: 3171.06 vs 3170.80 MiB, orchestrator fact "Image matrix"). And f16 itself must not drop below the A row by more than the noise.
5. **All Vulkan validation can run on strix-002; strix-001 is not needed.** The machine has RADV on gfx1151 (Mesa 26.0.3), and a working Vulkan build recipe exists: the flags above plus #240. Whether strix-001's driver matches was not checked (strix-001 was not touched).
6. **The rewrite must be based on origin/master (or include 6411b4315).** Otherwise the Vulkan backend does not compile. Record this in the PR description as a reason the rewrite is not based on #219's base.

**Status:** resolved for capabilities, for "does Vulkan run QSA" and for baselines. Register trick vs shared coopMatLoad is not decidable until the shared-memory shader exists; the reference numbers and the keep/drop rule above stand in for it.

## C12

**Question.** Does `LLAMA_FN_LAST_ATTN_TRIM` explain "11 of 12 QSA layers reach `qsa_prefill_impl1` at n_kv ≤ 32k" (critique.md:62, Q5 at :209; LEDGER §C12, L24)? It decides whether the trimmed shape (n_q 512 at ub4096) goes into the policy table and the reserve list.

**Answer: yes, confirmed at runtime.** With the trim off, all 12 QSA layers reach `qsa_prefill_impl1` in every ubatch. With the trim on (the default), the 12th call of each first-half ubatch is missing. That call is layer 47, and it goes to `qsa_decode` with n_q = 512 and the maskless selected-key route, for both f16 and q8_0.

**Setup.** strix-002, sole GPU user, every GPU command run under `flock ~/gpu.lock`. Model `~/models/qwen3.8-flash-next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`, real text `~/data/wikitext-2-raw/wiki.test.raw`, c 65536, b = ub = 4096, one chunk (16 ubatches up to n_kv 65536), `-ngl 99 -fa on -lm mmap -lzm on-direct`, graphs off.

- **Part A, the proposed experiment.** It uses plan A's stats hook, binary `~/dev/wt_planA/build/bin` (94e086847 plus the `PLANA_STATS` hook in `qsa_prefill_impl1`; `git -C ~/dev/wt_planA diff`). The hook prints one `PLANA call=` line per `qsa_prefill_impl1` call. I ran the binary only and did not modify the tree. Script `~/dev/wt_lab/c12/runA_trim.sh` (copy at `final219/c12/runA_trim.sh`):
  `flock ~/gpu.lock env PLANA_STATS=1 GGML_CUDA_DISABLE_GRAPHS=1 LLAMA_FN_LAST_ATTN_TRIM=<0|1> timeout 900 ~/dev/wt_planA/build/bin/llama-perplexity -m $M -f $W -c 65536 -b 4096 -ub 4096 --chunks 1 -ngl 99 -fa on -lm mmap -lzm on-direct -ctk <T> -ctv <T>`
  - Runs: trim 0 × {f16, q8_0} × 2 repetitions, and trim 1 × {f16, q8_0} × 1 as a control rerun of plan A's original `~/dev/wt_planA/m/A_{f16,q8_0}.log`.
- **Part B, route identity.** This answers which layer is missing and where it goes. It uses the lab build `~/dev/wt_lab/build/bin` (94e086847 plus the C1/C2 env-gated tracers; `llama-perplexity` relinked against the current lab source, `c12/build.log`).
  - `GGML_QSA_TRACE=1` prints each new (kernel, n_q, n_kv, ids, mask, K, V) once (`C1TRACE`).
  - `GGML_QSA_COUNT=1` prints the launch totals per node at exit (`C2COUNT exit`).
  - `-lv 4` makes the trim's one-time `LLAMA_LOG_INFO` line visible.
  - Script `~/dev/wt_lab/c12/runT_trim.sh`:
  `flock ~/gpu.lock env GGML_QSA_TRACE=1 GGML_QSA_COUNT=1 GGML_CUDA_DISABLE_GRAPHS=1 LLAMA_FN_LAST_ATTN_TRIM=<0|1> timeout 900 ~/dev/wt_lab/build/bin/llama-perplexity <same args> -ctk <T> -ctv <T> -lv 4`
- Logs are `~/dev/wt_lab/c12/logs/{A,T}_*.log` on strix-002, copied to `final219/c12/logs/`.

**Raw numbers, part A** (`qsa_prefill_impl1` calls per ubatch, by n_kv; from `grep "PLANA call" | awk '{print $3}' | sort | uniq -c`):

| run | n_kv 4096 … 32768 (8 ubatches) | n_kv 36864 … 65536 (8 ubatches) | total | n_q of every call | PPL |
|---|---|---|---|---|---|
| A_trim0_f16_r1 | 12 each | 12 each | 192 | 4096 | 3.2688 ± 0.03692 |
| A_trim0_f16_r2 | 12 each | 12 each | 192 | 4096 | 3.2688 ± 0.03692 |
| A_trim0_q8_0_r1 | 12 each | 12 each | 192 | 4096 | 3.2743 ± 0.03707 |
| A_trim0_q8_0_r2 | 12 each | 12 each | 192 | 4096 | 3.2743 ± 0.03707 |
| A_trim1_f16_r1 (control) | **11 each** | 12 each | 184 | 4096 | 3.2688 ± 0.03692 |
| A_trim1_q8_0_r1 (control) | **11 each** | 12 each | 184 | 4096 | 3.2743 ± 0.03707 |
| plan A original `m/A_f16.log`, `m/A_q8_0.log` (trim default) | 11 each | 12 each | 184 | 4096 | 3.2688 / 3.2743 |

- **The missing call is the last one of the ubatch.** For each first-half ubatch, I compared the `g4_mean_blk` sequence of the 11 trim-1 calls with calls 1–11 of the trim-0 run. They are identical for f16 and q8_0: 88 of 88 lines, 0 differences (command: awk over `A_trim0_<T>_r1.log` vs `A_trim1_<T>_r1.log`, output `f16 first-half lines: 88 vs 88; diff lines: 0`, same for q8_0). The selection is deterministic, so the 11 calls are the first 11 QSA layers and the missing call is the 12th.
- **Cost of the 12th layer on the G=4 prefill path** (trim 0, pack + union + attention ms from the hook, f16 r1): 8.67 ms at n_kv 4096, 13.42 at 8k, 14.90 at 16k, 17.25 at 32k. The per-ubatch hook sum trim0 − trim1 equals it: 106.14 − 97.46 = 8.68 ms at 4096. The hook does not time `qsa_decode`, so the time of the trimmed layer is not measured here. It is row T of C1.
- **PPL is identical with the trim on or off** (3.2688 f16, 3.2743 q8_0, all runs). That is expected: perplexity reads logits only for positions ≥ n_ctx/2, and those ubatches are never trimmed (below).

**Raw numbers, part B** (`T_*.log`):

| run | trim log line | non-decode QSA traces (n_q 4096) | `qsa_decode` traces | exit totals (FA launches, ubatch ≥ 128 q) |
|---|---|---|---|---|
| T_trim1_f16 | `build_layer_attn: LLAMA_FN_LAST_ATTN_TRIM: layer 47 attention runs query rows [3584, 4096) of 4096` | qsa_prefill n_kv 36864…65536 | `C1TRACE qsa_decode nq 512 nkv {4096,8192,…,32768} ids 2051 mask 0 K f16 V f16` (8 lines) | qsa_prefill_gated 176, qsa_prefill 8, **qsa_decode 8** |
| T_trim0_f16 | none | qsa_prefill n_q 4096 (8 shapes) | none | qsa_prefill_gated 184, qsa_prefill 8, qsa_decode 0 |
| T_trim1_q8_0 | same line, layer 47, [3584, 4096) | qsa_prefill (16 shapes) | `C1TRACE qsa_decode nq 512 nkv {4096…32768} ids 2051 mask 0 K q8_0 V q8_0` (8 lines) | qsa_prefill 184, **qsa_decode 8** |
| T_trim0_q8_0 | none | qsa_prefill (16 shapes) | none | qsa_prefill 192, qsa_decode 0 |

- Each run has 12 launches of `dense_tile_vec_mma nq=2-8 mask=1 ids=0` as well, one per QSA layer, at n_kv 256 (`C1TRACE dense_tile_vec_mma nq 2 nkv 256 ids -1 mask 1`). They are outside the 16 ubatches (the counter counts every FLASH_ATTN_EXT). Which call produces them is UNPROVEN; they take the same route with the trim on or off.
- For f16, 11 layers take the fused gated entry (`qsa_prefill_gated`, ggml-cuda.cu fusion) and the last layer takes the plain `qsa_prefill` entry. Both reach `qsa_prefill_impl1` (the part A counts). Why the last layer is not fused is UNPROVEN and does not affect the count.
- **Total QSA launches are the same:** 192 per run with the trim on or off (184 + 8 decode with trim 1, 192 prefill with trim 0). So the trim reroutes one layer and drops no layer.

**Mechanism (code, 94e086847, identical in wt_planA and wt_pr219v2; qwen4exp.cpp is untouched by the plan-A diff):**
1. `src/models/qwen4exp.cpp:938-951`. For `il == n_layer-1` (47) with the trim on (default, :1795) and `inp_out_ids` set, `q_first = ((n_tokens-1)/strip)*strip`, where strip = `min(n_tokens, 512)` (:1077-1079). For n_tokens 4096 that is 3584. `q_first` is reset to 0 if any row before it is an output row.
2. `qwen4exp.cpp:1886-1900`. The trim is taken only on the maskless input set, with one stream and a strip-aligned q_first. Then `n_q = n_tokens - q_first = 512` (:1904).
3. `ggml/src/ggml-cuda/fattn.cu:1154-1158`. The dispatcher tries `qsa_decode` first. `qsa-decode.cu:94` accepts `q->ne[1] <= QSA_DECODE_MAX_QUERIES` = 512 (`qsa-decode.cuh:5`), so the 512-query layer never reaches `qsa_prefill`.
4. `tools/perplexity/perplexity.cpp:540, :573` sets `first = n_ctx/2` and `need_logits = pos >= first`.
   - At c65536 the ubatches with n_kv ≤ 32768 (positions < 32768) need no logits, so they are trimmed.
   - From n_kv 36864 on, row 0 is an output, `q_first` falls to 0, and layer 47 runs all 4096 queries through `qsa_prefill`. This is exactly the 11/12 boundary in the table.

**Consequences (scope as stated):**
- **llama-bench.** `tools/llama-bench/llama-bench.cpp:1343-1347` states that only the last token of each batch is output (and reserves `n_outputs_max = 1`). So by the same code path, every llama-bench pp ubatch with n_ubatch > 512 sends layer 47 to `qsa_decode` with n_q = `n_tokens - q_first` (512 for ub a multiple of 512). This is code-verified only; no llama-bench run was traced here (UNPROVEN at runtime).
- **The server and any prefill whose only output is the last token** behave the same way. Code-verified by guard :945-950; not traced here.
- At **ub ≤ 512**, strip = n_tokens and q_first = 0, so there is no trim. Every layer is already ≤ 512 queries and goes to `qsa_decode` (fattn.cu:1154, qsa-decode.cu:94).
- In a **final partial ubatch** of n_tokens not a multiple of 512, the trimmed layer has n_q = n_tokens − 512·⌊(n_tokens−1)/512⌋ ∈ [1, 512] (formula :944-945; not run).

**Noise.**
- **The call counts are deterministic.** They are identical across 2 repetitions per type with trim 0 (192/192, 12 per ubatch), and the trim-1 control reproduces plan A's original logs (184, 11 per ubatch at ≤ 32k).
- The selection statistics are bit-identical between repetitions (`g4_mean_blk` per ubatch: f16 434.4/723.5/…/879.9 in r1 and r2).
- **Hook timings, r1 vs r2:**
  - The f16 per-ubatch QSA sum differs by 0.2–1.4 % (e.g. 106.14 vs 106.39 ms at 4k, 212.46 vs 215.46 at 48k).
  - q8_0 differs by ≤ 0.5 %, except for single-call spikes (n_kv 40960: 293 ms in both q8_0 runs against about 225 ms; n_kv 45056 r2: 367 ms; trim1 q8_0 n_kv 65536: 300 ms). These are isolated outliers inside one call of the hook. Their cause is UNPROVEN, and they do not affect the counts.
- **Wall time** (seconds per pass) was 53.2–62.6 s across runs. It is not used for any claim: the hook syncs after every call.

**Decision.**
- **C12 is resolved: the trim fully explains "11 of 12".**
- The **trimmed shape goes into the policy table and the reserve list** as its own row: last QSA layer (il 47), n_q = 512 (generally `n_tokens − q_first` ∈ [1, 512]) at any ub > 512, n_kv up to the context, maskless selected-key, ids width 2051, every K/V type. Today it runs on the `qsa_decode`/row path. That makes the row kernel part of prefill performance for every last-token-only prefill (perplexity's first half measured here; llama-bench and server code-verified).
- The gate and capture recipes that need all 12 layers on the G=4 prefill path must keep `LLAMA_FN_LAST_ATTN_TRIM=0` (as C3/C5 did). Perf numbers taken with the default trim must count the decode-path layer separately (C1 row T).
- The trim does not change perplexity (identical PPL in all 10 runs). The rewrite can therefore keep it unchanged, as plans W/X/Y/Z propose.

**Status:** resolved. Artifacts: `~/dev/wt_lab/c12/{runA_trim.sh,runT_trim.sh,logs/}` on strix-002, with a copy at `final219/c12/`. No large temporary files were created.

## C13

**Question.** What causes the PR's q8_0 regression at ub512 pp4096 d0? The author blames the qsa_prefill repack (LEDGER L12: comment 6563853593; PR description "ub 512, depth 0 … pp4096 −3%"). FACTS and plans W/X/Y say that n_q ≤ 512 goes to qsa_decode. Decides: is the fix in the work-split policy or in the prefill loader, and what correction goes on #219.

**Answer: the cause is the routing, not the repack.** At ub512 every QSA ubatch has n_q = 512, which the dispatcher hands to `qsa_decode` (the per-query WMMA kernel). `qsa_prefill` and its pack kernels are **never launched**: 0 launches in every PR run, on text and on random tokens, with `QSA_DIRECT` auto and with `QSA_DIRECT=0` (pack forced).
- Where the time goes. In the 4 ubatches past the dense shortcut (n_kv 2560–4096), `qsa_decode_wmma_partial<true>` takes 63–66 ms per ubatch. Base's dense masked q8_0 FA takes 24–38 ms for the same ubatches.
- That FA difference is +136 ms per pp4096 pass. It covers the whole end-to-end loss: +126 ms of graph time on text, +138 ms on random.
- Forcing the repack in the PR binary changes nothing: −0.03 % on text, +0.04 % on random.
- Sending the same 512-query ubatches to `qsa_prefill` (G=4) instead turns the regression into a gain: +1.6 % on text and +2.0 % on random, against −3.6 % / −4.2 % for the PR. With the full K+V repack forced on that path it is faster still: +1.8 % / +2.3 %.

### Code verification (PR 94e086847 in `~/dev/wt_pr219v2`; base 13a279bfa in `~/dev/wt_pr219v2_base`)
- **Dispatch order.** `ggml/src/ggml-cuda/fattn.cu:1154` tries `qsa_decode` first and `:1158` tries `qsa_prefill` second.
- **qsa_decode takes q8_0 up to 512 queries.**
  - Its support check accepts q8_0 at `qsa-decode.cu:92` (`k->type != F16 && k->type != Q8_0`) and up to `q->ne[1] > QSA_DECODE_MAX_QUERIES` queries at `:94`.
  - `QSA_DECODE_MAX_QUERIES` is 512 (`qsa-decode.cuh:5`, unchanged from base).
  - The q8_0 WMMA requirements are at `:101-102`.
- **Base routes q8_0 to dense FA.** Base `qsa-decode.cu:92-93` rejects anything that is not f16, so base q8_0 takes the masked dense FA.
- **qsa_prefill needs at least 128 queries** (`qsa-prefill.cu:616`) and is only reached when qsa_decode declines.
- **Where the repack lives.** The pack exists only inside `qsa_prefill`: `qsa_pack_k`/`qsa_pack_v` launches at `qsa-prefill.cu:692` and `:698`, selected by `QSA_DIRECT` at `:682-686`.
- **94e086847 (Iggy Jackson, "q8_0 direct K/V reads in qsa_prefill") touches only `qsa-prefill.cu`** (`git show --stat 94e086847`: 1 file, +73/−10).
  - So that commit cannot change this shape, where qsa_prefill never runs.
  - The author's "−4.5 % → −3 % at pp4096" after it is therefore not caused by the commit. Its cause is UNPROVEN; likely setup noise, given our base sd of up to 0.5 % per process and the 1.3 % spread between our r3 and C10 deltas.

### Setup (strix-002, 2026-11-04, 17:56–18:57 CET; every GPU run under `flock ~/gpu.lock`; no build)
- **Binaries, all pre-built:**
  - BASE `~/dev/wt_pr219v2_base/build/bin` (13a279bfa).
  - PR `~/dev/wt_pr219v2/build/bin` (94e086847).
  - LAB `~/dev/wt_lab/build/bin`: 94e086847 plus the env-gated C1 lab diff, `GGML_QSA_DECODE_MAXQ`, `GGML_QSA_TRACE` (C1 setup).
  - Text runs use the C10-patched llama-bench (`~/dev/wt_lab/c10/bin_{base,pr}`, C10 setup). For LAB text runs, `LD_LIBRARY_PATH=~/dev/wt_lab/c10/bin_pr:~/dev/wt_lab/build/bin` loads the lab libraries under the patched bench (checked with `ldd`).
  - Text corpus check: `C10: corpus 877755 tokens … fnv1a 48eaad26c3165780` in all text .err files. At d0 the timed prompt is wikitext.
- **Variants:**
  - `base`.
  - `pr` (auto).
  - `pr_d0` = PR + `QSA_DIRECT=0` (qsa_prefill full K/V pack forced).
  - `g4` = LAB + `GGML_QSA_DECODE_MAXQ=0` (128 ≤ n_q ≤ 512 → qsa_prefill, G=4).
  - `g4d0` = g4 + `QSA_DIRECT=0`.
  - `lab0` = LAB with no env (control that LAB equals PR).
- **Common args.** `M=~/models/qwen3.8-flash-next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`, `ARGS="-m $M -ngl 99 -fa 1 -lzm on-direct -ctk q8_0 -ctv q8_0 -b 4096 -ub 512 -p 4096 -n 0 -d 0 --progress -o json"`.
- **Part K (kernel profile).** `flock ~/gpu.lock env <ENV> GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_OVL=0 GGML_CUDA_OP_PROF=1 /opt/rocm/bin/rocprofv3 --kernel-trace --output-format csv -d logs/K_<v>_<wl>_<k> -o kt -- <BIN>/llama-bench $ARGS -r 3`, plus `[C10_TEXT=~/data/wikitext-2-raw/wiki.test.raw,~/dev/wt_lab/c10/corp/code.txt]` for text.
  - Runs: 2 processes per (variant, workload), interleaved base, pr, pr_d0, g4 (`c13/campaign.sh`, PART=K). g4d0 is in `campaign2.sh`.
  - Each process gives 32 graphs: 4 passes × 8 ubatches, where pass 0 is llama-bench's warm-up. Every graph has 12 FLASH_ATTN_EXT ops.
- **Analysis** (`c13/c13an.py`).
  - OP_PROF FLASH_ATTN_EXT ms per graph: the median over timed passes 1–3.
  - Kernel families from the rocprofv3 trace: `qsa_decode*`, `qsa3_*|qsa_pack_*` (= qsa_prefill), `flash_attn*` (dense), plus the `dequantize_block_q8_0_f16` K/V converts directly in front of dense FA. FA ops are rebuilt from runs of consecutive FA kernels; 384 ops = 32 × 12 in every run.
  - Per-kernel-name totals per timed pass.
- **Part E (end to end, no profiler, graphs on).** `flock ~/gpu.lock env <ENV> <BIN>/llama-bench $ARGS -r 5`.
  - Order: 3 rounds × {text, rand} × {base, pr, pr_d0, g4, lab0} (`campaign.sh`, PART=E), then 2 rounds × {text, rand} × {base, g4d0} (`campaign2.sh`).
  - That gives 5 base processes per workload. Analysis: `c13/c13e2e.py`.
- **Exit status.** All 50 processes exited with rc=0 (`campaign*_{K,E}.out`), and no `.err` contains `error|abort|failed to load`.
- **Artifacts.** `strix-002:~/dev/wt_lab/c13/` (logs 402 MB; the largest trace is < 20 MB). Copies of the scripts, the summaries and the `*_summary.json` files are in `SP/final219/c13/`.

### Results K: FA ms per ubatch (12 QSA layers), q8_0 ub512 pp4096 d0. Median of 3 timed passes; process 1 / process 2.
| ubatch (n_kv) | base text | PR text | PR QSA_DIRECT=0 text | g4 text | g4d0 text | base rand | PR rand | g4 rand | g4d0 rand |
|---|---|---|---|---|---|---|---|---|---|
| 0–3 (512–2048) | 6.29 / 11.28 / 15.54 / 21.94 | 6.32 / 11.28 / 15.54 / 22.28 | 6.30 / 11.35 / 15.50 / 22.15 | 6.29 / 11.26 / 15.52 / 22.10 | 6.33 / 11.38 / 15.55 / 22.15 | 6.38 / 11.43 / 15.69 / 22.18 | 6.39 / 11.39 / 15.64 / 22.18 | 6.37 / 11.37 / 15.64 / 22.04 | 6.38 / 11.35 / 15.64 / 22.24 |
| 4 (2560) | 23.97 / 23.78 | 63.72 / 63.18 | 63.24 | 20.02 / 20.00 | 18.38 / 18.29 | 23.93 / 23.80 | 63.02 / 63.20 | 20.02 / 19.93 | 18.16 / 18.19 |
| 5 (3072) | 28.31 / 28.23 | 64.22 / 63.96 | 64.19 | 21.44 / 21.37 | 19.43 / 19.37 | 28.36 / 28.25 | 64.34 / 64.17 | 23.22 / 23.38 | 21.04 / 21.01 |
| 6 (3584) | 32.38 / 32.35 | 65.07 / 64.87 | 65.33 | 22.93 / 22.87 | 20.85 / 20.84 | 32.67 / 32.63 | 64.98 / 64.99 | 25.33 / 25.22 | 22.93 / 22.93 |
| 7 (4096) | 37.35 / 37.78 | 65.56 / 65.63 | 65.98 | 23.21 / 23.34 | 21.48 / 21.41 | 37.96 / 38.29 | 65.65 / 65.73 | 26.93 / 26.91 | 24.47 / 24.48 |
| **pass FA sum** (OP_PROF; process 1 / 2) | 177.2 / 177.4 | 314.0 / 313.2 | 314.0 / 312.3 | 142.9 / 142.9 | 135.4 / 135.1 | 178.6 / 178.5 | 313.9 / 313.5 | 150.9 / 150.7 | 142.1 / 142.3 |
| **pass graph-op sum** | 3379.8 / 3375.3 | 3504.5 / 3503.3 | 3513.4 / 3505.2 | 3372.6 / 3330.1 | 3321.6 / 3321.5 | 2803.2 / 2800.5 | 2935.6 / 2943.7 | 2770.2 / 2770.2 | 2758.6 / 2757.1 |

(Single values are process 1. pr_d0 rand is in `c13_K.txt` and matches PR rand within 0.6 ms per ubatch. The full per-pass lists are in `c13_K.txt` / `c13_K2.txt`.)

**Launches per ubatch and kernels** (rocprofv3, identical in every pass and process):
- **ubatches 0–3 (n_kv ≤ 2048, dense shortcut)** are the same in every variant: 12 × `flash_attn_ext_f16<256,256,16,4,…>` plus 24 × `dequantize_block_q8_0_f16`.
- **ubatches 4–7:**
  - **base:** 12 `flash_attn_ext_f16` + 24 `dequantize_block_q8_0_f16`.
  - **PR and PR `QSA_DIRECT=0`:** 36 launches = 12 × (`qsa_decode_unique_ids`, `qsa_decode_wmma_partial<true>`, `qsa_decode_merge`). **0 launches of `qsa3_*`, `qsa_pack_k`, `qsa_pack_v`** (`qsa_prefill` column = 0 in every PR row of `c13_K.txt`).
  - **g4:** 48 = 12 × (`qsa3_rows_kernel`, `qsa_pack_v<true>`, `qsa3_attn_kernel<true,8,false,true>`, `qsa3_merge_kernel`). Auto mode reads K directly and packs V.
  - **g4d0:** 60 = 12 × (rows, `qsa_pack_k<true>`, `qsa_pack_v<true>`, `qsa3_attn_kernel<false,8,false,false>`, merge).
- **Per timed pass, PR − base** (mean of 2 processes each, top kernels):

  | kernel | text | rand |
  |---|---|---|
  | `qsa_decode_wmma_partial<true>` (48 launches vs 0) | +253.1 ms | +253.1 ms |
  | `flash_attn_ext_f16` (48 launches vs 96) | −119.2 ms | −120.1 ms |
  | `k_argsort_f32_i32` (selected-key ids) | +8.9 | +9.0 |
  | `top_k_wg_kernel` (base mask path) | −4.6 | −4.9 |
  | `qsa_decode_merge` | +2.9 | +2.9 |
  | `qsa_expand_complete_blocks_512` | +2.8 | +2.9 |
  | `qsa_decode_unique_ids` (absolute) | 1.1 | 1.2 |
  | **all kernels** | **+132.1** | **+137.1** |

  - The FA kernels (+253.1 − 119.2 + 2.9 + 1.1 ≈ +138) carry the whole delta. The selection overhead outside FA nets to about +7 ms.
- **`QSA_DIRECT=0` − auto on PR:** +0.6 ms (text) and −4.8 ms (rand) over all kernels. The biggest single changes are ±3 ms in unrelated MoE kernels, which is noise, and `qsa_decode_wmma_partial` moves −0.20 ms. **The repack switch has no effect at this shape.**
- **The repack itself, where it does run (g4d0).**
  - `qsa_pack_k` takes 0.62–0.65 ms and `qsa_pack_v` 0.78–0.83 ms per pass in total (48 launches each).
  - The packed `qsa3_attn_kernel<false,…>` takes 72.0 (text) / 78.4 (rand) ms per pass, against 80.6 / 88.5 for auto's direct-K `<true,…,Q8>`.
  - At d0, the repack therefore costs about 1.4 ms per pass and saves about 8–10 ms.

### Results E: end to end, t/s, pp4096 (avg of 5 samples per process)
| variant | text: per process | text mean (vs base) | rand: per process | rand mean (vs base) |
|---|---|---|---|---|
| base | 1222.1, 1220.4, 1216.1, 1220.5, 1213.9 | **1218.6** (sd 3.5, 0.28 %) | 1459.7, 1450.4, 1464.3, 1466.1, 1462.6 | **1460.6** (sd 6.2, 0.43 %) |
| pr | 1175.5, 1173.9, 1173.7 | 1174.4 (**−3.63 %**) | 1401.7, 1397.8, 1398.5 | 1399.3 (**−4.20 %**) |
| pr_d0 (`QSA_DIRECT=0`) | 1174.6, 1174.0, 1173.5 | 1174.0 (−3.66 %) | 1400.0, 1400.3, 1399.0 | 1399.8 (−4.17 %) |
| lab0 (control) | 1174.8, 1173.6, 1174.0 | 1174.1 (−3.65 %) | 1398.9, 1399.8, 1399.1 | 1399.3 (−4.20 %) |
| g4 (`DECODE_MAXQ=0`) | 1238.3, 1238.0, 1237.5 | 1237.9 (**+1.59 %**) | 1492.2, 1487.2, 1489.8 | 1489.7 (**+1.99 %**) |
| g4d0 (+ `QSA_DIRECT=0`) | 1240.7, 1240.9 | 1240.8 (+1.82 %) | 1495.5, 1491.9 | 1493.7 (+2.27 %) |

The raw sample lists are in `c13_E.txt`.

### Noise
- **End to end, between processes:**
  - base sd is 0.28 % on text and 0.43 % on random (5 processes). Every PR-family variant is ≤ 0.15 %.
  - The PR − base gap (−3.6 % / −4.2 %) is more than 8× the base sd.
  - pr vs pr_d0 vs lab0 differ by ≤ 0.04 %.
  - Two base samples are low outliers: 1390.3 (rand, E#2 sample 0) and 1196.4 (text, E2#2 sample 0). They are kept in the means.
- **Profile, process 1 vs process 2:**
  - Pass FA sums agree within 0.8 ms (≤ 0.5 %).
  - Per-ubatch FA medians agree within 0.6 ms.
  - Whole-graph sums agree within 1.3 % (one g4 text process ran 3411.8 / 3372.6 / 3337.3 ms across its passes; the second ran 3330 ± 2).
- **Consistency across methods.**
  - The e2e pass-time delta PR − base, from t/s, is 4096/1174.4 − 4096/1218.6 = 126.5 ms (text) and 117.6 ms (rand).
  - The profiled graph-op delta (graphs off) is +126.4 ms (text) and +137.8 ms (rand).
  - The profiled FA delta is +136.3 / +135.2 ms.
- **Cross-check against earlier data.**
  - C10 (text −3.70 %, random −4.66 %), r3 (L55, −4.7 %) and the C1 e2e (−4.2 %) agree with this run to within 0.5 points.
  - FACTS "dense 37.9 → QSA 65.2 ms per ubatch" matches ubatch 7 here (37.4–38.3 vs 65.6–66.0).
  - The earlier `QSA_DIRECT` sweep also shows no repack effect: `~/dev/wt_pr219v3_logs/perf/conf/pr_{auto,d0,d1,d2}.json` pp4096 gives 1380.0 / 1378.1 / 1376.1 / 1375.3, against base 1440.8 / 1441.6.

### Decision
1. **C13 is resolved against the author's attribution.** The ub512 d0 q8_0 regression comes from routing n_q ≤ 512 to `qsa_decode`. That is the work-split policy:
   - the dispatch order at fattn.cu:1154;
   - the 512-query bound at qsa-decode.cuh:5;
   - the q8_0 admission the PR added at qsa-decode.cu:92.
   - The qsa_prefill repack never runs at this shape (0 launches), and forcing it changes nothing (±0.04 %).
2. **The fix goes in the work-split policy, not the prefill loader.**
   - Send 128 ≤ n_q ≤ 512 to the G=4 kernel. This is consistent with C1 decision 3 (`QSA_DECODE_MAX_QUERIES` → 127).
   - Measured here, that turns −3.6 % / −4.2 % into +1.6 % / +2.0 % over base, on text and on random.
   - The text gate from C10 (≥ base, CI lower bound ≥ −0.5 %) is met by g4 at this shape: 1237.9 vs 1218.6.
3. **The loader matters for the rewrite, but in the opposite direction.**
   - At d0 / n_kv ≤ 4096, the full pack (g4d0) beats auto's direct-K read: FA 135 vs 143 ms per pass on text, +0.2–0.3 % e2e.
   - So the direct-vs-pack heuristic should be re-tuned on the new G=4 kernel at shallow n_kv, and not hard-coded to direct.
   - This is an input for the loader WP, not a cause of C13.
4. **Correction to post on #219.**
   - "At ub 512 every QSA ubatch has 512 queries and goes to qsa_decode (fattn.cu:1154, qsa-decode.cuh:5). qsa_prefill and its pack never launch there (rocprofv3: 0 launches; `QSA_DIRECT=0` vs auto −0.03 % / +0.04 %)."
   - "So 94e086847, which only touches qsa-prefill.cu, cannot move this number."
   - "The loss is `qsa_decode_wmma_partial<true>`: 63–66 ms per ubatch vs 24–38 ms for master's dense q8_0 FA at n_kv 2560–4096. Routing those ubatches to the G=4 prefill kernel gives +1.6 % (text) / +2.0 % (random) over master."

**Status:** resolved. No large temporary files were created (logs total 402 MB, no file > 1 GB). Nothing was built.

## C14

**Question.** Should bf16 K/V be converted to f16 with a range check (plans W/X/Z), or computed natively in bf16 (plan Y)? Decides: the bf16 trait of the sparse kernel, and whether bf16 drops to the generic tier on Vulkan (LEDGER §C14).

**Answer: convert, at LDS staging, with RNE (bit-equal to `ggml_get_to_fp16_cuda(BF16)`). bf16 then follows exactly the f16 numeric contract: Q f16, P f16, f16 WMMA, f32 accumulation. bf16 stays in the fast tier on HIP and Vulkan. Do not use native bf16 compute.**
1. **Range.** The check is met with a wide margin.
   - Over all 12 QSA layers, cells 0..135167, on text and on random tokens: max\|K\| = 15.875, max\|V\| = 36, max\|indexer K\| = 12.5625.
   - That is 1820× below the f16 limit 65504, and 113× below W/X's threshold of 4096 (2^12).
   - 0 elements are ≥ 4096, 0 overflow f16, and 0 are non-finite.
   - bf16 → f16 is exact for every element in [2^-14, 65504]. Only some of the f16-subnormal elements (< 2^-14, at most 9.9e-5 of all elements) round. The inexact ones are at most 3.0e-6 (K) and 8.3e-6 (V) of the elements, and their abs error is at most 2.98e-8 = 2^-25.
2. **Precision.** Native bf16 is **7.9× less accurate** than conversion on the real QSA selection.
   - The reason is that native bf16 WMMA forces Q and P into bf16 (8-bit mantissa) instead of f16 (11-bit).
   - Mean output rel-L2 against an f64 reference on the same stored K/V: CONV 6.7e-5 (text) / 4.6e-5 (random) vs NATIVE 5.3e-4 / 3.6e-4.
3. **Throughput.** Native buys nothing on gfx1151.
   - Raw bf16 WMMA = f16 WMMA: 50.3–50.8 TFLOP/s each, bf16/f16 = 0.998–1.000.
   - An LDS-staged attention microbench gives the same time for f16, bf16-convert (rtz or RNE) and bf16-native, within noise (≤ ±1.2 %).
   - The trap is conversion in registers per wave (no staging): that is 2.7× slower than f16 when K/V are L2-resident. **The bf16 trait must therefore be "LDS-staged".**
4. **Vulkan.** RADV on strix-002 has no bf16 shader type and no bf16 coopmat config (C11: `bf16: 0`, and the C11 bullet "bf16: NO as a shader type" citing ggml-vulkan.cpp:6306).
   - A native design would put bf16 on the scalar path there, which is plan Y's own fallback row "`sa_attend_scalar.comp` … bf16 without bf16-coopmat" (plan-v2-Y.md:215; X11 at :299).
   - Convert-at-staging keeps bf16 on the same coopmat f16 kernel as f16 and q8_0. **bf16 does not drop to the generic tier.**
   - The Vulkan cost of that conversion is **UNPROVEN** (not measured; HIP microbench only).
5. **Bound on the convert cost.** Converting bf16 at staging costs less than the q8_0 dequant, which the rewrite has to make fast anyway: Q8_LDS is 1.49× (L2-resident) and 1.07× (gather) the f16 time in the same microbench.

### Code verification (PR 94e086847 in `~/dev/wt_pr219v2`)
- **bf16 caches are not rotated.** Rotation is on only for `ggml_is_quantized(type)` (LEDGER L45, `src/llama-kv-cache.cpp:323-375`). Measured: `llama_kv_cache: n_rot_k = 0 … n_rot_v = 0` (`logs/cap_text.log`). So the dumped values are the raw cache values. "After rotation" in X PF-8 / W E9 / Z E-P7 is moot for bf16.
- **Today, HIP bf16 QSA takes the masked route.**
  - Log: `QSA attention: masked top-k (no selected-key kernels for this backend and cache type) (K bf16, V bf16)` (`logs/cap_text.log`).
  - Code: the QSA kernels accept only F16/Q8_0 (`ggml/src/ggml-cuda/qsa-decode.cu:92-93`, `qsa-prefill.cu:613-614`).
- **The dense FA it reaches already converts bf16 to f16, with no range check.**
  - Kernel choice: on RDNA3.5 at D=256, prefill picks `BEST_FATTN_KERNEL_MMA_F16` (`fattn.cu:723-731`), and that sets `need_f16_K/V` (`fattn.cu:774-777`).
  - The conversion: `launch_fattn` converts the whole K/V with `ggml_get_to_fp16_cuda(K->type)` (`fattn-common.cuh:1031-1050`). For BF16 this is `convert_unary_cont_cuda<nv_bfloat16>` (`convert.cu:695`, `:766-767`) → `ggml_cuda_cast<half>(nv_bfloat16)`, i.e. `__bfloat162float` followed by an implicit half RNE (`convert.cuh:35-41`). So an overflow gives ±inf, not saturation.
  - Plan Z's numeric contract ("bf16 saturates at ±65504", plan-v2-Z.md:86) does **not** match this existing behaviour.
  - The vec (decode) kernel reads bf16 directly as f32 (`vec_dot_fattn_vec_KQ_bf16` at `fattn-common.cuh:118-145`, selected at `:639-640`; `dequantize_V_bf16` at `:661-662`).
- **Today, Vulkan bf16 FA runs in f32 on the scalar path.**
  - `FA_COOPMAT1 && BF16 && !coopmat_bf16_support → FA_SCALAR` (`ggml-vulkan.cpp:1473-1475`).
  - For `bf16_kv` the scalar pipeline is the fp32 SPIR-V `flash_attn_f32_f16_fp32` (`ggml-vulkan.cpp:2314-2322`), with dequant via `bf16_to_fp32` (`vulkan-shaders/flash_attn_dequant.glsl:118-119`).
  - This was upstream's choice (6e093b80e, "vulkan: add Flash Attention support for BFloat16 KV cache (#23420)"). The sparse kernel does not have to copy it.
- **gfx1151 has no f32→bf16 convert instruction.**
  - The software RNE compiles to `v_bfe_u32 + v_add3_u32 + v_lshrrev_b32 + v_and_or_b32` per element, 8 of each per 8 P values (`logs/isa_attn_counts.txt`, mode 2).
  - f32→f16 is one `v_cvt_f16_f32` per element.
  - bf16→f16 is `v_lshlrev_b32`/`v_and_b32` + `v_cvt_pk_rtz_f16_f32` per pair, i.e. 1.5 VALU per element (mode 1: 132 + 127 + 128 for 256 elements).
  - `v_wmma_f32_16x16x16_bf16` exists and is what the bf16 builtin emits (`logs/isa_wmma_counts.txt`).

### Setup (strix-002, 2026-11-04 19:02–19:50 CET; every GPU run and build under `flock ~/gpu.lock`; I was the only GPU user)
- **Where the code lives.** Worktree `~/dev/wt_lab` (detached at 94e086847; the C1–C13 lab diffs are untouched and none of their env switches is set). C14 tools are in `~/dev/wt_lab/c14/`.
  - They link read-only against the unmodified PR libraries `~/dev/wt_pr219v2/build/bin`.
  - Copies are in `SP/final219/c14/` and `SP/final219/c14/remote/` (sources, `run.sh`, logs).
  - md5 local = remote: `c14_capture.cpp` 525d3b84…, `c14_analyze.cpp` edf99676…, `c14_wmma.cu` 3b9f067d…, `c14_attn.cu` 79c6e2f7…, `c14_attn2.cu` d7a3035f…, `isa_count.py` cebc539f….
- **Builds.**
  - `g++ -O2 -std=c++17 c14_capture.cpp -I$P/include -I$P/ggml/include -L$P/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$P/build/bin` with `P=~/dev/wt_pr219v2`.
  - `g++ -O3 -march=native -fopenmp c14_analyze.cpp`.
  - `/opt/rocm/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 c14_{wmma,attn,attn2}.cu`.
  - ISA from the same with `--save-temps`, counted by `isa_count.py` (innermost loop body). `-Rpass-analysis=kernel-resource-usage`: attn 194–198 VGPR, occupancy 7, no scratch; attn2 188–190 VGPR, occupancy 8, 16 KiB LDS, no scratch, identical across modes.
- **Capture (`run.sh`).**
  - `flock ~/gpu.lock env LLAMA_FN_LAST_ATTN_TRIM=0 ./c14_capture $M <wiki.test.raw|RANDOM> 135168 4096 dump_<src>.bin bf16 131072`, with `M=~/models/qwen3.8-flash-next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`.
  - Context: -ctk/-ctv bf16, FA on, n_ctx 139264, b = ub = 4096, ngl 99.
  - Text = the first 135168 wikitext-2 test tokens. Random = llama-bench's generator as one stream (`srand(1)`, `rand() % n_vocab`, BOS first).
  - At the ubatch with pos0 = 131072 (n_kv 135168), the eval callback dumps every `cache_*` bf16/f16 tensor that a node of that graph reads, rows 0..135167. That is 36 tensors: `cache_k_l{3..47}`, `cache_v_l*` and `cache_idx_k_l*`, 12 each.
  - Each of the 12 FLASH_ATTN_EXT nodes logged `bf16 K=cache_k_lN(256,135168,2,1) … mask=f16 ids=1 scale=0.0625 softcap=0`.
  - It also dumps 64 sampled query rows (q = 63, 127, …, 4095) × 24 heads of Q, their f16 mask rows (the actual top-k selection), and max\|Q\| over the full Q tensor.
  - Because the dump at 131k holds every cell, the range covers all depths 0..135167.
  - Both runs: `EXIT 0`, `C14 done pos0=131072 fa=12 caches=36`, wall 6:54 / 6:50, ROCm0 compute buffer 7186.36 MiB (`logs/cap_*_grep.log`).
- **Analysis (`c14_analyze dump.bin`), all on the CPU.**
  - **PART1:** per tensor, max\|x\|, counts ≥ 2^12 / ≥ 2^14 / f16-overflow / non-finite / f16-subnormal / inexact under RNE / flushed to 0, max conversion error, and a log2 histogram.
  - **PART2:** attention per (sampled query, head), 1536 rows per layer, in three variants:
    - REF: f64, Q·scale in f32, the stored bf16 K/V exactly.
    - CONV: Q·scale, K, V and P rounded to f16, f32 accumulation.
    - NATIVE: Q·scale and P rounded to bf16 (RNE), K/V as stored, f32 accumulation.
  - PART2 mask modes: **SEL** = the dumped FA mask (≈ 2051 visible cells, the real QSA selection), and **CAUSAL** = all cells ≤ query position (≈ 133152 cells, a dense stress case).
  - Metric: per row, rel-L2 of the 256-d output against REF.
- **Microbenches** (3 processes each, 10 timed reps per mode after 1 warm-up rep, modes interleaved round-robin inside a process).
  - **`c14_wmma 10`:** raw WMMA, 8 independent accumulators per wave, 320 workgroups × 256 threads × 4096 iterations. It also has synthetic conversion-in-loop variants.
  - **`c14_attn 10` (direct registers):** per wave one 16-query row tile; per 16-cell chunk it gathers K rows and V^T rows straight into WMMA operands, then QK (D/16 WMMA), exp, P pack and PV (D/16 WMMA).
  - **`c14_attn2 10` (LDS-staged):** a workgroup = 4 row tiles sharing one selection. 128 threads stage K [16 × D] and V^T [D × 16] into double-buffered LDS, converting on the way, with one barrier per chunk.
  - **Common to attn and attn2:**
    - D = 128 per wave (the model has 256: a 2-wave D split whose QK exchange is omitted).
    - 2048 selected cells per row tile, 6144 row tiles (24 heads × 4096 queries / 16), from a 135168-cell, 2-kv-head cache.
    - Selection SHARED: all row tiles of a head read the same 2048 cells, so K/V are L2-resident. Selection GROUP4: each 4-tile group reads its own random 512 blocks of 4 cells.
    - These are proxies, not the real kernel. The P fragment is not permlane-exchanged and there is no running max. The instruction mix that differs between modes is kept.
    - Modes: F16 (f16 cache); CONV (bf16 → f16, `v_cvt_pk_rtz`); CONVRNE (bf16 → f16 RNE, attn2 only); NATIVE (bf16 WMMA, Q bf16, P bf16 RNE in software); NATIVE_T (P bf16 truncated, attn only); Q8 (attn2 only: int8 + one f16 scale per 32 K-dims / per 4-cell V run, dequantized with the PR's exact trick `qsa-decode-wmma.cuh:9-12`).

### Results 1: range of the stored bf16 values (`remote/logs/summary_part1.txt`, from `an_{text,random}.txt`)
| corpus | cache | elements | max\|x\| (layer) | ≥ 2^12 | f16 overflow | non-finite | f16-subnormal (< 2^-14) | inexact bf16→f16 | flushed to 0 | max abs error | margin 65504/max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| text | K | 830,472,192 | 15.875 (l47) | 0 | 0 | 0 | 30,861 (3.7e-5) | 2,495 (3.0e-6) | 17 | 2.98e-8 | 4126× |
| text | V | 830,472,192 | 36 (l27) | 0 | 0 | 0 | 74,741 (9.0e-5) | 6,191 (7.5e-6) | 33 | 2.98e-8 | 1820× |
| text | indexer K | 207,618,048 | 12.5625 (l27) | 0 | 0 | 0 | 12,901 (6.2e-5) | 0 | 0 | 0 | 5214× |
| random | K | 830,472,192 | 15.375 (l47) | 0 | 0 | 0 | 29,665 (3.6e-5) | 2,487 (3.0e-6) | 17 | 2.98e-8 | 4260× |
| random | V | 830,472,192 | 33.75 (l27) | 0 | 0 | 0 | 82,738 (9.9e-5) | 6,904 (8.3e-6) | 39 | 2.98e-8 | 1941× |
| random | indexer K | 207,618,048 | 11.9375 (l27) | 0 | 0 | 0 | 11,895 (5.7e-5) | 0 | 0 | 0 | 5487× |

The 99.99th percentile of \|x\| is ≤ 2^3 in every tensor. max\|Q\| ≤ 14.17 and max\|Q·scale\| ≤ 0.886 in every layer (PART2 lines), so Q in f16 is safe too.

### Results 2: per layer, range and output error (SEL = the real selection; mean rel-L2 over 1536 rows)
| layer | max\|K\| text / rand | max\|V\| text / rand | max\|idx K\| text / rand | max\|Q\| text / rand | CONV text / rand | NATIVE text / rand | NATIVE/CONV text / rand |
|---|---|---|---|---|---|---|---|
| 3 | 7.28 / 7.13 | 6.5 / 5.03 | 5.41 / 5.38 | 9.84 / 10.17 | 9.05e-5 / 8.99e-5 | 7.16e-4 / 7.01e-4 | 7.91 / 7.80 |
| 7 | 8.06 / 8.19 | 10.63 / 10.69 | 8.63 / 8.75 | 9.05 / 7.78 | 8.60e-5 / 6.84e-5 | 6.78e-4 / 5.38e-4 | 7.89 / 7.88 |
| 11 | 8.44 / 7.34 | 17.13 / 16.5 | 10.0 / 9.75 | 8.76 / 8.07 | 7.23e-5 / 3.60e-5 | 5.69e-4 / 2.84e-4 | 7.87 / 7.89 |
| 15 | 9.75 / 7.97 | 12.31 / 8.13 | 8.63 / 5.72 | 7.78 / 7.12 | 4.77e-5 / 3.46e-5 | 3.76e-4 / 2.75e-4 | 7.89 / 7.96 |
| 19 | 9.75 / 8.06 | 24.63 / 14.5 | 10.81 / 6.59 | 10.75 / 9.03 | 6.98e-5 / 3.46e-5 | 5.56e-4 / 2.76e-4 | 7.97 / 7.96 |
| 23 | 10.19 / 8.38 | 7.63 / 6.72 | 8.75 / 8.31 | 11.14 / 8.49 | 6.87e-5 / 4.77e-5 | 5.44e-4 / 3.90e-4 | 7.92 / 8.18 |
| 27 | 9.44 / 8.06 | 36 / 33.75 | 12.56 / 11.94 | 10.08 / 8.15 | 6.07e-5 / 3.40e-5 | 4.78e-4 / 2.75e-4 | 7.88 / 8.10 |
| 31 | 11.19 / 9.81 | 14.94 / 8.88 | 5.41 / 3.86 | 9.61 / 9.37 | 5.38e-5 / 3.83e-5 | 4.25e-4 / 3.01e-4 | 7.91 / 7.86 |
| 35 | 11.38 / 9.25 | 29.88 / 8.38 | 6.03 / 4.66 | 14.17 / 11.69 | 8.99e-5 / 4.95e-5 | 7.16e-4 / 3.87e-4 | 7.97 / 7.83 |
| 39 | 12.56 / 10.0 | 12.44 / 10.88 | 9.69 / 8.81 | 11.67 / 7.92 | 8.03e-5 / 5.34e-5 | 6.37e-4 / 4.32e-4 | 7.93 / 8.08 |
| 43 | 9.63 / 9.25 | 7.38 / 6.03 | 4.69 / 4.69 | 12.34 / 8.05 | 4.08e-5 / 3.97e-5 | 3.25e-4 / 3.20e-4 | 7.97 / 8.06 |
| 47 | 15.88 / 15.38 | 10.06 / 7.47 | 9.06 / 9.44 | 13.31 / 10.84 | 4.40e-5 / 2.17e-5 | 3.55e-4 / 1.58e-4 | 8.07 / 7.30 |
| **all** (mean of layer means; max row) | | | | | 6.70e-5, max 6.0e-4 / 4.56e-5, max 6.5e-4 | 5.31e-4, max 4.3e-3 / 3.61e-4, max 4.9e-3 | **7.93 / 7.92** |

- **CAUSAL (dense stress, ≈ 133k cells).** CONV 7.70e-5 / 3.71e-5 vs NATIVE 4.76e-4 / 2.11e-4 (ratio 6.19 / 5.69). The outlier is layer 47: CONV 1.24e-4 / 1.26e-4, p99 1.3e-3 on random, ratio 1.37–2.53.
  - The probable cause is P underflow in f16: p < 2^-24 flushes to 0 over 133k cells, while the f32 row sum keeps them. That is UNPROVEN.
  - It would hit today's f16 caches the same way, and it does not appear on the real selection (SEL l47: 4.4e-5 / 2.2e-5).
- **Reproducibility of the analysis.** An 8192-token smoke capture (`logs/smoke_an.txt`, pos0 4096) gives the same picture: max\|V\| 36, NATIVE/CONV 7.83–8.16 per layer.
- **What the ratio means.** NATIVE/CONV ≈ 8 is 2^3, the mantissa gap between bf16 (8 bits) and f16 (11 bits), i.e. the expected cost of the Q and P rounding.

### Results 3: microbenches (gfx1151, `device AMD Radeon 8060S Graphics gfx1151 CUs 20 clock 2900000 kHz`)
**Raw WMMA, TFLOP/s, median of 10 per process (process 1 / 2 / 3; `logs/wmma_r*.txt`):**
| kernel | r1 | r2 | r3 |
|---|---|---|---|
| f16 WMMA | 50.84 | 50.29 | 50.28 |
| bf16 WMMA | 50.76 | 50.28 | 50.27 |
| synthetic: bf16→f16 RNE per element + f16 WMMA | 26.49 | 26.39 | 26.33 |
| synthetic: bf16→f16 `pk_rtz` + f16 WMMA | 44.76 | 42.12 | 42.20 |
| synthetic: f32→bf16 RNE (software) + bf16 WMMA | 17.09 | 17.06 | 17.04 |
| synthetic: f32→f16 (`v_cvt`) + f16 WMMA | 27.91 | 27.91 | 27.90 |

- Within-process spread for the two raw kernels: min–max 47.4–53.8.
- The synthetic rows convert 16 elements per WMMA, an upper bound on the VALU overhead. The compiler shared part of the `pk_rtz` work across accumulators (22 cvt for 64 pairs), so that row is optimistic.

**Attention proxy, ms per launch, median of 10 (process 1 / 2 / 3), sd within process in brackets.**

| mode | direct SHARED (`attn_r*`) | direct GROUP4 | LDS SHARED (`attn2_r*`, then `attn2b_r*`) | LDS GROUP4 |
|---|---|---|---|---|
| F16 | 14.73 / 15.53 / 12.30 [1.3–2.2] | 54.59 / 59.61 / 58.03 [3.6–10.2] | 10.73 / 10.88 / 10.97; 10.96 / 10.75 / 10.74 [0.18–0.23] | 17.26 / 17.14 / 17.29; 17.24 / 17.28 / 17.26 [0.13–0.45] |
| CONV (bf16→f16 `pk_rtz`) | 41.21 / 40.76 / 40.81 [0.5–0.8] | 48.61 / 47.69 / 48.75 [1.8–2.7] | 10.93 / 10.75 / 10.86; 10.62 / 10.58 / 10.59 [0.11–0.27] | 17.35 / 17.37 / 17.39; 17.47 / 17.39 / 17.30 [0.17–0.97] |
| CONVRNE (bf16→f16 RNE) | — | — | 10.88 / 10.83 / 10.62 [0.24–0.27] | 17.33 / 17.40 / 17.37 [0.20–0.63] |
| NATIVE (bf16 WMMA, P RNE) | 17.26 / 16.83 / 15.53 [1.2–2.3] | 61.30 / 58.74 / 58.22 [5.1–6.2] | 10.83 / 10.84 / 10.56; 11.09 / 10.43 / 10.69 [0.22–0.39] | 17.36 / 17.32 / 17.32; 17.74 / 17.33 / 17.31 [0.08–0.69] |
| NATIVE_T (P truncated) | 18.27 / 19.03 / 16.01 | 62.48 / 58.67 / 60.77 | — | — |
| Q8 (q8_0-like dequant) | — | — | 16.30 / 16.20 / 15.46; 16.11 / 15.76 / 16.44 [0.50–0.81] | 18.50 / 18.43 / 18.36; 18.64 / 18.27 / 18.37 [0.15–0.36] |

- **LDS-staged, ratio to F16** (median of the 6 process medians):
  - SHARED: CONV 0.988, CONVRNE 1.001, NATIVE 0.995, Q8 1.493.
  - GROUP4: CONV 1.007, CONVRNE 1.006, NATIVE 1.004, Q8 1.066.
  - Every bf16 mode is within ±1.2 % of F16. The within-process sd is 0.1–0.4 ms, i.e. 1–2.5 % SHARED and 0.5–2.6 % GROUP4.
- **Direct registers, ratio to F16.**
  - SHARED: CONV 2.8× (41.0 vs 14.7 ms; the F16 run itself is noisy, sd up to 2.2 ms). The ISA has +383 VALU per 16-cell chunk for 256 converted elements per lane (`isa_attn_counts.txt`: 525 vs 142 non-WMMA VALU, 16 WMMA).
  - SHARED: NATIVE ≈ 1.1× (+8 VALU per chunk).
  - GROUP4 (memory-bound): CONV is 0.85× F16 and NATIVE ≈ 1.03×. The F16 GROUP4 runs are the noisiest (sd up to 10 ms), so that ordering is not significant. Why CONV is lower is UNPROVEN.
- **LDS-staged ISA per chunk** (`isa_attn2b_counts.txt`): F16 77, CONV 125, CONVRNE 141, NATIVE 85 and Q8 156 non-WMMA VALU, for the same 16 WMMA. Each thread converts only 32 elements, because the 4 waves share the stage.

### Decision
- **bf16 trait = convert to f16 while staging K/V into LDS, RNE, bit-equal to `ggml_get_to_fp16_cuda(BF16)`.** It shares the f16 instance's compute path (Q f16, P f16, f16 WMMA/coopmat).
  - Freeze the staging strategy as LDS-staged for bf16 in the loader trait (X PF-3 / Z E-P3 / W E5). Direct-register conversion is 2.8× slower on L2-resident K/V.
  - No bf16 WMMA instance, and no plan-Y "ctype = bf16".
- **Range check: satisfied offline. No runtime check is needed for this model.**
  - Measured max 36 vs W E9's 4096 bar and Z E-P7's 16× margin (actual 1820×).
  - Define overflow in the numeric contract as to_fp16 does it today (RNE, ±inf), not as "saturates at ±65504" (Z §numeric contract). Pin that with a test-backend-ops case that has \|K\| > 65504.
  - Other models with QSA layers are not measured: UNPROVEN.
- **Vulkan: bf16 stays in the fast tier** via the same convert-at-staging into the f16 coopmat path (dequant as in `flash_attn_dequant.glsl`). It does not follow dense FA's `FA_SCALAR`/fp32 choice.
  - The Vulkan cost of the conversion is UNPROVEN and must be checked by the Vulkan perf gate (C11: bf16 masked route today 0.32×/0.40× of f16).
- **Dense FA caveat (for the dense shortcut / MTP).** The HIP dense path already converts bf16→f16 without saturation. Vulkan dense FA runs bf16 in f32. So bf16 output is not bit-identical across backends today. That is already the case and is not introduced by the rewrite.

### Large files
`dump_text.bin` and `dump_random.bin` (3.96 GB each) and the ISA intermediates are deleted. `~/dev/wt_lab/c14` now holds 18 MB. To recapture, use `run.sh`.

## C15

### C15 draft (written before the ub1024 deep-image run and the llama-bench trim runs; superseded where it differs, see "Corrections" in the final section below)

**Question.** Which route does `graph_reserve` build? What causes 510.2429 vs 510.2112 MiB (L27)? Does any runtime route need more memory than the reserve?

### Setup
- Build: strix-002 `~/dev/wt_lab/c15/tree`, a git worktree at PR head 94e086847, plus a trace-only patch (`~/dev/wt_lab/c15/c15_trace.patch`, 216 lines, generated by `~/dev/wt_lab/c15/patch.py`). The patch does nothing unless `GGML_C15=1` is set. With it set, it logs:
  - `C15A`: every gallocr plan, with the planned and current size of each buffer, and the tensor that sets each chunk's top (name, op, ne, offset, size). Code: ggml-alloc.c `reserve_n_impl` and `dyn_tallocr_alloc`.
  - `C15R`: why the allocator re-planned (`needs_realloc`).
  - `C15L`: every `graph_reserve` call, and every ubatch with the current buffer size vs `backend_buf_exp_size`. `!OVER` marks a buffer above the reserve.
  - `C15Q`: the QSA route chosen per layer: dense-shortcut or selection; blocks, scalar/maskless, key_limits, bias type and shape; and the QSA-cache state `cached`/`fast`.
  - With `GGML_C15_DUMP=<dir>` set, each growing plan is also dumped as a per-tensor TSV (buffer, offset, size).
- Build flags as in HARD RULES: `cmake --build build -j 16 --target llama-perplexity llama-server` (rc=0).
- Scripts: `~/dev/wt_lab/c15/c15_ppl.sh` (llama-perplexity on wikitext, real text), `c15_srv.sh` (the `~/repro_img/repro_kv.sh` flags: Signal-3.8 + mmproj + MTP sidecar, `-c 261888 -b/-ub 8192`) and `c15_np2.sh` (text model, `-np 2 -c 65536 -b/-ub 4096`, two concurrent requests that share a 60000-char wikitext prefix, then a third request with the same prefix). Every GPU run went through `flock ~/gpu.lock`.
- Commands: `~/dev/wt_lab/c15/c15_ppl.sh f16_ub512_r1 f16_ub512_r2 q8_ub512_r1 q8_ub512_r2 f16_ub4096 q8_ub4096 base_f16_ub512`. The perplexity runs use `-c 8192 -b 4096 --chunks 2 -ngl 99 -fa on -lm mmap -lzm on-direct -ctk/-ctv <kv> -lv 3`.
- Logs: `~/dev/wt_lab/c15/logs/{ppl_*,srv_*}.log`, with dumps in `logs/dump_<tag>/`.

### Raw numbers (destructor line `~llama_context: ... compute buffer size of X, does not match expectation of Y`, plus the C15 trace)
| run | ROCm0 runtime / reserve (MiB) | ROCm_Host runtime / reserve (MiB) | PPL |
|---|---|---|---|
| PR instr f16 ub512 r1 | 510.2429 / 510.2112 | 27.0274 / 25.1094 | 1.9428 |
| PR instr f16 ub512 r2 | 510.2429 / 510.2112 | 27.0274 / 25.1094 | 1.9428 |
| BASE 13a279bfa f16 ub512 (uninstrumented) | 510.2429 / 510.2112 | 27.0274 / 25.1094 | 1.9428 |
| PR uninstrumented f16 ub512 (L27, planB noise log) | 510.2429 / 510.2112 | 27.0274 / 25.1094 | 1.9428 |
| PR instr q8_0 ub512 r1 and r2 | 510.5085 / 510.4768 | 27.2930 / 25.3750 | 1.9532 |
| PR instr f16 ub4096 / q8_0 ub4096 | matches (no GROW plan after reserve) | matches | 1.9449 / 1.9519 |
| srv q8_0 A+B+C (image, MTP) | 3171.06 / 3171.06 (main), 2806.43 / 2806.43 (MTP) | **477.2315 / 403.1075** | answers 2749 / 12749 / 26999 |
| srv f16 B+C | 3170.80 / 3170.80 | **476.9659 / 402.8419** | 12749 / 26999 |
| srv q8_0 D (261333-token prompt, 1 image at pos 260284) | 3171.06 / 3171.06 (image plan 3117.19) | **819.0616 / 403.1075** | 11299 |
| np2 q8_0 (r2), q8_0 TRIM=0, f16 | no `!OVER` in any ubatch, no GROW after reserve, no mismatch line at exit | same | requests OK, 3rd request `cached 10767` |

Noise: the allocator is deterministic. In the repeated runs (f16 r1/r2, q8_0 r1/r2), every planned size matches to the byte and PPL matches to every printed digit. Between f16 and q8_0 the growth deltas are identical: ROCm0 +33280 B, host +2011136 B (ppl ub512); host +77724640 B (srv C). The instrumented and uninstrumented builds give the same sizes, so the trace patch does not perturb them.

### What graph_reserve builds (code + trace)
- `sched_reserve` calls `memory->init_full()` (src/llama-context.cpp:915) and `balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs)` (llama-context.cpp:3002). That gives a synthetic token ubatch with every position 0 and n_kv equal to the full cache.
- The resulting QSA route (`C15Q inputs` in ppl_f16_ub512_r1.log, reserve lines `pos0=0 posN=0`) is: selection (n_kv=8192 > width=2051), `blocks=1 scalar=1` (maskless), `key_limits=0` (the unbounded scorer, as intended by qwen4exp.cpp:1135-1141), `bias=i32`, `cached=0 fast=0` (`prep_blocks = n_blocks`). The cache flags follow from `qsa_prefix_matches` being false for the synthetic ubatch: src/llama-memory-hybrid-idx.cpp:1798-1810, qwen4exp.cpp:1495-1496.
- With `-np 2` the reserve builds n_seqs=2 with `n_stream=2 n_tps=2048` (srv_np2_q8_r2.log).

### Cause of 510.2429 vs 510.2112 MiB, and of the host 27.03 vs 25.11 MiB (ppl -c 8192 -ub 512)
1. **ROCm0 +33280 B: the QSA fast-cache route, which the reserve never builds.** From pos 5632 on, the runtime graph uses `cached=1 fast=1 prep_blocks=130` (qsa_fast needs `n_tokens <= 512`: llama-memory-hybrid-idx.cpp:1813-1822). That route adds per-layer GPU inputs (qwen4exp.cpp:1497-1507): for each of the 12 QSA layers, `members` i32[520], `positions` i32[520] and `rows` i64[130], i.e. 5200 B per layer. It also drops a reserve input of i32[8192].
   - These split inputs live for the whole GPU split. They raise the base of `result_output` (the logits, f32 [248320,512], 508559360 B, the top of ROCm0 in every plan).
   - Measured with `~/dev/wt_lab/c15/inp.py` on dump plans #1 (reserve) and #20 (last runtime GROW): the end of the input region moves from 21192960 to 21226240, i.e. +33280 B. That equals 535028480 − 534995200, exactly the L27 gap.
   - The gap opens in steps (plans #16-#20, `C15R ... 'leaf_108' size 24576 > size_max 22528` ... `32768 > 30720`): +512 B, then +8192 B per 512 tokens of n_kv, as the fast-route graph's per-n_kv inputs grow.
2. **ROCm_Host +2011136 B: the dense-shortcut route.** While n_kv <= top_k+r−1 = 2051 (qwen4exp.cpp:1871-1878), the layers attend dense through a full `attn_inp_kq_mask` f16 [n_kv, 512]. That mask grows to [2048,512] = 2097152 B (dump plan #8). The reserve, which is maskless at n_kv=8192, has no such mask (dump plan #1). Plans #5-#8 grow the host buffer by 512 KiB each time n_kv rises by 512. The net gain is 2097152 − 86016 B of reserve-only inputs (leaf_103/116/119).
3. **The same gap exists on base 13a279bfa** (base_f16_ub512 row). It predates PR #219. With ub4096 the gap disappears: the first ubatch already has n_kv=4096 > 2051, and qsa_fast refuses n_tokens > 512.

### Does any runtime route exceed the reserve?
- **Yes, the image (embd) ubatch, by a lot on the host, at any depth where n_kv is large.**
  - An embd ubatch has `ubatch.token == nullptr`, so block selection is ineligible (qwen4exp.cpp:1110). It takes the masked route (`C15Q inputs ... blocks=0 scalar=0 bias=f32[n_blocks,1024]`). That route uploads a full f16 KQ mask [n_kv, 1024] plus an f32 bias [n_kv/4, 1024].
  - At pos 140140 (5th image of request C), the host plan is 477.23 MiB vs a 403.11 MiB reserve (+74.12 MiB, f16 and q8_0 identical). The dump shows mask 297271296 B and bias 148635648 B.
  - At pos 260284 (request D), the host plan is 819.06 MiB (+415.95 MiB) from mask [261376,1024] = 535298048 B plus bias [65344,1024] = 267649024 B.
  - On ROCm0 the same image plan is 3117.19 MiB vs a 3171.06 MiB reserve, which leaves 53.87 MiB of margin at 261k for a 1024-token image. The top tensor is FLASH_ATTN_EXT with an alloc size of 560463872 B.
  - UNPROVEN: images with more tokens per ubatch (`--image-max-tokens` above 1024) would overrun ROCm0 too. Not measured.
- **Yes, slightly, for text:** the fast-cache route (+33280 B GPU) and the dense-shortcut route (+2 MiB host at ub512; it scales as 2048·ub·2 B, so up to 32 MiB at ub8192 when the first ubatch starts at n_kv <= 2051). The ub=8192 dense case is UNPROVEN on GPU: its plan did not exceed the reserve in the srv runs (no ROCm0 `!OVER`).
- **No overrun seen for:**
  - MTP context (ctx_type=1): 74 ubatches in srv_q8, 0 `!OVER`, including its masked image route at layer 48.
  - 2-slot shared prefix: n_seqs_unq=2 ubatches, 2-stream and 1-stream graphs, and prefix reuse `cached 10767`.
  - Trim: `LLAMA_FN_LAST_ATTN_TRIM` on (`q_first=3584` taken at layer 47 in np2) and off.
  - In the MTP server the trim is never taken: `embeddings_nextn` blocks it (qwen4exp.cpp:939-941), and srv logs show `q_first=0` everywhere.
- Anomaly, not reproduced: in `srv_np2_q8.log` (r1) the third request returned an empty body after about 7 min, and the server log shows no task for it. The rerun `srv_np2_q8_r2.log` with identical flags passed. Cause UNPROVEN.

### Decision
1. **Reserve-variant list for the rewrite.** In addition to text pp (ub, n_seq_max streams) and tg, the reserve must cover, or the design must remove:
   - (a) the dense-shortcut graph with its [min(n_ctx, top_k+r−1), ub] mask;
   - (b) the QSA fast-cache graph (ub <= 512, 12 × 3 inputs per layer);
   - (c) the embd/image ubatch at full n_kv.
   Recommendation: **remove (c) by design.** Give embd ubatches the maskless selection route instead of a full [n_kv, n_img] f16 mask plus an f32 bias. Otherwise the reserve needs a mask variant of roughly n_ctx·n_img·(2+1) B, which is about 800 MB host at 261k for a 1024-token image. Measured above, not extrapolated.
2. **Strict-reserve test.** After every ubatch, assert `ggml_backend_sched_get_buffer_size(backend) <= backend_buf_exp_size` per backend. The `~llama_context` comparison at src/llama-context.cpp:689-697 already computes this; the test turns it into a failure. Workloads:
   - (i) llama-perplexity `-c 8192 -b 4096 -ub 512` (catches a and b);
   - (ii) one image ubatch after ≥140k text tokens (catches c);
   - (iii) `-np 2` shared prefix with trim on and off;
   - (iv) an MTP server.
   Today PR fails (i) and (ii), and BASE fails (i) (base_f16_ub512 row). BASE on (ii) was not run with the trace: UNPROVEN. The test must therefore ship with the reserve fix in the same PR.

### C15 final

**Question.** Which route does `graph_reserve` build? What makes 510.2429 vs 510.2112 MiB (L27)? Does any runtime route (image/masked, trim, MTP, 2-slot shared prefix) need more compute memory than the reserve? Decides: the reserve-variant list and the strict-reserve test (LEDGER §C15).


**Corrections to the C15 draft above** (each re-checked against the logs in `~/dev/wt_lab/c15/logs/`):
- *"MTP context: 0 `!OVER`"* holds only at -ub 8192. At **-ub/-b 1024** the MTP context (ctx_type 1) overruns ROCm0 on the deep image ubatch, 1349.36 → 2558.52 MiB (`srv_srv_q8_D_ub1024.log`, plan #511, and the `~llama_context` warning).
- *"UNPROVEN: images … would overrun ROCm0 too"* is now **measured**, and the trigger is a smaller -ub, not more image tokens. At -ub 1024 the main context goes 1409.18 → 3034.61 MiB on ROCm0 with a 1024-token image at n_kv 253696 (same log, plan #510).
- *"Trim: … q_first=3584 taken at layer 47 in np2"* is wrong.
  - In np2, q_first > 0 is computed on output-free ubatches (`C15Q layer il=47 … q_first=3584 … n_outputs=0`), but the trim is **never taken**.
  - The once-per-process INFO line `LLAMA_FN_LAST_ATTN_TRIM: layer` is absent from `srv_np2_q8_r2.log`, `srv_np2_f16.log` and `srv_np2_q8.log` (count 0), while other INFO lines are present.
  - The trim was exercised only by llama-bench ub4096 (`bench_{q8_0,f16}_trim1_ub4096.log`, line present). There it adds 8224 B of host inputs over the reserve. At ub512 it can never fire (q_first = 0).
- The draft's cause for the image route (`ubatch.token`, qwen4exp.cpp:1110) is right. The final section now cites it.

**Answer.**
1. **`graph_reserve` builds exactly one QSA route.** That route is: text tokens, **all positions 0**, n_kv = the full cache, selection path, **maskless compact (I32 limits)**, unbounded scorer, the indexer cache present but **cached=0 / fast=0**, and **no trim**. With `-np 2` (non-unified KV, so n_stream=2) the pp reserve is maskless I32 and the tg reserve is the masked per-block F32 bias.
2. **The 510.2429 vs 510.2112 MiB mismatch is the per-layer fast-cache inputs, to the byte.** The reserve never builds them.
   - The exact difference is 33280 B = 535028480 − 534995200.
   - The 12 QSA layers each add `members` i32[520] + `positions` i32[520] + `rows` i64[130] (prep_blocks = (512+3)/4+2 = 130). On ROCm0 that is 62400 B of split-input copies.
   - The runtime graph drops the `blk_cells` i32[8192] input (32768 B), which those layers no longer read.
   - The 128-byte alignment padding is 12×(96+96+112) = 3648 B.
   - 62400 − 32768 + 3648 = **33280 B**.
   - Its ROCm_Host companion (27.0274 vs 25.1094 MiB) is the **dense-shortcut KQ mask**: f16 [n_kv ≤ 2048, 512], grown in 4 steps while n_kv ≤ top_k+r−1 = 2051. The reserve (n_kv = 8192) never takes the shortcut.
   - Both are **pre-existing**. Base 13a279bfa prints the identical pair, 510.2429/510.2112 and 27.0274/25.1094 (`logs/ppl_base_f16_ub512.log`). They are not from PR #219.
3. **Yes, one runtime route exceeds the reserve on the device by a large margin: the masked route of image (2-D position) ubatches.** Its inputs are O(n_kv × n_image_ubatch). In both the main and the MTP context:
   - **At -ub/-b 1024, c261888, image at n_kv 253696:**
     - ROCm0 main: **1409.18 → 3034.61 MiB (+1625.4 MiB, 2.15×)**.
     - ROCm0 MTP: **1349.36 → 2558.52 MiB (+1209.2 MiB)**.
     - ROCm_Host: 52.59 → 796.47 MiB.
     - The request still answers correctly (`'30099'`) because this UMA box can grow the buffer. Nothing reserved that memory.
   - **At -ub 8192** (the owner's command) the overshoot hides on ROCm0: the reserve is sized by the 8192-token pp graph, and the image ubatch plans 3117.19 of 3171.06 MiB at n_kv 261376 (53.9 MiB headroom). ROCm_Host still grows 403.11 → **819.06 MiB**.
   - The other unreserved routes are small:
     - dense-shortcut mask: ≤ 1.92 MiB host, ≤ 1.85 MiB ROCm0, at ub512.
     - fast-cache inputs: 33280 B ROCm0.
     - trim tile: 8224 B host.
   - The 2-slot runs, the MTP context on text, the bounded scorer and trim=0 never exceeded the reserve.
4. **Noise.** Buffer sizes are deterministic.
   - `ppl_f16_ub512_r1` = `r2` = base, to the byte. `ppl_q8_ub512_r1` = `r2`.
   - `np2_q8_r2` and the valid concurrent half of `np2_q8` agree (0 OVER).
   - The plan sizes repeat exactly.

**Decision.**
- **Reserve-variant list.** The only variant whose absence costs real memory is the **image / masked-route ubatch at full n_kv**. The rewrite must do one of two things:
  - (a) **Preferred: delete that route.** Make the maskless selected-key kernels accept 2-D image positions, so image ubatches stay on the I32 compact path, which is O(n_blocks + n_tokens). The masked route then no longer exists for HIP or Vulkan, in either the main or the MTP context.
  - (b) **Fallback: add that reserve variant to both contexts when mtmd is loaded.** The variant is an embd ubatch of min(n_ubatch, image chunk) tokens with 2-D positions, at n_kv = n_ctx.
  - The three small variants can be covered by one extra reserve pass with **sequential (non-degenerate) positions, n_outputs = 1, n_tokens = min(ub, 512)**, which builds the fast-cache and trim-tile inputs, plus one pass at **n_kv = top_k+r−1** (the dense shortcut). The alternative is a documented slack. They stay out of the plan only if the strict test below uses that slack.
- **Strict-reserve test.**
  - After every ubatch, assert `ggml_backend_sched_get_buffer_size(dev) ≤ backend_buf_exp_size[dev]` for every backend, ROCm_Host included. Use ≤, not the destructor's `==`.
  - Run it on:
    - ppl c8192 ub512, f16 and q8_0. Today it fails by 33280 B on ROCm0.
    - the deep-image request at ub1024. Today it fails by +1625 MiB on ROCm0 and +1209 MiB on ROCm0 in the MTP context.
    - np2, shared prefix.
    - llama-bench pp4096 @ d4096 with trim 1 and 0.
  - The C15 trace patch (`GGML_C15=1`, `!OVER` marker) is a working prototype of the check.

### Route that `graph_reserve` builds (code + trace)
- **Reserve ubatch.**
  - The ubatch is `balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs)` (`src/llama-context.cpp:3002`), with n_tokens = min(n_ctx, n_ubatch), n_seqs = n_seq_max, n_outputs = min(n_tokens, n_outputs_max) (`:897-898`, `:939`).
  - The sequence is pp, then tg, then pp again (`:943`, `:971`, `:993-996`).
  - The memory context is `memory->init_full()` (`:915`), so n_kv = the whole cache.
  - The model code itself states that the reserve positions are all 0 (`src/models/qwen4exp.cpp:1133-1138`, `degenerate_pos`).
- **Consequences in qwen4exp, PR tree.**
  - The shortcut is never taken because n_kv = n_ctx > width = 2051 (`:1877`).
  - blocks = 1 and scalar = 1 because text tokens have 1-D visibility, so bias = I32 (`:1453-1462`).
  - `score_key_limits` = {} (degenerate positions; `:1135-1141`).
  - storage = 1 but `qsa_prefix_matches` fails (positions are not `pos[0]+i`; `src/llama-memory-hybrid-idx.cpp:1798-1811`), so cached = 0 and fast = 0 (`qwen4exp.cpp:1494-1497`).
  - Trim needs `q_first > 0`, i.e. an output-free head of the ubatch; the reserve outputs every token (`:940-951`, `:1883`).
- **Measured.** The reserve-time `C15Q` lines are the ones with `pos0=0 posN=0`:

| run | reserve inputs (il=3) | cache |
|---|---|---|
| ppl c8192 ub512 | `n_kv=8192 n_blocks=2048 n_stream=1 n_tps=512 blocks=1 scalar=1 key_limits=0 bias=i32[3072]` (×2), plus tg `n_tps=1 bias=i32[2050]` (×4) and `n_tps=32` (×2) | `storage=1 cached=0 fast=0 prep_blocks=2048` |
| server c261888 ub8192 | `n_kv=261888 n_blocks=65472 n_tps=8192 scalar=1 key_limits=0 bias=i32[81856]` (×4), `n_tps=1` (×5), `n_tps=32` (×2) | `cached=0 fast=0 prep_blocks=65472` |
| np2 c65536 ub4096 | `n_kv=32768 n_stream=2 n_tps=2048 scalar=1 bias=i32[10240,2]` (×2); tg `n_tps=1 scalar=0 bias=f32[8192,1,2]` (×4); `n_tps=32 scalar=0 f32[8192,32,2]` (×2) | — |

- `graph_reserve` calls (`C15L`, ppl f16 ub512, in order):
  - `n_tokens=1` split_only ×3 and `n_tokens=32` split_only ×2. These are the `resolve_fused_ops` probes, `:712`.
  - Then 512 (alloc), 1 (alloc), 512 (alloc).
  - The server has the same pattern per context: ctx_type 0 = main, 1 = MTP.

### What makes 510.2429 vs 510.2112 MiB (ppl f16, c8192, b4096, ub512, `logs/ppl_f16_ub512_r1.log`)
- **Reserve plan #1** (pp, 9131 nodes): ROCm0 planned **534995200 B (510.2112 MiB)**.
  - The top of the chunk is `result_output` MUL_MAT f32 [248320,512], 508559360 B at offset 26435840.
  - Everything below it is persistent split inputs plus early tensors.
- **The ROCm0 buffer grows only in plans #16–#20**, the first 5 output ubatches with n_kv > 2051 (pos 5632..8191). Each step adds 8192 B: 534995712 → 535003904 → 535012096 → 535020288 → **535028480 (510.2429 MiB)**.
  - Each one is a `C15R` realloc on `leaf_108` (the QSA `cell_blk` I32 [n_kv]) growing 22528 → … → 32768 B.
  - In each the top tensor is still `result_output`, now at offset 26469120 instead of 26435840: it was pushed up by the larger input set below it.
- **Input diff, reserve plan #1 vs runtime plan #20** (`inp.py`, ROCm0 split inputs, `dump_f16_ub512_r1/plan_*_{0001,0020}.tsv`). A total of 21192712 → 21222344 B (diff 29632 B), made of:
  - 36 tensors present only at runtime: `leaf_{103,105,223,225,…}` i32[520] and `leaf_{106,226,…}` i64[130], one triple per QSA layer, i.e. `llm_graph_input_qsa_cache` members/positions/rows, `qwen4exp.cpp:1497-1507`. The trace confirms them: `C15Q cache il=* storage=1 cached=1 fast=1 prep_blocks=130` ×22 per layer.
  - Present only in the reserve: `leaf_116` i32[8192] (the `blk_cells` that fast layers stop reading).
  - Max end of the inputs: 21192960 → 21226240, which is **exactly +33280 B**, equal to the final buffer delta.
- **ROCm_Host (25.1094 → 27.0274 MiB).** This growth comes from plans #5–#8 (pos 0..2047, n_kv 512..2048 ≤ 2051, `route=dense-shortcut`). Each is a `C15R` on `attn_inp_kq_mask`: 524288 → 1048576 → 1572864 → 2097152 B, the f16 [n_kv, 512] mask. Its total is 26329120 → 28340256 B (+2011136).
- **q8_0** has the same mechanism and the same deltas, from a larger base: 510.4768 → 510.5085 MiB (+33280 B) and 25.3750 → 27.2930 MiB (`logs/ppl_q8_ub512_r{1,2}.log`).
- **ub4096 f16/q8_0**: no growth at all (`ppl_{f16,q8}_ub4096.log`: only plan #1 GROW, no warning). The first ubatch is 4096 tokens, so n_kv > 2051 from the start, and the fast cache needs ≤ 512 tokens (`llama-memory-hybrid-idx.cpp:1820`).

### Runtime routes vs the reserve (all on the instrumented PR build)

| workload | route(s) hit | reserve (ROCm0 / Host MiB) | runtime max (ROCm0 / Host MiB) | over? | log |
|---|---|---|---|---|---|
| ppl f16 c8192 ub512 (wikitext), r1, r2 | shortcut+mask; fast cache; selection | 510.2112 / 25.1094 | 510.2429 / 27.0274 | +33280 B / +1.92 MiB | `ppl_f16_ub512_r{1,2}.log` |
| same, base 13a279bfa | same | 510.2112 / 25.1094 | 510.2429 / 27.0274 | identical | `ppl_base_f16_ub512.log` |
| ppl q8_0 c8192 ub512, r1, r2 | same | 510.4768 / 25.3750 | 510.5085 / 27.2930 | +33280 B / +1.92 MiB | `ppl_q8_ub512_r{1,2}.log` |
| ppl f16 / q8_0 ub4096 | selection only | 4081.25 / 200.38 (f16), 4081.52 / 200.65 (q8) | same | no | `ppl_{f16,q8}_ub4096.log` |
| server Signal-3.8+MTP c261888 ub8192 q8_0, req A/B/C | text: selection, cache, bounded scorer (key_limits 8/16); image: **masked** (`blocks=0 scalar=0 bias=f32[n_blocks,1024]`) at pos 4/35038/70072/105106/140140 | main 3171.06 / 403.11; MTP 2806.43 / 402.75 | main 3171.06 / **477.23**; MTP 2806.43 / 402.75 | Host +74.1 MiB (5th image, n_kv 145152) | `srv_srv_q8.log` |
| same, f16 | same | 3170.80 / 402.84 | 3170.80 / **476.97** | Host +74.1 MiB | `srv_srv_f16.log` |
| server ub8192 q8_0, image after ~250k text | masked image at n_kv **261376** | 3171.06 / 403.11 | ROCm0 plan **3117.19** / **819.06** | ROCm0 no (53.9 MiB left); Host **+415.95 MiB** | `srv_srv_q8_D.log` |
| **server ub/b 1024 q8_0, image after 252k text (`D_text252k_img`)** | masked image at n_kv **253696** | main **1409.18** / 52.59; MTP **1349.36** / 52.31 | main **3034.61** / **796.47**; MTP **2558.52** / **796.19** | **ROCm0 +1625.4 MiB (main), +1209.2 MiB (MTP)**; Host +743.9 MiB | `srv_srv_q8_D_ub1024.log` |
| server np2 (kv_unified false), shared 60k-char wikitext prefix, 2 concurrent + 1 sequential, q8_0 (r2) / f16 / q8_0 trim0 | n_stream=2 tg masked F32 bias; split-stream pp I32; single-seq pp I32 | 1865.20 / 201.11 (q8), 1736.94 / 200.84 (f16) | same | no (0 OVER) | `srv_np2_q8_r2.log`, `srv_np2_f16.log`, `srv_np2_q8_trim0.log` |
| llama-bench pp4096 @ d4096 ub4096, trim 1 (random tokens) | trim fired (`layer 47 attention runs query rows [3584, 4096) of 4096`) | 1561.50 / 200.6329 (q8) | 1561.50 / 200.6407 | Host +8224 B (tile inputs) | `bench_q8_0_trim1_ub4096.log`, `bench_f16_trim1_ub4096.log` (200.3672 → 200.3751) |
| same, trim 0 | no trim | same | same | no | `bench_{q8_0,f16}_trim0_ub4096.log` |
| llama-bench pp2048 @ d4096 ub512, trim 1/0 | trim never fires (q_first = 0 at ub = strip = 512, `qwen4exp.cpp:944-945`); shortcut mask during the depth fill | 195.4359 / 25.3555 (q8), 195.1703 / 25.0899 (f16) | 197.2894 / 27.2930 (q8), 197.0238 / 27.0274 (f16) | **ROCm0 +1.85 MiB** (the logits do not dominate here: llama-bench reserves 1 output) | `bench_*_trim*_ub512.log` |
| MTP context (ctx_type 1) on text, all server runs | dense/selection | — | — | 0 of 74/61/44 MTP ubatches over at ub8192 | `srv_srv_{q8,f16,q8_D}.log` |

- **Image top tensors.**
  - **Main context, `srv_srv_q8_D_ub1024.log` plan #510:**
    - The ROCm0 top is `FLASH_ATTN_EXT` f32 [256,24,1024].
    - The ROCm0 inputs are `attn_inp_kq_mask` f16 [253696,1024] (519569408 B). The host-side plan holds the same mask plus the F32 per-block bias [63424,1024].
  - **MTP context, plan #511:**
    - The host top is `attn_inp_kq_mask` f16 [253696,1024].
    - ROCm0 holds [63424,2048] MUL_MAT, a GET_ROWS of [512,253696], and the mask copied to f32 [253696,512], each 519569408 B.
  - **At ub8192 (`srv_srv_q8_D.log` plan #80):** the host is `attn_inp_kq_mask` f16 [261376,1024] 535298048 B + `leaf_121` f32 [65344,1024] 267649024 B. The ROCm0 input copies are 535 + 2×268 + 2×134 MiB.
- **Why images take the masked route.**
  - Block selection requires `ubatch.token` (`qwen4exp.cpp:1110`, inside `qwen4exp_use_block_selection`). An embd ubatch has `token == nullptr`, so `blocks=0`.
  - Even on token ubatches, scalar visibility also excludes 2-D positions (`:1453-1454`, comment `:1468-1471`).
  - The trace confirms it: every embd ubatch logs `blocks=0 scalar=0 … bias=f32[n_blocks,1024]`. Each one is 1024 tokens (`--image-min-tokens 1024`).
- **The depth warning does not fire.** `QSA_MASKLESS_MAX_KV` (1048576, `qwen4exp.cpp:1102`) only warns above 1M keys, and this cliff appears well below that.
- **Answers and prompt throughput** (instrumented build, `-lv 3`, raw):
  - **ub8192:**
    - A `'2749'`: 61057 tok, 1087.3 t/s.
    - B `'12749'`: 69803 tok, 1098.4 t/s.
    - C `'26999'`: 180157 tok, 1050.2 t/s.
  - **ub1024:** D `'30099'` (correct): 253553 tok, 830.3 t/s (`srvD1024.out`).
  - Not a performance claim (logging on, 1 run).
- **llama-bench t/s** (random tokens, llama-bench.cpp:2226, -r 2; this is a memory experiment, not a perf one):

| test | trim | q8_0 | f16 |
|---|---|---|---|
| pp4096 @ d4096 ub4096 | 1 | 1936.52 ± 34.04 | 1991.30 ± 46.54 |
| pp4096 @ d4096 ub4096 | 0 | 1953.14 ± 5.77 | 2017.10 ± 11.75 |
| pp2048 @ d4096 ub512 | 1 | 1264.13 ± 12.37 | 1307.84 ± 5.15 |
| pp2048 @ d4096 ub512 | 0 | 1264.83 ± 13.10 | 1313.78 ± 11.92 |

  - Trim 1 vs 0 differs by less than 2 sd here. No trim speed claim follows from it.

### Setup and commands (strix-002, 2026-11-04 19:30–21:00 CET; every GPU run under `flock ~/gpu.lock`)
- **Tree.** A separate worktree `~/dev/wt_lab/c15/tree`, detached at 94e086847 and created with `git -C ~/dev/wt_lab worktree add --detach`. The existing `~/dev/wt_lab` tree and its C1–C14 diffs are untouched.
  - Trace patch: `SP/final219/c15/patch.py`. Resulting diff `SP/final219/c15/remote/c15_trace.diff` (md5 1b9ebc03…, equal to the remote `git diff`).
  - The patch only adds logging behind `GGML_C15=1`:
    - `C15A`: per-buffer planned / current size and the tensor that sets each chunk's top. `GGML_C15_DUMP=<dir>` writes a TSV of every allocated tensor for each plan that grows a buffer.
    - `C15R`: the realloc reason.
    - `C15L`: every `graph_reserve` call, and per ubatch the actual vs expected buffer size, with `!OVER`.
    - `C15Q`: the QSA route per layer: shortcut/selection, blocks/scalar/bias type, cache cached/fast, trim.
  - The C15 log lines were raised from INFO to WARN so that llama-perplexity shows them.
- **Build.** The task's HIP flags: `cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/lib/llvm/bin/clang -DCMAKE_PREFIX_PATH=/opt/rocm -DLLAMA_CURL=OFF`, then `cmake --build build -j 16 --target llama-perplexity llama-server llama-bench`.
- **Scripts.** Copies are in `SP/final219/c15/remote/`; the remote ones are in `~/dev/wt_lab/c15/`.
  - `c15_ppl.sh <tag…>`: `llama-perplexity -m $M -f wiki.test.raw -c 8192 -b 4096 -ub {512|4096} --chunks 2 -ngl 99 -fa on -lm mmap -lzm on-direct -ctk/-ctv {f16|q8_0} -lv 3`. The `base_f16_ub512` tag uses `~/dev/wt_pr219v2_base/build/bin`. PPL: f16 1.9428, q8_0 1.9532 (ub512); f16 1.9449, q8_0 1.9519 (ub4096).
  - `c15_srv.sh <tag> <kv> <env> <req…>`: `~/repro_img/repro_kv.sh` flags (Signal-3.8 + mmproj + MTP sidecar, c261888, `-ub/-b ${UB:-8192}`, `--image-min-tokens 1024`) plus `-lv 3`. Requests come from `${R:-~/repro_img}`. `D_text252k_img.json` = text lines 20000..30099 followed by 1 image (`~/dev/wt_lab/c15/`).
  - `c15_np2.sh <tag> <kv> <env>`: text model, `-c 65536 -np 2 -b 4096 -ub 4096`. Two concurrent `/completion` requests share a 60000-char wikitext prefix, then a third request runs alone.
  - `c15_trim.sh`: `llama-bench -ngl 99 -fa 1 -lm mmap -lzm on-direct -r 2 -n 0 -v` with `LLAMA_FN_LAST_ATTN_TRIM={1,0}`, `-p 2048 -ub 512 -d 4096` and `-p 4096 -ub 4096 -d 4096`, -b 4096.
  - Analysis helpers: `low.py`, `inp.py`.
- **Who ran what.** `np2_q8`, `np2_q8_trim0`, `np2_f16`, `np2_q8_r2`, `srv_f16` and `srv_q8_D` (request `D_text250k_img`, file no longer on disk; log `stop processing: n_tokens = 261338`) were launched by the orchestrator with these scripts.
  - The np2_q8 sequential result is void: the deadlock is described in the orchestrator's note. `np2_q8_r2` is its rerun with the fixed `wait $P0 $P1`.
  - Every other run was mine.
  - The first llama-bench pass had no `-v` (no trace). It was stopped, and every trim result above comes from the `-v` rerun.
- **Raw logs and TSV dumps** (27 MB total, no file > 1 GB): `~/dev/wt_lab/c15/logs/`. The instrumented tree is kept for reuse.

## C16

**Question.** FACTS says base q8_0 at c131k OOMs ("needs 21578 MiB"). llama-bench d131k on base q8_0 ran (L52). Which is true, and how should the PR state the "before" memory?
**Answer.** Both are true. They describe different setups.
- **The compute buffer is fixed.** Base 13a279bfa, q8_0/q8_0, `-c 131072 -b 4096 -ub 4096`, reserves **ROCm0 21578.66 MiB + ROCm_Host 1738.00 MiB**. That is 5 identical observations across both lazy modes. The same command reserves 4133.44 MiB with f16 on base and 4133.70 MiB with q8_0 on PR #219.
- **The OOM depends on the setup.** It happens only when `-lzm` is left at its default. With the default, the 28.8 GB `per_layer_token_embd` tensor stays resident (CPU model buffer 27465.95 MiB). The total committed memory is then 101775 + 29848 = 131623 MiB, which is more than the 127426 MiB of RAM. With `-lzm on-direct` (what llama-bench used in L52), that tensor is not loaded. The total is then 101775 + 2382 = 104157 MiB, and the context is created.
- **Decision.** State the goal and the "before" numbers as compute-buffer MiB, not as OOM: base q8_0 c131072 b/ub 4096 = 21578.66 MiB ROCm0 (5.22× f16's 4133.44); PR #219 = 4133.70; the target is ≤ f16 + ε. "OOM" may appear only as a qualified consequence: "with default lazy mode on a 128 GB Strix Halo, it OOMs".

### Setup
- strix-002, 2026-11-04 from 20:27 to 20:54 CET. Every run is under `flock ~/gpu.lock`. No build: the existing binaries are used, BASE = `~/dev/wt_pr219v2_base/build/bin` (13a279bfa) and PR = `~/dev/wt_pr219v2/build/bin` (94e086847).
- An orphaned `c15_trim.sh` (pid 3526617, parent 1, from the C15 agent) was still queuing llama-bench jobs on the same lock, so its jobs interleaved with mine. There was no concurrent GPU use: flock serializes, and `free -m` before each run shows about 124.8 GB available.
- Driver: `~/dev/wt_lab/c16/run.sh TAG BIN TYPE TEXT CHUNKS [extra]`. It runs:
  `flock ~/gpu.lock timeout 3600 $BIN/llama-perplexity -m ~/models/qwen3.8-flash-next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf -f TEXT -c 131072 -b 4096 -ub 4096 --chunks 1 -ngl 99 -fa on -ctk T -ctv T -lv 4 [extra]`
  - `-lv 4` is required. Without it the `sched_reserve` lines are not printed: first attempt, `c16/logs_v0/` on strix-002.
  - Campaign scripts: `c16/campaign.sh`, then `c16/campaign2.sh`.
- **Create-only runs** use TEXT = `c16/tiny.txt` (the first 20000 bytes of wiki.valid.raw, 4866 tokens). llama-perplexity creates the context, prints the reserve, then exits with "you need at least 262144 tokens". The exit code is 0.
- **Real-text runtime run** uses TEXT = `~/data/wikitext-2-raw/wiki.test.raw`, 1 chunk.
- Logs: `strix-002:~/dev/wt_lab/c16/logs/*.log`. Copies: `SP/final219/c16/` (logs, run.sh, campaign*.sh, campaign.out).

### Raw numbers (`sched_reserve` lines and `common_memory_breakdown_print`, MiB)

| run | tree | K/V | extra flags | load_mode / CPU model buf | ROCm0 compute | ROCm_Host compute | device self (model+ctx+compute) | host self | result |
|---|---|---|---|---|---|---|---|---|---|
| base_q8_mmap_lzm_r1 | base | q8_0 | `-lm mmap -lzm on-direct` | mmap / PLE ignored | **21578.66** | 1738.00 | 101775 = 78056+2140+21578 | 2382 | context OK, EXIT 0 |
| base_q8_mmap_lzm_r2 | base | q8_0 | same | mmap | **21578.66** | 1738.00 | 101775 | 2382 | context OK, EXIT 0 |
| base_q8_lzm_r1 | base | q8_0 | `-lzm on-direct` | none / PLE ignored | **21578.66** | 1738.00 | 101775 | 2382 | context OK, EXIT 0 |
| base_q8_lzm_r2 | base | q8_0 | same | none | **21578.66** | 1738.00 | 101775 | 2382 | context OK, EXIT 0 |
| base_q8_default_r1 | base | q8_0 | (none) | none / **CPU 27465.95** | n/a (fit projection 21578) | n/a (projection 1738) | 101775 projected | **29848** projected | **killed by the kernel OOM killer** during reserve (EXIT 137; journalctl 20:35:14 `Killed process 3530254 (llama-perplexit) anon-rss:20960636kB`) |
| 2026-11-01 historical | base | q8_0 | (none) | none / CPU 27465.95 | `allocating 21578.66 MiB … cudaMalloc failed: out of memory` | n/a | n/a | n/a | context failed (`~/dev/wt_pr219v2_logs/buf_base_q8_0_sk1_c131072.log` lines 78–84; this is the FACTS source) |
| base_f16_mmap_lzm_r1 | base | f16 | `-lm mmap -lzm on-direct` | mmap | 4133.44 | 201.44 | — | — | context OK |
| base_f16_default_r1 | base | f16 | (none) | none / CPU 27465.95 | 4133.44 | 201.44 | 85950 = 78056+3760+4133 | 28311 | context OK |
| pr_q8_mmap_lzm_r1 | PR | q8_0 | `-lm mmap -lzm on-direct` | mmap | 4133.70 | 201.70 | 84330 = 78056+2140+4133 | 845 | context OK |
| pr_q8_default_r1 | PR | q8_0 | (none) | none | 4133.70 | 201.70 | — | — | context OK |

KV buffers in every q8_0 run: 1632.00 + 204.01 MiB. In every f16 run: 3072.00 + 384.00 MiB.

### Noise
- **The compute buffer is deterministic.** Base q8_0 21578.66 MiB appears in 4 runs today (2 per lazy mode) and in the 2026-11-01 log, which makes 5 of 5 identical. The spread is 0.00 MiB.
- `-lm mmap` versus the default load mode, and `-lzm on-direct` versus off, do not change the compute buffer. They change only the weight residency (CPU model buffer 27465.95 MiB present or absent).

### Why the default setup OOMs (arithmetic from the breakdown lines above)
- With the default lazy mode: device self 101775 + host self 29848 = **131623 MiB > MemTotal 127426 MiB** (`free -m`). Weights, compute and KV are all carved out of the same unified LPDDR5X.
- `common_fit_params` still printed "will leave 21591 >= 1024 MiB of free device memory, no changes needed". It judges the ROCm0 side alone and does not count the 28110 MiB host side, which lives in the same physical RAM (`base_q8_default_r1.log`, `common_params_fit_impl` lines). This is a side defect of the fit logic on UMA, not of QSA.
- With `-lzm on-direct`: 101775 + 2382 = 104157 MiB, which fits. This is the L52 llama-bench setup (`d131k_base_q8_0_ub4096.json`: `"lazy_mode": "on-direct"`, n_ctx = 512 + 131072, llama-bench.cpp:1332). L52 and FACTS therefore do not contradict each other.
- The 83908.76 MiB image-server failure (L50: c261888, b/ub 8192, Signal-3.8) is a true OOM in any mode, because it exceeds the RAM even with no weights.

### Route check (code)
- On base, q8_0 always takes the masked route. `qwen4exp_use_block_selection` requires `type_k==GGML_TYPE_F16 && type_v==GGML_TYPE_F16` (base `src/models/qwen4exp.cpp:1112`).
- The log line `QSA attention: selected-key kernels (maskless block selection)` printed by base q8 runs is therefore misleading. It comes from the probe at `src/llama-memory-hybrid-idx.cpp:48-60`, which always builds **F16** K/V tensors whatever the cache type (lines 54-55), and is reported at line 125-126.
- The 21578.66 vs 4133.44 MiB difference is the masked route's query × all-key masks. The excess is 17445.22 MiB over 4096 × 131072 query-key pairs, which is 34.1 B per pair. This agrees with FACTS' "21-42 B per query × key".

### Real-text runtime at c131072: infeasible with llama-perplexity, for any K/V type and any tree
- `base_q8_mmap_lzm_eval` (wiki.test.raw, 1 chunk) created the context: 21578.66 MiB as above, "calculating perplexity over 1 chunks, n_ctx=131072, batch_size=4096".
  - It ran 19 graph reallocations, that is about 19 ubatches or 77.8k tokens of real text, with no GPU allocation failure.
  - The kernel OOM killer then killed it at 20:53:45 (EXIT 137, `free -m` after: used 125894 MiB, swap 6772 MiB).
- **Cause: a tool artefact, not QSA.** llama-perplexity keeps the logits of the second half of the chunk in host RAM: `logits.reserve(n_ctx * n_vocab)` and `logits.insert(...)` per batch (base `tools/perplexity/perplexity.cpp:511-513`, `:587-590`). From pos 65536 on, each ubatch adds 4096 × 248320 × 4 B = 3.79 GiB. The full half chunk needs 65536 × 248320 × 4 B = 62080 MiB. Together with the 78056 MiB of weights, that exceeds RAM whatever the KV type (f16 included).
- **The PR eval was stopped by me before it started computing,** for this reason. A real-text 131k runtime check needs a tool that does not keep the logits (for example llama-server or llama-cli). Not done: UNPROVEN beyond 77.8k tokens of runtime on real text.
- Runtime evidence for random tokens at d131k on base q8_0 already exists: L52 (pp512 247.0, pp2048 227.5 t/s, llama-bench).
- **COLLATERAL DAMAGE, needs owner attention.** The 20:53:45 OOM event killed other processes besides my llama-perplexity (journalctl -k):
  - UID 60578 (the GDM greeter session): gnome-session-c, gsd-*, at-spi*, dbus-broker*, dconf-service, xdg-permission-, fusermount3;
  - one UID 1000 `dbus-broker` (pid 1975874).
  
  I did not restart anything (no systemd). The 20:35:14 event killed only my process. Lesson for the rewrite's validation: on this 128 GB UMA box, **never run llama-perplexity with -c ≥ ~64k** on this 248k-vocab model.

### Decision
1. **Write the "before" numbers as compute-buffer MiB**, with flags: base q8_0, `-c 131072 -b 4096 -ub 4096 -fa on`, ROCm0 compute buffer **21578.66 MiB** (host 1738.00). Compare f16 **4133.44 MiB** and PR #219 q8_0 **4133.70 MiB**. At c32768 the figures are 6809.41, 4083.69 and 4083.95 (`~/dev/wt_pr219v2_logs/buf_*_c32768.log`).
2. **Memory goal:** for every supported K/V type, ROCm0 compute buffer ≤ the f16 figure + 1 MiB at the same `-c/-b/-ub`, measured from the `sched_reserve` line with `-lv 4`. The figure does not depend on `-lm`/`-lzm`, so either may be used. Drop "OOM at c131k" as a headline. If it is mentioned at all, word it as "with the default lazy mode (PLE resident, +27466 MiB host), the base q8_0 reserve no longer fits in 128 GB unified memory".
3. Fix FACTS line 15 and ledger L51 accordingly. L51's source is now located: `~/dev/wt_pr219v2_logs/buf_base_q8_0_sk1_c131072.log` (llama-perplexity, load_mode none, cudaMalloc failure).

## C17

**Question.** Depth quality (Track Q4): does pinned q8_0 stay near the floor at c32k?
**Answer.** Yes, up to 32k. Under QSA+MoE pins, scored on positions 28672-32767, q8_0/q8_0 KLD is **1.35e-4** against a floor of **1.01e-4**. That is an offset of **+3.4e-5**, the same as at c8192 (+3.0e-5 ± 0.8e-5). Both chunks are positive (+4.0e-5 and +2.8e-5). With only 2 chunks there is no chunk SE, so the 32k offset carries no σ of its own. Beyond 32k, q4_0 at depth, and the masked (tk) route at depth were not measured.

### Setup (source: results-Q.md section Q4 plus the raw logs, re-read for this entry; READ-ONLY, no GPU used)
- Machine strix-002, agent Q, 2026-11-04 between 00:18 and 01:00 (log mtimes). Worktree `~/dev/wt_measQ`, PR #219 head 94e086847 plus the measurement patch `~/dev/wt_measQ/q/qpin.patch` (results-Q.md:3).
- Driver `~/dev/wt_measQ/q/runQ.sh:14`: `flock ~/gpu.lock env $QENV ... llama-perplexity -m $M -f $W -c $CTX -b 4096 -ub $UB --chunks $CH -ngl 99 -fa on ...`. Common flags are in results-Q.md:5: `-lm mmap -lzm on-direct`, wiki.test.raw, `--kl-divergence-base`.
- Arms (`~/dev/wt_measQ/q/campaign2.sh:5-10`, all `-c 32768 --chunks 2 QKL_TAIL=4096`, replay against `ref32`): self f16 ub4096 qsa,moe; q8_0 ub4096 none / qsa / qsa,moe; f16 ub2048 qsa,moe / none.
- Reference: f16/f16 ub4096 hooked, recorded by `ref32_rec` (`campaign.sh:27`, `logs/ref32_rec.log:23`: `rows_rec=736256 moe_rec=3088384`). The ref32 base and pins were deleted afterwards (`ls ~/dev/wt_measQ/q/ref32` returns "No such file or directory"; results-Q.md:139).
- Log check: `logs/d32_q8_qsamoe.log:15`: `kl_divergence: computing over 2 chunks, n_ctx=32768, batch_size=4096`. Pin coverage is complete in every pinned arm: `rows_absent=0`, `moe_pin=3088384` (line 62 of each pinned log). The f16 ub2048 logs show `rows_absent=6144` with `rows_pin=691200`. That is a 2048-ub row-mapping remainder. The same reference rows were used and the KLD was not affected (UNPROVEN beyond that reading; the self-check shows the tool is exact at ub4096).

### Raw numbers (`strix-002:~/dev/wt_measQ/q/logs/<tag>.log`; chunk table lines 19-20 are cumulative, Mean KLD at line 31)

| tag | pins | Mean KLD | per-token ± (printed) | cum. chunk 1 / 2 | chunk 2 derived (2·mean − c1) | 99.9% KLD | same top p | MoE exact (QPIN_MOE layer=-1) | QSA overlap all pos |
|---|---|---|---|---|---|---|---|---|---|
| d32_self_qsamoe | qsa,moe | 0.000000 | 0.000000 | 0.00000 / 0.00000 | 0 | 0.000051 | 100.000% | 1.000000 | 1.000000 |
| d32_f16ub2048_qsamoe (**floor**) | qsa,moe | **0.000101** | 0.000002 | 0.00014 / 0.00010 | 0.62e-4 | 0.002057 | 99.573% | 0.966214 | 0.996327 |
| d32_q8_qsamoe | qsa,moe | **0.000135** | 0.000004 | 0.00018 / 0.00013 | 0.90e-4 | 0.003365 | 99.548% | 0.960713 | 0.995316 |
| d32_q8_qsa | qsa | 0.006036 | 0.000176 | 0.00972 / 0.00604 | 2.36e-3 | 0.179031 | 97.216% | 0.821494 | 0.982391 |
| d32_q8_none | none | 0.005776 | 0.000149 | 0.00902 / 0.00578 | 2.53e-3 | 0.162194 | 97.216% | 0.819993 | 0.982309 |
| d32_f16ub2048_none | none | 0.005512 | 0.000171 | 0.00882 / 0.00551 | 2.20e-3 | 0.186104 | 97.253% | 0.834478 | 0.984278 |

Paired q8_0 − floor under QSA+MoE pins, per chunk: chunk 1 = 0.00018 − 0.00014 = **+4.0e-5**; chunk 2 = 0.90e-4 − 0.62e-4 = **+2.8e-5**. The printed values are rounded, so each chunk difference carries about ±1e-5 of rounding. Mean difference = 0.000135 − 0.000101 = **+3.4e-5**.
- Minor correction to results-Q.md:125: Q lists the q8 chunk 2 as 0.8e-4 and the per-chunk differences as +4, +2. Re-deriving from the log gives 0.90e-4 (±0.06e-4 from rounding) and +2.8e-5. This does not change the reading.

Comparison with c8192 (results-Q.md:45-46, 59):

| metric | c8192, 8 chunks, pos 4096-8191 | c32768, 2 chunks, pos 28672-32767 |
|---|---|---|
| floor (f16 ub2048), QSA+MoE pinned | 1.08e-4 (chunk SE 0.12e-4) | 1.01e-4 (no SE) |
| q8_0/q8_0, QSA+MoE pinned | 1.37e-4 (chunk SE 0.14e-4) | 1.35e-4 (no SE) |
| q8 − floor, paired | +3.0e-5 ± 0.8e-5 | +3.4e-5 (chunks +4.0, +2.8) |
| q8 / floor | 1.27× | 1.34× |
| QSA overlap under pins, q8 / floor (results-Q.md:86,89) | 0.99673 / 0.99739 | 0.995316 / 0.996327 |
| unpinned q8 / floor | 75.3e-4 / 65.8e-4 | 57.8e-4 / 55.1e-4 |

### Noise
- **The self-check is exact at depth:** d32_self_qsamoe has KLD 0.000000, same top p 100%, and every QSA/MoE agreement at 1.000000 (`logs/d32_self_qsamoe.log:31,60,63,180-`). The 99.9% value of 0.000051 is the uint16 base quantization (results-Q.md:30).
- **The printed "±" is per-token**, about 6× too small according to Q's measurement at 8k (results-Q.md:7). It is not used here.
- **There is no chunk SE at 32k** (n=2). Estimate from the 8k data: the 8 paired differences [3, 5, −2, 6, 3, 3, 3, 3]e-5 (results-Q.md:59) have a per-chunk SD of 2.3e-5. If that SD holds at 32k (UNPROVEN), the 2-chunk paired mean has an SE of about 1.6e-5. On that basis:
  - the 32k offset of +3.4e-5 is about 2σ above zero;
  - the 32k vs 8k offset change (+0.4e-5) is well inside the noise;
  - a growth of the q8 cost beyond about 2× its 8k value (offset > ~6.6e-5) would have been visible at about 2σ.
  The observed spread of the two 32k differences (SD 0.85e-5) is smaller than the 8k per-chunk SD, which is consistent with this estimate.
- **The hook is itself a perturbation.** All arms are hooked and so is the reference, so comparisons between arms are consistent. Absolute numbers are in a hooked fusion configuration (results-Q.md:32-34).

### Proposed experiment vs what Q ran
Proposed: "Finish the QC ref32 block: c32768 ×2 chunks, QKL_TAIL=4096, q8 none/qsa/qsamoe vs f16 ub2048 none/qsamoe".
- **Run, all 5 arms:** d32_q8_none, d32_q8_qsa, d32_q8_qsamoe, d32_f16ub2048_none, d32_f16ub2048_qsamoe (`campaign2.sh:6-10`, logs listed above). Q also ran the self-check d32_self_qsamoe (`campaign2.sh:5`), which was not in the proposal. **No proposed arm is missing.**
- **Not run (none of these was in the proposal):**
  - no more than 2 chunks, so there is no chunk SE at 32k;
  - no QSA-only or MoE-only pin arm for the f16 ub2048 floor at 32k;
  - no masked-route q8_0 K / f16 V arm at depth;
  - no q4_0 arm and no indexer-type variant at depth;
  - no depth beyond 32k (results-Q.md:133).
- **Does it matter for the decision?** Not for the q8_0 ≤ 32k claim. Every arm needed to compare pinned q8 with the pinned floor at depth exists, the self-check proves the tool exact at 32k, and both chunks agree in sign and size with the 8k offset. The missing chunk SE limits the claim to "same offset as at 8k, within an estimated σ of about 1.6e-5 (UNPROVEN extrapolation)", not to a measured σ. The gaps do matter for anything wider: q4_0 or the tk route at depth, or any statement beyond 32k. Those stay UNPROVEN.

### Decision
1. **The quality claim extends to 32k for q8_0/q8_0** (PR #219 selected-key route, the default type). Pinned KLD at 28-32k is 1.35e-4 against a floor of 1.01e-4, a +3.4e-5 offset that equals the 8k offset (+3.0e-5 ± 0.8e-5). Q8_0 stays floor-class at depth, and there is no sign that its cost grows with depth. This replaces FACTS' "no valid depth-quality evidence" for q8_0 up to 32k. The wording must be limited to "≤ 32k, q8_0, 2 chunks".
2. **Not claimed:** q4_0 at depth, the q8_0 K / f16 V masked route at depth, and anything > 32k. These are UNPROVEN. A >32k check cannot use llama-perplexity at -c ≥ ~64k on this box (C16: host-RAM logits OOM). It would need about 98 KB of pins per position (results-Q.md:133).
3. **Depth tier for validation:** `-c 32768 --chunks 2 QKL_TAIL=4096`, QSA+MoE pinned, against a hooked f16 ub4096 reference recorded with the qpin patch. Compare the new kernel with the PR #219 kernel of the same type, paired by chunk, plus a self-replay that must give 0.000000.
   - Acceptance: the paired offset stays within the 8k-derived tolerance (about 1-2e-5 paired SE). This tier is a confirmation check, not the primary gate. The primary gate stays at c8192 ×8 chunks (C-Q2, results-Q.md:72), because only that tier has a measured σ.
   - If a measured σ is needed at depth, use ≥ 4 chunks at c32768. Whether wiki.test.raw holds enough tokens for that has not been checked: UNPROVEN.
   - The reference has to be re-recorded first, because ref32 was deleted. It takes about 9.6 GB of disk (results-Q.md:139) and must be deleted after the runs.

## C18

**Question.** Bound regime and loader staging: for f16 / q8_0 / q4_0 / bf16, K and V taken separately, which is faster, direct-register dequant or LDS-staged dequant? Is attention issue-bound or memory-bound in each regime? Decides: the per-type `staging` trait, and whether PR #219's direct q8 loaders are reused as-is (LEDGER §C18; plan-v2-X.md:190/264 PF-3, plan-v2-Z.md:127/225 E-P3, plan-v2-W.md:155/262 E5, plan-v2-Y.md:195/291 X3).

**Answer.**
- **The regime depends on the spread of the selected cells. It does not depend on the type.**
  - Real text up to n_kv 135k is issue-bound for every type. For f16 K direct + packed V, ns/chunk is 34.0 at wiki d4k and 33.7 at wiki d131k (mean of r1/r2). The time grows only with the G4 union (ch/q 45.2 → 59.2).
  - In that regime the PR's q8_0 K loader costs +11.6 % (d4k) to +13.0 % (d131k) against f16. q4_0 costs +17 % to +22 % and bf16→f16 +9 % to +12 %. This is dequant/issue cost.
  - The memory-bound regime starts with uniform spread ≥ 131k (f16) or ≥ 262k (quantized), and with random tokens at d131k for f16. There the bytes count. At 1M spread, q8_0 K+V is **0.39×** and q4_0 **0.32×** the f16 route time, and random-token d131k already favours q8_0/q4_0 K (−10 %).
  - This confirms and extends C6. Plan A's "issue-bound" and FACTS' "q8 faster at 131k" are both right, in different regimes.
- **Per-type staging trait for the prefill G4 kernel (n_q > 512):**
  - **K = split-register** (`X`): each half-wave dequantizes half the dims once, and the halves are exchanged with `v_permlanex16`. No LDS. Use it for q8_0, q4_0 and bf16→f16. For f16 and native bf16 bits, X and direct are equal within ±1.3 %.
  - Split cuts the fit-regime K penalty against f16 to **+3.9 % / +5.5 % (q8_0, d4k / d131k text), +2.2 % / +3.7 % (q4_0), +1.4 % / +4.3 % (bf16)**.
  - **LDS-staged K (`S`) is not adopted.** It equals split for q8_0/q4_0 (−0.7 % to +1.5 %), is up to +3.1 % for bf16 and +2 % to +5 % for f16 below 262k, and it adds LDS traffic and wave fences.
  - **V = packed by default for every type** (the PR's O(n_kv) f16 pack). Every direct-V loader is slower than packed+pack at every fixture up to n_kv 135k, text and random alike. The smallest losses are q8_0 +0.7 % and q4_0 +3.0 % at random d131k. On text they are +10 % (q8_0) and +15 % (q4_0) at wiki d131k.
  - **Direct V only for quantized types at large spread (≥ 262k measured).** There it wins: q8_0 X+X 20.34 vs packed 23.18 ms at 262k, and 24.60 vs 37.32 at 1M.
  - **f16/bf16 never use direct V.** Packed is faster at every n_kv measured, including 1M, where packed is 62.97 ms and the PR's VD is 65.91. So the PR's `n_q·64 < n_kv` switch is right for q8_0 but wrong for f16.
  - The direct-V loader is split-gather (`X`). For the quantized types, which are the only ones that use direct V, the staged-transposed `S` is slower everywhere except at 1M, where it wins by ≤ 1.8 %.
- **Decode kernel (`qsa_decode_wmma_partial`, n_q ≤ 512, which covers tg and ub ≤ 512): V = LDS-staged for every type. K = direct for f16 and q8_0, split for q4_0 and bf16.** Once V is staged, the PR's q8 decode K loader is within 1 % of split.
  - At ub512 wiki d131k, against the PR's verbatim decode kernel: q8_0 **6.43 → 3.77 ms (−41 %)** and f16 **4.69 → 3.37 ms (−28 %)**. At tg (n_q 1, d131k): q8_0 30.9 → 19.8 µs (−36 %) and f16 24.0 → 18.7 µs (−22 %).
  - Here staging wins because the PR's decode V gather issues 32 (f16) to 48 (q8_0) scalar loads per lane after the second barrier, with no prefetch (pr_dec.inc, the PR's qsa-decode-wmma.cuh).
- **PR #219's prefill q8 loaders and its decode q8 V loader are not reused as-is. Its exact q8→half helper and its decode q8 K loader are reused as-is.**
  - `qsa3_q8x2_to_h2` / `qd_q8x2_to_h2` (PR qsa-prefill.cu:13-16, qsa-decode-wmma.cuh:9-12) feed every new loader. All 4 905 prefill and 4 320 decode bitwise checks pass. Each check compares a variant with the first variant of its type pair, and the PR's verbatim kernels are among the variants checked.
  - Replaced:
    - the prefill q8 K branch: split is −6.7 % at the kernel level on wiki d131k (18.46 → 17.23 ms);
    - the prefill q8 DIRECT_V branch: 120 B/lane scratch spill, 24.24 ms at wiki d131k against 18.44 for split K + packed V;
    - the decode q8 V loader (numbers above).
  - Kept: the decode q8 K loader. Kq8_0D-Vq8_0S is 3.770 ms against 3.774 for Kq8_0X-Vq8_0S at ub512 wiki d131k.
- Not loader-related, but measured: the PR kernels themselves are **2-8.5 % (prefill) and 9-17 % (decode) slower than the same loaders without the mask/gate/o16 code paths**. The loop SALU count is 309 against 216 (`res/isa_loop_counts.md`). The cause is UNPROVEN beyond that count.

### Setup (strix-002, 2026-11-04 21:02–22:12 CET; every GPU run and every hipcc build under `flock ~/gpu.lock`; I was the only GPU user; no build in any shared tree; nothing in `~/dev/wt_pr219v2*` modified)
- **Where the code lives.** Worktree `~/dev/wt_lab`, detached at 94e086847. The earlier lab diffs are untouched and no env switch is set. C18 tools are in `~/dev/wt_lab/c18/`. Copies of every source, script, result and log are in `SP/final219/c18/remote/` (SP = `/tmp/claude-1000/-home-gaetan-puleo/78d4f4eb-9ba7-448e-8841-dce6638c30f6/scratchpad`).
  - md5 local = remote: `c18_attn.cu` 54825e9b…, `c18_dec.cu` 4308865a…, `c18_kernel.inc` 1a85aee6…, `pr_attn.inc` ed628f24…, `pr_dec.inc` 90b23e84….
  - The `c18_attn` binary was built before a `#ifndef C18_HELPERS_ONLY` guard was added to `c18_kernel.inc`. The guard does not change any kernel code.
- **PR code used verbatim.**
  - `pr_attn.inc` is byte-identical to PR qsa-prefill.cu:10-26, 241-601 and 630-670. I checked this today: the `sed` extract on strix-002, diffed, printed `IDENTICAL_TO_PR`.
  - `pr_union_kernels.inc` holds the PR's rows+merge plan kernels, as in C5/C6.
  - `pr_dec.inc` = PR `qsa-decode-wmma.cuh` (all 161 lines) + `qsa_decode_unique_ids` from qsa-decode.cu, extracted with `cat` / `sed` from `~/dev/wt_pr219v2` at 94e086847.
- **Fixtures** (`c18_cap.sh` → `logs/cap_*.log`; C5's `c5_capture`, linked against the unmodified PR libllama):
  - Command: `flock ~/gpu.lock env LLAMA_FN_LAST_ATTN_TRIM=0 ../c5/c5_capture $M <src> <ntok> 4096 cap/<tag> f16 <pos0s> 0 <qkv_pos0>`, with M = `~/models/qwen3.8-flash-next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`, ub = b = 4096, KV f16.
  - Every `C5CAP` line shows `n_q=4096 ns=2051 … mask=0`: all 12 QSA layers ran the maskless selected-key route with 4096 queries.
  - **Fixture sets:**
    - **wiki d4k**: wiki.test.raw, pos0 4096, n_kv 8192, real Q/K/V.
    - **rand d4k**: llama-bench's generator as one stream, `srand(1)` / `rand() % n_vocab`, pos0 4096, real Q/K/V.
    - **wiki d131k**: wiki.test.raw, pos0 131072, n_kv 135168, real Q/K/V (4.5 GB dump, deleted after the runs).
    - **rand d131k**: same generator, pos0 131072, rows only. The data is SYN (uniform ±1.7, as C6).
  - Determinism: the d4k rows of the 8192-token and 135168-token captures are byte-identical for both corpora. Commands `cmp wikiS_p4096.rows wikiL_p4096.rows` → `SAME_WIKI`, and the same for rand.
  - Random tokens are reported next to text, as the task asks.
- **Data per type.** The f16 K/V of each fixture are converted on the host with the ggml reference rules (`c18_attn.cu: build_cache`):
  - q8_0 as `quantize_row_q8_0_ref`: amax/127, `roundf`;
  - q4_0 as `quantize_row_q4_0_ref`: signed max/−8, `MIN(15, (int8)(x·id + 8.5))`;
  - bf16 by RNE.
- **Prefill harness `c18_attn`.**
  - The plan comes from the PR's own rows+merge kernels with the PR launch parameters.
  - Variant `K<T><L>-V<T><L>` = `c18_attn_kernel<KT,KL,VT,VL>` (`c18_kernel.inc`). It is the PR's `qsa3_attn_kernel` body line by line: Q prologue, WMMA order, online softmax, the 2 barriers and the epilogue are unchanged. Only the K and V loaders are policies.
  - **Types:**
    - `f16`;
    - `q8_0`;
    - `q4_0`: the exact `0x6400|q − 1032` trick, a single rounding, the same as `(half)((q−8)·d)`;
    - `bf16`: bf16 → f16 with saturation (`fmed3` ±65504), then f16 WMMA;
    - `bf16w`: bf16 bits fed to `wmma_f32_16x16x16_bf16`, with Q or P rounded to bf16 for that product.
  - **K loaders:**
    - `D`: per-lane direct, as the PR's `load_k`. For f16 and q8_0 it is the PR code verbatim. Lanes r and r+16 duplicate the load and the dequant.
    - `Dd`: D with the dequant deferred to the end of the iteration.
    - `X`: split. Half-wave p loads and dequantizes dims {8p..8p+7, 16+8p..16+8p+7} once, and the halves are exchanged with `__builtin_amdgcn_permlanex16`. Deferred.
    - `S`: LDS-staged. Lane l takes key l>>1 and half l&1 (coalesced), dequantizes once, and writes an f16 tile into a wave-private 1 KB slot. The slot aliases `part[0][w]`, which is free between barrier 2 and the next partials, so occupancy and LDS stay as in the PR (26496 B). The fragment is read back with 4 × b128. Deferred.
  - **V loaders:**
    - `P`: the PR's packed f16 (or bf16) layout. The pack is timed separately as `PACKV-<T>`; `c18_pack_v<T>` is the PR's `qsa_pack_v` indexing.
    - `D`: per-lane gather, as the PR's DIRECT_V. For f16 and q8_0 it is the PR code verbatim.
    - `X`: split gather: half p gathers dim 16p+r for 16 keys, then permlanex16. Dequant after barrier 2.
    - `S`: LDS-staged. Coalesced piece loads, dequantized once, written transposed into a wave-private slot (`part[1][w]`) with a bank swizzle, then read back with 2 × b128 per t. Done after barrier 2.
  - **References:** the PR's verbatim `qsa3_attn_kernel<true,8,false,Q8>` (V0 route) and `<true,8,true,Q8>` (VD route) for f16 and q8_0.
  - **Sweeps:**
    - K sweep: V fixed at f16 packed.
    - V sweep: K fixed at f16 direct, which is the PR code.
    - Pairs: K type = V type.
  - 58 kernels + 5 packs in all.
- **Decode harness `c18_dec`.**
  - The PR's `qsa_decode_wmma_partial` body with the K and V loaders as policies (`D` = the PR code, `X`, `S` as above). V `S` aliases `shared.scores[w]`.
  - The PR launch is used as is (qsa-decode.cu:136-158): `qsa_decode_unique_ids`, step 64, and the chunk/splits rule. It gives splits 1 / chunk 2064 at n_q 512, and splits 33 / chunk 64 at n_q 1 (`RES` lines).
  - Each fixture supplies the first n_q queries of every layer. These are the queries of a ub = n_q ubatch at the same pos0, in the same causal context. That the ub512 rows would be bit-equal is UNPROVEN, because GEMM numerics depend on the batch size.
  - The decode kernel is the route for every n_q ≤ 512 (QSA_DECODE_MAX_QUERIES 512, qsa-decode.cuh:5; it is checked first at fattn.cu:1154).
- **Resources** (`-Rpass-analysis=kernel-resource-usage`, `logs/build.log`, `logs/build_dec.log`, parsed by `resusage.py`):
  - Every prefill variant has 192 VGPRs, occupancy 8 and LDS 26496 B, as the PR has.
  - Scratch spills:
    - PR-VD-q8_0 120 B/lane;
    - Kf16D-Vq8_0D 132, Kq8_0D-Vq8_0D 108;
    - Kf16D-Vbf16D 72, Kbf16D-Vbf16D 76;
    - Kq8_0Dd 64;
    - the V-`S` pairs with K X or S: 36 each (Kf16S-Vf16S, Kq8_0X/S-Vq8_0S, Kq4_0S-Vq4_0S, Kbf16X/S-Vbf16S);
    - Kq4_0D-Vq4_0D and Kf16D-Vq4_0D: 8;
    - all other variants: 0.
  - All decode variants have 0 scratch, 95-121 VGPRs and occupancy 10-16.
  - Static loop instruction mix (CFG strongly-connected component holding the WMMAs, from `--save-temps` ISA, `isa_count.py` → `res/isa_loop_counts.md`), loop instrs / global_load / VALU:
    - Kq8_0D-Vf16P 772 / 11 / 415;
    - Kq8_0X-Vf16P 750 / 13 / 382 (+8 permlane);
    - Kf16D-Vf16P 655 / 12 / 314;
    - PR-V0-f16 829 / 20 / 371 with SALU 309 (mine: 216).
- **Commands.**
  - Build: `flock ~/gpu.lock /opt/rocm/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -Rpass-analysis=kernel-resource-usage c18_attn.cu -o c18_attn` (and `c18_dec.cu -o c18_dec`). Both printed BUILD 0.
  - Prefill (`campaign.sh`), per run r ∈ {r1, r2} as separate processes: `flock ~/gpu.lock ./c18_attn <rows> <qkv|SYN> 10 all all`. Fixtures:
    - `cap/wikiS_p4096.rows cap/wikiS_p4096.qkv`;
    - `cap/randS_p4096.rows cap/randS_p4096.qkv`;
    - `cap/wikiL_p131072.rows cap/wikiL_p131072.qkv`;
    - `cap/randL_p131072.rows SYN`.
  - Data control: `./c18_attn cap/wikiL_p131072.rows SYN 10 0,5,11 all`.
  - C6(a) combination: `./c18_attn SYN:1.8:<nk>:<s> SYN 10 all all` for nk ∈ {4096, 32768, 131072, 262144, 1048576} and s ∈ {1, 2}. This is the C6 generator, uniform spread, U = 1.8, 513 quads per query.
  - Decode (`campaign_dec.sh`), r1 and r2: `flock ~/gpu.lock ./c18_dec <rows> <qkv|SYN> <n_q> <reps> all all`, with n_q 512 (10 reps) on the 4 fixtures, and n_q 1 (50 reps) on wiki and rand d131k.
- **Timing.** 3 warm-up rounds, then n timed rounds. Each round launches every variant and every pack once, interleaved, with one hipEvent pair per launch. Per layer the median is taken, and the figure reported is the mean of the 12 per-layer medians.
- **Parsers:** `summ.py` → `res/summary_prefill.md`, `summ_dec.py` → `res/summary_dec.md`.

### Correctness
- **Bitwise.** Every variant is compared with the first variant of its (K type, V type) pair, the PR's verbatim kernels included. Result: **4905 prefill and 4320 decode checks, 0 words differ, 0 NaN** (`CHK` lines; summary_prefill.md and summary_dec.md "correctness:").
  - So every split and staged loader delivers exactly the halves of the PR loaders, which are bit-equal to `to_fp16` (PR comment, qsa-prefill.cu:11-12).
  - My reimplementation without the mask/gate paths gives the PR's output bit for bit: PR-V0/VD f16/q8_0 and PR-dec f16/q8_0 show `words_differ 0` against `Kf16D-Vf16P`, `Kq8_0D-Vq8_0P`, `Kf16D-Vf16D` and `Kq8_0D-Vq8_0D`.
- **Against a CPU f64 reference** that uses the dequantized cache values, 3 queries × 24 heads, first layer:
  - f16/q8_0/q4_0/bf16→f16: max |err|/max|ref| 0.5e-4 – 3.7e-4;
  - bf16w (bf16 Q or P): 0.5e-3 – 6.3e-3.

  bf16w is therefore not f16-class in numerics. Its model quality is UNPROVEN.

### Results: prefill G4 kernel (ms per ub4096 call, mean of 12 per-layer medians; mean of r1/r2, or of seeds s1/s2 for syn; `res/summary_prefill.md`)
Fixtures: wiki and rand at d4k (n_kv 8192) and d131k (n_kv 135168) are real captured rows. syn X = the C6 synthetic plan with uniform spread over X cells. G4 ch/q is 45.2 (wiki d4k), 56.7 (rand d4k), 59.2 (wiki d131k), 88.0 (rand d131k) and 57.75 (syn).

**K loaders** (V = f16 packed, pack excluded; reference `Kf16D-Vf16P` = the PR V0 f16 route, reimplemented):
| variant | wiki d4k | rand d4k | wiki d131k | rand d131k | syn 4k | syn 32k | syn 131k | syn 262k | syn 1M |
|---|---|---|---|---|---|---|---|---|---|
| Kf16D-Vf16P | 12.60 | 16.03 | 16.33 | 28.77 | 15.09 | 15.72 | 26.83 | 38.80 | 49.04 |
| Kf16X-Vf16P | 12.43 | 15.81 | 16.32 | 28.69 | 15.29 | 15.72 | 26.99 | 38.30 | 48.78 |
| Kf16S-Vf16P | 13.22 | 16.74 | 17.12 | 29.33 | 15.97 | 16.27 | 28.38 | 38.59 | 48.62 |
| PR-V0-f16 | 12.89 | 16.34 | 17.26 | 29.25 | 15.58 | 16.54 | 29.11 | 40.51 | 49.45 |
| Kq8_0D-Vf16P | 14.06 | 17.81 | 18.46 | 27.53 | 16.96 | 17.27 | 17.77 | 21.62 | 28.36 |
| Kq8_0Dd-Vf16P | 14.56 | 18.46 | 19.87 | 30.81 | 17.43 | 18.32 | 21.09 | 22.86 | 28.61 |
| Kq8_0X-Vf16P | 13.09 | 16.59 | 17.23 | 26.02 | 16.16 | 16.23 | 17.02 | 20.83 | 28.19 |
| Kq8_0S-Vf16P | 13.09 | 16.58 | 17.39 | 26.42 | 16.05 | 16.31 | 17.25 | 20.97 | 28.15 |
| PR-V0-q8_0 | 14.38 | 18.14 | 18.97 | 28.12 | 17.44 | 18.02 | 18.17 | 21.60 | 28.41 |
| Kq4_0D-Vf16P | 14.78 | 18.69 | 19.89 | 30.00 | 17.64 | 18.06 | 19.15 | 19.97 | 24.70 |
| Kq4_0Dd-Vf16P | 13.89 | 17.65 | 18.42 | 27.93 | 16.82 | 17.35 | 18.05 | 20.63 | 24.86 |
| Kq4_0X-Vf16P | 12.88 | 16.31 | 16.93 | 25.66 | 15.96 | 16.11 | 16.64 | 18.30 | 24.20 |
| Kq4_0S-Vf16P | 12.86 | 16.30 | 16.99 | 25.86 | 15.87 | 16.09 | 16.77 | 18.35 | 24.25 |
| Kbf16D-Vf16P | 13.74 | 17.40 | 18.36 | 30.57 | 16.94 | 17.26 | 26.82 | 38.36 | 48.62 |
| Kbf16Dd-Vf16P | 13.00 | 16.48 | 17.34 | 29.49 | 16.05 | 16.45 | 27.98 | 38.45 | 48.78 |
| Kbf16X-Vf16P | 12.78 | 16.22 | 17.04 | 29.21 | 15.81 | 16.17 | 27.89 | 38.25 | 48.84 |
| Kbf16S-Vf16P | 12.85 | 16.31 | 17.17 | 29.30 | 15.70 | 16.22 | 28.01 | 39.44 | 48.85 |
| Kbf16wD-Vf16P | 12.52 | 15.88 | 16.86 | 29.02 | 15.07 | 15.86 | 27.28 | 38.37 | 48.92 |
| Kbf16wX-Vf16P | 12.29 | 15.62 | 16.49 | 28.73 | 15.18 | 15.70 | 27.42 | 38.62 | 48.69 |
| Kbf16wS-Vf16P | 13.05 | 16.53 | 17.15 | 29.21 | 15.81 | 16.21 | 27.36 | 38.07 | 48.59 |

**V loaders** (K = f16 direct, PR code; `+ PACKV` = kernel + measured pack kernel of that type):
| variant | wiki d4k | rand d4k | wiki d131k | rand d131k | syn 4k | syn 32k | syn 131k | syn 262k | syn 1M |
|---|---|---|---|---|---|---|---|---|---|
| Kf16D-Vf16P + PACKV-f16 | 12.70 | 16.12 | 18.14 | 30.57 | 15.15 | 16.11 | 28.58 | 42.34 | 62.97 |
| Kf16D-Vf16D | 14.59 | 18.53 | 19.78 | 40.84 | 17.85 | 18.17 | 51.99 | 59.93 | 65.74 |
| Kf16D-Vf16X | 13.18 | 16.79 | 18.55 | 39.74 | 16.24 | 16.70 | 44.78 | 57.59 | 65.28 |
| Kf16D-Vf16S | 14.75 | 18.77 | 20.42 | 40.88 | 17.94 | 18.62 | 43.16 | 52.92 | 63.65 |
| PR-VD-f16 | 15.14 | 19.20 | 20.49 | 41.27 | 18.70 | 19.04 | 52.01 | 60.67 | 65.91 |
| Kf16D-Vq8_0P + PACKV-q8_0 | 12.63 | 16.06 | 17.54 | 29.98 | 15.11 | 15.96 | 28.02 | 41.14 | 58.17 |
| Kf16D-Vq8_0D | 16.67 | 21.18 | 23.48 | 37.76 | 20.43 | 22.52 | 28.27 | 36.60 | 46.39 |
| Kf16D-Vq8_0X | 14.71 | 18.78 | 19.31 | 30.19 | 18.02 | 18.46 | 23.97 | 35.09 | 45.48 |
| Kf16D-Vq8_0S | 15.39 | 19.58 | 20.34 | 31.57 | 18.64 | 18.85 | 25.15 | 35.49 | 44.67 |
| Kf16D-Vq4_0P + PACKV-q4_0 | 12.64 | 16.07 | 17.28 | 29.72 | 15.11 | 15.91 | 27.75 | 40.64 | 56.39 |
| Kf16D-Vq4_0D | 15.12 | 19.19 | 20.05 | 30.95 | 18.51 | 18.98 | 22.46 | 31.24 | 40.60 |
| Kf16D-Vq4_0X | 14.98 | 19.00 | 19.93 | 30.62 | 18.46 | 18.65 | 22.09 | 31.18 | 40.58 |
| Kf16D-Vq4_0S | 15.33 | 19.44 | 20.14 | 30.81 | 18.67 | 18.75 | 22.44 | 31.86 | 40.38 |
| Kf16D-Vbf16P + PACKV-bf16 | 12.74 | 16.18 | 18.28 | 30.72 | 15.15 | 16.32 | 28.73 | 42.44 | 63.12 |
| Kf16D-Vbf16D | 14.47 | 18.32 | 20.53 | 40.85 | 17.30 | 18.71 | 44.35 | 54.65 | 64.49 |
| Kf16D-Vbf16X | 14.11 | 17.94 | 19.39 | 40.21 | 17.39 | 17.70 | 43.14 | 55.20 | 65.38 |
| Kf16D-Vbf16S | 15.11 | 19.22 | 20.73 | 41.13 | 18.34 | 18.97 | 42.44 | 53.28 | 63.58 |
| Kf16D-Vbf16wP + PACKV-bf16w | 12.47 | 15.80 | 18.68 | 30.84 | 14.95 | 16.29 | 30.02 | 42.15 | 62.91 |
| Kf16D-Vbf16wX | 13.17 | 16.76 | 18.52 | 39.65 | 16.19 | 16.66 | 45.55 | 57.29 | 65.28 |
| Kf16D-Vbf16wS | 14.85 | 18.89 | 20.56 | 41.01 | 18.14 | 18.83 | 41.78 | 53.69 | 63.63 |

PACKV-f16 / q8_0 / q4_0 / bf16 (ms per call) are:
- wiki d4k: 0.093 / 0.027 / 0.036 / 0.141;
- wiki d131k: 1.812 / 1.212 / 0.950 / 1.955;
- rand d131k: 1.806 / 1.210 / 0.949 / 1.957;
- syn 262k: 3.545 / 2.342 / 1.838 / 3.642;
- syn 1M: 13.930 / 9.138 / 7.352 / 14.084.

**Pairs** (K type = V type; packed rows include the pack):
| variant | wiki d4k | rand d4k | wiki d131k | rand d131k | syn 4k | syn 32k | syn 131k | syn 262k | syn 1M |
|---|---|---|---|---|---|---|---|---|---|
| Kf16D-Vf16P + PACKV-f16 | 12.70 | 16.12 | 18.14 | 30.57 | 15.15 | 16.11 | 28.58 | 42.34 | 62.97 |
| Kf16X-Vf16X | 14.20 | 18.03 | 19.23 | 40.02 | 17.41 | 17.53 | 44.94 | 55.46 | 65.25 |
| PR-VD-f16 | 15.14 | 19.20 | 20.49 | 41.27 | 18.70 | 19.04 | 52.01 | 60.67 | 65.91 |
| Kq8_0D-Vq8_0P + PACKV-q8_0 | 13.80 | 17.50 | 19.23 | 27.74 | 16.78 | 17.41 | 18.86 | 23.96 | 37.48 |
| PR-V0-q8_0 + PACKV-q8_0 | 14.40 | 18.16 | 20.18 | 29.33 | 17.46 | 18.26 | 19.36 | 23.94 | 37.55 |
| PR-VD-q8_0 | 17.78 | 22.57 | 24.24 | 37.06 | 21.94 | 23.49 | 25.02 | 26.71 | 30.13 |
| Kq8_0D-Vq8_0D | 17.28 | 21.96 | 23.55 | 36.43 | 21.16 | 22.43 | 24.37 | 25.83 | 29.10 |
| Kq8_0X-Vq8_0X | 15.72 | 20.01 | 20.72 | 31.44 | 19.38 | 19.50 | 19.96 | 20.34 | 24.60 |
| Kq8_0X-Vq8_0S | 16.66 | 21.16 | 21.93 | 33.27 | 20.28 | 20.58 | 20.73 | 21.18 | 25.41 |
| Kq8_0S-Vq8_0S | 16.17 | 20.54 | 21.17 | 32.26 | 19.54 | 20.05 | 20.47 | 21.06 | 25.33 |
| Kq4_0D-Vq4_0P + PACKV-q4_0 | 14.73 | 18.64 | 20.55 | 30.77 | 17.39 | 18.39 | 20.07 | 21.98 | 32.07 |
| Kq4_0D-Vq4_0D | 16.45 | 20.89 | 21.85 | 33.30 | 20.19 | 20.71 | 21.16 | 21.27 | 21.49 |
| Kq4_0X-Vq4_0X | 15.58 | 19.79 | 20.56 | 31.17 | 19.49 | 19.53 | 19.61 | 19.80 | 20.05 |
| Kq4_0X-Vq4_0S | 16.17 | 20.50 | 21.23 | 32.07 | 19.93 | 20.00 | 19.89 | 20.01 | 19.90 |
| Kq4_0S-Vq4_0S | 15.86 | 20.12 | 20.64 | 31.18 | 19.30 | 19.82 | 19.77 | 19.95 | 20.38 |
| Kbf16D-Vbf16P + PACKV-bf16 | 13.75 | 17.40 | 20.05 | 32.29 | 16.77 | 17.92 | 30.57 | 41.77 | 62.71 |
| Kbf16X-Vbf16X | 14.67 | 18.68 | 20.00 | 40.48 | 18.07 | 18.79 | 43.71 | 56.03 | 65.35 |
| Kbf16S-Vbf16S | 15.93 | 20.27 | 21.98 | 42.60 | 19.24 | 20.68 | 42.33 | 55.46 | 65.42 |
| Kbf16wD-Vbf16wP + PACKV-bf16w | 12.38 | 15.70 | 18.44 | 30.49 | 14.84 | 16.42 | 28.95 | 42.73 | 62.75 |
| Kbf16wX-Vbf16wS | 15.05 | 19.18 | 20.41 | 41.01 | 18.49 | 19.02 | 41.44 | 53.65 | 63.56 |

**The best prefill option per type, against the f16 PR route (`Kf16D-Vf16P + PACKV-f16`).** "X+P" = split K + packed V. It is derived as `K<T>X-Vf16P` + `PACKV-<T>`. That kernel is ISA-identical to `K<T>X-V<T>P`, but it reads the f16-packed buffer. The same-ISA pair `Kq8_0D-Vq8_0P` (kernel only) vs `Kq8_0D-Vf16P` differs by 0 % to −3.6 % across the fixtures (for example 26.53 vs 27.53 ms at rand d131k), so derived rows carry up to about 4 %. The cause is buffer placement or data, UNPROVEN. The f16-packed buffer is the slower of the two, so the derived X+P rows are conservative.

| type | wiki d4k | rand d4k | wiki d131k | rand d131k | syn 131k | syn 262k | syn 1M |
|---|---|---|---|---|---|---|---|
| f16 PR route D+P | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| f16 X+P | 0.986 | 0.986 | 1.000 | 0.997 | 1.006 | 0.988 | 0.996 |
| q8_0 PR route D+P | 1.087 | 1.085 | 1.060 | 0.907 | 0.660 | 0.566 | 0.595 |
| q8_0 X+P | 1.033 | 1.031 | 1.017 | 0.891 | 0.637 | 0.547 | 0.593 |
| q8_0 X+X (direct V) | 1.239 | 1.241 | 1.142 | 1.028 | 0.698 | **0.480** | **0.391** |
| q4_0 PR-style D+P | 1.160 | 1.156 | 1.133 | 1.006 | 0.702 | 0.519 | 0.509 |
| q4_0 X+P | 1.017 | 1.013 | 0.986 | 0.870 | 0.614 | 0.475 | 0.501 |
| q4_0 X+X | 1.227 | 1.227 | 1.133 | 1.020 | 0.686 | **0.468** | **0.318** |
| bf16→f16 D+P | 1.083 | 1.079 | 1.105 | 1.056 | 1.070 | 0.986 | 0.996 |
| bf16→f16 X+P | 1.018 | 1.015 | 1.047 | 1.019 | 1.042 | 0.989 | 0.999 |
| bf16w (bf16 WMMA) D+P | 0.975 | 0.974 | 1.016 | 0.997 | 1.013 | 1.009 | 0.997 |

(Cross-type lines of `res/summary_prefill.md`. At syn 1M, q4_0 X+S is 0.316, 1 % below X+X.)

### Results: decode kernel (n_q ≤ 512 route; ms per call, mean of 12 per-layer medians; mean of r1/r2; `res/summary_dec.md`)
| variant | ub512 wiki d4k | ub512 rand d4k | ub512 wiki d131k | ub512 rand d131k | tg wiki d131k | tg rand d131k |
|---|---|---|---|---|---|---|
| Kf16D-Vf16D | 4.038 | 4.074 | 4.179 | 4.612 | 0.0205 | 0.0206 |
| Kf16X-Vf16D | 3.913 | 3.946 | 3.998 | 4.347 | 0.0204 | 0.0204 |
| Kf16S-Vf16D | 4.142 | 4.196 | 4.316 | 4.657 | 0.0208 | 0.0208 |
| Kf16D-Vf16X | 3.459 | 3.502 | 3.663 | 4.168 | 0.0190 | 0.0190 |
| Kf16D-Vf16S | 3.189 | 3.232 | 3.370 | 3.940 | 0.0187 | 0.0187 |
| Kf16X-Vf16X | 3.240 | 3.275 | 3.372 | 3.894 | 0.0189 | 0.0190 |
| Kf16X-Vf16S | 3.250 | 3.291 | 3.414 | 3.956 | 0.0188 | 0.0188 |
| Kf16S-Vf16S | 3.546 | 3.590 | 3.647 | 4.113 | 0.0192 | 0.0192 |
| **PR-dec-f16** (PR verbatim) | 4.389 | 4.437 | 4.692 | 5.213 | 0.0240 | 0.0240 |
| Kq8_0D-Vq8_0D | 4.976 | 5.033 | 6.015 | 7.019 | 0.0283 | 0.0283 |
| Kq8_0X-Vq8_0D | 4.320 | 4.388 | 5.096 | 5.833 | 0.0284 | 0.0284 |
| Kq8_0S-Vq8_0D | 4.842 | 4.922 | 5.976 | 7.002 | 0.0285 | 0.0285 |
| Kq8_0D-Vq8_0X | 4.913 | 4.986 | 5.996 | 7.050 | 0.0283 | 0.0284 |
| Kq8_0D-Vq8_0S | 3.460 | 3.494 | 3.770 | 4.112 | 0.0198 | 0.0199 |
| Kq8_0X-Vq8_0X | 4.476 | 4.536 | 5.194 | 5.937 | 0.0284 | 0.0284 |
| Kq8_0X-Vq8_0S | 3.486 | 3.537 | 3.774 | 4.122 | 0.0200 | 0.0201 |
| Kq8_0S-Vq8_0S | 3.873 | 3.930 | 4.201 | 4.588 | 0.0201 | 0.0201 |
| **PR-dec-q8_0** (PR verbatim) | 5.398 | 5.459 | 6.431 | 7.443 | 0.0309 | 0.0310 |
| Kq4_0D-Vq4_0D | 6.728 | 6.863 | 7.607 | 8.494 | 0.0334 | 0.0335 |
| Kq4_0X-Vq4_0D | 5.992 | 6.085 | 6.623 | 7.350 | 0.0333 | 0.0333 |
| Kq4_0S-Vq4_0D | 6.462 | 6.622 | 7.318 | 8.232 | 0.0334 | 0.0334 |
| Kq4_0D-Vq4_0X | 4.893 | 4.994 | 5.607 | 6.372 | 0.0287 | 0.0288 |
| Kq4_0D-Vq4_0S | 3.462 | 3.506 | 3.704 | 3.920 | 0.0201 | 0.0201 |
| Kq4_0X-Vq4_0X | 4.314 | 4.385 | 4.759 | 5.286 | 0.0282 | 0.0282 |
| Kq4_0X-Vq4_0S | 3.358 | 3.409 | 3.579 | 3.797 | 0.0197 | 0.0197 |
| Kq4_0S-Vq4_0S | 3.528 | 3.579 | 3.712 | 3.905 | 0.0201 | 0.0200 |
| Kbf16D-Vbf16D | 6.734 | 7.050 | 8.855 | 10.407 | 0.0330 | 0.0331 |
| Kbf16X-Vbf16D | 4.451 | 4.473 | 4.523 | 4.790 | 0.0216 | 0.0217 |
| Kbf16S-Vbf16D | 4.640 | 4.690 | 4.745 | 5.009 | 0.0220 | 0.0221 |
| Kbf16D-Vbf16X | 5.258 | 5.467 | 6.748 | 7.972 | 0.0291 | 0.0291 |
| Kbf16D-Vbf16S | 3.514 | 3.547 | 3.764 | 4.306 | 0.0198 | 0.0198 |
| Kbf16X-Vbf16X | 3.689 | 3.709 | 3.770 | 4.180 | 0.0197 | 0.0197 |
| Kbf16X-Vbf16S | 3.409 | 3.450 | 3.601 | 4.100 | 0.0191 | 0.0192 |
| Kbf16S-Vbf16S | 3.591 | 3.618 | 3.706 | 4.176 | 0.0194 | 0.0195 |

### Bound regime (derived from the tables above)
- **Text is issue-bound up to n_kv 135k for every type.**
  - Kf16D-Vf16P ns/chunk: wiki d4k 34.0, wiki d131k 33.7 (`ns/chunk` column of summary_prefill.md). Time grows with the union only.
  - The quantized-K penalty against f16 has the same sign at d4k and d131k:

    | K loader | d4k | d131k |
    |---|---|---|
    | q8_0 D | +11.6 % | +13.0 % |
    | q8_0 X | +3.9 % | +5.5 % |
    | q4_0 D | +17.3 % | +21.8 % |
    | q4_0 X | +2.2 % | +3.7 % |
    | bf16 D | +9.0 % | +12.4 % |
    | bf16 X | +1.4 % | +4.3 % |

    Fewer bytes do not help here. Cutting the issued work does: X removes the duplicated hi/lo dequant, and VALU drops 415 → 382 for q8_0 (`res/isa_loop_counts.md`).
- **Random tokens at d131k are partly memory-bound for f16.**
  - ns/chunk is 39.9, against 34.0 on text.
  - q8_0 / q4_0 K X are −9.6 % / −10.8 % against f16 K (26.02 / 25.66 vs 28.77 ms). The fewer bytes win.
- **The synthetic C6 sweep (fixed work, uniform spread)** — K sweep, ratio syn 1M / syn 4k:

  | K loader | ratio |
  |---|---|
  | f16 D | 3.25 |
  | bf16→f16 X | 3.09 |
  | bf16w X | 3.21 |
  | q8_0 X | 1.74 |
  | q4_0 X | 1.52 |

  - Against the 1.3× rule of Y X4 / C6: f16 D is 1.78× already at 131k; q8_0 X is 1.29× at 262k and 1.74× at 1M.
  - In the memory-bound regime the per-type cost follows the bytes: f16 512 B, q8_0 272 B, q4_0 144 B per head-row.
- **Consistency with C6.** The C6 V0 f16 synthetic numbers are 15.25 / 27.98 / 49.06 ms at 4k / 131k / 1M. Here they are 15.58 / 29.11 / 49.45 (PR-V0-f16), with a different data fill. q8_0 V0 at 1M: 28.44 in C6, 28.41 here.

### Noise
- **Run to run** (separate processes r1/r2, same fixture), |r1 − r2| / mean over the 68 rows of each fixture:

  | fixture | median | p90 | max |
  |---|---|---|---|
  | wiki d4k | 2.2 % | 2.5 % | 2.7 % |
  | rand d4k | 0.3 % | 0.5 % | 0.5 % |
  | wiki d131k | 0.3 % | 1.1 % | 1.7 % |
  | rand d131k | 0.1 % | 0.2 % | 0.2 % |

  - Decode, over 34 rows per fixture: median 0.1-0.8 %, max 1.4 %.
  - In wiki d4k the r2/r1 shift is uniform across variants (every one of the 70 rows has r2/r1 between 1.018 and 1.028). So ratios between variants in one run are tighter than the absolute spread.
- **Within a run**, sd/median per (variant, layer): median 0.84-0.97 %, p90 1.35-1.75 %, max 3.1 % (`wiki131k_r1`, `wiki4k_r1`, `rand131k_r1`, `dec512_wiki131k_r1`).
- **Synthetic seed to seed** (s1 vs s2: different rows and data):
  - median 0.2-1.7 % per nk;
  - max 9.1 % at syn 131k, the f16 transition point C6 reported (Kbf16Dd row).
  - The 131k syn cells are therefore ±5 % for f16-class loaders.
- **Data sensitivity.** Real vs SYN Q/K/V on wiki d131k (layers 0, 5, 11; 174 variant×layer cells): SYN/real median 1.002, range 0.985–1.023. The r2/r1 ratio over the same cells is 0.990–1.019. Timing does not depend on the data.
- **Decision margins against this noise:**
  - split vs PR q8 K: −6.7 % to −6.9 % on text, at least 3× the p90 noise;
  - packed vs direct V on text: ≥ 2.3 % for f16 (V X at wiki d131k: 18.55 vs 18.14, about 2× the p90 noise) and ≥ 10 % for the quantized types;
  - decode V-staged: ≥ 19 %;
  - LDS-staged vs split K: −0.7 % to +1.5 % for quantized types, which is inside the noise, so it is called a tie.

### Decision (rules: X PF-4 / PF-3 "per-type loader trait = the faster one at both depths" plan-v2-X.md:264; Z E-P3 plan-v2-Z.md:225; W E5 plan-v2-W.md:262; C6 decision 3: a direct V may replace the pack only if ≤ 2 % slower than V0 at the 4k fit cell and not slower than V0 + PACKV at 131k)
1. **`sa_loader<T>::staging` for the prefill attend kernel (n_q > 512).**

   | type | K | V, n_kv < 64·n_q | V, n_kv ≥ 64·n_q |
   |---|---|---|---|
   | f16 | direct | packed | packed |
   | q8_0 | split | packed | direct split-gather |
   | q4_0 | split | packed | direct split-gather |
   | bf16 → f16 | split | packed | packed |

   - "split" = per-half-wave dequant + `v_permlanex16`, with no LDS.
   - LDS-staged loses or ties on both K and V in this kernel at every depth, so no type gets it.
   - Packed V passes against every direct V at both depths (d4k and d131k, text and random):
     - The C6 rule fails for every direct V at the 4k cell. At syn 4k, direct V is +7.2 % (f16 X) and +19 % (q8_0 X) against packed.
     - Direct V wins only for quantized types at ≥ 262k spread: q8_0 −12 % at 262k, −34 % at 1M; q4_0 −1.7 % / −36 %.
     - Packed is never beaten for f16/bf16. At 1M it is 62.97 ms against the best direct, 63.65 (V S).
   - The 64·n_q crossover was measured only at ub4096, at 131k (packed better by 9.6 % for q8_0 on syn, 12 % on text) and at 262k (direct better). Its position for other n_q and between 131k and 262k is UNPROVEN.
   - The rewrite must re-measure it with its own kernel before freezing a threshold.
2. **The decode attend kernel (n_q ≤ 512: tg and ub ≤ 512) uses V = LDS-staged transposed (wave-private slot) for every type, with K = direct for f16/q8_0 and split for q4_0/bf16.** Measured gains against each type's direct D+D:

   | type | ub512 wiki d4k | ub512 wiki d131k | tg wiki d131k |
   |---|---|---|---|
   | f16 | −21 % | −19 % | −9 % |
   | q8_0 | −30 % | −37 % | −30 % |
   | q4_0 | −50 % | −53 % | −41 % |
   | bf16 | −49 % | −59 % | −42 % |

   (Best of D+S / X+S.)
3. **PR #219 reuse.**
   - **Taken as-is:**
     - `qsa3_q8x2_to_h2` / `qd_q8x2_to_h2`, the exact q8→half conversion (bit-identical outputs, see Correctness);
     - the decode q8 K loader (`qsa-decode-wmma.cuh:63-74`, within 1 % of split once V is staged);
     - the PR's `n_q·64 < n_kv` direct-V switch, as the threshold for q8_0, but no longer applied to f16.
   - **Replaced:**
     - the prefill q8 K loader (`qsa-prefill.cu` `load_k` Q8 branch) by split: −6.7 % kernel time on wiki d131k, −6.9 % on d4k;
     - the prefill DIRECT_V q8 loader. It spills 120 B/lane. At d131k text it takes 24.24 ms where packed + split K takes 18.44. At 1M it takes 30.13 where split-gather takes 24.60.
     - the decode q8 V loader by the LDS-staged V: −41 % at ub512 d131k (6.431 → 3.770 ms), −36 % at tg (30.9 → 19.8 µs);
     - the PR's f16 DIRECT_V at n_kv > 64·n_q. It is never faster than packed (+4.0 % at 1M, PR-VD-f16 65.91 vs PR-V0+PACKV 63.38).
   - The `Dd` (deferred-only) form is rejected. It is slower than split on every fixture, and q8_0 Dd spills 64 B.
   - These are the facts for the "what was taken from #219" list.
4. **The "no cliff" status after the trait choice** (UNPROVEN until it is re-measured on the rewrite's kernel):
   - **Prefill text:** best q8_0 / q4_0 / bf16 at 1.017-1.033 / 0.986-1.017 / 1.018-1.047 of the f16 route.
   - **Decode:** q8_0 / q4_0 / bf16 at +11.9 % / +6.2 % / +6.9 % against the best f16 at ub512 wiki d131k (3.770 / 3.579 / 3.601 vs 3.370 ms), and +6 % at tg.
   - A residual decode q8_0 K-dequant cost therefore remains. The next lever is the K side of the decode kernel, for example K staged together with V in one slot. That lever is not measured here.
5. **bf16.** Use the f16-class path: bf16 → f16 saturating, split K, packed V.
   - Native bf16 WMMA (`bf16w`) is the fastest bf16 option (0.974-1.016 of f16), but its CPU-reference error is 10-40× larger (Q/P rounded to bf16).
   - It may be adopted only after a model-quality (pinned KLD) check, which C18 did not run: UNPROVEN.
6. **Out of scope here and still open:** Vulkan loaders (no Vulkan code was run), RDNA other than gfx1151, and n_q between 513 and 4095 for the prefill kernel. Each is UNPROVEN.

### Large files
- Deleted after the runs: `cap/wikiS_p4096.qkv` (1.4 GB), `cap/randS_p4096.qkv` (1.4 GB), `cap/wikiL_p131072.qkv` (4.5 GB), and the duplicate `cap/{wiki,rand}L_p4096.rows`.
- Kept in `~/dev/wt_lab/c18/cap` for re-runs: 4 row files of 403 MB each (`wikiS_p4096`, `randS_p4096`, `wikiL_p131072`, `randL_p131072`; 1.6 GB in all).
- `~/dev/wt_lab/c18/isa` holds 12 MB.
- Results and logs (4.6 MB) are copied to `SP/final219/c18/remote/`.

## C19

**Question:** does `strip_views` (#219, 68da4191d, re-keyed by b60447506) still cut memory on the remaining non-scalar path? The author's numbers (L7) had not been checked.

**Answer: yes, but the experiment as proposed cannot show it.**
- Image request B puts its image at position 0. The image ubatch is then inside the dense shortcut, and every text ubatch is scalar. The strip views are built, but they are dead graph nodes. Bypassing `strip_views` therefore changes nothing on request B (a null result).
- When the non-scalar path really runs, bypassing `strip_views` costs a lot:
  - **+1120 MiB** of ROCm0 compute buffer at runtime for one 1024-token image ubatch 57k cells deep, on HIP q8_0;
  - +1125 MiB for the same case on HIP f16, and +1378 MiB on Vulkan f16;
  - **+8336 MiB** of reserved buffer at -c 65536 on the HIP masked route, and +8416 MiB on Vulkan q8_0;
  - the masked-route slope goes from 0.029 to 0.148 MiB/cell, which reproduces the author's "~0.15 MiB per cell".

### Setup (strix-002, 2026-11-04 22:20 to 2026-11-05 00:00 CET; every GPU run under `flock ~/gpu.lock`; I was the only agent)
- **Tree:** `~/dev/wt_lab/c19/tree`, a clean `git worktree add --detach` at 94e086847, with three changes:
  1. **qwen4exp.cpp**, inside the `strip_view` lambda (PR qwen4exp.cpp:1604-1610):
     - `LAB_C19_NO_SHARE=1` returns `make()` directly. That bypasses the map and gives a fresh view per layer and strip, as before 68da4191d.
     - `LAB_C19_LOG=1` logs every call (il, src, first, nq, nkv, new, bytes).
  2. **llama-context.cpp**, after `ggml_backend_sched_alloc_graph`: `LAB_C19_LOG=1` logs `ggml_backend_sched_get_buffer_size` of each backend whenever it changes (`C19 buf` lines).
     - This is needed because a Release build never prints ggml-alloc's "reallocating … buffer" line. It sits under `#ifndef NDEBUG` (PR ggml/src/ggml-alloc.c:1197-1204).
     - So the "realloc lines" metric of the proposed experiment is blind in these builds.
  3. **Vulkan build only:** `git apply ~/dev/wt_lab/c11/fix_6411b4315.diff`. It is the same truncation fix as C11, and PR 94e086847 does not compile Vulkan without it.
  - The full diff is `~/dev/wt_lab/c19/c19.patch` (copy in `SP/final219/c19/remote/c19.patch`).
  - With neither env var set, the lab binary is the PR. The pristine `~/dev/wt_pr219v2/build/bin/llama-server` gives identical buffers (see below).
- **Builds:**
  - HIP: the HIP flags from the rules, `cmake --build build -j 16 --target llama-server llama-perplexity`, rc 0.
  - Vulkan: `cmake -B build-vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DCMAKE_CXX_FLAGS=-I$HOME/.local/spirv-headers/include -DSPIRV-Headers_DIR=$HOME/.local/spirv-headers/share/cmake/SPIRV-Headers`, then `--target llama-server`, rc 0.
- **Server command:** `~/dev/wt_lab/c19/c19_srv.sh BIN TAG KV CTX "ENV…" REQ…`.
  - The flags are those of `~/repro_img/repro_kv.sh`: Signal-3.8 Q4_K_XL, MTP sidecar, mmproj, `-fa on -fit off --image-min-tokens 1024 -v`, all devices ROCm0 or `DEV=Vulkan0`.
  - Changed: **`-ub 4096 -b 4096`**, `-c CTX`, `-ctk/-ctv KV`.
  - Each run is one fresh server process. The batch drivers are `batch1-4.sh` and their outputs `batch1-4.out` (copies in `SP/final219/c19/remote/`). The per-log extract is `SP/final219/c19/log_extract.txt`.
- **Requests** (real text plus a real image, `~/repro_img/img.png`, 32×32 = 1024 image tokens, log `image_tokens->nx = 32`):
  - **B** = `~/repro_img/B_img_text60k.json`: 69803 tokens, so it does not fit -c 65536 and was run at -c 73728.
  - **B65** = B with the text cut to lines 10000..12299 (`c19/mkb65.py`): 58553 tokens, image first.
  - **B65t** = the same content with the image after the text (`c19/mkb65t.py`): the image ubatch starts at cell 57504 (log `non-consecutive token position 57504`) with n_kv 58624.
  - Every run answered correctly (`'12299'` for B65/B65t, `'12749'` for B; `batch*.out`).

### Why request B is a null test (code + logs)
- **The image ubatch takes the dense shortcut.**
  - At position 0 it has n_kv ≤ top_k + ratio − 1 = 2051, so the layer takes the dense shortcut (PR qwen4exp.cpp:1877, L23) and builds no selection at all.
  - In B65, all 1760 `C19 sv` calls have `src=attn_inp_kq_mask`. None has a bias src (`srv_share_c65k_r1.log`). That is 12 QSA layers + MTP layer 48 × 8 strips × 15 graphs.
- **Every text ubatch is scalar (compact).**
  - `bias` is therefore `nullptr` (qwen4exp.cpp:1612), and `tail_idxs` exists only when scalar (qwen4exp.cpp:1472-1476).
  - The strip loop takes the complete-block branch and `continue`s (qwen4exp.cpp:1644-1654) before `query_mask` is read (qwen4exp.cpp:1662-1668).
  - So the kq_mask strip views are built and never consumed.
- **The scheduler confirms it.** With `GGML_SCHED_DEBUG=1`, the inputs of the main 4096-token reserve graph are the same in both arms: split #1 ROCm0 has 210 inputs, 167.2 MiB, and no `attn_inp_kq_mask` (`logs/srv_dbg_{share,noshare}_c65k_init.log`, parsed by `c19/splits.py`).
- **The same holds on Vulkan f16**, which takes the selected-key route: B65 at 1498.56 / 1122.22 MiB in both arms (`srv_vk_{share,noshare}_f16_c65k_r1.log`).

### Raw numbers (MiB; "reserve" = `sched_reserve: <dev> compute buffer size` of the main context, then the MTP context; "runtime max" = largest `C19 buf` of the main context during the request)

**HIP, default route (q8_0 → selected-key, "QSA attention: selected-key kernels … (K q8_0, V q8_0)")**

| case | arm | reserve main / MTP | runtime max | runs |
|---|---|---|---|---|
| B65, -c 65536 (proposed experiment) | share | 1567.19 / 1401.22 | 1567.19 (no growth) | 3 (r1, r2, b4) |
| B65, -c 65536 (proposed experiment) | no-share | 1567.19 / 1401.22 | 1567.19 (no growth) | 3 |
| B65, -c 65536 (proposed experiment) | pristine PR binary | 1567.19 / 1401.22 | n/a (no log) | 1 |
| B unchanged, -c 73728 | share | 1568.00 / 1401.34 | n/a (batch 1) | 1 |
| B unchanged, -c 73728 | no-share | 1568.00 / 1401.34 | n/a | 1 |
| **B65t (deep image), q8_0, -c 65536** | **share** | 1567.19 / 1401.22 | **1567.19** | 2 (b4 r1, r2) + 2 earlier with identical reserve |
| **B65t (deep image), q8_0, -c 65536** | **no-share** | 1567.19 / 1401.22 | **2687.62 (+1120.43)** | 2 (r1 = r2) |
| B65t, f16, -c 65536 | share | 1566.92 / 1401.22 | 1566.92 | 1 |
| B65t, f16, -c 65536 | no-share | 1566.92 / 1401.22 | 2691.98 (+1125.06) | 1 |
| B65t, pristine PR binary | — | 1567.19 / 1401.22 | n/a | 1 |

- **Mechanism of the deep-image growth.** In B65t the 1024-token image ubatch takes the masked (non-compact) path.
  - Each layer calls `strip_view` for 2 strips × {bias, kq_mask}: 30015488 B (28.6 MiB) and 60030976 B (57.25 MiB) per view (`srv_img2_*_c65k_r1.log`, the lines after `decoding image batch`).
  - With no-share, the image graph's split #1 ROCm0 has **67 inputs, 2195.0 MiB**: 48 view copies plus the 114 MiB mask (`logs/srv_dbg_img2_noshare.log`, `splits.py`).
  - Copies are made per distinct src tensor when the consuming backend cannot read the input's buffer (PR ggml-backend.cpp:1694-1714). They are all allocated at the start of the split (ggml-backend.cpp:1833-1846).
  - ROCm cannot read the ROCm_Host buffer, because `integrated` is forced to false (PR ggml-cuda.cu:362) and `supports_buft` needs it for host buffers (ggml-cuda.cu:9422-9426).

**HIP, masked route forced (`LLAMA_QSA_SELECTED_KEY=0`, log "masked top-k"; the q8_0 route of BASE (L17) and of the author's scenario), B65**

| -c | share reserve main / MTP | no-share reserve main / MTP | Δ main |
|---|---|---|---|
| 32768 | 2585.39 / 1672.58 | 6809.39 / 1672.58 | 4224.00 |
| 65536 | 3337.77 / 2152.95 (3 runs, identical) | 11673.77 / 2152.95 (3 runs, identical) | **8336.00** |
| 98304 | 4490.14 / 2913.33 | 16538.14 / 2913.33 | 12048.00 |

- The runtime max equals the reserve in both arms (`srv_b4_msk_*`).
- The 32768 and 98304 rows are init-only runs, with no request (`srv_msk_*_init.log`).

**Vulkan0 (RADV), q8_0 (probe → masked route, "no selected-key kernels for this backend and cache type"), B65**

| -c | share reserve main / MTP | no-share reserve main / MTP | Δ main |
|---|---|---|---|
| 32768 | 2636.68 / 1121.72 | 6900.68 / 1121.72 | 4264.00 |
| 65536 | 3757.05 / 1122.22 (3 runs) | 12173.17 / 1122.22 (3 runs) | **8416.12** |
| 98304 | 4757.78 / 1154.72 | 17965.78 / 1154.72 | 13208.00 |

- On Vulkan the main buffer shrinks after warm-up and then grows again with depth. The largest runtime value is 3475.64 (share) against 10647.28 (no-share) (`srv_b4_vk_q8_B65_*`). Why Vulkan shrinks the buffer is UNPROVEN; it was not investigated.
- Vulkan f16, B65t (deep image, selected-key route): share stays at 1498.56 → 1497.27. No-share grows to **2875.01 (+1377.74)** (`srv_b4_vk_f16_img2_*`).

**Slopes (MiB per KV cell, main context reserve; endpoints 32768 → 98304)**

| route | share | no-share |
|---|---|---|
| HIP selected-key (default), init only | 0.0000992 (1563.94 / 1567.19 / 1570.44) | 0.0000992 (identical) |
| HIP masked (forced) | **0.0291** (0.0230 for 32k→64k, 0.0352 for 64k→96k) | **0.1484** (0.1484, 0.1484) |
| Vulkan q8_0 masked | **0.0324** (0.0342, 0.0305) | **0.1688** (0.1609, 0.1768) |

- The author's claim was 7193 → 2969 MiB at 32k and "~0.15 MiB per context cell" (68da4191d message). Here, at 32768 on the HIP masked route, the bypass gives 6809.39 MiB and the PR gives 2585.39 MiB, with a slope of 0.148 MiB/cell. **The mechanism and the size of the effect are reproduced.** The ~400 MiB absolute gap is not explained (UNPROVEN; different tree and flags).
- The author's 10.7 GB at 262k and his "perplexity bit-identical" were not measured here: UNPROVEN.
- What was checked here instead:
  - Every share and no-share run gave the same answer.
  - By code, the shared view is the same `ggml_view_*` of the same src with the same (first, n_query). The other view parameters (n_kv, n_stream, nb) are fixed by the src within one graph (qwen4exp.cpp:1612-1617). So sharing cannot change values. That is a code argument, not a bit-compare.

### Noise
- **Buffer sizes are deterministic.** Every repeated arm gave bit-identical reserve and runtime sizes:
  - HIP B65 share ×3 and no-share ×3;
  - HIP B65t q8_0 share ×4 and no-share ×2 with the runtime log (2687.62 both);
  - HIP masked ×3 per arm;
  - Vulkan q8_0 ×3 per arm.

  Noise on the memory metric is 0.00 MiB. The smallest decision margin (+1120 MiB) is therefore exact.
- **Prompt t/s** (server `prompt_per_second`, one sample per run; not a decision input):

  | case | share | no-share |
  |---|---|---|
  | HIP B65 | 1053.3 / 1068.5 / 1035.3 | 1055.6 / 987.6* / 1031.6 |
  | HIP B65t q8_0 | 1066.8 / 1053.5 / 1044.2 / 1058.4 | 1056.4 / 1059.0 / 1049.3 / 1020.6 |
  | HIP masked | 932.4 / 942.9 / 916.0 | 905.0 / 896.3 / 912.4 |
  | Vulkan q8_0 B65 | 646.4 / 759.3 / 763.4 | 682.8 / 708.4 / 689.7 |
  | Vulkan f16 B65t | 1157.7 | 1133.7 |

  - \* This run overlapped my own CPU-side build-vk compile attempts.
  - Spread between runs: about ±2 % on HIP and up to ±8 % on Vulkan q8_0.
  - The masked-route no-share arm is 2-3 % slower than share on HIP. That is about the size of the spread, so it is not claimed.

### Decision
1. **strip_views is load-bearing and must be recovered.**
   - On the remaining non-scalar path (an image or other 2-D ubatch deeper than the dense shortcut), the PR's sharing keeps the compute buffer at its reserve. Without it the buffer grows by +1120 MiB (HIP q8_0), +1125 MiB (HIP f16) and +1378 MiB (Vulkan f16) for one 1024-token image at -c 65536 ub4096. The growth scales with n_kv × strips × QSA layers (view bytes above).
   - On any masked route (Vulkan q8_0 today; HIP whenever the probe fails), it saves 8.3-8.4 GiB at 65536 cells and cuts the slope 5×.
2. **"What was taken from #219" wording (sourced):**
   - "strip_views (68da4191d; key on strip size from b60447506), ported as-is.
   - Measured on strix-002 (MEASUREMENTS C19):
     - without it, a deep image ubatch grows the ROCm0 compute buffer from 1567 to 2688 MiB at -c 65536 ub4096;
     - on the masked route, the buffer is 11674 MiB instead of 3338 MiB at 65536 cells (0.148 vs 0.029 MiB/cell);
     - the author's 0.15 MiB/cell is reproduced.
   - Request B with its image at position 0 does not exercise it."
3. **Regression test for the compute-buffer slope.**
   - X's G4 (< 0.002 MiB/cell) is met only where strip_views is not exercised: 0.0001 on the default HIP reserve.
   - The masked route of the PR, with sharing, is at 0.029 (HIP) and 0.032 (Vulkan) MiB/cell. So a single global G4 threshold would fail the PR's own masked route. The residual 0.029-0.032 MiB/cell in the shared arm is not attributed: UNPROVEN.
   - A slope test read from `sched_reserve` on the default route cannot catch a strip_views regression. The reserve graph is scalar there, and Release builds print no realloc line.
   - The test must therefore have two parts:
     - **(a) Reserve slope on a forced masked route** (`LLAMA_QSA_SELECTED_KEY=0`, or Vulkan q8_0), at two contexts. Pass < 0.06 MiB/cell. That is about 2× the shared value and 0.4× the unshared one.
     - **(b) A non-scalar ubatch at depth** (text, then a 1024-token image at ≥ 32k cells), with `ggml_backend_sched_get_buffer_size` read after `ggml_backend_sched_alloc_graph`. Pass: no growth above the reserve. Shared growth here is 0; unshared growth is +1120 MiB.
   - The thresholds are inferred from this run's numbers: UNPROVEN as a general rule.
4. **Request B (image at position 0) must not be the image test for this property.** Use B65t-like requests with the image deep in the context. Request C has its later images deep (180157 tokens, needs -c ≥ 180k) and probably exercises it too. That is UNPROVEN: C was not run here.

### Large files
- None were created (> 1 GB). Logs total 85 MB in `~/dev/wt_lab/c19/logs`.
- The worktree with its build/ and build-vk/ is 1.3 GB at `~/dev/wt_lab/c19/tree`, kept for re-runs.
- No processes were left running (`pgrep` empty).

## R1-The 'one CB rule everywhere' (D6) is und

**Question.** Does PLAN-v3's W8 rule for non-compact ubatches (D6: complete-block + tail, built as in `c4_only.diff` hunk 3, then invisible cells → −1) change behaviour against PR219 on unified multi-slot layouts: a shared prefix that ends inside a block, and a second slot holding an image? C4's evidence covered only single-sequence layouts.

**Answer. Yes. The objection holds, and the effect is larger than the objection says.** W8 as written is fully defined. It is wrong on these layouts: it drops visible cells that PR219 keeps, among them the query's own cell. It also wipes out image understanding in a unified 2-slot run. **D6 must not be applied to non-compact layouts as written.**

Two code mechanisms cause it. Both are confirmed by in-graph dumps and by a CPU port. The PR219 line numbers refer to `~/dev/wt_pr219v2` @ 94e086847.

- **M1: the spare ("dead") block.** The non-compact scan groups cells by (position block, exact sequence set) (`src/llama-memory-hybrid-idx.cpp:721`). Only groups that fill all 4 slots get a block id (`:796`). Every other cell is mapped to one spare block (`:845`).
  - PR219 (TE) gives the spare block +1e9 (`:982`), and it gives the own block +1e9 too (`:975`). Then `top_k(expanded, 2051)` (`src/models/qwen4exp.cpp:1674`) keeps all of these "forced" cells, as long as there are ≤ 2051 of them.
  - Hunk 3 (CB) removes every 1e9 block from the block choice (`c4_only.diff:53-54`, `non_tail`). It keeps only `top_k(expanded, r−1 = 3)` of the forced cells (`:60`).
  - In f32, 1e9 + score rounds to one value (`F_distinct_vals` ≈ 1 below). So which 3 survive is set by top_k tie order, and the own cell is often lost.
  - Triggers: `seq_cp` of a prefix of length 4k+1/2/3 (a split block holds {0,1} cells plus own cells), and an image whose temporal-position block is not full.
- **M2: an oversized "complete" group.** Ranking by (pos, y, x) runs only when exactly one sequence is in the cache (`:755`, `dup && is_pos_2d && one_seq`). With 2 sequences, all 1036 image cells share one position and fall into one group. If the text cells next to the image fill the other 3 slots, that group counts as complete. It then holds 1039 cells.
  - `cell_blk` maps all 1039 cells to the block (`:845`), so TE scores every one of them (`qwen4exp.cpp:1659`).
  - `blk_cells` keeps only 4 member slots, last writer wins (`:842`). CB expands a block through `blk_cells` (`c4_only.diff:56`), so it can select at most 4 of the 1036 image cells.
- **Why C4 did not see this.** C4's REF matched in-graph CB only on layouts where every reference group was exactly one llama block ("reference groups that are one llama block: 2547165/2547165", `c4/an_p2_s3.txt:5,16`). Those were single-sequence ranked layouts. PR219 says itself that a split block has no complete group (`src/prefix.h:12-14`).

### Setup (strix-002; GPU lock held for every run; sources and outputs mirrored in `SP/final219/r1/`)
- **Binary.** `~/dev/wt_lab/build4/bin/lab-r1`, built from `wt_lab` @ 94e086847 with the existing default-off lab patches; the C4 switch is `LAB_QSA_C4`.
  - The only new file is `tools/mtmd/lab-r1.cpp` (md5 a41972959c46b9be757d0bcd56648146), added to `tools/mtmd/CMakeLists.txt`. Build: `cmake build4 && cmake --build build4 -j 16 --target lab-r1`. Nothing else recompiled: libggml-hip is unchanged from 06:05, and the lab sources predate it.
  - `-np 2 --kv-unified` is set in code (`params.n_parallel = 2; params.kv_unified = true`), because the MTMD argument parser rejects both flags.
- **Mode switch** (`LAB_QSA_C4`, c4_only.diff):
  - **0** = PR219 behaviour, with TE rows feeding the masked FA.
  - **2** = the W8 rule: hunk-3 CB rows feed the masked FA. The kq_mask added in `build_attn_qsa` drops invisible cells, which is the same set as W8's −1 filter.
  - **1** = TE feeds attention, and the CB rows are built and dumped next to it.
- **Run command** (`r1/run.sh`): `flock ~/gpu.lock env LAB_QSA_C4=<0|1|2> LAB_R1_SCEN=<text|image> LAB_TEXT_FILE=~/data/wikitext-2-raw/wiki.test.raw … lab-r1 -m ~/models/signal3.8/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf [--mmproj mmproj-Signal-3.8-Flash-Next-f16.gguf --image ~/dev/wt_lab/c4/img_nyt.jpeg --image-min-tokens 1024] -c <16384|40960> -b UB -ub UB -ngl 99 -fa on -ctk f16 -ctv f16 -lm mmap -lzm on-direct -fit off -n 160 -t 8`.
- **Text workload** (`LAB_LENS=8192,8193,8194,8195`, one process, `llama_memory_clear` between lengths):
  1. seq 0 decodes wikitext tokens [0, L);
  2. `llama_memory_seq_cp(0, 1, -1, -1)`;
  3. one batch with 40 own tokens per sequence (seq 1's text starts 100000 tokens further on);
  4. one batch with 100 teacher-forced wikitext answer tokens per sequence, i.e. 200 fixed answer tokens per request pair.
  - L = 8192 (block-aligned) is the control: it stays compact, so both modes must be identical.
- **Image workload.**
  - seq 0: 2000 wikitext tokens, then 100 answer tokens.
  - seq 1: prompt `r1/p3.txt` (`c4/mkprompt.py wiki p3.txt 136500 5500`): **32779 text tokens, the 1036-token NYT image at n_past 32779, then 1573 text tokens plus the headline/newspaper/date/photo question** (`logs/img_te_gen.log`, LAB_CHUNK lines).
  - Free generation in both modes. The pinned answer is TE's greedy output (`answer_img.txt`, 95 tokens), so TE is favoured by construction, as in C4. It is teacher-forced together with seq 0's 100 tokens in one 2-sequence batch.
- **Campaign.**
  - Text, `r1/text_campaign.sh`: te_s1, te_s2, cb_s1, cb_s2 at ub 8192; te_u4, cb_u4 at ub 4096. Then one fresh process per length for te/cb (`txt_*_one*`), plus te1 (mode 1).
  - Image, `r1/img_campaign.sh`: te_gen, cb_gen, then te_s1, te_s2, cb_s1, cb_s2, te_u4, cb_u4, te1_s and the dump.
- **Metric.** The C4 `kld.py` method, split per request (`r1/kld2.py`): KLD(P_A‖P_B) over the fixed-answer positions, the summed answer log-prob, and top-1 agreement. Raw lines are in `SP/final219/r1/kld_results.txt`.
- **Dumps** (mode 1, layers 3/23/43): `lab_te_rows`, `lab_cb_rows`, `lab_blk_cells`, `lab_blk_score`, `indexer_score_tokens`. They were taken on text L=8194 (cont + ans batches) and on the image run (image ubatch + the text chunk after it), and analysed by `r1/r1_analyze.py`.
  - visible = finite `indexer_score_tokens` (score + block bias + kq_mask).
  - forced = visible cells with a value > 5e8.
  - own cell = the cell that a row sees and the previous row of the same sequence does not (known for 100 % of rows).

### Step 1: CPU-only oracle, no GPU (`r1/r1_oracle.py`, output `r1/oracle.out` md5 32b6e3e1…)
- **What it ports:** the grouping, block ids, member slots, spare block and bias of `set_input_qsa_scan` (unranked, since 2 sequences are present), the TE top_k, and hunk-3 CB. Block scores are random f32 and 1e9 is added in f32. Tie order is reported both ways.

| layout | queries | visible forced cells mean (min–max) | forced cells CB drops, mean (max) | TE drops | own cell dropped by CB, ties low-first / high-first |
|---|---|---|---|---|---|
| L=8192 (control) | 80 | 1.50 (0–3) | 0 | 0 | same as TE (17 / 17, random-score effect, also 17 under TE) |
| L=8193 | 80 | 5.30 (2–7) | 2.33 (4), 76 rows | 0 | 58 / 15 (TE 15) |
| L=8194 | 80 | 5.40 (3–7) | 2.40 (4), 78 rows | 0 | 77 / 62 (TE 17) |
| L=8195 | 80 | 5.50 (4–7) | 2.50 (4), 80 rows | 0 | 65 / 15 (TE 15) |
| 2 seqs, image at pos 28930 (= 2 mod 4, C4 p2 depth), image rows | 346 | 520.5 (3–1038) | 517.5 (1035) | 0 | 345 / 0 |
| same, text rows after the image | 443 | 1040.5 (1039–1042) | 1037.5 (1039) | 0 | 436 / 321 (TE 108) |
| 2 seqs, image at pos 32779 (= 3 mod 4, the measured p3 layout) | 346 / 525 | 0 / 1.5 | 0 / 0; **M2 instead: one complete group of 1039 cells, member list 4** | 0 | n/a |

- **Reading the table.**
  - The control and the CB−TE own-cell counts under "high-first" equal TE's. These are random-score artefacts: TE also misses an own cell whose ordinary block loses on a random score. The tie-independent quantity is the forced-cell drop.
  - The oracle's TE−CB on the M2 layout (3) depends on the score. With real scores it is ~500–1000 (below).
  - **The oracle reproduces the GPU dumps exactly where they are comparable:**
    - text L=8194, cont batch forced histogram {3:2, 4:20, 5:20, 6:20, 7:18}, against the dump {3:1, 4:10, 5:10, 6:10, 7:9} per sequence × 2;
    - p3 layout: 0 forced on image rows, and the {0..3} cycle on text rows.

### Step 2: in-graph dumps, GPU (mode 1, TE and CB built in the same graph; `r1/an_t1_cont.txt`, `an_t1_ans.txt`, `an_img.txt`, `an_txt.txt`)
| phase / layer | rows | visible forced cells mean (min–max) | forced dropped by CB, mean (max), rows | TE − CB visible cells, mean (max) | own cell dropped: CB / TE |
|---|---|---|---|---|---|
| text L8194 cont (2 × 40), L3 / L23 / L43 | 80 | 5.40 (3–7) | 2.40 (4), 78/80 | 2.40 (4) | 60 / 0; 52 / 0; 57 / 0 |
| text L8194 ans (2 × 100), L3 / L23 / L43 | 200 | 5.50 (4–7) | 2.50 (4), 200/200 | 2.50 (4) | **150 / 0; 125 / 0; 141 / 0** |
| image ubatch (seq 1, 2 seqs in cache), L3 | 1036 | 0 | 0 (M2, not M1) | **518.5 (1035)** | **1035 / 0** |
| image ubatch, L23 / L43 | 1036 | 0 | 0 | 498.6 (1035) / 223.1 (1035) | 1035 / 53; 1035 / 510 |
| text after the image, L3 / L23 / L43 | 1573 | 1.50 (0–3) | 0 | **1024.1 / 1024.7 / 644.8 (max 1035)** | 0 / 0; 0 / 0; 0 / 2 |

- **Forced cells are exact ties.** `F_distinct_vals` is 1.00–1.19 per row, so CB's 3 survivors are chosen by top_k tie order. On text, CB keeps exactly 3 forced cells in every row and loses the query's own cell in 62–75 % of rows (125–150 of 200 answer rows, 52–60 of 80 cont rows). TE loses it in 0 rows.
- **On the image request, CB loses up to 1035 of the 1036 image cells.**
  - Image rows lose them through the 4-member list (M2).
  - TE rows keep the oversized block whole. For text queries after the image, TE − CB ≈ 1024 cells, i.e. the image.
- **CB rows hold 3–4 invisible or duplicate entries per image row** (`CB_invis_or_dup`). They come from the tail top-3 repeating cells of the best block (hunk-3 comment).

### Step 3: end to end, new rule vs PR219, per request
**Text** (200 fixed tokens = 100 per request; KLD mean over all 200, with per-seq detail in kld_results.txt):

| L | rule: te vs cb ub8192 / ub4096 / fresh process | noise: te repeat, cb repeat, te ub8192→4096, cb ub8192→4096, fresh vs cleared | rule ÷ max noise | top1, rule / noise |
|---|---|---|---|---|
| 8192 (control) | 0 / 0 / – | 0, 0, 0, 0 | – | 100 % / 100 % |
| 8193 | 6.59e-2 / 5.67e-2 / 6.28e-2 | 1.40e-2, 1.40e-2, 1.28e-2, 1.58e-2, 1.62e-2 | 3.5–4.1× | 85.0–88.5 % / 93–94.5 % |
| 8194 | 9.40e-2 / 8.80e-2 / 8.04e-2 | 1.42e-2, 2.28e-2, 1.91e-2, 1.68e-2, 1.77e-2 | 3.5–4.1× | 84–89 % / 90.5–94.5 % |
| 8195 | 7.96e-2 / 8.01e-2 / 8.38e-2 | 0.96e-2, 1.78e-2, 1.22e-2, 1.62e-2, 0.93e-2 | 4.5–4.7× | 84–87 % / 93–95 % |

- **Per request.** Seq 1's rule KLD is 0.083–0.149, against at most 0.034 noise. Seq 0's is 0.031–0.044, against at most 0.012.
- **Rule vs noise.** The rule effect exceeds every noise pair in all 9 (L, arm) cells. The smallest ratio is 5.67e-2 / 2.28e-2 = 2.5×.
- **Summed log-prob is not decisive at n = 200.**
  - te → cb: −4.68 / −2.03 / +8.85 (8193), −3.50 / +1.12 / −4.80 (8194), −6.84 / −9.35 / −11.41 (8195).
  - Noise pairs swing −3.9 to +11.0.
- **PR219's masked route is not run-to-run deterministic on these layouts** (te_s1 vs te_s2 KLD 0.96–1.42e-2; mode 1 vs mode 0 1.64e-2), while the compact control is bit-identical. The cause is UNPROVEN. The exact ties shown above, under GPU top_k, are the candidate.
  - So the earlier single run t1_dump (−106.78 seq 0) is within run-to-run spread, and no memory-clear effect is claimed: fresh vs cleared is 0.93–1.77e-2, the same size as a repeat.

**Image request** (seq 1, 95 pinned-answer tokens; seq 0 is the text-only slot, 100 tokens):

| pair | seq 1 KLD mean (max) | seq 1 log-prob | seq 1 argmax | seq 0 KLD mean |
|---|---|---|---|---|
| te_s1 vs cb_s1 | **1.821e-1 (4.93)** | −10.58 → **−32.13** | 95/95 → 87/95 | 4.18e-3 |
| te_s2 vs cb_s2 | 1.749e-1 (5.02) | −10.67 → −31.77 | 95/95 → 88/95 | 5.15e-3 |
| te_u4 vs cb_u4 | 1.737e-1 (4.68) | −10.85 → −31.86 | 94/95 → 87/95 | 3.41e-3 |
| noise: te rep / cb rep / te mode1 / te ub / cb ub | 9.04e-4 / 3.44e-3 / 8.29e-4 / 9.86e-4 / 2.98e-3 | Δ ≤ 0.36 | – | 3.47e-3 – 6.67e-3 |

- **Image request:** the rule effect is ≥ 50× the largest noise (1.737e-1 / 3.44e-3), with −21.0 to −21.6 nats on 95 tokens.
- **The text-only slot (seq 0) is within noise.** It has only complete blocks and ≤ 3 forced cells, so CB = TE there.
- **Free generation** (`logs/img_te_gen.log`, `logs/img_cb_gen.log`; this is the image-content check):
  - **PR219 (TE):** "MEN WALK ON MOON", The New York Times, Monday, July 21, 1969, astronaut on the lunar surface. Correct.
  - **W8 rule (CB):** **"GIANT WAVE SWAMPS SHIP", The Daily Mirror, Friday, September 15, 1961**, a wave over a ship. Wrong on every field. The model no longer sees the image.

### Decision
1. **R1 is confirmed. D6 and W8 are rejected for non-compact layouts as written.**
   - "CB reproduces C4's REF" holds only where every group is one llama block. Single-sequence ranked layouts give exactly that (C4); unified multi-sequence layouts do not.
   - W8's done-when is unmeetable as stated, and "answers correct" is not a quality proof. The measured case here fails both: answer and KLD.
2. **The rule this PR ships for non-compact ubatches: keep PR219's TE set.**
   - Rows = `top_k(expanded, 2051)` exactly as `qwen4exp.cpp:1658-1674` builds them today. Cells the kq_mask hides go to −1, and the rows are sorted ascending before the plan (C3 decision 4: unsorted rows cost about 2.5 ms/layer, and only non-compact ubatches pay it). They feed the sparse op.
   - Attention then runs over exactly PR219's set, so new and PR219 differ only by the op's numerics. Q10 checks this.
   - TE and the masked half of `build_attn_qsa` are removed as *attention routes* only. TE stays as the non-compact *row producer*.
   - Rejecting these layouts with an error is not an option: they arise from ordinary `-np N --kv-unified` serving (prompt-cache `seq_cp`, images in any slot).
3. **Any future CB-style rule for these layouts must be stated against llama's real groups.** The candidate: every visible forced cell (spare block, own block) is kept as an extra entry, which set semantics allow, with n_sel = 2048 + |forced| where needed. Oversized groups, i.e. more than r cells mapped to one block id, are expanded through `cell_blk`, not through the 4-slot `blk_cells`.
   - Measured bounds of |forced|: 7 on text, 1042 on the M1 image layout (oracle). An M2 group is 1039 cells.
   - Its CPU oracle starts from `r1/r1_oracle.py`, which already ports the grouping. It must pass Q10 before replacing TE. This is UNPROVEN until then.
4. **New quality gate Q10 (unified 2-slot, non-compact), run as this experiment** (`r1/text_campaign.sh`, `r1/img_campaign.sh` with `LAB_QSA_C4` replaced by new-vs-PR219 binaries):
   - Text control L = 4k: KLD = 0 against PR219 (compact path).
   - Text L ∈ {4k+1, 4k+2, 4k+3}: per-request KLD over the 2 × 100 fixed tokens ≤ the same-rule noise of that L, measured in the same campaign: max over repeat and ub8192/4096 of PR219. That was 1.40–2.28e-2 (all) here, and ≤ 3.4e-2 per request.
   - Image (p3, 2 slots): seq 1 KLD ≤ the PR219 same-rule noise (9.0e-4 – 3.4e-3 here). Seq 0 KLD ≤ 6.7e-3. The free answer must give the headline, the newspaper and the date correctly.
   - Thresholds come from 2 repeats + 1 ub pair per cell. Their tightness is UNPROVEN; one more repeat per arm would firm them.
5. **U2 is closed as "CB does not match PR219 on these layouts."** The census check of W8 ("non-compact rows in the census") remains. It must use these two workloads, not a deep image alone.

### Large files and state
- **Deleted:** the logits (7.0 GB, `~/dev/wt_lab/r1/logits`) and the image dumps (1.6 GB, `dump/img`).
- **Kept:** `dump/t1` (55 MB), the logs and the analysis texts (`~/dev/wt_lab/r1`, 57 MB), and the mirror in `SP/final219/r1/`.
- No processes are left running (`pgrep -a lab-r1` empty).
- Nothing outside `~/dev/wt_lab` was modified.

## R1-Skipping the non-compact reserve for sin

**Question.** PLAN-v3 §7.1.4 (PLAN-v3.md:304) builds the non-compact reserve variant only when `n_seq_max > 1 && kv_unified`. Can a context outside that condition (non-unified multi-slot = split streams; single sequence with mtmd) still produce a non-compact ubatch, and what does it cost unreserved? Decides: the condition of §7.1.4, and whether a predicate failure without the variant must be a hard error.

### Code verification (PR head 94e086847, `~/dev/wt_pr219v2`)
- **Split streams reach the same predicate with both sequences in `rows`.** `qsa_scalar_visibility` loops over the streams but passes the *whole* ubatch to `qsa_scalar_visibility_cells` for each stream's cells (src/llama-memory-hybrid-idx.cpp:1723-1735). That function builds `rows` from every token of the ubatch (src/prefix.h:18-27), so `one_seq = rows.count()==1` (prefix.h:35) is false for a 2-stream ubatch. As soon as either stream holds a 2-D cell, `pos_2d` is non-empty (prefix.h:40; text ubatches of this arch have `is_pos_2d()` true because `n_pos >= 3`, src/llama-batch.h:50-53) and prefix.h:45-46 returns false. Result: every split-stream pp ubatch with n_tps >= 128 goes non-compact (`scalar=0`, F32 bias + f16 KQ mask) once any slot holds image cells, although each stream holds exactly one sequence.
- Split-stream ubatches with n_tps < 128 are non-compact regardless of images (`(n_stream == 1 || n_tps >= 128)`, src/models/qwen4exp.cpp:1453-1454); that is the C15 tg route, covered by the existing tg reserve and deleted by PLAN-v3 §4.4.
- **`kv_unified` default (closes U18).** `common_params::kv_unified = false` (common/common.h:649). The server forces it to true only when `-np` is left at auto (`n_parallel < 0` → 4 slots + unified, tools/server/server.cpp:168-172; `-np` default -1, common/arg.cpp:1429). So **every explicit `-np N` server is split-stream by default**, which is outside the §7.1.4 condition. Confirmed at runtime: `kv_unified = 'false'` (srv_split_img_q8.log, `load_model: initializing, n_slots = 2, n_ctx_slot = 65536`).
- **`n_pos_per_embd > 1` is true for every qwen4exp model, text-only included.** `llama_model_rope_type` returns IMROPE for LLM_ARCH_QWEN4EXP unconditionally (src/llama-model.cpp:3788-3790) and `n_pos_per_embd()` is 4 for IMROPE (src/llama-hparams.cpp:285-287). A condition "n_pos_per_embd > 1" therefore means "always" on this arch: it would reserve the 3 B × n_ctx × n_ubatch residual (PLAN-v3.md:292) on every run, e.g. 3 × 261888 × 8192 B = 6.0 GiB host at the owner's c261888 ub8192 (arithmetic), plus the strip copies.

### Setup and commands (strix-002, 2026-11-05 01:14–01:45 CET, every GPU run under `flock ~/gpu.lock` inside the script)
- Build: the C15 instrumented PR tree `~/dev/wt_lab/c15/tree` (detached 94e086847 + `c15_trace.patch`, built 2026-11-04 19:39; no source newer than `build/bin/libllama.so`, checked with `find -newer`). Trace on with `GGML_C15=1` (C15Q route per layer, C15L per ubatch actual vs expected buffer sizes with `!OVER`, C15A plans with `GROW`, dumps in `GGML_C15_DUMP`).
- Script `~/dev/wt_lab/r1s/r1s.sh TAG KV EXTRA REQ0 REQ1 DELAY` (copy: SP/final219/r1s/r1s.sh): Signal-3.8 + mmproj + MTP sidecar, the `~/repro_img/repro_kv.sh` flags except `--ctx-size 131072 --parallel 2 -ub 1024 -b 1024`, `-ctk/-ctv q8_0`, `-lv 3`. REQ0 is sent, then REQ1 after DELAY = 5 s, both concurrent.
- Requests (`~/dev/wt_lab/r1s/mkreq.py`, copy SP/final219/r1s/mkreq.py, same line generator as `~/repro_img/mkreq.py`, real text, not random tokens): `S0_img` = 1 image (`~/repro_img/img.png`, `--image-min-tokens 1024`) + lines 10000..11999 (51053 prompt tokens, answer 11999); `S0_txt` = same text without the image (50027 tokens); `S1_txt` = lines 30000..31999 (50027 tokens); `T_deep` = lines 30000..32517 (62977 tokens, answer 32517), sized so that its last batch before the server's `n_prompt − (4 + n_ubatch)` checkpoint break (tools/server/server-context.cpp:4310-4325) leaves 511 tokens for the other slot.
- The first request lands in slot 1 (seq 1) and the second in slot 0 (`seq0=1` on the image ubatch); slot 0 fills each batch first, so 2-stream ubatches form at slot 0's checkpoint break.
- Commands:
  - `./r1s.sh split_img_q8 q8_0 "" S0_img S1_txt 5`
  - `./r1s.sh split_img_deep_q8_r1 q8_0 "" S0_img T_deep 5` and `..._r2` (repeat)
  - control, no image: `./r1s.sh split_txt_deep_q8 q8_0 "" S0_txt T_deep 5`
  - control, unified: `./r1s.sh unified_img_deep_q8 q8_0 "--kv-unified" S0_img T_deep 5`
- Logs: `~/dev/wt_lab/r1s/logs/srv_<tag>.log`, dumps `logs/dump_<tag>/` (24 MB total).

### Raw numbers
All answers correct: S0_img `'11999'`, S1_txt `'31999'`, T_deep `'32517'`, S0_txt `'11999'` in every run.

| run | 2-stream ubatch (ctx) | C15Q route of that ubatch | buffer growth at that ubatch (C15A GROW) | `~llama_context` at exit, actual vs reserve |
|---|---|---|---|---|
| split_img_q8 | 149+149 @ pos [48128..] / [2232..], main (ctx_type 0) and MTP (ctx_type 1) | `n_kv=48384 n_stream=2 n_tps=149 scalar=0 bias=f32[12096,149,2]` (il=3) | main Host 56959008 → 59969472 B (+2.87 MiB); MTP Host 56668160 → 59687296 B (+2.88 MiB); ROCm0 none (plan 375.73 MiB < 763.45) | main Host 57.1913 vs 51.4615 MiB; MTP Host 56.9222 vs 51.1836 MiB |
| split_img_deep_q8_r1 | 511+511 @ pos [61440..] / [2084..2594], MTP only (main split the same batch into 513 seq-0 + 511 seq-1 one-stream ubatches) | `il=48 n_kv=61952 n_stream=2 n_tps=511 scalar=0 bias=f32[15488,511,2]` | **MTP ROCm0 777183360 → 939552768 B (+154.85 MiB); MTP Host 56668160 → 243803104 B (+178.47 MiB)** | **MTP ROCm0 896.0273 vs 741.1798 MiB; MTP Host 232.5088 vs 51.1836 MiB**; main Host 54.3203 vs 51.4615 |
| split_img_deep_q8_r2 | identical | identical | identical to the byte (939552768 / 243803104) | identical |
| split_txt_deep_q8 (control, no image) | same 511+511 shape, MTP | `il=48 n_kv=61952 n_stream=2 n_tps=511 scalar=1 bias=i32[15999,2,1]` | none (ROCm0 777183360 = exp) | main Host 54.3203 vs 51.4615; MTP Host 54.0430 vs 51.1836 (both = the dense-shortcut mask of C15, plans #16-#19 at n_kv ≤ 2048); no ROCm0 line |
| unified_img_deep_q8 (control, `--kv-unified`) | one stream, 2 sequences in the cache | e.g. `il=3 n_kv=100608 n_stream=1 n_tps=1024 scalar=0 bias=f32[25152,1024,1]` (seq-1 ubatches after the image) | grows through plan #255 | main ROCm0 1529.1448 vs 769.4189 MiB (+759.73); MTP ROCm0 1225.7578 vs 741.5587 (+484.20); Host 384.6248 vs 51.5904 and 384.3475 vs 51.3125 (+333.03 each) |

- Top tensors of the split deep overrun (dump plan #146, MTP): host `attn_inp_kq_mask` f16 [61952,511,1,2] 126629888 B; ROCm0 `attn_inp_kq_mask (copy)` f32 [61952,511,1,2], `indexer_score_tokens-48` f32 [61952,511,2], GET_ROWS [511,61952,2], 253259776 B each; top chunk `(view) (cont)` f32 [15488,511,2]. This is the non-compact scorer of §4.3 (mask + F32 bias), not the image route.
- In the main context, a text-only one-stream ubatch of the image slot right after the image is compact (`il=3 n_kv=3840 n_stream=1 n_tps=511 scalar=1`, srv_split_img_deep_q8_r1.log): single-sequence text after an image is already compact in the PR.
- Why the main context split 513+511 into two one-stream ubatches while MTP formed one 2-stream ubatch: UNPROVEN (candidate: the `n_keep_tail` argument of `split_equal`, src/llama-memory-hybrid-idx.cpp:186-191). The main context did form a 2-stream image-affected ubatch in split_img_q8 (149+149), so both contexts are exposed.
- **Noise.** The allocator is deterministic: r1 and r2 give the same plan sizes to the byte and the same route lines. Prompt t/s (logging on, 1 sample per config, not a perf claim): T_deep 846.9 / 832.1 / 840.2 / 827.9; S0_img 374.2 / 373.2 (split), 308.0 (unified).

### Run 2 (single sequence on W9): infeasible
- W9 does not exist: there is no `~/dev/wt_rewrite` on strix-002 (`ls ~/dev/`), so the extended predicate of §4.2 and `LLAMA_STRICT_RESERVE` are not implemented anywhere. Request C and the back-to-back-images-at-depth > 32k layout cannot be checked against them. U1 stays UNPROVEN.
- What the PR shows for single sequence: only embd (image) ubatches are non-compact (`ubatch.token` required, src/models/qwen4exp.cpp:1110; C15), and text ubatches after images are compact (above, and C15 request C). After §4.2, any layout the extended predicate rejects falls back to the same non-compact scorer, whose unreserved cost C15 measured at +1625.4 MiB ROCm0 main / +1209.2 MiB MTP (ub1024, n_kv 253696).

### Answer
1. **The objection holds for split streams, measured.** A non-unified `-np 2` server (the default for any explicit `-np`) produces non-compact 2-stream pp ubatches as soon as one slot holds image cells: `scalar=0 bias=f32[15488,511,2]` vs `scalar=1 bias=i32` for the identical ubatch shape without the image. Unreserved, this costs **+154.85 MiB ROCm0 and +181.32 MiB ROCm_Host above the reserve in the MTP context at n_kv 61952, ub1024** (2/2 runs identical to the byte), and +2.9 MiB host in both contexts with a 149+149 ubatch. The cost scales as n_kv × n_tps × n_stream (the [n_kv, n_tps, 2] f32 tensors above).
2. The cause is a predicate artifact: each stream holds one sequence, but the per-stream check passes `rows` of both streams (src/llama-memory-hybrid-idx.cpp:1731 + src/prefix.h:18-35).
3. The unified control shows the case §7.1.4 does cover: +759.73 MiB ROCm0 main, +484.20 MiB MTP, +333 MiB host each, unreserved in the PR today.
4. The single-sequence half (U1) is not settled: no W9 build exists.

### Decision
- **§7.1.4 condition becomes `n_seq_max > 1` (unified or split).** Both are measured to reach the non-compact scorer. Also **fix the split-stream predicate** so each stream is checked with only its own tokens' sequence ids. A split-stream text ubatch then has the single-sequence status of §4.2, and the 511+511 image-slot case becomes compact. The variant stays reserved for `n_seq_max > 1` until the strict-reserve test shows 0 non-compact split-stream ubatches on the run above.
- **Do not use `n_pos_per_embd > 1` as the trigger.** It is true for every qwen4exp model (src/llama-model.cpp:3788-3790), so it would reserve about 6.0 GiB host on every text run at c261888 ub8192.
- **Single-sequence contexts without the variant:** when the §4.2 predicate rejects a ubatch, `llama_decode` must return an error with a log line naming the layout. It must never build an unreserved non-compact graph. This stays the rule until U1 is proven.
- **Add an M4 / `LLAMA_STRICT_RESERVE=1` recipe** with 0 violations required on W9:
  - (a) this R1 workload (`r1s.sh … S0_img T_deep`, `-np 2`, split, then `--kv-unified`);
  - (b) `-np 1`, request C (5 images, 180k);
  - (c) a synthetic single-sequence prompt with two images back to back across a ubatch boundary at depth > 32k.
  - On (b) and (c), 0 predicate rejections are also required on image ubatches.
- Status: the split-stream gap is resolved by measurement. The single-sequence half stays inconclusive (W9 absent).

## R1-The new quantized dense-shortcut route (

**Question.** D4 routes every non-f16 K/V pair through all-visible rows plus the sparse op while n_kv ≤ 2051. No existing gate scores that route. Is it quality-neutral against PR219's dense FA? Is the adopted producer (C1 lab `LLAMA_QSA_ALLVIS=2`, `ggml_view_2d(m, m->ne[0], n_tokens, m->nb[1], 0)`) wrong for a 4-D `[n_kv, n_tps, 1, n_stream]` mask? This decides whether Q11 is added, how the producer is specified, and which test is added.

**Answer.**
1. **Quality: no loss measured.** All-visible rows with the sparse op, against PR219's dense shortcut, pass the paired 3·SE_paired rule on all 6 K/V pairs, on two references, at ub512 and on the decode-like ub16 path. The difference is ≤ 0 on 21 of 22 (pair, reference, region) cells. The one positive cell is q4_0 at +0.42 SE.
2. **Producer: the objection is confirmed, but its mechanism is not.**
   - The 2-D view does not read pad rows. There are none in this tree. The bytes it reads are correct.
   - The view does have the wrong shape for n_stream = 2. The graph build then **aborts**: `ggml.c:5758: GGML_ASSERT(top_k->ne[1] == a->src[0]->ne[1]) failed`, rc 134, measured.
   - A 4-D producer built over the whole mask fixes this.
     - At n_stream = 1 it gives logits bit-identical to the 2-D producer.
     - At n_stream = 2 it gives logits bit-identical to n_stream = 1, in 32/32 sequences.
3. **PR219 has no GPU kernel for part of this route.** In the shortcut, its maskless op falls back to the CPU without notice for q4_0 and bf16. It also falls back for mixed pairs and for split streams at n_q < 128 (code and trace below). The rewrite's op must cover these cases.

### Code verification (PR head 94e086847, `~/dev/wt_pr219v2`)
- **No pad rows.** `KQ_MASK_PAD` occurs nowhere in `ggml/ src/ common/ tools/` (grep is empty).
  - The mask is `ggml_new_tensor_4d(type, n_kv, n_tokens/n_stream, 1, n_stream)` (src/llama-graph.cpp:43). It is contiguous, so stream s's rows follow stream s−1's rows.
  - The lab view `ggml_view_2d(m, ne0, n_tokens, nb1, 0)` (SP/final219/c1/lab.diff:139) therefore reads exactly the right values, but its shape is `[n_kv, n_tps·n_stream]`.
- **Why the 2-D shape aborts.** `build_attn_mha` permutes q to `[d, n_tps, H, n_stream]` (llama-graph.cpp:3032-3034). `ggml_flash_attn_ext_add_top_k` asserts `top_k->ne[1] == q->ne[1]` (ggml.c:5758). With 2 streams, 2·n_tps ≠ n_tps.
  - The dense shortcut condition has no stream term (`if (shortcut && n_kv <= width)`, qwen4exp.cpp:1877), so every split-stream context reaches this branch.
- **Kernel coverage of the maskless op on HIP (PR).**
  - qsa_decode accepts only K = V ∈ {f16, q8_0} and only one stream (`v->type != k->type`, f16/q8_0 at qsa-decode.cu:91-92; `q->ne[3] != 1` at :97).
  - qsa_prefill accepts f16 or q8_0 for each of K and V separately, n_q ≥ 128, and n_stream > 1 only when maskless (qsa-prefill.cu:613-620).
  - Any other maskless op makes `ggml_cuda_flash_attn_ext_supported` return false (fattn.cu:1199-1208), so the scheduler assigns it to the CPU.
- **Gate coverage (the claim's premise).**
  - Q1 and Q3 use the ub4096 reference (PLAN-v3.md:577, :583, :585), where the first ubatch already has n_kv 4096.
  - Q6 and Q7 are as the objection says (:588-589).
  - **Correction to the proposed Q11.**
    - At c8192 ub512, only ubatches 0–3 (positions 0–2047) are in the shortcut. The decode-like tail of a chunk has n_kv ≈ 8192.
    - Default llama-perplexity scoring starts at n_ctx/2 (perplexity.cpp `first = n_ctx/2`). So Q11 as proposed never scores a shortcut position directly. It needs `QKL_TAIL`, or a chunk size ≤ 2048.

### Setup (strix-002, 2026-11-05 01:44–04:24 CET, every GPU run under `flock ~/gpu.lock`)
- **W7 does not exist** (`ls ~/dev | grep -c wt_rewrite` → 0), so the W7-vs-PR219 run is not possible. The "before W7" variant was run instead, on an extended copy of the C1 lab binary.
- **Binary.** `~/dev/wt_lab/r1av/tree` is a git worktree at 94e086847 with three changes (full diff: `r1av_full.diff`, md5 264f3bd03deb5627ea27f25af73ae2e0):
  - `c1/lab.diff` (ALLVIS=2 and the `GGML_QSA_TRACE` kernel trace);
  - the `~/dev/wt_measQ` QPIN hook diff (`qpin_measQ.diff`);
  - `r1av_patch.py`, which adds:
    - **`LLAMA_QSA_ALLVIS=3`**: the same step / scale_bias / mul / cast construction over the whole 4-D mask, with the assert `m->ne[2]==1 && m->ne[1]*m->ne[3]==n_tokens`;
    - **`QKLD_CHUNK`**: a per-chunk KLD line at %.9e (W12), split at position `QKL_SPLIT` = 2048. "lo" = positions < 2048, computed in the shortcut. "hi" = the rest.
  - With no env set, this binary is the hooked PR219 binary (ALLVIS=0 = PR219 dense shortcut).
  - Build: the mandated HIP flags, `--target llama-perplexity -j 16`.
- **Runner** `run.sh TAG REF MODE PIN CTK CTV B UB CTX CHUNKS ENV…`:
  `flock ~/gpu.lock env [QPIN…] GGML_QSA_TRACE=1 <ENV> llama-perplexity -m ~/models/qwen3.8-flash-next-UD-Q4_K_XL/…00001-of-00004.gguf -f ~/data/wikitext-2-raw/wiki.test.raw -c CTX -b B -ub UB --chunks CHUNKS -ngl 99 -fa on -lm mmap -lzm on-direct -ctk CTK -ctv CTV --kl-divergence-base … [--kl-divergence]`
  - Real text only. Paired analysis: `pair.py` (mean per-chunk diff, SE_paired = sd/√n, pass if diff ≤ 3·SE_paired).
- **Campaign A** (`campA.sh`), the proposed Q11:
  - Reference: f16 b/ub 4096 c8192 × 8 with QPIN record and `QKL_TAIL=8191`, so positions 1..8190 are scored.
  - Arms: `-b 512 -ub 512`, `QPIN_PIN=qsa,moe`, ALLVIS=0 vs ALLVIS=2.
- **Campaign D** (`campD.sh`), the direct shortcut gate:
  - Reference: f16 b/ub 2048 c2048 × 16, which is dense FA. MoE ids are recorded. No QSA rows exist in the shortcut.
  - Arms are MoE-pinned at ub512 for the 6 pairs, and at ub16 for q8_0 and f16. ub16 is the per-query qsa_decode path; QPIN needs ≥ 16 tokens per ubatch.
  - Every ubatch has n_kv ≤ 2048, and positions 1024..2046 are scored.
- **Campaign B** (`campBC.sh`), the task's quick read: unpinned, base logits from ALLVIS=0 of the same type, c8192 × 8, b/ub 512, `QKL_TAIL=8191`.
- **Campaign C** (split streams): `-c 1024 -b 2048 -ub 2048`.
  - With these flags `n_parallel = n_batch/n_ctx = 2` (perplexity.cpp:2012) and `kv_unified` keeps its default false (common/common.h:649). That gives n_stream = 2, n_tps = 1024 and n_kv 1024. The log shows `n_seq=2`.
  - The np1 arm runs `-b 1024 -ub 1024`.
- **Unit test** `test_allvis_rows.cpp` (md5 9e89eeb70e8bf45d6d27bc238d2edd9f). It builds both constructions with ggml on ROCm0 and on the CPU and compares the rows with {j : mask = 0} per (stream, row). It also checks the shape the FA top_k setter asserts.
  - Synthetic causal split-stream masks: np2 at n_tps 1 / 128 with n_kv 256 / 512 / 2048, different lengths per stream; plus np1 n_tps 512.
  - Recorded C4 image-ubatch FA masks: p1 layers 3 and 43 `[6656, 1036]`, p2 layer 3 `[30208, 1036]`.

### Raw numbers
**Unit test** (`test_allvis_rows.out`, RESULT PASS):
- The 4-D construction gives `values_mismatch = 0` with shape `[n_kv, n_tps, 1, n_stream]` in all 8 cases on both backends.
  - Covered: np2 n_tps 1 (4096 and 512 entries), np2 n_tps 128 (524288 and 131072), np1 512 (1048576), and the 3 recorded image masks (6895616 / 6895616 / 31295488 entries; 2124836 visible each).
- The 2-D view also gives 0 value mismatches, but shape `[n_kv, 2·n_tps, 1, 1]` → `fa_top_k_assert FAIL` in all 4 np2 cases.
- No mask value other than 0 and −inf exists in any case (`other_mask_values=0`).
- **The recorded image masks are not the raw kq_mask.** They are the masked-route FA masks (kq_mask ∪ TE selection): 2124836/1036 = 2051.0 visible cells per row, with n_kv > 2051. They test the elementwise construction on an irregular image-ubatch zero pattern. A raw M-RoPE image kq_mask inside the shortcut (an image within the first ~1000 cells) was not recorded: **UNPROVEN** beyond the elementwise argument.

**Campaign A (QSA + MoE pinned, f16 ub4096 sparse reference, n = 8 chunks)**. Values are KLD; diff = ALLVIS=2 − ALLVIS=0 (SE_paired, t):

| K/V | ALLVIS=0 all | ALLVIS=2 all | lo diff | hi diff | all diff |
|---|---|---|---|---|---|
| q8_0 | 1.6317e-4 | 1.5254e-4 | −1.71e-5 (8.63e-6, −1.98) | −8.47e-6 (3.47e-6, −2.44) | −1.06e-5 (3.69e-6, −2.88) |
| q4_0 | 6.3192e-3 | 6.3329e-3 | +5.91e-5 (1.41e-4, +0.42) | −1.5e-6 (5.54e-5, −0.03) | +1.37e-5 (3.53e-5, +0.39) |
| bf16 | 1.7473e-4 | 1.6840e-4 | −2.37e-5 (7.52e-6, −3.15) | −5.6e-7 (2.03e-6, −0.28) | −6.33e-6 (3.02e-6, −2.09) |
| q8_0K/f16V | 1.4038e-4 | 1.3439e-4 | −3.00e-5 (7.70e-6, −3.90) | +2.01e-6 (7.08e-6, +0.28) | −5.99e-6 (6.36e-6, −0.94) |
| f16K/q8_0V | 1.6152e-4 | 1.4728e-4 | −2.72e-5 (8.17e-6, −3.33) | −9.94e-6 (5.95e-6, −1.67) | −1.42e-5 (5.18e-6, −2.75) |
| f16 (control; D4 keeps dense) | 1.3638e-4 | 1.1367e-4 | −3.62e-5 (5.55e-6, −6.53) | −1.82e-5 (1.47e-5, −1.24) | −2.27e-5 (1.15e-5, −1.97) |

- **Noise.**
  - Replay into itself (`self_qm`): −0.000000.
  - q8_0 repeats: ALLVIS=0 r2 − r1 and ALLVIS=2 r2 − r1 are 0.000e+00 in 8/8 chunks for lo, hi and all (deterministic).
  - **ALLVIS=3 − ALLVIS=2 (q8_0, np1) = 0.000e+00 in 8/8 chunks.**
  - Reorder: q8_0 ALLVIS=0 ub1024 − ub512 gives lo +1.19e-7 (2.53e-7), hi +8.94e-6 (7.77e-6), all +6.73e-6 (5.83e-6).
- **Census.** In every arm, rows_absent = 0 and moe_absent = 0, with moe_pin = 3145720. rows_pin is 589824 (cb, q8_0 and f16) or 1179648 (tk, other pairs, PR219's masked route past the shortcut, L72).
- **Shortcut kernels** (C1TRACE, n_kv ≤ 2048).
  - ALLVIS=0: `dense_tile_vec_mma nq 512 mask 1` (f16: one `fa7`).
  - ALLVIS=2:
    - q8_0, f16: `qsa_decode nq 512 mask 0`;
    - mixed pairs: `qsa_prefill nq 512 mask 0`;
    - **q4_0, bf16: no HIP trace, i.e. CPU fallback.** These are hooked runs with logging, so this is not a perf claim: 14.76 → 17.62 s per pass (q4_0) and 13.72 → 17.63 s (bf16).
- **Bias of this reference.** At ub4096 the reference ran the CB sparse kernel at every position, so ALLVIS=2 shares its kernel family. Campaign D removes this by using a dense reference.

**Campaign D (MoE pinned, f16 dense ub2048 reference, n = 16 chunks, all scored positions in the shortcut)**. diff = ALLVIS=2 − ALLVIS=0 (SE_paired, t):

| arm | ALLVIS=0 | ALLVIS=2 | diff |
|---|---|---|---|
| q8_0 ub512 | 1.9817e-4 | 1.6244e-4 | −3.57e-5 (5.22e-6, −6.85) |
| q4_0 ub512 | 7.5295e-3 | 7.4056e-3 | −1.24e-4 (1.08e-4, −1.15) |
| bf16 ub512 | 1.7173e-4 | 1.4280e-4 | −2.89e-5 (7.74e-6, −3.74) |
| q8_0K/f16V ub512 | 1.8131e-4 | 1.4563e-4 | −3.57e-5 (3.66e-6, −9.76) |
| f16K/q8_0V ub512 | 1.8851e-4 | 1.4667e-4 | −4.18e-5 (4.15e-6, −10.08) |
| f16 ub512 (control) | 1.6158e-4 | 1.3440e-4 | −2.72e-5 (5.72e-6, −4.75) |
| q8_0 ub16 (qsa_decode nq 16 vs dense nq 16) | 6.8998e-3 | 6.7445e-3 | −1.55e-4 (9.21e-5, −1.69) |
| f16 ub16 | 6.7452e-3 | 6.5629e-3 | −1.82e-4 (9.75e-5, −1.87) |

- **Noise.**
  - `D_self` −0.000000.
  - q8_0 ub512 repeats of both arms: 0.000e+00 in 16/16 chunks.
  - ALLVIS=0 ub1024 − ub512: 0.000e+00 in 16/16 chunks. Dense FA is batch-invariant here.
- **ub16 floor.** ALLVIS=0 ub16 − ub512 = +6.70e-3 (4.06e-4). ub16 changes GDN chunking and MoE batching (C8, plan-v2-Z.md:350), so the ub16 cells have about 18× larger SE_paired.
- **Cause UNPROVEN.** In D, ALLVIS=2 is closer to the f16 dense reference than PR219's dense kernel is, for every pair including f16. The cause is not established. PR219's vec kernel quantizes Q to q8_1 for quantized K (fattn-vec.cuh:97), but the f16 control shows the same sign, so this does not explain it alone.

**Campaign B (unpinned, base = ALLVIS=0 of the same type, n = 8).** Mean KLD:

| K/V | ALLVIS=0 rerun | **ALLVIS=2** (lo / hi) | ALLVIS=0 ub1024 (lo / hi) |
|---|---|---|---|
| q8_0 | −0.000000 | **8.274e-3** (1.077e-2 / 7.44e-3) | 5.144e-3 (3.5e-6 / 6.86e-3) |
| q4_0 | 0.000000 | **1.8111e-2** (2.12e-2 / 1.71e-2) | 1e-6 (2.9e-6 / 0) |
| bf16 | 0.000000 | **8.309e-3** (1.07e-2 / 7.50e-3) | 1e-6 (2.2e-6 / 0) |

- **ALLVIS=0 vs ALLVIS=0 reorder noise is no yardstick here.** With dense FA in the shortcut, ub1024 vs ub512 differs by about 3e-6 in lo for every type. Dense FA and the MoE matmuls are batch-invariant, so this is not a kernel change.
- q8_0's hi value of 6.86e-3 at ub1024 is a kernel change: past the shortcut, ub1024 moves the CB route from qsa_decode (G1) to qsa_prefill (G4), the same mechanism as C1 D3. It has the same size as ALLVIS=2's hi value, 7.44e-3.
- The unpinned values sit at the known MoE-cascade floor: L65 gives 0.0113–0.0123 for ub reorders on the PR binary, and L69 shows that QSA-only pinning removes none of it. So the unpinned read can only say "floor class". It cannot gate, which matches Q9's status.
- q4_0 is about 2× q8_0 unpinned. Under MoE pins the q4_0 difference is +0.42 SE (A) and −1.15 SE (D), so the factor is cascade amplitude, not a route loss. Its cause is UNPROVEN.

**Campaign C (split streams, n_stream 2, n_tps 1024, n_kv 1024, 32 sequences, unpinned vs ALLVIS=0 np2):**
- **ALLVIS=2 (the C1 lab producer): rc = 134, `ggml.c:5758: GGML_ASSERT(top_k->ne[1] == a->src[0]->ne[1]) failed`,** for q8_0 and q4_0 (`logs/C_*_av2.log`).
- ALLVIS=3 runs:
  - q8_0: `qsa_prefill nq 1024 nkv 1024 mask 0`, KLD 1.1464e-2;
  - q4_0: no HIP trace (CPU), KLD 2.3148e-2.
- **ALLVIS=3 np2 − ALLVIS=3 np1: 0.000e+00 in 32/32 sequences for both types.** ALLVIS=0 np1 vs the np2 base: 0.000000 for both types.
  - So with the 4-D rows, each stream computes exactly its single-stream result. This is the graph-level check of the per-stream rows.
- **Not measurable on PR219:** split streams at n_tps < 128 (tg with `-np 2`). The maskless op has no HIP kernel there (qsa-decode.cu:97 needs ne[3] = 1, qsa_prefill needs n_q ≥ 128). The same holds for mixed pairs at n_q < 128 (qsa-decode.cu:91-92). The prototype would run these on the CPU.

### Decision
1. **Add Q11 to §15, in two forms. Both compare new vs PR219, are paired per chunk at %.9e, and pass at ≤ 3·SE_paired (one-sided), separately for lo (< 2048), hi and all.**
   - **Q11a** (the proposed form, made to score the shortcut):
     - c8192 × 8, `-b 512 -ub 512`, `QKL_TAIL=8191`, `QKL_SPLIT=2048`;
     - reference f16 ub4096 recorded with QPIN; `QPIN_PIN=qsa,moe`;
     - pairs q8_0, q4_0, bf16, q8_0K/f16V, f16K/q8_0V, plus f16 as control.
   - **Q11b** (direct, dense reference):
     - c2048 × 16, reference f16 b/ub 2048, MoE pins;
     - the same 6 pairs at ub512, plus q8_0 and f16 at ub16 (the per-query kernel).
   - **Q11c** (split streams): `-c 1024 -b 2048 -ub 2048` (np2, split) must equal `-b 1024` (np1) to 0.000000 per sequence for q8_0 and q4_0.
     - Also add a split-stream decode case: `-np 2` without `--kv-unified`, teacher-forced, the new op at n_tps 1.
   - **PR219 arm:** `~/dev/wt_lab/r1av/tree/build/bin` with `LLAMA_QSA_ALLVIS` unset. It is the hooked PR219 binary plus the QKLD_CHUNK print; `wt_measQ` lacks that print.
   - **Census for W7:**
     - the trace or `GGML_SCHED_DEBUG` must show 0 FA ops assigned to the CPU on any shortcut ubatch;
     - every shortcut ubatch (0–3 per c8192 chunk, every ubatch at c2048) must carry `qsa_rows_av-<il>`.
     - (The proposal's "0 FLASH_ATTN_EXT with src[5]" applies to W7's new op, not to the PR binary, where the sparse op *is* FLASH_ATTN_EXT + src[5].)
   - **Expected sizes and thresholds:** taken from the measured SE_paired here, i.e. 3.5e-6 – 8.6e-6 for non-q4 types at ub512 and about 1e-4 for q4_0 and ub16.
   - **Prototype verdict:** the all-visible prototype already passes Q11a and Q11b on every pair. **The non-f16 half of D4 is supported on quality.**
2. **Producer specification (§4.1 row 2):**
   - Build the rows over the whole mask `[n_kv, n_tps, 1, n_stream]`:
     `cast(m, F32) → step(scale_bias(·, 1, 1)) → mul(·, arange(0, n_kv) + 1) → scale_bias(·, 1, −1) → cast I32`.
   - Assert `m->ne[2] == 1 && m->ne[1]·m->ne[3] == n_tokens`. The output is `[n_kv, n_tps, 1, n_stream]`, which is the ALLVIS=3 hunk of `r1av_full.diff`.
   - Drop the "KQ_MASK_PAD rows" wording: this tree has no pad rows (llama-graph.cpp:43). The defect of the 2-D view is its shape, and it is measured as an abort, not as wrong rows.
3. **Add `tests/test-qsa-allvis-rows.cpp`** (port of `test_allvis_rows.cpp`). It runs on CPU and on every GPU backend (HIP, Vulkan) and checks rows == {j : mask = 0} plus the 4-D shape:
   - np2 at n_tps 1 and 128;
   - np1;
   - the recorded C4 image-ubatch masks (`~/dev/wt_lab/c4/dump/p1_c4dump/img1_fa_mask_{3,43}_0.bin`, `p2_c4dump/img1_fa_mask_3_0.bin`);
   - a raw M-RoPE image kq_mask inside the shortcut, still to be recorded (UNPROVEN here).
4. **Kernel coverage that D4 implies:** the new op must run on the GPU for every pair in the shortcut, including:
   - q4_0 and bf16 at any n_q (PR219: CPU fallback, measured);
   - mixed pairs at n_q < 128 (PR219: no kernel, qsa-decode.cu:91-92);
   - split streams at n_tps < 128 (PR219: no kernel, qsa-decode.cu:97).
   - Without this, D4 brings back a silent CPU fallback. C1's decode perf margin (0.980–0.984) was measured only for q8_0 and must be re-checked on the new op for every pair.
5. **Status: resolved by measurement** for the risk read, the producer defect and the test. The Q11 run on W7 itself is not done, because W7 does not exist (no `~/dev/wt_rewrite`). It is the W7 done-when.

### Files and state
- strix-002 `~/dev/wt_lab/r1av/` (431 MB, mostly `tree/build`): scripts, `logs/` (932 KB), `r1av_full.diff`, the test and its output. Mirror: `SP/final219/r1av/` (1.1 MB, including `campA_analysis.txt`, `campD_analysis.txt`, `campBC_analysis.txt`).
- **Deleted:** `ref8t` (37 GB), `ref2k` (7.7 GB), and the `refB_*` / `refC_*` base logits (31 GB each, removed after each type).
- No process left running. Nothing outside `~/dev/wt_lab` was modified. The worktree `~/dev/wt_lab/r1av/tree` was added to the wt_lab repo's worktree list, following the `c15/tree` precedent.

## R1--nkvo: the plan names --no-kv-offload as

**Question.** §10 says "A CPU KV buffer with CPU ops is legal", and its init error tells users to "use another -ctk/-ctv or --no-kv-offload". With `-nkvo`, where does the scheduler put a node whose only src is the GPU-produced rows (the future `SPARSE_ATTN_PLAN`) and the attend node? Does a W7 pin (`ggml_backend_sched_set_tensor_backend(..., backend_cpu)`, the way llama-graph.cpp:3147 pins the non-FA path) fix it, and what does the fix cost?

**Answer.**
1. **The objection is confirmed by measurement.**
   - With `-nkvo`, every QSA-layer FLASH_ATTN_EXT in the PR runs on ROCm0: 300 of 300 nodes in the printed graphs, f16 and q8_0. Each one reads a `ROCm0#cache_k_l<il>` / `ROCm0#cache_v_l<il>` copy, so the K and V caches are copied in as split inputs on every graph.
   - A stand-in plan node (a `ggml_cont` whose only src is the rows) also lands on ROCm0: 240 of 240 nodes when it is not pinned. Pinning only the attend node leaves this stand-in on ROCm0 (240/240), which breaks the §10 rule.
2. **The proposed pin works for runtime graphs.** Pinning the stand-in plan and the attend node (LAB pin 3) puts both on the CPU in 228 of 240 graph prints. The attend node then reads `cache_k_l*` / `cache_v_l*` directly, with no copy. The rows (I32, 8 KiB at n_q 1, 4 MiB at n_q 512) are copied to the CPU instead. Pinning the rows tail as well (pin 7) changes no FA or plan placement.
3. **Two gaps the objection did not name, both measured:**
   - **(a) The pin is lost in the `-fit` probe context.** The 12 remaining unpinned prints are all from one graph: the first pp reserve (n_tokens 512) of the probe context that `common_fit_params` creates. `graph_reserve(..., sizes)` calls `ggml_backend_sched_reserve_size` (PR llama-context.cpp:3024-3026). That function calls `ggml_backend_sched_reset` *before* `split_graph` (ggml-backend.cpp:2296, upstream code, blame b1f3a6e5db), and the reset wipes every user assignment made during the build.
     - A backtrace proves it: `ggml_backend_sched_reset ← ggml_backend_sched_reserve_size ← llama_context::graph_reserve ← sched_reserve ← llama_context ← … common_fit_params`. After the reset, `split_graph` runs with is_reset=1, and the FA node's cause is `2.sup` instead of `usr`.
     - Consequence: a §10 guard that runs after every reserve would fire at init in that probe context, even with the pin. The remedy would then fail exactly as the objection predicts.
     - The same reset already drops upstream's own non-FA pin (llama-graph.cpp:3149) and the MoE-cache pins (llama-graph.cpp:2679) in that probe. [code reading; not measured for those two]
   - **(b) The dense shortcut FA is a third node to pin.** In pin 3 and pin 7, 60 of 300 FA prints have no rows src: the n_kv ≤ top_k+r−1 shortcut (`rows=none`). They stay on ROCm0 with K/V copied, because the lab pin sits inside `if (top_k)`. §10 also covers "every QSA-layer FLASH_ATTN_EXT (the f16 shortcut)", so the shortcut needs the pin too.
4. **Quality (paired KLD, wikitext c4096 × 4 chunks, ub512).** Pinned CPU attend is indistinguishable from the unpinned GPU-with-copies route. Both differ from any GPU reference by about 0.014 mean KLD, and the reference itself is not deterministic:
   - **The PR219 masked route is non-deterministic on the GPU.** Two identical runs of the masked route with offload on give mean KLD **0.010303 ± 0.000409** (f16) and **0.010408 ± 0.000383** (q8_0) against each other. The sparse (maskless) GPU route gives KLD 0.000000 on rerun, twice for each type.
   - **The pinned CPU route is deterministic.** Pin 3 run 1 and run 2 give identical KLD statistics: f16 0.014361 ± 0.000502 both times, q8_0 0.014007 ± 0.000540 both times. The unpinned runs vary (f16 0.014282 vs 0.014254).
   - A known nondeterminism source is "decode unique_ids dedup nondeterministic" (rewrite219/FACTS.md:34). That it explains this 0.0103 is **UNPROVEN**.
   - **Consequence for the recipe.** The proposed "short paired KLD vs the GPU route" can only resolve differences well below 0.01 if both arms are deterministic. On W7, where `-nkvo` takes the sparse op on the CPU and the reference is the sparse GPU route (deterministic today), it is feasible. Against PR219's masked route it is not.
5. **Cost (first-pass seconds for 4096 tokens, c4096 ub512, one sample per run).**
   - f16: GPU route 4.36 / 4.56 / 4.87; `-nkvo` unpinned 11.78 / 12.61 / 12.86; `-nkvo` pin 3 13.87 / 14.08 / 14.09 (pin 1: 14.20).
   - q8_0: GPU route 4.45 / 4.89 / 4.95; `-nkvo` unpinned 12.33 / 12.57 / 12.95; `-nkvo` pin 3 20.61 / 21.80 / 23.39 (pin 1: 21.28).
   - So at c4096, pinned CPU attention costs about +12 % (f16) and about +70 % (q8_0) over copying K/V to the GPU. Both `-nkvo` variants run about 2.7–4.7× slower than GPU KV. That ratio also includes the GDN recurrent state, which `-nkvo` moves to the CPU as well (`llama_kv_cache: layer 3: dev = CPU`; 3483 CPU splits per print set).
   - At depth: the unpinned copy is the full K and V cache views of each QSA layer per graph (f16 c4096: 4 MiB K + 4 MiB V per layer [log split inputs]). It grows linearly with n_ctx [arithmetic]. CPU attend cost at depth was not measured: **UNPROVEN**.

### Decision
- **Accept the objection and amend the plan text.** Keep `-nkvo` as a supported route, implemented as the DECIDES first option, with three additions:
  1. **W7 pins** (when `!cparams.offload_kqv`): `SPARSE_ATTN_PLAN`, `SPARSE_ATTN`, **and the QSA-layer dense-shortcut FLASH_ATTN_EXT**, all to `backend_cpu`. Pinning the rows producer tail is optional: no measured placement benefit (pin 7 = pin 3 for FA and plan placement). The rows copy to the CPU is 8 KiB–4 MiB per layer per graph.
  2. **§10 guard scope.** Do not run the guard on `graph_reserve(..., sizes != nullptr)`, the `ggml_backend_sched_reserve_size` path, which resets user pins (ggml-backend.cpp:2296). Alternatively, remove that reset in our ggml: `graph_reserve` already resets at llama-context.cpp:2985 before the build. Either choice must be stated in W7. Otherwise the guard throws in the `-fit` probe for `-nkvo`. Note also that the fit probe's compute-buffer estimate is made with the pins lost, so under `-nkvo` it counts the K/V copies on ROCm0. [mechanism measured; the size effect is not measured]
  3. **W7 validation recipe.**
     - `-nkvo` with `-ctk/-ctv` f16 and q8_0, c4096 × 4, ub512 and ub16.
     - Census: CPU `SPARSE_ATTN` > 0, GPU `SPARSE_ATTN` = 0, 0 QSA-layer FLASH_ATTN_EXT on ROCm0, and no `cache_[kv]_l*` split inputs to ROCm0 (`GGML_SCHED_DEBUG=2 -v`). The guard stays silent, including in the fit probe.
     - Paired KLD of `-nkvo` vs the same binary's GPU sparse route: first verify that both arms are deterministic (rerun KLD = 0), then apply the §15 3·SE_paired rule.
     - Report the q8_0 CPU-attend cost next to f16.
- **Fallback.** If W7 does not carry items 1 and 2, drop `--no-kv-offload` from the §10 message and make `-nkvo` with QSA an explicit init error. The measurement shows that without both items the remedy either violates the guard (unpinned: 300/300 FA on ROCm0) or makes the guard throw at init (fit probe).
- **Side finding for §15 (PR219 baseline).** The PR219 masked route has a run-to-run mean KLD of 0.0103 on HIP (f16 and q8_0). It must never be used as a paired reference without pinning or derandomizing it.

### Commands (strix-002, 2026-11-05 04:30–05:55, every GPU run under `flock ~/gpu.lock`; I was the only agent on the GPU)
- **PR placement check** (the objection's own command, plus `-v`, needed because the sched print is GGML_LOG_DEBUG):
  `flock ~/gpu.lock env GGML_SCHED_DEBUG=2 ~/dev/wt_pr219v2/build/bin/llama-perplexity -m ~/models/qwen3.8-flash-next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf -f ~/data/wikitext-2-raw/wiki.test.raw -c 4096 -ub 512 --chunks 1 -fa on -nkvo -ctk {f16,q8_0} -ctv {f16,q8_0} -v | grep -E "FLASH_ATTN|## SPLIT|..."`
  - Without `-v`, nothing is printed (logs/pr_nkvo_f16.log).
- **Lab binary.**
  - `~/dev/wt_lab/r1nkvo/tree`: a git worktree at 94e086847 (added to the wt_lab worktree list) plus `r1nkvo_lab.diff`.
  - Env-gated changes in build_attn_mha's FA branch:
    - `LAB_PLAN_STANDIN=1`: `top_k = ggml_cont(top_k)`, named `lab_plan_standin-<il>`;
    - `LAB_NKVO_PIN` bitmask, applied only when `!offload_kqv`: 1 = FA, 2 = stand-in, 4 = rows tail.
  - Debug changes: SET_CAUSE enabled, and `LAB_TRACE` prints at sched_reset and split_graph, with a backtrace at level 2.
  - HIP build flags as in the rules, `--target llama-perplexity`.
  - The placement runs (`place.sh`) used the build before the cause and trace hooks. The KLD runs used the final build; with the env unset, its graph equals PR219's.
- **Placement:** `place.sh` (f16 and q8_0 × pin 0/1/3/7, all with the stand-in). Cause and trace runs: `LAB_TRACE=1|2 GGML_SCHED_DEBUG=2 LAB_PLAN_STANDIN=1 LAB_NKVO_PIN=3 ... -nkvo -ctk f16`.
- **KLD campaign 1:** `kld.sh`.
  - Reference: `--kl-divergence-base` of the GPU sparse route (offload on); the base is 4.07 GB, deleted after use.
  - Arms: gpu_rerun ×2, nkvo_unpinned ×2, nkvo_pin1, nkvo_standin_pin3 ×2.
  - Common flags: `-c 4096 --chunks 4 -ub 512 -fa on`.
- **KLD campaign 2:** `kld2.sh`.
  - Reference: the GPU masked route (`LLAMA_QSA_SELECTED_KEY=0`, offload on).
  - Arms: gpumasked_rerun, nkvo_unpinned, nkvo_standin_pin3.

### Raw numbers
**Placement** (FA = QSA-layer FLASH_ATTN_EXT prints, all graphs of one run; `parse2.py`, `placement_summary.txt`):

| run | FA backend · K src · rows src | stand-in | K/V split inputs | PPL (1 chunk) |
|---|---|---|---|---|
| PR f16 | ROCm0·copy·direct 240, ROCm0·copy·none 60 | — | 300 / 300 | 1.2642 ± 0.02487 (non-debug run: 1.2634) |
| PR q8_0 | same | — | 300 / 300 | 1.2552 ± 0.02410 |
| lab f16 pin0 | ROCm0·copy 240 + 60 | ROCm0 240 | 300 / 300 | 1.2648 |
| lab f16 pin1 | **CPU·direct·rows copied to CPU 228**, ROCm0·copy 12, ROCm0·copy·none 60 | **ROCm0 240** | 72 / 72 | 1.2654 |
| lab f16 pin3 | CPU·direct·direct 228, ROCm0 12, ROCm0·none 60 | CPU 228 (src copied), ROCm0 12 | 72 / 72 | 1.2654 |
| lab f16 pin7 | as pin3 | CPU 228 (src direct), ROCm0 12 | 72 / 72 | 1.2654 |
| lab q8_0 pin0/1/3/7 | identical pattern to f16 | identical | 300 / 72 / 72 / 72 | 1.2713 / 1.2605 / 1.2605 / 1.2605 |

- Order of the FA prints under pin 3 (`logs/lab_pin3_order.log`): graphs 1–5 are fit-probe resolve_fused_ops reserves (CPU); **graph 6 is the fit probe's first pp reserve (ROCm0)**; graphs 7–16 are reserves (CPU); graphs 17–21 are dense-shortcut runtime graphs (ROCm0, no rows); graphs 22–25 are sparse runtime graphs (CPU).
- Trace (`logs/lab_pin3_trace.log`): `LAB pin il=3 ... n_q=512 ... fa_backend_after_set=CPU`, then `LAB sched_reset sched=0x30545a90 is_reset=0`, then `LAB split_graph ... is_reset=1`, then FA `[ROCm0 2.sup]`.
- Backtrace (`logs/lab_pin3_bt2_LAB.log`): `ggml_backend_sched_reset+0x9d ← ggml_backend_sched_reserve_size+0x32 ← llama_context::graph_reserve(...)+0x7d3 ← llama_context::sched_reserve()+0x427 ← llama_context::llama_context ← llama_init_from_model ← ... common_fit_params ← common_init_result`.

**KLD** (mean KLD ± SE, same-top-p %):

| reference | arm | f16 | q8_0 |
|---|---|---|---|
| GPU sparse (PPL 2.0810 / 2.0872) | gpu rerun 1, 2 | 0.000000, 0.000000 (100 %) | −0.000000, −0.000000 (100 %) |
| | nkvo unpinned r1, r2 | 0.014282 ± 0.000568 (97.03), 0.014254 ± 0.000575 (96.74) | 0.014320 ± 0.000657 (96.89), 0.014236 ± 0.000502 (96.64) |
| | nkvo pin1 | 0.014361 ± 0.000502 (97.07) | 0.014007 ± 0.000540 (96.80) |
| | nkvo standin pin3 r1, r2 | 0.014361 ± 0.000502, 0.014361 ± 0.000502 (97.07) | 0.014007 ± 0.000540, 0.014007 ± 0.000540 (96.80) |
| GPU masked (PPL 2.0814 / 2.0809) | gpu masked rerun | **0.010303 ± 0.000409** (97.48), max 1.60 | **0.010408 ± 0.000383** (97.61), max 0.78 |
| | nkvo unpinned | 0.013655 ± 0.000484 (97.09) | 0.014379 ± 0.000529 (96.79) |
| | nkvo standin pin3 | 0.014357 ± 0.000596 (97.01) | 0.014117 ± 0.000499 (96.95) |

- **Noise.**
  - Sparse GPU route: 0 (4/4 reruns bit-identical in KLD).
  - Masked GPU route: 0.0103–0.0104 run to run.
  - `-nkvo` unpinned: ±3e-5 between r1 and r2 against a fixed base.
  - `-nkvo` pinned: 0 (r1 = r2).
- **Timing.** First-pass seconds from each log's `seconds per pass` line, listed in Answer 5.

### Files and state
- **strix-002 `~/dev/wt_lab/r1nkvo/`** (444 MB, mostly `tree/build`): `tree/` (lab worktree), `place.sh`, `kld.sh`, `kld2.sh`, `r1nkvo_lab.diff`, `logs/`, `kld/*.log` and `kld/*_campaign.out`.
- **Mirror:** SP/final219/r1nkvo/ (11 MB): all of the above except the tree, plus `parse2.py` and `placement_summary.txt`.
- **Deleted:** `kld/base_{f16,q8_0}.bin` and `kld/basemasked_{f16,q8_0}.bin` (4.07 GB each).
- No process left running. Nothing outside `~/dev/wt_lab/r1nkvo` was modified.

## R1-The Q8 detection-power calibration canno

**Question.** (a) Under QSA pins, does QPIN replay erase a defect planted in the producer of `qsa_rows_cb-<il>`? (b) Does Q8's "paired t against the floor" calibrate Q1? (c) Is Q8 scheduled too late (only in W13)? Fix proposed: plant the defects downstream of the pin point, measure defect-on minus defect-off at the same type and pins, per chunk at full precision, and derive delta_T. If any defect's one-sided 95 % LCB is ≤ delta_T, Q5 becomes primary.

**Answer.**
- **(a) Confirmed, by code and by measurement.**
  - Code: `qpin_rows` overwrites all W entries of every recorded query row: `if (S.pin_qsa) for j < W: cur[j] = ref[j]` (`~/dev/wt_measQ/q/qpin.patch:109-116`). It then writes all W·nq entries back (`:118`). The hooked tensor is the final rows tensor (`:300`, `cb(top_k, … "qsa_rows_cb" …)` after `qwen4exp_finish_strips`, PR qwen4exp.cpp:1681). That tensor is the `top_k` passed to `build_attn_mha` (:1735).
  - Measurement: the same defects applied to the computed rows before the pin copy (`LAB_DEF_PRE=1`) change nothing. Every chunk of `q8_d1_pre − q8_off` and `q8_d2_pre − q8_off` is +0.000000e+00, at %.9e per-chunk precision (8/8 chunks each). That is with 49152 cells, and 65512 blocks, corrupted in the producer.
- **(b) Confirmed.** PLAN-v3.md:590 has Q8 measure "each as a paired t against the floor", and the floor is f16 ub2048. Q1 (PLAN-v3.md:583) is "new(T) − PR219(T), … Pass if mean diff ≤ 3·SE_paired". These are different comparisons, so Q8 as written does not calibrate Q1.
- **(c) Confirmed.** The ordering is in the plan text:
  - Q8 is reached only through W13's "Q1–Q9" (PLAN-v3.md:545).
  - W7 requires "Q1–Q5" (:487) and W10 requires "Q1" (:516). Both rely on Q1's power before it is measured (U11, :653).
- **The calibration was run now, not after W7/W12.**
  - It ran on the hooked PR219 binary (QPIN patch plus a downstream defect hook plus %.9e per-chunk KLD output). Detection power is a property of the metric, the pins and the model, so it does not have to wait for the new kernels.
  - Its result goes further than the objection: **the pinned e2e KLD cannot carry the quality claim for kernel-local changes, in either of the two forms tested.**
    1. **Q1 form** (KLD of each arm against the f16 reference, difference on − off, q8_0).
       - D1 is detected: t 6.08, LCB 8.21e-4.
       - **D2 is not detected**: t 1.39, LCB −2.22e-4, even though 8/8 chunks are positive.
       - **D3 is invisible**: +4.9e-7, t 0.20.
       - With delta_T = 3 × SE_paired of a benign same-type reorder (q8 ub2048 − q8 ub4096: SE 2.48e-5, so delta_T = 7.45e-5), **D2 and D3 fail LCB > delta_T. By the DECIDES rule, Q5 becomes primary.**
    2. **Direct form** (KLD against the clean same-type pinned output).
       - D3 gives 9.47e-5, but **a 1e-7 relative perturbation of the same tensor gives 9.39e-5**, and a benign ub2048 reorder gives 1.63e-4 (larger than D3).
       - Pinned KLD saturates at about 1e-4 for any bit-level perturbation (new finding, below). It cannot separate a small numeric defect from a correct reordering.
  - **Q1 at T = f16 cannot pass a correct rewrite.**
    - PR219(f16) against the PR219-f16 reference is −3.6e-8 (exact replay).
    - A benign f16 reorder (ub2048) is +1.116e-4 with t 9.40.
    - So any new f16 kernel that is not bit-identical fails Q1's 3·SE rule.

### Sources and setup
- **Lab tree.** `strix-002:~/dev/wt_lab/r1q8/tree` is a worktree at 94e086847 with `~/dev/wt_measQ/q/qpin.patch` plus `labedit.py`. The full diff is `~/dev/wt_lab/r1q8/r1q8_lab.diff` (mirror: `SP/final219/r1q8/r1q8_lab.diff`). It was built with the HIP flags of the rules, `--target llama-perplexity`.
- **Measurement-only additions** (all downstream of the pin copy unless `LAB_DEF_PRE=1`):
  - `LAB_DEF=d1`: at layer 23, set the last valid tail slot (row slots W−3..W−1) of every query row to −1.
  - `LAB_DEF=d2`: at layer 3, set the 4 cells of the median valid block of every query row to −1. A −1 is a hole that the PR consumer skips (`qsa-prefill.cu:49`, `ent[j] = visible ? key : QSA3_SENT`). Invalid blocks are written as −1 by the producer (qwen4exp.cpp:1199-1208).
  - `LAB_DEF=d3`: `attn_output-23 × (1 + LAB_D3_EPS)`, default 1e-3. This equals V × (1 + ε): the softmax weights sum to 1 and the gate and wo are linear in the attention output.
    - The tensor is observed in every arm, the reference included, so all arms have the same graph splits.
    - The hook fires on the MUL_MAT and on its `(reshaped)` view, which share the same data (`SP/final219/r1q8/dbg/d3_{f16,q8_0}.log`: `LAB_DBG ao name=attn_output-23 op=MUL_MAT …` then `… (reshaped) op=RESHAPE … data=<same>`). The effective factor is therefore (1 + ε)². The ε sweep below shows this does not matter.
  - `LAB_DEF_PRE=1`: apply d1/d2 to the computed rows before the pin copy (the producer-side sanity check).
  - `R1Q8_SUMMARY` counters, and `QKLD_CHUNK i n kld same_top` per chunk at %.9e (tools/perplexity).
- **Workload.** `llama-perplexity -m <Q4_K_XL> -f wiki.test.raw -c 8192 -b 4096 -ub {4096|2048} --chunks 8 -ngl 99 -fa on -lm mmap -lzm on-direct -ctk T -ctv T`. KLD is scored over positions 4096–8191 of each chunk (32 760 tokens). Every command ran as `flock ~/gpu.lock env QPIN=replay QPIN_DIR=ref8/pins QPIN_PIN=qsa,moe …`.
- **Scripts:**
  - `~/dev/wt_lab/r1q8/run.sh TAG MODE CTK CTV UB [ENV…]` with base = hooked f16 ub4096 reference (`ref8`, recorded by `run.sh ref record f16 f16 4096`);
  - `run2.sh TAG {base|kld} …` with base = hooked q8_0 ub4096 clean pinned output (`q8base`);
  - `campaign.sh`, `campaign2.sh`. The ε 0 / 1e-2 / 1e-6 / 1e-7 arms ran as `./run2.sh q8_d3e{0,2,6,7} kld q8_0 q8_0 4096 LAB_DEF=d3 LAB_D3_EPS=…`.
- **Logs:** `~/dev/wt_lab/r1q8/{logs,logs2,dbg}/`, and `campaigns.out` (all mirrored to `SP/final219/r1q8/`).
- **Analysis:** `analyze.py logs` → `analysis_vs_f16ref.out`; `analyze2.py logs2` → `analysis_direct_q8base.out`. One-sided 95 % LCB = mean − t₀.₉₅,₇ · SE, with t₀.₉₅,₇ = 1.8946.

### Raw numbers
**Defect counters** (`R1Q8_SUMMARY`, main context, every defect arm):
- d1 dropped 49152 cells, and **all 49152 were the query's own cell** (`d1_dropped_self=49152`). 49152 = 8 × 8192 × 3/4, so in PR219's rows the last tail slot is always the query itself, and positions with p mod 4 = 3 have no tail. D1 therefore means "layer 23 cannot attend to the current token for 75 % of queries".
- d2 dropped 65512 blocks (all rows except the 3 tail-only positions per chunk × 8).
- d3 scaled 32 tensors (16 ubatches × {MUL_MAT, reshape}).
- QPIN: `rows_pin=757760 rows_absent=0 moe_pin=3112960 moe_absent=0 cb_tensors=192` in every replay arm.

**A. Q1 form: KLD against the hooked f16 ub4096 reference** (per-chunk %.6e; `analysis_vs_f16ref.out`)

| arm | mean | chunk SE | per-chunk values |
|---|---|---|---|
| f16_off (self-replay) | −3.60e-08 | 6.0e-08 | −1.24e-7 … +1.79e-7 (uint16 base quantization) |
| q8_off = q8_off_r2 | 1.375064e-04 | 1.354e-05 | 2.031e-4 1.310e-4 1.404e-4 1.795e-4 1.009e-4 1.252e-4 1.340e-4 8.601e-5 |
| q8_d1 | 1.330276e-03 | 2.09e-04 | |
| q8_d2 | 7.454779e-04 | 4.48e-04 | 3.821e-3 2.720e-4 8.957e-4 2.501e-4 1.309e-4 2.415e-4 1.776e-4 1.750e-4 |
| q8_d3 = q8_d3_r2 | 1.380001e-04 | 1.45e-05 | |
| q8_d3e4 (ε 1e-4) | 1.390963e-04 | 1.35e-05 | |
| q8_d1_pre, q8_d2_pre | 1.375064e-04 | | identical to q8_off in every chunk |
| f16_d1 / f16_d2 / f16_d3 | 1.266425e-03 / 6.959759e-04 / 7.310475e-05 | 1.97e-4 / 4.23e-4 / 8.0e-6 | |
| q8_ub2048 | 1.657226e-04 | 2.93e-05 | |
| f16_ub2048 (floor) | 1.116028e-04 | 1.19e-05 | |

| paired (A − B), same pins | mean | SE_paired | t | LCB95 |
|---|---|---|---|---|
| q8_d1 − q8_off | +1.192769e-03 | 1.961e-04 | 6.08 | **+8.21e-04** |
| q8_d2 − q8_off | +6.079715e-04 | 4.382e-04 | **1.39** | **−2.22e-04** (8/8 chunks > 0) |
| q8_d3 − q8_off | +4.937e-07 | 2.491e-06 | **0.20** | −4.23e-06 |
| q8_d3e4 − q8_off | +1.590e-06 | 1.677e-06 | 0.95 | −1.59e-06 |
| f16_d1 − f16_off | +1.266461e-03 | 1.967e-04 | 6.44 | +8.94e-04 |
| f16_d2 − f16_off | +6.960119e-04 | 4.229e-04 | 1.65 | −1.05e-04 |
| f16_d3 − f16_off | +7.314075e-05 | 8.038e-06 | 9.10 | +5.79e-05 |
| q8_d{1,2}_pre − q8_off | 0 (every chunk 0.000000e+00) | 0 | | |
| q8_off_r2 − q8_off, q8_d3_r2 − q8_d3 | 0 (every chunk) | 0 | | |
| **q8_ub2048 − q8_off** (benign same-type reorder) | +2.821617e-05 | **2.484e-05** | 1.14 | −1.88e-05 |
| **f16_ub2048 − f16_off** (benign f16 reorder) | **+1.116388e-04** | 1.188e-05 | **9.40** | +8.91e-05 |
| q8_off − f16_ub2048 (C8 statistic, full precision) | +2.590360e-05 | **4.691e-06** | 5.52 | +1.70e-05 |

**B. Direct form: KLD against the clean q8_0 ub4096 pinned output** (`analysis_direct_q8base.out`)

| arm | mean KLD | chunk SE | min–max |
|---|---|---|---|
| q8_off, q8_d2_pre, q8_d3e0 (ε = 0) | 7.0e-08 | 8.7e-08 | identical per chunk (base quantization) |
| q8_d1 | 1.322291e-03 | 2.15e-04 | 7.04e-4 – 2.54e-3 |
| q8_d2 | 7.439161e-04 | 4.46e-04 | 1.37e-4 – 3.83e-3 |
| **q8_d3e7 (ε 1e-7)** | **9.387759e-05** | 9.56e-06 | 6.49e-5 – 1.43e-4 |
| q8_d3e6 (ε 1e-6) | 9.497600e-05 | 9.58e-06 | |
| q8_d3e4 (ε 1e-4) | 9.544101e-05 | 9.95e-06 | |
| q8_d3 (ε 1e-3) | 9.473362e-05 | 9.58e-06 | |
| q8_d3e2 (ε 1e-2) | 1.057716e-04 | 1.12e-05 | |
| q8_ub2048 (benign reorder) | 1.625561e-04 | 2.28e-05 | 9.19e-5 – 2.91e-4 |
| f16 ub4096 (type change) | 1.370528e-04 | 1.35e-05 | |

| paired (A − B), direct form | mean | SE | t | LCB95 | chunks > 0 |
|---|---|---|---|---|---|
| q8_d1 − q8_ub2048 | +1.160e-03 | 2.03e-04 | 5.70 | +7.74e-04 | 8/8 |
| q8_d2 − q8_ub2048 | +5.81e-04 | 4.38e-04 | 1.33 | −2.48e-04 | 8/8 |
| **q8_d3 − q8_ub2048** | **−6.78e-05** | 2.11e-05 | −3.22 | | **0/8** |
| q8_d3 − q8_d3e7 (ε 1e-3 vs 1e-7) | +8.6e-07 | 4.5e-07 | 1.91 | +9e-09 | 5/8 |
| q8_d3e2 − q8_d3e7 (ε 1e-2 vs 1e-7) | +1.19e-05 | 1.9e-06 | 6.12 | +8.2e-06 | 8/8 |

### Noise
- **Run to run: zero.** q8_off_r2 equals q8_off and q8_d3_r2 equals q8_d3 in every chunk at %.9e, and self-replay f16 is ≤ 3.2e-7 per chunk (`analysis_vs_f16ref.out`). Every SE above is therefore chunk-to-chunk variation of the effect itself, not run noise.
- **Full-precision paired SE of q8_0 − floor: 4.69e-6** (U11). The 5-decimal table gave 0.82e-5 (C8), so C8's rounding diagnosis is confirmed. The mean is unchanged: +2.59e-5 here, +2.9e-5 in C8.
- **Benign same-type reorder** (q8 ub2048 vs ub4096): SE_paired 2.48e-5. One chunk dominates it (chunk 3: +2.0e-4; the other chunks lie between −8.9e-6 and +2.4e-5).
- **Saturation (new).** Under QSA+MoE pins, scaling one layer's attention output by 1 + 1e-7 (about 1 f32 ulp) gives KLD 9.39e-5 against the clean output. This is 0.99× the 1e-3 defect's 9.47e-5 and 0.58× a ub2048 reorder.
  - ε 0 gives 7e-8, so the get/set round trip itself is exact.
  - From ε 1e-7 to 1e-2 (10⁵×), KLD grows only 1.13× (9.39e-5 → 1.058e-4).
  - So the pinned network still has a chaotic amplifier that turns any bit change into a KLD of about 1e-4. The f16 ub2048 floor (1.1e-4) and q8_0's cost (1.37e-4) sit at that same level.
  - **Mechanism UNPROVEN.** Hypothesis: activation quantization in the quantized-weight matmuls (q8_1 rounding at every MMQ call, a discrete decision on every element that the pins do not cover). The experiment that would test it: the same ε sweep on a `-DGGML_CUDA_FORCE_CUBLAS=ON` build, with its own base.
- **Why the Q1 form hides D3.** The direct KLD of D3 is 9.47e-5, yet Q1's on − off is +4.9e-7: the cross term with q8_0's own deviation cancels it chunk by chunk (per-chunk diffs ±1e-5, mixed signs). A KLD difference against a third distribution is not a divergence between the two arms.

### Decision
1. **The objection holds on all three points**, (a) by code and by a measured exact 0, and (b) and (c) by the plan text. A producer-side defect can never test the pinned gate, because it is erased.
2. **delta_T (from the same-type benign reorder q8 ub2048 vs ub4096): delta_T = 3 × 2.48e-5 = 7.45e-5.** The LCB check per defect:
   - D1 (self cell dropped at L23): LCB 8.21e-4, **pass**.
   - D2 (one block per query at L3): LCB −2.22e-4, **fail**.
   - D3 (V × (1 + 1e-3) at L23): LCB −4.2e-6, **fail**.
   - **So Q5 becomes the primary gate for kernel-local changes (attend, plan, loaders, rotation).** Q1 is demoted to a coarse e2e screen. It reliably catches only selection defects of D1's size (KLD ≥ about 1e-3, t about 6), and D2-sized defects only through a sign test (8/8 chunks, p = 1/256), not through the 3·SE rule.
3. **Rewrite Q1's rule before W7.**
   - (i) Drop the f16 arm's "≤ 3·SE_paired": a benign f16 reorder fails it at t 9.40.
   - (ii) For every type, replace it with a fixed tolerance against the benign-reorder band: new(T) − PR219(T) ≤ max over chunks of |q8_ub2048 − q8_off| (2.0e-4), as an e2e screen.
   - (iii) Add the direct form (KLD of new(T) against PR219(T)'s pinned output) as a reported number. It must stay ≤ the ub2048 reorder value of that type (1.63e-4 for q8_0) + 3 × its chunk SE.
   - Neither form detects numeric defects below the saturation level (about 1e-4). No MDE below that may be claimed for e2e KLD.
4. **Schedule.** Q8 (this experiment, re-run on the branch with the same hook ported after `qpin_rows`) moves into W7's done-when, before its quality verdict. It also becomes part of W12's tooling (`r1q8_lab.diff`). The plan's Q8 row (PLAN-v3.md:590) must be changed from "paired t against the floor" to "on − off, same type, same pins, defects downstream of the pin".
5. **Q5 must therefore be strong enough to be primary.**
   - Its plan-bytes exactness catches D1/D2-class selection bugs by construction.
   - Its attend rel-L2 ≤ 4e-4 bound catches a V × (1 + 1e-3) scaling: rel-L2 = 1e-3 > 4e-4.
   - This last claim is by arithmetic, not measured on a fixture: UNPROVEN until the C18 fixture run includes a planted × (1 + 1e-3).
6. **Files.**
   - Deleted: `ref8/base.kld` (16 GB), `q8base.kld` (16 GB), `ref8/pins` (6.0 GB).
   - Kept on strix-002: `~/dev/wt_lab/r1q8/` (861 MB: `tree/`, `tree2/` (tree plus an env-gated `LAB_DBG` print, used only for the `dbg/` runs), logs, scripts).
   - Mirror: `SP/final219/r1q8/` (logs, logs2, dbg, analyses, scripts, diff).
   - No process left running.

## R1-The e2e quality gates exercise only one 

**Question.** Q1 and Q3 (PLAN-v3.md:583, :585) score one shape only: the f16 ub4096 reference (:577), c8192 or c32768. Does that leave G1, G4 at n_q 128–512, direct split-gather V, the all-visible shortcut and the Vulkan kernels without an e2e arm? Can pinned arms at -ub 512, -ub 64 and c32768 -ub 256 replay into the ub4096 pins? What noise and thresholds do they have? Can a Vulkan arm resolve anything?

**Answer.**
1. **The claim holds.** It is confirmed by the plan text, by code, and by a measured launch census on PR219 (table A).
   - At ub4096, every scored QSA call is n_q 4096 on `qsa_prefill` with packed V: 23 calls per chunk.
   - The shortcut never runs.
   - The one n_q 512 call is layer 47 of the first ubatch, and it is dead.
   - G1, G4 at n_q 128–512, direct V and the shortcut get no e2e coverage from Q1/Q3 as written.
2. **The three proposed arms are feasible.** In every replay arm (30 HIP, 11 Vulkan), rows_absent = 0 and moe_absent = 0, provided the reference is recorded with `LLAMA_FN_LAST_ATTN_TRIM=0`.
   - That reference holds rows_rec = 786432 = 12 × 65536. The default-trim reference had 757760 (R1-Q8 campaign).
   - Each arm reaches its new-design route in hundreds of layer-calls per chunk: G1 1120, G4@512 140, G4@256 1336 (of which direct V 731), shortcut 44 / 352 / 88 live calls (table B).
3. **Run-to-run noise is exactly zero.** All 9 HIP self-pairs and both Vulkan self-pairs are 0.000e+00 in every chunk at %.9e.
4. **Q1's 3·SE_paired rule cannot be used for these arms.** A benign route change of PR219 itself fails it:
   - f16 ub512 vs ub4096 (decode vs prefill kernel): t 5.24;
   - f16 ub64: t 6.59;
   - q8_0 ub64: t 3.94;
   - f16 ub256 @32k: t 4.34.
   - So a correct rewrite that is not bit-identical fails 3·SE, as R1-Q8 found at ub4096. The arms need the band rule, with a measured band per ub (Decision 3).
5. **A Vulkan arm of the form new-Vulkan(T) vs new-HIP(T) is not resolvable.**
   - Under identical QSA+MoE pins, PR219+#240 Vulkan differs from HIP by KLD 5.52e-3 for f16 (SE_paired 7.45e-4), 5.38e-3 for q8_0 and 4.79e-3 for q4_0. That is q4_0-class.
   - Its 3·SE is 2.2e-3, about 2× a D1-size defect (R1-Q8: +1.19e-3).
   - **Pins do work on Vulkan.** Against a Vulkan reference, f16 ub2048 is 4.98e-4 pinned and 8.82e-3 observe-only (17.7× lower).
   - So the Vulkan arm must be Vulkan-internal: new-Vulkan(T) vs PR219+#240-Vulkan(T), both against a Vulkan f16 reference.
6. **New finding: PR219 q4_0 is bit-identical between ub512 and ub4096.** All 8 chunks match at %.9e, and PPL(Q) is 3.293703 in both.
   - The census shows that q4_0 uses one kernel (`dense_tile_vec_mma`, mask + ids) at every ub.
   - So on HIP every non-attention op is ub-invariant between 512 and 4096. The f16/q8_0 ub512 shifts come only from the attention route (`qsa_prefill` n_q 4096 vs `qsa_decode` n_q 512). This is an inference from the measurement.
   - At ub64 that invariance breaks (q4_0 +8.4e-5). Which non-attention op changes there is UNPROVEN.

### Code and plan verification (94e086847, `~/dev/wt_pr219v2`)
- **Plan text.**
  - §15 records "one f16/f16 ub4096 reference" (PLAN-v3.md:577).
  - Q1 (:583) gives c8192 × 8 and no ub. Q3 (:585) gives c32768 × 2.
  - The only Q1 data ever measured are at ub4096, with the floor at ub2048 (results-Q.md Q2 table; R1-Q8 table A).
  - Q7 (:589) is an acceptance-rate CI, not a numeric KLD.
- **Shortcut predicate.** `if (shortcut && n_kv <= width)` with width = top_k + r − 1 = 2051, on the indexer cache n_kv (qwen4exp.cpp:1870-1878). The f16/q8/q4 shortcut on PR219 is dense FA with a mask and no ids (census).
- **Decode route.** It is tried first: fattn.cu:1154-1158. It accepts 1 ≤ n_q ≤ 512 (`QSA_DECODE_MAX_QUERIES`, qsa-decode.cuh:5; qsa-decode.cu:94) and only K = V ∈ {f16, q8_0} (:92-93). So on PR219 every f16/q8_0 QSA call with n_q ≤ 512 is `qsa_decode`. That is the stand-in for G1 and for G4 at n_q 128–512.
- **Direct V.** PR219's `qsa_prefill` switches to direct V only when n_q·64 < n_kv (qsa-prefill.cu:684-686). At n_q 4096 that needs n_kv > 262144, so neither Q1 nor Q3 reaches it.
  - In the plan's design (D8, PLAN-v3.md:44, :221), direct V is G4 with n_q ≥ 128 and n_kv ≥ 64·n_q. At c32768 that needs n_q ≤ 512, which the n_q 256 arm gives.
- **QPIN.** `qpin_begin_ubatch` skips ubatches of fewer than 16 tokens (r1av_full.diff, src/llama-context.cpp hunk: `if (ub.n_tokens < 16 …) return;`). So **n_q 1 decode and MTP n_q 2–8 cannot be pinned**, and no perplexity arm covers them.

### Setup (strix-002, 2026-11-05 06:50–10:30 CET; only this agent used the GPU; every GPU command under `flock ~/gpu.lock`)
- **HIP binary.** `~/dev/wt_lab/r1av/tree/build/bin`, used as built. It is 94e086847 plus `r1av_full.diff` (md5 264f3bd03deb5627ea27f25af73ae2e0), containing:
  - the QPIN record/replay hook;
  - `QKL_TAIL`;
  - `QKLD_CHUNK` per-chunk KLD at %.9e (the W12 precondition);
  - the `GGML_QSA_TRACE` unique-shape trace.
  - `LLAMA_QSA_ALLVIS` is unset, so this is PR219's own dense shortcut.
- **Vulkan binary.** New worktree `~/dev/wt_lab/r1ub/tree` = 94e086847 + `r1av_full.diff` + `c11/fix_6411b4315.diff` (#240). The working diff md5 is 0739857d93ba32fc7aeac249e562208c.
  - Built with `cmake -B build-vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DCMAKE_CXX_FLAGS=-I$HOME/.local/spirv-headers/include -DSPIRV-Headers_DIR=$HOME/.local/spirv-headers/share/cmake/SPIRV-Headers` and `cmake --build build-vk -j 16 --target llama-perplexity`, BUILD_RC=0 (`build_vk.log`).
- **Census binary.** `~/dev/wt_lab/build/bin` (94e086847 + the C1/C2 env-gated tracers, as in C12), unpinned, 1 chunk.
- **Runner** `~/dev/wt_lab/r1ub/run.sh TAG REFDIR MODE CTK CTV UB CTX CHUNKS {hip|vk} [ENV…]`:
  `flock ~/gpu.lock env [QPIN=record QPIN_DIR=REF/pins | QPIN=replay QPIN_DIR=REF/pins QPIN_STATS=… QPIN_PIN=qsa,moe] GGML_QSA_TRACE=1 <ENV> timeout 5400 <BIN>/llama-perplexity -m ~/models/qwen3.8-flash-next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf -f ~/data/wikitext-2-raw/wiki.test.raw -c CTX -b 4096 -ub UB --chunks CHUNKS -ngl 99 -fa on -lm mmap -lzm on-direct -ctk CTK -ctv CTV --kl-divergence-base REF/base.kld [--kl-divergence]`
  - Mode `observe` is QPIN replay without `QPIN_PIN`.
  - Arms use the default trim. References use `LLAMA_FN_LAST_ATTN_TRIM=0`.
- **Census** `census.sh TAG CTK CTV UB CTX`:
  `flock ~/gpu.lock env GGML_CUDA_DISABLE_GRAPHS=1 GGML_QSA_COUNT=1 GGML_QSA_TRACE=1 ~/dev/wt_lab/build/bin/llama-perplexity … -c CTX -b 4096 -ub UB --chunks 1 …`
  - The first six census runs (f16 at all four shapes, q8_0 ub4096 and ub512) were made without `GGML_CUDA_DISABLE_GRAPHS=1`. HIP graph replays bypass the counter, which gave f16 ub64 204 + 588 instead of 384 + 1152 launches.
  - All six were rerun with graphs off (`census_redo.out`). Only the graph-off logs are kept and used.
- **References** (all f16/f16, ub4096, trim 0, QPIN record):
  - `ref8t0`, c8192 × 8 (HIP): rows_rec 786432, moe_rec 3112960;
  - `ref32t0`, c32768 × 2, `QKL_TAIL=4096` (HIP): rows_rec 786432;
  - `ref32x8`, c32768 × 8, `QKL_TAIL=4096` (HIP): rows_rec 3145728 = 12 × 262144, moe_rec 12353536;
  - `vref8t0`, c8192 × 8 (Vulkan): rows_rec 786432.
  - The 32k tier was extended from × 2 to × 8 because 2 chunks give 1 dof (t₀.₉₅,₁ = 6.31). In that tier, q8_0 − f16 at ub4096 has t 2.89 but LCB −1.59e-4.
- **Scripts.** `campaign.sh`, `campaign32b.sh`, `campaign_vk.sh`, `campaign_vk2.sh`, `census_all.sh`, `census_redo.sh`. Analysis: `analyze.py logs` → `analysis.out`; `census_sum.py census/*.log` → `census_summary.out`.
  - Paired difference = per-chunk B − A, SE = sd/√n, one-sided LCB95 = mean − t₀.₉₅,ₙ₋₁·SE.
- **Runs.**
  - 33 HIP perplexity runs: 3 records and 30 replays.
  - 13 Vulkan runs: 1 record and 12 replay/observe. One of the 12 died with a device loss; another of them is its rerun (Noise).
  - 18 census runs (12 + 6 redone).
- **Mirror.** `SP/final219/r1ub/` holds `logs/` (87 files), `census/`, the scripts, `analysis.out` and `census_summary.out`.

### Raw numbers

**A. Measured launch census per chunk, PR219 HIP** (`census_summary.out`). Launches are summed over the 12 QSA layers. The warmup's 12 `dense_tile_vec_mma nq=2` launches are excluded.

| arm | ubatches | shortcut (n_kv ≤ 2048): kernel, launches | past the shortcut, f16 and q8_0 | past the shortcut, q4_0 (masked: mask + ids) |
|---|---|---|---|---|
| c8192 ub4096 | 2 | none | `qsa_prefill(_gated)` n_q 4096, 23 (n_kv 4096, 8192) + `qsa_decode` n_q 512, 1 (L47, trimmed) | `dense_tile_vec_mma` n_q 4096, 24 |
| c8192 ub512 | 16 | 4 ubatches, 48 (f16: 36 dense_tile + 12 fa7_gated; q8/q4: 48 dense_tile) | `qsa_decode` n_q 512, 144 (12 n_kv, 2560…8192) | dense_tile n_q 512, 144 |
| c8192 ub64 | 128 | 32 ubatches, 384 (dense_tile n_q 64, n_kv 256…2048) | `qsa_decode` n_q 64, 1152 (24 n_kv, 2304…8192) | dense_tile n_q 64, 1152 |
| c32768 ub256 | 128 | 8 ubatches, 96 (f16: 84 + 12 fa7_gated) | `qsa_decode` n_q 256, 1440 (120 n_kv, 2304…32768) | dense_tile n_q 256, 1440 |

**B. Reach of the new-design routes in each arm, per chunk.**
- These are arithmetic on the measured shapes of table A, applying D3 (G1 iff n_q < 128), D4 (shortcut iff n_kv ≤ 2051) and D8 (direct V iff q8_0/q4_0 V and n_kv ≥ 64·n_q) (PLAN-v3.md:39-45, :221).
- "live" excludes layer 47 at non-output positions, which feeds only unscored logits (C12 mechanism 4).
- Outputs are positions ≥ n_ctx/2 at c8192, and ≥ 28672 at c32768 with `QKL_TAIL=4096`.

| arm | shortcut calls (live) | G1 (live) | G4 n_q 128–512 (live) | G4 n_q 4096 (live) | direct V, q8_0/q4_0 (live) |
|---|---|---|---|---|---|
| Q1 c8192 ub4096 | 0 | 0 | 1 (0) | 23 (23) | 0 |
| (i) c8192 ub512 | 48 (44) | 0 | 144 (140) | 0 | 0 (64·512 = 32768 > 8192) |
| (ii) c8192 ub64 | 384 (352) | 1152 (1120) | 0 | 0 | n/a (G1 stages V, D9) |
| (iii) c32768 ub256 | 96 (88) | 0 | 1440 (1336) | 0 | 780 (731): ubatches 63–127, n_kv 16384…32768; packed 660 (605) |

- Across all arms, G1 is reached only at n_q = 64. n_q 1–63 and 65–127 are never reached e2e (Decision 5).

**C. Pin census of every replay arm** (QPIN_SUMMARY in `logs/*.log`).
- rows_absent = 0 and moe_absent = 0 in all arms.
- rows_pin:
  - 757760 at ub4096, c8k (cb route);
  - 589824 = 8 × 6144 × 12 at ub512/ub64, cb. The shortcut positions have no rows tensor, and layer 47 is not trimmed at ub ≤ 512.
  - 2949120 at c32k × 8 ub256 = 8 × 30720 × 12;
  - the tk (masked, q4_0) arms show 2× these values, because the hook also fires on the rows view (results-Q tool note).
- moe_pin: 3112960 at c8k and 12353536 at c32k × 8.

**D. Per-arm KLD, QSA+MoE pinned, against the HIP f16 ub4096 trim-0 reference** (`analysis.out`; chunk SE in parentheses).

| arm | f16 | q8_0 | q4_0 |
|---|---|---|---|
| c8192 ub4096 | −3.93e-8 (7.1e-8) (self) | 1.371668e-4 (1.25e-5) | 5.786380e-3 (6.31e-4) |
| c8192 ub512 r1 = r2 | 1.289040e-4 (2.46e-5) | 1.469425e-4 (1.67e-5) | 5.786380e-3 (6.31e-4), bit-identical to ub4096 |
| c8192 ub64 r1 = r2 | 1.873024e-4 (2.85e-5) | 2.193766e-4 (3.15e-5) | 5.870390e-3 (6.72e-4) |
| c32768 × 8 ub4096 | 5.75e-8 (5.0e-8) (self) | 1.453076e-4 (3.37e-5) | 5.669339e-3 (9.74e-4) |
| c32768 × 8 ub256 | 1.976457e-4 (4.55e-5) | 2.281067e-4 (5.43e-5) | 5.749668e-3 (1.05e-3) |
| c32768 × 2 ub256 r1 = r2 | 2.262e-4 | 2.56e-4 | 5.430e-3 |

- Per-chunk values are in `analysis.out`. For example, f16 ub512: 1.522e-4 1.070e-4 2.845e-4 1.441e-4 7.35e-5 1.075e-4 9.64e-5 6.61e-5.
- The first two chunks of the × 8 tier equal the × 2 tier's values (f16 ub256: 3.244766e-4 and 1.279699e-4 in both). The references recorded separately are consistent.

**E. Paired differences (B − A, per chunk, n = 8)**

| pair | mean | SE_paired | t | LCB95 | max\|d\| | chunks > 0 |
|---|---|---|---|---|---|---|
| every r2 − r1 (HIP ub512 ×3 types, ub64 ×3, c32k ub256 ×3; Vulkan f16, q8_0) | 0 | 0 | | | 0 | 0/8 |
| **route change on PR219, same type:** | | | | | | |
| f16 ub512 − f16 ub4096 | +1.289e-4 | 2.46e-5 | **5.24** | +8.2e-5 | 2.85e-4 | 8/8 |
| q8_0 ub512 − q8_0 ub4096 | +9.8e-6 | 1.07e-5 | 0.92 | −1.0e-5 | 8.1e-5 | 5/8 |
| q4_0 ub512 − q4_0 ub4096 | 0 (bit-identical) | 0 | | | 0 | 0/8 |
| f16 ub64 − f16 ub4096 | +1.873e-4 | 2.84e-5 | **6.59** | +1.34e-4 | 3.23e-4 | 8/8 |
| q8_0 ub64 − q8_0 ub4096 | +8.22e-5 | 2.09e-5 | **3.94** | +4.3e-5 | 1.73e-4 | 8/8 |
| q4_0 ub64 − q4_0 ub4096 | +8.40e-5 | 6.38e-5 | 1.32 | −3.7e-5 | 4.34e-4 | 6/8 |
| f16 ub256 − ub4096 @32k × 8 | +1.976e-4 | 4.55e-5 | **4.34** | +1.11e-4 | 4.40e-4 | 8/8 |
| q8_0 ub256 − ub4096 @32k × 8 | +8.28e-5 | 2.63e-5 | **3.14** | +3.3e-5 | 1.87e-4 | 7/8 |
| q4_0 ub256 − ub4096 @32k × 8 | +8.03e-5 | 8.42e-5 | 0.95 | −7.9e-5 | 5.91e-4 | 4/8 |
| **type vs f16 at the same ub:** | | | | | | |
| q8_0 − f16, ub4096 | +1.372e-4 | 1.25e-5 | 10.98 | +1.14e-4 | | 8/8 |
| q8_0 − f16, ub512 | +1.80e-5 | 1.36e-5 | 1.32 | −7.8e-6 | 7.3e-5 | 7/8 |
| q8_0 − f16, ub64 | +3.21e-5 | 6.5e-6 | 4.90 | +2.0e-5 | 7.1e-5 | 8/8 |
| q8_0 − f16, ub4096 @32k × 8 | +1.453e-4 | 3.37e-5 | 4.31 | +8.1e-5 | | 8/8 |
| q8_0 − f16, ub256 @32k × 8 | +3.05e-5 | 9.7e-6 | 3.13 | +1.2e-5 | | 8/8 |
| q4_0 − f16, ub4096 / ub512 / ub64 | +5.79e-3 / +5.66e-3 / +5.68e-3 | 6.3e-4 / 6.2e-4 / 6.5e-4 | ≈ 9 | | | 8/8 |

- **The "q8_0 cost" depends on the ub of the reference.** It is +1.37e-4 at ub4096, where the reference is the f16 arm itself, and only +1.8e-5 to +3.2e-5 once f16 has its own route-change offset (ub512, ub64, ub256).
- Pinned KLDs against a third distribution do not add. Any change of bits saturates near 1e-4 (R1-Q8 saturation finding). This is a second, independent instance of it.

**F. Vulkan** (PR219+#240 binary, ub4096 c8192 × 8)

| arm | reference | mean KLD (chunk SE) |
|---|---|---|
| vk f16, pinned, r1 = r2 | HIP f16 `ref8t0` | 5.519015e-3 (7.45e-4) |
| vk q8_0, pinned (tk / masked route), r1 = r2 | HIP | 5.521400e-3 (7.39e-4) |
| vk q4_0, pinned (tk) | HIP | 1.057718e-2 (1.24e-3) |
| vk f16, observe only | HIP | 2.052341e-2 (3.20e-3) |
| v f16 self, pinned | Vulkan f16 `vref8t0` | −4.8e-9 (6.9e-8) |
| v f16 ub2048, pinned | Vulkan | 4.980874e-4 (4.82e-5) |
| v f16 ub2048, observe only | Vulkan | 8.817473e-3 (1.52e-3) |
| v q8_0 ub4096, pinned | Vulkan | 5.706370e-4 (5.39e-5) |
| v q8_0 ub4096, observe only | Vulkan | 9.467617e-3 (1.65e-3) |

| paired | mean | SE_paired | t |
|---|---|---|---|
| vk f16 − HIP f16 (same pins, Q1 form) | +5.519e-3 | 7.45e-4 | 7.41 |
| vk q8_0 − HIP q8_0 | +5.384e-3 | 7.28e-4 | 7.40 |
| vk q4_0 − HIP q4_0 | +4.791e-3 | 7.27e-4 | 6.59 |
| vk q8_0 − vk f16 (both vs HIP ref) | +2.4e-6 | 3.49e-5 | 0.07 |
| v q8_0 − v f16 ub2048 (both vs Vulkan ref) | +7.25e-5 | 1.09e-5 | 6.63 |

- Before the pin is applied, the computed selection and experts of Vulkan f16 against the HIP reference agree less than HIP q8_0 does: QSA overlap 0.9842 vs 0.9967, MoE exact 0.7446 vs 0.9603 (QPIN_QSA/QPIN_MOE layer=-1, `vk_f16_u4096_r1.log`, `h_q8_0_u4096.log`).
- The PR219 Vulkan routes, from the hook tensors: f16 is maskless selected-key (cb_tensors 192). q8_0 and q4_0 are the masked route (tk_tensors 384), as in C11.

### Noise
- **Run to run: 0.** 11 self-pairs at %.9e: 9 HIP across 3 ubs × 3 types, plus Vulkan f16 and q8_0. Self-replays at ub4096 are ≤ 3.7e-7 per chunk (the uint16 base quantization).
  - So every SE in this section is chunk-to-chunk variation of the effect itself. No extra repeats would shrink it; only more chunks would.
- **Cross-route SE per ub** (the noise that new-vs-PR219 arms will see, with PR219's own route change as the proxy):

  | arm | f16 | q8_0 | q4_0 |
  |---|---|---|---|
  | ub512 | 2.46e-5 | 1.07e-5 | 0 |
  | ub64 | 2.84e-5 | 2.09e-5 | 6.38e-5 |
  | ub256 @32k × 8 | 4.55e-5 | 2.63e-5 | 8.42e-5 |

- **Vulkan device loss.** The first `vk_q8_0_u4096_r2` died at 07:36:49 CET with `vk::Queue::submit: ErrorDeviceLost`. The kernel journal shows `amdgpu_gem_va_update_vm … Couldn't update BO_VA (-12)` (ENOMEM) at the same second (`logs/vk_q8_0_u4096_r2_devlost.log.txt`; `journalctl -k`).
  - The rerun reproduced r1 exactly.
  - Its cause is UNPROVEN. Free memory afterwards was 44 GiB available, with GTT 82.0 of 133.1 GB used.

### Decision
1. **The objection is confirmed.** See table A, table B and the code lines above.
2. **The ub of every Q1/Q3 arm (to be written into PLAN-v3 §15):**

   | arm | context | ub | routes it reaches (new design) |
   |---|---|---|---|
   | Q1-A (existing) | c8192 × 8 | 4096 | G4 n_q 4096, packed V |
   | Q1-B (new, i) | c8192 × 8 | 512 | shortcut (f16 dense, others all-visible sparse) + G4 n_q 512 |
   | Q1-C (new, ii) | c8192 × 8 | 64 | shortcut + G1 n_q 64 |
   | Q3-A | c32768 × **8** (not × 2), `QKL_TAIL=4096` | 4096 | G4 n_q 4096 at depth |
   | Q3-B (new, iii) | c32768 × 8, `QKL_TAIL=4096` | 256 | G4 n_q 256; direct V for q8_0/q4_0 V at n_kv ≥ 16384 (731 live calls per chunk) |

   - Each arm runs for f16, q8_0 and q4_0 against PR219 of the same type at the same ub. Pins come from one f16 ub4096 reference per context, recorded with `LLAMA_FN_LAST_ATTN_TRIM=0` (rows_absent = 0 measured in all arms).
3. **Pass rule.** The source's "item-1 rule" is not defined in any file I could read (grep of the scratchpad). I apply R1-Q8 decision item 3, the band rule:
   - The rule is mean(new(T)_U − PR219(T)_U) ≤ B_U.
   - B_U = max over T of [mean(T_U − T_4096) + 3·SE], from PR219's own route change. Measured:
     - **B_4096 = 2.0e-4** (R1-Q8: max chunk of the q8_0 ub2048 reorder);
     - **B_512 = 2.03e-4** (from f16);
     - **B_64 = 2.75e-4** (from q4_0; f16 gives 2.73e-4);
     - **B_256@32k = 3.34e-4** (f16; q4_0 gives 3.33e-4).
   - Also report the direct form (KLD of new(T) against PR219(T)'s pinned output).
   - Do not use 3·SE_paired: correct route changes fail it at t 3.1–6.6 (table E).
   - These arms are coarse e2e screens. A D1-size defect (+1.19e-3) and a D2-size defect (+6.1e-4 mean) exceed every band. A D3-size defect (V × (1 + 1e-3)) is invisible (R1-Q8). **Q5 stays the primary gate for kernel-local numerics.**
4. **Vulkan Q1 arm.** Do not pair new-Vulkan(T) with new-HIP(T):
   - The cross-backend pinned KLD (5.5e-3, SE 7.4e-4) exceeds q4_0's whole cost, and 3·SE = 2.2e-3.
   - Instead, record a PR219+#240 Vulkan f16 ub4096 trim-0 reference (`vref8t0` recipe above). Then run new-Vulkan(T) vs PR219+#240-Vulkan(T) for each S_fast type under those pins, with the band rule.
   - **B_vk = 4.98e-4 + 3 × 4.82e-5 = 6.4e-4** (Vulkan f16 ub2048 route change).
   - PR219 Vulkan q8_0/q4_0 are on the masked route. As on HIP, a cross-route pair is valid under pins. Measured: Vulkan q8_0 masked lies +7.25e-5 above the Vulkan floor.
   - bf16 on Vulkan was not run here: UNPROVEN.
   - Budget one rerun per Vulkan arm, because of the device loss above.
5. **Still not gated e2e after these arms:**
   - G1 at n_q ≠ 64. n_q 1 decode and MTP n_q 2–8 cannot be pinned: QPIN skips ubatches of fewer than 16 tokens.
   - G4 at n_q ∈ (128, 512) other than 256 and 512.
   - Direct V with n_q 512 (needs n_kv ≥ 32768 at ub512, i.e. a c65536 tier: not measured).
   - These stay on Q5 (fixtures) and Q7. Q7 is not a numeric KLD gate.
   - Optional: a c32768 ub64 arm would add G1 at n_kv up to 32768. Not run here.
6. **Files.**
   - Deleted after the runs: `ref8t0` (22 GB), `vref8t0` (22 GB), `ref32t0` (10 GB), `ref32x8` (40 GB).
   - Kept on strix-002: `~/dev/wt_lab/r1ub/` (867 MB: `tree/` with `build-vk`, logs, census, scripts).
   - Mirror: `SP/final219/r1ub/` (1.6 MB). No process left running.

## R1-M-Lab E: noise of the quality thresholds (Q3/Q4/Q6/Q7)

**Objection.** "Q3/Q4/Q6/Q7 thresholds sit on point estimates with no measured noise" (PLAN-v3.md:585-589).

**Answers, one per gate.**
- **Q3: no run. The objection holds for the current text and is already fixed by R1-e2e.** R1-e2e Decision 2 moves Q3-A to c32768 × 8. That tier has a measured SE: q8_0 − f16 @32k × 8 = +1.453e-4, SE 3.37e-5, and B_256@32k = 3.34e-4 (R1-e2e table E, Decision 3). The fix is wording only (see Decision Q3).
- **Q4: the objection holds. Agreement noise is now measured, and it changes the rule.** Run to run the noise is 0. The per-chunk SE is 1.3e-5 for f16 KV + q8 idx, 3.0e-5 for q8_0, and 2.3e-4 for q4_0 (16 chunks).
  - A benign PR219 reorder (ub4096 → ub2048) moves f16 KV + q8 idx agreement by −1.54e-3, which is 85 paired SE.
  - So a pure "k × SE" margin would fail a correct rewrite. Decision Q4 adopts a band per type.
  - The forced kind-1 q4_0 route is feasible. It uses CPU attend at 577–632 s per 16-chunk arm. On kind 1 the ratio is q8 idx / f16 idx = 0.999802 (SE 3.1e-6).
  - The two estimators are not interchangeable. Kind 1 minus tk, same type, is −3.7e-4 to −5.9e-4, at t −11 to −16.
- **Q6: the objection holds.** C4 had 1 ub pair per rule; there are now 6 per rule. The same-rule ub noise reaches 2.554e-3 (p2) and 2.878e-3 (p1) for f16, and 2.872e-3 for q8_0 p2.
  - The TE → CB component at the same ub is 0.94–1.91e-3 (p2) and 1.15–2.45e-3 (p1), which is below the tolerance.
  - The gate is coarse: q8_0 vs f16 at the same rule and ub (0.84–2.46e-3) also sits inside the tolerance.
- **Q7: the objection holds for prompt sizing.** A maskless q4_0 draft (forced route, CPU attend) minus PR219's masked q4_0 draft gives −0.18 pp, SD 0.695 pp, 95% CI [−0.58, +0.22]. Against the f16 draft it gives −2.55 pp, CI [−4.17, −0.93].
  - **This settles U19: the q4_0 draft's acceptance loss is the q4_0 KV quantization, not the masked route.**
  - A same-type pair needs ≥ 6 prompts for a 95% CI half-width ≤ 1 pp, and ≥ 7 for 80% power, at the largest measured same-type SD (0.92 pp, L/X prompts only).

### Setup (strix-002, 2026-11-05 10:30–12:59 CET; I was the only GPU user; every build and GPU run under `flock ~/gpu.lock`)
- **Q4 binary.** `~/dev/wt_lab/r1av/tree/build/bin/llama-perplexity`.
  - The tree was at `r1av_full.diff` (md5 264f3bd03deb5627ea27f25af73ae2e0 before this pass, saved as `r1e/r1av_tree_before_r1e.diff`). I added `r1e/r1e_patch.py` (env-gated, inert without `QPIN`).
  - The patch adds a `QPIN_QSA_CHUNK chunk=c src={cb|tk1|tk2} n overlap jaccard exact` line (%.9f) per chunk and per hook fire to QPIN_STATS.
    - `tk1` is the first fire of `qsa_rows_tk-<il>` in a ubatch, i.e. the computed rows.
    - `tk2` is the second fire (the view, after pinning).
  - The full working diff is now `r1e/r1e_full.diff`, md5 8c599625e7e4149a5fc3868bf4d2a337. Build: `flock ~/gpu.lock cmake --build build -j 16 --target llama-perplexity`, BUILD_RC=0 (`r1e/build.log`). The cmake configuration is the existing HIP one of r1av (R1-e2e Setup).
- **Probe override.** No new code was needed. PR219 already has `LLAMA_QSA_SELECTED_KEY` (llama-memory-hybrid-idx.cpp:129 in PR numbering, :140 in this tree; L15).
  - With `LLAMA_QSA_SELECTED_KEY=1`, the q4_0 startup line reads `selected-key kernels (maskless block selection) (K q4_0, V q4_0)` (`c2/server_r1e_sk1_q4d.log`).
  - The perplexity arms report `cb_tensors=384 tk_tensors=0` (kind 1). Without the override they report `cb_tensors=0 tk_tensors=768` (`r1e/logs/e_q4_idxf16_cb.stats` vs `e_q4_idxq8_tk.stats`).
  - The maskless q4_0 FA has no HIP kernel. It ran on the CPU (no HIP trace line; 38.46 s per pass vs about 12 s, `r1e/logs/feas_q4_idxq8_cb.log`). That is slow but valid, as R1-dense-shortcut predicted. It completed rc=0 in every forced arm.
- **Runner.** `r1e/run.sh` is `r1ub/run.sh` with the log directory changed (diff shown in the session):
  `flock ~/gpu.lock env QPIN=replay QPIN_DIR=REF/pins QPIN_STATS=… QPIN_PIN=qsa,moe GGML_QSA_TRACE=1 <ENV> timeout 5400 …/llama-perplexity -m ~/models/qwen3.8-flash-next-UD-Q4_K_XL/…-00001-of-00004.gguf -f ~/data/wikitext-2-raw/wiki.test.raw -c 8192 -b 4096 -ub UB --chunks 16 -ngl 99 -fa on -lm mmap -lzm on-direct -ctk T -ctv T --kl-divergence-base REF/base.kld --kl-divergence`
- **Reference** `ref16t0`: f16/f16, ub4096, c8192 × **16**, `LLAMA_FN_LAST_ATTN_TRIM=0`, QPIN record. rc=0 in 200 s, `rows_rec=1572864 moe_rec=6225920 cb_tensors=384` (`r1e/campA.out`).
  - wiki.test.raw holds 16 chunks at c8192 (QKLD_CHUNK 0–15 in every arm), so the 2 disjoint sets of 8 exist.
  - In every arm, rows_absent = 0 and moe_absent = 0.
- **Arms.** Scripts `r1e/campA.sh`, `campB.sh`, `campC.sh`: 1 record + 18 replays, all rc=0.
  - f16: self, ub2048, idx q8 (×2), idx q8 ub2048.
  - q8_0: auto (q8 idx), idx f16, auto ub2048.
  - q4_0 tk (PR219 masked): idx q4, q8, f16, plus q8 and q4 at ub2048.
  - q4_0 kind 1 (forced): idx q4, q8 (×2: `feas_q4_idxq8_cb` copied as `e_q4_idxq8_cb`, then `_r2`), f16, plus q8 ub2048.
  - The indexer type is set with `QPIN_IDX_TYPE` (C9).
  - Analysis: `python3 r1e/q4_analyze.py r1e/logs` → `r1e/q4_analysis.out`; band pairs → `r1e/q4_band.out`.
- **Q6.** `r1e/q6.sh` runs `c4/run.sh` unchanged: `lab-c4` (build4), Signal-3.8 + mmproj, `img_nyt.jpeg`, `-c 40960 -b UB -ub UB -ctk f16 -ctv f16 --image-min-tokens 1024 -fit off`, score mode against `answer_p{1,2}.txt`.
  - Arms: LAB_QSA_C4 ∈ {0 (TE), 2 (CB)} × ub {8192, 4096, 2048, 1024} × {p2, p1} = 16 runs, all EXIT 0 (`r1e/q6.out`).
  - `r1e/q6q8.sh` runs the same 8 arms for p2 with `-ctk q8_0 -ctv q8_0`, through `r1e/run_c4_kv.sh`. That script is c4/run.sh with only the -ctk/-ctv pair parametrised (diff shown in the session). All EXIT 0.
  - Pairs: `r1e/q6_pairs.sh` (c4 `kld.py`) → `r1e/q6_pairs.out` and `r1e/q6q8_pairs.out`.
- **Q7.** `r1e/q7.sh`, `c2/run_c2.sh` unchanged (main KV q8_0, `--spec-type draft-mtp,ngram-mod`, `NOGRAPH=0`, 14 requests). It made 3 runs:
  - `r1e_repro_q4d` (PR219 masked q4_0 draft);
  - `r1e_sk1_q4d` (`LLAMA_QSA_SELECTED_KEY=1`, q4_0 draft);
  - `r1e_repro_f16d`.

  Analysis: `python3 r1e/q7_analyze.py A B` (paired per prompt, Student t, prompt-count search) → `r1e/q7_analysis.out`.
- **Mirror.** `SP/final219/r1e/` holds `r1e/` (scripts, 39 logs and stats, all .out files, both diffs), `c2/` (the 3 server logs and jsonl files) and `c4/logs/r1e_*` (24 logs).

### Raw numbers: Q4 (c8192 × 16, QSA+MoE pinned, `r1e/q4_analysis.out`, `q4_band.out`)
**A. Agreement per arm.** The estimator is the mean over query positions ≥ 2052 of |C∩R|/|R|. Per-chunk SD is across the 16 chunks. SE = SD/√n.

| arm (KV, idx, route, ub) | mean (16) | per-chunk SD | SE (16) | set 0–7 mean (SE) | set 8–15 mean (SE) | pinned KLD (16) |
|---|---|---|---|---|---|---|
| f16, f16, cb, 4096 (self) | 1.000000 | 0 | 0 | 1.000000 | 1.000000 | −2.6e-8 |
| f16, f16, cb, 2048 | 0.997417 | 9.30e-5 | 2.33e-5 | 0.997386 (2.84e-5) | 0.997449 (3.50e-5) | 9.58e-5 |
| f16, q8, cb, 4096 (r1 = r2) | 0.998528 | 5.35e-5 | 1.34e-5 | 0.998509 (2.03e-5) | 0.998546 (1.59e-5) | −2.6e-8 |
| f16, q8, cb, 2048 | 0.996987 | 9.89e-5 | 2.47e-5 | 0.996952 | 0.997023 | 9.6e-5 |
| q8_0, q8 (auto), cb, 4096 | 0.996743 | 1.20e-4 | 3.00e-5 | 0.996726 (3.46e-5) | 0.996759 (5.07e-5) | 1.179e-4 |
| q8_0, f16, cb, 4096 | 0.997148 | 1.16e-4 | 2.90e-5 | 0.997135 | 0.997162 | 1.179e-4 |
| q8_0, q8, cb, 2048 | 0.996684 | 1.35e-4 | 3.38e-5 | 0.996645 | 0.996723 | 1.31e-4 |
| q4_0, q4 (auto), tk | 0.974770 | 8.78e-4 | 2.20e-4 | 0.974650 | 0.974891 | 4.933e-3 |
| q4_0, q8, tk | 0.986079 | 8.96e-4 | 2.24e-4 | 0.986240 (3.52e-4) | 0.985918 (2.90e-4) | 4.933e-3 |
| q4_0, f16, tk | 0.986247 | 9.00e-4 | 2.25e-4 | 0.986410 (3.54e-4) | 0.986084 (2.90e-4) | 4.933e-3 |
| q4_0, q8 / q4, tk, ub2048 | bit-identical to ub4096 | | | | | 4.933e-3 |
| **q4_0, q4, kind 1 (forced)** | 0.974182 | 9.24e-4 | 2.31e-4 | 0.974081 | 0.974284 | 4.995e-3 |
| **q4_0, q8, kind 1 (r1 = r2)** | 0.985678 | 9.40e-4 | 2.35e-4 | 0.985858 (3.50e-4) | 0.985498 (3.24e-4) | 4.995e-3 |
| **q4_0, f16, kind 1** | 0.985873 | 9.44e-4 | 2.36e-4 | 0.986058 (3.52e-4) | 0.985688 (3.24e-4) | 4.995e-3 |
| q4_0, q8, kind 1, ub2048 | 0.985734 | 8.95e-4 | 2.24e-4 | 0.985885 | 0.985582 | 4.832e-3 |

- **The tk estimator check.** In every chunk of every tk arm, the second fire `tk2` has overlap 1.000000000. The results-Q estimator 2·ov_all − 1 equals the first-fire value within 3.3e-16 (max |corr − direct|).
  - So results-Q's correction (C9 "Correction caveat") is now verified, not assumed.
- **Reproduction of C9 on the same chunks (set 0–7).**
  - The cb arms reproduce C9 to 6 decimals: 0.998509, 0.997386, 0.996726, 0.997135 (C9 table).
  - The tk q4 arms differ from C9: 0.986240 / 0.986410 / 0.974650 here vs 0.98615 / 0.98633 / 0.97441. The difference is ≤ 2.4e-4, which is < 1 set-SE (3.5e-4).
  - The references differ: C9 used measQ `ref8` with the default trim, this pass used r1av `ref16t0` with trim 0. That this is the cause is UNPROVEN.

**B. Paired per-chunk differences (B − A, n = 16; the two sets of 8 in brackets)**

| pair | mean | SD | SE_paired | t | sets 0–7 / 8–15 |
|---|---|---|---|---|---|
| any r2 − r1 (f16 idx q8; q4 kind 1 idx q8) | 0 | 0 | 0 | | 0 / 0 |
| **benign reorder, PR219 ub2048 − ub4096, same type:** | | | | | |
| f16 idx f16 (vs self = 1) | −2.583e-3 | 9.30e-5 | 2.33e-5 | −111 | −2.614e-3 / −2.551e-3 |
| f16 idx q8 | −1.540e-3 | 7.28e-5 | 1.82e-5 | −85 | −1.557e-3 / −1.523e-3 |
| q8_0 auto | −5.9e-5 | 7.85e-5 | 1.96e-5 | −3.0 | −8.0e-5 / −3.7e-5 |
| q4_0 tk (idx q8, idx q4) | 0 (bit-identical) | 0 | 0 | | |
| q4_0 kind 1 idx q8 | +5.6e-5 | 1.25e-4 | 3.12e-5 | +1.8 | +2.7e-5 / +8.5e-5 |
| **estimator / route, kind 1 − tk, same KV and idx:** | | | | | |
| q4_0 idx q8 | −4.01e-4 | 1.34e-4 | 3.35e-5 | −12.0 | −3.82e-4 / −4.21e-4 |
| q4_0 idx f16 | −3.74e-4 | 1.37e-4 | 3.44e-5 | −10.9 | −3.52e-4 / −3.96e-4 |
| q4_0 idx q4 | −5.88e-4 | 1.43e-4 | 3.57e-5 | −16.5 | −5.69e-4 / −6.07e-4 |
| **indexer type, same KV and route:** | | | | | |
| q4 KV kind 1: idx q8 − idx f16 | −1.95e-4 | 1.22e-5 | 3.05e-6 | −64 | −2.00e-4 / −1.90e-4 |
| q4 KV tk: idx q8 − idx f16 | −1.68e-4 | 8.70e-6 | 2.17e-6 | −77 | |
| q4 KV kind 1: idx q4 − idx f16 | −1.169e-2 | 6.66e-4 | 1.66e-4 | −70 | |
| q8 KV: idx q8 − idx f16 | −4.06e-4 | 1.99e-5 | 4.98e-6 | −82 | |
| f16 KV idx q8 − f16 ub2048 floor | +1.110e-3 | 7.75e-5 | 1.94e-5 | +57 | |

**C. The C9 ratio rule computed inside one build** (per-chunk ratio q8 idx / f16 idx, same KV, same estimator)

| KV, estimator | ratio, mean (SE) | sets 0–7 / 8–15 | C9 (tk-corrected, 8 chunks) |
|---|---|---|---|
| q4_0, kind 1 | **0.999802** (3.08e-6), one-sided LCB95 0.999796 | 0.999797 / 0.999807 | — |
| q4_0, tk | 0.999830 (2.19e-6) | 0.999827 / 0.999832 | 0.98615 / 0.98633 = 0.99982 |
| q4_0, kind 1, idx q4 / idx f16 | 0.988142 (1.68e-4) | 0.987854 / 0.988430 | 0.9879 (tk) |
| q8_0, cb, idx q8 / idx f16 | 0.999593 (5.0e-6) | | 0.99959 |

### Raw numbers: Q6 (`r1e/q6_pairs.out`, `r1e/q6q8_pairs.out`; pinned-answer KLD, `kld.py` A ∥ B)
- **Consistency.** The current build4 reproduces C4: the ub8192 and ub4096 reruns are KLD 0.000e+00 against the C4 `p{1,2}_{te_s1,cb_s1,te_ub4096,cb_ub4096}.bin` files in 8/8 comparisons. So C4's "repeats bit-identical" holds across the libllama rebuild of 2026-11-05 00:36.
- **ub1024 runs.** The 1036-token image spans 2 ubatches there. They completed with the correct answer length (n_answer 97 / 92).

| prompt, KV | rule | 8192-4096 | 8192-2048 | 8192-1024 | 4096-2048 | 4096-1024 | 2048-1024 | mean (SD) | **max** |
|---|---|---|---|---|---|---|---|---|---|
| p2 f16 | TE | 2.348e-3 | 1.308e-3 | 2.554e-3 | 2.423e-3 | 2.418e-3 | 2.495e-3 | 2.258e-3 (0.47e-3) | **2.554e-3** |
| p2 f16 | CB | 1.103e-3 | 1.688e-3 | 2.310e-3 | 1.910e-3 | 2.239e-3 | 2.250e-3 | 1.917e-3 (0.47e-3) | 2.310e-3 |
| p1 f16 | TE | 1.680e-3 | 1.294e-3 | 2.269e-3 | 2.124e-3 | 1.933e-3 | 1.773e-3 | 1.845e-3 (0.35e-3) | 2.269e-3 |
| p1 f16 | CB | 1.532e-3 | 2.066e-3 | 1.990e-3 | 2.114e-3 | 2.878e-3 | 2.857e-3 | 2.240e-3 (0.53e-3) | **2.878e-3** |
| p2 q8_0 | TE | 1.298e-3 | 1.719e-3 | 1.379e-3 | 2.422e-3 | 2.001e-3 | 1.743e-3 | 1.760e-3 (0.41e-3) | 2.422e-3 |
| p2 q8_0 | CB | 2.768e-3 | 2.105e-3 | 1.902e-3 | 2.872e-3 | 1.893e-3 | 2.312e-3 | 2.309e-3 (0.43e-3) | **2.872e-3** |

| TE vs CB at the same ub | 8192 | 4096 | 2048 | 1024 | max |
|---|---|---|---|---|---|
| p2 f16 | 9.397e-4 | 1.905e-3 | 1.753e-3 | 1.521e-3 | 1.905e-3 |
| p1 f16 | 1.266e-3 | 1.148e-3 | 2.452e-3 | 1.550e-3 | 2.452e-3 |
| p2 q8_0 | 1.785e-3 | 1.852e-3 | 1.562e-3 | 1.428e-3 | 1.852e-3 |

- **f16 vs q8_0, same rule and same ub (p2), 8 pairs:** 0.843e-3 to 2.461e-3, mean 1.72e-3.
- Same top-1 is 97.8–100% in every pair. Log-prob shifts are within ±0.89 nats across all pairs.
- The 6 pairs per rule share runs, so they are not independent samples. The max is a sample max.

### Raw numbers: Q7 (`r1e/q7_analysis.out`; acceptance = draft_n_accepted/draft_n per request, paired per prompt, Student t, df 13)

| pair | mean | SD | 95% CI | identical trunk text |
|---|---|---|---|---|
| r1e_repro_q4d − g1_repro_q4d (C2) | 0 | 0 | | 14/14 |
| r1e_repro_f16d − g1_repro_f16d (C2) | 0 | 0 | | 14/14 |
| **maskless q4_0 draft − PR219 masked q4_0 draft** | **−0.179 pp** | **0.695 pp** | **[−0.580, +0.222]** | 14/14 |
| same, L/X prompts only (n = 8) | −0.313 pp | 0.921 pp | | |
| maskless q4_0 − f16 draft (report only) | −2.550 pp | 2.810 pp | [−4.173, −0.928] | 14/14 |
| PR219 masked q4_0 − f16 (C2, recomputed) | −2.371 pp | 2.538 pp | [−3.836, −0.906] | 14/14 |
| C2: q8_0 − f16, repro, L/X only | −0.167 pp | 0.871 pp | | |
| C2: q8_0 − f16, MTP only, L/X only | +0.141 pp | 0.767 pp | | |

- **Per prompt, maskless − masked:** S0–S5 +0.00 (6/6 identical), L0 +0.00, L1 −0.74, L2 −1.03, L3 −0.81, L4 −1.21, L5 +1.08, X0 +1.05, X1 −0.84. Totals: 2906/4684 = 0.6204 vs 2910/4672 = 0.6229.
- **Route census** (`summ_c2.py`):
  - The masked draft has 2502 `dense_tile_vec_mma nq=1 nkv>2051 mask ids q4_0` launches.
  - The forced draft has **0** HIP launches past 2051. It prints 64 `C2CPU_FA node_146 nq 1 … ids 1 mask 0 K q4_0 V q4_0` lines; 64 is the print cap (`n_cpu++ < 64`, wt_lab ggml-cpu/ops.cpp:9697).
  - So every past-shortcut draft attention ran maskless on the CPU.
  - Both drafts run the same 1866 shortcut launches (n_kv ≤ 2051, dense masked).
  - **S0–S5 never leave the draft's shortcut, so they cannot see any change past 2051. That gives 6 structural zeros.**
- **Prompt counts** for a same-type pair (`q7_analyze.py` search with Student t):

  | per-prompt SD | n for 95% CI half-width ≤ 1 pp | n for 80% power, one-sided α 0.05, true Δ = 0 passes LCB ≥ −1 pp |
  |---|---|---|
  | 0.695 | 5 | 5 |
  | 0.768 | 5 | 6 |
  | 0.921 | 6 | 7 |
  | 2.81 (vs f16) | 33 | 51 |

  - At SD 0.921, the half-width is 0.53 pp with 14 prompts and 0.77 pp with the 8 L/X prompts.

### Noise
- **Run to run: 0** for every harness.
  - Q4: two agreement r2 − r1 pairs of 16/16 chunks at %.9f, plus every pinned KLD equal.
  - Q6: 8/8 old-vs-new comparisons at 0.
  - Q7: 2 × 14/14 prompts equal to C2.
- So all spreads above are chunk-to-chunk or prompt-to-prompt variation, and only more chunks or prompts shrink them.
- **Q4 per-chunk SE** (16 chunks, chunk-to-chunk): 1.34e-5 (f16 + q8 idx), 2.3e-5 (floor), 3.0e-5 (q8_0), 2.2–2.4e-4 (q4_0).
  - The paired SE is much smaller: 2e-6 to 3.6e-5, because the chunk effect cancels.
  - The two sets of 8 agree within 1.5 combined SE for every arm (table A).

### Decision
**Q4: replace `≥ 0.9985` and `≥ 0.9861` (PLAN-v3.md:586).**
1. **Recipe.**
   - Reference: f16 ub4096 trim-0 (`ref16t0` recipe above), c8192 × 16 chunks.
   - QPIN_STATS emits `QPIN_QSA_CHUNK`: port `r1e/r1e_patch.py` into the measurement build.
   - Every arm runs QSA+MoE pinned.
2. **Paired band gate per type T, same estimator on both sides.**
   - Rule: mean_c(new_T − PR219_T) ≥ −B_T, with **B_T = |Δ_reorder,T| + k·SE_paired,reorder,T** and **k = 3**.
   - Δ_reorder is PR219's own ub4096 → ub2048 shift (table B). Measured:

   | type | PR219 value (16 chunks) | B_T | gate |
   |---|---|---|---|
   | f16 KV, f16 idx (self) | 1.000000 | 2.583e-3 + 7.0e-5 = **2.65e-3** | ≥ 0.99735 |
   | f16 KV, q8 idx | 0.998528 | 1.540e-3 + 5.5e-5 = **1.60e-3** | ≥ 0.99693 |
   | q8_0 KV, auto (q8) idx | 0.996743 | 5.9e-5 + 5.9e-5 = **1.18e-4** | ≥ 0.996625 |
   | q4_0 KV, q8 idx (new `auto`), kind 1 | 0.985678 (PR219 forced with `LLAMA_QSA_SELECTED_KEY=1`) | 5.6e-5 + 9.4e-5 = **1.50e-4** | ≥ 0.985528 |

   - **A pure k·SE margin is rejected.** PR219's own benign reorder is −85 SE_paired for f16 + q8 idx. As with the KLD arms (R1-e2e item 4), a correct but non-bit-identical rewrite would fail it.
   - Detection: a q4_0 indexer on f16 KV (C9: 0.9794, −1.9e-2) and on q4 KV (−1.17e-2) both exceed every B_T by ≥ 70×. A sub-band selection defect is not covered here; it stays with Q5 (plan exactness).
3. **q4_0: pair kind 1 against kind 1, never against tk.**
   - The new design's q4_0 is maskless complete-block, which is kind 1.
   - PR219's partner is the forced kind-1 run (`LLAMA_QSA_SELECTED_KEY=1`, CPU attend, 577–632 s per 16-chunk arm, rc=0 in 5/5 arms). The env var and probe are deleted only in the new build (D15), not in the PR219 measurement binary.
   - Kind 1 − tk is −4.0e-4 (t −12). That is 2.7× B_q4, so a cross-estimator pair would fail a correct build.
   - **Caveat.** The forced partner's attention runs on the CPU, so its numerics differ from a GPU kernel. That difference is inside table B's kind-1 numbers and is covered by the band through Δ_reorder only to the extent the ub2048 shift represents it: UNPROVEN.
4. **The C9 ratio rule inside one build, kind 1:** q8 idx / f16 idx ≥ 0.99. Measured 0.999802, SE 3.1e-6. The q4 indexer gives 0.988142 (SE 1.7e-4) and fails it by 11 SE. So 0.99 separates the two indexer types, and the measured margin of q8 above it is 0.0098 = 3200 SE.
   - The earlier C9 values (tk-corrected 0.98615/0.98633) are superseded for the gate. On tk the same ratio is 0.999830, so the ratio is estimator-robust, while the levels are not.
5. **Other types** (bf16, mixed q8K/f16V and f16K/q8V): run each type's own ub2048 shift on PR219 and apply the same B_T formula. Not measured here: UNPROVEN. The plan's "PR219 − 0.001" would be looser than q8_0's measured B and tighter than f16's.

**Q6: images (PLAN-v3.md:588).**
1. **Tolerance τ_{P,T} = max over the 6 ub pairs of both rules (TE and CB), per prompt and KV type:**
   - **τ_p2,f16 = 2.554e-3**;
   - **τ_p1,f16 = 2.878e-3**;
   - **τ_p2,q8_0 = 2.872e-3**;
   - τ_p1,q8_0 not measured: UNPROVEN, so run the 8-arm matrix on PR219 before use.

   Both rules apply because the gated pair spans them: new = CB everywhere (R1 D6 / C4), PR219 = TE on the image ubatch.
2. **Gate.** KLD(new_T, CB ∥ PR219_T, TE) at ub 8192 ≤ τ_{P,T} for T ∈ {f16, q8_0}, against PR219 **of the same type** (not PR219 f16). Keep the content check (headline, newspaper, date) and the `~/repro_img` A/B/C answers.
3. **The TE → CB component is accounted for explicitly.** At ub 8192 it is 9.397e-4 (p2) and 1.266e-3 (p1) for f16, and 1.785e-3 (p2) for q8_0. That uses 37%, 44% and 62% of τ, so a correct CB rewrite has 1.6e-3, 1.6e-3 and 1.1e-3 of headroom at ub8192.
   - The thinnest point is p1 f16 TE vs CB at ub2048: 2.452e-3 = 85% of τ. So the gate is evaluated at ub8192 only, with the other ubs reported.
4. **Stated weakness.** q8_0 vs f16 at the same rule and ub is 0.84–2.46e-3, inside τ. This 97-token gate cannot resolve the whole q8_0 cost. It is a coarse screen that catches only defects whose pinned-answer KLD exceeds about 3e-3. Q1 and Q5 remain the primary gates.

**Q7: MTP (PLAN-v3.md:589).**
1. **Pair same type to same type.** The gate is per draft type T ∈ {default (f16), q8_0, q4_0, bf16}: paired per prompt new(T) − PR219(T), 95% one-sided LCB ≥ −1 pp.
   - For q4_0, PR219(T) is the masked draft (`g1_repro_q4d`, bit-identical to `r1e_repro_q4d`).
   - The comparison against the f16 draft is reported only: q4_0 costs −2.55 pp there (CI [−4.17, −0.93]), so a correct q4_0 build would fail an f16-referenced gate.
2. **Prompt count.**
   - The largest measured same-type SD is 0.921 pp (maskless − masked q4_0, L/X prompts). It needs n ≥ 6 for a CI half-width ≤ 1 pp and n ≥ 7 for 80% power.
   - **Only prompts that pass n_kv 2051 in the draft carry information.** Require ≥ 8 L/X-class prompts (the current L0–L5, X0, X1 give half-width 0.77 pp at that SD) and keep S0–S5 as shortcut coverage.
   - bf16 draft: SD not measured, UNPROVEN.
3. **U19 is separated.** Masked → maskless at the same q4_0 type changes acceptance by −0.18 pp (CI [−0.58, +0.22]) out of the −2.55 pp loss vs f16. The loss is the q4_0 draft KV quantization, not the masked top-k route.
   - **Caveat.** The maskless attend ran on the CPU, so this is route plus CPU numerics. A route effect at least as large as the q4_0 cost is excluded; a sub-0.6 pp route effect is not.
4. **The W-build kernel is not needed for this measurement.** The same-type pairing ships regardless.

**Q3: no run (PLAN-v3.md:585 and :593).**
- **Replace Q3's rule.** The current rule is "c32768 × 2 … within ±3.2e-5 (2 × the 8k-derived per-chunk SE estimate 1.6e-5)". Replace it with the R1-e2e Q3-A tier:
  - c32768 × 8, `QKL_TAIL=4096`, ub4096;
  - band rule mean(new_T − PR219_T) ≤ B;
  - cite the measured q8_0 − f16 @32k × 8 = +1.453e-4, SE 3.37e-5;
  - Q3-B ub256 uses B_256@32k = 3.34e-4 (R1-e2e table E, Decision 3).
- **Delete the sentence "The c8192 × 8 tier is the only one with a measured σ; 32k is a confirmation tier"** (:593). C8 showed the 8k σ was resolution-limited (5-decimal tables), so the t ≈ 3.6 and MDE 1.6e-5 figures are UNPROVEN as noise (C8 "Noise", decision 2–3).
- **Replace it with:** "Per-chunk SEs come from full-precision QKLD_CHUNK output (W12): c8192 × 8 (R1-e2e table E), c32768 × 8 (R1-e2e), c8192 × 16 for agreement (R1-M-Lab E)."
- The 32k tier is no longer confirmation-only: it has its own measured SE.

### Large files
- Deleted on strix-002:
  - `~/dev/wt_lab/r1e/ref16t0` (43 GB);
  - `~/dev/wt_lab/c4/logits/*.bin` (3.4 GB, including the C4 originals);
  - `~/dev/wt_lab/c4/dump/` (2.4 GB). This clears the C4 "Large files" item.
- Kept: `~/dev/wt_lab/r1e/` (672 KB) and `~/dev/wt_lab/c4/` (6.6 MB, logs and scripts). The r1av tree carries the r1e patch, which is inert without `QPIN`.
- No process left running (ps check after the last run).

## OWNER DECISION 2026-11-05: f16/bf16 V-pack cap is DEFERRED (binding for every plan revision, Implement and Prove)
The repository owner has not decided yet whether f16/bf16 V is capped (packed only while n_kv < 64·n_q, PR219 behaviour, direct V beyond) or always packed (PLAN-v3 D8, pack reserve sized at n_ctx). Do NOT settle it in the plan. Instead:
- **Implement both behind one switch.** The f16/bf16 pack limit is a single per-type trait/constant. Default = CAPPED, i.e. PR219's behaviour: reserve min(n_kv, 64·n_q − 1) cells × 1 KiB, and direct V at n_kv ≥ 64·n_q. Setting the alternative (always packed, reserve n_ctx × 1 KiB) must be a one-line change, with the reserve formula following the trait. For the f16 direct-V loader, choose from the C18 V-loader table. At syn 131k / 262k / 1M: Kf16D-Vf16S 43.16 / 52.92 / 63.65 ms, Kf16D-Vf16X 44.78 / 57.59 / 65.28, PR-VD-f16 52.01 / 60.67 / 65.91.
- **Gate the default (capped).** new f16 reserve ≤ PR219 reserve + PR219 pool high-water at -c 524288 and -c 1048576 (ub4096).
- **Acceptance test in Prove (decision data for the owner, not pass/fail).** Measure both settings on the rewrite:
  - f16, real text (wikitext) end-to-end prefill t/s, ub4096, at depth ~300k (n_kv just past 262144) and at the deepest depth that fits;
  - compute-buffer reserve and peak VRAM for each;
  - the plan's repetition/noise rule.
  Report it in PROOF and in the PR description as an open owner decision, with the kernel-level C18 numbers (packed+pack 42.34 vs best direct 52.92 ms at syn 262k; 62.97 vs 63.65 at syn 1M) next to the e2e result.

## R1-M-Lab A, shrunk to the op-level correctnes

**Question.** For each type pair and shape the rewrite ships, what is the distribution of op output error, so that every W1/W3/W4/W5/Q5 threshold is a measured value instead of the unsourced 5e-7 / 4e-4 (PLAN-v3.md:436, :587)? Five measurements: (1) GPU vs CPU-contract NMSE; (2) GPU vs f64 max|err|/max|ref| p99.9 / p100 under the C18 oracle and under the §3 oracle (PLAN-v3.md:146-155); (3) CPU FA+src[5] vs the §3 contract for {f32, f16, bf16, q8_0, q4_0}²; (4) contract vs f64; (5) whether three planted defects land above the band. Before this, the only evidence was 3 queries × 24 heads × layer 0 (MEASUREMENTS.md:2299-2301, C18).

**Answer.**
- **(1) GPU vs §3 contract, per-op NMSE, p100 over every pair, variant and layer** (`SP/final219/bands/kdecide.md`, `compact.md`). The band depends on the shape and the data. It hardly depends on the type pair: the 10 pairs fall within ×1.12 of each other at prefill and at decode n_q 512, and within ×1.9 at n_q 1 on wiki (1.21e-9 for q8_0 to 2.27e-9 for q4_0).

  | shape (kernel) | wiki d4k (real Q/K/V, 12 layers) | SYN 262k (C6 spread, U 1.8) |
  |---|---|---|
  | prefill n_q 4096 (G4 attend, C18 harness) | **5.78e-10** | **1.34e-9** (seed 1 and seed 2) |
  | decode n_q 512 (`qsa_decode_wmma_partial`, splits 1) | **5.15e-10** | **1.52e-9** |
  | decode n_q 1 (splits 33, chunk 64) | **2.27e-9** | **5.73e-8** |

  - Per-row max|err|/max|ref| vs contract, p100: 4.65e-4 for prefill wiki and ≤ 3.9e-4 for every other cell.
  - Every variant of a pair is bitwise equal to the PR-verbatim kernel of that pair. The test covered 264 comparisons over every layer and fixture, with 0 words differing. So the bands hold for the PR kernels and for the shipped C18 loaders alike.
  - The n_q 1 band is ×4 (wiki) to ×40 (SYN) wider than n_q 512. In the split-K decode, each split rounds P to f16 against its own local max, so those roundings are no longer the ones the contract makes. For n_q 1 SYN, NMSE vs f64 C18 is 2.3e-8 for the GPU and 3.7e-8 for the contract, and the two add up to the 5.7e-8 GPU-vs-contract figure. The arithmetic is measured; the mechanism is UNPROVEN.
- **(2) GPU vs f64, max|err|/max|ref| p99.9 / p100 per row (query, head), max over pairs:**

  | shape / fixture | C18 oracle (q·scale and K/V in f16) | §3 oracle (q·scale f32, stored K/V, no f16) |
  |---|---|---|
  | prefill wiki d4k | 1.67e-4 / 3.68e-4 | 1.79e-3 / **2.75e-3** (q8_0) |
  | prefill SYN 262k | 3.33e-4 / **4.41e-4** (q8_0) | 6.78e-4 / 9.17e-4 |
  | decode n_q 512 wiki / SYN | 1.67e-4 / 2.93e-4; 3.45e-4 / 4.29e-4 | 1.71e-3 / 2.57e-3; 6.77e-4 / 8.37e-4 |
  | decode n_q 1 wiki / SYN | 1.73e-4 / 1.73e-4; 2.63e-4 / 2.63e-4 | 1.37e-3 / 1.37e-3; 5.57e-4 / 5.57e-4 |

  **The plan's "max|err|/max|ref| ≤ 4e-4 vs f64" fails on the clean kernels.** Under the §3 oracle, every type exceeds it on real text, f16 included (f16 is 1.83e-3). Under the C18 oracle, SYN 262k exceeds it at p100 (4.41e-4).

  Most of the §3-oracle error is the f16 rounding of q·scale, which the contract itself prescribes (PLAN-v3.md:150). On f32/f32, contract vs §3 f64 is 1.78e-3, while CPU FA, which keeps Q in f32, is at 1.2e-5 (`compact_cpu.md`). C18's 0.5e-4 – 3.7e-4 was the C18 oracle on 72 rows. The p100 over 405 504 rows is 3.7e-4 on wiki and 4.4e-4 on SYN.
- **(3) CPU FA + src[5] vs the §3 contract: W1's "equals CPU FA within f32 rounding" (PLAN-v3.md:422) is false for all 25 pairs.** The closest pairs, K f16 with V f32/bf16, differ by the contract's own P rounding: 1.2e-9 on wiki, 3.7e-8 on SYN. Per-op NMSE p100 over wiki (12 layers) and SYN 262k (`compact_cpu.md`):

  | K \ V | f32 | f16 | bf16 | q8_0 | q4_0 |
  |---|---|---|---|---|---|
  | f32 | 7.4e-8 | **1.1e-3** | 7.4e-8 | 1.2e-7 | 1.1e-7 |
  | f16 | 3.7e-8 | **1.7e-3** | 3.7e-8 | 7.8e-8 | 7.3e-8 |
  | bf16 | 2.5e-6 | **1.7e-3** | 2.5e-6 | 2.6e-6 | 2.6e-6 |
  | q8_0 | 1.3e-5 | **1.7e-3** | 1.3e-5 | 1.3e-5 | 1.3e-5 |
  | q4_0 | 1.3e-5 | **1.7e-3** | 1.3e-5 | 1.3e-5 | 1.3e-5 |

  The causes, read in the code:
  - **f16 V:** CPU FA accumulates VKQ in f16 with `ggml_vec_mad_f16` (PR ops.cpp:9119). The per-row maxrel reaches 0.15.
  - **K q8_0 / q4_0:** Q is converted to the K type's `vec_dot_type`, which is q8_0 for both (ops.cpp:9014, :9058; ggml-cpu.c:451 for q4_0, :519 for q8_0).
  - **K bf16:** Q becomes bf16 (vec_dot_type BF16, ggml-cpu.c:639-642). Measured: CPU FA bf16/f32 vs f64 §3 has NMSE 5.2e-7, while f16/f32 has 8.1e-9.
  - **K f16 or f32 with any non-f16 V:** CPU FA keeps P in f32, and for K f32 it also keeps Q in f32: f32/f32 vs f64 §3 is 7.3e-13. So it differs from the contract by the contract's own f16 roundings, up to 1.2e-7 on SYN.

  The PR's own op test shows the same thing (`logs/tbo_QSA_*.log`, 78 cases plus 1 not supported, all pass):
  - GPU vs CPU-FA NMSE is 4.3e-5 – 4.7e-5 for f16 V with 2051 selected cells, 5.4e-6 – 5.7e-6 with 257, 1.56e-6 – 1.79e-6 for q8_0/q8_0, and 7.7e-8 for f16 K / q8_0 V.
  - So **the 5e-7 "NMSE vs CPU" of W3 fails on 76 of the 78 supported PR shapes against today's CPU FA.** The two that pass are 128×512×17 f16 (3.8e-7) and f16 K / q8_0 V (7.7e-8). The threshold is only meaningful against the new CPU op (= contract).
  - These cases pass today only because the maskless `err()` returns max(GPU-vs-f64, CPU-vs-f64) against 5e-4 (tests/test-backend-ops.cpp:11605, :11416).
  - GPU vs f64 §3 on these shapes: f16 3.5e-8 – 4.7e-8; q8_0 7.3e-8 – 9.0e-8; the GQA 4/8 cases 1.8e-14 – 3.1e-14. That last figure means GQA 4/8 run a non-WMMA f32 path, so a band set on the WMMA kernels does not cover them.
- **(4) Contract vs f64, every pair:**
  - C18 oracle: NMSE ≤ 1.21e-9 (wiki) / 3.67e-8 (SYN), maxrel p100 ≤ 3.67e-4 / 4.49e-4.
  - §3 oracle: NMSE ≤ 5.15e-8 (wiki, q8_0/q8_0) / 1.55e-7 (SYN), maxrel p99.9 ≤ 1.80e-3 and p100 ≤ 2.74e-3 (wiki).
  - The contract's own NMSE distance from f64 is 2× (wiki) to 27× (SYN) the GPU-vs-contract band under the C18 oracle, and 89× (wiki) to 116× (SYN) under the §3 oracle.
- **(5) Planted defects (CPU, applied to the contract, measured against the clean contract), min over layers and pairs at the same row count** (`kdecide.md`):

  | shape | fixture | band | d1 tail cell | d2 one block | d3 V×(1+1e-3) | max k |
  |---|---|---|---|---|---|---|
  | prefill | wiki | 5.78e-10 | 5.05e-4 | 7.63e-7 | 8.80e-7 | 1320 |
  | prefill | SYN | 1.34e-9 | 4.70e-4 | 1.98e-3 | 1.04e-6 | 776 |
  | decode n_q 512 | wiki | 5.15e-10 | 5.05e-4 | 7.63e-7 | 8.80e-7 | 1482 |
  | decode n_q 512 | SYN | 1.52e-9 | 4.72e-4 | 2.02e-3 | 1.04e-6 | 684 |
  | decode n_q 1 | wiki | 2.27e-9 | 2.11e-5 | **1.38e-8** | 8.73e-7 | **6.1** |
  | decode n_q 1 | SYN | 5.73e-8 | 8.36e-4 | 9.73e-4 | 1.03e-6 | 18.0 |

  - **All three defects exceed 2 × the band in every cell, so k = 2.**
  - With one global band (5.73e-8, set by n_q 1 SYN), **d2 at n_q 1 on real text is not detected** (1.38e-8 < 1.15e-7). The band must therefore be per (shape class, fixture).
  - Detection also holds under the triangle bound NMSE(defective GPU, contract) ≥ (√D − √B)². For the tightest cell, n_q 1 wiki d2, that gives 4.9e-9 > 2B = 4.5e-9.
  - The per-row maxrel metric is thinner. Its max k is 2.7 at prefill wiki, set by d3: 1.28e-3 against a band of 4.65e-4.
  - The plan's 5e-7 would have detected d3 only at ≤ 1.76×, and would not have detected d2 at n_q 1 (1.38e-8 < 5e-7).

### Setup (strix-002, 2026-11-05 13:11–13:58 CET; I was the only GPU user; every GPU run and every build under `flock ~/gpu.lock`; nothing in `~/dev/wt_pr219v2*` modified)
- **Fixture (step 0).**
  - Command: `flock ~/gpu.lock env LLAMA_FN_LAST_ATTN_TRIM=0 ../c5/c5_capture $M ~/data/wikitext-2-raw/wiki.test.raw 8192 4096 cap2/wikiS f16 4096 0 4096`, run in `~/dev/wt_lab/c18`.
    - M = `~/models/qwen3.8-flash-next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`.
    - `c5_capture` links `~/dev/wt_pr219v2/build/bin/libllama.so.0` (ldd), the PR build, used read-only.
    - The output went to `cap2/` so the kept rows were not overwritten.
  - Result: EXIT 0 in 94 s (`logs/cap_campaign.out`), 12 `C5CAP … n_q=4096 ns=2051 nk=8192 mask=0` lines.
  - `cmp cap2/wikiS_p4096.rows cap/wikiS_p4096.rows` → **ROWS_IDENTICAL**. md5 is da31456343863b233c1fcdcf0eea2e9e for both.
  - `cap2/` (the 1.4 GB .qkv and the duplicate .rows) was deleted after the runs. The 4 kept C18 row files are untouched.
  - wikiL d131k was not recaptured, and no d32k/d65k fixture was added.
- **Code.** All of it is in `~/dev/wt_lab/bands/`. Copies are in `SP/final219/bands/src/` (my sources) and `SP/final219/bands/remote/` (everything fetched: sources, res, logs).
  - md5: `bands_ref.h` 17b3ac60…, `bands_attn.cu` 1ce47bf2…, `bands_dec.cu` 51e98fc1…, `bands_cpu.cpp` aa8911cb….
  - `bands_ref.h` holds what every harness shares: the fixture reader, the SYN generator (C18's, verbatim), the caches (C18's `build_cache`), dequant, the three oracles and the error statistics. Host arithmetic is not FMA-contracted (`#pragma clang fp contract(off)`, `g++ -ffp-contract=off`).
  - **Oracles** (per (query, head); the row's valid cells deduplicated and ascending):
    - **O_C18**, f64: q = f16(f16(q)·f16(1/16)); K/V = f16(to_float(x)); bf16 is saturated to ±65504 first. This is c18_attn.cu:290.
    - **O_PLAN**, f64: q = q·scale in f32; K/V = to_float(x) with no f16 rounding.
    - **O_CON**, the §3 contract in f32:
      - q·scale → f16; K/V → to_float → f16;
      - f32 scores, summed sequentially over d;
      - per-cell online softmax with `expf`;
      - P → f16 for PV, while l sums the unrounded f32 p, as the HIP kernel does (c18_kernel.inc:562);
      - f32 PV, then acc / l.
  - **Same data in every harness.** FNV-1a hashes of rows, Q and every cache are identical between `bands_attn` and `bands_cpu`: 0 mismatches on layers 0, 3, 5, 11, and equal on SYN.
    - The q8_0, q4_0 and bf16 caches equal `ggml_quantize_chunk` byte for byte: `QUANT_CHECK … identical 1`, 48/48.
    - A first smoke build with FMA contraction gave q4_0 `identical 0`, which is why contraction is now off.
  - **`bands_attn`** is c18_attn.cu with the timing loop replaced by a check mode.
    - Kernels and plan are C18's `c18_kernel.inc` / `pr_attn.inc` / `pr_union_kernels.inc` from `~/dev/wt_lab/c18`, unchanged (md5 as in C18 Setup).
    - Added instantiations: K{q8_0,q4_0,bf16}X-V<same>P and Kf16D-V{q8_0,q4_0,bf16}P.
  - **`bands_dec`** is c18_dec.cu lines 1-279 verbatim (the decode kernels and `pr_dec.inc`) plus a new main.
    - The partials are merged on the host with the PR's `qsa_decode_merge` arithmetic (qsa-decode.cu:65-79: f32, `expf`, same order).
    - SYN rows are taken as the first n_q queries of the 4096-query C6 plan.
  - **`bands_cpu`** links the CPU backend of the new tree.
    - It builds `ggml_flash_attn_ext` + `ggml_flash_attn_ext_add_top_k(…, 0)` + `ggml_prec_set_acc(F32)`, no mask, scale 1/16, the graph of `test_qsa_prefill_maskless`.
    - It runs all 4096 queries.
    - The defects use the R1-Q8 lab definitions (`SP/final219/r1q8/r1q8_lab.diff:73-87`):
      - d1 sets the last valid tail slot (W−3..W−1) to −1. That is 3072 cells per wiki layer, because q%4 = 3 rows have no tail slot.
      - d2 sets the median valid block to −1: 4096 blocks per layer.
      - d3 multiplies the stored V by 1.001 before the f16 rounding. After re-rounding to f16 the effective NMSE is 0.88e-6 to 1.2e-6, not exactly 1e-6. R1-Q8's d3 scaled attn_output instead.
- **Builds** (`logs/build_*.log`, BUILD 0 each):
  - `flock ~/gpu.lock /opt/rocm/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -I../c18 -Rpass-analysis=kernel-resource-usage bands_attn.cu -o bands_attn`, and the same for `bands_dec.cu`.
  - `g++ -O3 -march=native -ffp-contract=off -std=c++17 -I$T/ggml/include bands_cpu.cpp -L$T/build/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,$T/build/bin -lpthread -o bands_cpu`, with T = `~/dev/wt_lab/bands/tree`.
- **Tree for step 3.** Created with `git -C ~/dev/wt_lab worktree add --detach ~/dev/wt_lab/bands/tree 94e086847`.
  - One-line patch (`tree_bands.diff`): after `ud->tc->err(…)` at tests/test-backend-ops.cpp:1637, print `BANDS_NMSE_VS_CPU … nmse(f2=CPU, f1=ROCm0)`.
  - HIP build with the hard-rule flags, `cmake --build build --target test-backend-ops -j 16`: CMAKE 0, BUILD 0.
  - Runs, one at a time: `flock ~/gpu.lock ./build/bin/test-backend-ops -o QSA_PREFILL_MASKLESS -b ROCm0` (26/26 passed), then `-o QSA_DECODE_MASKLESS` (52/52 passed; GQA 4 q8_0 is "not supported").
- **Campaign** (`campaign.sh`; `logs/campaign.out`, every step EXIT 0). R = `../c18/cap2/wikiS_p4096.rows`, Q = `../c18/cap2/wikiS_p4096.qkv`.
  - `flock ~/gpu.lock ./bands_attn $R $Q 0,5,11 8 ship wiki4k 32`. This runs all 4096 queries × 24 heads on layers 0, 5 and 11, and every 8th query on the other 9 layers: 405 504 rows per variant.
  - `… ./bands_attn SYN:1.8:262144:{1,2} SYN all 1 ship+direct syn262k_s{1,2} 32`: 98 304 rows per variant, and adds Kq8_0X-Vq8_0X and Kq4_0X-Vq4_0X.
  - `… ./bands_dec $R $Q {512,1} ship dec{512,1}_wiki4k 32` and `… ./bands_dec SYN:1.8:262144:1 SYN {512,1} ship dec{512,1}_syn262k 32`. Variants: Kf16D-Vf16S, Kq8_0D-Vq8_0S, Kq4_0X-Vq4_0S, Kbf16X-Vbf16S, each type's D+D, PR-dec-f16 and PR-dec-q8_0.
  - `… ./bands_attn $R $Q 0,5,11 8 ship wiki4k_r2 32`, a repeat run.
  - CPU, without flock because they are CPU-only:
    - `./bands_cpu $R $Q 0,5,11 1 cpu_wiki4k 32 1 all` (all 25 pairs);
    - `./bands_cpu $R $Q all 8 cpu_wiki4k_s8 32 1 gpu`;
    - `./bands_cpu SYN:1.8:262144:1 SYN all 1 cpu_syn262k 32 1 all`;
    - and, for the decode-sized defect rows, `… all 4096 cpu_wiki4k_q0 …`, `SYN … all 4096 cpu_syn262k_q0 …` and `SYN … all 8 cpu_syn262k_s8 …`.
- **Parsers:** `src/analyze.py` → `analysis.md` (all tables); `src/compact.py` → `compact.md`; `src/compact_cpu.py` → `compact_cpu.md`; `src/kdecide.py` → `kdecide.md`. All are in `SP/final219/bands/`, raw outputs in `remote/res/`.

### Noise
- **Run to run: zero.** Every BAND / REFCMP / HASH line of `attn_wiki4k_r1.out` and `attn_wiki4k_r2.out` is identical (1213 lines; `diff` → `R1_R2_IDENTICAL_LINES`). The kernels are deterministic and the oracles are fixed, so the bands carry no sampling noise from repetition.
- **Seed to seed (SYN 262k s1 vs s2):** band 1.336e-9 vs 1.333e-9 (0.3 %); §3 maxrel p100 9.17e-4 vs 9.1e-4.
- **Layer to layer (wiki):** per-layer NMSE vs contract ranges over 2.5e-10 – 5.8e-10 (×2.3).
  - The p100 is taken over all 12 layers.
  - The bands are p100 over 0.4 M (prefill), 0.15 M (n_q 512) and 288 (n_q 1, 24 rows × 12 layers) rows.
  - The n_q 1 band therefore rests on the fewest samples. It is the cell with the smallest headroom (6.1).
- Defect metrics are also deterministic. Their spread over layers is the "min" in the tables: d2 on wiki goes from 7.6e-7 (512 rows) to 1.75e-4, and at n_q 1 from 1.38e-8 to 9.3e-6.

### Decision
**Index 1 (k).**
- **k = 2.** By the rule, the smallest k ≥ 2 such that all three defects exceed k × p100. It is met in every (shape, fixture) cell (`kdecide.md`). No defect is UNPROVEN for Q5.
- **This holds only with per-shape bands.** The thresholds to write, as NMSE of GPU vs the new CPU op (contract), are 2 × p100:

  | shape | wiki d4k | SYN 262k |
  |---|---|---|
  | prefill (G4) | 1.16e-9 | 2.67e-9 |
  | decode n_q 512 (G1, 1 split) | 1.03e-9 | 3.03e-9 |
  | decode n_q 1 (G1 split-K) | 4.53e-9 | 1.15e-7 |

  Source: `SP/final219/bands/kdecide.md`, `remote/res/*.out`.
- W3/W4/W5/Q5 replace "NMSE vs CPU ≤ 5e-7" with the row for their shape. The 5e-7 is dropped: it misses d2 at n_q 1 and gives d3 only a ×1.76 margin.
- The one-band form (5.73e-8 for everything) fails d2 at n_q 1 on text, so it is rejected.
- **R1-Q8 decision 5** ("Q5's rel-L2 ≤ 4e-4 catches V × (1+1e-3)") is **refuted as written**. The clean kernels already exceed 4e-4 against the §3 f64: rl2 p100 1.15e-3 and maxrel p100 2.75e-3 for q8_0 on wiki. So that gate would fail clean code.
- It is **closed in its replacement form**: GPU-vs-contract NMSE at k = 2 catches d3 in every cell, with headroom ≥ 15 (`kdecide.md`, min 8.73e-7 / 5.73e-8). d3's per-row rel-L2 is 0.96e-3 – 1.04e-3 at p50.

**Index 10 (W1, §3).**
- W1's "equals CPU FA + src[5] within f32 rounding" (PLAN-v3.md:422) is replaced by measured per-pair tolerances: NMSE(CPU op, CPU FA) ≤ 2 × p100 of (3).
  - V = f16 (any K): 3.4e-3;
  - K ∈ {q8_0, q4_0}, V ≠ f16: 2.6e-5;
  - K = bf16, V ≠ f16: 5.1e-6;
  - K ∈ {f32, f16}, V ∈ {f32, bf16, q8_0, q4_0}: 2.3e-7.
- That comparison is a coarse cross-check only.
  - For f16-V, quantized-K and bf16-K pairs (tolerances 5.1e-6 to 3.4e-3) it cannot see d2 or d3 (about 1e-6).
  - For K ∈ {f32, f16} with non-f16 V (2.3e-7), it sees d3 by only about 3.8× and misses d2 at n_q 1 (1.38e-8).
  - The correctness gate is the GPU-vs-contract band plus the contract-vs-f64 check.
- **W1 gains a contract-vs-f64 check** against the measured band, ×2: NMSE vs O_C18 ≤ 2.4e-9 (wiki) / 7.3e-8 (SYN); NMSE vs O_PLAN ≤ 1.0e-7 (wiki) / 3.1e-7 (SYN); max over the 25 pairs, `compact_cpu.md`.
- **§3 is rewritten with the exact oracle inputs** of O_CON / O_C18 / O_PLAN above. Two decisions are needed:
  - P is rounded against the running max of the online softmax, and l sums the unrounded p (the kernel rule, c18_kernel.inc:559-562).
  - The scale is applied in f32 before the f16 rounding of q. The kernels instead do f16(q)·f16(scale), which is identical only while scale is a power of two (1/16 here).
- **Every threshold cites its log path** (`SP/final219/bands/{kdecide,compact,compact_cpu}.md`, `remote/res/*.out`, `remote/logs/tbo_*.log`).

**Index 14.**
- **The bands are set on the PR/W0 kernels.** The PR-verbatim `qsa3_attn_kernel` V0/VD and `pr_dec` ran in the same processes and are bitwise equal to the shipped loaders: 264/264 comparisons, 0 words differ.
- On the ported QSA_*_MASKLESS shapes, only GPU vs CPU FA and GPU vs f64 §3 were measured (numbers above). These use random uniform data and the old CPU FA.
  - The GPU-vs-contract band on those shapes is **UNPROVEN** until W1's CPU op exists.
  - W1 then runs the same 78 shapes and records GPU-vs-CPU-op NMSE. They must fall under the matching table row, and an excess is a finding.
  - The GQA 4/8 path (f32, NMSE 1e-14 vs f64) is outside this band.
- **The 4e-4 vs-f64 figure becomes report-only**, per shape and type: table (2) and `compact.md`.
- **The generic and portable tiers stay UNPROVEN** until W5 runs them through `bands_attn` (or its port).
- **Q5 is restated:**
  - Attend vs the contract on the recaptured wiki d4k fixture, with the k = 2 prefill/decode rows above.
  - The recapture command is above. Its rows are byte-identical to the kept `cap/wikiS_p4096.rows`, so the fixture is reproducible from the kept rows plus one ~94 s capture.
  - Or on SYN 262k, with the PR-verbatim kernel in the same run as a bitwise reference.
  - The f64 numbers are reported, not gated.

**The four objections behind this lab:**
1. *"The op-level numeric thresholds are unsourced and attributed to C18, which did not measure them."* **Upheld.**
   - C18 measured 72 rows on one layer. Its 3.7e-4 rises to 4.4e-4 at p100 over 98k SYN rows (C18 oracle), and to 2.75e-3 under the §3 oracle.
   - 5e-7 had no measurement behind it. Against today's CPU FA it is too tight: the PR's own shapes sit 2-94× above it, and the wiki fixture up to 3400×. Against the contract it is too loose to catch d2 at n_q 1.
   - Both are now replaced by measured values, above.
2. *"D3 G1/G4 threshold (n_q < 128) was set against the OLD G1 kernel at shallow depth only."* **Not measured here.** Timing was excluded from this lab, and G1-vs-G4 is deferred to W4, per the index 0 defer entry. It stays UNPROVEN.
   - One fact from this lab bears on it: G1 split-K (n_q 1) has a 4× (text) to 40× (SYN) wider numeric band than the 1-split decode or G4.
   - So moving the G1/G4 boundary also moves which band row applies.
3. *"W1's 'equals CPU FA+src[5] within f32 rounding' is false, and the op-test thresholds are unsourced or thin."* **Upheld, by measurement and by code.**
   - For all 25 pairs, CPU FA differs from the contract by more than f32 rounding. The smallest gap is 1.2e-9 (K f16, V f32/bf16, the contract's P rounding). The largest are 1.7e-3 (f16 V), 1.3e-5 (quantized K) and 2.6e-6 (bf16 K).
   - The causes are ops.cpp:9119 (the f16 VKQ accumulator) and :9014/:9058 (Q converted to the K type's vec_dot_type).
   - The thresholds are now measured, and the per-row maxrel form is flagged as thin (max k 2.7).
4. *"Kernel correctness bands are unsourced, and the fixtures they need were deleted."* **Resolved.**
   - wiki d4k was recaptured with C18's command. Its rows are byte-identical (`cmp`), and the bands were measured on it and on SYN 262k.
   - The .qkv was deleted again after the runs. The capture is deterministic and takes about 94 s.

### Large files
- Deleted: `~/dev/wt_lab/c18/cap2/wikiS_p4096.qkv` (1.4 GB) and `cap2/wikiS_p4096.rows` (403 MB, a duplicate of the kept one).
- Kept: `~/dev/wt_lab/bands` (399 MB with the step-3 tree build). The 4 C18 row files are untouched.

## R1-M-Lab C, shrunk: code arithmetic now, plus

**Question.** Part 1, by code (PR head 94e086847, `~/dev/wt_pr219v2`): the G1 partial size and the f16 V-pack delta. Part 2, on the GPU:
- (Index 12) what does the PR219 default server (no `--parallel`) reserve?
- (Index 16) on that server, does a ~4096-token text ubatch in a second slot, while another slot holds a deep image, reach `strip_views` for both srcs? What are the share-vs-no-share delta and the runtime peak?

**Short answer.**
- **G1:** the formula is confirmed, and '<= 8 MiB' is false. The proposed maximum (9.0 MiB at n_q 127) is too low, because G1 also takes 128 <= n_q <= 512. The real maximum is **12.42 MiB at n_q 263**.
- **Default server:** it is 4 slots, unified (log). Reserve at c131072 ub4096 q8_0: ROCm0 1609.88 MiB main and 1465.23 MiB MTP; ROCm_Host 202.06 and 201.75 MiB.
- **Layout as proposed (text-only second slot): null.** Slot B's text ubatches are compact, with **0 bias views** in 5/5 runs.
  - Sent one after the other, the layout never forms at all: the default `--cache-idle-slots` saves slot A to host RAM and clears it from the KV cache first.
- **Images in two slots** (re-run as the DECIDES rule requires): **census > 0 for both srcs**.
  - Share vs no-share: **+9014.36 MiB** ROCm0 main runtime peak (3546.90 vs 12561.26), 2/2 runs identical to the byte.
  - With sharing, the default server still runs past its own reserve: **+1937.02 MiB ROCm0 main, +803.36 MiB ROCm0 MTP, +837.21 MiB ROCm_Host in each context**, at n_kv 71680.

### Part 1: code arithmetic (no GPU)

**G1 partials** (`ggml/src/ggml-cuda/qsa-decode.cu`):
- Inputs: `ns = ids->ne[0]`, `nq = q->ne[1]`, `nh = q->ne[2]`, `step = wmma ? 64 : 128` (:134). The chunk and split rule is at :137-142. The allocation is `ggml_cuda_pool_alloc<float> partial(..., nq*nh*splits*258)` (:145), so the size is nq·nh·splits·258·4 B.
- ns: `width = min(n_kv, indexer_top_k + r - 1)` (src/models/qwen4exp.cpp:1672), with top_k 2048 and r 4 (log `indexer.top_k = 2048`, `compress_ratios … 4`), so ns ≤ 2051.
- nh = 24 (log `head_count = 24`). With `n_head_kv = 2` the wmma condition `q->ne[2] == 12*k->ne[2]` holds, so step = 64.

Splits and sizes (python, the code's integer rule; max over ns ≤ 2051 is at ns = 2051):

| n_q | 8 | 9 | 64 | 65 | 66 | 127 | 128 | 255 | **263** | 264 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| splits | 33 | 26 | 5 | 5 | 4 | 3 | 3 | 2 | 2 | 1 | 1 |
| MiB | 6.24 | 5.53 | 7.56 | 7.68 | 6.24 | 9.00 | 9.07 | 12.05 | **12.42** | 6.24 | 12.09 |

- The claimed numbers are confirmed: 9,436,608 B at n_q 127, 7.68 MiB at 65, 6.24 MiB at 66. The size is non-monotone; it drops at 8→9, 65→66, 263→264 and elsewhere. So '<= 8 MiB' (PLAN-v3.md:291) is **false**.
- **Correction to the proposal: G1 is not limited to n_q < 128.**
  - `qsa_decode_supported` accepts `q->ne[1] <= QSA_DECODE_MAX_QUERIES` = **512** (qsa-decode.cuh:5, qsa-decode.cu:94).
  - It is tried **before** prefill (fattn.cu:1154-1157). Prefill takes `q->ne[1] >= 128` (qsa-prefill.cu:616).
  - So every ubatch with 128 ≤ n_q ≤ 512 on one stream runs G1: ub512, and the tail ubatch of any prompt. The strips are concatenated before attention (`qwen4exp_finish_strips`, qwen4exp.cpp:1225-1233, called at :1681), so attention sees the whole n_tps.
  - Max over n_q' ≤ 127 is 9.00 MiB (n_q 127). Max over n_q' ≤ 512 is **13,027,968 B = 12.42 MiB (n_q 263)**.
- **Per n_stream:** G1 requires `q->ne[3] == 1` (qsa-decode.cu:96), so the bound is per one-stream op. A split-stream op (ne[3] > 1) never takes G1. It goes to prefill, which needs ≥ 128 queries (qsa-prefill.cu:616-626), or to the masked path. So G1 contributes 0 for n_stream > 1.

**f16 V pack** (qsa-prefill.cu):
- `qsa_direct = kv_direct_ok && (cfg==1 || (cfg<0 && n_q*64 < nk))` (:685). `packed_v(ctx.pool(), qsa_direct ? 0 : ggml_nelements(v))` uint16 (:688). The proposal cites :684/:686; the lines are :685/:688 at 94e086847.
- V per cell is 256 × n_head_kv 2 = 512 elements × 2 B = **1024 B/cell** per stream. `nelements` includes ne[3], so the size is × n_stream.
- `qsa_kdirect` is true for f16/f16 and q8_0/q8_0 at cfg -1, so K is never packed there (:686-687).
- Pack thresholds: ub4096 packs while n_kv ≤ 262144 cells (≤ 256 MiB); ub512 packs while n_kv ≤ 32768 cells (≤ 32 MiB).

Always-packed vs PR (capped):

| config | always-packed at n_kv = n_ctx | PR at the same n_kv | Δ vs PR instantaneous | PR pool high-water if the prompt crossed the threshold | Δ vs high-water |
|---|---|---|---|---|---|
| ub4096 c524288 | 512 MiB | 0 (direct) | **+512 MiB** | 256 MiB | +256 MiB |
| ub4096 c1048576 | 1024 MiB | 0 | **+1024 MiB** | 256 MiB | +768 MiB |
| ub512 c131072 | 128 MiB | 0 | **+128 MiB** | 32 MiB | +96 MiB |

- **Why the high-water column exists.** HIP builds use the legacy pool: `GGML_HIP_NO_VMM` defaults ON (ggml/CMakeLists.txt:222), and the log shows `VMM: no`, `NO_VMM = 1`. `free()` keeps the buffer in `buffer_pool` and calls `cudaFree` only when all 256 slots are full (ggml-cuda.cu:586-598). So a PR server whose prompt grew through n_kv ≈ 262143 keeps the 256 MiB pack buffer.
- The high-water column is arithmetic. The real pool high-water also depends on best-fit reuse of other allocations (ggml-cuda.cu:503-540): UNPROVEN, not measured. It was dropped from this run as (c).
- The C18 cost of the cap at 1M is +4.0 %: PR-VD-f16 65.91 vs packed 63.38 ms (C18).
- Per the binding OWNER DECISION 2026-11-05, this section reports the numbers. It does not choose between cap and always-pack.

### Part 2 setup (strix-002, 2026-11-05 14:00-14:27 CET; I was the only agent; every server run under `flock ~/gpu.lock`)
- **Binary:** `~/dev/wt_lab/c19/tree/build/bin/llama-server`. This is the C19 lab tree: PR 94e086847 + `LAB_C19_NO_SHARE` / `LAB_C19_LOG` (`~/dev/wt_lab/c19/c19.patch`). With no env var set it is the PR, with buffers identical to the pristine binary (C19). It was not rebuilt.
- **Scripts:** `~/dev/wt_lab/r1c/` (copies in `SP/final219/r1c/`).
- **Common flags** are those of `c19_srv.sh` / `~/repro_img/repro_kv.sh` **minus `--parallel`**: MTP sidecar `mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, `-dev ROCm0 --spec-draft-device ROCm0 --threads 8 --load-mode none --lazy-mode on-direct --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-ngl all -ngl 99 --n-gpu-layers-draft 99 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.05 --presence-penalty 0.0 --repeat-penalty 1.0 --reasoning-preserve --jinja -fa on -fit off`.
- **(a)** `runA.sh` → `srv.sh text a_c<C>_ub<U>_<KV> <KV> <C> <U> ""`:
  - text model `~/models/qwen3.8-flash-next-UD-Q4_K_XL/*00001-of*.gguf`, `-lv 4`, `-b/-ub U`, `-ctk/-ctv KV`, init only;
  - C/U ∈ {131072/4096, 261888/8192}, KV ∈ {f16, q8_0}.
- **(b)** Signal-3.8 Q4_K_XL + `-mm mmproj-Signal-3.8-Flash-Next-f16.gguf --mmproj-device ROCm0 --image-min-tokens 1024 -v`, `-c 131072 -b 4096 -ub 4096 -ctk/-ctv q8_0`. Arms: `LAB_C19_LOG=1` (share) and `LAB_C19_LOG=1 LAB_C19_NO_SHARE=1` (no-share). Three layouts:
  1. `runB.sh` / `srv.sh` (as proposed, sequential): `A_B65t_cp.json` (c19 B65t + `cache_prompt:true`), then `T12k_cp.json`. T12k is real text lines 40000..40479, 12027 prompt tokens, answer 40479 (`mkreq.py`). One run per arm.
  2. `runB2.sh` / `srv2.sh` (concurrent, text-only slot B): `A_B65t_long.json` (B65t + `ignore_eos`, `max_tokens 1200`, so slot A keeps generating). T12k is sent once the log shows `task 0 | generate_draft` (slot A is in generation, with its image cells at 57504..58527 in the cache). Two runs per arm.
  3. `runB3.sh` / `srv3.sh … TI12k_cp.json` (concurrent, **images in two slots**): as layout 2, but slot B's request is `TI12k_cp.json` (`mkreq2.py`): the same image (`~/repro_img/img.png`, 1024 tokens) first, then the same 12k text and question, 13053 prompt tokens. Two runs per arm.
- **Census** (`parse2.py`). One "graph" is a consecutive run of `C19 sv` lines with one (context, n_kv); il 48 is the MTP context. Per graph, it counts new=1 and reuse calls by src, plus the bytes of the new views.
  - The bias src is the unnamed `inp->bias` input. It logs as `src=` (empty) in the first QSA layer and as `leaf_121` afterwards. My first parser (`parse.py`) dropped the empty-name lines; `parse2.py` fixes that.
  - Runtime peak = the max `C19 buf` per context and backend. `CPU` in these lines is the ROCm_Host buffer: 202.06 = `sched_reserve: ROCm_Host` 202.06.
- **Logs:** `~/dev/wt_lab/r1c/logs/srv_{a_*,b_*,b2_*,b3_*}.log` (21 MB). Full parse: `parse_all.out`.

### (a) PR219 default-server reserve (MiB, `sched_reserve`; main context, then MTP; one run each, buffer noise 0.00 per C19)
Every run logs `n_parallel is set to auto, using n_parallel = 4 and kv_unified = true` and `load_model: initializing, n_slots = 4, n_ctx_slot = <C>, kv_unified = 'true'` (both contexts `n_seq_max = 4`, `kv_unified = true`).

| -c / -b,-ub | KV | ROCm0 main | ROCm_Host main | ROCm0 MTP | ROCm_Host MTP | KV buffers main (ROCm0) | KV buffers MTP |
|---|---|---|---|---|---|---|---|
| 131072 / 4096 (M1) | f16 | 1609.61 | 201.80 | 1465.23 | 201.75 | 3072.00 + 384.00 | 256.00 + 32.00 |
| 131072 / 4096 (M1) | q8_0 | 1609.88 | 202.06 | 1465.23 | 201.75 | 1632.00 + 204.01 | 256.00 + 32.00 |
| 261888 / 8192 (owner) | f16 | 3315.03 | 403.59 | 2938.45 | 403.50 | 6138.00 + 767.25 | 511.50 + 63.94 |
| 261888 / 8192 (owner) | q8_0 | 3315.30 | 403.86 | 2938.45 | 403.50 | 3260.81 + 407.61 | 511.50 + 63.94 |

- (b) re-reserves identical values on the Signal model: 1609.88 / 202.06 and 1465.23 / 201.75, in 10/10 runs.
- After `load_model`, the server re-reserves the MTP context, then the main one, with the same sizes. It also logs `reserve_compute_meta: ROCm0 248.10 MiB` (mmproj, Signal only).
- Side observation: the MTP KV is the same size in the f16 and q8_0 runs (256.00 MiB at c131072). The MTP context does not follow `-ctk/-ctv` here. The cause is UNPROVEN (not investigated).

### (b) Raw numbers (MiB)
**Answers:** correct in all 10 runs. A: `'12299'` (layout 1), `'12299. the quick brown fox…'` (layouts 2-3, ignore_eos). T12k and TI12k: `'40479'` (`runB*.out`, `logs/res*_*.txt`).

**Slot assignment** (log): A → `id 3` (LRU), B → `id 2`, in every run.

**Layout 1, sequential (null by construction).** Before slot B starts, the log shows:
- `slot process_sing: id 3 | saving idle slot to prompt cache`;
- `prompt_save: saving prompt with length 58558, total state size = 1066.185 MiB`;
- `slot prompt_clear: id 3 | clearing prompt with 58558 tokens` (`srv_b_share.log`).

The code path is `cache_idle_slots` (default enabled with `--cache-ram` 8192, common/arg.cpp:1761, common/common.h:711, tools/server/server-context.cpp:3080-3083). Slot B then starts on an empty cache: n_kv 4096, 7936, 12032. Its 3 text ubatches have bias new = 0 and kq_mask views only, and no runtime growth in either arm: max = reserve, 1609.88 / 1465.23.

**Layout 2, concurrent, text-only slot B** (`srv_b2_{share,noshare}_r{1,2}.log`; r1 = r2 for every number below):

| graph (main ctx) | n_kv | n_tps / strips | share: new / reuse | no-share: new |
|---|---|---|---|---|
| slot A text ubatches (×14) | 4096..57600 | 4096/8, 1209/3, 3047/6 | kq_mask 8 / 88, **bias 0** | kq_mask 96, bias 0 |
| slot A image ubatch | 58624 | 1024 / 2 | bias 2 / 22, kq_mask 2 / 22 | bias 24 (687.0 MiB), kq_mask 24 (1374.0 MiB) |
| **slot B text ubatches** | 62720, 66560, 70656 | 4088 / 8, 3835 / 8, 4088 / 8 | kq_mask 8 / 88, **bias 0** | kq_mask 96, **bias 0** |

- Runtime peak ROCm0 main: share 1609.88 (= reserve), no-share 2687.62. The growth is reached at the slot-A image ubatch, and slot B adds nothing in either arm.
- MTP ROCm0: 1465.23 in both arms. ROCm_Host: 222.74 main and 222.46 MTP in both arms (+20.68 / +20.71, at the image ubatch).
- **Why slot B is compact (code):** `qsa_scalar_visibility_cells` skips every cell whose sequence set does not meet the ubatch's `rows` *before* the 2-D test (src/prefix.h:39-40). A text-only sequence therefore collects no `pos_2d`. The other-sequence check (prefix.h:43-46) runs only when `pos_2d` is non-empty.
- So **an image in another slot never makes a text-only slot non-compact.**
- The 4 leftover tokens (n_tps 4088 vs the server's `n_tokens = 4092` batch share) went to a ubatch with < 128 queries, a single strip, which makes no `strip_view` call. Which ubatch they went into is UNPROVEN.
- A slot-B ubatch of a full 4096 cannot form while slot A is decoding: the 4096-token batch also carries slot A's 1+3 draft tokens. The 4088-token ubatches have the same 8-strip structure (7 × 512 + 504) as a 4096 one.

**Layout 3, concurrent, images in two slots** (`srv_b3_{share,noshare}_r{1,2}.log`; r1 = r2 for every buffer number):

| graph | ctx | n_kv | n_tps / strips | share: new / reuse (new MiB) | no-share: new (MiB) |
|---|---|---|---|---|---|
| slot A text (×14) | main | ≤ 57600 | 4096/8 … | kq_mask 8 / 88, **bias 0** | kq_mask 96, bias 0 |
| slot A image | main | 58624 | 1024 / 2 | bias 2 / 22 (57.25), kq_mask 2 / 22 (114.5) | bias 24 (687.0), kq_mask 24 (1374.0) |
| slot B image | main | 59648 | 1024 / 2 | bias 2 / 22 (58.25), kq_mask 2 / 22 (116.5) | bias 24 (699.0), kq_mask 24 (1398.0) |
| **slot B text #1** | main | 63744 | 4088 / 8 | **bias 8 / 88 (248.51), kq_mask 8 / 88 (497.03)** | bias 96 (2982.16), kq_mask 96 (5964.33) |
| **slot B text #2** | main | 67584 | 3833 / 8 | bias 8 / 88 (247.05), kq_mask 8 / 88 (494.10) | bias 96 (2964.59), kq_mask 96 (5929.17) |
| **slot B text #3** | main | 71680 | 4088 / 8 | bias 8 / 88 (279.45), kq_mask 8 / 88 (558.91) | bias 96 (3353.44), kq_mask 96 (6706.88) |
| slot B text #1-3 | MTP (il 48) | same | same | bias 8, kq_mask 8 (one QSA layer: nothing to share) | identical |

Runtime `C19 buf` (MiB):

| backend | ctx | reserve | share peak | no-share peak | share − reserve | no-share − share |
|---|---|---|---|---|---|---|
| ROCm0 | main | 1609.88 | **3546.90** (3268.35 at n_kv 63744) | **12561.26** (11385.47 at 63744) | **+1937.02** | **+9014.36** |
| ROCm0 | MTP | 1465.23 | 2268.59 (2113.80 at 63744) | 2268.59 | +803.36 | 0.00 |
| ROCm_Host | main | 202.06 | 1039.27 (946.36 at 63744) | 1039.27 | +837.21 | 0.00 |
| ROCm_Host | MTP | 201.75 | 1038.96 (946.05 at 63744) | 1038.96 | +837.21 | 0.00 |

- **Host growth fits the proportional form exactly.** 3 B × n_kv × n_tps = 3 × 71680 × 4088 = 838.36 MiB, against the measured 837.21 (Δ 0.14 %). At n_kv 63744 the formula gives 745.54 MiB against the measured 744.30 (946.36 − 202.06).
- That is the f16 KQ mask (2 B/cell) plus the f32 per-block bias (4 B / ratio 4 = 1 B/cell) of a non-compact ubatch, in each context.
- **Per-view bytes (logged `bytes`):**
  - bias view = n_kv × 512 B per 512-query strip: 63744 × 512 = 31.13 MiB against 248.51 / 8 = 31.06 (the last strip has 504 queries);
  - kq_mask view = n_kv × 1024 B.
  - At n_ctx: 64 + 128 MiB per strip at 131072, and 127.9 + 255.75 MiB at 261888.

**Noise.**
- Buffers and census are deterministic: r1 = r2 to the 0.01 MiB in every layout and arm, consistent with C19's 0.00 MiB.
- Prompt t/s (server `prompt_per_second`, logging on, not a perf claim):
  - A: 1009.5 / 1010.7 (layout 1), 1005.6 / 1019.6 / 1023.1 / 1015.9 (layout 2), 997.7 / 1016.3 / 1007.5 / 1014.8 (layout 3);
  - slot B in layout 3: share 636.3 / 679.0, no-share 628.9 / 598.1. That spread is about ±6 %, so no share/no-share perf difference is claimed.

### Answers to the four objections
1. **"D17's reserve-covers-every-route is unproven where G and topology switch on n_q; G1 size unsourced."**
   - G1 is now sourced (qsa-decode.cu:134-145), and it is non-monotone in n_q.
   - The routes switch on n_q in three places:
     - G1 for n_q ≤ 512 on one stream, tried first (fattn.cu:1154);
     - prefill for n_q ≥ 128 otherwise (qsa-prefill.cu:616);
     - the V pack for n_kv ≤ 64·n_q (qsa-prefill.cu:685).
   - So a reserve sized at n_q = n_ubatch does not bound G1 or the pack. Both need the max over n_q' ≤ n_ubatch: G1 12.42 MiB at n_q 263, the pack at n_kv < 64·n_q'.
2. **"Every default llama-server gets the non-compact reserve variant, ungated."**
   - Measured: the default server is 4 slots, unified (log).
   - Its reserve does not cover the non-compact route. Images in two concurrent slots overrun it by +1937 MiB ROCm0 main, +803 MiB ROCm0 MTP and +837 MiB host per context at n_kv 71680, ub4096, with sharing on.
   - A text-only second slot is compact (prefix.h:39-46). Sequential use is protected by `cache_idle_slots` (it clears idle slots from the KV cache).
   - So the non-compact variant is reachable on the default server only with ≥ 2 image-bearing slots in the cache at once. The gate must be measured with layout 3, not with layout 1.
3. **"The strip_views 'not lost' proof can pass vacuously."**
   - Confirmed for the proposed layout: slot B's bias census is 0 and the no-share arm adds 0 MiB there. A layout-1 test would pass with strip_views deleted.
   - Layout 3 is not vacuous: bias and kq_mask are both new > 0 on slot-B text ubatches, and the no-share delta is +9014.36 MiB.
4. **"Memory model contradicts the plan's table; f16 regression at depth not gated."**
   - Corrected arithmetic, against the PR:
     - always-pack is +512 / +1024 MiB at c524288 / c1M ub4096 at n_kv = n_ctx;
     - it is +256 / +768 MiB against the PR's legacy-pool high-water (a 256 MiB pack retained);
     - at ub512 it is +128 MiB already at c131072.
   - The f16 depth gate is the one in the OWNER DECISION section (capped default; reserve ≤ PR reserve + PR pool high-water at c524288 / c1M). The pool high-water itself is still unmeasured: it was dropped as (c) and is to be measured at acceptance.

### Decision
- **Index 5.**
  - §7 replaces '<= 8 MiB' with **G1 partial = n_q·n_h·splits(n_q, ns)·258·4 B, one stream only** (qsa-decode.cu:96, :145).
  - Its maximum is stated over **n_q' ≤ min(n_q, 512)**, not ≤ 127: **12.42 MiB (13,027,968 B) at n_q 263, ns 2051, nh 24**. For ubatches < 263 the maximum is lower (9.00 MiB for n_q ≤ 127).
  - `get_alloc_size` takes the max over every G and every n_q' ≤ n_q.
  - The ub < 128 strict-reserve recipes go into W9, run deferred. No measurement here.
- **Index 17.**
  - The §7 sentence takes the Δ table above (instantaneous and high-water columns).
  - The owner's choice is unchanged and deferred per the binding OWNER DECISION. Options: cap at 64·n_q, at a cost of +4.0 % at 1M (C18 65.91 vs 63.38 ms); or always pack, at +512 / +1024 MiB at c524288 / c1M.
  - The pack and G1 bounds are stated per n_stream: the pack is × n_stream (nelements includes ne[3]); G1 is n_stream = 1 only.
- **Index 12.**
  - U18 is closed (code + log: 4 slots, `kv_unified = 'true'`).
  - **The PR219 default-server M-gate numbers for the PR description** are table (a) and the layout-3 rows above:
    - reserve at c131072 ub4096 q8_0: 1609.88 / 202.06 main, 1465.23 / 201.75 MTP;
    - reserve at c261888 ub8192 q8_0: 3315.30 / 403.86 main, 2938.45 / 403.50 MTP;
    - layout-3 runtime peak with sharing: 3546.90 / 1039.27 main, 2268.59 / 1038.96 MTP.
  - **The proportional host form, applied literally.**
    - 3 B × 131072 × 4096 = **1536 MiB**. PR reserve + runtime peak, both measured, is **1039.27 MiB** host. **1536 > 1039.27, so the proportional form is rejected by the rule as written.**
    - At the owner config it is 3 × 261888 × 8192 B = 6.0 GiB per context.
  - **Caveat (sourced).** The proportional form is not slack.
    - Measured host growth equals 3 B × n_kv × n_tps within 0.14 %. The rule's runtime peak was taken at n_kv 71680, not at n_ctx.
    - At n_kv = n_ctx the PR itself would need 202.06 + 1536 = 1738 MiB host per context. That is a projection from the formula fit: UNPROVEN at that depth.
  - **The bound therefore becomes a fixed strip set, as the rule prescribes.**
    - Per 512-query strip at n_ctx: bias n_ctx × 512 B + mask n_ctx × 1024 B, i.e. 192 MiB at c131072 and 383.6 MiB at c261888.
    - Plus the per-strip device copies, measured with sharing at n_kv 71680: (3546.90 − 1609.88) / 8 strips = 242.1 MiB per strip, main.
    - The strip count S is fixed and independent of n_ubatch. W9 then has to split non-compact ubatches longer than S × 512 tokens.
    - With S = 8 the bound equals the ub4096 proportional value. The owner's ub8192 then costs half of 6.0 GiB host.
- **Index 16.**
  - Layout 1 (the proposed one) gives a bias count of **0** in 5/5 runs. As DECIDES requires, the layout was changed to images in two slots and re-run (2 runs per arm, identical). It gives **census > 0 for both srcs**.
  - **Layout 3 becomes the W8/W9 regression test** (`~/dev/wt_lab/r1c/runB3.sh` recipe): default server, c131072 ub4096 q8_0, slot A = B65t generating, slot B = TI12k.
    - **Pass:** the main ROCm0 runtime peak stays below the share peak + half the measured share-vs-no-share delta: 3546.90 + 4507.18 = **8054.08 MiB**.
    - **Also required, so the test cannot pass vacuously:** census on slot-B text ubatches with bias new > 0 **and** kq_mask new > 0, and bias + kq_mask reuse = 88 per graph.
    - The C19 fallback (forced masked route) is not needed: the natural selected-key route reaches strip_views in layout 3.
  - **Labels for §12 and §4.3:**
    - "+1120 MiB" and "0.148 → 0.029 MiB/cell" are PR-tree measurements, C19: `-np 1`, c65536, HIP.
    - The default-server equivalent here: +1077.74 MiB for one deep image (layout 2 no-share, 1609.88 → 2687.62), and +9014.36 MiB for two image slots (layout 3), both PR-tree measurements.
- **Status:** resolved. Part 1 by code. Index 12 and 16 by measurement: 10 server runs, deterministic, all answers correct.

### Large files
- None created (> 1 GB). The logs are 21 MB in `~/dev/wt_lab/r1c/logs`, kept. The C19 logs (85 MB) were not touched.
- No processes were left running (`pgrep -a llama-server` empty after the runs).

## R2-M-LAB-RES (shrunk): re-split mechanism by 

**Question.** (a) By code: where does a conservative pre-placement test of a non-compact ubatch hook in, can `split_equal` be re-run at a smaller n_ubatch while keeping `n_keep_tail` and multi-sequence ubatches, and which reserve shape and flag force the non-compact scorer at reserve time? (b) By measurement on strix-002, per backend (ROCm0, ROCm_Host) and per context (main, MTP): what does the forced variant (n_tps = min(ub, 1024) per stream, n_kv = full, n_seqs = n_seq_max) add over the unforced reserve, as a byte formula in n_ctx and n_stream? From the same `-fit on` logs: does the fit-probe estimate miss the variant, and does the probe build the MTP context? Decides Index 0 (mechanism and location), Index 7 (§7.1.4 variant spec, M5 formula, ceiling), Index 6 (probe fix).

**Short answer.**
- **Index 0, by code.** The hook goes into `llama_kv_cache::prepare` (src/llama-kv-cache.cpp:816-879). It runs right after `apply_ubatch(sinfo_new, ubatch)` (:856) and before the restore loop (:861). The re-split is a deterministic replay inside `llama_memory_hybrid_idx::init_batch` (src/llama-memory-hybrid-idx.cpp:168-231). It does not need to go one level up into `llama_context::decode`.
  - `split_equal` keeps `n_keep_tail` for any n_ubatch' > n_keep_tail (src/llama-batch.cpp:764-823).
  - **The re-split unit must be total tokens, not tokens per stream.** The keep-tail rule can emit one sequence alone, extended up to n_ubatch' tokens (:787-799). So a cap of "S × 512 per stream × n_seqs" can put more than S × 512 tokens on one stream.
  - The unit is therefore **n_ubatch' = min(n_ubatch, S × 512) in total**, which also bounds n_tps per stream.
- **Index 7, measured.** The variant spec as posed (n_tps = 1024 *per stream*, so T = 2048 tokens at `-np 2 -ub 1024`) reserves a graph larger than any real ubatch: **+1338.69 MiB ROCm0** in the main context, against +288.50 MiB for the same variant with T = min(ub, 1024) in total.
  - The ROCm_Host part of the variant follows one exact byte formula on 11 configurations (0 B residual): **V_host = 3·K·T + 12·K·s + 51244·T + 32 B** (MTP context; main is ≤ 0.29 MiB higher).
  - The pre-registered S = 2 device ceiling **is exceeded** on `-c 131072 -ub 1024` (default server): +940.54 MiB main ROCm0 against a ceiling of 886.25. The rule's fallback is **S = 1**. At S = 1 every measured configuration is under the ceiling.
- **Index 6, measured.** The fit probe misses the variant on every forced configuration (gaps up to −2539.06 MiB ROCm0, −818.79 MiB host).
  - The probe does build the MTP context.
  - Unforced, the probe equals the real reserve to the byte.
  - A per-backend max over the reserve_size results reproduces the forced real reserve **to the byte in 64/64 cases**. So the fix is adopted.

### Step 0: code (PR head 94e086847, `~/dev/wt_pr219v2`, read-only)

**Hook point.**
- `llama_kv_cache::prepare` (src/llama-kv-cache.cpp:816-879) loops over the ubatches. For each one it calls `find_slot(ubatch, false)` (:833), saves the cells, calls `apply_ubatch(sinfo_new, ubatch)` (:856), and after the loop restores everything in reverse order (:861-872).
- So right after :856 the attention cells hold exactly the state that `llama_kv_cache_context::apply` will build at graph time. `apply` calls `kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur])` with the same sinfo (:3064), and `n_kv = kv->get_n_kv(sinfos[i_cur])` (:3065).
- `llama_memory_hybrid_idx::init_batch` (src/llama-memory-hybrid-idx.cpp:168-231) works in three steps:
  - it forms **all** ubatches first (`split_equal`, or `split_seq` when embd_all; :175-200);
  - it then prepares the recurrent cache (:207);
  - it then prepares the attention cache (:214), and copies `heads_idx = heads_attn` (:223). The indexer cache is not prepared separately.
- **Constraint found:** the graph-time predicate `qsa_scalar_visibility` reads the **indexer** cells, `mem->get_mem_idx()->get_cells(seq)` with `get_idx()->get_n_kv()` (src/llama-memory-hybrid-idx.cpp:1731, :1738). The init-time test can only see the simulated **attention** cells. The design therefore rests on the invariant "attention cells ≡ indexer cells after every apply" (same slots :223; edits forwarded to both, e.g. seq_rm :281-300). That invariant is asserted by the implication test below. As a general claim it is UNPROVEN.
- **Not in `decode`.** `llama_context::decode` calls `memory->init_batch(*balloc, cparams.n_ubatch, embd_all)` once per batch (src/llama-context.cpp:2267). It only sees the finished context; the placement exists only inside `prepare`. A decode-level re-split could only re-run the whole batch at the smaller n_ubatch. That shrinks compact ubatches too, and nothing is gained, because `split_equal` already keeps n_keep_tail (below).

**`split_equal` and n_keep_tail** (src/llama-batch.cpp:665-834):
- **Where n_keep_tail comes from.** `n_keep_tail = n_rs_seq + 1` when n_rs_seq > 0 (hybrid-idx.cpp:186-191). n_rs_seq is **3 in the main context and 0 in the MTP context**: `common/common.cpp:1716` gives `need_n_rs_seq()` = draft.n_max = 3 (common/common.h:461-475), and `cparams_dft.n_rs_seq = 0` (common/common.cpp:1261). Logs: `llama_context: n_rs_seq = 3` and then `= 0` (`logs/srv_B_u.log:2627, :6946`).
- **R1-RES's open point, now explained by code.** R1-RES left UNPROVEN why the main context split 513+511 into two one-stream ubatches while MTP formed one 511+511 two-stream ubatch.
  - With 511 tokens in seq 1, the expand loop stops at 511 per sequence (can_expand, :731-741).
  - seq 0 then has 2 < 4 tokens left, so `n_keep = 0` (:771-783). seq 0 is extended to completion (511 + 2 = 513 ≤ 1024, :787-799) and emitted alone. seq 1's 511 tokens are returned to the pool (:815-820).
  - MTP has n_keep_tail 0 and skips that block.
- **Re-run at a smaller n_ubatch'.** `split_reset()` clears `used` (:622-628), and `split_equal` is a deterministic function of `used`. So replaying the same calls reproduces the same ubatches.
  - Size bound: the expand loop stops at `(size + 1)·n_seqs > n_ubatch` (:755), and the keep-tail extension only runs when `idxs.size() + n_remaining(0) <= n_ubatch` (:788). **So every ubatch has ≤ n_ubatch' tokens.**
  - The keep-tail logic works for any `n_ubatch > n_keep_tail` (assert :764). The truncation branch (:800-810) pops at most n_keep_tail tokens and never empties the ubatch.
  - With n_ubatch' = min(n_ubatch, S × 512) and n_rs_seq = 3, the assert holds (512 > 4). An init-time check `n_rs_seq + 1 < S × 512` replaces the abort for large `--spec-draft-n-max`.
- **Multi-sequence ubatches.**
  - The choice of sequence sets (:672-701) does not depend on n_ubatch, except for the `> n_ubatch` break (:698), which needs more than 512 sequences.
  - Split caches use `sequential = !unified` (:189-191), which takes consecutive seq ids (:692-695), so stream count = n_seqs.
  - **A per-stream cap fails:** passing n_ubatch' = S·512·n_seqs lets the n_keep = 0 branch extend one sequence to S·512·n_seqs tokens on one stream (:787-799). The cap must be on total tokens.
- **2-D positions.** `split_equal` never reads positions; `ubatch_add` copies every n_pos axis per token. A re-split cuts an image's token run at a 512 boundary. The default server already makes the same cut: ub 512 (`n_ubatch = 512`, `logs/srv_A_u.log:2621`) with `--image-min-tokens 1024` splits every image. Image (embd) ubatches are always non-compact, because `ubatch.token` is required both in `qwen4exp_use_block_selection` (qwen4exp.cpp:1110) and in `qsa_scalar_visibility` (hybrid-idx.cpp:1721). So image ubatches longer than S × 512 in multi-sequence contexts are re-split too.
- **seq_cp-shared blocks (R1-CB layout).** A re-split changes neither the cells nor their sequence sets. The rows of a sub-ubatch are a subset of the parent's rows. The shared-block test (src/prefix.h:32, :52-62) can then only accept more, never less. Only the size bound matters, and it holds by construction.

**Mechanism (Index 0, written now).**
1. **Init-time conservative predicate** `NC0(u)`. It returns true when all of the following hold:
   - `n_kv(sinfo) > indexer_top_k + r − 1`;
   - and either `u.token == nullptr`, or `!(n_swa == 0 && (n_stream_u == 1 || n_tps_u >= 128) && qsa_scalar_visibility_cells(cells_sim, n_kv, r, u))`.

   It is evaluated on the simulated attention cells after kv-cache.cpp:856. Compared with qwen4exp.cpp:1453, it drops the `blocks` term and the `qsa_prefix_matches` shortcut. Both only turn non-compact into compact, so NC0 is a superset of the graph-time non-compact set. Exception: `blocks` is also false at n_kv > QSA_MASKLESS_MAX_KV = 1048576 (qwen4exp.cpp:1102, :1112). Beyond 1M every ubatch is non-compact, so NC0 must also be true there.
2. **Hook.** `prepare(ubatches, hook)` takes an optional callback `bool(i, ubatch, sinfo, cells)`. It returns false iff `NC0(u) && u.n_tokens > U`, with U = min(n_ubatch, S × 512). prepare then records `i_fail`, stops, and restores the cells as today (:861-872).
3. **Replay in init_batch.**
   - Keep a per-ubatch cap vector `cap`. Loop: `split_reset`; form the ubatches with `split_equal(k < cap.size() ? cap[k] : n_ubatch, !unified, n_keep_tail)` (or `split_seq`); run `attn->prepare(ubatches, hook)`. On `i_fail`, set `cap[i_fail] = U` and repeat.
   - Ubatches 0..i_fail−1 replay identically, and ubatch i_fail then has ≤ U tokens (:755, :788). So i_fail strictly increases, and the loop runs at most #ubatches times.
   - The recurrent `prepare` (:207) moves after the accepted attention prepare and runs once on the final ubatches.
   - Count replays in a context counter: `n_qsa_resplit`, reported by the W9 runs.
4. **Always-on graph-build check (no env gate).** In `build_qsa_top_k`, right after `scalar` (qwen4exp.cpp:1453), the check fires when `!scalar && n_kv > indexer_top_k + r − 1 && n_tokens > S × 512`. It then sets a flag on the graph result. `process_ubatch` checks that flag after `model.build_graph` (llama-context.cpp:1838-1844 area). It logs an error naming n_kv, n_tps, n_stream and the layout, and returns `GGML_STATUS_FAILED`, so `llama_decode` returns an error and never allocates the graph.
   - The reserve graphs are not affected: the pp/tg reserves are compact (`LAB_NC qsa_input … forced=0 scalar=1` in every log), and the variant has T ≤ S × 512.
   - A reused graph keeps the same compact flags (`can_reuse` compares them, qwen4exp.cpp:1283-1287).
5. **Implication unit test** (CPU only, specified now, run at W9 acceptance): `tests/test-qsa-resplit.cpp`.
   - **Layouts:**
     - §4.2 single sequence with 2-D cells at several depths;
     - R1-CB unified, two sequences with a seq_cp prefix that ends inside a block;
     - R1-C layout 3, two sequences each holding an image;
     - split `-np 2` with one image slot.
   - **Batches:** random token batches with 1-4 sequences and lengths 1..3·n_ubatch, plus embd batches with 2-D positions.
   - **Grid:** n_keep_tail ∈ {0, 4}, n_ubatch ∈ {512, 1024, 4096, 8192}, S ∈ {1, 2}.
   - **Procedure:** for every ubatch from init_batch-with-hook, apply it to the real caches (attention and indexer), then evaluate the graph-time predicate (the exact :1453 expression, including `blocks` and `qsa_prefix_matches`).
   - **Asserts:**
     - (i) graph-NC(u) ⇒ NC0(u);
     - (ii) NC0(u) ⇒ u.n_tokens ≤ S × 512;
     - (iii) every batch token appears in exactly one ubatch;
     - (iv) the last n_keep_tail tokens of every sequence share one ubatch;
     - (v) attention cells ≡ indexer cells after every apply;
     - (vi) replays ≤ #ubatches.
   - Pass: 0 violations over the grid with a fixed seed set.

**Forcing the non-compact scorer at reserve.**
- The flag is the `scalar` term at qwen4exp.cpp:1453. Forcing it false gives `compact = maskless = false`, the f32 bias `[n_blocks, n_tps, n_stream]`, and the masked strip route (:1455-1462).
- Reserve shape: `graph_reserve(T, n_seqs, n_out, mctx)` with `mctx = memory->init_full()` (llama-context.cpp:915), so n_kv = the full cache size per stream. The logs confirm `n_kv = 131072` at c131072 unified, and `n_kv = 65536 … n_stream = 2` at `-np 2` c131072.
- **A one-stream ubatch in a split cache cannot be reserved.** `LAB_FORCE_NC=3` (T = 1024 on n_seqs = 1 with `-np 2`) aborts in the probe: `GGML_ASSERT(a->ne[2] == b->ne[2]) failed` at ggml.c:4155, from `ggml_set_rows` in `build_attn_qsa` (`logs/srv_D_f3.log`). The init_full context always spans all streams. So whether a reserve of shape [K, T/s, s] covers a real one-stream non-compact ubatch [K, T, 1] in a split cache is UNPROVEN. That goes to the W9 STRICT runs (AT-C12/M4).
- For unified caches n_seqs does not matter: `LAB_FORCE_NC=3` (n_seqs = 1) equals `=1` (n_seqs = 4) to the byte on every backend and context (`srv_B_f3.log` vs `srv_B_f.log`).

**Fit probe** (code):
- `common_get_device_memory_data_impl` loads the model with `no_alloc = true` (common/fit.cpp:56-58), creates the context, and reads `llama_get_memory_breakdown`. In no_alloc mode the compute term is `backend_buf_exp_size[i]` (llama-context.cpp:4329-4334).
- Only the **first** pp reserve fills it: `graph_reserve(..., no_alloc, no_alloc ? backend_buf_exp_size.data() : nullptr)` (:943-944). The tg reserve (:971) and the second pp reserve (:993, :996) pass `sizes = nullptr`, so they only split the graph (:3024-3029). The assignment from `ggml_backend_sched_get_buffer_size` is skipped when no_alloc (:1007-1009).
- `ggml_gallocr_reserve_n_size` overwrites `sizes[i] = 0` and sums the chunks of the current graph only (ggml-alloc.c:1235-1244).
- `ggml_backend_sched_reserve_size` calls `ggml_backend_sched_reset` (ggml-backend.cpp:2296) before `split_graph`. `graph_reserve` already resets at llama-context.cpp:2985, and the only caller of `reserve_size` is llama-context.cpp:3026 (`grep -rn ggml_backend_sched_reserve_size` over the tree: the header, the definition and this call). The non-size `ggml_backend_sched_reserve` does not reset before splitting (ggml-backend.cpp:2305-2319). So the reset at :2296 is redundant for allocation and only wipes the pins set during graph build (R1-NKVO).
- **MTP in the probe: yes.** When `spec_mtp` is set, `common_init_result` builds `cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP` and passes it as `common_fit_extra_model` (common/common.cpp:1248-1275). `add_extra_memory` measures it with the same function (common/fit.cpp:203-251).
  - Log: `operator(): getting device memory data for the extra model at a context size of 131072`, then a second memory-breakdown table (`logs/smoke_lv4_B_f.log` lines 22-26).
  - With the lab patch, the probe's MTP context logs `LAB_NC variant: ctx_type=1 … no_alloc=1` (every forced log).

### Step 1: setup and commands (strix-002, 2026-11-05 15:13-15:40 CET; I was the only agent; every build and server run under `flock ~/gpu.lock`)
- **Tree:** `git -C ~/dev/wt_lab worktree add --detach ~/dev/wt_lab/r2res/tree 94e086847` (a clean checkout; wt_pr219v2 was not used).
- **Lab patch** `~/dev/wt_lab/r2res/r2res.patch` (copy in SP/final219/r2res/r2res.patch), 2 files:
  - `src/models/qwen4exp.cpp`: `scalar = !llama_lab_force_nc && …` at :1453, plus a `LAB_NC qsa_input` log line under `LAB_NC_LOG=1`.
  - `src/llama-context.cpp`: in `sched_reserve`, after the second pp reserve and before the sizes are recorded, when `LAB_FORCE_NC ≥ 1`:
    - one extra `graph_reserve` with `llama_lab_force_nc = true`, at n_kv = init_full;
    - `s = kv_unified ? 1 : n_seq_max`, `n_tps = min(n_ubatch, LAB_NC_TPS)` (default 1024 = S × 512 with S = 2; LAB_NC_TPS=512 gives S = 1);
    - `T = n_tps·s` for mode 1 (the spec as posed: n_tps per stream), and `T = n_tps` in total for mode 2. Mode 3 is mode 2 on one sequence;
    - `n_seqs = n_seq_max`, `n_outputs = max(n_seqs, min(T, n_outputs_pp))`.
  - **In no_alloc (probe) mode** the extra reserve goes through `reserve_size` into a local array, so `backend_buf_exp_size` keeps the PR's value (the probe estimate stays the PR's). The variant's own size is logged as `variant_alone`.
  - The forcing is active only inside that one reserve call.
  - The mode-3 and `LAB_NC_TPS` knobs were added after the first 12 runs. Without those env vars the code path is unchanged.
- **Build:** `cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/lib/llvm/bin/clang -DCMAKE_PREFIX_PATH=/opt/rocm -DLLAMA_CURL=OFF && cmake --build build --target llama-server -j 16` (rc 0, three incremental rebuilds).
- **Server** `~/dev/wt_lab/r2res/srv.sh TAG KV "EXTRA" "ENV"` (copy SP/final219/r2res/srv.sh). These are the R1-C Signal-3.8 + mmproj + MTP flags minus `--parallel`, with **`-fit on`**:
  - `env LAB_NC_LOG=1 $ENV llama-server -lv 5 -m ~/models/signal3.8/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf -mm ~/models/signal3.8/mmproj-Signal-3.8-Flash-Next-f16.gguf --mmproj-device ROCm0 --image-min-tokens 1024 --spec-draft-model ~/models/qwen3.8-flash-next-mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf -dev ROCm0 --spec-draft-device ROCm0 --metrics --host 127.0.0.1 --port 8099 --threads 8 --load-mode none --lazy-mode on-direct --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-draft-ngl all -ngl 99 --n-gpu-layers-draft 99 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.05 $EXTRA --presence-penalty 0.0 --repeat-penalty 1.0 -ctk $KV -ctv $KV --reasoning-preserve --jinja -fa on -fit on`
  - Init only: the script waits for `listening on`, then kills the server.
  - `-lv 5` is used because the probe forwards its internal llama logs only at verbosity ≥ LOG_LEVEL_DEBUG = 5 (common/common.cpp:1276, common/log.h:24). One `-lv 4` smoke run, `logs/smoke_lv4_B_f.log`, gave identical sizes.
- **Runs:**
  - `run.sh`: the 5 cells × {unset, `LAB_FORCE_NC=1`}, plus `B_f_r2` (noise) and `D_f2` (mode 2).
  - `run2.sh`: `D_f3` (mode 3, aborted as above), `B_f3`, `E_u`, `E_f`, `D262_u`, `D262_f2`, `D262_f`.
  - `run3.sh`: S = 1 (`LAB_NC_TPS=512`): `E_f_s1`, `B_f_s1`, `C_f_s1`, `D_f2_s1`, `D262_f2_s1`.
  - Cells: A = default server (no -c/-ub: auto 4 slots unified, n_ctx 262144 from fit, ub 512); B = `-c 131072 -ub 4096 -b 4096`; B' = B with f16; C = `-c 261888 -ub 8192 -b 8192`; D = `-np 2 -c 131072 -ub 1024 -b 1024` (split); E = `-c 131072 -ub 1024 -b 1024` (unified, extra: separates n_stream from ub against D); D262 = `-np 2 -c 262144 -ub 1024 -b 1024` (extra: split slope).
  - Every cell is q8_0 except B'.
- **Parsing:** `parse.py` (SP/final219/r2res/parse.py) labels each `sched_reserve` pair as probe main, probe MTP (after the extra-model line), or real main / real MTP / MTP re-reserve (after `successfully fit params`), and reads the `LAB_NC variant` byte lines. Output: `~/dev/wt_lab/r2res/parse.json` (copy in SP/final219/r2res/). Fits: `fit.py` (SP/final219/r2res/fit.py).
- **Logs:** `~/dev/wt_lab/r2res/logs/srv_<tag>.log` + `.cmd` (50 files, 37 MB).

### Raw numbers: `sched_reserve` (MiB) and the forced increment
Each cell shows the unforced reserve; the forced reserve (mode 1 as posed unless stated); and the increment = forced − unforced, computed from the logged bytes.
- In every unforced run, **probe = real to 0.01 MiB** on both backends and in both contexts (all 7 unforced runs: A_u, B_u, Bf16_u, C_u, D_u, E_u, D262_u).
- The MTP re-reserve after `load_model` equals the first MTP reserve in every run.

| cell | ctx | unforced ROCm0 / Host | forced S=2 ROCm0 / Host | **Δ S=2 ROCm0 / Host** | forced S=1 ROCm0 / Host | **Δ S=1 ROCm0 / Host** |
|---|---|---|---|---|---|---|
| A (c262144 ub512, T = 512) | main | 1323.34 / 28.30 | 1744.36 / 412.29 | **+421.02 / +383.99** | (= S=2: T = min(512, ·)) | +421.02 / +383.99 |
| A | MTP | 1285.03 / 28.03 | 1673.03 / 412.02 | **+388.00 / +383.99** | (= S=2) | +388.00 / +383.99 |
| B (c131072 ub4096) | main | 1609.88 / 202.06 | 1715.95 / 435.82 | **+106.08 / +233.76** | 1609.88 / 218.79 | 0.00 / +16.73 |
| B | MTP | 1465.23 / 201.75 | 1465.23 / 435.54 | **0.00 / +233.79** | 1465.23 / 218.52 | 0.00 / +16.77 |
| B' f16 | main | 1609.61 / 201.80 | 1609.61 / 435.56 | **0.00 / +233.76** | — | — |
| B' f16 | MTP | 1465.23 / 201.75 | 1465.23 / 435.54 | **0.00 / +233.79** | — | — |
| C (c261888 ub8192, owner) | main | 3315.30 / 403.86 | 3315.30 / 820.57 | **0.00 / +416.71** | 3315.30 / 411.92 | 0.00 / +8.06 |
| C | MTP | 2938.45 / 403.50 | 2938.45 / 820.29 | **0.00 / +416.79** | 2938.45 / 411.64 | 0.00 / +8.15 |
| D (np2 split c131072 ub1024), mode 1 T=2048 | main | 763.45 / 51.46 | 2102.14 / 485.88 | **+1338.69 / +434.41** | — | — |
| D, mode 1 | MTP | 741.18 / 51.18 | 1817.60 / 485.59 | **+1076.42 / +434.40** | — | — |
| D, mode 2 T=1024 (S=1: T=512) | main | 763.45 / 51.46 | 1051.95 / 243.82 | +288.50 / +192.36 | 763.45 / 122.79 | 0.00 / +71.33 |
| D, mode 2 | MTP | 741.18 / 51.18 | 941.55 / 243.54 | +200.37 / +192.36 | 741.18 / 122.52 | 0.00 / +71.34 |
| E (unified c131072 ub1024) | main | 775.42 / 51.84 | 1715.95 / 435.82 | **+940.54 / +383.98** | 942.86 / 218.79 | +167.44 / +166.95 |
| E | MTP | 742.06 / 51.56 | 1389.55 / 435.54 | +647.49 / +383.98 | 871.53 / 218.52 | +129.47 / +166.96 |
| D262 (np2 c262144 ub1024), mode 1 | main | 1404.58 / 52.59 | 3943.64 / 871.38 | +2539.06 / +818.79 | — | — |
| D262, mode 1 | MTP | 1350.30 / 52.31 | 3083.10 / 871.09 | +1732.80 / +818.78 | — | — |
| D262, mode 2 | main | 1404.58 / 52.59 | 1885.45 / 437.32 | +480.88 / +384.73 | 1404.58 / 220.29 | 0.00 / +167.71 |
| D262, mode 2 | MTP | 1350.30 / 52.31 | 1743.05 / 437.04 | +392.75 / +384.73 | 1350.30 / 220.02 | 0.00 / +167.71 |

- **Forced reserve = max(unforced, variant alone) per backend, to the byte, in every forced run.** This is gallocr's grow-only buffer behaviour. The variant's own size `V` (`variant_alone` in the probe lines, bytes) is therefore what matters. **The increment is max(0, V − P_pp), not a strip count × n_ctx.**
- **The forced route is the one intended.** Every forced variant logs `qsa_input … blocks=1 blk_bias=1 forced=1 scalar=0`, e.g. `il=3 r=4 n_kv=131072 n_blocks=32768 n_tps=1024 n_stream=1` (B) and `il=3 n_kv=65536 n_tps=1024 n_stream=2 n_tokens=2048` (D, mode 1). The pp/tg reserves log `forced=0 scalar=1` (compact).

**V, the variant alone (bytes, q8_0):**

| config | K (n_kv per stream) | T | s | V host main | V host MTP | V ROCm0 main | V ROCm0 MTP |
|---|---|---|---|---|---|---|---|
| A | 262144 | 512 | 1 | 432321600 | 432035872 | 1829094912 | 1754294400 |
| B = E (identical) | 131072 | 1024 | 1 | 456991808 | 456699936 | 1799308800 | 1457049728 |
| B' f16 | 131072 | 1024 | 1 | 456713280 | 456699936 | **1555760640** | 1457049728 |
| C | 261888 | 1024 | 1 | 860428352 | 860136480 | 3274390016 | 2764686464 |
| D mode 1 | 65536 | 2048 | 2 | 509477440 | 509173792 | 2204255232 | 1905893504 |
| D mode 2 | 65536 | 1024 | 2 | 255664704 | 255373344 | 1103053824 | 987287680 |
| D262 mode 2 | 131072 | 1024 | 2 | 458564160 | 458272800 | 1977041920 | 1827721344 |
| D262 mode 1 | 131072 | 2048 | 2 | 913703488 | 913399840 | 4135207936 | 3232866432 |
| E S=1 (= B S=1) | 131072 | 512 | 1 | 229422144 | 229136416 | 988661248 | 913860736 |
| C S=1 | 261888 | 512 | 1 | 431925312 | 431639584 | 1827453440 | 1752652928 |
| D S=1 | 65536 | 512 | 2 | 128758336 | 128473120 | 665699328 | 477653120 |
| D262 S=1 | 131072 | 512 | 2 | 230994496 | 230709280 | 1170588672 | 881879168 |

**Fits (`fit.py`):**
- **Host, exact.** On all 11 configurations (unified and split, T 512/1024/2048, K 65536..262144), **V_host(MTP) = 3·K·T + 12·K·s + 51244·T + 32 B, with residual 0 B on every point**. V_host(main) = V_host(MTP) + 285216..303648 B (≤ 0.29 MiB).
  - Reading: 3 B per (cell × query) is the f16 KQ mask (2 B) plus the f32 per-block bias (4 B / r 4); 12 B per cell per stream are the per-cell inputs; 51244 B per token are the other host inputs.
  - The first fit used 7 S=2 points; the 4 S=1 points were predicted out of sample with 0 B error.
- **ROCm0 is not linear in K·T.** The best 4-term fit leaves up to 138.35 MiB of residual (main). V/(K·T) ranges from 12.2 to 19.8 B.
  - **Unified at T = 512, exact:** V_dev = 6412·K + 141.36 MiB (main) and 6412·K + 70.03 MiB (MTP). Fitted on E_s1 and C_s1, it predicts A (K 262144, T 512) to the byte (1829094912 and 1754294400).
  - **Split at T = 512:** 7704·K + 153.36 MiB (main) and 6168·K + 70.03 MiB (MTP), each from only 2 points (UNPROVEN elsewhere).
- **The requested form fails.** Increment = a + b × n_ctx × n_stream over B, C, D (n_tps 1024 per stream), per (ctx, backend): residuals are host 8.68 / 8.63 MiB and ROCm0 669.45 / 538.21 MiB (main / MTP), all > 1 MiB.
  - The cause is the subtraction of P_pp: P depends on ub (B ub4096 202.06 MiB host vs D ub1024 51.46), and on the device side the gallocr peak is not additive.
  - E vs B shows it directly: **identical V, and Δ ROCm0 main = +940.54 at ub1024 against +106.08 at ub4096.**

**Type independence (PLAN-v3.md:316, cell B').**
- Host: f16 and q8_0 give the same increment (+233.76 / +233.79). V_host differs by 0.27 MiB, which is the same difference as in the unforced pp reserve (201.80 vs 202.06).
- **ROCm0: not type-independent.** The main variant alone is **1555760640 B for f16 against 1799308800 B for q8_0 (+232.27 MiB for q8_0)**, so the increment is 0.00 (f16) against +106.08 MiB (q8_0).
  - MTP is identical, consistent with the MTP KV not following `-ctk/-ctv` (R1-C side observation).
  - Cause of the +232.27 MiB: UNPROVEN (not investigated). At S = 1, f16 was not measured.

**Pre-registered ceiling.**
- The ceiling, per context: host ≤ S·n_ctx·1536·n_stream + 1 MiB, device ≤ S·n_ctx·3541·n_stream + 1 MiB, with n_ctx = `-c`. "Per-stream" means n_ctx = n_ctx_seq instead.
- S=2 ceilings (MiB): A 769 / 1771.50; B, B', E 385 / 886.25; C 768.25 / 1769.77; D 769 / 1771.50 (per-stream 385 / 886.25); D262 1537 / 3542.
- Results:

| config | S=2 result | S=1 ceilings (host / dev) | S=1 result |
|---|---|---|---|
| A | pass (384 / 421 / 388) | 385 / 886.25 | pass, host margin 1.01 MiB |
| B | pass | 193 / 443.63 | pass (16.7 / 0) |
| B' | pass | — | — |
| C | pass | 384.63 / 885.38 | pass (8.1 / 0) |
| D mode 1 | pass with `-c`; **fails per-stream** (host 434.41 > 385, dev 1338.69 > 886.25) | — | — |
| D mode 2 | pass both readings | 385 (per-stream 193) / 886.25 | pass (71.3 / 0) |
| **E** | **FAIL: ROCm0 main 940.54 > 886.25 (+54.29 MiB)**; host 383.98 ≤ 385 (margin 1.02) | 193 / 443.63 | pass (167.0 / 167.4) |
| D262 | pass | 769 / 1771.50 | pass (167.7 / 0) |

  - In short: all five pre-registered cells pass at S=2 when n_ctx = `-c`. The extra unified ub1024 cell E exceeds the device ceiling, so S = 2 is not a valid general bound. Every configuration measured at S=1 is under the S=1 ceiling.

### Probe estimate vs real reserve (forced; probe value = pp #1 sizes, bytes logged as `before` in the probe)
- **Gaps (probe − real, MiB):** the probe misses the whole increment.
  - Main ROCm0 / Host: A −421.02 / −383.99; B −106.08 / −233.76; B' 0.00 / −233.76; C 0.00 / −416.71; D −1338.69 / −434.41; D mode 2 −288.50 / −192.36; E −940.54 / −383.98; D262 mode 1 −2539.06 / −818.79; D262 mode 2 −480.88 / −384.73.
  - At S=1: B 0 / −16.73; C 0 / −8.06; D 0 / −71.33; E −167.44 / −166.95; D262 0 / −167.71.
  - MTP is the same, with its own numbers (table above).
- **More than 1 MiB below the real reserve in 16/16 forced runs (14 distinct configurations) on ROCm_Host, and in 10/16 on ROCm0 (main context).**
- **The fix, checked against the measurement.** The probe's own variant size, obtained through `reserve_size` in no_alloc mode, gives **max(pp #1, variant) − real forced reserve = 0 B in 64/64 cases** (16 forced runs × 2 contexts × 2 backends; `SP/final219/r2res/probe_gap.txt`).
- The probe's pp #1 equals the real pre-variant size (`before`) in all 32 (run, context) pairs. The PR's tg and pp #2 reserves never exceed pp #1 in these cells, so today's probe is exact only because pp #1 happens to be the max.
- **The probe builds the MTP context** (`ctx_type=1 … no_alloc=1` in every forced log; the extra-model memory table in every log).

### Noise
- **Deterministic to the byte.**
  - `B_f` and `B_f_r2` agree on every byte of every reserve and LAB line.
  - Mode 3 equals mode 1 on unified B to the byte.
  - Every probe/real pair of the unforced runs agrees to 0.01 MiB.
  - This is consistent with C19/R1-C (0.00 MiB).
- No perf was measured: init-only runs, no requests.

### Answers to the three objections
1. **"The S = 2 re-split has no mechanism, and a missed prediction builds an unreserved graph in multi-sequence contexts."**
   - Mechanism specified above, all by code: hook after kv-cache.cpp:856, replay in hybrid-idx `init_batch`, total-token cap U = min(n_ubatch, S × 512), n_keep_tail kept by split_equal's own rule (:764-823).
   - A missed prediction can no longer build an unreserved graph. The always-on graph-build check makes `llama_decode` return an error for any non-compact ubatch with n_tokens > S × 512 (no env gate).
   - The implication test (graph-NC ⇒ NC0, plus five more asserts) is specified and runs at W9.
   - Remaining UNPROVEN: the attention ≡ indexer cells invariant (assert (v)), and coverage of one-stream non-compact ubatches in split caches by the s-stream reserve (cannot be reserved through init_full: D_f3 abort). Both go to W9 / AT-C12.
2. **"With -fit on, the fit estimate counts only the pp graph and loses the pins, so D17's 'reserve covers every route by construction' does not hold there."**
   - Confirmed for sizes by code (llama-context.cpp:943-944, :971, :993-996, :1007-1009; ggml-alloc.c:1239-1241).
   - Confirmed by measurement: the probe misses the variant in 16/16 forced runs (host), by up to 818.79 MiB host and 2539.06 MiB ROCm0 per context.
   - The pins part is the ggml-backend.cpp:2296 reset (code; size effect under `-nkvo` stays with AT-D4).
   - The proposed per-backend max reproduces the real reserve to the byte (64/64).
3. **"Non-compact reserve variant: the reserve graph shape and how the route is forced are unspecified, its size at n_ctx is unmeasured, and M5 cannot fail."**
   - The shape and the forcing flag are now specified and measured: `scalar` false at qwen4exp.cpp:1453, `init_full` n_kv.
   - The size is measured at n_kv = n_ctx in 7 configurations at S=2 and 5 at S=1.
   - There is an exact host formula and an exact unified device formula at T = 512.
   - M5 now has byte bounds that can fail (Decision, Index 7).

### Decision
- **Index 0 (mechanism, from the code reading).**
  - The re-split is the **pre-placement predicate NC0 hooked into `llama_kv_cache::prepare`** (after src/llama-kv-cache.cpp:856, before the restore at :861), **with a deterministic replay in `llama_memory_hybrid_idx::init_batch`** that re-runs `split_equal` at a per-ubatch cap. n_keep_tail is kept, so the re-split does not move up into decode.
  - **Correction to the proposal:** the cap is **n_ubatch' = min(n_ubatch, S × 512) total tokens per ubatch**, not S × 512 per stream. The n_keep = 0 branch (src/llama-batch.cpp:787-799) puts up to n_ubatch' tokens on one stream.
  - The recurrent prepare moves after the accepted attention prepare.
  - The init check `n_rs_seq + 1 < S × 512` replaces the :764 abort.
  - The always-on graph-build check and the implication unit test are written as specified above (test at W9).
- **Index 7 (§7.1.4 variant spec and M5).**
  - **The S = 2 device ceiling is exceeded: E, +940.54 > 886.25 MiB ROCm0 main.** So the named fallback applies: **S = 1** (`QSA_NONCOMPACT_STRIPS = 1`). At S = 1 every measured configuration is inside the S = 1 ceiling. The other fallback (gating on mtmd or seq_cp) is not needed.
  - The perf cost of S = 1 falls on non-compact ubatches only: multi-slot images, seq_cp blocks, and every image ubatch > 512 tokens in multi-sequence contexts. That cost stays UNPROVEN (U27, AT-C12).
  - **§7.1.4 text:**
    - one extra reserve graph with `scalar` forced false (a graph-param flag replacing the lab global) at `init_full` n_kv;
    - **T = min(n_ubatch, S × 512) tokens in total**, n_seqs = n_seq_max, s = kv_unified ? 1 : n_seq_max, so n_tps = T / s per stream;
    - n_outputs = max(n_seqs, min(T, n_outputs_pp));
    - in the main and MTP contexts whenever n_seq_max > 1;
    - the re-split unit equals the reserve's T (total tokens).
  - **Replaced:** the "n_tps = S × 512 per stream" shape. It reserves graphs with more tokens than n_ubatch: +1338.69 MiB ROCm0 main at D against +288.50 for the total-token shape at S = 2.
  - **M5's "+ S = 2 bound" is replaced, per context and per backend, by:** new reserve ≤ max(PR219 reserve, V) + 1 MiB, where K = n_ctx / s, T = min(ub, 512), and
    - **V_host = 3·K·T + 12·K·s + 51244·T + 32 B (+ 0.29 MiB in main)**, exact on 11 configurations;
    - **V_ROCm0 (unified, T = 512) = 6412·K + 141.36 MiB main, 6412·K + 70.03 MiB MTP**, exact on 3 configurations. Split: 7704·K + 153.36 MiB main and 6168·K + 70.03 MiB MTP (2 points, UNPROVEN elsewhere).
    - The q8_0 values bound f16, since f16 measured lower at S = 2. f16 at S = 1 is UNPROVEN.
  - **Instantiated M5 bounds (main ROCm0 / Host; MTP ROCm0 / Host, MiB):**

    | config | main | MTP |
    |---|---|---|
    | c131072 ub4096 | 1609.88 / 218.79 | 1465.23 / 218.52 |
    | c261888 ub8192 | 3315.30 / 411.92 | 2938.45 / 411.64 |
    | default server (c262144 ub512) | 1744.36 / 412.29 | 1673.03 / 412.02 |
    | c131072 ub1024 unified | 942.86 / 218.79 | 871.53 / 218.52 |

    Each value + 1 MiB.
  - The PR description reports the default-server cost: +421.02 / +383.99 MiB main and +388.00 / +383.99 MiB MTP.
  - The ub8192 and ub4096 rows of the M5 grid at the other n_ctx follow from the formulas. Their P is unmeasured, so whether V_ROCm0 < P there is UNPROVEN.
- **Index 6 (fit probe).** The probe estimate is below the forced real reserve by more than 1 MiB on every backend: 16/16 forced runs on host, 10/16 on ROCm0 (main). **The plan adopts the fix:**
  1. drop the `ggml_backend_sched_reset` at ggml-backend.cpp:2296. It is redundant: graph_reserve resets at llama-context.cpp:2985, and :3026 is the only caller (grep on 94e086847);
  2. in no_alloc mode, keep an explicit per-backend max over pp, tg, pass A, pass B and the variant, each through `reserve_size`, because ggml-alloc.c:1239-1241 overwrites the sizes. Measured: this max equals the real reserve to the byte in 64/64 cases;
  3. run the §10 guard on the fit path.

  The `-nkvo` size stays at AT-D4.
- **Moved to W9 / AT-C12/M4** (as the task states), with these additions:
  - the STRICT run of a one-stream non-compact ubatch of T tokens in a split cache (the reserve cannot express it);
  - the re-split counter > 0 on R1-C layout 3 and on the R1-RES workload;
  - 0 graph-build-check errors on the image matrix (requests A/B/C).
- **Status: resolved.** Index 0 by code; Index 7 and Index 6 by measurement (24 init-only server runs plus 1 smoke run, deterministic to the byte).

### Large files / processes
- None > 1 GB. Logs 37 MB in `~/dev/wt_lab/r2res/logs`, kept. The build is 314 MB in `~/dev/wt_lab/r2res/tree/build`, kept for re-runs.
- `pgrep -a llama-server` is empty after the runs. No systemd unit was touched, and wt_pr219v2 was not modified (read-only greps only).

## R2-M-LAB-BANDS-OP (shrunk): GPU-vs-O_CON band

**Question.** Does the §14.2 SYN row (PLAN-v3.md:562-568) hold on test-backend-ops' own inputs for the GQA-12 shape classes that the ported QSA_*_MASKLESS cases instantiate? The test is: 2 × p100 of the GPU-vs-O_CON NMSE ≤ the SYN row of the dispatched (G, splits) class (2.67e-9 G4, 3.03e-9 G1 1 split, 1.15e-7 G1 split-K). The second part: do d1, d2 and d3 still exceed 2 × the band on those inputs?

**Answer.**
- **The SYN row does not stand for test-backend-ops.** It holds for G4 and fails for both G1 classes on the ported shapes (`SP/final219/bandsop/compact_op.md`, `analysis_op.md`):
  - **G4** (qsa_prefill, n_q 513 / 640): p100 1.02e-9 at n_sel 2051 and 6.04e-10 at 257. 2 × p100 = 2.04e-9 ≤ 2.67e-9, so it **holds**, with ×1.31 headroom. 4/4 ported cases pass on test-backend-ops dumps (TBO), and 4/4 on the UNI generator.
  - **G1, 1 split** (qsa_decode, splits 1): p100 2.51e-8 (n_sel 17), 5.36e-8 (257), 6.17e-8 (2051). 2 × p100 is 5.0e-8 to 1.23e-7 against 3.03e-9, so it **fails by ×17 to ×41**. 0/12 TBO cases and 0/16 UNI cells pass.
  - **G1, split-K** (splits 2-33): p100 6.7e-8 (n_sel 257), 7.82e-8 (2051, splits 2-5), 8.54e-8 (2051, splits 17-33). 2 × p100 is 1.34e-7 to 1.71e-7 against 1.15e-7, so it **fails by ×1.17 to ×1.49**. 2/37 TBO cases pass: the sparse `blocks` 3 and 40 rows, at 3.91e-8 and 5.63e-8.
- **The defects are caught with margin on these inputs.** d1, d2 and d3 exceed k × p100 with min k = 12.1, set by d3 at n_q 1 / 32768 / blocks 8000 q8_0. Per class, the min k at the class p100 is: G1 split-K 12.1 / 13.3 / 15.5, G1 1 split 16.8 / 19.7 / 42.3, G4 1032 / 1757. Against the SYN row itself every D is ≥ 9.0 × the row. So under the SYN row nothing is missed. The SYN row fails because it would reject the clean PR kernel (false failures), not because it misses defects.
  - One exception, by construction: d1 cannot be applied on the n_q 1 `blocks` rows (4096/b100, 32768/b8000, 65536/b16000). Query 0 has q % 4 = 0, so it has no tail cell (tests/test-backend-ops.cpp:11560 at 94e086847), and D1 = 0 because there is nothing to drop. d1 is tested on those shapes through the multi-query cases (for example n_q 2 / b100: D1 2.4e-3).
- **Mechanism, measured (`SP/final219/bandsop/res2_summary.md`):**
  - **G1 1 split depends on cell order.** The PR decode visits cells in row order: `qsa_decode_unique_ids` keeps each row position (qsa-decode.cu:107-125), and the split is the position range (:137-142). O_CON visits them deduplicated and ascending (bands_ref.h `row_cells`), and P is rounded to f16 against the running max.
    - The PR cell formula `1+(j*37+q*13)%(keys-4)` (test-backend-ops.cpp:11567) wraps modulo 4092 at keys 4096, so the order is scrambled.
    - **E1:** with the same UNI rows sorted ascending, 512×2051×4096 drops from 6.17e-8 to 1.75e-9 (3 seeds). With the formula at n_kv 262144, where it does not wrap, the value is 1.70e-9.
    - For n_sel 257 / 17 the sorted value is still 7.2e-9 / 2.2e-8. That is still above 3.03e-9, so n_sel matters too.
  - **G1 split-K depends on the data amplitude.**
    - UNI ±1.0 gives 6.5e-8 to 8.5e-8 (ascending or not; E1 leaves n_q 1 / 9 at 7.3e-8 to 7.9e-8).
    - **E2:** uniform ±1.7 (the SYN amplitude) on the same PR rows gives 5.5e-8 to 6.4e-8 on the non-sparse rows (3.5e-8 to 3.7e-8 on blocks 40).
    - SYN 262k n_q 1 over seeds 1-3 gives 5.35e-8 to 5.85e-8.
  - **G4 is insensitive to row order.** The plan sorts and merges each group of 4 rows (qsa3_rows_kernel / qsa3_merge_kernel).
    - At n_q 4096 / 2051 / 262144, the PR rows on UNI give 3.77e-10.
    - SYN rows on UNI data give 1.44e-9. SYN data on SYN rows gives 1.34e-9 (R1-M-Lab A). The UNI generator at ±1.7 on SYN rows gives 1.53e-9 at decode n_q 512 (= SYN 1.52e-9).
  - The type spread f16 vs q8_0 is ≤ ×1.06 in every cell (max 1.058, `analysis_op.md`).
- **New finding: the PR decode is not run-to-run deterministic on the ported rows.** The rows contain repeated keys. j = 1 is forced to key 2, and the formula produces key 2 again wherever (37j + 13q) % (keys − 4) = 1 (:11567-11569). For example, 255 of 512 rows at 512×2051. Which occurrence survives is decided by an atomicCAS race (qsa-decode.cu:120), and that changes the cell's tile or split.
  - Observed: up to 18 distinct unique-id hashes over 6 harness runs of one dump (512×257), and the output differs run to run by NMSE up to 6.9e-11 (maxabs 2.8e-5).
  - On the same dump, 2 of 5 harness runs were bitwise equal to test-backend-ops' ROCm0 output, and 3 were not.
  - Rows without duplicates (all `blocks` rows) are deterministic.
  - So §14.1's "each shape run twice per backend, outputs memcmp-equal" would fail on the PR decode for these cases. The rewrite needs a deterministic dedup (keep-first on the host reproduces the kernel's usual result: `UIDS … host_dedup first` hash = the kernel's hash on 001/003), or the case list must avoid duplicates. Which of the two is a plan decision, UNPROVEN here.
- **Route facts the plan relies on** (QSA_ROUTE census, `remote/op/logs/tbo_r1_*.log`):
  - Every ported same-type case with n_q ≤ 512 runs qsa_decode, including the QSA_PREFILL_MASKLESS 128/129/130/200/97/70-query cases. fattn.cu:1154 tries decode first; QSA_DECODE_MAX_QUERIES = 512 (qsa-decode.cuh:5).
  - qsa_prefill runs only for n_q 513/640 (V packed, K direct: `direct 0 kdirect 1`) and for the 3 mixed pairs (`kdirect 0`, both packed, qsa-prefill.cu:681-686).
  - So the comment at test-backend-ops.cpp:16558 ("prefill … reads direct when the attention is small against the cache: 130 or 200 queries against 32k / 64k keys") is wrong for same-type pairs: those run on decode. **No ported case reaches the qsa_prefill VD (both-direct) path.** It is bitwise equal to V0 on the 513/640 dumps (`BITWISE … words_differ_vs_pair_ref 0`).

### Per-shape results (p100 over TBO r1/r2 × 3 harness repeats and UNI seeds 1-3; f16 and q8_0 PR-verbatim; full table with every type and defect: `SP/final219/bandsop/analysis_op.md`)

| class | splits/chunk | n_q | n_sel | n_kv | blocks | TBO p100 | UNI p100 | 2 × max | SYN row | holds | min k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G4 | – | 513 | 2051 | 4096 | 0 | 1.02e-9 | 1.02e-9 | 2.04e-9 | 2.67e-9 | yes | 1042 |
| G4 | – | 640 | 257 | 4096 | 0 | 5.76e-10 | 6.04e-10 | 1.21e-9 | 2.67e-9 | yes | 1763 |
| G1 | 1/32 | 128 | 17 | 512 | 0 | 2.39e-8 | 2.51e-8 | 5.02e-8 | 3.03e-9 | NO | 42.4 |
| G1 | 1/272 | 64-512 (6 shapes) | 257 | 4096 | 0 | 5.15e-8 – 5.26e-8 | 5.2e-8 – 5.36e-8 | ≤ 1.07e-7 | 3.03e-9 | NO | ≥ 19.9 |
| G1 | 1/2064 | 512 | 2051 | 4096 | 0 | 5.98e-8 | 6.17e-8 | 1.23e-7 | 3.03e-9 | NO | 17.2 |
| G1 | 5/64, 3/96 | 1, 8, 9, 16 | 257 | 4096 | 0 | 5.82e-8 – 6.47e-8 | 5.88e-8 – 6.7e-8 | ≤ 1.34e-7 | 1.15e-7 | NO | ≥ 15.5 |
| G1 | 33/64, 26/80, 17/128 | 1, 2, 4, 8, 9, 16 (12 shapes, b0/b100/b8000/b16000/b66000) | 2051 | 4096 – 270336 | various | 7.25e-8 – 8.25e-8 | 7.54e-8 – 8.54e-8 | ≤ 1.71e-7 | 1.15e-7 | NO | ≥ 12.1 |
| G1 | 5/416 – 2/1040 | 64 – 256 (9 shapes) | 2051 | 4096 – 270336 | 0/7000/15000/16000/66000 | 6.9e-8 – 7.82e-8 | 7.0e-8 – 7.8e-8 | ≤ 1.56e-7 | 1.15e-7 | NO | ≥ 13.6 |
| G1 | 3/688 | 130 / 97 | 2051 | 4096 | 40 / 3 | 5.63e-8 / 3.91e-8 | 5.87e-8 / 3.48e-8 | 1.17e-7 / 7.8e-8 | 1.15e-7 | NO / yes | 17.7 / 27.1 |

UNI-vs-SYN comparison at n_kv 262144 / n_sel 2051 (UNI ±1.0, 3 seeds; "SYN rows" = the C6 plan with UNI data). These are not ported cases:

| class | n_q | UNI, PR rows | UNI, SYN rows | ±1.7, SYN rows (E2) | SYN (R1-A seed 1; seeds 2-3 here) |
|---|---|---|---|---|---|
| G4 | 4096 | 3.77e-10 | 1.44e-9 | – | 1.334e-9 / 1.336e-9 (s1/s2, R1-A) |
| G1 1 split | 512 | 1.70e-9 | 1.71e-9 | 1.53e-9 | 1.516e-9; 1.52e-9 – 1.53e-9 |
| G1 split-K (33) | 1 | 7.97e-8 | 7.40e-8 | 5.58e-8 | 5.733e-8; 5.35e-8 – 5.85e-8 |

The SYN n_q 1 row is itself exceeded by a SYN seed: seed 3 gives 5.85e-8, and 2 × 5.85e-8 = 1.17e-7 > 1.15e-7. That row rests on 24 rows per seed.

### Setup (strix-002, 2026-11-05 17:35–18:13 CET; I was the only GPU user; every build and GPU run under `flock ~/gpu.lock`; nothing in `~/dev/wt_pr219v2*` modified)
- **Step 2, done first: test-backend-ops' own inputs.**
  - Lab tree: `~/dev/wt_lab/bands/tree` (94e086847, R1-M-Lab A's worktree). Patch `~/dev/wt_lab/bands/tree_bandsop.diff` (copy in `SP/final219/bandsop/remote/`), written by `patch_tree_op.py` / `patch_tree_op2.py`. It adds four things next to R1-A's `BANDS_NMSE_VS_CPU` line:
    - `BANDS_DUMP=<dir>`: one statement in `test_qsa_prefill_maskless::err()` (ratio 12 only) that writes the case's Q, the K/V cache bytes reordered to [key][head], the rows, and the ROCm0 and CPU outputs;
    - `BANDS_ROUTE=1`: one line each in the qsa_decode launcher (after qsa-decode.cu:142) and the qsa_prefill launcher (after qsa-prefill.cu:686), printing G, splits, chunk and direct/kdirect. This is the census for this lab;
    - `BANDS_PARTIAL=<dir>`: dumps the decode partial buffer.
  - Build: `flock ~/gpu.lock cmake --build build --target test-backend-ops -j 16`, using the existing HIP configure with the hard-rule flags (CMakeCache: GGML_HIP ON, gfx1151, Release, core-10.1 clang). BUILD 0 twice (`op/logs/op_tree_build*.log`).
  - Runs, one op per invocation:
    - `BANDS_ROUTE=1 BANDS_DUMP=op/dump_rN flock ~/gpu.lock ./tree/build/bin/test-backend-ops -o QSA_DECODE_MASKLESS -b ROCm0`, then the same with `-o QSA_PREFILL_MASKLESS`, for N = 1, 2. 52/52 and 26/26 passed on each run.
    - That gives 71 GQA-12 dumps per run: 47 decode + 24 prefill, 3 of them mixed pairs.
- **Step 1, fixture.** `op/bands_ref.h` (md5 f5cbf8ee…) is R1-A's file plus:
  - `UNI:A:nk:seed:n_q:ns:blocks[:sort]`:
    - Q/K/V come from `std::uniform_real_distribution<float>(-A, A)` on `std::default_random_engine`. That is init_tensor_uniform's generator (test-backend-ops.cpp:71-72), with a fixed seed per tensor.
    - K/V go f32 → f16 by RNE. q8_0 uses `quantize_row_q8_0_ref` on the f32 values, as `ggml_quantize_chunk` does (ggml-quants.c:278-301, :2850-2855, imatrix unused).
    - Rows are `tbo_rows`, the maskless formula verbatim (:11545-11574): blocks 0 is the plain formula, blocks > 0 the complete-block rows, blocks −1 the C6 SYN rows. `sort` = 1 sorts each row.
  - `TBO:<dump>` reads a dump, with the exact cache bytes.
  - **Citation correction:** the task cites :11432-11440. Those lines are the *masked* `test_qsa_prefill` formula (key 3 → 4, `stream*7`). The maskless cases use :11545-11574.
  - **The values cannot be reproduced bit for bit.** init_tensor_uniform seeds its engine from `std::random_device` (:71), so every test-backend-ops run draws new values. FNV validation is therefore applied where it is possible:
    - **Rows:** FNV-1a of the UNI rows equals the dump rows for **142/142 dumps**, both runs (`op/res/validate_inputs.txt`, `validate_inputs.sh`).
    - **Values:** distribution only. Over 142 dumps, TBO K/V have var 0.3329–0.3338 (U(−1,1) is 0.3333) and every 10-bin histogram bin is within 0.0012 of 0.1. UNI gives 0.3329–0.3338 and 0.0016. Q is noisier on both sides because it has 6144 values at n_q 1.
    - **Kernel path:** the harness PR-verbatim kernels on the dumped inputs reproduce test-backend-ops' ROCm0 output:
      - G4 (bands_attn PR-V0 / VD) is **bitwise equal in 28/28 comparisons** (7 dumps × r1/r2 × V0/VD: the 513/640 f16 and q8_0 dumps, interleaved and not), with 0 words differing.
      - Decode: the partials are **bitwise equal** to test-backend-ops' partial dump (`PARTIAL_CMP … words_differ 0` of 204 336, case 1×2051). The merge alone differs at NMSE 1.2e-14 to 1.4e-14 (maxabs 1.1e-8), even on the dumped partials. That is the PR merge kernel copied verbatim (qsa-decode.cu:65-79), built with the same clang. The cause is UNPROVEN, and it is 5 orders below every band.
      - 1-split decode cases with no duplicate race are bitwise equal (case 008, 2 of 5 runs). The others differ by the dedup race above.
  - The bands use the device merge (PR kernel). Host vs device merge differ by NMSE ≤ 3.9e-15 (`HOSTMERGE` lines).
- **Harness.** `op/bands_dec.cu` (22390af8…) and `op/bands_attn.cu` (a2083974…) are R1-A's harnesses plus:
  - the UNI/TBO fixtures;
  - `BANDS_DEFECTS=1`, the d1/d2/d3 R1-Q8 definitions applied to O_CON and measured against the clean O_CON (`run_defects`);
  - the TBO bitwise check;
  - the device merge, `UIDS` dedup diagnostics (`BANDS_DEDUP=first|last`) and `PARTIAL_CMP`.
  - Variants: PR-verbatim only, `PR-dec-f16,PR-dec-q8_0` (qsa_decode_wmma_partial, launch rule qsa-decode.cu:135-142) and `PR-V0-*`, `PR-VD-*` (qsa3_attn_kernel). The 10 shipped C18 loader pairs were skipped (bitwise equal to PR-verbatim, R1-M-Lab A index 14).
  - Build: `flock ~/gpu.lock /opt/rocm/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -I. -I../../c18 bands_{dec,attn}.cu -o bands_{dec,attn}`, BUILD 0.
  - Edits made after the main campaign (the ROWS_HASH/VALSTAT prints and the optional `sort` field) are additive. `syn_dec1` / `syn_dec512` / `syn_attn` reproduce R1-A's dec1/dec512/attn_syn262k_s1 gpu_vs_contract NMSE to all 5 printed digits (5.7326e-8, 1.5156e-9, 1.3342e-9, and q8_0 likewise).
- **Step 3.** `op/campaign_op.sh` (`op/logs/campaign_op.out`, 512 runs, all EXIT 0):
  - **TBO:** `./campaign_op.sh tbo r1` and `tbo r2` run every same-type GQA-12 dump on its census route. Decode: `BANDS_DEFECTS=1 ./bands_dec TBO:<dump> SYN 0 PR-dec-f16,PR-dec-q8_0 <tag> 32`, plus 2 repeats with defects off. G4: `./bands_attn TBO:<dump> SYN all 1 PR-V0-f16,PR-V0-q8_0,PR-VD-f16,PR-VD-q8_0 <tag> 32`. The 3 mixed dumps were skipped (`SKIP … mixed`); their route is packed-both `qsa3_attn_kernel<false, 8>`, which the harness does not instantiate.
  - **UNI:** `./campaign_op.sh uni`. 37 unique ported GQA-12 shapes, plus n_q {1, 512, 4096} × n_sel 2051 × n_kv 262144 with PR rows and with SYN rows, seeds 1-3, both types, defects on. n_q > 512 goes to bands_attn, the rest to bands_dec, matching the census route.
  - **Follow-up:** `op/campaign_op2.sh` (`op/logs/campaign_op2.out`, all EXIT 0). E1 is the sorted rows (`UNI:1.0:…:1`), E2 is ±1.7 data (`UNI:1.7:…`), and SYN 262k n_q 1/512 adds seeds 2 and 3.
  - Defects: d1 = last valid tail slot → −1, d2 = median valid block → −1, d3 = V × 1.001 before the f16 rounding (bands_ref.h `defect_row`, R1-Q8 `r1q8_lab.diff:73-87`). On the plain-formula rows, d1 drops slot W−3 (the slots after it are tail padding, :11570), and d2 drops 4 scattered cells.
- **Parsers:** `SP/final219/bandsop/src/analyze_op.py` → `analysis_op.md`; `compact_op.md` comes from it; `summarize_res2.py` → `res2_summary.md`. Raw outputs: `SP/final219/bandsop/remote/op/res/` and `res2/`.

### Noise
- **Seed to seed (UNI, 3 seeds):** per-case NMSE spread min→max is ≤ ×1.20 (TBO over r1/r2 × repeats: ≤ ×1.18). Examples: 1×2051×32768/b8000 7.12e-8 → 8.54e-8; 513×2051 G4 1.00e-9 → 1.02e-9.
- **TBO vs UNI on the same shape** agree within that spread. Examples: 1×2051×4096 TBO 8.25e-8 vs UNI 8.22e-8; 512×2051 TBO 5.98e-8 vs UNI 6.17e-8; 513×2051 G4 1.02e-9 vs 1.02e-9.
- **Run to run:**
  - G4 is bitwise stable.
  - Decode is bitwise stable except on rows with duplicate keys (the race above). There the NMSE between runs is ≤ 6.9e-11, ≥ 340× below the smallest G1 p100.
- **Type to type:** f16 vs q8_0 differ by ≤ ×1.06.
- **SYN n_q 1:** seeds 1-3 range 5.35e-8 to 5.85e-8.

### Decision (rule fixed before the run)
- **Not every UNI cell meets 2 × p100 ≤ SYN row:** on the ported shapes, 68 of 76 UNI cells and 47 of 53 TBO cells (shape × type) fail, all of them G1. Counting the n_kv 262144 comparison cells too, 125 of 139 fail. So the SYN row does **not** stand for test-backend-ops.
- **Option 1 (switch `test_sparse_attn::initialize_tensors` to the SYN generator) is rejected.**
  - Switching only the data does not fix G1: on the PR rows, ±1.7 data still gives 1.2e-8 to 5.2e-8 at 1 split (E2, `res2_summary.md`) against 3.03e-9.
  - Switching to the SYN rows would drop the coverage the ported cases exist for. The SYN plan is fixed at 4096 × 2051 with ascending complete blocks (bands_ref.h `syn_rows`). It has no n_sel 17/257 rows, no partial or empty `blocks` rows, no scattered physical cells and no 270336 uint16 case.
- **Option 2 is adopted.** §14.2 gets UNI rows, NMSE(GPU, CPU op) ≤ 2 × p100 measured here on the PR/W0 kernels, keyed on the census (G, splits) and the n_sel bucket:

  | class (census) | n_sel | band (2 × p100) | set by | min k at this band |
  |---|---|---|---|---|
  | G4 | 257 | 1.21e-9 | 640×257 q8_0 | 1757 |
  | G4 | 2051 | 2.04e-9 | 513×2051 q8_0 | 1032 |
  | G1, 1 split | 17 | 5.02e-8 | 128×17×512 | 42.3 |
  | G1, 1 split | 257 | 1.07e-7 | 127×257 q8_0 | 19.7 |
  | G1, 1 split | 2051 | 1.23e-7 | 512×2051 | 16.8 |
  | G1, splits 2-5 | 257 | 1.34e-7 | 1×257 q8_0 | 15.5 |
  | G1, splits 2-5 | 2051 | 1.56e-7 | 64×2051 (TBO) | 13.3 |
  | G1, splits 17-33 | 2051 | 1.71e-7 | 1×2051×32768/b8000 q8_0 | 12.1 |

  - Every defect margin is > 2× (min 12.1), so option 2 meets the rule. The wiki and SYN rows of §14.2 stay for the Q5 / fixture gates.
  - Split counts not measured here (6-16, > 33) are UNPROVEN until measured.
  - The table depends on the kernel's cell order (E1). If the rewrite's G1 kernel visits cells in a different order than the PR decode, its NMSE on these rows may differ from these rows. That is fine for detection (k ≥ 12) but must be re-recorded at W3/W4 (U34), on the new CPU op.
- **Pre-registered rules applied:**
  - The band is selected from the census (G, splits), and here G1 vs G4 differ by ×60 at equal n_q (512 vs 513).
  - No band is set on the kernel it gates: all rows come from PR-verbatim kernels.
  - GQA 4/8 move to W5 (not measured).
  - n_stream 2, the type spread and U34 move to W3/W4.
  - Rot is not exercised by these cases.
- **Added to the plan from this lab:**
  - (a) A deterministic dedup, or duplicate-free ported rows, before §14.1's memcmp determinism test can pass on the decode path.
  - (b) The test-backend-ops.cpp:16558 comment and §14.1's reach list must not claim qsa_prefill direct-V coverage. No ported case reaches it (route census). Add a case with n_q ≥ 513 and n_q × 64 < n_kv, or force `QSA_DIRECT=1`.
  - (c) The mixed-pair packed-both route (3 ported cases) has no band. It is UNPROVEN (U8/U34) until it is measured on a PR counterpart, the PR's own `qsa3_attn_kernel<false, 8>`.

**The two objections behind this lab:**
1. *"The op-test NMSE bands were never measured on test-backend-ops' input distribution or shape classes, and for kernels with no PR counterpart they would be calibrated on the kernel under test."*
   - **First half: upheld, and now measured.** 142 dumps of test-backend-ops' own inputs plus the UNI generator cover 37 shapes. G4 meets the SYN row. G1 1-split misses it by ×17 to ×41 and G1 split-K by ×1.17 to ×1.49. The table above replaces it.
   - **Second half: upheld as a rule, not resolved by measurement here.** Every row above is measured on PR-verbatim kernels, so none is calibrated on the kernel under test. The routes without a PR counterpart in this lab get no band from it and stay UNPROVEN: the mixed packed-both route, GQA 4/8 (non-WMMA f32 path), and the generic/portable/Vulkan tiers. Their bands must anchor on O_CON vs f64 (W1 contract-vs-f64 check) or on a PR kernel of the same class, never on themselves (U8, U7, U34).
2. *"The §14.2 numeric bands are applied as binding test-backend-ops gates on a data distribution they were not measured on."*
   - **Upheld.** On test-backend-ops' distribution (uniform ±1, the PR cell formula with its row order and repeated keys), the clean PR kernels exceed the SYN 1-split row by ×17 to ×41 and the split-K row by up to ×1.49. Binding the SYN row would fail clean code.
   - The cause is mostly row order (G1 1 split) and data amplitude (split-K), measured by E1 and E2. The binding gate for test-backend-ops is now the UNI table, measured on that distribution.

### Large files
- Deleted: `~/dev/wt_lab/bands/op/dump_r1` and `dump_r2` (3.3 GB each; largest file 564 MB) and the one-case `op/part` partial dump. No file over 1 GB remains under `~/dev/wt_lab/bands` (`find -size +1G` empty).
- Kept: `~/dev/wt_lab/bands/op/{res,res2,logs}` (4.5 MB) and the patched lab tree. Local copies are in `SP/final219/bandsop/{src,remote}`, with `analysis_op.md`, `compact_op.md` and `res2_summary.md`.

## OWNER DECISION 2026-11-05: D22 / §20.6 = option b (binding)
Backends with no SPARSE_ATTN kernel (CUDA, MUSA, Metal, SYCL, OpenCL, CANN, WebGPU, and Vulkan devices that can create no tier) get an init error that names the remedies. The masked route is deleted everywhere. The owner's reason: this fork does not support those backends anyway. Option a (a logged masked backstop) is rejected.

## OWNER DECISION 2026-11-07: reduced scope to finish sooner (binding)

- W8, W9, W10, W11 and W13 run builds, correctness tests for what they change, and the gates the plan marks binding for them. Long end-to-end performance and quality campaigns are not run inside those items; each one skipped is listed under "Moved to Prove" in IMPLEMENTATION.md and Prove runs it.
- Prove's end-to-end matrix is reduced to KV types f16, q8_0, q4_0, bf16 and the mixed pairs q8_0/f16 and f16/q8_0. Other types (q4_1, q5_0, q5_1, iq4_nl, f32, ...) are proven at op level only and are reported as such in PROOF and in the PR.
- Deep contexts in Prove: the depths in Prove item 4, plus about 262k (about 300k real text) and the deepest context that fits. Other intermediate depth steps are skipped. Deferred acceptance tests that ask for larger matrices run on this reduced matrix, and the reduction is recorded next to each result.
- No correctness check is dropped. The f16 V-pack cap is still measured in both settings on the reduced depth set.
- W7 (in progress when this was decided) is not affected.

## OWNER DECISION 2026-11-08: W7 failures deferred ("note pour plus tard à fixer") (binding)

The W7 gate failures and open points (Q1 direct form q4_0/bf16, AT-D2 CPU perf, AT-B1 Vulkan q8_0/q4_0, AT-B5 12 MiB, AT-D1 wording, iq4_nl MTP M1, -nkvo open cells, thin Q4 margins) are recorded in TODO-LATER.md. They do not block W8 and later items. W7 gets no repair pass. Prove still measures them and reports their results, but they do not make Prove fail and do not trigger a prove-fix. The PR lists them under "Known issues / follow-ups" with numbers and is opened as a draft.

## OWNER DECISION 2026-11-10: Q7 failure deferred like the W7 items (binding)
The owner's "note pour plus tard à fixer" rule applies to Q7 (TODO-LATER item 9): report it with exact numbers in the PR's "Known issues / follow-ups"; it does not block and gets no fix pass. Other Prove failures (for example the -np 2 split STRICT_RESERVE violations) are NOT deferred.
