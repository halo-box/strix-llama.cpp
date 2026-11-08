# QSA validation tooling (PLAN-v3 §13 W12)

Measurement sources, patches and the small summaries the QSA sparse-attention PR cites. Nothing here is built by
default; nothing here changes the library. Paths inside the lab scripts (`~/dev/wt_lab/...`, `~/models/...`) are the
ones they ran with on strix-002 and are kept as run. `EV/` below is the checksummed evidence copy
`strix-002:~/dev/wt_lab/evidence/final219` (PLAN-v3 header and E0).

## In the library: `tools/perplexity` (commit "perplexity : per-chunk KLD at full precision")

- `QKLD_CHUNK <chunk> all= lo= nlo= hi= nhi=` after every chunk of `--kl-divergence`, at `%.9e`, straight from the
  chunk's `kld_values` (the cumulative table is printed with 5 decimals). Chunk index 0-based.
- `QKL_TAIL=N`: score the last N positions of each chunk (2 <= N <= n_ctx - 1). Use the same N for the base and its arms.
- `QKL_SPLIT=P`: lo / hi split point of the QKLD_CHUNK line (default 2048).
- `QKL_HASH=1`: `QKL_HASH {base|kld} <chunk> <hash> n=<floats>` per chunk, a 64-bit FNV-1a hash of the bits of the
  scored logits, printed when the base is recorded and under `--kl-divergence`: a self-replay is bit-identical iff the
  hashes match chunk by chunk (the uint16 base leaves a self-replay KLD of a few 1e-7 per chunk, so the KLD cannot show it).

## `patches/`: the PR219 measurement arm (W0-lab)

W0-lab = the W12 commit + these patches, applied in order by `patches/apply.sh <tree>`:

| patch | what | origin |
|---|---|---|
| `01-qpin.patch` | QPIN record / replay of `qsa_rows_cb-` (kind 1), `qsa_rows_tk-` (kind 2) and `ffn_moe_topk-` (kind 3), name table unchanged (§4.5); `QPIN_IDX_TYPE`; the hook names in qwen4exp.cpp | `~/dev/wt_measQ/q/qpin.patch` rebased (its perplexity hunks are now the library commit) |
| `02-r1e-qsa-chunk.patch` | `QPIN_QSA_CHUNK chunk= src={cb,tk1,tk2} n overlap jaccard exact` per chunk (%.9f) | `EV/r1e/r1e/r1e_patch.py`, `r1e_full.diff` |
| `03-r1q8-defect-hook.patch` | post-pin defects `LAB_DEF=d1|d2|d3` (`LAB_D3_EPS`, `LAB_DEF_PRE`), `R1Q8_SUMMARY`, and the `QKLD_CHUNK <c> n= kld= same_top=` line. One change from the round-1 diff: `attn_output-<LAB_D3_L>` is observed only under `LAB_DEF=d3`. In the round-1 diff it was observed in every QPIN run, which splits the graph and changes the numerics (with it, a trim-0 reference reproduces R1-Q8's chunks, not R1-UB's); a d3 base is recorded with `LAB_DEF=d3 LAB_D3_EPS=0` so base and arms share the splits | `EV/r1q8/r1q8_lab.diff` |
| `04-c1-lab.patch` | `GGML_QSA_TRACE` (C1TRACE unique shapes), `GGML_QSA_MASKED_DENSE`, `GGML_QSA_DECODE_MAXQ`, `LLAMA_QSA_ALLVIS=1|2|3` | c1 lab diff as carried by `EV/r1av/r1av_full.diff` |
| `06-w13-measurement-hooks.patch` | not a W0-lab patch: the measurement-only env hooks that W13 took out of the branch, restored on a lab tree of a W13+ head: `LLAMA_QSA_PREFER_DENSE` (AT-W7PERF, AT-B4), `LLAMA_QSA_SHORTCUT_PIN` and `LLAMA_QSA_HOST_KV_COPY` (AT-W7PERF (c)/(d), AT-D4), `LLAMA_QSA_FORCE_NONCOMPACT` (AT-C12 / U38, the single-sequence decode error), `LAB_OVL_PRINT` (AT-D1). Applied by `harness/w7/lab_tree.sh` and `harness/w13/hooks_tree.sh` when the tree lacks them | `git diff 0da383ddd 0da383ddd~1` |
| `c15/c15_trace.diff`, `c15/c15_patch.py` | C15 reserve / strip-view trace (GGML_C15) | `EV/c15/remote/c15_trace.diff`, `EV/c15/patch.py` |

The four stacked patches reproduce the round-1 W0-lab tree in every file except `perplexity.cpp`, whose QKL_TAIL /
QKLD_CHUNK code is the library commit (same arithmetic and print format), and the one `attn_output` line of patch 03
above. Without their environment variables the patches do not change the graph.

## `w12/`: the W12 done-when and AT-W12-BANDS campaign (§16.5)

- `run.sh` one perplexity arm (modes record / replay / observe / pinbase / pinkld; HIP, Vulkan, round-1 Vulkan).
- `campaign.sh S A C D V B E` the phases: S = QKL_HASH bit-identity of every self-replay, A = ref8t0 (self-replay, control, H1, direct form incl. H3), C = ref16t0 (H2,
  agreement control), D = ref32x8 (H4), V = W0-Vulkan (V1, V2), B = ref8L (Q11a lo/hi split, U40), E = ref2k (Q11b shift).
  Each reference is deleted after its phase.
- `analyze_w12.py LOGDIR EVDIR` pre-registered formulas -> `summaries/w12_bands.md`.

## `harness/w1/`: the W1 done-when (CPU SPARSE_ATTN_PLAN / SPARSE_ATTN, §13 W1)

- `cap.sh` recaptures C3's rows (C3's `c3_capture` and arguments) and R1-A's wiki d4k fixture (C18's `c5_capture`
  "wikiS" with `.qkv`) with the PR219 build, into `~/dev/wt_lab/w1/cap` (deleted after the run, 4.4 GB).
- `compare_c3.py` checks the recapture against C3's logs (per-layer row statistics) and the
  `test-sparse-attn-plan-oracle --shuffle 7` verdicts -> `summaries/w1_plan_c3.md`.
- `w1_ocon.cpp` the CPU op against `harness/bandsop/bands_ref.h` (unchanged): bits vs O_CON, NMSE vs O_C18 / O_PLAN,
  NMSE of CPU FA + src[5]; `w1_ocon overflow` the V overflow rule. `vhash.cpp` the V-cache hash of a fixture.
  `campaign_w1.sh` the runs; `analyze_w1.py RESDIR` the verdicts -> `summaries/w1_ocon.md`.

## `harness/w2/`: the W2 done-when (HIP SPARSE_ATTN_PLAN, §13 W2)

- `cap.sh` recaptures C3s rows at d32k and d131k with C3s `c3_capture` and its pos0 list (PR219 build, read-only)
  into `~/dev/wt_lab/w2/cap` (deleted after the run, 2.4 GB). `rowstat.cpp` counts the rows the rows kernel can
  read in place.
- `w2_plan.cpp` the C3 plan-stage harness on the op: R plan nodes per graph on ROCm0, wall time per op, whole-dst
  memcmp against the CPU backend and the C3 oracle (`tests/sparse-attn-ref.h`). `campaign_w2.sh` the runs;
  `analyze_w2.py RESDIR` the table and the gate -> `summaries/w2_plan_c3.md`. `mutants_w2.sh` the mutation check of
  the op tests on ROCm0.

## `harness/w3/`: the W3 done-when (HIP SPARSE_ATTN G4 attend, §13 W3)

- `build_hip.sh` the HIP build with the rule flags; `tu_check.sh` (W2's) and `tu_all.sh` the `-Werror` TU builds of the
  W3 TUs with the resource remarks, parsed by `scripts/hip/gcn-cdna-vgpr-check.py parse_assembly` (spills, scratch).
- `tests_w3.sh TAG` the test-backend-ops runs (one op per run, `-j 1`, under flock): G4 cases with graphs on / off and
  with `GGML_CUDA_SA_FORCE=vdirect` / `vpacked`, all SPARSE_ATTN cases, CPU, SPARSE_ATTN_PLAN, the four QSA_* ops, the
  plan oracle ctest. `final_w3.sh` the same at the committed tree plus the Vulkan / CPU-only builds and their runs.
- `prband_run.sh`, `prband_an.py` the PR-verbatim G4 kernel (R2-BOP `bands_attn` TBO mode) on the W3 cases' own inputs
  (`GGML_SA_TEST_DUMP`), joined with the new kernel's NMSE -> `summaries/w3_prband.md` (the G4 band rows).
- `w3_attn.cu`, `w3_build.sh`, `w3_campaign.sh`, `w3_summ.py` the kernel-time harness on the C18 rows and the
  V-loader crossover -> `summaries/w3_kernel_time.md`; `w3_attrib.sh` the attribution lab (non-finite check, fixup).

## `harness/w4/`: the W4 done-when (HIP SPARSE_ATTN G1 attend and the G policy, §13 W4)

- `build_hip.sh` HIP build with the rule flags. `tu_all.sh` the `-Werror` TU builds of the W4 TUs
  (`sparse-attn-attend-g1.cu`, `sparse-attn-merge.cu`, `sparse-attn-attend-g4.cu`, `ggml-cuda.cu`), parsed for
  spills. `final_w4.sh` runs every build, TU check and test-backend-ops run (one op per run) at the committed tree.
- `w4_dec.cu` + `build.sh` is the W4 kernel harness. It includes, in one process:
  - the head of R2-BOP's `bands_dec.cu`, verbatim: the PR decode kernel and C18's `c18_dec_kernel`;
  - the worktree's G1 / G4 / pack / merge kernels;
  - `bands_ref.h` fixtures and oracles;
  - `sa_ref::plan` plans.
  - Modes: `band` (NMSE vs O_CON, planted defects, run-to-run determinism) and `time` (the C18 protocol).
- `uni.sh`, `tbo.sh`, `uni_an.py`, `tbo_an.py` run the PR-verbatim decode (R2-BOP `bands_dec`):
  - on UNI fixtures for the G1 split classes 6-16 and > 33;
  - on the W4 test-backend-ops G1 dumps.
  - They give the band rows -> `summaries/w4_bands.md`.
- `syn1.sh`, `syn1_an.py`: the decode n_q 1 SYN 262k fixture enlarged to 12 layers x 3 seeds
  (`BANDS_SYN_LAYERS=12`), with the re-set §14.2 (b) row -> `summaries/w4_bands.md`.
- `c18gate.sh`, `c18gate_an.py`: the G1 kernel gates against C18's decode harness kernel, same run ->
  `summaries/w4_c18gate.md`.
- `ata.sh`, `ata_plan.sh` (+ `trunc_rows.py` and W2's `w2_plan`), `ata_an.py`: AT-A G1 vs G4. They give the fitted rule
  in `sparse-attn-attend-g1.cu`, the D8 re-derivation and the trimmed-layer row -> `summaries/w4_ata.md`.
- `w4_reserve.cpp`, `atc5.sh`: AT-C5 at op level. The reserve goes through `ggml_backend_sched`, then every runtime
  n_q' <= ub and n_kv' is run with the policy G and with forced g4 / g1, under `GGML_SCHED_DEBUG_REALLOC=1` ->
  `summaries/w4_atc5.md`.
- `w4_procs.cpp` checks the HIP policy procs as llama resolves them: the group rule grid, the prefer_dense table and
  `GGML_CUDA_SA_FORCE`.

## `harness/w5/`: the W5 done-when (HIP generic and portable tiers, §13 W5)

- `final_w5.sh TAG` the W5 final at the commits under test, every step under flock: HIP clean rebuild, the multi-arch
  compile gate from an empty directory (`build-multiarch`, `GPU_TARGETS="gfx1100;gfx1151;gfx90a;gfx942"`), the Vulkan /
  CPU-only builds, the TU checks, `-o SPARSE_ATTN` on ROCm0 (default, `GGML_CUDA_SA_FORCE=portable`, `=generic`, graphs
  off, `=vdirect`, `=vpacked`) and on the CPU, SPARSE_ATTN_PLAN, the QSA_* regression ops on ROCm0 and Vulkan0, the plan
  oracle ctest. `runs1.sh`, `runs2.sh` the same runs as first executed (before the commits).
- `build_hip.sh` the HIP build with the rule flags (reconfigure: the new top-level `.cu` files are globbed at configure
  time). `tu_all.sh`, `tu_check.sh` the `-Werror` TU builds of the W5 TUs (`sparse-attn-generic.cu`,
  `sparse-attn-portable.cu`, `sparse-attn-pack.cu`, `sparse-attn-merge.cu`, the G1 / G4 TUs, `ggml-cuda.cu`), parsed for
  spills by `scripts/hip/gcn-cdna-vgpr-check.py parse_assembly`. `t.sh TAG ENV... CMD...` one test-backend-ops run under
  flock.
- `sarep.py LOG` the per-tier / per-class summary of the `GGML_SA_TEST_REPORT` lines (max NMSE / band). `f64rep.py LOG` the
  per-tier / per-type GPU-vs-f64 errors of the same lines (reported, never gated). `pairs.py LOG...` the S² pair coverage
  per tier and class, and the SA_GQAREP results.
- `w5_time.cpp`, `build_time.sh`, `time_runs.sh`, `time_an.py`: the op time per K = V type on the C18 wiki d4k / d131k rows
  with SYN data, through ggml (plan + R attends minus plan) -> `summaries/w5_time.md` (recorded, not gated: U8, P4 / P7).
- `arch_check.sh` the per-arch check of the multi-arch build's code objects (which SPARSE_ATTN kernels carry device code on
  each of the 4 targets), and the attribution of the build's warnings.
- Results: `summaries/w5_tests.md` (bands, GQA replication, tiers, resources, compile gate).

## `harness/w6/`: the W6 done-when (Vulkan plan and attend, portability, §13 W6, AT-B2, AT-B3, AT-B7)

- `all_final.sh` the whole W6 final on the tree at HEAD, every step under flock: the timing campaign, `final_w6.sh`, the mutants. `final_w6.sh`: `runs1.sh` (the two ops on RADV default, forced scalar, portable_sg32, portable, vdirect, vstaged, vpack, and on lavapipe), `runs2.sh` (vulkaninfo, the system lavapipe's crash, validation layer runs, `test-sparse-attn-vk` on both devices, `GGML_VK_SA_MAX_Q`, `GGML_VK_SA_CAP_MASK`), the HIP, CPU and CPU-only regressions.
- `w6_attn_report.py LOG...` passes, "not supported" by reason, max band ratio per tier and V variant, K/V pairs and rot cases seen. `summ.py` writes the summaries.
- `w6_time.cpp`, `time_campaign.sh`, `w6_time_an.py`: AT-B2 and AT-B3 at op level -> `summaries/w6_time.md`.
- `mutants_w6.py`, `mutants_val.py`: planted defects and the positive controls of the validation-layer runs.
- `lvp_try.c`, `lvp_bisect.py`: one SPIR-V on one ICD (compile + dispatch): how the system lavapipe's crash was isolated.
- Tools outside the tree, user directories, nothing installed in the system: `~/dev/wt_lab/w6/tools` (vulkan-validation-layers 1.4.341, vulkaninfo: `dnf download`, `rpm2cpio`), `~/dev/wt_lab/w6/lvp43` (lavapipe from Fedora 43: mesa-vulkan-drivers 25.3.6, llvm-libs 21.1.8, libdisplay-info), `lvpenv.sh` its environment.
- Results: `summaries/w6_tests.md`, `summaries/w6_time.md`.

## Other tools

| path | what | origin |
|---|---|---|
| `census_check.py` | §14.3 graph-level reach checker: `expect` (trace x select dump == backend census per key), `keyset` (hooked-arm census), `recipe` (§14.3 zero / nonzero rows), `selftest`. The log contract is in its docstring | new (W12) |
| `bench/analyze.py` | §16 statistic: mean of reps 1..n per process, hierarchical bootstrap (4000 draws, seed 1), ratio and ratio-of-ratios, PASS / FAIL / RERUN | C10 `analyze.py` changed to the mean (`EV/bs_skeptic.py`) |
| `bench/c10/` | C10 real-text llama-bench patch (`c10.diff`, `patch_c10.py`), its build and campaign scripts; `analyze_c10_round0.py` is the round-0 median analysis kept for reproduction | `EV/c10/` |
| `kfd/` | C7 hipMalloc / pool shim (`c7shim.c`) and KFD eviction sampler (`c7samp.py`), runner and analysis | `EV/c7/` |
| `harness/c3`, `c5`, `c18` | plan, union and attend / decode kernel harnesses | `EV/c3`, `EV/c5`, `EV/c18` |
| `harness/bands` | R1-A numeric-band harnesses (O_CON reference `bands_ref.h`, `bands_attn.cu`, `bands_dec.cu`, CPU) | `EV/bands/src` |
| `harness/bandsop` | R2-BOP harnesses with the UNI / TBO fixtures (`UNI:` generator = init_tensor_uniform's distribution, `TBO:` dumps written by `tree_bandsop.diff` from test-backend-ops) | `EV/bandsop/src`, `EV/bandsop/remote/tree_bandsop.diff` |
| `campaigns/` | `r1/text_campaign.sh`, `r1/img_campaign.sh` (+ `run.sh`), `r1c/runB3.sh` (+ `srv3.sh`), `r1s/r1s.sh`, `r2res/srv.sh`, `r2res/parse.py`, the r1ub / r1e / r1q8 runners and analyzers | `EV/r1`, `EV/r1c`, `EV/r1s`, `EV/r2res`, `EV/r1ub`, `EV/r1e`, `EV/r1q8` |
| `summaries/` | `kdecide.md`, `compact*.md`, `analysis_op.md`, `res2_summary.md`, `probe_gap.txt`, the R1 analysis outputs; W12: `w12_bands.md` (analyzer output), `w12_chunks.tsv` (every arm's per-chunk KLD, logits hashes and agreement), `w12_campaign*.out` (run lines), `w12_hookobs_campaign.out` (the aborted first pass with the round-1 r1q8 hook); W1: `w1_ocon.md`, `w1_plan_c3.md`; W2: `w2_plan_c3.md`; W3: `w3_prband.md`, `w3_kernel_time.md`, `w3_resources_tests.md`; W4: `w4_bands.md`, `w4_c18gate.md`, `w4_ata.md`, `w4_atc5.md`, `w4_resources_tests.md`; W5: `w5_tests.md`, `w5_time.md` | as named; W12 logs in `strix-002:~/dev/wt_lab/w12/logs`, W1 results in `strix-002:~/dev/wt_lab/w1/res`, W2 results in `strix-002:~/dev/wt_lab/w2/res`, W3 results in `strix-002:~/dev/wt_lab/w3/{c18,prband,attrib}`, W4 results in `strix-002:~/dev/wt_lab/w4/{prband,bands,reserve}`, W5 logs in `strix-002:~/dev/wt_rewrite_logs/W5` and timing in `strix-002:~/dev/wt_lab/w5/time` |
