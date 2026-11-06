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
| `summaries/` | `kdecide.md`, `compact*.md`, `analysis_op.md`, `res2_summary.md`, `probe_gap.txt`, the R1 analysis outputs; W12: `w12_bands.md` (analyzer output), `w12_chunks.tsv` (every arm's per-chunk KLD, logits hashes and agreement), `w12_campaign*.out` (run lines), `w12_hookobs_campaign.out` (the aborted first pass with the round-1 r1q8 hook) | as named; W12 logs in `strix-002:~/dev/wt_lab/w12/logs` |
