# W2: HIP SPARSE_ATTN_PLAN (PLAN-v3 §13 W2)

Machine strix-002 (gfx1151), 2026-11-06, every build / test / capture / harness run under `flock ~/gpu.lock`, no other GPU user.
Raw results: `strix-002:~/dev/wt_lab/w2/res/` (harness), `strix-002:~/dev/wt_rewrite_logs/W2/` (builds, op tests, mutants).

## Rows (recapture of C3's rows)

- `harness/w2/cap.sh`: C3's own `c3_capture` binary and arguments (`EV/c3/scripts/cap.sh`: pos0 list `32768,131072`,
  `LLAMA_FN_LAST_ATTN_TRIM=0`, PR219 build read-only), text f16, random f16, text q8_0.
- Identity with C3: `test-sparse-attn-plan-oracle --shuffle 7` prints C3's per-layer statistics (valid cells per query,
  ascending rows, mean padded G4 union); at d131k they equal `EV/c3/res/*_p131072_r1.out` on **36/36 layers**, and the
  oracle verdicts are 144/144 IDENTICAL (`res/oracle_*.log`).
- Observation: a first capture with the pos0 list `131072` alone (`cap_partial/`, deleted) gave different d131k rows
  (mean G4 union differs from C3 by up to 4.6 entries per group, e.g. text layer 10: 859.4 vs 854.8). The hooked
  ubatch at pos0 32768 changes the cache the later ubatch reads; the mechanism is UNPROVEN (candidate: the eval callback
  splits the graph and changes fusions, hence numerics). The gated runs use the C3-identical capture only.
- `rowstat.cpp`: every row of the 36 layers has its valid cells strictly ascending with every invalid entry after them
  (4096/4096 rows per layer, `res/rowstat_*.out`), so production rows take the rows kernel's no-copy path.

## Harness (`harness/w2/w2_plan.cpp`, `campaign_w2.sh`, `analyze_w2.py`)

One graph of R = 8 plan nodes over the same sel on ROCm0, 3 warm-ups, 20 reps; per rep the wall time of
`ggml_backend_graph_compute` / 8 = one plan op including its three launches and the gaps between them. Per layer the
median over the reps, then the mean over the 12 layers (C3's statistic), worst layer in brackets. Each layer and G: the
whole dst of the first and last GPU node memcmp-equal to the CPU backend's dst and to the C3 oracle (`sa_ref::plan`).

| run | G4 mean [worst] ms/layer | G1 mean [worst] ms/layer | layers x G bit-exact (CPU op and C3 oracle) |
|---|---|---|---|
| rand_f16_p131072_nographs | 0.832 [1.029] | 0.604 [0.604] | 24 / 24 |
| rand_f16_p131072_r1 | 0.831 [1.026] | 0.603 [0.604] | 24 / 24 |
| rand_f16_p131072_r2 | 0.832 [1.028] | 0.602 [0.603] | 24 / 24 |
| rand_f16_p131072_shuf7 | 1.908 [2.104] | 1.715 [1.716] | 24 / 24 |
| text_f16_p131072_nographs | 0.718 [0.790] | 0.604 [0.604] | 24 / 24 |
| text_f16_p131072_r1 | 0.719 [0.789] | 0.603 [0.603] | 24 / 24 |
| text_f16_p131072_r2 | 0.719 [0.790] | 0.603 [0.604] | 24 / 24 |
| text_f16_p131072_shuf7 | 1.795 [1.860] | 1.718 [1.719] | 24 / 24 |
| text_q8_0_p131072_nographs | 0.721 [0.792] | 0.603 [0.604] | 24 / 24 |
| text_q8_0_p131072_r1 | 0.720 [0.792] | 0.603 [0.604] | 24 / 24 |
| text_q8_0_p131072_r2 | 0.719 [0.791] | 0.603 [0.604] | 24 / 24 |

**Gate (G4 mean ≤ 1.0 ms/layer on text and random, C3 d3; every layer bit-exact): PASS.** The worst random layer is
1.03 ms (layer 2); the gate is on the mean, as C3's budget check is.

Same-run C3 harness (`c3_plan`, PR-verbatim rows kernel + C3 merges, hipEvent timing; `res/c3plan_*.out`). These merges
do not write the plan's padding up to cap (39.9–45.2 MB per layer on these rows: (8204 − mean entries) x 6 B x 1024
groups), which the op must (ggml.h: every byte of
dst defined):

| rows | T0 (PR rows+merge) | P_asc_l512 (the merge W2 ports) |
|---|---|---|
| rand_f16_p131072 | 1.491 [1.942] | 0.883 [1.172] |
| text_f16_p131072 | 1.189 [1.344] | 0.732 [0.847] |

Kernel split (rocprofv3 `--kernel-trace --stats`, 576 calls each, average / min µs; `prof_final_*`):

| rows | sa_plan_merge<4, LDS> | sa_plan_rows<LDS> | sa_plan_init |
|---|---|---|---|
| text f16 | 458.2 / 370.6 | 249.3 / 151.6 | 1.0 |
| random f16 | 571.4 / 395.5 | 244.7 / 157.2 | 1.0 |

History: the first version (one load in flight per lane in the rows and merge copy loops, scalar padding stores) measured
G4 1.042 (text) / 1.188 (random) ms and failed the gate (`res_v1/`); the merge was 879.6 µs average on random rows
(`res_v1/prof1`). Unrolled row loads (4 per thread in the rows kernel, 2 per row x 4 rows in the merge copy) and
16/8-byte padding stores gave the table above, bit-exact unchanged.

## Op tests (test-backend-ops, `-o SPARSE_ATTN_PLAN`, one op per run, `-j 1`)

- ROCm0 `--fail-unsupported`: **185/185** with HIP graphs on, with `GGML_CUDA_DISABLE_GRAPHS=1`, and with
  `GGML_CUDA_SA_PLAN_GLOBAL=1` (every row and group on the global-memory kernels). Each case: whole dst memcmp vs the
  CPU backend, the C3 oracle (`verify_reference`), 2 determinism runs, and the census (exactly one `sa_plan/G<g>` and
  none of the other group).
- CPU `--fail-unsupported`: 185/185 in the HIP, Vulkan and CPU-only builds.
- 16 of the 185 cases are new in W2: n_sel 9000 (2 x 9000 x 4 B > 60 KiB of LDS), 2 streams, rows asc / perm / dup /
  mix, n_kv 4096 and 65536, G 1 and 4: the global-memory rows kernel without forcing.
- Mutants (temporary edits of sparse-attn-plan.cu, restored, md5 equal): padding quad 0 instead of the last real quad:
  149 of 185 FAIL; no dedup after the sort: 87 FAIL; n_bad not counted: 94 FAIL. Restored: 185/185.
- Before the census counted HIP graph replays, 59 cases (73 in a second run) failed with "census sa_plan/G<g>=0" on ROCm0 with graphs on
  (a replayed graph from an earlier case of the same shape dispatches nothing on the host); with graphs off, 185/185.
  After the fix the in-model census is equal with graphs on and off (llama-bench pp4096 + tg8, -r 2: `qsa_expand=267`
  both ways).
