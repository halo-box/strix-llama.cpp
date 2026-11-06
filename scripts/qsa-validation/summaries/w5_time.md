# W5 op time per K = V type (recorded, not gated: PLAN-v3 §13 W5, U8, P4 / P7)

Harness `harness/w5/w5_time.cpp`, campaign `time_runs.sh`, summary `time_an.py`, raw `strix-002:~/dev/wt_lab/w5/time/res`, binary at 431fd7def. Rows: C18 wiki d4k (`wikiS_p4096`: n_q 4096, n_sel 2051, n_kv 8192) and d131k (`wikiL_p131072`: n_q 4096, n_sel 2051, n_kv 135168); SYN data (uniform +-1.7), GQA 12, 2 kv heads, plan G = 4. The time is the wall time of ggml_backend_graph_compute: attend = (median(plan + R attends) - median(plan)) / R, R = 4, reps 10 (portable R = 1, reps 3, layers 0, 5, 11), two processes per arm. It is not the C18 kernel-level protocol of the W3 / W4 rows, so these numbers are not compared with those rows.

Arms: `default` = the backend's choice at these shapes: packed V (n_kv < 64 * n_q), the fast tier for S_fast and the generic tier for S_generic. `generic` = GGML_CUDA_SA_FORCE=generic. `vdirect` = GGML_CUDA_SA_FORCE=vdirect (direct V, the route the CAPPED rule takes at n_kv >= 64 * n_q). `portable` = GGML_CUDA_SA_FORCE=portable on gfx1151; its speed on other archs stays UNPROVEN (U15).

| rows | arm | type | layers | attend ms r1 | attend ms r2 | r2/r1 | / default-arm f16 same layers (r1, r2) | / same-arm f16 (r1, r2) |
|---|---|---|---|---|---|---|---|---|
| wikiL_p131072 | default | f16 | 12 | 17.810 | 17.809 | 1.000 | 1.000, 1.000 | 1.000, 1.000 |
| wikiL_p131072 | default | bf16 | 12 | 17.993 | 17.982 | 0.999 | 1.010, 1.010 | 1.010, 1.010 |
| wikiL_p131072 | default | q8_0 | 12 | 17.704 | 17.728 | 1.001 | 0.994, 0.995 | 0.994, 0.995 |
| wikiL_p131072 | default | q4_0 | 12 | 17.148 | 17.158 | 1.001 | 0.963, 0.963 | 0.963, 0.963 |
| wikiL_p131072 | default | f32 | 12 | 25.266 | 25.244 | 0.999 | 1.419, 1.417 | 1.419, 1.417 |
| wikiL_p131072 | default | q4_1 | 12 | 18.136 | 18.158 | 1.001 | 1.018, 1.020 | 1.018, 1.020 |
| wikiL_p131072 | default | q5_0 | 12 | 19.227 | 19.232 | 1.000 | 1.080, 1.080 | 1.080, 1.080 |
| wikiL_p131072 | default | q5_1 | 12 | 18.853 | 18.823 | 0.998 | 1.059, 1.057 | 1.059, 1.057 |
| wikiL_p131072 | default | iq4_nl | 12 | 19.665 | 19.662 | 1.000 | 1.104, 1.104 | 1.104, 1.104 |
| wikiL_p131072 | generic | f16 | 12 | 18.289 | 18.307 | 1.001 | 1.027, 1.028 | 1.000, 1.000 |
| wikiL_p131072 | generic | q8_0 | 12 | 18.408 | 18.434 | 1.001 | 1.034, 1.035 | 1.007, 1.007 |
| wikiL_p131072 | vdirect | f16 | 12 | 18.993 | 18.958 | 0.998 | 1.066, 1.065 | 1.000, 1.000 |
| wikiL_p131072 | vdirect | bf16 | 12 | 19.368 | 19.344 | 0.999 | 1.088, 1.086 | 1.020, 1.020 |
| wikiL_p131072 | vdirect | q8_0 | 12 | 20.481 | 20.477 | 1.000 | 1.150, 1.150 | 1.078, 1.080 |
| wikiL_p131072 | vdirect | q4_0 | 12 | 20.230 | 20.234 | 1.000 | 1.136, 1.136 | 1.065, 1.067 |
| wikiL_p131072 | vdirect | f32 | 12 | 42.896 | 42.944 | 1.001 | 2.409, 2.411 | 2.258, 2.265 |
| wikiL_p131072 | vdirect | q4_1 | 12 | 26.174 | 26.172 | 1.000 | 1.470, 1.470 | 1.378, 1.381 |
| wikiL_p131072 | vdirect | q5_0 | 12 | 28.740 | 28.711 | 0.999 | 1.614, 1.612 | 1.513, 1.514 |
| wikiL_p131072 | vdirect | q5_1 | 12 | 27.964 | 27.928 | 0.999 | 1.570, 1.568 | 1.472, 1.473 |
| wikiL_p131072 | vdirect | iq4_nl | 12 | 29.533 | 29.530 | 1.000 | 1.658, 1.658 | 1.555, 1.558 |
| wikiL_p131072 | portable | f16 | 3 | 227.547 | 227.549 | 1.000 | 12.462, 12.504 | 1.000, 1.000 |
| wikiL_p131072 | portable | q8_0 | 3 | 221.073 | 221.330 | 1.001 | 12.108, 12.162 | 0.972, 0.973 |
| wikiS_p4096 | default | f16 | 12 | 12.330 | 12.497 | 1.014 | 1.000, 1.000 | 1.000, 1.000 |
| wikiS_p4096 | default | bf16 | 12 | 12.244 | 12.319 | 1.006 | 0.993, 0.986 | 0.993, 0.986 |
| wikiS_p4096 | default | q8_0 | 12 | 12.873 | 12.968 | 1.007 | 1.044, 1.038 | 1.044, 1.038 |
| wikiS_p4096 | default | q4_0 | 12 | 12.643 | 12.769 | 1.010 | 1.025, 1.022 | 1.025, 1.022 |
| wikiS_p4096 | default | f32 | 12 | 13.185 | 13.257 | 1.005 | 1.069, 1.061 | 1.069, 1.061 |
| wikiS_p4096 | default | q4_1 | 12 | 13.062 | 13.174 | 1.009 | 1.059, 1.054 | 1.059, 1.054 |
| wikiS_p4096 | default | q5_0 | 12 | 14.014 | 14.041 | 1.002 | 1.137, 1.124 | 1.137, 1.124 |
| wikiS_p4096 | default | q5_1 | 12 | 13.899 | 14.036 | 1.010 | 1.127, 1.123 | 1.127, 1.123 |
| wikiS_p4096 | default | iq4_nl | 12 | 14.248 | 14.302 | 1.004 | 1.156, 1.144 | 1.156, 1.144 |
| wikiS_p4096 | generic | f16 | 12 | 12.765 | 12.720 | 0.996 | 1.035, 1.018 | 1.000, 1.000 |
| wikiS_p4096 | generic | q8_0 | 12 | 13.617 | 13.653 | 1.003 | 1.104, 1.093 | 1.067, 1.073 |
| wikiS_p4096 | vdirect | f16 | 12 | 14.309 | 14.385 | 1.005 | 1.161, 1.151 | 1.000, 1.000 |
| wikiS_p4096 | vdirect | bf16 | 12 | 14.612 | 14.749 | 1.009 | 1.185, 1.180 | 1.021, 1.025 |
| wikiS_p4096 | vdirect | q8_0 | 12 | 16.207 | 16.332 | 1.008 | 1.314, 1.307 | 1.133, 1.135 |
| wikiS_p4096 | vdirect | q4_0 | 12 | 16.247 | 16.385 | 1.008 | 1.318, 1.311 | 1.135, 1.139 |
| wikiS_p4096 | vdirect | f32 | 12 | 21.610 | 21.648 | 1.002 | 1.753, 1.732 | 1.510, 1.505 |
| wikiS_p4096 | vdirect | q4_1 | 12 | 19.911 | 19.986 | 1.004 | 1.615, 1.599 | 1.392, 1.389 |
| wikiS_p4096 | vdirect | q5_0 | 12 | 21.976 | 22.145 | 1.008 | 1.782, 1.772 | 1.536, 1.539 |
| wikiS_p4096 | vdirect | q5_1 | 12 | 21.572 | 21.707 | 1.006 | 1.750, 1.737 | 1.508, 1.509 |
| wikiS_p4096 | vdirect | iq4_nl | 12 | 22.106 | 22.124 | 1.001 | 1.793, 1.770 | 1.545, 1.538 |
| wikiS_p4096 | portable | f16 | 3 | 213.180 | 213.088 | 1.000 | 17.149, 16.902 | 1.000, 1.000 |
| wikiS_p4096 | portable | q8_0 | 3 | 210.887 | 211.078 | 1.001 | 16.965, 16.743 | 0.989, 0.991 |

## Reading (recorded, not gated; P4 itself is an end-to-end gate on the W7 build and is not decided here)

Time ratios per run are stable: r2/r1 is within 1.4 % in every row above.

- **Runtime-switch overhead alone:** f16 through the generic tier against f16 through the fast tier: 1.027 / 1.028 at d131k, 1.035 / 1.018 at d4k. q8_0 likewise: 18.41 / 17.70 ms at d131k, 13.62 / 12.87 ms at d4k.
- **Default arm, S_generic against f16:** at d4k f32 1.07, q4_1 1.06, q5_1 1.13, q5_0 1.13, iq4_nl 1.15; at d131k q4_1 1.02, q5_1 1.06, q5_0 1.08, iq4_nl 1.10, **f32 1.42**.
- **Direct V, against f16 direct V (same arm):** at d131k q4_1 1.38, q5_1 1.47, q5_0 1.51, iq4_nl 1.56, **f32 2.26**; at d4k q4_1 1.39, q5_1 1.51, f32 1.51, q5_0 1.54, iq4_nl 1.54.
- **P4 signal:** P4 needs new_T / new_f16 >= 0.90, i.e. a time ratio <= 1.11, and lets an S_generic pair below it ship listed as "not first-class". At op level, on the default arm, f32 at d131k and q5_0, q5_1, iq4_nl at d4k are past 1.11; on the direct-V route every S_generic type is. Whether they miss P4 end to end is UNPROVEN until the P4 cells run.
- **Portable on gfx1151 (forced):** 12.1 to 12.5 x default f16 at d131k, 16.7 to 17.1 x at d4k, on layers 0, 5, 11. This is the only portable speed measured; on real non-RDNA3.5 hardware it stays UNPROVEN (U15).
