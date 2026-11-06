# W3: G4 attend kernel time on the C18 rows (PLAN-v3 §13 W3 done-when) and the V-loader crossover

Harness `harness/w3/w3_attn.cu` (built by `w3_build.sh` from the worktree's kernel sources; C18's timing method:
3 warm-up rounds, 10 timed rounds, every variant launched once per round with a hipEvent pair, per-layer median, mean
over the 12 layers; r1 / r2 separate processes), fixtures `~/dev/wt_lab/c18/cap/wikiS_p4096.rows` (wiki d4k) and
`wikiL_p131072.rows` (wiki d131k) with SYN data (uniform +-1.7, C18's generator; C18: SYN/real 1.002, range
0.985-1.023). Reference in the same process: the PR-verbatim `qsa3_attn_kernel<true, 8, false>` + C18 `PACKV-f16`
(the PR's qsa_pack_v indexing). New rows: X+P = sa_attend_g4 (packed) + sa_attend_fixup launch + sa_pack_v (+ its
4-byte memset); X+X = sa_attend_g4 (direct) + fixup. Gate: best row / PR route <= C18 target x 1.04 (f16 1.0;
q8_0 1.033 / 1.017; q4_0 1.017 / 0.986; bf16 1.018 / 1.047 at d4k / d131k). Logs `strix-002:~/dev/wt_lab/w3/c18/res`.

## Committed kernel
## C18 rows (SYN data), mean of r1 / r2; ms per ub4096 call, mean of 12 per-layer medians

### wiki4k (runs 2): PR-verbatim f16 route PR-V0-f16 + PACKV-f16 = 12.6924 ms; C18 Kf16D-Vf16P + PACKV-f16 = 12.1577 ms

| type | X+P (attend+fixup + pack) | X+X direct | best | best / PR route | gate (target x 1.04) | pass | best / C18 route | per-run best / PR |
|---|---|---|---|---|---|---|---|---|
| f16 | 12.1183 | 13.7647 | X+P | 0.955 | 1.040 | yes | 0.997 | 0.955 / 0.954 |
| bf16 | 12.2175 | 14.0602 | X+P | 0.963 | 1.059 | yes | 1.005 | 0.963 / 0.962 |
| q8_0 | 12.6275 | 15.5016 | X+P | 0.995 | 1.074 | yes | 1.039 | 0.996 / 0.994 |
| q4_0 | 12.4052 | 15.3463 | X+P | 0.977 | 1.058 | yes | 1.020 | 0.978 / 0.977 |

### wiki131k (runs 2): PR-verbatim f16 route PR-V0-f16 + PACKV-f16 = 18.7657 ms; C18 Kf16D-Vf16P + PACKV-f16 = 18.1213 ms

| type | X+P (attend+fixup + pack) | X+X direct | best | best / PR route | gate (target x 1.04) | pass | best / C18 route | per-run best / PR |
|---|---|---|---|---|---|---|---|---|
| f16 | 17.9398 | 18.9430 | X+P | 0.956 | 1.040 | yes | 0.990 | 0.956 / 0.956 |
| bf16 | 18.0112 | 19.2760 | X+P | 0.960 | 1.089 | yes | 0.994 | 0.960 / 0.960 |
| q8_0 | 17.5564 | 20.1963 | X+P | 0.936 | 1.058 | yes | 0.969 | 0.935 / 0.936 |
| q4_0 | 17.1381 | 20.0449 | X+P | 0.913 | 1.025 | yes | 0.946 | 0.913 / 0.914 |

C18 KERNEL-TIME GATE: PASS

## V-loader crossover (SYN plans, C6/C18 generator U 1.8; packed = attend + pack, direct = attend; ms)

| n_q | n_kv | 64*n_q | rule | f16 P / D / D/P | bf16 P / D / D/P | q8_0 P / D / D/P | q4_0 P / D / D/P |
|---|---|---|---|---|---|---|---|
| 512 | 16384 | 32768 | packed | 2.33 / 2.53 / 1.085 | 2.40 / 2.56 / 1.065 | 2.41 / 2.85 / 1.182 | 2.38 / 2.87 / 1.205 |
| 512 | 32768 | 32768 | direct | 2.64 / 2.52 / 0.952 | 2.69 / 2.56 / 0.953 | 2.48 / 2.95 / 1.190 | 2.46 / 2.93 / 1.190 |
| 512 | 65536 | 32768 | direct | 3.24 / 2.81 / 0.868 | 3.22 / 2.87 / 0.890 | 2.84 / 2.95 / 1.039 | 2.76 / 2.92 / 1.058 |
| 512 | 131072 | 32768 | direct | 5.33 / 4.73 / 0.888 | 5.38 / 4.73 / 0.880 | 3.51 / 3.01 / 0.857 | 3.27 / 2.90 / 0.889 |
| 512 | 262144 | 32768 | direct | 8.11 / 5.90 / 0.728 | 8.11 / 5.91 / 0.728 | 5.18 / 3.11 / 0.602 | 4.49 / 2.95 / 0.656 |
| 512 | 1048576 | 32768 | direct | 19.57 / 6.91 / 0.353 | 19.57 / 6.95 / 0.355 | 12.88 / 3.66 / 0.284 | 10.81 / 3.08 / 0.285 |
| 1024 | 16384 | 65536 | packed | 4.23 / 4.66 / 1.101 | 4.31 / 4.71 / 1.092 | 4.33 / 5.26 / 1.214 | 4.32 / 5.27 / 1.221 |
| 1024 | 32768 | 65536 | packed | 4.56 / 4.66 / 1.022 | 4.61 / 4.74 / 1.028 | 4.49 / 5.29 / 1.178 | 4.38 / 5.30 / 1.212 |
| 1024 | 65536 | 65536 | direct | 5.19 / 5.49 / 1.059 | 5.18 / 5.48 / 1.057 | 4.76 / 5.42 / 1.139 | 4.68 / 5.31 / 1.133 |
| 1024 | 131072 | 65536 | direct | 8.35 / 10.28 / 1.231 | 8.27 / 10.40 / 1.258 | 5.45 / 5.35 / 0.980 | 5.30 / 5.31 / 1.002 |
| 1024 | 262144 | 65536 | direct | 12.77 / 13.12 / 1.028 | 12.78 / 13.20 / 1.032 | 7.56 / 5.50 / 0.727 | 6.55 / 5.29 / 0.808 |
| 1024 | 1048576 | 65536 | direct | 25.77 / 15.85 / 0.615 | 25.81 / 16.05 / 0.622 | 16.19 / 6.50 / 0.401 | 13.71 / 5.39 / 0.393 |
| 4096 | 16384 | 262144 | packed | 15.20 / 17.10 / 1.125 | 15.35 / 17.59 / 1.146 | 15.89 / 19.40 / 1.221 | 15.54 / 19.21 / 1.237 |
| 4096 | 32768 | 262144 | packed | 15.84 / 17.32 / 1.093 | 16.00 / 17.74 / 1.109 | 16.08 / 19.55 / 1.215 | 15.73 / 19.37 / 1.231 |
| 4096 | 65536 | 262144 | packed | 17.09 / 23.30 / 1.364 | 17.09 / 23.44 / 1.372 | 16.60 / 19.72 / 1.188 | 16.31 / 19.52 / 1.197 |
| 4096 | 131072 | 262144 | packed | 30.08 / 45.42 / 1.510 | 30.97 / 46.93 / 1.515 | 17.64 / 20.01 / 1.135 | 17.15 / 19.59 / 1.142 |
| 4096 | 262144 | 262144 | direct | 41.96 / 56.70 / 1.351 | 42.44 / 57.06 / 1.345 | 23.10 / 20.42 / 0.884 | 20.11 / 19.74 / 0.982 |
| 4096 | 1048576 | 262144 | direct | 62.73 / 64.98 / 1.036 | 62.83 / 65.95 / 1.050 | 37.38 / 24.76 / 0.662 | 31.58 / 20.09 / 0.636 |


The CAPPED rule (packed iff n_kv < 64 * n_q) against these points: for q8_0 / q4_0 direct V is slower than packed at
n_kv = 64 n_q for n_q 512 and 1024 (D/P 1.19 / 1.19 and 1.14 / 1.13) and wins from 256 n_q at n_q 512 (0.86 / 0.89),
from 128 n_q at n_q 1024 (0.98 / 1.00) and from 64 n_q at n_q 4096 (0.88 / 0.98); for f16 / bf16 the direct split-gather V is slower than packed + pack at n_q 1024 / 131072 (1.23 / 1.26)
and n_q 4096 / 262144 (1.35 / 1.35), at 1M it wins only for n_q <= 1024. Recorded for AT-O (owner decision deferred);
the trait is unchanged (default CAPPED for every type).

## Attribution lab (`harness/w3/w3_attrib.sh`, layers 0, 5, 11, 10 reps; first kernel version, ms)

Ship = the first version (non-finite V check in the WMMA loop for packed V too), nocheck = that check compiled out,
nofix = the fixup launch removed. The check cost ~4 % of the packed kernel (12.90 -> 12.41 ms d4k, 17.12 -> 16.52 ms
d131k, f16); the fixup launch costs nothing measurable. The committed kernel moves the packed-V check into the pack.

```
wikiL_p131072_nocheck.out C18-Kf16D-Vf16P=16.6420 NEW-f16-P=16.5153 NEW-q8_0-P=16.6503 PACKV-f16=1.8141 PR-V0-f16=17.1998 
wikiL_p131072_nofix.out C18-Kf16D-Vf16P=16.5756 NEW-f16-P=17.1022 NEW-q8_0-P=17.3157 PACKV-f16=1.8129 PR-V0-f16=17.1433 
wikiL_p131072_ship.out C18-Kf16D-Vf16P=16.5742 NEW-f16-P=17.1156 NEW-q8_0-P=17.3207 PACKV-f16=1.8131 PR-V0-f16=17.1380 
wikiS_p4096_nocheck.out C18-Kf16D-Vf16P=12.4494 NEW-f16-P=12.4111 NEW-q8_0-P=13.0335 PACKV-f16=0.0927 PR-V0-f16=12.9892 
wikiS_p4096_nofix.out C18-Kf16D-Vf16P=12.4846 NEW-f16-P=13.0105 NEW-q8_0-P=13.6526 PACKV-f16=0.0927 PR-V0-f16=12.9991 
wikiS_p4096_ship.out C18-Kf16D-Vf16P=12.3345 NEW-f16-P=12.9043 NEW-q8_0-P=13.5095 PACKV-f16=0.0927 PR-V0-f16=12.8759 
```

## First kernel version (same campaign, for the record)
### wiki4k (runs 2): PR-verbatim f16 route PR-V0-f16 + PACKV-f16 = 12.6905 ms; C18 Kf16D-Vf16P + PACKV-f16 = 12.1588 ms

| type | X+P (attend+fixup + pack) | X+X direct | best | best / PR route | gate (target x 1.04) | pass | best / C18 route | per-run best / PR |
|---|---|---|---|---|---|---|---|---|
| f16 | 12.6740 | 13.7377 | X+P | 0.999 | 1.040 | yes | 1.042 | 0.999 / 0.998 |
| bf16 | 12.8225 | 14.0328 | X+P | 1.010 | 1.059 | yes | 1.055 | 1.011 / 1.010 |
| q8_0 | 13.2042 | 15.4668 | X+P | 1.040 | 1.074 | yes | 1.086 | 1.042 / 1.039 |
| q4_0 | 12.9328 | 15.3078 | X+P | 1.019 | 1.058 | yes | 1.064 | 1.019 / 1.019 |

### wiki131k (runs 2): PR-verbatim f16 route PR-V0-f16 + PACKV-f16 = 18.7985 ms; C18 Kf16D-Vf16P + PACKV-f16 = 18.1512 ms

| type | X+P (attend+fixup + pack) | X+X direct | best | best / PR route | gate (target x 1.04) | pass | best / C18 route | per-run best / PR |
|---|---|---|---|---|---|---|---|---|
| f16 | 18.7203 | 18.9661 | X+P | 0.996 | 1.040 | yes | 1.031 | 0.996 / 0.996 |
| bf16 | 18.8334 | 19.2866 | X+P | 1.002 | 1.089 | yes | 1.038 | 1.002 / 1.002 |
| q8_0 | 18.3469 | 20.2073 | X+P | 0.976 | 1.058 | yes | 1.011 | 0.976 / 0.976 |
| q4_0 | 17.8503 | 20.0386 | X+P | 0.950 | 1.025 | yes | 0.983 | 0.949 / 0.950 |

C18 KERNEL-TIME GATE: PASS

## Correctness lines of the committed-kernel campaign (new kernel vs PR-V0-f16 output: f16 nmse <= 2.1e-12; other types differ by their own quantization; packed vs direct of one type bitwise equal)

syn_1024_1048576.out NEW-bf16-D:nmse_max=5.31e-06,PvsD_words+nan=0 NEW-bf16-P:nmse_max=5.31e-06,PvsD_words+nan=0 NEW-f16-D:nmse_max=1.00e-12,PvsD_words+nan=0 NEW-f16-P:nmse_max=1.00e-12,PvsD_words+nan=0 NEW-q4_0-D:nmse_max=8.17e-03,PvsD_words+nan=0 NEW-q4_0-P:nmse_max=8.17e-03,PvsD_words+nan=0 NEW-q8_0-D:nmse_max=2.74e-05,PvsD_words+nan=0 NEW-q8_0-P:nmse_max=2.74e-05,PvsD_words+nan=0
syn_1024_131072.out NEW-bf16-D:nmse_max=5.30e-06,PvsD_words+nan=0 NEW-bf16-P:nmse_max=5.30e-06,PvsD_words+nan=0 NEW-f16-D:nmse_max=2.12e-12,PvsD_words+nan=0 NEW-f16-P:nmse_max=2.12e-12,PvsD_words+nan=0 NEW-q4_0-D:nmse_max=8.19e-03,PvsD_words+nan=0 NEW-q4_0-P:nmse_max=8.19e-03,PvsD_words+nan=0 NEW-q8_0-D:nmse_max=2.75e-05,PvsD_words+nan=0 NEW-q8_0-P:nmse_max=2.75e-05,PvsD_words+nan=0
syn_1024_16384.out NEW-bf16-D:nmse_max=5.32e-06,PvsD_words+nan=0 NEW-bf16-P:nmse_max=5.32e-06,PvsD_words+nan=0 NEW-f16-D:nmse_max=1.12e-12,PvsD_words+nan=0 NEW-f16-P:nmse_max=1.12e-12,PvsD_words+nan=0 NEW-q4_0-D:nmse_max=8.22e-03,PvsD_words+nan=0 NEW-q4_0-P:nmse_max=8.22e-03,PvsD_words+nan=0 NEW-q8_0-D:nmse_max=2.76e-05,PvsD_words+nan=0 NEW-q8_0-P:nmse_max=2.76e-05,PvsD_words+nan=0
syn_1024_262144.out NEW-bf16-D:nmse_max=5.31e-06,PvsD_words+nan=0 NEW-bf16-P:nmse_max=5.31e-06,PvsD_words+nan=0 NEW-f16-D:nmse_max=1.33e-12,PvsD_words+nan=0 NEW-f16-P:nmse_max=1.33e-12,PvsD_words+nan=0 NEW-q4_0-D:nmse_max=8.19e-03,PvsD_words+nan=0 NEW-q4_0-P:nmse_max=8.19e-03,PvsD_words+nan=0 NEW-q8_0-D:nmse_max=2.74e-05,PvsD_words+nan=0 NEW-q8_0-P:nmse_max=2.74e-05,PvsD_words+nan=0
syn_1024_32768.out NEW-bf16-D:nmse_max=5.32e-06,PvsD_words+nan=0 NEW-bf16-P:nmse_max=5.32e-06,PvsD_words+nan=0 NEW-f16-D:nmse_max=8.60e-13,PvsD_words+nan=0 NEW-f16-P:nmse_max=8.60e-13,PvsD_words+nan=0 NEW-q4_0-D:nmse_max=8.21e-03,PvsD_words+nan=0 NEW-q4_0-P:nmse_max=8.21e-03,PvsD_words+nan=0 NEW-q8_0-D:nmse_max=2.75e-05,PvsD_words+nan=0 NEW-q8_0-P:nmse_max=2.75e-05,PvsD_words+nan=0
syn_1024_65536.out NEW-bf16-D:nmse_max=5.31e-06,PvsD_words+nan=0 NEW-bf16-P:nmse_max=5.31e-06,PvsD_words+nan=0 NEW-f16-D:nmse_max=2.12e-12,PvsD_words+nan=0 NEW-f16-P:nmse_max=2.12e-12,PvsD_words+nan=0 NEW-q4_0-D:nmse_max=8.20e-03,PvsD_words+nan=0 NEW-q4_0-P:nmse_max=8.20e-03,PvsD_words+nan=0 NEW-q8_0-D:nmse_max=2.75e-05,PvsD_words+nan=0 NEW-q8_0-P:nmse_max=2.75e-05,PvsD_words+nan=0
syn_4096_1048576.out NEW-bf16-D:nmse_max=5.30e-06,PvsD_words+nan=0 NEW-bf16-P:nmse_max=5.30e-06,PvsD_words+nan=0 NEW-f16-D:nmse_max=1.14e-12,PvsD_words+nan=0 NEW-f16-P:nmse_max=1.14e-12,PvsD_words+nan=0 NEW-q4_0-D:nmse_max=8.17e-03,PvsD_words+nan=0 NEW-q4_0-P:nmse_max=8.17e-03,PvsD_words+nan=0 NEW-q8_0-D:nmse_max=2.74e-05,PvsD_words+nan=0 NEW-q8_0-P:nmse_max=2.74e-05,PvsD_words+nan=0
syn_4096_131072.out NEW-bf16-D:nmse_max=5.31e-06,PvsD_words+nan=0 NEW-bf16-P:nmse_max=5.31e-06,PvsD_words+nan=0 NEW-f16-D:nmse_max=1.32e-12,PvsD_words+nan=0 NEW-f16-P:nmse_max=1.32e-12,PvsD_words+nan=0 NEW-q4_0-D:nmse_max=8.19e-03,PvsD_words+nan=0 NEW-q4_0-P:nmse_max=8.19e-03,PvsD_words+nan=0 NEW-q8_0-D:nmse_max=2.75e-05,PvsD_words+nan=0 NEW-q8_0-P:nmse_max=2.75e-05,PvsD_words+nan=0
syn_4096_16384.out NEW-bf16-D:nmse_max=5.32e-06,PvsD_words+nan=0 NEW-bf16-P:nmse_max=5.32e-06,PvsD_words+nan=0 NEW-f16-D:nmse_max=1.29e-12,PvsD_words+nan=0 NEW-f16-P:nmse_max=1.29e-12,PvsD_words+nan=0 NEW-q4_0-D:nmse_max=8.23e-03,PvsD_words+nan=0 NEW-q4_0-P:nmse_max=8.23e-03,PvsD_words+nan=0 NEW-q8_0-D:nmse_max=2.76e-05,PvsD_words+nan=0 NEW-q8_0-P:nmse_max=2.76e-05,PvsD_words+nan=0
syn_4096_262144.out NEW-bf16-D:nmse_max=5.30e-06,PvsD_words+nan=0 NEW-bf16-P:nmse_max=5.30e-06,PvsD_words+nan=0 NEW-f16-D:nmse_max=1.26e-12,PvsD_words+nan=0 NEW-f16-P:nmse_max=1.26e-12,PvsD_words+nan=0 NEW-q4_0-D:nmse_max=8.19e-03,PvsD_words+nan=0 NEW-q4_0-P:nmse_max=8.19e-03,PvsD_words+nan=0 NEW-q8_0-D:nmse_max=2.74e-05,PvsD_words+nan=0 NEW-q8_0-P:nmse_max=2.74e-05,PvsD_words+nan=0
syn_4096_32768.out NEW-bf16-D:nmse_max=5.32e-06,PvsD_words+nan=0 NEW-bf16-P:nmse_max=5.32e-06,PvsD_words+nan=0 NEW-f16-D:nmse_max=1.11e-12,PvsD_words+nan=0 NEW-f16-P:nmse_max=1.11e-12,PvsD_words+nan=0 NEW-q4_0-D:nmse_max=8.20e-03,PvsD_words+nan=0 NEW-q4_0-P:nmse_max=8.20e-03,PvsD_words+nan=0 NEW-q8_0-D:nmse_max=2.75e-05,PvsD_words+nan=0 NEW-q8_0-P:nmse_max=2.75e-05,PvsD_words+nan=0
syn_4096_65536.out NEW-bf16-D:nmse_max=5.30e-06,PvsD_words+nan=0 NEW-bf16-P:nmse_max=5.30e-06,PvsD_words+nan=0 NEW-f16-D:nmse_max=1.43e-12,PvsD_words+nan=0 NEW-f16-P:nmse_max=1.43e-12,PvsD_words+nan=0 NEW-q4_0-D:nmse_max=8.20e-03,PvsD_words+nan=0 NEW-q4_0-P:nmse_max=8.20e-03,PvsD_words+nan=0 NEW-q8_0-D:nmse_max=2.75e-05,PvsD_words+nan=0 NEW-q8_0-P:nmse_max=2.75e-05,PvsD_words+nan=0
syn_512_1048576.out NEW-bf16-D:nmse_max=5.30e-06,PvsD_words+nan=0 NEW-bf16-P:nmse_max=5.30e-06,PvsD_words+nan=0 NEW-f16-D:nmse_max=8.31e-13,PvsD_words+nan=0 NEW-f16-P:nmse_max=8.31e-13,PvsD_words+nan=0 NEW-q4_0-D:nmse_max=8.17e-03,PvsD_words+nan=0 NEW-q4_0-P:nmse_max=8.17e-03,PvsD_words+nan=0 NEW-q8_0-D:nmse_max=2.74e-05,PvsD_words+nan=0 NEW-q8_0-P:nmse_max=2.74e-05,PvsD_words+nan=0
syn_512_131072.out NEW-bf16-D:nmse_max=5.30e-06,PvsD_words+nan=0 NEW-bf16-P:nmse_max=5.30e-06,PvsD_words+nan=0 NEW-f16-D:nmse_max=1.88e-12,PvsD_words+nan=0 NEW-f16-P:nmse_max=1.88e-12,PvsD_words+nan=0 NEW-q4_0-D:nmse_max=8.18e-03,PvsD_words+nan=0 NEW-q4_0-P:nmse_max=8.18e-03,PvsD_words+nan=0 NEW-q8_0-D:nmse_max=2.74e-05,PvsD_words+nan=0 NEW-q8_0-P:nmse_max=2.74e-05,PvsD_words+nan=0
syn_512_16384.out NEW-bf16-D:nmse_max=5.33e-06,PvsD_words+nan=0 NEW-bf16-P:nmse_max=5.33e-06,PvsD_words+nan=0 NEW-f16-D:nmse_max=4.35e-13,PvsD_words+nan=0 NEW-f16-P:nmse_max=4.35e-13,PvsD_words+nan=0 NEW-q4_0-D:nmse_max=8.22e-03,PvsD_words+nan=0 NEW-q4_0-P:nmse_max=8.22e-03,PvsD_words+nan=0 NEW-q8_0-D:nmse_max=2.76e-05,PvsD_words+nan=0 NEW-q8_0-P:nmse_max=2.76e-05,PvsD_words+nan=0
syn_512_262144.out NEW-bf16-D:nmse_max=5.30e-06,PvsD_words+nan=0 NEW-bf16-P:nmse_max=5.30e-06,PvsD_words+nan=0 NEW-f16-D:nmse_max=1.49e-12,PvsD_words+nan=0 NEW-f16-P:nmse_max=1.49e-12,PvsD_words+nan=0 NEW-q4_0-D:nmse_max=8.18e-03,PvsD_words+nan=0 NEW-q4_0-P:nmse_max=8.18e-03,PvsD_words+nan=0 NEW-q8_0-D:nmse_max=2.74e-05,PvsD_words+nan=0 NEW-q8_0-P:nmse_max=2.74e-05,PvsD_words+nan=0
syn_512_32768.out NEW-bf16-D:nmse_max=5.31e-06,PvsD_words+nan=0 NEW-bf16-P:nmse_max=5.31e-06,PvsD_words+nan=0 NEW-f16-D:nmse_max=1.06e-12,PvsD_words+nan=0 NEW-f16-P:nmse_max=1.06e-12,PvsD_words+nan=0 NEW-q4_0-D:nmse_max=8.20e-03,PvsD_words+nan=0 NEW-q4_0-P:nmse_max=8.20e-03,PvsD_words+nan=0 NEW-q8_0-D:nmse_max=2.75e-05,PvsD_words+nan=0 NEW-q8_0-P:nmse_max=2.75e-05,PvsD_words+nan=0
syn_512_65536.out NEW-bf16-D:nmse_max=5.31e-06,PvsD_words+nan=0 NEW-bf16-P:nmse_max=5.31e-06,PvsD_words+nan=0 NEW-f16-D:nmse_max=2.13e-12,PvsD_words+nan=0 NEW-f16-P:nmse_max=2.13e-12,PvsD_words+nan=0 NEW-q4_0-D:nmse_max=8.19e-03,PvsD_words+nan=0 NEW-q4_0-P:nmse_max=8.19e-03,PvsD_words+nan=0 NEW-q8_0-D:nmse_max=2.75e-05,PvsD_words+nan=0 NEW-q8_0-P:nmse_max=2.75e-05,PvsD_words+nan=0
wiki131k_r1.out NEW-bf16-D:nmse_max=5.39e-06,PvsD_words+nan=0 NEW-bf16-P:nmse_max=5.39e-06,PvsD_words+nan=0 NEW-f16-D:nmse_max=1.56e-12,PvsD_words+nan=0 NEW-f16-P:nmse_max=1.56e-12,PvsD_words+nan=0 NEW-q4_0-D:nmse_max=8.30e-03,PvsD_words+nan=0 NEW-q4_0-P:nmse_max=8.30e-03,PvsD_words+nan=0 NEW-q8_0-D:nmse_max=2.79e-05,PvsD_words+nan=0 NEW-q8_0-P:nmse_max=2.79e-05,PvsD_words+nan=0
wiki131k_r2.out NEW-bf16-D:nmse_max=5.39e-06,PvsD_words+nan=0 NEW-bf16-P:nmse_max=5.39e-06,PvsD_words+nan=0 NEW-f16-D:nmse_max=1.56e-12,PvsD_words+nan=0 NEW-f16-P:nmse_max=1.56e-12,PvsD_words+nan=0 NEW-q4_0-D:nmse_max=8.30e-03,PvsD_words+nan=0 NEW-q4_0-P:nmse_max=8.30e-03,PvsD_words+nan=0 NEW-q8_0-D:nmse_max=2.79e-05,PvsD_words+nan=0 NEW-q8_0-P:nmse_max=2.79e-05,PvsD_words+nan=0
wiki4k_r1.out NEW-bf16-D:nmse_max=5.37e-06,PvsD_words+nan=0 NEW-bf16-P:nmse_max=5.37e-06,PvsD_words+nan=0 NEW-f16-D:nmse_max=1.34e-12,PvsD_words+nan=0 NEW-f16-P:nmse_max=1.34e-12,PvsD_words+nan=0 NEW-q4_0-D:nmse_max=8.31e-03,PvsD_words+nan=0 NEW-q4_0-P:nmse_max=8.31e-03,PvsD_words+nan=0 NEW-q8_0-D:nmse_max=2.78e-05,PvsD_words+nan=0 NEW-q8_0-P:nmse_max=2.78e-05,PvsD_words+nan=0
wiki4k_r2.out NEW-bf16-D:nmse_max=5.37e-06,PvsD_words+nan=0 NEW-bf16-P:nmse_max=5.37e-06,PvsD_words+nan=0 NEW-f16-D:nmse_max=1.34e-12,PvsD_words+nan=0 NEW-f16-P:nmse_max=1.34e-12,PvsD_words+nan=0 NEW-q4_0-D:nmse_max=8.31e-03,PvsD_words+nan=0 NEW-q4_0-P:nmse_max=8.31e-03,PvsD_words+nan=0 NEW-q8_0-D:nmse_max=2.78e-05,PvsD_words+nan=0 NEW-q8_0-P:nmse_max=2.78e-05,PvsD_words+nan=0
