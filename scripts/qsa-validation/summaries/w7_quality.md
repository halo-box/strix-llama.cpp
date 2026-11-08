# W7 quality gates (PLAN-v3 §15)

Source: strix-002:~/dev/wt_lab/w7/q/logs (qrun.sh arms; new = W7 lab tree, w0 = W0-lab12). Command: python3 scripts/qsa-validation/harness/w7/q_an.py ~/dev/wt_lab/w7/q/logs. Phases run: A, D, H, B, E, C, N; P partly (Q3-B q4_0 ub256 missing); S not run (Q11c missing). Q12 (phase N) ran on the lab tree at dcaa19f16 (host-KV copy default).
# W7 quality gates (PLAN-v3 §15)

## Run health and self-replay

- arms: 140; arms with rows_absent or moe_absent != 0: [('w0_p_q4_0_u256', 'NO SUMMARY')]
- ref8t0 self-replay on W0-lab: QKL_HASH equal in 8/8 chunks

## Q1 (new(T)_U - PR219(T)_U, c8192 x 8, pinned)

| arm | mean diff | SE | band | verdict |
|---|---|---|---|---|
| Q1-A f16_f16 ub4096 | +1.369e-04 | 2.88e-05 | 2.000e-04 | PASS  |
| Q1-A q8_0_q8_0 ub4096 | -2.654e-06 | 3.17e-06 | 2.000e-04 | PASS  |
| Q1-A q4_0_q4_0 ub4096 | -1.505e-05 | 4.95e-05 | 2.000e-04 | PASS  |
| Q1-A bf16_bf16 ub4096 | -4.116e-05 | 2.00e-05 | 2.000e-04 | PASS  |
| Q1-A q8_0_f16 ub4096 | -4.985e-06 | 5.81e-06 | 2.000e-04 | PASS  |
| Q1-A f16_q8_0 ub4096 | -2.314e-05 | 2.70e-05 | 2.000e-04 | PASS  |
| Q1-B f16_f16 ub512 | -2.271e-05 | 2.52e-05 | 2.030e-04 | PASS  |
| Q1-B q8_0_q8_0 ub512 | -1.199e-05 | 9.57e-06 | 2.030e-04 | PASS  |
| Q1-B q4_0_q4_0 ub512 | -7.264e-05 | 8.24e-05 | 2.030e-04 | PASS  |
| Q1-C f16_f16 ub64 | +1.778e-05 | 1.67e-05 | 2.750e-04 | PASS  |
| Q1-C f16_f16 ub64 forced G1 | -1.002e-05 | 9.39e-06 | 2.750e-04 | PASS  |
| Q1-C q8_0_q8_0 ub64 | +2.621e-06 | 9.44e-06 | 2.750e-04 | PASS  |
| Q1-C q8_0_q8_0 ub64 forced G1 | -1.794e-05 | 1.41e-05 | 2.750e-04 | PASS  |
| Q1-C q4_0_q4_0 ub64 | -1.075e-04 | 8.90e-05 | 2.750e-04 | PASS  |
| Q1-C q4_0_q4_0 ub64 forced G1 | -4.388e-05 | 1.08e-04 | 2.750e-04 | PASS  |

## Q1 direct form (KLD(new(T) || PR219(T) pinned), LAB_DEF=d3 eps 0 on both)

| type | mean KLD | SE | band | verdict |
|---|---|---|---|---|
| f16_f16 | 1.375e-04 | 2.89e-05 | 1.403e-04 | PASS |
| q8_0_q8_0 | 1.503e-04 | 1.63e-05 | 2.310e-04 | PASS |
| q4_0_q4_0 | 2.690e-03 | 3.46e-04 | 1.441e-03 | FAIL |
| bf16_bf16 | 1.802e-04 | 1.94e-05 | 1.032e-04 | FAIL |
| q8_0_f16 | 1.266e-04 | 1.53e-05 | 1.807e-04 | PASS |
| f16_q8_0 | 1.517e-04 | 2.81e-05 | 1.623e-04 | PASS |

## Q1 direct form, stage 2 (§15 two-stage rule: the same band on chunks 8-15 of the c8192 x 16 reference)

| type | mean KLD chunks 8-15 | SE | band | verdict |
|---|---|---|---|---|
| q4_0_q4_0 | 2.082e-03 | 3.15e-04 | 1.441e-03 | FAIL |
| bf16_bf16 | 1.250e-04 | 1.09e-05 | 1.032e-04 | FAIL |

## Q2 floor context (ratio to the f16 ub2048 pinned floor)

- f16 ub2048 pinned floor on W0-lab = 1.0777e-04 (plan value 1.116e-04)
- new_a_bf16_bf16_u4096: 1.3121e-04, ratio 1.218
- new_a_f16_f16_u4096: 1.3683e-04, ratio 1.270
- new_a_f16_f16_u512: 1.0619e-04, ratio 0.985
- new_a_f16_f16_u64: 2.0508e-04, ratio 1.903
- new_a_f16_f16_u64g1: 1.7728e-04, ratio 1.645
- new_a_f16_q8_0_u4096: 1.2159e-04, ratio 1.128
- new_a_q4_0_q4_0_u4096: 5.7713e-03, ratio 53.553
- new_a_q4_0_q4_0_u512: 5.7137e-03, ratio 53.019
- new_a_q4_0_q4_0_u64: 5.7629e-03, ratio 53.475
- new_a_q4_0_q4_0_u64g1: 5.8265e-03, ratio 54.065
- new_a_q8_0_f16_u4096: 1.2130e-04, ratio 1.126
- new_a_q8_0_q8_0_u4096: 1.3451e-04, ratio 1.248
- new_a_q8_0_q8_0_u512: 1.3495e-04, ratio 1.252
- new_a_q8_0_q8_0_u64: 2.2200e-04, ratio 2.060
- new_a_q8_0_q8_0_u64g1: 2.0143e-04, ratio 1.869

## Q3 (c32768 x 8, QKL_TAIL=4096)

| arm | mean diff | SE | band | verdict |
|---|---|---|---|---|
| Q3-A f16 ub4096 | +1.442e-04 | 3.52e-05 | 1.801e-04 | PASS  |
| Q3-B f16 ub256 | +2.689e-06 | 3.35e-06 | 3.340e-04 | PASS  |
| Q3-A q8_0 ub4096 | +4.456e-06 | 1.01e-05 | 1.801e-04 | PASS  |
| Q3-B q8_0 ub256 | +4.376e-06 | 8.41e-06 | 3.340e-04 | PASS  |
| Q3-A q4_0 ub4096 | -1.073e-04 | 8.33e-05 | 1.801e-04 | PASS  |
| Q3-B q4_0 ub256 | missing | | 3.340e-04 | MISSING  |

## Q4 selection agreement (c8192 x 16)

| type | new mean | w0 mean | new - w0 | bound -B_T | abs threshold | verdict |
|---|---|---|---|---|---|---|
| f16_f16 | 0.997356 | 1.000000 | -2.64e-03 (below) | -1.178e-04 | 0.99735 (margin +6.3e-06) | PASS |
| f16k_q8idx | 0.996932 | 0.998528 | -1.60e-03 (below) | -1.178e-04 | 0.99693 (margin +2.0e-06) | PASS |
| q8_0_q8_0 | 0.996737 | 0.996743 | -5.24e-06 (ok) | -1.178e-04 | 0.996625 (margin +1.1e-04) | PASS |
| q4_0_q4_0 | 0.985712 | 0.985678 | +3.44e-05 (ok) | -1.178e-04 | 0.985528 (margin +1.8e-04) | PASS |
| bf16_bf16 | 0.997256 | 0.997257 | -1.51e-06 (ok) | -1.178e-04 | 0.99714 (margin +1.2e-04) | PASS |
| q8_0_f16 | 0.996857 | 0.996860 | -2.50e-06 (ok) | -1.178e-04 | 0.996742 (margin +1.2e-04) | PASS |
| f16_q8_0 | 0.997237 | 0.997239 | -1.67e-06 (ok) | -1.178e-04 | 0.997121 (margin +1.2e-04) | PASS |

## Q8 planted defects (on - off, same type, same pins, c8192 x 8)

| arm | type | on - off | SE | LCB95 | pos chunks | rule | verdict |
|---|---|---|---|---|---|---|---|
| new d1 | f16 | +1.187e-03 | 1.96e-04 | +8.158e-04 | 8/8 | LCB95 > 7.45e-5 | PASS |
| new d2 | f16 | +5.706e-04 | 4.29e-04 | -2.420e-04 | 8/8 | sign >= 7/8, |new - PR219| <= 1.86e-3 (PR219 +7.037e-04) | PASS |
| new d3 | f16 | -2.936e-08 | 1.19e-06 | -2.277e-06 | 5/8 | report only | PASS |
| new d1 | q8_0 | +1.160e-03 | 1.88e-04 | +8.044e-04 | 8/8 | LCB95 > 7.45e-5 | PASS |
| new d2 | q8_0 | +5.744e-04 | 4.17e-04 | -2.153e-04 | 8/8 | sign >= 7/8, |new - PR219| <= 1.86e-3 (PR219 +6.080e-04) | PASS |
| new d3 | q8_0 | -1.395e-06 | 1.72e-06 | -4.646e-06 | 3/8 | report only | PASS |

## Q11a (c8192 x 8, b/ub 512, QKL_TAIL=8191, split 2048)

| type | part | mean diff | SE | band | verdict |
|---|---|---|---|---|---|
| f16_f16 | lo | +0.000e+00 | 0.00e+00 | 2.073e-04 | PASS |
| f16_f16 | hi | -1.405e-05 | 1.70e-05 | 2.030e-04 | PASS |
| f16_f16 | all | -1.053e-05 | 1.27e-05 | 2.030e-04 | PASS |
| q8_0_q8_0 | lo | -2.866e-05 | 6.04e-06 | 2.073e-04 | PASS |
| q8_0_q8_0 | hi | -1.179e-05 | 7.00e-06 | 2.030e-04 | PASS |
| q8_0_q8_0 | all | -1.601e-05 | 6.16e-06 | 2.030e-04 | PASS |
| q4_0_q4_0 | lo | +1.561e-04 | 1.51e-04 | 2.073e-04 | PASS |
| q4_0_q4_0 | hi | -8.315e-05 | 7.49e-05 | 2.030e-04 | PASS |
| q4_0_q4_0 | all | -2.335e-05 | 7.41e-05 | 2.030e-04 | PASS |
| bf16_bf16 | lo | -3.050e-05 | 4.04e-06 | 2.073e-04 | PASS |
| bf16_bf16 | hi | -5.722e-05 | 6.20e-06 | 2.030e-04 | PASS |
| bf16_bf16 | all | -5.054e-05 | 4.60e-06 | 2.030e-04 | PASS |
| q8_0_f16 | lo | -2.892e-05 | 8.87e-06 | 2.073e-04 | PASS |
| q8_0_f16 | hi | -1.239e-06 | 3.79e-06 | 2.030e-04 | PASS |
| q8_0_f16 | all | -8.157e-06 | 4.22e-06 | 2.030e-04 | PASS |
| f16_q8_0 | lo | -2.880e-05 | 5.58e-06 | 2.073e-04 | PASS |
| f16_q8_0 | hi | -1.132e-05 | 5.79e-06 | 2.030e-04 | PASS |
| f16_q8_0 | all | -1.569e-05 | 5.07e-06 | 2.030e-04 | PASS |

## Q11b (c2048 x 16, MoE pins, dense f16 b/ub 2048 reference)

| arm | mean diff | SE | band | verdict |
|---|---|---|---|---|
| Q11b f16_f16 ub512 | +0.000e+00 | 0.00e+00 | 2.030e-04 | PASS  |
| Q11b q8_0_q8_0 ub512 | -2.157e-05 | 1.40e-05 | 2.030e-04 | PASS  |
| Q11b q4_0_q4_0 ub512 | -2.656e-04 | 1.18e-04 | 2.030e-04 | PASS  |
| Q11b bf16_bf16 ub512 | -3.900e-05 | 6.06e-06 | 2.030e-04 | PASS  |
| Q11b q8_0_f16 ub512 | -3.403e-05 | 5.16e-06 | 2.030e-04 | PASS  |
| Q11b f16_q8_0 ub512 | -3.569e-05 | 5.74e-06 | 2.030e-04 | PASS  |
| Q11b f16_f16 ub16 | +0.000e+00 | 0.00e+00 | UNPROVEN (F4) | report only |
| Q11b q8_0_q8_0 ub16 | +0.000e+00 | 0.00e+00 | UNPROVEN (F4) | report only |

## Q11c (np2 split == np1, unpinned, KLD per chunk)

- q8_0 np2 rerun: missing
- q8_0 np1: missing
- q4_0 np2 rerun: missing
- q4_0 np1: missing

## Q12 host KV (c8192 x 8, pinned, new(T, -nkvo) - PR219(T, -nkvo))

| arm | mean diff | SE | band | rerun new | rerun w0 | verdict |
|---|---|---|---|---|---|---|
| -nkvo f16 ub4096 | -2.804e-05 | 3.36e-05 | 2.000e-04 | 0.00e+00 | 0.00e+00 | PASS |
| -nkvo f16 ub512 | +1.746e-05 | 1.67e-05 | 2.030e-04 | 0.00e+00 | 0.00e+00 | PASS |
| -nkvo q8_0 ub4096 | +3.484e-05 | 3.60e-05 | 2.000e-04 | 0.00e+00 | 0.00e+00 | PASS |
| -nkvo q8_0 ub512 | +1.236e-05 | 1.85e-05 | 2.030e-04 | 0.00e+00 | 0.00e+00 | PASS |

## Overall

FAILED: direct q4_0_q4_0, direct bf16_bf16, Q3-B q4_0 ub256 missing, Q11c q8_0 np2 rerun missing, Q11c q8_0 np1 missing, Q11c q4_0 np2 rerun missing, Q11c q4_0 np1 missing
