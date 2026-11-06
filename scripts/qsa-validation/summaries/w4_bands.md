# W4 band rows (PLAN-v3 §14.2): PR-verbatim decode on UNI and TBO, enlarged n_q 1 SYN fixture

Results: strix-002:~/dev/wt_lab/w4/prband/{uni,tbo}, ~/dev/wt_lab/w4/bands/syn. PR-verbatim = R2-BOP bands_dec (PR-dec-<T> = qsa_decode_wmma_partial + qsa_decode_merge verbatim).

## UNI fixtures (test-backend-ops inputs, PR cell-formula rows), split classes 6-16 and > 33, 3 seeds, f16 + q8_0

| class | splits | n_q | n_sel | n_kv | samples | NMSE p100 (PR-dec) | 2 x p100 | defect Dmin | k = Dmin / p100 |
|---|---|---|---|---|---|---|---|---|---|
| 34+ | 40 | 1 | 2560 | 16384 | 6 | 7.611e-08 | 1.522e-07 | 1.041e-06 | 13.7 |
| 34+ | 52 | 9 | 4096 | 16384 | 6 | 7.664e-08 | 1.533e-07 | 1.066e-06 | 13.9 |
| 34+ | 64 | 1 | 4096 | 16384 | 6 | 7.767e-08 | 1.553e-07 | 1.061e-06 | 13.7 |
| 34+ | 64 | 4 | 4096 | 16384 | 6 | 7.556e-08 | 1.511e-07 | 1.062e-06 | 14.1 |
| 6-16 | 7 | 43 | 2051 | 8192 | 6 | 7.861e-08 | 1.572e-07 | 1.058e-06 | 13.5 |
| 6-16 | 8 | 64 | 4096 | 16384 | 6 | 7.803e-08 | 1.561e-07 | 1.060e-06 | 13.6 |
| 6-16 | 9 | 1 | 513 | 16384 | 6 | 6.870e-08 | 1.374e-07 | 1.062e-06 | 15.5 |
| 6-16 | 9 | 2 | 513 | 16384 | 6 | 6.644e-08 | 1.329e-07 | 1.068e-06 | 16.1 |
| 6-16 | 9 | 8 | 513 | 16384 | 6 | 6.733e-08 | 1.347e-07 | 1.066e-06 | 15.8 |
| 6-16 | 9 | 32 | 2051 | 8192 | 6 | 7.900e-08 | 1.580e-07 | 1.055e-06 | 13.4 |
| 6-16 | 11 | 24 | 2051 | 8192 | 6 | 7.814e-08 | 1.563e-07 | 1.051e-06 | 13.5 |
| 6-16 | 13 | 20 | 2051 | 8192 | 6 | 7.844e-08 | 1.569e-07 | 1.049e-06 | 13.4 |

CLASS splits 34+ n_sel big: p100 7.7665e-08 (set by n_q 1 n_sel 4096 n_kv 16384 splits 64) -> row 2 x p100 = 1.5533e-07; min Dmin 1.0405e-06, min k 13.4
CLASS splits 6-16 n_sel big: p100 7.9002e-08 (set by n_q 32 n_sel 2051 n_kv 8192 splits 9) -> row 2 x p100 = 1.5800e-07; min Dmin 1.0486e-06, min k 13.3

## TBO: the W4 test-backend-ops G1 dumps (88 same-type dumps; class = the new kernel census class)

| dump | case | new splits (class) | PR splits | PR NMSE vs O_CON | new NMSE vs CPU op | new / PR | defect Dmin | census class (test) |
|---|---|---|---|---|---|---|---|---|
| sa_0000 | type_k=f16,type_v=f16,n_q=9,n_kv=4096,n_sel=257,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 5 (2-5) | 5 | 4.3011e-08 | 4.8930e-08 | 1.138 | 1.065e-06 | sa_attend/G1/f16/f16/splits2-5/gqa12 |
| sa_0003 | type_k=q8_0,type_v=q8_0,n_q=9,n_kv=4096,n_sel=257,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 5 (2-5) | 5 | 4.0910e-08 | 4.7280e-08 | 1.156 | 1.074e-06 | sa_attend/G1/q8_0/q8_0/splits2-5/gqa12 |
| sa_0004 | type_k=f16,type_v=f16,n_q=9,n_kv=32768,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 26 (17-33) | 26 | 6.7801e-08 | 6.8840e-08 | 1.015 | 1.065e-06 | sa_attend/G1/f16/f16/splits17-33/gqa12 |
| sa_0007 | type_k=q8_0,type_v=q8_0,n_q=9,n_kv=32768,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 26 (17-33) | 26 | 7.1591e-08 | 6.7740e-08 | 0.946 | 1.076e-06 | sa_attend/G1/q8_0/q8_0/splits17-33/gqa12 |
| sa_0008 | type_k=f16,type_v=f16,n_q=1,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 33 (17-33) | 33 | 6.3040e-08 | 6.2970e-08 | 0.999 | 1.067e-06 | sa_attend/G1/f16/f16/splits17-33/gqa12 |
| sa_0009 | type_k=q8_0,type_v=q8_0,n_q=1,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 33 (17-33) | 33 | 7.0377e-08 | 6.8420e-08 | 0.972 | 1.076e-06 | sa_attend/G1/q8_0/q8_0/splits17-33/gqa12 |
| sa_0010 | type_k=f16,type_v=f16,n_q=2,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 33 (17-33) | 33 | 7.4733e-08 | 7.4890e-08 | 1.002 | 1.058e-06 | sa_attend/G1/f16/f16/splits17-33/gqa12 |
| sa_0011 | type_k=q8_0,type_v=q8_0,n_q=2,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 33 (17-33) | 33 | 7.1544e-08 | 7.0310e-08 | 0.983 | 1.065e-06 | sa_attend/G1/q8_0/q8_0/splits17-33/gqa12 |
| sa_0012 | type_k=f16,type_v=f16,n_q=4,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 33 (17-33) | 33 | 7.0585e-08 | 7.0240e-08 | 0.995 | 1.067e-06 | sa_attend/G1/f16/f16/splits17-33/gqa12 |
| sa_0013 | type_k=q8_0,type_v=q8_0,n_q=4,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 33 (17-33) | 33 | 7.4469e-08 | 7.3970e-08 | 0.993 | 1.073e-06 | sa_attend/G1/q8_0/q8_0/splits17-33/gqa12 |
| sa_0014 | type_k=f16,type_v=f16,n_q=8,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 33 (17-33) | 33 | 6.7049e-08 | 6.6200e-08 | 0.987 | 1.068e-06 | sa_attend/G1/f16/f16/splits17-33/gqa12 |
| sa_0015 | type_k=q8_0,type_v=q8_0,n_q=8,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 33 (17-33) | 33 | 7.2371e-08 | 7.2040e-08 | 0.995 | 1.061e-06 | sa_attend/G1/q8_0/q8_0/splits17-33/gqa12 |
| sa_0016 | type_k=f16,type_v=f16,n_q=9,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 26 (17-33) | 26 | 6.6042e-08 | 6.6100e-08 | 1.001 | 1.074e-06 | sa_attend/G1/f16/f16/splits17-33/gqa12 |
| sa_0017 | type_k=q8_0,type_v=q8_0,n_q=9,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 26 (17-33) | 26 | 7.0917e-08 | 7.0830e-08 | 0.999 | 1.063e-06 | sa_attend/G1/q8_0/q8_0/splits17-33/gqa12 |
| sa_0018 | type_k=f16,type_v=f16,n_q=64,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 5 (2-5) | 5 | 4.5640e-08 | 5.3610e-08 | 1.175 | 1.065e-06 | sa_attend/G1/f16/f16/splits2-5/gqa12 |
| sa_0019 | type_k=q8_0,type_v=q8_0,n_q=64,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 5 (2-5) | 5 | 4.6006e-08 | 5.3820e-08 | 1.170 | 1.064e-06 | sa_attend/G1/q8_0/q8_0/splits2-5/gqa12 |
| sa_0020 | type_k=f16,type_v=f16,n_q=65,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 5 (2-5) | 5 | 4.6462e-08 | 5.3610e-08 | 1.154 | 1.068e-06 | sa_attend/G1/f16/f16/splits2-5/gqa12 |
| sa_0021 | type_k=q8_0,type_v=q8_0,n_q=65,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 5 (2-5) | 5 | 4.6379e-08 | 5.4460e-08 | 1.174 | 1.066e-06 | sa_attend/G1/q8_0/q8_0/splits2-5/gqa12 |
| sa_0022 | type_k=f16,type_v=f16,n_q=66,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 4 (2-5) | 4 | 4.0769e-08 | 5.0000e-08 | 1.226 | 1.067e-06 | sa_attend/G1/f16/f16/splits2-5/gqa12 |
| sa_0023 | type_k=q8_0,type_v=q8_0,n_q=66,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 4 (2-5) | 4 | 4.2029e-08 | 5.0560e-08 | 1.203 | 1.068e-06 | sa_attend/G1/q8_0/q8_0/splits2-5/gqa12 |
| sa_0024 | type_k=f16,type_v=f16,n_q=127,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 3 (2-5) | 3 | 3.0674e-08 | 4.1980e-08 | 1.369 | 1.073e-06 | sa_attend/G1/f16/f16/splits2-5/gqa12 |
| sa_0025 | type_k=q8_0,type_v=q8_0,n_q=127,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 3 (2-5) | 3 | 3.0928e-08 | 4.2120e-08 | 1.362 | 1.061e-06 | sa_attend/G1/q8_0/q8_0/splits2-5/gqa12 |
| sa_0026 | type_k=f16,type_v=f16,n_q=16,n_kv=512,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=allvis,kv_layout=1,at_end=0,scale=0.062500 | 17 (17-33) | 17 | 5.2103e-08 | 6.9920e-08 | 1.342 | 1.086e-06 | sa_attend/G1/f16/f16/splits17-33/gqa12 |
| sa_0027 | type_k=q8_0,type_v=q8_0,n_q=16,n_kv=512,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=allvis,kv_layout=1,at_end=0,scale=0.062500 | 17 (17-33) | 17 | 4.6506e-08 | 6.1590e-08 | 1.324 | 1.089e-06 | sa_attend/G1/q8_0/q8_0/splits17-33/gqa12 |
| sa_0028 | type_k=f16,type_v=f16,n_q=16,n_kv=2048,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=allvis,kv_layout=1,at_end=0,scale=0.062500 | 17 (17-33) | 17 | 7.5648e-08 | 7.5650e-08 | 1.000 | 1.078e-06 | sa_attend/G1/f16/f16/splits17-33/gqa12 |
| sa_0029 | type_k=q8_0,type_v=q8_0,n_q=16,n_kv=2048,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=allvis,kv_layout=1,at_end=0,scale=0.062500 | 17 (17-33) | 17 | 6.5383e-08 | 6.5380e-08 | 1.000 | 1.060e-06 | sa_attend/G1/q8_0/q8_0/splits17-33/gqa12 |
| sa_0030 | type_k=f16,type_v=f16,n_q=9,n_kv=4096,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 26 (17-33) | 26 | 7.0262e-08 | 7.0700e-08 | 1.006 | 1.063e-06 | sa_attend/G1/f16/f16/splits17-33/gqa12 |
| sa_0031 | type_k=f16,type_v=f16,n_q=9,n_kv=32768,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 26 (17-33) | 26 | 6.9432e-08 | 6.8840e-08 | 0.991 | 1.065e-06 | sa_attend/G1/f16/f16/splits17-33/gqa12 |
| sa_0032 | type_k=q8_0,type_v=q8_0,n_q=9,n_kv=4096,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 26 (17-33) | 26 | 6.9554e-08 | 7.0530e-08 | 1.014 | 1.079e-06 | sa_attend/G1/q8_0/q8_0/splits17-33/gqa12 |
| sa_0033 | type_k=q8_0,type_v=q8_0,n_q=9,n_kv=32768,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 26 (17-33) | 26 | 6.9142e-08 | 6.7740e-08 | 0.980 | 1.077e-06 | sa_attend/G1/q8_0/q8_0/splits17-33/gqa12 |
| sa_0034 | type_k=f16,type_v=f16,n_q=66,n_kv=4096,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 4 (2-5) | 4 | 3.8582e-08 | 4.8090e-08 | 1.246 | 1.066e-06 | sa_attend/G1/f16/f16/splits2-5/gqa12 |
| sa_0035 | type_k=q8_0,type_v=q8_0,n_q=66,n_kv=4096,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 4 (2-5) | 4 | 4.0142e-08 | 4.9730e-08 | 1.239 | 1.065e-06 | sa_attend/G1/q8_0/q8_0/splits2-5/gqa12 |
| sa_0036 | type_k=f16,type_v=f16,n_q=66,n_kv=4096,n_sel=257,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=empty,kv_layout=1,at_end=0,scale=0.062500 | 1 (1) | 1 | 9.0426e-09 | 2.7120e-09 | 0.300 | 1.069e-06 | sa_attend/G1/f16/f16/splits1/gqa12 |
| sa_0037 | type_k=q8_0,type_v=q8_0,n_q=66,n_kv=4096,n_sel=257,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=empty,kv_layout=1,at_end=0,scale=0.062500 | 1 (1) | 1 | 9.2048e-09 | 2.6750e-09 | 0.291 | 1.072e-06 | sa_attend/G1/q8_0/q8_0/splits1/gqa12 |
| sa_0042 | type_k=f16,type_v=f16,n_q=66,n_kv=4096,n_sel=257,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=poison,kv_layout=1,at_end=0,scale=0.062500,poison | 1 (1) | 1 | 6.9670e-09 | 3.7730e-09 | 0.542 | 1.061e-06 | sa_attend/G1/f16/f16/splits1/gqa12 |
| sa_0043 | type_k=q8_0,type_v=q8_0,n_q=66,n_kv=4096,n_sel=257,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=poison,kv_layout=1,at_end=0,scale=0.062500,poison | 1 (1) | 1 | 6.9856e-09 | 3.7130e-09 | 0.532 | 1.071e-06 | sa_attend/G1/q8_0/q8_0/splits1/gqa12 |
| sa_0046 | type_k=f16,type_v=f16,n_q=65,n_kv=4096,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.088388 | 5 (2-5) | 5 | 4.4994e-08 | 5.2590e-08 | 1.169 | 1.061e-06 | sa_attend/G1/f16/f16/splits2-5/gqa12 |
| sa_0047 | type_k=q8_0,type_v=q8_0,n_q=65,n_kv=4096,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.088388 | 5 (2-5) | 5 | 4.2796e-08 | 5.0280e-08 | 1.175 | 1.056e-06 | sa_attend/G1/q8_0/q8_0/splits2-5/gqa12 |
| sa_0048 | type_k=f16,type_v=f16,n_q=1,n_kv=4096,n_sel=257,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 5 (2-5) | 5 | 4.7131e-08 | 4.8420e-08 | 1.027 | 1.078e-06 | sa_attend/G1/f16/f16/splits2-5/gqa12 |
| sa_0049 | type_k=f16,type_v=f16,n_q=1,n_kv=4096,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 33 (17-33) | 33 | 8.1126e-08 | 8.1470e-08 | 1.004 | 1.068e-06 | sa_attend/G1/f16/f16/splits17-33/gqa12 |
| sa_0050 | type_k=f16,type_v=f16,n_q=9,n_kv=4096,n_sel=257,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 5 (2-5) | 5 | 5.0213e-08 | 4.8170e-08 | 0.959 | 1.076e-06 | sa_attend/G1/f16/f16/splits2-5/gqa12 |
| sa_0051 | type_k=f16,type_v=f16,n_q=9,n_kv=4096,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 26 (17-33) | 26 | 7.4070e-08 | 7.2570e-08 | 0.980 | 1.058e-06 | sa_attend/G1/f16/f16/splits17-33/gqa12 |
| sa_0052 | type_k=f16,type_v=f16,n_q=64,n_kv=4096,n_sel=257,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 1 (1) | 1 | 6.9544e-09 | 1.8610e-09 | 0.268 | 1.064e-06 | sa_attend/G1/f16/f16/splits1/gqa12 |
| sa_0053 | type_k=f16,type_v=f16,n_q=64,n_kv=4096,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 5 (2-5) | 5 | 5.6483e-08 | 5.5960e-08 | 0.991 | 1.059e-06 | sa_attend/G1/f16/f16/splits2-5/gqa12 |
| sa_0054 | type_k=q8_0,type_v=q8_0,n_q=1,n_kv=4096,n_sel=257,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 5 (2-5) | 5 | 5.3257e-08 | 4.9270e-08 | 0.925 | 1.055e-06 | sa_attend/G1/q8_0/q8_0/splits2-5/gqa12 |
| sa_0055 | type_k=q8_0,type_v=q8_0,n_q=1,n_kv=4096,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 33 (17-33) | 33 | 7.5579e-08 | 7.3870e-08 | 0.977 | 1.107e-06 | sa_attend/G1/q8_0/q8_0/splits17-33/gqa12 |
| sa_0056 | type_k=q8_0,type_v=q8_0,n_q=9,n_kv=4096,n_sel=257,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 5 (2-5) | 5 | 4.9465e-08 | 4.9990e-08 | 1.011 | 1.068e-06 | sa_attend/G1/q8_0/q8_0/splits2-5/gqa12 |
| sa_0057 | type_k=q8_0,type_v=q8_0,n_q=9,n_kv=4096,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 26 (17-33) | 26 | 7.1677e-08 | 7.0970e-08 | 0.990 | 1.078e-06 | sa_attend/G1/q8_0/q8_0/splits17-33/gqa12 |
| sa_0058 | type_k=q8_0,type_v=q8_0,n_q=64,n_kv=4096,n_sel=257,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 1 (1) | 1 | 7.0178e-09 | 1.8630e-09 | 0.265 | 1.072e-06 | sa_attend/G1/q8_0/q8_0/splits1/gqa12 |
| sa_0059 | type_k=q8_0,type_v=q8_0,n_q=64,n_kv=4096,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 5 (2-5) | 5 | 5.5277e-08 | 5.4570e-08 | 0.987 | 1.052e-06 | sa_attend/G1/q8_0/q8_0/splits2-5/gqa12 |
| sa_0060 | type_k=f16,type_v=f16,n_q=1,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 33 (17-33) | 33 | 7.7803e-08 | 7.4070e-08 | 0.952 | 1.077e-06 | sa_attend/G1/f16/f16/splits17-33/gqa12 |
| sa_0061 | type_k=q8_0,type_v=q8_0,n_q=1,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 33 (17-33) | 33 | 7.2026e-08 | 7.2160e-08 | 1.002 | 1.065e-06 | sa_attend/G1/q8_0/q8_0/splits17-33/gqa12 |
| sa_0062 | type_k=f16,type_v=f16,n_q=2,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 33 (17-33) | 33 | 7.3699e-08 | 7.3850e-08 | 1.002 | 1.074e-06 | sa_attend/G1/f16/f16/splits17-33/gqa12 |
| sa_0063 | type_k=q8_0,type_v=q8_0,n_q=2,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 33 (17-33) | 33 | 6.6969e-08 | 6.4830e-08 | 0.968 | 1.062e-06 | sa_attend/G1/q8_0/q8_0/splits17-33/gqa12 |
| sa_0064 | type_k=f16,type_v=f16,n_q=4,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 33 (17-33) | 33 | 6.9666e-08 | 6.8510e-08 | 0.983 | 1.066e-06 | sa_attend/G1/f16/f16/splits17-33/gqa12 |
| sa_0065 | type_k=q8_0,type_v=q8_0,n_q=4,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 33 (17-33) | 33 | 6.8118e-08 | 6.6420e-08 | 0.975 | 1.060e-06 | sa_attend/G1/q8_0/q8_0/splits17-33/gqa12 |
| sa_0066 | type_k=f16,type_v=f16,n_q=8,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 33 (17-33) | 33 | 7.0550e-08 | 6.9490e-08 | 0.985 | 1.068e-06 | sa_attend/G1/f16/f16/splits17-33/gqa12 |
| sa_0067 | type_k=q8_0,type_v=q8_0,n_q=8,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 33 (17-33) | 33 | 6.8879e-08 | 6.8470e-08 | 0.994 | 1.078e-06 | sa_attend/G1/q8_0/q8_0/splits17-33/gqa12 |
| sa_0068 | type_k=f16,type_v=f16,n_q=9,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 26 (17-33) | 26 | 6.7468e-08 | 6.7510e-08 | 1.001 | 1.071e-06 | sa_attend/G1/f16/f16/splits17-33/gqa12 |
| sa_0069 | type_k=q8_0,type_v=q8_0,n_q=9,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 26 (17-33) | 26 | 7.1140e-08 | 7.1150e-08 | 1.000 | 1.069e-06 | sa_attend/G1/q8_0/q8_0/splits17-33/gqa12 |
| sa_0070 | type_k=f16,type_v=f16,n_q=64,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 5 (2-5) | 5 | 4.5917e-08 | 5.3920e-08 | 1.174 | 1.066e-06 | sa_attend/G1/f16/f16/splits2-5/gqa12 |
| sa_0071 | type_k=q8_0,type_v=q8_0,n_q=64,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 5 (2-5) | 5 | 4.4389e-08 | 5.2530e-08 | 1.183 | 1.070e-06 | sa_attend/G1/q8_0/q8_0/splits2-5/gqa12 |
| sa_0072 | type_k=f16,type_v=f16,n_q=65,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 5 (2-5) | 5 | 4.5863e-08 | 5.4020e-08 | 1.178 | 1.071e-06 | sa_attend/G1/f16/f16/splits2-5/gqa12 |
| sa_0073 | type_k=q8_0,type_v=q8_0,n_q=65,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 5 (2-5) | 5 | 4.5983e-08 | 5.3450e-08 | 1.162 | 1.061e-06 | sa_attend/G1/q8_0/q8_0/splits2-5/gqa12 |
| sa_0074 | type_k=f16,type_v=f16,n_q=66,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 4 (2-5) | 4 | 3.9158e-08 | 4.8410e-08 | 1.236 | 1.067e-06 | sa_attend/G1/f16/f16/splits2-5/gqa12 |
| sa_0075 | type_k=q8_0,type_v=q8_0,n_q=66,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 4 (2-5) | 4 | 4.0355e-08 | 4.9070e-08 | 1.216 | 1.065e-06 | sa_attend/G1/q8_0/q8_0/splits2-5/gqa12 |
| sa_0076 | type_k=f16,type_v=f16,n_q=127,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 3 (2-5) | 3 | 3.0783e-08 | 4.2290e-08 | 1.374 | 1.074e-06 | sa_attend/G1/f16/f16/splits2-5/gqa12 |
| sa_0077 | type_k=q8_0,type_v=q8_0,n_q=127,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 3 (2-5) | 3 | 3.0991e-08 | 4.2100e-08 | 1.358 | 1.072e-06 | sa_attend/G1/q8_0/q8_0/splits2-5/gqa12 |
| sa_0078 | type_k=f16,type_v=f16,n_q=263,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 2 (2-5) | 2 | 1.9086e-08 | 2.9900e-08 | 1.567 | 1.065e-06 | sa_attend/G1/f16/f16/splits2-5/gqa12 |
| sa_0079 | type_k=q8_0,type_v=q8_0,n_q=263,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 2 (2-5) | 2 | 1.8534e-08 | 2.9400e-08 | 1.586 | 1.061e-06 | sa_attend/G1/q8_0/q8_0/splits2-5/gqa12 |
| sa_0080 | type_k=f16,type_v=f16,n_q=1,n_kv=16384,n_sel=17,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 1 (1) | 1 | 2.3553e-08 | 9.9600e-09 | 0.423 | 1.048e-06 | sa_attend/G1/f16/f16/splits1/gqa12 |
| sa_0081 | type_k=f16,type_v=f16,n_q=1,n_kv=16384,n_sel=257,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 5 (2-5) | 5 | 4.1080e-08 | 4.6540e-08 | 1.133 | 1.046e-06 | sa_attend/G1/f16/f16/splits2-5/gqa12 |
| sa_0082 | type_k=f16,type_v=f16,n_q=1,n_kv=16384,n_sel=513,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 9 (6-16) | 9 | 5.5428e-08 | 5.7350e-08 | 1.035 | 1.060e-06 | no band row for sa_attend/G1/f16/f16/splits6-16/gqa12 |
| sa_0083 | type_k=f16,type_v=f16,n_q=1,n_kv=16384,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 33 (17-33) | 33 | 7.2654e-08 | 7.0470e-08 | 0.970 | 1.035e-06 | sa_attend/G1/f16/f16/splits17-33/gqa12 |
| sa_0084 | type_k=f16,type_v=f16,n_q=1,n_kv=16384,n_sel=4096,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 64 (34+) | 64 | 8.4414e-08 | 8.2900e-08 | 0.982 | 1.079e-06 | no band row for sa_attend/G1/f16/f16/splits34+/gqa12 |
| sa_0095 | type_k=q8_0,type_v=q8_0,n_q=1,n_kv=16384,n_sel=17,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 1 (1) | 1 | 2.1752e-08 | 1.0210e-08 | 0.469 | 1.054e-06 | sa_attend/G1/q8_0/q8_0/splits1/gqa12 |
| sa_0096 | type_k=q8_0,type_v=q8_0,n_q=1,n_kv=16384,n_sel=257,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 5 (2-5) | 5 | 4.3965e-08 | 4.6870e-08 | 1.066 | 1.072e-06 | sa_attend/G1/q8_0/q8_0/splits2-5/gqa12 |
| sa_0097 | type_k=q8_0,type_v=q8_0,n_q=1,n_kv=16384,n_sel=513,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 9 (6-16) | 9 | 5.3175e-08 | 5.4490e-08 | 1.025 | 1.045e-06 | no band row for sa_attend/G1/q8_0/q8_0/splits6-16/gqa12 |
| sa_0098 | type_k=q8_0,type_v=q8_0,n_q=1,n_kv=16384,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 33 (17-33) | 33 | 7.5829e-08 | 7.4680e-08 | 0.985 | 1.111e-06 | sa_attend/G1/q8_0/q8_0/splits17-33/gqa12 |
| sa_0099 | type_k=q8_0,type_v=q8_0,n_q=1,n_kv=16384,n_sel=4096,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 64 (34+) | 64 | 8.1039e-08 | 8.0110e-08 | 0.989 | 1.093e-06 | no band row for sa_attend/G1/q8_0/q8_0/splits34+/gqa12 |
| sa_0100 | type_k=f16,type_v=f16,n_q=512,n_kv=4096,n_sel=257,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 1 (1) | 1 | 7.1894e-09 | 1.8940e-09 | 0.263 | 1.065e-06 | sa_attend/G1/f16/f16/splits1/gqa12 |
| sa_0101 | type_k=f16,type_v=f16,n_q=512,n_kv=4096,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 1 (1) | 1 | 1.7253e-09 | 1.0990e-09 | 0.637 | 1.069e-06 | sa_attend/G1/f16/f16/splits1/gqa12 |
| sa_0102 | type_k=f16,type_v=f16,n_q=129,n_kv=4096,n_sel=257,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 1 (1) | 1 | 7.0430e-09 | 1.8700e-09 | 0.266 | 1.065e-06 | sa_attend/G1/f16/f16/splits1/gqa12 |
| sa_0103 | type_k=f16,type_v=f16,n_q=130,n_kv=32768,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=prblk,kv_layout=1,at_end=0,scale=0.062500,prblk=7000 | 3 (2-5) | 3 | 4.3232e-08 | 4.2320e-08 | 0.979 | 1.070e-06 | sa_attend/G1/f16/f16/splits2-5/gqa12 |
| sa_0104 | type_k=q8_0,type_v=q8_0,n_q=512,n_kv=4096,n_sel=257,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 1 (1) | 1 | 7.0524e-09 | 1.8890e-09 | 0.268 | 1.065e-06 | sa_attend/G1/q8_0/q8_0/splits1/gqa12 |
| sa_0105 | type_k=q8_0,type_v=q8_0,n_q=512,n_kv=4096,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 1 (1) | 1 | 1.6316e-09 | 1.0430e-09 | 0.639 | 1.077e-06 | sa_attend/G1/q8_0/q8_0/splits1/gqa12 |
| sa_0106 | type_k=q8_0,type_v=q8_0,n_q=129,n_kv=4096,n_sel=257,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 1 (1) | 1 | 6.8813e-09 | 1.8330e-09 | 0.266 | 1.060e-06 | sa_attend/G1/q8_0/q8_0/splits1/gqa12 |
| sa_0107 | type_k=q8_0,type_v=q8_0,n_q=130,n_kv=32768,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=prblk,kv_layout=1,at_end=0,scale=0.062500,prblk=7000 | 3 (2-5) | 3 | 4.3008e-08 | 4.2030e-08 | 0.977 | 1.071e-06 | sa_attend/G1/q8_0/q8_0/splits2-5/gqa12 |

CLASS G1 splits 1     n_sel <=17  dumps   2 PR_p100 2.3553e-08 (set by type_k=f16,type_v=f16,n_q=1,n_kv=16384,n_sel=17,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500) 2xp100 4.7106e-08 new_max 1.0210e-08 max_new/PR 0.469 min_Dmin 1.048e-06 min_k 44.5
CLASS G1 splits 1     n_sel <=257 dumps  10 PR_p100 9.2048e-09 (set by type_k=q8_0,type_v=q8_0,n_q=66,n_kv=4096,n_sel=257,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=empty,kv_layout=1,at_end=0,scale=0.062500) 2xp100 1.8410e-08 new_max 3.7730e-09 max_new/PR 0.542 min_Dmin 1.060e-06 min_k 115.2
CLASS G1 splits 1     n_sel >257  dumps   2 PR_p100 1.7253e-09 (set by type_k=f16,type_v=f16,n_q=512,n_kv=4096,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500) 2xp100 3.4506e-09 new_max 1.0990e-09 max_new/PR 0.639 min_Dmin 1.069e-06 min_k 619.7
CLASS G1 splits 17-33 n_sel >257  dumps  36 PR_p100 8.1126e-08 (set by type_k=f16,type_v=f16,n_q=1,n_kv=4096,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500) 2xp100 1.6225e-07 new_max 8.1470e-08 max_new/PR 1.342 min_Dmin 1.035e-06 min_k 12.8
CLASS G1 splits 2-5   n_sel <=257 dumps   8 PR_p100 5.3257e-08 (set by type_k=q8_0,type_v=q8_0,n_q=1,n_kv=4096,n_sel=257,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500) 2xp100 1.0651e-07 new_max 4.9990e-08 max_new/PR 1.156 min_Dmin 1.046e-06 min_k 19.6
CLASS G1 splits 2-5   n_sel >257  dumps  26 PR_p100 5.6483e-08 (set by type_k=f16,type_v=f16,n_q=64,n_kv=4096,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500) 2xp100 1.1297e-07 new_max 5.5960e-08 max_new/PR 1.586 min_Dmin 1.052e-06 min_k 18.6
CLASS G1 splits 34+   n_sel >257  dumps   2 PR_p100 8.4414e-08 (set by type_k=f16,type_v=f16,n_q=1,n_kv=16384,n_sel=4096,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500) 2xp100 1.6883e-07 new_max 8.2900e-08 max_new/PR 0.989 min_Dmin 1.079e-06 min_k 12.8
CLASS G1 splits 6-16  n_sel >257  dumps   2 PR_p100 5.5428e-08 (set by type_k=f16,type_v=f16,n_q=1,n_kv=16384,n_sel=513,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500) 2xp100 1.1086e-07 new_max 5.7350e-08 max_new/PR 1.035 min_Dmin 1.045e-06 min_k 18.9

## Decode n_q 1 SYN 262k fixture, 12 layers x 3 seeds (PLAN-v3 §14.2 (b))

```
samples per variant: {'PR-decM-f16': 36, 'PR-decM-q8_0': 36, 'G1M-f16': 36, 'G1M-bf16': 36, 'G1M-q8_0': 36, 'G1M-q4_0': 36}
VARIANT G1M-bf16       p100 6.0215e-08 (syn1_s2 L10) mean 5.4407e-08
VARIANT G1M-f16        p100 5.9983e-08 (syn1_s1 L6) mean 5.4780e-08
VARIANT G1M-q4_0       p100 6.0437e-08 (syn1_s3 L4) mean 5.5120e-08
VARIANT G1M-q8_0       p100 5.9022e-08 (syn1_s2 L9) mean 5.4993e-08
VARIANT PR-decM-f16    p100 5.9983e-08 (syn1_s1 L6) mean 5.4780e-08
VARIANT PR-decM-q8_0   p100 5.9044e-08 (syn1_s2 L9) mean 5.4993e-08
PR_p100 5.9983e-08 (PR-decM-f16 syn1_s1 L6) -> re-set row 2 x p100 = 1.1997e-07 (old row 1.15e-7)
NEW_G1_max 6.0437e-08 (syn1_s3 L4) <= row: True ; ratio 0.504
DEFECT Dmin 1.0109e-06 k = Dmin / PR_p100 16.9 ; Dmin / row 8.4
DETERMINISM all run pairs bitwise equal: True (216 pairs)
```
