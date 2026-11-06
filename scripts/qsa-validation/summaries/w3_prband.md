# W3: PR-verbatim G4 kernel on the W3 test-backend-ops inputs (band rows of §14.2 for G4)

Method: `GGML_SA_TEST_DUMP=<dir> test-backend-ops test -o SPARSE_ATTN -b ROCm0 -p group=4 --fail-unsupported` writes
the f16 / q8_0, GQA 12, 2 kv heads, 1 stream cases with n_kv % 4 == 0 in R2-BOP's TBO format (rows canonical: valid
cells ascending and unique). `harness/w3/prband_run.sh` runs R2-BOP's `bands_attn TBO:<dump>` (PR-verbatim
`qsa3_attn_kernel` V0 and VD, the PR's own plan builder) on every same-type dump (mixed pairs: no PR-verbatim G4
instance in that harness; gated and overflow dumps skipped), `prband_an.py` joins it with the new kernel's NMSE vs the
CPU op (`GGML_SA_TEST_REPORT=1` lines). Two runs (test-backend-ops draws new input values per run:
init_tensor_uniform seeds from std::random_device). Logs: `strix-002:~/dev/wt_lab/w3/prband/{res,res2}`,
`~/dev/wt_rewrite_logs/W3/{t3,t5}_g4.*`. Run 1 was taken with the first kernel version (non-finite V check in the WMMA
loop for packed V too); run 2 with the committed kernel (packed V: non-finite halves removed by the pack and the
touching blocks recomputed by the fixup, so the poison dumps are recomputed by the fixup and their ratio is ~0.04).

Result: band rows (2 x p100 over both runs, PR-verbatim): G4 n_sel <= 257: 2 x 3.899e-9 = 7.80e-9; G4 n_sel > 257:
2 x 1.206e-9 = 2.41e-9. PR-V0 == PR-VD and the dumped CPU op == R2-BOP's O_CON (nmse 0) on every dump.

## Run 1
| dump | case | PR-V0 nmse vs O_CON | PR-VD | new kernel nmse vs CPU op | new / PR | words differ new vs PR (nmse) | CPU op vs O_CON |
|---|---|---|---|---|---|---|---|
| sa_0000 | type_k=f16,type_v=f16,n_q=129,n_kv=4096,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=blocks,kv_layout=1,at_end=0,scale=0.062500 | 1.1477e-09 | 1.1477e-09 | 1.1480e-09 | 1.000 | 10816 / 792576 (8.49e-13) | 0.0e+00 |
| sa_0003 | type_k=q8_0,type_v=q8_0,n_q=129,n_kv=4096,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=blocks,kv_layout=1,at_end=0,scale=0.062500 | 1.2058e-09 | 1.2058e-09 | 1.2060e-09 | 1.000 | 10906 / 792576 (4.91e-13) | 0.0e+00 |
| sa_0004 | type_k=f16,type_v=f16,n_q=513,n_kv=65536,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=blocks,kv_layout=1,at_end=0,scale=0.062500 | 8.5568e-10 | 8.5568e-10 | 8.5590e-10 | 1.000 | 41277 / 3151872 (9.53e-13) | 0.0e+00 |
| sa_0005 | type_k=q8_0,type_v=q8_0,n_q=513,n_kv=65536,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=blocks,kv_layout=1,at_end=0,scale=0.062500 | 8.5435e-10 | 8.5435e-10 | 8.5450e-10 | 1.000 | 42499 / 3151872 (8.43e-13) | 0.0e+00 |
| sa_0006 | type_k=f16,type_v=f16,n_q=128,n_kv=8192,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 6.3896e-10 | 6.3896e-10 | 6.3870e-10 | 1.000 | 12679 / 786432 (1.66e-12) | 0.0e+00 |
| sa_0007 | type_k=q8_0,type_v=q8_0,n_q=128,n_kv=8192,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 6.4454e-10 | 6.4454e-10 | 6.4440e-10 | 1.000 | 10734 / 786432 (6.29e-13) | 0.0e+00 |
| sa_0008 | type_k=f16,type_v=f16,n_q=129,n_kv=8192,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 6.7187e-10 | 6.7187e-10 | 6.7180e-10 | 1.000 | 11428 / 792576 (4.83e-13) | 0.0e+00 |
| sa_0009 | type_k=q8_0,type_v=q8_0,n_q=129,n_kv=8192,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 6.5175e-10 | 6.5175e-10 | 6.5210e-10 | 1.001 | 9124 / 792576 (5.16e-13) | 0.0e+00 |
| sa_0010 | type_k=f16,type_v=f16,n_q=263,n_kv=8192,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 6.5087e-10 | 6.5087e-10 | 6.5110e-10 | 1.000 | 22709 / 1615872 (9.38e-13) | 0.0e+00 |
| sa_0011 | type_k=q8_0,type_v=q8_0,n_q=263,n_kv=8192,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 6.4701e-10 | 6.4701e-10 | 6.4730e-10 | 1.000 | 17200 / 1615872 (6.66e-13) | 0.0e+00 |
| sa_0012 | type_k=f16,type_v=f16,n_q=512,n_kv=8192,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 6.4896e-10 | 6.4896e-10 | 6.4900e-10 | 1.000 | 37634 / 3145728 (5.77e-13) | 0.0e+00 |
| sa_0013 | type_k=q8_0,type_v=q8_0,n_q=512,n_kv=8192,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 6.5666e-10 | 6.5666e-10 | 6.5660e-10 | 1.000 | 36049 / 3145728 (5.79e-13) | 0.0e+00 |
| sa_0014 | type_k=f16,type_v=f16,n_q=513,n_kv=8192,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 6.6045e-10 | 6.6045e-10 | 6.6060e-10 | 1.000 | 36900 / 3151872 (1.01e-12) | 0.0e+00 |
| sa_0015 | type_k=q8_0,type_v=q8_0,n_q=513,n_kv=8192,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 6.3054e-10 | 6.3054e-10 | 6.3060e-10 | 1.000 | 37311 / 3151872 (5.42e-13) | 0.0e+00 |
| sa_0016 | type_k=f16,type_v=f16,n_q=4096,n_kv=8192,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=blocks,kv_layout=1,at_end=0,scale=0.062500 | 9.6806e-10 | 9.6806e-10 | 9.6810e-10 | 1.000 | 312876 / 25165824 (8.36e-13) | 0.0e+00 |
| sa_0017 | type_k=f16,type_v=f16,n_q=66,n_kv=4096,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 9.8198e-10 | 9.8198e-10 | 9.8230e-10 | 1.000 | 6239 / 405504 (6.34e-13) | 0.0e+00 |
| sa_0018 | type_k=q8_0,type_v=q8_0,n_q=66,n_kv=4096,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 9.2062e-10 | 9.2062e-10 | 9.2100e-10 | 1.000 | 4183 / 405504 (3.56e-13) | 0.0e+00 |
| sa_0019 | type_k=f16,type_v=f16,n_q=66,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=empty,kv_layout=1,at_end=0,scale=0.062500 | 1.6330e-09 | 1.6330e-09 | 1.6330e-09 | 1.000 | 4063 / 405504 (4.21e-16) | 0.0e+00 |
| sa_0020 | type_k=q8_0,type_v=q8_0,n_q=66,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=empty,kv_layout=1,at_end=0,scale=0.062500 | 1.5503e-09 | 1.5503e-09 | 1.5500e-09 | 1.000 | 3473 / 405504 (3.01e-16) | 0.0e+00 |
| sa_0025 | type_k=f16,type_v=f16,n_q=66,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=poison,kv_layout=1,at_end=0,scale=0.062500,poison | 3.7471e-09 | 3.7471e-09 | 3.7470e-09 | 1.000 | 5360 / 405504 (3.16e-16) | 0.0e+00 |
| sa_0026 | type_k=q8_0,type_v=q8_0,n_q=66,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=poison,kv_layout=1,at_end=0,scale=0.062500,poison | 3.7588e-09 | 3.7588e-09 | 3.7590e-09 | 1.000 | 7062 / 405504 (4.51e-16) | 0.0e+00 |
| sa_0029 | type_k=f16,type_v=f16,n_q=512,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 5.6550e-10 | 5.6550e-10 | 5.6620e-10 | 1.001 | 37465 / 3145728 (1.09e-12) | 0.0e+00 |
| sa_0030 | type_k=f16,type_v=f16,n_q=512,n_kv=4096,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 1.0438e-09 | 1.0438e-09 | 1.0440e-09 | 1.000 | 40452 / 3145728 (7.60e-13) | 0.0e+00 |
| sa_0031 | type_k=f16,type_v=f16,n_q=129,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 5.7099e-10 | 5.7099e-10 | 5.7050e-10 | 0.999 | 7845 / 792576 (4.80e-13) | 0.0e+00 |
| sa_0032 | type_k=f16,type_v=f16,n_q=513,n_kv=4096,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 9.7527e-10 | 9.7527e-10 | 9.7530e-10 | 1.000 | 36348 / 3151872 (4.61e-13) | 0.0e+00 |
| sa_0033 | type_k=f16,type_v=f16,n_q=640,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 5.8198e-10 | 5.8198e-10 | 5.8180e-10 | 1.000 | 54223 / 3932160 (1.42e-12) | 0.0e+00 |
| sa_0034 | type_k=f16,type_v=f16,n_q=130,n_kv=32768,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=prblk,kv_layout=1,at_end=0,scale=0.062500,prblk=7000 | 6.0962e-10 | 6.0962e-10 | 6.1010e-10 | 1.001 | 7513 / 798720 (6.45e-13) | 0.0e+00 |
| sa_0035 | type_k=f16,type_v=f16,n_q=200,n_kv=65536,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=prblk,kv_layout=1,at_end=0,scale=0.062500,prblk=15000 | 5.5717e-10 | 5.5717e-10 | 5.5760e-10 | 1.001 | 15488 / 1228800 (1.03e-12) | 0.0e+00 |
| sa_0036 | type_k=q8_0,type_v=q8_0,n_q=512,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 5.6591e-10 | 5.6591e-10 | 5.6530e-10 | 0.999 | 36827 / 3145728 (5.55e-13) | 0.0e+00 |
| sa_0037 | type_k=q8_0,type_v=q8_0,n_q=512,n_kv=4096,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 9.8998e-10 | 9.8998e-10 | 9.9020e-10 | 1.000 | 39395 / 3145728 (8.97e-13) | 0.0e+00 |
| sa_0038 | type_k=q8_0,type_v=q8_0,n_q=129,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 5.9141e-10 | 5.9141e-10 | 5.9140e-10 | 1.000 | 7690 / 792576 (2.53e-16) | 0.0e+00 |
| sa_0039 | type_k=q8_0,type_v=q8_0,n_q=513,n_kv=4096,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 1.0041e-09 | 1.0041e-09 | 1.0040e-09 | 1.000 | 37744 / 3151872 (6.48e-13) | 0.0e+00 |
| sa_0040 | type_k=q8_0,type_v=q8_0,n_q=640,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 5.7948e-10 | 5.7948e-10 | 5.7930e-10 | 1.000 | 50725 / 3932160 (1.01e-12) | 0.0e+00 |
| sa_0041 | type_k=q8_0,type_v=q8_0,n_q=130,n_kv=32768,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=prblk,kv_layout=1,at_end=0,scale=0.062500,prblk=7000 | 6.0093e-10 | 6.0093e-10 | 6.0110e-10 | 1.000 | 9284 / 798720 (7.61e-13) | 0.0e+00 |
| sa_0042 | type_k=q8_0,type_v=q8_0,n_q=200,n_kv=65536,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=prblk,kv_layout=1,at_end=0,scale=0.062500,prblk=15000 | 5.4977e-10 | 5.4977e-10 | 5.4930e-10 | 0.999 | 16818 / 1228800 (7.27e-13) | 0.0e+00 |
| sa_0046 | type_k=f16,type_v=f16,n_q=130,n_kv=270336,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=prblk,kv_layout=1,at_end=0,scale=0.062500,prblk=66000 | 5.4662e-10 | 5.4662e-10 | 5.4650e-10 | 1.000 | 13902 / 798720 (1.64e-12) | 0.0e+00 |
| sa_0047 | type_k=f16,type_v=f16,n_q=129,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 1.2866e-09 | 1.2866e-09 | 1.2870e-09 | 1.000 | 9819 / 792576 (3.29e-16) | 0.0e+00 |
| sa_0048 | type_k=f16,type_v=f16,n_q=129,n_kv=16384,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 1.0654e-09 | 1.0654e-09 | 1.0650e-09 | 1.000 | 8830 / 792576 (2.77e-16) | 0.0e+00 |
| sa_0053 | type_k=q8_0,type_v=q8_0,n_q=129,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 1.2910e-09 | 1.2910e-09 | 1.2910e-09 | 1.000 | 9763 / 792576 (3.37e-16) | 0.0e+00 |
| sa_0054 | type_k=q8_0,type_v=q8_0,n_q=129,n_kv=16384,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 1.0773e-09 | 1.0773e-09 | 1.0770e-09 | 1.000 | 8493 / 792576 (2.79e-16) | 0.0e+00 |

BUCKET n_sel>257 dumps 26 PR_p100 1.2058e-09 (set by type_k=q8_0,type_v=q8_0,n_q=129,n_kv=4096,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=blocks,kv_layout=1,at_end=0,scale=0.062500) band_2xp100 2.4116e-09 new_max 1.2060e-09 max_new/PR 1.001
BUCKET n_sel<=257 dumps 14 PR_p100 3.7588e-09 (set by type_k=q8_0,type_v=q8_0,n_q=66,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=poison,kv_layout=1,at_end=0,scale=0.062500,poison) band_2xp100 7.5176e-09 new_max 3.7590e-09 max_new/PR 1.001
VD == V0 everywhere: True  CPU op == O_CON everywhere: True

## Run 2 (committed kernel)

| dump | case | PR-V0 nmse vs O_CON | PR-VD | new kernel nmse vs CPU op | new / PR | words differ new vs PR (nmse) | CPU op vs O_CON |
|---|---|---|---|---|---|---|---|
| sa_0000 | type_k=f16,type_v=f16,n_q=129,n_kv=4096,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=blocks,kv_layout=1,at_end=0,scale=0.062500 | 1.1839e-09 | 1.1839e-09 | 1.1840e-09 | 1.000 | 8580 / 792576 (3.86e-13) | 0.0e+00 |
| sa_0003 | type_k=q8_0,type_v=q8_0,n_q=129,n_kv=4096,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=blocks,kv_layout=1,at_end=0,scale=0.062500 | 1.1608e-09 | 1.1608e-09 | 1.1600e-09 | 0.999 | 10396 / 792576 (1.14e-12) | 0.0e+00 |
| sa_0004 | type_k=f16,type_v=f16,n_q=513,n_kv=65536,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=blocks,kv_layout=1,at_end=0,scale=0.062500 | 8.4960e-10 | 8.4960e-10 | 8.4960e-10 | 1.000 | 35593 / 3151872 (7.30e-13) | 0.0e+00 |
| sa_0005 | type_k=q8_0,type_v=q8_0,n_q=513,n_kv=65536,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=blocks,kv_layout=1,at_end=0,scale=0.062500 | 8.6214e-10 | 8.6214e-10 | 8.6210e-10 | 1.000 | 42041 / 3151872 (8.95e-13) | 0.0e+00 |
| sa_0006 | type_k=f16,type_v=f16,n_q=128,n_kv=8192,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 6.4621e-10 | 6.4621e-10 | 6.4640e-10 | 1.000 | 7567 / 786432 (7.23e-13) | 0.0e+00 |
| sa_0007 | type_k=q8_0,type_v=q8_0,n_q=128,n_kv=8192,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 6.6125e-10 | 6.6125e-10 | 6.6140e-10 | 1.000 | 8576 / 786432 (1.34e-13) | 0.0e+00 |
| sa_0008 | type_k=f16,type_v=f16,n_q=129,n_kv=8192,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 6.3722e-10 | 6.3722e-10 | 6.3660e-10 | 0.999 | 10651 / 792576 (8.56e-13) | 0.0e+00 |
| sa_0009 | type_k=q8_0,type_v=q8_0,n_q=129,n_kv=8192,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 6.4588e-10 | 6.4588e-10 | 6.4620e-10 | 1.000 | 10009 / 792576 (1.27e-12) | 0.0e+00 |
| sa_0010 | type_k=f16,type_v=f16,n_q=263,n_kv=8192,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 6.5023e-10 | 6.5023e-10 | 6.5010e-10 | 1.000 | 24053 / 1615872 (1.77e-12) | 0.0e+00 |
| sa_0011 | type_k=q8_0,type_v=q8_0,n_q=263,n_kv=8192,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 6.5380e-10 | 6.5380e-10 | 6.5400e-10 | 1.000 | 16359 / 1615872 (4.98e-13) | 0.0e+00 |
| sa_0012 | type_k=f16,type_v=f16,n_q=512,n_kv=8192,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 6.5467e-10 | 6.5467e-10 | 6.5470e-10 | 1.000 | 38148 / 3145728 (9.61e-13) | 0.0e+00 |
| sa_0013 | type_k=q8_0,type_v=q8_0,n_q=512,n_kv=8192,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 6.4676e-10 | 6.4676e-10 | 6.4670e-10 | 1.000 | 36989 / 3145728 (4.31e-13) | 0.0e+00 |
| sa_0014 | type_k=f16,type_v=f16,n_q=513,n_kv=8192,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 6.5729e-10 | 6.5729e-10 | 6.5750e-10 | 1.000 | 39734 / 3151872 (8.20e-13) | 0.0e+00 |
| sa_0015 | type_k=q8_0,type_v=q8_0,n_q=513,n_kv=8192,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 6.4007e-10 | 6.4007e-10 | 6.4000e-10 | 1.000 | 33963 / 3151872 (5.59e-13) | 0.0e+00 |
| sa_0016 | type_k=f16,type_v=f16,n_q=4096,n_kv=8192,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=blocks,kv_layout=1,at_end=0,scale=0.062500 | 1.0051e-09 | 1.0051e-09 | 1.0050e-09 | 1.000 | 298140 / 25165824 (7.46e-13) | 0.0e+00 |
| sa_0017 | type_k=f16,type_v=f16,n_q=66,n_kv=4096,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 9.6524e-10 | 9.6524e-10 | 9.6490e-10 | 1.000 | 6289 / 405504 (2.72e-13) | 0.0e+00 |
| sa_0018 | type_k=q8_0,type_v=q8_0,n_q=66,n_kv=4096,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 9.4574e-10 | 9.4574e-10 | 9.4650e-10 | 1.001 | 5297 / 405504 (7.54e-13) | 0.0e+00 |
| sa_0019 | type_k=f16,type_v=f16,n_q=66,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=empty,kv_layout=1,at_end=0,scale=0.062500 | 1.5400e-09 | 1.5400e-09 | 1.5390e-09 | 0.999 | 4501 / 405504 (1.42e-12) | 0.0e+00 |
| sa_0020 | type_k=q8_0,type_v=q8_0,n_q=66,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=empty,kv_layout=1,at_end=0,scale=0.062500 | 1.5426e-09 | 1.5426e-09 | 1.5430e-09 | 1.000 | 4254 / 405504 (3.74e-16) | 0.0e+00 |
| sa_0025 | type_k=f16,type_v=f16,n_q=66,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=poison,kv_layout=1,at_end=0,scale=0.062500,poison | 3.7698e-09 | 3.7698e-09 | 1.4690e-10 | 0.039 | 405252 / 405504 (3.68e-09) | 0.0e+00 |
| sa_0026 | type_k=q8_0,type_v=q8_0,n_q=66,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=poison,kv_layout=1,at_end=0,scale=0.062500,poison | 3.8989e-09 | 3.8989e-09 | 1.4110e-10 | 0.036 | 405245 / 405504 (3.83e-09) | 0.0e+00 |
| sa_0029 | type_k=f16,type_v=f16,n_q=512,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 5.8962e-10 | 5.8962e-10 | 5.8960e-10 | 1.000 | 36174 / 3145728 (5.36e-13) | 0.0e+00 |
| sa_0030 | type_k=f16,type_v=f16,n_q=512,n_kv=4096,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 9.9407e-10 | 9.9407e-10 | 9.9420e-10 | 1.000 | 36257 / 3145728 (8.71e-13) | 0.0e+00 |
| sa_0031 | type_k=f16,type_v=f16,n_q=129,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 6.0337e-10 | 6.0337e-10 | 6.0340e-10 | 1.000 | 10635 / 792576 (3.65e-16) | 0.0e+00 |
| sa_0032 | type_k=f16,type_v=f16,n_q=513,n_kv=4096,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 9.5583e-10 | 9.5583e-10 | 9.5600e-10 | 1.000 | 33787 / 3151872 (6.43e-13) | 0.0e+00 |
| sa_0033 | type_k=f16,type_v=f16,n_q=640,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 5.6301e-10 | 5.6301e-10 | 5.6280e-10 | 1.000 | 51855 / 3932160 (1.17e-12) | 0.0e+00 |
| sa_0034 | type_k=f16,type_v=f16,n_q=130,n_kv=32768,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=prblk,kv_layout=1,at_end=0,scale=0.062500,prblk=7000 | 6.2182e-10 | 6.2182e-10 | 6.2130e-10 | 0.999 | 12147 / 798720 (8.03e-13) | 0.0e+00 |
| sa_0035 | type_k=f16,type_v=f16,n_q=200,n_kv=65536,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=prblk,kv_layout=1,at_end=0,scale=0.062500,prblk=15000 | 5.5432e-10 | 5.5432e-10 | 5.5450e-10 | 1.000 | 12785 / 1228800 (6.76e-13) | 0.0e+00 |
| sa_0036 | type_k=q8_0,type_v=q8_0,n_q=512,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 5.6563e-10 | 5.6563e-10 | 5.6480e-10 | 0.999 | 44098 / 3145728 (1.49e-12) | 0.0e+00 |
| sa_0037 | type_k=q8_0,type_v=q8_0,n_q=512,n_kv=4096,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 1.0365e-09 | 1.0365e-09 | 1.0360e-09 | 1.000 | 37189 / 3145728 (1.02e-12) | 0.0e+00 |
| sa_0038 | type_k=q8_0,type_v=q8_0,n_q=129,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 5.5447e-10 | 5.5447e-10 | 5.5280e-10 | 0.997 | 10084 / 792576 (1.64e-12) | 0.0e+00 |
| sa_0039 | type_k=q8_0,type_v=q8_0,n_q=513,n_kv=4096,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 9.7589e-10 | 9.7589e-10 | 9.7590e-10 | 1.000 | 43685 / 3151872 (9.69e-13) | 0.0e+00 |
| sa_0040 | type_k=q8_0,type_v=q8_0,n_q=640,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.062500 | 5.5052e-10 | 5.5052e-10 | 5.5060e-10 | 1.000 | 44006 / 3932160 (1.68e-13) | 0.0e+00 |
| sa_0041 | type_k=q8_0,type_v=q8_0,n_q=130,n_kv=32768,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=prblk,kv_layout=1,at_end=0,scale=0.062500,prblk=7000 | 6.0904e-10 | 6.0904e-10 | 6.0910e-10 | 1.000 | 8999 / 798720 (5.64e-13) | 0.0e+00 |
| sa_0042 | type_k=q8_0,type_v=q8_0,n_q=200,n_kv=65536,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=prblk,kv_layout=1,at_end=0,scale=0.062500,prblk=15000 | 5.5343e-10 | 5.5343e-10 | 5.5340e-10 | 1.000 | 17008 / 1228800 (1.27e-12) | 0.0e+00 |
| sa_0046 | type_k=f16,type_v=f16,n_q=130,n_kv=270336,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=prblk,kv_layout=1,at_end=0,scale=0.062500,prblk=66000 | 5.4871e-10 | 5.4871e-10 | 5.4820e-10 | 0.999 | 7862 / 798720 (6.27e-13) | 0.0e+00 |
| sa_0047 | type_k=f16,type_v=f16,n_q=129,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 1.3049e-09 | 1.3049e-09 | 1.3050e-09 | 1.000 | 8915 / 792576 (2.70e-13) | 0.0e+00 |
| sa_0048 | type_k=f16,type_v=f16,n_q=129,n_kv=16384,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 1.0986e-09 | 1.0986e-09 | 1.0990e-09 | 1.000 | 8021 / 792576 (7.59e-13) | 0.0e+00 |
| sa_0053 | type_k=q8_0,type_v=q8_0,n_q=129,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062500 | 1.3061e-09 | 1.3061e-09 | 1.3060e-09 | 1.000 | 7360 / 792576 (2.66e-16) | 0.0e+00 |
| sa_0054 | type_k=q8_0,type_v=q8_0,n_q=129,n_kv=16384,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.062500 | 1.0852e-09 | 1.0852e-09 | 1.0840e-09 | 0.999 | 10596 / 792576 (1.15e-12) | 0.0e+00 |

BUCKET n_sel>257 dumps 26 PR_p100 1.1839e-09 (set by type_k=f16,type_v=f16,n_q=129,n_kv=4096,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=blocks,kv_layout=1,at_end=0,scale=0.062500) band_2xp100 2.3678e-09 new_max 1.1840e-09 max_new/PR 1.001
BUCKET n_sel<=257 dumps 14 PR_p100 3.8989e-09 (set by type_k=q8_0,type_v=q8_0,n_q=66,n_kv=4096,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=poison,kv_layout=1,at_end=0,scale=0.062500,poison) band_2xp100 7.7978e-09 new_max 1.5430e-09 max_new/PR 1.000
VD == V0 everywhere: True  CPU op == O_CON everywhere: True
