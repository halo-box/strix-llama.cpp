# W5 resources and tests (final_w5.sh at 431fd7def)

Clean HIP rebuild: the sha256 of libggml-hip.so and test-backend-ops are unchanged (PRE = the binaries of the first runs, runs1.sh / runs2.sh; POST = clean rebuild). `rc=1` on the ROCm0 SPARSE_ATTN runs comes from the 20 rot != 0 cases under `--fail-unsupported` (in-op rotation: W10): see the table below.

```
HEAD 431fd7def status=[1 changed] date=2026-11-06T14:11:28+01:00
PRE 4ed20bd71b79e88ec60a9977793729de5303509c59e998c6e1d857a91e7b0b8a  build/bin/libggml-hip.so.0.26.0
PRE df6c36f3fc16fdbf110a61a13e5c362a108b196ca093427ce50c2d093050ffc4  build/bin/test-backend-ops
EXIT hip_cfg 0 warnings=0 errors=0
EXIT hip_build 0 warnings=85 errors=0
POST 4ed20bd71b79e88ec60a9977793729de5303509c59e998c6e1d857a91e7b0b8a  build/bin/libggml-hip.so.0.26.0
POST df6c36f3fc16fdbf110a61a13e5c362a108b196ca093427ce50c2d093050ffc4  build/bin/test-backend-ops
EXIT ma_cfg 0 warnings=0 errors=0
EXIT ma_build 0 warnings=124 errors=0
EXIT vk_build 0 warnings=0 errors=0
EXIT cpu_build 0 warnings=0 errors=0
EXIT tu_sparse-attn-attend-g4.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-merge.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-pack.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-portable.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-generic.cu 0 warnings=0 errors=0
EXIT tu_ggml-cuda.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-attend-g1.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-attend-g4-f16.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-attend-g4-q4_0.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-attend-g4-bf16.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-attend-g4-q8_0.cu 0 warnings=0 errors=0
sparse-attn-generic.cu       kernels=10 spilling=0 sa_kernels=10 sa_spilling=0 sa_max_vgprs=209 sa_max_scratch=0
sparse-attn-portable.cu      kernels=1 spilling=0 sa_kernels=1 sa_spilling=0 sa_max_vgprs=53 sa_max_scratch=0
sparse-attn-pack.cu          kernels=5 spilling=0 sa_kernels=5 sa_spilling=0 sa_max_vgprs=28 sa_max_scratch=0
sparse-attn-merge.cu         kernels=1 spilling=0 sa_kernels=1 sa_spilling=0 sa_max_vgprs=8 sa_max_scratch=0
sparse-attn-attend-g1.cu     kernels=48 spilling=0 sa_kernels=48 sa_spilling=0 sa_max_vgprs=131 sa_max_scratch=0
sparse-attn-attend-g4.cu     kernels=0 spilling=0 sa_kernels=0 sa_spilling=0 sa_max_vgprs=-1 sa_max_scratch=-1
sparse-attn-attend-g4-f16.cu kernels=56 spilling=0 sa_kernels=56 sa_spilling=0 sa_max_vgprs=192 sa_max_scratch=0
sparse-attn-attend-g4-bf16.cu kernels=56 spilling=0 sa_kernels=56 sa_spilling=0 sa_max_vgprs=192 sa_max_scratch=0
sparse-attn-attend-g4-q8_0.cu kernels=56 spilling=0 sa_kernels=56 sa_spilling=0 sa_max_vgprs=192 sa_max_scratch=0
sparse-attn-attend-g4-q4_0.cu kernels=56 spilling=0 sa_kernels=56 sa_spilling=0 sa_max_vgprs=192 sa_max_scratch=0
ggml-cuda.cu                 kernels=1 spilling=0 sa_kernels=0 sa_spilling=0 sa_max_vgprs=-1 sa_max_scratch=-1
TOTAL sa kernels=289 spilling=0 scratch>0=0
EXIT attn_graphs rc=1  1190/1210 tests passed fail=2 notsup=20  CENSUS COVERAGE: ROCm0: 492/492 compiled keys of {sa_attend/,sa_fixup/,sa_generic/,sa_merge,sa_pack/,sa_portable/} launched (2 excluded by a forcing switch): OK
EXIT portable rc=1  1190/1210 tests passed fail=2 notsup=20  CENSUS COVERAGE: ROCm0: 2/2 compiled keys of {sa_attend/,sa_fixup/,sa_generic/,sa_merge,sa_pack/,sa_portable/} launched (492 excluded by a forcing switch): OK
EXIT generic rc=1  1190/1210 tests passed fail=2 notsup=20  CENSUS COVERAGE: ROCm0: 24/24 compiled keys of {sa_attend/,sa_fixup/,sa_generic/,sa_merge,sa_pack/,sa_portable/} launched (470 excluded by a forcing switch): OK
EXIT attn_nographs rc=1  1190/1210 tests passed fail=2 notsup=20  CENSUS COVERAGE: ROCm0: 492/492 compiled keys of {sa_attend/,sa_fixup/,sa_generic/,sa_merge,sa_pack/,sa_portable/} launched (2 excluded by a forcing switch): OK
EXIT vdirect rc=1  1190/1210 tests passed fail=2 notsup=20  CENSUS COVERAGE: ROCm0: 388/388 compiled keys of {sa_attend/,sa_fixup/,sa_generic/,sa_merge,sa_pack/,sa_portable/} launched (106 excluded by a forcing switch): OK
EXIT vpacked rc=1  1190/1210 tests passed fail=2 notsup=20  CENSUS COVERAGE: ROCm0: 393/393 compiled keys of {sa_attend/,sa_fixup/,sa_generic/,sa_merge,sa_pack/,sa_portable/} launched (101 excluded by a forcing switch): OK
EXIT cpu_attn rc=0  1210/1210 tests passed fail=0 notsup=0 
EXIT plan_rocm rc=0  185/185 tests passed fail=0 notsup=0  CENSUS COVERAGE: ROCm0: 2/2 compiled keys of {sa_plan/} launched (0 excluded by a forcing switch): OK
EXIT plan_cpu rc=0  185/185 tests passed fail=0 notsup=0 
EXIT QSA_PREFILL_MASKLESS rc=0  26/26 tests passed fail=0 notsup=0 
EXIT QSA_DECODE_MASKLESS rc=0  52/52 tests passed fail=0 notsup=1 
EXIT QSA_PREFILL rc=0  26/26 tests passed fail=0 notsup=0 
EXIT QSA_DECODE rc=0  92/92 tests passed fail=0 notsup=0 
EXIT vk_QSA_PREFILL_MASKLESS rc=0  14/14 tests passed fail=0 notsup=12 
EXIT vk_QSA_DECODE_MASKLESS rc=0  34/34 tests passed fail=0 notsup=19 
EXIT vk_QSA_PREFILL rc=0  26/26 tests passed fail=0 notsup=0 
EXIT vk_QSA_DECODE rc=0  92/92 tests passed fail=0 notsup=0 
EXIT vk_SPARSE_ATTN_PLAN rc=0  0/0 tests passed fail=0 notsup=185 
EXIT vk_attn rc=0  0/0 tests passed fail=0 notsup=1210 
EXIT vkbuild_cpu_attn rc=0  1210/1210 tests passed fail=0 notsup=0 
EXIT cpuonly_attn rc=0  1210/1210 tests passed fail=0 notsup=0 
EXIT ctest rc=0 100% tests passed, 0 tests failed out of 1
FINAL_DONE
```

## Cases not supported and compare errors, per ROCm0 SPARSE_ATTN run (log grep)

```
final_attn_graphs notsup=20 notsup_with_rot0=0 compare_errors=0
final_portable notsup=20 notsup_with_rot0=0 compare_errors=0
final_generic notsup=20 notsup_with_rot0=0 compare_errors=0
final_attn_nographs notsup=20 notsup_with_rot0=0 compare_errors=0
final_vdirect notsup=20 notsup_with_rot0=0 compare_errors=0
final_vpacked notsup=20 notsup_with_rot0=0 compare_errors=0
```

## S^2 pair coverage, bands, GQA 4 / 8 replicated heads (pairs.py; NMSE vs the CPU op / the band of the census class x type spread)

```
== final_attn_graphs.log
S^2 pairs seen 81 / 81, with a generic member 65 / 65; reports over band or without a band: 0
  fast G1              pairs 16 (with a generic member  0) max nmse/band 0.468
  fast G4              pairs 16 (with a generic member  0) max nmse/band 0.767
  generic G1           pairs 65 (with a generic member 65) max nmse/band 0.383
  generic G4 direct    pairs 65 (with a generic member 65) max nmse/band 0.322
  generic G4 packed    pairs 65 (with a generic member 65) max nmse/band 0.358
  GQAREP fast     G1 cases 10 max nmse/row 0.000 max nmse 0.000e+00 bitwise 10/10
  GQAREP fast     G4 cases 10 max nmse/row 0.000 max nmse 0.000e+00 bitwise 10/10
  GQAREP generic  G1 cases  4 max nmse/row 0.000 max nmse 0.000e+00 bitwise 4/4
  GQAREP generic  G4 cases  4 max nmse/row 0.000 max nmse 0.000e+00 bitwise 4/4
== final_generic.log
S^2 pairs seen 81 / 81, with a generic member 65 / 65; reports over band or without a band: 0
  generic G1           pairs 81 (with a generic member 65) max nmse/band 0.473
  generic G4 direct    pairs 81 (with a generic member 65) max nmse/band 0.731
  generic G4 packed    pairs 81 (with a generic member 65) max nmse/band 0.500
  GQAREP generic  G1 cases 14 max nmse/row 0.000 max nmse 0.000e+00 bitwise 14/14
  GQAREP generic  G4 cases 14 max nmse/row 0.000 max nmse 0.000e+00 bitwise 14/14
== final_portable.log
S^2 pairs seen 81 / 81, with a generic member 65 / 65; reports over band or without a band: 0
  portable G1          pairs 81 (with a generic member 65) max nmse/band 0.002
  portable G4          pairs 81 (with a generic member 65) max nmse/band 0.006
  GQAREP portable G1 cases 14 max nmse/row 0.000 max nmse 0.000e+00 bitwise 14/14
  GQAREP portable G4 cases 14 max nmse/row 0.000 max nmse 0.000e+00 bitwise 14/14
```

Band sources: §14.2 (c), sa_band_for_backend in tests/test-backend-ops.cpp: sa_generic/G4 -> the G4 row, sa_generic/G1/splits<n> -> the G1 row of that class, sa_portable/G4 -> the G4 row, sa_portable/G1 -> "G1, 1 split", all rows measured on PR-verbatim kernels (w3_prband.md, R2-BOP, w4_bands.md), never on these tiers; the case-level ratio includes the §14.2 (c) type spread. The GQAREP row is the class row without the spread.

## Per-class summary (sarep.py)

### final_attn_graphs

```
SPARSE_ATTN fast G1 splits1              cases=  69 max_ratio=0.325  type_k=f16,type_v=f16,n_q=1,n_kv=16384,n_sel=17,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.062
SPARSE_ATTN fast G1 splits17-33          cases= 150 max_ratio=0.468  type_k=q8_0,type_v=q8_0,n_q=16,n_kv=2051,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=allvis,kv_layout=1,at_end=0,scal
SPARSE_ATTN fast G1 splits2-5            cases= 157 max_ratio=0.384  type_k=q8_0,type_v=q8_0,n_q=1,n_kv=16384,n_sel=257,group=1,gqa=8,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.0
SPARSE_ATTN fast G1 splits34+            cases=  48 max_ratio=0.468  type_k=q8_0,type_v=q8_0,n_q=1,n_kv=16384,n_sel=4096,group=1,gqa=4,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.
SPARSE_ATTN fast G1 splits6-16           cases=  48 max_ratio=0.400  type_k=q8_0,type_v=q8_0,n_q=1,n_kv=16384,n_sel=513,group=1,gqa=4,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.0
SPARSE_ATTN fast G4                      cases= 274 max_ratio=0.767  type_k=q8_0,type_v=q8_0,n_q=9,n_kv=2051,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=allvis,kv_layout=1,at_end=1,scale
SPARSE_ATTN generic G1 splits1           cases=  11 max_ratio=0.164  type_k=q5_0,type_v=q4_1,n_q=1,n_kv=16384,n_sel=17,group=1,gqa=8,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.06
SPARSE_ATTN generic G1 splits17-33       cases=  17 max_ratio=0.383  type_k=f32,type_v=f32,n_q=9,n_kv=8192,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.06
SPARSE_ATTN generic G1 splits2-5         cases=  84 max_ratio=0.341  type_k=f32,type_v=iq4_nl,n_q=9,n_kv=4096,n_sel=257,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.
SPARSE_ATTN generic G1 splits34+         cases=   6 max_ratio=0.250  type_k=q5_0,type_v=q4_1,n_q=1,n_kv=16384,n_sel=4096,group=1,gqa=4,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.
SPARSE_ATTN generic G1 splits6-16        cases=   6 max_ratio=0.198  type_k=q5_0,type_v=q4_1,n_q=1,n_kv=16384,n_sel=513,group=1,gqa=4,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.0
SPARSE_ATTN generic G4 direct            cases=  89 max_ratio=0.322  type_k=q5_1,type_v=q5_1,n_q=513,n_kv=65536,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=blocks,kv_layout=1,at_end=0,sc
SPARSE_ATTN generic G4 packed            cases=  95 max_ratio=0.358  type_k=iq4_nl,type_v=iq4_nl,n_q=65,n_kv=4096,n_sel=2051,group=4,gqa=12,n_stream=2,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,sca
SPARSE_ATTN_GATED fast G4                cases= 110 max_ratio=0.410  type_k=q8_0,type_v=q8_0,n_q=129,n_kv=4096,n_sel=2051,group=4,gqa=12,n_stream=2,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=
SPARSE_ATTN_GQAREP fast G1 splits17-33   cases=   4 max_ratio=0.236  type_k=q4_0,type_v=bf16,n_q=1,n_kv=16384,n_sel=2051,group=1,gqa=4,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.
SPARSE_ATTN_GQAREP fast G1 splits2-5     cases=  16 max_ratio=0.328  type_k=f16,type_v=f16,n_q=66,n_kv=4096,n_sel=2051,group=1,gqa=8,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.06
SPARSE_ATTN_GQAREP fast G4               cases=  20 max_ratio=0.398  type_k=q8_0,type_v=q8_0,n_q=66,n_kv=4096,n_sel=2051,group=4,gqa=8,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.
SPARSE_ATTN_GQAREP generic G1 splits1    cases=   4 max_ratio=0.021  type_k=q5_0,type_v=f16,n_q=129,n_kv=16384,n_sel=257,group=1,gqa=4,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.
SPARSE_ATTN_GQAREP generic G1 splits2-5  cases=   4 max_ratio=0.282  type_k=q4_1,type_v=q4_1,n_q=66,n_kv=4096,n_sel=2051,group=1,gqa=8,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.
SPARSE_ATTN_GQAREP generic G4 direct     cases=   4 max_ratio=0.127  type_k=q5_0,type_v=f16,n_q=129,n_kv=16384,n_sel=257,group=4,gqa=8,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.
SPARSE_ATTN_GQAREP generic G4 packed     cases=   4 max_ratio=0.349  type_k=q4_1,type_v=q4_1,n_q=66,n_kv=4096,n_sel=2051,group=4,gqa=4,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.
TOTAL reports 1220 max ratio 0.767
```

### final_generic

```
SPARSE_ATTN generic G1 splits1           cases=  80 max_ratio=0.242  type_k=q8_0,type_v=q8_0,n_q=1,n_kv=16384,n_sel=17,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.0
SPARSE_ATTN generic G1 splits17-33       cases= 167 max_ratio=0.473  type_k=q8_0,type_v=q8_0,n_q=1,n_kv=4096,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=pr,kv_layout=1,at_end=0,scale=0.0
SPARSE_ATTN generic G1 splits2-5         cases= 241 max_ratio=0.465  type_k=q8_0,type_v=q8_0,n_q=1,n_kv=16384,n_sel=257,group=1,gqa=4,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.0
SPARSE_ATTN generic G1 splits34+         cases=  54 max_ratio=0.472  type_k=q8_0,type_v=q8_0,n_q=1,n_kv=16384,n_sel=4096,group=1,gqa=8,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.
SPARSE_ATTN generic G1 splits6-16        cases=  54 max_ratio=0.437  type_k=q8_0,type_v=q8_0,n_q=1,n_kv=16384,n_sel=513,group=1,gqa=4,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,scale=0.0
SPARSE_ATTN generic G4 direct            cases= 209 max_ratio=0.731  type_k=q8_0,type_v=q8_0,n_q=9,n_kv=2051,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=allvis,kv_layout=1,at_end=1,scale
SPARSE_ATTN generic G4 packed            cases= 249 max_ratio=0.500  type_k=q8_0,type_v=q8_0,n_q=129,n_kv=4096,n_sel=2051,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=blocks,kv_layout=1,at_end=0,sca
SPARSE_ATTN_GATED generic G4 direct      cases=  50 max_ratio=0.142  type_k=f16,type_v=f16,n_q=129,n_kv=16384,n_sel=257,group=4,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.
SPARSE_ATTN_GATED generic G4 packed      cases=  60 max_ratio=0.400  type_k=f16,type_v=f16,n_q=129,n_kv=4096,n_sel=2051,group=4,gqa=12,n_stream=2,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.
SPARSE_ATTN_GQAREP generic G1 splits1    cases=   4 max_ratio=0.021  type_k=q5_0,type_v=f16,n_q=129,n_kv=16384,n_sel=257,group=1,gqa=4,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.
SPARSE_ATTN_GQAREP generic G1 splits17-33 cases=   4 max_ratio=0.215  type_k=q4_0,type_v=bf16,n_q=1,n_kv=16384,n_sel=2051,group=1,gqa=4,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0
SPARSE_ATTN_GQAREP generic G1 splits2-5  cases=  20 max_ratio=0.316  type_k=q8_0,type_v=q8_0,n_q=66,n_kv=4096,n_sel=2051,group=1,gqa=4,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.
SPARSE_ATTN_GQAREP generic G4 direct     cases=   8 max_ratio=0.131  type_k=q5_0,type_v=f16,n_q=129,n_kv=16384,n_sel=257,group=4,gqa=4,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.
SPARSE_ATTN_GQAREP generic G4 packed     cases=  20 max_ratio=0.394  type_k=f16,type_v=f16,n_q=66,n_kv=4096,n_sel=2051,group=4,gqa=4,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.06
TOTAL reports 1220 max ratio 0.731
```

### final_portable

```
SPARSE_ATTN portable G1                  cases= 596 max_ratio=0.002  type_k=f16,type_v=f16,n_q=65,n_kv=4096,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.0
SPARSE_ATTN portable G4                  cases= 458 max_ratio=0.006  type_k=f16,type_v=f16,n_q=66,n_kv=4096,n_sel=2051,group=4,gqa=4,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.06
SPARSE_ATTN_GATED portable G4            cases= 110 max_ratio=0.002  type_k=q8_0,type_v=f16,n_q=129,n_kv=16384,n_sel=257,group=4,gqa=4,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.
SPARSE_ATTN_GQAREP portable G1           cases=  28 max_ratio=0.000  type_k=f16,type_v=f16,n_q=66,n_kv=4096,n_sel=2051,group=1,gqa=4,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.06
SPARSE_ATTN_GQAREP portable G4           cases=  28 max_ratio=0.004  type_k=q8_0,type_v=q8_0,n_q=66,n_kv=4096,n_sel=2051,group=4,gqa=8,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end=0,scale=0.
TOTAL reports 1220 max ratio 0.006
```

## GPU vs f64 per tier and type (f64rep.py; reported, never gated, §14.2)

### final_attn_graphs

```
fast      bf16    cases= 243 max_f64_c18=2.640e-04 max_f64_plan=3.637e-04 max_nmse/band=0.462
fast      f16     cases= 317 max_f64_c18=2.784e-04 max_f64_plan=3.547e-04 max_nmse/band=0.738
fast      q4_0    cases= 251 max_f64_c18=2.624e-04 max_f64_plan=3.999e-04 max_nmse/band=0.462
fast      q8_0    cases= 310 max_f64_c18=2.861e-04 max_f64_plan=4.021e-04 max_nmse/band=0.767
generic   bf16    cases=  30 max_f64_c18=2.424e-04 max_f64_plan=3.784e-04 max_nmse/band=0.333
generic   f16     cases=  42 max_f64_c18=2.354e-04 max_f64_plan=3.402e-04 max_nmse/band=0.321
generic   f32     cases=  89 max_f64_c18=2.819e-04 max_f64_plan=3.904e-04 max_nmse/band=0.383
generic   iq4_nl  cases=  73 max_f64_c18=2.521e-04 max_f64_plan=3.736e-04 max_nmse/band=0.358
generic   q4_0    cases=  30 max_f64_c18=2.456e-04 max_f64_plan=3.682e-04 max_nmse/band=0.326
generic   q4_1    cases=  88 max_f64_c18=3.043e-04 max_f64_plan=3.752e-04 max_nmse/band=0.355
generic   q5_0    cases=  86 max_f64_c18=2.561e-04 max_f64_plan=3.752e-04 max_nmse/band=0.359
generic   q5_1    cases=  73 max_f64_c18=2.566e-04 max_f64_plan=3.787e-04 max_nmse/band=0.348
generic   q8_0    cases=  45 max_f64_c18=2.819e-04 max_f64_plan=3.904e-04 max_nmse/band=0.340
```

### final_portable

```
portable  bf16    cases= 273 max_f64_c18=2.821e-04 max_f64_plan=3.414e-04 max_nmse/band=0.002
portable  f16     cases= 359 max_f64_c18=2.925e-04 max_f64_plan=3.639e-04 max_nmse/band=0.006
portable  f32     cases=  89 max_f64_c18=3.218e-04 max_f64_plan=3.646e-04 max_nmse/band=0.001
portable  iq4_nl  cases=  73 max_f64_c18=2.841e-04 max_f64_plan=3.625e-04 max_nmse/band=0.002
portable  q4_0    cases= 281 max_f64_c18=2.905e-04 max_f64_plan=3.902e-04 max_nmse/band=0.003
portable  q4_1    cases=  88 max_f64_c18=2.770e-04 max_f64_plan=3.687e-04 max_nmse/band=0.002
portable  q5_0    cases=  86 max_f64_c18=2.770e-04 max_f64_plan=3.645e-04 max_nmse/band=0.002
portable  q5_1    cases=  73 max_f64_c18=2.975e-04 max_f64_plan=3.467e-04 max_nmse/band=0.002
portable  q8_0    cases= 355 max_f64_c18=3.218e-04 max_f64_plan=3.807e-04 max_nmse/band=0.002
```

## Kernel resources (tu_all.sh final: -Werror TU builds, gcn-cdna-vgpr-check.py parse_assembly)

```
EXIT tu_sparse-attn-attend-g4.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-merge.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-pack.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-portable.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-generic.cu 0 warnings=0 errors=0
EXIT tu_ggml-cuda.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-attend-g1.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-attend-g4-f16.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-attend-g4-q4_0.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-attend-g4-bf16.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-attend-g4-q8_0.cu 0 warnings=0 errors=0
sparse-attn-generic.cu       kernels=10 spilling=0 sa_kernels=10 sa_spilling=0 sa_max_vgprs=209 sa_max_scratch=0
sparse-attn-portable.cu      kernels=1 spilling=0 sa_kernels=1 sa_spilling=0 sa_max_vgprs=53 sa_max_scratch=0
sparse-attn-pack.cu          kernels=5 spilling=0 sa_kernels=5 sa_spilling=0 sa_max_vgprs=28 sa_max_scratch=0
sparse-attn-merge.cu         kernels=1 spilling=0 sa_kernels=1 sa_spilling=0 sa_max_vgprs=8 sa_max_scratch=0
sparse-attn-attend-g1.cu     kernels=48 spilling=0 sa_kernels=48 sa_spilling=0 sa_max_vgprs=131 sa_max_scratch=0
sparse-attn-attend-g4.cu     kernels=0 spilling=0 sa_kernels=0 sa_spilling=0 sa_max_vgprs=-1 sa_max_scratch=-1
sparse-attn-attend-g4-f16.cu kernels=56 spilling=0 sa_kernels=56 sa_spilling=0 sa_max_vgprs=192 sa_max_scratch=0
sparse-attn-attend-g4-bf16.cu kernels=56 spilling=0 sa_kernels=56 sa_spilling=0 sa_max_vgprs=192 sa_max_scratch=0
sparse-attn-attend-g4-q8_0.cu kernels=56 spilling=0 sa_kernels=56 sa_spilling=0 sa_max_vgprs=192 sa_max_scratch=0
sparse-attn-attend-g4-q4_0.cu kernels=56 spilling=0 sa_kernels=56 sa_spilling=0 sa_max_vgprs=192 sa_max_scratch=0
ggml-cuda.cu                 kernels=1 spilling=0 sa_kernels=0 sa_spilling=0 sa_max_vgprs=-1 sa_max_scratch=-1
TOTAL sa kernels=289 spilling=0 scratch>0=0
K _Z12sa_attend_g1ILi8ELi8ELi12EEv10sa_g1_args vgprs=146 spill=0 scratch=0
K _Z12sa_attend_g1ILi8ELi8ELi4EEv10sa_g1_args vgprs=146 spill=0 scratch=0
K _Z12sa_attend_g1ILi8ELi8ELi8EEv10sa_g1_args vgprs=146 spill=0 scratch=0
K _Z12sa_attend_g4ILi8ELi8ELi0ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi8ELi8ELi0ELi0ELi4EEv10sa_g4_args vgprs=119 spill=0 scratch=0
K _Z12sa_attend_g4ILi8ELi8ELi0ELi0ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi8ELi8ELi1ELi0ELi12EEv10sa_g4_args vgprs=209 spill=0 scratch=0
K _Z12sa_attend_g4ILi8ELi8ELi1ELi0ELi4EEv10sa_g4_args vgprs=136 spill=0 scratch=0
K _Z12sa_attend_g4ILi8ELi8ELi1ELi0ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z15sa_attend_fixupILi8ELi8ELi0EEv10sa_g4_argsi vgprs=30 spill=0 scratch=0
K _ZL18sa_attend_portable16sa_portable_args vgprs=53 spill=0 scratch=0
K _ZL9sa_pack_vILi0EEvPKcPtmmmijmPhPji vgprs=13 spill=0 scratch=0
K _ZL9sa_pack_vILi1EEvPKcPtmmmijmPhPji vgprs=13 spill=0 scratch=0
K _ZL9sa_pack_vILi2EEvPKcPtmmmijmPhPji vgprs=18 spill=0 scratch=0
K _ZL9sa_pack_vILi3EEvPKcPtmmmijmPhPji vgprs=18 spill=0 scratch=0
K _ZL9sa_pack_vILi8EEvPKcPtmmmijmPhPji vgprs=28 spill=0 scratch=0
```

## Multi-arch compile gate and device guards (O25)

`final.out`: EXIT ma_cfg 0, EXIT ma_build 0 from an empty `build-multiarch` (GPU_TARGETS="gfx1100;gfx1151;gfx90a;gfx942"), errors=0. The earlier build of the same sources (`ma_build.log`, EXIT 0, errors=0) also completed: it ran under nohup on strix-002 and outlived the orchestrator's SSH connection.

`harness/w5/arch_check.sh` compiles the three TUs with the multi-arch build's own command (`arch_check.out`). sa_attend_portable has device code with no v_permlanex16 on all four targets; the WMMA / pack bodies use v_permlanex16 only on the RDNA3 targets, and on gfx90a / gfx942 they are the NO_DEVICE_CODE stubs, never selected:

```
TU sparse-attn-portable rc=0 warnings=0 errors=0
sparse-attn-portable gfx1100 kernels: 1 with cross-lane ops (permlanex16): 0 sa_attend_portable descriptor: 1
sparse-attn-portable gfx1151 kernels: 1 with cross-lane ops (permlanex16): 0 sa_attend_portable descriptor: 1
sparse-attn-portable gfx90a kernels: 1 with cross-lane ops (permlanex16): 0 sa_attend_portable descriptor: 1
sparse-attn-portable gfx942 kernels: 1 with cross-lane ops (permlanex16): 0 sa_attend_portable descriptor: 1
TU sparse-attn-generic rc=0 warnings=0 errors=0
sparse-attn-generic gfx1100 kernels: 10 with cross-lane ops (permlanex16): 145 sa_attend_portable descriptor: 0
sparse-attn-generic gfx1151 kernels: 10 with cross-lane ops (permlanex16): 145 sa_attend_portable descriptor: 0
sparse-attn-generic gfx90a kernels: 10 with cross-lane ops (permlanex16): 0 sa_attend_portable descriptor: 0
sparse-attn-generic gfx942 kernels: 10 with cross-lane ops (permlanex16): 0 sa_attend_portable descriptor: 0
TU sparse-attn-pack rc=0 warnings=0 errors=0
sparse-attn-pack gfx1100 kernels: 5 with cross-lane ops (permlanex16): 5 sa_attend_portable descriptor: 0
sparse-attn-pack gfx1151 kernels: 5 with cross-lane ops (permlanex16): 5 sa_attend_portable descriptor: 0
sparse-attn-pack gfx90a kernels: 5 with cross-lane ops (permlanex16): 0 sa_attend_portable descriptor: 0
sparse-attn-pack gfx942 kernels: 5 with cross-lane ops (permlanex16): 0 sa_attend_portable descriptor: 0
ARCH_DONE
```

Warnings of the multi-arch build (`final_ma_build.log`): 72 host C++ warnings (54 deprecated enum arithmetic, 16 `u8path`, 1 `ggml_flash_attn_ext_set_prec`, 1 overloaded-virtual) and 52 "loop not unrolled", which is 13 per target if evenly split; the clean gfx1151 build also has 13 (`final_hip_build.log`). The 3 TUs above have warnings=0 on all four targets, and every TU of `tu_all.sh` builds under -Werror on gfx1151 (`tu_final.out`). The G1 / G4 TUs were not compiled alone for the other three targets, and which TUs emit the 13 was not attributed: UNPROVEN.
