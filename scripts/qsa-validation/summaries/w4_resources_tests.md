# W4 resources and tests (final_w4.sh at 330ece83f)

```
HEAD 330ece83f status=[4 changed]
EXIT hip_build 0 warnings=0
EXIT vk_build 0 warnings=75
EXIT cpu_build 0 warnings=72
EXIT tu_sparse-attn-merge.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-attend-g4.cu 0 warnings=0 errors=0
EXIT tu_ggml-cuda.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-attend-g1.cu 0 warnings=0 errors=0
sparse-attn-attend-g1.cu     kernels=48 spilling=0 sa_kernels=48 sa_spilling=0 sa_max_vgprs=131 sa_max_scratch=0
sparse-attn-merge.cu         kernels=1 spilling=0 sa_kernels=1 sa_spilling=0 sa_max_vgprs=8 sa_max_scratch=0
sparse-attn-attend-g4.cu     kernels=0 spilling=0 sa_kernels=0 sa_spilling=0 sa_max_vgprs=-1 sa_max_scratch=-1
ggml-cuda.cu                 kernels=1 spilling=0 sa_kernels=0 sa_spilling=0 sa_max_vgprs=-1 sa_max_scratch=-1
TOTAL sa kernels=49 spilling=0 scratch>0=0
EXIT attn_graphs rc=1  854/940 tests passed fail=2 notsup=86  CENSUS COVERAGE: ROCm0: 469/469 compiled keys of {sa_attend/,sa_fixup/,sa_merge,sa_pack/} launched (0 excluded by a forcing switch): OK
EXIT attn_nographs rc=1  854/940 tests passed fail=2 notsup=86  CENSUS COVERAGE: ROCm0: 469/469 compiled keys of {sa_attend/,sa_fixup/,sa_merge,sa_pack/} launched (0 excluded by a forcing switch): OK
EXIT g1_vdirect rc=1  472/558 tests passed fail=2 notsup=86  CENSUS COVERAGE: ROCm0: 241/369 compiled keys of {sa_attend/,sa_fixup/,sa_merge,sa_pack/} launched (100 excluded by a forcing switch): not gated (-p filter)
EXIT g4_graphs rc=0  382/382 tests passed fail=0 notsup=0  CENSUS COVERAGE: ROCm0: 228/469 compiled keys of {sa_attend/,sa_fixup/,sa_merge,sa_pack/} launched (0 excluded by a forcing switch): not gated (-p filter)
EXIT cpu_attn rc=0  940/940 tests passed fail=0 notsup=0 
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
EXIT vk_attn rc=0  0/0 tests passed fail=0 notsup=940 
EXIT vkbuild_cpu_attn rc=0  940/940 tests passed fail=0 notsup=0 
EXIT cpuonly_attn rc=0  940/940 tests passed fail=0 notsup=0 
EXIT ctest rc=0 100% tests passed, 0 tests failed out of 1
DONE
```

## Kernel resources (tu_all.sh final: -Werror TU builds, gcn-cdna-vgpr-check.py parse_assembly)

```
EXIT tu_sparse-attn-merge.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-attend-g4.cu 0 warnings=0 errors=0
EXIT tu_ggml-cuda.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-attend-g1.cu 0 warnings=0 errors=0
sparse-attn-attend-g1.cu     kernels=48 spilling=0 sa_kernels=48 sa_spilling=0 sa_max_vgprs=131 sa_max_scratch=0
sparse-attn-merge.cu         kernels=1 spilling=0 sa_kernels=1 sa_spilling=0 sa_max_vgprs=8 sa_max_scratch=0
sparse-attn-attend-g4.cu     kernels=0 spilling=0 sa_kernels=0 sa_spilling=0 sa_max_vgprs=-1 sa_max_scratch=-1
ggml-cuda.cu                 kernels=1 spilling=0 sa_kernels=0 sa_spilling=0 sa_max_vgprs=-1 sa_max_scratch=-1
TOTAL sa kernels=49 spilling=0 scratch>0=0
K _Z12sa_attend_g1ILi0ELi0ELi12EEv10sa_g1_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g1ILi0ELi0ELi4EEv10sa_g1_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g1ILi0ELi0ELi8EEv10sa_g1_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g1ILi0ELi1ELi12EEv10sa_g1_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g1ILi0ELi1ELi4EEv10sa_g1_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g1ILi0ELi1ELi8EEv10sa_g1_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g1ILi0ELi2ELi12EEv10sa_g1_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g1ILi0ELi2ELi4EEv10sa_g1_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g1ILi0ELi2ELi8EEv10sa_g1_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g1ILi0ELi3ELi12EEv10sa_g1_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g1ILi0ELi3ELi4EEv10sa_g1_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g1ILi0ELi3ELi8EEv10sa_g1_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g1ILi1ELi0ELi12EEv10sa_g1_args vgprs=130 spill=0 scratch=0
K _Z12sa_attend_g1ILi1ELi0ELi4EEv10sa_g1_args vgprs=130 spill=0 scratch=0
K _Z12sa_attend_g1ILi1ELi0ELi8EEv10sa_g1_args vgprs=130 spill=0 scratch=0
K _Z12sa_attend_g1ILi1ELi1ELi12EEv10sa_g1_args vgprs=130 spill=0 scratch=0
K _Z12sa_attend_g1ILi1ELi1ELi4EEv10sa_g1_args vgprs=130 spill=0 scratch=0
K _Z12sa_attend_g1ILi1ELi1ELi8EEv10sa_g1_args vgprs=130 spill=0 scratch=0
K _Z12sa_attend_g1ILi1ELi2ELi12EEv10sa_g1_args vgprs=130 spill=0 scratch=0
K _Z12sa_attend_g1ILi1ELi2ELi4EEv10sa_g1_args vgprs=130 spill=0 scratch=0
K _Z12sa_attend_g1ILi1ELi2ELi8EEv10sa_g1_args vgprs=130 spill=0 scratch=0
K _Z12sa_attend_g1ILi1ELi3ELi12EEv10sa_g1_args vgprs=130 spill=0 scratch=0
K _Z12sa_attend_g1ILi1ELi3ELi4EEv10sa_g1_args vgprs=130 spill=0 scratch=0
K _Z12sa_attend_g1ILi1ELi3ELi8EEv10sa_g1_args vgprs=130 spill=0 scratch=0
K _Z12sa_attend_g1ILi2ELi0ELi12EEv10sa_g1_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g1ILi2ELi0ELi4EEv10sa_g1_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g1ILi2ELi0ELi8EEv10sa_g1_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g1ILi2ELi1ELi12EEv10sa_g1_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g1ILi2ELi1ELi4EEv10sa_g1_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g1ILi2ELi1ELi8EEv10sa_g1_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g1ILi2ELi2ELi12EEv10sa_g1_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g1ILi2ELi2ELi4EEv10sa_g1_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g1ILi2ELi2ELi8EEv10sa_g1_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g1ILi2ELi3ELi12EEv10sa_g1_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g1ILi2ELi3ELi4EEv10sa_g1_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g1ILi2ELi3ELi8EEv10sa_g1_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g1ILi3ELi0ELi12EEv10sa_g1_args vgprs=130 spill=0 scratch=0
K _Z12sa_attend_g1ILi3ELi0ELi4EEv10sa_g1_args vgprs=130 spill=0 scratch=0
K _Z12sa_attend_g1ILi3ELi0ELi8EEv10sa_g1_args vgprs=130 spill=0 scratch=0
K _Z12sa_attend_g1ILi3ELi1ELi12EEv10sa_g1_args vgprs=130 spill=0 scratch=0
K _Z12sa_attend_g1ILi3ELi1ELi4EEv10sa_g1_args vgprs=130 spill=0 scratch=0
K _Z12sa_attend_g1ILi3ELi1ELi8EEv10sa_g1_args vgprs=130 spill=0 scratch=0
K _Z12sa_attend_g1ILi3ELi2ELi12EEv10sa_g1_args vgprs=130 spill=0 scratch=0
K _Z12sa_attend_g1ILi3ELi2ELi4EEv10sa_g1_args vgprs=130 spill=0 scratch=0
K _Z12sa_attend_g1ILi3ELi2ELi8EEv10sa_g1_args vgprs=130 spill=0 scratch=0
K _Z12sa_attend_g1ILi3ELi3ELi12EEv10sa_g1_args vgprs=130 spill=0 scratch=0
K _Z12sa_attend_g1ILi3ELi3ELi4EEv10sa_g1_args vgprs=130 spill=0 scratch=0
K _Z12sa_attend_g1ILi3ELi3ELi8EEv10sa_g1_args vgprs=130 spill=0 scratch=0
K _ZL15sa_merge_splitsPKfPfi vgprs=8 spill=0 scratch=0
```

## G1 band reach (final_attn_graphs.log SA_REPORT, max NMSE / band per census split class)

- splits1/: max ratio 0.252 over 69 cases
- splits2-5: max ratio 0.418 over 157 cases
- splits6-16: max ratio 0.410 over 48 cases
- splits17-33: max ratio 0.466 over 150 cases
- splits34: max ratio 0.483 over 48 cases
