# W3: resources and tests at the committed tree (tree of 05a1b6cd6; the run printed a1b0aa2dd, the same tree before a message-only amend)

## Resources (`harness/w3/tu_all.sh final`: each W3 HIP TU built with the build's own command + `-Werror
## -Wno-tautological-compare -Rpass-analysis=kernel-resource-usage --save-temps`, code objects parsed with
## `scripts/hip/gcn-cdna-vgpr-check.py parse_assembly`; K lines: kernel, VGPRs, spilled VGPRs, scratch bytes)

```
EXIT tu_sparse-attn-attend-g4.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-pack.cu 0 warnings=0 errors=0
EXIT tu_ggml-cuda.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-attend-g4-bf16.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-attend-g4-f16.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-attend-g4-q4_0.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-attend-g4-q8_0.cu 0 warnings=0 errors=0
sparse-attn-attend-g4-f16.cu     kernels=56 spilling=0 sa_kernels=56 sa_spilling=0 sa_max_vgprs=192 sa_max_scratch=0
sparse-attn-attend-g4-bf16.cu    kernels=56 spilling=0 sa_kernels=56 sa_spilling=0 sa_max_vgprs=192 sa_max_scratch=0
sparse-attn-attend-g4-q8_0.cu    kernels=56 spilling=0 sa_kernels=56 sa_spilling=0 sa_max_vgprs=192 sa_max_scratch=0
sparse-attn-attend-g4-q4_0.cu    kernels=56 spilling=0 sa_kernels=56 sa_spilling=0 sa_max_vgprs=192 sa_max_scratch=0
sparse-attn-attend-g4.cu         kernels=0 spilling=0 sa_kernels=0 sa_spilling=0 sa_max_vgprs=-1 sa_max_scratch=-1
sparse-attn-pack.cu              kernels=4 spilling=0 sa_kernels=4 sa_spilling=0 sa_max_vgprs=18 sa_max_scratch=0
ggml-cuda.cu                     kernels=1 spilling=0 sa_kernels=0 sa_spilling=0 sa_max_vgprs=-1 sa_max_scratch=-1
TOTAL sa kernels=228 spilling=0 scratch>0=0
K _Z12sa_attend_g4ILi0ELi0ELi0ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi0ELi0ELi0ELi4EEv10sa_g4_args vgprs=126 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi0ELi0ELi0ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi0ELi0ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi0ELi0ELi1ELi4EEv10sa_g4_args vgprs=127 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi0ELi0ELi1ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi0ELi1ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi0ELi1ELi0ELi4EEv10sa_g4_args vgprs=122 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi0ELi1ELi0ELi8EEv10sa_g4_args vgprs=163 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi0ELi1ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi0ELi1ELi1ELi4EEv10sa_g4_args vgprs=123 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi0ELi1ELi1ELi8EEv10sa_g4_args vgprs=164 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi1ELi0ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi1ELi0ELi0ELi4EEv10sa_g4_args vgprs=126 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi1ELi0ELi0ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi1ELi0ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi1ELi0ELi1ELi4EEv10sa_g4_args vgprs=127 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi1ELi0ELi1ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi1ELi1ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi1ELi1ELi0ELi4EEv10sa_g4_args vgprs=109 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi1ELi1ELi0ELi8EEv10sa_g4_args vgprs=163 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi1ELi1ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi1ELi1ELi1ELi4EEv10sa_g4_args vgprs=110 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi1ELi1ELi1ELi8EEv10sa_g4_args vgprs=164 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi2ELi0ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi2ELi0ELi0ELi4EEv10sa_g4_args vgprs=126 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi2ELi0ELi0ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi2ELi0ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi2ELi0ELi1ELi4EEv10sa_g4_args vgprs=127 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi2ELi0ELi1ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi2ELi1ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi2ELi1ELi0ELi4EEv10sa_g4_args vgprs=135 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi2ELi1ELi0ELi8EEv10sa_g4_args vgprs=156 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi2ELi1ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi2ELi1ELi1ELi4EEv10sa_g4_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi2ELi1ELi1ELi8EEv10sa_g4_args vgprs=157 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi3ELi0ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi3ELi0ELi0ELi4EEv10sa_g4_args vgprs=126 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi3ELi0ELi0ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi3ELi0ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi3ELi0ELi1ELi4EEv10sa_g4_args vgprs=127 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi3ELi0ELi1ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi3ELi1ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi3ELi1ELi0ELi4EEv10sa_g4_args vgprs=132 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi3ELi1ELi0ELi8EEv10sa_g4_args vgprs=157 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi3ELi1ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi3ELi1ELi1ELi4EEv10sa_g4_args vgprs=132 spill=0 scratch=0
K _Z12sa_attend_g4ILi0ELi3ELi1ELi1ELi8EEv10sa_g4_args vgprs=158 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi0ELi0ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi0ELi0ELi0ELi4EEv10sa_g4_args vgprs=126 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi0ELi0ELi0ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi0ELi0ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi0ELi0ELi1ELi4EEv10sa_g4_args vgprs=127 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi0ELi0ELi1ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi0ELi1ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi0ELi1ELi0ELi4EEv10sa_g4_args vgprs=122 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi0ELi1ELi0ELi8EEv10sa_g4_args vgprs=163 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi0ELi1ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi0ELi1ELi1ELi4EEv10sa_g4_args vgprs=123 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi0ELi1ELi1ELi8EEv10sa_g4_args vgprs=164 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi1ELi0ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi1ELi0ELi0ELi4EEv10sa_g4_args vgprs=126 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi1ELi0ELi0ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi1ELi0ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi1ELi0ELi1ELi4EEv10sa_g4_args vgprs=127 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi1ELi0ELi1ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi1ELi1ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi1ELi1ELi0ELi4EEv10sa_g4_args vgprs=109 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi1ELi1ELi0ELi8EEv10sa_g4_args vgprs=163 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi1ELi1ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi1ELi1ELi1ELi4EEv10sa_g4_args vgprs=110 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi1ELi1ELi1ELi8EEv10sa_g4_args vgprs=164 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi2ELi0ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi2ELi0ELi0ELi4EEv10sa_g4_args vgprs=126 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi2ELi0ELi0ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi2ELi0ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi2ELi0ELi1ELi4EEv10sa_g4_args vgprs=127 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi2ELi0ELi1ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi2ELi1ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi2ELi1ELi0ELi4EEv10sa_g4_args vgprs=135 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi2ELi1ELi0ELi8EEv10sa_g4_args vgprs=156 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi2ELi1ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi2ELi1ELi1ELi4EEv10sa_g4_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi2ELi1ELi1ELi8EEv10sa_g4_args vgprs=157 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi3ELi0ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi3ELi0ELi0ELi4EEv10sa_g4_args vgprs=126 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi3ELi0ELi0ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi3ELi0ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi3ELi0ELi1ELi4EEv10sa_g4_args vgprs=127 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi3ELi0ELi1ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi3ELi1ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi3ELi1ELi0ELi4EEv10sa_g4_args vgprs=132 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi3ELi1ELi0ELi8EEv10sa_g4_args vgprs=157 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi3ELi1ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi3ELi1ELi1ELi4EEv10sa_g4_args vgprs=132 spill=0 scratch=0
K _Z12sa_attend_g4ILi1ELi3ELi1ELi1ELi8EEv10sa_g4_args vgprs=158 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi0ELi0ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi0ELi0ELi0ELi4EEv10sa_g4_args vgprs=126 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi0ELi0ELi0ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi0ELi0ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi0ELi0ELi1ELi4EEv10sa_g4_args vgprs=127 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi0ELi0ELi1ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi0ELi1ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi0ELi1ELi0ELi4EEv10sa_g4_args vgprs=122 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi0ELi1ELi0ELi8EEv10sa_g4_args vgprs=163 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi0ELi1ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi0ELi1ELi1ELi4EEv10sa_g4_args vgprs=123 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi0ELi1ELi1ELi8EEv10sa_g4_args vgprs=164 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi1ELi0ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi1ELi0ELi0ELi4EEv10sa_g4_args vgprs=126 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi1ELi0ELi0ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi1ELi0ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi1ELi0ELi1ELi4EEv10sa_g4_args vgprs=127 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi1ELi0ELi1ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi1ELi1ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi1ELi1ELi0ELi4EEv10sa_g4_args vgprs=111 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi1ELi1ELi0ELi8EEv10sa_g4_args vgprs=163 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi1ELi1ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi1ELi1ELi1ELi4EEv10sa_g4_args vgprs=112 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi1ELi1ELi1ELi8EEv10sa_g4_args vgprs=164 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi2ELi0ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi2ELi0ELi0ELi4EEv10sa_g4_args vgprs=126 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi2ELi0ELi0ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi2ELi0ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi2ELi0ELi1ELi4EEv10sa_g4_args vgprs=127 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi2ELi0ELi1ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi2ELi1ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi2ELi1ELi0ELi4EEv10sa_g4_args vgprs=135 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi2ELi1ELi0ELi8EEv10sa_g4_args vgprs=156 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi2ELi1ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi2ELi1ELi1ELi4EEv10sa_g4_args vgprs=131 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi2ELi1ELi1ELi8EEv10sa_g4_args vgprs=157 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi3ELi0ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi3ELi0ELi0ELi4EEv10sa_g4_args vgprs=126 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi3ELi0ELi0ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi3ELi0ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi3ELi0ELi1ELi4EEv10sa_g4_args vgprs=127 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi3ELi0ELi1ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi3ELi1ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi3ELi1ELi0ELi4EEv10sa_g4_args vgprs=132 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi3ELi1ELi0ELi8EEv10sa_g4_args vgprs=157 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi3ELi1ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi3ELi1ELi1ELi4EEv10sa_g4_args vgprs=132 spill=0 scratch=0
K _Z12sa_attend_g4ILi2ELi3ELi1ELi1ELi8EEv10sa_g4_args vgprs=158 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi0ELi0ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi0ELi0ELi0ELi4EEv10sa_g4_args vgprs=119 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi0ELi0ELi0ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi0ELi0ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi0ELi0ELi1ELi4EEv10sa_g4_args vgprs=119 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi0ELi0ELi1ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi0ELi1ELi0ELi12EEv10sa_g4_args vgprs=190 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi0ELi1ELi0ELi4EEv10sa_g4_args vgprs=119 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi0ELi1ELi0ELi8EEv10sa_g4_args vgprs=152 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi0ELi1ELi1ELi12EEv10sa_g4_args vgprs=191 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi0ELi1ELi1ELi4EEv10sa_g4_args vgprs=120 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi0ELi1ELi1ELi8EEv10sa_g4_args vgprs=161 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi1ELi0ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi1ELi0ELi0ELi4EEv10sa_g4_args vgprs=119 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi1ELi0ELi0ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi1ELi0ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi1ELi0ELi1ELi4EEv10sa_g4_args vgprs=119 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi1ELi0ELi1ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi1ELi1ELi0ELi12EEv10sa_g4_args vgprs=191 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi1ELi1ELi0ELi4EEv10sa_g4_args vgprs=120 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi1ELi1ELi0ELi8EEv10sa_g4_args vgprs=152 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi1ELi1ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi1ELi1ELi1ELi4EEv10sa_g4_args vgprs=120 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi1ELi1ELi1ELi8EEv10sa_g4_args vgprs=161 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi2ELi0ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi2ELi0ELi0ELi4EEv10sa_g4_args vgprs=119 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi2ELi0ELi0ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi2ELi0ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi2ELi0ELi1ELi4EEv10sa_g4_args vgprs=119 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi2ELi0ELi1ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi2ELi1ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi2ELi1ELi0ELi4EEv10sa_g4_args vgprs=132 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi2ELi1ELi0ELi8EEv10sa_g4_args vgprs=153 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi2ELi1ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi2ELi1ELi1ELi4EEv10sa_g4_args vgprs=133 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi2ELi1ELi1ELi8EEv10sa_g4_args vgprs=154 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi3ELi0ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi3ELi0ELi0ELi4EEv10sa_g4_args vgprs=119 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi3ELi0ELi0ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi3ELi0ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi3ELi0ELi1ELi4EEv10sa_g4_args vgprs=119 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi3ELi0ELi1ELi8EEv10sa_g4_args vgprs=171 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi3ELi1ELi0ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi3ELi1ELi0ELi4EEv10sa_g4_args vgprs=136 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi3ELi1ELi0ELi8EEv10sa_g4_args vgprs=154 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi3ELi1ELi1ELi12EEv10sa_g4_args vgprs=192 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi3ELi1ELi1ELi4EEv10sa_g4_args vgprs=137 spill=0 scratch=0
K _Z12sa_attend_g4ILi3ELi3ELi1ELi1ELi8EEv10sa_g4_args vgprs=155 spill=0 scratch=0
K _Z15sa_attend_fixupILi0ELi0ELi0EEv10sa_g4_argsi vgprs=27 spill=0 scratch=0
K _Z15sa_attend_fixupILi0ELi0ELi1EEv10sa_g4_argsi vgprs=31 spill=0 scratch=0
K _Z15sa_attend_fixupILi0ELi1ELi0EEv10sa_g4_argsi vgprs=27 spill=0 scratch=0
K _Z15sa_attend_fixupILi0ELi1ELi1EEv10sa_g4_argsi vgprs=31 spill=0 scratch=0
K _Z15sa_attend_fixupILi0ELi2ELi0EEv10sa_g4_argsi vgprs=28 spill=0 scratch=0
K _Z15sa_attend_fixupILi0ELi2ELi1EEv10sa_g4_argsi vgprs=32 spill=0 scratch=0
K _Z15sa_attend_fixupILi0ELi3ELi0EEv10sa_g4_argsi vgprs=28 spill=0 scratch=0
K _Z15sa_attend_fixupILi0ELi3ELi1EEv10sa_g4_argsi vgprs=32 spill=0 scratch=0
K _Z15sa_attend_fixupILi1ELi0ELi0EEv10sa_g4_argsi vgprs=27 spill=0 scratch=0
K _Z15sa_attend_fixupILi1ELi0ELi1EEv10sa_g4_argsi vgprs=31 spill=0 scratch=0
K _Z15sa_attend_fixupILi1ELi1ELi0EEv10sa_g4_argsi vgprs=27 spill=0 scratch=0
K _Z15sa_attend_fixupILi1ELi1ELi1EEv10sa_g4_argsi vgprs=31 spill=0 scratch=0
K _Z15sa_attend_fixupILi1ELi2ELi0EEv10sa_g4_argsi vgprs=28 spill=0 scratch=0
K _Z15sa_attend_fixupILi1ELi2ELi1EEv10sa_g4_argsi vgprs=32 spill=0 scratch=0
K _Z15sa_attend_fixupILi1ELi3ELi0EEv10sa_g4_argsi vgprs=28 spill=0 scratch=0
K _Z15sa_attend_fixupILi1ELi3ELi1EEv10sa_g4_argsi vgprs=32 spill=0 scratch=0
K _Z15sa_attend_fixupILi2ELi0ELi0EEv10sa_g4_argsi vgprs=28 spill=0 scratch=0
K _Z15sa_attend_fixupILi2ELi0ELi1EEv10sa_g4_argsi vgprs=32 spill=0 scratch=0
K _Z15sa_attend_fixupILi2ELi1ELi0EEv10sa_g4_argsi vgprs=28 spill=0 scratch=0
K _Z15sa_attend_fixupILi2ELi1ELi1EEv10sa_g4_argsi vgprs=32 spill=0 scratch=0
K _Z15sa_attend_fixupILi2ELi2ELi0EEv10sa_g4_argsi vgprs=28 spill=0 scratch=0
K _Z15sa_attend_fixupILi2ELi2ELi1EEv10sa_g4_argsi vgprs=32 spill=0 scratch=0
K _Z15sa_attend_fixupILi2ELi3ELi0EEv10sa_g4_argsi vgprs=29 spill=0 scratch=0
K _Z15sa_attend_fixupILi2ELi3ELi1EEv10sa_g4_argsi vgprs=33 spill=0 scratch=0
K _Z15sa_attend_fixupILi3ELi0ELi0EEv10sa_g4_argsi vgprs=28 spill=0 scratch=0
K _Z15sa_attend_fixupILi3ELi0ELi1EEv10sa_g4_argsi vgprs=32 spill=0 scratch=0
K _Z15sa_attend_fixupILi3ELi1ELi0EEv10sa_g4_argsi vgprs=28 spill=0 scratch=0
K _Z15sa_attend_fixupILi3ELi1ELi1EEv10sa_g4_argsi vgprs=32 spill=0 scratch=0
K _Z15sa_attend_fixupILi3ELi2ELi0EEv10sa_g4_argsi vgprs=29 spill=0 scratch=0
K _Z15sa_attend_fixupILi3ELi2ELi1EEv10sa_g4_argsi vgprs=33 spill=0 scratch=0
K _Z15sa_attend_fixupILi3ELi3ELi0EEv10sa_g4_argsi vgprs=28 spill=0 scratch=0
K _Z15sa_attend_fixupILi3ELi3ELi1EEv10sa_g4_argsi vgprs=32 spill=0 scratch=0
K _ZL9sa_pack_vILi0EEvPKcPtmmmijmPhPj vgprs=13 spill=0 scratch=0
K _ZL9sa_pack_vILi1EEvPKcPtmmmijmPhPj vgprs=13 spill=0 scratch=0
K _ZL9sa_pack_vILi2EEvPKcPtmmmijmPhPj vgprs=18 spill=0 scratch=0
K _ZL9sa_pack_vILi3EEvPKcPtmmmijmPhPj vgprs=18 spill=0 scratch=0
```

## Builds and tests (`harness/w3/final_w3.sh`; logs `strix-002:~/dev/wt_rewrite_logs/W3/final_*.log`)

The 8 warnings of each build are in tests/test-backend-ops.cpp at lines W3 does not touch (6 deprecated enum-enum
operations, a hidden max_err overload, a deprecated ggml_flash_attn_ext_set_prec), as in W2.

```
HEAD a1b0aa2dd status=[1 changed]
EXIT final_hip_build 0 warnings=8
EXIT final_vk_build 0 warnings=8
EXIT final_cpu_build 0 warnings=8
EXIT tu_sparse-attn-attend-g4.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-pack.cu 0 warnings=0 errors=0
EXIT tu_ggml-cuda.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-attend-g4-bf16.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-attend-g4-f16.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-attend-g4-q4_0.cu 0 warnings=0 errors=0
EXIT tu_sparse-attn-attend-g4-q8_0.cu 0 warnings=0 errors=0
sparse-attn-attend-g4-f16.cu     kernels=56 spilling=0 sa_kernels=56 sa_spilling=0 sa_max_vgprs=192 sa_max_scratch=0
sparse-attn-attend-g4-bf16.cu    kernels=56 spilling=0 sa_kernels=56 sa_spilling=0 sa_max_vgprs=192 sa_max_scratch=0
sparse-attn-attend-g4-q8_0.cu    kernels=56 spilling=0 sa_kernels=56 sa_spilling=0 sa_max_vgprs=192 sa_max_scratch=0
sparse-attn-attend-g4-q4_0.cu    kernels=56 spilling=0 sa_kernels=56 sa_spilling=0 sa_max_vgprs=192 sa_max_scratch=0
sparse-attn-attend-g4.cu         kernels=0 spilling=0 sa_kernels=0 sa_spilling=0 sa_max_vgprs=-1 sa_max_scratch=-1
sparse-attn-pack.cu              kernels=4 spilling=0 sa_kernels=4 sa_spilling=0 sa_max_vgprs=18 sa_max_scratch=0
ggml-cuda.cu                     kernels=1 spilling=0 sa_kernels=0 sa_spilling=0 sa_max_vgprs=-1 sa_max_scratch=-1
TOTAL sa kernels=228 spilling=0 scratch>0=0
EXIT g4_graphs rc=0  382/382 tests passed fail=0 notsup=0  CENSUS COVERAGE: ROCm0: 228/228 compiled keys of {sa_attend/,sa_fixup/,sa_pack/} launched (0 excluded by a forcing switch): OK
EXIT g4_nographs rc=0  382/382 tests passed fail=0 notsup=0  CENSUS COVERAGE: ROCm0: 228/228 compiled keys of {sa_attend/,sa_fixup/,sa_pack/} launched (0 excluded by a forcing switch): OK
EXIT g4_vdirect rc=0  382/382 tests passed fail=0 notsup=0  CENSUS COVERAGE: ROCm0: 128/128 compiled keys of {sa_attend/,sa_fixup/,sa_pack/} launched (100 excluded by a forcing switch): OK
EXIT g4_vpacked rc=0  382/382 tests passed fail=0 notsup=0  CENSUS COVERAGE: ROCm0: 132/132 compiled keys of {sa_attend/,sa_fixup/,sa_pack/} launched (96 excluded by a forcing switch): OK
EXIT attn_all rc=0  382/382 tests passed fail=0 notsup=219  CENSUS COVERAGE: ROCm0: 228/228 compiled keys of {sa_attend/,sa_fixup/,sa_pack/} launched (0 excluded by a forcing switch): OK
EXIT cpu_attn rc=0  601/601 tests passed fail=0 notsup=0 
EXIT plan_rocm rc=0  185/185 tests passed fail=0 notsup=0  CENSUS COVERAGE: ROCm0: 2/2 compiled keys of {sa_plan/} launched (0 excluded by a forcing switch): OK
EXIT plan_cpu rc=0  185/185 tests passed fail=0 notsup=0 
EXIT QSA_PREFILL_MASKLESS rc=0  26/26 tests passed fail=0 notsup=0 
EXIT QSA_DECODE_MASKLESS rc=0  52/52 tests passed fail=0 notsup=1 
EXIT QSA_PREFILL rc=0  26/26 tests passed fail=0 notsup=0 
EXIT QSA_DECODE rc=0  92/92 tests passed fail=0 notsup=0 
EXIT ctest rc=0 100% tests passed, 0 tests failed out of 1
DONE
EXIT vk_attn_g4 rc=0  0/0 tests passed fail=0 notsup=382
EXIT vk_plan rc=0  0/0 tests passed fail=0 notsup=185
EXIT vk_QSA_PREFILL_MASKLESS rc=0  14/14 tests passed fail=0 notsup=12
EXIT vk_QSA_DECODE_MASKLESS rc=0  34/34 tests passed fail=0 notsup=19
EXIT vk_QSA_PREFILL rc=0  26/26 tests passed fail=0 notsup=0
EXIT vk_QSA_DECODE rc=0  92/92 tests passed fail=0 notsup=0
EXIT cpuonly_attn rc=0  601/601 tests passed fail=0 notsup=0
EXIT vkbuild_cpu_attn rc=0  601/601 tests passed fail=0 notsup=0
DONE
```
