#!/bin/bash
# compile and print VGPR/scratch of c5 kernels
cd ~/dev/wt_lab/c5 && /opt/rocm/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -Rpass-analysis=kernel-resource-usage c5_attn.cu -o ${1:-c5_attn} > logs/build_attn.log 2>&1; echo BUILD $?
grep -E "Function Name|VGPRs:|ScratchSize" logs/build_attn.log | grep -A2 "attn_kernel" | paste - - - | sed -E 's/.*Name: _Z[0-9]+//; s/EvPK.*VGPRs:/ VGPR/; s/\.\/[a-z0-9_]+\.inc:[0-9]+:0: ScratchSize \[bytes\/lane\]:/ scratch/'
