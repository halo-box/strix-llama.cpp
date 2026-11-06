#!/bin/bash
# W5 O25: which SPARSE_ATTN kernels carry device code on each target of the multi-arch build, and the warnings of the portable /
# generic / pack TUs per arch (compiled alone, with the multi-arch build's own command). Read-only for the build tree.
cd ~/dev/wt_rewrite/build-multiarch; L=~/dev/wt_rewrite_logs/W5/arch; rm -rf $L; mkdir -p $L
B=/opt/rocm/core-10.1/lib/llvm/bin
for F in sparse-attn-portable sparse-attn-generic sparse-attn-pack; do
  CMD=$(python3 -c "
import json
for c in json.load(open('compile_commands.json')):
    if c['file'].endswith('ggml-cuda/$F.cu'): print(c['command'])
")
  mkdir -p $L/$F; ( cd $L/$F; CMD2=${CMD/-o CMakeFiles\/ggml-hip.dir\/__\/ggml-cuda\/$F.cu.o/-o $F.o}; eval "$CMD2" --save-temps > compile.log 2>&1; echo "TU $F rc=$? warnings=$(grep -c 'warning:' compile.log) errors=$(grep -c 'error:' compile.log)" )
  for A in gfx1100 gfx1151 gfx90a gfx942; do
    S=$(ls $L/$F/*${A}*lto.s 2>/dev/null | head -1)
    if [ -n "$S" ]; then
      echo "$F $A kernels: $(grep -E '^\s*\.amdhsa_kernel ' "$S" | sed 's/.*\.amdhsa_kernel //' | sed 's/\.kd$//' | sort -u | wc -l) with cross-lane ops (permlanex16): $(grep -c 'v_permlanex16' "$S") sa_attend_portable descriptor: $(grep -c '\.amdhsa_kernel .*sa_attend_portable' "$S")"
    else
      echo "$F $A no .lto.s"
    fi
  done
done
echo ARCH_DONE
