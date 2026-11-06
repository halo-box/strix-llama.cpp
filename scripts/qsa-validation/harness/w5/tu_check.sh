#!/bin/bash
# W2: compile one HIP TU with the build's own command plus -Werror -Wno-tautological-compare and the metrics flags
# (-Rpass-analysis=kernel-resource-usage --save-temps, as GGML_HIP_EXPORT_METRICS=On adds), in a scratch directory.
# usage: tu_check.sh <file under ggml/src/ggml-cuda> <outdir>
set -u
F=$1; O=$2; mkdir -p $O; cd ~/dev/wt_rewrite/build
CMD=$(python3 -c "
import json,sys
for c in json.load(open('compile_commands.json')):
    if c['file'].endswith('ggml-cuda/expand.cu'): print(c['command'])
")
CMD=${CMD//expand.cu.o/$(basename $F).o}
CMD=${CMD//ggml-cuda\/expand.cu/ggml-cuda\/$F}
CMD=${CMD/-o CMakeFiles\/ggml-hip.dir\/__\/ggml-cuda\//-o $O\/}
cd $O && eval "$CMD" -Werror -Wno-tautological-compare -Rpass-analysis=kernel-resource-usage --save-temps
