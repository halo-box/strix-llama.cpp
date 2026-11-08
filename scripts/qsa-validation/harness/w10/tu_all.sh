#!/bin/bash
# W10: -Werror + resource-usage TU builds of the HIP TUs the in-op rotations change (in parallel, run under flock), then the
# repo VGPR checker's parser (scripts/hip/gcn-cdna-vgpr-check.py parse_assembly) over their code objects: 0 spills, 0 scratch
# (PLAN-v3 §6.2 resource rule); plus the same TUs compiled with the multi-arch build's command (gfx1100;gfx1151;gfx90a;gfx942,
# the W5 compile gate) for the device code every arch carries (portable, merge, fwht).
# usage: tu_all.sh <tag>
T=${1:-t}; L=~/dev/wt_rewrite_logs/W10; cd ~/dev/wt_rewrite
CHK=scripts/qsa-validation/harness/w5/tu_check.sh
FILES="sparse-attn-generic.cu sparse-attn-portable.cu sparse-attn-merge.cu sparse-attn-attend-g1.cu sparse-attn-attend-g4.cu sparse-attn-attend-g4-f16.cu sparse-attn-attend-g4-bf16.cu sparse-attn-attend-g4-q8_0.cu sparse-attn-attend-g4-q4_0.cu fwht.cu"
for F in $FILES; do
  rm -rf $L/tu_${T}_${F%.cu}
  ( bash $CHK $F $L/tu_${T}_${F%.cu} > $L/tu_${T}_${F%.cu}.log 2>&1; echo "EXIT tu_$F $? warnings=$(grep -c "warning:" $L/tu_${T}_${F%.cu}.log) errors=$(grep -c "error:" $L/tu_${T}_${F%.cu}.log)" ) &
done
wait
python3 - $L $T $FILES <<"PY"
import importlib.util, sys, pathlib
spec = importlib.util.spec_from_file_location("c", "scripts/hip/gcn-cdna-vgpr-check.py"); c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
L, T = pathlib.Path(sys.argv[1]), sys.argv[2]
allk = {}
for F in sys.argv[3:]:
    d = L / ("tu_%s_%s" % (T, F[:-3])); ks = {}
    for p in sorted(d.glob("*.lto.s")): ks.update(c.parse_assembly(p.read_text(), "gfx1151"))
    sa = {k: v for k, v in ks.items() if "sa_attend" in k or "sa_pack" in k or "sa_merge" in k or "sa_portable" in k or "fwht" in k}
    allk.update(sa)
    print("%-30s kernels=%d spilling=%d sa_kernels=%d sa_spilling=%d sa_max_vgprs=%d sa_max_scratch=%d" % (F, len(ks), sum(v["spill_stores"] > 0 for v in ks.values()), len(sa), sum(v["spill_stores"] > 0 for v in sa.values()), max([v["vgprs"] for v in sa.values()] or [-1]), max([v["scratch_bytes"] for v in sa.values()] or [-1])))
print("TOTAL sa+fwht kernels=%d spilling=%d scratch>0=%d" % (len(allk), sum(v["spill_stores"] > 0 for v in allk.values()), sum(v["scratch_bytes"] > 0 for v in allk.values())))
for k, v in sorted(allk.items()):
    print("K %s vgprs=%d spill=%d scratch=%d" % (k, v["vgprs"], v["spill_stores"], v["scratch_bytes"]))
PY
# multi-arch compile of the TUs whose device code exists on every arch
MA=~/dev/wt_rewrite/build-multiarch
for F in sparse-attn-portable sparse-attn-merge fwht sparse-attn-attend-g1 sparse-attn-attend-g4-q8_0; do
  CMD=$(python3 -c "
import json
for c in json.load(open('$MA/compile_commands.json')):
    if c['file'].endswith('ggml-cuda/$F.cu'): print(c['command'])
")
  D=$L/ma_${T}_$F; rm -rf $D; mkdir -p $D
  ( cd $MA/ggml/src/ggml-hip && CMD2=${CMD/-o CMakeFiles\/ggml-hip.dir\/__\/ggml-cuda\/$F.cu.o/-o $D/$F.o}; eval "$CMD2" -Werror -Wno-tautological-compare > $D/compile.log 2>&1; echo "MULTIARCH $F rc=$? warnings=$(grep -c 'warning:' $D/compile.log) errors=$(grep -c 'error:' $D/compile.log)" )
done
echo TU_DONE
