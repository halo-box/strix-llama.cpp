#!/bin/bash
# W4: -Werror + resource-usage TU builds of the HIP TUs W4 adds or changes (in parallel, one build slot under flock), then
# the repo VGPR checker's parser (scripts/hip/gcn-cdna-vgpr-check.py parse_assembly) over their code objects
# usage: tu_all.sh <tag>
T=${1:-t}; L=~/dev/wt_rewrite_logs/W4; cd ~/dev/wt_rewrite
FILES="sparse-attn-attend-g1.cu sparse-attn-merge.cu sparse-attn-attend-g4.cu ggml-cuda.cu"
for F in $FILES; do
  rm -rf $L/tu_${T}_${F%.cu}
  ( $L/tu_check.sh $F $L/tu_${T}_${F%.cu} > $L/tu_${T}_${F%.cu}.log 2>&1; echo "EXIT tu_$F $? warnings=$(grep -c "warning:" $L/tu_${T}_${F%.cu}.log) errors=$(grep -c "error:" $L/tu_${T}_${F%.cu}.log)" ) &
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
    sa = {k: v for k, v in ks.items() if "sa_attend" in k or "sa_pack" in k or "sa_merge" in k}
    allk.update(sa)
    print("%-28s kernels=%d spilling=%d sa_kernels=%d sa_spilling=%d sa_max_vgprs=%d sa_max_scratch=%d" % (F, len(ks), sum(v["spill_stores"] > 0 for v in ks.values()), len(sa), sum(v["spill_stores"] > 0 for v in sa.values()), max([v["vgprs"] for v in sa.values()] or [-1]), max([v["scratch_bytes"] for v in sa.values()] or [-1])))
print("TOTAL sa kernels=%d spilling=%d scratch>0=%d" % (len(allk), sum(v["spill_stores"] > 0 for v in allk.values()), sum(v["scratch_bytes"] > 0 for v in allk.values())))
for k, v in sorted(allk.items()):
    print("K %s vgprs=%d spill=%d scratch=%d" % (k, v["vgprs"], v["spill_stores"], v["scratch_bytes"]))
PY
