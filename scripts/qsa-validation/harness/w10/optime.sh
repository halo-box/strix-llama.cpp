#!/bin/bash
# W10 op-time A/B (indicative; the full interleaved campaign is Prove's): the W5 op-time harness (w5_time.cpp: plan + R
# attends on the C18 wiki rows, SYN data, G4 at the C18 prefill shape) built once against the ggml headers and run against
# the W9 head's libraries (the saved build/bin) and this build's, rot 0 and, for this build, rot (256, 64) (env
# W10_ROT_Q / W10_ROT_O of the copy); two processes per arm, interleaved.
S=~/dev/wt_rewrite; D=~/dev/wt_lab/w10/time; BASE=${BASE:-~/dev/wt_lab/w10/base_bin}; C=~/dev/wt_lab/c18/cap
mkdir -p $D && cd $D
sed 's/ggml_sparse_attn(ctx, q, k, v, plan, 1.0f \/ 16.0f, 0, 0)/ggml_sparse_attn(ctx, q, k, v, plan, 1.0f \/ 16.0f, getenv("W10_ROT_Q") ? atoi(getenv("W10_ROT_Q")) : 0, getenv("W10_ROT_O") ? atoi(getenv("W10_ROT_O")) : 0)/' \
  $S/scripts/qsa-validation/harness/w5/w5_time.cpp > w10_time.cpp
grep -c W10_ROT_Q w10_time.cpp | xargs echo "rot env patched:"
g++ -O2 -std=c++17 -I $S/ggml/include w10_time.cpp -o w10_time -L $S/build/bin -lggml -lggml-base > build.log 2>&1; echo "BUILD rc=$?"
for f in ${ROWS:-wikiS_p4096 wikiL_p131072}; do
  for r in r1 r2; do
    flock ~/gpu.lock env LD_LIBRARY_PATH=$BASE ./w10_time $C/$f.rows 10 4 ${TYPES:-f16,q8_0} > ${f}_w9_$r.out 2>&1
    echo "RUN $f w9 $r rc=$? $(grep '^SUM' ${f}_w9_$r.out | awk '{print $2"="$6}' | tr '\n' ' ')"
    flock ~/gpu.lock env LD_LIBRARY_PATH=$S/build/bin ./w10_time $C/$f.rows 10 4 ${TYPES:-f16,q8_0} > ${f}_w10_$r.out 2>&1
    echo "RUN $f w10 $r rc=$? $(grep '^SUM' ${f}_w10_$r.out | awk '{print $2"="$6}' | tr '\n' ' ')"
    flock ~/gpu.lock env LD_LIBRARY_PATH=$S/build/bin W10_ROT_Q=256 W10_ROT_O=64 ./w10_time $C/$f.rows 10 4 ${TYPES:-f16,q8_0} > ${f}_w10rot_$r.out 2>&1
    echo "RUN $f w10rot $r rc=$? $(grep '^SUM' ${f}_w10rot_$r.out | awk '{print $2"="$6}' | tr '\n' ' ')"
  done
done
echo TIME_DONE
