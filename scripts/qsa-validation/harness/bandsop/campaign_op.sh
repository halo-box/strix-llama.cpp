#!/bin/bash
# R2-M-LAB-BANDS-OP campaign. Every GPU run under flock ~/gpu.lock, one at a time.
# usage: campaign_op.sh tbo <run> | uni
cd ~/dev/wt_lab/bands/op
V_DEC=PR-dec-f16,PR-dec-q8_0; V_G4=PR-V0-f16,PR-V0-q8_0,PR-VD-f16,PR-VD-q8_0
g() { local t=$1; shift; echo "START $t $(date +%T)"; flock ~/gpu.lock env "$@" > res/$t.out 2> logs/$t.err; echo "EXIT $t $? $(date +%T)"; }
if [ "$1" = tbo ]; then
  r=$2
  for f in dump_$r/QSA_DECODE_MASKLESS_*.bdmp dump_$r/QSA_PREFILL_MASKLESS_*.bdmp; do
    b=$(basename $f .bdmp); op=${b%_*}; n=${b##*_}
    route=$(grep -B2 "BANDS_DUMP op/$f " logs/tbo_${r}_$op.log | grep QSA_ROUTE | tail -1)
    case "$route" in
      *"vt f16"*"kt"*) ;;
    esac
    kt=$(echo "$route" | sed 's/.* kt \([^ ]*\) vt \([^ ]*\).*/\1/'); vt=$(echo "$route" | sed 's/.* vt \([^ ]*\).*/\1/')
    if [ "$kt" != "$vt" ]; then echo "SKIP $b mixed $kt/$vt ($route)"; continue; fi
    tag=t_${r}_${b}
    if echo "$route" | grep -q "QSA_ROUTE decode"; then
      g $tag BANDS_DEFECTS=1 ./bands_dec TBO:$f SYN 0 $V_DEC $tag 32
      for rep in 2 3; do g ${tag}_rep$rep BANDS_DEFECTS=0 ./bands_dec TBO:$f SYN 0 $V_DEC ${tag}_rep$rep 32; done
    else
      g $tag BANDS_DEFECTS=1 ./bands_attn TBO:$f SYN all 1 $V_G4 $tag 32
    fi
  done
elif [ "$1" = uni ]; then
  S="1:257:4096:0 8:257:4096:0 9:257:4096:0 16:257:4096:0 64:257:4096:0 127:257:4096:0 128:257:4096:0 256:257:4096:0 512:257:4096:0
     1:2051:4096:0 8:2051:4096:0 9:2051:4096:0 16:2051:4096:0 64:2051:4096:0 127:2051:4096:0 128:2051:4096:0 256:2051:4096:0 512:2051:4096:0
     128:17:512:0 129:257:4096:0 130:2051:4096:0
     130:2051:4096:40 97:2051:4096:3 130:2051:32768:7000 200:2051:65536:15000 70:2051:65536:16000 130:2051:270336:66000
     1:2051:4096:100 2:2051:4096:100 9:2051:4096:100 1:2051:32768:8000 4:2051:32768:8000 1:2051:65536:16000 9:2051:65536:16000 9:2051:270336:66000
     513:2051:4096:0 640:257:4096:0
     1:2051:262144:0 512:2051:262144:0 4096:2051:262144:0 1:2051:262144:-1 512:2051:262144:-1 4096:2051:262144:-1"
  for seed in 1 2 3; do for s in $S; do
    IFS=: read nq ns nk bl <<< "$s"; tag=u_${nq}_${ns}_${nk}_b${bl}_s${seed}
    if [ $nq -gt 512 ]; then g $tag BANDS_DEFECTS=1 ./bands_attn UNI:1.0:$nk:$seed:$nq:$ns:$bl SYN all 1 $V_G4 $tag 32
    else g $tag BANDS_DEFECTS=1 ./bands_dec UNI:1.0:$nk:$seed:$nq:$ns:$bl SYN 0 $V_DEC $tag 32; fi
  done; done
  # same-binary regression against R1-M-Lab A's SYN 262k runs (res/dec1_syn262k.out, dec512_syn262k.out, attn_syn262k_s1.out)
  g syn_dec1   BANDS_DEFECTS=0 ./bands_dec SYN:1.8:262144:1 SYN 1 $V_DEC syn_dec1 32
  g syn_dec512 BANDS_DEFECTS=0 ./bands_dec SYN:1.8:262144:1 SYN 512 $V_DEC syn_dec512 32
  g syn_attn   BANDS_DEFECTS=0 ./bands_attn SYN:1.8:262144:1 SYN all 1 PR-V0-f16,PR-V0-q8_0 syn_attn 32
fi
echo DONE $(date +%T)
