#!/bin/bash
cd ~/dev/wt_lab/r1
for t in "te_s1 0 8192" "te_s2 0 8192" "cb_s1 2 8192" "cb_s2 2 8192" "te_u4 0 4096" "cb_u4 2 4096"; do
  set -- $t
  ./run.sh txt_$1 text $2 $3 LAB_LENS=8192,8193,8194,8195 LAB_LOGITS_OUT=$PWD/logits/txt_$1
done
echo TEXT_DONE
