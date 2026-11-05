#!/bin/bash
cd ~/dev/wt_lab/r1
while ! grep -q TEXT_DONE text_campaign.out; do sleep 10; done
P=LAB_PROMPT_FILE=$PWD/p3.txt
./run.sh img_te_gen image 0 8192 $P
./run.sh img_cb_gen image 2 8192 $P
~/hfenv/bin/python -I - <<PY
s=open("logs/img_te_gen.log").read()
a=s.index("LAB_GEN ")+len("LAB_GEN "); b=s.index("\nLAB_GEN_END")
open("answer_img.txt","w").write(s[a:b]+"<|im_end|>")
PY
A="$(cat answer_img.txt)"
for t in "te_s1 0 8192" "te_s2 0 8192" "cb_s1 2 8192" "cb_s2 2 8192" "te_u4 0 4096" "cb_u4 2 4096" "te1_s 1 8192"; do
  set -- $t
  ./run.sh img_$1 image $2 $3 $P LAB_ANSWER="$A" LAB_LOGITS_OUT=$PWD/logits/img_$1
done
rm -rf dump/img; mkdir -p dump/img
./run.sh img_dump image 1 8192 $P LAB_DUMP_DIR=$PWD/dump/img LAB_DUMP_LAYERS=3,23,43 LAB_DUMP_PHASES=img,txt LAB_ANSWER="$A"
# order / clear-state check for the text scenario: one length per fresh process, mode 0 and mode 1, no dump
for L in 8193 8194 8195; do
  ./run.sh txt_te_one$L text 0 8192 LAB_LENS=$L LAB_LOGITS_OUT=$PWD/logits/txt_te_one
  ./run.sh txt_cb_one$L text 2 8192 LAB_LENS=$L LAB_LOGITS_OUT=$PWD/logits/txt_cb_one
done
./run.sh txt_te1_one8194 text 1 8192 LAB_LENS=8194 LAB_LOGITS_OUT=$PWD/logits/txt_te1_one
echo IMG_DONE
