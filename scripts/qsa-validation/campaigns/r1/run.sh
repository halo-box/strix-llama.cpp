#!/bin/bash
# run.sh TAG SCEN C4MODE UB [extra env ...]   (text: LAB_LENS; image: gen unless LAB_ANSWER set)
TAG=$1; SCEN=$2; C4=$3; UB=$4; shift 4
W=~/dev/wt_lab/r1; B=~/dev/wt_lab/build4/bin; S=~/models/signal3.8
mkdir -p $W/logits $W/logs
IMG=(); CTX=16384
if [ "$SCEN" = image ]; then IMG=(--mmproj $S/mmproj-Signal-3.8-Flash-Next-f16.gguf --image ~/dev/wt_lab/c4/img_nyt.jpeg --image-min-tokens 1024); CTX=40960; fi
flock ~/gpu.lock env LAB_QSA_C4=$C4 LAB_R1_SCEN=$SCEN LAB_TEXT_FILE=$HOME/data/wikitext-2-raw/wiki.test.raw "$@" $B/lab-r1 \
  -m $S/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf "${IMG[@]}" \
  -c $CTX -b $UB -ub $UB -ngl 99 -fa on -ctk f16 -ctv f16 -lm mmap -lzm on-direct -fit off \
  -n 160 -t 8 > $W/logs/$TAG.log 2>&1
echo "EXIT $? $TAG"; grep -E "LAB_SCORE|LAB_GEN|LAB_CHUNK|LAB_PROMPT|error|abort|ASSERT" $W/logs/$TAG.log | head -30
