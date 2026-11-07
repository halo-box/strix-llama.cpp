#!/bin/bash
# PLAN-v3 §14.1, U30: recorded masks for test-qsa-allvis-rows: C4's image masks (lab-c4 with C4's flags, prompts p1 and p2 past the shortcut) and a raw M-RoPE image kq_mask inside the shortcut (a short prompt, one image).
. "$(dirname "$0")/env.sh"
OUT=$W7/allvis; mkdir -p $OUT
C4=~/dev/wt_lab/c4
LC=~/dev/wt_lab/build4/bin/lab-c4
rec() { # tag prompt_file ctx
  local tag=$1 p=$2 c=$3 D=$OUT/$1; rm -rf $D; mkdir -p $D
  flock ~/gpu.lock env LAB_QSA_C4=0 LAB_PROMPT_FILE=$p LAB_DUMP_DIR=$D LAB_DUMP_LAYERS=3 timeout 1800 $LC -m $S/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf --mmproj $S/mmproj-Signal-3.8-Flash-Next-f16.gguf --image $C4/img_nyt.jpeg \
     -c $c -b 8192 -ub 8192 -ngl 99 -fa on -ctk f16 -ctv f16 -lm mmap -lzm on-direct --image-min-tokens 1024 -fit off -n 4 -t 8 > $OUT/$tag.log 2>&1
  echo "rec $tag rc=$?; $(ls $D | grep -c mask) mask files; $(ls -la $D | grep mask | head -4 | awk '{print $5, $9}' | tr '\n' ';')"
}
echo "Describe <__media__> in one short sentence." > $OUT/p0.txt
rec s0 $OUT/p0.txt 4096
for p in p1 p2; do [ -d $OUT/$p ] && ls $OUT/$p | grep -q mask || rec $p $C4/$p.txt 40960; done
ARGS=""
for d in s0 p1 p2; do
  m=$(ls $OUT/$d/manifest* 2>/dev/null | head -1)
  while IFS=$'\t' read -r ph tag lay idx ty ne0 ne1 ne2 ne3 fn ex; do
    [ "$tag" = fa_mask ] && ARGS="$ARGS $OUT/$d/$fn $ne0 $ne1" && echo "mask $d $fn $ty ne=$ne0,$ne1,$ne2,$ne3"
  done < "$m"
done
flock ~/gpu.lock timeout 900 $T/build/bin/test-qsa-allvis-rows $ARGS > $OUT/test_rec.log 2>&1; echo "test-qsa-allvis-rows recorded masks rc=$? $(tail -1 $OUT/test_rec.log)"; grep -E "ROCm0 .*bin" $OUT/test_rec.log | cut -c1-170
if [ -x $T/build-vk/bin/test-qsa-allvis-rows ]; then flock ~/gpu.lock timeout 900 $T/build-vk/bin/test-qsa-allvis-rows $ARGS > $OUT/test_rec_vk.log 2>&1; echo "vk rc=$? $(tail -1 $OUT/test_rec_vk.log)"; fi
# keep the logs, delete the dumps
find $OUT -name '*.bin' -delete
echo ALLVIS_REC_DONE
