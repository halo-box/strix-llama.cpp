#!/bin/bash
# C5 attend-prototype timing campaign on replayed rows; one flock per process (one rows file)
cd ~/dev/wt_lab/c5
VL=0,1,2,3,4,5,6,7,13,14
run() { # out rows type data
  flock ~/gpu.lock ./c5_attn cap/$2 $3 20 $4 all $VL > res/attn_$1.out 2>&1; echo "EXIT $1 $?"
}
run wiki_f16_p32768_real_r1 wiki_f16_p32768.rows f16 cap/wiki_f16_p32768.qkv
run wiki_f16_p32768_real_r2 wiki_f16_p32768.rows f16 cap/wiki_f16_p32768.qkv
run wiki_f16_p32768_syn wiki_f16_p32768.rows f16 SYN
for c in code chat randc; do run ${c}_f16_p32768_syn ${c}_f16_p32768.rows f16 SYN; done
run bench_f16_p32768_syn bench32768_f16_p32768.rows f16 SYN
for p in 8192 65536 131072; do run wiki_f16_p${p}_syn wiki_f16_p$p.rows f16 SYN; done
for p in 8192 131072; do run bench_f16_p${p}_syn bench${p}_f16_p$p.rows f16 SYN; run randc_f16_p${p}_syn randc_f16_p$p.rows f16 SYN; done
run wiki_q8_0_p32768_real wiki_q8_0_p32768.rows q8_0 cap/wiki_q8_0_p32768.qkv
run wiki_q8_0_p131072_syn wiki_q8_0_p131072.rows q8_0 SYN
