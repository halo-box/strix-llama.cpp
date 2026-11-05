#!/bin/bash
# R1-M-Lab A campaign: GPU bands (bands_attn / bands_dec), then CPU bands (bands_cpu). Every GPU run under flock ~/gpu.lock.
cd ~/dev/wt_lab/bands
R=../c18/cap2/wikiS_p4096.rows; Q=../c18/cap2/wikiS_p4096.qkv
g() { local t=$1; shift; echo "START $t $(date +%T)"; flock ~/gpu.lock "$@" > res/$t.out 2> logs/$t.err; echo "EXIT $t $? $(date +%T)"; }
c() { local t=$1; shift; echo "START $t $(date +%T)"; "$@" > res/$t.out 2> logs/$t.err; echo "EXIT $t $? $(date +%T)"; }
g attn_wiki4k_r1   ./bands_attn $R $Q 0,5,11 8 ship wiki4k 32
g attn_syn262k_s1  ./bands_attn SYN:1.8:262144:1 SYN all 1 ship+direct syn262k_s1 32
g dec512_wiki4k    ./bands_dec $R $Q 512 ship dec512_wiki4k 32
g dec1_wiki4k      ./bands_dec $R $Q 1 ship dec1_wiki4k 32
g dec512_syn262k   ./bands_dec SYN:1.8:262144:1 SYN 512 ship dec512_syn262k 32
g dec1_syn262k     ./bands_dec SYN:1.8:262144:1 SYN 1 ship dec1_syn262k 32
g attn_wiki4k_r2   ./bands_attn $R $Q 0,5,11 8 ship wiki4k_r2 32
g attn_syn262k_s2  ./bands_attn SYN:1.8:262144:2 SYN all 1 ship+direct syn262k_s2 32
c cpu_wiki4k       ./bands_cpu $R $Q 0,5,11 1 cpu_wiki4k 32 1 all
c cpu_wiki4k_s8    ./bands_cpu $R $Q all 8 cpu_wiki4k_s8 32 1 gpu
c cpu_syn262k      ./bands_cpu SYN:1.8:262144:1 SYN all 1 cpu_syn262k 32 1 all
echo DONE $(date +%T)
