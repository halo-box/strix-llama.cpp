#!/bin/bash
# AT-W12-BANDS + W12 done-when campaign on W0-lab (PLAN-v3 §13 W12, §16.5 AT-W12-BANDS).
#   campaign.sh PHASE...   phases: S (QKL_HASH bit-identity of every self-replay) A (ref8t0: self, control, H1, direct form incl. H3) C (ref16t0: H2 + agreement control)
#                                  D (ref32x8: H4) V (W0-Vulkan: V1, V2) B (ref8L: Q11a lo/hi split, U40) E (ref2k: Q11b shift)
# Each reference (base.kld + pins) is deleted after its phase; logs and .stats stay in $LOGS.
set -u
D=$(cd "$(dirname "$0")" && pwd)
R=${W12_REFS:-~/dev/wt_lab/w12/refs}
RUN=$D/run.sh
T0=LLAMA_FN_LAST_ATTN_TRIM=0
pairs="bf16:bf16 q8_0:f16 f16:q8_0"
tagof() { echo "$1" | sed 's/:/_/'; }   # bf16:bf16 -> bf16_bf16
phase_A() {
  local F=$R/ref8t0
  $RUN ref8t0_rec $F record f16 f16 4096 8192 8 hip $T0
  $RUN a_f16_f16_u4096 $F replay f16 f16 4096 8192 8 hip            # QPIN self-replay
  $RUN a_q8_0_q8_0_u4096 $F replay q8_0 q8_0 4096 8192 8 hip        # control (R1-UB table D: 1.371668e-4, SE 1.25e-5)
  $RUN a_f16_f16_u2048 $F replay f16 f16 2048 8192 8 hip
  for p in $pairs; do k=${p%:*}; v=${p#*:}; t=$(tagof $p)            # H1: 12 arms
    for u in 4096 2048; do
      $RUN a_${t}_u${u} $F replay $k $v $u 8192 8 hip
      $RUN a_${t}_u${u}_sk1 $F replay $k $v $u 8192 8 hip LLAMA_QSA_SELECTED_KEY=1
    done
  done
  # direct form: clean same-type pinned base, its ub2048 reorder and its eps = 1e-7 saturation (H3 for q4_0, extended
  # to every Q1-A type so no direct-form band is borrowed). Every arm of this block runs with LAB_DEF=d3 (eps 0 for the
  # base, its self check and the ub2048 arm), because observing attn_output-23 splits the graph: base and arms must
  # share the same splits (R1-Q8 Setup).
  local E0="LAB_DEF=d3 LAB_D3_EPS=0"
  for p in f16:f16 q4_0:q4_0 q8_0:q8_0 $pairs; do k=${p%:*}; v=${p#*:}; t=$(tagof $p)
    export W12_KBASE=$R/kbase_${t}.kld
    $RUN db_${t}_base $F pinbase $k $v 4096 8192 8 hip $E0
    $RUN db_${t}_self $F pinkld $k $v 4096 8192 8 hip $E0
    $RUN db_${t}_u2048 $F pinkld $k $v 2048 8192 8 hip $E0
    $RUN db_${t}_d3e7 $F pinkld $k $v 4096 8192 8 hip LAB_DEF=d3 LAB_D3_EPS=1e-7
    rm -f $W12_KBASE; unset W12_KBASE
  done
  rm -rf $F
}
phase_C() {
  local F=$R/ref16t0
  $RUN ref16t0_rec $F record f16 f16 4096 8192 16 hip $T0
  $RUN c_f16_f16_u4096 $F replay f16 f16 4096 8192 16 hip            # QPIN self-replay
  $RUN c_q8_0_q8_0_u4096 $F replay q8_0 q8_0 4096 8192 16 hip        # agreement control (R1-E: 0.996743)
  for p in $pairs; do k=${p%:*}; v=${p#*:}; t=$(tagof $p)            # H2: kind 1 ub4096/ub2048, default (tk) ub4096
    $RUN c_${t}_u4096_sk1 $F replay $k $v 4096 8192 16 hip LLAMA_QSA_SELECTED_KEY=1
    $RUN c_${t}_u2048_sk1 $F replay $k $v 2048 8192 16 hip LLAMA_QSA_SELECTED_KEY=1
    $RUN c_${t}_u4096 $F replay $k $v 4096 8192 16 hip
  done
  rm -rf $F
}
phase_D() {
  local F=$R/ref32x8
  $RUN ref32x8_rec $F record f16 f16 4096 32768 8 hip $T0 QKL_TAIL=4096
  for t in f16 q8_0 q4_0; do
    for u in 4096 2048; do $RUN d_${t}_${t}_u${u} $F replay $t $t $u 32768 8 hip QKL_TAIL=4096; done
  done
  rm -rf $F
}
phase_V() {
  local F=$R/vref8t0
  $RUN vref8t0_rec $F record f16 f16 4096 8192 8 vk $T0                       # V1
  $RUN v_f16_f16_u4096 $F replay f16 f16 4096 8192 8 vk                       # self-replay
  $RUN v_old_f16_u4096_r1 $F replay f16 f16 4096 8192 8 vkold                 # V2 (i)
  $RUN v_old_f16_u4096_r2 $F replay f16 f16 4096 8192 8 vkold                 # V2 (ii)
  $RUN v_f16_f16_u2048 $F replay f16 f16 2048 8192 8 vk                       # V2 (iii)
  $RUN v_f16_f16_u4096_nodq8 $F replay f16 f16 4096 8192 8 vk GGML_VK_DENSE_DQ8=0 GGML_VK_HC_DQ8=0   # V2 (iv)
  rm -rf $F
}
phase_B() {
  local F=$R/ref8L
  $RUN ref8L_rec $F record f16 f16 4096 8192 8 hip $T0 QKL_TAIL=8191
  for p in f16:f16 q8_0:q8_0 q4_0:q4_0 $pairs; do k=${p%:*}; v=${p#*:}; t=$(tagof $p)
    $RUN b_${t}_u4096 $F replay $k $v 4096 8192 8 hip QKL_TAIL=8191 QKL_SPLIT=2048
    W12_B=512 $RUN b_${t}_u512 $F replay $k $v 512 8192 8 hip QKL_TAIL=8191 QKL_SPLIT=2048
  done
  rm -rf $F
}
phase_E() {
  local F=$R/ref2k
  W12_B=2048 $RUN ref2k_rec $F record f16 f16 2048 2048 16 hip $T0
  for p in f16:f16 q8_0:q8_0 q4_0:q4_0 $pairs; do k=${p%:*}; v=${p#*:}; t=$(tagof $p)
    W12_B=2048 $RUN e_${t}_u2048 $F replay $k $v 2048 2048 16 hip
    W12_B=2048 $RUN e_${t}_u512 $F replay $k $v 512 2048 16 hip
  done
  for t in f16 q8_0; do W12_B=2048 $RUN e_${t}_${t}_u16 $F replay $t $t 16 2048 16 hip; done
  rm -rf $F
}
phase_S() {
  # bit-identity of every self-replay (QKL_HASH=1): each reference is recorded again and replayed by itself; the
  # per-chunk logits hashes of the record and of the replay must match, and the replay's QKLD_CHUNK values must equal
  # those of the phase that used the reference (so the re-recorded reference is the same one)
  local H=QKL_HASH=1
  $RUN s_ref8t0_rec $R/s8 record f16 f16 4096 8192 8 hip $T0 $H;                  $RUN s_ref8t0_self $R/s8 replay f16 f16 4096 8192 8 hip $H; rm -rf $R/s8
  $RUN s_ref16t0_rec $R/s16 record f16 f16 4096 8192 16 hip $T0 $H;               $RUN s_ref16t0_self $R/s16 replay f16 f16 4096 8192 16 hip $H; rm -rf $R/s16
  $RUN s_ref32x8_rec $R/s32 record f16 f16 4096 32768 8 hip $T0 QKL_TAIL=4096 $H; $RUN s_ref32x8_self $R/s32 replay f16 f16 4096 32768 8 hip QKL_TAIL=4096 $H; rm -rf $R/s32
  $RUN s_vref8t0_rec $R/sv record f16 f16 4096 8192 8 vk $T0 $H;                  $RUN s_vref8t0_self $R/sv replay f16 f16 4096 8192 8 vk $H; rm -rf $R/sv
  $RUN s_ref8L_rec $R/sL record f16 f16 4096 8192 8 hip $T0 QKL_TAIL=8191 $H;     $RUN s_ref8L_self $R/sL replay f16 f16 4096 8192 8 hip QKL_TAIL=8191 QKL_SPLIT=2048 $H; rm -rf $R/sL
  W12_B=2048 $RUN s_ref2k_rec $R/s2 record f16 f16 2048 2048 16 hip $T0 $H;       W12_B=2048 $RUN s_ref2k_self $R/s2 replay f16 f16 2048 2048 16 hip $H; rm -rf $R/s2
}
for ph in "$@"; do echo "PHASE $ph start $(date +%T)"; phase_$ph; echo "PHASE $ph done $(date +%T)"; done
echo CAMPAIGN_DONE $(date +%T)
