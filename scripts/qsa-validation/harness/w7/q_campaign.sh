#!/bin/bash
# W7 quality gates (PLAN-v3 §13 W7, §15): pinned arms, new (W7 lab tree) vs w0 (W0-lab12, PR219), one reference at a time, each deleted after
# its phase. Phases: A (ref8t0: self, Q1-A/B/C, Q2, Q8), N (ref8t0 again: Q12), D (direct form), B (ref8L: Q11a), E (ref2k: Q11b), C (ref16t0: Q4),
# P (ref32x8: Q3-A/B), S (Q11c split streams, unpinned).
set -u
D=$(cd "$(dirname "$0")" && pwd)
R=~/dev/wt_lab/w7/q/refs
RUN=$D/qrun.sh
T0=LLAMA_FN_LAST_ATTN_TRIM=0
ALL="f16:f16 q8_0:q8_0 q4_0:q4_0 bf16:bf16 q8_0:f16 f16:q8_0"
THREE="f16:f16 q8_0:q8_0 q4_0:q4_0"
tagof() { echo "$1" | sed 's/:/_/'; }
both() { # tag ref mode ctk ctv ub ctx chunks [env]: the PR219 arm then the new arm
  local tag=$1 ref=$2 mode=$3 k=$4 v=$5 ub=$6 c=$7 ch=$8; shift 8
  $RUN w0_$tag $ref $mode w0 $k $v $ub $c $ch "$@"
  $RUN new_$tag $ref $mode new $k $v $ub $c $ch "$@"
}
need_ref8() { [ -f $R/ref8t0/base.kld ] || $RUN ref8t0_rec $R/ref8t0 record w0 f16 f16 4096 8192 8 $T0; }
phase_A() {
  need_ref8
  $RUN w0_a_self $R/ref8t0 replay w0 f16 f16 4096 8192 8
  for p in $ALL; do k=${p%:*}; v=${p#*:}; both a_$(tagof $p)_u4096 $R/ref8t0 replay $k $v 4096 8192 8; done     # Q1-A
  $RUN w0_a_f16_f16_u2048 $R/ref8t0 replay w0 f16 f16 2048 8192 8                                               # Q2 floor
  for p in $THREE; do k=${p%:*}; v=${p#*:}; both a_$(tagof $p)_u512 $R/ref8t0 replay $k $v 512 8192 8; done     # Q1-B
  for p in $THREE; do k=${p%:*}; v=${p#*:}; both a_$(tagof $p)_u64 $R/ref8t0 replay $k $v 64 8192 8               # Q1-C
    # AT-A routes n_q 64 at n_kv <= 8192 to G4: the forced arm keeps G1 gated e2e (PLAN-v3 §14.3 "Forced arms after AT-A")
    $RUN new_a_$(tagof $p)_u64g1 $R/ref8t0 replay new $k $v 64 8192 8 GGML_CUDA_SA_FORCE=g1
  done
  for t in f16 q8_0; do                                                                                         # Q8: on - off, same type
    for d in d1 d2 d3; do both a8_${t}_$d $R/ref8t0 replay $t $t 4096 8192 8 LAB_DEF=$d; done
    both a8_${t}_off $R/ref8t0 replay $t $t 4096 8192 8 LAB_DEF=d3 LAB_D3_EPS=0
  done
}
phase_N() {
  need_ref8
  export QARGS="-nkvo"
  for t in f16 q8_0; do for u in 4096 512; do
    both n_${t}_u$u $R/ref8t0 replay $t $t $u 8192 8
    both n_${t}_u${u}_r2 $R/ref8t0 replay $t $t $u 8192 8
  done; done
  unset QARGS
}
phase_Z() {  # -ngl 0 (AT-D2 chunks, CPU only: the pinned rule on the first NGL0_CHUNKS chunks of the same reference)
  need_ref8
  export QNGL="-ngl 0" QTIMEOUT=28800
  for t in f16 q8_0; do both z_${t}_u4096 $R/ref8t0 replay $t $t 4096 8192 ${NGL0_CHUNKS:-2}; done
  unset QNGL QTIMEOUT
}
phase_D() {
  need_ref8
  for p in $ALL; do k=${p%:*}; v=${p#*:}; t=$(tagof $p)
    export QKBASE=$R/kbase_$t.kld
    $RUN w0_db_${t}_base $R/ref8t0 pinbase w0 $k $v 4096 8192 8 LAB_DEF=d3 LAB_D3_EPS=0
    $RUN new_db_${t} $R/ref8t0 pinkld new $k $v 4096 8192 8 LAB_DEF=d3 LAB_D3_EPS=0
    rm -f $QKBASE; unset QKBASE
  done
}
phase_H() {  # §15 two-stage rule: a Q1 direct-form type that failed stage 1, re-scored on chunks 8-15 of the c8192 x 16 reference (HTYPES)
  local F=$R/ref16h
  $RUN ref16h_rec $F record w0 f16 f16 4096 8192 16 $T0
  for p in ${HTYPES:?}; do k=${p%:*}; v=${p#*:}; t=$(tagof $p)
    export QKBASE=$R/kbaseh_$t.kld
    $RUN w0_hb_${t}_base $F pinbase w0 $k $v 4096 8192 16 LAB_DEF=d3 LAB_D3_EPS=0
    $RUN new_hb_${t} $F pinkld new $k $v 4096 8192 16 LAB_DEF=d3 LAB_D3_EPS=0
    # explanatory, not a gate: the PR219 arm itself with eps = 1e-7 at layer 23 (the band's saturation arm) and at layer 3, the first QSA layer
    $RUN w0_hs23_${t} $F pinkld w0 $k $v 4096 8192 16 LAB_DEF=d3 LAB_D3_EPS=1e-7
    $RUN w0_hs3_${t} $F pinkld w0 $k $v 4096 8192 16 LAB_DEF=d3 LAB_D3_EPS=1e-7 LAB_D3_L=3
    rm -f $QKBASE; unset QKBASE
  done
  rm -rf $F
}
phase_K() {  # explanatory, not a gate: PR219's own route changes in the direct form, on ref8t0 (the first 8 chunks of phase D's arms)
  need_ref8
  for spec in q8_0:q8_0:0 q4_0:q4_0:1 bf16:bf16:1 f16:f16:0; do k=${spec%%:*}; r=${spec#*:}; v=${r%:*}; o=${r#*:}; t=${k}_$v
    export QKBASE=$R/kbasek_$t.kld
    $RUN w0_kb_${t}_base $R/ref8t0 pinbase w0 $k $v 4096 8192 8 LAB_DEF=d3 LAB_D3_EPS=0
    $RUN w0_kk_${t}_route $R/ref8t0 pinkld w0 $k $v 4096 8192 8 LAB_DEF=d3 LAB_D3_EPS=0 LLAMA_QSA_SELECTED_KEY=$o
    $RUN w0_kk_${t}_eps3 $R/ref8t0 pinkld w0 $k $v 4096 8192 8 LAB_DEF=d3 LAB_D3_EPS=1e-7 LAB_D3_L=3
    rm -f $QKBASE; unset QKBASE
  done
}
phase_B() {
  local F=$R/ref8L
  $RUN ref8L_rec $F record w0 f16 f16 4096 8192 8 $T0 QKL_TAIL=8191
  export QB=512
  for p in $ALL; do k=${p%:*}; v=${p#*:}; both b_$(tagof $p)_u512 $F replay $k $v 512 8192 8 QKL_TAIL=8191 QKL_SPLIT=2048; done
  unset QB
  rm -rf $F
}
phase_E() {
  local F=$R/ref2k
  export QB=2048
  $RUN ref2k_rec $F record w0 f16 f16 2048 2048 16 $T0
  for p in $ALL; do k=${p%:*}; v=${p#*:}; both e_$(tagof $p)_u512 $F replay $k $v 512 2048 16; done
  for p in f16:f16 q8_0:q8_0; do k=${p%:*}; v=${p#*:}; both e_$(tagof $p)_u16 $F replay $k $v 16 2048 16; done
  unset QB
  rm -rf $F
}
phase_C() {
  local F=$R/ref16t0
  $RUN ref16t0_rec $F record w0 f16 f16 4096 8192 16 $T0
  for p in f16:f16 q8_0:q8_0; do k=${p%:*}; v=${p#*:}; both c_$(tagof $p) $F replay $k $v 4096 8192 16; done
  # the W0 arm takes the indexer type from QPIN_IDX_TYPE (lab patch 01); a W11+ new arm from LLAMA_ARG_CACHE_TYPE_K_IDX (its lab tree
  # drops the QPIN_IDX_TYPE hunk), and its q4_0 arm runs the shipped auto mapping (q4_0 -> q8_0 indexer) with no override
  both c_f16k_q8idx $F replay f16 f16 4096 8192 16 QPIN_IDX_TYPE=q8_0 LLAMA_ARG_CACHE_TYPE_K_IDX=q8_0
  $RUN w0_c_q4_0_q4_0 $F replay w0 q4_0 q4_0 4096 8192 16 QPIN_IDX_TYPE=q8_0 LLAMA_QSA_SELECTED_KEY=1
  $RUN new_c_q4_0_q4_0 $F replay new q4_0 q4_0 4096 8192 16 QPIN_IDX_TYPE=q8_0
  for p in bf16:bf16 q8_0:f16 f16:q8_0; do k=${p%:*}; v=${p#*:}; t=$(tagof $p)
    $RUN w0_c_$t $F replay w0 $k $v 4096 8192 16 LLAMA_QSA_SELECTED_KEY=1
    $RUN new_c_$t $F replay new $k $v 4096 8192 16
  done
  rm -rf $F
}
phase_P() {
  local F=$R/ref32x8
  $RUN ref32x8_rec $F record w0 f16 f16 4096 32768 8 $T0 QKL_TAIL=4096
  for t in f16 q8_0 q4_0; do
    both p_${t}_u4096 $F replay $t $t 4096 32768 8 QKL_TAIL=4096
    both p_${t}_u256 $F replay $t $t 256 32768 8 QKL_TAIL=4096
  done
  rm -rf $F
}
phase_S() {  # Q11c: -c 1024 -b 2048 (np2, split streams) equals -b 1024 (np1), unpinned, per sequence; new binary only (R1-AV campaign C)
  for t in q8_0 q4_0; do
    local F=$R/refC_$t
    QB=2048 $RUN s_${t}_np2base $F base new $t $t 2048 1024 32
    QB=2048 $RUN s_${t}_np2rerun $F kld new $t $t 2048 1024 32
    QB=1024 $RUN s_${t}_np1 $F kld new $t $t 1024 1024 32
    rm -rf $F
  done
}
for ph in "$@"; do echo "PHASE $ph start $(date +%T)"; phase_$ph; echo "PHASE $ph done $(date +%T)"; done
echo Q_CAMPAIGN_DONE $(date +%T)
