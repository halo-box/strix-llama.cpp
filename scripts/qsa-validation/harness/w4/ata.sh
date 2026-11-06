#!/bin/bash
# AT-A (PLAN-v3 §16.5): G1 vs G4 at n_q 2-512, kernel level, C18 protocol (3 warm-ups, hipEvent per launch, mean of 12
# per-layer medians, two processes r1 / r2 interleaved). Fixtures: the kept C18 rows (wiki / rand d4k and d131k) with SYN
# data, plus the SYN:1.8 plans at 32768 and 65536 cells (12 layers, BANDS_SYN_LAYERS). Arms per type (f16, bf16, q8_0,
# q4_0): (a) G1M = sa_attend_g1 (+ sa_merge_splits when splits > 1); (b) G4P = sa_pack_v over the full n_kv + sa_attend_g4
# packed + fixup; (c) G4D = sa_attend_g4 direct V + fixup; PACK alone reported. Plus the trimmed layer: the last 512
# queries of layer 11 at ub4096 pos0 d131k. Every run under flock ~/gpu.lock.
cd ~/dev/wt_lab/w4/bands || exit 1; mkdir -p ata
C=~/dev/wt_lab/c18/cap
NQ="2 4 8 16 32 64 96 127 128 256 512"
for r in r1 r2; do
  for fx in wikiS_p4096 randS_p4096 wikiL_p131072 randL_p131072 syn32768 syn65536; do
    for nq in $NQ; do
      reps=10; [ $nq -le 16 ] && reps=30
      case $fx in
        syn*) nk=${fx#syn}; flock ~/gpu.lock env BANDS_SYN_LAYERS=12 ./w4_dec SYN:1.8:$nk:1 SYN $nq atA ${fx}_${nq}_$r time $reps > ata/${fx}_${nq}_$r.out 2>&1 ;;
        *)    flock ~/gpu.lock ./w4_dec $C/$fx.rows SYN $nq atA ${fx}_${nq}_$r time $reps > ata/${fx}_${nq}_$r.out 2>&1 ;;
      esac
      echo "EXIT ${fx}_${nq}_$r $?"
    done
  done
  flock ~/gpu.lock ./w4_dec $C/wikiL_p131072.rows SYN 512 atA trim512_$r time 10 last 11 > ata/trim512_$r.out 2>&1; echo "EXIT trim512_$r $?"
done
echo DONE
