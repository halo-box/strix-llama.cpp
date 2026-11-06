#!/bin/bash
# W4 (PLAN-v3 §14.2 (b)): the decode n_q 1 SYN 262k fixture enlarged to 12 layers x 3 seeds (BANDS_SYN_LAYERS=12), the
# PR-verbatim decode (PR-decM-f16 / PR-decM-q8_0: partial + qsa_decode_merge) and the new G1 kernel (G1M-<T>) against
# O_CON, planted defects on; every run under flock ~/gpu.lock
cd ~/dev/wt_lab/w4/bands || exit 1
for seed in 1 2 3; do
  flock ~/gpu.lock env BANDS_DEFECTS=1 BANDS_SYN_LAYERS=12 ./w4_dec SYN:1.8:262144:$seed SYN 1 band syn1_s$seed band > syn/syn1_s$seed.out 2> syn/syn1_s$seed.err
  echo "EXIT syn1_s$seed $?"
done
echo DONE
