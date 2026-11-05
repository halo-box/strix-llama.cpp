#!/bin/bash
# (b) layout 2: images in two slots (slot A: B65t deep image, generating; slot B: image + 12k text)
cd ~/dev/wt_lab/r1c
for r in r1 r2; do
echo "##### b3_share_$r $(date +%T)";   flock ~/gpu.lock ./srv3.sh b3_share_$r   "LAB_C19_LOG=1" TI12k_cp.json
echo "##### b3_noshare_$r $(date +%T)"; flock ~/gpu.lock ./srv3.sh b3_noshare_$r "LAB_C19_LOG=1 LAB_C19_NO_SHARE=1" TI12k_cp.json
done
echo RUNB3_DONE $(date +%T)
