#!/bin/bash
# W6: on the tree at HEAD: the timing campaign, every test run, the mutants
cd ~/dev/wt_rewrite; L=~/dev/wt_rewrite_logs/W6
flock ~/gpu.lock bash -c 'cmake --build build-vk -j 16 --target test-backend-ops test-sparse-attn-vk > ~/dev/wt_rewrite_logs/W6/vk_build_t.log 2>&1'
rm -rf ~/dev/wt_lab/w6/time/res/*
scripts/qsa-validation/harness/w6/time_campaign.sh > $L/time_campaign.out 2>&1
python3 scripts/qsa-validation/harness/w6/w6_time_an.py ~/dev/wt_lab/w6/time/res > $L/time_an.txt
scripts/qsa-validation/harness/w6/final_w6.sh
sha256sum build-vk/bin/libggml-vulkan.so.0.26.0 build-vk/bin/test-backend-ops build-vk/bin/test-sparse-attn-vk > $L/sha_final.txt
python3 scripts/qsa-validation/harness/w6/mutants_w6.py > $L/mutants.out 2>&1
sha256sum build-vk/bin/libggml-vulkan.so.0.26.0 build-vk/bin/test-backend-ops build-vk/bin/test-sparse-attn-vk > $L/sha_after_mutants.txt
echo ALL_DONE >> $L/mutants.out
