#!/bin/bash
# W6: validation layer runs (plan cases; K/V attend smoke), the standalone Vulkan test on RADV and lavapipe, the V-variant census on lavapipe limits, vulkaninfo records
cd ~/dev/wt_rewrite
L=~/dev/wt_rewrite_logs/W6
T=build-vk/bin/test-backend-ops
TOOLS=~/dev/wt_lab/w6/tools/usr
LVP="VK_ICD_FILENAMES=$HOME/dev/wt_lab/w6/lvp43/icd.json GGML_VK_VISIBLE_DEVICES=0"
LVPLD=$HOME/dev/wt_lab/w6/lvp43/usr/lib64
BL=$PWD/build-vk/bin
# AT-B7 (0): limits of the devices
env VK_LAYER_PATH= LD_LIBRARY_PATH=$TOOLS/lib64 $TOOLS/bin/vulkaninfo 2>/dev/null > $L/vulkaninfo_radv.txt; echo "vulkaninfo radv $?" >> $L/runs2.out
env $LVP LD_LIBRARY_PATH=$LVPLD:$TOOLS/lib64 $TOOLS/bin/vulkaninfo 2>/dev/null > $L/vulkaninfo_lvp.txt; echo "vulkaninfo lvp $?" >> $L/runs2.out
env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json GGML_VK_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$BL flock ~/gpu.lock timeout 60 $T test -o SPARSE_ATTN_PLAN -b Vulkan0 -j 1 > $L/plan_lvp_system_mesa.log 2>&1; echo "plan lavapipe system mesa rc=$?" >> $L/runs2.out
# the validation layer, with GPU-AV (shader buffer accesses): every plan case, then the unsorted rows are in rows=perm,dup,neg,oob,mix
VAL="VK_LAYER_PATH=$TOOLS/share/vulkan/explicit_layer.d VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT"
env $VAL LD_LIBRARY_PATH=$TOOLS/lib64:$BL flock ~/gpu.lock timeout 1500 $T test -o SPARSE_ATTN_PLAN -b Vulkan0 --fail-unsupported -j 1 > $L/val_plan_radv.log 2>&1; echo "val plan radv rc=$?" >> $L/runs2.out
env $VAL $LVP LD_LIBRARY_PATH=$LVPLD:$TOOLS/lib64:$BL flock ~/gpu.lock timeout 1500 $T test -o SPARSE_ATTN_PLAN -b Vulkan0 --fail-unsupported -j 1 > $L/val_plan_lvp.log 2>&1; echo "val plan lvp rc=$?" >> $L/runs2.out
# attend under the layer, one shape per tier (the workspace binding)
for F in "" scalar portable vstaged vpack; do
  env $VAL GGML_VK_SA_FORCE=$F LD_LIBRARY_PATH=$TOOLS/lib64:$BL flock ~/gpu.lock timeout 900 $T test -o SPARSE_ATTN -b Vulkan0 -j 1 -p 'type_k=(f16|q8_0),type_v=(f16|q8_0),n_q=(9|65),n_kv=4096,n_sel=257.*rot_q=0,rot_o=0,rows=mix' > $L/val_attn_radv_$F.log 2>&1; echo "val attn radv [$F] rc=$?" >> $L/runs2.out
done
env $VAL $LVP LD_LIBRARY_PATH=$LVPLD:$TOOLS/lib64:$BL flock ~/gpu.lock timeout 900 $T test -o SPARSE_ATTN -b Vulkan0 -j 1 -p 'type_k=(f16|q8_0),type_v=(f16|q8_0),n_q=(9|65),n_kv=4096,n_sel=257.*rot_q=0,rot_o=0,rows=mix' > $L/val_attn_lvp.log 2>&1; echo "val attn lvp rc=$?" >> $L/runs2.out
# the standalone test
flock ~/gpu.lock timeout 600 build-vk/bin/test-sparse-attn-vk > $L/vkt_radv.log 2>&1; echo "vkt radv rc=$?" >> $L/runs2.out
env $LVP LD_LIBRARY_PATH=$LVPLD:$BL flock ~/gpu.lock timeout 900 build-vk/bin/test-sparse-attn-vk > $L/vkt_lvp.log 2>&1; echo "vkt lvp rc=$?" >> $L/runs2.out
# the q_base loop of the host (GGML_VK_SA_MAX_Q): every tier, plan and attend
for F in "" scalar portable; do
  env GGML_VK_SA_FORCE=$F GGML_VK_SA_MAX_Q=7 flock ~/gpu.lock timeout 900 $T test -o SPARSE_ATTN -b Vulkan0 -j 1 -p 'type_k=(f16|q8_0),type_v=(f16|q8_0),n_q=(9|65),n_kv=4096,n_sel=257.*rot_q=0,rot_o=0,rows=mix' > $L/maxq_attn_$F.log 2>&1; echo "maxq attn [$F] rc=$?" >> $L/runs2.out
done
env GGML_VK_SA_MAX_Q=7 flock ~/gpu.lock timeout 300 $T test -o SPARSE_ATTN_PLAN -b Vulkan0 --fail-unsupported -j 1 > $L/maxq_plan.log 2>&1; echo "maxq plan rc=$?" >> $L/runs2.out
# no pipeline: every case unsupported, nothing dispatched (rc reported by the driver)
env GGML_VK_SA_CAP_MASK=15 flock ~/gpu.lock timeout 300 $T test -o SPARSE_ATTN_PLAN -b Vulkan0 -j 1 > $L/mask15_plan.log 2>&1; echo "mask15 plan rc=$?" >> $L/runs2.out
env GGML_VK_SA_CAP_MASK=7 flock ~/gpu.lock timeout 300 $T test -o SPARSE_ATTN -b Vulkan0 -j 1 > $L/mask7_attn.log 2>&1; echo "mask7 attn rc=$?" >> $L/runs2.out
echo RUNS2_DONE >> $L/runs2.out
