#!/bin/bash
# A lab tree of a W13+ head with the measurement-only env hooks of patches/06-w13-measurement-hooks.patch, for the Prove
# runs that set them (AT-W7PERF / AT-B4: LLAMA_QSA_PREFER_DENSE; AT-W7PERF (c)/(d), AT-D4: LLAMA_QSA_SHORTCUT_PIN,
# LLAMA_QSA_HOST_KV_COPY; AT-C12 / U38: LLAMA_QSA_FORCE_NONCOMPACT; AT-D1: LAB_OVL_PRINT). Without the variables the tree
# computes what the head computes. hooks_tree.sh [commit] [hip|vk|both]; binaries in $LAB/build/bin and $LAB/build-vk/bin.
set -e
C=${1:-HEAD}
WHAT=${2:-both}
LAB=~/dev/wt_lab/w13/hooks
P=~/dev/wt_rewrite/scripts/qsa-validation/patches
mkdir -p ~/dev/wt_lab/w13
cd ~/dev/wt_rewrite
if [ -d $LAB ]; then
  git -C $LAB reset -q --hard
  git -C $LAB checkout -q --detach $(git rev-parse $C)
else
  git worktree add --detach $LAB $C
fi
cd $LAB
if ! grep -q 'getenv("LLAMA_QSA_PREFER_DENSE")' src/models/qwen4exp.cpp; then
  git apply $P/06-w13-measurement-hooks.patch
fi
git status --short
T="--target llama-perplexity llama-bench llama-server llama-cli"
if [ $WHAT != vk ]; then
  cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/lib/llvm/bin/clang -DCMAKE_PREFIX_PATH=/opt/rocm -DLLAMA_CURL=OFF > ~/dev/wt_lab/w13/hooks_cfg.log 2>&1
  flock ~/gpu.lock cmake --build build -j 16 $T > ~/dev/wt_lab/w13/hooks_build.log 2>&1
  echo HOOKS_HIP_rc=$?
fi
if [ $WHAT != hip ]; then
  cmake -B build-vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > ~/dev/wt_lab/w13/hooks_vk_cfg.log 2>&1
  flock ~/gpu.lock cmake --build build-vk -j 16 $T > ~/dev/wt_lab/w13/hooks_vk_build.log 2>&1
  echo HOOKS_VK_rc=$?
fi
echo $(git -C $LAB rev-parse --short HEAD) + 06
