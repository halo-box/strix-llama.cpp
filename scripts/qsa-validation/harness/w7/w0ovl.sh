#!/bin/bash
# The W0-lab arm with LAB_OVL_PRINT for AT-D1: W0-lab12's commit + patches/apply.sh + the print, in its own worktree (W0-lab12 itself is not touched).
set -e
Z=~/dev/wt_lab/w7/w0ovl
cd ~/dev/wt_rewrite
[ -d $Z ] || git worktree add --detach $Z 11e7e784f
cd $Z
git diff --quiet && scripts/qsa-validation/patches/apply.sh $Z
python3 - <<'PY'
p='ggml/src/ggml-cuda/ggml-cuda.cu'; s=open(p).read()
old='''        ctx->ovl_regions[reg.s_first] = reg;
        prev_b = c.b;
    }
}
'''
new='''        ctx->ovl_regions[reg.s_first] = reg;
        prev_b = c.b;
    }
    static const bool ovl_print = getenv("LAB_OVL_PRINT") != nullptr;
    if (ovl_print) {
        int n_kind[5] = { 0, 0, 0, 0, 0 };
        for (const auto & it : ctx->ovl_regions) { n_kind[it.second.kind]++; }
        fprintf(stderr, "LAB_OVL nodes=%d kind1=%d kind2=%d kind4=%d\\n", cg->n_nodes, n_kind[1], n_kind[2], n_kind[4]);
    }
}
'''
assert s.count(old)==1 or 'LAB_OVL' in s
s=s.replace(old,new)
open(p,'w').write(s)
PY
cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/lib/llvm/bin/clang -DCMAKE_PREFIX_PATH=/opt/rocm -DLLAMA_CURL=OFF > ~/dev/wt_lab/w7/w0ovl_cfg.log 2>&1
flock ~/gpu.lock cmake --build build -j 16 --target llama-bench > ~/dev/wt_lab/w7/w0ovl_build.log 2>&1
echo W0OVL_rc=$?
