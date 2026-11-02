# integration/strix-vulkan

A test branch that carries the Strix Halo Vulkan work that has no upstream PR merged yet, on top of halo-box master.
It is not a release branch and nothing here is approved by the fact that it is on the branch.

Base: halo-box master f5c0a7a34 (already contains #214, #215, #216, #223, #224, #228, #229, #241, #242).

Scope of this branch: only changes to `ggml/src/ggml-vulkan/` and `tests/test-backend-ops.cpp` (shaders included).
Everything that touches `src/`, `common/` or `tools/`, and every PR that is still under review, is on the optional
branch `integration/strix-vulkan-pending`, which sits on top of this one.

Every unit is its own branch `hb/unit-<name>`, merged here with `git merge --no-ff`, so `git log --first-parent` lists the units.
Every commit builds on its own (checked one by one with the shaderc glslc). There is no reconcile commit.

## Units

| unit | branch | PR | state | depends on |
|---|---|---|---|---|
| hx | `hb/unit-hx` | none yet (body drafted) | ready | - |
| sub0 | `hb/unit-sub0` | none yet (body drafted) | ready | - |
| pfq | `hb/unit-pfq` | none yet (body drafted) | ready; decode prefetch queue on by default only on Strix Halo (fix-pfq-default folded in) | - |
| gt1 | `hb/unit-gt1` | none yet (body drafted) | ready | - |
| gt5 | `hb/unit-gt5` | none yet (body drafted) | ready (GDN q/k norm fold and conv fold) | - |
| gdn-conv-window | `hb/unit-gdn-conv-window` | none yet | ready | gt5 |
| hccmb | `hb/unit-hccmb` | none yet (body drafted) | ready | - |
| fn-decode-hc | `hb/unit-fn-decode-hc` | none yet | ready (fixup squashed into the unit commit) | hccmb |
| hc-bf16res | `hb/unit-hc-bf16res` | none yet | ready | hccmb |
| hc-w8a8 | `hb/unit-hc-w8a8` | none yet | ready | hccmb |
| fa-out-gate | `hb/unit-fa-out-gate` | none yet | ready | - |
| fa-sel-lean | `hb/unit-fa-sel-lean` | none yet | ready | - |
| get-rows-narrow | `hb/unit-get-rows-narrow` | none yet | ready | - |
| argmax-2pass | `hb/unit-argmax-2pass` | none yet | ready | - |
| gemm-ragged-tail | `hb/unit-gemm-ragged-tail` | none yet | ready (fix-ragged-hc folded in) | hc-w8a8 |
| qsa-vis | `hb/unit-qsa-vis` | none yet | ready | - |
| dsv4-hc-boundary | `hb/unit-dsv4-hc-boundary` | none yet | ready (the mix buffer is marked as written) | hccmb |
| test-qwen4exp-cases | `hb/unit-test-qwen4exp-cases` | none yet | tests only | - |
| test-engage-script | `hb/unit-test-engage-script` | none yet | script and a gfx1151 CI step (never dispatched) | - |

## How to test

- Build: `cmake -B build -DGGML_VULKAN=ON -DLLAMA_BUILD_TESTS=ON -DVulkan_GLSLC_EXECUTABLE=<shaderc glslc>` and `cmake --build build --target test-backend-ops`.
- Full suite: `build/bin/test-backend-ops -b Vulkan0`.
- Per unit: the op filters named in each unit's commit message, for example `-o 'HC_LAYER|HC_MOE_EPI|HC_POST_NORM'` (hccmb), `-o 'GDN_CONV_PREFILL|GATED_DELTA_NET'` (gt5), `-o TOP_K` (gt1), `-o MUL_MAT`.
- Fused-path engagement: `GGML_VK_TEST_REQUIRE_<NAME>=1` makes the run fail if the fused path does not run, and `GGML_VK_*=0` switches a path off for the comparison run.

## Not on this branch

- On `integration/strix-vulkan-pending` (on top of this branch): every unit that touches `src/`, `common/` or `tools/`, the PRs still under review (#208, #218, #225, #226, #227, #230, #231), expert-repack with its dependants and test-hooks.
- Held, in neither branch: #219 (changes requested), the quantized-K/V maskless port that depends on it, EXL3 (#201/#204), #217 (closed).
- Dropped: the unreviewed "draft, untested" MOE_GLU_GEMV commit message (the unit is rebuilt with one real commit on the pending branch), the `fixup!` commit (squashed), the reconcile commit that imported an unpublished tree (every change is a unit now).
- test-census-qsa-strip (master census expects the QSA_EXPAND fusion in the strip case) is a separate PR on master; qsa-vis carries the same expectation here.

Known limits: the dq8 GEMM test-break switch of master is still compiled in on this branch (test-hooks, on the pending branch, gates it).
