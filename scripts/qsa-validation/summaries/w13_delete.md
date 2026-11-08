# W13: delete the old code, CI baseline, merge-tree (PLAN-v3 §11, §13 W13)

Branch qsa-rewrite, base ea5285ddf, W13 commits 08b431a31..11b42ee8e (strix-002). Logs: `~/dev/wt_rewrite_logs/W13/`,
`~/dev/wt_lab/w13/` (guard, smoke, vk, vgpr).

## Deleted (located by symbol)
- HIP: qsa-prefill.cu/.cuh, qsa-decode.cu/.cuh, qsa-decode-wmma.cuh; fattn.cu QSA dispatch, maskless abort and
  supports branch, `ggml_cuda_qsa_gate_ok`; ggml-cuda.cu qsa-prefill gated branch; common.cuh `GGML_QSA_MAX_KEYS`.
- Vulkan: flash_attn_sel.comp, flash_attn_sel_cm.comp, flash_attn_sel_prep.comp, their pipelines, push constants,
  `ggml_vk_flash_attn_sel{,_pipeline}`, the maskless FA dispatch and supports_op branches, `GGML_VK_SEL_CM`.
- llama: `LLAMA_QSA_DENSE_SHORTCUT`, `llm_graph_input_qsa::maskless`; `qsa_decode_error` renamed `qsa_ubatch_error`.
- Measurement-only env hooks (to `patches/06-w13-measurement-hooks.patch`): `LLAMA_QSA_PREFER_DENSE`,
  `LLAMA_QSA_SHORTCUT_PIN`, `LLAMA_QSA_HOST_KV_COPY`, `LLAMA_QSA_FORCE_NONCOMPACT` (+ `QSA_NC_FORCED`), `LAB_OVL_PRINT`.
- Tests: QSA_PREFILL, QSA_DECODE, QSA_PREFILL_MASKLESS, QSA_DECODE_MASKLESS.
- ggml: `ggml_flash_attn_ext_add_top_k` asserts a mask.

## Done-when checks
| check | result |
|---|---|
| `grep -r "qsa_decode\|qsa_prefill\|flash_attn_sel\|QSA_DIRECT\|LLAMA_QSA_SELECTED_KEY\|QSA_MASKLESS_MAX_KV\|LLAMA_QSA_PREFER_DENSE"` (tree without build dirs, .git, scripts/qsa-validation) | empty (rc 1). 30 files of scripts/qsa-validation match: PR219-arm harnesses and evidence, kept by design |
| `grep -rn "ctx.pool()" ggml/src/ggml-cuda/sparse-attn-*` | empty (rc 1) |
| HIP quality check in the pinned image (two CI build steps, `-Werror`, `gcn-cdna-vgpr-check.py --baseline`) | -Werror build rc 0; test_vgpr_check 18 OK; old baseline rc 1 (qsa-decode 0 < 4, qsa-prefill 0 < 6); regenerated baseline PASS, 8868 kernels, 162 TUs, 0 spilling |
| CUDA CI red on master | run 41354919590 (ee30b949e, jobs cuda + musa failure); master head 62a3f4144 run 42326610175 also cuda + musa failure |
| merge-tree | `harness/w13/merge_tree.sh` (table below) |

## merge-tree of the W13 head (11b42ee8e) against master and the open PRs
| ref | head | result |
|---|---|---|
| origin/master (contains #234, #259) | 62a3f4144 | conflicts: qsa-prefill.cu (modify/delete: deleted here, modified by #234), ggml-vulkan.cpp (3 hunks), scripts/hip/gfx1151-vgpr-baseline.json (6), tests/CMakeLists.txt (1) |
| #234 (merged 2026-11-07) | 0ec95bc5c | same four files |
| #225 | e6849cfc6 | clean |
| #247 | 9b7c19212 | ggml-cuda.cu (1 hunk) |
| #248 | 26dbe6369 | tests/CMakeLists.txt (2 hunks) |
| #230 | a3d51093b | clean |

## Correctness (W13 build, HIP build/ and Vulkan build-vk/)
- test-backend-ops ROCm0: SPARSE_ATTN_PLAN 185/185, SPARSE_ATTN 1390/1390, FLASH_ATTN_EXT 4051/4051, QSA_EXPAND 56/56,
  QSA_SCORE 13/13, QSA_STRIP 6/6. Vulkan0: 92/92, 744/744, 5383/5383, 56/56, 13/13, 6/6.
- ctest -R "test-qsa|test-sparse-attn": 11/11 (HIP), 11/11 (Vulkan).
- W7 guard harness: pass q8_0/f16/q4_0/bf16, fake device, reject, reject-decode, allvis PASS; split-mode tensor and
  n_ctx cap init errors as in W7.
- KLD smoke (W11 harness, base = W9-head copy): q4_0 auto 0.028154 / PPL 2.124025, q4_0 override KLD 0 / 2.127075,
  f16 KLD 0 / 2.108191, q8_0 KLD 0 / 2.113919: equal to the W11 rows digit for digit.
- Vulkan c8192 1 chunk: q8_0 PPL 2.1006, f16 2.0936; 24 sparse + 12 dense trace rows, all on Vulkan0, census
  sa_attend/vk/cm/direct 24, sa_plan/G1 24.
