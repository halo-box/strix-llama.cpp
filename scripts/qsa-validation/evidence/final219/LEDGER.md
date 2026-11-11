# Evidence ledger: QSA rewrite with first-class quantized KV (halo-box/strix-llama.cpp, PR #219)

Snapshot: 2026-11-04 00:14 CET, built READ-ONLY. No build and no GPU run was made for this ledger. The only commands run on strix-002 were git, grep, sed, ls and CPU-only python over existing logs.

**Abbreviations used in the Source column**
- `PR` means `~/dev/wt_pr219v2` at 94e086847.
- `BASE` means `~/dev/wt_pr219v2_base` at 13a279bfa.
- `LOGS` means `~/dev/wt_pr219v3_logs`.
- `QC` means `~/dev/wt_measQ/q/campaign.out`.
- `QL` means `~/dev/wt_measQ/q/logs`.
- `SP` means the scratchpad directory `/tmp/claude-1000/-home-gaetan-puleo/78d4f4eb-9ba7-448e-8841-dce6638c30f6/scratchpad`.

**Status values**
- **measured**: read from a log, with the value given.
- **code-verified**: I read it at the stated file:line and commit during this pass.
- **contested**: the sources disagree, or the claim rests only on estimates yet drives a design decision. Each one has an entry in section 2.
- **unverified**: taken from a brief or plan and not re-checked here, or inferred.

**State of the inputs**
- `rewrite219/results-Q.md` and `rewrite219/results-P.md` **do not exist**.
- Track Q is still running: `campaign.sh` (pid 3049934) is on `q4_idxf16_qsamoe`. The Q values below come from QC and QL as of this snapshot.
- Track P has not started: there is no `~/dev/wt_measP`. Every Track P item from MEASURE.md is therefore listed as contested (C1–C7, C12).

---

## 1. Facts

### 1.1 PR #219: provenance and recoverable content

| id | claim | status | source |
|---|---|---|---|
| L1 | PR #219 is OPEN and MERGEABLE, base master, head 94e086847. GitHub author login `iggy`, name "Iggy Jackson". One CHANGES_REQUESTED review by gaetan-puleo (5726896660). Our earlier review is issue comment 6554261242 | measured | `gh pr view 219 --json …`; `gh api …/pulls/219/reviews`; `…/issues/219/comments` |
| L2 | The PR has 3 commits on 13a279bfa, all authored `Iggy Jackson <iggy@theiggy.com>`, each with trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. The Co-authored-by trailer for ported code is therefore `Co-authored-by: Iggy Jackson <iggy@theiggy.com>` | code-verified | `git log --format="%H %an <%ae>" 13a279bfa..94e086847` in PR |
| L3 | **68da4191d mixes two unrelated changes:** (a) q8_0 K/V in qsa_decode WMMA and qsa_prefill pack, plus a `selected_key_q8` flag with a "ROCm" name check; (b) the `strip_views` sharing of score-strip views (models.h `strip_views` map; qwen4exp.cpp strip_view lambda around the bias/kq_mask views). Recovering strip_views alone means a partial port of this commit: only the qwen4exp.cpp/models.h strip_views hunks, plus the key change from b60447506 | code-verified | `git show 68da4191d` in PR |
| L4 | b60447506 does four things: replaces `selected_key_q8` and the ROCm check with a probe using the real K/V types at n_q {1,513}; keys strip_views on (input, first, strip size); adds q8_0 and mixed tests; puts the types in the log line | code-verified | `git show b60447506 --stat` + message in PR |
| L5 | 94e086847 adds direct q8_0 K/V reads in qsa3_attn_kernel (template flag Q8). The commit message says all QSA_DIRECT modes are bit-identical to the pack (author claim) | code-verified (code); unverified (bit-identity claim) | `git show 94e086847` in PR |
| L6 | strip_views today: `std::map<std::tuple<const ggml_tensor*,int64_t,int64_t>, ggml_tensor*> strip_views` at models.h:2642, used at qwen4exp.cpp:1604-1606. It serves the non-compact (token-expanded/masked) scorer views of bias_all and kq_mask, which every plan keeps for non-scalar ubatches | code-verified | PR src/models/models.h:2642, src/models/qwen4exp.cpp:1604-1606 |
| L7 | The strip_views memory claim (q8_0 ub4096 compute buffer 7193 → 2969 MiB at 32k; ~0.15 MiB/cell growth; 10.7 GB at 262k; "perplexity bit-identical") was measured by the PR author only | unverified | commit message 68da4191d |
| L8 | The q8→half helper exists in 3 copies in the PR: `qsa3_q8x2_to_h2` (qsa-prefill.cu:22), `qsa_q8_to_h` (qsa-prefill.cu:630), `qd_q8x2_to_h2` (qsa-decode-wmma.cuh:9). All say they match `dequantize_block_q8_0_f16` (convert.cu:47) | code-verified | PR ggml/src/ggml-cuda/* |
| L9 | The PR's bool template is `qsa3_attn_kernel<bool DIRECT, int WPE, bool DIRECT_V, bool Q8>` | code-verified | PR qsa-prefill.cu:254 |
| L10 | Instance `qsa3_attn_kernel<1,8,1,1>` (q8 K+V direct) has `vgpr_spill_count: 28` and private segment 116; the other qsa3 instances have 0 spills | measured | `SP/notes.txt` (compiler metadata dump of the PR build) |
| L11 | The PR tests: q8 decode-maskless at q {1,9,64,512} × sel {257,2051}; prefill-maskless at 129/513/640 q × 4096 keys; 130 q × 32k; 200 q × 64k; 70 q × 64k; mixed q8/f16 at 513 q and 130 q. Cases with ≤512 queries and equal types are taken by qsa_decode (dispatch order, L22), so 129/130/200/70 never run qsa_prefill. No test has n_q > 512 with nk > 64·n_q, the only shape that reaches `<1,8,1,1>` | code-verified | `git show 94e086847 b60447506 -- tests/test-backend-ops.cpp`; PR fattn.cu:1154-1160; qsa-prefill.cu:684 |
| L12 | The author's reply (comment 6563853593) attributes the ub512 d0 slowdown to the qsa_prefill repack. His q8 tables (comment 6560590152) come from his own setup: ub512 pp4096 d0 −6.2%, later "−3%" after 94e086847 | measured (as posted) | gh issue comments 6563853593, 6560590152 |
| L13 | origin/master is ee30b949e, 6 commits ahead of 13a279bfa. Those commits touch only ggml-vulkan.cpp (+216/−27) and test-backend-ops.cpp (+99) among the QSA-relevant files. Open PR #251 (fix/qwen4exp-qsa-layer-types, conversion/qwen4exp.py) fixes compress_ratios being written as 0 for `indexed_attention` layers, which turns QSA off at runtime. Open PR #248 skips QSA tests on Windows shared builds | measured | `git rev-list --count 13a279bfa..origin/master`, `git diff --stat` in PR; `gh pr list`, `gh pr view 251` |

### 1.2 Route choice, probe, selection (llama side)

| id | claim | status | source |
|---|---|---|---|
| L14 | Probe `qsa_selected_key_supported`: a no_alloc context (`data=nullptr`), n_q ∈ {1, 513}, real type_k/type_v, k/v with 4·n_sel keys. All QSA layers must pass, together with `offload` | code-verified | PR src/llama-memory-hybrid-idx.cpp:50-66, :120-128 |
| L15 | `LLAMA_QSA_SELECTED_KEY` overrides the probe result in both directions | code-verified | PR llama-memory-hybrid-idx.cpp:129; BASE :124 |
| L16 | The indexer key cache is created with the attention `type_k` (inherits -ctk) and is not rotated (`rope_type NONE`; the comment says the keys are raw) | code-verified | PR llama-memory-hybrid-idx.cpp:115 (and :100-104) |
| L17 | BASE eligibility requires `type_k==F16 && type_v==F16`, so every quantized cache takes the masked route on BASE | code-verified | BASE src/models/qwen4exp.cpp:1107-1112 |
| L18 | Selected-key eligibility also requires `ubatch.token` (so image/embedding ubatches are excluded), causal, ratio>1, n_kv > top_k+ratio−1, flash_attn, offload_kqv, D=256, and n_kv ≤ 1048576 (QSA_MASKLESS_MAX_KV) | code-verified | PR qwen4exp.cpp:1102-1124 |
| L19 | Complete-block (maskless) rows are used only when the ubatch is `scalar`: blocks && n_swa==0 && (n_stream==1 \|\| n_tps≥128) && `qsa_scalar_visibility`. Otherwise the token-expanded `ggml_top_k(expanded, width)` plus masked FA is used | code-verified | PR qwen4exp.cpp:1453-1456, :1657-1678 |
| L20 | `qsa_scalar_visibility_cells` returns false for: a 2-D (image) cell at the linear position of a query in the ubatch; several sequences with any 2-D cell; a block shared by two sequence sets. Text ubatches after an image in a single-sequence cache stay scalar unless a query sits at an image position | code-verified | PR src/prefix.h:15-60 |
| L21 | Complete-block selection: top_k → argsort through an f32 cast of block ids (:1192) → cells through an f32 cast (:1204), which caps exact ids at 2^24. Output is block-ascending, cells mapped through `block_cells`, so not guaranteed ascending under permutation, with -1 for invisible blocks. The token-expanded path returns top-k (score) order | code-verified | PR qwen4exp.cpp:1182-1206, :1674 |
| L22 | The code comments name complete blocks + tail as the reference: "the reference returns indexer_top_k + compress_ratio - 1: whole blocks plus the tail" (:1671). The -1 sentinels are understood "only [by] the maskless kernel", and that is why non-scalar ubatches use token-expanded (:1468-1472) | code-verified | PR qwen4exp.cpp:1468-1472, :1671 |
| L23 | Dense shortcut: when n_kv ≤ top_k+ratio−1 (2051), the layer only stores indexer keys and attends dense. `LLAMA_QSA_DENSE_SHORTCUT=0` disables it | code-verified | PR qwen4exp.cpp:1866-1878 |
| L24 | `LLAMA_FN_LAST_ATTN_TRIM` (default on) runs the last layer's attention only for the last strip, `min(n_tokens,512)` queries, when every output row is in it. It requires the maskless route and one stream. So at ub4096 layer 47 sends n_q ≤ 512 to qsa_decode | code-verified | PR qwen4exp.cpp:936-951, :1077-1079, :1787-1797, :1870-1900 |
| L25 | `LLAMA_MAX_SEQ` = 256 | code-verified | PR src/llama-cparams.h:8 |
| L26 | The MoE expert choice is `selected_experts = ggml_argsort_top_k(...)` (the pin point used by QPIN) | code-verified | PR src/llama-graph.cpp:2362 |
| L27 | The `"does not match expectation"` reserve warning exists, and was observed at 510.2429 vs 510.2112 MiB (ROCm0) and 27.03 vs 25.11 MiB (ROCm_Host) | code-verified + measured | PR src/llama-context.cpp:695; `~/dev/wt_planB/planb/noise_f16_ub512.log:60-61` |

### 1.3 HIP backend

| id | claim | status | source |
|---|---|---|---|
| L28 | On HIP, `ggml_cuda_flash_attn_ext` tries qsa_decode, then qsa_prefill, then fa7, then aborts on maskless FA+src[5]. Masked FA+src[5] goes to fa7/tile/vec/mma, which do not read src[5] | code-verified | PR ggml/src/ggml-cuda/fattn.cu:1150-1192 |
| L29 | `ggml_cuda_flash_attn_ext_supported` returns decode‖prefill support for maskless FA+src[5] (HIP only), so an unsupported op falls to another backend (the CPU) | code-verified | PR fattn.cu:1196-1208 |
| L30 | qsa_decode needs RDNA3.5; K∈{f16,q8_0} with **v->type == k->type**; 1 ≤ n_q ≤ 512 (QSA_DECODE_MAX_QUERIES); n_kv ≤ 2^20; a single stream. q8_0 also needs GQA 12 plus 16-byte alignment of K (WMMA only). The WMMA predicate is duplicated at :102 and :132-133 | code-verified | PR qsa-decode.cu:84-103, :128-148; qsa-decode.cuh:5; common.cuh:109 |
| L31 | qsa_prefill needs RDNA3.5; n_q ≥ 128; K and V each in {f16, q8_0} (so mixed is accepted); GQA 12; K alignment checked; **V data alignment not checked**; ids ≤ 2560 wide | code-verified | PR qsa-prefill.cu:608-627 |
| L32 | qsa_prefill auto mode: K is read direct; V is read direct only when n_q·64 < nk; mixed types always pack. Packs and union scratch (`packed_k/v`, `ublk`, `umask`, `ucount`, `srow`, `sflag`) come from `ctx.pool()` | code-verified | PR qsa-prefill.cu:678-707 |
| L33 | qsa_decode scratch (`unique_ids`, `partial` nq·nh·splits·258 floats) also comes from `ctx.pool()` | code-verified | PR qsa-decode.cu:130, :145 |
| L34 | `GGML_HIP_NO_VMM` defaults to ON, and the PR build has it ON, so the pool is the legacy (non-VMM) pool | code-verified | PR ggml/CMakeLists.txt:222; `PR/build/CMakeCache.txt:512` |
| L35 | ggml-cuda already lets FA over-allocate through `get_alloc_size` (`ggml_cuda_flash_attn_ext_get_alloc_size`) | code-verified | PR ggml-cuda.cu:1020-1024; fattn.cu:751 |
| L36 | HIP dense FA K/V types (vec instances): f32, f16, bf16, q4_0, q4_1, q5_0, q5_1, q8_0. No iq4_nl | code-verified | PR fattn.cu:528-541 |
| L37 | CPU FA implements selected keys from src[5] (the CPU oracle, and the silent-fallback target) | code-verified | PR ggml/src/ggml-cpu/ops.cpp:8946, :8988 |

### 1.4 Vulkan backend

| id | claim | status | source |
|---|---|---|---|
| L38 | Vulkan maskless FA+src[5] is supported only when `ggml_vk_flash_attn_sel_pipeline` returns a pipeline. That needs **K and V f16**, D=256, GQA ∈ {4, 8, 12}, ids ≤ 4096 and 32-bit offsets | code-verified | PR ggml-vulkan.cpp:10577-10617, :21303-21306 |
| L39 | Vulkan's masked top-k sparse path (`ggml_vk_flash_attn_top_k`) handles only D=512, MQA (k ne2=1), q ne2=64, K==V same tensor. It does not apply to QSA's D256/GQA12. Masked QSA on Vulkan therefore runs dense FA | code-verified | PR ggml-vulkan.cpp:10714-10740 |
| L40 | Vulkan dense FA rejects bf16 mixed with non-bf16 | code-verified | PR ggml-vulkan.cpp:21352-21354 |
| L41 | Vulkan FA dequant switches on the FaTypeK/FaTypeV spec constants for f32, q4_0, q4_1, q5_0, q5_1, q8_0, iq4_nl, bf16 (f16 is native) | code-verified | PR vulkan-shaders/flash_attn_dequant.glsl:123-142 |
| L42 | The Vulkan buffer-type `get_alloc_size` returns `ggml_nbytes` (no FA workspace hook today) | code-verified | PR ggml-vulkan.cpp:17125-17128 |
| L43 | strix-002 has a Vulkan stack: mesa-vulkan-drivers 26.0.3 (radeon_icd), vulkan-loader 1.4.341, glslc in ~/.local/bin, and build-vk directories in other worktrees. On 2026-10-13 ggml_vulkan reported `Radeon 8060S Graphics (RADV STRIX_HALO) … fp16: 1 \| bf16: 0 \| warp size: 64 \| shared memory: 65536 \| int dot: 1 \| matrix cores: KHR_coopmat` (no coopmat2, no bf16). Neither PR tree has a Vulkan build | measured (as of 2026-10-13; current Mesa not re-probed) | `rpm -qa`; `ls /usr/share/vulkan/icd.d`; `~/dev/sync-validate/6205701a4/vk/bench/small-BF16-base-1.log` |
| L44 | Plans W, X and Z put RADV acceptance and experiments on **strix-001**, which the hard rules forbid. All Vulkan validation must run on strix-002 | code-verified (plan text) | rewrite219/plan-v2-W.md §1.3 G1; plan-v2-X.md PF-9; plan-v2-Z.md (G6 / E-P6) |

### 1.5 Rotation and gate

| id | claim | status | source |
|---|---|---|---|
| L45 | K rotation is on iff `ggml_is_quantized(type_k)` and D%64==0. n_rot_k doubles from 64 while it divides D (so 256 for D256). V rotation is on iff `ggml_is_quantized(type_v)`, with n_rot_v = 64. bf16/f16 are not rotated. `LLAMA_ATTN_ROT_DISABLE` turns rotation off | code-verified | PR src/llama-kv-cache.cpp:323-375 |
| L46 | For QSA the rotation is applied in `build_attn_qsa`: q and k rotated (:1698-1700), v rotated (:1703-1704), output un-rotated after FA in both the maskless (:1737-1738) and masked (:1780-1781) branches. FACTS cites llama-graph.cpp ~3344, but that is the generic `build_attn`, not the QSA path | code-verified | PR src/models/qwen4exp.cpp:1688-1784; llama-graph.cpp:3299-3345 |

### 1.6 MTP

| id | claim | status | source |
|---|---|---|---|
| L47 | `graph_mtp` builds the draft block through `build_layer_attn(inp_attn, mctx_hyb, …)`. The draft is sparse when its context has a hybrid-idx memory and the block has indexer q/k proj+norm, and dense otherwise. The ":1857 MTP draft attends dense" comment is stale | code-verified | PR qwen4exp.cpp:495-541 vs :1857-1858 |
| L48 | Both shipped MTP sidecars contain `blk.48.indexer.{q,k}_proj.weight` and `{q,k}_norm.weight` plus the indexer metadata keys | measured | `strings` on `~/models/qwen3.8-flash-next-mtp/mtp-Qwen3.8-Flash-Next{,-shared}-Q8_0.gguf` |
| L49 | At runtime the MTP draft context took the sparse route with **f16** K/V while the main context was q8_0: `create_memory: MTP context uses a hybrid-idx memory (sparse draft attention)`, then `QSA attention: selected-key kernels (maskless block selection) (K f16, V f16)`. The draft KV type does not follow -ctk/-ctv (no -ctkd was given) | measured | `~/repro_img/server_pr_q8_0.log` lines 7072, 1.31.119 entry; source of the message: PR src/llama-model.cpp:3221 |

### 1.7 Memory

| id | claim | status | source |
|---|---|---|---|
| L50 | Image matrix (Signal-3.8 + MTP, -c 261888, b/ub 8192). PR q8_0 main ROCm0 compute buffer 3171.06 MiB; PR f16 and base f16 3170.80 MiB; draft context 2806.43 MiB; mmproj meta 248.10 MiB. Base q8_0 failed at context creation: `allocating 83908.76 MiB … cudaMalloc failed` | measured | `~/repro_img/server_{base_f16,pr_f16,pr_q8_0,base_q8_0}.log` |
| L51 | FACTS: compute buffer f16 4084 MiB c32k / 4133 MiB c131k; q8_0 6809 MiB c32k; q8_0 at c131k needs 21578 MiB and OOMs (master, b/ub 4096). The run logs behind these numbers were not re-located | unverified | rewrite219/FACTS.md "original issue" |
| L52 | Base q8_0 llama-bench at d131072, b/ub 4096 **did run**: pp512 247.0, pp2048 227.5 t/s. "Base q8_0 OOMs at c131k" is therefore setup-dependent. `perf/failed_oom/d131k_base_q8_0_ub4096.err` was a weight-load failure (78056.46 MiB), not a compute-buffer OOM | measured | LOGS/perf/r2/d131k_base_q8_0_ub4096.json; LOGS/perf/failed_oom/d131k_base_q8_0_ub4096.err |
| L53 | Large leftover files from other agents: `~/dev/critique219/base_f16_ub4096.kld` (4.07 GB), `~/dev/wt_measQ/q/ref8/base.kld` (16.3 GB), `ref8/pins` (6.0 GB), the last two still in use by the running Q campaign. Not mine; flagged for cleanup | measured | `ls -la`, `du -sh` |

### 1.8 Performance (llama-bench random tokens unless stated)

| id | claim | status | source |
|---|---|---|---|
| L54 | llama-bench feeds random tokens: `tokens[i] = std::rand() % n_vocab` with `srand(1)` | code-verified | PR tools/llama-bench/llama-bench.cpp:2220-2228 |
| L55 | r3 clean rerun, q8_0 ub512 d0, PR vs mean(base, base2): pp2048 1536.6 vs 1539.9 (−0.2%); **pp4096 1377.8 vs 1446.1 (−4.7%)**; pp8192 1316.8 vs 1349.2 (−2.4%). Base drift ≤0.1%. FACTS/orchestrator report −4.9% for pp4096, which is the r2 value; r3 gives −4.7% | measured | LOGS/perf/r3/d0_{base,pr,base2}_q8_0_ub512.json |
| L56 | r3 f16 ub512 d0, PR vs base: within ±0.3% at every pp | measured | LOGS/perf/r3/d0_*_f16_ub512.json |
| L57 | r3 q8_0 pp512 d32k: PR 1170.3 (ub4096) / 1169.7 (ub512) against base 672.4 / 673.2 and base2 622.6 / 676.1, i.e. +74% / +74% vs base (+88% vs base2 at ub4096, where base2 has a 476 sample). The orchestrator's +76% / +73% are within this range | measured | LOGS/perf/r3/d32kp512_*_q8_0_ub{4096,512}.json |
| L58 | r3 PR q8 vs PR f16 at pp512 d32k: 1170.3 vs 1254.1 (−6.7%, ub4096); 1169.7 vs 1247.9 (−6.3%, ub512) | measured | LOGS/perf/r3/d32kp512_pr_*.json |
| L59 | The ~421 t/s sample appears **3 times** in PR q8_0 ub512 pp512 d0 runs: r3 `[1380,421,1578,1582,1585]`; conf/pr_auto `[422,1573,1581,1580]`; conf/pr_d2 (QSA_DIRECT=2) `[1136,421,1560,1570]`. pr_d1 has 694; pr_d0 (pack) none. Base runs show dips but none near 421: conf/base2 `[1139,1178,955,966]`; r3 base q8 sample0 1247; r3 base2 f16 ub4096 sample0 899 | measured | LOGS/perf/r3/d0_pr_q8_0_ub512.json; LOGS/perf/conf/{pr_auto,pr_d0,pr_d1,pr_d2,base1,base2}.json |
| L60 | r2 (FACTS): q8 d131k ub4096 PR pp2048 1596.5 vs base 227.5 (7.0×), pp512 1040.3 vs 247.0 (4.2×); tg d32k PR q8 23.9 vs base 21.4 (+11.7%) vs PR f16 24.3. **Do not use** r2/d0_pr_q8_0_ub512.json (contaminated) | measured | LOGS/perf/r2/*.json |
| L61 | Op profile, ub512 d0 pp4096: base q8 (masked dense FA) vs PR q8 (QSA): 37.9 vs 65.2 ms per ubatch. g34 (pp512 graph at 131k): FA 64.9 ms f16 vs 85.2 ms q8. Depth-fill graphs at 131k: 391 ms f16 vs 351 ms q8 | unverified (per-graph aggregation by FACTS/critique; raw lines exist) | LOGS/perf/r2/prof_*_ub512_p4096_d0.log, prof_pr_*_ub4096_p512_d131072.log (OPPROF lines) |
| L62 | One 585.363 ms qsa3 attention call at nk 45056, nq 4096 (f16, wikitext), with a normal union (g4_mean_blk 838.8, pack 0.806 ms, union 1.174 ms) | measured | `~/dev/wt_planA/m/A_f16.log:143` |
| L63 | Union inflation, G4 union / per-query blocks, all-layer mean: 2.11×/2.48×/2.60×/2.66× at 12k/28k/45k/61k on random tokens, against 1.74× on wikitext at 61k. Attention time per layer per ubatch: 25.9 vs 16.1 ms at 61k. The wikitext number is A's log re-aggregated, and A's sampler covers only one layer per ubatch | measured (partial coverage) | rewrite219/critique.md table; `~/dev/critique219/bench_rand.err`; `~/dev/wt_planA/m/A_f16.log` |
| L64 | Image matrix prompt t/s (A 60k text / B 1 image + 60k / C 5 images + 150k, 180157 tokens): base f16 1106/1127/1055; PR f16 1124/1130/1061; PR q8_0 1131/1120/1050. Answers correct in every case (2749/12749/26999). These requests test text recall, not image understanding. "non-consecutive" warnings: 0/4/20 in each build | measured | `~/repro_img/matrix.log` |

### 1.9 Quality

| id | claim | status | source |
|---|---|---|---|
| L65 | Unpinned reorder floor (critique, PR binary, c8192 × 2 chunks): f16 ub4096 rerun 0.000000; ub2048 0.011885 ± 0.000586; ub1024 0.011292 ± 0.000587; ub512 0.012271 ± 0.000613 (plan B). q8_0 0.01264 ± 0.00058 | measured | `~/dev/critique219/kld.out`; `~/dev/wt_planB/planb/noise.out`; `planb/h1.out` |
| L66 | Plan B H1: KLD q8 KV + f16 indexer 0.012861; f16 KV + q8 indexer 0.011329; q8 control 0.012637. None of these can be told apart from the floor (L65) | measured | `~/dev/wt_planB/planb/h1.out` |
| L67 | QPIN works. The record run is f16 ub4096 c8192 × 8 chunks, PPL 3.2819 ± 0.0370, 757760 QSA rows and 3112960 MoE ids. Replaying it into itself with qsa+moe pins, or with no pins, gives KLD −0.000000 (same top p 100%) | measured | QC lines `ref8_rec`, `self_qm`, `self_none` |
| L68 | A **plain** f16 run (QPIN env off, same binary) against the QPIN record reference gives KLD 0.008817 ± 0.000198, chunk SE 0.001788. Recording itself changes numerics against plain inference; the cause (cb_eval graph splits?) is UNVERIFIED | measured | QC `plain_f16`; QL/plain_f16.log (parse.py) |
| L69 | Floor ladder, f16 ub2048 vs the ub4096 reference (KLD ± token-SE, chunk SE): none 0.006580 (chunk SE 0.001184); QSA-only 0.006632 (0.001239); **QSA+MoE 0.000108 ± 0.000002 (chunk SE 0.000012)**, 99.9% 0.003634, same top p 99.679%. QSA-only pinning removes nothing; MoE pinning is required | measured | QC; QL via `python3 ../parse.py` |
| L70 | Type costs, same ladder (none / QSA / QSA+MoE): q8_0 0.007529 / 0.007293 / **0.000137 (chunk SE 0.000014)**; q8_0K+f16V 0.006929 / 0.007005 / **0.000126 (0.000017)**; q4_0 0.020001 / 0.019411 / **0.005786 (0.000628)** | measured | QC; QL via parse.py |
| L71 | Pinned q8_0 against the pinned floor, per chunk (CPU arithmetic on the parse.py per-chunk values, rounded to 1e-5): differences +3, +5, −2, +6, +3, +3, +3, +3 ×1e-5; mean 3.0e-5, paired t ≈ 3.6 (7 of 8 positive). Unpaired with chunk SEs: 1.6σ. q4_0 pinned is 54× the floor | measured (derived) | QL/q8_qsamoe.log, QL/f16ub2048_qsamoe.log, `parse.py` per-chunk output |
| L72 | The q4_0 and q8K/f16V runs use the PR's **masked/token-expanded** route (tk_tensors=384, cb_tensors=0, rows_absent=57344), while f16 and q8_0 use complete-block. Their pinned KLD therefore mixes two selection rules | measured (counters); route consistent with L30/L14 | QC QPIN_SUMMARY lines; PR qsa-decode.cu:93; llama-memory-hybrid-idx.cpp:51 |
| L73 | Selection agreement (computed vs reference rows, under qsa+moe pins): f16 ub2048 0.997386 (exact 0.2556); q8_0 0.996726; q8K/f16V 0.998577; q4_0 0.987205; f16 KV + q8 indexer 0.998509 (exact 0.386); f16 + q4 indexer **0.979373**; q8_0 KV + f16 indexer 0.997135 | measured | QC lines `*_qsamoe`, `f16_idxq{8,4}_qsamoe`, `q8_idxf16_qsamoe` |
| L74 | With QSA rows pinned, the indexer type does not affect KLD: f16+idx q8 and f16+idx q4 both give KLD −0.000000; q8+idx f16 gives 0.000137, identical to q8+idx q8. Pinned KLD cannot judge the indexer cache type (W §7.3 and Z §7.3 are right; critique Q7 is not) | measured | QC lines `f16_idxq8_qsamoe`, `f16_idxq4_qsamoe`, `q8_idxf16_qsamoe` |
| L75 | Still pending at the snapshot: q4_idxf16_qsamoe (running), q4_idxq8_qsamoe, all of Q4 depth (ref32 record, d32 q8 none/qsa/qsamoe, d32 f16 ub2048 none/qsamoe), and the unpinned indexer KLD runs (f16_idxq8, q8_idxf16, q4_idxf16, q4_idxq8) | measured (process list + campaign.sh) | `ps` on strix-002; `~/dev/wt_measQ/q/campaign.sh` |
| L76 | FACTS: "no valid depth-quality evidence (the 58.8k check compared 1 token)". The masked and selected-key routes pick different keys (logprob −0.493 vs −0.459 at 58k) | unverified (not re-checked) | rewrite219/FACTS.md; LOGS/quality/deep*.json |

### 1.10 Plan positions (what each source asserts; these drive the contested items)

| id | claim | status | source |
|---|---|---|---|
| L77 | Op count: critique, W, Y and Z want two ops (integer plan + attend). X wants one op with a private plan | code-verified (plan text) | critique §5.1; plan-v2-W §0.2; plan-v2-X §3.1, §6.1; plan-v2-Y §2.1; plan-v2-Z §0 |
| L78 | Dense mode / shortcut. A and the critique table: a dense mode for small n_kv. W, X and Y: no dense mode, since the ub512 regression is the n_q≤512→decode policy. W routes the dense shortcut through the sparse op. Y keeps the dense shortcut on masked FA_EXT with an intersected type matrix. Z keeps f16/f16 on FA_EXT and sends other pairs to the sparse op with all-visible rows | code-verified (plan text) | plan-A; critique §4; W §0.3, §7.4; X §6.6; Y §0.4, §0.8; Z §7.8 |
| L79 | MTP. Critique: keep the draft dense. Y: the MTP draft stays on masked FA_EXT. W, X and Z: the draft already runs sparse, so it uses the new op. The code and the runtime log side with W/X/Z (L47-L49) | code-verified (plan text + code) | critique §5 table "MTP"; Y §0.8; W §1.2; X §0.2; Z §7.1 |
| L80 | Image / non-scalar selection rule. X and Y: one generalized complete-block rule. W and Z: keep token-expanded plus a validity filter for non-scalar ubatches, and treat a rule change as out of scope (model fidelity) | code-verified (plan text) | X §0.1, §3.2; Y §0.7; W §7.1; Z §2.7, §7.4 |

---

## 2. Contested items: the experiment that settles each one, and the decision it drives

Every GPU experiment below runs on strix-002 under `flock ~/gpu.lock`, one run per locked section, with no full test-backend-ops and `-j 1` for it. Each table gives the question, the evidence on each side, the experiment, and the decision it drives.

### C1. Dense mode yes or no (Track P4)

| | |
|---|---|
| **Question** | Does a sparse op at small n_kv / n_q lose to dense FA, so that a dense mode or dense shortcut must stay? Does G=4 beat base dense q8 at ub512 d0? |
| **Evidence** | Base dense q8 FA 37.9 ms vs PR QSA 65.2 ms per ub512 ubatch (L61, unverified aggregation). Positions split (L78) |
| **Experiment** | Use the op profiler (`GGML_CUDA_OP_PROF=1 GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_OVL=0`) with the BASE and PR binaries:<br>- n_kv ∈ {512, 1k, 2k, 4k, 8k, 16k}, n_q ∈ {128, 512, 1024, 4096}, f16 and q8_0;<br>- PR with LLAMA_QSA_DENSE_SHORTCUT=0 and the default; BASE masked dense;<br>- also time the trimmed-last-layer case (n_q 512 at ub4096);<br>- report FA ms per ubatch, n ≥ 5.<br>Then rerun the same cells on a G=4 prototype with all-visible rows once one exists |
| **Decision it drives** | Keep or delete FLASH_ATTN_EXT on QSA layers, and the shortcut threshold. Sparse ≥ 0.98× dense (W's rule) → delete it; otherwise keep it for f16/f16 only. It also fixes the G1/G4 crossover n_q |
| **GPU** | yes |

### C2. MTP: sparse or dense (Track P6)

| | |
|---|---|
| **Question** | Does the MTP draft run sparse QSA with the shipped sidecar, and must the rewrite route it through the new op? |
| **Evidence** | Code says sparse when an indexer is present (L47). The sidecar has the indexer (L48). The server log shows a sparse route with f16 draft KV (L49). The critique and Y say "dense / masked FA_EXT" (L79). Still missing: an op census confirming that the draft's FA nodes reach qsa_decode/prefill, and the draft behaviour with `-ctkd q8_0` |
| **Experiment** | `llama-server` (PR build) with the repro_kv.sh flags plus `-ctkd q8_0 -ctvd q8_0`, a short text request, `GGML_CUDA_OP_PROF=1` or a debug counter on the qsa_decode/qsa_prefill launches, filtered to the draft context. Log the draft acceptance rate for f16 vs q8_0 draft KV |
| **Decision it drives** | The new op covers the MTP draft: placement guard, reserve, and an acceptance-rate gate. Dropping the critique/Y "keep the draft dense" means a quantized draft KV needs the same type coverage as the trunk |
| **GPU** | yes |

### C3. One op or two

| | |
|---|---|
| **Question** | One op with a private plan (X) or a plan op plus an attend op (critique, W, Y, Z)? |
| **Evidence** | Plan positions only (L77); no plan-cost or allocator test exists. Plan-cost budgets disagree: B ≤0.3, W ≤1.0, Y ≤1.3 (≤0.5 with bucketing), Z ≤1.3 ms/layer. Today's rows+merge is 1.3 ms/layer (plan A M2, unverified here) |
| **Experiment** | (a) Prototype the plan stage (validate, sort, dedup, merge, optional bucket) as a standalone kernel on captured rows at ub4096, d131k, all 12 layers. Time it with hipEvents, n ≥ 10, text and random rows. (b) Check that a plan tensor with one consumer survives gallocr reuse, graph splits and HIP graph capture: allocate it in a 2-split test graph and memcmp the outputs over 20 iterations |
| **Decision it drives** | If the plan cost is under budget and (b) passes, go with two ops (bit-exact plan vs CPU). If the plan must fuse into attend for cost, go with one op and a debug plan dump. This freezes the ggml.h contract |
| **GPU** | yes |

### C4. Selection rule for images and non-scalar ubatches (Track P7)

| | |
|---|---|
| **Question** | Does the token-expanded rule (used today for image/embedding ubatches and any non-scalar visibility, L18-L20) differ from complete-block on real image prompts? Is complete-block the reference rule? |
| **Evidence** | Code comments name complete-block + tail as the reference (L22). W and Z refuse to change the rule in this PR (L80). The image matrix passes but tests only text recall (L64) |
| **Experiment** | On the PR f16 build with Signal-3.8 + mmproj, request B (1 image + 60k):<br>- dump `indexer_top_k-L` for one QSA layer through cb_eval for the image ubatch and the first text ubatch after it;<br>- compute both rules offline on the same `indexer_score` tensor; report the set difference, the attention-mass difference, and the fraction of rows where token-expanded omits the query's own cell;<br>- add an image-understanding prompt (describe the image), compare its answer under both rules, and compute pinned KLD over a fixed answer |
| **Decision it drives** | Either one rule everywhere (X/Y), or keep token-expanded plus a −1 filter for non-scalar (W/Z). The image test set; whether strip_views stays load-bearing |
| **GPU** | yes |

### C5. Union and bucketing payoff on real text vs random tokens (Track P2)

| | |
|---|---|
| **Question** | What do the per-query union for G ∈ {1, 2, 4, 8} and the bucketed tile count look like on real text against random tokens, over all 12 layers? |
| **Evidence** | 1.74× (text, one layer per ubatch) vs 2.66× (random) at 61k (L63). B's bucketing tile count is analytical only |
| **Experiment** | Extend B's `qsa_union` (cb_eval on `indexer_top_k-L`, all 12 layers) to f16 ub4096 runs on wikitext, a code corpus (llama.cpp src) and a chat corpus (~/dev/llm-eval-corpora), plus llama-bench random tokens, at depth {8k, 32k, 64k, 131k}. Report the union for G ∈ {1,2,4,8}, tile-chunks per query for ascending vs bucketed order, and the permutation rate. Then time G4-ASC, G4-BUCKET and G8 prototypes on replayed rows |
| **Decision it drives** | Whether bucketing enters the plan layout (W/X/Z: ≥10% at d32k on text; Y: ≥15%), the surviving G set, and the bench weighting |
| **GPU** | yes (stats capture needs model runs) |

### C6. Cause of the depth-dependent FA cost

| | |
|---|---|
| **Question** | Union inflation from random tokens (critique), MALL misses or memory-bound behaviour (A), or something else? |
| **Evidence** | 13.7 / 8.9 ms/layer growth at 131k (FACTS); the attention-per-block slope is 0.0173-0.019 ms (A and critique); the text vs random union gap (L63) |
| **Experiment** | (a) Synthetic plans with a fixed 513 tiles per query, cells spread uniformly over n_kv 4k → 1M, f16 and q8. Time the attention kernel, and check whether time/tile grows more than 20%. (b) Time the same depth on text vs random fills (C10 harness) |
| **Decision it drives** | Grid order (MALL/L2-friendly, interleaved kv-heads), V prefetch depth, and how much the loader design matters |
| **GPU** | yes |

### C7. Cause of the ~421 t/s stall and the 585 ms FA call (Track P5)

| | |
|---|---|
| **Question** | Pool growth (hipMalloc/hipFree with an implicit sync on NO_VMM), graph recapture, lazy loading (-lzm on-direct), or the kernel itself? |
| **Evidence** | 421 appears 3× on PR, in QSA_DIRECT auto and mode 2, not in mode 0 (L59). Base dips to 955 but never near 421 (L59). The 585 ms call had a normal union (L62). Scratch comes from the pool (L32-L34) |
| **Experiment** | `rocprofv3 --hip-trace` (or an LD_PRELOAD hipMalloc/hipFree logger) on `llama-bench -ctk q8_0 -ctv q8_0 -ub 512 -p 512 -d 0 -r 30`:<br>- PR build, QSA_DIRECT auto/0/1/2, and BASE;<br>- repeat with `-lzm` off;<br>- correlate per-sample t/s with allocation timestamps;<br>- rerun the A-style 0-64k wikitext sweep with per-call hipEvents and the pool high-water mark |
| **Decision it drives** | Is "no ctx.pool in sparse kernels" enough to remove the stall (design), or is a root-cause work package needed before landing? It also sets the stall gate threshold |
| **GPU** | yes |

### C8. Pinned-KLD floor and type costs

| | |
|---|---|
| **Question** | Is QSA+MoE pinned KLD the primary quality gate, and with which statistic? |
| **Evidence** | Floor 1.08e-4 (L69): below W's 1e-3 threshold but above X/Y's "≤1e-4". q8_0 is 1.27× the floor, paired t ≈ 3.6 but unpaired 1.6σ (L71). Z's attention-local floor (T5) and planted-defect detection power were not measured. q4/mixed are mixed-route (L72). The recording perturbs plain numerics (L68) |
| **Experiment** | With QPIN on wt_measQ:<br>- (a) Z's T5: multiply the QSA FA output by (1 + 2^-12·ε) at ub4096, pinned QSA+MoE, 8 chunks;<br>- (b) planted defects: drop the last tail cell in layer 23; drop one block per query in layer 3; V×(1+1e-3) in one layer. Report each one's paired per-chunk t against the floor;<br>- (c) a second f16 ub2048 pinned run with a different chunk set (`--chunks 8` offset), to check floor stability;<br>- (d) plain vs plain rerun, to confirm L68 is the record hook and not nondeterminism |
| **Decision it drives** | The primary e2e gate (paired per-chunk pinned KLD vs floor + kσ), the stated minimum detectable effect (MDE), and whether layer-local replay must be primary |
| **GPU** | yes |

### C9. Indexer cache type

| | |
|---|---|
| **Question** | Should `--cache-type-k-idx auto` stay = -ctk, or map sub-8-bit types to q8_0/f16? |
| **Evidence** | Selection agreement: q8 indexer 0.9985 is at the floor (0.9974); the q4 indexer is clearly worse at 0.9794 (L73). Pinned KLD is blind to the indexer (L74). The unpinned indexer KLD runs and the q4-KV indexer runs are pending (L75). Plan A: q4 + f16 idx gives 0.0287 (1.6σ, unpaired) |
| **Experiment** | Finish QC's pending runs: q4_idxf16_qsamoe, q4_idxq8_qsamoe, the unpinned f16_idxq8, q8_idxf16, q4_idxf16, q4_idxq8. Add 16-chunk paired unpinned KLD at c8192 for q4 KV with q4/q8/f16 indexer, plus passkey at 32k |
| **Decision it drives** | The default mapping: auto → q8_0 for <8-bit types if its agreement is ≥0.99× f16's and its paired KLD is within 1σ (W E12). Otherwise f16. Also whether `--cache-type-k-idx` ships at all |
| **GPU** | yes |

### C10. Real-text perf of base and PR (Track P1)

| | |
|---|---|
| **Question** | Base vs PR, f16 vs q8_0, ub {512, 4096}, depth {0, 32k, 131k}, on real text against random tokens |
| **Evidence** | Every bench number so far is random-token (L54-L60), except the server image matrix (L64) and A's wikitext attention timings (L63) |
| **Experiment** | Patch llama-bench in a measurement worktree with `--prompt-file` (tokenized wikitext + code, cycled for depth fill and pp), or use llama-batched-bench / server with wikitext prompts. Run interleaved base/PR/base, n ≥ 12, on both workloads, and report the bootstrap CI of the ratio |
| **Decision it drives** | The baseline numbers for every perf gate; whether the PR's ub512 d0 regression and the q8-vs-f16 gaps hold on text |
| **GPU** | yes |

### C11. Vulkan capabilities on strix-002

| | |
|---|---|
| **Question** | What does RADV on strix-002 offer today (Mesa 26.0.3): coopmat1 vs coopmat2, bf16 (needed by Y's bf16-WMMA plan), subgroup size 64 vs 32, and the shared-memory limit? Does a Vulkan build of the PR tree run QSA at all? |
| **Evidence** | Measured once on 2026-10-13: KHR_coopmat, bf16 0, warp 64, 64 KiB shared (L43). Plans target strix-001, which is forbidden (L44) |
| **Experiment** | Build `build-vk` (-DGGML_VULKAN=ON) in the rewrite worktree. Run `llama-cli --list-devices` or `test-backend-ops -o FLASH_ATTN_EXT -b Vulkan0` (per-op, -j1) and record the `ggml_vulkan: 0 = …` line. Run the flash_attn_sel f16 path once on the model at d32k pp2048 |
| **Decision it drives** | The Vulkan kernel inventory (coopmat1 vs scalar), bf16 on Vulkan (convert vs native), the register-trick vs shared-memory coopMatLoad question (Q10 / E10), and the Vulkan perf gates |
| **GPU** | yes |

### C12. Last-layer trim confirmation (Track P3)

| | |
|---|---|
| **Question** | Does LLAMA_FN_LAST_ATTN_TRIM explain "11 of 12 layers reach qsa_prefill"? |
| **Evidence** | Code-verified (L24). Not yet confirmed at runtime |
| **Experiment** | Rerun A's stats hook (wt_planA build) on wikitext c65536 ub4096 with `LLAMA_FN_LAST_ATTN_TRIM=0`. Expect 12 calls per ubatch at nk ≤ 32k |
| **Decision it drives** | The trimmed shape (n_q 512 at ub4096) must appear in the policy table and the reserve list. It also confirms that every llama-bench pp ubatch runs layer 47 through the decode/row path |
| **GPU** | yes |

### C13. Cause of the ub512 d0 q8 regression

| | |
|---|---|
| **Question** | Author: the qsa_prefill repack (L12). FACTS and W/X/Y: n_q ≤ 512 → qsa_decode, slower than base's dense q8 FA |
| **Evidence** | Code: 512-query ubatches go to qsa_decode first (L28, L30). The 94e086847 prefill fix leaves r3 at −4.7% (L55) |
| **Experiment** | Op-profile the PR q8 ub512 pp4096 d0 run: FA time split by kernel (decode vs prefill launches), against BASE. Covered by the C1 sweep |
| **Decision it drives** | The fix belongs in the work-split policy (G=4 or a dense mode for ≤512 queries), not in the prefill loader. The PR comment must correct the author's attribution |
| **GPU** | yes |

### C14. bf16 handling

| | |
|---|---|
| **Question** | Convert bf16 → f16 with a range check (W/X/Z), or use native bf16 compute (Y)? |
| **Evidence** | No max\|K\|, \|V\| measurement exists. Vulkan reported bf16: 0 (L43). bf16 is not rotated (L45) |
| **Experiment** | Dump the K/V caches of all 12 layers at 131k (text and random) with -ctk bf16 -ctv bf16 and compute max\|K\|, max\|V\| on the CPU. Microbench bf16 WMMA vs f16 WMMA throughput on gfx1151 |
| **Decision it drives** | The bf16 trait (convert or native), and whether bf16 drops to the generic tier on Vulkan |
| **GPU** | yes |

### C15. Reserve coverage

| | |
|---|---|
| **Question** | Which route does `graph_reserve` build? Which tensor explains 510.2429 vs 510.2112 MiB (L27)? Does any runtime route (image/masked, trim, MTP, shared prefix) need more memory than the reserve? |
| **Evidence** | Plan reasoning (X: the reserve uses all-zero positions) |
| **Experiment** | Log the gallocr sizes per graph kind at reserve and at runtime (text, image request B, a 2-slot shared prefix, MTP, trim=0/1) on the PR build; diff the node sizes |
| **Decision it drives** | The reserve-variant list and the strict-reserve test |
| **GPU** | yes |

### C16. Base q8_0 OOM at c131k

| | |
|---|---|
| **Question** | FACTS says OOM (21578 MiB needed). llama-bench d131k base q8 ran (L52). The image server at c261888 OOMs at 83908.76 MiB (L50) |
| **Experiment** | `llama-cli`/`llama-perplexity` on BASE, -c 131072 -b/-ub 4096 -ctk/-ctv q8_0, with and without `-lzm on-direct`; record the `compute buffer size` line |
| **Decision it drives** | How the memory goal and the "before" numbers in the PR description are stated: as compute buffer MiB, not as OOM |
| **GPU** | yes |

### C17. Depth quality (Track Q4)

| | |
|---|---|
| **Question** | Does q8_0 (pinned) stay near the floor at depth? |
| **Evidence** | Pending (L75); no valid depth evidence yet (L76) |
| **Experiment** | Finish QC's ref32 block: c32768 × 2 chunks with QKL_TAIL=4096, q8 none/qsa/qsamoe against f16 ub2048 none/qsamoe |
| **Decision it drives** | Whether the quality claim may extend past 8k, and the size of the depth tier in validation |
| **GPU** | yes |

### C18. Bound regime and loader staging

| | |
|---|---|
| **Question** | Direct-register vs LDS-staged dequant per type (f16/q8/q4/bf16); is attention issue-bound or memory-bound per regime? |
| **Evidence** | A says issue-bound (q8 7% slower per block ≤64k on text). FACTS says q8 is faster at 131k on random tokens (351 vs 391 ms) |
| **Experiment** | Loader microbench on captured d4k/d131k fixtures, K and V separately; combine with C6(a) |
| **Decision it drives** | The `staging` trait per type, and whether PR #219's direct q8 loaders are reused as-is |
| **GPU** | yes |

### C19. strip_views payoff in the rewritten graph

| | |
|---|---|
| **Question** | strip_views must be kept (owner). The open question is only whether its memory effect is real on the remaining non-scalar path. The author's numbers are unverified (L7) |
| **Experiment** | PR build vs PR with the strip_view lambda bypassed (measurement worktree only), image request B at -c 65536 ub4096; compare the ROCm0 compute buffer and the realloc lines |
| **Decision it drives** | A regression test for the compute-buffer slope (X's G4: <0.002 MiB/cell), and the wording of "what was taken from #219" |
| **GPU** | yes |

---

## 3. Notes for the PR description and the #219 comment (sourced)

**Taken from #219, by fact id**
- strip_views (L3, L4, L6).
- The q8_0 dequant arithmetic shared with `dequantize_block_q8_0_f16` (L8).
- The direct q8 K-chunk = q8_0-block-w addressing in qsa3 (L5).
- The probe-with-real-types idea, superseded by an init-time guard.
- The q8/mixed test shapes (L11); they need census assertions because of the decode-first dispatch.

**Replaced, and why**
- The probe and override, which let an op reach the CPU silently (L14, L15, L29).
- The bool template flags and the spilling instance (L9, L10).
- Pool scratch (L32-L34).
- K==V in decode (L30).
- Unchecked V alignment (L31).

**Corrections to post on #219**
- The ub512 d0 cause is the decode routing, not the repack (L28, L30, L55; settled by C13).
- The "prefill" q8 tests ≤512 q run decode (L11).
- The MTP draft is sparse with f16 KV by default (L49).
- Pinned measurements: q8_0 costs 1.27× the reorder floor, versus 54× for q4_0 (L69-L71).
