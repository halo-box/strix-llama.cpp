# PLAN v3 (revision round 2) — QSA sparse attention with first-class quantized KV (halo-box/strix-llama.cpp, branch `qsa-rewrite`)

This is the definitive plan. It says **what to do**, in order. Every design decision cites its evidence:
- `Lnn` = fact in `EV/LEDGER.md` §1;
- `Cnn` = experiment in `EV/MEASUREMENTS.md` (`Cnn dk` = its decision k);
- `R1-x` = a round-1 lab section of `EV/MEASUREMENTS.md` (`R1-CB` = "R1-The 'one CB rule everywhere'…", line 2708; `R1-RES` = "R1-Skipping the non-compact reserve…", :2847; `R1-AV` = "R1-The new quantized dense-shortcut route…", :2906; `R1-NKVO` = "R1--nkvo…", :3080; `R1-Q8` = "R1-The Q8 detection-power calibration…", :3181; `R1-UB` = "R1-The e2e quality gates exercise only one…", :3321; `R1-E` = "R1-M-Lab E…", :3550; `OWNER` = "OWNER DECISION 2026-11-05", :3797; `R1-A` = "R1-M-Lab A…", :3807; `R1-C` = "R1-M-Lab C…", :4010);
- `R2-x` = a round-2 lab section (`R2-RES` = "R2-M-LAB-RES (shrunk): re-split mechanism…", :4222; `R2-BOP` = "R2-M-LAB-BANDS-OP (shrunk): GPU-vs-O_CON band", :4507);
- `PR file:line` = `~/dev/wt_pr219v2` at 94e086847 on strix-002 (QSA files outside ggml-vulkan and test-backend-ops are identical on origin/master, L13); `master file:line` = origin/master ea5285ddf where the file differs from the PR tree (ggml-vulkan.cpp, test-backend-ops.cpp);
- `review` = our issue comment 6554261242 on #219; `author` = Iggy Jackson's comments 6560590152 / 6563853593;
- `O<n>` = round-1 objection n, `P2-<n>` = round-2 objection n, listed in the Review log (§20).

**Evidence location (P2-27).** The evidence was produced in `SP/final219` (`SP` = `/tmp/claude-1000/-home-gaetan-puleo/78d4f4eb-9ba7-448e-8841-dce6638c30f6/scratchpad`), a plain directory on the root disk that systemd-tmpfiles empties at boot and ages after 30 days (`/usr/lib/tmpfiles.d/tmp.conf`: `D /tmp 1777 root root 30d`; `findmnt /tmp` empty). Before W1, work item **E0** (§13) copies it to durable storage. Every citation in this plan is written against that durable copy:
- `EV` = `strix-002:~/dev/wt_lab/evidence/final219` = a checksummed copy of `SP/final219` (80 MB `du -shc`, 2587 files with the two extras below; per-file SHA-256 in `EV/SHA256SUMS`, E0), plus `SP/bs_skeptic.py` → `EV/bs_skeptic.py` and `SP/skept_ocon/` → `EV/skept_ocon/`. Relative paths are unchanged (`EV/X` = `SP/final219/X`).
- The small summaries and harness sources that the PR cites are also committed under `scripts/qsa-validation/` in W12, and the PR description and the #219 comment cite repo paths or `EV` paths only, never `/tmp`.

Anything without a measurement or a code reading behind it is in §18 (UNPROVEN), together with the experiment that would settle it. There are no time, effort or calendar figures anywhere in this plan.

---

## 0. Scope, base, deliverable

- **Goal (owner).** Quantized KV caches are first-class in the QSA sparse-attention subsystem:
  - no memory or performance cliff for any supported K/V type, on HIP and on Vulkan;
  - never a silent CPU fallback;
  - the Hadamard rotation of quantized caches is kept (its computation may change);
  - output quality is provably preserved;
  - PR #219's `strip_views` fix is kept.
- **Branch.** `qsa-rewrite`, from **origin/master at branch time** (today ea5285ddf, "Merge pull request #251"; it is 8 commits ahead of #219's base 13a279bfa, `gh api …/compare/13a279bfa...master`). The base SHA is written into the W0 commit message and the PR description, and every "W0" reference arm below is built from that SHA (§0 "Base and landing order").
  - Reason 1: the Vulkan backend does not compile at 13a279bfa or 94e086847. The fix is 6411b4315 (#240), which is in master (C11, C11 d6).
  - Reason 2: #251 fixes compress_ratios being written as 0 for `indexed_attention` layers, which turns QSA off (L13). #251 (dc80afcfe) touches only `conversion/qwen4exp.py` (gh compare), so it matters only for re-converted GGUFs, not for the existing `~/models` GGUF used by every gate.
  - Master also carries #241, #242, #244, #245 (dense q8_0 × f16 GEMM `mul_mm_dq8`, on by default) and #246 (HC_UP_MIX on the dq8 GEMM, on by default), all in ggml-vulkan.cpp (+189/−27) (gh compare 13a279bfa...ea5285ddf). They change this model's dense-projection numerics on Vulkan (bdb70fc13, ee30b949e: tiling, split_k off, hc-up GEMM moved from int8 MMQ to f16 B; the model has 503 q8_0 tensors and `blk.N.hc_*` tensors, `~/dev/wt_pr219v3_logs/host/pr_q4_ub4096.log`). Every Vulkan perf **and quality** reference is therefore re-measured on W0-Vulkan, never taken from C11 or the PR219+#240 tree (O22, O36, O40, P2-25).
- **Base and landing order (P2-18, P2-24).** Open PRs touch the same code (`gh pr view <n> --json files`, 2026-11-05):
  - **#234** (pwilkin, APPROVED, MERGEABLE, head 777479045, +865/−2239): qsa-prefill.cu (deletes `qsa_pack_k` and the `QSA_DIRECT` / `GGML_QSA_RANKSORT` code at PR qsa-prefill.cu:564-640), fattn.cu, ggml-cuda.cu (OVL), ggml-alloc.c (−181), ggml-backend.cpp, llama-memory-hybrid-idx.cpp (incremental QSA cache on CPU, 5cbb82c45), qwen4exp.cpp, test-backend-ops.cpp, `scripts/hip/gfx1151-vgpr-baseline.json` and `gcn-cdna-vgpr-check.py`, plus non-QSA HIP numerics (mmq, mmvq, mmb, gated_delta_net, getrows, hyperconn). Measured (temp clone, git only): the three #219 commits cherry-pick cleanly onto ea5285ddf; on ea5285ddf + #234, 68da4191d and 94e086847 conflict in qsa-prefill.cu (`git merge-tree`), b60447506 is clean.
  - **#225** (qwen4exp.cpp), **#248** (tests/CMakeLists.txt QSA tests), **#247** (test-backend-ops.cpp, ggml-cuda.cu), **#230** (ggml-backend.cpp sched, RPC; relevant to the RPC note of §2).
  - **Rule.** (1) Immediately before W0, re-run the dry-run on the exact base SHA: a temporary index (`GIT_INDEX_FILE=… git read-tree <base>`, then `git apply --cached --check` of 68da4191d, b60447506, 94e086847 in order), and record the result in the W0 commit message. (2) If #234 is in the base: W0 becomes "cherry-pick with conflict resolution"; Iggy stays the author of every commit; each conflicted commit gets a `[conflict resolved: ggml/src/ggml-cuda/qsa-prefill.cu]` line; §12 rows for 68da4191d / 94e086847 say "adapted (conflict resolved)" instead of "as-is"; the §11 deletion list is re-derived against the new master by grep of the symbols (not by line numbers); W13 regenerates the VGPR baseline from #234's baseline file and checker; the `-ngl 0` recipes (AT-D2, Q12) are re-run on the new base because #234 moves the incremental QSA cache to the CPU. (3) If #234 is still open when this PR is ready, the PR description states that the two PRs conflict in qsa-prefill.cu and the VGPR baseline, and that whichever lands second rebases (owner's call; the orchestrator does not merge). (4) #225/#248/#247/#230: before W13, `git merge-tree` of the branch against each open head; any conflict is listed in the PR description with the file.
  - **Reference arms follow the base (P2-24).** Every PR219 reference arm used by a gate is **W0-lab** = <actual base> + the three #219 commits + the W12 measurement patches, built in `~/dev/wt_lab/w0lab` (HIP `build/`, Vulkan `build-vk/`). The round-1 arms `~/dev/wt_lab/r1av/tree` and `~/dev/wt_lab/r1q8/tree` (94e086847 on 13a279bfa) are history: they are used only to check that W0-lab reproduces them while the base's HIP code equals 13a279bfa's (true for ea5285ddf: L13, the 8 commits touch only Vulkan and conversion).
  - **Rebase protocol.** If the base moves during the work (a rebase of `qsa-rewrite`), these are re-run on the new W0-lab before any later gate is scored: W0 done-when; the W12 control reproduction; every W12 band calibration (AT-W12-BANDS) whose arm's numerics the base change can reach (any change under ggml/src/ggml-cuda, ggml-vulkan, src/llama-graph*, src/models/qwen4exp.cpp); the Q1/Q4 references; the P-campaign references (they are interleaved per campaign anyway, §16).
- **One branch, one PR.** The internal commits are the work items of §13, each buildable and testable. The first work item imports #219's commits as they are. The last deletes the old code.
- **Commit trailers.**
  - A commit taken as-is from #219 keeps its author `Iggy Jackson <iggy@theiggy.com>` and its message (L2; `git log --format='%h %an <%ae> %s' 13a279bfa..94e086847` on strix-002 prints this exact name and email for 68da4191d, b60447506, 94e086847).
  - Every commit that ports or adapts #219 code adds `Co-authored-by: Iggy Jackson <iggy@theiggy.com>`.
  - Every commit also carries the session attribution line the orchestrator gives.
- **Machine rules for every validation step.** Run on strix-002 only, never strix-001 (L44, C11 d5). Wrap every GPU command in `flock ~/gpu.lock`. Run test-backend-ops per op with `-o`, never the full suite, never `-j>1`. Never run llama-perplexity at `-c ≥ 64k` on this box: host logits OOM, and the OOM killer once took other users' processes (C16). Delete KLD bases, pin files and dumps > 1 GB after each campaign.

---

## 1. Decisions at a glance

| # | Decision | Evidence |
|---|---|---|
| D1 | **Two ops:** `GGML_OP_SPARSE_ATTN_PLAN` (integer; its whole dst is bit-identical on CPU, HIP and Vulkan) and `GGML_OP_SPARSE_ATTN` (attend). The plan has exactly one consumer and is read-only. **Every scratch area lives outside `ggml_nbytes(dst)`**: on GPU backends in the `get_alloc_size` over-allocation of the op's own dst (plan: sort/dedup scratch; attend: split-K / V-pack workspace), on the CPU in the graph work buffer. Both ops are added to `ggml_op_alloc_size_may_expand` | C3 d1, d2 (plan 0.733/0.883 ms per layer, 1056/1056 bit-exact; gallocr/splits/graphs 20/20); O1, O21, O29 |
| D2 | **Plan layout: ASC order only, G ∈ {1, 4}.** No BUCKET, no G8, no G2 | C5 d1, C5 d2 |
| D3 | **Work split G is a measured backend policy `G(n_q_per_stream, n_kv, T_k, T_v, shortcut)`.** HIP ships with the default `n_q < 128 ? 1 : 4` (C1 d3, C13 d1–d2) until W4's two-way crossover fit (§16.5 AT-A) replaces it; the **shortcut** (all-visible rows, n_kv ≤ 2051) gets its own G decision from AT-W7PERF, because all-visible rows of consecutive causal queries share almost their whole union (P2-9). Vulkan: G = 1 | C1 d3, C13; O12; P2-9 |
| D4 | **Dense shortcut (n_kv ≤ top_k + r − 1 = 2051) is a measured backend policy `prefer_dense(T_k, T_v, n_q_band, n_kv)`.** Default: f16/f16 keeps the existing dense causal FLASH_ATTN_EXT (no src[5]); every other pair runs the sparse op with all-visible rows. A pre-registered rule flips a (pair, n_q band) to dense on a backend where its sparse-in-shortcut cell is below 0.98 × dense (AT-W7PERF on HIP with n_q bands {1, 2–8, 9–32, 33–127, 128–511, 512–1023, ≥ 1024}; AT-B4 on Vulkan). No pair is assumed at parity in a band nobody measured: the nearest data point to a cliff is q8_0 n_q 128 masked all-visible G1 7.44/14.98/25.96 ms vs dense 2.13/3.21/5.41 at n_kv 512/1024/2048 (MEASUREMENTS.md:49) | C1 d2; R1-AV (quality: Q11a/Q11b pass on every pair); O4, O23; P2-9 |
| D5 | **Past the shortcut, no masked attention route on QSA layers, for any type, backend, ubatch kind, `-fa` or `-nkvo` setting.** Masked FLASH_ATTN_EXT with src[5] is deleted for QSA | C1 d1, C15, C16, C11; O5, O34 |
| D6 | **Two selection producers, each proven against PR219:** compact visibility → complete-block (CB) + tail (as today); **non-compact visibility → PR219's token-expanded set `top_k(expanded, 2051)`** with cells the kq_mask hides set to −1. TE is deleted as an *attention route*, kept as the non-compact *row producer* | C4 d1–d2 (CB = REF on single-sequence ranked layouts); R1-CB (CB on unified multi-sequence layouts drops forced cells: KLD 5.7–9.4e-2 text, 0.17–0.18 image, wrong free answer); O2 |
| D7 | **Compact (I32-limit) visibility extended to 2-D ubatches whose stream holds exactly one sequence, with a predicate that is total by construction for that case (rank-based limits), and checked per stream for split streams.** "One sequence" means **the stream's cell array holds cells of at most one sequence** (`n_seq_present ≤ 1`, the condition under which the scan ranks by (pos, y, x), PR llama-memory-hybrid-idx.cpp:677-683, :755), not "the ubatch carries one sequence". Predicate and scan derive it from **one shared function** `qsa_stream_one_seq(cells)`, so they cannot diverge. The non-compact scorer remains for any ubatch of a sequence with 2-D cells while another sequence has cells in the same unified stream, and for blocks split across sequence sets; there `strip_views` stays load-bearing | C15 decision (a); C4; R1-RES (split-stream predicate uses other streams' sequence ids, prefix.h:18-27, :35; the unified control `unified_img_deep_q8` is non-compact for the image slot with a text-only second slot); R1-CB (CB over unranked groups: 1035 of 1036 image cells lost); O3; P2-1 |
| D8 | **Loader traits (HIP G4):** K = direct for f16, split-register (`v_permlanex16`) for q8_0/q4_0/bf16. V: **one per-type pack-limit trait.** q8_0/q4_0: packed iff n_kv < 64·n_q, else direct split-gather. f16/bf16: **default CAPPED** (packed iff n_kv < 64·n_q, else direct `Vf16S`, PR219's behaviour); the alternative "always packed" is a one-line trait change (OWNER decision deferred, §16.5 AT-O) | C18 d1, C6 d3; OWNER; R1-C (always-pack is +512 / +1024 MiB at c524288 / c1M); O43 |
| D9 | **Loader traits (HIP G1):** V = LDS-staged transposed (wave-private slot) for every type. K = direct for f16/q8_0 (#219's decode q8 K loader as it is), split for q4_0/bf16 | C18 d2 |
| D10 | **bf16 = convert to f16 at load/stage with to_fp16 semantics** (RNE, ±inf on overflow), then the f16 compute path. No bf16 WMMA instance | C14 decision, C18 d5 |
| D11 | **Grid order:** the PR's (kv head on blockIdx.y). No kv-head interleave by default. No V prefetch in this PR | C6 d2, d4 |
| D12 | **No `ctx.pool()` in any sparse kernel.** Every workspace is in the reserved compute buffer (memory accounting, not a stall fix) | C7 decision |
| D13 | **Rotation stays mathematically where it is** (K/V stored rotated iff quantized). Its computation moves into the op (`rot_q` prologue, `rot_o` epilogue), so the HIP gate fusion matches for quantized caches too. Own work item (W10) with its own rollback | L45, L46; C12 (q8_0: 0 `qsa_prefill_gated`, f16: 176); review §2. Benefit UNPROVEN (U6) |
| D14 | **Indexer cache `auto`, qwen4exp only:** sub-8-bit -ctk → q8_0, f16 → f16, q8_0 → q8_0. GLM5_NEXT keeps `type_k`. `--cache-type-k-idx` ships as an override only | C9 d1, d4, d5; R1-E (q8 idx / f16 idx = 0.999802 on kind 1); O26 |
| D15 | **Placement guard instead of the probe; explicit pins for host KV.** supports_op is static. When a QSA layer's K buffer is a host buffer (`-nkvo`, `-ngl 0`, partial `-ngl`), llama pins `SPARSE_ATTN_PLAN`, `SPARSE_ATTN` and the QSA-layer shortcut attention node to `backend_cpu` (the shortcut pin only if AT-W7PERF (d) keeps it, §10). The guard runs on **split_only** reserve graphs (placements survive) and after every `sched_alloc_graph`; `ggml_backend_sched_reserve_size` stops resetting user pins (fix of ggml-backend.cpp:2296, §7.1.7) | L14, L15, L29, C2 d3; R1-NKVO (unpinned: 300/300 FA on ROCm0; pinned plan+attend: CPU 228/240, the other 12 = fit probe after reset, ggml-backend.cpp:2296); R2-RES (reset redundant: graph_reserve resets at llama-context.cpp:2985, only caller :3026); O5, O24, O34; P2-12, P2-13 |
| D16 | **MTP draft uses the same op, policies, pins and guard.** The draft KV type stays independent of -ctk (default f16) | C2 d1–d4, L47–L49 |
| D17 | **Reserve covers every route by construction, and a violation is an error, never an unreserved graph.** (1) `get_alloc_size` of **both** ops = max over every G the policy can return (and, for the attend, every n_q' ≤ n_q) at that n_kv. (2) Reserve passes A/B. (3) A non-compact variant whenever `n_seq_max > 1` (unified or split), with **T = min(n_ubatch, S × 512) tokens in total, S = 1**. (4) Non-compact ubatches longer than T are re-split before placement by the NC0 hook in `llama_kv_cache::prepare` with a deterministic replay in `llama_memory_hybrid_idx::init_batch` (§4.3), in the main and MTP contexts. (5) An **always-on graph-build check** (no env gate) makes `llama_decode` return an error for any non-compact ubatch with n_tokens > T (and in single-sequence contexts for any non-compact ubatch). (6) The `-fit` probe takes the per-backend max over every reserve graph. (7) `LLAMA_STRICT_RESERVE` is a validation check on top | C15 decision; R1-RES; R1-C (G1 partial non-monotone in n_q; max 12.42 MiB at n_q 263); R2-RES (mechanism by code; S = 2 ceiling exceeded at unified c131072 ub1024, +940.54 > 886.25 MiB → S = 1; probe fix exact in 64/64); O3, O20, O35; P2-2, P2-13, P2-14, P2-15 |
| D18 | **Quality gates:** Q5 (op-local: plan bytes exact, attend NMSE vs the CPU contract within measured per-class bands at k = 2, band selected from the census (G, splits), measured on PR-verbatim kernels only) is **primary for kernel-local changes**. Pinned e2e KLD (Q1/Q3/Q11) is a coarse screen with pre-registered bands taken from PR219's own route changes, **floored so that no band is 0** (§15), never "≤ 3·SE_paired". Co-gates: selection agreement (Q4), images (Q6), MTP (Q7), planted defects (Q8), non-compact multi-sequence (Q10), host KV (Q12). A failed band gate gets one pre-registered confirmation stage on held-out data (§15) | R1-Q8 (3·SE fails a benign f16 reorder at t 9.40 and passes D2; pinned KLD saturates near 1e-4); R1-UB (bands); R1-A, R2-BOP (k = 2 bands); O6–O11, O27, O30, O38; P2-3–P2-8, P2-20 |
| D19 | **Perf gates are non-inferiority tests against a reference run in the same interleaved campaign**, on real text (C10 corpus), with the mean of reps (rep 0 dropped), a hierarchical bootstrap CI, KFD-eviction filtering, a margin δ per cell and a terminating stopping rule. Random tokens are a stress row only | C10 d1, C7; skeptic re-bootstrap of C10 raw logs (O15: the median is bimodal, the mean at 2× resolves ±0.7–0.95 %); O15, O39 |
| D20 | **Vulkan:** a KHR-coopmat f16 kernel derived from `flash_attn_sel_cm.comp`, K/V types through `flash_attn_dequant.glsl`, bf16 converted at staging, **with a direct-fragment and an LDS-staged V variant for quantized V, and a diagnostic dequant-once pack arm (shipped only under M1), the fastest shipped per type (AT-B3)**. A **portable tier** with no subgroup-size, shuffle or 46 KB shared-memory requirement (workgroup reductions in shared memory, subgroup size a spec constant) for every device that cannot create the coopmat or 64-lane pipelines; supports_op is true only where a pipeline was created. No coopmat2. G = 1. Over-allocated sparse nodes get an unconditional `ggml_vk_sync_buffers` before their first dispatch, and their workspace is bound through an explicit sub-buffer covering [round_up(nbytes, minStorageBufferOffsetAlignment), alloc_size) | C11 d1–d4, C14 d4; C18 (HIP direct V 1.23–1.67× staged); master ggml-vulkan.cpp:4800-4816 (= PR :4735-4752: sel pipelines need sg 64 + shuffle + arithmetic + fp16 + 46,720 B shared), master :7028-7053 (= PR :6963-6989: `ggml_vk_tensor_subbuffer` range = ggml_nbytes); O14, O17; P2-11, P2-16, P2-19 |
| D21 | **`-fa off` keeps a QSA route.** The hybrid-idx cache of qwen4exp is created with `v_trans = false` whatever `-fa` says; QSA layers always run the sparse op past the shortcut; the f16/f16 shortcut under `-fa off` runs the existing non-FA dense causal `build_attn_mha`, which already handles non-transposed V by `ggml_cont(ggml_transpose(v))` (PR llama-graph.cpp, non-FA branch of `build_attn_mha`). Eligibility no longer tests `cparams.flash_attn` or `cparams.offload_kqv`. This applies to f16/bf16/f32 V only: `-fa off` with a quantized V is already an init error (llama-context.cpp:663). `v_trans = false` is cache-wide; non-QSA attention layers in the same cache (ratio ≤ 1, e.g. a pre-#251 GGUF whose ratios are all 0, L13) then take the `ggml_cont(ggml_transpose(v))` branch of `build_attn_mha` per layer under `-fa off` (PR llama-graph.cpp:3127-3130), a perf cost listed in §19 | PR llama-model.cpp:3022 (`attn_v_trans = !cparams.flash_attn`), PR qwen4exp.cpp:1109-1112; O34 |
| D22 | **Backends without a SPARSE_ATTN kernel (OWNER decision 2026-11-05: deletion, option b):** QSA on CUDA, MUSA, Metal, SYCL, OpenCL, CANN, WebGPU and on Vulkan devices where no tier's pipeline can be created becomes an init error whose message names the real remedies (host KV, or a HIP/Vulkan build), listed by name in §19; the alternative (keep the masked route as a logged backstop on those backends only) is offered to the owner in §20.6 | PR qwen4exp.cpp:1743-1777 (masked route is backend-agnostic); PR ggml-metal-device.m:1743-1790 (Metal FA supports_op does not look at src[5]: under `-fa on` today Metal probably computes a maskless FA + src[5] as dense, UNPROVEN); P2-17 |

---

## 2. Op contracts (ggml/include/ggml.h)

The two enum values are **appended right before `GGML_OP_COUNT`**, so no existing op value moves; both `GGML_OP_NAME`/`GGML_OP_SYMBOL` static asserts (PR ggml.c:1223, :1338) and `ggml/include/ggml-rpc.h:14` (`static_assert(GGML_OP_COUNT == 101 …)`) are updated, and **`RPC_PROTO_MINOR_VERSION` is bumped** (O21). Reason: the RPC hello checks only major/minor (ggml-rpc.cpp:432) and deserialize does no op validation (:1687 `result->op = (ggml_op) tensor->op`); inserting after `GGML_OP_LIGHTNING_INDEXER` (PR ggml.h:631) with a patch bump would shift every later op (incl. UNARY, GLU) by 2 between a new client and an old server, for every model. The PR text states the minor bump (dual-Strix-Halo RPC users, #230).

### 2.1 `ggml_sparse_attn_plan`

```c
// Canonicalise per-query selection rows into a grouped attention plan. Integer only; CPU, HIP and Vulkan produce
// identical dst bytes (every byte of ggml_nbytes(dst)). S(i,s) = { c : exists j, sel[j,i,0,s] == c, 0 <= c < n_kv }
// (set semantics: order, duplicates, -1 and out-of-range entries are irrelevant; out-of-range values other than -1
// are counted in hdr.n_bad).
GGML_API struct ggml_tensor * ggml_sparse_attn_plan(
        struct ggml_context * ctx,
        struct ggml_tensor  * sel,      // I32 [n_sel, n_q, 1, n_stream]; nb[0] == 4; nb[1..3] % 4 == 0; n_sel >= 1
        int64_t               n_kv,     // keys visible to the op (cells 0..n_kv-1 of the K/V views); n_kv <= 2^24
        int32_t               group);   // G in {1, 4}: consecutive queries of one stream sharing one union
// dst: I8 [ggml_sparse_attn_plan_nbytes(n_sel, n_q, n_stream, n_kv, group)], layout below
GGML_API size_t ggml_sparse_attn_plan_nbytes(int64_t n_sel, int64_t n_q, int64_t n_stream, int64_t n_kv, int32_t group);
// scratch (not part of dst): ggml_sparse_attn_plan_scratch_nbytes(n_sel, n_q, n_stream) = n_stream * n_q * round4(n_sel) * 4
GGML_API size_t ggml_sparse_attn_plan_scratch_nbytes(int64_t n_sel, int64_t n_q, int64_t n_stream);
```

**Layout of dst** (little-endian; offsets 256-byte aligned; every byte defined):

```
hdr    { u32 magic 'QSAP', version 1, G, n_q, n_stream, n_groups, cap, n_bad; u64 off_counts, off_quads, off_masks; u64 reserved = 0 }
counts u32 [n_stream][n_groups]           // entries used, padded to a multiple of 4 (4 entries = one 16-key WMMA chunk)
quads  u32 [n_stream][n_groups][cap]      // physical quad id = cell >> 2, strictly ascending within a group
masks  u16 [n_stream][n_groups][cap]      // bit 4*j + c: query j of the group attends cell 4*quad + c (G=1 uses bits 0..3)
n_groups = ceil(n_q / G);  cap = round4(min(G * n_sel, ceil(n_kv / 4)))
entries past counts[s][g] and alignment gaps: quad = last valid quad of the group (or 0), mask = 0
```

- **Scratch outside dst (O1a, O29).** The sort/dedup rows scratch is not in `ggml_nbytes(dst)`:
  - HIP and Vulkan: `get_alloc_size(dst) = max over G' ∈ G_range of ggml_sparse_attn_plan_nbytes(n_sel, n_q, n_stream, n_kv, G') + scratch_nbytes` (+ the Vulkan alignment round-up, §6.3), through the existing HIP hook pattern (ggml-cuda.cu:1020-1024 → fattn.cu:751, C3 code verification) and the new Vulkan hook (§6.3). G_range = every G the backend's `group` policy can return ({1, 4} on HIP, {1} on Vulkan). Reason (P2-15): nbytes(dst) depends on G, and when n_q % 4 ≠ 0 the G4 dst is larger than the G1 dst (n_q 127, n_sel 2051, deep n_kv: G1 1,564,408 B vs G4 1,575,680 B with 256 B section alignment [arithmetic]); once AT-A makes G depend on n_kv, a runtime ubatch can get a G whose dst exceeds the reserved size_max, and gallocr re-plans (ggml-alloc.c:1281-1291, :1336-1350; ggml-backend.cpp:1945-1975). The dst tensor itself keeps the exact `plan_nbytes` of its own G (whole-dst bit-exact contract unchanged); only the allocation is the max.
  - CPU: the graph work buffer (`ggml_graph_plan` work-size switch, PR ggml-cpu.c:2452 / :2784).
  - `GGML_OP_SPARSE_ATTN_PLAN` and `GGML_OP_SPARSE_ATTN` are added to `ggml_op_alloc_size_may_expand` (PR ggml-backend.cpp:2560-2573); otherwise the `GGML_ASSERT(size <= ggml_nbytes(tensor) || … )` at PR ggml-backend.cpp:199-206 aborts on the first over-allocation. ggml-rpc.cpp:1044 keys its remote alloc-size query on the same list.
  - Consequence: test-backend-ops compares the whole dst (`tensor_to_float` over `ggml_nbytes`, PR test-backend-ops.cpp:1611-1645), and "bit-exact" is literally a whole-dst equality (§14).
- **Why physical quads and this cap.** Correct under permuted cells and holes. `G·n_sel` is the true worst case when every selected cell lands in its own quad (C5: qpb 1.001 in fresh caches; permuted layouts UNPROVEN, U2b). At ub4096, G4, n_sel 2051: cap 8204, the PR's own cap (PR qsa-prefill.cu:702).
- **Size at ub4096, G4** [arithmetic]: dst (quads + masks + counts + hdr) 1024 × 8204 × 6 B ≈ 50.4 MB; scratch 4096 × 2052 × 4 B = 33.6 MB; total 84 MB, independent of n_kv, the PR's in-op pool use (C3).
- **G1, n_q ≤ 127:** about 127 × 2052 × 10 B = 2.6 MB.
- **Invariants.**
  - The dst has exactly one consumer, a `GGML_OP_SPARSE_ATTN`; the llama guard asserts it (§10). C3 (b) proves single-consumer reuse through gallocr, 2/3-split schedules and HIP graph capture.
  - The plan is never written by its consumer (C3 d2).
  - `ggml_op_can_inplace` must not list either op (ggml-alloc.c:22-50, C3).

### 2.2 `ggml_sparse_attn`

```c
// out[:,h,i,s] = H_o( sum_{c in S(i,s)} softmax_c( f16(scale * H_q q[:,i,h,s]) . f16(k[:,c,h/r,s]) ) * f16(v[:,c,h/r,s]) )
//   r = n_head / n_head_kv; H_q / H_o: orthonormal Sylvester-Hadamard over consecutive blocks of rot_q / rot_o dims
//   (0 = identity), the transform llama_mul_mat_hadamard applies today; empty S -> 0.
GGML_API struct ggml_tensor * ggml_sparse_attn(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,      // F32 [D, n_q, n_head, n_stream]        nb[0] == 4, any other strides (FA layout)
        struct ggml_tensor  * k,      // T_k [D, n_kv_k, n_head_kv, n_stream]   cache view, stored rotated if quantized
        struct ggml_tensor  * v,      // T_v [D, n_kv_v, n_head_kv, n_stream]   not transposed
        struct ggml_tensor  * plan,   // ggml_sparse_attn_plan(..., n_kv, G) with the same n_q, n_stream
        float                 scale,
        int32_t               rot_q,  // 0, 64, 128 or 256, dividing D
        int32_t               rot_o); // 0, 64, 128 or 256, dividing D
// dst: F32 [D, n_head, n_q, n_stream]  (the FLASH_ATTN_EXT output layout, so wo and the gate graph are unchanged)
```

- **Build-time asserts** (ggml.c):
  - n_head % n_head_kv == 0; D equal for q, k and v; rot sizes as listed;
  - plan->src[0] is a `sel` with ne[1] == q->ne[1] and ne[3] == q->ne[3];
  - **bounds (O1b): `k->ne[1] >= 4·ceil(n_kv/4)` and `v->ne[1] >= 4·ceil(n_kv/4)`**, because a quad load reads 4 cells. Production K/V views have n_kv padded to a multiple of 256 (llama-kv-cache.cpp:1347-1352), so the assert holds; the PR kernels relied on round4 (PR qwen4exp.cpp:1103 static_assert). Op tests build n_kv = 2051 with views of 2052 rows ending exactly at the end of their buffer (§14);
  - T_k and T_v are any ggml types; the backends restrict them (§5).
- **Masked-off and padding cells (O31).** A cell whose mask bit is 0, and every padding entry, contributes exactly nothing to the output **even if its stored bytes are NaN or ±inf** (including a NaN/inf q8_0/q4_0 scale d). Kernels must select (score −inf, V row zeroed by select) rather than multiply by 0, because 0·NaN = NaN in WMMA. The existing kernels already meet this on the base `poison` test (PR test-backend-ops.cpp:11410-11450, authored by Piotr Wilkin per `git blame`, not #219), which §14 ports to every S_fast type.
- **Numeric contract** (documented in ggml.h; the CPU reference implements it as the oracle O_CON of §3; O30):
  1. K and V elements are converted to f16 with ggml-cuda `ggml_get_to_fp16_cuda(T)` semantics: RNE, ±inf on overflow. Bit equality with HIP's conversion is claimed only for f16, bf16, q8_0 and q4_0 (for q8_0, `(half)(q·d)`, bit-equal to `dequantize_block_q8_0_f16`, L8, C18: 4905 + 4320 bitwise checks). For q4_1/q5_1 (d·q + m) and the Vulkan `flash_attn_dequant.glsl` path (FLOAT_TYPE arithmetic, L41), the contract is met within the §14 bands, not bitwise.
  2. q is transformed by H_q in f32 when rot_q > 0, multiplied by `scale` **in f32**, and rounded to f16. (The PR kernels compute f16(q)·f16(scale), identical only while scale is a power of two, 1/16 for this model; the new kernels apply the f32 product.)
  3. Scores are f32, summed sequentially over d in the oracle. Online softmax: P is rounded to f16 against the running max for PV, while the normaliser l sums the unrounded f32 p (the kernel rule, c18_kernel.inc:559-562). PV accumulates in f32. acc / l and H_o run in f32.
  4. The summation order is a pure function of (shape, G, V-direct/packed choice, split count, device) **and of the op's inputs**; cells are visited in the plan's ascending quad order, so duplicates in `sel` cannot change it (the plan dedups deterministically, §3). Output is deterministic and memcmp-stable across runs for a fixed (ub, b) and a fixed allocator placement (R1-A noise: r1 = r2, 1213 identical lines; R1-UB: 11 self-pairs exactly 0). The placement matters because the gate-fusion overlap check (ggml-cuda.cu:6776-6785) can refuse fusion depending on where gallocr puts the MUL output, and fused and unfused outputs are not claimed bit-equal. QPIN self-replay therefore uses identical ub/b. This determinism is a property of the op given its inputs; the TE rows producer on non-compact ubatches (`top_k` tie order on GPU) is not claimed deterministic (R1-CB names GPU `top_k` tie order as the candidate cause of PR219's masked-route nondeterminism). Contrast: PR219's decode kernel is **not** deterministic on rows with duplicate keys (atomicCAS dedup race, PR qsa-decode.cu:107-125; R2-BOP: up to 18 distinct unique-id sets in 6 runs, output NMSE up to 6.9e-11).
- **Alignment.** supports_op checks `nb[1..3] % 16 == 0` for K and V (every S type's 256-dim row is a multiple of 16 B: f16 512, q8_0 272, q4_0 144, q4_1 160, q5_0 176, q5_1 192, iq4_nl 144, bf16 512, f32 1024 [arithmetic from ggml-common.h block sizes]). compute aborts only on a misaligned base pointer. The PR never checked V alignment (L31).

### 2.3 Existing FLASH_ATTN_EXT + src[5]

- `ggml_flash_attn_ext_add_top_k` stays for DSA (Vulkan `flash_attn_top_k`, D512/MQA, L39).
- It gains `GGML_ASSERT(a->src[3] != NULL)`, so a maskless FA + src[5] can no longer be built (W13).

---

## 3. CPU reference semantics (ggml/src/ggml-cpu/ops.cpp)

- **`ggml_compute_forward_sparse_attn_plan`.** Threading (O24): phase 1 splits the n_stream·n_q query rows over `nth` threads (collect valid cells, sort ascending, dedup, map to (quad, 4-bit mask)) into the work-buffer scratch; a barrier; phase 2 splits the n_stream·n_groups groups over `nth` (k-way merge by quad into strictly ascending entries, OR-ing masks shifted by 4·j, pad counts to a multiple of 4 with mask-0 entries, count bad values). Output: the whole dst, byte-identical to HIP and Vulkan. The algorithm is the `c3_plan.cu` `std::map`-per-group reference (EV/c3/c3_plan.cu); a CPU unit test (`tests/test-sparse-attn-plan-oracle.cpp`) memcmps the op against that oracle on the C3 captured rows (§14).
- **`ggml_compute_forward_sparse_attn`**, threaded over **(s, i, kv head)** work items split across `nth`, the r = n_head / n_head_kv query heads of one kv head in one item, implements **O_CON** (R1-A Setup):
  1. FWHT q over rot_q blocks in f32 (butterfly order of ggml-cpu's hinted mul_mat); ×scale in f32; round to f16.
  2. Walk the group's entries whose mask has this query's bits.
  3. Dequantize each visited K/V cell **once per kv head** with `to_float` of the type, round to f16, and use it for all r heads (the PR CPU FA dequantizes a non-f16 V row once per (query, head), ops.cpp:9134-9135, which is the plausible structural cause of R1-NKVO's q8_0-vs-f16 asymmetry; code reading, not measured).
  4. f32 scores, each summed sequentially over d, with the SIMD lanes across the r heads (`for d: kd = k[d]; for h < r: s[h] += qT[d][h]·kd`, the loop of `EV/bands/src/bands_ref.h:239`); per-cell online softmax with `expf`; P → f16 for PV against the running max; l sums the unrounded f32 p.
  5. f32 PV; acc / l; FWHT over rot_o in f32.
  - The TU is compiled with `-ffp-contract=off` (as the bands build is, `EV/bands/src/bands_cpu.cpp:8`), because GCC's GNU-mode default `fp-contract=fast` may fuse `s += q·k` into an FMA and break bitwise equality with `bands_ref.h` (UNPROVEN either way; W1 checks it bitwise).
  - Vertical SIMD keeps the contract order. Skeptic microbenchmark (`EV/skept_ocon/mb.cpp`, `g++ -O3 -march=native -ffp-contract=off`, one thread, i7-1370P AVX2, QK of one (query, kv head) at n_sel 2051, D 256, 12 heads): the bands_ref.h:239 loop 0.272 ms against 0.312 ms for a W0-style horizontal F16C dot; outputs agree within max|diff| 3.1e-6. Full-op CPU speed is gated by AT-D2/AT-D4.
- **Roles.** It is the oracle, and the production backend when a QSA layer's KV is on the CPU (`-nkvo`, `-ngl 0`, partial `-ngl`), with the D15 pins. Its graph work buffer (plan scratch 33.6 MB at ub4096, plus attend scratch) is host memory grown by `ggml_graph_plan` at compute time, outside the sched buffers, so `LLAMA_STRICT_RESERVE` does not cover it; the init log line states its size for host-KV contexts.
- **Oracles in the test driver** (R1-A Setup, `EV/bands/src/bands_ref.h`):
  - **O_CON** = the CPU op itself;
  - **O_C18** (f64): q = f16(f16(q)·f16(scale)), K/V = f16(to_float(x)), bf16 saturated to ±65504 first (c18_attn.cu:290). This saturation differs from the §2.2 contract (±inf on overflow); the bands never exercised overflow, so the overflow case of §14.1 is checked against O_CON only, and W1 verifies that `bands_ref.h` O_CON rounds bf16 overflow to ±inf (UNPROVEN until W1);
  - **O_PLAN** (f64): q·scale in f32, K/V = to_float(x) with no f16 rounding.
  - Gated: GPU vs O_CON (§14 bands) and O_CON vs O_C18 / O_PLAN (W1). Reported only: GPU vs O_C18 / O_PLAN max|err|/max|ref| per shape and type (R1-A: clean kernels reach 2.75e-3 vs O_PLAN, so the old 4e-4 cannot be a gate).

---

## 4. Producers (src/models/qwen4exp.cpp, src/llama-memory-hybrid-idx.*, src/prefix.h)

### 4.1 Routes, per QSA layer and ubatch

| case | rows producer | attention node |
|---|---|---|
| n_kv ≤ top_k + r − 1 and `prefer_dense(dev, T_k, T_v, n_q, n_kv)` (default: K = V = f16) | none (indexer keys stored only) | existing dense causal FLASH_ATTN_EXT (no src[5]), fa7 + gate fusion; under `-fa off`, the existing non-FA dense `build_attn_mha` (D21) |
| n_kv ≤ top_k + r − 1, otherwise | **all-visible rows `qsa_rows_av-<il>`**, built over the whole mask `[n_kv, n_tps, 1, n_stream]`: `cast(m, F32) → step(scale_bias(·, 1, 1)) → mul(·, arange(0, n_kv) + 1) → scale_bias(·, 1, −1) → cast I32`, with `GGML_ASSERT(m->ne[2] == 1 && m->ne[1]·m->ne[3] == n_tokens)`; output `[n_kv, n_tps, 1, n_stream]` (the `ALLVIS=3` hunk of `EV/r1av/r1av_full.diff`) | sparse plan + attend |
| compact (scalar) visibility | **CB + tail**, as today (PR qwen4exp.cpp:1182-1206), including the fused HIP `qsa_expand` matcher (expand.cu, ggml-cuda.cu:4782) | sparse plan + attend |
| non-compact visibility (§4.3) | **PR219's TE set**: `top_k(expanded, 2051)` exactly as PR qwen4exp.cpp:1658-1674 builds it, then cells hidden by the kq_mask set to −1. Mechanism: `expanded` is already score + bias + kq_mask (PR qwen4exp.cpp:1666-1668), so a gather of `expanded` at the selected indices (`ggml_get_rows` with `expanded` viewed as [1, n_kv, n_tps·n_stream] rows and the indices as [n_sel, n_tps·n_stream], i.e. one row per (query, stream)) followed by `select(value == −inf → −1, index)` gives exactly TE ∩ visible with no second mask read, for split streams too. Rows go unsorted to the plan, whose sort handles them (set semantics; about 2.5 ms/layer, non-compact ubatches only, C3 d4) | sparse plan + attend |

- **The attended set on non-compact ubatches equals PR219's** (TE ∩ visible), so new vs PR219 differs only by the op's numerics; Q10 checks it (R1-CB decision 2).
- **Why not CB there (R1-CB).** Two mechanisms in PR llama-memory-hybrid-idx.cpp: (M1) incomplete groups go to a spare block with +1e9 (:796, :845, :982), TE keeps every visible forced cell, CB keeps 3 by tie order; (M2) with 2 sequences in the cache (any second sequence, text-only included: the scan ranks only when `n_seq_present ≤ 1`, :677-683, :755) an image falls into one 1039-cell "complete" group while `blk_cells` keeps 4 slots. Measured: text seq_cp prefix L = 8193–8195, KLD te vs cb 5.67e-2 to 9.40e-2 against noise ≤ 2.28e-2; image slot KLD 0.174–0.182 against noise ≤ 3.44e-3; the free answer goes from correct to wrong on every field. A future CB rule for these layouts must keep every visible forced cell (n_sel = 2048 + |forced|, measured |forced| ≤ 7 text, ≤ 1042 image) and expand oversized groups through `cell_blk`; it is not in this PR (U2).
- **Producer under split streams (O4).** The 4-D construction is the only accepted form: the 2-D view of the C1 lab aborts at ggml.c:5758 (`GGML_ASSERT(top_k->ne[1] == a->src[0]->ne[1])`) for np2 (R1-AV, rc 134 for q8_0 and q4_0); the 4-D form gives np2 = np1 exactly in 32/32 sequences and bit-identical rows to the 2-D form at np1 (R1-AV). This tree has no KQ_MASK_PAD rows (PR llama-graph.cpp:43).
- **GPU coverage the shortcut implies (R1-AV decision 4).** Every pair routed to the sparse op in the shortcut must have a GPU kernel at every n_q and n_stream: q4_0 and bf16 at any n_q, mixed pairs at n_q < 128, split streams at n_tps < 128 (PR219 has none of these: qsa-decode.cu:91-92, :97; the PR's maskless op falls back to the CPU in the R1-AV lab for q4_0/bf16). The §10 guard and the CPU-launch counter make a miss an error, never a fallback.
- **`-fa`.** The sparse route ignores `-fa` (D21). Alibi or soft-cap on a QSA layer is a model-load error (L18).
- **Non-causal attention.** `cparams.causal_attn == false` on a QSA layer returns an error from `llama_decode` with a message (today it takes the scan/masked path, PR llama-memory-hybrid-idx.cpp:446-447). mtmd only switches to non-causal for Gemma projectors (PR tools/mtmd/mtmd.cpp:2326-2340), so no Qwen-VL flow reaches it; any other caller gets the error.

### 4.2 Compact visibility predicate, extended (D7)

- **Today.** `qsa_scalar_visibility_cells` (PR src/prefix.h:15-60) rejects any ubatch whose own queries have 2-D positions; on qwen4exp every ubatch is 2-D (n_pos ≥ 3, IMROPE unconditional: PR llama-batch.h:50-53, llama-model.cpp:3788-3790, llama-hparams.cpp:285-287). Eligibility also requires `ubatch.token` (L18, qwen4exp.cpp:1110). The split-stream check passes the whole ubatch's sequence ids for every stream (PR llama-memory-hybrid-idx.cpp:1719-1738, prefix.h:18-27), so `one_seq` (prefix.h:35) is false whenever a ubatch carries two slots (R1-RES).
- **Change.**
  1. Accept embedding ubatches.
  2. **Per-stream sequence set:** each stream is checked with only the sequence ids of its own tokens.
  3. **2-D rows in a one-sequence stream (P2-1).** The compact predicate for a ubatch whose sequences hold 2-D cells holds **iff the stream's cell array holds cells of exactly one sequence** (`qsa_stream_one_seq(cells) := n_seq_present ≤ 1` over the stream's non-empty cells; the same condition that makes the scan rank, PR llama-memory-hybrid-idx.cpp:677-683, :755). Both the predicate (prefix.h) and the scan call this one function; a unit-test assert checks that for every ubatch the scan's "ranked" branch was taken iff the predicate accepted it (the two cannot diverge). Not "the ubatch carries one sequence": in the PR code `one_seq` requires every non-empty cell's sequence set to equal `rows` and returns false when `pos_2d` is non-empty and any other sequence is present (PR prefix.h:34-46); R1-RES's unified control (`unified_img_deep_q8`, only the S0_img slot has an image, T_deep is text) shows the image slot's ~50 ubatches after the second slot got cells all `scalar=0` (n_kv 66816 → 114176, `~/dev/wt_lab/r1s/logs/srv_unified_img_deep_q8.log` from line 3133), while every text-slot ubatch was compact. For one sequence the kq_mask makes cell j visible to query q iff key(j) ≤lex key(q) on (pos, y, x) (PR llama-kv-cache.cpp:1792-1807, `is_2d_gt` in llama-kv-cells.h:23). The visible set is therefore a prefix of that order whatever the physical cell order, and the compact limits are **rank-based**: block-start / tail-start values in the (pos, y, x) rank order that `qsa_rebuild_prefix` already builds by counting sort (prefix.h:83-118; the qsa_prefix ranking at llama-memory-hybrid-idx.cpp:755-785). Not "cell index + 1".
  4. The predicate is total for one sequence except on degenerate input (pos ≥ 2^24, a duplicate (pos, y, x) key); those return a `llama_decode` error naming the layout. **Which APIs can create a duplicate key** is listed at W9 by code reading of `llama_memory_seq_div` (grp-attn), `seq_add` with ext positions not shifted, and `seq_cp`; a duplicate reachable from a supported qwen4exp flow is routed to the non-compact path (with its re-split) instead of an error (UNPROVEN until the W9 reading; U35).
  5. **Text ubatches in a multi-sequence unified stream** stay compact as today when no sequence of the ubatch holds 2-D cells (PR prefix.h:39-46: `pos_2d` empty).
- **Proof obligation.** `tests/test-qsa-visibility.cpp` (CPU) compares, for every row, the compact limits against the `set_input_kq_mask` predicate on recorded layouts (p1/p2 of C4, B65t of C19, request C of L64, the R1-RES split-stream image ubatch) and synthetic ones (several images, back to back across a ubatch boundary, image at position 0, text after images, depth > 32k, holes after seq_rm). **Negative cases (P2-1):** unified stream, seq A holding an image plus text seq B holding cells: the predicate must **reject** A's ubatches (text and image) and **accept** B's text ubatches; the same with B's cells removed by `seq_rm` (A accepted again); split streams with the same two sequences (each stream accepted). The scan/predicate agreement assert runs on every case. It sits in the `NOT WIN32 OR NOT BUILD_SHARED_LIBS` block of tests/CMakeLists.txt (it uses src/ internals; O37).
- **Image ubatches and the fast cache.** Image positions repeat one linear position, and `qsa_apply` rejects `!u.token` (PR llama-memory-hybrid-idx.cpp:1757-1760), so every image ubatch invalidates the incremental prefix and forces `qsa_recover`. Reserve pass A (§7.1) and the fast-cache inputs must therefore be valid on the ubatch after an image; W9 tests it.

### 4.3 Non-compact visibility (residual)

- **Remaining triggers (P2-1):**
  - **any ubatch of a sequence that holds 2-D cells while another sequence has cells in the same unified stream** (text ubatches of that sequence included). One image slot is enough: R1-RES's unified control (S0_img + text T_deep) is non-compact for the image slot's ubatches (§4.2 step 3). On the default server (4 slots, unified) every prompt of an image-bearing slot is non-compact as soon as another slot holds cells;
  - a block shared by two sequence sets (`seq_cp` inside a block: server-context.cpp:804-816);
  - SWA is not a trigger for this model: n_swa == 0 in the scalar condition (L19) and the model's GGUF has no SWA layers (UNPROVEN beyond L19; U26).
  - Not a trigger: a text-only ubatch whose sequences hold no 2-D cells in a multi-sequence unified stream (PR prefix.h:39-46), and any ubatch in a stream whose cells belong to one sequence.
- **What runs.** Today's non-compact scorer (`blk_bias` f32 [n_blocks, n_tps], kq_mask strip views) with **`strip_views`** (one view per (input, first query, strip size), shared by all layers; 68da4191d + b60447506, L6), producing the TE rows of §4.1.
- **Bounded size: the re-split mechanism (O35, P2-2, P2-14; R2-RES by code, PR tree 94e086847).**
  - **Cap.** U = min(n_ubatch, S × 512) **tokens in total per ubatch** (not per stream: split_equal's n_keep = 0 branch can put up to n_ubatch' tokens on one stream, PR llama-batch.cpp:787-799), with **S = 1** (`QSA_NONCOMPACT_STRIPS = 1`). S = 2 was the round-1 choice; its pre-registered device ceiling is exceeded on the unified `-c 131072 -ub 1024` server (+940.54 MiB ROCm0 main > 886.25, R2-RES), so the named fallback S = 1 applies; at S = 1 every measured configuration is under the S = 1 ceiling (R2-RES).
  - **NC0, the init-time conservative predicate.** NC0(u) is true when `n_kv(sinfo) > indexer_top_k + r − 1` and the §4.2 compact predicate (the same `qsa_stream_one_seq`-based function the graph uses) rejects u on the **simulated** cells. It drops the graph-time terms that can only turn non-compact into compact (the incremental `qsa_prefix_matches` shortcut), so NC0 ⊇ graph-time non-compact. It reads the attention cells, while the graph-time predicate reads the indexer cells (PR llama-memory-hybrid-idx.cpp:1731, :1738); the design rests on "attention cells ≡ indexer cells after every apply" (same slots :223, edits forwarded to both, e.g. seq_rm :281-300), which the unit test asserts (as a general claim UNPROVEN, U36).
  - **Hook.** `llama_kv_cache::prepare(ubatches, hook)` (PR src/llama-kv-cache.cpp:816-879) calls `hook(i, ubatch, sinfo, cells)` right after `apply_ubatch(sinfo_new, ubatch)` (:856), where the cells hold exactly the state `apply` builds at graph time (:3064-3065). The hook returns false iff NC0(u) and u.n_tokens > U; prepare records `i_fail`, stops, and restores the cells as today (:861-872).
  - **Replay** in `llama_memory_hybrid_idx::init_batch` (PR :168-231): keep a per-ubatch cap vector; loop {`split_reset`; form ubatches with `split_equal(k < cap.size() ? cap[k] : n_ubatch, !unified, n_keep_tail)` (or `split_seq` when embd_all); `attn->prepare(ubatches, hook)`; on `i_fail`, `cap[i_fail] = U`}. `split_equal` is a deterministic function of `used` (`split_reset` :622-628), so ubatches 0..i_fail−1 replay identically and ubatch i_fail then has ≤ U tokens (:755, :788); i_fail strictly increases and the loop runs at most #ubatches times. n_keep_tail is kept by split_equal's own rule for any n_ubatch' > n_keep_tail (:764-823); an init check `n_rs_seq + 1 < S × 512` replaces the :764 abort. The recurrent `prepare` (:207) moves after the accepted attention prepare and runs once on the final ubatches. A context counter `n_qsa_resplit` counts replays.
  - **Both contexts.** The main context has n_keep_tail 4 (n_rs_seq 3, common/common.cpp:1716) and the MTP context 0 (common/common.cpp:1261; logs `srv_B_u.log:2627, :6946`); this explains R1-RES's 513+511 vs 511+511 split by code (R2-RES). The same init_batch code serves both, so both are re-split; no decode-level re-split is needed (`llama_context::decode` calls init_batch once per batch, llama-context.cpp:2267).
  - **Always-on graph-build check (no env gate).** In `build_qsa_top_k`, right after the compact decision, a non-compact ubatch with n_kv > indexer_top_k + r − 1 and n_tokens > U (in single-sequence contexts: any non-compact ubatch) sets a flag on the graph result; `process_ubatch` checks it after `model.build_graph` (PR llama-context.cpp:1838-1844 area), logs n_kv, n_tps, n_stream and the layout, and returns `GGML_STATUS_FAILED`, so `llama_decode` returns an error and never allocates that graph. Reserve graphs are unaffected (pp/tg reserves compact, R2-RES logs `forced=0 scalar=1`; the variant has T ≤ U). A reused graph keeps the same compact flag (`can_reuse` compares it, PR qwen4exp.cpp:1283-1287).
  - **Image cuts.** A re-split cuts an image's token run at a U boundary; the default server already does the same at ub512 with `--image-min-tokens 1024` (R2-RES).
  - **Perf** of S = 1 falls on non-compact ubatches only (image-bearing slots in a multi-sequence unified stream, seq_cp blocks); UNPROVEN (U27), reported by AT-C12.
- **What strip_views saves (PR-tree measurements):** +1120 MiB at runtime for one deep image ubatch at `-np 1`, c65536, HIP, and 0.148 → 0.029 MiB/cell on the masked route (C19); +1077.74 MiB for one deep image and +9014.36 MiB for two image slots on the default server (R1-C layouts 2 and 3).

### 4.4 Split streams, multi-sequence, trim, MTP, > 1M

- **Split streams** (non-unified multi-slot, the default for an explicit `-np N`: common.h:649, server.cpp:168-172): the sparse ops take n_stream on ne[3] at any n_tps. Today's `(n_stream == 1 || n_tps ≥ 128)` condition (PR qwen4exp.cpp:1453-1456) and the per-stream masked tg route (C15) are deleted.
- **Shared-prefix multi-sequence, unified:** compact while the predicate holds (prefix.h:48-58), otherwise §4.3.
- **`LLAMA_FN_LAST_ATTN_TRIM`** stays (C12; PPL identical on/off). Trim applies to the **compact CB** route only (today's condition keys on `maskless`, PR qwen4exp.cpp:1883, and the shortcut never trims, :1876-1878); the TE route trims only together with slicing its mask rows from q_first; the all-visible shortcut never trims (its producer asserts m->ne[1]·ne[3] == n_tokens, §4.1, which conflicts with the sel ne[1] == q ne[1] assert of §2.2). The trimmed layer passes n_q = n_tokens − q_first ∈ [1, 512] and gets G from the policy. W7's "pp2048 d0 ub2048" cell exercises shortcut + single output.
- **MTP draft context.** Builds through `build_layer_attn` with the hybrid-idx memory (PR qwen4exp.cpp:495-541, L47): same producers, ops, policies, pins and guard. Routes nq 1, nq 2–8, nq ≥ 128 when not cache-only (qwen4exp.cpp:491), gated prefill, shortcut (C2 d2). Draft KV `cache_type_k = F16` unless `-ctkd` (PR common/common.h:348/373; C2).
- **> 1M keys.** `QSA_MASKLESS_MAX_KV` (PR qwen4exp.cpp:1102) and `GGML_QSA_MAX_KEYS` (ggml-cuda/common.cuh:109) are deleted. Quad ids are u32; the producer's f32 casts (PR qwen4exp.cpp:1192, :1204, L21) cap exact ids at 2^24 cells. All K/V offsets in the sparse kernels are `size_t` (as PR qsa-prefill.cu:351/:432 does). **Context creation rejects `n_ctx_seq` above the largest n_kv an op test has run (§14.1: 2^21 + 4 cells, the first f16 K tensor ≥ 2 GiB, the signed-int offset boundary) with an init error** until U12 is closed; 2^24 stays the hard ceiling. 1M runtime UNPROVEN (U12).

### 4.5 Rows naming (for the quality tooling)

- The rows tensor that feeds `ggml_sparse_attn_plan` is `qsa_rows_cb-<il>` on the compact route, `qsa_rows_te-<il>` on the non-compact route, and `qsa_rows_av-<il>` on the shortcut.
- MoE ids stay at `ffn_moe_topk-<il>` (PR llama-graph.cpp:2362, L26).
- The QPIN hook matches only `qsa_rows_cb-` (kind 1), `qsa_rows_tk-` (kind 2) and `ffn_moe_topk-` (kind 3) (`~/dev/wt_measQ/q/qpin.patch:173-175`). The rebased hook (W12) keeps that table: `qsa_rows_av-` and `qsa_rows_te-` are **deliberately not matched and not pinned**. The all-visible rows are a deterministic function of the mask (§4.1); the TE rows occur only on non-compact ubatches, and no pinned gate runs on non-compact ubatches (Q10 is an unpinned same-campaign comparison). Every QPIN reference is recorded on the PR219 arm (W0-lab), which keeps the PR naming, so the record-mode assert (qpin.patch:67) still guards every reference.

---

## 5. Dispatch and cost rule: where each decision lives

| decision | owner | rule | evidence |
|---|---|---|---|
| producer (§4.1) | qwen4exp graph build | shortcut test `n_kv ≤ indexer_top_k + r − 1`; inside it, the `prefer_dense` policy; compact or non-compact from the predicate | C1 d2, C4 d2, R1-AV |
| G and shortcut policy | backend procs `ggml_backend_sparse_attn_group(dev, n_q_per_stream, n_kv, T_k, T_v, shortcut)` and `ggml_backend_sparse_attn_prefer_dense(dev, T_k, T_v, n_q, n_kv)` (the proc maps n_q to the D4 band internally), resolved as `ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "<name>")` (PR ggml-backend.h:207; `ggml_backend_dev_get_proc_address` does not exist, O21) for the device of the layer's K buffer. Without the procs: G = 1, prefer_dense = (T_k == T_v == f16). CPU: G = 1, prefer_dense default | HIP G: `n_q < 128 ? 1 : 4` until AT-A (past the shortcut) and AT-W7PERF (in the shortcut); Vulkan G: 1; prefer_dense flips per (pair, n_q band) by the pre-registered 0.98 rule (AT-W7PERF HIP, AT-B4 Vulkan) | C1 d3, C13 d2; C11; O12, O23; P2-9 |
| kernel instance, V pack vs direct, split count | backend compute, from one shape-only function `sa_select(op, dev)` that `supports_op`, `get_alloc_size` and the census also use | V direct iff n_kv ≥ 64·n_q for q8_0/q4_0, and for f16/bf16 under the default CAPPED trait (OWNER). G1: V staged. **G1 split count `splits(n_q, n_sel, n_head, CUs)` is a shape-only function written in sparse-attn-attend-g1.cu; its partial buffer is `n_stream · n_q · n_head · splits · (D + 2) · 4 B`** (with the PR rule, qsa-decode.cu:134-145: max 9.00 MiB for n_q ≤ 127 and 12.42 MiB at n_q 263 per stream, R1-C Part 1) | C18 d1–d3; R1-C |
| supports_op | backend | T_k, T_v ∈ S_backend (§6), D = 256, n_head/n_head_kv ∈ {4, 8, 12}, strides as in §2.2, rot ∈ {0,64,128,256}, **and a pipeline/kernel for this (op, types, tier) was created on this device** (Vulkan: the pipeline pointer is non-null; never a null-pipeline dispatch, P2-16); Vulkan also 32-bit offsets (L38, evaluated with `ggml_nbytes(k)` at the reserve n_kv = n_ctx; f32 K past UINT32_MAX elements is an init error) and **every binding ≤ `maxStorageBufferRange`** (dst range, workspace range, K/V views; P2-19). **Never** n_q or pointers. Vulkan rejects G = 4 plans | L14, L29; P2-16, P2-19 |
| HIP tier selection per arch | backend | WMMA tiers (G4, G1, generic) are selected at **runtime only on `GGML_CUDA_CC_IS_RDNA3_5`** (the arch the PR gates every QSA kernel to, PR qsa-decode.cu:85, fattn.cu fa7) until validated elsewhere; every other HIP arch (RDNA3 dGPU gfx1100/1101, RDNA4, CDNA) gets the portable tier. Today those archs run the WMMA/MFMA dense masked FA; their QSA speed on the portable tier is UNPROVEN (U15) and listed in §19 | PR qsa-decode.cu:85 |
| placement | llama: pins (D15) + guard (§10) | host K buffer → pinned to CPU; every sparse node on the backend of its layer's K buffer | D15, R1-NKVO |
| workspace size | Plan: `get_alloc_size = max over G' ∈ policy range of plan_nbytes(G') + scratch` (§2.1). Attend: `get_alloc_size(dst) = nbytes(dst) + max over every G ∈ policy range and every n_q' ≤ n_q of ws(G, n_q', n_kv, T_k, T_v, n_stream)`, evaluated by `sa_select`. V-pack bytes use `min(n_kv, 64·n_q' − 1)` cells × 1 KiB × n_stream (CAPPED), or n_kv cells for f16/bf16 under the always-pack trait. So the reserve at (n_q = ub, n_kv = n_ctx) bounds every smaller ubatch whatever G and split count it gets | C3 code verification; C15; R1-C (G1 non-monotone in n_q: 65 → 7.68, 66 → 6.24 MiB); O20 |

---

## 6. Kernel inventory

### 6.1 Type sets

| set | types | HIP | Vulkan |
|---|---|---|---|
| S_fast | f16, bf16, q8_0, q4_0 (all 16 K×V pairs) | type-templated loaders (D8/D9/D10) | spec-constant dequant into the coopmat kernel |
| S_generic | f32, q4_1, q5_0, q5_1, iq4_nl, and any pair with one of them | one LDS-staged loader with a runtime `switch(type)` over the existing per-type dequant | the same coopmat kernel (spec constants cover all of them, L41) |

- Any other type, D ≠ 256, or GQA ∉ {4, 8, 12} is rejected by supports_op: an init error (§10), never a CPU fallback.
- S_generic perf and GQA 4/8 perf are UNPROVEN (U8); §16 P4 states what the PR text says if a generic pair misses.

### 6.2 HIP (`ggml/src/ggml-cuda/sparse-attn-*.cu`, flat, so the existing globs pick them up)

The files sit at the top level of `ggml/src/ggml-cuda/` with a `sparse-attn-` prefix: the HIP and CUDA CMake globs only match top-level `*.cu` plus `template-instances/` (ggml-hip/CMakeLists.txt:69, ggml-cuda/CMakeLists.txt:117; O37).

| file | kernels | instances |
|---|---|---|
| `sparse-attn-plan.cu` | `sa_plan_rows` (one block per query; validate, sort only when not already ascending, dedup into the over-allocated scratch); `sa_plan_merge<G>` (512-lane ASC merge for G4, direct emit for G1). Ported from master's `qsa3_rows_kernel` + `qsa3_merge_kernel` (PR qsa-prefill.cu:28-240); the 512-lane merge is C3's `P_asc 512` | 1 + 2 |
| `sparse-attn-q8.cuh` | `sa_q8x2_to_h2`: #219's `qsa3_q8x2_to_h2` / `qd_q8x2_to_h2` (PR qsa-prefill.cu:22, qsa-decode-wmma.cuh:9), one copy instead of three (L8). q4_0 `0x6400\|q − 1032` trick and bf16 → f16 RNE, as in C18's `c18_kernel.inc` | — |
| `sparse-attn-loaders.cuh` | `sa_load_k<T, DIRECT\|SPLIT>`, `sa_load_v<T, PACKED\|SPLITGATHER\|DIRECT_F16\|STAGED>`, with the select rule of §2.2 for masked/padding cells | — |
| `sparse-attn-pack.cu` | `sa_pack_v<T>` (PR `qsa_pack_v` indexing, qsa-prefill.cu:654) into the attend over-allocation | 4 + 1 generic |
| `sparse-attn-attend-g4.cu` | `sa_attend_g4<KT, VT, VL, EPI>`: the PR's `qsa3_attn_kernel` body (WMMA f16→f32, 8 waves × 32 dims, rows = 4·GQA, online softmax, 2 barriers), with no mask path (the PR's runtime mask/gate/o16 paths cost 2–8.5 %, C18). EPI ∈ {plain, gated + f16/bf16 aux store} | KT 4 × (VT,VL) 8 × EPI 2 |
| `sparse-attn-attend-g1.cu` | `sa_attend_g1<KT, VT>`: the PR's `qsa_decode_wmma_partial` body with V staged (C18 decode harness), K q8_0 = #219's decode q8 K loader (qsa-decode-wmma.cuh:63-74); split-K over chunks; n_stream on the grid | 16 |
| `sparse-attn-merge.cu` | `sa_merge_splits`: deterministic fixed-order merge + rot_o epilogue for G1 | 1 |
| `sparse-attn-generic.cu` | G4 and G1 attend with the runtime-switch loader, S_generic pairs | 2 |
| `sparse-attn-portable.cu` | scalar FMA attend for HIP devices without RDNA3/3.5 WMMA; all of S via to_fp16 into LDS | 1 |
| `sparse-attn-fwht.cuh` (D13) | `fwht_block<N>` device function, shared with `fwht.cu` | — |

- **Device guards (O25).** Every sparse-attn file's host code is under `#if defined(GGML_USE_HIP)`; WMMA/`v_permlanex16` device bodies are under the RDNA3 arch macro with `NO_DEVICE_CODE` otherwise, as the PR does (qsa-prefill.cu:2, :260; qsa-decode-wmma.cuh:19), **and the host selects them only when `GGML_CUDA_CC_IS_RDNA3_5(cc)`** (§5 "HIP tier selection per arch"). The portable tier compiles for every HIP arch and is what every other HIP arch runs.
- **CUDA (non-HIP ggml-cuda builds): `supports_op` returns false for both ops**, so a QSA model on CUDA gets the §10 init error (D22). No nvcc exists on strix-002 (`ssh strix-002 which nvcc`: not found), so no CUDA kernel can be compiled or proven here. Today such a build takes the masked route (fattn.cu:1198-1207: only ggml-cuda checks `dst->src[5] && !src[3]`, `#else return false` → probe fails → masked). Master's CUDA/MUSA CI is red on run 41354919590 (W13), so no CI run of that route in this fork is known. This removal is listed in §19 "what it replaces".
- **Resource rule:** 0 VGPR spills in every instance, enforced by the **existing** repo gate `.github/workflows/hip-quality-check.yml` (`-Werror -Wno-tautological-compare`, `scripts/hip/gcn-cdna-vgpr-check.py --baseline scripts/hip/gfx1151-vgpr-baseline.json`), not a new script (O37). W13 regenerates the baseline in the pinned image (§13 W13).
- **Census:** the fork's existing proc names `ggml_backend_debug_dispatch_census_{reset,count,skip}` (implemented today only in master ggml-vulkan.cpp, PR :18942-19016; test-backend-ops.cpp:521-563 consumes them), implemented for HIP per sparse instance. **Key strings** (exact, listed in `ggml/src/ggml-cuda/sparse-attn-census.h`, generated from the instantiation table so the expected list cannot drift): `sa_plan/G<g>`, `sa_attend/G4/<KT>/<VT>/<VL>/<EPI>`, `sa_attend/G1/<KT>/<VT>/splits<n>`, `sa_pack/<VT>`, `sa_merge`, `sa_generic/G<g>`, `sa_portable`, `qsa_expand`; plus a proc `ggml_backend_debug_dispatch_census_list` that enumerates every compiled key. Host-side counts are graph captures, not launches, when HIP graphs are on (ggml-cuda.cu:7142, :7398), so every count-valued census recipe runs with `GGML_CUDA_DISABLE_GRAPHS=1` (as C12 did); presence/zero assertions are valid either way.
- **Pool use outside the sparse kernels.** D12 covers the sparse-attn kernels only. The TE rows producer's `top_k` on HIP uses `ctx.pool()` for narrow shapes (top-k.cu:577-599; only the RDNA3.5 "wide" path is pool-free, :548-557), so small non-compact ubatches still allocate unreserved pool memory. M3 runs on the TE route too (AT-C12 layout), and §19 states it.
- **Forcing switch:** `GGML_CUDA_SA_FORCE=g1|g4|portable|generic|vdirect|vpacked` forces a kernel family, tier or V loader for tests on gfx1151 (g1/g4 override the group policy; tests build the plan with the forced G). It changes the kernel choice inside the backend only, never placement.
- **Gate fusion.** The matcher (ggml-cuda.cu:6739-6795) is retargeted to `SPARSE_ATTN → [views] → CONT(gate) → SIGMOID → MUL` for the G4 EPI=gated instance, including the bf16/f16 `mmb` aux store (`ggml_cuda_bf16_for_mmb_reader`, :6768-6770); it lands in W3 so the whole-graph op test can reach it (O28). The FA branch of the matcher stays until W13.
- **OVL kind-4 region (O16).** The end predicate at ggml-cuda.cu:6955-6961 (`find(i + 1, i + 48, … op == GGML_OP_FLASH_ATTN_EXT)`) becomes `op == FLASH_ATTN_EXT || op == SPARSE_ATTN` in W7. The `SPARSE_ATTN_PLAN` node is built right after the rows producer, which runs before `Qcur_full` (PR qwen4exp.cpp:1901 vs :1908), so it sits before the region; `add_alloc_dep` to `end` (:7019-7028) then keeps rows and plan alive up to the attend node, which §7 accounts for.

### 6.3 Vulkan (`ggml/src/ggml-vulkan/vulkan-shaders/`)

| shader | derived from | role |
|---|---|---|
| `sparse_attn_plan_rows.comp` | `flash_attn_sel_prep.comp` (dedup), ascending sort added | G1 plan, whole-dst bit-exact to the CPU. The source shader uses only `barrier()` and shared memory, `local_size_x = 256` (PR flash_attn_sel_prep.comp:17, :54-57, :90-131); only its host gate ties it to 64 lanes (master ggml-vulkan.cpp:4800-4805), so it gets **its own gate**: any device with 256 invocations per workgroup and the shared memory it declares |
| `sparse_attn_cm.comp` | `flash_attn_sel_cm.comp` (one 64-lane subgroup per (query, kv head, stream, split), register-built fragments; K/V bound as `uvec4` and built from 16-byte loads, :37-38, :150-151, :225) | RADV RDNA3 fast path. K/V via `flash_attn_dequant.glsl` (`FaTypeK`/`FaTypeV` spec constants, L41); bf16 via `bf16_to_fp32` → f16 at staging (C14 d4). **V variants: direct-fragment, LDS-staged, and the diagnostic pack arm (`sparse_attn_pack.comp`: K+V → f16 into the SPARSE_ATTN over-allocation when n_kv < k_T·n_q, the HIP trait form)**, selected per type by AT-B3; the pack arm ships for a type only under the AT-B3 M1 condition. The pipeline gate stays RADV + RDNA3 (master ggml-vulkan.cpp:4825-4828 = PR :4760-4763) |
| `sparse_attn.comp` | `flash_attn_sel.comp` (scalar) | 64-lane devices that are not RADV RDNA3 and pass the existing sel gate (`sel_sg64` + shuffle + arithmetic + fp16 + 64·GQA invocations + 46,720 B shared at GQA 12, master :4800-4816); correctness tier (7.8× slower on the op, C11 d3) (U28) |
| `sparse_attn_portable.comp` | new (scalar FMA, no subgroup operations) | **every other Vulkan device (P2-16)**: one workgroup per (query, kv head, stream, split), subgroup size a spec constant that is never assumed, reductions through shared memory and `barrier()`, ≤ 8 KiB shared memory, ≤ 128 invocations per workgroup, fp16 storage not required (K/V converted to f32 at load via `flash_attn_dequant.glsl`, rounded to f16 values by `packHalf2x16`/`unpackHalf2x16` to keep the §2.2 contract). Device requirements: Vulkan 1.2, storage buffers, `shaderInt16` not required. The source `flash_attn_sel.comp` hard-codes `LANES = 64` and `subgroupShuffleXor(s, 32)` (PR :56-58, :129), so it cannot serve these devices |
| `sparse_attn_reduce.comp` | the sel split reduce | split-K merge + rot_o epilogue (shared-memory reduction variant for the portable tier) |

- **Devices that lose QSA.** A device that cannot create even the portable pipelines (below Vulkan 1.2, or with less shared memory / fewer invocations than stated) gets supports_op = false and the §10 init error, which names the device and the missing capability (§10). Such devices are listed in §19 by capability, not by vendor; vendor subgroup sizes (NVIDIA 32, Intel 8–32, MoltenVK 32, Mali 16) are general knowledge, UNPROVEN on strix-002. Today such devices take the masked route through dense Vulkan FA (master ggml-vulkan.cpp:21465-21467: only maskless src[5] needs the sel pipelines), which is untested on any non-64-lane device (UNPROVEN).
- **Workspace:** a `get_alloc_size` hook for SPARSE_ATTN and SPARSE_ATTN_PLAN in the Vulkan buffer type (today `ggml_nbytes`, L42, master ggml-vulkan.cpp:17287 = PR :17125-17128). Each tensor's `round_up(nbytes, minStorageBufferOffsetAlignment) + workspace` must stay below the buffer type's max size, the 1 GiB suballocation block (PR ggml-vulkan.cpp:5378-5384, :17120-17123; the master line is re-cited at W6); W6 asserts this at the reserve for -c 1048576 (Vulkan workspace = split partials, plan scratch and, only if AT-B3 ships it, the pack).
- **Binding of the workspace (P2-19).** Every existing dispatch binds dst through `ggml_vk_tensor_subbuffer`, whose range is `ggml_nbytes(tensor)` (master :7028-7053), so a shader reading [nbytes, alloc_size) through that binding is outside the descriptor range (undefined behaviour). SPARSE_ATTN_PLAN and SPARSE_ATTN therefore bind **two** descriptors: dst with range `ggml_nbytes(dst)`, and the workspace as an explicit `vk_subbuffer{buf, off + round_up(nbytes, minStorageBufferOffsetAlignment), ws_bytes}`; they never reach the workspace through `ggml_vk_tensor_subbuffer`. The round-up is included in get_alloc_size. supports_op/reserve reject a shape where either binding exceeds `maxStorageBufferRange` (checked separately from the 1 GiB block, as the existing code does for the sel split buffer, PR :10909).
- **Synchronisation of over-allocated nodes (O17).** The barrier tracker computes ranges with `ggml_nbytes` (ggml-vulkan.cpp:16243-16262, :16310-16321), so bytes in [nbytes, alloc_size) are invisible to it. Rule: an unconditional `ggml_vk_sync_buffers` before the first dispatch of every SPARSE_ATTN_PLAN and SPARSE_ATTN node, plus the usual barrier between split and reduce passes (the `prealloc_split_k_need_sync` pattern, :10688-10710).
- **Prealloc buffers.** The sparse path uses no `prealloc_x/y/split_k` (they are sized lazily at compute time outside the scheduler, ggml_vk_preallocate_buffers :16056-16186). M1 logs them per type (§16).
- **Not used:** coopmat2 (absent, C11), bf16 coopmat (absent, C11 d2), G4 on Vulkan (U7). The Vulkan `SPARSE_ATTN_PLAN` / `SPARSE_ATTN` case lists exclude G = 4 through an explicit per-backend **expected-unsupported list** in test-backend-ops (`expected_unsupported(backend_name, op, params)`), so `--fail-unsupported` passes on listed cases and fails on any other "not supported".
- **Census:** the existing Vulkan dispatch census (PR ggml-vulkan.cpp:18942-19016) extended to the new pipelines, with the tier in the key (`sa_attend/vk/<cm|scalar|portable>/<VL>/<KT>/<VT>`).

### 6.4 CPU

The plan and attend references of §3, for all of S, threaded as stated. Both ops are added to the ggml-cpu.c n_tasks and work-size switches (PR :2452, :2784, default `GGML_ABORT("op not implemented")`).

### 6.5 Other backends

- **Meta (tensor-split) backend:** `ggml-backend-meta.cpp:1070` aborts on unknown ops. QSA under `LLAMA_SPLIT_MODE_TENSOR` is rejected at context creation with a message (O21); no meta handler in this PR.
- **RPC:** the op-count assert and protocol version are updated (§2); a `-DGGML_RPC=ON` build is a W1 done-when.

---

## 7. Memory model

Shapes: 12 QSA layers, H_kv 2, n_head 24, D 256, indexer dim 128, n_sel 2051, ub4096 unless stated. "Meas" = measured; everything else is arithmetic from shapes.

| item | where reserved | c32k | c131k | c1M |
|---|---|---|---|---|
| K+V cache f16/bf16 · q8_0 · q4_0 | KV init | 768 · 408 · 216 MiB | 3072 · 1632 · 864 MiB (f16, q8_0 meas C16) | 24576 · 13056 · 6912 MiB |
| indexer cache f16 · q8_0 | KV init | 96 · 51 MiB | 384 · 204 MiB (meas C16) | 3072 · 1632 MiB |
| rows I32 [2051, 4096], per layer, freed after the plan | compute buffer | 32 MiB | 32 MiB | 32 MiB |
| plan dst (G4) + its scratch over-allocation, one live at a time | compute buffer | 50.4 + 33.6 MB | same | same |
| attend dst F32 [256, 24, 4096] | compute buffer | 96 MiB | 96 MiB | 96 MiB |
| attend V pack, **default CAPPED**, every type: min(n_kv, 64·n_q − 1) × 1 KiB × n_stream | dst over-alloc | 32 MiB | 128 MiB | 256 MiB |
| attend V pack, f16/bf16 under the **always-pack** trait (owner alternative): n_kv × 1 KiB × n_stream | dst over-alloc | 32 MiB | 128 MiB | 1024 MiB (+512 MiB at c524288 / +1024 MiB at c1M vs PR at n_kv = n_ctx; +256 / +768 MiB vs the PR's retained 256 MiB legacy-pool pack, R1-C) |
| G1 split-K partials: n_stream · n_q · 24 · splits · 258 · 4 B, max over n_q' ≤ min(ub, 512) (PR rule) | dst over-alloc | ≤ 12.42 MiB × n_stream | same | same |
| dense shortcut mask f16 [≤ 2048, ub] (existing) | compute buffer (host input) | ≤ 16 MiB | same | same |
| non-compact variant (§4.3, §7.1.4), only when `n_seq_max > 1`, S = 1, T = min(ub, 512) total, K = n_ctx / s: **V_host = 3·K·T + 12·K·s + 51244·T + 32 B** (MTP; main ≤ +0.29 MiB), exact on 11 configurations; **V_ROCm0 (unified, T = 512) = 6412·K + 141.36 MiB main / 6412·K + 70.03 MiB MTP**, exact on 3; split 7704·K + 153.36 / 6168·K + 70.03 MiB (2 points). The reserve grows by **max(0, V − P_pp)** per backend (gallocr grow-only) | compute buffer (reserve variant, §7.1) | V_host 73.4 MiB, V_ROCm0 341.7 MiB main [formula, K 32768] | meas c131072 ub4096 q8_0: +0.00 ROCm0 / +16.73 host main; unified ub1024: +167.44 / +166.95 (R2-RES) | V_host 1.5 GiB, V_ROCm0 6.4 GiB main [formula, K 1048576; UNPROVEN outside the measured K ≤ 262144] |
| Vulkan0 compute buffer, W0 reference (maskless f16, C11 shape) | compute buffer | 1494.23 MiB (ub4096), 226.22 MiB (ub512) (C11, PR tree + #240; re-measured on W0 by AT-B5) | — | — |
| **deleted:** masked attention route (base q8_0) | — | 6809.41 − 4083.69 MiB (meas C16) | 21578.66 − 4133.44 MiB (meas C16) | not allocatable |
| **deleted:** QSA pool scratch (unreserved) | pool | — | — | — (PR pool high-water 650.7 MiB f16 / 617.0 MiB q8_0 at c64k ub4096, all pool users, meas C7) |

- **Resulting rule (corrected, O43).** Under the default CAPPED trait, every type's attention memory is independent of the type except for its own KV bytes, and q8_0/q4_0 never exceed f16. Under the always-pack alternative f16/bf16 exceed q8_0/q4_0 by up to 768 MiB at c1M; that is the owner's deferred choice (AT-O), not a claim of this plan.
- **OVL lifetime.** Under kind-4 regions, `add_alloc_dep` keeps rows (32 MiB) and the plan alive until the attend node; M1/M2 are measured with OVL at its default (mask 15) so they include it.
- **The residual non-compact variant** is identical in host bytes for f16 and quantized caches (R2-RES B': +233.76 MiB host both). **On ROCm0 it is not type-independent:** the main-context variant alone is 1555760640 B for f16 against 1799308800 B for q8_0 (+232.27 MiB for q8_0) at c131072 ub4096 T 1024 (R2-RES); cause UNPROVEN (U37). M1's "every T ≤ new f16 + 1 MiB" is therefore evaluated on the reserve **including** the variant, and a q8_0 excess there is a finding that W9 must explain or remove before M1 can pass. The default-server cost goes into the PR description: +421.02 / +383.99 MiB main and +388.00 / +383.99 MiB MTP (ROCm0 / host, R2-RES cell A, S = 1 = S = 2 at ub512).
- **ub8192 column (owner's server, L50, L64)** [arithmetic, same shapes]: plan dst + scratch ≈ 101 + 67 MB, rows 64 MiB, attend dst 192 MiB, CAPPED V pack min(n_kv, 64·8192 − 1) × 1 KiB = 256 MiB at c261888. The pool items that move into the reserve (§16 M2) make M5's PR219-relative bound carry their size explicitly: M5 lists "moved pool items" as a separate allowed term with its arithmetic, instead of hiding them in a strip allowance.

### 7.1 Reservation (D17)

1. `get_alloc_size` hooks on HIP (exists) and Vulkan (new), sized as in §5 (plan: max over G; attend: max over G and n_q' ≤ n_q, at n_kv = n_ctx). The existing reserve at n_kv = n_ctx (PR llama-context.cpp:915, :3002) then covers every runtime n_kv, n_q and G.
2. **Extra reserve pass A**: `graph_reserve` with sequential positions, n_tokens = min(ub, 512), n_outputs = 1 (QSA fast-cache inputs +33280 B ROCm0, trim tile +8224 B host, C15).
3. **Extra reserve pass B** at n_kv = top_k + r − 1 (dense-shortcut mask +1.92 MiB host at ub512, C15). Mechanism: `sched_reserve` uses `memory->init_full()` (llama-context.cpp:915), which always gives the full n_kv, so pass B uses a hybrid-idx memory context constructed with an explicit `n_kv_override = 2048` (the padded shortcut n_kv), added for the reserve only. Every extra pass runs before `backend_buf_exp_size` is recorded (llama-context.cpp:1008) in the allocating context, and enters the fit-probe max in the no_alloc context (item 7).
4. **Non-compact reserve variant (P2-14; R2-RES)** whenever `n_seq_max > 1`, **unified or split**, in the main and MTP contexts:
   - one extra reserve graph built with a graph-param flag `force_noncompact = true` (replacing the lab global of `EV/r2res/r2res.patch`), which makes the §4.2 decision false at `init_full` n_kv (the empty-cache reserve ubatch is otherwise compact: PR prefix.h:36 skips empty cells, C15);
   - **T = min(n_ubatch, S × 512) tokens in total, S = 1**; n_seqs = n_seq_max; s = kv_unified ? 1 : n_seq_max, so n_tps = T / s per stream; n_kv = n_ctx / s per stream (init_full); n_outputs = max(n_seqs, min(T, n_outputs_pp));
   - the re-split unit of §4.3 is the same T (total tokens), so the reserve bounds every runtime non-compact ubatch in token count;
   - **size:** V_host and V_ROCm0 of §7 (exact formulas); the reserve grows by max(0, V − P_pp) per backend; forced reserve = max(unforced, variant alone) to the byte in every measured run (R2-RES);
   - **one shape the reserve cannot express:** a one-stream non-compact ubatch of T tokens in a split cache (a one-stream reserve aborts at ggml.c:4155 `GGML_ASSERT(a->ne[2] == b->ne[2])` from `ggml_set_rows` in `build_attn_qsa`, `logs/srv_D_f3.log`). Whether the [K, T/s, s] reserve covers [K, T, 1] at runtime is UNPROVEN (U38); W9 runs it under STRICT (AT-C12), and a violation there makes the split-cache variant use T per stream (n_tps = T, total T·s) instead, re-measured;
   - **pre-registered ceiling (kept from R2-RES):** host increment ≤ S·n_ctx·1536·n_stream + 1 MiB, device increment ≤ S·n_ctx·3541·n_stream + 1 MiB per context, n_ctx = `-c` (n_stream = 1 unified). All S = 1 cells measured pass (R2-RES). If a W9/AT-C12 cell exceeds it at S = 1, the fallback is to reserve the variant only when an mtmd projector is loaded or a `seq_cp` has happened, with the graph-build check (§4.3) turning every other non-compact ubatch into a decode error; that fallback is a PR-blocking owner decision, not automatic;
   - not keyed on `n_pos_per_embd > 1`, which is true for every qwen4exp model (llama-model.cpp:3788-3790).
5. **Single-sequence contexts (n_seq_max == 1)** reserve no variant. If the §4.2 predicate rejects a ubatch there, the always-on graph-build check (§4.3) makes `llama_decode` return an error with a log line naming the layout; it never builds an unreserved non-compact graph.
6. **Strict-reserve check (validation):** after every ubatch, `ggml_backend_sched_get_buffer_size(b) ≤ backend_buf_exp_size[b]` for every backend, ROCm_Host included (≤, not the destructor's ==, PR llama-context.cpp:689-697). Enabled by `LLAMA_STRICT_RESERVE=1`, on in every validation run. Because gallocr never shrinks, the check also hooks the existing realloc callback (llama-context.cpp:881; ggml-backend.cpp:1953) so the **first** growing ubatch and its node are logged. The production guarantee does not rest on this env var: it rests on items 1–5 and the graph-build check.
7. **Fit probe (P2-13; R2-RES).** (a) Remove the `ggml_backend_sched_reset` at the start of `ggml_backend_sched_reserve_size` (ggml-backend.cpp:2296): graph_reserve already resets before building the graph (llama-context.cpp:2985) and llama-context.cpp:3026 is the only caller (`grep -rn ggml_backend_sched_reserve_size`: header, definition, this call), so the reset only wipes the pins set during graph build (R1-NKVO backtrace). It also affects the upstream pins at llama-graph.cpp:147, :2679, :3149 and llama-context.cpp:3479, which the same pre-existing bug hits today; W7 runs AT-D4's fit-probe placement census to confirm the pins survive (UNPROVEN until then). (b) In no_alloc mode, every reserve graph (pp, tg, pass A, pass B, the non-compact variant) goes through `reserve_size` into a local array and `backend_buf_exp_size[b]` = the per-backend **max** (ggml-alloc.c:1235-1244 overwrites sizes per graph, so passing `sizes` alone would keep the last graph). Measured on the PR tree with the variant: this max equals the real reserve to the byte in 64/64 cases, while today's probe misses it in 16/16 forced runs on host and 10/16 on ROCm0, by up to −2539.06 MiB ROCm0 and −818.79 MiB host (R2-RES, `EV/r2res/probe_gap.txt`). The probe builds the MTP context (common/common.cpp:1248-1275), so the same applies there. (c) The §10 guard also runs on the fit path. AT-C18 gates it.

---

## 8. Rotation and gate (D13)

- **Storage rotation unchanged.** K rotated iff `ggml_is_quantized(type_k)` and D % 64 == 0, n_rot_k = 256; V rotated iff `ggml_is_quantized(type_v)`, n_rot_v = 64; `LLAMA_ATTN_ROT_DISABLE` kept (PR llama-kv-cache.cpp:323-375, L45). k_cur/v_cur rotated by the hinted FWHT mul_mat before `cpy_k`/`cpy_v` (PR qwen4exp.cpp:1698-1704; `GGML_HINT_SRC0_IS_HADAMARD`, ggml.h:484). The indexer cache stays unrotated (L16).
- **q and output rotation move into the op** (W10): the graph passes `rot_q = n_rot_k`, `rot_o = n_rot_v` (0 when that side is not quantized) and deletes the q rotation (:1699) and both output un-rotations (:1737-1738, :1780-1781).
- **Why.** The output un-rotation sits between FA and the sigmoid gate, so the gate fusion never matches for quantized caches: 0 `qsa_prefill_gated` launches for q8_0 vs 176 for f16 (C12 part B); review OP_PROF +20.4 ms (rotation) + 17.7 ms (unfused gate) per ub4096 pp4096 pass.
- **Order of landing.** W7 lands with rot_q = rot_o = 0 and graph-side rotation, comparable numerically with PR219. W10 moves rotation in-kernel; rollback = pass 0 again.

---

## 9. Indexer cache (D14)

- **Scope: `LLM_ARCH_QWEN4EXP` only**, main and MTP contexts. `LLM_ARCH_GLM5_NEXT`, which builds the same `llama_memory_hybrid_idx` with `params.type_k` (PR llama-model.cpp:2994-3021) and reads the indexer cache (glm5-next.cpp:890-958), keeps `type_k_idx = type_k` (O26).
- New `cparams.type_k_idx`, default `auto` for qwen4exp:
  - q4_0, q4_1, q5_0, q5_1, iq4_nl → q8_0. Measured for q4_0: on kind 1, q8 idx 0.985678 vs f16 idx 0.985873, ratio 0.999802 (SE 3.1e-6); the q4 indexer gives ratio 0.98814 and fails the C9 0.99 rule by 11 SE (R1-E). The other sub-8-bit types follow the conservative extension (C9 d2, U8).
  - f16 → f16, q8_0 → q8_0, bf16 → bf16, f32 → f32 (bf16 and f32 UNPROVEN, U9).
- **Cost** for q4_0 users: indexer bytes ×1.89 (C9 d3); at c131k 204 MiB (q8_0) instead of 108 MiB [arithmetic].
- CLI `--cache-type-k-idx auto|<type>` is an override only (C9 d4); it replaces the measurement-only `QPIN_IDX_TYPE`.
- **Session/state files: no format change, no version bump (P2-26).** The indexer cache is an ordinary `llama_kv_cache` built with its type (PR llama-memory-hybrid-idx.cpp:97-117), its state is the last section of every hybrid-idx stream (:369-371, read back at :392), and `state_write_data` already writes each layer's K type (PR llama-kv-cache.cpp:2467-2470) while `state_read_data` rejects a mismatch with `mismatched key type (%d != %d, layer %d)` (:2789-2796), turned into `failed to restore kv cache` (:2405-2406) and re-thrown by hybrid-idx after dropping the sequence (:403-410). So an old qwen4exp file saved with a sub-8-bit -ctk is rejected by the existing check once W11 maps its indexer to q8_0, and GLM5_NEXT files and qwen4exp f16/q8_0 files (indexer type unchanged) load as before. A new field would need a `LLAMA_SESSION_VERSION` / `LLAMA_STATE_SEQ_VERSION` bump (include/llama.h:46, :49; strict equality at llama-context.cpp:4096, :4156), which would invalidate every saved file of every model. W11 only wraps that error in hybrid-idx `state_read` with a clearer message ("saved with indexer type X; this context uses Y; re-save the session"), without changing the stream. If the indexer's `n_rot_k` differed between the old and new type, the earlier `incompatible key rotation` check (:2774-2777) fires first; the exact message W11 tests is confirmed when the test is written.

---

## 10. Placement pins, guard and errors (D15)

- **Pins.** When a QSA layer's K buffer is a host buffer (`!cparams.offload_kqv`, or the layer on CPU with `-ngl 0` / partial `-ngl`), llama calls `ggml_backend_sched_set_tensor_backend(sched, node, backend_cpu)` for that layer's `SPARSE_ATTN_PLAN` and `SPARSE_ATTN`, the way PR llama-graph.cpp:3147-3150 pins the non-FA path. The **shortcut** attention node (`FLASH_ATTN_EXT` or non-FA dense output) is pinned only if AT-W7PERF (d) shows pin-on / pin-off ≥ 0.98; the R1-NKVO timed arms never had it pinned (60 `ROCm0·copy·none` shortcut FA prints under pin3/pin7, R1-NKVO placement table), so its cost is unmeasured until then. Pinning the rows producer tail is not done: R1-NKVO pin 7 = pin 3 for FA and plan placement. Measured with pins on plan + attend: both on CPU, reading K/V directly, only the rows (8 KiB – 4 MiB per layer) copied (R1-NKVO).
- **Host-KV fallback (P2-10).** AT-D4 gates `-nkvo` per type. Pre-registered order if a type misses: (a) the CPU op as specified in §3 (dequantize once per kv head, vertical SIMD) is already the default; any further CPU change must stay inside the §14 bands against O_CON; (b) a **named GPU placement with the host-KV copy** (the PR219 route: plan and attend on the K-buffer-less GPU of the layer's other weights, K/V copied by the scheduler), enabled per type by the AT-D4 result, logged in the init line as `route=host-kv-copy`, and accepted by the guard as a named route only when `offload_kqv == false`. A type that misses after (a) and (b) is a PR-blocking FAIL that goes to the owner with the ratios.
- **Guard — when (P2-12).** `ggml_backend_sched_reserve` ends with `ggml_backend_sched_reset` (ggml-backend.cpp:2305-2318), which memsets `hv_tensor_backend_ids` to −1 (:2280-2289), so after it `ggml_backend_sched_get_tensor_backend` returns NULL for every node (:1047-1048, :2549-2556). The init guard therefore runs on a **split_only** `graph_reserve` (split_graph without reset, as `resolve_fused_ops` does at llama-context.cpp:712-725 and :3024-3029), once for each reserve graph: pp, tg, pass A, pass B and the non-compact variant, in the main and MTP contexts, and in the fit-probe context (§7.1.7). The runtime guard runs after `ggml_backend_sched_alloc_graph` of every newly built graph (alloc_graph does not reset, ggml-backend.cpp:2418-2442). The guard inspects **llama's gf** (the graph llama built), not `sched->graph`, which also holds alloc-dep NONE nodes and input copies (ggml-backend.cpp:1800-1870) that would break the "exactly one consumer" check.
- **Non-vacuity.** Each guard pass counts the nodes it inspected with a non-NULL backend; a pass on a context with QSA layers past the shortcut that inspects fewer than 12 × (plan + attend) such nodes fails as "guard inspected nothing". A unit test asserts the count on a passing config (W7).
- **Guard — what.** Every `SPARSE_ATTN_PLAN` / `SPARSE_ATTN` node and every QSA-layer shortcut attention node is assigned to the backend of the device that holds that layer's K buffer (or to the CPU / the named host-KV-copy route when pinned); each plan has exactly one consumer, a `SPARSE_ATTN` with the same n_q and n_stream.
- **On a violation at init (P2-17, D22):** throw `"QSA layer %d (%s): sparse attention K=%s V=%s is not supported on device %s (backend %s): %s"`, where the last field is the cause from the backend: `no sparse-attention kernel in this backend` (ggml_backend_reg_name: CUDA, MUSA, Metal, SYCL, OpenCL, CANN, WebGPU, …), `Vulkan device lacks <capability> for every tier`, or `type pair not supported`; followed by the remedies that apply to that cause: "run QSA on the CPU with `-ngl 0` or `--no-kv-offload`, or use a HIP (RDNA3.5) or Vulkan build" for the first two, "use another -ctk/-ctv (supported here: <list>)" for the third. At decode time, return an error from `llama_decode` and never compute.
- **Other init errors:** a QSA model with `LLAMA_SPLIT_MODE_TENSOR`; a QSA model on a ggml-cuda (non-HIP) device (§6.2); `n_ctx_seq` above the tested maximum (§4.4).
- **Init log line,** one per context: route, K and V types, indexer type, G and prefer_dense policy source, pins applied. This is #219's "types in the log line" (b60447506), adapted.
- **Deleted:** the probe `qsa_selected_key_supported` (PR llama-memory-hybrid-idx.cpp:50-66), `selected_key_attn` (:128), `LLAMA_QSA_SELECTED_KEY` (:129).

---

## 11. What is deleted (W13)

- **ggml-cuda:** `qsa-prefill.cu/.cuh`, `qsa-decode.cu/.cuh`, `qsa-decode-wmma.cuh`; the QSA dispatch, maskless abort and `_supported` branches in fattn.cu (PR :1150-1208); `ggml_cuda_qsa_gate_ok` (:1086) and the FA branch of `fa_gate_qsa` (ggml-cuda.cu:6741-6795, retargeted in W3); env `QSA_DIRECT`, `GGML_QSA_RANKSORT`, `GGML_QSA_MAX_KEYS`, `QSA_DECODE_MAX_QUERIES`.
- **Vulkan:** `flash_attn_sel.comp`, `flash_attn_sel_cm.comp`, `flash_attn_sel_prep.comp`; `ggml_vk_flash_attn_sel*` and the maskless FA supports_op branches (PR ggml-vulkan.cpp:10577-10625, :21303-21306 = master :21465 area; located by symbol at W13); env `GGML_VK_SEL_CM` (replaced by `GGML_VK_SA_FORCE`). DSA `flash_attn_top_k*` is kept.
- **llama:** the probe, `selected_key_attn`, `LLAMA_QSA_SELECTED_KEY`; `QSA_MASKLESS_MAX_KV`; the masked half of `build_attn_qsa` (PR qwen4exp.cpp:1742-1784) and `llm_graph_input_qsa::maskless`; the TE route as an attention route (the TE producer stays for non-compact rows, D6); `LLAMA_QSA_DENSE_SHORTCUT`; the `(n_stream == 1 || n_tps ≥ 128)` restriction; the QSA q/out rotation nodes (W10).
- **Tests:** `QSA_PREFILL`, `QSA_DECODE`, `QSA_PREFILL_MASKLESS`, `QSA_DECODE_MASKLESS`, replaced by the §14 tests. `QSA_SCORE`, `QSA_STRIP`, `QSA_EXPAND` are kept.
- **ggml.h:** `ggml_flash_attn_ext_add_top_k` asserts a mask (§2.3).
- **Who else uses FA + src[5]** (`grep -rn "build_attn_mha\|ggml_flash_attn_ext_add_top_k" src/models`, re-run at W13): qwen4exp and deepseek4 (deepseek4.cpp:760, which passes a mask, so the §2.3 assert and the W13 maskless deletions in fattn.cu and ggml-vulkan are safe for DSV4). glm5-next passes `sel` as kq_mask and no top_k (glm5-next.cpp:1083), so it is untouched.
- **Deleted for every backend (D22):** the masked attention route was the only QSA implementation that runs on backends without a SPARSE_ATTN kernel; §19 lists the affected backends by name.

---

## 12. Recovered from PR #219

Head 94e086847, 3 commits on 13a279bfa, all authored by Iggy Jackson <iggy@theiggy.com> (L2). W0 cherry-picks all three onto the base (§0): as they are while #234 is not in the base (dry-run clean on ea5285ddf); with a recorded conflict resolution in qsa-prefill.cu if it is (then the "taken as-is" rows below for 68da4191d and 94e086847 read "adapted (conflict resolved), author kept").

| #219 piece | commit / file / function | final status | reason and source |
|---|---|---|---|
| strip_views map | 68da4191d `src/models/models.h` (`strip_views`), `qwen4exp.cpp` `strip_view` lambda in `build_qsa_top_k` (PR :1604-1617) | **taken as-is** (cherry-pick, author kept) | Load-bearing on the non-compact path. PR-tree measurements: +1120 MiB (HIP q8_0), +1125 MiB (HIP f16), +1378 MiB (Vulkan f16) without it on a deep image ubatch, `-np 1`, c65536; 0.148 → 0.029 MiB/cell on the masked route (C19 d1, d2); +9014.36 MiB on the default server with two image slots (R1-C layout 3). Final-tree positive control: AT-C16 |
| strip_views key (input, first, strip size) | b60447506 models.h tuple key | **taken as-is** | Fixes the latent mixed-ratio assert of review §5.2 |
| exact q8 → half conversion | 68da4191d `qd_q8x2_to_h2`; 94e086847/68da4191d `qsa3_q8x2_to_h2`; `qsa_q8_to_h` (:630) | **taken as-is, deduplicated** into `sparse-attn-q8.cuh` (Co-authored-by) | Bit-identical to `dequantize_block_q8_0_f16` (review; C18: 9225 bitwise checks; R1-A: 264/264 PR-verbatim comparisons bitwise equal) |
| decode q8 K loader | 68da4191d `qsa-decode-wmma.cuh:63-74` | **taken as-is** into `sa_attend_g1` (Co-authored-by) | Within 1 % of split once V is staged (C18 d3) |
| decode q8 V loader | 68da4191d V gather | **replaced** by LDS-staged V | −41 % at ub512 d131k, −36 % at tg (C18 d2, d3) |
| q8 pack conversion | 68da4191d `qsa_pack_k/v<true>` (qsa-prefill.cu:635, :654) | **adapted** into `sa_pack_v<q8_0>` into the dst over-allocation (Co-authored-by); `qsa_pack_k` dropped | C18 d1 |
| direct q8 K read in qsa3 (block-w addressing) | 94e086847 `qsa3_attn_kernel<…,Q8>` K branch | **adapted**: block-w addressing kept, split register layout (Co-authored-by) | C18 d3 |
| direct q8 V read in qsa3 | 94e086847 `DIRECT_V && Q8` | **replaced** by split-gather at n_kv ≥ 64·n_q | spills 120 B/lane (C18 d3; L10) |
| `n_q·64 < n_kv` direct-V switch | 94e086847 `qsa_direct` (qsa-prefill.cu:685) | **taken as the threshold** for every type's V under the default CAPPED trait (owner may switch f16/bf16 to always-pack) | C18 d1; OWNER |
| bool template flags | `qsa3_attn_kernel<DIRECT, WPE, DIRECT_V, Q8>` | **replaced** by type-level loader policies | L9, L10 |
| q8 admission in decode | 68da4191d `qsa-decode.cu:92-102` | **replaced** by static supports_op over S² and GQA {4,8,12} | L30; C13 d1 |
| `selected_key_q8` + "ROCm" name check | 68da4191d | already removed by b60447506 | L3, L4 |
| probe with real types at n_q {1, 513} | b60447506 | **replaced** by static supports_op + pins + guard + init error; the idea (decide with the real cache types) is kept in the guard | L14, L15, L29, C2 d3 |
| `LLAMA_QSA_SELECTED_KEY` override | b60447506 (:129) | **deleted** in the branch; used read-only in the PR219 measurement binary as the kind-1 partner for q4_0 Q4 arms (R1-E) | L15; R1-E |
| log line with K/V types | b60447506 | **adapted** into §10's init line (Co-authored-by) | — |
| q8_0 / mixed tests | b60447506 `test_qsa_{prefill,decode}_maskless(type_k, type_v)` | **adapted**: shapes and the maskless cell formula (test-backend-ops.cpp:11545-11574) ported into `SPARSE_ATTN` cases with census assertions, plus the shapes #219 lacked (Co-authored-by). The rows keep their duplicate keys (the new plan dedups deterministically); the comment at test-backend-ops.cpp:16558 claiming qsa_prefill direct-V reach is not carried over (no ported case reaches it, R2-BOP route census), and a case with n_q ≥ 513 and 64·n_q < n_kv is added | L11; R2-BOP |
| `QSA_DIRECT` env | 94e086847 | **deleted** (routing is shape-only; `GGML_CUDA_SA_FORCE=vdirect\|vpacked` for tests) | C13 |
| commit-message claims | 68da4191d "perplexity bit-identical", "10.7 GB at 262k"; 94e086847 "all QSA_DIRECT modes bit-identical" | bit-identity of strip sharing supported (review §3, KLD 0); 10.7 GB at 262k UNPROVEN (U17) | C19 |

Not from #219 (kept for attribution clarity): the `poison` test case (base, Piotr Wilkin), the TE producer and the non-compact scorer (base).

---

## 13. Ordered work items

**Order of landing: E0 (not a commit), W0, W12, W1, W2, W3, W4, W5, W6, W7, W8, W9, W10, W11, W13.** (W12 keeps its id; it lands right after W0 so every later gate can run, O7, O27, O33.) Each item builds (HIP `build/`, Vulkan `build-vk/`, flags as in the rules), passes its own tests, and leaves the tree usable. CI runs on the PR head only; internal commits W0–W12 still contain 94e086847's 28-spill `<1,8,1,1>` instance (L10) and would fail the repo's zero-spill gate on their own; this is stated in the PR description (O37). GPU validation is run by one agent at a time under `flock ~/gpu.lock`.

### E0 — Make the evidence durable (before W1; not a commit) (P2-27)
- **What.** From the orchestrator host, with `SP` as defined in the header: `cd "$SP" && rsync -a --exclude=SHA256SUMS --exclude=IMPLEMENTATION.md final219/ bs_skeptic.py skept_ocon strix-002:~/dev/wt_lab/evidence/final219/` (the work log `IMPLEMENTATION.md` keeps growing and is not evidence); write `SHA256SUMS` on both sides over the same file set (`LC_ALL=C`, `find . -type f ! -name SHA256SUMS ! -name IMPLEMENTATION.md -print0 | sort -z | xargs -0 sha256sum`, the local side mapping `bs_skeptic.py` and `skept_ocon/` to `./bs_skeptic.py` and `./skept_ocon/`) and compare them with `cmp`. Files already mirrored under `~/dev/wt_lab/{bands,c1..c19,r1,r1av,r1c,r1e,r1nkvo,r1q8,r1s,r1ub,r2res}` stay where they are; the evidence copy is the one every `EV/` citation resolves to. Path fix: the r1e patch is `EV/r1e/r1e/r1e_patch.py` (the round-1 plan cited `…/r1e/r1e_patch.py`, which does not exist).
- **Done when:** checksums equal on both sides; outside the header paragraph that defines `SP` and `EV` ("Evidence location"), the plan, the PR description and the #219 comment cite no path under `SP` or `/tmp` (`grep -n '[S]P/\|/[t]mp/'` returns only that paragraph; the bracketed form does not match its own line).

### W0 — Import #219 on the base
- **What.** Re-run the §0 dry-run on the base SHA and record it. `git switch -c qsa-rewrite <base>`; `git cherry-pick 68da4191d b60447506 94e086847`, authorship and messages unchanged. On ea5285ddf the dry-run is clean (temp index `git apply --cached --check`: 3 × OK; master's test-backend-ops hunks 4371–4410, 6672, 13173, 13450, 14718, 16675 vs the PR's 11514, 11599, 16548; #219 does not touch ggml-vulkan). With #234 in the base, 68da4191d and 94e086847 conflict in qsa-prefill.cu (§0); a conflict is resolved inside the cherry-pick with a `[conflict resolved: <file>]` line; Iggy stays the author. Build HIP and Vulkan.
- **Also:** build **W0-lab** (§0) in `~/dev/wt_lab/w0lab` (detached worktree; never a second tree inside wt_rewrite), HIP `build/` and Vulkan `build-vk/` with C11's flags, and the C10-patched llama-bench linked against each (`c11/build_bench_vk.sh` for Vulkan). W0-lab is the PR219 arm of every gate.
- **Done when:**
  - HIP `-o QSA_PREFILL_MASKLESS`, `-o QSA_DECODE_MASKLESS`, `-o QSA_PREFILL`, `-o QSA_DECODE` on ROCm0 pass;
  - Vulkan0 test-backend-ops counts are recorded on W0 and become the W0 reference; they are compared with the C11 counts (14/14 + 12 not supported, 34/34 + 19, 26/26, 92/92) for the QSA_* ops only, because master carries #241/#242/#244/#245/#246 and +99 test-backend-ops lines, so a count change outside QSA_* is expected;
  - `sched_reserve` at -c 131072 b/ub 4096 gives ROCm0 4133.70 MiB for q8_0 (C16), if the base's HIP code equals 13a279bfa's (else recorded as the new W0 value);
  - llama-perplexity c8192 × 2 gives the C15 PPL for f16/q8_0 at ub4096 (1.9449 / 1.9519), same condition.

### W12 — Validation tooling and per-chunk KLD precision (lands directly after W0)
- **What.**
  - `tools/perplexity`: per-chunk mean KLD printed straight from `kld_values` (not reconstructed from the cumulative table) at `%.9e` (`QKLD_CHUNK`), with `QKL_SPLIT` (the R1-AV hunk of `EV/r1av/r1av_full.diff`).
  - Committed under `scripts/qsa-validation/` as sources and patches, not built by default:
    - the QPIN patch rebased onto the branch (`~/dev/wt_measQ/q/qpin.patch`; name table unchanged, §4.5), with the per-chunk agreement line `QPIN_QSA_CHUNK` (`EV/r1e/r1e/r1e_patch.py`) and the post-pin defect hook (`EV/r1q8/r1q8_lab.diff`);
    - the C10 llama-bench text patch (`EV/c10/c10.diff`, `analyze.py`, changed to the mean-of-reps statistic of §16);
    - the C7 shim and KFD sampler (`EV/c7/remote/c7shim.c`, `c7samp.py`);
    - the C15 trace patch; the C3/C5/C18 and `bands` kernel harnesses (`EV/bands/src/`, with the R2-BOP UNI/TBO fixtures, `EV/bandsop/src/`); the census checker (§14.3); `r1/text_campaign.sh`, `r1/img_campaign.sh`, `r1c/runB3.sh`, `r1s/r1s.sh`, `r2res/srv.sh`, `r2res/parse.py`;
    - the small summaries the PR cites (`kdecide.md`, `compact*.md`, `analysis_op.md`, `res2_summary.md`, `probe_gap.txt`, the R1/R2 `*_analysis.out`).
  - **The PR219 measurement arm** is **W0-lab** (§0) with `LLAMA_QSA_ALLVIS` unset. It carries the c1 lab diff, QPIN, QKLD_CHUNK, the r1e patch and the r1q8 post-pin hook, all inert without env. `~/dev/wt_lab/r1av/tree` and `~/dev/wt_lab/r1q8/tree` are used only for the reproduction check below. `wt_measQ` (5-decimal cumulative print, perplexity.cpp:1868) is not used for any gated pair.
  - The owner may drop this commit from the final PR; the patches stay in the PR description's artefact list either way.
- **Done when (run on the W0-lab build):**
  - QPIN self-replay gives per-chunk KLD 0 at %.9e (≤ 3.7e-7, the uint16 base quantization) at c8192 × 8 and c32768 × 8, ub4096, `LLAMA_FN_LAST_ATTN_TRIM=0` reference;
  - **control (P2-31):** against the re-recorded trim-0 f16 ub4096 reference `ref8t0` (c8192 × 8), q8_0 ub4096 pinned reproduces **mean KLD 1.371668e-4, chunk SE 1.25e-5** (R1-UB table D, r1av tree, ref8t0, MEASUREMENTS.md:3433/3441), **per-chunk equal at %.9e** to the r1av arm's chunks, as R1-UB's self-pairs are; and agreement 0.996743 on c8192 × 16 against ref16t0 (R1-E). (1.375064e-4 is the R1-Q8 value against the default-trim reference ref8, rows_pin 757760, MEASUREMENTS.md:3239; it is not the control for a trim-0 reference.) Equality is expected only while the base's HIP code equals 13a279bfa's; otherwise the r1av values are history and the bands are re-derived (next item);
  - **AT-W12-BANDS (§16.5)** passes before any W7+ Q1/Q3-A/Q4/Q1-vk arm is scored. Until it has passed, a gate whose band is not yet measured stays UNPROVEN and the PR cannot claim it.

### W1 — Op contracts, ggml touchpoints and CPU reference
- **What.**
  - `GGML_OP_SPARSE_ATTN_PLAN` / `GGML_OP_SPARSE_ATTN` in ggml.h/ggml.c (enum after `GGML_OP_LIGHTNING_INDEXER`), constructors, `_nbytes`, `_scratch_nbytes`, names/symbols, the §2 asserts (bounds rule included).
  - **Checklist (O21):** `ggml_op_alloc_size_may_expand` (ggml-backend.cpp:2560-2573) += both ops; ggml-rpc.h:14 static_assert + `RPC_PROTO_PATCH_VERSION` bump; ggml.c:1223 / :1338 asserts; ggml-cpu.c n_tasks and work-size switches (:2452, :2784); meta backend: reject at context creation (§6.5).
  - CPU forwards of §3, threaded.
  - test-backend-ops: `test_sparse_attn_plan` (err() override = exact whole-dst integer comparison, `max_err = 0`, the `test_get_rows_packed` precedent at test-backend-ops.cpp:12974-12977) and `test_sparse_attn` (`run_whole_graph` with `fusion_test_nodes = {attend}`, so the plan node is never NMSE-compared). `tests/test-sparse-attn-plan-oracle.cpp` (CPU vs c3 `std::map` oracle, memcmp).
  - `--fail-unsupported` (a "not supported" result, a missing census proc or a missing engagement opt-in is a FAIL for the selected ops; today `engagement_required` returns false and skips, test-backend-ops.cpp:556-563).
- **Done when:**
  - both ops pass on CPU for every §14 shape; the plan equals the c3 oracle byte for byte on the C3 captured rows;
  - **the CPU op is bitwise equal to `EV/bands/src/bands_ref.h` O_CON** on the wiki d4k, SYN 262k and UNI fixtures (R2-BOP generator), f16/q8_0/q4_0/bf16 (the TU built with `-ffp-contract=off`, §3); bf16 overflow → ±inf in both;
  - **contract vs f64 (R1-A, k = 2):** NMSE(O_CON, O_C18) ≤ 2.4e-9 (wiki d4k) / 7.3e-8 (SYN 262k); NMSE(O_CON, O_PLAN) ≤ 1.0e-7 / 3.1e-7 (`EV/bands/compact_cpu.md`);
  - **coarse cross-check vs CPU FA + src[5]** (PR ops.cpp:8946), NMSE(CPU op, CPU FA) ≤ 2 × p100 of R1-A (3): V = f16 3.4e-3; K ∈ {q8_0, q4_0}, V ≠ f16 2.6e-5; K = bf16, V ≠ f16 5.1e-6; K ∈ {f32, f16}, V ∉ {f16} 2.3e-7 (with the src[5] index remap: FA's src[5] = first n_kv_raw keys plus key n_kv_raw + row[j], ops.cpp:8988-8990). Not a correctness gate (it cannot see D2/D3, R1-A);
  - builds compile: CPU-only, `-DGGML_RPC=ON`, HIP, Vulkan; a test that over-allocates the SPARSE_ATTN dst does not hit the ggml-backend.cpp:199-206 assert.

### W2 — HIP plan kernels
- **What.** `sparse-attn-plan.cu` (§6.2), supports_op/compute/get_alloc_size for `SPARSE_ATTN_PLAN`, the HIP census procs.
- **Done when:**
  - `-o SPARSE_ATTN_PLAN -b ROCm0 --fail-unsupported`: whole-dst bit-exact against the CPU on every §14 shape (permuted, duplicate, −1, out-of-range, all −1; G1/G4; 1–2 streams; n_kv = 2051 views ending at their buffer end);
  - the C3 harness on recaptured d131k rows (`scripts/cap.sh` of C3) ≤ 1.0 ms/layer mean, text and random (C3 d3), bit-exact;
  - HIP `-Werror` build (`CMAKE_HIP_FLAGS='-Werror -Wno-tautological-compare'`) and `gcn-cdna-vgpr-check.py` report 0 spills for the new TUs.

### W3 — HIP G4 attend, fast types, gate fusion
- **What.** `sparse-attn-q8.cuh` (#219 helper, Co-authored-by), loaders, pack, `sparse-attn-attend-g4.cu` for S_fast², EPI plain and gated; `ggml_cuda_sparse_attn_get_alloc_size` (§5 max rule); supports_op/compute with rot = 0; the gate-fusion matcher retargeted to SPARSE_ATTN; the V pack-limit trait (default CAPPED, OWNER).
- **Done when:**
  - `-o SPARSE_ATTN -b ROCm0 --fail-unsupported` for all 16 pairs × {packed, direct V} × 1/2 streams × GQA {4, 8, 12}: **NMSE(GPU, CPU op) ≤ the §14.2 band of the class the census reports for the case (G, splits, n_sel bucket)**, where a band exists; classes without a measured band follow the §14.2 transfer rules; f64 errors reported per shape and type;
  - `test_sparse_attn_gated` (§14) reaches every EPI=gated instance and the fusion-reject case;
  - census: every G4 instance (plain and gated) launched at least once;
  - 0 spills (`-Werror` build + vgpr check);
  - kernel timing on C18 rows (`~/dev/wt_lab/c18/cap`, kept) with SYN data (C18: SYN/real 1.002, range 0.985–1.023): each type's best row ≤ the **same-run** PR-verbatim reference × the C18 target ratio × (1 + 0.04) (the ±4 % placement uncertainty C18 states for its derived X+P rows, minor note): f16 X+P / D+P ≤ 1.0; q8_0 X+P ≤ 1.033 (d4k) / 1.017 (d131k); q4_0 ≤ 1.017 / 0.986; bf16 ≤ 1.018 / 1.047;
  - direct V vs packed + pack re-measured at syn 262k and 1M, and the 64·n_q crossover at n_q ∈ {512, 1024, 4096}, for q8_0/q4_0 and for f16/bf16 under both traits (records the data AT-O needs).

### W4 — HIP G1 attend and the G policy
- **What.** `sparse-attn-attend-g1.cu` + `sparse-attn-merge.cu` for S_fast²: V staged; K direct for f16/q8_0 (#219 decode q8 K loader, Co-authored-by), split for q4_0/bf16; n_stream on the grid; split-K partials in the over-allocation with the shape-only split formula of §5. HIP policy procs (`group`, `prefer_dense`).
- **Done when:**
  - `-o SPARSE_ATTN` with n_q ∈ {1, 2, 4, 8, 9, 64, 65, 66, 127, 263} × n_stream {1, 2} passes the §14.2 UNI band of the census-reported (G1, splits, n_sel) class; split counts with no measured row (6–16, > 33) are measured first on the PR-verbatim decode kernel with the R2-BOP harness (U34) before the case is gated;
  - **U34 as a gate:** the new GPU-vs-CPU-op NMSE on every ported #219 shape is ≤ its UNI row (R2-BOP), re-recorded on the new CPU op; the planted d1/d2/d3 still exceed 2 × the row;
  - G1 kernel gates on the C18 decode fixtures, each ≤ same-run C18 harness value × 1.014 (C18 decode noise max 1.4 %): q8_0 D+S ≤ 3.770 ms, q4_0 X+S ≤ 3.579 ms, bf16 X+S ≤ 3.601 ms (ub512-equivalent, wiki d131k); tg q8_0 ≤ 19.9 µs, q4_0 ≤ 19.7 µs, bf16 ≤ 19.1 µs (O13);
  - **AT-A (§16.5)**: the two-way G1/G4 crossover fit, including n_q {2, 4, 8} (MTP verify widths, C2) and {16, 32, 64, 96, 127, 128, 256, 512}; the HIP `group` proc encodes the fitted rule;
  - **AT-C5 (§16.5)**: reserved partials and plan dst ≥ the §5 max formulas, 0 strict-reserve violations at ub {64, 65, 66, 127, 128, 129}, including a forced G flip between reserve and runtime;
  - census coverage and 0 spills.

### W5 — HIP generic and portable tiers
- **What.** `sparse-attn-generic.cu`, `sparse-attn-portable.cu`, `GGML_CUDA_SA_FORCE`.
- **Done when:**
  - `-o SPARSE_ATTN` passes for every S² pair with a generic member, and for the portable tier forced on gfx1151 (all of S), against bands set by the §14.2 transfer rules (anchored on O_CON vs f64 and on the PR-verbatim kernel of the nearest class, never on the tier itself; U8);
  - GQA 4/8: the band is transferred by rule (§14.2) and the G4 WMMA GQA 4/8 instance is additionally compared with the GQA-12 instance run on the same per-head data (heads replicated), which must agree to the GQA-12 row;
  - **compile gate:** `cmake -DGGML_HIP=ON -DGPU_TARGETS="gfx1100;gfx1151;gfx90a;gfx942"` builds with 0 errors (single build slot, O25);
  - generic kernel time at C18's wiki d4k / d131k recorded against f16 (P4/P7).

### W6 — Vulkan plan and attend
- **What.** The shaders of §6.3 (V variants and the pack arm, the scalar and the portable tier), supports_op for both ops with the pipeline-exists rule, the `get_alloc_size` hooks with the alignment round-up, the explicit workspace binding and the `maxStorageBufferRange` check (§6.3), the §6.3 sync rule, the Vulkan policy procs (G = 1; prefer_dense default), the census, `GGML_VK_SA_FORCE=scalar|portable|vdirect|vstaged|vpack`. The old sel pipelines stay until W13 (they are the in-build op reference of AT-B2).
- **Done when:**
  - `-o SPARSE_ATTN_PLAN -b Vulkan0 --fail-unsupported`: whole-dst bit-exact against the CPU for G = 1 cases (G = 4 excluded by supports_op, listed);
  - `-o SPARSE_ATTN -b Vulkan0 --fail-unsupported` passes every S² pair (cm every V variant, forced scalar, forced portable) within the §14 bands, including the bf16 D256 case, the V-side overflow case and the poison cases;
  - **non-64-lane devices (P2-16, AT-B7):** the same two op runs on lavapipe (`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`, installed on strix-002) and on RADV with the portable tier forced at required subgroup size 32, 0 failures and 0 "not supported" for S_fast; supports_op false (no dispatch) when a pipeline is absent;
  - the plan cases with unsorted rows (whose bit-exactness depends on the scratch) pass on RADV and lavapipe, once with `VK_LAYER_KHRONOS_validation` enabled and 0 out-of-range access reported (P2-19);
  - Vulkan rot_q/rot_o bitwise equal to `fwht.comp` on the same input (unit test);
  - a Vulkan stress graph where gallocr places the SPARSE_ATTN workspace over a freed tensor still being read is memcmp-stable over 20 iterations; the test **asserts from `tensor->data` and get_alloc_size that the sparse node's [nbytes, alloc_size) range overlaps a tensor read by the preceding dispatch**, and fails if it does not (so it cannot pass trivially);
  - per tensor, round_up(nbytes) + workspace < the 1 GiB block and each binding ≤ `maxStorageBufferRange` at the -c 1048576 reserve;
  - **AT-B2, AT-B3 and AT-B7 (§16.5)** pass.

### W7 — Graph switch (text producers, all types, MTP, pins, guard)
- **What.**
  - qwen4exp: compact CB rows → `ggml_sparse_attn_plan` + `ggml_sparse_attn` for every type pair (rot 0, graph-side rotation); shortcut → `prefer_dense` policy, otherwise the 4-D all-visible producer; G from the policy; split streams on ne[3] at any n_tps; trim passes its n_q; rows named per §4.5.
  - Eligibility drops `type_k/type_v`, `ubatch.token` (moved to W9), `selected_key_attn`, `QSA_MASKLESS_MAX_KV`, **`cparams.flash_attn` and `cparams.offload_kqv`**; hybrid-idx cache for qwen4exp created with `v_trans = false` (D21); the `n_ctx_seq` cap of §4.4.
  - Pins (D15) and guard (§10: split_only reserve graphs, llama's gf, non-vacuity count) in the main, MTP and fit-probe contexts; the init log line; the fit-probe fix (§7.1.7: no reset in reserve_size, per-backend max).
  - The always-on graph-build check (§4.3), the decode error for single-sequence rejections and the multi-sequence re-split land in W9, together with the extended predicate; until then non-compact ubatches take the old route (W7) or the TE route (W8).
  - OVL kind-4 end predicate retargeted (§6.2); plan node placed before the region.
  - The two census sources of §14.3: `LLAMA_QSA_TRACE=1` (llama side: producer × route × context, counted per **executed** ubatch at compute time, printed at exit) and the backend dispatch census read by llama-perplexity, llama-bench and llama-server at exit through the census procs (G, V loader, splits, EPI, instance). The "CPU launch" check is llama-side: at every guard pass, any QSA-layer `SPARSE_ATTN` / `FLASH_ATTN_EXT` node whose `ggml_backend_sched_get_tensor_backend` is the CPU while the layer's K buffer is on a GPU is counted, and nonzero fails the run (ggml-cpu itself knows nothing about layer placement) (O32, P2-22).
  - Non-compact ubatches still take the old masked route in this item; W8 removes it.
- **Done when:**
  - route census per the §14.3 table on llama-perplexity c8192 ub {16, 512, 4096} and llama-bench pp4096 @ d4096, for f16/q8_0/q4_0/bf16/q8_0K-f16V/f16K-q8_0V: 0 FLASH_ATTN_EXT + src[5]; every shortcut ubatch carries `qsa_rows_av-<il>` unless prefer_dense; 0 CPU launches (`GGML_CUDA_DISABLE_GRAPHS=1` for counts);
  - MTP server (C2's `run_c2.sh` flags) with draft KV default, `-ctkd q8_0`, `-ctkd q4_0`: census all sparse past 2051; 0 CPU launches;
  - the guard fires as an init error on a fake GPU device registered with `ggml_backend_device_register` (ggml-backend.h:237) that rejects SPARSE_ATTN, and on a real HIP build with a forced-reject hook; it does not fire on any validation run;
  - **HIP shortcut policy cells (pre-registered, AT-W7PERF §16.5):** the op grid n_q {1, 2, 4, 8, 16, 32, 64, 127, 512, 1024} × n_kv {512, 1024, 2048}, G1 and G4 forced; llama-bench text pp512/pp2048 at d0 ub {64, 512, 1024, 2048} and tg128 at d0/1024/1900, sparse vs dense for q8_0, q4_0, bf16, q8_0K/f16V, f16K/q8_0V (one build; dense via a measurement-only `LLAMA_QSA_PREFER_DENSE` env that W13 deletes); a (pair, n_q band) whose LB of sparse/dense < 0.98 (§16 statistic) is set to dense in the HIP `prefer_dense` proc; the shortcut G per band is set by the AT-A noise rule;
  - quality: Q1 (A/B/C), Q2, Q3 (A/B), Q4, Q5, Q8 (on the W7 build, defects after the pin), Q11 (a/b/c), Q12; perf P1–P7; memory M1–M3;
  - **AT-D1–D5 (§16.5)** pass (OVL, `-ngl 0`, partial `-ngl`, `-nkvo` with its binding perf gate, `-fa auto/off`); **AT-C18** (fit probe) passes;
  - the guard's non-vacuity count is ≥ 12 × (plan + attend) on a passing config with QSA layers past the shortcut (unit test).

### W8 — TE rows for non-compact ubatches; masked route deleted
- **What.** The non-compact scorer's `top_k(expanded, 2051)` rows with the kq_mask −1 filter (§4.1 row 4) feed the sparse op; the masked attention half of `build_attn_qsa` is deleted; strip_views unchanged.
- **Done when:**
  - CPU unit test: on `r1/r1_oracle.py`-ported layouts (seq_cp prefix L ∈ {8192, 8193, 8194, 8195}, 2-slot image), the new rows' visible set equals PR219's TE ∩ visible for every row;
  - **Q10** passes (after AT-Q10Q6 calibration); Q6 passes;
  - **AT-C16a** (layout 3, `r1c/runB3.sh`, before the W9 re-split, 8-strip ubatches as on PR219): census on slot-B text ubatches with bias new > 0 and kq_mask new > 0, reuse 88 per src per graph (8 strips × 11 reusing layers, R1-C layout 3 table); main ROCm0 runtime peak ≤ 8054.08 MiB.

### W9 — Compact visibility for single-sequence images + reserve variants
- **What.** §4.2 (per-stream sequence sets, the shared `qsa_stream_one_seq` function, rank-based predicate, embedding ubatches) with its CPU test incl. the negative cases; reserve passes A/B; the non-compact variant for `n_seq_max > 1` with **S = 1** (§7.1.4); the NC0 hook, the replay and the always-on graph-build check (§4.3; in single-sequence contexts it turns any non-compact ubatch into a decode error), with `n_qsa_resplit`; `tests/test-qsa-resplit.cpp`; `LLAMA_STRICT_RESERVE` with the realloc-callback log; the duplicate-key API reading (§4.2 step 4).
- **Done when (all with `LLAMA_STRICT_RESERVE=1`, 0 violations, correct answers):**
  - the visibility test passes on all recorded, synthetic and negative layouts, with the scan/predicate agreement assert;
  - **`tests/test-qsa-resplit.cpp` (CPU, R2-RES spec):** layouts §4.2 single sequence with 2-D cells at several depths; R1-CB unified two sequences with a seq_cp prefix ending inside a block; R1-C layout 3; split `-np 2` with one image slot. Batches: random token batches with 1–4 sequences and lengths 1..3·n_ubatch, plus embd batches with 2-D positions. Grid n_keep_tail {0, 4} × n_ubatch {512, 1024, 4096, 8192} × S {1, 2}. For every ubatch from init_batch-with-hook, apply it to the real attention and indexer caches and evaluate the graph-time predicate. Asserts: (i) graph-NC(u) ⇒ NC0(u); (ii) NC0(u) ⇒ u.n_tokens ≤ U; (iii) every batch token in exactly one ubatch; (iv) the last n_keep_tail tokens of every sequence share one ubatch; (v) attention cells ≡ indexer cells after every apply; (vi) replays ≤ #ubatches. Pass: 0 violations over a fixed seed set;
  - **re-split in both contexts (P2-2):** (1) unified 2-slot server, A (image) and B both send ≥ 4096-token prompts at once at ub 4096, so split_equal forms multi-sequence ubatches; (2) the same with `-np 2` split and the MTP sidecar. Pass: 0 graph-build non-compact ubatches above U in either context, 0 graph-build-check errors, 0 strict-reserve violations, answers correct, and `n_qsa_resplit` > 0 in both contexts' logs;
  - **one-stream non-compact ubatch in a split cache** (the shape the reserve cannot express, §7.1.4) under STRICT: 0 violations, else the §7.1.4 fallback;
  - re-split counter > 0 on R1-C layout 3 and on the R1-RES workload; 0 graph-build-check errors on the image matrix (requests A/B/C);
  - B65t (C19), the C15 deep-image D request at ub1024, request C (L64);
  - llama-perplexity c8192 ub512 f16/q8_0; np2 shared prefix with trim on/off; llama-bench pp4096 @ d4096 trim 1/0;
  - **M4 recipes (R1-RES):** (a) `r1s.sh … S0_img T_deep` with `-np 2` split, then `--kv-unified`; (b) `-np 1` request C; (c) a synthetic single-sequence prompt with two images back to back across a ubatch boundary at depth > 32k; on (b) and (c) also 0 predicate rejections on image ubatches; on (a) split, 0 non-compact split-stream ubatches (then the split-stream variant may be dropped);
  - **AT-C12, AT-C16b, AT-C17 (§16.5)**.

### W10 — Rotation and output un-rotation in the op
- **What.** `fwht_block<N>` shared with fwht.cu; rot_q prologue / rot_o epilogue in every HIP and Vulkan kernel; the graph passes n_rot_k / n_rot_v and deletes the QSA rotation nodes.
- **Done when:**
  - `-o SPARSE_ATTN` with rot ∈ {64, 256} within the §14 bands; HIP in-kernel q transform bitwise equal to `fwht.cu`, Vulkan to `fwht.comp` (unit tests);
  - C12-style census (graphs off) shows the gated G4 epilogue for q8_0 and q4_0;
  - OP_PROF of ub4096 pp4096 d0 and d32k q8_0: (rotation + gate + attention) below W7's (U6);
  - Q5, Q1-A/B and P1–P6 re-run and pass;
  - **Rollback:** revert this commit only.

### W11 — Indexer cache mapping
- **What.** §9.
- **Done when:**
  - **Q4** passes (bands of §15);
  - the KV buffer line shows the indexer type;
  - **session files (P2-26):** a qwen4exp session saved on W0 with `-ctk q4_0` fails to load on W11 with the existing per-layer check (`mismatched key type`, or `incompatible key rotation` if that check fires first, §9) wrapped in the new hybrid-idx message naming the first QSA layer; f16/q8_0 qwen4exp files and a GLM5_NEXT file saved on W0 load; `LLAMA_SESSION_VERSION` and `LLAMA_STATE_SEQ_VERSION` are unchanged (`git diff include/llama.h` shows no version change);
  - a synthetic hybrid-idx load with `LLM_ARCH_GLM5_NEXT` and `-ctk q4_0` (unit test constructing the memory, or a GLM5-Next GGUF if one exists on strix-002) shows `type_k_idx = q4_0`, unchanged.

### W13 — Delete the old code, CI baseline
- **What.** Everything in §11, the old QSA tests, the `ggml_flash_attn_ext_add_top_k` mask assert, measurement-only env vars. **Regenerate `scripts/hip/gfx1151-vgpr-baseline.json`** in the pinned image (`rocm/dev-ubuntu-26.04@sha256:4f5ed1bf…`, compiler d6f6cb69, via podman on strix-002) with the scripts/hip/README.md procedure: the qsa-prefill (6) / qsa-decode (4) TU entries are replaced by the sparse-attn TUs, and `minimum_kernel_count` (8155) is updated; otherwise the checker fails on a missing or reduced TU (gcn-cdna-vgpr-check.py:75-79).
- **Done when:**
  - no reference to deleted symbols (`grep -r "qsa_decode\|qsa_prefill\|flash_attn_sel\|QSA_DIRECT\|LLAMA_QSA_SELECTED_KEY\|QSA_MASKLESS_MAX_KV\|LLAMA_QSA_PREFER_DENSE"` empty outside the changelog/PR text); `grep -rn "ctx.pool()" ggml/src/ggml-cuda/sparse-attn-*` empty;
  - HIP quality check locally: `-Werror` build + `gcn-cdna-vgpr-check.py --baseline` on the regenerated baseline passes;
  - master's red CUDA CI run 41354919590 (jobs cuda, musa) is recorded in the PR text as pre-existing;
  - `git merge-tree` of the head against #234, #225, #247, #248, #230 (if still open) recorded in the PR description (§0);
  - the full §14–§16.5 campaign passes on the final head: every census row, Q1–Q12, P1–P9, M1–M6, AT-* (AT-C16b, not AT-C16a, on the final head).

---

## 14. Tests that provably reach each path

### 14.1 Op tests (test-backend-ops, per op, `-j 1`, `--fail-unsupported`)

- **`SPARSE_ATTN_PLAN`** (err() = exact whole-dst comparison, max_err 0):
  - n_q ∈ {1, 2, 8, 127, 128, 129, 512, 513, 4096}; n_sel ∈ {1, 257, 2051, 4096}; n_kv ∈ {4, 2051, 4096, 65536, 270336, 1048576}, with K/V-free plan cases and n_kv = 2051 against a view of 2052 rows ending at the buffer end;
  - rows ascending / permuted / duplicates / −1 / out-of-range / all −1; cells permuted across quads; G ∈ {1, 4} (G 4 on the Vulkan expected-unsupported list, §6.3); n_stream ∈ {1, 2}; interleaved and strided rows.
- **`SPARSE_ATTN`** (`run_whole_graph`, `fusion_test_nodes = {attend}`; err() = NMSE vs CPU against the §14.2 band of the class the census reports for the case):
  - the same n_q set plus {4, 9, 64, 65, 66, 263}; n_kv ∈ {≤ 2051 all-visible, 2051 at view end, 4096, 32768, 65536, 270336, 1048576 at n_q 1}; **plus n_kv = 2^21 + 4, f16 and q8_0, n_q 1** (the first f16 K view ≥ 2 GiB, signed-int offset boundary; size_t offsets required), on HIP and CPU. On Vulkan, cells whose f32 K/V view exceeds the 1 GiB buffer max (PR ggml-vulkan.cpp:5378-5384, :17120-17123), e.g. n_kv 1048576 f32, are on the Vulkan expected-unsupported list instead of failing allocation;
  - every S² pair; GQA {4, 8, 12}; rot {0, 64, 256}; n_stream {1, 2}; interleaved/permuted K/V views (#219's `interleaved` and `blocks` scatter);
  - **both V-loader regimes** for every S_fast pair: n_kv < 64·n_q and n_kv ≥ 64·n_q; and one case with n_q ≥ 513 and 64·n_q < n_kv for direct V on G4 (R2-BOP: no ported case reaches it);
  - empty rows;
  - **poison** (ported from the base test, every S_fast type): NaN/inf f16/bf16 values and NaN/inf q8_0/q4_0 scales d, in masked-off cells of attended quads and in the cells padding entries point to; output finite and within band. **Reach is asserted at build time, not by census (P2-21):** the case computes the CPU plan dst (deterministic) inside `initialize_tensors` and asserts that, for every G and type, at least one group has counts % 4 ≠ 0 and its padded quad holds a poisoned cell outside S; the non-NaN output check then proves the select rule. (The dispatch census counts host-side dispatches, test-backend-ops.cpp:533-547, and cannot see padding reads inside a launch.)
  - **overflow**, deterministic: rot 0, a single-key S; **V-side only for bf16 V (value > 65504) and q8_0 V (d·q overflow)** (an f16 V cannot store a finite value > 65504; for f16 V the only case is a stored inf). Expected ±inf of the same sign on both backends; compared against CPU only. The case **overrides err() to compute NMSE over finite elements only** and asserts that the set of inf positions equals the expected set computed from the single-key V row (the default nmse() sums (a−b)² with no finite filter, test-backend-ops.cpp:324-337, so inf−inf = NaN, and `err > max_err` is false for NaN, :1636: the case would pass vacuously). The K-side ±inf conversion rule is tested in a loader unit test of the to_fp16 helpers, not end to end;
  - **determinism (P2-21):** a new test_case hook `n_determinism_runs` (default 1; 2 for SPARSE_ATTN_PLAN and SPARSE_ATTN) makes eval mode compute backend1 twice and memcmp the outputs; a difference is a FAIL (eval mode computes each case once today, test-backend-ops.cpp:1595-1670). On the ported rows with duplicate keys the PR decode fails this (R2-BOP: atomicCAS dedup race); the new plan dedups after sorting, so the new kernels must pass it with the duplicates kept.
- **`test_sparse_attn_gated`** (whole graph, O28): SPARSE_ATTN → view → CONT(gate) → SIGMOID → MUL, variants followed by a MUL_MAT that takes the f16 and the bf16 mmb reader paths; `fusion_test_nodes = {mul}`; `checks_engagement`/`verify_engagement` assert gated-census count == number of SPARSE_ATTN nodes; every S_fast pair × rot {0, 64, 256} × **both V-loader regimes**; one case where the overlap check must reject fusion (census gated = 0, result still correct).
- **Suite-level census accumulator.** The per-case census reset (test-backend-ops.cpp:1657-1658) stays; a suite-level accumulator sums the per-case counts, and at the end of `-o SPARSE_ATTN` the driver compares the accumulated key set with `ggml_backend_debug_dispatch_census_list` (§6.2): every compiled key launched at least once, forced tiers in their own runs.
- **`tests/test-qsa-allvis-rows.cpp`** (port of `test_allvis_rows.cpp`, CPU/HIP/Vulkan): rows == {j : mask = 0} and the 4-D shape for np2 at n_tps 1 and 128, np1, and the recorded C4 image masks (`~/dev/wt_lab/c4/dump/…`: deleted by R1-E cleanup; re-recorded at W7 with C4's `run.sh`), plus a raw M-RoPE image kq_mask inside the shortcut (to record; UNPROVEN until then, U30).
- **Ported from #219** (b60447506 case list, Co-authored-by): decode q {1, 9, 64, 512} × sel {257, 2051}; prefill 129/513/640 × 4096; 130 × 32k; 200 × 64k; 70 × 64k; mixed q8/f16 at 513 and 130; GQA 4. Census routes on PR219 (R2-BOP): every same-type case with n_q ≤ 512 ran qsa_decode (G1), qsa_prefill ran only n_q 513/640 and the 3 mixed pairs. These shapes are **gated** by the UNI rows of §14.2 (U34 is a gate, W4).

### 14.2 Numeric bands (k = 2: NMSE(GPU, CPU op) ≤ 2 × p100 measured on PR-verbatim kernels)

**Band selection (P2-20).** The band of a case is the row of the class the **census reports for that case** — (G, split count, n_sel bucket) — never a row picked from n_q (R2-BOP: G1 vs G4 differ by ×60 at n_q 512 vs 513). Every row is measured on **PR-verbatim kernels** against O_CON; **no band is ever measured on the kernel it gates**.

**(a) test-backend-ops inputs: UNI rows** (R2-BOP; init_tensor_uniform's distribution, U(−1, 1), the maskless cell formula with its row order and repeated keys; f16 and q8_0; GQA 12; 1 stream; 37 ported shapes, 142 dumps + 3 UNI seeds; `EV/bandsop/compact_op.md`, `analysis_op.md`):

| class (census) | n_sel | band (2 × p100) | set by | min k of d1/d2/d3 at this band |
|---|---|---|---|---|
| G4 | 257 | 7.80e-9 (W3: n_sel ≤ 257, measured on the W3 mix/empty/poison rows, 2× PR-verbatim p100 3.899e-9; scripts/qsa-validation/summaries/w3_prband.md; old 1.21e-9 used PR-formula rows only) | 640×257 q8_0 | 1757 |
| G4 | 2051 | 2.41e-9 (W3: n_sel > 257, same source; old 2.04e-9) | 513×2051 q8_0 | 1032 |
| G1, 1 split | 17 | 5.02e-8 | 128×17×512 | 42.3 |
| G1, 1 split | 257 | 1.07e-7 | 127×257 q8_0 | 19.7 |
| G1, 1 split | 2051 | 1.23e-7 | 512×2051 | 16.8 |
| G1, splits 2–5 | 257 | 1.34e-7 | 1×257 q8_0 | 15.5 |
| G1, splits 2–5 | 2051 | 1.56e-7 | 64×2051 (TBO) | 13.3 |
| G1, splits 17–33 | 2051 | 1.71e-7 | 1×2051×32768/b8000 q8_0 | 12.1 |

- The G1 rows are ×17–×41 (1 split) and ×1.17–×1.49 (split-K) above the SYN rows, because the PR decode visits cells in row order while O_CON visits them ascending (E1: 512×2051 drops 6.17e-8 → 1.75e-9 with sorted rows) and because of the data amplitude (E2) (R2-BOP mechanism). The new G1 kernel reads the plan's ascending quads, so it is expected to sit well below these rows; the rows stay upper bounds with min k ≥ 12.1 for detection.
- Split counts 6–16 and > 33 have no row: measured on the PR-verbatim decode kernel with the R2-BOP harness before the first case of that class is gated (W4). Seed-to-seed ≤ ×1.20, f16 vs q8_0 ≤ ×1.06, G4 bitwise stable run to run (R2-BOP noise).

**(b) Fixture rows for Q5** (R1-A; wiki d4k and SYN 262k; `EV/bands/kdecide.md`, `remote/res/*.out`):

| shape class | wiki d4k fixture | SYN 262k |
|---|---|---|
| prefill (G4) | 1.16e-9 | 2.67e-9 |
| decode n_q 512 (G1, 1 split) | 1.03e-9 | 3.03e-9 |
| decode n_q 1 (G1 split-K) | 4.53e-9 | 1.15e-7 |

- Measured on the PR/W0 kernels, bitwise equal to the shipped loaders (264/264). Run-to-run 0, seed to seed 0.3 %. **The SYN n_q 1 row is exceeded by SYN seed 3** (2 × 5.85e-8 = 1.17e-7 > 1.15e-7, R2-BOP) and rests on 24 rows per seed (288 rows, R1-A noise) with the smallest defect headroom (6.1×). Before it is a hard gate, the decode n_q 1 fixture is enlarged to all 12 QSA layers × 3 seeds and the row is re-set to 2 × p100 over that set (W4).
- **Sensitivity (measured, R1-A (5); R2-BOP):** d1/d2/d3 exceed k × p100 in every (shape, fixture) cell; tightest margin 6.1× (fixtures) and 12.1× (UNI). d1 cannot apply on the n_q 1 `blocks` rows (no tail cell, test-backend-ops.cpp:11560) and is covered there by the multi-query cases.

**(c) Transfer rules for classes with no PR counterpart (P2-20, written now).**
- **Type spread:** a pair without its own row uses the row of its class × the measured type spread (×1.12 at prefill and decode n_q 512, ×1.9 at n_q 1 on wiki, R1-A (1); ×1.06 f16 vs q8_0 on UNI, R2-BOP).
- **n_stream 2:** the output of each stream must equal, **bitwise**, the n_stream 1 output on that stream's slice (same kernel, same shape per stream); no band needed.
- **rot ≠ 0:** H_q / H_o are orthonormal, so the rotated case uses the rot = 0 band of its class; the in-kernel FWHT is bitwise equal to `fwht.cu` / `fwht.comp` by unit test (W10).
- **GQA 4/8:** the PR path is f32 (NMSE 1.8e-14 to 3.1e-14 vs f64, R1-A; MEASUREMENTS.md:3855), so it cannot anchor the new WMMA GQA 4/8 instances; they use the GQA-12 row of their class (same per-head arithmetic), checked by the replicated-heads comparison of W5.
- **Generic, portable tiers, Vulkan, mixed packed-both:** anchored on O_CON vs f64 (the W1 contract-vs-f64 bound) plus the nearest PR-verbatim class row × the type spread; the mixed packed-both route is measured on the PR's own `qsa3_attn_kernel<false, 8>` (R2-BOP skipped it: UNPROVEN until then, U8/U34). If no anchor exists, the class is **report-only** and the PR text says so; it is never gated on itself.
- f64 numbers (GPU vs O_C18 / O_PLAN) are reported per shape and type, never gated.

### 14.3 Graph-level reach: expected census rows per recipe

**Two sources (P2-22).**
- **llama side, `LLAMA_QSA_TRACE=1`:** producer (av / cb / te / dense), route and context, counted **per executed ubatch at compute time** (in `process_ubatch` after `graph_compute`, so reused graphs are counted; llama-context.cpp:1821 skips the rebuild on reuse), printed at exit. Count-valued rows use it.
- **backend side, the dispatch census** (§6.2 keys): G, V loader, splits, EPI and kernel instance, which only the backend knows (`sa_select`, §5). llama-perplexity, llama-bench and llama-server read it at exit through `ggml_backend_debug_dispatch_census_count` and print `QSA_CENSUS <key> <count>`; HIP counts use `GGML_CUDA_DISABLE_GRAPHS=1`.
- **Expected rows are generated, not hand-written.** The census checker (`scripts/qsa-validation/census_check.py`) reads every recorded ubatch shape from the trace (n_q per stream, n_kv, n_stream, types, shortcut) and evaluates the **shipped** `group()` / `prefer_dense()` policy and the same `sa_select` logic (exported by a test-only proc `ggml_backend_sparse_attn_select_dump`), then requires equality with the backend census per key. The table below is therefore the expected content under today's default policy; after AT-A and AT-W7PERF it is regenerated, and any row that no longer reaches its path gets a forced arm.
- **Forced arms after AT-A.** If the fitted rule no longer routes Q1-B (ub512) or Q3-B (ub256) to G4 with direct V, those arms are also run with `GGML_CUDA_SA_FORCE=g4` (and `vdirect` for Q3-B) so G4@256/512 and direct V stay gated e2e (R1-UB: Q3-B is the only e2e arm with 731 live direct-V calls).

| recipe | expected nonzero rows (per QSA layer), default policy | expected zero rows |
|---|---|---|
| Q1-A / P2 ub4096 c8192 | compact CB, G4, V packed (n_kv < 262144) | shortcut, G1, masked, CPU |
| Q1-B / Q11a ub512 c8192 | shortcut: dense (f16/f16 or prefer_dense) or `av` sparse; then compact CB G4 n_q 512, V packed for n_kv < 32768 and direct beyond | masked, CPU |
| Q1-C ub64 | shortcut (32 ubatches/chunk, trace count), then compact CB G1 n_q 64 | G4, masked, CPU |
| Q3-B ub256 c32768 | compact CB G4 n_q 256; V direct for n_kv ≥ 16384 (q8_0/q4_0, and f16/bf16 under CAPPED) | masked, CPU |
| Q11b ub16 c2048 | shortcut `av` sparse G<shortcut policy> n_q 16 (or dense) | masked, CPU |
| Q10 text / image (unified 2-slot) | non-compact `te` rows → sparse; compact on the L = 4k control | masked, CPU |
| Q7 / MTP server | main: compact G4/G1; draft: shortcut, G1 nq 1 and nq 2–8 past 2051 | masked, CPU |
| P1 q8_0 ub512 pp4096 d0 | shortcut for n_kv ≤ 2048, then compact G4 packed | masked, CPU |
| P2 d32k ub512 | compact G4, V direct (n_kv ≥ 32768) | masked, CPU |
| Q12 / AT-D2/D4 `-nkvo`, `-ngl 0` | CPU SPARSE_ATTN > 0, CPU shortcut attention > 0 (if pinned, §10) | GPU SPARSE_ATTN (unless the named host-KV-copy route is active), ROCm0 QSA FA, `cache_[kv]_l*` split inputs to ROCm0 |
| AT-C16a layout 3 (W8) | non-compact on slot-B text ubatches; strip-view bias/kq_mask new > 0, reuse 88/src/graph | — |
| AT-C16b layout 3 (W9, W13) | non-compact on slot-B text ubatches of ≤ 512 tokens; strip-view new 1 / reuse 11 per src per graph | — |

- **Proof of reach at op level.** The suite-level census accumulator (§14.1) checks every compiled key; forced tiers get their own runs. This replaces the PR's "≤ 512-query prefill tests that run decode" defect (L11).
- **Hooked-arm census.** The QPIN hook splits the graph at `ffn_moe_topk`, which breaks the HIP topk_moe fusion (ggml-cuda.cu:3237, :5582; ggml-backend.cpp:2083-2116), the likely cause of L68's 0.0088 (U20). Every hooked Q-arm runs the backend census and must show the same sparse-attention key set (EPI gated, qsa_expand) as the unhooked production run.

---

## 15. Quality protocol

### 15.0 Measured bands from W12 (binding; copied by the orchestrator from IMPLEMENTATION.md §W12 and scripts/qsa-validation/summaries/w12_bands.md on qsa-rewrite 52c2c1a95)
These replace any provisional value written further down in this section.
- **H1.** B_T,4096 = 2.0e-4 for bf16 and both mixed pairs.
- **Direct form.** f16 1.403e-4, q8_0 2.310e-4, q4_0 1.441e-3, bf16 1.032e-4, q8_0K/f16V 1.807e-4, f16K/q8_0V 1.623e-4. q4_0 saturation is 1.058e-3, which explains the larger q4_0 band.
- **H2 (Q4 agreement thresholds).** bf16 0.997140, q8_0K/f16V 0.996742, f16K/q8_0V 0.997121.
- **H4.** B_4096@32k = 1.801e-4.
- **Vulkan.** vref8t0 is invalid. B_vk = 3.214e-4 against the new W0-Vulkan reference.
- **U40.** B_512,lo = 2.073e-4.
- **Q11b.** 2.03e-4. The ub16 arms remain UNCALIBRATED/UNPROVEN, because PR219 ub16 is +6.7e-3 from ub2048 with MoE pinned, cause unknown.
- **Self-replay.** Gated on QKL_HASH equality, not on the old 3.7e-7 value, which is reported only.
- **PR219 arm.** Use ~/dev/wt_lab/w0lab12, not w0lab. Use the R1-UB chunk values; R1-Q8's 1.375064e-4 came from a graph altered by the attn_output-23 hook.

- **References.** One f16/f16 ub4096 QPIN reference per context, recorded on the PR219 arm **W0-lab** (§0, W12) with `LLAMA_FN_LAST_ATTN_TRIM=0` (rows_absent = 0 in all 41 R1-UB replay arms): c8192 × 8 (Q1, Q8, Q11a, Q12), c8192 × 16 (Q4, R1-E; chunks 8–15 also serve as the c8192 holdout), c32768 × 8 with `QKL_TAIL=4096` (Q3), c2048 × 16 dense f16 b/ub 2048 (Q11b), and a **W0-Vulkan** f16 ub4096 trim-0 reference (AT-W12-BANDS V1, built from W0-lab `build-vk`) for the Vulkan arm. Pins are keyed by position, so they replay at any ub ≥ 16 (QPIN skips ubatches < 16 tokens, qpin.patch:200).
- **Every pinned arm runs hooked**, with per-chunk output at %.9e. Pinned run-to-run noise is 0 (R1-UB 11 self-pairs; R1-Q8; R1-E), so for pinned gates every SE below is chunk-to-chunk variation and re-running cannot flip a verdict. **Determinism does not remove sampling variation:** a band verdict still depends on which chunks and prompts were sampled. Hence the two-stage rule below. **Q6 and Q10 are not pinned**: Q6 is deterministic run to run (R1-E) but its τ is a sample statistic; Q10's PR219 arm runs the masked route, which is not deterministic run to run (R1-CB te_s1 vs te_s2 0.96–1.42e-2; R1-NKVO 0.0103), so Q10 verdicts use the median of ≥ 3 PR219 repeats (AT-Q10Q6).
- **Never** use the PR219 masked route as a paired KLD reference without pins (R1-NKVO), **except in Q10**, which has no pinnable reference (non-compact multi-sequence server workload) and is scored with the AT-Q10Q6 noise model instead of a paired band.
- **No e2e MDE below the pinned saturation level (~1e-4) is claimed** (R1-Q8: a 1e-7 relative perturbation gives 9.39e-5). Q5 is primary for kernel-local numerics.
- **Band floor (P2-4, P2-23).** PR219 runs q4_0, bf16, q8_0K/f16V and f16K/q8_0V on routes where a reorder can be **exactly 0**: q4_0 is bit-identical between ub512 and ub4096 (MEASUREMENTS.md:3346-3348: one `dense_tile_vec_mma` kernel at every ub) and between ub2048 and ub4096 (R1-E table A); bf16 takes the same masked dense route (PR qsa-prefill.cu:613 kv_ok F16/Q8_0 only; qsa-decode.cu:91-92; log MEASUREMENTS.md:1667-1668), so its reorder is likely 0 (inferred); the mixed pairs probably run masked qsa_prefill at n_q ≥ 128 and may be nonzero (inferred from the code, no census). A rewrite that is not bit-identical cannot pass a band of 0. **Rule: every per-type band is floored, B_T = max(measured B_T, max over U of the B_U of the same ub tier), and a type whose PR219 reorder is bit-identical takes the floor, never 0.** The direct form uses max(that type's ub2048 reorder, that type's ε = 1e-7 saturation value) + 3·SE (q8_0 saturation 9.39e-5, R1-Q8; q4_0 measured in AT-W12-BANDS H3). The same floor applies to Q4's B_T and to the Q11b tier band.
- **Two-stage rule (P2-8).** Stage 1 scores every gate on its pre-registered data. A gate that fails stage 1 is re-scored **once**, with the same band and rule, on pre-registered held-out data: c8192 gates (Q1, Q8, Q11a, Q12) on chunks 8–15 of the c8192 × 16 reference; Q4 on chunks 16–31 of a c8192 × 32 reference (wiki.test.raw supports 262k tokens at c32768 × 8, so × 32 at c8192 fits); Q3 on 8 chunks of the C10 corpus (877,755 tokens) with its own c32768 reference (whether wiki.test.raw holds c32768 × 16 is UNPROVEN, so it is not assumed); Q6/Q10 on a fresh calibration and a fresh arm; Q7 on ≥ 14 new L/X prompts. **The PR is blocked only if both stages fail.** A stage-2 pass is reported in the PR text with both values. The confirmation does not apply to Q5 (op-local, deterministic, k ≥ 6.1) or to any "= 0" determinism check.
- **Expected family-wise false-fail rate (P2-8).** After AT-W12-BANDS and AT-Q10Q6, each band gate's false-fail probability under a benign change is estimated from its calibration (the fraction of the calibration's own benign pairs, leave-one-out, that exceed the band); the product rule under independence gives an upper-bound family-wise rate for stage 1, and its square (both stages) for blocking. Both are reported in the PR text. Q7 at n = 14 has P(pass | Δ = 0) = 0.986 per draft type and ≈ 5.6 % family-wise over 4 types (skeptic Monte Carlo, 200k draws, σ 0.921).
- **Delete** every KLD base and pin set after its campaign (sizes: 22 GB at c8192 × 8, 43 GB at × 16, 40 GB at c32768 × 8, R1-UB, R1-E).

| id | gate | rule | source |
|---|---|---|---|
| Q1 | **e2e screen**, QSA+MoE pinned, new(T)_U − PR219(T)_U, c8192 × 8: **Q1-A ub4096** for T ∈ {f16, q8_0, q4_0, bf16, q8_0K/f16V, f16K/q8_0V}; **Q1-B ub512**, **Q1-C ub64** for T ∈ {f16, q8_0, q4_0} | mean per-chunk diff ≤ B_T,U = max(measured B_T,U, B_U), with **B_4096 = 2.0e-4** (the max chunk of \|q8_ub2048 − q8_off\|, R1-UB Decision 3; *not* a mean + 3·SE, which over f16 would be 1.116e-4 + 3 × 1.188e-5 = 1.47e-4), **B_512 = 2.03e-4, B_64 = 2.75e-4** (R1-UB d2). bf16 / mixed: max(their AT-W12-BANDS H1 band, B_U). **Direct form** KLD(new(T) ‖ PR219(T) pinned output) ≤ max(T's ub2048 reorder, T's ε = 1e-7 saturation) + 3 chunk SE (q8_0: 1.63e-4 + 3 × 2.28e-5 = 2.31e-4; others AT-W12-BANDS; never 0). `mean ≤ 3·SE_paired` is printed, never gated | R1-Q8 d3; R1-UB d2–d3; AT-W12-BANDS |
| Q2 | floor context | report each arm's ratio to the f16 ub2048 pinned floor (1.116e-4, R1-Q8) | C8, R1-Q8 |
| Q3 | depth | **Q3-A** c32768 × 8, ub4096, `QKL_TAIL=4096`, f16/q8_0/q4_0: mean diff ≤ **B_4096@32k = max over T ∈ {f16, q8_0, q4_0} of (mean + 3·SE) of PR219's paired ub2048 − ub4096 at c32768 × 8** (AT-W12-BANDS H4; **UNPROVEN until measured**, U39; context: q8_0 − f16 @32k × 8 = +1.453e-4, SE 3.37e-5, so mean + 3·SE 2.46e-4; q4_0 paired SE at 32k 8.42e-5, max chunk 5.91e-4; C17 f16 ub2048 floor at c32768 × 2: chunks 1.4e-4 and 0.62e-4, no SE). **Q3-B** same at ub256: ≤ B_256@32k = 3.34e-4 (G4 n_q 256; direct V at n_kv ≥ 16384) | R1-UB; R1-E (Q3); C17 |
| Q4 | selection agreement, c8192 × 16, QSA+MoE pinned, per-chunk `QPIN_QSA_CHUNK` | mean_c(new_T − PR219_T) ≥ −B_T, B_T = max(\|PR219 kind-1 ub2048 shift\| + 3·SE, the smallest measured S_fast B_T): f16 KV f16 idx ≥ 0.99735; f16 KV q8 idx ≥ 0.99693; q8_0 KV ≥ 0.996625; q4_0 KV q8 idx ≥ 0.985528. **The PR219 partner is forced to kind 1 (`LLAMA_QSA_SELECTED_KEY=1`) for q4_0, bf16, q8_0K/f16V and f16K/q8_0V**, never tk (the tk estimator's structural offset is kind 1 − tk = −4.01e-4, t −12.0, R1-E table B); bf16 and mixed bands from AT-W12-BANDS H2. Plus, inside the new build on kind 1: q8 idx / f16 idx ≥ 0.99 (measured 0.999802) | R1-E Q4; AT-W12-BANDS |
| Q5 | **primary, op-local** | (i) plan whole-dst bytes CPU = HIP = Vulkan on C3 captured rows and the kept C18 rows; (ii) attend NMSE(GPU, CPU op) ≤ the §14.2 band of the census-reported class: fixture rows on the wiki d4k fixture (recaptured with C18's command, byte-identical rows, R1-A step 0; delete the .qkv after) and on SYN 262k, UNI rows on the ported shapes, with the PR-verbatim kernel in the same run as a bitwise reference; (iii) the planted op-level defects must exceed the band in every cell | R1-A; R2-BOP |
| Q6 | images | C4 p1/p2: KLD(new_T CB-or-TE ‖ PR219_T TE) at ub8192 ≤ τ_T. **τ_T = mean + 3·SD of ≥ 6 independent PR219 noise draws** (ub pairs {8192, 6144, 4096, 3072, 2048, 1536, 1024} × {TE, CB}, AT-Q10Q6 (3)); if that rule's leave-one-out false-fail rate on the calibration exceeds 1/6, τ_T = the max of ≥ 19 draws (a 95 % quantile) instead. Provisional, superseded by AT-Q10Q6: the round-1 sample maxima p2 f16 2.554e-3, p1 f16 2.878e-3, p2 q8_0 2.872e-3 (R1-E: p1 f16 TE vs CB at ub2048 is already 85 % of τ). τ_p1,q8_0 UNPROVEN until measured. Content check (headline, newspaper, date). `~/repro_img` A/B/C answer 2749/12749/26999; B65t answers 12299 | R1-E Q6; C4; L64; C19; AT-Q10Q6 |
| Q7 | MTP | C2 harness, greedy: per draft type T ∈ {default f16, q8_0, q4_0, bf16}, paired per prompt new(T) − PR219(T) (q4_0 partner: PR219's masked draft `g1_repro_q4d`), one-sided 95 % LCB ≥ −1 pp on **≥ 14 L/X-class prompts** (prompts that pass n_kv 2051 in the draft). Sizing: √n ≥ σ·(z_0.9875 + t_0.95) gives n ≥ 13.6 at σ 0.921 pp for ≤ 5 % family-wise false fail over 4 types; at n = 14 the one-sided margin is t_0.95,13 · 0.921/√14 = 0.436 pp (at n = 8 it was 0.617 pp; the round-1 "0.77 pp" was the two-sided half-width). The round-1 data give LCB −0.930 at n = 8, 0.07 pp from failing. S0–S5 (6 structural zeros, R1-E) are reported as shortcut coverage and **excluded from the LCB**. vs f16 draft: reported only (q4_0 −2.55 pp). bf16 SD UNPROVEN; trunk text identical across draft types | R1-E Q7 |
| Q8 | planted defects, downstream of the pin (post-`qpin_rows` hook, `r1q8_lab.diff`), on − off, same type, same pins, c8192 × 8, on the W7 build | **D1** (drop last tail cell L23): LCB95 > δ_T = 7.45e-5 required (PR219: +8.21e-4). **D2**: sign test ≥ 7 of 8 chunks > 0 (PR219: 8/8 for f16 and q8_0, `EV/r1q8/analysis_vs_f16ref.out:20,25`) and \|new − PR219\| ≤ 3·√2·SE_PR219 = 1.86e-3. **D3: report-only** (its Q1-form value is cross-term noise with per-chunk diffs ±1e-5 of mixed sign; a fresh build lands within 1 SE of PR219's +4.9e-7 with probability ≈ 0.52; Q5 covers it) | R1-Q8 |
| Q9 | unpinned context | reported, not gated (L65) | C8, C9 |
| Q10 | non-compact multi-sequence, unified 2-slot (`r1/text_campaign.sh`, `r1/img_campaign.sh`, new vs PR219, unpinned) | **Control L = 4k:** (a) new rerun vs new = 0 (determinism of the new build on this layout); (b) new vs each PR219 repeat ≤ **floor_ctl** = mean + 3·SD of the 3 × 3 cross pairs PR219 build4 vs PR219 built with `QSA_DECODE_MAX_QUERIES 127` (a compile-time define, qsa-decode.cuh:5: the unpinned kernel-change floor of this layout, AT-Q10Q6 (1)). Bit identity with PR219 is **not** required: the new build sends the 200-query answer batch to G4 (D3 default) while PR219 sends it to qsa_decode (fattn.cu:1154), and an unpinned kernel-family change costs about 6.86e-3 (R1-AV campaign B). **Cells L ∈ {4k+1, 4k+2, 4k+3} and image p3:** per-request KLD over 2 × 100 fixed tokens between new and the **median of ≥ 3 PR219 repeats** ≤ τ = mean + 3·SD of ≥ 6 PR219 noise draws of that cell (AT-Q10Q6 (2); quantile fallback as Q6). Free answer correct on headline, newspaper and date. KLD 0 is used only for new-vs-new (CB vs TE producer in the same new binary, which is what R1-CB measured) | R1-CB d4; AT-Q10Q6 |
| Q11 | shortcut route | **Q11a** c8192 × 8, b/ub 512, `QKL_TAIL=8191`, `QKL_SPLIT=2048`, QSA+MoE pins, pairs q8_0, q4_0, bf16, q8_0K/f16V, f16K/q8_0V, f16: mean diff ≤ B_512 (floored) for lo (< 2048), hi and all; **the lo-region band is borrowed** from B_512, which was scored over positions 4096–8191 (labelled, U40; measured in W12 with AT-W12-BANDS H1's arms split at 2048). **Q11b** c2048 × 16, MoE pins, dense f16 b/ub 2048 reference, the same pairs at ub512 plus q8_0 and f16 at ub16: mean diff ≤ max(PR219's own ub512-vs-ub2048 shift + 3·SE, B_512) (shift measured in W12; band UNPROVEN until then). **Q11c** `-c 1024 -b 2048` (np2 split) equals `-b 1024` (np1) to 0 per sequence, q8_0 and q4_0; plus a split-stream decode case at n_tps 1. Prototype: all 22 cells ≤ 0 or +0.42 SE (R1-AV) | R1-AV d1 |
| Q12 | host-KV routes (`-nkvo`, `-ngl 0`) | **Primary, pinned (P2-6):** QSA+MoE pinned against the c8192 × 8 reference, ub4096 and ub512, f16 and q8_0: mean per-chunk diff new(T, `-nkvo`) − PR219(T, `-nkvo`) ≤ B_U (floored); PR219 `-nkvo` runs the GPU kernels on copied K/V (R1-NKVO), so the paired diff carries the CPU-vs-GPU attention arithmetic. Both arms rerun-KLD = 0 first. **Coarse screen, unpinned** (labelled coarse): mean KLD vs the same binary's GPU sparse route ≤ the R1-NKVO pinned-CPU value + 3·SE: f16 ≤ 0.01587, q8_0 ≤ 0.01563 (c4096 × 4; the round-1 point thresholds 0.014361 / 0.014007 had no margin, and both unpinned q8_0 arms, 0.014320 and 0.014236, already exceed 0.014007). **`-ngl 0`:** the same pinned rule; its unpinned screen has no measured basis (R1-NKVO measured `-nkvo` only) and is reported only | R1-NKVO; AT-D2/D4 |
| Q1-vk | Vulkan, per S_fast type | new-Vulkan(T) vs **W0-Vulkan(T)** under the W0-Vulkan reference's pins: mean diff ≤ B_vk = mean + 3·SE of W0-Vulkan's own f16 ub2048 − ub4096 (AT-W12-BANDS V2 (iii)); **UNPROVEN until measured** (the round-1 B_vk = 6.4e-4 came from the PR219+#240 tree, which lacks master's default-on Vulkan GEMM changes). The old `vref8t0` reference is used only if AT-W12-BANDS V2 (i) shows the master delta is below 6.4e-4 and not within 3·SE of it. Never paired with new-HIP (cross-backend 5.5e-3). bf16 on Vulkan UNPROVEN until run. One rerun per arm budgeted (device loss seen once, cause UNPROVEN) | R1-UB d4; AT-W12-BANDS |

Per-chunk SEs come from full-precision QKLD_CHUNK output: c8192 × 8 (R1-UB table E), c32768 × 8 (R1-UB), c8192 × 16 for agreement (R1-E). Not gated e2e after these arms, and left to Q5 and Q7: G1 at n_q ∉ {16, 64} (n_q 1 decode and MTP nq 2–8 cannot be pinned), G4 at n_q ∈ (128, 512) other than 256/512, direct V at n_q 512 (needs a c65536 tier, which llama-perplexity cannot run here, C16).

---

## 16. Performance protocol

- **Workloads.** Text = the C10 corpus (877,755 tokens, FNV-1a 48eaad26c3165780) through the C10 llama-bench patch. Random = unpatched llama-bench (L54), reported as a stress row. Named random cells where quantized types must not fall below f16 are reported explicitly (e.g. rand d131k q8_0 X+P 0.891 of f16 at kernel level, C18).
- **Design.** Interleaved processes R N R N R N R in **one campaign**, where R is the reference measured in that campaign (PR219 = W0 build for neutrality and ratio cells; base 13a279bfa for P1); a warm-up llama-bench instance precedes the first timed process of every cell (the d0 ub4096 pp512 warm-up artifact, C10 [−12.95, +0.15] even with rep 0 dropped). 5 reps per process, rep 0 dropped. Every sample records KFD `evicted_ms` and `pgmigrate_success` (C7 sampler); eviction samples are excluded and rerun, and their rate is gated by P6.
- **Statistic (O15, O39).** Per process, the **mean** of reps 1–4 (the median is bimodal under the fixed rep-position ramp: skeptic re-bootstrap of `EV/c10/logs/d{32768,131072}_f16_*_text.json`, script `EV/bs_skeptic.py`; with the mean at 2× processes, d131k ub4096 pp512 gives [−0.87, +0.95]). Ratio new/R with a 95 % CI from the hierarchical bootstrap (processes within build, then reps; 4000 draws, seed 1; `analyze.py`). Ratios of ratios (P3) bootstrap all four arms of one interleaved campaign jointly.
- **Verdict and stopping rule.** A cell with threshold θ: PASS if CI LB ≥ θ; FAIL if CI UB < θ; otherwise rerun with 2× processes, then 4× (28 per build); still unresolved → FAIL. The rule terminates.

| id | cell(s) | threshold θ (on new/R unless stated) | source |
|---|---|---|---|
| P1 | text q8_0 ub512 pp4096 d0, new vs base 13a279bfa in the same campaign | **≥ 0.995** (C10 decision 2: CI lower bound ≥ −0.5 % vs base; C13 predicts +1.6 %). No absolute t/s term | C10 d2, C13 d4 |
| P2 | text f16, every C10 cell (d0 pp 512–8192; d32k/d131k pp 512, 4096; ub 512, 4096), new vs W0 | ≥ 0.99 (non-inferiority δ 1 %) | C10 d3; O15 |
| P3 | text, per (depth, ub, pp): (new_q8/new_f16) / (W0_q8/W0_f16), four arms interleaved | ≥ 0.99, **and** an absolute floor new_q8/new_f16 ≥ 0.90 in every cell (the P4 floor; C10 d4 names pp512 d131k, PR 0.896, as "the target the rewrite has to close") | C10 d4 |
| P4 | text q4_0, bf16, q8_0K/f16V, f16K/q8_0V and each S_generic type, new_T / new_f16, same cells | ≥ 0.90 (PR219's worst q8 cell 0.896, C10 d4). An S_generic pair below 0.90 still ships (support is not dropped), and the PR text lists it as "not first-class" with its measured ratio | C10 d4; U8 |
| P5 | decode tg128, text depth; **tg with MTP** | shortcut (d0/1024/1900): new_T vs W0_T ≥ 0.98 for every S_fast pair, else the pre-registered `prefer_dense` flip (AT-W7PERF); depth (d32k/d131k): new_T vs W0_T ≥ 0.99 for f16 and q8_0; **new_T / new_f16 ≥ 0.94** for every S_fast pair incl. q8_0K/f16V and f16K/q8_0V (C18 d4: q8_0 +11.9 % at ub512, about +6 % at tg). **MTP cells (P2-9, minor note):** llama-server tg with the MTP sidecar (verify width 2–8, G1 at depth in the main context), W0 vs new, f16 and q8_0 main KV, draft default and `-ctkd q8_0`, at d0, d1024 and d32768: new / W0 ≥ 0.98 and new_q8_0 / new_f16 ≥ 0.94 | C1 d2; C18 d4; O13; P2-9 |
| P6 | stall | (a) in every perf cell, any eviction-free sample < 0.9 × its cell median fails; (b) eviction-sample rate new vs base, interleaved, ≥ 30 reps, at ub512 pp512 d0 (f16, q8_0, q4_0) and ub4096 pp2048 d32k: **reported only** (at the measured rates, PR 4/286 vs base 1/128 reps, C7, a one-sided Fisher test at ≥ 30 reps has essentially no power, so it cannot fail; the PR text states the counts and the rate ratio it could have detected) | C7 decision; O19 |
| P7 | kernel level | the W3/W4 rows (same-run references, stated margins); plan ≤ 1.0 ms/layer (C3 d3) | C3, C18 |
| P8 | Vulkan | **AT-B1** (new/W0 Vulkan, interleaved) on d4096, d32768, d131072 × ub {512, 4096} × pp {512, 2048}: f16 ≥ 0.99; q8_0, bf16 and each S_fast pair ≥ 0.95 × new f16; **a miss after every AT-B3 arm is a PR-blocking FAIL that goes to the owner with the ratios** (P2-11). Plus **AT-B3** op-level per-pair arms, AT-B4 shortcut and AT-B7 portability | O14, O22; P2-11 |
| P9 | images and MTP (server prompt t/s) | requests A/B/C, f16 and q8_0, 3 interleaved server runs per build (W0 vs new): ≥ 0.97 with the stopping rule (δ 3 %; server noise is UNPROVEN, C19 "one sample per run", so the first campaign measures it) | L64; O39 |

- **Memory gates** (deterministic, `sched_reserve -lv 4`; C16 d2):
  - **M1:** for every T ∈ S, the ROCm0 and Vulkan0 compute buffer ≤ new f16 + 1 MiB, at -c {32768, 131072} b/ub {4096, 512}, main and MTP; plus `np 2` (split streams) and default-server (`n_seq_max` 4, with the non-compact variant) rows; on Vulkan also `prealloc_size_x/y/split_k` logged (VK_LOG_MEMORY) per type with the same ≤ f16 + 1 MiB rule. The variant's measured q8_0-over-f16 ROCm0 excess (+232.27 MiB variant alone, R2-RES) must be explained or removed for M1 to pass on those rows (U37).
  - **M2 (total footprint, O18):** (new compute buffer + new pool high-water) ≤ (W0 compute buffer + W0 pool high-water) + 1 MiB, both by the C7 shim in the same campaign, per device, main and MTP, f16 and q8_0, at -c {65536, 131072} ub4096. Additionally, new pool high-water ≤ W0 pool high-water − W0's QSA pool share (qsa-prefill.cu:678-707, qsa-decode.cu:130/145, attributed by the shim's call stacks). Prediction by arithmetic: equality (the QSA pool items move into the compute buffer one for one, §7); whether gallocr absorbs them under the existing peak is UNPROVEN (U32). **Pre-registered fallback if M2 fails:** the attend workspace moves from the dst over-allocation to a separate workspace tensor that is the attend's own src, freed right after the attend (today tying it to the dst keeps it alive until the dst's consumer is allocated; an in-place MUL frees it via `ggml_gallocr_free_extra_space`, ggml-alloc.c:769/777), and M2 is re-run once; a second fail goes to the owner with the numbers.
  - **M3:** 0 hipMalloc in any rep window after warm-up (C7 shim).
  - **M4:** `LLAMA_STRICT_RESERVE` = 0 violations on the W9 list (R1-RES recipes included), and 0 graph-build-check errors on every run that is expected to succeed.
  - **M5 (default server, O35, P2-14; R2-RES):** llama-server without `--parallel` (auto 4 slots, unified), the default (no -c/-ub), -c {131072, 261888} × ub {1024, 4096, 8192}, f16 and q8_0, main and MTP, per backend: **new reserve ≤ max(PR219 reserve, V) + moved pool items + 1 MiB**, with K = n_ctx / s, T = min(ub, 512) (S = 1), V_host = 3·K·T + 12·K·s + 51244·T + 32 B (+0.29 MiB in main), V_ROCm0 (unified) = 6412·K + 141.36 MiB main / 6412·K + 70.03 MiB MTP, split 7704·K + 153.36 / 6168·K + 70.03 MiB (UNPROVEN beyond its 2 points). Instantiated (main ROCm0 / host; MTP ROCm0 / host, MiB, each + 1 MiB): c131072 ub4096 1609.88 / 218.79, 1465.23 / 218.52; c261888 ub8192 3315.30 / 411.92, 2938.45 / 411.64; default server (c262144 ub512) 1744.36 / 412.29, 1673.03 / 412.02; c131072 ub1024 unified 942.86 / 218.79, 871.53 / 218.52 (R2-RES, measured q8_0; f16 measured lower at S = 2, f16 at S = 1 UNPROVEN). Runtime max ≤ reserve (AT-C12).
  - **M6 (f16 at depth, OWNER):** default CAPPED: new f16 reserve ≤ PR219 reserve + PR219 pool high-water at -c 524288 and -c 1048576, ub4096 (AT-C17).
  - **Before-numbers in the PR text** are compute-buffer MiB, not OOM (C16 d1).

### 16.5 Prove-phase acceptance tests (deferred labs; run on the rewrite, scope unchanged)

**AT-A — G1 vs G4 at n_q 128–512 (D3), in W4.**
- Where: `~/dev/wt_rewrite` once W4 exists. Kernel timing of the rewrite's kernels with the C18 protocol (3 warm-ups, hipEvent per launch, mean of 12 per-layer medians, two processes r1/r2 interleaved). Fixtures: kept C18 rows (`~/dev/wt_lab/c18/cap`: wiki/rand d4k and d131k) with SYN data, plus SYN:1.8 at 32768 and 65536 cells.
- Shapes: n_q {2, 4, 8, 16, 32, 64, 96, 127, 128, 256, 512}, plus the trimmed layer n_q 512 at ub4096 pos0 on d131k.
- Arms per T ∈ {f16, bf16, q8_0, q4_0}: (a) new G1 staged V; (b) new G4 packed + PACKV-<T> over the full n_kv; (c) new G4 direct V (f16/bf16 included); (d) plan-stage cost at that n_q (C3 harness).
- E2E confirmation: C10-style f16 ub512 pp512 at d131k, measured policy vs forced G4, §16 design.
- **Pass criterion (fixed now):** G1 is picked where it beats min(G4 packed + pack, G4 direct) by > 3× the p90 run-to-run noise; elsewhere G4. The HIP `group` proc encodes the fitted (n_q, n_kv, T_v) rule; W4's rule is two-way. D8 is re-derived at n_q 512 from (b) vs (c). The e2e cell must be ≥ 0.99 under §16. The reserve covers any G through the §5 max rule. Prediction to check (arithmetic, UNPROVEN: no ledger row gives the G4 side directly; it is derived from C1 G4 n_q 512 at n_kv 16384 plus PACKV-f16 1.771 ms at 131k, MEASUREMENTS.md:720): f16/bf16 at n_q 512 d131k pick G1 (3.37–3.60 vs about 3.85–4.6 ms); q8_0/q4_0 pick G4 direct; at d4k every type picks G4. **D8's CAPPED 64·n_q direct-V switch is re-derived at n_q 128 and 256 as well** from (b) vs (c), because the PR never executed G4 at n_q 128–512 (n_q ≤ 512 went to qsa_decode, C13) and C18 measured the crossover only at ub4096; Q3-B relies on direct V at n_kv ≥ 16384 with n_q 256. The trimmed-layer row becomes a permanent W4 gate row.

**AT-B — Vulkan (W6/W7).**
- Build: build-vk with C11's flags; W0 reference = W0-lab `build-vk` (§0: the actual base + 68da4191d b60447506 94e086847 + the W12 patches) with the C10-patched llama-bench linked against it (`c11/build_bench_vk.sh`), in `~/dev/wt_lab/w0lab`. All runs under `flock ~/gpu.lock`; one rerun per arm budgeted.
- **AT-B1 (P8, P2-11):** C10_TEXT, `-ngl 99 -fa 1 -b 4096 -ub 4096,512 -p 512,2048 -n 0 -d 4096,32768,131072 -r 5 -o json`, f16, q8_0, bf16 and each S_fast pair; ≥ 4 new processes interleaved with ≥ 4 W0; §16 statistic. Pass: f16 new/W0 ≥ 0.99; every other pair ≥ 0.95 × new f16 in **every** cell. A pair that misses after all AT-B3 arms is a **PR-blocking FAIL that goes to the owner with the measured ratios** ("ships anyway" applies only to S_generic, P4). Reason for the grid: on HIP the q8/f16 gap is worst at pp512 and at depth (PR 0.944 at d32k, 0.896 at d131k for ub512 pp512, against 0.929–0.966 for pp4096 at d131k, MEASUREMENTS.md:1169, :1251). C11 absolutes (1259.9 / 919.6) are history only.
- **AT-B2 (op reference in-build):** `GGML_VK_PERF_LOGGER=1` at C11's shape (n_kv 34816, 11 × 2048 q + 1 × 512 q, GQA 12) and at one d131072 text cell, new SPARSE_ATTN f16 vs the old sel_cm pipeline in the same build, ≥ 5 reps × 3 processes, text and random. Pass: new f16 op ≤ 1.10 × the in-build sel_cm median; otherwise §17's Vulkan rollback, evaluated against this multi-sample number.
- **AT-B3 (quantized staging):** `GGML_VK_PERF_LOGGER=1` op timing of SPARSE_ATTN with `GGML_VK_SA_FORCE=vdirect|vstaged|vpack` (pack time included), q8_0, q4_0, bf16, at C11's shape (n_kv 34816, 11 × 2048 q + 1 × 512 q, GQA 12) and at d4096 and d131072, text and random, 3 interleaved processes, ratio to new f16 of the same build; each arm's Vulkan0 `sched_reserve` recorded against f16. Ship per type the fastest of vdirect / vstaged; **vpack ships for a type only where it beats the better of the two by more than the measured noise and M1/AT-B5 still holds (T ≤ new f16 + 1 MiB), or f16 reserves the same bytes**; otherwise it is not shipped (a quantized-only pack would put T above f16 by up to min(n_kv, 64·n_q)·1 KiB per tensor, about 256 MiB for K+V at n_kv 131k [arithmetic], against M1). Context: the HIP G1 analogue chose staged V for every type (C18 d2, −30 % to −59 % vs direct; quantized +6 % to +12 % of f16 at the op), and C18's "packed V beats every direct V" is G4-only, so whether vpack wins on Vulkan G1 is UNPROVEN either way; the attention op is 5.9 % of the Vulkan graph at C11's shape (8.35 of 1591 ms, C11 op-level). Diagnostic gate: ≤ 1.15 × new f16, or ≤ the HIP C18 ratio + noise (HIP direct D+D vs f16, ub512 wiki d4k: q8_0 1.23×, q4_0/bf16 1.67×). Binding gate: AT-B1.
- **AT-B4 (shortcut):** op level n_q {2, 4, 8, 16, 32, 64, 127, 512, 1024, 2048} × n_kv {512, 1024, 2048}, and e2e `-p 2048 -n 0 -d 0 -ub 64,512,1024,2048`, sparse all-visible vs dense FA of the same pair T in one build, the 0.98 flip per (pair, n_q band) of D4; decode tg128 at d0/d1024/d1900, 3 interleaved processes; q8_0 and q4_0 with `GGML_SCHED_DEBUG` and the CPU-launch counter. Pre-registered: any cell where sparse(T) < 0.98 × dense(T) makes the Vulkan `prefer_dense` proc return dense for T (for pairs Vulkan dense FA supports; Vulkan bf16-mixed stays sparse, L40). Also: 0 CPU FA / CPU sparse nodes, every attend node on Vulkan0; decode at depth ≥ W0 of the same type.
- **AT-B5 (memory):** Vulkan0 `sched_reserve -lv 4` per type at -c {32768, 131072}, ub {4096, 512}, main and MTP, new vs W0. Pass: M1 (every T ≤ new f16 + 1 MiB); new f16 ≤ W0 Vulkan f16 at the same flags (history: C11 1494.23 / 226.22 MiB).
- **AT-B6 (quality):** Q1-vk.
- **AT-B7 (portability, P2-16, W6).** (0) No GPU: confirm by reading that `flash_attn_sel_prep.comp` uses only `barrier()` and shared memory; record the lavapipe limits with `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json vulkaninfo --summary` (subgroup size and size-control range). (1) Under flock: `test-backend-ops -o SPARSE_ATTN_PLAN`, then a separate `-o SPARSE_ATTN --fail-unsupported`, on lavapipe and on RADV with the portable tier forced at required subgroup size 32; one op per invocation, never `-j > 1`. (2) A check that supports_op returns false, never a null-pipeline dispatch, when no pipeline exists (forced capability mask). Pass: both op runs 0 failures and 0 "not supported" for S_fast on both devices; supports_op false when the pipeline is absent; every device class that cannot run any tier is listed in §19 with its error text.

**AT-C — reserve (W4/W9).** All with `LLAMA_STRICT_RESERVE=1 -lv 4`, under flock.
- **AT-C5:** `llama-bench -ub {64, 65, 66, 127, 128, 129} -b 4096 -p 4096 -d 4096,32768,131072 -n 32`, f16/q8_0/q4_0; explicit 127- and 65-token final ubatches at d131k and c4096; `llama-perplexity c8192 -ub 127`; an MTP server context with ub < 128; **a forced G flip between reserve and runtime** (reserve with the policy's G, runtime with `GGML_CUDA_SA_FORCE=g4` at n_q {65, 127, 129}) with `GGML_SCHED_DEBUG_REALLOC` set (an unexpected re-plan aborts, ggml-backend.cpp:1945-1975). Pass: 0 violations; 0 re-plans; reserved partial size ≥ the §5 formula max over n_q' ≤ n_q and over G; reserved plan dst ≥ max over G of plan_nbytes (§2.1).
- **AT-C17:** init-only llama-server -c 524288 and 1048576, -ub 4096, f16, new and PR219; for PR219 also one pp4096 ubatch just below and one just above 262144 cells under the C7 hipMalloc shim (pool high-water); then `-np 4` on the new build. Pass (owner decision deferred, default CAPPED): new f16 ≤ PR219 reserve + PR219 pool high-water (M6); under the always-pack trait, new f16 − (PR219 reserve + pool high-water) equals the stated delta (+512 / +1024 MiB instantaneous, +256 / +768 MiB vs the retained pack) within 1 MiB; partials and pack scale as the stated n_stream formulas.
- **AT-C12:** default-server grid on both builds: the default (no -c/-ub) and -c {131072, 261888} × ub {512, 1024, 4096, 8192} × {f16, q8_0}, plus `-np 2 -c 131072 -ub 1024` (split), sched_reserve per backend, main and MTP; runtime peak with Signal-3.8 + mmproj + MTP, requests A and B, and two concurrent requests sharing a prefix (seq_cp). **Image-slot layout (P2-1, `r1s/r1s.sh`-style default server, 4 slots unified):** A = S0_img, then a second ≥ 8k-token turn on A while B = T_deep is resident; census must show A's ubatches non-compact (TE, re-split to ≤ 512 tokens) and B's compact; Q10-style per-request KLD of A's fixed tokens, new vs the median of ≥ 3 PR219 repeats, ≤ the AT-Q10Q6 τ of the image cell. Pass: every cell's new reserve ≤ the M5 bound (§16); runtime max ≤ reserve (0 strict violations); 0 graph-build-check errors; answers 2749 / 12749; non-compact ubatch prompt t/s reported vs PR219 (U27); M3 on the TE route (pool use of the TE `top_k`, §6.2) reported.
- **AT-C16 (P2-28)**, layout 3 (`r1c/runB3.sh`: default server, c131072 ub4096 q8_0, slot A = B65t generating, slot B = TI12k); a test-only no-share switch (`LAB_C19_NO_SHARE` ported as a test hook).
  - **AT-C16a (W8, before the re-split; 8-strip ubatches of 4088 tokens as on PR219):** strip-view census bias and kq_mask new > 0 on slot-B text ubatches, reuse 88 per src per graph (8 strips × 11 reusing layers, R1-C layout 3 table, MEASUREMENTS.md:4127-4134); main ROCm0 **runtime peak** ≤ 8054.08 MiB.
  - **AT-C16b (W9 and W13, S = 1, ubatches ≤ 512 tokens):** census new 1 / reuse 11 per src per graph; **reserve-to-reserve** no-share − share ≥ 0.5 × 11 × (n_ctx × 512 + n_ctx × 1024) B = 0.5 × 2112 MiB = 1056 MiB at c131072 (per-view bytes from R1-C: bias n_kv × 512 B, mask n_kv × 1024 B per strip, MEASUREMENTS.md:4151-4154); runtime peak ≤ reserve (strict). Runtime-peak and reserve criteria are not mixed (the round-1 4507.18 MiB was half of a runtime-peak delta, 12561.26 − 3546.90 MiB at n_kv 71680, and cannot be reached by a reserve delta at S ≤ 2).
- **AT-C18 (fit probe, P2-13; W7 for `-np 1`, W9 for `-np auto`).** `-fit on`, f16 and q8_0, with `-np 1` and with `-np auto` (4 slots unified), with and without `-nkvo`, at -c 131072, ub 512 and ub 4096: log the probe's per-backend `backend_buf_exp_size` in the no_alloc context next to the real context's `sched_reserve` lines (the R2-RES `parse.py` labels). Pass: |estimate − real| ≤ 1 MiB per backend, main and MTP; the guard ran on the probe context with a non-NULL backend on ≥ 12 × (plan + attend) nodes; under `-nkvo` the probe shows the pins (0 ROCm0 plan/attend nodes).

**AT-D — HIP graph placement (W7/W13).** Builds: W0 (reference) and W7, again at W13. An env-gated print of `ovl_regions` counts by kind at the end of `ggml_cuda_ovl_plan`. All runs under flock.
- **AT-D1 (OVL):** `LAB_OVL_PRINT=1 llama-bench -p 4096 -n 0 -d 4096,32768 -b/-ub 4096`, f16 and q8_0, -r 1, W0 and W7; U16: 20 iterations with graphs on, OVL=15 vs OVL=11, memcmp; value: OVL=15 vs 11, pp4096 at d32768 and d131072, ub4096, f16/q8_0, §16 design, W0 and W7. Pass: W7's kind-4 regions per graph = W0's per type and depth (expected 12); memcmp identical; W7's OVL15/OVL11 ratio not below W0's (CI of the ratio of ratios contains 1 or lies above).
- **AT-D2 (`-ngl 0`):** `GGML_SCHED_DEBUG=2 -v llama-perplexity -c 4096 --chunks 4 -ub 512 -fa on -ngl 0`, f16 and q8_0, W7 vs W0-lab, 3 interleaved processes (re-run if #234's CPU incremental QSA cache is in the base, §0): route census, placement of plan / attend / shortcut FA, guard, PPL, s/pass; Q12 KLD; pp512 @ d4096 and tg32 @ d4096. Pass: sparse plan and attend on the device of their K buffer in 100 % of prints; guard 0; Q12; CPU perf new/W0 ≥ 1 − 2 × the W0 between-process spread (no unsourced 0.95).
- **AT-D3 (partial `-ngl`):** an `-ngl` value that splits the QSA layers 3, 7, …, 47 between CPU and ROCm0. Pass: placement census per layer = its K buffer device; guard 0.
- **AT-D4 (`-nkvo`, P2-10):** placement and quality: `llama-perplexity -c 8192 -b 4096 --chunks 2 -ngl 99 -nkvo -fa on`, f16 and q8_0, ub 512 and 4096 (and ub16 for the pin recipe), W7 vs W0-lab. Census incl. the `-fit` probe context. Pass: 0 GPU-placed plan / attend / shortcut-FA nodes in any context (the reserve_size path included, after the §7.1.7 fix) unless the named host-KV-copy route is active and logged; no `cache_[kv]_l*` split inputs to ROCm0 (same exception); guard silent incl. the fit probe; PPL within 3 × the chunk SE of C15 (f16 1.9428/1.9449, q8_0 1.9532/1.9519); Q12. **Binding perf gate:** the AT-W7PERF (c) cells; history: R1-NKVO pinned PR CPU FA +12 % f16, +70 % q8_0 vs the PR219 copy route at c4096.
- **AT-D5 (`-fa auto` / `-fa off`):** `-nkvo -fa auto` f16 ub512: log the FA resolution (llama-context.cpp:753-755) and check the plan's stated route matches; `-fa off` f16, ub 512 and 4096, offload on and `-nkvo`: on PR219 and on W7, then W13. Pass: under `-fa off` (and with `-nkvo`), QSA layers keep the sparse route (census > 0, 0 masked nodes, guard 0); PPL within 3 × the chunk SE of the PR219 same-flag run; never use PR219's masked route as a paired KLD reference (R1-NKVO). Old `-fa off` session files of qwen4exp fail on the `v_trans` check (llama-kv-cache.cpp:2768) with the existing message (§17).

**AT-W12-BANDS — per-type, c32768 and Vulkan e2e bands (P2-4, P2-23, P2-25, P2-33), in W12, before any W7+ Q1/Q3-A/Q4/Q1-vk arm is scored.**
- Arms: W0-lab HIP (`LLAMA_QSA_ALLVIS` unset) and W0-Vulkan `build-vk` (C11 Vulkan flags, target llama-perplexity), plus the round-1 trees for reproduction only. Every run under `flock ~/gpu.lock`, `%.9e` QKLD_CHUNK, `QPIN_PIN=qsa,moe`, `rows_absent = moe_absent = 0` checked in every arm. Recipes `r1ub/run.sh`, `analyze.py`, `q4_analyze.py`.
- (H1) Reference ref8t0 (c8192 × 8, f16 ub4096, `LLAMA_FN_LAST_ATTN_TRIM=0`). Arms: bf16, q8_0K/f16V and f16K/q8_0V, each at ub4096 and ub2048, on the default route and with `LLAMA_QSA_SELECTED_KEY=1` (12 arms). Q11a's lo/hi split is printed for the ub512 arms of the same reference (`QKL_SPLIT=2048`) to measure the lo-region band (U40).
- (H2) Reference ref16t0 (c8192 × 16). Agreement arms for the same 3 types: kind 1, ub4096 and ub2048, and kind 1 minus tk.
- (H3) A clean q4_0 ub4096 pinned base, then q4_0 with `LAB_DEF=d3 LAB_D3_EPS=1e-7` in direct form (the q4_0 saturation floor).
- (H4) Reference ref32x8 (c32768 × 8, `QKL_TAIL=4096`, trim 0). Arms: f16, q8_0 and q4_0 at ub4096 and ub2048.
- (V1) A vref8t0-style reference on W0-Vulkan. (V2) Arms against it: (i) the old `~/dev/wt_lab/r1ub/tree/build-vk` at f16 ub4096; (ii) a repeat of (i); (iii) W0-Vulkan f16 ub2048; (iv) optional: W0-Vulkan with `GGML_VK_DENSE_DQ8=0 GGML_VK_HC_DQ8=0`. One rerun per Vulkan arm on device loss.
- Each reference and its pin set is deleted after its arms (22 / 43 / 40 / 16 / 22 GB), one reference at a time; the `.stats` files are kept.
- **Pass:** every band below computed by its pre-registered formula and frozen in §15 before any W7+ arm it gates is scored: QPIN self-replay per-chunk KLD 0 (≤ 3.7e-7) on each new reference; each per-type band > 0 (floored, §15); B_4096@32k = max over T ∈ {f16, q8_0, q4_0} of (mean + 3·SE) of the paired per-chunk ub2048 − ub4096, cross-checked against C17's f16 chunks (1.4e-4, 0.62e-4); Q4 B_T from the kind-1 ub2048 shift + 3·SE; the q4_0 saturation floor recorded next to q8_0's 9.39e-5. Vulkan: (ii) = (i) at %.9e (determinism); if (i)'s mean is ≥ 6.4e-4 or within 3·SE of it, the old vref8t0 reference is declared invalid and Q1-vk pairs new-Vulkan(T) with W0-Vulkan(T) only, with B_vk = mean + 3·SE of (iii). A band still unmeasured keeps its gate UNPROVEN, and the PR cannot claim that gate.

**AT-Q10Q6 — calibration of the Q10/Q6 thresholds (P2-3, P2-5, P2-29), before Q10/Q6 are scored in W8/W9.**
- (1) A lab tree `~/dev/wt_lab/r1q10/tree` = W0-lab + the build4 lab patches (`LAB_QSA_C4`, `tools/mtmd/lab-r1.cpp`) with `#define QSA_DECODE_MAX_QUERIES 127` in `ggml/src/ggml-cuda/qsa-decode.cuh` (a compile-time define, qsa-decode.cuh:5; setting it in the environment does nothing). HIP flags, targets lab-r1 and lab-c4. lab-r1 text L = 8192 (`r1/text_campaign.sh`, `LAB_QSA_C4=0`): build4 default × 3 and the 127 build × 3, per-request KLD over the 2 × 100 fixed tokens.
- (2) L = 8193, 8194, 8195 and image p3 (`r1/img_campaign.sh`): ≥ 4 more PR219 repeats per arm at ub8192, plus the ub4096 pair, so each cell has ≥ 6 draws.
- (3) Q6 on lab-c4 (`r1e/q6.sh`, `r1e/q6q8.sh`): add ub 6144, 3072 and 1536 to {8192, 4096, 2048, 1024} for p1 and p2 f16 and for p2 q8_0; run the missing p1 q8_0 matrix (TE and CB × ≥ 4 ubs).
- Delete the logits after the KLD pass (R1-CB produced 7.0 GB); keep `kld_results` and the pair files.
- **Calibration valid when:** 127 vs 127 KLD = 0 and build4 vs build4 KLD = 0; floor_ctl = mean + 3·SD over the 3 × 3 cross pairs; each Q10 cell and each Q6 (prompt, type) has ≥ 6 independent draws; the leave-one-out false-fail rate of mean + 3·SD is reported next to that of the sample max, and mean + 3·SD is kept only if its rate is ≤ 1/6, else the ≥ 19-draw quantile rule is used (with the extra draws run); τ_p1,q8_0 filled.
- **Gates then pass when:** Q10 control: new rerun vs new = 0 and new vs PR219 ≤ floor_ctl; Q10 cells: new vs the PR219 median ≤ τ; Q6: KLD(new_T ‖ PR219_T TE) at ub8192 ≤ τ_T; answers correct (headline, newspaper, date; 2749 / 12749 / 26999; B65t 12299).

**AT-W7PERF — shortcut grid (AT-A extension, AT-B4-HIP) and the binding `-nkvo` gate (P2-9, P2-10), on W7.** On `~/dev/wt_rewrite` at W7, under `flock ~/gpu.lock`, HIP build.
- **Op level**, C18 protocol (3 warm-ups, hipEvent per launch, mean of 12 per-layer medians, 3 interleaved processes): SPARSE_ATTN all-visible with forced G1 and forced G4 against causal FLASH_ATTN_EXT of the same pair, for q8_0, q4_0, bf16, q8_0K/f16V, f16K/q8_0V on n_q {2, 4, 8, 16, 32, 64, 127, 1024} × n_kv {512, 1024, 2048}; CPU SPARSE_ATTN per layer, f16 vs q8_0.
- **E2E**, C10-text llama-bench with the §16 statistic: (a) `-ub 64 -b 4096 -p 2048 -n 0 -d 0`, and `-ub 1024` for q4_0, bf16 and the mixed pairs, sparse vs `LLAMA_QSA_PREFER_DENSE`; (b) tg-with-MTP, verify 2–8, d0/d1024, `-ctkd q8_0`; (c) `-ngl 99 -nkvo -fa 1 -b 4096 -ub 512,4096 -p 512,2048 -d 0,4096,32768 -n 0 -r 5`, and `-p 0 -n 32 -d 4096,32768`, for f16, q8_0, q4_0 and bf16, W7 vs W0-lab, interleaved; (d) shortcut pin on vs off, pp512 and tg32 at d0/d1024, with a `GGML_SCHED_DEBUG=2 -v` placement census in the same arms; (e) on any miss, fallback (a) and then (b) of §10 on the same grid.
- **Pass, shortcut:** per (pair, n_q band), sparse ≥ 0.98 × dense at op level and in the e2e `-ub 64` / `-ub 1024` cells, else `prefer_dense` returns dense for that band (pre-registered, no re-decision); in each band the shortcut G is G4 where G4 beats G1 by > 3 × the p90 run-to-run noise (the AT-A rule), else G1; the tg-with-MTP cell passes P5 (new_T / new_f16 ≥ 0.94).
- **Pass, `-nkvo`:** new(T, `-nkvo`) / W0-lab(T, `-nkvo`) ≥ 0.98 in every pp512/pp2048 cell at d0, d4096 and d32768 and every tg32 cell at d4096 and d32768 (§16 statistic); new_q8_0 / new_f16 under `-nkvo` ≥ 0.94; the census shows 0 GPU-placed plan/attend/shortcut-FA nodes unless fallback (b) is active and logged; the shortcut stays pinned to the CPU only if pin-on / pin-off ≥ 0.98 in (d). A miss after fallbacks (a) and (b) is a PR-blocking FAIL that goes to the owner with the ratios.
- Context: C1 shows parity at n_q 1 (0.980–0.984) and 3.4–4× slower masked all-visible G1 at n_q 128 (MEASUREMENTS.md:49); R1-NKVO pinned PR CPU FA +12 % f16 vs +70 % q8_0 at c4096; the PR219 copy route copies about 427 MB per graph for q8_0 at d32768 [arithmetic, UNPROVEN], so a per-cell mixed result is possible.

**AT-O — f16/bf16 V-pack cap (owner decision data, not pass/fail).** Measure both trait settings on the rewrite: f16 real-text e2e prefill t/s, ub4096, at depth ~300k (n_kv just past 262144) and at the deepest depth that fits; compute-buffer reserve and peak VRAM for each; §16 statistic. Report in PROOF and in the PR description as an open owner decision with the C18 kernel numbers (packed+pack 42.34 vs best direct 52.92 ms at syn 262k; 62.97 vs 63.65 at syn 1M) (OWNER).

---

## 17. Rollback

- **Inside the branch.** Each work item is a separate commit series that builds and passes its gates. W10 and W11 are independent: each can be dropped by reverting its commits (W10: rot params 0; W11: auto = -ctk).
- **Policies.** `prefer_dense` and `group` are backend procs: flipping a pair back to dense in the shortcut, or a G choice, is a one-line policy change gated by the same cells.
- **Vulkan.** If AT-B2 misses the in-build sel_cm f16 median by more than 10 %, keep `flash_attn_sel_cm.comp` as the f16 instance of SPARSE_ATTN on RADV, adapted to read the G1 plan, and delete only its FLASH_ATTN_EXT wiring in W13. If an S_fast pair misses AT-B1 after every AT-B3 arm, that is a **PR-blocking FAIL** that goes to the owner with the measured ratios (P2-11); "ships on the sparse op with its measured ratio in the PR text" applies only to S_generic pairs (P4).
- **After merge.** Revert the merge commit. There is no runtime switch back to the masked route (L15). Session files: no session or state version is bumped (§9); qwen4exp files with a sub-8-bit -ctk are rejected by the existing per-layer type check after W11 (and after a revert) and must be re-saved; qwen4exp files saved with `-fa off` (v_trans = true) must be re-saved after W7 (D21).
- **Measurement artefacts:** C18 rows in `~/dev/wt_lab/c18/cap` (1.6 GB, kept); wiki d4k Q/K/V recapturable with the R1-A command (rows byte-identical); `~/dev/wt_lab/{r1,r1av,r1q8,r1ub,r1e,r1c,r1s,r1nkvo,bands,r2res}` (logs, trees, scripts) kept; their large files already deleted; the evidence copy `EV` (E0).

---

## 18. UNPROVEN, with the experiment that would prove it

| id | claim | experiment |
|---|---|---|
| U1 | The rank-based single-sequence predicate (§4.2) is exact for every layout (several images, depth > 31.5k, back to back, holes) | W9 visibility test on recorded (request C, B65t, R1-RES) and synthetic layouts; M4 recipes (b)/(c) with 0 rejections |
| U2 | A CB-style rule for non-compact multi-sequence layouts (keep forced cells, expand oversized groups) matches PR219's TE | Port `r1/r1_oracle.py`, implement as an alternative producer, pass Q10; not in this PR |
| U2b | Plan cap and union behaviour under permuted cells (qpb 1.001 measured only in fresh caches, C5) | C5-style union capture after defrag / seq_rm holes |
| U3 | G1/G4 crossover (n_q 2–512) on the new kernels | AT-A |
| U4 | 64·n_q direct-V crossover at other n_q and 131k–262k | W3 sweep |
| U5 | Pack-K at shallow n_kv no longer pays with split K | W3 variant timing; adopt only if ≥ 2 % |
| U6 | In-kernel rotation + gate fusion recovers +20.4 + 17.7 ms per pass | W10 OP_PROF A/B |
| U7 | Vulkan quantized kernel speed, bf16 staging cost, Vulkan bands, whether Vulkan G4 pays | AT-B2/B3; band measurement at W6 |
| U8 | S_generic speed, quality, bands, indexer mapping; GQA 4/8 speed | W5 timing + bands; P4 cells; Q1/Q4 arms per generic type |
| U9 | bf16 KV quality on HIP beyond Q1-A/Q11; bf16/f32 indexer; bf16 on Vulkan | Q1-A, Q4, Q1-vk arms for bf16 |
| U10 | Quality beyond 32k; q4_0 and mixed at depth | server-based teacher-forced KLD tool (llama-perplexity cannot run ≥ 64k here, C16) |
| U11 | — closed: full-precision paired SE of q8_0 − floor is 4.69e-6 (R1-Q8) | — |
| U12 | Real-text runtime at 131k and anything at 1M; n_kv beyond the op-tested 2^21 + 4 | llama-server -c 131072 / 1048576, q8_0 and q4_0, one long text request, strict reserve, census + answer; raise the §4.4 cap only with a larger op case |
| U13 | q8_0 decode margin in the shortcut (0.980 at d1024, C1) on the new kernel | P5 shortcut cells; `prefer_dense` flip if it misses |
| U14 | Which kernel path triggers KFD evictions (C7) | owner action with root: amdgpu/kfd tracepoints |
| U15 | Portable tier on real non-RDNA3.5 HIP hardware (compile gated by W5; runtime only forced on gfx1151) | `-o SPARSE_ATTN` on such a device |
| U16 | OVL regions with plan/attend on sparse layers | AT-D1 |
| U17 | #219's "10.7 GB at 262k"; residual 0.029–0.032 MiB/cell of the shared arm | C19 slope runs at -c {131072, 262144} on the non-compact variant |
| U18 | — closed: default server = 4 slots, `kv_unified = true` (common/arg.cpp:1429, server.cpp:168-172; R1-C log); explicit `-np N` = split (common.h:649) | — |
| U19 | — closed: the q4_0 draft acceptance loss is quantization, not route (−0.18 pp, CI [−0.58, +0.22], R1-E; CPU-attend caveat) | — |
| U20 | L68's 0.0088 is the hook's graph split | plain vs plain rerun; the §14.3 hooked-arm census |
| U22 | Transformers' `Qwen4ExpTextQSAIndexer` is the source of the CB rule | read the transformers source |
| U23 | In-situ plan cost is 0.22 ms above the harness (C3) | hipEvent timing of `sa_plan_*` in-model |
| U24 | Text-vs-random absolute gap cause (C10) | not needed for any decision |
| U25 | Bucketing at depth-gated scope; kv-head interleave ≥ 262k; V prefetch | later, separately measured changes |
| U26 | No SWA layer in this model's GGUF beyond L19's scalar condition | read `n_swa` / swa layer pattern from the GGUF metadata at load (`-lv 4`) |
| U27 | Perf cost of splitting non-compact ubatches at U = min(ub, 512) tokens (S = 1); the variant's size is now measured (R2-RES formulas) | AT-C12 prompt t/s vs PR219 on layout 3 and the image-slot layout |
| U28 | Speed on non-RADV / non-RDNA3 Vulkan devices (scalar tier, 7.8× slower on the op, C11 d3) | not possible on strix-002; listed in the PR text |
| U29 | — closed for the variant: the probe misses it by up to −2539.06 MiB ROCm0 / −818.79 MiB host and the per-backend max fix is exact in 64/64 (R2-RES); the `-nkvo` size effect of the pins is measured by AT-C18 | AT-C18 |
| U30 | All-visible rows on a raw M-RoPE image kq_mask inside the shortcut | record one in W7 and add it to `test-qsa-allvis-rows` |
| U31 | Q10/Q6 thresholds: noise distribution and false-fail rate of the mean + 3·SD rule | AT-Q10Q6 |
| U32 | M2 equality: gallocr places the moved QSA items under the existing peak | M2 run |
| U33 | Bands for bf16 and mixed pairs in Q1/Q4 and the Q11b tier (floored meanwhile) | AT-W12-BANDS H1/H2; W12 Q11b shift |
| U34 | Band rows for split counts 6–16 and > 33, the mixed packed-both route, GQA 4/8 and the non-PR tiers on test-backend-ops inputs; the new kernels' NMSE on the ported shapes (now a gate against the R2-BOP UNI rows) | R2-BOP harness on the PR-verbatim kernels (W4); W3/W4/W5 gates |
| U35 | Which llama APIs can create a duplicate (pos, y, x) key in a qwen4exp flow | W9 code reading of seq_div, seq_add with ext positions, seq_cp |
| U36 | Attention cells ≡ indexer cells after every apply (NC0 reads the attention cells, the graph reads the indexer cells) | `tests/test-qsa-resplit.cpp` assert (v) |
| U37 | Cause of the q8_0-over-f16 ROCm0 excess of the non-compact variant (+232.27 MiB variant alone at c131072 ub4096 T 1024, R2-RES); f16 at S = 1 | W9 `sched_reserve -lv 4` per node; M1 default-server rows |
| U38 | Whether the [K, T/s, s] split-cache variant covers a one-stream non-compact ubatch [K, T, 1] | W9 STRICT run of that shape (AT-C12) |
| U39 | B_4096@32k (Q3-A band); B_vk on W0-Vulkan and the size of master's Vulkan GEMM delta (Q1-vk band) | AT-W12-BANDS H4; V1/V2 |
| U40 | Q11a lo-region (< 2048) band, borrowed from B_512 | AT-W12-BANDS H1 split |
| U41 | Speed of the Vulkan portable tier; whether today's masked route works on non-64-lane Vulkan devices | AT-B7 on lavapipe / RADV sg32 (correctness); no non-AMD device on strix-002 for speed |
| U42 | Under `-fa on`, Metal (and likely SYCL, OpenCL, WebGPU, CANN) accept FA + src[5] without a mask and compute dense attention today (ggml-metal-device.m:1743-1790 reading) | no such device here; stated as UNPROVEN in §19 |
| U43 | Effect of #234 (CPU incremental QSA cache) on the `-ngl 0` path measured by R1-NKVO | AT-D2 re-run on the new base if #234 lands |
| U44 | Shortcut sparse/dense ratio at n_q 2–127 and 1024 for non-f16 pairs | AT-W7PERF |

---

## 19. Material for the PR description and for the #219 review comment (all sourced)

**What we found about #219**
- It works and it is the right direction: q8_0 at depth is +65 % (d32k) to +290 % (d131k) at pp512 on real text against master (C10); the image matrix passes on f16 and q8_0 (L64); the compute buffer equals f16's (4133.70 vs 4133.44 MiB at c131k, C16; 3171.06 vs 3170.80 MiB in the server, L50).
- **The ub512 d0 q8_0 regression is the routing, not the repack.** At ub512 every QSA ubatch has 512 queries and goes to `qsa_decode` (fattn.cu:1154, qsa-decode.cuh:5); `qsa_prefill` never launches there (rocprofv3: 0 launches). The loss is `qsa_decode_wmma_partial<true>`, 63–66 ms per ubatch against 24–38 ms for master's dense q8 FA; routing to G4 gives +1.6 % (text) over master (C13 d4).
- The "prefill" q8 tests with ≤ 512 queries run decode, and no test reaches `<1,8,1,1>`, which spills 28 VGPRs (L10, L11).
- The MTP draft runs sparse QSA with f16 KV by default. With `-ctkd q4_0` it silently takes the masked route: 12568.67 MiB draft compute and −2.37 pp acceptance (C2); the acceptance loss is the q4_0 quantization, not the route (R1-E).
- **Vulkan:** the PR base does not compile without #240; q8_0 and bf16 run the masked route at 0.575× / 0.323× of f16 on text (C11).
- **The ~421 t/s sample is most likely a KFD eviction, not a PR allocation** (0 hipMalloc in 486 rep windows; the closest reproduction, 512 t/s, coincides with +662 ms evicted; base showed one 689 t/s rep with +402 ms eviction, C7). The specific historical ~421 samples (L59) had no eviction counter and are attributed by analogy only (C7 Limits), and the triggering kernel path is UNPROVEN.
- **strip_views is verified**, with numbers from the PR tree: −1120 MiB for one deep image ubatch at c65536 (C19), −9014 MiB on a default server with two image slots (R1-C).
- **Quality under pins:** q8_0 costs +2.6e-5 over the reorder floor at full precision (SE 4.69e-6, R1-Q8); q4_0 on the masked route costs 54× the floor (C8).
- **Latent issues the rewrite fixes, measured on the PR tree:** split-stream image ubatches, and on a unified cache **any image-bearing slot while another slot holds cells** (one image slot is enough: R1-RES unified control, image slot `scalar=0` from n_kv 66816 on with a text-only second slot), reach the non-compact scorer above the reserve (+155 to +1937 MiB ROCm0, R1-RES, R1-C). (Correction of our round-1 reading of R1-C, which said "reachable on the default server only with ≥ 2 image-bearing slots", MEASUREMENTS.md:4169: that measured slot B's ubatches only.) The `-fit` probe misses that variant by up to −2539 MiB ROCm0 (R2-RES). `-nkvo` copies K/V to ROCm0 every graph (300/300 QSA FA on ROCm0, R1-NKVO). The PR's decode kernel is not run-to-run deterministic on rows with duplicate keys (atomicCAS dedup race, R2-BOP). The PR's masked route is not run-to-run deterministic on HIP (KLD 0.0103, R1-NKVO); the rewrite removes the masked attention route, but keeps GPU `top_k` for the non-compact TE rows, whose tie order is the candidate cause (R1-CB), so non-compact ubatches are **not** claimed deterministic.

**What the new PR recovers** — the §12 table: strip_views as-is (author kept; "adapted (conflict resolved)" if #234 is in the base); the q8 helper and the decode q8 K loader as-is (deduplicated); the q8 pack and the block-w addressing adapted; the `n_q·64 < n_kv` direct-V threshold kept for every type's V by default; the q8_0/mixed test shapes ported; the log line adapted. The three commits keep Iggy Jackson as author; porting commits carry `Co-authored-by: Iggy Jackson <iggy@theiggy.com>`.

**What it replaces, and why:**
- the bool-flag kernels and the spilling DIRECT_V (C18, L10); the decode V gather (C18); n_q ≤ 512 → decode (C13);
- the probe and the env override (L14, L15, L29); pool scratch (D12); K = V in decode (L30);
- the masked route (C1, C15, C16); token-expanded selection as an attention route (TE stays as the non-compact row producer; CB kept for compact ubatches, C4, R1-CB);
- behaviour changes:
  - **backends without a SPARSE_ATTN kernel (D22, owner decision §20.6):** QSA on CUDA, MUSA, Metal, SYCL, OpenCL, CANN and WebGPU becomes an init error whose message names the remedies (host KV with `-ngl 0` / `--no-kv-offload`, or a HIP/Vulkan build). Today CUDA takes the masked route (fattn.cu:1198-1207); Metal and the others under `-fa on` probably accept the maskless FA + src[5] and compute dense attention (U42, UNPROVEN), and under `-fa off` take the masked route;
  - **Vulkan devices** that can create none of the three tiers (§6.3 requirements) lose QSA with an init error naming the missing capability; non-RADV-RDNA3 devices run the scalar or portable tier (speed UNPROVEN, U28/U41);
  - **HIP archs other than RDNA3.5** (gfx1100/1101, RDNA4, CDNA) run the portable tier (§5; speed UNPROVEN, U15);
  - `LLAMA_SPLIT_MODE_TENSOR` with QSA becomes an init error; `n_ctx_seq` above the op-tested 2^21 + 4 cells becomes an init error;
  - RPC: `RPC_PROTO_MINOR_VERSION` bumped (two ops appended before GGML_OP_COUNT), so old RPC servers are refused at hello;
  - `-fa off` keeps QSA sparse but changes `v_trans` (old `-fa off` session files must be re-saved; non-QSA attention layers in the same cache pay a `ggml_cont(transpose)` per layer); `-fa off` with a quantized V stays an init error (llama-context.cpp:663);
  - sub-8-bit -ctk gets a q8_0 indexer on qwen4exp (GLM5_NEXT unchanged); no session/state version bump; old sub-8-bit qwen4exp files are rejected by the existing per-layer type check;
  - multi-sequence contexts reserve one non-compact variant (S = 1): default server +421.02 / +383.99 MiB main and +388.00 / +383.99 MiB MTP (ROCm0 / host, R2-RES); non-compact ubatches are re-split to ≤ 512 tokens;
  - host-KV contexts: the CPU graph work buffer is outside the sched reserve; the TE `top_k` on HIP still uses pool memory for narrow shapes (top-k.cu:577-599).
- **Coordination:** #234 conflicts with this PR in ggml/src/ggml-cuda/qsa-prefill.cu and the VGPR baseline (`git merge-tree`, §0); whichever lands second rebases. Overlaps with #225, #247, #248, #230 as recorded in W13.
- open owner decision: f16/bf16 V pack capped (default) vs always packed (AT-O).

---

## 20. Review log

### 20.1 Round 1 — confirmed objections and how the plan addresses them

| O | objection (short) | resolution |
|---|---|---|
| O1 | Op contract: scratch inside dst, OOB quad loads, W1 CPU gate impossible | Scratch moved out of dst (§2.1, D1); bounds assert `ne[1] ≥ 4·ceil(n_kv/4)` + view-end op test (§2.2, §14.1); W1 equality replaced by R1-A per-pair tolerances + contract-vs-f64 bands (W1) |
| O2 | One CB rule undefined for split/shared blocks; no multi-sequence gate | Measured wrong (R1-CB); D6 changed: non-compact rows = PR219 TE ∩ visible (§4.1); Q10 added; CB-style alternative is U2 |
| O3 | Non-compact reserve skipped for split streams / single sequence | Variant for `n_seq_max > 1` unified or split (§7.1.4, R1-RES); per-stream predicate fix (§4.2); single-sequence predicate total by construction + decode error on rejection (§7.1.5); M4 recipes (a)–(c) (W9) |
| O4 | Quantized shortcut route ungated; 2-D producer single-stream | Q11a/b/c (R1-AV: prototype passes); 4-D producer specified with asserts (§4.1); `test-qsa-allvis-rows`; GPU coverage of every shortcut pair (§4.1) |
| O5 | `-nkvo` placement | Pins for host KV (D15, §10), guard skipped on the reserve_size path, recipe AT-D4, Q12; `--no-kv-offload` removed from the error text |
| O6 | Q1 has no margin; f16 arm = hidden bit identity | Q1 rule replaced by fixed bands B_U from PR219's own route changes (§15), f16 included; 3·SE printed only (R1-Q8, R1-UB) |
| O7 | Full-precision KLD and QPIN arrive after the gates; PR219 arm unpatched | W12 lands right after W0 (§13); PR219 arm = r1av / r1q8 trees with `QKLD_CHUNK` at %.9e; control reproduction in W12 |
| O8 | Q8 cannot detect selection defects (pin erases them) | Q8 defects injected after the pin (`r1q8_lab.diff`), on − off same type; run in W7; δ_T = 7.45e-5; Q5 primary (R1-Q8) |
| O9 | e2e gates reach one kernel route | Q1-B ub512, Q1-C ub64, Q3-B ub256, Q11, Q1-vk; trim-0 references (R1-UB); remaining ungated routes listed (§15) |
| O10 | Q3/Q4/Q6/Q7 thresholds on point estimates | Q3 → R1-UB tiers ×8; Q4 → R1-E bands; Q6 → R1-E τ, same type; Q7 → same-type pairing, prompt count from SD 0.921 (R1-E) |
| O11 | Op thresholds unsourced, misattributed to C18 | Replaced by R1-A measured per-shape bands (k = 2) with sensitivity proof; f64 report-only (§14.2) |
| O12 | D3 threshold set against the old G1 at shallow depth | G is a measured policy proc; AT-A two-way fit incl. pack cost; default CAPPED pack removes the always-pack cost at depth (D3, D8) |
| O13 | Decode at depth for q4_0/bf16/mixed not gated vs f16 | P5: new_T/new_f16 ≥ 0.94 at depth for every S_fast pair; W4 G1 kernel gates for q4_0/bf16 |
| O14 | No quantized Vulkan kernel measured; P8 too coarse | Two V variants (D20); AT-B3 op-level per pair; AT-B1 per pair; §17 extension |
| O15 | P2–P4 statistics cannot settle pp512 at depth | Mean of reps (rep 0 dropped), warm-up instance, non-inferiority θ, terminating 2×/4× rule (§16). The objection's "~40 processes" remedy is not adopted: the skeptic's re-bootstrap shows the non-convergence is the median statistic, not the process count |
| O16 | OVL kind-4 hard-wired to FLASH_ATTN_EXT | End predicate retargeted, plan node before the region, lifetime in §7 (§6.2, W7); AT-D1 |
| O17 | Vulkan barrier elision does not see over-allocation | Unconditional `ggml_vk_sync_buffers` before every sparse node (§6.3); stress test in W6 |
| O18 | Memory gates miss total footprint and Vulkan prealloc | M2 = total footprint + QSA-share rule; M1 logs Vulkan prealloc (§16) |
| O19 | P6 weaker than C7 | P6 rewritten as C7 states it, all cells, q4_0/q8_0 ub512 d0 cells added |
| O20 | Reserve vs G switching on n_q; G1 size unsourced | §5 max over G and n_q' ≤ n_q; G1 formula and 12.42 MiB max (R1-C); AT-C5 |
| O21 | ggml/backend touchpoints missing | W1 checklist (may_expand, RPC, OP_COUNT asserts, ggml-cpu switches, meta reject, reg proc API) and build gates |
| O22 | Vulkan gates calibrated on another base | All Vulkan references re-measured on W0 (AT-B1/B2/B5) |
| O23 | Shortcut rules rest on HIP data; Vulkan type cliff | `prefer_dense` backend policy with pre-registered 0.98 rule (D4, AT-B4, W7); Vulkan memory/quality rows (AT-B5, Q1-vk) |
| O24 | CPU production path ungated; single-threaded plan | CPU threading (§3); pins (§10); AT-D2/D3; Q12 |
| O25 | Non-RDNA3.5 ggml-cuda devices | Device guards; multi-arch HIP compile gate (W5); CUDA supports_op false → init error, listed in §19 |
| O26 | Indexer auto hits GLM5_NEXT | D14 scoped to qwen4exp; session reader rule keeps GLM5 files readable; W11 test |
| O27 | W7–W10 gates not runnable; Q1 no power | W12 moved; Q8 in W7; bands; Q7 same-type (duplicate of O6–O8) |
| O28 | Gated epilogue unreachable; census skip | `test_sparse_attn_gated`; matcher retarget in W3; HIP census on existing procs; missing census = FAIL |
| O29 | Plan bit-exactness vs rows scratch in dst | Scratch out of dst; exact whole-dst err(); attend tests use `fusion_test_nodes` (duplicate of O1a) |
| O30 | W1 CPU-FA equality false; thresholds thin; oracle inputs | Same as O1c/O11; oracles O_CON/O_C18/O_PLAN defined exactly (§3) |
| O31 | Overflow test unpassable; poison missing | Overflow respecified (V-side, same-sign inf, CPU only) + K-side loader unit test; poison ported for every S_fast type incl. scales and padding quads; select rule in the contract (§2.2, §14.1) |
| O32 | No expected census table; new routes not quality-tested | §14.3 table; TRACE and CPU counter in W7 "What"; Q1-B/C, Q3-B, Q11 for coverage |
| O33 | W7/W10 need W12 tooling | W12 moved (duplicate of O7) |
| O34 | No route for `-nkvo` or `-fa off` after W8 | Eligibility drops flash_attn and offload_kqv; `v_trans = false` for qwen4exp (D21); pins; AT-D4/D5 |
| O35 | Default server gets an ungated non-compact reserve; U18 settled | U18 closed; S = 2 fixed strip set with ubatch re-split (round 2: S = 1 and the specified mechanism, P2-2/P2-14) (§4.3, §7); M5 + AT-C12. The objection's alternative "reserve only when mtmd is loaded" is not adopted: unified seq_cp layouts are non-compact without any image (§4.3) |
| O36 | Vulkan baselines from another tree, single sample | AT-B1/B2 on W0, multi-sample, text + random (duplicate of O22) |
| O37 | Repo CI gates missing | Existing HIP quality check used; baseline regenerated in W13; flat file names for the globs; test placement; internal-commit spill state stated; master CUDA red run recorded |
| O38 | Correctness bands unsourced; fixtures deleted | R1-A bands; wiki d4k recapture is byte-identical (rows `cmp`); Q5 restated (§15) |
| O39 | Perf gates cannot pass at parity | Non-inferiority θ vs same-campaign reference, stopping rule, W4 margins, P1 absolute dropped, P9 rule (§16) |
| O40 | Vulkan gates rest on another tree and a random single sample | AT-B1/B2 (duplicate of O22/O36) |
| O41 | strip_views proof can pass vacuously | AT-C16 on layout 3 with census > 0 and a no-share delta (R1-C); §12/§4.3 numbers labelled PR-tree |
| O42 | Q7 unpassable for q4_0 | Same-type pairing; f16-draft comparison reported only (R1-E) |
| O43 | Memory model contradicts its table; f16 depth ungated | §7 corrected; per-type pack trait default CAPPED per OWNER; M6 + AT-C17; AT-O for the owner |

**Partial disagreements, with sources** (the plan changed anyway, but not as the objection proposed):
- O4's "2-D view gives wrong rows for a 4-D mask" and "KQ_MASK_PAD rows": this tree has no pad rows (PR llama-graph.cpp:43; `grep -rn KQ_MASK_PAD` empty) and the 2-D view reads correct values; the defect is the shape, measured as an abort (R1-AV). The proposed Q11 "decode-like tail in the shortcut" does not exist at c8192 (scored positions are past n_kv 2051); Q11a uses `QKL_TAIL=8191 QKL_SPLIT=2048` and Q11b a c2048 tier instead (R1-AV).
- O5 / R1-NKVO: pinning the rows tail is not needed (pin 7 = pin 3).
- O14: the proposed d131k / tg Vulkan cells barely add sensitivity (attention share does not grow with depth at fixed top-k); the plan keeps the op-level per-pair gates as binding diagnostics and AT-B1 as the binding e2e gate.
- O17: "any later node in the freed workspace" and "the reduce pass" are already ordered by the dst checks (skeptic); the rule adopted is the narrower pre-dispatch sync.
- O22/O36: the W6 op reference does not depend on the base (no FA hunk in the master diff), so AT-B2 uses the in-build sel_cm, which is cheaper than a cross-tree A/B.
- O20/R1-C: G1 in PR219 runs up to n_q 512, not only < 128 (qsa-decode.cuh:5, fattn.cu:1154), so the max is 12.42 MiB at n_q 263, not 9.0 MiB at 127.
- O35/R1-C: the owner's own server (`~/repro_img/repro_kv.sh`) passes `--parallel 1` (n_seq_max 1), so it is not affected. *(Round-2 correction, P2-1: the round-1 text added "a text-only second slot is compact (prefix.h:39-46)"; that holds for the text slot's own ubatches only. The image slot next to it is non-compact, and the mtmd-only gate was rejected for that reason as well as for seq_cp.)*

### 20.2 Deferred acceptance tests (owner rule: not run now, written as Prove-phase tests)
- Round 1 index 0 (Lab A, second half) → AT-A in W4.
- Round 1 index 1 (Lab B, Vulkan) → AT-B1–B6 in W6/W7; plan edits made now: Vulkan `prefer_dense` branch (D4, §5), staged-V variant (§6.3), §17 extension, Vulkan compute-buffer row (§7).
- Round 1 index 2 (Lab C GPU part 2) → AT-C5 (W4), AT-C17, AT-C12, AT-C16 (W8/W9).
- Round 1 index 3 (Lab D) → AT-D1–D5 in W7/W13; the OVL retarget, CPU threading, eligibility change and pins are written now.
- OWNER decision (V-pack cap) → AT-O.

### 20.3 Rejected by skeptics
- None in round 1.

### 20.4 Minor notes (round 1) and where each went
- Rank-based compact limits (§4.2 step 3). Static G1 workspace max (§5, §7). Non-causal attention error (§4.1). Vulkan G4 excluded from Vulkan case lists (§6.3, §14.1). SWA claim labelled (§4.3, U26). Image ubatches force `qsa_recover` (§4.2). Bit equality claimed only for f16/bf16/q8_0/q4_0 (§2.2). Hooked-arm census for the MoE fusion split (§14.3). Re-recorded ref must reproduce (W12 control). Family-wise alpha (round 1 answered "deterministic band rules, no correction"; **withdrawn in round 2**, P2-8: two-stage rule and reported family-wise rate, §15). P2 ratio CI "contains 1" (§16 uses ratios with θ; AT-D1 wording fixed). P1 absolute dropped (§16). Determinism memcmp test (§14.1). Overflow test output (§14.1). S_generic quality/perf statement (P4, U8). D4 using (n_q, n_kv) and the mixed-pair / bf16 extrapolation: `prefer_dense` takes (n_q, n_kv) and W7 measures bf16 and mixed cells at d0 ub {512, 2048}. W3 X+P tolerance widened to the stated ±4 % with same-run references (W3). P6 wording (P6 rewritten). MTP verify widths in the crossover (AT-A). Named random cells (§16). Census under HIP graphs (§6.2, §14.3). Pass B mechanism (§7.1.3). Attend dst and rows-scratch peak in §7 (scratch kept always reserved; sizing it 0 for ascending producers is not done because the op cannot know the producer's guarantee). Vulkan 1 GiB max buffer (§6.3, W6). Determinism per fixed (ub, b, G, V choice) (§2.2). OVL lifetime in M1/M2 (§7). U16 uses OVL=11 (AT-D1). Vulkan dequant not bit-equal (§2.2). Vulkan rot unit test (W6, W10). Vulkan supports_op 32-bit offsets at the reserve n_kv (§5). Non-RADV Vulkan devices (U28). Enum placement and OP_NAME/SYMBOL asserts (§2). #251 is conversion-only (§0). W0 conflict claim checked (W0). Existing census API reused (§6.2). Census captures vs launches (§6.2). Placement-guard test with a registered fake device (W7).

### 20.5 Round 2 — confirmed objections and how the plan addresses them

Every round-2 objection was confirmed by its skeptic; none is refuted. Where the plan adopts a remedy different from the one proposed, the reason is given with its source.

| P2 | objection (short) | resolution |
|---|---|---|
| P2-1 | Compact trigger for unified multi-sequence image layouts misstated (image slot next to a text slot) | D7, §4.2 step 3: compact for 2-D ubatches iff the **stream's** cells hold one sequence, from one shared `qsa_stream_one_seq` used by predicate and scan, with an agreement assert; §4.3 trigger list rewritten, "(rare)" removed (§4.3, U27); R1-C sentence corrected in §19 and §20.1; negative cases in `test-qsa-visibility`; AT-C12 image-slot layout (A image + ≥ 8k follow-up while B resident, STRICT, Q10-style KLD) |
| P2-2 | S × 512 re-split has no mechanism; unreserved graph in production | Mechanism by code (R2-RES): NC0 hook after kv-cache.cpp:856, deterministic replay in hybrid-idx init_batch, **total-token** cap U = min(ub, S × 512) (correction: not per stream, llama-batch.cpp:787-799), n_keep_tail kept by split_equal (:764-823), both contexts; **always-on graph-build check** → `llama_decode` error (no env gate); `tests/test-qsa-resplit.cpp` with asserts (i)–(vi); W9 two-context concurrency runs with `n_qsa_resplit` > 0 (§4.3, D17, W9) |
| P2-3 | Q10 control "KLD 0" between two binaries | Control = new rerun vs new = 0 plus new vs PR219 ≤ the measured unpinned kernel-change floor (AT-Q10Q6 (1)); KLD 0 only new-vs-new (§15 Q10). The memcmp-first alternative is not taken: the 200-query batch runs G4 on new and qsa_decode on PR219 (D3, fattn.cu:1154), so op-level bit identity cannot hold |
| P2-4 | Per-type bands go to 0 on PR219's ub-invariant masked route | Band floor B_T = max(measured, B_U); direct form max(reorder, ε-saturation) + 3·SE, never 0; PR219 partner forced to kind 1 for bf16 and the mixed pairs; Q3-A measured at c32768 (H4); the "measured, not borrowed" claim removed (§15, AT-W12-BANDS). Skeptic correction adopted: Q1-A for q4_0 used B_4096, so only its direct form was at 0; B_4096 is a q8_0 max chunk, relabelled |
| P2-5 | Q10/Q6 thresholds false-fail by design; PR219 partner not deterministic | τ = mean + 3·SD of ≥ 6 draws (≥ 19-draw quantile if its leave-one-out false-fail rate > 1/6); Q10 vs the median of ≥ 3 PR219 repeats; §15 determinism sentence corrected to exclude Q10 and to state that Q6's τ is a sample statistic. Skeptic correction adopted: `QSA_DECODE_MAX_QUERIES` is a compile-time define, so AT-Q10Q6 builds a lab binary |
| P2-6, P2-32 | Q12 is a point estimate with no margin | Primary Q12 = pinned paired diff vs PR219 `-nkvo` with the floored B_U; the unpinned value is a coarse screen at value + 3·SE (0.01587 / 0.01563); `-ngl 0` unpinned screen reported only (no measured basis) (§15 Q12) |
| P2-7 | Q8 D3 "within chunk SE" is a coin flip | D3 report-only; D2 = sign test ≥ 7/8 chunks > 0 (PR219 8/8 on both bases) and \|new − PR219\| ≤ 3·√2·SE; D1 unchanged (§15 Q8) |
| P2-8 | Q7 underpowered; §15 rejects family-wise control without adjudication | Q7 n ≥ 14 L/X prompts, one-sided margin fixed (0.617 → 0.436 pp at n 14; 0.77 was two-sided); S0–S5 excluded from the LCB; "no family-wise correction" withdrawn; two-stage rule on pre-registered holdouts, PR blocked only if both fail; expected family-wise false-fail rate reported (§15) |
| P2-9 | Shortcut non-f16 pairs at n_q 2–127 / 1024 unmeasured, flip cannot fire | Shortcut has its own G decision (D3); prefer_dense per (pair, n_q band) (D4); AT-W7PERF op grid + e2e `-ub 64/1024` + tg-with-MTP; P5 MTP cells. Skeptic correction adopted: HIP q8_0 at n_q 1024 is measured (C1: G4 wins), so the 1024 gap applies to q4_0, bf16 and the mixed pairs |
| P2-10 | `-nkvo` CPU pin is a type-asymmetric regression, only reported | Binding `-nkvo` perf gate per type (AT-W7PERF (c), AT-D4) at d0/d4096/d32768 incl. tg32 at depth; pre-registered fallbacks: CPU op dequantizes once per kv head (§3, now the default), then a named, logged host-KV-copy GPU route accepted by the guard; shortcut pin only if (d) passes (§10) |
| P2-11 | Vulkan pack variant omitted; AT-B1 one depth; §17 pre-accepts any miss | AT-B1 on d4096/d32768/d131072 × ub {512, 4096} × pp {512, 2048}; an S_fast miss is a PR-blocking FAIL (§16 P8, §17); vpack is an AT-B3 arm that ships only under M1. Skeptic correction adopted: the pack evidence is G4-only and Vulkan's G1 analogue on HIP chose staged V, so vpack is diagnostic, not mandatory; a quantized-only pack would break M1 |
| P2-12 | Guard after `ggml_backend_sched_reserve` sees NULL backends | Init guard on split_only reserve graphs (as resolve_fused_ops), every reserve variant, main/MTP/fit-probe; runtime guard after alloc_graph; inspects llama's gf; non-vacuity count ≥ 12 × (plan + attend) with a unit test (§10, W7) |
| P2-13 | `-fit` estimate counts only the pp graph and loses pins | Fix adopted as measured (R2-RES): drop the reset in reserve_size, per-backend max over every reserve graph in no_alloc (exact 64/64), guard on the fit path; AT-C18 gates it (§7.1.7). Skeptic detail adopted: ggml-alloc.c overwrites sizes per graph, hence an explicit max |
| P2-14 | Non-compact variant shape/forcing unspecified, size unmeasured, M5 cannot fail | Fully specified and measured (R2-RES): forcing flag, T = min(ub, 512) total, n_seqs, s, n_outputs; S = 1 because the S = 2 ceiling is exceeded (+940.54 > 886.25 MiB); exact host formula and unified device formula; M5 now a per-backend byte bound with instantiated values; ceiling and owner-decision fallback pre-registered; one-stream split shape → W9 STRICT (§7.1.4, M5, U38) |
| P2-15 | Plan dst size depends on G | get_alloc_size(PLAN) = max over G of plan_nbytes + scratch (§2.1, §5); AT-C5 forced G flip at n_q {65, 127, 129} with realloc abort |
| P2-16 | Vulkan "scalar tier for every other device" is 64-lane only | Portable Vulkan tier with stated requirements (§6.3); plan shader gets its own gate; supports_op = pipeline exists (§5); AT-B7 on lavapipe and RADV sg32 (W6); devices that lose QSA listed in §19 with the error text |
| P2-17 | Masked-route deletion removes QSA from every backend without a kernel | Explicit OWNER decision (D22, §20.6), default deletion with every backend named in §19 and an init message that names the real remedies (§10); the "never compiled or run" wording replaced by the sourced CUDA CI fact (§6.2). Skeptic correction adopted: under `-fa on` those backends may already compute dense attention silently (U42) |
| P2-18, P2-24 | Open PR #234 rewrites the same lines; no merge-order rule | §0 "Base and landing order": dry-run on the actual base, conflict-resolution rule with authorship kept, §12 wording, §11 re-derived by symbol, VGPR baseline from #234's files, `-ngl 0` re-run; reference arms = W0-lab on the actual base; rebase protocol; #225/#248/#247/#230 merge-tree in W13 |
| P2-19 | Vulkan workspace outside the bound descriptor range | Two descriptors with an explicit workspace sub-buffer and alignment round-up; `maxStorageBufferRange` check; W6 unsorted-row plan cases on RADV and lavapipe with the validation layer (§6.3, W6) |
| P2-20, P2-30 | Op bands never measured on test-backend-ops inputs; circular calibration | Measured (R2-BOP): SYN row holds for G4, fails for G1; UNI rows adopted, selected by census (G, splits, n_sel); transfer rules for classes without a PR counterpart, never calibrated on the kernel under test; U34 a gate (§14.2, W3–W5) |
| P2-21 | Op tests pass vacuously | Overflow err() over finite elements + inf-position check; poison reach asserted at build time from the CPU plan; determinism hook computes twice and memcmps; stress test asserts the overlap (§14.1, W6) |
| P2-22 | §14.3 census cannot observe what it asserts; hard-coded rows | Two sources (llama trace per executed ubatch; backend census through procs with exact keys); expected rows generated from the shipped policy and `sa_select`; forced-G arms after AT-A (§14.3, §6.2) |
| P2-23 | Zero bands (duplicate of P2-4) | As P2-4 |
| P2-25 | Q1-vk reference lacks master's default-on Vulkan GEMM changes | W0-Vulkan reference, B_vk re-derived from its own reorder, old reference only if the master delta is measured small (AT-W12-BANDS V1/V2); Q1-vk UNPROVEN until then |
| P2-26 | Session-file indexer field unnecessary and a global version break | Field removed; existing per-layer type check relied on; message wrapped; no version bump; W11 test rewritten (§9, W11, §17) |
| P2-27 | Evidence in a boot-cleared /tmp | E0 copies it to `EV` with checksums; citations rewritten to `EV/`; r1e path fixed; summaries committed in W12 (header, E0, W12) |
| P2-28 | AT-C16 criteria copied from 8-strip runtime numbers | Split into AT-C16a (W8, 8 strips, runtime peak) and AT-C16b (W9/W13, S = 1: 1 new / 11 reuse, reserve-to-reserve ≥ 1056 MiB at c131072 from the per-view formula) (§16.5, §14.3) |
| P2-29 | Q10 control is a hidden bit-identity requirement (duplicate of P2-3) | As P2-3 (option b) |
| P2-31 | W12 control value misquoted for the trim-0 reference | 1.371668e-4 (R1-UB table D, ref8t0) with per-chunk equality at %.9e (W12) |
| P2-33 | Q3-A borrows the c8192 band for c32768 | B_4096@32k measured in AT-W12-BANDS H4; Q3-A UNPROVEN until then (U39) |

### 20.6 OWNER decision (P2-17): backends without a SPARSE_ATTN kernel — DECIDED: option b

- **Decided 2026-11-05 by the owner: option b (init error).** Reason given: this fork does not support CUDA, MUSA, Metal, SYCL, OpenCL, CANN or WebGPU anyway. Implement option b exactly as below; do not implement option a. The PR description states it, with the backends named.

- **Default in this plan (option b):** delete the masked route everywhere (D5); QSA on CUDA, MUSA, Metal, SYCL, OpenCL, CANN, WebGPU, and on Vulkan devices that can create no tier, is an init error naming the remedies; §19 lists them. Reasons: the masked route is the memory cliff the goal forbids (base q8_0 21578.66 vs 4133.44 MiB at c131k, C16), and keeping it means keeping the masked half of `build_attn_qsa` (PR qwen4exp.cpp:1742-1784) as untested code on backends nobody here can run.
- **Option a:** keep the masked FA route as a logged, non-silent backstop used only when the layer's device backend reports no SPARSE_ATTN support for any type (never on HIP gfx1151 or RADV, so the HIP/Vulkan goal is unaffected), with the §10 guard checking it and the init log line naming it. Cost: the masked half of `build_attn_qsa` and its mask inputs stay; under `-fa on` those backends' FA supports_op does not check src[5] (Metal: ggml-metal-device.m:1743-1790), so option a would also need a backend capability query to avoid the silent dense result of U42.
- (Resolved: the owner chose option b.)

### 20.7 Deferred acceptance tests (round 2; owner rule: not run now, written as Prove-phase tests, scope unchanged)
- Round 2 index 1 (LAB-BANDS-E2E) → **AT-W12-BANDS** in W12 (§16.5); plan edits made now: band floor, kind-1 partner, Q3-A and Q1-vk labelled UNPROVEN (U39, §15), the "measured, not borrowed" claim removed.
- Round 2 index 2 (LAB-Q10Q6) → **AT-Q10Q6** before W8/W9 scoring; plan edits made now: Q10 control rule, mean + 3·SD / quantile τ, median of ≥ 3 PR219 repeats, §15 determinism sentence, τ_p1,q8_0 UNPROVEN.
- Round 2 index 3 (LAB-W7PERF) → **AT-W7PERF** on W7 (shortcut grid with its own G, per-band flip, MTP cell, binding `-nkvo` gate with fallbacks); the Phase 0 PR-tree bounds are dropped (bounds only, they decide nothing).
- Round 2 index 4 (LAB-VK) → AT-B1 extension, AT-B3 vpack arm, **AT-B7** portability on W6; plan edits made now: §17 FAIL rule, portable tier spec (§6.3), supports_op pipeline-exists rule (§5), W6 non-64-lane done-when, §19 device list.

### 20.8 Round-2 lab results integrated
- **R2-RES** (24 init-only server runs + 1 smoke, deterministic to the byte): re-split mechanism and the total-token correction (§4.3); S = 1 (§4.3, §7.1.4); V formulas and M5 bounds (§7, §16 M5); probe fix (§7.1.7); q8_0 ROCm0 variant excess (U37); one-stream split reserve impossible (U38); R1-RES's 513+511 split explained by n_keep_tail (§4.3).
- **R2-BOP** (142 dumps, 37 shapes, PR-verbatim kernels): UNI band rows and census-based selection (§14.2); SYN n_q 1 row exceeded by seed 3 (§14.2 (b)); PR decode nondeterministic on duplicate keys (§2.2, §14.1, §19); no ported case reaches qsa_prefill direct V (§12, §14.1); mixed packed-both route without a band (§14.2 (c), U34).

### 20.9 Rejected by skeptics (round 2; not acted on as blocking, small fixes taken where noted)
- **"Q5 op bands are gated on input distributions and split counts that were never calibrated."** Rejected as blocking: the W6 part was already covered by §14.2 (bands measured at W3–W6 before use); the 13–34× evidence compared different metrics; the 33-split row mechanistically bounds 2–32 splits. Taken: the wording inconsistency between W3/W4 (binding) and §14.1/U34 (report-only) is gone, because R2-BOP measured the UNI rows and U34 is now a gate.
- **"The QPIN hook claim in §4.5 is false, and the planned rows renaming removes the hook's reference-purity assert."** Rejected as blocking: every reference is recorded on the PR219 arm, which keeps the PR naming, so the record-mode assert (qpin.patch:67) still guards it; no gate pins non-compact ubatches; all-visible rows are deterministic. Taken: §4.5 now says `av` (and `te`) are deliberately not matched, and non-compact rows have their own name.
- **"The CPU op is both the sequential-sum oracle and the production kernel, while AT-D2 demands CPU speed parity."** Rejected: vertical SIMD across the r GQA heads keeps the sequential-over-d order (bands_ref.h:239) and was not slower than a W0-style horizontal dot in the skeptic's microbenchmark (0.272 vs 0.312 ms, `EV/skept_ocon/mb.cpp`); the q8_0/f16 `-nkvo` ratio is W0's own CPU FA baseline. Taken: §3 threads over (s, i, kv head) with K/V dequantized once per kv head, and the TU is built with `-ffp-contract=off` with a W1 bitwise check against bands_ref.h.

### 20.10 Minor notes (round 2) and where each went
- Trim eligibility after deleting `maskless`: §4.4 (compact CB only; TE with mask-row slicing; never the av shortcut).
- CPU op SIMD order: §3 (vertical SIMD keeps the contract order; the note's "allow any f32 dot order" is not adopted, because the shipped op is the oracle and the skeptic measurement shows no speed reason to diverge).
- Masked-route determinism claim in §19: reworded (§19, §2.2 item 4 scoped to the op given its inputs).
- > 1M keys: n_kv 2^21 + 4 op case and an `n_ctx_seq` cap (§4.4, §14.1, U12).
- TE −1 filter: gather of `expanded` with the view shapes specified (§4.1 row 4).
- `v_trans = false` cache-wide and non-QSA layers: D21 and §19.
- Duplicate (pos, y, x) keys: API list at W9, non-compact route instead of an error if reachable (§4.2 step 4, U35).
- B_4096 label: §15 Q1 (max chunk, not mean + 3·SE).
- QPIN `av`: §4.5.
- SYN n_q 1 band on 288 rows: §14.2 (b) (fixture enlarged before it is a hard gate).
- O_C18 bf16 saturation vs the contract's ±inf: §3.
- Q11a lo-region band borrowed: §15 Q11a, U40, AT-W12-BANDS H1.
- Q7 S0–S5 excluded from the LCB: §15 Q7.
- P1 θ: adopted C10's bound, θ = 0.995 (§16 P1).
- P3 absolute floor new_q8/new_f16 ≥ 0.90: §16 P3.
- AT-A ledger prediction unsourced: labelled arithmetic/UNPROVEN with its derivation (AT-A).
- §19 "~421 t/s not PR-specific": reworded per C7 Limits (§19).
- P6(b) Fisher test without power: reported only (§16 P6).
- MTP generation throughput gate: P5 MTP cells.
- ub8192 column and M5's hidden pool items: §7 bullet, M5 "moved pool items" term.
- D8 CAPPED at G4 n_q 128–512: re-derived at n_q 128/256 in AT-A.
- M2 fallback: §16 M2 (separate workspace tensor, then owner).
- P6(b) power (duplicate): §16 P6.
- Vulkan 1M f32 cells above the 1 GiB buffer max: Vulkan expected-unsupported list (§14.1, §6.3).
- CPU graph work buffer outside the sched reserve: §3 roles, §19.
- TE `top_k` pool use on HIP: §6.2, AT-C12 M3, §19.
- Strict check logs the first growing ubatch via the realloc callback: §7.1.6.
- Determinism per allocator placement (gate-fusion overlap check): §2.2 item 4.
- RPC op numbering: §2 (ops appended before GGML_OP_COUNT, minor version bump), §19.
- HIP arch mapping: §5 "HIP tier selection per arch", §6.2, §19, U15.
- DSV4 FA + src[5] user: §11.
- Vulkan line citations from the PR tree: master lines cited where given (§6.3, D20, §11), others located by symbol at W6/W13.
- W0 Vulkan counts vs C11: W0 done-when (QSA_* ops only; W0 counts become the reference).
- `-fa off` with quantized V already an init error: D21, §19.
- Init message names the backend registry and the cause: §10.
- `--fail-unsupported` and intentionally unsupported cases: per-backend expected-unsupported list (§6.3, §14.1).
- `test_sparse_attn_gated` both V-loader regimes: §14.1.
- Suite-level census accumulator and the expected instance list: §14.1, §6.2 (`census_list` proc, generated header).
- CPU-backend launch counter cannot live in ggml-cpu: llama-side check over `ggml_backend_sched_get_tensor_backend` (W7).
- Overflow test for f16 V impossible: restricted to bf16 and q8_0 V (§14.1).
