# IMPLEMENTATION log: qsa-rewrite (halo-box/strix-llama.cpp)

This is a work log that keeps growing. It is **not** part of the evidence set: E0 leaves it out of the copy and out of `SHA256SUMS`.

## Base

- Worktree `strix-002:~/dev/wt_rewrite`, branch `qsa-rewrite`. It was created by E0 with `git worktree add -b qsa-rewrite ~/dev/wt_rewrite ea5285ddfc4eb2e457886c4105c77a0cece5f205`, run in `~/dev/wt_gdn` (remote `origin` = https://github.com/halo-box/strix-llama.cpp).
- **Base SHA: `ea5285ddfc4eb2e457886c4105c77a0cece5f205`.** This is origin/master after `git fetch origin` on 2026-11-05 (`git rev-parse origin/master`), with subject "Merge pull request #251 from halo-box/fix/qwen4exp-qsa-layer-types". It is the base the plan names (PLAN-v3 §0 "Branch").
- Commit identity: `user.email=12990773+gaetan-puleo@users.noreply.github.com` (`git config --show-origin` reports it from `~/dev/wt_gdn/.git/config`, which all worktrees of wt_gdn share). E0 ran `git config user.email` with this value. Earlier commits in this repository already carry this author and committer email (`git log --all --author=Puleo`, for example 9b7c19212 and 468bc6455), so it was probably set already. That it was set before E0 is UNPROVEN: git keeps no history of config values.

## E0: make the evidence durable (not a commit)

**What changed**
- Evidence copy `EV` = `strix-002:~/dev/wt_lab/evidence/final219/`, a new untracked directory inside the wt_lab worktree. It holds `SP/final219/` (all of it except `SHA256SUMS` and this `IMPLEMENTATION.md`), plus `SP/bs_skeptic.py` → `EV/bs_skeptic.py` and `SP/skept_ocon/` → `EV/skept_ocon/` (the files `mb` and `mb.cpp`).
- Copy method: `cd "$SP" && tar -cf - --exclude=SHA256SUMS --exclude=IMPLEMENTATION.md -C final219 . -C .. bs_skeptic.py skept_ocon | ssh strix-002 'tar -xpf - -C ~/dev/wt_lab/evidence/final219'`. The plan names rsync, but rsync fails here: `protocol version mismatch -- is your shell clean?` / `rsync error: protocol incompatibility (code 2) at compat.c(622)`, because the strix-002 login banner writes export lines to stdout. tar over ssh reads only stdin on the remote side, so the banner does no harm.
- Checksums. Each side builds its list over the same file set, in path order with `LC_ALL=C`:
  - local: `SP/final219/SHA256SUMS`;
  - remote: `EV/SHA256SUMS`, built on strix-002 by `find . -type f ! -name SHA256SUMS ! -name IMPLEMENTATION.md -print0 | sort -z | xargs -0 sha256sum`.
- `PLAN-v3.md` edits, made before the final copy so that the EV copy carries them. A pre-edit backup sits outside the evidence set at `/tmp/claude-1000/-home-gaetan-puleo/78d4f4eb-9ba7-448e-8841-dce6638c30f6/scratchpad/PLAN-v3.pre-E0.bak.md`.
  - Header `EV` line (:13): "75 MB, `du -sh`" becomes "80 MB `du -shc`, 2587 files …; per-file SHA-256 in `EV/SHA256SUMS`". 2587 files is the line count of both SHA256SUMS. 80M is the output of `du -sh .` on EV.
  - E0 "What" (:451): the copy command now runs from `cd "$SP"` with relative sources, so it contains no `SP/` path. It names the two exclusions and the shared `find … | sort -z | xargs -0 sha256sum` recipe, compared with `cmp`.
  - E0 "Done when" (:452): the grep is now `grep -n '[S]P/\|/[t]mp/'`, which does not match its own line, and the definition is placed correctly. It lives in the header paragraph "Evidence location", not in §0.
- r1e path (P2-27). `EV/r1e/r1e/r1e_patch.py` exists: `ls SP/final219/r1e/r1e/` lists `r1e_patch.py`. The plan already cites that path at :467. The only other mention is the E0 note at :451 that records the old, wrong path. `MEASUREMENTS.md:3570,3730` say `r1e/r1e_patch.py` relative to the R1-E lab directory. They were left unchanged because they are evidence and are now checksummed.
- Citation resolution. Every `EV/…` path in PLAN-v3.md was checked against the local tree: 23 distinct paths, all of which exist. The brace glob `EV/c10/logs/d{32768,131072}_f16_*_text.json` matches the files `d32768_f16_{0..}_*_text.json` and `d131072_f16_{0..6}_*_text.json`. `EV/X` is only the placeholder in the definition.

**Done-when check**
- Checksums equal: `wc -l` gives 2587 on both lists, `cmp` reports them equal (`CMP_EQUAL`), and both files hash to `7e378bc985f0c6a192ad43f3630795c9284c51194b1af3c4f5b90a23ac83cd63` (sha256sum of the local file and of `EV/SHA256SUMS` on strix-002). An independent check also passes: the local list was piped into `sha256sum -c --quiet -` inside EV and printed `VERIFY_ALL_OK`. The plan's own hash is `aced0f46cdac5c63b4f3e2fff30c69e1a860ed3ccfb454f84962e90fdc8acce1 ./PLAN-v3.md` on both sides.
- No `SP/` or `/tmp/` citations. `grep -n '[S]P/\|/[t]mp/' PLAN-v3.md`, run on the local copy and on `EV/PLAN-v3.md`, returns only :12 and :13, the "Evidence location" definition paragraph. The PR description and the #219 comment are not written yet: only their source material exists, in plan §19 (:877–910), and the grep finds nothing there. **W13 / the orchestrator must re-run the same grep on the final PR text and on the #219 comment before posting.**
- Caveat: if `SP/final219/PLAN-v3.md` (or any other evidence file) is edited after this point, EV and its SHA256SUMS have to be refreshed with the same commands, or the edit has to go into EV directly.

**Commits:** none. E0 is "not a commit" (PLAN-v3 §13 E0).

**Build:** HIP was built at the base in `~/dev/wt_rewrite/build` with the rule flags (`-DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/lib/llvm/bin/clang -DCMAKE_PREFIX_PATH=/opt/rocm -DLLAMA_CURL=OFF`, `cmake --build build -j 16`). The results were `cfg_rc=0` and `build_rc=0`, and the build produced `build/bin/test-backend-ops`, `llama-perplexity` and `llama-cli`. The logs are `strix-002:~/dev/wt_rewrite_E0_cfg.log` and `~/dev/wt_rewrite_E0_build.log`. The build log has 85 `warning:` lines and one line matching "error", which is the non-fatal `-- UI: download dist.tar.gz.sha256 from b12299 failed`. Vulkan was not built because E0 touches no Vulkan code.

**Tests:** for E0 the plan requires only the checksum comparison and the grep, and both pass (above). E0 changes no code, so no test-backend-ops subset was run. `git status --short` in wt_rewrite is empty.

## W0: import #219 on the base

All commands ran on strix-002. Every build, GPU run and test ran under `flock ~/gpu.lock`, and no other agent was active. Logs are in `strix-002:~/dev/wt_rewrite_logs/W0/` (668 KB, no file over 100 MB).

**Base check (§0 rule (1))**
- `git fetch origin` gives `origin/master = ea5285ddfc4eb2e457886c4105c77a0cece5f205`. That is the E0 base, so the base did not move.
- #234 is not in the base. `git fetch origin pull/234/head` gives head `7774790458a5f467376a5a109b69cda9ed145821`, and `git merge-base --is-ancestor 777479045 ea5285ddf` is false. `git log --oneline 13a279bfa..ea5285ddf` lists #240, #242, #241, #244, #245, #246, dc80afcfe and the #251 merge only.
- The base's HIP code equals 13a279bfa's. `git diff --stat 13a279bfa ea5285ddf -- ggml/src/ggml-cuda src ggml/src/ggml-alloc.c ggml/src/ggml-backend.cpp ggml/src/ggml.c ggml/include common tools tests` shows only `tests/test-backend-ops.cpp | 99 +`. The full 13a279bfa..ea5285ddf stat is conversion/qwen4exp.py plus ggml-vulkan files plus test-backend-ops.cpp.
- Dry-run: `GIT_INDEX_FILE=$(mktemp -u) git read-tree ea5285ddf`, then for each of 68da4191d, b60447506 and 94e086847 in order, `git diff --binary c^ c | git apply --cached --check` followed by `git apply --cached`. Result: `68da4191d check OK`, `b60447506 check OK`, `94e086847 check OK`, and the temporary tree is `449aba6970e93af2278908bb341239f6c7ab70b2`. The temporary index was deleted afterwards.

**What changed (branch `qsa-rewrite` in `~/dev/wt_rewrite`)**
- `git cherry-pick -x 68da4191d b60447506 94e086847`, then `git commit --amend -F` on each commit. The amend keeps the original subject and body. It adds one paragraph, `[qsa-rewrite W0] …`, placed before the trailers, which records the base SHA, the dry-run result (OK, OK, OK, tree 449aba697), and the fact that #234 is not in the base. It then appends this session's `Claude-Session:` line. The original trailers stay, followed by git's `(cherry picked from commit …)` line.
- No `[conflict resolved]` line was added, because #234 is not in the base and so there was no conflict.
- Commits (author `Iggy Jackson <iggy@theiggy.com>`, committer `12990773+gaetan-puleo@users.noreply.github.com`; source `git log --format='%h %an <%ae> | %ce | %s' ea5285ddf..HEAD`):
  - `509e8ee3b` qwen4exp : q8_0 K/V for the QSA selected-key kernels, share score-strip input views (from 68da4191d)
  - `21fd1b73d` qwen4exp : probe selected-key support with the cache types, q8_0 QSA tests (from b60447506)
  - `0e1121792` hip : q8_0 direct K/V reads in qsa_prefill (from 94e086847)
- The HEAD tree is `449aba6970e93af2278908bb341239f6c7ab70b2` (`git rev-parse HEAD^{tree}`), identical to the dry-run tree. `git status --short` is empty, and nothing was pushed.
- An earlier attempt left no blank line between the body and the W0 paragraph. It was undone with `git reset --hard ea5285ddf` inside wt_rewrite and redone. Those commits (7e2daf321, d7f84fdb0, eb67be78a) are unreferenced.

**W0-lab (PR219 arm of every gate, §0 / P2-24)**
- Worktree: `git worktree add --detach ~/dev/wt_lab/w0lab 0e1121792`, a detached worktree outside wt_rewrite.
- Uncommitted lab patches, all of which do nothing unless their env var is set:
  - (1) `git apply EV/r1e/r1e/r1e_full.diff`, which is byte-identical to the current `git diff` of `~/dev/wt_lab/r1av/tree` (`cmp`). It contains the c1 lab diff (GGML_QSA_TRACE, GGML_QSA_MASKED_DENSE, GGML_QSA_DECODE_MAXQ, LLAMA_QSA_ALLVIS 1/2/3), QPIN with the `qsa_rows_cb/tk` hook names, QPIN_IDX_TYPE, QKL_TAIL, `QKLD_CHUNK … all= lo= hi=` at %.9e with QKL_SPLIT, and the r1e `QPIN_QSA_CHUNK`.
  - (2) `python3 -I ~/dev/wt_lab/w0lab_src/w0lab_r1q8_hook.py`. This script adds the r1q8 post-pin defect hook verbatim from `EV/r1q8/r1q8_lab.diff`, which equals `git diff` of `~/dev/wt_lab/r1q8/tree` (`cmp`). The hook covers `lab_def_state`/`lab_init`/`lab_def_row`, LAB_DEF d1/d2/d3 in `qpin_rows`/`qpin_cb`, `R1Q8_SUMMARY`, and r1q8's `QKLD_CHUNK n= kld= same_top=` line.
  - Both QKLD_CHUNK formats are printed. Their regexes are disjoint, so `r1ub/analyze.py`, `r1av/pair.py`, `r1e/q4_analyze.py` (`all=`) and `r1q8/analyze*.py` (`n=\d+ kld=`) all parse W0-lab logs unchanged.
- A `grep '^[+-]'` diff of the combined `~/dev/wt_lab/w0lab_src/w0lab_lab.diff` (582 lines, sha256 `de60fdcc…c98c21`) against r1e_full.diff shows only the r1q8 additions listed above. The inputs are also kept in `~/dev/wt_lab/w0lab_src/`: `r1e_full.diff` sha256 `986ee653…5f1bccb` and `w0lab_r1q8_hook.py` sha256 `a00491e7…e9328f`.
- Builds:
  - HIP `build/` with the rule flags.
  - Vulkan `build-vk/` with C11's flags: `-DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DCMAKE_CXX_FLAGS=-I$HOME/.local/spirv-headers/include -DSPIRV-Headers_DIR=$HOME/.local/spirv-headers/share/cmake/SPIRV-Headers` (MEASUREMENTS.md C11 "Configure").
  - The C10-patched llama-bench (`~/dev/wt_lab/c10/src`), linked against each build by `~/dev/wt_lab/w0lab_src/build_bench_w0lab.sh build|build-vk`. This is the c10/build.sh and c11/build_bench_vk.sh recipe with the tree and build directory as parameters and the defines read from the build's own `flags.make`. Outputs: `~/dev/wt_lab/w0lab/build/bench_c10/llama-bench` and `~/dev/wt_lab/w0lab/build-vk/bench_c10/llama-bench`. ldd resolves to each build's `bin/` (logs `w0lab_bench_build.log` and `w0lab_bench_build-vk.log`).
  - The C10 source is valid on the base because tools/llama-bench is unchanged between 13a279bfa and ea5285ddf (diff stat above).

**Build results**
- W0 HIP `~/dev/wt_rewrite/build`, `cmake --build build -j 16`: build_rc=0 (`~/dev/wt_rewrite_W0_build.log`).
- W0 Vulkan `~/dev/wt_rewrite/build-vk`, C11 flags, full build: cfg_rc=0 and build_rc=0 (`rw_vk_cfg.log`, `rw_vk_build.log`, 75 `warning:` lines).
- W0-lab HIP: cfg_rc=0 and build_rc=0 (`w0lab_hip_*.log`, 86 warnings). W0-lab Vulkan: cfg_rc=0 and build_rc=0 (`w0lab_vk_*.log`, 76 warnings). Both bench builds: rc=0.

**Tests and done-when**
- HIP ROCm0, `test-backend-ops test -o <OP> -b ROCm0`, one op per run. The results below come from W0 (`hip_<OP>.log`). W0-lab (`hip_w0lab_<OP>.log`) gives the same counts and identical per-case result lines (`cmp`).
  - QSA_PREFILL_MASKLESS: 26/26 passed.
  - QSA_DECODE_MASKLESS: 52/52 passed, plus 1 not supported (`queries=9,keys=4096,selected=257,ratio=4,…,type_k=q8_0,type_v=q8_0`; 68da4191d admits q8_0 decode only at GQA 12).
  - QSA_PREFILL: 26/26 passed.
  - QSA_DECODE: 92/92 passed.
  - 0 FAIL in all four, and every run returned rc 0. **Pass.**
- Vulkan0, `test-backend-ops test -o <OP> -b Vulkan0`. These are the **W0 reference counts** (`vk_rw_<OP>.log`). W0-lab gives the same (`vk_w0lab_<OP>.log`).
  - QSA_PREFILL_MASKLESS: 14/14 plus 12 not supported.
  - QSA_DECODE_MASKLESS: 34/34 plus 19 not supported.
  - QSA_PREFILL: 26/26.
  - QSA_DECODE: 92/92.
  - These equal the C11 counts (14/14 + 12, 34/34 + 19, 26/26, 92/92). The sorted per-case result lines are also identical to C11's `~/dev/wt_lab/c11/tbo_<OP>.log`: sha256 prefixes 48effa7b0fa4, 923c7a047f94, 93a065a39f7d and 2d1423414f4e on all three of C11, W0 and W0-lab. So the QSA_* ops are unchanged by master's Vulkan commits. Ops outside QSA_* were not run, as the plan requires.
- `sched_reserve` uses the C16 recipe: `llama-perplexity -f ~/dev/wt_lab/c16/tiny.txt -c 131072 -b 4096 -ub 4096 --chunks 1 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -lv 4 -lm mmap -lzm on-direct`. It gives `ROCm0 compute buffer size = 4133.70 MiB` and `ROCm_Host … = 201.70 MiB` on both W0 and W0-lab (`c16_rw_q8.log`, `c16_w0lab_q8.log`; rc 0, ending with the expected "need at least" exit). This equals C16 `pr_q8_mmap_lzm_r1` (4133.70 / 201.70, MEASUREMENTS.md:2080). **Pass.**
- PPL uses the C15 recipe: `llama-perplexity -f wiki.test.raw -c 8192 -b 4096 -ub 4096 --chunks 2 -ngl 99 -fa on -lm mmap -lzm on-direct -ctk/-ctv T -lv 3`. f16 gives `PPL = 1.9449 +/- 0.03433` ([1]2.1025, [2]1.9449) and q8_0 gives `PPL = 1.9519 +/- 0.03464` ([1]2.1197, [2]1.9519), identical on W0 and W0-lab (`c15_{rw,w0lab}_{f16,q8_0}_ub4096.log`). This equals C15 at ub4096 (1.9449 / 1.9519, MEASUREMENTS.md:2038). **Pass.** W0-lab giving the same numbers as W0 shows that its lab patches have no effect when their env vars are unset, at least for these two runs.
- The full test-backend-ops suite was never run. No file over 1 GB was created.

## W12: validation tooling and per-chunk KLD precision

Everything ran on strix-002. Every build, GPU run and test ran under `flock ~/gpu.lock`, and no other agent was active. Logs: `strix-002:~/dev/wt_rewrite_logs/W12/` (builds, tests, commit messages) and `strix-002:~/dev/wt_lab/w12/` (campaign: `campaign.out`, `campaign_S.out`, `logs/`, `logs_hookobs/`; 2.9 MB in total). The committed copy of the results is `scripts/qsa-validation/summaries/w12_*`.

**Commits (branch `qsa-rewrite` in `~/dev/wt_rewrite`, author and committer `12990773+gaetan-puleo@users.noreply.github.com`, not pushed; `git log --format='%h %an <%ae> | %s' ea5285ddf..HEAD`)**
- `d14e4bd4d` perplexity : per-chunk KLD at full precision (QKLD_CHUNK), QKL_TAIL, QKL_SPLIT
- `003427142` scripts : QSA validation tooling (QPIN arm patches, harnesses, census checker, bench statistic)
- `11e7e784f` perplexity : QKL_HASH, a per-chunk hash of the scored logits (bit-identity of a self-replay)
- `52c2c1a95` scripts : W12 / AT-W12-BANDS results on W0-lab (summaries)
- None of these commits ports #219 code, so none carries a Co-authored-by trailer. HEAD is `52c2c1a95` and `git status --short` is empty.

**What changed**
- `tools/perplexity/perplexity.cpp`:
  - `qkl_first_scored()`: `QKL_TAIL=N` scores the last N positions (2 ≤ N ≤ n_ctx − 1, otherwise rejected). It is used by `perplexity()` (base recording) and by `kl_divergence()`. The KLD buffers are sized from it.
  - `qkl_split()` / `QKL_SPLIT` (default 2048).
  - `QKLD_CHUNK <c> all= lo= nlo= hi= nhi=` at %.9e, computed from that chunk's `kld_values`. The arithmetic and the format are those of the R1-AV hunk (`EV/r1av/r1av_full.diff`), so the existing analyzers parse it.
  - `qkl_hash_print()`: with `QKL_HASH=1`, prints `QKL_HASH {base|kld} <c> <hash> n=` (FNV-1a over the bits of the scored logits).
  - Without the env variables, the only output change is the QKLD_CHUNK line.
- `scripts/qsa-validation/` (124 files, index in its `README.md`):
  - `patches/01-qpin.patch`: `~/dev/wt_measQ/q/qpin.patch` rebased onto the branch, name table unchanged.
  - `patches/02-r1e-qsa-chunk.patch`: QPIN_QSA_CHUNK.
  - `patches/03-r1q8-defect-hook.patch`: LAB_DEF d1/d2/d3, R1Q8_SUMMARY, and the `QKLD_CHUNK n= kld=` line.
  - `patches/04-c1-lab.patch`: GGML_QSA_TRACE, MASKED_DENSE, DECODE_MAXQ, LLAMA_QSA_ALLVIS.
  - `patches/apply.sh`.
  - `patches/c15/` (c15_trace.diff, c15_patch.py).
  - `w12/{run.sh,campaign.sh,analyze_w12.py}`.
  - `census_check.py` (§14.3; modes expect/keyset/recipe; `selftest` PASS).
  - `bench/analyze.py` (§16 mean-of-reps hierarchical bootstrap, 4000 draws, seed 1; ratio and ratio-of-ratios; PASS/FAIL/RERUN), plus `bench/c10/`.
  - `kfd/` (C7 shim and sampler), `harness/{c3,c5,c18,bands,bandsop}` (bandsop includes the UNI/TBO fixtures and `tree_bandsop.diff`), `campaigns/` (r1 text/img, r1c runB3, r1s, r2res srv/parse, and the r1ub/r1e/r1q8 runners and analyzers), and `summaries/` (kdecide.md, compact*.md, analysis_op.md, res2_summary.md, probe_gap.txt, the R1 analysis outputs, and the W12 results).
- How the patches were built: the four stacked patches were generated in a scratch worktree (since removed). That worktree applied the round-1 pieces in order on `d14e4bd4d`. Applied to the branch, they reproduce `~/dev/wt_lab/w0lab` (W0's lab tree) byte for byte (`cmp`) in fattn.cu, qsa-decode.cu, qsa-prefill.cu, llama-memory-hybrid-idx.cpp and qwen4exp.cpp. Two files differ: perplexity.cpp (now the library commit) and one line of llama-context.cpp (finding F1).

**PR219 measurement arm.** The arm is `~/dev/wt_lab/w0lab12`, a new detached worktree at `11e7e784f` with `patches/apply.sh` applied. It is W0-lab built from the committed patches.
- The old `~/dev/wt_lab/w0lab` (W0's lab tree) was not modified. An attempt to reset it was denied by the permission classifier, so the new tree was made instead. `~/dev/wt_lab/w0lab` is still at 0e1121792 plus the uncommitted round-1 lab diff (md5 0337775d… = `w0lab_src/w0lab_lab.diff`).
- **Later gates should use `w0lab12` as "W0-lab"**, or recreate the arm with `apply.sh` on the branch.

**Builds**
- `wt_rewrite` HIP `build/` (rule flags, `-j 16`): rc 0 after each change. `build_hip_ppl.log` and `build_hip_hash.log` show 0 warnings, because those builds were incremental.
- `wt_rewrite` Vulkan `build-vk/`: rc 0 (`build_vk.log`).
- `w0lab12` HIP `build/` and Vulkan `build-vk/` (C11 flags): configure and build rc 0, 85 and 75 `warning:` lines (`w0lab12_{hip,vk}_build.log`). The incremental rebuilds `*_build2.log` and `*_build3.log` are rc 0.

**Tests (per op, one op per run, `-b ROCm0` / `-b Vulkan0`, under flock; logs `hip_head_<OP>.log`, `vk_head_<OP>.log` at HEAD 11e7e784f, plus `hip_rw_<OP>.log` at d14e4bd4d)**
- HIP:
  - QSA_PREFILL_MASKLESS: 26/26.
  - QSA_DECODE_MASKLESS: 52/52, plus 1 not supported.
  - QSA_PREFILL: 26/26.
  - QSA_DECODE: 92/92.
- Vulkan:
  - QSA_PREFILL_MASKLESS: 14/14, plus 12 not supported.
  - QSA_DECODE_MASKLESS: 34/34, plus 19 not supported.
  - QSA_PREFILL: 26/26.
  - QSA_DECODE: 92/92.
- 0 FAIL and rc 0 everywhere. These are the W0 reference counts.
- `census_check.py selftest`: SELFTEST PASS.
- The full suite was never run.

**Done-when (W0-lab12; analyzer `scripts/qsa-validation/summaries/w12_bands.md`, per-chunk data in `w12_chunks.tsv`)**
- Campaign: `campaign.sh A C D V B E`, then `campaign.sh S`. That is 105 perplexity runs, all rc 0, with rows_absent = moe_absent = 0 in every replay arm (`campaign.out`, `campaign_S.out`).
- **QPIN self-replay: PASS.**
  - c8192 × 8: max |chunk| 3.697e-7.
  - c32768 × 8: max |chunk| 2.320e-7.
  - Both are ≤ 3.7e-7.
  - In addition, all 6 references are bit-identical on replay: the `QKL_HASH` of the record equals that of the self-replay in every chunk (8, 16, 8, 8, 8 and 16 chunks). The phase-S re-record reproduces each campaign self-replay at %.9e.
- **Control: PASS.**
  - q8_0 ub4096 pinned against the re-recorded ref8t0 gives mean **1.371668e-4, chunk SE 1.253e-5**. All 8 chunks are equal at %.9e to `EV/r1ub/logs/h_q8_0_u4096.log`.
  - The f16 self-replay chunks are equal to `h_f16_u4096.log`.
  - The c32768 × 8 f16, q8_0 and q4_0 ub4096 arms equal `h32x8_*_u4096.log` per chunk (q8_0 1.453076e-4, q4_0 5.669339e-3).
  - Agreement against ref16t0 is **0.996743** (SE 3.00e-5), and every chunk equals `EV/r1e/r1e/logs/e_q8_idxq8.stats` at %.9f.
- **AT-W12-BANDS: PASS** (analyzer verdict). The bands below are frozen in `w12_bands.md`, and the orchestrator should copy them into PLAN-v3 §15.
  - H1 (Q1-A, c8192 × 8): every raw band is below the floor, so **B_T,4096 = 2.0e-4** for each type. The raw values (mean + 3 SE of ub2048 − ub4096) are:
    - bf16 default route: bit-identical (masked `dense_tile_vec_mma`);
    - bf16 kind 1: 1.021e-4;
    - q8_0K/f16V: 1.852e-5;
    - f16K/q8_0V: 3.713e-5.
    - For the mixed pairs, the default route (tk, masked qsa_prefill) and kind 1 (cb) give identical pinned KLD in every chunk.
  - Direct form (band = max(ub2048 reorder, ε 1e-7 saturation) + 3 SE). Every arm of a type, its clean base included, ran with `LAB_DEF=d3`, so base and arms share the graph splits (F1).

    | type | ub2048 reorder | ε 1e-7 saturation | band |
    |---|---|---|---|
    | f16 | 1.084e-4 | 7.016e-5 | 1.403e-4 |
    | q8_0 | 1.687e-4 | 9.216e-5 | 2.310e-4 (equal to the plan's 2.31e-4) |
    | q4_0 (H3) | −2.2e-9 (bit-identical route) | **1.058e-3** (SE 1.28e-4) | **1.441e-3** |
    | bf16 | −1.4e-8 | 7.847e-5 | 1.032e-4 |
    | q8_0K/f16V | 1.341e-4 | 8.246e-5 | 1.807e-4 |
    | f16K/q8_0V | 1.250e-4 | 8.192e-5 | 1.623e-4 |

  - H2 (Q4, ref16t0, kind-1 partner). In all three types the |ub2048 shift| + 3 SE term is below the 1.178e-4 floor, so B_T = 1.178e-4.

    | type | kind-1 mean | threshold | kind 1 − tk |
    |---|---|---|---|
    | bf16 | 0.997257 | **0.997140** | +1.83e-4 |
    | q8_0K/f16V | 0.996860 | **0.996742** | −3.32e-4 |
    | f16K/q8_0V | 0.997239 | **0.997121** | −2.98e-4 |

  - H4 (c32768 × 8): **B_4096@32k = 1.801e-4**, set by f16 (+1.073e-4, SE 2.43e-5). q8_0 gives 1.106e-5, and q4_0 is bit-identical.
    - C17 cross-check: f16 ub2048 chunks 0 and 1 are 1.365e-4 and 6.51e-5 here, against C17's 1.4e-4 and 0.62e-4.
  - V1/V2 (Vulkan):
    - The W0-Vulkan self-replay is bit-identical (hash).
    - (ii) equals (i) at %.9e.
    - (i), the round-1 PR219+#240 Vulkan binary against the W0-Vulkan reference, gives **5.546e-3** (SE 7.57e-4). That is ≥ 6.4e-4, so **the old vref8t0 is invalid**, and Q1-vk pairs new-Vulkan(T) with W0-Vulkan(T) only.
    - (iii) W0-Vulkan f16 ub2048 − ub4096 = 2.427e-4 (SE 2.62e-5), so **B_vk = 3.214e-4**.
    - (iv) `GGML_VK_DENSE_DQ8=0 GGML_VK_HC_DQ8=0` on W0-Vulkan gives 5.546185e-3, against 5.546173e-3 for the old binary. Turning master's dq8 GEMM defaults off moves W0-Vulkan to within 1.2e-9 of the round-1 binary.
  - U40 (Q11a lo/hi; ref8L = c8192 × 8, QKL_TAIL=8191; ub512 at -b 512 against ub4096):
    - **B_512,lo = 2.073e-4**, set by f16 lo (+1.599e-4, SE 1.58e-5).
    - hi max 1.806e-4.
    - q8_0, q4_0, bf16 and the mixed pairs all give ≤ 5.41e-5.
  - Q11b (ref2k = c2048 × 16, dense f16 b/ub 2048): ub512 − ub2048 is bit-identical for q8_0, q4_0, bf16 and both mixed pairs. f16 gives +1.616e-4 (SE 7.9e-6). With the floor, every Q11b band is **2.03e-4**.

**Findings (sources: the files named)**
- **F1: the round-1 r1q8 hook was not inert.**
  - The hook's `qpin_cb` asked to observe `attn_output-23` in every QPIN run (`EV/r1q8/r1q8_lab.diff`), and observing a tensor splits the graph.
  - With that hook, the first pass (`w12/logs_hookobs/`, stopped after 5 arms) reproduced R1-Q8's chunks at %.9e, not R1-UB's: q8_0 2.030895e-4 … (mean 1.375064e-4), f16 ub2048 1.643766e-4 …, and the f16 self-replay chunks (`EV/r1q8/logs/{q8_off,f16_ub2048,f16_off}.log`). This was on a **trim-0** reference.
  - So the plan's attribution of 1.375064e-4 to "the default-trim reference ref8" (PLAN-v3 §13 W12 / P2-31) is wrong: the difference is the observation, not the trim.
  - Fix in patch 03: the tensor is observed only under `LAB_DEF=d3`. The control then reproduces R1-UB exactly.
  - Consequence: R1-Q8's numbers (B_4096 = 2.0e-4, the 9.39e-5 saturation, q8_0 direct 1.63e-4) were measured on the observed graph. The unobserved-graph values here are close (direct q8_0 1.687e-4 / 9.216e-5), but they are not the same measurement.
- **F2: "≤ 3.7e-7" is not a bound.**
  - The ref16t0, vref8t0 and ref2k self-replays print max chunks of 4.672e-7, 3.787e-7 and 7.508e-7.
  - The ref16t0 chunks are identical at %.9e to R1-E's own round-1 `e_f16_self.log`.
  - Bit identity is now shown by `QKL_HASH` (all 6 references). The analyzer reports the 3.7e-7 figure without gating on it. This deviates from the literal AT-W12-BANDS wording, and the stated reason is in the analyzer and in commit 11e7e784f.
  - The extension self-checks of the direct-form bases are reported, not gated. db_f16 shows 3.926e-7; the gating H3 q4_0 base shows 1.327e-7.
- **F3: q4_0 saturates 11× higher than q8_0.** Under pins, an ε = 1e-7 perturbation of attn_output-23 gives KLD 1.058e-3 against the clean q4_0 output (q8_0: 9.216e-5). So the q4_0 direct-form band is 1.441e-3, not ~2e-4.
- **F4: PR219 at ub16 is far from ub2048 at c2048 × 16, even with MoE pinned.**
  - f16: +6.745e-3 (SE 4.04e-4). q8_0: +6.702e-3.
  - Pre-pin MoE exact agreement is 0.740 at ub16, against 0.969 at ub512 (`e_*_u16.stats`, QPIN_MOE layer=-1).
  - C1TRACE kernels: f16 ub2048 runs `fa7` at nq 2048; f16 ub512 runs `dense_tile_vec_mma` at nq 512; ub16 runs `dense_tile_vec_mma` at nq 16. For q8_0, ub512 is bit-identical to ub2048, yet ub16 sits at +6.7e-3. Which op causes the ub16 shift (attention at nq 16 or a non-attention op) is UNPROVEN.
  - The Q11b band max(ub512 shift + 3 SE, B_512) = 2.03e-4 is therefore not calibrated for the ub16 arms. Their new-vs-PR219 comparison is same-ub, and its band stays **UNPROVEN** (a PR219 ub16 rerun or reorder arm would settle it).
- **F5:** with the master Vulkan dq8 GEMM defaults, Vulkan f16's own ub2048 reorder is 2.43e-4. Round 1 had 4.98e-4 on the PR219+#240 tree.

**Large files.** Every reference and base was deleted after its phase. That covers ref8t0, ref16t0, ref32x8, vref8t0, ref8L, ref2k, the 6 kbase_*.kld files, the 6 phase-S references, and the aborted first ref8t0 (22 GB). `find ~/dev/wt_lab/w12 ~/dev/wt_rewrite_logs/W12 -size +1G` finds nothing. The scratch worktree `~/dev/wt_lab/w12stack` was removed (`git worktree remove`). No process of mine is left running.

## W1: op contracts, ggml touchpoints, CPU reference

Everything ran on strix-002, with no other agent active. Every build, test, capture and harness run ran under `flock ~/gpu.lock`. Logs: `strix-002:~/dev/wt_rewrite_logs/W1/` (builds, test-backend-ops runs, oracle runs, scripts `builds_w1.sh`, `gpu_w1.sh`, `final_w1.sh`, commit messages `msgs/`) and `strix-002:~/dev/wt_lab/w1/` (captures log `cap.out`, harness campaign `campaign.out`, raw outputs `res/*.out`, binaries `bin/`). Branch HEAD before W1: `52c2c1a95`.

**Commits (branch `qsa-rewrite` in `~/dev/wt_rewrite`, author and committer `Gaetan Puleo <12990773+gaetan-puleo@users.noreply.github.com>`, not pushed; `git log --format='%h %an <%ae> | %ce | %s' ea5285ddf..HEAD`)**
- `d5feb2b93` ggml : GGML_OP_SPARSE_ATTN_PLAN and GGML_OP_SPARSE_ATTN, CPU reference
- `4e6328f61` tests : sparse attention op tests, C3 plan oracle, --fail-unsupported, determinism runs. Carries `Co-authored-by: Iggy Jackson <iggy@theiggy.com>`: its case list includes the q8_0 / mixed K/V shapes #219 added to test-backend-ops (`git show 21fd1b73d -- tests/test-backend-ops.cpp`, = b60447506). The QSA_*_MASKLESS cell formulas it reuses are not #219's (`git blame` of tests/test-backend-ops.cpp:11592-11620 at 52c2c1a95: Nathan Wilson, 341b56eb1c), and the comments say so.
- `943afa774` scripts : W1 harness and results (CPU sparse attention vs bands_ref.h, C3 plan oracle)
- Every commit ends with this session's `Claude-Session:` line. `git status --short` is empty after the commits.

**What changed**
- `ggml/include/ggml.h`:
  - `GGML_OP_SPARSE_ATTN_PLAN` and `GGML_OP_SPARSE_ATTN` appended right before `GGML_OP_COUNT` (PLAN-v3 §2).
  - Contract comments for both ops.
  - `GGML_SPARSE_ATTN_PLAN_MAGIC` (bytes "QSAP") and `_VERSION`.
  - `struct ggml_sparse_attn_plan_hdr` (64 bytes) and `struct ggml_sparse_attn_plan_layout`.
  - Functions: `ggml_sparse_attn_plan_get_layout`, `_nbytes`, `_scratch_nbytes`, `ggml_sparse_attn_plan`, `ggml_sparse_attn`.
- `ggml/src/ggml.c`:
  - Names and symbols added; the two static asserts go from 101 to 103.
  - Layout functions: section offsets are 256-byte aligned and the last section is not padded. Check of the plan's own arithmetic, executed against the built `libggml-base` (`~/dev/wt_lab/w1/nbytes.c`, output `res/nbytes.out`): `ggml_sparse_attn_plan_nbytes(2051, 127, 1, 1<<24, G)` gives G1 1564408 B and G4 1575680 B, the PLAN-v3 §2.1 values. At ub4096: scratch 33619968 B (§2.1 "33.6 MB") and G4 dst 50409728 B (§2.1 "≈ 50.4 MB").
  - Constructors with the §2 asserts: I32 sel with nb[0] = 4, nb[1..3] % 4 = 0, ne[2] = 1; 1 ≤ n_kv ≤ 2^24; G ∈ {1, 4}; GQA divides; D equal; K/V rows readable by to_float (nb[0] = type size); the bounds `k->ne[1], v->ne[1] ≥ 4·ceil(n_kv/4)`; rot ∈ {0, 64, 128, 256} dividing D.
  - op_params: plan = (n_kv, G); attend = (scale, rot_q, rot_o).
- `ggml/include/ggml-rpc.h`: `RPC_PROTO_MINOR_VERSION` 0 → 1, and the static_assert moved to 103 with a message naming the minor version.
- `ggml/src/ggml-backend.cpp`: both ops added to `ggml_op_alloc_size_may_expand`.
- `ggml/src/ggml-backend-meta.cpp`: `supports_op` returns false for both ops, and the split-state switch aborts with a named message for them.
- CPU:
  - `ggml/src/ggml-cpu/sparse-attn.cpp` (new): `ggml_compute_forward_sparse_attn_plan` (phase 1 per row, `ggml_barrier`, phase 2 per group) and `ggml_compute_forward_sparse_attn` (items (s, i, kv head); K/V row dequantized once per item for the r heads; template scores loop with the r heads in the lanes for r ∈ {1, 2, 4, 8, 12, 16}, otherwise a generic loop; FWHT in the ggml-cpu hinted-mul_mat order). Work-size helpers.
  - `ops.h`: declarations.
  - `ggml-cpu.c`: forward dispatch, n_tasks = n_threads, work sizes.
  - `ggml-cpu.cpp`: `supports_op` for SPARSE_ATTN requires to_float for K and V (or F32).
  - `ggml-cpu/CMakeLists.txt`: `sparse-attn.cpp` with `-ffp-contract=off`. `build-cpu/ggml/src/CMakeFiles/ggml-cpu.dir/flags.make` shows `sparse-attn.cpp.o_OPTIONS = -ffp-contract=off`.
- Tests:
  - `tests/test-backend-ops.cpp`:
    - `g_fail_unsupported` and `--fail-unsupported`.
    - `engagement_required` treats the opt-in as given and requires the census under that flag.
    - New `test_case` hooks: `n_determinism_runs`, `verify_reference`, `expected_unsupported`, `weights_end_at_buffer_end`. `eval()` runs determinism then the reference check.
    - The `run_tests` NOT_SUPPORTED → FAIL rule.
    - `sa_*` row generators and view helpers.
    - `test_sparse_attn_plan` (169 cases) and `test_sparse_attn` (301 cases), with the §14.2 (a) band table `sa_band_row` / `sa_band_for_backend` and `make_test_cases_sparse_attn`.
  - `tests/sparse-attn-ref.h` (new): the C3 oracle `sa_ref::plan` and O_CON `sa_ref::attend`.
  - `tests/test-sparse-attn-plan-oracle.cpp` (new, ctest).
  - `tests/CMakeLists.txt`: `-ffp-contract=off` for both test TUs. The oracle test sits under `NOT LLAMA_USE_SYSTEM_GGML` because it includes ggml-backend-impl.h.
- `scripts/qsa-validation/`: `harness/w1/{cap.sh, campaign_w1.sh, w1_ocon.cpp, vhash.cpp, analyze_w1.py, compare_c3.py}`, `summaries/w1_ocon.md`, `summaries/w1_plan_c3.md`, and a README section.

**Deviations from the plan text, with reasons**
- **CPU forwards in their own TU** (`ggml-cpu/sparse-attn.cpp`), not in `ops.cpp`. Building `ops.cpp` with `-ffp-contract=off` would change the FMA contraction of every other CPU op, the host-KV FA included. The plan's requirement is on "the TU" (PLAN-v3 §3).
- **RPC: the minor version is bumped**, as PLAN-v3 §2 and the task say. The W1 checklist line in PLAN-v3 §13 says `RPC_PROTO_PATCH_VERSION`, which contradicts §2. The ops are appended **before GGML_OP_COUNT** (§2 and the task), not "after GGML_OP_LIGHTNING_INDEXER" as the W1 "What" line says.
- **Meta backend.** The ggml side is done (supports_op false, named abort). The llama-side init error for a QSA model under `LLAMA_SPLIT_MODE_TENSOR` is PLAN-v3 §10 "Other init errors". It needs `src/` (not in W1's file list) and the QSA producer switch, so **it is not implemented in W1; W7 must add it**.
- **C3 captured rows no longer existed.** `~/dev/wt_lab/c3/cap/` holds only logs (`ls`). They were recaptured with C3's own `c3_capture` binary and arguments (`harness/w1/cap.sh`, PR219 build, read-only).
  - The recapture reproduces C3's per-layer statistics (valid_per_q, sorted_rows, union_asc) on **72/72 layers**, checked against `EV/c3/res/*_r1.out` with `compare_c3.py`.
  - The R1-A wiki d4k fixture (`c18/cap2`) was also gone and was recaptured. Its rows, Q and K hashes equal R1-A's on **84/84** keys, and its V f16 hashes on **12/12** (`summaries/w1_ocon.md`, `res/vhash.out`).
  - All captures (4.4 GB) were deleted after use. `du -sh ~/dev/wt_lab/w1` reports 2.6M, and `find … -size +100M` finds nothing.
- **Poison reach is asserted on the C3-oracle plan bytes**, not on a CPU-backend plan computed inside `initialize_tensors`. The two are byte-identical: 169/169 test-backend-ops plan cases, 1080 synthetic plans and 288 captured-row plans.
- **Not in W1:** the census list proc `ggml_backend_debug_dispatch_census_list`, whose contract is defined here in `sa_band_for_backend` (NULL-terminated key array) and is implemented by W2/W3/W6; the suite-level census accumulator (needs those procs); `test_sparse_attn_gated` (W3, gate fusion). On a non-CPU backend, a SPARSE_ATTN case with no census class FAILS ("band class unknown").

**Finding F-W1-1: `bands_ref.h` O_CON saturates bf16 instead of overflowing to ±inf.** PLAN-v3 §3 left this UNPROVEN.
- `w1_ocon overflow`: bf16 V values 7e4 / −1e5 (bf16 RNE 70144 / −99840). The CPU op gives ±inf at all 52 expected positions with the expected sign, which is the §2.2 contract. bands_ref.h O_CON gives **65504** (`build_deq` clamps bf16 to ±65504 before `rh`, `bands_ref.h:311`).
- q8_0 V with d·q = ±127000 gives ±inf in both.
- So the done-when "bf16 overflow → ±inf in both" holds for the CPU op but **not for bands_ref.h**. The plan already routes the overflow case to O_CON (the CPU op) only (§3, §14.1).
- Bit equality with bands_ref.h holds on every fixture used, none of which has a bf16 value past 65504.

**Builds** (`builds_w1.sh`, then `final_w1.sh` at the committed tree; every rc 0)
- HIP `build/` (rule flags, `-j 16`): rc 0, 85 `warning:` lines (E0: 85). Logs `hip_cfg.log`, `hip_build.log`, `final_hip_build.log`.
- Vulkan `build-vk/` (C11 flags): rc 0, 75 warnings (W0: 75).
- RPC `build-rpc/` (`-DGGML_RPC=ON`, CPU + RPC): rc 0. It produces `ggml-rpc-server` and `libggml-rpc.so`.
- CPU-only `build-cpu/`: rc 0.
- The only line matching "error" in any build log is the non-fatal `-- UI: download dist.tar.gz.sha256 … failed`, as in E0. No warning names a sparse-attn file.

**Tests** (per op, one op per run, `-j 1`, under flock; the full suite was never run)
- **test-backend-ops on the CPU backend** (`test -o <OP> -b CPU --fail-unsupported`; CPU against CPU-ref, error bound 0, plus `verify_reference` and 2 determinism runs):
  - `SPARSE_ATTN_PLAN` **169/169**, `SPARSE_ATTN` **301/301**.
  - Same result in the CPU-only, HIP, Vulkan and RPC builds (`final_{cpuonly,hipbuild_cpu,vkbuild_cpu}_*.log`, `rpcbuild_cpu_*.log`).
  - The attend cases are checked **bit for bit against the O_CON host reference** (`sa_ref::attend`). The plan cases are checked **byte for byte against the C3 oracle**.
- **Mutation check of those references** (temporary edit, restored, rebuilt; `mutant_*.log`):
  - P not rounded to f16 in the reference: 85 of 91 selected SPARSE_ATTN cases FAIL.
  - A stray mask bit for cell 5 in the oracle: 24 of 40 selected plan cases FAIL.
- `test-sparse-attn-plan-oracle` (ctest, `final_ctest_oracle.log` 1/1): 1080 synthetic plans IDENTICAL, plus the over-allocation check.
  - `OVER_ALLOC plain_buffer=3679200 over_allocating_buffer=4741792 outputs IDENTICAL -> OK`. A buffer type that over-allocates both ops goes through gallocr without the ggml-backend.cpp alloc-size assert.
- **Plan vs C3 oracle on the recaptured C3 rows** (6 files × 12 layers × G {4, 1} × {as captured, shuffled seed 7}): **288/288 IDENTICAL** (`oracle_c3_*.log`, `summaries/w1_plan_c3.md`).
- **CPU op vs `bands_ref.h` O_CON** (`w1_ocon`, `summaries/w1_ocon.md`, OVERALL PASS):
  - Fixtures:
    - wiki d4k, layers 0, 5, 11, stride 1, all 25 pairs of {f32, f16, bf16, q8_0, q4_0} (R1-A's `cpu_wiki4k` configuration);
    - wiki d4k, all 12 layers, stride 8, the 16 S_fast pairs (`cpu_wiki4k_s8`);
    - SYN 262k, stride 1, 25 pairs (`cpu_syn262k`);
    - 7 UNI fixtures (R2-BOP shapes, seed 1), 25 pairs each.
  - **3,350,040,576 values compared, 0 different.** G = 1 and G = 4 plans give identical output for every pair.
  - Contract vs f64, max over layers: every wiki / SYN pair is within NMSE(op, O_C18) ≤ 2.4e-9 / 7.3e-8 and NMSE(op, O_PLAN) ≤ 1.0e-7 / 3.1e-7.
    - Example, wiki q8_0/q8_0: 1.172e-9 and 5.151e-8.
    - Example, SYN bf16/bf16: 3.658e-8 and 7.409e-8.
  - Coarse CPU FA + src[5] cross-check within the four PLAN-v3 W1 bounds for all 66 wiki/SYN rows (example, SYN bf16/bf16 2.517e-6 ≤ 5.1e-6). UNI rows are reported, not gated.
- **GPU regression after the enum change** (`gpu_w1.out`). The four QSA ops equal the W0 reference counts:
  - ROCm0: QSA_PREFILL_MASKLESS 26/26; QSA_DECODE_MASKLESS 52/52 + 1 not supported; QSA_PREFILL 26/26; QSA_DECODE 92/92.
  - Vulkan0: 14/14 + 12; 34/34 + 19; 26/26; 92/92.
  - SPARSE_ATTN_PLAN / SPARSE_ATTN on ROCm0 and Vulkan0: 169 and 301 "not supported", as expected at W1 (no GPU kernel yet). With `--fail-unsupported` these become FAILs until W2 (HIP plan), W3–W5 (HIP attend) and W6 (Vulkan) land.

**Done-when (PLAN-v3 §13 W1)**
- Both ops pass on CPU for every §14 shape in the case lists: **PASS** (169 + 301).
- Plan = C3 oracle byte for byte on C3 captured rows: **PASS** (on the recapture, which is shown to be C3's rows by statistics).
- CPU op bitwise = bands_ref.h O_CON on wiki / SYN / UNI: **PASS**. bf16 overflow → ±inf holds for the CPU op only (F-W1-1).
- Contract vs f64 bands: **PASS**.
- CPU FA cross-check: **PASS**.
- CPU-only, RPC, HIP and Vulkan builds: **PASS**.
- Over-allocation without the assert: **PASS**.
- Open for W7: the llama-side init error under tensor split.

## W2: HIP plan kernels

Everything ran on strix-002, with no other agent active. Every build, test, capture, profile and harness run ran under `flock ~/gpu.lock`. Logs: `strix-002:~/dev/wt_rewrite_logs/W2/` (builds, test-backend-ops runs, TU checks, mutants, scripts `tu_check.sh`, `tests_w2.sh`, `final_w2.sh`, `mutants_w2.sh`, commit messages `msgs/`) and `strix-002:~/dev/wt_lab/w2/` (captures log `cap.out` + `logs/`, harness results `res/`, first-version results `res_v1/`, profiles `prof*`, binaries `bin/`, `w2_plan_summary.md`). Branch HEAD before W2: `943afa774`.

**Commits (branch `qsa-rewrite` in `~/dev/wt_rewrite`, author and committer `Gaetan Puleo <12990773+gaetan-puleo@users.noreply.github.com>`, not pushed; `git log --format='%h %an <%ae> | %ce | %s' 943afa774..HEAD`)**
- `c2da2cdc0` hip : SPARSE_ATTN_PLAN kernels, get_alloc_size and dispatch census
- `2a8637673` tests : SPARSE_ATTN_PLAN census engagement, LDS-overflow cases
- `82615f74f` scripts : W2 plan harness and results (HIP SPARSE_ATTN_PLAN on C3 rows)
- No `Co-authored-by: Iggy Jackson` trailer: the ported union builder (`qsa3_rows_kernel` / `qsa3_merge_kernel`, qsa-prefill.cu:28-240) has no #219 line (`git blame --line-porcelain -L 28,240 ea5285ddf -- ggml/src/ggml-cuda/qsa-prefill.cu`: Piotr Wilkin 148 lines, Gaetan Puleo 65); the commit body names that origin. Every commit ends with this session's `Claude-Session:` line. `git status --short` is empty after the commits.

**What changed**
- `ggml/src/ggml-cuda/sparse-attn-plan.cu` (new, HIP host and device code under `GGML_USE_HIP`; CUDA builds get `supported` = false, `get_alloc_size` = nbytes, compute = abort):
  - `sa_plan_init` (header with n_bad = 0, zero bytes up to off_counts, the two section gaps).
  - `sa_plan_rows<LDS>` (one 256-thread block per query row): validate, count n_bad (block sum, one `atomicAdd` into `hdr.n_bad`); a row whose valid cells are strictly ascending with every invalid entry after them gets only the marker `SA_PLAN_RAW` in `scratch[row][0]` and is read from sel by the merge; otherwise compaction in row order, and when not strictly ascending an in-place bitonic sort (flip form, no power-of-two padding) and a dedup into the scratch row. Global-memory variant for rows past the LDS budget: sort in place, mark repeats, replace them by the sentinel, sort again.
  - `sa_plan_merge<G, LDS>` (G = 4: 512 lanes, C3's P_asc 512; G = 1: 256 lanes): G-way merge by quad with C3's quantile lane split (quantiles of the longest row), count pass, block scan, write pass, then the ggml.h padding up to cap (`sa_plan_fill`: 16-byte quad and 8-byte mask stores). Row loads unrolled (4 per thread in rows, 2 per row x G rows in the merge copy).
  - `ggml_cuda_sparse_attn_plan_supported` (every HIP device; G in the instance table; int-indexing and u32 n_bad limits), `ggml_cuda_sparse_attn_plan_get_alloc_size` (max dst over the G table + `ggml_sparse_attn_plan_scratch_nbytes`; scratch placed right after `ggml_nbytes(dst)`), `ggml_cuda_sparse_attn_plan` (asserts dst owns the over-allocation through `ggml_backend_buffer_get_alloc_size`; LDS or global kernel per `smpb − 4 KiB`; `GGML_CUDA_SA_PLAN_GLOBAL=1` forces the global kernels, a test switch).
- `ggml/src/ggml-cuda/sparse-attn-census.h` (new): `GGML_CUDA_SA_PLAN_GROUPS(X)` = {1, 4} drives the template instantiation and dispatch switch of the merge, the get_alloc_size G range and the generated keys `sa_plan/G1`, `sa_plan/G4`; plus `qsa_expand`; the id enum and the NULL-terminated key array.
- `ggml/src/ggml-cuda/sparse-attn.cuh` (new): declarations (not in the W2 file list; needed by ggml-cuda.cu).
- `ggml/src/ggml-cuda/ggml-cuda.cu`: get_alloc_size branch, compute and supports_op cases for `GGML_OP_SPARSE_ATTN_PLAN`; HIP census storage (one relaxed atomic per key) and procs `ggml_backend_debug_dispatch_census_{reset,count,get,list,skip}` in `ggml_backend_cuda_reg_get_proc_address`; `GGML_CUDA_DISPATCH_CENSUS_LOG=1`; `qsa_expand` counted where `ggml_cuda_op_qsa_expand` runs; HIP graph capture records the census ids into the graph and every replay counts them again.
- `ggml/src/ggml-cuda/common.cuh`: `ggml_cuda_graph::census_ids` (not in the W2 file list; needed for the replay count).
- `tests/test-backend-ops.cpp`: census engagement in `test_sparse_attn_plan` (exactly one `sa_plan/G<g>`, zero of the other group; CPU exempt); 16 cases with n_sel 9000 (reach the global rows kernel without forcing).
- `scripts/qsa-validation/harness/w2/{cap.sh, rowstat.cpp, w2_plan.cpp, campaign_w2.sh, analyze_w2.py, mutants_w2.sh}`, `summaries/w2_plan_c3.md`, README section.

**Deviations from the plan text, with reasons**
- **Census counts launches, not captures.** PLAN-v3 §6.2 says HIP counts are graph captures and that presence assertions are valid with graphs on. The per-case presence check failed on 59 cases with graphs on (`hip_SPARSE_ATTN_PLAN.log`; 73 in the rerun `dbg_graphs.csv`: "census sa_plan/G1=0 sa_plan/G4=0 (want 1 and 0)"; a case of the same shape and addresses as the previous one replays its graph and dispatches nothing on the host), 0 with `GGML_CUDA_DISABLE_GRAPHS=1` (`dbg_nographs.csv`). Fix: the capture records the ids, a replay counts them. In-model check: llama-bench pp4096 + tg8, `-r 2`, `qsa_expand=267` with graphs on and off (`final_bench_census.log`, `final_bench_census_nographs.log`).
- **The W2 C3 harness is a new harness on the op** (`w2_plan.cpp`): wall time of a graph of 8 plan nodes / 8, so launches and gaps are included; C3's `c3_plan` timed raw kernels with hipEvents. Same rows, same statistic (per-layer median, mean over 12 layers). C3's `c3_plan` was re-run in the same session on the same rows as the reference.
- **-Werror and spill check per TU.** The two changed HIP TUs were compiled with the build's own command + `-Werror -Wno-tautological-compare -Rpass-analysis=kernel-resource-usage --save-temps` (`tu_check.sh`), and their code objects parsed with `parse_assembly` of `scripts/hip/gcn-cdna-vgpr-check.py`. Not a full `-Werror` tree build, and not the full checker (it requires the CI image's exact compiler identity; baseline regeneration is W13).
- **Recapture.** A first capture with the pos0 list `131072` alone gave d131k rows that differ from C3's (mean G4 union up to 4.6 entries per group off, e.g. text layer 10: 859.4 vs 854.8). The mechanism is UNPROVEN (candidate: the hooked ubatch at pos0 32768 changes graph splits and numerics). The capture was redone with C3's list `32768,131072`, which reproduces C3 on 36/36 layers; only that one is used.
- **Not in W2:** the suite-level census accumulator (§14.1) still waits for attend keys (W3). The list proc already returns `qsa_expand`, which `-o SPARSE_ATTN` never reaches; the W3 accumulator must compare the `sa_*` keys only, or run a qsa_expand case.

**Finding F-W2-1 (code reading, not executed): PR #219's merge can emit a quad twice.** `qsa3_rows_kernel` passes a non-decreasing row with a repeated cell unsorted (its check is `ent[j] > ent[j+1]`, qsa-prefill.cu), and the merge's complete-quad shortcut then advances 4 entries on 4,5,6,7,7 and meets 7 again in the same quad, so quad 1 is emitted twice. The new kernels dedup before the merge; the dup cases of §14.1 pass.

**Builds** (every rc 0)
- HIP `build/` (rule flags, reconfigured for the new file, `-j 16`): `hip_build.log`, `hip_build2.log` (full rebuild after the common.cuh change: 13 warnings, all `ld.lld: warning: ... loop not unrolled`, also 13 in W1's `hip_build.log`), `final_hip_build.log` and `commit_hip_build.log` 0 warnings.
- TU checks: `sparse-attn-plan.cu` and `ggml-cuda.cu` 0 warnings, 0 errors under `-Werror`. Parser over the code objects: 7 `sa_plan` kernels, 0 spilling, 0 scratch bytes, VGPRs init 6, rows 56 / 11 (LDS / global), merge G1 13 / 13, G4 32 / 24 (`tu_final_sparse-attn-plan.log`, `final_w2.sh` output).
- Vulkan `build-vk/` and CPU-only `build-cpu/` incremental: rc 0, 8 warnings each, all in `tests/test-backend-ops.cpp` at lines the W2 diff does not touch (deprecated enum-enum, hidden overload, deprecated set_prec).

**Tests** (per op, one op per run, `-j 1`, under flock; the full suite was never run)
- `-o SPARSE_ATTN_PLAN -b ROCm0 --fail-unsupported`: **185/185** with HIP graphs on, with `GGML_CUDA_DISABLE_GRAPHS=1`, and with `GGML_CUDA_SA_PLAN_GLOBAL=1` (`final_rocm_plan_{graphs,nographs,global}.log`), again 185/185 at the committed tree (`commit_rocm_plan.log`). Each case: whole-dst memcmp vs CPU (err() = memcmp), C3 oracle (`verify_reference`), 2 determinism runs, census engagement. Shapes: the W1 list (§14.1: n_q 1–4096, n_sel 1/257/2051/4096, n_kv 4–1048576, every row pattern, G 1/4, 1–2 streams, plain/interleaved/strided rows) + 16 n_sel 9000 cases.
- CPU `--fail-unsupported` 185/185 in the HIP, Vulkan and CPU-only builds (`final_{hipbuild,vkbuild,cpuonly}_cpu_plan.log`). `ctest -R test-sparse-attn-plan-oracle` 1/1.
- Mutants of sparse-attn-plan.cu on ROCm0 (`mutant_*.log`, restored, md5 d7e34e42… before and after; the committed file, 9de4dbbe…, differs from it in one source comment only, and its build passes 185/185 in `commit_rocm_plan.log`): padding quad 0 → 149 of 185 FAIL; no dedup → 87 FAIL; n_bad not counted → 94 FAIL; restored → 185/185.
- Regression on ROCm0 after the ggml-cuda.cu / common.cuh changes, equal to W0/W1: QSA_PREFILL_MASKLESS 26/26; QSA_DECODE_MASKLESS 52/52 + 1 not supported; QSA_PREFILL 26/26; QSA_DECODE 92/92. `-o SPARSE_ATTN -b ROCm0`: 301 not supported (attend lands in W3–W5).
- **C3 harness** (`w2_plan`, `scripts/qsa-validation/summaries/w2_plan_c3.md`, `res/*.out`): rows recaptured with C3's `c3_capture` and arguments, equal to C3 on 36/36 d131k layers (stats) with 144/144 oracle verdicts IDENTICAL (`res/oracle_*.log`). G4 mean [worst layer] ms/layer: text f16 **0.719** [0.789], text q8_0 0.720 [0.792], random f16 **0.831** [1.026] (r1; r2 and graphs-off within 0.002); shuffled (sort path, reported only) 1.795 / 1.908. G1 0.603. Every (layer, G, run) bit-exact vs the CPU op and the C3 oracle (W2PLAN_DONE mismatches=0 in all 11 runs). Same-run `c3_plan` P_asc_l512 (no padding writes): text 0.732, random 0.883; PR T0 1.189 / 1.491. Kernel split (rocprofv3, average µs): merge 458 / 571, rows 249 / 245, init 1 (text / random).
- First version (one load in flight per lane, scalar padding stores) measured G4 1.042 / 1.188 ms and failed the gate (`res_v1/`); the final code fixed it with unrolled loads and vector padding stores, bit-exactness unchanged.

**Done-when (PLAN-v3 §13 W2)**
- Whole-dst bit-exact vs CPU on every §14.1 plan case (`--fail-unsupported`): **PASS** (185/185, three kernel modes).
- C3 harness on recaptured d131k rows ≤ 1.0 ms/layer mean, text and random, bit-exact: **PASS** (0.719 / 0.831).
- `-Werror` and 0 spills for the new TU: **PASS** (TU-level, see deviations).

**Cleanup:** captures deleted (`cap/` 2.3 GB, `cap_partial/` 1.2 GB); `du -sh ~/dev/wt_lab/w2` = 5.2M; no file > 100 MB under `~/dev/wt_lab/w2` or `~/dev/wt_rewrite_logs/W2`. The TU save-temps directories (≤ 20 MB each) are kept as the spill evidence.

## W3: HIP G4 attend, fast types, gate fusion

Everything ran on strix-002, with no other agent active. Every build, test, harness and profile run ran under `flock ~/gpu.lock`. Logs: `strix-002:~/dev/wt_rewrite_logs/W3/` (builds `build_*.log`, `final_*.log`, TU checks `tu_final*`, test runs `f1_*`..`f3_*`, `final_*`, report runs `t*_g4.log`, scripts, commit messages `msg_c*.txt`) and `strix-002:~/dev/wt_lab/w3/` (`c18/` kernel-time harness and `res/`, `res_v1/`; `prband/` PR-verbatim band runs `res/`, `res2/`, `summary*.md`; `attrib/` attribution lab; `exp/` register experiments). Branch HEAD before W3: `82615f74f`.

**Commits (branch `qsa-rewrite` in `~/dev/wt_rewrite`, author and committer `Gaetan Puleo <12990773+gaetan-puleo@users.noreply.github.com>`, not pushed; `git log --format='%h %an <%ae> | %ce | %s' 82615f74f..HEAD`)**
- `ed8d3d70e` hip : sparse-attn K/V conversion helpers and loaders (q8_0 helper from #219) — `Co-authored-by: Iggy Jackson <iggy@theiggy.com>` (sa_q8x2_to_h2 is #219's qsa3_q8x2_to_h2 / qd_q8x2_to_h2).
- `c0ef4a848` hip : SPARSE_ATTN G4 attend for S_fast pairs, V pack, gate fusion — `Co-authored-by: Iggy Jackson <iggy@theiggy.com>` (kernel body ported from qsa3_attn_kernel, of which 57 lines are Iggy's in `git blame --line-porcelain -L 241,601 HEAD -- ggml/src/ggml-cuda/qsa-prefill.cu`: Piotr Wilkin 211, Gaetan Puleo 93, Iggy Jackson 57; plus the #219 q8 helper and the n_kv >= 64·n_q direct-V switch).
- `05a1b6cd6` tests : SPARSE_ATTN G4 cases, gate-fusion test, suite census coverage.
- `d7927391d` scripts : W3 harnesses and results (HIP G4 attend: bands, kernel time, resources).
- Every commit ends with this session's `Claude-Session:` line; `git status --short` is empty after the commits. (Earlier local versions of the last three, f2a448a60 / c3baa3607 / bb547d821 / a1b0aa2dd, were rewritten with `git reset --soft` / `--amend` before anything was pushed; a1b0aa2dd and 05a1b6cd6 have the same tree, the amend changed the message only.)

**What changed**
- `ggml/src/ggml-cuda/sparse-attn-q8.cuh` (new): `sa_q8x2_to_h2` (#219 helper), `sa_q4x2_to_h2` (C18 trick), `sa_bf2_to_h2` / `sa_bf1_to_h` (one RNE rounding, ±inf on overflow — C18's helper saturated to ±65504, the op contract does not), `sa_f2h`, WMMA fragment typedefs.
- `ggml/src/ggml-cuda/sparse-attn-loaders.cuh` (new): type ids (census order f16, bf16, q8_0, q4_0), piece loaders/converters for the split K loader (C18 "X"), gather loaders for the split-gather direct V, `sa_mask_cells` (V select), traits `sa_k_loader_of` and `sa_vpack_trait` (default CAPPED for every type, OWNER decision), the census-token → template-id macros.
- `ggml/src/ggml-cuda/sparse-attn-attend-g4.cuh` (new): `sa_attend_g4<KT, VT, VL, EPI, GQA>` (the PR qsa3_attn_kernel body as C18's c18_attn_kernel, generalised to GQA 4/8/12 = 1/2/3 row tiles, streams on grid.z, the ggml.h plan layout, q·scale in f32, EPI plain / gated with f16/bf16 aux store, select for masked cells) and `sa_attend_fixup<KT, VT, EPI>` (exact per-cell recomputation of flagged blocks, CPU-op arithmetic).
- `ggml/src/ggml-cuda/sparse-attn-attend-g4-{f16,bf16,q8_0,q4_0}.cu` (new): 48 attend + 8 fixup instances per K type (tables generated from the census X-macros).
- `ggml/src/ggml-cuda/sparse-attn-pack.cu` (new): `sa_pack_v<T>` (PR qsa_pack_v indexing per stream and kv head, one block per quad; non-finite halves written as 0, per-quad flag, any-flag) and `ggml_cuda_sparse_attn_pack_v` (4-byte memset of the any-flag, then the pack; census `sa_pack/<VT>`).
- `ggml/src/ggml-cuda/sparse-attn-attend-g4.cu` (new): `sa_select` (shape-only: G, types, GQA, V loader, workspace offsets/bytes and the reserved max), `ggml_cuda_sparse_attn_supported` (RDNA3.5 only, S_fast², D 256, GQA {4,8,12}, rot 0, G4 plan, strides %16, grid limits), `ggml_cuda_sparse_attn_get_alloc_size` (round_up(nbytes,256) + max(packed layout at min(4·ceil(n_kv/4), 64·n_q) cells, direct wave-flag layout at n_q)), `ggml_cuda_sparse_attn` / `_gated` (pack, attend, fixup, census), `ggml_cuda_sparse_attn_gate_ok`, `_ws_range`, `_census_skip`, `GGML_CUDA_SA_FORCE=vpacked|vdirect`. CUDA (non-HIP) builds: supported = false, compute aborts.
- `ggml/src/ggml-cuda/sparse-attn-census.h`: X-macro instance tables and keys `sa_attend/G4/<KT>/<VT>/<packed|direct>/<plain|gated>/gqa<n>` (192), `sa_fixup/G4/<KT>/<VT>/<EPI>` (32), `sa_pack/<VT>` (4); N = 231 ≤ 256 (the uint8 graph-capture ids).
- `ggml/src/ggml-cuda/sparse-attn.cuh`: declarations.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: get_alloc_size / compute / supports_op cases for `GGML_OP_SPARSE_ATTN`; the gate-fusion matcher branch `SPARSE_ATTN → [views] → CONT(gate) → SIGMOID → MUL` (FA branch kept until W13), gate rows `R = (s·n_q + i)·n_head + h` at `gate->nb[1]` (qwen4exp's strided Qcur_full view), overlap check over q/k/v/plan/gate and the node's own workspace; the census skip proc consults the forcing switch; the HIP graph node properties include the source types.
- `ggml/src/ggml-cuda/common.cuh`: `node_properties::node_src_type`.
- `tests/test-backend-ops.cpp`: 192 G4 cases (16 pairs × packed/direct × 1/2 streams × GQA 4/8/12, n_q 129, n_kv 4096 / 16384, n_sel 257); `test_sparse_attn_gated` (96 F32 cases = every gated instance, BF16 / BF16-only / F16-only (dense-R2 Q8_0 weight) reader variants, 2 streams, the overlap-reject variant); the suite census accumulator and coverage check (`test_case::census_families`, honouring the census skip proc); G4 band rows 7.80e-9 / 2.41e-9 and the §14.2 (c) type spread ×1.12; `GGML_SA_TEST_REPORT` and `GGML_SA_TEST_DUMP` hooks.
- `tests/sparse-attn-ref.h`: `sa_ref::f64_maxrel` (O_C18 / O_PLAN f64 oracles for the report).
- `scripts/qsa-validation/harness/w3/*`, `summaries/w3_{prband,kernel_time,resources_tests}.md`, README section.

**Deviations from the plan text, with reasons**
- **K loader split for f16 too** (D8 says direct for f16). C18 measured them equal (Kf16X-Vf16P vs Kf16D-Vf16P: 12.43 / 12.60 ms wiki d4k, 16.32 / 16.33 d131k, MEASUREMENTS.md C18 K table); with direct f16 K the GQA-12 instances with a direct quantized V and the gated epilogue need scratch (single-instance compiles in `~/dev/wt_lab/w3/exp`: Kf16-D Vq8_0-direct gated GQA12 ScratchSize 56 B, Kf16 Vq4_0-direct plain 20 B; with split K and the final flag design: 0).
- **Direct V for f16/bf16 = split-gather (X)**, not `Vf16S` (D8). C18: V-S with split K spills 36 B (C18 Resources); X is faster than the PR's VD at 262k / 1M (57.59 / 65.28 vs 60.67 / 65.91 ms). Final kernel: 0 spills.
- **Contract item 4 under the shared PV WMMA.** A non-finite V half of a cell attended by only some of the 4 queries of a group gives 0·inf = NaN to the others (found by the G4 overflow cases). An in-loop exact path did not fit 192 VGPRs at GQA 12 (lab: up to 979 spilled VGPRs; a WMMA class-count variant 8–128 B scratch). Final design: packed V — `sa_pack_v` writes non-finite halves as 0 and flags the quad; direct V — an in-loop check (one packed FMA per dword) removes them and flags the block; `sa_attend_fixup` recomputes every block that touches a flagged quad / was flagged with the CPU op's arithmetic. The in-loop check cost ~4 % of the packed kernel (attribution lab `summaries/w3_kernel_time.md`: 12.90 → 12.41 ms d4k, 17.12 → 16.52 ms d131k with the check compiled out; the fixup launch costs nothing measurable), hence its move into the pack for packed V.
- **Census keys** carry a `/gqa<n>` level (each compiled instance has its own key; the §6.2 key is its prefix and still counts) and new `sa_fixup/...` keys.
- **G4 band rows changed.** The R2-BOP G4 rows (2.04e-9 / 1.21e-9) were measured on PR cell-formula rows only; the PR-verbatim kernel itself exceeds 1.21e-9 on the W3 case list's mix / empty / poison rows (PR-V0 vs O_CON 1.2866e-9 on the 129×4096×257 mix dump; p100 3.759e-9 run 1, 3.899e-9 run 2, `summaries/w3_prband.md`). New rows = 2 × p100 of the PR-verbatim kernel over both runs: n_sel ≤ 257 7.80e-9, n_sel > 257 2.41e-9 (p100 1.206e-9). Detection margin stays large (R2-BOP min k at G4 1032 / 1757 at p100 ~1.0e-9 / 6.0e-10, i.e. planted defects ≥ ~1.05e-6 NMSE, ~135× the new 257 band; arithmetic). The new kernel's NMSE is 0.997–1.001 × the PR kernel's on every dump the fixup does not recompute, and its output differs from PR-V0 in ~1.2–1.4 % of words at NMSE ≤ 1.7e-12 (TBO_BITWISE; the f32 q·scale prologue vs the PR's f16(q)·f16(scale) is the only arithmetic difference by code reading — UNPROVEN as the sole cause).
- **HIP graph replay bug fixed** (not in the W3 file list: `common.cuh`): graphs whose only change was an F16 ↔ BF16 source at the same address replayed the stale kernel (W3 bf16-after-f16 cases NMSE ~0.5, census keys missing: `t2_g4.log`); node properties now include source types.
- **`test_sparse_attn_gated`**: rot 64/256 left to W10 (op unsupported until then); the scheduler-allocated variants (bf16only, f16only, reject) skip the memcmp determinism re-run (it runs through `ggml_backend_sched_graph_compute`, a different computation: `nondeterministic … MUL_MAT at byte 0` in the first run) and the CPU O_CON check (their buffers are reused). Determinism is checked on every directly allocated case (all 96 F32 gated cases and all plain cases).
- **n_stream 2 transfer rule** (bitwise per-stream equality) not implemented separately: the (G4, n_sel) class has a band, which the 2-stream cases meet.
- **Not in W3**: G1 (W4), generic/portable (W5), rot (W10), Vulkan (W6). supports_op is false for those shapes (`attn_all`: 219 not supported, 382/382 pass).

**Findings**
- F-W3-1: the 64·n_q direct-V switch is early for q8_0/q4_0 at small n_q: direct is 1.19× / 1.19× packed at n_kv = 64·n_q for n_q 512, 1.14× / 1.13× for n_q 1024; it wins from 256·n_q (n_q 512), 128·n_q (n_q 1024), 64·n_q (n_q 4096). For f16/bf16 under CAPPED, direct is 1.35× packed at n_q 4096 / n_kv 262144 and 1.23–1.26× at n_q 1024 / 131072 (`summaries/w3_kernel_time.md`). Recorded for AT-O; trait unchanged.
- F-W3-2: the HIP graph cache keyed nodes without source types (above).

**Builds** (`final_w3.sh`, at the committed tree)
- HIP `build/` (rule flags, `-j 16`): rc 0, 8 warnings, all in tests/test-backend-ops.cpp at lines W3 does not touch (as W2). Vulkan `build-vk/`: rc 0, 8 warnings (same). CPU-only `build-cpu/`: rc 0, 8 warnings (same).
- TU checks (`tu_all.sh final`, `-Werror -Wno-tautological-compare` + resource remarks, `gcn-cdna-vgpr-check.py parse_assembly`): 7 TUs 0 warnings 0 errors; **228 sparse-attn kernels (192 attend, 32 fixup, 4 pack), 0 spilling, 0 scratch**, max 192 VGPRs (`summaries/w3_resources_tests.md`). Not the full repo checker / baseline (W13).

**Tests** (one op per run, `-j 1`, under flock; the full suite was never run; `final_w3.out`)
- `-o SPARSE_ATTN -b ROCm0 -p group=4 --fail-unsupported`: **382/382**, graphs on and off; `GGML_CUDA_SA_FORCE=vdirect`: 382/382 and `vpacked`: 382/382 (every case in both V regimes). Census coverage: **228/228 compiled keys launched** (forced runs: 128/128 and 132/132 with 100 / 96 keys excluded by the forcing switch).
- `-o SPARSE_ATTN -b ROCm0` (all cases): 382/382, 219 not supported (G1, S_generic, rot ≠ 0: W4/W5/W10).
- Band reach: max NMSE/band 0.745 over 274 plain G4 cases, 0.399 over the gated ones (`t5_g4.log` SA_REPORT). f64 errors (reported, not gated) per pair: max|err|/max|ref| vs O_C18 2.19e-4–2.82e-4, vs O_PLAN 2.37e-4–3.59e-4 over the 16 S_fast pairs.
- Gate fusion: every EPI=gated instance launched (census), engagement gated = 1 / plain = 0 per gated case, the reject case gated = 0 / plain = 1, BF16 / BF16-only / F16-only reader variants pass (wo GEMM NMSE ~1.1e-7 ≤ 5e-4).
- CPU `-o SPARSE_ATTN --fail-unsupported`: 601/601 (HIP, Vulkan and CPU-only builds). SPARSE_ATTN_PLAN: ROCm0 185/185 (census 2/2), CPU 185/185. `ctest test-sparse-attn-plan-oracle` 1/1.
- Regression, equal to the W0 reference counts: ROCm0 QSA_PREFILL_MASKLESS 26/26, QSA_DECODE_MASKLESS 52/52 + 1 not supported, QSA_PREFILL 26/26, QSA_DECODE 92/92; Vulkan0 14/14 + 12, 34/34 + 19, 26/26, 92/92. Vulkan SPARSE_ATTN G4 / PLAN: all on the expected-unsupported list (382 / 185 not supported, rc 0).

**Kernel time on C18 rows (done-when; `summaries/w3_kernel_time.md`, mean of r1/r2, 12 layers, SYN data; best row / same-run PR-verbatim route PR-V0-f16 + PACKV-f16; gate = C18 target × 1.04)**
- wiki d4k (PR route 12.69 ms): f16 0.955 (≤ 1.040), bf16 0.963 (≤ 1.059), q8_0 0.995 (≤ 1.074), q4_0 0.977 (≤ 1.058). Against C18's own Kf16D-Vf16P + pack (12.16 ms): 0.997 / 1.005 / 1.039 / 1.020.
- wiki d131k (PR route 18.77 ms): f16 0.956 (≤ 1.040), bf16 0.960 (≤ 1.089), q8_0 0.936 (≤ 1.058), q4_0 0.913 (≤ 1.025). Against C18 route (18.12): 0.990 / 0.994 / 0.969 / 0.946.
- **PASS** on every row; best row is X+P for every type. Direct vs packed+pack at syn 262k / 1M and the crossover at n_q 512 / 1024 / 4096 for all four types: table in the summary (F-W3-1).
- Correctness inside the harness: new f16 vs PR-V0-f16 NMSE ≤ 2.1e-12 on every fixture; packed vs direct of one type bitwise equal.

**Done-when (PLAN-v3 §13 W3)**
- All 16 pairs × V regimes × 1/2 streams × GQA {4,8,12} within the band of the census class (type spread where the pair has no row): **PASS** (382/382, max ratio 0.745), with the G4 band rows re-measured on the PR-verbatim kernel (deviation above).
- Gated test reaches every EPI=gated instance and the fusion reject: **PASS**.
- Suite census covers every compiled key: **PASS** (228/228).
- 0 spills: **PASS** (228 kernels, TU-level checker parse).
- C18 kernel-time rows: **PASS** (above).

**Cleanup**: TBO dumps deleted (`~/dev/wt_lab/w3/dump` 3.0 GB, `dump2` 2.8 GB); intermediate TU save-temps deleted (the `tu_final_*` ones kept as spill evidence, `~/dev/wt_rewrite_logs/W3` 221 MB); `find ~/dev/wt_lab/w3 ~/dev/wt_rewrite_logs/W3 -size +100M` finds nothing. A one-off mistake: the first PR-band run wrote its 40 result files into `~/dev/wt_lab/bands/op/res/` (R2-BOP's lab dir) because the script changed directory before resolving its output path; they were moved to `~/dev/wt_lab/w3/prband/res/` (`ls … | grep -c sa_` = 0 afterwards) and the script now resolves the path first. No process of mine is left running.

## W4: HIP G1 attend and the G policy

Everything ran on strix-002, and no other agent was active. Every build, test, harness run and profile ran under `flock ~/gpu.lock`. The full test-backend-ops suite was never run, and every run used one op with `-j 1`.

Where things are:
- Logs: `strix-002:~/dev/wt_rewrite_logs/W4/`. This holds the builds `hip_build_b*.log` and `final_*_build.log`, the TU checks `tu_c1*` and `tu_final*`, the test runs `t1_g1`, `t2_g1dump`, `t3_all` and `final_*.log`, the scripts `chain1.sh`, `chain2.sh` and `final_w4.sh`, and the commit messages in `msgs/`.
- Results: `strix-002:~/dev/wt_lab/w4/`:
  - `prband/`: the PR-verbatim band runs, in `uni/` and `tbo/`;
  - `bands/`: the W4 harness `w4_dec`, plus `syn/`, `c18/`, `ata/`, `ataplan/`, `ata_summary.md`;
  - `reserve/`: AT-C5, plus `w4_procs`.

Branch HEAD before W4: `d7927391d`.

**Commits** (branch `qsa-rewrite` in `~/dev/wt_rewrite`). Author and committer are `Gaetan Puleo <12990773+gaetan-puleo@users.noreply.github.com>`. Nothing was pushed. Listed with `git log --format='%h %an <%ae> | %s' d7927391d..HEAD`:
- `bb5490b5e` hip : SPARSE_ATTN G1 attend, deterministic split merge, HIP group / prefer_dense policies.
  - Carries `Co-authored-by: Iggy Jackson <iggy@theiggy.com>`.
  - The kernel body ports `qsa_decode_wmma_partial`. `git blame --line-porcelain` of qsa-decode-wmma.cuh at the base gives Piotr Wilkin 114 lines, Iggy Jackson 44 and delgon 3.
  - Its q8_0 K loader is #219's decode q8 K loader, taken as is: qsa-decode-wmma.cuh:63-74 (`git blame -L 60,80` shows 509e8ee3b1, the cherry-pick of 68da4191d). The merge keeps #219's qsa_decode_merge arithmetic.
- `7f7083f41` tests : SPARSE_ATTN G1 cases, band rows for split classes 6-16 and > 33.
  - Carries `Co-authored-by: Iggy Jackson <iggy@theiggy.com>`, because it adds the G1 variants of the ported #219 shapes from the b60447506 case list.
- `330ece83f` hip : AT-A G1 / G4 crossover rule in the HIP group policy.
- `315fe35ac` scripts : W4 harnesses and results (HIP G1 attend: bands, C18 gates, AT-A, AT-C5).
- Every commit ends with this session's `Claude-Session:` line. `git status --short` is empty after the commits.

**What changed**
- `ggml/src/ggml-cuda/sparse-attn-attend-g1.cuh` (new): `sa_attend_g1<KT, VT, GQA>`.
  - Ported body: PR #219's `qsa_decode_wmma_partial`, as C18's `c18_dec_kernel` took it over.
  - Loaders (C18 decision 2, D9):
    - V: LDS-staged and transposed in a wave-private slot, for every type;
    - K: direct for f16 and q8_0 (q8_0 = #219's decode q8 K loader), split with `v_permlanex16` for q4_0 and bf16.
  - Cells come from the G = 1 plan (ascending quads plus mask bits). Each lane forms its own cell from the chunk's uniform quads and masks, as the PR's `ir[start + r]` did. Masked and padding cells are never loaded and score −inf.
  - q·scale is computed in f32 and then rounded to f16 (the op contract).
  - Grid: x = (stream, query, kv head), y = split.
  - With one split the kernel writes dst. Otherwise it writes the PR partial rows (256 + max + normaliser).
  - Host-side helpers:
    - `sa_g1_splits(n_q, n_sel, n_kv)`: the shape-only split count. It is #219's decode rule (qsa-decode.cu:134-145, step 64), clamped to the plan's chunks (`cap / 4`) and to 65535;
    - `sa_g1_split_class`;
    - `sa_g1_partial_bytes`;
    - `sa_g1_partial_bytes_max`: the max over n_q' ≤ n_q, stopping at n_q' = 8·ceil(n_sel/64), after which splits = 1.
- `ggml/src/ggml-cuda/sparse-attn-attend-g1.cu` (new):
  - the 48 instances (S_fast² × GQA {4, 8, 12});
  - `ggml_cuda_sa_g1_launch` and its census count;
  - the HIP policies:
    - `ggml_cuda_sparse_attn_group`: past the shortcut, the AT-A rule `ggml_cuda_sa_g1_beats_g4`. In the shortcut, its own decision, which stays at the default `n_q < 128 ? 1 : 4` until AT-W7PERF. `GGML_CUDA_SA_FORCE=g1|g4` is read at every call;
    - `ggml_cuda_sparse_attn_prefer_dense`: a table per pair × the D4 n_q bands; the default is dense for f16/f16 only.
- `ggml/src/ggml-cuda/sparse-attn-merge.cu` (new): `sa_merge_splits`, with one block per output row.
  - It merges the splits in a fixed order, so it is deterministic.
  - It uses #219's merge arithmetic; splits with a zero normaliser are skipped by select.
  - The rot_o epilogue comes in W10; supports_op rejects rot ≠ 0 until then.
- `sparse-attn-attend-g4.cu` (`sa_select`, supports_op, get_alloc_size, dispatch):
  - G1 path: splits, and the partial workspace at `round_up(nbytes, 256)`.
  - `ws_max` is now the max of three things: G4 packed, G4 flags, and the G1 partials over n_q' ≤ n_q.
  - G1 grid limit: `n_stream · n_q · n_head_kv ≤ INT_MAX`.
  - Gate fusion stays G4-only.
- `sparse-attn-census.h`:
  - G1 keys `sa_attend/G1/<KT>/<VT>/splits<1|2-5|6-16|17-33|34+>/gqa<n>`. This is the §6.2 key with the class lower bound as `<n>`, extended by GQA as the G4 keys are. There are 240 of them;
  - one `sa_merge` key;
  - 469 keys in total.
- `ggml-cuda.cu`, `common.cuh`:
  - census ids recorded into HIP graphs go from `uint8_t` to `uint16_t`, because 469 > 256;
  - procs `ggml_backend_sparse_attn_group` and `ggml_backend_sparse_attn_prefer_dense`.
- `ggml/include/ggml-backend.h`: typedefs `ggml_backend_sparse_attn_group_t` and `ggml_backend_sparse_attn_prefer_dense_t`.
- `sparse-attn.cuh`: declarations.
- `tests/test-backend-ops.cpp`, 339 new SPARSE_ATTN cases (940 in all, up from 601):
  - the W4 n_q set × 1/2 streams × S_fast with G1;
  - every G1 census key: S_fast² × GQA × the 5 classes, at n_q 1;
  - complete-quad rows;
  - the G1 variants of the ported #219 shapes that PR219 ran through qsa_decode.
- Other test changes:
  - G1 band rows for 6-16 and > 33;
  - `sa_merge` in the census families;
  - suite census coverage is gated only without `-p`;
  - `GGML_SA_TEST_DUMP_MAX_KV`.
- `scripts/qsa-validation/harness/w4/*`, `summaries/w4_{bands,c18gate,ata,atc5,resources_tests}.md`, and a README section.

**Deviations from the plan text, with reasons**
- **Census key shape.** The plan names `sa_attend/G1/<KT>/<VT>/splits<n>`. The shipped key is `.../splits<class>/gqa<g>`, where the class is the §14.2 band class.
  - One key per exact split count would be unbounded, so it could not be enumerated by `census_list`.
  - The test's band lookup parses the class's lower bound with `atoi` (6 for "6-16", 34 for "34+").
- **Split formula inputs.** The plan's `splits(n_q, n_sel, n_head, CUs)` is the PR rule. The shipped function takes (n_q, n_sel, n_kv):
  - the PR rule uses neither n_head nor the CU count;
  - n_kv is added for the clamp to the plan's chunk count, so no split is empty by shape.
  - Keeping the PR rule makes every G1 grid the one C18 and R2-BOP measured.
- **Per-split range.** Split z takes chunks `[z·len, (z+1)·len)` with `len = ceil(C/S)`, in 32-bit.
  - A first version used `[z·C/S, (z+1)·C/S)` with 64-bit divisions and a register-indexed key array. It was 1.6–3.5 % slower than C18 at tg on wiki d131k (q8_0 20.7 vs 20.0 µs, `res/tsmoke1.out`) and would have failed the ×1.014 gate.
  - The final kernel is 8–15 % faster (gates below).
- **AT-C5 at op level only.** In W4 the model graph does not build SPARSE_ATTN (the graph switch is W7). So the `llama-bench` / `llama-perplexity -ub 127` / MTP-server parts of AT-C5 cannot reach the op. The parts that can were run through `ggml_backend_sched` with the llama layout (`w4_reserve`):
  - reserve at (ub, n_ctx) with the policy G;
  - every runtime shape and a forced G flip, with `GGML_SCHED_DEBUG_REALLOC=1`.
  - The model-level recipes are re-run when W7 / W9 wire the op (UNPROVEN until then).
- **AT-A e2e confirmation** (C10 f16 ub512 pp512 d131k, policy vs forced G4) also needs W7. It is UNPROVEN until then.
- **The plan's AT-A prediction did not hold.** The plan said "f16/bf16 at n_q 512 d131k pick G1". The measured G1 / best-G4 ratio there is 1.16 (`w4_ata.md`), so G4 with direct V is picked. The other two predictions held: q8_0/q4_0 pick G4 direct, and at d4k every type picks G4 at n_q 512.

**Finding F-W4-1 (for W7's guard / pins).** An unpinned SPARSE_ATTN_PLAN whose `sel` is a host graph input is placed on the CPU by the scheduler. The HIP attend then gets a copy of the plan dst as `src[3]`, and it aborts with `SPARSE_ATTN: no HIP kernel for this node: malformed SPARSE_ATTN node`.
- Source: `~/dev/wt_lab/w4/reserve/res_v1/*.err`, the first AT-C5 run, 54/54 rc 134.
- Its get_alloc_size also falls back to `ggml_nbytes`.
- n_sel, n_kv and G live in the plan node, not in its bytes, so the backend cannot recover them.
- Plan §10 already requires plan and attend on the backend of the layer's K buffer. W7 must pin both (and the guard must check co-placement). With both pinned, AT-C5 passes (below).

**Finding F-W4-2 (D8 re-derivation, AT-O data).** Under CAPPED, direct V is chosen for every type at n_kv ≥ 64·n_q. For q8_0 / q4_0 it is chosen too early at small n_q (`w4_ata.md` D8 table):
- n_q 128, n_kv 8192: packed is faster by ×1.125–1.156;
- n_q 256, n_kv 32768: ×1.034 / ×1.060;
- n_q 512, n_kv 32768: ×1.157 / ×1.171;
- n_q 512, n_kv 65536: ×1.031 / ×1.054.
- f16 / bf16 prefer direct at every n_q 128 cell, and at n_q ≥ 256 from n_kv 32768 (bf16 at 512 / 32768 is within 0.4 %).

The threshold was **not changed**. A per-type limit above 64·n_q would make q8_0 / q4_0 pack more cells than f16, which breaks M1 (every T ≤ f16 + 1 MiB). The pack-limit trait is the owner's deferred AT-O decision. Recorded with F-W3-1.

**Builds** (`final_w4.sh` at `330ece83f`; the scripts commit `315fe35ac` adds no code; every rc 0)
- HIP `build/` (rule flags, `-j 16`): 0 warnings at the final incremental build. The full reconfigure build `hip_build_b1.log` has 85 warnings: 45 in clip-graph.h, 13 `ld.lld`, and 8 in test-backend-ops.cpp at lines W4 does not touch. E0 also had 85.
- Vulkan `build-vk/`: 75 warnings, as in W1. CPU-only `build-cpu/`: 72. None of them is in ggml-backend.h, a sparse-attn file, or a test-backend-ops line W4 changed (grep in `final_{vk,cpu}_build.log`).
- `tu_all.sh final` (`-Werror -Wno-tautological-compare` + resource remarks, parsed by `gcn-cdna-vgpr-check.py parse_assembly`):
  - `sparse-attn-attend-g1.cu`, `sparse-attn-merge.cu`, `sparse-attn-attend-g4.cu` and `ggml-cuda.cu` compile with 0 warnings and 0 errors;
  - **49 new kernels (48 G1, 1 merge): 0 spilling, 0 scratch**, max 131 VGPRs (`tu_final.out`).
  - This is not the repo checker with its baseline (W13).

**Tests** (`final_*.log`)
- `-o SPARSE_ATTN -b ROCm0 --fail-unsupported`, HIP graphs on and off: **854/940 pass**, with 0 failed comparisons.
  - The 86 "not supported" cases are exactly the S_generic pairs (66: W5) and rot ≠ 0 (20: W10). `grep` confirms none is an S_fast rot-0 case.
  - Under `--fail-unsupported` they turn the run's rc to 1, as the 219 did at W3.
  - **Census coverage 469/469 compiled keys launched (OK)**: every G1 instance × split class, `sa_merge`, and all G4 keys.
  - Every case passes 2 determinism runs (memcmp).
- G1 band reach (`final_attn_graphs.log` SA_REPORT, G1 cases only), max NMSE/band per class: 1 split 0.252 (69 cases), 2-5 0.418 (157), 6-16 0.410 (48), 17-33 0.466 (150), 34+ 0.483 (48). The earlier run `t3_all.log` gave max 0.487.
- Other runs:
  - `-p group=4`: 382/382, the W3 set, unchanged;
  - CPU `-o SPARSE_ATTN --fail-unsupported`: 940/940 in the HIP, Vulkan and CPU-only builds;
  - SPARSE_ATTN_PLAN: ROCm0 185/185 (census 2/2), CPU 185/185;
  - `ctest test-sparse-attn-plan-oracle`: 1/1 (`final_ctest_oracle.log`).
- Regression, equal to the W0 reference counts:
  - ROCm0: QSA_PREFILL_MASKLESS 26/26, QSA_DECODE_MASKLESS 52/52 + 1 not supported, QSA_PREFILL 26/26, QSA_DECODE 92/92;
  - Vulkan0: 14/14 + 12, 34/34 + 19, 26/26, 92/92;
  - Vulkan0 SPARSE_ATTN / PLAN: all "not supported" (940 / 185, rc 0; W6).

**Done-when (PLAN-v3 §13 W4)**
- **n_q set × n_stream {1, 2} within the UNI band of the census class: PASS.**
  - Every case of n_q {1, 2, 4, 8, 9, 64, 65, 66, 127, 263} × {1, 2} streams × S_fast passes its census-class band.
  - Classes reached: 17-33 (n_q 1–9), 2-5 (64–263).
- **Split classes 6–16 and > 33 measured first on the PR-verbatim decode (R2-BOP harness): done, then gated** (`summaries/w4_bands.md`).
  - Two inputs were used:
    - UNI: 12 shapes × f16/q8_0 × 3 seeds, planted defects on;
    - TBO: R2-BOP `bands_dec` on the 88 same-type G1 dumps of the W4 cases (`GGML_SA_TEST_DUMP`).
  - 6-16 row = 2 × 7.900e-8 = **1.58e-7**. It is set by UNI 32×2051×8192 (9 splits); the TBO max is 5.54e-8. Min k = Dmin/p100 = 13.3.
  - > 33 row = 2 × 8.441e-8 = **1.69e-7**. It is set by TBO 1×4096×16384 f16 (64 splits); the UNI max is 7.77e-8. Min k = 12.3.
  - Every existing UNI row holds over the TBO dumps (class 2 × p100 ≤ row in each class), so they are unchanged.
  - The new kernel's NMSE is up to 1.586× the PR's on canonical (sorted) TBO rows, but ≤ 0.49× its band everywhere. The likely mechanism is more online-softmax chunks on scattered rows (each cell in its own quad); this is UNPROVEN.
- **U34 as a gate: PASS.**
  - Every ported #219 shape run on G1 is ≤ its UNI row. That covers decode q {1, 9, 64} × sel {257, 2051}, 9×257 GQA 4, 70×64k, and the G1 variants of 512×{257, 2051}, 129×257, 130×32k / 270k and 200×64k. The max ratio is 0.453 (1×2051 q8_0 PR rows; `t3_all.log` SA_REPORT).
  - The G4 variants passed W3's G4 rows.
  - The UNI rows are measured against O_CON, which W1 showed bitwise equal to the new CPU op (3.35e9 values, UNI fixtures included).
  - The planted d1/d2/d3 still exceed 2 × the row: min k ≥ 12.3 in every class, so Dmin ≥ 6.1 × the row.
- **n_q 1 fixture enlarged: done** (`syn1_*`, 12 SYN 262k layers × 3 seeds × 24 rows per type).
  - PR-verbatim p100 is 5.998e-8, so the **re-set §14.2 (b) row is 1.20e-7**. The old row was 1.15e-7; R2-BOP's seed-3 value was 1.17e-7.
  - New G1 max 6.04e-8 (0.504 × row).
  - Defect Dmin/p100 = 16.9.
  - 216/216 run pairs are bitwise deterministic.
  - On these complete-quad rows the new G1 output and the PR's give the same NMSE (f16 p100 identical at s1 L6).
- **C18 decode gates: PASS** (`summaries/w4_c18gate.md`). Wiki d131k rows with SYN data; same-run C18 harness kernel × 1.014; mean of r1/r2; r1/r2 spread ≤ 0.6 %.

  | case | new | C18 | ratio | plan's absolute bound |
  |---|---|---|---|---|
  | q8_0 D+S, n_q 512 | 3.309 ms | 3.754 | 0.881 | 3.770 |
  | q4_0 X+S, n_q 512 | 3.018 | 3.596 | 0.839 | 3.579 |
  | bf16 X+S, n_q 512 | 2.969 | 3.555 | 0.835 | 3.601 |
  | tg q8_0 | 18.3 µs | 19.9 | 0.920 | 19.9 µs |
  | tg q4_0 | 17.6 | 20.8 | 0.846 | 19.7 µs |
  | tg bf16 | 17.2 | 19.9 | 0.864 | 19.1 µs |

  - With the split merge included, every tg pair still passes: G1M ≤ C18 + PR merge × 1.014.
  - Against the PR-verbatim decode: q8_0 0.511 (n_q 512) and 0.579 (tg); f16 0.633 and 0.719.
  - rand d131k is reported: every ratio ≤ 0.948.
- **AT-A: done, rule encoded** (`summaries/w4_ata.md`, `330ece83f`).
  - 264 cells; noise p90 0.33 %, so the G1 margin is 1 %.
  - G1 thresholds (largest n_q), by V type and n_kv class 8192 / 32768 / 65536 / 135168:

    | V type | 8192 | 32768 | 65536 | 135168 |
    |---|---|---|---|---|
    | f16 | 32 | 16 | 16 | 64 |
    | bf16 | 32 | 32 | 16 | 64 |
    | q8_0 | 32 | 32 | 32 | 96 |
    | q4_0 | 32 | 64 | 64 | 128 |

  - It includes MTP verify widths 2/4/8, which give G1 everywhere.
  - D8 was re-derived at n_q 128/256/512 (F-W4-2).
  - Trimmed-layer gate row (last 512 queries of L11, ub4096 pos0 d131k): the policy picks G4 with direct V, the fastest arm for all four types. PASS.
  - `w4_procs` checks the proc grid, the prefer_dense table, and that `GGML_CUDA_SA_FORCE` is honoured (0 wrong).
- **AT-C5 (op level): PASS** (`summaries/w4_atc5.md`). Grid: f16/q8_0/q4_0 × ub {64, 65, 66, 127, 128, 129} × n_ctx of `-p 4096 -n 32 -d {4096, 32768, 131072}`, reserve with the final policy.
  - **54/54 runs PASS** (`atc5.out`): 62,532 runtime shapes allocated through the scheduler, 9,720 of them computed; 10 runs reserved with G1 and 44 with G4.
  - Reserved plan alloc = max over G of plan_nbytes + scratch, exact.
  - Reserved attend alloc ≥ round_up(nbytes) + the G1 partial max over n_q' ≤ ub, recomputed in the harness from the PR rule (equal to the byte), and ≥ every runtime node's alloc.
  - The forced G flip, at every n_q' including 65 / 127 / 129: 0 re-plans (realloc callback count 0, and `GGML_SCHED_DEBUG_REALLOC=1` never aborted), 0 strict violations (buffer sizes unchanged), 0 non-finite outputs.
  - The same grid with the pre-AT-A default policy (`res_defaultpolicy/`) also passed 54/54.
- **Census coverage and 0 spills: PASS** (469/469; 49 new kernels, 0 spills).

**Cleanup**
- TBO dumps deleted: `~/dev/wt_lab/w4/dump`, 2.0 GB.
- Cut rows files deleted: `bands/rows`.
- `find ~/dev/wt_lab/w4 ~/dev/wt_rewrite_logs/W4 -size +100M` finds nothing.
- No process I started is still running.

## W5: HIP generic and portable tiers

Everything ran on strix-002, and no other agent was active. Every build, test and timing run ran under `flock ~/gpu.lock`. The full test-backend-ops suite was never run, and every run used one op with `-j 1`.

**Resume.** A previous W5 agent was cut off when the orchestrator host rebooted. This agent continued its uncommitted work on top of `315fe35ac`; nothing was reset or discarded (backup: `strix-002:~/dev/wt_lab/w5_interrupted_backup_20261106`).
- The resume note says the multi-arch build died with the SSH connection. The logs show otherwise: it ran under nohup and finished (`runs2.out`: `EXIT ma_build 0 ... errors=0`, `RUNS2_DONE`; `ma_build.log` ends at `[100%] Built target test-chat`). The gate was still rerun from an empty directory, as the note asked.
- Local edits that never reached the tree: the README section (no W5 line in the tree's README), re-applied from the transcript. The two summaries and the op-time campaign had never been produced (`~/dev/wt_lab/w5` was empty), so they were produced here.
- Before the commits, the campaign's `w5_time` harness was smoke-tested on one layer.

Where things are:
- Logs: `strix-002:~/dev/wt_rewrite_logs/W5/`: `final.out`, `final_*.log`, `tu_final.out`, `arch_check.out`, `time.out`, the first-run logs (`t1_attn` to `t7_cpu`, `r_*`, `ma_build`, `runs1.out`, `runs2.out`), commit messages in `msgs/`.
- Timing: `strix-002:~/dev/wt_lab/w5/time/res/`.
- Results in the tree: `scripts/qsa-validation/summaries/w5_tests.md`, `w5_time.md`; harness in `scripts/qsa-validation/harness/w5/`.

Branch HEAD before W5: `315fe35ac`.

**Commits** (branch `qsa-rewrite` in `~/dev/wt_rewrite`). Author and committer are `Gaetan Puleo <12990773+gaetan-puleo@users.noreply.github.com>`. Nothing was pushed. Listed with `git log --format='%h %an <%ae> | %s' 315fe35ac..HEAD`:
- `3b93e6acd` hip : SPARSE_ATTN generic and portable tiers, GGML_CUDA_SA_FORCE generic|portable. Trailer `Co-authored-by: Iggy Jackson <iggy@theiggy.com>` (name and email from `git log -1 94e086847`): the generic instances extend the kernel bodies ported from #219 in W3 / W4.
- `431fd7def` tests : SPARSE_ATTN generic-tier cases, GQA 4/8 replicated-heads check, forced tiers.
- `cdfaeac9f` scripts : W5 harnesses and results.

**What changed**
- `ggml/src/ggml-cuda/sparse-attn-generic.cu` (new): `sa_attend_g4<SA_T_GEN, SA_T_GEN, packed|direct, plain, 4|8|12>`, `sa_attend_fixup<SA_T_GEN, SA_T_GEN, plain>`, `sa_attend_g1<SA_T_GEN, SA_T_GEN, 4|8|12>`, `ggml_cuda_sa_gen_g1_launch`.
- `ggml/src/ggml-cuda/sparse-attn-portable.cu` (new): `sa_attend_portable` (scalar, LDS, `__syncthreads` only, any plan G, every type of S), `ggml_cuda_sa_portable_launch`.
- `sparse-attn-loaders.cuh`: `SA_T_GEN`, `sa_elem_t`, `sa_any_elem`, `sa_any_piece`, `sa_any_quad4`: to_float as ggml-quants.c in f32, then one RNE rounding to f16.
- `sparse-attn-attend-g4.cu`: `sa_select` picks the tier (`sa_device_portable`, `sa_type_in_S`), `sa_force` parses `GGML_CUDA_SA_FORCE` once (vpacked, vdirect, generic, portable; g1 / g4 stay the group policy's), `sa_run` dispatches the tiers, `gate_ok` is fast-tier only, `census_skip` takes the device.
- `sparse-attn-attend-g4.cuh`, `sparse-attn-attend-g1.cuh`: generic branches in the K / V loaders, `kt_g` / `vt_g` arguments, `sa_g4_waves_per_eu` (the direct-V GQA 12 generic instance targets 7 waves per EU).
- `sparse-attn-pack.cu`: `sa_pack_v<SA_T_GEN>`. `sparse-attn-census.h`: the `sa_generic/*` and `sa_portable/G<g>` keys (494 in all). `ggml-cuda.cu`: the census skip proc passes the device. `sparse-attn.cuh`: declarations.
- `tests/test-backend-ops.cpp`: band lookup for the new census keys (`sa_band_for_backend`), `test_sparse_attn_gqa_rep`, `test_sparse_attn_gated` under unfused tiers, f32 V overflow, 270 new SPARSE_ATTN cases (1210 in all, 940 before).

**Deviations from the plan's wording**
- §6.1 says "one LDS-staged loader" for S_generic. The generic tier reuses the S_fast loader shapes instead (split K; packed or split-gather V on G4; LDS-staged V on G1), with the conversion as a runtime switch (`sparse-attn-loaders.cuh` header comment). Generic K and direct-V conversions run after the barriers, because converting before them spilled the GQA 12 instances (`tu_c1.out`: `spilling=2`).
- §6.2 calls the portable tier "scalar FMA". It runs with FP contraction off, the CPU op's per-cell arithmetic (`sparse-attn-portable.cu` header comment).
- §6.2 names the census keys `sa_generic/G<g>` and `sa_portable`. They are extended with the V loader, GQA, split class and plan G (`sparse-attn-census.h`).

**Build** (`final.out`, HEAD `431fd7def`, only the W5 scripts uncommitted)
- HIP clean rebuild: EXIT 0, errors=0. The sha256 of `libggml-hip.so` (`4ed20bd7...`) and `test-backend-ops` (`df6c36f3...`) are identical before and after, so the first-run logs (`t1`-`t7`, `r_*`) ran on the same binaries as the commits.
- Vulkan `build-vk`: EXIT 0. CPU-only `build-cpu`: EXIT 0. Both errors=0.
- TU checks (`tu_final.out`): 11 TUs under -Werror, 289 SA kernels, 0 spilling, 0 with scratch. New kernels: generic G4 119-209 VGPRs, generic G1 146, fixup 30, generic pack 28, portable 53.

**Done-when results**
- **Multi-arch compile gate: PASS.** `GPU_TARGETS="gfx1100;gfx1151;gfx90a;gfx942"` from an empty `build-multiarch`: EXIT ma_cfg 0, EXIT ma_build 0, errors=0 (`final_ma_build.log`). `sa_attend_portable` has device code with no `v_permlanex16` on all four targets (`arch_check.out`). The 52 "loop not unrolled" warnings are not attributed to TUs (UNPROVEN), and the 3 tier TUs alone show warnings=0 on all four targets.
- **`-o SPARSE_ATTN` generic and forced portable: PASS.** Each ROCm0 run reports 1190/1210 passed, 20 not supported, rc=1. All 20 are rot != 0 cases (`notsup_with_rot0=0` in every run), which are W10's; `--fail-unsupported` is what makes rc 1 (0 compare errors in every run). Max NMSE / band:

  | run | log | pairs seen | census | generic / portable G1 | G4 |
  |---|---|---|---|---|---|
  | default | `final_attn_graphs.log` | 81/81, 65/65 generic-member pairs on G1, G4 packed, G4 direct | 492/492 | generic 0.383 | generic 0.358 packed, 0.322 direct |
  | `GGML_CUDA_SA_FORCE=generic` | `final_generic.log` | 81/81 on the same 3 classes | 24/24 | generic 0.473 | generic 0.500 packed, 0.731 direct |
  | `GGML_CUDA_SA_FORCE=portable` | `final_portable.log` | 81/81 on G1 and G4 | 2/2 | portable 0.002 | portable 0.006 |

  Bands are set by §14.2 (c), never measured on these tiers: `sa_generic/G4` takes the G4 row, `sa_generic/G1/splits<n>` the G1 row of that class, `sa_portable/G4` the G4 row and `sa_portable/G1` "G1, 1 split", each times the type spread (`tests/test-backend-ops.cpp`, `sa_band_for_backend`). Rows come from PR-verbatim kernels (§14.2 (a), W3 `w3_prband.md`, W4 `w4_bands.md`). f64 errors are reported per tier and type in `w5_tests.md`, not gated: max `f64_c18` 3.04e-4 default, 3.22e-4 portable.
- **GQA 4/8 replicated heads: PASS.** Every replicated head equals its GQA 4 / 8 head bitwise on the G4 and G1 instances: default run 28 cases (fast 20, generic 8), forced generic 28, forced portable 28 (`pairs.py`, `SA_GQAREP` lines). The gate is the GQA 12 row; the observed NMSE is 0. The check is not vacuous: the same cases report nonzero NMSE against the CPU op (forced generic, `SPARSE_ATTN_GQAREP`: up to 0.394 of the band, `sarep.py`).
- **Generic kernel time: recorded** (`summaries/w5_time.md`; not gated; r2/r1 within 1.4 %). Time ratios to default-arm f16 on the C18 wiki rows:

  | type | d4k default | d131k default | d4k direct V, / f16 direct | d131k direct V, / f16 direct |
  |---|---|---|---|---|
  | f32 | 1.07 | 1.42 | 1.51 | 2.26 |
  | q4_1 | 1.06 | 1.02 | 1.39 | 1.38 |
  | q5_0 | 1.13 | 1.08 | 1.54 | 1.51 |
  | q5_1 | 1.13 | 1.06 | 1.51 | 1.47 |
  | iq4_nl | 1.15 | 1.10 | 1.54 | 1.56 |

  Forced portable on gfx1151 is 12.1-12.5 x f16 at d131k and 16.7-17.1 x at d4k, on layers 0, 5, 11.

**Other tests** (`final.out`, all rc=0 unless stated)
- `-o SPARSE_ATTN` on ROCm0 with graphs off, `=vdirect` and `=vpacked`: each 1190/1210, 20 rot cases not supported, rc=1 as above; census 492/492, 388/388, 393/393.
- `-o SPARSE_ATTN` on the CPU: 1210/1210 (HIP, Vulkan and CPU-only builds).
- SPARSE_ATTN_PLAN: ROCm0 185/185, CPU 185/185. Plan oracle ctest: pass.
- QSA_PREFILL_MASKLESS 26/26, QSA_DECODE_MASKLESS 52/52, QSA_PREFILL 26/26, QSA_DECODE 92/92 on ROCm0.
- The same four on Vulkan0: 14/14, 34/34, 26/26, 92/92. SPARSE_ATTN and SPARSE_ATTN_PLAN on Vulkan0: all not supported until W6.

**Open points for the owner**
- **Possible P4 misses.** At op level, f32 at d131k and q5_0, q5_1, iq4_nl at d4k are past a time ratio of 1.11 on the default arm (P4's 0.90 speed), and on the direct-V route every S_generic type is, f32 reaching 2.26 x at d131k. P4 is decided end to end later, and §16 lets such a pair ship listed as "not first-class". Against the owner goal of no performance cliff for any supported type, f32 direct V is the case to look at. Whether these miss P4: UNPROVEN until the P4 cells run.
- **Portable on non-RDNA3.5 hardware (U15).** Only the compile gate and the forced runs on gfx1151 exist. Its runtime on those archs is UNPROVEN, and at 12-17 x f16 on gfx1151 it may be slow there.

**Cleanup**
- Temp TU and `arch` directories deleted (`~/dev/wt_rewrite_logs/W5` is 7.7 MB). `find ~/dev/wt_lab/w5 ~/dev/wt_rewrite_logs/W5 -size +100M` finds nothing.
- `build-multiarch` (754 MB) is kept. No process this agent started is still running.

## W6: Vulkan plan and attend, portability

Everything ran on strix-002, with no other agent active. Every build, test, capture and timing run ran under `flock ~/gpu.lock`. The full test-backend-ops suite was never run, and every run used one op with `-j 1`. A previous W6 agent left nothing: the tree was clean at `cdfaeac9f` and `~/dev/wt_rewrite_logs/W6` did not exist.

Logs: `strix-002:~/dev/wt_rewrite_logs/W6/`. Timing: `strix-002:~/dev/wt_lab/w6/time/res/`, and the first version (slow loaders, K select) in `res_v1/`. Results in the tree: `scripts/qsa-validation/summaries/w6_tests.md`, `w6_time.md`; harness in `scripts/qsa-validation/harness/w6/`.

Branch HEAD before W6: `cdfaeac9f`.

**Commits** (`git log --format='%h %an <%ae> | %ce | %s' cdfaeac9f..HEAD`). Author and committer are `Gaetan Puleo <12990773+gaetan-puleo@users.noreply.github.com>`. Nothing was pushed.
- `06d8072de` vulkan : SPARSE_ATTN_PLAN and SPARSE_ATTN, coopmat / scalar / portable tiers. Trailer `Co-authored-by: Iggy Jackson <iggy@theiggy.com>`: `sa_q8x2` / `sa_q4x2` adapt #219's q8 bit trick (via `sparse-attn-q8.cuh`). #219 does not touch ggml-vulkan (C11), so nothing else here comes from it.
- `34b1e7223` tests : Vulkan SPARSE_ATTN cases, limits, capability, rotation and workspace stress checks.
- `b1cb6764c` vulkan : SPARSE_ATTN cm kernel: no K select, uniform V branch.
- `65cd5a3d9` scripts : W6 harnesses and results (Vulkan op runs, AT-B2, AT-B3, AT-B7, mutants).

**Results the owner should read first**
- **AT-B3 diagnostic gate: missed for q8_0 and q4_0, met for bf16.** Plan text: ≤ 1.15 × new f16, or ≤ the HIP C18 ratio + noise (q8_0 1.23×, q4_0 and bf16 1.67×); "Binding gate: AT-B1", which W7 runs. Shipped arm (V direct), attend / new f16, logger, 3 processes (`summaries/w6_time.md`):

  | type | C11 text / rand | d4096 text / rand | d131072 text / rand | gate |
  |---|---|---|---|---|
  | bf16 | 1.353 / 1.304 | 1.382 / 1.387 | 1.291 / 1.094 | met everywhere |
  | q4_0 | 2.058 / 1.897 | 2.104 / 2.114 | 1.895 / 1.325 | met only at d131072 rand |
  | q8_0 | 2.241 / 2.064 | 2.307 / 2.306 | 2.045 / 1.439 | missed everywhere |

- **The pack arm gets closer, and M1 forbids it.** It meets the q4_0 gate everywhere and the q8_0 gate only at d4096 (1.26-1.41 elsewhere). bf16 is worse with it at d131072. vpack, same table order: q8_0 1.260 / 1.266, 0.964 / 0.970, 1.409 / 1.311; q4_0 1.211 / 1.220, 0.966 / 0.974, 1.362 / 1.275; bf16 1.178 / 1.189, 0.947 / 0.956, 1.429 / 1.327 (pack time included). Attend node allocation 124,755,584 B vs 53,452,416 B at C11's shape (+68 MiB), 380,608,128 B vs 103,784,064 B at d131072 (+264 MiB): T above new f16 + 1 MiB, so by the AT-B3 rule it is not shipped. It runs under `GGML_VK_SA_FORCE=vpack` only.
- **Risk for AT-B1 (W7), arithmetic, UNPROVEN at graph level:** q8_0 attend 15.84 ms vs f16 7.07 ms per layer at C11's shape (logger). Over the 12 layers (one at 512 q) against C11's 1591 ms graph that is about +105 ms, about +6.6 %, and AT-B1 asks ≥ 0.95 × new f16. If AT-B1 misses, the pack arm against M1 is the owner's call (§17).
- **Staged V lost everywhere,** also for f16 (1.66-2.05 × direct), so V direct is shipped for every type (`ggml_vk_sa_default_vl`).
- **AT-B2: PASS.** New f16 op / in-build sel_cm (≤ 1.10), logger: C11 text 0.917, rand 0.914; d131072 text 0.920, rand 0.958; d4096 text 0.916, rand 0.910. With the plan node added (the old op has its pre-pass inside): 1.071, 1.055, 1.031, 1.039, 1.047, 1.041.
- **AT-B2 / AT-B3 ran at op level: real rows (captured text and random), SYN Q / K / V.** The graph uses the op from W7 on. Numbers on model data are UNPROVEN until then. `sched_reserve` per arm is replaced by the node's `get_alloc_size` for the same reason.

**What changed**
- `ggml/src/ggml-vulkan/vulkan-shaders/` (new):
  - `sparse_attn_plan_rows.comp`: G = 1 plan. 32-bit storage only, the sort in the storage buffer (bitonic, flip form), a two-level sum instead of a Hillis-Steele scan, masks as whole dwords by `atomicOr`.
  - `sparse_attn_cm.comp`: from `flash_attn_sel_cm.comp`. Steps of 4 plan entries, score -inf and zero V column for masked cells, split z takes steps z, z + n_split, ..., V direct or staged, `ROT_Q` spec constant with H_q in registers, a `ROT_DUMP` test hook.
  - `sparse_attn.comp`: scalar tier, from `flash_attn_sel.comp`. `sparse_attn_portable.comp`: portable tier. `sparse_attn_reduce.comp`: split merge without the clamp. `sparse_attn_pack.comp`: pack arm. `sparse_attn_load.glsl`: `flash_attn_dequant.glsl` wrappers and the packed f16 helpers.
- `fwht.comp`: `FWHT_INPLACE` (one binding), for H_o. `vulkan-shaders-gen.cpp`: the new SPIR-V.
- `ggml-vulkan.cpp`: push constants; pipeline gates and creation in `ggml_vk_load_shaders` (`sa_tier_ok`, `pipeline_sparse_attn` map, plan, reduce, pack, in-place fwht); `ggml_vk_sa_force`, `ggml_vk_sa_tier`, `ggml_vk_sa_select`, `ggml_vk_sa_plan_supported`, `ggml_vk_sa_alloc_size`, `ggml_vk_sa_ws_subbuffer`, `ggml_vk_sparse_attn_plan`, `ggml_vk_sparse_attn`; `get_alloc_size` hook; `supports_op` cases; census (`ggml_vk_census_key_add`, `ggml_vk_sa_census_list`, `ggml_vk_sa_census_skip`); policy procs; `ggml_backend_debug_vk_sa_binding_limit`; flops estimate. `ggml-vulkan-types.h`: `vk_sa_tier`, `vk_sa_vl`, `vk_sa_key`, device members.
- `tests/test-backend-ops.cpp`: `expected_unsupported(ggml_backend_t)`, Vulkan keys in `sa_band_for_backend`, 81 cases at n_q 65, n_kv 4096 (1291 SPARSE_ATTN cases, 1210 before). `tests/test-sparse-attn-vk.cpp` (new), `tests/CMakeLists.txt`.

**Deviations from the plan's wording**
- **Plan shader.** §6.3 derives it from `flash_attn_sel_prep.comp`. It shares only the workgroup shape: the hash table and 33 KB of shared memory are replaced by a sort in the scratch binding, because the plan contract has no n_sel limit and the test list has n_sel 9000. Its gate: up to 256 invocations, a power of two ≥ 16, and (WG + WG / 16 + 2) × 4 B shared. AT-B7 (0): `flash_attn_sel_prep.comp` does use only `barrier()` and shared memory (`:90-131`, no subgroup extension).
- **Portable tier.** §6.3 names `flash_attn_dequant.glsl` and a subgroup-size spec constant. That file needs 16-bit storage and int8, so the tier reads 32-bit words with its own loaders and integer f16 conversions, and it reads no subgroup value at all. One invocation per query head, not a reduction: no reduction is left to do. It never splits, so `sparse_attn_reduce.comp` serves cm and scalar only, as on HIP.
- **H_o** is a dispatch of `fwht.comp` (one-binding variant), not part of the reduce. H_q is in each kernel. The HIP rot stays with W10; §13 W6 lists the Vulkan unit test, so Vulkan has it now.
- **Census key.** §6.3's `sa_attend/vk/<tier>/<VL>/<KT>/<VT>`, plus `sa_merge/vk/splits<class>` for the band row.
- **Expected-unsupported by size.** §14.1 expects cells past 1 GiB on the list. On RADV they pass (n_kv 2^21 + 4 f16, a 2 GiB K view, `run_radv_SPARSE_ATTN.log`), so the list only applies where the device reports it.

**Build** (rc 0, `final.out`, `sha_final.txt` = `sha_after_mutants.txt`): HIP `build/`, Vulkan `build-vk/` all targets, CPU-only `build-cpu/`. ctest `test-sparse-attn` in build-vk: 2/2. `grep -c warning`: `vk_build_t.log` 0.

**Done-when results** (`summaries/w6_tests.md`)

| run (`--fail-unsupported`) | SPARSE_ATTN_PLAN | SPARSE_ATTN | census | max NMSE / band |
|---|---|---|---|---|
| RADV default (cm, V direct) | 92/92 | 711/711 | 85/85 | 0.518 |
| forced scalar | 92/92 | 711/711 | 84/84 | 0.613 |
| forced portable | 92/92 | 711/711 | 81/81 | 0.003 |
| forced portable, required subgroup 32 | 92/92 | 711/711 | 81/81 | 0.002 |
| forced vdirect | 92/92 | 711/711 | 85/85 | 0.500 |
| forced vstaged | 92/92 | 711/711 | 85/85 | 0.509 |
| forced vpack | 92/92 | 711/711 | 93/93 | 0.506 |
| lavapipe | 92/92 | 703/703 | 81/81 | 0.003 |

- **Whole-dst plan bit-exact:** PASS, in every row above. Each case compares with the CPU op and the C3 oracle, and runs twice. "Not supported": 93 plan, all G = 4, listed.
- **Every S² pair, bf16 D256, V overflow, poison, determinism:** PASS. 81 pairs seen in every run; 580 "not supported", all G = 4, listed. rot cases run on every tier.
- **AT-B7:** PASS with the lavapipe deviation below. RADV at required subgroup 32: 0 failures, only the G = 4 list. That the driver ran at 32 is by the pipeline's `requiredSubgroupSize`; no readback (UNPROVEN beyond the create call).
  - **Lavapipe "0 not supported" is not literally met:** 8 cases past its 128 MiB `maxStorageBufferRange` are rejected and listed: f16 n_kv 270336 (2), 1048576, 2097156; q8_0 n_kv 270336, 1048576, 2097156; f32 n_kv 1048576. Rejecting them is P2-19's own rule. The same limit rejects an attend dst past 128 MiB, ub 8192 (`vkt_lvp.log`, limits).
  - **The system lavapipe cannot run these shaders.** Mesa 26.0.3 + LLVM 22.1 crashes while compiling compute shaders with `barrier()`, including the pre-existing `flash_attn_sel_prep.comp` (`plan_lvp_system_mesa.log`: rc 139; SIGSEGV in `llvm::InstCombinerImpl::simplifyMaskedScatter`, and "LLVM ERROR: Cannot select" on `X86ISD::MGATHER`; `lvp_try` + `lvp_bisect.py`). `GALLIUM_OVERRIDE_CPU_CAPS=avx` and `GALLIVM_PERF=no_opt` do not fix every shader. So lavapipe ran from Fedora 43 packages in `~/dev/wt_lab/w6/lvp43`: Mesa 25.3.6, LLVM 21.1.8. Limits (`vulkaninfo_lvp.txt`): subgroup 8 (min = max = 8), 1024 invocations, 32768 B shared, range 134217728, offset alignment 16.
  - **supports_op false with no pipeline:** PASS. `GGML_VK_SA_CAP_MASK=15`: 185/185 plan cases "not supported"; `=7`: 1291/1291 attend cases; nothing dispatched, rc 0 (`mask15_plan.log`, `mask7_attn.log`). `test-sparse-attn-vk capability` on both devices.
  - **Devices that lose QSA, by capability (for §19):** the plan needs 16 invocations and 76 B shared at its smallest; portable needs 12 invocations and 2 KiB shared; a binding past `maxStorageBufferRange` loses that shape; rot_o ≠ 0 needs an fwht pipeline (one Intel Windows driver range lacks it). The §10 error text is W7's.
- **Unsorted-row plan cases with the validation layer:** 0 messages, with a caveat. RADV and lavapipe, GPU-AV on: 92/92, 0 VUID, every tier's attend smoke 8/8, 0 VUID (`val_*.log`). The layer was not installed: `vulkan-validation-layers` 1.4.341 is extracted to `~/dev/wt_lab/w6/tools`.
  - **Caveat, no positive control:** a mutant binding half the scratch was **not** reported by GPU-AV (`mut_short_scratch_gpuav.log`: 0 VUID). So "0 out-of-range access" from the layer proves little (cause UNPROVEN). The same mutant is caught by the whole-dst compare (0/1 passed), and the bound itself is by construction: the binding is the explicit sub-buffer, asserted against the allocation.
- **Rotation:** PASS. H_q in the kernel (`rot_dump`) and H_o bitwise equal to `fwht.comp` on the same input: rot 64, 128, 256 × n_q 3, 65 × f16, q8_0. 36/36 on RADV (3 tiers), 24/24 on lavapipe (default = portable, and forced portable; no scalar).
- **Stress:** PASS as specified, with a caveat. The test asserts from `tensor->data` and `get_alloc_size` that the workspace overlaps T, read by the preceding dispatch, and that dst is clear of T and of the unsynced writes. 19 of 19 iterations stable and bitwise equal to the graph with T kept alive: attend workspace and plan scratch on RADV, plan scratch on lavapipe (the portable tier has no workspace).
  - **Caveat: it does not catch the removal of the rule.** With either `ggml_vk_sync_buffers` deleted, the test still passes, and synchronization validation reports 0 hazards, submit-time validation on (`mutants.out`; `mut_sync_*_syncval.log`, run one commit before HEAD: `b1cb6764c` touches only the cm kernel's selects). Whether the race can occur on RADV at all is UNPROVEN. The rule stays by the O17 reasoning.
- **Limits at the -c 1048576 reserve:** PASS. ub 8192: plan alloc 168,132,864 B, attend alloc 204,447,360 B against the 1 GiB block; K/V views f16 and bf16 1 GiB, q8_0 570,425,344 B, q4_0 301,989,888 B, within the binding limit on RADV.
- **Other mutants** (`mutants.out`), all caught: no sort (27/92), no V zeroing (0/9), no score select (0/9), wrong H_q butterfly (8 failed checks), wrong q8_0 V scale (0/2). Dropping only the K select is not a defect, so the code dropped it (`b1cb6764c`).
- **`GGML_VK_SA_MAX_Q=7`** (the host loop for `maxComputeWorkGroupCount[0]`): plan 92/92, attend smoke 8/8 per tier.

**Regressions** (`final.out`): HIP SPARSE_ATTN_PLAN 185/185; SPARSE_ATTN 1271/1291, census 492/492, rc 1 from the 20 rot ≠ 0 "not supported" (`grep -cv "rot_q=0,rot_o=0"` = 20, W10's); QSA_PREFILL_MASKLESS 26/26; QSA_DECODE_MASKLESS 52/53, rc 1 from the one q8_0 case W2 already lists; QSA_PREFILL 26/26; QSA_DECODE 92/92. Vulkan: QSA_PREFILL_MASKLESS 14/14, QSA_DECODE_MASKLESS 34/34, QSA_PREFILL 26/26, QSA_DECODE 92/92, as C11. CPU: 185/185, 1291/1291.

**Not done here (later items):** the §10 error text, the Vulkan `prefer_dense` flip (AT-B4), AT-B1, AT-B5, the Q gates. The policy procs return G = 1 and the default prefer_dense.

**Cleanup:** the two captured rows files are deleted (189 MB each; `time_campaign.sh` recaptures them). No file over 1 GB under `~/dev/wt_lab/w6` or the log directory. Kept for later items: `tools/` and `lvp43/`. No process this agent started is still running.

## W7: graph switch, pins, guard, fit probe

**Status: implemented and committed (not pushed); the correctness checks pass on the final build; several gates FAIL or are open (listed under "Verdicts" and "Owner decisions"); the campaigns the owner cut short are under "Moved to Prove".** The owner stopped W7's measurement chains on 2026-11-07 around 21:00 CET (chain8 / 9 / 10 / 12, outputs kept in `strix-002:~/dev/wt_lab/w7/`); a run whose log is incomplete is treated as not run. All paths below are on strix-002 unless stated: `W7 = ~/dev/wt_lab/w7`, harnesses `scripts/qsa-validation/harness/w7/` (H), summaries `scripts/qsa-validation/summaries/` (S). Backup of the interrupted state: `~/dev/wt_lab/w7_wrapup_backup_20261107/` (HEAD, status, tracked.diff, untracked.tgz).

**Done-when scorecard** (details and sources below):
| done-when item | result |
|---|---|
| census rows = shipped policy (6 pairs, ub 16 / 512 / 4096, bench d4096) | PASS 24/24 (final build) |
| MTP census (draft default, q8_0, q4_0) | PASS 3/3 |
| guard fires on a fake device and on forced reject, not on validation runs; non-vacuity unit test | PASS |
| AT-W7PERF shortcut grid and per-band flip (HIP) | PASS, policy committed |
| AT-W7PERF MTP cell (b) | not run → Moved to Prove |
| AT-W7PERF binding -nkvo gate (c) | (a) FAIL all four types; (b) PASS q4_0, f16 / q8_0 PASS except 4 unresolved d0 cells, bf16 not run; (b) shipped for the four types |
| AT-W7PERF (d) shortcut pin | not run → Moved to Prove |
| Q1 | A/B/C PASS; direct form FAIL q4_0 and bf16 at both stages (owner) |
| Q2 / Q3 / Q4 / Q5 / Q8 / Q11 / Q12 | report / PASS except Q3-B q4_0 not run / PASS (plan thresholds, thin f16 margins) / not re-run, kernels unchanged / PASS / a, b PASS, c not run / PASS (-nkvo), -ngl 0 not run |
| P1-P7 | P1 PASS; P2, P3, P4, P6N not run; P5 not completed; P7 = W3/W4 rows |
| M1-M3 | M1 35/36 (iq4_nl MTP, W11); M2, M3 not run |
| AT-D1-D5 | D1 count criterion not met as worded (W0 = 0), U16 PASS, value not run; D2 placement / PPL PASS, CPU perf FAIL; D3 PASS; D4 PASS (copy route); D5 PASS |
| AT-C18 (-np 1) | PASS 32/32 |
| Vulkan (AT-B, run here because W6 left them to W7) | AT-B4 op PASS, policy committed, e2e not run; AT-B5 M1 PASS, one MTP cell +12 MiB over W0; Q1-vk PASS 6/6; **AT-B1 FAIL q8_0, q4_0** |

### What changed (files, functions)

- `ggml/src/ggml-backend.cpp` `ggml_backend_sched_reserve_size`: no `ggml_backend_sched_reset` (the reset wiped the pins set while llama built the graph; `llama_context::graph_reserve` resets before it builds, PLAN-v3 §7.1.7). `ggml/include/ggml-backend.h`: proc typedefs `ggml_backend_sparse_attn_unsupported_reason_t`, `ggml_backend_sparse_attn_select_dump_t` (782b1bb4a).
- `ggml/src/ggml-cuda/ggml-cuda.cu`: OVL kind-4 end predicate `FLASH_ATTN_EXT || SPARSE_ATTN` (§6.2), `LAB_OVL_PRINT` region count by kind (AT-D1), procs `ggml_backend_sparse_attn_unsupported_reason` and `_select_dump`; `sparse-attn-attend-g4.cu`: `GGML_CUDA_SA_REJECT` test hook, unsupported reason, census select dump (782b1bb4a). `sparse-attn-attend-g1.cu` `ggml_cuda_sparse_attn_group` (shortcut G: 1 to n_q 32, 4 above) and `ggml_cuda_sparse_attn_prefer_dense` (7-band table per pair from AT-W7PERF) (0294c0274). No kernel source changed in W7 (`git diff --stat 65cd5a3d9..HEAD -- ggml/`: policy procs, hook and reason only).
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`: `ggml_backend_vk_sparse_attn_unsupported_reason` (782b1bb4a); `ggml_backend_vk_sparse_attn_prefer_dense`, the AT-B4 op-level table (6e07acd3d).
- `src/models/qwen4exp.cpp`: compact CB rows and the 4-D all-visible shortcut rows feed `ggml_sparse_attn_plan` + `ggml_sparse_attn` for every pair (`build_qsa_plan`, `build_attn_qsa`), G and `prefer_dense` from the K-buffer device's procs (`qsa_policy`), `LLAMA_QSA_PREFER_DENSE` (measurement only, W13 deletes), pins for host KV (plan, attend, shortcut attention to `backend_cpu`), the named host-KV copy route `LLM_QSA_PLACE_HOST_KV_COPY` (`qwen4exp_host_kv_copy`, `LLAMA_QSA_HOST_KV_COPY`), shortcut pin switch `LLAMA_QSA_SHORTCUT_PIN` (998c46803); the -nkvo default (a2c37af52, below).
- `src/llama-context.cpp` / `.h`: `qsa_guard` on split_only reserve graphs and after every `sched_alloc_graph` (main, MTP, fit probe), non-vacuity count, llama-side CPU-launch counter, init error text (`qsa_unsupported_error`), `LLAMA_QSA_TRACE` per executed ubatch, fit-probe per-backend max over every reserve graph (998c46803). `src/llama-graph.{h,cpp}`: `llm_graph_qsa_node` records. `src/llama-model.cpp`: qwen4exp hybrid-idx memory with `v_trans = false` (D21) and the `--split-mode tensor` / `n_ctx_seq` > 2^21 + 4 init errors at both qwen4exp memory sites (789a85c12). `src/llama-memory-hybrid-idx.{h,cpp}`: the old type gate removed (998c46803).
- `tools/perplexity`, `tools/llama-bench`, `tools/server`: print the backend dispatch census at exit (`QSA_CENSUS <key> <count>`) (998c46803).
- Tests (adb011978): `tests/test-qsa-guard.cpp` (fake device FAKE0 registered with `ggml_backend_device_register`, forced reject, decode reject, non-vacuity assert), `tests/test-qsa-allvis-rows.cpp`, `tests/test-qsa-select-dump.cpp`.
- Scripts: W7 harnesses (81aeb1fda, dcaa19f16, and this wrap-up's commit), census_check fields; summaries (this wrap-up's commit).

### Commits (branch qsa-rewrite, not pushed)

782b1bb4a ggml procs / reserve_size / OVL; 998c46803 qwen4exp graph switch, pins, guard, fit probe, trace (Co-authored-by: Iggy Jackson <iggy@theiggy.com>, the init log line adapts #219's b60447506); adb011978 tests; 81aeb1fda harnesses; 0294c0274 HIP AT-W7PERF shortcut policy; 789a85c12 D21 V layout and QSA context checks at the qwen4exp memory sites; **a2c37af52** qwen4exp: -nkvo takes the host-KV copy route for f16, q8_0, q4_0, bf16; **dcaa19f16** harness updates (-nkvo phases, eviction handling, Q1 stage 2, Q4 per-type bound, Vulkan and all-visible recorders); **6e07acd3d** Vulkan AT-B4 shortcut policy; **0b0e901ed** scripts: W7 results summaries and harness fixes; **32667ca00** scripts: AT-B1 summary. Authorship: Gaetan Puleo <12990773+gaetan-puleo@users.noreply.github.com> on all (`git log --format='%h %ae'`).

**Which build ran what** (provenance): the AT-W7PERF op/e2e grids and the first census ran at 0294c0274 (chain2-6); every perf campaign of chain7-12 (ND, NDB, P1, D2B, P5, P23) ran the bench binary `W7/bench/bin_w7/llama-bench`, which loads `~/dev/wt_rewrite/build/bin/lib{llama,ggml}*.so` (`ldd`), built at 789a85c12 (`~/dev/wt_rewrite_logs/W7/hip_build8.log`, 04:07, links libllama and libggml-hip), with the route forced by env; the quality arms ran the lab tree `W7/lab` at 0294c0274 + QPIN patches (`chain6.out: LAB_BUILD_rc=0 0294c0274`; it lacks 789a85c12's llama-model.cpp hunk, which only changes `-fa off` V layout and init checks); the wrap-up checks ran at dcaa19f16 (HIP build/ rebuilt 2026-11-07 21:28, `hip_build_final.log`; lab tree reset to dcaa19f16, `W7/lab_tree_final.out`), Vulkan at 6e07acd3d (`vk_build_policy.log`; the Vulkan lab tree at dcaa19f16: the policy commit only changes the shortcut, which c8192 ub4096 Q1-vk arms never enter, n_kv ≥ 4096 > 2051).

### Build

- HIP (`build/`, flags of the rules), final: rc 0, 0 errors, 3 warnings, all in `tests/gguf-model-data.cpp` (deprecated `u8path`, not W7 code) (`~/dev/wt_rewrite_logs/W7/hip_build_final.log`). It is current for HEAD 32667ca00: `git diff --stat a2c37af52..HEAD -- . ':!scripts'` lists only `ggml/src/ggml-vulkan/ggml-vulkan.cpp`.
- Vulkan (`build-vk/`): rc 0 at dcaa19f16 (`vk_build_final.log`) and at 6e07acd3d (`vk_build_policy.log`); the two `ggml-vulkan.cpp` warnings (9691 unused `dst_buf_ctx`, 10758 unused `coopmat2`) predate W7 (`git blame`: 89ef7d894d, 7f7438e79c, Nathan Wilson). Vulkan lab tree (`W7/lab/build-vk`, target llama-perplexity): the harness's first configure missed the SPIR-V headers include (`W7/vk/build_lab.log:376` `spirv/unified1/spirv.hpp: No such file`); reconfigured with the main build-vk's `CMAKE_CXX_FLAGS=-I$HOME/.local/spirv-headers/include` and `SPIRV-Headers_DIR` (from `build-vk/CMakeCache.txt`): rc 0 (`W7/vk/build_lab2.log`); vk.sh VB fixed accordingly.

### Correctness checks on the final build (dcaa19f16 HIP, 6e07acd3d Vulkan)

Source `W7/chainF1.out` (S/w7_census_guard.md, w7_c18.md, w7_at_d.md) unless stated.
- **Per-op tests** (`H/optests.sh`, one op per run, `-j 1`, under flock): CPU SPARSE_ATTN_PLAN 185/185, ROCm0 SPARSE_ATTN_PLAN 185/185, CPU SPARSE_ATTN 1291/1291, **ROCm0 SPARSE_ATTN 1271/1291, rc 1**: the 20 are "not supported" with `--fail-unsupported`, every one has rot ≠ 0 (`grep "not supported" W7/optests/ROCm0_SPARSE_ATTN.log | grep -vc "rot_q=0,rot_o=0,"` = 20, `-c` with rot 0 = 0), 0 compare errors: rotation in the kernel is W10's (same 20 at W5/W6, IMPLEMENTATION.md W5/W6). QSA_PREFILL_MASKLESS 26/26, QSA_DECODE_MASKLESS 52/52, QSA_PREFILL 26/26, QSA_DECODE 92/92; Vulkan0 SPARSE_ATTN_PLAN 92/92, SPARSE_ATTN 711/711. Census coverage ROCm0 492/492 compiled keys launched.
- **Guard** (`H/guard.sh`): pass f16 / q8_0 / q4_0 / bf16: PASS, every reserve line `guard=24/24` and "inspected 24, 2 x layers = 24" (the unit test's non-vacuity assert, `tests/test-qsa-guard.cpp:194-202`, `W7/guard/pass_q8.log`); fake device FAKE0: init error "QSA layer 3 (cb): sparse attention K=q8_0 V=q8_0 is not supported on device FAKE0 (backend FAKE): no sparse-attention kernel in this backend; run QSA on the CPU with -ngl 0 or --no-kv-offload, or use a HIP (RDNA3.5) or Vulkan build"; `GGML_CUDA_SA_REJECT=1` on ROCm0: the same init error ("rejected by the test hook") and the decode error from `process_ubatch` (layer 3, av); `-sm tensor`: init error (meta backend has no SPARSE_ATTN handler); `-c 2097664`: init error "n_ctx_seq = 2097664 is above the tested maximum of 2097156 cells"; test-qsa-allvis-rows PASS (0 failures).
- **Route census** (`H/census.sh`, 6 pairs × llama-perplexity c8192 ub {4096, 512, 16} + llama-bench pp4096 @ d4096, `GGML_CUDA_DISABLE_GRAPHS=1`, expected rows generated by `test-qsa-select-dump` from the shipped policy): **24/24 EXPECT PASS, 0 diffs, 0 masked lines, cpu_launch 0**, guard inspected_min 12 (perplexity) / 24 (bench), cpu_launch 0 in every run; Q1-B recipe check PASS on the 6 ub512 runs; `CENSUS_DONE rc_all=0`.
- **MTP census** (`H/mtp_census.sh`, C2 flags, main q8_0, requests S0 S1 L0 L1 X0): draft KV default, `-ctkd q8_0`, `-ctkd q4_0`: EXPECT PASS, masked 0, cpu 0; every MTP ubatch past 2051 (n_kv 5632-29184) route=sparse, below it dense; guard main inspected_min 12, mtp ctx 1019 passes (MTP cache-only graphs inspect 0 by design, `src/llama-context.cpp:953`).
- **AT-C18, `-np 1`** (`H/at_c18.sh`, `-fit on`, f16 / q8_0, ub 512 / 4096, ± `-nkvo`, c131072): **32/32 rows |probe − real| = 0.00 MiB** per backend, main and MTP; the guard ran on the probe context (4 lines, inspected ≥ 2 × layers); under `-nkvo` the probe shows the named route on ROCm0 (12 plan + 12 attend, `route=host-kv-copy`) — the pins of the CPU route are not exercised there because the copy route is now the default for f16 / q8_0. **PASS.** (`-np auto` is W9's.)
- **AT-D4** (`-nkvo`, c8192 × 2, b 4096, ub 4096 / 512 / 16, f16 / q8_0, W7 vs W0-lab, `H/at_d.sh D4`): every QSA trace line `route=host-kv-copy ctx=main backend=ROCm0` (36 / 204 / 396 per run), masked 0, no CPU census entry, guard silent (rc 0) incl. passes 7 / 35 / 69. `cache_[kv]_l*` split inputs to ROCm0 present (772 / 772 / 748 `GGML_SCHED_DEBUG=2` lines): allowed, the named host-KV copy route is active and logged (AT-D4 exception). PPL new / W0: f16 1.9477 / 1.9471 (ub4096), 1.9442 / 1.9460 (512), 1.9387 / 1.9376 (16); q8_0 1.9453 / 1.9602, 1.9453 / 1.9508, 1.9406 / 1.9447; C15 f16 1.9428 / 1.9449, q8_0 1.9532 / 1.9519; every |new − C15| ≤ 0.008 against a per-run SE of 0.034. Q12 (below) PASS. **AT-D4 placement and quality: PASS** for the shipped route; the binding perf part is AT-W7PERF (c) below.
- **AT-D5** (`H/at_d.sh D5`): `-fa off` ub 512 / 4096 offloaded: sparse 144 / 24 + dense shortcut 60 / 12 trace lines, masked 0, guard 0; `-fa off -nkvo`: host-kv-copy only, masked 0. PPL new vs PR219 same flags: 1.9565 / 1.9573, 1.9485 / 1.9574, 1.9475 / 1.9505, 1.9477 / 1.9469 (|Δ| ≤ 0.009, SE 0.034). `-nkvo -fa auto` f16 ub512: FA resolved enabled (no "Flash Attention not supported, set to disabled" warning, `llama-context.cpp:745-751`; the "enabled" line is INFO and not printed at this log level, `W7/at_d/d5_nkvo_auto_lv3.log`), 108/108 QSA nodes `host-kv-copy` on ROCm0: the route the plan states for -nkvo. **PASS.** (Old `-fa off` session files: not tested here, §17.)
- **AT-D3** (partial `-ngl 24`, at 789a85c12, `W7/chain11.out`): QSA layers 3-23 on CPU, 27-47 on ROCm0, each = its K buffer's device; guard 0; **PASS**.
- **U30** (`H/allvis_rec.sh`, `W7/allvis2.out`): the raw M-RoPE image kq_mask inside the shortcut (s0: ne 1280 × 1036, 1036 image tokens) and C4's p1 / p2 image masks re-recorded: `test-qsa-allvis-rows` 0 mismatches on ROCm0, mask values only 0 / −inf; dumps deleted. Closed (ROCm0; Vulkan not run on these masks).
- **Vulkan census** (`vk.sh VC`, 6 pairs × c8192 × 1 chunk × ub 4096 / 512 on Vulkan0, build-vk 6e07acd3d; S/w7_vulkan.md): 12/12 rc 0, 0 masked, 0 `QSA_CENSUS cpu` entries, guard cpu_launch 0, every attend key `sa_attend/vk/cm/direct/<K>/<V>` on Vulkan0. The shortcut follows the new table: at ub512 the four shortcut ubatches (n_q 512) are dense for f16, q8_0, q4_0, q8_0K/f16V, f16K/q8_0V (144 sparse nodes = 12 compact ubatches × 12 layers) and sparse for bf16 (192 = 144 + 48), as `ggml_backend_vk_sparse_attn_prefer_dense` sets for band 512-1023. Vulkan per-op tests: above (at dcaa19f16; 6e07acd3d changes only the policy proc, which test-backend-ops does not call).

### Verdicts of the W7 gates from the collected data

**Quality** (S/w7_quality.md = `python3 H/q_an.py W7/q/logs`; new = W7 lab tree, PR219 = W0-lab12; c8192 × 8 pinned unless stated):
- **Q1-A/B/C: PASS** (15/15 arms; e.g. Q1-A q4_0 −1.5e-5 ± 5.0e-5, bf16 −4.1e-5 ± 2.0e-5, band 2.0e-4; Q1-C forced G1 arms PASS).
- **Q1 direct form: PASS for f16, q8_0, q8_0K/f16V, f16K/q8_0V; FAIL for q4_0 and bf16 at stage 1 and at stage 2 → PR-blocking for those two types under §15 ("blocked only if both stages fail"), goes to the owner.** Stage 1 (chunks 0-7): q4_0 2.690e-3 ± 3.46e-4 vs band 1.441e-3; bf16 1.802e-4 ± 1.94e-5 vs 1.032e-4. **Stage 2 (chunks 8-15 of the c8192 × 16 reference, computed from the recorded phase H arms `new_hb_*`): q4_0 2.082e-3 ± 3.15e-4 vs 1.441e-3: FAIL; bf16 1.250e-4 ± 1.09e-5 vs 1.032e-4: FAIL.** Explanatory arms (not gates, same logs): PR219 itself with ε = 1e-7 at layer 23 (where the band's saturation arm sits) gives q4_0 7.20e-4, bf16 5.84e-5 on chunks 8-15; at layer 3 (the first QSA layer) q4_0 1.720e-3, bf16 8.97e-5. The new build's direct-form KLD is 1.21× (q4_0) and 1.39× (bf16) the layer-3 ε arm. Reading (UNPROVEN as a cause): the band calibrates a perturbation at the last QSA layer, while the new kernels perturb every QSA layer from layer 3, and a 1e-7 perturbation at layer 3 already reaches 83 % (q4_0) / 72 % (bf16) of the new build's value; the e2e screen Q1-A for the same types passes with mean diffs of −1.5e-5 and −4.1e-5. Phase K (PR219 route-change arms) was not run.
- **Q2** (report): f16 ub2048 floor on W0-lab 1.0777e-4 (plan 1.116e-4); ratios 0.985-2.06 for f16 / q8_0 / bf16 / mixed arms, q4_0 ≈ 53 (its quantization floor).
- **Q3-A: PASS** f16 +1.44e-4 ± 3.5e-5, q8_0 +4.5e-6, q4_0 −1.07e-4 (band B_4096@32k 1.801e-4, AT-W12-BANDS). **Q3-B: PASS** f16 +2.7e-6, q8_0 +4.4e-6 (band 3.34e-4); **q4_0 ub256 not run** (phase P stopped: `w0_p_q4_0_u256` NO SUMMARY) → Moved to Prove.
- **Q4: PASS on the plan's per-type thresholds** (PLAN-v3 Q4 row; MEASUREMENTS.md:3738-3739 give B = 2.65e-3 for f16 KV f16 idx, 1.60e-3 for f16 KV q8 idx): f16 0.997356 ≥ 0.99735 (margin +6.3e-6), f16K q8 idx 0.996932 ≥ 0.99693 (+2.0e-6), q8_0 0.996737 ≥ 0.996625, q4_0 0.985712 ≥ 0.985528, bf16 0.997256 ≥ 0.99714, q8_0K/f16V 0.996857 ≥ 0.996742, f16K/q8_0V 0.997237 ≥ 0.997121. The harness had used one relative bound (1.178e-4) for every type, under which the two f16-KV rows fail (new − PR219 = −2.64e-3 and −1.60e-3); for f16 KV the PR219 arm is the reference build itself (agreement 1.000000), so the relative form compares new with the reference's self-agreement. **The two f16-KV margins (6e-6, 2e-6) are thin; reported for the owner.**
- **Q8: PASS** d1 f16 LCB95 +8.16e-4, q8_0 +8.04e-4 (> 7.45e-5); d2 8/8 chunks > 0, |new − PR219| 1.3e-4 / 3.4e-5 (≤ 1.86e-3); d3 report-only (−2.9e-8, −1.4e-6).
- **Q11a: PASS** (18/18 cells, lo / hi / all, 6 pairs). **Q11b: PASS** 6/6 ub512 arms (band 2.03e-4); ub16 report-only (band UNPROVEN, F4). **Q11c: not run** (phase S) → Moved to Prove.
- **Q12 (-nkvo, pinned, phase N at dcaa19f16, both arms on the copy route): PASS**: f16 ub4096 −2.8e-5 ± 3.4e-5, ub512 +1.7e-5, q8_0 ub4096 +3.5e-5, ub512 +1.2e-5 (bands 2.0e-4 / 2.03e-4); reruns new and PR219 0 per chunk. Coarse unpinned screen and `-ngl 0` (phase Z): not run → Moved to Prove.
- **Q5**: not re-run in W7; W7 changes no kernel source (see "What changed"), so W1-W6's op-local results stand for the same kernels.

**AT-W7PERF (HIP)** (S/w7_at_op.md, w7_at_e2e.md, at 0294c0274):
- **Shortcut grid and per-band flip: done, policy committed (0294c0274).** Op grid 15 pairs × n_q {1 … 1024} × n_kv {512, 1024, 2048}, G1 / G4 forced, 3 processes, noise p90 0.39 %: shortcut G = 1 to n_q 32, 4 above; prefer_dense dense in bands 2-8 and 9-32 for every pair and in 1 and 33-127 for most. E2E cells (pp512 / pp2048 d0, ub 64 / 512 / 1024 / 2048, tg128 d0 / 1024 / 1900, 5 pairs): 54 PASS, 1 unresolved (q8_0K/f16V tg d0 0.9475 [0.906, 0.985], a band the shipped table runs dense, `sparse-attn-attend-g1.cu` `dense[2][0][0] = 1`). **PASS.**
- **(b) tg-with-MTP cell: not run** → Moved to Prove (with P5's MTP cells).
- **(c) binding -nkvo gate** (S/w7_nkvo.md): fallback (a), plan + attend on the CPU (phase ND, 1 new process per type, LLAMA_QSA_HOST_KV_COPY=0): **FAIL for f16, q8_0, q4_0, bf16** (54 of 56 cells FAIL, new / W0-lab 0.48-0.95; one f16 d0 cell unresolved, one bf16 tg cell PASS). Fallback (b), the named host-KV copy route (phase NDB, 4 new processes interleaved with 4 W0, =1): **q4_0 PASS in 14/14 cells; f16 and q8_0 PASS in every cell at d4096 and d32768 and in every tg32 cell**; f16 d0 pp2048 ub512, pp512 ub4096, pp2048 ub4096 and q8_0 d0 pp512 ub512 are unresolved (point 0.991-0.998, CI contains 0.98; the §16 stopping rule asks for 2× processes, not run); new_q8_0 / new_f16 ≥ 0.94 under (b): 11 PASS, 3 unresolved (d0 pp512 ub512 0.966, d4096 pp512 ub512 0.957, d32768 pp2048 ub512 0.965), 0 FAIL. **bf16 under (b): not run.** By the pre-registered order the route of f16, q8_0, q4_0 and bf16 under -nkvo is now (b) (a2c37af52); other types keep (a) (no cell measured). **Verdict: (c) PASS for q4_0; f16 and q8_0 PASS except 4 unresolved d0 cells; bf16 UNPROVEN (its (a) FAILs, its (b) unmeasured).** The census condition holds: 0 GPU-placed plan / attend / shortcut nodes outside the named, logged route (AT-D4, AT-C18 above).
- **(d) shortcut pin on / off: not run** (NPIN). It now only matters for the types that keep fallback (a); the pin stays on for them (QWEN4EXP_SHORTCUT_PIN_DEFAULT) — UNPROVEN per §10 ("pinned only if (d) passes") → Moved to Prove; if (d) fails, the default flips to unpinned.

**Perf P1-P7:**
- **P1: PASS** 1.0274 [1.0227, 1.0325] (θ 0.995), q8_0 ub512 pp4096 d0, new vs base 13a279bfa, 4 base + 3 new processes (S/w7_perf_p1_d2.md).
- **P2, P3: not run** (P23: 2 of 48 processes completed, the 3rd incomplete) → Moved to Prove.
- **P4: not run**; **P6N: not run**; P6 (a) on the analysed phases: 0 eviction-free samples below 0.9 × cell median (perf_an totals line, S/w7_nkvo.md, w7_perf_p1_d2.md).
- **P5: not completed** (chain12: 12 of 42 processes completed, the 13th incomplete) → Moved to Prove. Indicative only (1 process per arm, no verdict): see `W7/perf/p5_*.json`.
- **P7**: kernel-level rows are W3/W4's (same kernels).

**AT-D1 (OVL)** (at 789a85c12, `W7/chain11.out`): W7 has 12 kind-4 regions in each of the 17 full graphs and 0 in the 1022 partial ones; **W0-lab has 0 kind-4 regions in all 1035 graphs**, against the plan's "expected 12" for W0. The criterion "W7's kind-4 per graph = W0's" is therefore not met as worded, while W7 meets the plan's expected value; why W0 forms no kind-4 region is UNPROVEN (`ggml-cuda.cu:7051-7056` needs Qcur_full within 16 nodes before Vcur, Kcur within 8 after and the attention within 48; not traced on W0). U16: 20/20 chunk hashes identical OVL=15 vs 11 for f16 and q8_0: PASS. The value part (D1V): not run → Moved to Prove. Owner to note the W0 premise.

**AT-D2 (`-ngl 0`)** (at 789a85c12): placement 100 % on the K buffer's device (CPU), guard 0 (passes 35, inspected 636 / 840, cpu_launch 0), PPL new / W0 f16 2.0835 / 2.0851, q8_0 2.0858 / 2.0866: PASS. **CPU perf: FAIL** (D2B, 3 + 3 interleaved processes, S/w7_perf_p1_d2.md): f16 pp512 @ d4096 0.713 [0.699, 0.733] vs threshold 0.9185 (1 − 2 × W0 spread 0.0408), tg32 0.888 vs 0.9917; q8_0 pp512 0.693 vs 0.985, tg32 0.892 vs 0.987. Q12 at -ngl 0 (phase Z): not run. Cause: see the investigation below.

**Memory** (S/w7_mem.md, at 789a85c12): **M1: 35/36 cells PASS, 1 FAIL**: iq4_nl c32768 ub4096 MTP ROCm0 +32.29 MiB over f16 (main −3.44). Cause found in code: HIP `GET_ROWS` takes IQ4_NL rows only when `ne[0] % 256 == 0` (`ggml-cuda.cu:9203-9209`) and the indexer key length is 128 (model kv 32, `W7/mem/srv_m1_iq4_nl_c32768_u4096.log`), so the iq4_nl indexer cache's gather does not run on ROCm0 (the 26-vs-2 split count was seen by the first W7 agent; not re-verified here, UNPROVEN). PLAN-v3:383 (§9, W11) maps the iq4_nl indexer to q8_0, which removes this case; M1 iq4_nl is re-checked at W11. Not a QSA attention fallback (census shows no CPU attention). np2 and default-server rows: q8_0 / q4_0 ≤ f16 + 0.27 MiB. **M2, M3: not run** (`H/m23.sh`) → Moved to Prove.

**Vulkan (AT-B)** (S/w7_vulkan.md, w7_vk_at_b4_op.md):
- **AT-B4 op level: done, policy committed (6e07acd3d).** 16 pairs × n_q {1 … 2048} × n_kv {512, 1024, 2048}, G 1, 3 processes, noise p90 1.25 %: sparse all-visible is 0.42-0.76 × dense in every band for f16K/q8_0V, f16K/q4_0V, q8_0/q8_0, q8_0K/q4_0V, q4_0K/q8_0V, q4_0/q4_0 → dense in all bands; q8_0K/f16V and q4_0K/f16V dense except 33-127 (1.02 / 1.06); bf16/bf16 dense in 1, 2-8, 9-32 (0.51-0.93), sparse from 33 (1.38-1.53); bf16-mixed pairs sparse (no Vulkan dense kernel). Before this the Vulkan proc returned dense for f16/f16 only, i.e. every quantized shortcut on Vulkan ran at 0.42-0.76 of dense. **The e2e half of AT-B4 (`vk.sh VE`) was not run** → Moved to Prove; the 0 CPU-node condition holds (census above).
- **AT-B5: M1 part PASS** (every T ≤ new f16 + 0.27 MiB on Vulkan0, main and MTP, c32768 / 131072 × ub 4096 / 512); the PR219 cliff is gone (W0 q8_0 at c131072 ub4096 6796.63 MiB main vs new 1504.04). **"new f16 ≤ W0 f16": FAIL in one cell, c131072 ub512 MTP: new 629.28 vs W0 617.28 MiB (+12.00)**; the main context there is −24.00, and every other f16 cell is ≤ W0 (0.00 to −256.00 MiB). Cause UNPROVEN (not attributed per tensor).
- **Q1-vk / AT-B6: PASS for all six pairs** (new-Vulkan(T) − W0-Vulkan(T), B_vk 3.214e-4): f16 +2.391e-4 ± 2.6e-5 (74 % of the band), q8_0 +7.5e-7, q4_0 −1.0e-4, bf16 −5.1e-6, q8_0K/f16V −2.1e-5, f16K/q8_0V −1.3e-5; rows_absent = moe_absent = 0 on every arm. Three first-pass arms were lost to vk::DeviceLostError under host memory pressure (kernel `Couldn't update BO_VA (-12)`) and re-run once, as the plan budgets.
- **AT-B1 (P8, binding; S/w7_vk_at_b1.md; 4 W0 f16 + 4 new processes per pair, round-robin, d4096 / d32768 / d131072 × ub 512 / 4096 × pp 512 / 2048, -r 5): FAIL for q8_0 and q4_0 → PR-blocking, to the owner with the ratios.** new q8_0 / new f16: 6 cells FAIL, 6 unresolved, 0 PASS; worst pp2048 ub4096 0.921 [0.919, 0.924] (d4096), 0.926 (d32768), 0.932 (d131072); pp2048 ub512 0.945-0.949 (FAIL). new q4_0 / new f16: 3 FAIL (pp2048 ub4096 0.916, 0.926, 0.928), 6 unresolved, 3 PASS. **bf16 / f16: 12/12 PASS (0.978-0.991); q8_0K/f16V: 12/12 PASS (0.972-0.989); f16K/q8_0V: 11 PASS, 1 unresolved (d4096 pp2048 ub4096 0.952 [0.949, 0.955]).** f16 new / W0: 11 PASS, 1 unresolved (d4096 pp512 ub512 0.997 [0.988, 1.007]). 0 eviction samples, 0 P6 (a) samples. The shortfall is on the pairs with both K and V quantized (q8_0/q8_0, q4_0/q4_0; both mixed pairs pass), in line with W6's AT-B3 op-level result (q8_0 attend 15.84 ms vs f16 7.07 ms per layer at C11's shape, IMPLEMENTATION.md W6) and its arithmetic risk note; the AT-B3 arms (vdirect / vstaged / vpack) were measured in W6, and the plan's next step is the owner's (vpack against M1, §17). The harness's last line (`echo VK_DONE`) did not run: vk.sh was edited during the run (bash read the shifted tail, `chainV.out`: "line 124: syntax error near unexpected token done" after "PHASE V1M done"); every phase had finished.

### The -nkvo slowdown of fallback (a): cause (from code and the recorded traces, no new campaign)

Observed (phase ND, S/w7_nkvo.md; `W7/perf/nk_*_pp_*.json` avg_ns): with plan + attend pinned to the CPU, prompt processing past the shortcut runs at 0.48-0.71 of W0-lab and tg32 at 0.79-1.03; d0 (all shortcut) cells at 0.51-0.98.
- **Where the work runs.** Under fallback (a) all 12 QSA layers' `SPARSE_ATTN_PLAN` and `SPARSE_ATTN` run on the CPU backend (`W7/nk1/q8_0_new.log`: `QSA_CENSUS cpu:sparse_attn 204`, `cpu:sparse_attn_plan 204`; every trace line `backend=CPU`), and so does the shortcut attention (f16: `route=dense backend=CPU` — the CPU's default `prefer_dense` is dense only for f16 / f16; q8_0: `producer=av route=sparse backend=CPU n_sel=n_kv`). W0 (PR219) leaves the attention on ROCm0 and lets the scheduler copy the host K / V (R1-NKVO: 300/300 FA on ROCm0; same copy as route (b)).
- **The cost is the CPU attend, flat over depth.** one new f16 process takes 2.873 s per 512 tokens (ub512 pp512) at d4096 and 2.832 s at d32768, one W0 process 1.503 / 1.569 s (`nk_f16_pp_1_new.json`, `nk_f16_pp_0_w0.json` avg_ns); per 512-token ubatch the CPU route adds 1.35 s (d4096) and 1.25 s (d32768) for f16, 1.43 / 1.23 s for q8_0 (from the ND table means: 512 / t/s new − 512 / t/s W0). The CPU op costs 44.64 ms per layer at n_q 127 × 2048 selected keys and 23.26 ms at n_q 64 with 8 threads (`W7/at_op/cpu.out`), i.e. 0.34 ms per query and layer; for 512 queries × 12 layers that is 2.1 s at 8 threads, about 1.05 s at the bench's 16 threads if it scaled linearly (scaling UNPROVEN). Arithmetic, consistent with the measured extra 1.23-1.43 s: **the CPU sparse attend over 2051 keys is the cause** (it does not grow with depth, which is why new / W0 rises from 0.53 at d4096 to 0.56 at d32768 while W0's copy grows). q8_0 / f16 at the op: 52.79 / 44.64 ms = 1.18 (same file), which matches q8_0's larger shortfall.
- **Why that op is slow.** `ggml/src/ggml-cpu/sparse-attn.cpp:217-270`: each K / V row is converted value by value to float and rounded through f16 (`sa_load_row`), and the scores accumulate sequentially over d in scalar float (`sa_scores`), the bit-exact contract of W1; PR219's CPU path is ggml's vectorized FLASH_ATTN_EXT with a selected-key gather (`~/dev/wt_lab/w0lab12/ggml/src/ggml-cpu/ops.cpp:8946-8988`). The same op is what AT-D2 (`-ngl 0`) measures: 0.69-0.71 at pp512 d4096. Not profiled per instruction (UNPROVEN beyond the per-layer timing).
- **tg.** f16 tg32: +16.3 ms per token at d4096 (W0 71.4 → new 87.7 ms), +4.3 ms at d32768 (84.9 → 89.2): the CPU attend at n_q 1 is 0.5-1.0 ms per layer (n_q 2: 1.036 ms, `at_op/cpu.out`), 6-12 ms over 12 layers; W0's per-token copy grows with depth (+13.5 ms from d4096 to d32768), the new route's does not (+1.5 ms). The remainder at d4096 (about 4-10 ms) is attributed to the 24 extra GPU↔CPU split transitions per token (UNPROVEN, not traced).
- **d0.** All shortcut: q8_0 runs the all-visible sparse op on the CPU with n_sel = n_kv (trace above), whose cost grows with n_kv, hence pp2048 d0 q8_0 0.62 against f16 (CPU dense FA) 0.89.
- **Consequence (committed, a2c37af52):** fallback (b) puts plan, attend and shortcut on the layer's GPU over the copied K / V, as PR219 does; NDB measures it at or above W0-lab. A faster CPU op (vectorized dequant-once, §3) would be needed for `-ngl 0` (AT-D2) and for the types still on fallback (a); not done in W7.

### Owner decisions / open FAILs (with their numbers)

1. **Q1 direct form q4_0 and bf16 fail both stages** (2.690e-3 / 2.082e-3 vs 1.441e-3; 1.802e-4 / 1.250e-4 vs 1.032e-4); explanatory PR219 ε arms above. PR-blocking for these two types by §15 unless the owner rules on the band.
2. **AT-D2 `-ngl 0` CPU perf FAIL** (0.69-0.71 pp512, 0.89 tg32 at d4096 vs thresholds 0.918-0.992); cause above (CPU op).
3. **AT-D1 premise**: W0 has 0 kind-4 regions (plan expected 12).
4. **Q4 f16-KV margins** 6e-6 and 2e-6 over the plan's thresholds.
5. **M1 iq4_nl MTP +32.29 MiB** (indexer GET_ROWS, fixed by W11's mapping).
6. **AT-B5 Vulkan MTP f16 c131072 ub512 +12.00 MiB over W0** (main −24.00).
7. **AT-B1 Vulkan q8_0 and q4_0 below 0.95 × new f16** (worst 0.921 / 0.916, pp2048 ub4096 d4096); PR-blocking by P8 / P2-11 unless the owner rules (options in §17: vpack against M1).
8. **-nkvo bf16** ships on the copy route without a measured (b) cell (its (a) failed); the generic types keep the CPU route with no measured cell.

### Moved to Prove (campaigns not completed in W7; harness command, criterion)

All commands from `~/dev/wt_rewrite` on strix-002, every GPU run under flock (the harnesses take it); analysis `python3 H/perf_an.py W7/perf <PHASE>` or `python3 H/q_an.py W7/q/logs`.
| item | command | criterion |
|---|---|---|
| Q3-B q4_0 ub256 (phase P remainder) | `H/q_campaign.sh P` (records ref32x8 again; the old one is deleted) | Q3-B q4_0 mean diff ≤ 3.34e-4 (and Q3-A/B re-scored on the same reference) |
| Q11c (phase S) | `H/q_campaign.sh S` | np2 split = np1 per sequence, KLD 0 per chunk, q8_0 and q4_0; plus the split-stream decode case at n_tps 1 |
| Q12 at `-ngl 0` (phase Z), Q12 coarse unpinned screen | `H/q_campaign.sh Z`; coarse: unpinned `-nkvo` vs the same binary's GPU route, c4096 × 4 | pinned rule ≤ B_U; coarse f16 ≤ 0.01587, q8_0 ≤ 0.01563 |
| P2 / P3 (P23) | `H/perf.sh P23` then `perf_an.py W7/perf P2 P3` | P2 ≥ 0.99 every C10 cell; P3 ratio of ratios ≥ 0.99 and new_q8 / new_f16 ≥ 0.90 |
| NDB remainder: bf16 and the unresolved d0 / cross-type cells | `NKTYPES=bf16 H/perf.sh NDB`; `NKTYPES="f16 q8_0" PROCS="8 9 10 11 12 13 14 15" H/perf.sh NDB` (needs the matching W0 processes of the stopping rule) | new / W0-lab ≥ 0.98 every pp512 / pp2048 cell at d0 / d4096 / d32768 and tg32 at d4096 / d32768; new_q8 / new_f16 ≥ 0.94 |
| NPIN, AT-W7PERF (d) | `H/perf.sh NPIN` | pin-on / pin-off ≥ 0.98, else QWEN4EXP_SHORTCUT_PIN_DEFAULT = false |
| P5 remainder (+ MTP cells, AT-W7PERF (b)) | `H/perf.sh P5`; `H/mtp_perf.sh` | shortcut tg new / W0 ≥ 0.98 per S_fast pair (else prefer_dense flip); depth f16 / q8_0 ≥ 0.99; new_T / new_f16 ≥ 0.94; MTP new / W0 ≥ 0.98 and new_q8 / new_f16 ≥ 0.94 |
| P4 | `H/perf.sh P4` | new_T / new_f16 ≥ 0.90 (S_generic below it ships, listed) |
| P6N | `H/perf.sh P6N` | reported only (eviction-sample counts and rate ratio) |
| AT-D1 value (D1V) | `H/perf.sh D1V`; `perf_an.py W7/perf D1V` | (W7 OVL15/OVL11) / (W0 OVL15/OVL11) CI contains 1 or lies above |
| M2, M3 | `H/m23.sh` | M2: new compute + pool high-water ≤ W0's + 1 MiB per device, main / MTP, f16 / q8_0, c65536 / 131072 ub4096; M3: 0 hipMalloc after warm-up |
| AT-B4 e2e (Vulkan) | `scripts/qsa-validation/harness/w7/vk.sh VE` | per (pair, n_q band) sparse ≥ 0.98 × dense in the e2e cells, else flip to dense in `ggml_backend_vk_sparse_attn_prefer_dense`; decode at depth ≥ W0 of the same type |
| AT-B1 unresolved cells (f16 d4096 pp512 ub512; f16K/q8_0V d4096 pp2048 ub4096) | `vk.sh V1M` with process indices 4-7 added (8 per arm, the stopping rule) | f16 new / W0 ≥ 0.99; new_T / new f16 ≥ 0.95 |
| U30 masks on Vulkan | `H/allvis_rec.sh` (its Vulkan branch runs `build-vk/bin/test-qsa-allvis-rows`) | 0 mismatches |
| Q1 phase K (explanatory) | `H/q_campaign.sh K` | none (explains the direct-form failure) |

### Cleanup

- Deleted (each > 1 GB, after its campaign): `W7/q/refs/ref32x8` (40 GB, phase P reference; Q3-B q4_0 ub256 re-records it), `W7/q/refs/ref8t0` (22 GB, after phase N), `W7/vk/refs/vref8t0` (22 GB, deleted by VQ, and its hard-linked copy `W7/vk/refs_keep_vref8t0` after the Q1-vk rerun); the U30 mask dumps (`*.bin`, by allvis_rec.sh). `find W7 -size +1G` is empty; `W7` holds 1.7 GB (build trees, logs).
- Not touched: `~/dev/wt_pr219v2*`, other agents' worktrees, systemd units, strix-001. A `sleep 597` shell of the previous W7 agent (pid 4029405) was left alone (not started by this agent).

## W8: TE rows for non-compact ubatches; masked route deleted

**Status: implemented and committed (not pushed). Both builds pass. The CPU unit test (TE ∩ visible equality) passes, and so do the per-op subsets, the guard tests and AT-C16a. Q10 and Q6 were not run: they need the AT-Q10Q6 calibration campaign, which the owner's scope note moves to Prove (see "Moved to Prove").** Every command ran on strix-002, every build and GPU run under `flock ~/gpu.lock`, and this agent was the only one active. Logs: `~/dev/wt_rewrite_logs/W8/` (builds, op tests, unit tests) and `~/dev/wt_lab/w8/` (AT-C16a, guard, smoke runs). All of it is 6.4 MB, with no file over 100 MB.

**Done-when scorecard**
| done-when item (PLAN-v3:560-563) | result |
|---|---|
| CPU unit test: new rows' visible set = PR219's TE ∩ visible for every row, on the r1_oracle layouts (seq_cp prefix L 8192-8195, 2-slot image) | **PASS** on CPU, and also on ROCm0 and Vulkan0. 2 mutants killed |
| Q10 (after AT-Q10Q6) | not run → Moved to Prove (the AT-Q10Q6 calibration does not exist yet: no AT-Q10Q6 / floor_ctl entry in IMPLEMENTATION.md or MEASUREMENTS.md, `grep -n "Q10Q6\|floor_ctl"` empty) |
| Q6 | not run → Moved to Prove (needs τ_T from AT-Q10Q6 (3)). Partial content check from the AT-C16a runs: B65t answers `12299`, which is Q6's expected B65t answer |
| AT-C16a (layout 3, 8-strip slot-B ubatches): bias new > 0, kq_mask new > 0, reuse 88/src/graph; main ROCm0 runtime peak ≤ 8054.08 MiB | **PASS**: bias 8 / 88 and kq_mask 8 / 88 on all three slot-B text graphs; peak 2396.59 MiB, r1 = r2 |

### What changed (files, functions)

- `src/models/qwen4exp-te.h` (new). It holds two functions:
  - `qwen4exp_te_expand(score, cell_blk, add)` carries the base's expanded construction op for op (get_rows of the permuted score by cell_blk, cont-permute back, + mask or per-cell bias). `git blame` attributes these lines to Piotr Wilkin 8026a5cd06 and Daniel Han 6c84c7d5d8, so they are base code and not #219's.
  - `qwen4exp_te_select(expanded, width)` returns `top_k = cont(top_k(expanded, width))` as [width, n_query, 1, n_stream]. It then builds the **TE rows**: one GET_ROWS of expanded, viewed as [1, n_kv, n_query, n_stream], at the picked indices; `step(x + FLT_MAX)` (1 for any finite value, 0 for −inf or NaN); then `(idx + 1)·visible − 1` and a cast to I32. The CB rows use the same op family (`qwen4exp_select_complete_blocks`).
  - Both functions are shared with the test, so the test runs the production ops.
- `src/models/qwen4exp.cpp`:
  - `build_qsa_top_k`: the non-compact strip calls the two helpers and appends `sel.rows` to the strips. The joined rows are named `qsa_rows_te-<il>` (PLAN-v3 §4.5; the old name was `qsa_rows_tk`), and the per-strip top_k keeps the name `indexer_top_k`. A new `GGML_ASSERT(score_blocks == n_blocks)` (:1677) holds because `qwen4exp_score_key_limits` returns no bound for a non-compact ubatch (`!compact` → `{}`, :1161-1162).
  - `build_layer_attn`: every non-shortcut ubatch now builds a plan. The producer is `cb` when the input set is maskless and `te` otherwise. The `"masked"` producer is gone.
  - `build_attn_qsa`: the masked half is deleted for every backend (D22). That half was fill(−inf) + set_rows of zeros at top_k + kq_mask, then `build_attn_mha(..., top_k)`, i.e. FLASH_ATTN_EXT + src[5]. The function now asserts a SPARSE_ATTN_PLAN and takes no `top_k` parameter (`models.h` declaration updated).
  - Comments that named the masked route were updated. strip_views is unchanged (`git show 79308ac04 -- src/models/qwen4exp.cpp` does not touch the `strip_view` lambda).
- `src/llama-context.cpp`: `qsa_guard` no longer exempts a `"masked"` producer (both places), so TE nodes go through the full placement and plan checks. The `LLAMA_QSA_TRACE` route string lost its `masked` case. `src/llama-graph.h`: producer comment updated.
- `tests/test-qsa-te-rows.cpp` (new, `llama_build_and_test`, so ctest runs it on CPU builds).
  - It is a C++ port of `EV/r1/r1_oracle.py` (layouts, the grouping by (position block, exact sequence set), the blk_bias rules, the kq_mask visibility rule incl. `is_2d_gt`).
  - Cases: text L ∈ {8192, 8193, 8194, 8195} with a seq_cp prefix and 40 own cells per sequence (1 strip, and 3 strips of 32); the two image layouts of the oracle (image at 28930 and at 32779, sequence 0 holding 2000 text cells, and 3 sequence-0 queries whose 2000 visible cells are fewer than 2051, so top_k must fill those rows with hidden cells); split streams (n_stream 2).
  - In one graph, for each backend of the build, it compares the new rows with **PR219's masked-route ops copied verbatim** (fill / set_rows / add), evaluated on the same top_k output.
  - Pass rule: the cell set of the rows equals {j : PR219 mask = 0} for every row; no duplicates; every listed cell was picked by top_k; a host cross-check {top_k ∩ kq_mask = 0} equals PR219's set; and no visible cell has a −inf block bias.

**Why a gather of expanded equals PR219's TE ∩ visible.** PR219 attended top_k(expanded) ∩ {kq_mask = 0}. The two sets can differ only on a cell that is mask-visible while its block bias is −inf. The bias is −inf only for a block whose representative cell lacks the query's sequence, or for a block past the complete ones (`llama-memory-hybrid-idx.cpp:932-935`). Cells are grouped by exact sequence set (`seq_get_all(grp_first) == seq_get_all(j)`, :686), or there is only one sequence (`one_seq`), and every cell outside a complete group maps to the dead block, which has bias 1e9. So a visible cell always has a finite bias. The test asserts this on its layouts (`visible_with_ninf_bias=0` on every case). Gathering the mask instead would give the same set at the cost of a second read of the mask, so the plan's form (§4.1 row 4) is kept.

### Commits (branch qsa-rewrite, not pushed)

- **79308ac04** `qwen4exp : TE rows for non-compact ubatches on GGML_OP_SPARSE_ATTN; the masked route is deleted` (7 files, +523 −107).
- **a6912ec39** `scripts : W8 AT-C16a harness (R1-C layout 3 server run) and the C19 logging lab patch`. It adds `scripts/qsa-validation/harness/w8/at_c16a.sh` and `parse_c19.py` (a copy of `EV/r1c/parse2.py`), plus `scripts/qsa-validation/patches/05-c19-log.patch`. That patch is the C19 lab logging (`~/dev/wt_lab/c19/c19.patch`, the llama-context and strip_view hunks without LAB_C19_NO_SHARE) rebased on 79308ac04. It is measurement-only and is not part of the product; `git apply --check` passes on HEAD.
- Author and committer on both: Gaetan Puleo <12990773+gaetan-puleo@users.noreply.github.com>. **No Co-authored-by trailer.** No #219 code is ported or adapted:
  - the expanded / top_k scorer and the deleted masked route are base code (`git show ea5285ddf:src/models/qwen4exp.cpp` has `indexer_score_tokens` at :1659 and `ggml_top_k(ctx0, expanded` at :1664);
  - #219's strip_views is used unchanged.

### Build

- **HIP** `build/` (rule flags, incremental `cmake --build build -j 16`): rc 0 with 0 errors (`hip_build1.log`; the only line matching "error" is the non-fatal `-- UI: download dist.tar.gz.sha256 ... failed`).
  - For AT-C16a, `05-c19-log.patch` was applied to the working tree, only `llama-server` was rebuilt (`hip_build_lab.log`, rc 0), and the patch was then reverted with `git apply -R`.
  - Final rebuild after the revert: rc 0, 0 `warning:` lines (`hip_build_final.log`). `git status --short` is empty, so `build/` is current for HEAD a6912ec39 (the second commit is scripts only).
- **Vulkan** `build-vk/` (incremental): rc 0, 0 `warning:` lines (`vk_build1.log`), built from the 79308ac04 content before the lab patch. W8 changes no Vulkan source, but the graph change runs on Vulkan, so the new test was also run there.

### Tests run (correctness; outputs)

- **CPU unit test** `build/bin/test-qsa-te-rows CPU` (`te_rows_cpu.log`): **RESULT PASS**, 11/11 cases. Every row has mismatch 0, host_mismatch 0, dup 0, not_picked 0 and visible_with_ninf_bias 0, and every shape is [2051, n_tps, 1, n_stream]. Details:
  - text cases: 80 rows each (160 for the split2 case);
  - image_pos28930: 792 rows; image_pos32779 (4 strips): 874 rows;
  - each image layout has 3 underfull rows, from which 156 hidden cells are dropped (= (2051 − 2000) + (2051 − 1999) + (2051 − 1998)).
  - Run time 2.8 s.
- **Same test on every backend** (`build/bin/test-qsa-te-rows`, `te_rows_all_hip.log`: ROCm0 and CPU; `build-vk/bin/test-qsa-te-rows Vulkan`, `te_rows_vk.log`: Vulkan0): RESULT PASS, the same counts on all three backends.
- **Mutants** (`scratchpad/w8/mutants.out`; header restored afterwards, md5 equal):
  - M1, rows = top_k (no −inf filter): FAIL, mismatch 3 on both image cases, i.e. exactly the underfull rows.
  - M3, index without the +1: FAIL in all 11 cases (e.g. L8193: mismatch 80, not_picked 30784).
  - The test therefore cannot pass trivially.
- **Per-op test-backend-ops**, one op per run, `-j 1`, under flock (`~/dev/wt_rewrite_logs/W8/optests/`). Every run has rc 0 and 0 FAIL:

  | backend | op | tests passed |
  |---|---|---|
  | ROCm0 | SPARSE_ATTN_PLAN (`--fail-unsupported`) | 185/185 |
  | CPU | SPARSE_ATTN_PLAN (`--fail-unsupported`) | 185/185 |
  | ROCm0 | GET_ROWS | 591/591 |
  | ROCm0 | TOP_K | 587/587 |
  | ROCm0 | STEP | 12/12 |
  | ROCm0 | SCALE | 7/7 |
  | Vulkan0 | SPARSE_ATTN_PLAN (`--fail-unsupported`) | 92/92 |
  | Vulkan0 | GET_ROWS | 647/647 |
  | Vulkan0 | TOP_K | 587/587 |

  The plan op's row patterns include permuted rows, −1 and duplicates at n_sel 2051 (`SA_ROWS_MIX`, `SA_ROWS_NEG`, test-backend-ops.cpp:13232-13245, :14424-14430), which is the TE row shape. SPARSE_ATTN was not re-run: W8 changes no kernel, and W7's runs stand (IMPLEMENTATION.md W7).
- **Guard tests** on the final build (`~/dev/wt_lab/w8/guard/`; W8 removed the guard's masked exemption):
  - `test-qsa-guard` pass q8_0: PASS; pass f16: PASS.
  - fake device: PASS (init error "QSA layer 3 (cb): ... not supported on device FAKE0 (backend FAKE): no sparse-attention kernel in this backend; ...").
  - reject: PASS (init error, "rejected by the test hook GGML_CUDA_SA_REJECT").
  - reject-decode: PASS (`process_ubatch: QSA layer 3 (av): ...`).
  - `test-qsa-allvis-rows`: PASS.
- **Smoke run of the real model, compact routes unchanged** (`~/dev/wt_lab/w8/smoke_ppl_q8_0_u{4096,512}.log`). Command: the W7 census recipe, `llama-perplexity -c 8192 -b 4096 -ub {4096,512} --chunks 1 -ngl 99 -fa on -lm mmap -lzm on-direct -ctk q8_0 -ctv q8_0` with `LLAMA_QSA_TRACE=1 GGML_CUDA_DISABLE_GRAPHS=1`.
  - PPL 2.1139 ± 0.05827 at both ubs, equal to W7's census logs (`~/dev/wt_lab/w7/census/ppl_q8_0_q8_0_u{4096,512}.log`).
  - The sorted `LLAMA_QSA_TRACE` lines are identical to W7's (`diff` empty).
- **AT-C16a** (`H8/at_c16a.sh w8_share_r1` and `w8_share_r2`, where H8 = `scripts/qsa-validation/harness/w8`; outputs `~/dev/wt_lab/w8/w8_c16a_r{1,2}.out`, logs `~/dev/wt_lab/w8/logs/srv_w8_share_r{1,2}.log`, parsed by `parse_c19.py` into `parse_w8_share_r{1,2}.out`). The binary is HEAD 79308ac04 + 05-c19-log.patch, run with the R1-C layout 3 recipe (`EV/r1c/srv3.sh` flags, default 4-slot unified server, c131072 ub4096 q8_0).
  - **Strip-view census on the slot-B text graphs** (main): n_kv 63744 / n_tps 4088, 67584 / 3833 and 71680 / 4088, each with 8 strips. Every one has bias new 8 / reuse 88 and kq_mask new 8 / reuse 88, with new MiB bias 248.51 / 247.05 / 279.45 and kq_mask 497.03 / 494.10 / 558.91. These are identical to PR219's R1-C layout 3 table (MEASUREMENTS.md:4130-4132) and to a re-parse of PR219's log `~/dev/wt_lab/r1c/logs/srv_b3_share_r1.log` with the same parser. **Pass** (bias > 0, kq_mask > 0, reuse 88 = 8 × 11).
  - **Runtime peak** (`C19 buf` max per context and backend). Criterion ≤ 8054.08 MiB: **PASS**.

    | context, backend | new peak (MiB) | PR219 share peak, same parser (MiB) |
    |---|---|---|
    | main, ROCm0 | **2396.59** | 3546.90 |
    | MTP, ROCm0 | 1677.70 | 2268.59 |
    | host (CPU buffer), main | 1039.27 | 1039.27 |
    | host (CPU buffer), MTP | 1038.96 | 1038.96 |

    Reserve: 1609.88 main / 1465.23 MTP ROCm0, equal to PR219. The 1150.31 MiB of main ROCm0 saved fits the deletion of the masked route's per-layer full-width mask copies (fill / set_rows / add over [n_kv, n_tps]); that attribution per tensor is UNPROVEN.
  - **Route census** (`LLAMA_QSA_TRACE`, 453 trace lines in r1):
    - producers: cb 222, te 218, dense 13; **masked 0** (`grep -c producer=masked` = 0 in both runs); every line `backend=ROCm0`.
    - `te` lines: the slot-B text ubatches (n_q 4088 / 3833 / 4088, 12 main layers each, G4), the two image ubatches (n_q 1024 at n_kv 58624 / 59648), and the decode / draft ubatches of slot A after slot B got cells (n_q 1 to 4).
    - Guard: `LLAMA_QSA_GUARD ctx=main passes=40 inspected=948 cpu_launch=0`, `ctx=mtp passes=637 cpu_launch=0`; no QSA error in the log.
  - **Answers**: A `'12299. the quick brown fox jumps over the lazy dog number 12'`, TI12k `'40479'`, in both runs (the PR219 answers, `EV/r1c/runB3.out`).
  - **Determinism of the census and buffers**: r1 = r2 for every parsed line except the context pointers (`diff` of the parse outputs).
  - Indicative only (one process per run, not a gate): TI12k prompt 728.19 / 723.87 t/s against PR219's 636.31 / 679.01 (`runB3.out`).

### Moved to Prove (owner scope note 2026-11-07; harness / command and pass criterion)

| item | command | criterion |
|---|---|---|
| AT-Q10Q6 (1), Q10 control floor | PLAN-v3:810. Lab tree `~/dev/wt_lab/r1q10/tree` = W0-lab (`~/dev/wt_lab/w0lab12`) + the build4 lab patches (`LAB_QSA_C4`, `tools/mtmd/lab-r1.cpp` = `EV/r1/lab-r1.cpp`) with `#define QSA_DECODE_MAX_QUERIES 127` in `ggml/src/ggml-cuda/qsa-decode.cuh`. HIP flags, targets lab-r1 and lab-c4. Then `r1/text_campaign.sh` (`LAB_QSA_C4=0`): build4 × 3 and the 127 build × 3 | 127 vs 127 KLD = 0; build4 vs build4 = 0; floor_ctl = mean + 3·SD over the 3 × 3 cross pairs (PLAN-v3:814) |
| AT-Q10Q6 (2), Q10 cell τ | `r1/img_campaign.sh` and `r1/text_campaign.sh`: ≥ 4 more PR219 repeats per arm at ub8192 plus the ub4096 pair, for L 8193 / 8194 / 8195 and image p3 (PLAN-v3:811) | ≥ 6 draws per cell; τ = mean + 3·SD if its leave-one-out false-fail rate ≤ 1/6, else the ≥ 19-draw quantile (PLAN-v3:814) |
| AT-Q10Q6 (3), Q6 τ_T | `r1e/q6.sh`, `r1e/q6q8.sh` on lab-c4: ub 6144 / 3072 / 1536 added to {8192, 4096, 2048, 1024} for p1 / p2 f16 and p2 q8_0; the p1 q8_0 matrix (TE and CB × ≥ 4 ubs) (PLAN-v3:812) | ≥ 6 draws per (prompt, type); τ_p1,q8_0 filled (PLAN-v3:814) |
| Q10 on the W8 head | lab-r1 built on qsa-rewrite 79308ac04 or later. LAB_QSA_C4 selects nothing there, because non-compact ubatches always take TE. Then `r1/text_campaign.sh` and `r1/img_campaign.sh` arms; per-request KLD over the 2 × 100 fixed tokens (`r1/kld2.py`) | control L = 4k: new rerun vs new = 0, and new vs each PR219 repeat ≤ floor_ctl. Cells L 4k+1 / +2 / +3 and image p3: new vs the median of ≥ 3 PR219 repeats ≤ τ. Free answer correct on headline, newspaper and date (PLAN-v3:727, :815). Census: `producer:te` nonzero, `producer:masked` and `cpu:` zero (census_check.py 'Q10' row) |
| Q6 on the W8 head | `r1e/q6.sh` / `r1e/q6q8.sh` with lab-c4 built on the W8 head, C4 p1 / p2 at ub8192 | KLD(new_T ‖ PR219_T TE) ≤ τ_T; content check; `~/repro_img` A / B / C answers 2749 / 12749 / 26999; B65t 12299 (PLAN-v3:723, :815). (B65t already answers 12299 in both AT-C16a runs above) |

### Notes

- No new item for `TODO-LATER.md`: no W8 check failed.
- Not touched: `~/dev/wt_pr219v2*`, the W7 lab tree `~/dev/wt_lab/w7/lab`, other agents' worktrees, systemd units, strix-001. No process this agent did not start was killed: the AT-C16a servers were started by the harness and stopped by it with SIGINT.
- Cleanup: no file over 1 GB was created (`find ~/dev/wt_lab/w8 ~/dev/wt_rewrite_logs/W8 -size +100M` is empty). The `/tmp/te_h.bak` used by the mutant runs on strix-002 was deleted.

## W9: one-sequence compact predicate, non-compact re-split, reserve variants

**Status: implemented and committed (not pushed). HIP and Vulkan builds pass with 0 warnings. Every correctness test run here passes: the visibility test with its negatives and the agreement assert, test-qsa-resplit (i)–(vi) with 0 violations, the guard tests, and the strict-reserve runs (0 violations on every run). The concurrent 2-slot runs show n_qsa_resplit > 0 in both contexts. AT-C18 at `-np auto` passes 32/32. The M5 cell measured here (c131072 ub4096) is within its bound. AT-C16b's criterion cannot be met as written: with S = 1 the strip-view path never runs (see "AT-C16b" below). AT-C12's grid, AT-C17 and the rest of M5 are moved to Prove.**

All commands ran on strix-002, and every build and GPU job ran under `flock ~/gpu.lock`.
- Logs: `~/dev/wt_rewrite_logs/W9/` (builds and unit tests) and `~/dev/wt_lab/w9/` (strict, guard, srv, c18). The layout 3 run is in `~/dev/wt_lab/w8/logs/srv_w9_l3_r1.log`.
- Chain outputs: `~/dev/wt_lab/w9/chain{1,2,3}.out`.
- An earlier W9 agent wrote the code and ran `chain1`/`chain2` before a login expiry interrupted it. This agent reviewed the uncommitted diff and made one change: it moved `#include <functional>` in `src/llama-kv-cache.h` into the system-include block. It then re-ran the build and the unit tests on that tree (`chain3`), added the missing runs (`-np 1` image smoke, R1-C layout 3, Vulkan build and tests), and committed.

**Done-when scorecard (PLAN-v3:566-576)**
| done-when item | result |
|---|---|
| visibility test on recorded, synthetic and negative layouts, with the scan/predicate agreement assert | **PASS**: 53 ubatches, 17556 rows compared with the KQ mask, 44 agreement checks, 0 failures (`ut_test-qsa-visibility.log`, `ut_resume_*.log`, `ut_vk_*.log`). 3 mutants killed |
| test-qsa-resplit (i)–(vi), 0 violations, fixed seeds | **PASS**: 320 batches, 2738 ubatches, 2316 graph-time non-compact = 2316 NC0, 1863 replays. `violations: refused 0, (i) 0, (ii) 0, (iii) 0, (iv) 0, (v) 0, (vi) 0` (`ut_test-qsa-resplit.log`). Mutant "hook always true" is killed: (ii) 288, FAILED |
| re-split in both contexts: (1) unified 2-slot, A image + B ≥ 4096-token prompts at ub4096; (2) the same with `-np 2` split + MTP | (1) **PASS**, q8_0 and f16: main n_qsa_resplit 82, MTP 81, strict violations 0, decode errors 0, TE n_q max 512, answers 11999 / 32517 (`chain2.out`, `srv/srv_uni_{q8,f16}_u4096.log`). (2) split + MTP: 0 violations, 0 decode errors, answers correct, but **n_qsa_resplit = 0 in both contexts by construction**. Each split stream holds one sequence, so no ubatch is non-compact (te_lines 0, `srv_split_q8_u4096.log`); test-qsa-resplit layout d gives the same 0. The criterion "> 0 in both contexts" holds for the MTP sidecar in the unified run (81) |
| one-stream non-compact ubatch in a split cache under STRICT (U38) | **PASS**: `LLAMA_QSA_FORCE_NONCOMPACT=1`, `--parallel 2` split, T_deep alone. 1639 trace lines all `n_stream=1`, n_qsa_resplit 108 / 108, TE n_q max 512, strict violations 0 in both contexts, answer 32517 (`srv_split1s_forced_q8_u4096.log`). The [K, T/s, s] reserve covers [K, T, 1] at q8_0 c131072 ub4096, so the §7.1.4 fallback is not needed |
| re-split counter > 0 on R1-C layout 3 and on the R1-RES workload; 0 graph-build errors on the image matrix | layout 3: **n_qsa_resplit 22 main / 22 MTP**, 0 decode errors, 0 strict violations, answers `12299…` / `40479` (`srv_w9_l3_r1.log`). R1-RES (`r1s` recipe, ub1024, S0_img + T_deep): unified 45 main / 46 MTP; split 0 / 0 by construction; 0 decode errors on both. Requests A/B/C of `~/repro_img` were not run (moved to Prove) |
| B65t, C15 deep-image D at ub1024, request C | B65t: answer `12299…`, 0 strict violations, 0 decode errors (layout 3 run above). C15 D and request C: moved to Prove (deep context) |
| llama-perplexity c8192 ub512 f16/q8_0; llama-bench pp4096 @ d4096 trim 1/0 | **PASS** (strict 0 violations, 0 decode errors). PPL f16 1.9419 (ub512) / 1.9485 (ub4096); q8_0 1.9530 / 1.9530. Every value is within one chunk SE (±0.034) of W12's C15 references (f16 1.9449, q8_0 1.9519, IMPLEMENTATION.md:89). bench: trim 1 logs the "attention runs query rows" line once, trim 0 never, strict 0 on both types. The "np2 shared prefix with trim on/off" cell was not run (moved to Prove) |
| M4 recipes (a) `r1s` split then unified; (b) `-np 1` request C; (c) single sequence, two images back to back across a ubatch boundary at depth > 32k | (a) **PASS**: split 0 violations and 0 non-compact split-stream ubatches (te_lines 0, so the split-stream variant may be dropped); unified 0 violations (`srv_r1s_{split,uni}_q8_u1024.log`). (b) moved to Prove. Smoke instead: `-np 1` with S0_img (an image + 12k lines) at q8_0 ub4096 and f16 ub1024 gives 0 decode errors, te_lines 0 (every image and text ubatch compact), strict 0, answer 11999 (`srv_np1_*_img.log`). (c) CPU only: the visibility test's request-C layout has two images back to back across a 512-token boundary, and its B65t layout has an image past 32k. The GPU recipe is moved to Prove |
| AT-C12 | U38 part done (above). The image-slot layout part, run on the r1s default-server recipe, was not run as specified (4 slots + a second ≥ 8k turn + Q10-style KLD vs PR219): moved to Prove. Strict and re-split on the 2-slot version are above |
| AT-C16b | **criterion not applicable as written**, see below. Layout 3: runtime peak = reserve (1609.88 / 1465.23 MiB), strict 0 |
| AT-C17 | moved to Prove (init at -c 524288 / 1048576 is a deep-context run) |
| M5 byte bounds | the measured cell is within bound (below); the rest of the grid is moved to Prove |
| AT-C18 (`-np auto`) | **PASS** 32/32: f16/q8_0 × ub 512/4096 × ±`-nkvo`, main and MTP, every backend's estimate − real = +0.00 MiB. Probe guard ≥ 2 × layers; under `-nkvo` the probe shows the pins (`c18_auto.out`, `c18/parsed.json`) |

### What changed (files, functions)

- **`src/prefix.h`**:
  - `qsa_stream_one_seq(cells)` (:15) is the one function that decides whether a stream holds a single sequence. Both the compact predicate and the non-compact scan call it, the scan through `qsa_scan_ranks` (:26; called at llama-memory-hybrid-idx.cpp:765 and :840), so the two cannot disagree.
  - `qsa_compact_visibility_stream(cells, count, ratio, u, i0, n)` (:94) implements §4.2:
    - embedding ubatches are judged like token ubatches;
    - only the stream's own tokens' sequence ids count;
    - a sequence holding 2-D cells is compact iff the stream holds only that sequence (rank-based limits);
    - a position ≥ 2^24 or a duplicate (pos, y, x) key (`qsa_seq_has_dup_key`, :61) is rejected;
    - multi-sequence text with no 2-D cells keeps today's rule, including the shared-block test.
  - It returns a `qsa_compact_reason` code that the decode error names. `qsa_scalar_visibility_cells` is now a wrapper. The old rule that a query at an image's own position is not compact is gone: the rank order makes it compact (test-qsa-prefix expectations updated).
- **`src/llama-memory-hybrid-idx.{h,cpp}`**:
  - `qsa_compact_reason(kv, n_stream, n_kv, ratio, ubatch)` (static, :303) runs the predicate per stream on either cache.
  - The context's `qsa_scalar_visibility` / `qsa_compact_reason` call it on the indexer cells (graph time).
  - `qsa_nc0` (:162) is NC0 = the selection path plus predicate rejection, read on the attention cells inside `prepare`.
  - `init_batch` (:172):
    - deterministic replay: the failing ubatch is re-formed with U = `qsa_noncompact_cap(n_ubatch)` = min(ub, 1 × 512) tokens in total;
    - the kept NC0 prefix is asserted unchanged;
    - asserts that each replay moves forward (:251) and that replays < #ubatches;
    - the recurrent prepare moved after the accepted attention prepare.
  - The constructor collects the QSA compress ratios and top-k. It throws at init when n_rs_seq + 1 ≥ U (:120), instead of the split_equal abort.
  - `init_full_n_kv(n_kv)` (:299) and a context constructor with an n_kv-limited attention and indexer view serve pass B.
  - Counters and test hooks: `qsa_n_resplit`, `qsa_last_replays`, `qsa_last_nc0`, `qsa_last_scan_ranked`, `qsa_resplit_configure`.
  - `LLAMA_QSA_FORCE_NONCOMPACT` is a validation-only hook, listed for deletion in W13.
- **`src/llama-kv-cache.{h,cpp}`**:
  - `prepare(ubatches, hook, i_fail)` (:819) runs the hook right after each `apply_ubatch` (:866). On false it stops, and the existing restore loop puts the cells back.
  - New `llama_kv_cache_context(kv, n_kv)`.
- **`src/llama-memory-hybrid.{h,cpp}`**: `llama_memory_hybrid_context(mem, n_kv)`.
- **`src/llama-graph.{h,cpp}`**:
  - `llm_graph_qsa_reserve` flags `LLM_GRAPH_QSA_RESERVE_NONCOMPACT` / `_FAST`, carried by `llm_graph_params::qsa_reserve` and `llm_graph_context::qsa_reserve`, are reserve-only.
  - `llm_graph_result::set_qsa_decode_error` / `get_qsa_decode_error`, cleared in `reset()`.
- **`src/models/qwen4exp.cpp`**:
  - `qwen4exp_use_block_selection` drops `ubatch.token`.
  - `build_qsa_top_k`:
    - the variant forces non-compact (:1474);
    - **the always-on graph-build check** (:1476) sets the decode error for a non-compact ubatch on the selection path when n_tokens > U, or for any non-compact ubatch in a single-sequence context;
    - pass A forces the fast-cache route at ≤ 512 tokens (:1531).
- **`src/llama-context.{h,cpp}`**:
  - `sched_reserve` adds the QSA passes (:1310-1360) and logs the per-backend reserve after each one:
    - `passA-trim` (n_tokens > 512, sequential positions ending at the last cell, one output in the last row);
    - `passA-fast` (≤ 512 tokens);
    - `passB` (dense shortcut at n_kv = ⌊(top_k + r − 1)/256⌋·256 = 2048, tokens capped at n_kv per stream);
    - `pp-0out` (MTP context: a prompt ubatch with no output; without it the strict run measured 1456.06 > 1403.23 MiB, `srv_uni_q8_u4096_img.log`, ubatch n_outputs = 0);
    - `noncompact` (n_seq_max > 1: T = min(ub, 512) tokens, `LLM_GRAPH_QSA_RESERVE_NONCOMPACT`).
  - Each pass enters the fit-probe max and runs the guard on its split_only graph. The guard log prints the inspected/required counts of that pass.
  - `graph_reserve` takes a `reserve_variant` (flags, `seq_pos`, `last_output`). n_outputs = 0 is allowed only with a variant.
  - `process_ubatch` returns the graph-build check's error before allocation (:2263) and counts it. After allocation, `strict_reserve_check` (:3783, `LLAMA_STRICT_RESERVE=1`) logs the first growth per backend past the reserve, with the ubatch, n_outputs, nodes, splits and the 3 largest nodes.
  - The realloc callback counts re-plans (:3826).
  - At exit, `LLAMA_QSA_RESPLIT ctx=… n_qsa_resplit=… decode_errors=…` and `LLAMA_STRICT_RESERVE ctx=… violations=… reallocs=…` are printed (:688).
- **Tests**:
  - `tests/qsa-mem-fixture.h`: the production hybrid-idx memory without a model file.
  - `tests/test-qsa-visibility.cpp`:
    - layouts C4, B65t, request C (back-to-back images across 512), image at pos 0, holes after seq_rm, split with an image stream;
    - negatives: an image next to another sequence in one unified stream (accepted again after seq_rm), a seq_cp block split, a duplicate key;
    - per-row comparison of the limits with the KQ mask, plus the agreement assert.
  - `tests/test-qsa-resplit.cpp`: R2-RES spec, layouts a / b / c / d / a1(np1), grid n_keep_tail {0, 4} × ub {512, 1024, 4096, 8192} × S {1, 2}.
  - Both are in the `NOT WIN32 OR NOT BUILD_SHARED_LIBS` block of `tests/CMakeLists.txt`. `tests/test-qsa-prefix.cpp` expectations are updated.
- **Scripts**: `scripts/qsa-validation/harness/w9/concurrent.sh` (two concurrent requests, strict + trace, prints the pass fields) and `at_c18_auto.sh` (W7's AT-C18 without `--parallel`).

### Code readings required by the plan

- **U35, which APIs can create a duplicate (pos, y, x) key in a qwen4exp flow** (§4.2 step 4). Read on HEAD db71ab244.
  - `seq_add` / `seq_div` (grp-attn, context shift) cannot create one: they abort. qwen4exp's rope type is IMROPE (llama-model.cpp:3806-3809), so `n_pos_per_embd()` = 4 (llama-hparams.cpp:285-288). `llama_memory_hybrid::seq_add/seq_div` call the attention cache first (llama-memory-hybrid.cpp:162-168, wrapped by llama-memory-hybrid-idx.cpp:419-436), which hits `GGML_ASSERT(hparams.n_pos_per_embd() == 1 …)` (llama-kv-cache.cpp:643, :693). This is pre-existing behaviour, not changed by W9.
  - `seq_cp` within a stream adds dst to every cell of src in [p0, p1) (llama-kv-cache.cpp:511-548). If dst already holds cells at those positions, dst gets two cells per position, which is a duplicate key. Whether a supported server flow calls `seq_cp` without a prior `seq_rm` of dst is UNPROVEN (not read). Either way `seq_cp` needs n_seq_max > 1, so the ubatch takes the non-compact route with the re-split and the reserved variant, not a decode error.
  - A client-built embedding batch can start at the last stored position: the M-RoPE rule for embd batches is X ≤ Y (llama-batch.cpp:425-433, against X < Y for tokens at :414-423). That puts the image's first cell on the key (X, X, X) of the text cell at X, which is a duplicate. The mtmd helper never does this: it places an image at `n_past`, the next free position (tools/mtmd/mtmd-helper.cpp:155, :196). In a single-sequence context this is a decode error that names the duplicate key, as §4.2 step 4 specifies. test-qsa-visibility covers it (`dup-image`, EXPECT_NONCOMPACT).
- **U37, the q8_0-over-f16 ROCm0 excess of the variant.** It does not appear in the rewrite. The variant graph alone at c131072 ub4096 `-np auto` (probe lines "this graph", `c18/srv_{f16,q8_0}_u4096.log`) measures:
  - main ROCm0 846.52 (f16) vs 846.79 (q8_0), +0.27 MiB; host 218.53 vs 218.79;
  - MTP 831.52 vs 831.52.

  The +232.27 MiB of R2-RES was measured on PR219's code at T 1024 / S = 2. Its node-level cause was not attributed: UNPROVEN, no longer needed for M1. M1's ≤ f16 + 1 MiB holds on these rows: final reserves q8_0 − f16 = +0.27 MiB main ROCm0 at ub4096 (1609.88 vs 1609.61) and at ub512 (846.79 vs 846.52).
- **AT-C16b.** With S = 1 every non-compact ubatch has ≤ 512 tokens.
  - `qwen4exp_score_strip` returns n_tps as long as n_stream · n_blocks · n_idx_h · 512 ≤ 2^27 (qwen4exp.cpp:1088-1100), which holds up to 262144 cells at ratio 4.
  - The strip loop then takes `whole = first == 0 && n_query == n_tps` and uses the original bias and kq_mask with **no view** (qwen4exp.cpp:1640, :1649-1652).
  - So "strip-view new 1 / reuse 11 per src per graph" is never observed on the slot-B text ubatches, and reserve no-share − share is 0 by construction, not ≥ 1056 MiB. The layout 3 run confirms it: the C19 census lists views only for the compact 4096-token prompt ubatches of slot A (8 strips, kq_mask new 8 / reuse 88, #219's sharing at work). No view appears for the TE ubatches (te 499 trace lines, n_q ≤ 512).
  - The criterion's premise (views on S = 1 non-compact ubatches) holds only past 262144 cells. This is a plan-criterion conflict for the owner: Prove can run AT-C16b at -c 524288, where strips halve. Runtime peak ≤ reserve does hold at layout 3: C19 BUFMAX = reserve, 1609.88 main / 1465.23 MTP ROCm0, strict 0.

### Commits (branch qsa-rewrite, not pushed)

- **db71ab244** `qsa : one-sequence compact predicate, non-compact re-split, reserve variants` (13 files, +647 −89; includes the test-qsa-prefix expectation updates).
- **3dc0a330b** `tests : QSA visibility (predicate vs KQ mask) and re-split tests on the hybrid-idx memory` (4 files, +600).
- **0639561f3** `scripts : W9 harnesses (concurrent two-slot strict / re-split runs, AT-C18 at -np auto)` (2 files, +70).
- Author and committer: Gaetan Puleo <12990773+gaetan-puleo@users.noreply.github.com>. **No Co-authored-by trailer.** W9 ports nothing from #219:
  - the predicate it rewrites is base code (`git blame` → 6dd6ea1290, Piotr Wilkin; `git merge-base --is-ancestor 6dd6ea1290 13a279bfa` = true, and 13a279bfa is the merge base of HEAD and 94e086847);
  - #219's strip_views is used unchanged;
  - `git grep "STRICT_RESERVE\|n_qsa_resplit\|qsa_stream_one_seq\|force_noncompact" 94e086847` is empty.
- Each commit builds on its own: the src commit carries its own test-prefix update, and the new tests come in the next commit.

### Build

- **HIP** `build/` (rule flags, `cmake --build build -j 16`):
  - final tree before the commits: rc 0, 0 `warning:`, 0 ` error:` (`hip_build_resume.log`, `hip_build_final2.log` after reverting the lab patch);
  - lab build for layout 3 (05-c19-log.patch, llama-server only): rc 0 (`hip_build_lab.log`). The patch was reverted with `git apply -R`; `md5sum` of llama-context.cpp / qwen4exp.cpp / models.h equals the pre-patch copies.
- **Vulkan** `build-vk/`: rc 0, 0 warnings, 0 errors (`vk_build1.log`). W9 changes no Vulkan source; the build checks the shared libllama and the tests.
- The commits hold exactly the built tree (`git status --short` empty after committing).

### Tests run (correctness; outputs)

- **Unit tests (CPU)**. HIP build (`ut_*.log` from chain2, `ut_resume_*.log` from chain3): all seven pass. Vulkan build (`ut_vk_*.log`): test-qsa-prefix, test-qsa-visibility, test-qsa-resplit and test-qsa-te-rows, all four pass.
  - test-qsa-prefix PASS (20000 randomized steps);
  - test-qsa-recover PASS;
  - test-qsa-fast PASS;
  - test-qsa-visibility PASS (53 / 17556 / 44 / 0);
  - test-qsa-resplit PASS (counts above; per layout, non-compact ubatches added: a 567, b 824, c 925, d 0, a1 0; replays a 439, b 660, c 764, d 0, a1 0);
  - test-qsa-allvis-rows PASS;
  - test-qsa-te-rows PASS (failures 0).
- **Mutants** (run by the earlier W9 agent, outputs in its transcript and `scratchpad/w9mut.out`; the sources were restored with md5 equal to the saved copies):
  - predicate accepts 2-D cells next to another sequence → FAIL `neg-A-text`;
  - scan never ranks → abort at the scan invariant llama-memory-hybrid-idx.cpp:927 `group_members` assert;
  - ranked tail start + r → 72 FAIL lines (`c4-image: row 0 lists a cell twice`, limits 2604 vs mask 2601);
  - NC0 hook always true → `(ii) 288`, "the re-split never engaged", FAILED.
  - The tests cannot pass trivially.
- **Per-op test-backend-ops**: not run. W9 changes no ggml op or kernel (`git show --stat db71ab244` lists only `src/llama-*`, `src/models/qwen4exp.cpp`, `src/prefix.h`, `tests/test-qsa-prefix.cpp`). W8's subsets stand.
- **Guard** (`test-qsa-guard`, `~/dev/wt_lab/w9/guard/`): pass q8_0 PASS, pass f16 PASS (the passB guard line requires 12 dense nodes), fake PASS, reject PASS, reject-decode PASS (`llama_decode returned -3`).
- **Strict reserve, llama-perplexity** (`-lv 4 -c 8192 -b 4096 --chunks 2 -ngl 99 -fa on`, `strict/ppl_*`): 0 violations and 0 "does not match" lines in all four runs.
  - ub512 ROCm0 reserve: q8_0 517.29 MiB, f16 517.03 MiB. The steps are pp/tg 510.48, passA-fast 510.51, passB 517.29.
  - W8's runtime at c8192 ub512 q8_0 grew past its own reserve (510.5085 > 510.4768 ROCm0, 27.2930 > 25.3750 host, `~/dev/wt_lab/w8/smoke_ppl_q8_0_u512.log`). The new reserve covers both values.
  - passB's +6.78 MiB ROCm0 is the dense-shortcut graph at n_kv 2048 with 512 output rows, a shape the reserve must cover.
  - ub4096: 4081.52 / 4081.25 MiB, unchanged by the passes.
- **Strict reserve, llama-bench** pp4096 @ d4096, ub4096, trim 1/0, q8_0 and f16: 0 violations, trim line 1 / 0 (`strict/bench_*`).
- **Single-sequence decode error**: `LLAMA_QSA_FORCE_NONCOMPACT=1` llama-perplexity at `-np 1` gives `process_ubatch: QSA layer 3: a non-compact ubatch (forced-by-LLAMA_QSA_FORCE_NONCOMPACT) of 512 tokens … (n_seq_max = 1, single-sequence context: no non-compact reserve)`, then `perplexity : failed to decode`, with decode_errors=1 (`strict/np1_forced_q8.log`). llama-perplexity itself exits 0 after a failed decode, so the log is what to check.
- **Server runs** (`concurrent.sh`, Signal-3.8 + mmproj + MTP, c131072, `LLAMA_STRICT_RESERVE=1`, `~/dev/wt_lab/w9/srv/`):

  | tag | flags | n_qsa_resplit main / MTP | strict violations | decode errors | TE n_q max | answers |
  |---|---|---|---|---|---|---|
  | uni_q8_u4096 | `--parallel 2 --kv-unified`, ub4096 | 82 / 81 | 0 / 0 | 0 | 512 | 11999, 32517 |
  | uni_f16_u4096 | same, f16 | 82 / 81 | 0 / 0 | 0 | 512 | 11999, 32517 |
  | split_q8_u4096 | `--parallel 2` | 0 / 0 | 0 / 0 | 0 | (no TE) | 11999, 32517 |
  | split1s_forced_q8_u4096 | `--parallel 2`, forced NC, one request | 108 / 108 | 0 / 0 | 0 | 512 | 32517 |
  | r1s_split_q8_u1024 | `--parallel 2`, ub1024 (M4 a) | 0 / 0 | 0 / 0 | 0 | (no TE) | 11999, 32517 |
  | r1s_uni_q8_u1024 | `--parallel 2 --kv-unified`, ub1024 (M4 a) | 45 / 46 | 0 / 0 | 0 | 512 | 11999, 32517 |
  | np1_q8_u4096_img | `--parallel 1`, S0_img alone | 0 / 0 | 0 / 0 | 0 | (no TE, cb 169) | 11999 |
  | np1_f16_u1024_img | `--parallel 1`, f16 ub1024 | 0 / 0 | 0 / 0 | 0 | (no TE, cb 651) | 11999 |
  | layout 3 (`w8/at_c16a.sh w9_l3_r1` + lab patch, default 4-slot server, ub4096 q8_0) | — | 22 / 22 | 0 / 0 | 0 | 512 | `12299…`, `40479` |

  - The guard on layout 3: `ctx=main passes=71 inspected=1692 cpu_launch=0`, `ctx=mtp passes=98 cpu_launch=0`, producer=masked 0.
  - Indicative only, one process per run, no gate (the U27 cost of S = 1; Prove measures it):
    - S0_img prompt runs at 345.6 t/s next to T_deep in the unified run, against 1062.2 t/s alone at `-np 1`;
    - layout 3's TI12k prompt runs at 578.78 t/s, against W8's 728.19 (8 strips of 4088 tokens) and PR219's 636.31 / 679.01 (`EV/r1c/runB3.out`).
- **AT-C18 at `-np auto`** (`at_c18_auto.sh`, `c18_auto.out`): 32/32 rows PASS, all +0.00 MiB.
- **M5, the cell measured here** (q8_0, c131072 ub4096, default 4 unified slots, `c18/srv_q8_0_u4096.log`). Every value is within its bound (PLAN-v3:760, instantiated + 1 MiB):
  - main ROCm0 1609.88 ≤ 1610.88;
  - main host 218.79 ≤ 219.79;
  - MTP ROCm0 1465.23 ≤ 1466.23;
  - MTP host 218.52 ≤ 219.52.

  Before pass B's token count was capped by n_kv, the MTP cell was 1476.16 MiB, over its bound (earlier W9 agent, transcript). The cap is in the committed code.

### Moved to Prove (owner scope note 2026-11-07; harness / command and pass criterion)

| item | command | criterion |
|---|---|---|
| AT-C12 grid | init-only `llama-server` (Signal-3.8 + mmproj + MTP, `-lv 4`, `LLAMA_STRICT_RESERVE=1`), default (no -c/-ub) and -c {131072, 261888} × ub {512, 1024, 4096, 8192} × {f16, q8_0}, plus `-np 2 -c 131072 -ub 1024`, on qsa-rewrite and PR219; parse `sched_reserve` + "QSA reserve" lines per backend, main and MTP (R2-RES `parse.py`) | every cell's new reserve ≤ the M5 bound; runtime max ≤ reserve (0 strict violations); 0 graph-build-check errors |
| AT-C12 image-slot layout | `r1s/r1s.sh`-style default server (4 slots unified): A = S0_img, then a second ≥ 8k-token turn on A while B = T_deep is resident, `LLAMA_QSA_TRACE=1 LLAMA_STRICT_RESERVE=1`; Q10-style per-request KLD of A's fixed tokens vs the median of ≥ 3 PR219 repeats | census: A's ubatches TE with n_q ≤ 512, B's cb; KLD ≤ the AT-Q10Q6 image τ; answers 2749 / 12749 (`~/repro_img`); 0 strict, 0 decode errors; non-compact prompt t/s reported vs PR219 (U27); M3 on the TE route |
| M5 remaining cells | as AT-C12 grid: c261888 ub8192, default (c262144 ub512), c131072 ub1024 unified, f16 and q8_0, main and MTP, with PR219 reserves | new ≤ max(PR219, V) + moved pool items + 1 MiB per backend (PLAN-v3:760 instantiated values) |
| AT-C17 / M6 | init-only `llama-server -c 524288` and `-c 1048576 -ub 4096` f16, new and PR219 (PR219 also one pp4096 ubatch below / above 262144 cells under the C7 shim), then `-np 4` on new | new f16 ≤ PR219 reserve + PR219 pool high-water (M6) |
| AT-C16b | `scripts/qsa-validation/harness/w8/at_c16a.sh` with `05-c19-log.patch` and `LLAMA_STRICT_RESERVE=1`, at `--ctx-size 524288` (so that ≤ 512-token TE ubatches have strips < n_tps), share vs a no-share test hook | owner decision on the criterion (see "AT-C16b" above). At c131072 the census is 0 views by construction; runtime peak ≤ reserve already passes at c131072 |
| M4 (b) | `-np 1` request C (`~/repro_img`, 5 images, ~150k tokens) with `LLAMA_STRICT_RESERVE=1 LLAMA_QSA_TRACE=1` | 0 strict violations, 0 predicate rejections on image ubatches (no `producer=te`, no decode error), correct answer |
| M4 (c) on GPU | a synthetic single-sequence prompt with two images back to back across a ubatch boundary at depth > 32k, same env | as (b) |
| C15 deep-image D at ub1024; requests A/B/C image matrix | `r1e`/`c15` recipes on qsa-rewrite with strict + trace | 0 graph-build-check errors, 0 strict, answers 2749 / 12749 / 26999 |
| np2 shared prefix with trim on/off | two concurrent requests sharing a prefix (seq_cp) on a `--parallel 2 --kv-unified` server, `LLAMA_FN_LAST_ATTN_TRIM=1/0`, strict | 0 violations, correct answers |
| U27 | S0_img/T_deep and layout 3 prompt t/s, new vs PR219, ≥ 3 interleaved processes (§16 design) | reported (no fixed gate in PLAN-v3:868) |

### Notes

- No new item for `TODO-LATER.md`. Two points for the owner:
  - **AT-C16b's criterion** cannot be met by the S = 1 design at c ≤ 262144 (above).
  - **"`n_qsa_resplit` > 0 in both contexts" on the split + MTP run** is 0 by construction (split streams hold one sequence each). The > 0 requirement is met by both contexts of the unified runs.
- Cost of the new reserve passes, from the logs above:
  - c8192 ub512: +6.78 MiB ROCm0 from pass B (517.29 vs 510.51);
  - c131072 ub4096 default server: none on ROCm0; the variant takes host from 202.07 to 218.79 MiB (main);
  - c131072 ub512 default server: the variant takes ROCm0 from 660.34 to 846.79 MiB (main) and from 639.53 to 831.52 MiB (MTP).
- Not touched: `~/dev/wt_pr219v2*`, other agents' worktrees, systemd units, strix-001. Only processes started by this item's harnesses were stopped (SIGINT by the harness).
- Cleanup: no file over 100 MB (`find ~/dev/wt_lab/w9 ~/dev/wt_rewrite_logs/W9 ~/dev/wt_lab/w8/logs -size +100M` is empty; total 24 MB + 1.3 MB). The earlier agent's mutant backups in `/tmp/w9mut` (256 KB) on strix-002 were deleted.

## W10: rotation and output un-rotation in the op

**Status: implemented and committed (not pushed).** The HIP build passes with 0 warnings and the Vulkan build passes; its 8 warnings are all at test-backend-ops.cpp lines W10 does not touch. Every correctness check run for this item passes:
- HIP SPARSE_ATTN 1390/1390 with 0 "not supported", under the default and under every forcing switch;
- H_q and H_o are bitwise equal to fwht.cu in every HIP tier, and the op is bitwise equal to the graph-side form it replaces;
- the gated G4 census shows q8_0 and q4_0 (23/24 gated at ub4096);
- 0 spills over 549 kernels;
- the real model gives a KLD of 0 against the W9 head (Same top p 100 %).

OP_PROF of rotation + gate + attention is below W9 at d0 and d32k. This was measured once per arm and is indicative only; the proper A/B is moved to Prove. Vulkan keeps W6's H_o dispatch inside the op rather than a kernel epilogue (deviation below).

All commands ran on strix-002, and every build and GPU job ran under `flock ~/gpu.lock`. No other agent was active. Logs and outputs:
- `~/dev/wt_rewrite_logs/W10/` holds builds, op tests (`optests/`, `optests_final2/`), the TU checks (`tu_final.out`, `tu_final2.out`), the ISA comparison (`isa_rot0.out`) and the commit messages (`W10msg/`);
- `~/dev/wt_lab/w10/` holds the census (`census/`, `census2/`), smoke (`smoke/`, `smoke2/`), op time (`time/`), OP_PROF (`opprof*/`) and the chain outputs (`chain1.out`, `chain2.out`);
- `base_bin/` is a copy of the W9 head's `build/bin` (HEAD 0639561f3), taken before any W10 edit. It is the A/B baseline, and Prove can reuse it.

There was no earlier W10 agent: the tree was clean at 0639561f3 (`git status --short` empty).

### What changed (files, functions)
- **`ggml/src/ggml-cuda/sparse-attn-fwht.cuh`** (new; the plan's `fwht_block<N>`):
  - `fwht_block<N, WS>` is the warp FWHT: scale first, then butterflies from stride 1 up, shuffles below the warp size and registers above it;
  - `fwht_row<D, WS>` applies it to one 256-dim row over its blocks of rot = 64, 128 or 256;
  - `sa_fwht_smem<D>` is the same arithmetic through shared memory, for any wave size;
  - FP contraction is off inside all three (`SA_FP_CONTRACT_OFF` = `_Pragma("clang fp contract(off)")` on clang-based compilers).
- **`fwht.cu`**: `fwht_cuda` now calls `fwht_block`, so the hinted mul_mat and the op share one transform.
  - Effect on fwht.cu itself: fwht.cu was compiled alone to gfx1151 assembly before and after. All 16 kernels keep their FMA counts: 0 in every f32 instance, and the f16 instances' `v_fma_mix` (the conversion × scale) unchanged. 12 kernels are instruction-identical. The N = 256 / 512 warp kernels differ only in schedule and dual-issue pairing, with no FMA either way (`/tmp/w10asm` comparison, opcode histogram; the files are deleted). So the f32 arithmetic of fwht.cu is unchanged by code reading plus this ISA check.
- **`sparse-attn-attend-g4.cuh`** (`sa_attend_g4<KT, VT, VL, EPI, GQA, ROT>`):
  - **ROT is a new bool template parameter.** With ROT = true and `rot_q` != 0, a prologue reads whole query rows (wave w takes rows w, w + 8, …). It applies `fwht_row` in f32, then f16(x · scale) into the per-wave stage the existing read-back expects.
  - With ROT = true and `rot_o` != 0, the epilogue takes whole rows of acc / l from the staging tile, applies H_o, then the store. When gated, the sigmoid gate comes after H_o, element by element with the unrotated expression, and also feeds the o16 copy.
  - `sa_attend_fixup` (not templated) does H_q / H_o through `sa_fwht_smem`, plus a `fixup_all` test hook.
  - `sa_g4_args` gains `rot_q`, `rot_o`, `rot_q_scale`, `rot_o_scale`, `rot_dump` and `fixup_all`.
  - `sa_g4_waves_per_eu`: the generic GQA 12 instance gets 7 waves when VL = direct (as in W5) or ROT = true.
- **`sparse-attn-attend-g1.cuh`** (`sa_attend_g1<KT, VT, GQA, ROT>`):
  - H_q prologue: wave w takes heads w and w + 8.
  - With one split, H_o of each head row is staged through a new `float ostage[8][256]` member of the existing 8 KB union, 8 heads at a time, so LDS does not grow.
  - With more splits the partials stay unrotated.
- **`sparse-attn-merge.cu`**: `sa_merge_splits` applies H_o through `sa_fwht_smem`. `ggml_cuda_sa_merge_splits` takes `rot_o`.
- **`sparse-attn-portable.cu`**: H_q / H_o through `sa_fwht_smem` over the padded `qs` rows (ld 257).
- **Instance tables** (`sparse-attn-attend-g4-{f16,bf16,q8_0,q4_0}.cu`, `sparse-attn-attend-g1.cu`, `sparse-attn-generic.cu`):
  - two entries per census index (2·i = ROT false, 2·i + 1 = ROT true), so the census keys and ids are unchanged;
  - instance counts per K type: G4 48 → 96, G1 48 → 96, generic G4 6 → 12, generic G1 3 → 6.
- **`sparse-attn-attend-g4.cu`**:
  - `sa_select` accepts rot ∈ {0, 64, 128, 256}, where it used to reject rot ≠ 0.
  - `sa_run` passes the rotation parameters to every tier and picks the ROT instance when `rot_q` or `rot_o` is non-zero. It passes `rot_o` to the merge.
  - Test hooks:
    - `GGML_CUDA_SA_ROT_DUMP=1` is read per dispatch: an op with `rot_q` ≠ 0 writes H_q q (f32) into dst and launches nothing after the attend;
    - `GGML_CUDA_SA_FORCE=fixup` makes the G4 fixup recompute every block.
  - **`sparse-attn.cuh`**: launch parameters gain `rot_q`, `rot_o` and `rot_dump`. `ggml_cuda_sa_rot_scale(rot)` = `1 / sqrtf(rot)`, fwht.cu's host expression.
- **`src/models/qwen4exp.cpp`** (`build_attn_qsa`):
  - the q rotation and the `kqv_out` un-rotation nodes are deleted;
  - `ggml_sparse_attn(…, rot_q = self_k_rot ? self_k_rot->ne[0] : 0, rot_o = self_v_rot ? self_v_rot->ne[0] : 0)`;
  - the k_cur / v_cur storage rotation is unchanged, and so is `LLAMA_ATTN_ROT_DISABLE`;
  - rollback = pass 0, 0.
- **Tests**:
  - `tests/test-backend-ops.cpp`: 99 new SPARSE_ATTN cases (1390 in all, 1291 before).
    - Rotated G4 packed / direct and G1 one-split for every S_fast type at (256, 64), (64, 256), (128, 128).
    - GQA 4 / 8, two streams, mixed pairs, the five S_generic types.
    - Rotated empty and poison rows; the V overflow cases with H_q (rot_o 0), which reach the fixup.
    - `test_sparse_attn_gated` takes rot: every S_fast type × V regime × (256, 64) / (64, 256), the three 16-bit reader variants, 2 streams with GQA 8, and the rejection.
  - `tests/test-sparse-attn-rot.cpp` (new, HIP, ctest; it skips without ROCm):
    - **hq**: the dump hook against fwht.cu;
    - **ho**: op(rot_o) against fwht.cu(op(0));
    - **graph**: op(rot_q, rot_o) against fwht.cu(q) → op(0) → fwht.cu(out);
    - **gated**: the fused rotated epilogue against the unfused graph-side form, with census engagement;
    - shapes: G4 packed / direct and G1 one split / merge, each × f16, q8_0, q4_1 × rot 64 / 128 / 256.
  - `tests/test-qsa-select-dump.cpp`: gated = true for every V type.
- **Scripts** (`scripts/qsa-validation/harness/w10/`): `optests.sh`, `tu_all.sh`, `isa_rot0.sh`, `census.sh`, `smoke_kld.sh`, `optime.sh`, `opprof.sh`, `opprof_an.py`.

### Deviations from the plan text, with reasons
- **ROT template instances instead of a runtime branch only.** The first version used a runtime branch in every instance, and its rot 0 attend was slower than W9 in an op-time A/B (W5 harness `w5_time.cpp` copied as `~/dev/wt_lab/w10/time/w10_time.cpp`, C18 wiki rows, two processes per arm, interleaved; first `optime.sh` run):
  - wiki d4k: f16 12.32 / 12.56 → 12.90 / 13.15 ms per layer, about +4.7 %; q8_0 12.78 / 13.16 → 13.58 / 13.80;
  - wiki d131k: f16 18.11 / 18.07 → 18.79 / 18.70; q8_0 18.01 / 17.92 → 18.77 / 18.74.

  With ROT as a template parameter, every ROT = false instance is instruction-identical to the W9 kernel (`isa_rot0.sh`: device assembly of each TU from the tree and from `git archive HEAD`, branch label numbers normalised; `isa_rot0.out`):
  - 48 / 48 per G4 TU (f16, bf16, q8_0, q4_0);
  - 48 / 48 for G1;
  - 9 / 9 for the generic tier;
  - the 33 fixup kernels are not templated and differ.

  The second `optime.sh` run confirms it:
  - wiki d4k: W9 f16 12.12 / 12.48 against W10 rot 0 12.30 / 12.50; q8_0 12.66 / 13.06 against 12.90 / 13.13;
  - wiki d131k: W9 18.05 / 17.95 against W10 rot 0 18.00 / 17.95; q8_0 17.95 / 17.88 against 17.88 / 17.87.

  The rotated op itself (rot 256 / 64) costs:
  - wiki d4k: 13.05 / 13.27 (f16) and 13.77 / 13.99 (q8_0);
  - wiki d131k: 18.88 / 18.87 (f16) and 18.88 / 18.88 (q8_0).

  The rotation and the gate were moved out of the graph to pay for this, and OP_PROF below shows the net.
- **Vulkan: H_o stays the in-op dispatch of `fwht.comp` that W6 shipped** (W6 "Deviations"), not an epilogue in every Vulkan kernel. H_q is already in every Vulkan kernel (W6). Both are bitwise equal to fwht.comp (`test-sparse-attn-vk rot`, 36/36 below), and both run inside the SPARSE_ATTN op. The plan's reason for the epilogue is the HIP gate fusion (D13), and Vulkan has no gate fusion. Moving H_o into the cm / scalar / portable kernels and restructuring the split reduce (one workgroup per 32 dims today) was not done. **OWNER: confirm or ask for the kernel epilogue.**
- **H_q is checked with a dump hook, and additionally end to end.** The plan asks for "HIP in-kernel q transform bitwise equal to fwht.cu". The hook checks that directly. The `graph` check also shows the whole op equal to the graph-side form.
- **`GGML_CUDA_SA_FORCE=fixup`** is new. Without it, the fixup's rotation path would be reached only by the overflow cases (rot_o 0).

### Commits (branch qsa-rewrite, not pushed; `git log --format='%h %an <%ae> | %ce | %s' 0639561f3..HEAD`)
- `d2297b7a9` hip : fwht_block, the FWHT shared by fwht.cu and the sparse-attention kernels
- `84f1e4bea` hip : SPARSE_ATTN in-op rotations: rot_q prologue and rot_o epilogue in every tier
- `a120beefa` tests : SPARSE_ATTN rotation cases on every kernel family, gated rotations, test-sparse-attn-rot
- `20deb5659` qwen4exp : q rotation and output un-rotation inside SPARSE_ATTN (rot_q = n_rot_k, rot_o = n_rot_v)
- `426040b7e` scripts : W10 harnesses (op tests, rotation unit test, TU and ISA checks, census, smoke, op time, OP_PROF)
- Author and committer: Gaetan Puleo <12990773+gaetan-puleo@users.noreply.github.com>. 27 files, +1322 −124. `git status --short` is empty.
- **No Co-authored-by trailer.**
  - W10 ports nothing from #219: `git grep -c "rot_q\|fwht_block" 94e086847 -- ggml/src/ggml-cuda` is empty.
  - fwht.cu is base code: its last commits are by bri-prism, Johannes Gäßler and Aman Gupta, and `git merge-base --is-ancestor` puts it in ea5285ddf.
  - The kernel bodies W10 extends already carry Iggy Jackson's trailer from W3/W4. W10 adds new prologue / epilogue code only.
- Each intermediate commit building on its own is UNPROVEN: only the final tree was built. By construction, `d2297b7a9` and `84f1e4bea` touch no test or graph file, and the test list in `a120beefa` only adds cases.

### Build
- **HIP** `build/` (rule flags, `cmake --build build -j 16`):
  - final tree: rc 0, 0 `warning:` (`hip_build5.log`);
  - earlier W10 builds (`hip_build1.log`): rc 0, 8 warnings, all in test-backend-ops.cpp at lines 463, 466, 9386, 11455, 17540-17549, none touched by W10.
- **Vulkan** `build-vk/`: rc 0, 8 warnings (the same test-backend-ops.cpp lines; `vk_build1.log`). W10 changes no Vulkan source; the build covers test-backend-ops, test-qsa-select-dump and libllama. It was built before the ROT-template change, which touches only `ggml-cuda` TUs that build-vk does not compile.
- **TU checks** (`tu_all.sh final2`, `-Werror -Wno-tautological-compare` + resource remarks, `gcn-cdna-vgpr-check.py parse_assembly`; `tu_final2.out`):
  - 10 TUs: 0 warnings, 0 errors;
  - **549 sa_* / fwht kernels, 0 spilling, 0 scratch**; G4 max 192 VGPRs; the generic tier is at 7 waves for GQA 12 (max 209, as in W5).
  - The first W10 version (runtime branch only) had 1 spill: the generic packed-V GQA 12 instance, 192 VGPRs, 8 B scratch (`tu_final.out`). That spill is what motivated the waves rule above.
- **Multi-arch compile** (the `build-multiarch` command, gfx1100 / gfx1151 / gfx90a / gfx942, `-Werror`): sparse-attn-portable, sparse-attn-merge, fwht, sparse-attn-attend-g1 and sparse-attn-attend-g4-q8_0 are rc 0, 0 warnings.

### Tests run (correctness; outputs)
- **Per-op test-backend-ops** (`optests.sh`, one op per run, under flock, `--fail-unsupported`).

  Final tree (`optests_final2/`):

  | run | result |
  |---|---|
  | ROCm0 SPARSE_ATTN, default | 1390/1390, 0 not supported (W9: 1271/1291 with the 20 rot cases unsupported) |
  | graphs off (`GGML_CUDA_DISABLE_GRAPHS=1`) | 1390/1390 |
  | `GGML_CUDA_SA_FORCE=vdirect` | 1390/1390 |
  | `vpacked` | 1390/1390 |
  | `generic` | 1390/1390 |
  | `portable` | 1390/1390 |
  | `fixup` | 1390/1390 |
  | SPARSE_ATTN_PLAN, ROCm0 and CPU | 185/185 each |

  - Census coverage: 492/492 compiled keys launched, unchanged.
  - Before the ROT-template change (`optests/`): CPU SPARSE_ATTN 1390/1390. The CPU op is unchanged since.
  - Vulkan0 (`build-vk`): SPARSE_ATTN 744/744; the 646 "not supported" are all G = 4 (listed). Of its 119 rot cases, 53 run (G1) and pass. SPARSE_ATTN_PLAN 92/92.
  - Bands: rot cases use the rot 0 band of their class (§14.2). 0 compare failures in every run.
- **`test-sparse-attn-rot`** (final tree; one process per `GGML_CUDA_SA_FORCE` of "" / generic / portable / fixup; `optests_final2/rot_*.log`):
  - every run passes: 0 failed of 144 (default, fixup) and of 136 (generic, portable);
  - each run has 136 bitwise lines:
    - H_q: 36 (4 shapes × 3 types × rot 64 / 128 / 256);
    - H_o: 36;
    - graph: 60 (12 shape × type combinations × 5 rot pairs);
    - gated: 4.
  - **H_q and H_o are bitwise equal to fwht.cu in every tier**: fast G4 packed / direct, G1 single split / merge, generic, portable, and the fixup forced.
  - **The op with (rot_q, rot_o) is bitwise equal to fwht.cu(q) → op(rot 0) → fwht.cu(out)**, the qwen4exp graph of W9.
  - **gated**: in-op rotation + fused gate against graph-side rotation + unfused gate is bitwise, 0 of 399360 values differ (f16 and q8_0, packed and direct). Census per case: in-op rotation gated 1 / plain 0; graph-side rotation gated 0 / plain 1.
- **`test-sparse-attn-vk rot`** (build-vk, RADV): OK, 0 failed checks, 36 bitwise lines (H_q and H_o against fwht.comp, every tier). This is W6's check, re-run.
- **Gated G4 census** (C12-style, HIP graphs off, `census.sh`: llama-perplexity c8192 b4096 --chunks 1 with `LLAMA_QSA_TRACE=1`, census_check.py `expect` against the shipped policy rows; `~/dev/wt_lab/w10/census{,2}/`). Every run is EXPECT PASS with 0 diffs and 0 masked lines:

  | K/V | ub4096 gated / plain | ub512 gated / plain |
  |---|---|---|
  | q8_0 | 23 / 1 (W7: 0 / 24 plain, `~/dev/wt_lab/w7/census/ppl_q8_0_q8_0_u4096.log`) | 184 / 8 |
  | q4_0 | 23 / 1 | 184 / 8 |
  | f16 | 23 / 1 (as W7) | 136 / 8 |
  | bf16 | 23 / 1 | 184 / 8 |
  | q8_0 K / f16 V | 23 / 1 | 184 / 8 |
  | f16 K / q8_0 V | 23 / 1 | 184 / 8 |

  - Every run's PPL equals W7's census run to the printed digits, e.g. q8_0 2.1139 ± 0.05827, q4_0 2.1271, f16 2.1082 / 2.0978.
  - The f16 K / q8_0 V ub512 PPL appears only in census/ppl_f16_q8_0_u512.log: 2.1152, the same as W7's ub4096 run.
  - census2 re-ran q8_0 / q4_0 / f16 on the final tree with identical results.
- **Smoke on the real model** (`smoke_kld.sh`: Qwen3.8-Flash-Next UD-Q4_K_XL, c8192 b/ub4096, 1 chunk, the W9 head's logits as the KL base; base files deleted):

  | K/V | Mean KLD | Max KLD | Same top p | PPL W9 → W10 |
  |---|---|---|---|---|
  | q8_0 | 0.000000 | 5.1e-5 | 100 % | 2.1139 → 2.113919 |
  | q4_0 | 0.000000 | 5.5e-5 | 100 % | 2.1271 → 2.127075 (ratio 1.000044 ± 0.000044) |
  | f16 | 0.000000 | 5.8e-5 | 100 % | its route is unchanged |

  - The q8_0 and q4_0 values hold on both trees (smoke and smoke2).
  - The max KLD of the unchanged f16 route sets the floor of this comparison. The q8_0 / q4_0 maxima are at or below it.
- **OP_PROF** (U6; `opprof.sh` + `opprof_an.py`, q8_0, ub4096, graphs off, OVL 0, trim 0; one process per arm; the W9 head from `base_bin`). Per 4096-token pass over the 12 QSA layers, the sum of attend (with the fused gate), hadamard mul_mats and unfused gate (CONT+SIGMOID+MUL), excluding the plan:

  | depth | W9 | W10 | diff |
  |---|---|---|---|
  | d0 (-c 4096) | 133.17 ms | 104.83 ms | −28.35 ms |
  | d32k (-c 36864, mean over the chunk's 9 ubatches, depth 0..32768) | 236.97 ms | 194.85 ms | −42.13 ms |

  `~/dev/wt_lab/w10/opprof2/summary` is in chain2.out.
  - W10's hadamard mul_mats fall from 12.51 + 10.54 ms to 0.61 + 0.40 ms (only k_cur / v_cur remain), and the unfused gate from 19.40 to 1.67 ms (the 1 plain layer).
  - **Below W9 in this run, but indicative only**: W9's d32k attend alone measured 176.77, 165.43 and 195.18 ms in three single runs (chain1, `opprof_q8_l`, chain2). One process per arm cannot gate this. Moved to Prove.
  - The first W10 run at d32k (chain1, runtime-branch build) measured 301.66 ms for the fused attend. A re-run of the same build gave 172.43 ms. The outlier's cause is UNPROVEN.

### Done-when scorecard (PLAN-v3:581-585)
| item | result |
|---|---|
| `-o SPARSE_ATTN` with rot ∈ {64, 256} within the §14 bands | **PASS**: 1390/1390 on ROCm0 in every tier, CPU 1390/1390, Vulkan G1 rot cases pass |
| HIP in-kernel q transform bitwise = fwht.cu, Vulkan = fwht.comp | **PASS**: HIP 36 H_q + 36 H_o bitwise per tier (4 tiers); Vulkan 36/36 |
| C12-style census (graphs off) shows the gated G4 epilogue for q8_0 and q4_0 | **PASS**: 23/24 (ub4096) and 184/192 (ub512) gated for both, from 0 for q8_0 at W7 |
| OP_PROF (rotation + gate + attention) below W7's at ub4096 pp4096 d0 and d32k q8_0 | **indicative PASS** against the W9 head 0639561f3 (W7 + W8 + W9; a W7-head arm was not run): −28.35 / −42.13 ms, one process per arm. Proper A/B, including a W7-head arm, moved to Prove |
| Q5, Q1-A/B, P1–P6 re-run and pass | not run here (owner scope note): moved to Prove. Supporting evidence: the ROT = false instances are the W9 kernels instruction for instruction, the rotated op is bitwise the W9 graph-side form, and the model KLD vs W9 is 0 |
| Rollback = revert this commit only | `20deb5659` alone restores graph-side rotation (rot 0, 0). The kernel commits are inert at rot 0 (ISA-identical rot 0 instances) |

### Moved to Prove (owner scope note 2026-11-07; harness / command and pass criterion)
| item | command | criterion |
|---|---|---|
| U6 OP_PROF A/B | `scripts/qsa-validation/harness/w10/opprof.sh` (BASE = W9 head bin, e.g. `~/dev/wt_lab/w10/base_bin`) run ≥ 3 interleaved processes per arm, KV q8_0 (and q4_0), d0 and d32k | median per-pass (rotation + gate + attention) of W10 < W9 / W7, outside the arms' spread |
| op-time of the rotated op | `optime.sh` with ≥ 3 processes per arm | report rot (256, 64) vs rot 0 vs W9 (no gate in the plan) |
| Q5 | the R2-BOP / W3 op-local band harnesses (`bandsop`, W3 `prband`, W4 `tbo` / `uni`) on the final head, rot 0 and rot (256, 64) | NMSE(GPU, CPU op) ≤ the §14.2 band of the census class on the fixture, SYN and UNI rows; planted defects exceed it |
| Q1-A / Q1-B | W12/W7 `q_campaign.sh` pinned arms (c8192 × 8; ub4096 and ub512; T ∈ {f16, q8_0, q4_0, bf16, q8_0/f16, f16/q8_0}) new vs PR219 | PLAN-v3 §15 Q1 bands (difference form; the direct-form q4_0 / bf16 misses are TODO-LATER #1) |
| P1–P6 | the §16 bench designs (`perf.sh` and siblings) on the final head | PLAN-v3 §16 P1–P6 criteria |
| Vulkan quality with in-op H_q (graph change) | Q1-vk recipe | B_vk band |

### Notes
- No new TODO-LATER item. One owner decision: **the Vulkan H_o placement** (in-op dispatch vs kernel epilogue), see Deviations.
- Not touched: `~/dev/wt_pr219v2*`, other agents' worktrees, systemd units, strix-001. The only processes stopped were ones this item started (none left running).
- Cleanup: the KL-divergence bases (about 2 GB each) were deleted by `smoke_kld.sh`. TU save-temps and the ISA archive (about 1.1 GB) were deleted; `~/dev/wt_rewrite_logs/W10` is now 4.8 MB. `~/dev/wt_lab/w10/base_bin` (135 MB) is kept for Prove's A/B. `find … -size +1G` finds nothing. `/tmp/w10asm` on strix-002 was deleted.

## W11: indexer cache mapping, no session format change

**Status: implemented and committed (not pushed).** The HIP build passes (0 warnings in the files W11 touches: `grep warning: ~/dev/wt_rewrite_logs/W11/build{1,2}.log`). Every correctness check run for this item passes:
- `test-qsa-idx-type` (new) passes, and so do the 13 related ctest entries;
- the real-model session check (file saved on W0, loaded on W11) passes;
- the real-model KLD smoke passes, with the W10 values reproduced exactly for every arm whose indexer type does not change;
- the q4_0 route census passes, with the same gating as W10;
- an MTP server smoke works: both contexts get a q8_0 indexer.

One finding predates W11 and is reported for the owner: the MTP server's census shows the fusible G4 attends ungated (0/12), and the result is identical with the pre-W11 indexer type (Notes).

All commands ran on strix-002. Every build and GPU job ran under `flock ~/gpu.lock`, and no other agent was active. Logs and outputs:
- `~/dev/wt_rewrite_logs/W11/`: builds, `test-qsa-idx-type.log`, `ctest.log`, commit messages `msg/`;
- `~/dev/wt_lab/w11/`: `session/`, `session.out`, `smoke/`, `smoke.out`, `census/`, `census_ovr/`, `mtp/`, `mtp_ovr/`, `mtp_smoke.out`.

There was no earlier W11 agent: the tree was clean at 426040b7e (`git status --short` empty).

### What changed (files, functions)
- **`src/llama-model.cpp`**:
  - `llama_model_type_k_idx(arch, type_k, type_k_idx)` (:1124). For `LLM_ARCH_QWEN4EXP`, an explicit type wins. Otherwise (`GGML_TYPE_COUNT` = auto) q4_0, q4_1, q5_0, q5_1 and iq4_nl map to q8_0, and every other type stays as it is (f16, q8_0, bf16, f32; PLAN-v3:383-384). Every other architecture returns `type_k` (GLM5_NEXT, O26).
  - `qsa_resolve_type_k_idx` (:1147) resolves the type and warns when an override is given to a non-qwen4exp model. It checks the block size against `indexer_head_size` and logs `indexer K cache type = X (auto|override, -ctk T)`.
  - Called at the three hybrid-idx sites (:3086 GLM5_NEXT, :3292 qwen4exp MTP, :3399 qwen4exp main).
  - Declared in `src/llama-model.h:991-993`.
- **`include/llama.h:428`**: `llama_context_params.type_k_idx`. The default is `GGML_TYPE_COUNT` = auto (`src/llama-context.cpp:5120`). It is validated in `llama_init_from_model` (:5176-5179) and carried by `llama_memory_params.type_k_idx` (`src/llama-memory.h:24`, filled at `llama-context.cpp:562`). **`LLAMA_SESSION_VERSION` 11 and `LLAMA_STATE_SEQ_VERSION` 4 are unchanged** (`include/llama.h:46,49`; `git diff 426040b7e HEAD -- include/llama.h | grep VERSION` is empty).
- **`src/llama-memory-hybrid-idx.{h,cpp}`**:
  - The constructor takes the resolved `type_k_idx` (h:40-41) and builds the indexer `llama_kv_cache` with it (cpp:108). The log line becomes `creating indexer KV cache, size = N cells, type = X`.
  - New: `get_type_k_idx()` (h:86).
  - `state_read` marks the indexer section (cpp:494). A failure there is re-thrown through `state_read_idx_error` (cpp:522), after the existing `state_drop`, as `failed to restore the sparse-attention indexer cache (first QSA layer N): the session was saved with indexer type X, this context uses Y; re-save the session[, or load it with --cache-type-k-idx X]`. The hint is given for qwen4exp only. Any other indexer-section failure keeps its original text inside a generic wrapped message, and failures outside the indexer section are re-thrown unchanged.
  - The stream is unchanged.
- **`src/llama-kv-cache.{h,cpp}`**: the existing per-layer check `mismatched key type` now also records the saved type (cpp:2816, reset at :2380). `state_read_k_type_mismatch()` (h:172) exposes it. The message and the check are as before.
- **`common/`**:
  - `common_params.cache_type_k_idx` (common.h:666, default auto) is passed on at common.cpp:1746.
  - New option `--cache-type-k-idx auto|TYPE` / `LLAMA_ARG_CACHE_TYPE_K_IDX` (arg.cpp:2490). It replaces the measurement-only `QPIN_IDX_TYPE` (PLAN-v3:386).
  - The MTP draft context inherits the option through `common_base_params_to_speculative`, which copies `params`. Auto then maps from the draft's own `-ctkd`.
- **tests**:
  - New `tests/test-qsa-idx-type.cpp` (CMake entry next to test-qsa-resplit).
  - Constructor call sites updated in `tests/qsa-mem-fixture.h` and `tests/test-qsa-prefix.cpp`.
- **scripts**:
  - New `harness/w11/{session.sh, smoke_kld.sh, mtp_smoke.sh}` and `summaries/w11_idx.md`.
  - `harness/w7/lab_tree.sh` drops 01-qpin's `QPIN_IDX_TYPE` hunk on a W11+ tree, because it no longer applies. The three lab patches then apply on the W11 head (checked on a `git archive` copy, rc 0; **the lab tree build at W11 is UNPROVEN**, not built here).
  - `harness/w7/q_campaign.sh` phase C passes `LLAMA_ARG_CACHE_TYPE_K_IDX=q8_0` next to `QPIN_IDX_TYPE=q8_0` for the f16-K q8-idx arms.

Nothing was taken from #219 in W11, so no commit carries a Co-authored-by trailer. The per-layer type check it relies on is base code (`llama-kv-cache.cpp` state_read_data).

### Commits (branch qsa-rewrite, not pushed)
All authored by Gaetan Puleo <12990773+gaetan-puleo@users.noreply.github.com> (`git log --format='%h %ae' -4`):
- `9a61f16b8` qwen4exp : indexer key cache type auto (q8_0 for a sub-8-bit -ctk), --cache-type-k-idx override, clearer restore error
- `2d80c06be` tests : test-qsa-idx-type, the indexer type mapping and the session check of an old indexer type
- `866d436e7` scripts : W11 harnesses (session files, KLD smoke, MTP smoke) and summary
- `6ad61a115` scripts : W7 lab tree and Q4 campaign on the W11 indexer override

Rollback: revert `9a61f16b8`, and auto becomes -ctk again (PLAN-v3:830). `2d80c06be` must be reverted with it, because the test calls the new function.

### Build
- HIP `build/` (flags as in the rules; existing cache), `cmake --build build -j 16` under flock.
  - First build: rc 2. `tests/test-qsa-prefix.cpp:115` called the old constructor (`~/dev/wt_rewrite_logs/W11/build1.log`). Fixed in the same commit.
  - Second build: rc 0 (`build2.log`). Its only warnings are the 3 pre-existing `u8path` deprecations in `tests/gguf-model-data.cpp`. Across both logs, the only warning in a file W11 touches is the pre-existing `u8path` deprecation at `common/common.cpp:1034`.
- Vulkan `build-vk`: **not built**. W11 changes no file under `ggml/` (`git diff --stat 426040b7e HEAD` lists no ggml path), so the task rule (build Vulkan when the item touches Vulkan) does not apply.
- test-backend-ops: **not run**, for the same reason: no ggml op changes. SPARSE_ATTN and the indexer ops are unchanged, and q8_0 indexers were already reachable with `-ctk q8_0`.

### Tests run (correctness; outputs)
1. **`test-qsa-idx-type`** (`~/dev/wt_rewrite_logs/W11/test-qsa-idx-type.log`, rc 0, `test-qsa-idx-type: OK`):
   - mapping, 9 × 10 cases: qwen4exp auto sub-8-bit → q8_0, others unchanged, override wins; GLM5_NEXT = type_k always.
   - `create_memory`, 13 rows, checked on the actual indexer tensor type:
     - qwen4exp main q4_0 → q8_0, iq4_nl → q8_0, f16/q8_0/bf16 unchanged;
     - override q4_0 → q4_0, q8_0+override f16 → f16;
     - qwen4exp **MTP** q4_0 → q8_0 (layer 1);
     - **GLM5_NEXT main and MTP -ctk q4_0 → q4_0**, and with `--cache-type-k-idx q8_0` still q4_0 (warning `--cache-type-k-idx q8_0 is ignored for glm5-next`). This is the synthetic GLM5-Next load of the done-when: no GLM5-Next GGUF exists on strix-002 (`ls ~/models`).
   - session states, written by the production `state_write` and read by `state_read`, with 40 cells each:
     - old qwen4exp q4_0 state (q4_0 indexer) → rejected. The log shows `state_read_data: mismatched key type (8 != 2, layer 0)`, then the wrapped message. The cells of every cache are dropped.
     - The same state with `--cache-type-k-idx q4_0` loads.
     - qwen4exp f16 and q8_0, GLM5_NEXT q4_0 and f16, and a new qwen4exp q4_0 state all load.
     - An attention-section mismatch (`-ctk q8_0`) keeps `failed to restore kv cache` and is not wrapped.
2. **ctest** `-R "test-qsa|test-arg-parser|test-llama-context|test-batch-alloc"` -j1 under flock: **14/14 passed** (`ctest.log`), including:
   - test-qsa-prefix, -fast, -recover, -allvis-rows, -te-rows, -visibility, -resplit, -idx-type;
   - test-llama-context, test-arg-parser.
3. **Session files on the real model** (`harness/w11/session.sh`, Qwen3.8-Flash-Next UD-Q4_K_XL, `llama-completion --prompt-cache`, saved by W0-lab12 `~/dev/wt_lab/w0lab12/build/bin` with `-fa on`, loaded by this build; `~/dev/wt_lab/w11/session.out`, logs `session/*.log`):
   - `-ctk q4_0` → **rejected** by the existing check: `E state_read_data: mismatched key type (8 != 2, layer 3)`, followed by `E llama_state_load_file: error loading session file: failed to restore the sparse-attention indexer cache (first QSA layer 3): the session was saved with indexer type q4_0, this context uses q8_0; re-save the session, or load it with --cache-type-k-idx q4_0` (`session/load_w11_q4_0.log:18-19`). Layer 3 is the first QSA layer. The `incompatible key rotation` check did not fire first, because q4_0 and q8_0 are both quantized and get the same `n_rot_k` (`llama-kv-cache.cpp:339-367`).
   - The same file with `--cache-type-k-idx q4_0`: `loaded a session with prompt size of 48 tokens`.
   - `-ctk f16` and `-ctk q8_0`: both loaded, 48 tokens.
   - Session files (~118 MB each) deleted.
4. **KLD smoke** (`harness/w11/smoke_kld.sh`, c8192 b/ub4096, 1 chunk; KL base = the pre-W11 build `~/dev/wt_lab/w10/base_bin`, against which W10's smoke measured these same numbers; `~/dev/wt_lab/w11/smoke.out`):

   | arm | Mean KLD | Max KLD | Same top p | PPL |
   |---|---|---|---|---|
   | q4_0 auto (indexer q8_0) | 0.028154 ± 0.001406 | 2.708 | 96.068 % | 2.1271 → 2.124025 |
   | q4_0 `--cache-type-k-idx q4_0` | 0.000000 | 5.5e-5 | 100 % | 2.127075 (ratio 1.000044 ± 0.000044) |
   | f16 auto | 0.000000 | 5.8e-5 | 100 % | 2.108191 |
   | q8_0 auto | 0.000000 | 5.1e-5 | 100 % | 2.113919 |

   - The override, f16 and q8_0 rows equal W10's smoke rows digit for digit (this file, W10 "Smoke on the real model"). So W11 changes nothing except the indexer type.
   - The q4_0 auto row is the intended change. It is a one-chunk smoke and says nothing about quality: that is Q4's job (Moved to Prove).
   - KL bases deleted.
5. **Census** (W10 `census.sh`, graphs off, `OUT=~/dev/wt_lab/w11/census PAIRS=q4_0:q4_0`): ub4096 23/24 gated and ub512 184/192 gated, `EXPECT PASS 0 diffs`. These equal W10's census2. With `LLAMA_ARG_CACHE_TYPE_K_IDX=q4_0`, ub4096 is also 23/24, PASS (`census_ovr/`).
6. **MTP server smoke** (`harness/w11/mtp_smoke.sh`, Signal-3.8 + MTP head, `-ctk/-ctv/-ctkd/-ctvd q4_0`, c32768, request S0; `~/dev/wt_lab/w11/mtp_smoke.out`):
   - Both contexts log `indexer K cache type = q8_0 (auto, -ctk q4_0)`.
   - **The KV buffer lines show the indexer type**: `K (q8_0): 51.00 MiB` (main, 12 layers) and `K (q8_0): 4.25 MiB` (MTP, 1 layer). With the q4_0 override they are 27.00 / 2.25 MiB (`mtp_ovr/`). The ratio of 1.89 matches C9 d3 and PLAN-v3:385.
   - The QSA lines show `idx=q8_0` and `cpu=0` for every main/mtp graph, and the request completes (prompt 999, predicted 320, draft accepted 215/311).
   - census_check: every route key OK, except `gated share of the fusible G4 attends: 0/12 (floor 0.9)`. The override run gives the same 0/12 (`mtp_ovr/server_q4_0.log.expect`), so W11 does not cause it (see Notes).
7. **Done-when scorecard**

   | item | result |
   |---|---|
   | Q4 passes (floored bands, kind-1 partners) | **not run here** (owner scope note: c8192 × 16 pinned campaign) → Moved to Prove. The lab tooling is adapted so phase C runs on W11 (commit 6ad61a115) |
   | the KV buffer line shows the indexer type | **PASS** (test 6) |
   | old q4_0 qwen4exp session rejected through the existing per-layer check, with the wrapped message naming the first QSA layer | **PASS** (tests 1 and 3: `mismatched key type (8 != 2, layer 3)` + wrapped message, first QSA layer 3) |
   | f16/q8_0 qwen4exp files and a GLM5_NEXT file load | **PASS** for qwen4exp f16/q8_0 on the real model (test 3). GLM5_NEXT: synthetic state round trip only (test 1), because there is no GLM5-Next GGUF on strix-002, so a real GLM5_NEXT file saved on W0 is **UNPROVEN** |
   | `LLAMA_SESSION_VERSION` / `LLAMA_STATE_SEQ_VERSION` unchanged | **PASS** (`git diff 426040b7e HEAD -- include/llama.h`: the only change is the new `type_k_idx` field at :428, not a version) |
   | synthetic hybrid-idx load with GLM5_NEXT and -ctk q4_0 shows type_k_idx = q4_0 | **PASS** (test 1, main and MTP) |

### Moved to Prove (owner scope note 2026-11-07; harness / command and pass criterion)
| item | command | criterion |
|---|---|---|
| Q4 selection agreement on W11 | `scripts/qsa-validation/harness/w7/lab_tree.sh <W11 head or final head>`, then `scripts/qsa-validation/harness/w7/q_campaign.sh C` (records ref16t0 on W0-lab12, then the replay arms; the new q4_0 arm runs auto, f16k_q8idx gets `LLAMA_ARG_CACHE_TYPE_K_IDX=q8_0`), then `q_an.py` | PLAN-v3 §15 Q4 row: mean_c(new − PR219 kind-1) ≥ −B_T, i.e. f16 ≥ 0.99735, f16K q8 idx ≥ 0.99693, q8_0 ≥ 0.996625, **q4_0 KV (auto q8 idx) ≥ 0.985528**, bf16 ≥ 0.997140, q8_0K/f16V ≥ 0.996742, f16K/q8_0V ≥ 0.997121, with PR219 partners at `LLAMA_QSA_SELECTED_KEY=1` for q4_0, bf16 and the mixed pairs; plus q8 idx / f16 idx ≥ 0.99 on kind 1. Stage 2 on chunks 16–31 of c8192 × 32 if stage 1 fails |
| M1 iq4_nl MTP memory (TODO-LATER #6) | `scripts/qsa-validation/harness/w7/mem.sh` cell iq4_nl, c32768, ub4096, MTP (`-ctkd iq4_nl`), then `mem_an.py` | ROCm0 compute buffer ≤ f16's + 1 MiB (M1). W11 maps the iq4_nl indexer to q8_0, which removes the 128-wide IQ4_NL GET_ROWS case (`ggml-cuda.cu:9203-9209`) |
| Vulkan with the q8_0 indexer for q4_0 KV | `build-vk` llama-perplexity c8192 b/ub4096 1 chunk `-ctk q4_0 -ctv q4_0` vs `--cache-type-k-idx q4_0` (as `smoke_kld.sh` with `NEW=build-vk/bin`), then Q1-vk | runs with 0 CPU attention nodes; Q1-vk band B_vk |
| GLM5_NEXT real session file | only if a GLM5-Next GGUF becomes available: save on W0, load on the head | loads (indexer type unchanged) |

### Notes
- **Finding for the owner (it predates W11):** in the MTP llama-server run (Signal-3.8, `-ub 4096`, prompt 999 tokens), the census shows **0/12 fusible G4 attends gated**. The perplexity census of the same build gives 23/24, and so did W10's. The run with the pre-W11 indexer type (`LLAMA_ARG_CACHE_TYPE_K_IDX=q4_0`) also gives 0/12, so W11 does not cause it. That a W10-head binary gives 0/12 is UNPROVEN: no W10-head binary was kept, because `build/` is now W11. census_check.py says the gate fusion's overlap check depends on where gallocr places the MUL output (census_check.py:104-106). So on the server graph the gate is probably unfused (inferred, UNPROVEN). Prove should run `mtp_smoke.sh` on the final head and the W10 head, plus the server-path perf cells.
- No new TODO-LATER item. TODO-LATER #6 (M1 iq4_nl) is expected to clear here and is re-checked in Prove.
- Plan deviations:
  - The plan's "new `cparams.type_k_idx`" lives in `llama_context_params` and `llama_memory_params`, not in `llama_cparams`. It is used only at memory creation, as `type_k`/`type_v` are.
  - Saving the saved type in `llama_kv_cache` (`k_type_mismatch_ref`) is an in-memory addition: no stream change.
- Not touched: `~/dev/wt_pr219v2*`, other agents' worktrees, systemd units, strix-001. The only processes stopped were ones this item started: two llama-server runs, both stopped by their scripts. `pgrep llama-server` is empty afterwards.
- Cleanup: the session files and KL bases were deleted by the scripts. `find ~/dev/wt_lab/w11 ~/dev/wt_rewrite_logs/W11 -size +100M` is empty; `~/dev/wt_lab/w11` is 4.1 MB. A temporary `git archive` copy (`~/dev/wt_lab/w11/labcheck`) was removed.

## W13: delete the old code, CI baseline, merge-tree checks

**Status: implemented and committed (not pushed), head `1dbee4d9a`.** The HIP and Vulkan builds pass. Every correctness check run for this item passes:
- the per-op test-backend-ops subsets on ROCm0 and Vulkan0;
- ctest for the QSA tests;
- the guard harness;
- the KLD smoke, which reproduces W11 digit for digit;
- a Vulkan smoke.

The HIP quality check (`-Werror` build plus `gcn-cdna-vgpr-check.py --baseline`) passes in the pinned image against the regenerated baseline. The merge-tree results are recorded below. The full §14–§16.5 campaign and AT-C16b are moved to Prove, as the owner's scope note asks.

All commands ran on strix-002. Every build and GPU job ran under `flock ~/gpu.lock`, and no other agent was active. Logs:
- `~/dev/wt_rewrite_logs/W13/`: builds `hip_build.log` and `vk_build.log`, `build_rc.txt`, `tests/` (per-op logs and `summary.txt`), `merge_tree_final.txt`, commit messages `m1..m10.txt`;
- `~/dev/wt_lab/w13/`: `guard/` and `guard.out`, `smoke/` and `smoke.out`, `vk/` and `vk.out`, `vgpr/` (pinned-image logs, `check_old_baseline.log`, `check_new_baseline.log`, `baseline_before.json`).

There was no earlier W13 agent: the tree was clean at 6ad61a115 (`git status --short` was empty).

### What changed (files, functions), located by symbol
- **Tests** (`tests/test-backend-ops.cpp`): the four classes `test_qsa_prefill`, `test_qsa_decode`, `test_qsa_prefill_maskless` and `test_qsa_decode_maskless` are deleted, along with every registration of them.
  - QSA_SCORE, QSA_STRIP and QSA_EXPAND stay (PLAN-v3 §11 "Tests").
  - Comments that named the deleted classes or #219 kernels by symbol now name the source by commit and file: SA_ROWS_PR / PRBLK, the G1 band comment, and the ported-shapes comments.
- **HIP** (`ggml/src/ggml-cuda`):
  - Deleted the files `qsa-prefill.cu/.cuh`, `qsa-decode.cu/.cuh` and `qsa-decode-wmma.cuh`, which removes `QSA_DIRECT`, `GGML_QSA_RANKSORT` and `QSA_DECODE_MAX_QUERIES`.
  - `fattn.cu`: removed the QSA dispatch in `ggml_cuda_flash_attn_ext`, the maskless abort, the maskless branch of `ggml_cuda_flash_attn_ext_supported`, `ggml_cuda_qsa_gate_ok`, and the QSA exclusion in `ggml_cuda_fa7_gate_ok`. The matching declarations in `fattn.cuh` are removed too.
  - `ggml-cuda.cu`: the FA gate fusion keeps only fa7. The qsa-prefill gated branch, `fa_gate_qsa`, and the `src[5]` overlap read are removed. QSA layers use the SPARSE_ATTN gate fusion from W3.
  - `common.cuh`: removed `GGML_QSA_MAX_KEYS`.
  - The provenance comments in `sparse-attn-merge.cu` and `sparse-attn-attend-g1.cuh` now cite the #219 files and the commit (94e086847) instead of the symbols.
- **Vulkan**:
  - Deleted the shaders `flash_attn_sel.comp`, `flash_attn_sel_cm.comp` and `flash_attn_sel_prep.comp`, and their entries in `vulkan-shaders-gen.cpp`.
  - Deleted the three `pipeline_flash_attn_sel*` members (`ggml-vulkan-types.h`), the pipeline creation including `GGML_VK_SEL_CM`, and `vk_op_flash_attn_sel_push_constants`.
  - Deleted `ggml_vk_flash_attn_sel_pipeline`, `ggml_vk_flash_attn_sel`, and the maskless branches of the FLASH_ATTN_EXT dispatch and of supports_op.
  - DSA's `flash_attn_top_k` is untouched.
  - The comments of `sparse_attn.comp`, `sparse_attn_cm.comp` and `ggml_vk_sa_splits` now cite master ea5285ddf.
- **ggml**: `ggml_flash_attn_ext_add_top_k` gains `GGML_ASSERT(a->src[3] != NULL)` (`ggml/src/ggml.c`), and the `ggml.h` comment says so (PLAN-v3 §2.3). Its callers were re-checked by `grep -rn "build_attn_mha\|ggml_flash_attn_ext_add_top_k" src/models`:
  - deepseek4.cpp:760 passes `kq_mask` = `ggml_concat(raw_mask, csa_mask)`, which is non-null;
  - glm5-next.cpp:1083 passes no top_k;
  - qwen4exp no longer calls it;
  - `test_flash_attn_ext_top_k` passes a mask (`test-backend-ops.cpp:11452`).
- **llama**:
  - Deleted `LLAMA_QSA_DENSE_SHORTCUT` (`qwen4exp.cpp`, `build_layer_attn`).
  - Deleted `llm_graph_input_qsa::maskless`, which was set and checked exactly where `compact` was, with the same value. Its two readers now read `compact`.
  - Renamed `qsa_decode_error(s)` to `qsa_ubatch_error(s)` (llama-graph.h/.cpp, llama-context.h/.cpp, qwen4exp.cpp). The printed key `decode_errors=` is unchanged, because `harness/w9/concurrent.sh` reads it.
  - Already absent before W13: the probe, `LLAMA_QSA_SELECTED_KEY` and `QSA_MASKLESS_MAX_KV` (grep on `src/` at 6ad61a115, apart from the `common.cuh` comment deleted here), the masked half of `build_attn_qsa`, and the q/out rotation nodes (W8, W10).
- **Measurement-only env hooks**: removed from the branch and kept as `scripts/qsa-validation/patches/06-w13-measurement-hooks.patch`, the exact reverse of 0da383ddd. `git apply --check` passes on the head. The hooks:
  - `LLAMA_QSA_PREFER_DENSE`;
  - `LLAMA_QSA_SHORTCUT_PIN` and `LLAMA_QSA_HOST_KV_COPY`, whose shipped defaults are unchanged;
  - `LLAMA_QSA_FORCE_NONCOMPACT` and `QSA_NC_FORCED`;
  - `LAB_OVL_PRINT`.

  `harness/w7/lab_tree.sh` applies the patch on a W13+ tree. The new `harness/w13/hooks_tree.sh` builds a hooks tree for the Prove runs that set these variables, and the headers of `at_e2e_shortcut.sh`, `vk.sh`, `at_d.sh` and `perf.sh` say so.

  Kept on purpose:
  - test hooks used by in-tree tests: `GGML_CUDA_SA_FORCE` (test-backend-ops, test-sparse-attn-rot), `GGML_CUDA_SA_REJECT` (test-qsa-guard), `GGML_CUDA_SA_ROT_DUMP` (test-sparse-attn-rot), and `GGML_VK_SA_FORCE` and `GGML_VK_SA_CAP_MASK` (test-sparse-attn-vk);
  - forced-path switches for op runs: `GGML_CUDA_SA_PLAN_GLOBAL` and `GGML_VK_SA_MAX_Q`;
  - the census log, and the plan's `LLAMA_QSA_TRACE` (§14.3) and `LLAMA_STRICT_RESERVE` (D17).
- **CI baseline**: `scripts/hip/gfx1151-vgpr-baseline.json` was regenerated (details under Build). #234 is not in the base ea5285ddf, so the regeneration starts from the base's own baseline file and checker.
- **`tests/test-qsa-guard.cpp`**: the fake device's caps initializer now lists all 5 members. The pinned-image log showed `-Wmissing-field-initializers` there.

Nothing from #219 was ported in W13, so no commit carries a Co-authored-by trailer. The #219 pieces the rewrite keeps were ported earlier with the trailer (sparse-attn-q8.cuh, the G1 q8 K loader, the G4 block-w addressing). W13 deletes only the #219-era kernels that those ports replace.

### Commits (branch qsa-rewrite, not pushed; all by Gaetan Puleo <12990773+gaetan-puleo@users.noreply.github.com>, `git log --format='%h %an <%ae>' 6ad61a115..HEAD`)
- `08b431a31` tests : delete QSA_PREFILL, QSA_DECODE and their _MASKLESS cases
- `d14c6c337` hip : delete the QSA FLASH_ATTN_EXT kernels (qsa-prefill, qsa-decode) and the maskless FA dispatch
- `740d48097` vulkan : delete the maskless selected-key FA (flash_attn_sel, _cm, _prep)
- `74fc732f3` ggml : ggml_flash_attn_ext_add_top_k asserts a mask
- `bee4e2076` qwen4exp : delete LLAMA_QSA_DENSE_SHORTCUT and llm_graph_input_qsa::maskless
- `0da383ddd` qsa : measurement-only env hooks move out of the branch (lab patch 06)
- `953b794a0` scripts : lab patch 06, the measurement-only env hooks W13 took out
- `8d1faf602` hip : regenerate the gfx1151 VGPR baseline for the sparse-attn TUs
- `11b42ee8e` tests : test-qsa-guard initializes every ggml_backend_dev_caps member
- `1dbee4d9a` scripts : W13 harnesses (op tests, VGPR baseline in the pinned image, merge-tree, smoke) and summary

The tests are deleted first and the assert comes last, so no intermediate commit builds a maskless FA + src[5] while the assert is in place. Only the final head was built and tested. That each intermediate commit builds on its own is UNPROVEN.

### Build
- **HIP** `build/` (rule flags; `~/dev/wt_rewrite_logs/W13/build_rc.txt`): configure rc 0, build rc 0.
  - The 86 `warning:` lines in `hip_build.log` are all in base or unchanged code: the `u8path` deprecations in common/ and tests/gguf-model-data.cpp, `ggml_flash_attn_ext_set_prec` deprecated at test-backend-ops.cpp:11455, the enum-bitwise warnings at test-backend-ops.cpp:17332-17341, and `ld.lld ... loop not unrolled`.
  - Nothing warns in a file W13 changed, except test-backend-ops.cpp at lines W13 did not touch.
- **Vulkan** `build-vk/`: build rc 0. The 76 warnings are the same set plus two in base code that this full recompile of ggml-vulkan.cpp shows:
  - ggml-vulkan.cpp:10707 unused `coopmat2` (`git blame`: 7f7438e79c, Nathan Wilson 2026-07-28);
  - :9640 unused `dst_buf_ctx` (89ef7d894d, 2026-08-06).

  Neither line comes from this branch.
- **HIP quality check, pinned image** (`harness/w13/vgpr_in_image.sh`). The image is podman `docker.io/rocm/dev-ubuntu-26.04@sha256:4f5ed1bf6532a4b9920400401b0ad0f706af356dae61e40afefbfd6c8073f0ce`, pulled for this item (`podman_pull.log`; local image id 90fa45f2d0fb, 22.1 GB). The source is a `git archive` of 953b794a0, and the run has its own clean build directory (`~/dev/wt_lab/w13/vgpr/out/rc.txt`, all rc 0). The steps follow `hip-quality-check.yml`, with apt packages as in CI:
  - **`-Werror -Wno-tautological-compare` build (EXPORT_METRICS Off): rc 0**, 167 HIP objects. The 17 `warning:` lines are gcc C++ warnings (test-backend-ops.cpp ×2, test-qsa-guard.cpp ×1, fixed in 11b42ee8e, stl_algobase ×1) and 13 `ld.lld loop not unrolled`. None is fatal under the CI flags, which apply `-Werror` to HIP sources.
  - `test_vgpr_check.py`: 18 tests OK.
  - EXPORT_METRICS On build: rc 0, 167 `*.lto.s` files.
  - Checker against the **old** baseline: rc 1, with `qsa-decode: kernel coverage 0 < 4` and `qsa-prefill: kernel coverage 0 < 6` (`check_old_baseline.log`). This is the failure the plan predicted (gcn-cdna-vgpr-check.py:75-79).
  - Regeneration: the scripts/hip/README.md python, verbatim (`harness/w13/vgpr_regen.sh`), run at HEAD = 953b794a0, the built source.
    - `minimum_kernel_count` goes from 8155 to 8868, the TUs from 154 to 162, and the copies from 8763 to 9476.
    - Removed: qsa-decode 4 and qsa-prefill 6.
    - Added: sparse-attn-attend-g1 96, -g4-{bf16,f16,q4_0,q8_0} 104 each, -generic 19, -merge 1, -pack 5, -plan 7, -portable 1.
    - Other TU counts grew, and each change comes from the base (6805faa86 is an ancestor of ea5285ddf, and each file has 3-16 commits in 6805faa86..ea5285ddf): fattn 38→41, gated_delta_net 30→35, hyperconn 8→12, mmb 318→473, mmid 12→18, rope 48→50, unary 192→195. This branch only deletes lines in fattn.cu and does not touch the others (`git diff --stat ea5285ddf HEAD`).
  - Checker against the **new** baseline: **`PASS: zero VGPR spills with complete baseline coverage.`** It parsed 8868 kernels across 162 TUs, with 0 spilling symbols (`check_new_baseline.log`).
  - **Deviation:** only the regeneration python and the checker ran on the host (python3 reading the image's `.lto.s`). The compiler string is identical in both places: `AMD clang version 24.0.0git (https://github.com/ROCm/llvm-project.git d6f6cb691863e9258034ab48c1dfce21689d5606)` (`compiler_version.txt` from the image, and `clang --version` on the host).

### Tests run (correctness; outputs)
1. **test-backend-ops**, per op, one process per op, under flock (`harness/w13/optests.sh`; `~/dev/wt_rewrite_logs/W13/tests/summary.txt`, per-op logs alongside):

   | backend | SPARSE_ATTN_PLAN | SPARSE_ATTN | FLASH_ATTN_EXT | QSA_EXPAND | QSA_SCORE | QSA_STRIP |
   |---|---|---|---|---|---|---|
   | ROCm0 | 185/185 | 1390/1390 | 4051/4051 (1343 not supported) | 56/56 | 13/13 | 6/6 |
   | Vulkan0 | 92/92 (93 not supported: G4 is on the Vulkan expected-unsupported list) | 744/744 (646 not supported) | 5383/5383 (11 not supported) | 56/56 | 13/13 | 6/6 |

   - FLASH_ATTN_EXT covers the DSA top-k cases, which keep their mask, so the new assert does not fire there.
   - SPARSE_ATTN and SPARSE_ATTN_PLAN include the determinism and census-coverage checks of §14.1, unchanged.
   - Whether these not-supported counts equal those of the pre-W13 build is UNPROVEN. No pre-W13 build was kept, and W13 deletes no supported case of these ops: the deleted supports branches covered only `src[5] && !src[3]`, which only the deleted tests built.
   - `-o QSA_PREFILL`, `QSA_DECODE` and both `_MASKLESS` names list 0 cases.
2. **ctest** `-R "test-qsa|test-sparse-attn"` -j1: **11/11 on HIP and 11/11 on Vulkan** (`ctest_hip.log`, `ctest_vk.log`).
3. **Guard harness** (W7 `guard.sh` with its output redirected to `~/dev/wt_lab/w13/guard`; `guard.out`), all as in W7:
   - pass q8_0/f16/q4_0/bf16: PASS;
   - fake device: PASS, init error "...not supported on device FAKE0 ... no sparse-attention kernel in this backend; run QSA on the CPU with -ngl 0 or --no-kv-offload...";
   - reject and reject-decode: PASS (`GGML_CUDA_SA_REJECT`);
   - allvis: `RESULT PASS (failures 0)`;
   - split-mode tensor and the n_ctx cap: init errors.
4. **KLD smoke** on the real model (`harness/w11/smoke_kld.sh`, `OUT=~/dev/wt_lab/w13/smoke`, base = `~/dev/wt_lab/w10/base_bin`; `smoke.out`):
   - q4_0 auto: KLD 0.028154 ± 0.001406, PPL 2.124025;
   - q4_0 override: KLD 0, PPL 2.127075;
   - f16: KLD 0, PPL 2.108191;
   - q8_0: KLD 0, PPL 2.113919.

   Every number equals the W11 smoke row digit for digit (this file, W11 test 4). So W13 changes nothing that the HIP graph computes. KL bases were deleted by the script.
5. **Vulkan smoke** (c8192, b/ub 4096, 1 chunk, `LLAMA_QSA_TRACE=1`; `~/dev/wt_lab/w13/vk.out`, `vk/ppl_*.log`):
   - PPL: q8_0 2.1006 and f16 2.0936. These are the same 4-digit values as the W7 Vulkan logs `~/dev/wt_lab/w7/vk/logs/c_{q8_0_q8_0,f16_f16}_u4096.log`. That those runs used identical arguments is UNPROVEN, because the logs do not record their command line.
   - Trace: 36 rows each, all `backend=Vulkan0`, 24 `route=sparse`.
   - Census: `sa_attend/vk/cm/direct/<T>/<T> 24` and `sa_plan/G1 24`.
6. **Done-when greps** (on 1dbee4d9a):
   - `grep -rn "qsa_decode\|qsa_prefill\|flash_attn_sel\|QSA_DIRECT\|LLAMA_QSA_SELECTED_KEY\|QSA_MASKLESS_MAX_KV\|LLAMA_QSA_PREFER_DENSE" --exclude-dir=build --exclude-dir="build-*" --exclude-dir=.git --exclude-dir=qsa-validation .` → **empty (rc 1)**.
   - An extended grep for `GGML_QSA_RANKSORT`, `GGML_QSA_MAX_KEYS`, `QSA_DECODE_MAX_QUERIES`, `GGML_VK_SEL_CM`, `selected_key_attn`, `LLAMA_QSA_DENSE_SHORTCUT`, `LLAMA_QSA_FORCE_NONCOMPACT`, `LAB_OVL_PRINT`, `LLAMA_QSA_HOST_KV_COPY`, `LLAMA_QSA_SHORTCUT_PIN` and `qsa_gate_ok` over ggml/src/tests/common/tools/include/.github is **empty**. The only remaining mentions are the hyphenated #219 file names (`qsa-prefill.cu`, `qsa-decode*.cu*` "of 94e086847") in provenance comments.
   - **Deviation:** the 30 files under `scripts/qsa-validation` that match are the PR219-arm harnesses and evidence (they build and measure the PR219 reference). Prove needs them, so they are excluded from the grep and kept.
   - `grep -rn "ctx.pool()" ggml/src/ggml-cuda/sparse-attn-*` → **empty (rc 1)**.
7. **CUDA CI**:
   - `gh run view 41354919590`: "CI (CUDA, ubuntu)" at ee30b949e, conclusion failure, failed jobs `cuda` and `musa`. It is pre-existing and must be recorded in the PR text.
   - The current master head 62a3f4144 (run 42326610175) also fails `cuda` and `musa`.
   - Master's "HIP quality check" last ran with success at b96cba366 (41201114479), and failed at 27849371cc (42150234495, event pull_request).
8. **merge-tree** (`harness/w13/merge_tree.sh`; `~/dev/wt_rewrite_logs/W13/merge_tree_final.txt`; head 1dbee4d9a; refs fetched with `git fetch origin master` and `pull/N/head`). #234 is **merged** (2026-11-07, merge commit c1529aed0, in origin/master 62a3f4144, `gh pr view 234`). The others are open.

   | ref | head | result |
   |---|---|---|
   | origin/master | 62a3f4144 | rc 1: `qsa-prefill.cu` modify/delete (deleted here, modified by #234), `ggml-vulkan.cpp` 3 text hunks (around the deleted sel pipeline creation and supports_op), `scripts/hip/gfx1151-vgpr-baseline.json` 6 hunks, `tests/CMakeLists.txt` 1 hunk |
   | #234 | 0ec95bc5c | rc 1: the same four files |
   | #225 | e6849cfc6 | clean |
   | #247 | 9b7c19212 | rc 1: `ggml/src/ggml-cuda/ggml-cuda.cu` 1 hunk |
   | #248 | 26dbe6369 | rc 1: `tests/CMakeLists.txt` 2 hunks |
   | #230 | a3d51093b | clean |

   Consequences for the orchestrator:
   - Under PLAN-v3 §0 (3), whichever of #234 and this PR lands second rebases. #234 has landed, so this branch must rebase onto master to merge. The resolution is mechanical for `qsa-prefill.cu` (keep the deletion). The VGPR baseline must be regenerated again on the rebased tree with `harness/w13/vgpr_in_image.sh` and `vgpr_regen.sh`.
   - The §0 rebase protocol then re-runs the W0 done-when, the W12 control, the band calibrations and the Q1/Q4 references on the new W0-lab.
   - That rebase is the owner's call and was not done here.

### Moved to Prove (owner scope note 2026-11-07; harness / command and pass criterion)
| item | command | criterion |
|---|---|---|
| Full §14–§16.5 campaign on the final head (every census row, Q1–Q12, P1–P9, M1–M6, AT-*), the W13 done-when | the harnesses of W7–W11's "Moved to Prove" tables, run on the final head. Runs that set the measurement hooks (AT-W7PERF, AT-B4, AT-D1, AT-D4, AT-C12 / U38) use `scripts/qsa-validation/harness/w13/hooks_tree.sh <final head>` and the binaries in `~/dev/wt_lab/w13/hooks/build{,-vk}/bin`. The Q campaign lab tree (`harness/w7/lab_tree.sh`) applies patch 06 itself | the criteria of PLAN-v3 §14–§16.5 and §15.0 (bands of W12), each as listed in its W-item's table |
| AT-C16b on the final head | `scripts/qsa-validation/harness/w8/at_c16a.sh` with `05-c19-log.patch` and `LLAMA_STRICT_RESERVE=1`, at `--ctx-size 524288`, share vs no-share (see W9's "AT-C16b") | PLAN-v3 §16.5 AT-C16b: census new 1 / reuse 11 per src per graph; reserve-to-reserve no-share − share ≥ 1056 MiB at c131072 (owner decision pending on the criterion, W9); runtime peak ≤ reserve |
| HIP quality check after the rebase onto master (#234 merged) | `harness/w13/vgpr_in_image.sh` in the pinned image on a `git archive` of the rebased head, then `harness/w13/vgpr_regen.sh` | `-Werror` build rc 0; `PASS: zero VGPR spills with complete baseline coverage` |
| merge-tree on the final (rebased) head | `scripts/qsa-validation/harness/w13/merge_tree.sh` | list every conflicting file in the PR description (§0 (4)) |
| Pre-existing items (TODO-LATER 1-8, W7–W11 Prove tables) | unchanged | unchanged |

### Notes
- **Owner-facing finding:** master moved under the branch. #234 and #259 merged after the base (`git log --first-parent ea5285ddf..origin/master`: 62a3f4144 #259, c1529aed0 #234). The PR as it stands therefore conflicts with master in the four files above.
- Open PRs #270 ("vulkan : maskless selected-key attention with a quantized K/V cache") and #271 ("qwen4exp : maskless block selection with a quantized K/V cache") appeared after the plan was written (`gh pr list`). Their heads were fetched (`pr/270`, `pr/271`) but **not merge-tree'd**: the plan's list is #234/#225/#247/#248/#230. They target the maskless FA path this item deletes, so the owner should know that they overlap this PR in intent. Their overlap in code is UNPROVEN.
- The pinned image (22.1 GB, `podman images`: docker.io/rocm/dev-ubuntu-26.04 `<none>` 90fa45f2d0fb) is kept for the Prove re-run after the rebase. Remove it with `podman rmi 90fa45f2d0fb` when it is no longer needed.
- Cleanup:
  - The pinned-image build dir (4.4 GB) and the source export (187 MB) were deleted; `metrics*.log` were gzipped.
  - `~/dev/wt_lab/w13` is 17 MB, and `find ~/dev/wt_lab/w13 -size +100M` is empty.
  - The smoke script deleted the KL bases.
- Not touched: `~/dev/wt_pr219v2*`, other agents' worktrees, systemd units, strix-001. No process was killed. Nothing was pushed or posted (the only `gh` calls were the read-only `pr list` / `pr view` / `run view` / `run list`).

## Prove-fix 1: fixes for PROOF-initial.md §2 (head `97865cacb`)

**Status: 4 commits on `qsa-rewrite` (not pushed), on top of `1dbee4d9a`. HIP and Vulkan builds pass with rc 0 and 0 warnings. The CPU unit tests pass, including a new one. AT-C18 was re-scored on Prove's own logs and passes. No GPU check could be rerun: the GPU lock `~/gpu.lock` is still held by the Vulkan process that hung during Prove (PROOF-initial §13). That process is in state D and holds 91.6 GB GTT. So every GPU-side verdict below is NOT RUN, and the rerun harness is ready (see "Reruns").**

Facts at the start of this step (strix-002, 2026-11-10 13:12):
- `ps -o pid,stat,etime,wchan:30 -p 4041225` → `D 01:17:52 drm_suballoc_insert`, cmd `build-vk/bin/llama-perplexity ... -ctk bf16 -ctv bf16 --kl-divergence-base .../c1/det_model/x_2.kld`;
- `flock -n ~/gpu.lock true` → LOCK_HELD;
- `mem_info_gtt_used` → 91631312896;
- `free -g` → 31 GB available of 124.

These were the same at the end (`flock -n` → LOCK_HELD at the last check). This agent did not signal the process (the rule: never kill processes you did not start), and it has no root to reset the GPU.

**Builds ran without `flock ~/gpu.lock`.** A build is not a GPU command (hipcc compiles offline for `GPU_TARGETS=gfx1151`), and the lock cannot be taken while the hung process holds it. This agent was the only one on strix-002, as the task states. No GPU command was run. The CPU unit tests ran from the CPU-only tree `build-cpu` (`GGML_HIP=OFF`, `GGML_VULKAN=OFF`, `build-cpu/CMakeCache.txt`), so they do not open the GPU device.

Logs:
- `strix-002:~/dev/wt_rewrite_logs/PF/`: `build_hip.log`, `build_hip2.log`, `build_vk.log`, `build_cpu_tests.log`, `cfg_cpu.log`, `ut_cpu_*.log`;
- patch scripts, commit messages and the rerun harness: `strix-002:~/dev/wt_lab/prove_fix/` (`p1_onestream.py`, `p2_guardlog.py`, `p3_ata.py`, `p4_gate.py`, `msg/c1..c4.txt`, `commit.sh`, `rerun.sh`, `h/`).

### Scorecard against PROOF-initial §2

| # | Prove failure | this step | verdict now |
|---|---|---|---|
| 2.1 | reserve < runtime, HIP `-np 2` split ub8192 (+22.6 MiB, 10 violations per type) | **fixed in code** (a16582997): one-stream reserve passes | NOT RUN on GPU (rerun `np2`) |
| 2.2 | reserve < runtime, Vulkan ub16, f16/f16, bf16/bf16, f16K/q8_0V (+44,800 B) | analysed, **not fixed** (below) | open |
| 2.3 | MTP census gated share 56/96 | **fixed in code** (97865cacb): gate fusion writes to scratch on overlap | NOT RUN on GPU (rerun `mtp`) |
| 2.4 | AT-C18 guard-on-probe FAIL (4) | **fixed** (f12eb329d): cache-only guard line + checker | **PASS** on Prove's logs (re-scored); new log line NOT RUN |
| 2.5 | Q7 MTP acceptance | deferred by the owner (TODO-LATER #9: "no prove-fix work on it") | not worked on |
| 2.6 | Q10 control floor, image p3 seq1 | not fixed (below) | open |
| 2.7 | Q12 coarse q8_0 0.015717 > 0.01563 | not fixed (below) | open |
| 2.8 | AT-A one cell (V q8_0, class 135168: 96 shipped vs 64 measured) | **fixed** (4552df2a4): the table encodes the fitted rule | criterion PASS by construction (16/16 cells = `CPP` line); e2e cell NOT RUN |
| 2.9 | P5 MTP f16 default d32k 0.898 | not fixed (below) | open |
| 2.10 | stall 1/3232 | not fixed (needs a rerun) | open |
| 2.11 | unresolved perf cells (§16 2× reruns) | not run (GPU) | open |
| 2.12 | AT-B2 not runnable | not fixed (below) | open |
| 2.13 | AT-C17 PR219 pool fill rejected | cause found, not run (below) | open |
| 2.14 | Vulkan q8_0 depth, AT-B1, AT-B4 e2e, AT-B5, Vulkan determinism bf16 / mixed | not run: the GPU is still hung | open |
| 2.15 | AT-A e2e cell, U30 masks on Vulkan, AT-C12 image-slot KLD, d131072 D1V, 1M depth | not run (GPU; 1M is a host-OOM risk) | open |

### What changed (files, functions)

1. **One-stream reserve in a split multi-sequence context** (a16582997, Prove §2.1).
   - **Cause.** With split streams (`kv_unified = false`) and `n_seq_max > 1`, every reserve graph viewed all `n_seq_max` streams. The dummy slot info of the full-cache context holds every stream (`llama_kv_cache_context(kv)`, `src/llama-kv-cache.cpp:3029-3044` at 1dbee4d9a), and `get_k` / `get_v` view `sinfo.s1 - sinfo.s0 + 1` streams (`llama-kv-cache.cpp:1385, :1406`). The hybrid-idx context's `get_n_stream()` was the cache's stream count (`ns_ubatch`, `llama-memory-hybrid-idx.cpp:1303-1304` at 1dbee4d9a).
   - A server prompt ubatch holds one sequence, so at runtime it is a one-stream graph of up to `n_ubatch` rows. No reserve graph had that shape:
     - the reserve log says `QSA reserve passB: n_tokens = 4096, n_seqs = 2` and `noncompact: n_tokens = 512, n_seqs = 2`;
     - there are no passA passes, because they need `kv_unified || n_seqs == 1` (`llama-context.cpp:1326` at 1dbee4d9a);
     - every violation is on `ubatch n_tokens=8192 n_seqs=1 n_outputs=0 ... n_stream=1` (`PV/img/srv_np2split_f16.log:9406-9735`);
     - the last violations are f16 3144.30 > 3122.75, q8_0 3144.56 > 3123.02 and q8_0K/f16V 3144.55 > 3123.00 MiB, each +22594688 B with 10 violations (`grep 'STRICT_RESERVE violation' PV/img/srv_np2split_*.log`).
   - **Change.**
     - `llama_kv_cache_context(kv, n_kv, n_stream = 0)` (`src/llama-kv-cache.cpp:3046-3058`): with `n_stream > 0`, it keeps the first `n_stream` streams of the dummy slot info.
     - The parameter is carried by `llama_memory_hybrid_context(mem, n_kv, n_stream)`, by `llama_memory_hybrid_idx_context(mem, n_kv, n_stream)` (where `ns_ubatch = {n_stream}`, so the graph's `get_n_stream()` is 1) and by `llama_memory_hybrid_idx::init_full_n_kv(n_kv, n_stream)`.
     - `sched_reserve` (`src/llama-context.cpp:1364-1379`): when `!kv_unified && n_seq_max > 1`, four passes run on the one-stream view at the full per-stream n_kv:
       - `1s-pp`: n_ubatch rows;
       - `1s-0out`: 0 outputs, the shape of the violating ubatches;
       - `1s-trim`: when n_ubatch > 512, sequential positions and the last row the only output;
       - `1s-fast`: the fast-cache variant.
     - The trim and the fast cache both need `n_stream == 1` (`qwen4exp.cpp:1511`, `:1969`), which a one-sequence ubatch of a split cache has. Each pass enters the fit probe's max and the placement guard, like the other QSA passes (`llama-context.cpp:1381` loop, `:1428` guard loop).
     - Unified contexts and `n_seq_max == 1` get no new pass.
   - **Test.** `tests/test-qsa-reserve-view.cpp` (CPU, the production hybrid-idx memory of `qsa-mem-fixture.h`) checks the stream count and n_kv of `init_full`, of pass B (`init_full_n_kv(kv/2)`) and of the one-stream view, for split 2 and 4 sequences, unified, and one sequence, plus clamping. Output: `views kv_size=4096 n_seq_max=2 unified=0: full 2 streams, pass B 2 streams, one-stream 1`, `n_seq_max=4 ...: full 4 ..., one-stream 1`, `unified=1: full 1`, then `test-qsa-reserve-view: OK` (`build-cpu/bin/test-qsa-reserve-view`, rc 0).
   - **What this does not prove.** The new passes give the reserve the runtime shape (one stream, 8192 rows, 0 outputs) at the largest per-stream n_kv. Whether every runtime graph of that shape at a smaller n_kv then stays within the reserve is the strict rerun's job (`rerun.sh np2`), because gallocr plans each new graph topology from scratch (see 2.2). The reserve grows by whatever the one-stream graph needs (≥ 22.6 MiB at ub8192, from the violations), which affects the np2 rows of M5 / AT-C12. UNPROVEN until the rerun prints `QSA reserve 1s-*` and the compute buffer sizes.

2. **Guard line of a cache-only graph, and the AT-C18 checker** (f12eb329d, Prove §2.4).
   - **Cause.** At ub ≥ 4096 with one sequence, the MTP context builds a cache-only graph for a no-output prompt ubatch (`src/models/qwen4exp.cpp:497-501`, `cache_only`). That graph stores K / V and has no attention node. `qsa_guard` sets `cache_only`, requires 0 nodes and returns "" (`llama-context.cpp:966`, `:1066`). The log line, however, printed `std::max(qsa_guard_last_required, 1)` and `route=dense-shortcut` (1dbee4d9a `llama-context.cpp:1107-1115`), so it read `graph=pp-0out route=dense-shortcut ... backend=none plan=0 attend=0 ... guard=0/1` (`PV/c18/srv_f16_u4096.log`). `at_c18_an.py` tests `inspected >= required` per probe line, so it failed exactly the four ub4096 runs. At ub512 the same pass has attention (`guard=2/2`, `srv_f16_u512.log`), because `cache_only` needs `n_ubatch >= 4096`.
   - Prove's reading of the `passB ... guard=12/12` line as "< 2 × layers" comes from the checker's label: the code compares inspected with the guard's required count (12 for a shortcut graph: one attend per layer).
   - **Change.**
     - `qsa_guard_reserve` prints `route=cache-only` when the graph has no attention node, prints the guard's own required count, and does not look up a backend for a missing attend node (`llama-context.cpp:1107-1117`).
     - `at_c18_an.py` requires `inspected >= required` on every probe line, and accepts required 0 only on a cache-only line: `route=cache-only`, or `plan=0 attend=0` in logs written before this change. The label is corrected.
   - **Re-score on the Prove logs** (`strix-002:~/dev/wt_lab/prove/c18`, no GPU):
     - the 1dbee4d9a checker (`git show 1dbee4d9a:...at_c18_an.py`) gives `AT-C18 FAIL (4)` (f16/q8_0 × u4096 ± nkvo);
     - the new checker gives **`AT-C18 PASS`**: 32/32 memory rows (unchanged), and 8/8 runs with every guard line inspected ≥ required.
   - The new log line itself is NOT RUN (rerun `c18`).

3. **AT-A cell** (4552df2a4, Prove §2.8).
   - `ggml_cuda_sa_g1_beats_g4`'s table, row V q8_0, column 135168, goes from 96 to 64 (`ggml/src/ggml-cuda/sparse-attn-attend-g1.cu:71`).
   - The source is Prove's re-fit on the head's kernels: `PV/w4bands/ata.md`, line `CPP { ..., { 32, 32, 32, 64 }, ... }`, with noise p90 0.0051 over 1188 samples and margin 0.9848. The deciding row is `wikiL_p131072 | 135168 | 96 | q8_0 | ... | t_G1 / t_G4 0.994 | no` (`ata.md:198`).
   - The other 15 cells already equal the `CPP` line. `ata.md` is now in the branch as `scripts/qsa-validation/summaries/prove_ata.md`.
   - Effect: a q8_0-V layer at n_kv > 94118 (the class boundary, `sparse-attn-attend-g1.cu:78`) with 65-96 queries per stream now takes G4.
   - `git grep` finds no other copy of the table (HIP only; Vulkan has its own policy).
   - The AT-A e2e confirmation cell is NOT RUN.

4. **Gate fusion that does not depend on gallocr's placement** (97865cacb, Prove §2.3).
   - **Cause.** The fusion (`ggml-cuda.cu:6764-6843`) was refused when the MUL output, or the BF16 copy for the next mmb reader, overlapped a buffer the attend reads (Q, K, V, plan, gate, V-pack workspace).
   - Q is dead after the attend, so gallocr may hand its memory to the MUL. `census_check.py` already notes that this makes the gated share allocation-dependent (`scripts/qsa-validation/census_check.py:105-107`).
   - In `PV/mtp/server_c_default.log`, all 96 G4 attends are in the main context: 8 ubatch graphs × 12 layers (`LLAMA_QSA_TRACE ... G=4`, n_q 995-8192). The census splits them `packed/gated 56`, `packed/plain 40`. A refusal for a structural reason (output flag, a second reader, the pattern) would fail whole graphs of the same structure, and 40 is not a multiple of 12. The other refusal conditions are shape- or graph-structure-based, so the overlap test is the allocation-dependent one. That the 40 refusals are all overlap refusals is still UNPROVEN until the rerun.
   - **Change.** When an output overlaps a read, the gated attend writes to `ggml_cuda_pool_alloc` scratch, which is outside the compute buffer. One `cudaMemcpyAsync` on the same stream then moves the result to `mul->data` or `dst16` (`ggml-cuda.cu:6805-6841`).
   - The kernel, its arguments and the non-overlapping case are unchanged, so outputs that were fused before are bit-identical by construction.
   - The cost is one device copy of the MUL output, only where the fusion was refused before. Before, that case ran the unfused attend, then SIGMOID, then MUL.
   - Pool high-water (M2) with the scratch: UNPROVEN (rerun `census` / M2).

### Not fixed, with the reason

- **2.2 Vulkan ub16.** Every violation is on the runtime graph `n_tokens=16 n_outputs=16 producer=cb n_stream=1 nodes=7158` (`PV/c1/vkcensus/c_f16_f16_u16.log:12-21`).
  - The excess grows by 4096 B per 256-cell n_kv step, i.e. 16 B per cell: +7936 B at n_kv 4352, then up to +44800 B at 6656. It then stops: there are no new violations up to n_kv 8192, although n_kv keeps growing within the chunk.
  - 16 B per cell is the size of the scorer's F32 [n_blocks, 16] tensors (n_blocks = n_kv / 4, `qwen4exp.cpp:1271`, `:1640-1654`).
  - At ub16 every reserve graph is already at the full n_kv 8192:
    - `pp`, the compact CB route: the zero-position reserve ubatch passes `qsa_compact_visibility_stream` on empty cells, and `score_key_limits` is empty below 128 tokens (`qwen4exp.cpp:1147`);
    - `passA-fast`, the fast route, with positions at the end of the context and 16 outputs.
  - Every n_kv-dependent tensor of these routes is non-decreasing in n_kv, including the plan (`ggml_sparse_attn_plan_get_layout`, `ggml.c:6651-6669`).
  - gallocr plans a graph from scratch whenever its node count or a node size differs from the last plan (`ggml_gallocr_needs_realloc`, `ggml-alloc.c:1293-1333`; `reallocs=34` in the log). A fresh plan's peak is not monotone in the tensor sizes, so a smaller graph can peak higher. The non-monotone pattern (growth up to 6656, then none) fits that.
  - So the most likely cause is allocator fragmentation, not a missing reserve shape. This is **UNPROVEN**: no GPU run could log the per-plan peak.
  - Only K f16 / bf16 violate. q8_0 / q4_0 K add the K Hadamard nodes, a different topology (UNPROVEN as the reason).
  - A fix needs either an allocator change (reuse the reserved plan for a smaller graph of the same topology) or a reserve at several n_kv per graph. Neither was made here. The first needs a GPU run to show that it holds.
- **2.6 Q10.** The control floor is degenerate: SD 0, because PR219 is deterministic at L 8192 (PROOF §5.8). Prove marks it as the owner's calibration decision. Image p3 seq1 (3.408e-3 > τ 1.553e-3) needs logits runs on the GPU to attribute. No code cause was identified.
- **2.7 Q12 coarse q8_0** (0.015717 ± 0.000542 > 0.01563). This is an unpinned screen of `-nkvo` (host-kv-copy route) against the same binary's GPU route. f16 passes (0.014491), and the pinned Q12 passes for q8_0 (PROOF §5.3). The margin over the threshold (+8.7e-5) is 0.16 SE. Separating route noise from a q8_0 copy-route effect needs GPU reruns. Not fixed.
- **2.9 P5 MTP f16 d32k 0.898.** The accepted drafts differ (70/386 vs 76/341) and so do the texts. Attributing it needs GPU runs. Not fixed. Note: 4552df2a4 does not touch f16 V, and 97865cacb only changes cases where the fusion was refused.
- **2.10 / 2.11.** These need GPU reruns (§16 with 2× processes). Not run.
- **2.12 AT-B2.** It compares against the `sel_cm` pipeline that W13 deleted (740d48097). An in-build reference would mean restoring deleted code. Not done: it needs an owner ruling (keep AT-B2 as NOT APPLICABLE, or compare against the W12 build).
- **2.13 AT-C17 fill.** The server capped the slot at the training context: `the slot context (524288) exceeds the context the model can serve (262144 ...) - capping`, then `request (266240 tokens) exceeds the available context size (262144 tokens)` (`PV/c17/srv_pr_fill266k.log:423, :480`).
  - The fill needs a linear `--rope-scale 2`, which the cap honours (`tools/server/server-context.cpp:4866-4872`), or `llama-bench -d 266240` with the C7 shim.
  - Not run (GPU).
- **2.14 / 2.15.** Not run: the GPU lock is held by the hung process. The 1M-depth runs also risk the host OOM killer again (PROOF §13: it killed desktop processes twice).

### Commits (branch qsa-rewrite, not pushed; `git log --format='%h %an <%ae> | %s' 1dbee4d9a..HEAD`)
- f12eb329d `llama : QSA guard line of a cache-only reserve graph; AT-C18 checker` (2 files, +14 −6)
- a16582997 `llama : reserve one-stream QSA graphs in a split multi-sequence context` (9 files, +108 −17, includes the test)
- 4552df2a4 `hip : G1 policy cell V q8_0 at the 135168 n_kv class re-fitted to 64` (2 files, +368 −1, includes `summaries/prove_ata.md`)
- 97865cacb `hip : sparse-attention gate fusion writes to scratch when the output overlaps a read` (1 file, +19 −6)

All are by Gaetan Puleo <12990773+gaetan-puleo@users.noreply.github.com>, with the session trailer. **No Co-authored-by**: none of the four ports #219 code. The reserve passes, the guard and its log line, the G policy table and the gate fusion are this branch's own (W3, W4, W7, W9). Each commit touches a disjoint set of hunks. f12eb329d's `llama-context.cpp` is 1dbee4d9a plus the guard-line hunks only (`commit.sh`). Only the final tree was compiled (below). The intermediate commits were not built one by one: UNPROVEN that each builds alone, though each one's hunks only use symbols that exist at its parent.

### Build (final tree 97865cacb, rule flags, existing `build/` and `build-vk/` configurations)
- **HIP** `build/`: two incremental builds, rc 0, 0 `warning:` lines (`build_hip.log`, `build_hip2.log`). They recompiled `ggml-cuda.cu`, `sparse-attn-attend-g1.cu` and the changed `src/` files (`grep` in the logs).
- **Vulkan** `build-vk/`: rc 0, 0 warnings (`build_vk.log`).
- **CPU** `build-cpu/` (reconfigured for the new test): the QSA test targets built with rc 0. Its 9 warnings are all `std::filesystem::u8path` deprecations in `common/` (common.cpp, download.cpp, hf-cache.cpp, preset.cpp), which this step does not touch.

### Tests run (CPU only; GPU NOT RUN)
| test | result |
|---|---|
| `build-cpu/bin/test-qsa-reserve-view` (new) | rc 0, `OK`, 4 layouts (above) |
| `build-cpu/bin/test-qsa-resplit` | rc 0, `PASS` (`ut_cpu_test-qsa-resplit.log`) |
| `build-cpu/bin/test-qsa-visibility` | rc 0, `PASS` |
| `build-cpu/bin/test-qsa-prefix` | rc 0, `PASS: 20000 randomized append/rollback steps, ...` |
| AT-C18 checker on Prove's logs | head checker `FAIL (4)` → new checker `PASS` |

### Reruns (ready; NOT RUN here)
`bash ~/dev/wt_lab/prove_fix/rerun.sh setup ctest ops np2 vkc mtp c18 census smoke` (setup already ran: `PV`'s harness copied to `prove_fix/h`, retargeted, with the fixed AT-C18 checker; `rerun.out` "SETUP ok 97865cacb"). Every GPU job is under `flock ~/gpu.lock`, one op per test-backend-ops process.

| phase | what | pass criterion |
|---|---|---|
| ctest | `ctest -R "test-qsa\|test-sparse-attn" -j1` in build and build-vk | all pass (now 12 per tree with test-qsa-reserve-view) |
| ops | test-backend-ops `-o SPARSE_ATTN_PLAN`, `-o SPARSE_ATTN` on ROCm0 (graphs on / off) and Vulkan0 | every case passes, census coverage 100 % |
| np2 | Prove's `srv.sh` `-np 2` split ub8192 A+B for f16, q8_0, q8_0K/f16V; `-np 2 --kv-unified` q8_0; np1 f16 | strict violations 0 in both contexts, answers 2749 / 12749, CPU 0; the `QSA reserve 1s-*` lines and the new compute buffer sizes reported |
| vkc | Vulkan ub16 census, six pairs (and f16 ub4096 / 512) | strict violations 0 (expected to stay FAIL for f16 / bf16 / f16K-q8_0V: 2.2 is not fixed) |
| mtp | `h/w7/mtp_census.sh` (default / q8 / q4 draft) | every route row OK and gated share ≥ 0.9 |
| c18 | `h/w9/at_c18_auto.sh` with the fixed checker | `AT-C18 PASS`, and the pp-0out line reads `route=cache-only ... guard=0/0` |
| census | `h/w10/census.sh` | EXPECT PASS, and the gated share per run |
| smoke | `h/w11/smoke_kld.sh` with `BASE=~/dev/wt_lab/w13/hooks/build/bin` (1dbee4d9a + measurement-only patch 06) | f16 / q8_0 / q4_0 c8192: KLD at base resolution (the gate scratch only changes cases that ran unfused before; the AT-A cell does not apply at n_kv 8192) |

The other Prove checks that 97865cacb / 4552df2a4 can move are Q1-A, P2/P3 and P5 (main HIP, G4 layers). They should be re-run with the Prove harness on this head once the GPU is back.

### Cleanup
- Nothing over 1 GB was created; `~/dev/wt_lab/prove_fix` holds scripts, messages and a copy of the harness.
- Not touched: `~/dev/wt_pr219v2*`, other worktrees, systemd units, strix-001, and the hung process 4041225 (not signalled).
- Nothing was pushed or posted.

## Final-fix 2: the confirmed review bugs (head `ed4bd3ea5`)

**Status:** 9 commits on `qsa-rewrite`, on top of `97865cacb`. Nothing was pushed.
- The review list had 10 entries. Two of them are one bug: the NC0 entry at `llama-memory-hybrid-idx.cpp:240` appears twice. So there are 9 distinct bugs and 9 commits.
- Each commit holds its fix and its regression test.
- The HIP, Vulkan and CPU builds pass with rc 0. No warning is in a touched line.
- Every test that can run without the GPU was run twice: once on the tree with the tests but none of the fixes, once on the final tree.
- **The GPU was still unavailable.** `flock -n ~/gpu.lock` gives LOCK_HELD at the start and at the end. The hung pid 4041225 is still in state D (`ps`: `D 02:39:01`), and `mem_info_gtt_used` is 91631312896. So no RADV or ROCm run was made, and every check that needs those devices is marked NOT RUN.

**What ran without the GPU, and why that is not a GPU job.** Builds compile offline (as in Prove-fix 1). The CPU tests ran from `build-cpu` (`GGML_HIP=OFF`, `GGML_VULKAN=OFF`). The Vulkan tests ran on **lavapipe** only, a CPU Vulkan device, through `strix-002:~/dev/wt_lab/final_fix/lvp.sh`:
- `VK_ICD_FILENAMES` is the lvp43 ICD alone (Mesa 25.3.6 / LLVM 21.1.8, PROOF-initial §3.1), with `GGML_VK_VISIBLE_DEVICES=0`;
- `VK_LOADER_LAYERS_DISABLE=~implicit~` and `NODEVICE_SELECT=1`, so no loader layer opens a DRM node;
- `timeout 1200`;
- every log line prints `device: llvmpipe (LLVM 21.1.8, 256 bits)`.

These runs were not wrapped in `flock` (the lock is held by the hung process), and none of them opens the AMD GPU.

**Method.** The local mirror holds one test commit (Tn) and one fix commit (Fn) per bug.
- The "before" tree is `97865cacb` plus every Tn and no Fn: tests and test hooks, no fix.
- The "after" tree is every Tn and Fn. The 9 branch commits are Tn+Fn squashed, in order. Applying the 9 patches to `97865cacb` gives exactly the after tree (`git diff --cached --quiet after` → IDENTICAL).
- Logs: `strix-002:~/dev/wt_rewrite_logs/F2/` (`before_*`, `after_*`).
- Patches, commit messages and the GPU rerun harness: `strix-002:~/dev/wt_lab/final_fix/` (`commit/patches/1-bug5.patch` … `9-bug7.patch`, `commit/msg/m*.txt`, `rerun_gpu.sh`, `lvp.sh`).

### Scorecard

| # | bug (review title, short) | commit | regression test | before | after |
|---|---|---|---|---|---|
| 5 | `qsa_supported_types` overflows its 16-tensor ctx | 9e84db9e7 | `test-qsa-supported-types` (new, CPU) | **abort**: `ggml_new_object: not enough space ... (needed 6256, available 5888)`, `ggml.c:1959: GGML_ASSERT(obj_new) failed`, rc 134 | rc 0: `'f16, bf16, q8_0, q4_0, f32, q4_1, q5_0, q5_1, iq4_nl'`, OK |
| 6 (+10) | NC0 runs on every ubatch, decode included | 7da229bf8 | `test-qsa-resplit`, new decode phase (CPU) | rc 1: `decode 16` violations, e.g. `ubatch 0 of 4 tokens <= U = 512 was given an NC0 evaluation` | rc 0, PASS: `decode 0`; decode steps 0 evaluations, prompt steps 2 |
| 1 | Vulkan `prefer_dense` ignores the device | b7394c04f | `test-sparse-attn-vk dense` | lavapipe (portable): dense 64 of 252 cells per forcing, 192 FAIL | dense 0 of 252, 0 FAIL |
| 3 | `unsupported_reason` never returns the limit reason | ae956e69a | `test-sparse-attn-vk reason` | GQA 16 and an f32 view of 134225920 B both read `type pair not supported` (3 FAIL) | `n_head / n_head_kv not in {4, 8, 12}`; the binding-limit reason; q2_K is still `type pair not supported`; 0 FAIL |
| 4 | vpack reserve not monotone in n_kv | e46cd6c94 | `test-sparse-attn-vk vpack` | needs the cm tier: on lavapipe "skipped" | **NOT RUN** on RADV |
| 2 | `sa_tier_ok` cleared and set again on every walk, read unlocked | 482b5f0f8 | `test-sparse-attn-vk race` (+ ThreadSanitizer build) | TSan on lavapipe: **1 `data race`**, write `ggml-vulkan.cpp:4595` (the clear) in `ggml_vk_load_shaders` against read `:10883` in `ggml_vk_sa_tier`, rc 66 | TSan: 0 reports, rc 0. The value check reads 0 differences both before and after on lavapipe (see 2.) |
| 8 | over RPC the SPARSE_ATTN workspace is never reserved | d665efc53 | `test-sparse-attn-rpc-view` (new, every device) | lavapipe: **SIGSEGV** in `ggml_vk_sa_select` ← `ggml_backend_vk_buffer_type_get_alloc_size` (gdb, `before_lvp_rpcview_gdb.log`) | 27 rows equal (9 Vulkan, 18 CPU), OK. HIP rows **NOT RUN** |
| 9 | gate-fusion scratch in the pool (D12) | 7a96532ad | `test-sparse-attn-gate-scratch` (new, CPU) | 22 failed checks (`mul over q: an output goes to pool scratch`; `no room for both` fused, expected refused) | 0 failed; MUL over Q goes to `scratch`, `no room for both` refused |
| 7 | at_end cases blind on HIP (272 B padding) | ed4bd3ea5 | the at_end cases themselves | on HIP: **NOT RUN** | lavapipe: the 8 at_end cases are unchanged (4 OK, 4 G = 4 expected-unsupported) |

### What changed, per bug

1. **Bug 5** (`src/llama-context.cpp`, `src/llama-context.h`).
   - The probe context holds `2 * std::size(types)` tensor objects: one K tensor plus one `ggml_sparse_attn` result per type.
   - The function is exported as `llama_qsa_supported_types` for the test.
   - Source of the 2 objects per type: `ggml_sparse_attn` creates one tensor (`ggml.c:6746`).
2. **Bug 6** (`src/llama-memory-hybrid-idx.{h,cpp}`, `tests/test-qsa-resplit.cpp`).
   - In the prepare hook, a ubatch with `n_tokens <= U` records `QSA_NC0_SKIPPED` and returns true. That is the value `!(nc0 && n_tokens > U)` always had there.
   - The graph-time predicate still routes the ubatch.
   - Test hooks: `qsa_nc0_eval_all` (the old every-ubatch evaluation) and `qsa_last_nc0_evals`.
   - The resplit grid runs with the hook, so assert (i) still covers every ubatch. Its counts are identical before and after: `320 batches, 2738 ubatches, 2316 graph-time non-compact, 2316 NC0, 1863 replays` (`before_/after_test-qsa-resplit.log`).
   - New decode phase, on unified `-np 4` and split `-np 2` at about 6k cells per sequence (past `top_k + r - 1 = 2051`):
     - a decode step gets 0 evaluations;
     - a 2 × 700-token prompt step (above U) gets them;
     - the ubatches and replays equal those of a twin fixture with every ubatch evaluated.
   - Not changed: the graph-time predicate's own repeated scan and the dup-key set walk in `can_reuse` / build, which the review also names. Only the init-time NC0 work, whose result could not matter, was removed. The tg cost before and after is UNPROVEN: no host timing was taken.
3. **Bug 1** (`ggml-vulkan.cpp`, `ggml_backend_vk_sparse_attn_prefer_dense`). It returns dense only when all three hold:
   - the table says dense;
   - `ggml_vk_sa_tier(device) == VK_SA_TIER_CM` (the table was measured on cm only);
   - the device's `supports_op` takes a FLASH_ATTN_EXT of the shortcut's shape (`ggml_vk_sa_dense_fa_supported`: D 256, 24 heads, 2 KV heads, f16 mask).

   Every other case keeps the sparse "av" route, which the portable tier runs. Limits of this check:
   - lavapipe has subgroup shuffle and vote, so its FLASH_ATTN_EXT is supported (`FLASH_ATTN_EXT f16/f16 supported` in `after_lvp_vk.log`). There, only the tier gate is exercised. The FA gate (a device without vote or shuffle) has no device here: UNPROVEN by run, by code reading only (`ggml-vulkan.cpp` FA supports_op `!coopmat2 && !(subgroup_shuffle && subgroup_vote)`).
   - Unchanged residual: a shape that the sparse op rejects (for example GQA 16) is no longer rescued by a dense pick on a non-cm tier.
4. **Bug 3** (`ggml-vulkan.cpp`).
   - `vk_sa_selection::why` is set at every early return of `ggml_vk_sa_select`, and `ggml_vk_sa_plan_why` gives the plan's causes.
   - Reasons:
     - `type pair not supported` only for a type outside the set;
     - one binding-limit reason for every limit check (bindings, workspace, suballocation block);
     - one reason per shape cause.
   - `ggml_backend_vk_sparse_attn_unsupported_reason` returns them.
   - The llama rule (`qsa_unsupported_error`: prefix "type pair" gets the type remedy, else the CPU / `-nkvo` remedy) is unchanged. HIP already returned distinct reasons (`sparse-attn-attend-g4.cu:354-372`).
5. **Bug 4** (`ggml-vulkan.cpp`, `ggml_vk_sa_select`).
   - The pack reserve is taken at `n_kv4' = min(n_kv4, 64 n_q - 4, the largest n_kv4 whose pack binds, 64 × maxComputeWorkGroupCount[0])`. That is the largest pack a runtime op of the context can ask, since a pack happens at `n_kv4 < 64 n_q`.
   - The dispatch's pack decision is unchanged.
   - Effect on M1 / the AT-B3 vpack rejection: the vpack arm's reserve grows. The size is UNPROVEN (needs RADV).
6. **Bug 2** (`ggml-vulkan.cpp`, `ggml-vulkan-types.h`).
   - `sa_tier_ok` is `std::atomic<bool>[3]`.
   - The walk computes `ok_portable`, `ok_scalar` and `ok_cm` into locals and stores each one once at the end of the block (relaxed), never clearing first. `ggml_vk_sa_tier` loads them atomically.
   - The value check (another thread reads supports_op and get_alloc_size during 12 on-demand compiles) can only see the bug on a cm or scalar device. Lavapipe's cm and scalar gates are false, so a clear changes no value: before, `7687300 reads ..., 0 differ`; after, `2786042 reads ..., 0 differ`.
   - The discriminating run is ThreadSanitizer:
     - setup: clang 22 (`libclang_rt.tsan`; gcc's libtsan is not installed: `cannot find /usr/lib64/libtsan.so.2.0.0`), `GGML_SANITIZE_THREAD=ON`, `GGML_OPENMP=OFF`, RelWithDebInfo, the same lavapipe;
     - before: exactly one report, the clear at `:4595` racing the read at `:10883`;
     - after: none.
   - The TSan build trees were deleted after the runs. Their build logs are kept in `F2/tsan_build_{before,after}.log`.
   - The review's second point, that `ggml_vk_sa_ws_subbuffer` recomputes get_alloc_size instead of checking the size gallocr reserved, is not changed: the backend does not know the reserved size. Open.
7. **Bug 8** (`ggml.c`, `ggml.h`, `sparse-attn-attend-g4.cu`, `ggml-vulkan.cpp`).
   - `ggml_sparse_attn_plan` stores `n_sel, n_q, n_stream` in op_params 2-4. The RPC serializes op_params (`deserialize_tensor` sets them, review source `ggml-rpc.cpp:1641-1695`).
   - HIP `sa_select` and Vulkan `ggml_vk_sa_select` read n_sel from there. Vulkan `ggml_vk_sa_plan_why` uses the op_params shape when the plan has no src.
   - The plan node's own alloc size still reads its direct src, which the server links (`rpc_server::get_alloc_size`, `ggml-rpc.cpp:1484-1488`).
   - Residual, not changed: the RPC client's alloc-size cache key leaves out src shapes and types (`ggml-rpc.cpp:1054-1078`). Its comment relies on the reserve querying the largest shapes first. Whether every QSA context's first SPARSE_ATTN query is the largest one is UNPROVEN.
   - The HIP RPC end-to-end run (`--rpc` to a HIP `rpc-server`) is NOT RUN.
8. **Bug 9** (`ggml-cuda.cu`, new `ggml-cuda/sparse-attn-gate-scratch.h`).
   - The placement is a host function. An output that overlaps a read goes to graph memory the fused subgraph leaves unwritten, in this order:
     1. the attend's own dst (`nbytes(node) == nbytes(mul)`: same element count, f32, contiguous; gallocr never places it over its srcs, and its workspace starts at `round_up(nbytes)`);
     2. the CONT output;
     3. the SIGMOID output.
   - A scratch must overlap no read, no final output and no other scratch. Otherwise the fusion is refused and the unfused nodes run.
   - The pool path is gone; `GGML_ASSERT` guards that it is never chosen.
   - NOT RUN (GPU):
     - that the MTP census gated share stays ≥ 0.9 (a refusal now needs both outputs hazardous and the CONT / SIGMOID memory unusable);
     - that the fused outputs are bitwise equal to before;
     - the pool high-water.
9. **Bug 7** (`tests/test-backend-ops.cpp`). For a case with `weights_end_at_buffer_end()`:
   - after initialization, the bytes between the last weight tensor and the buffer's end (the HIP padding of a quantized storage: `ggml_row_size(q8_0, 256)` = 272 B, `ggml-cuda.cu:1044-1049`, zeroed by `init_tensor` `:870-879`) become an I8 tensor filled with 0xFF;
   - 0xFF is NaN as an f16 / bf16 element and as the f16 scale of q8_0 / q4_0 and the other block types;
   - the bytes must still be 0xFF after the compute.

   So an over-read that reaches the arithmetic (a score, or `p × V` with p = 0) turns the output NaN (the existing NaN check fails the case), and a write is reported. A read whose value is discarded before any arithmetic is still not visible: this is stated in the commit and is UNPROVEN to be covered. The "fails before" demonstration (an injected one-quad over-read in a HIP loader) was not written and is NOT RUN.

### Build (final tree `ed4bd3ea5`)
- **HIP** `build/` (rule flags, existing configuration), full rebuild (ggml.h changed): rc 0 (`after_build_hip.log`).
  - 85 `warning:` lines, none in a touched file.
  - The 13 `ld.lld: ... loop not unrolled` lines are also in the W0 build (`W0/w0lab_hip_build.log`: 13).
  - The rest are in `tools/mtmd`, `common/` and `tests/test-backend-ops.cpp:463, 466, 9427, 11496, 17373-17382`, none of which this step touched.
- **Vulkan** `build-vk/`: full build rc 0 (`after_build_vk.log`). Its 75 warnings include `ggml-vulkan.cpp:9646` (`dst_buf_ctx` unused; the same code is at `:9640` in `97865cacb`) and `:10713` (`ggml_vk_fa_kv_native`'s unused `coopmat2`). Both lines are untouched.
- **CPU** `build-cpu/`: full build rc 0. Its warnings are all in untouched files.

### Tests run on the final tree
| run | result |
|---|---|
| `build-cpu` `ctest -R "test-qsa\|test-sparse-attn" -j1` | 15/15 passed (`after_ctest_cpu.log`) |
| lavapipe `test-sparse-attn-vk` (all: limits, capability, rot, stress, dense, reason, vpack, race) | `OK: 0 failed checks` (`after_lvp_vk_all.log`) |
| lavapipe `test-backend-ops -b Vulkan0 -o SPARSE_ATTN_PLAN` / `-o SPARSE_ATTN` `--fail-unsupported` | 92/92 and 736/736, rc 0; 654 "not supported" lines, the count in PROOF-initial §3.1 |
| lavapipe `test-qsa-allvis-rows`, `test-qsa-te-rows` (run directly) | PASS, rows on `[Vulkan0` and `[CPU` |
| lavapipe `build-vk` ctest subset | 14/15 passed. `test-qsa-te-rows` aborts intermittently (see below) |

**Not caused by this step: `test-qsa-te-rows` heap corruption on lavapipe.**
- Symptom: `corrupted size vs. prev_size while consolidating`, SIGABRT in a worker thread whose frames are in an unsymbolized module (gdb, `F2/terows_gdb.log`).
- Rates over 20 runs each:
  - head: 9/20;
  - head with `MESA_SHADER_CACHE_DISABLE=true`: 10/20;
  - **`97865cacb` (a clean `git archive` built with the same flags): 11/20**.
- It predates these commits. The test uses only the TE ops (`qwen4exp-te.h`), none of which this step touches. Cause: UNPROVEN (lavapipe / LLVM, or the test). Open.

### GPU reruns (ready, NOT RUN)
`bash ~/dev/wt_lab/final_fix/rerun_gpu.sh`. Every job is under `flock ~/gpu.lock`, with one op per test-backend-ops process.

| what | pass criterion |
|---|---|
| RADV `test-sparse-attn-vk dense reason vpack race`, then all | 0 failed checks. `vpack` runs there (cm tier) and must show the reserve ≥ every smaller n_kv. `dense`: cells dense only on tier cm |
| HIP `test-sparse-attn-rpc-view` | every HIP row `server view == linked`. Before the fix it would have been `nbytes` against `nbytes + ws` (code reading, `sparse-attn-attend-g4.cu:349-352`) |
| HIP / RADV `test-backend-ops -o SPARSE_ATTN_PLAN`, `-o SPARSE_ATTN` | all pass. On ROCm0 the q8_0 at_end cases now carry a poisoned 272 B tail |
| `ctest -R "test-qsa\|test-sparse-attn" -j1` in `build` and `build-vk` | all pass, except the known te-rows issue, which is lavapipe-only so far (UNPROVEN on RADV) |
| also from Prove-fix 1 `rerun.sh`: `mtp`, `census`, `np2`, `vkc` | the gated share with the new scratch placement (bug 9), and strict reserve with the vpack and RPC changes |

### Commits (`git log --format='%h %an <%ae> | %s' 97865cacb..HEAD`)
- 9e84db9e7 `llama : QSA init error lists the supported types without overflowing its context`
- 7da229bf8 `llama : QSA NC0 runs only on the ubatches the re-split can act on`
- b7394c04f `vulkan : QSA dense shortcut only where the device runs FLASH_ATTN_EXT, on the cm tier`
- ae956e69a `vulkan : SPARSE_ATTN unsupported reason names the cause`
- e46cd6c94 `vulkan : V-pack workspace reserve holds the pack of every smaller n_kv`
- 482b5f0f8 `vulkan : sparse-attention tier gates are set once per walk, never cleared first`
- d665efc53 `ggml : SPARSE_ATTN alloc size never reads the plan's own source (RPC)`
- 7a96532ad `hip : gated sparse-attention fusion writes a hazardous output to graph memory, not the pool`
- ed4bd3ea5 `tests : sparse-attention at_end cases guard the bytes past the last K/V storage`

All commits are by Gaetan Puleo <12990773+gaetan-puleo@users.noreply.github.com>, with the session trailer.
- **No Co-authored-by:** none of them ports #219 code. The touched code is this branch's own: W3 / W4 / W6 / W7 / W9, the gate fusion of 97865cacb, and the test-backend-ops at_end cases.
- Only the final tree was compiled. The intermediate commits were not built one by one, so it is UNPROVEN that each one builds alone. Each patch was checked to apply in order (`git apply --cached`).

### Cleanup
- Deleted: the TSan tree (`~/dev/wt_lab/final_fix/tsan_before`, 926 MB) and the base te-rows tree (`base_vk`, 890 MB). Their build logs are kept in `F2/`.
- Core dumps of the te-rows aborts may be in the systemd-coredump store. They were not removed (no root).
- Not touched: `~/dev/wt_pr219v2*`, other worktrees, systemd units, strix-001, and pid 4041225.
- Nothing was pushed or posted.

## F3: fixes for eight confirmed review bugs (7 commits on top of `ed4bd3ea5`)

All work ran on strix-002 in `~/dev/wt_rewrite` (branch `qsa-rewrite`). Logs are in `strix-002:~/dev/wt_rewrite_logs/F3/`. Nothing was pushed or posted.

### Environment constraint: no GPU run was possible
- `~/gpu.lock` was held for the whole step by pid 4041224 (`timeout 3600 … build-vk/bin/llama-perplexity …`). Its child, pid 4041225, was in state `D` with an elapsed time of 05:08:23 when this step started. Sources: `lslocks` printed `timeout 4041224 FLOCK WRITE … /home/gaetan/gpu.lock`, and `ps -eo pid,ppid,etime,stat,cmd` printed `4041225 … 05:08:23 D`.
- That process was not started by this step, so it was not touched. A `flock ~/gpu.lock cmake -B build-rpc-hip …` started by this step waited on the lock for more than 13 minutes. It was killed (pids 143283 and 143236, both started by this step).
- Consequences:
  - **No GPU device was opened.** Nothing ran on ROCm0 or RADV.
  - Builds ran without the lock. A build uses no GPU, and the previous step's builds at 15:25 ran while the same process held the lock.
  - The tests ran on the CPU, on the CPU + RPC build and on lavapipe (`~/dev/wt_lab/final_fix/lvp.sh`, a CPU Vulkan ICD).
- **Every HIP / RADV check is NOT RUN.** They are scripted in `~/dev/wt_rewrite_logs/F3/gpu_rerun.sh`, which wraps every job in `flock`.
- The "before" HIP binaries are kept for that script in `~/dev/wt_lab/F3_before/{hip,rpchip}` (215 MB, run with `LD_LIBRARY_PATH` set to that directory). They were built from the before tree described below: `build/bin` at 17:07 and `build-rpc-hip/bin` at 17:11 (`ls -la` in the snapshot).

### Method (before / after)
- **Before tree.** HEAD `ed4bd3ea5` plus all new tests and the two behaviour-preserving refactors that the tests need, with no fix:
  - `llama_qsa_reserve_passes` without the 1s-passB passes;
  - `qwen4exp-qsa-check.h` with the old predicate: `scalar` alone, `blocks` and `swa` ignored.
  The fix files were restored from `git show HEAD:…`, and their mtimes were reset to the commit time so that make kept the HEAD objects.
- **Builds of the before tree.**
  - Directories: `build-cpu`, `build-rpc` (CPU + RPC), `build` (HIP), `build-vk`, and the new `build-rpc-hip`.
  - Results: all rc=0 (`before_*.log`, second attempt). The first attempt failed on two test compile errors: private `ggml_{get,set}_op_params_i32`, and a missing `<initializer_list>`. Both were fixed in the tests.
- **Run.** `run.sh before` (logs `F3/before/`), then the after tree: `run.sh after` (`F3/after/`).
- **Caveat about stale objects.** The first "after" build reused stale objects, because tar kept the local mtimes, which were older than the before objects. `build-rpc` failed to link with `undefined reference to ggml_sparse_attn_shape_why` (`after2_build-rpc.log`), which exposed it. Every changed file was then `touch`ed and all five directories were rebuilt. Those rebuilds are the `after_*.log` results.
- **Final state.** After the final incremental build (`final_*.log`), all five directories are rc=0. No `error:` appears, and no `warning:` appears in a touched file. The grep over `after_*.log` matched nothing.

### Results, bug by bug

| # | bug | fix (file:line on HEAD `574689270`) | test | before | after |
|---|---|---|---|---|---|
| 1 | G1 split-K merge drops a NaN-normaliser split | `sparse-attn-merge.cu:37` `count != 0.0f` (NaN kept, so the row is 0); ggml.h contract item 5 | test-backend-ops SPARSE_ATTN `,nanscore` (f16/q8_0/bf16, n_q 1/3, n_sel 2051, n_kv 8192, G 1/4) + CPU non-vacuity check (query 0 of KV head 0 all zero, KV head 1 not) | CPU 1/1 OK, lavapipe 2/2 OK (`before/{cpu,lvp}_tbo_nanscore.log`); **HIP NOT RUN** | CPU OK, lavapipe OK (6 G1 cases run, G4 unsupported on lvp) (`after/…`); **HIP NOT RUN** (`gpu_rerun.sh` `hip_nanscore_before/after`) |
| 2 | supports_op / compute trust ggml_sparse_attn's shape invariants (RPC) | `ggml_sparse_attn_shape_why` / `ggml_sparse_attn_plan_shape_why` (`ggml.c:6770,6793`, asserted by the constructors at `:6708,:6765`); used by HIP `sparse-attn-attend-g4.cu:220` (supports_op and sa_run's abort), `sparse-attn-plan.cu:595`, CPU `ggml-cpu.cpp:498,501` and `sparse-attn.cpp:31,299`, Vulkan `ggml-vulkan.cpp:11028,11356,22590` | new `test-sparse-attn-shape-guard`: 13 malformed nodes (K/V views short of n_kv, V heads != K heads, streams, plan n_q / size / n_kv, dst shape, plan sel shape) × type × G, each of which must be unsupported where the well-formed node is | CPU: `52 malformed nodes checked, 52 failed: FAILED`; lavapipe: `78 … 78 failed` | CPU `52 … 0 failed: OK`, lavapipe `78 … 0 failed: OK`; HIP NOT RUN |
| 3 | no one-stream shortcut reserve in split multi-sequence contexts | pass list moved to `llama_qsa_reserve_passes` (`llama-context.cpp:914`, used at `:1406`); `1s-passB` / `1s-passB-0out` at `:973-974` (one stream at n_kv_b, min(ub, n_kv_b) rows) | new `test-qsa-reserve-passes` (5 ub × 4 np × split/unified × 3 cache sizes) | 30 `FAIL … no reserve graph builds the one-stream shortcut` lines, `FAILED` | `OK`; e.g. `ub 512 np 8 split n_kv 32768: passB noncompact 1s-pp 1s-0out 1s-fast 1s-passB 1s-passB-0out` |
| 4 | non-compact check fires when block selection is ineligible | `qwen4exp-qsa-check.h:16` (false unless `blocks && !swa`), call at `qwen4exp.cpp:1467`; `llama_set_causal_attn` asks for a reserve with QSA layers (`llama-context.cpp:2137`) | new `test-qsa-noncompact-check` | 20 FAIL lines (`blocks ineligible: a decode error` ×10, `SWA layer: a decode error` ×10), `FAILED` | `OK` |
| 5 + 8 | RPC alloc-size cache keyed by dst only (duplicates) | `ggml-rpc.cpp:1059` `rpc_alloc_size_src_keyed`; key gains every src's type/op/op_params/ne/nb (`:1112,:1124`) | new `test-rpc-sparse-attn` part (1): in-process rpc-server; dst shapes with 2–3 plans, RPC alloc size == server device's | CPU server: equal, **vacuous** (`0 of a dst shape whose size differs`: the CPU alloc size is nbytes); **HIP NOT RUN** (`hiprpc_before`) | same (vacuous on CPU); HIP NOT RUN |
| 6 | QSA guard cannot fail on RPC; 8.1 client accepts an 8.0 server | `RPC_CMD_SUPPORTS_OP` (`ggml-rpc.cpp:81`, server `:1563`, dispatch `:2630`, client `:3005-3059`, cached) for SPARSE_ATTN / SPARSE_ATTN_PLAN; `RPC_PROTO_MIN_SERVER_MINOR_VERSION 1` (`ggml-rpc.h:18`, hello `ggml-rpc.cpp:445-447`); deserialize rejects op >= GGML_OP_COUNT (`:1786`) | `test-rpc-sparse-attn` part (2) (supports_op through RPC == server device's, for f16 / q8_0 / q2_K / i32 attends and their plans); new `test-rpc-hello` (fake server 8.1 / 8.2 / 8.0 / 7.1, forked client) | part (2): `SPARSE_ATTN i32/i32: server device 0, through RPC 1` → FAIL; hello: `server 8.0: client goes on (want rejects)` → FAILED | part (2) 10/10 equal, OK; hello: 8.1 goes on, 8.2 / 8.0 / 7.1 rejected (`RPC server version mismatch: 8.0.0 (this client takes 8.1 to 8.1)`), OK |
| 7 | supported-types test matched by substring | `test-qsa-supported-types.cpp:18` `split_list`, whole-entry match, entry count 9, self-check of the checker | mutation: `types[]` without `GGML_TYPE_F16` in `llama_qsa_supported_types` (build-cpu, reverted afterwards) | old test with the mutation: `'bf16, q8_0, …'` → `test-qsa-supported-types: OK` (blind; `before/cpu_supported_types_mutF16.log`) | new test with the mutation: `missing f16`, `8 entries, want 9`, `FAILED` (`after/cpu_supported_types_mutF16.log`); without the mutation: `OK` |

Notes per bug:
- **1.** Vulkan was checked by reading the code: `sparse_attn_reduce.comp:54-68` sums `exp(m - m_max) * l` over the splits without a select, so a NaN l gives `L` NaN and `inv_l` 0. Lavapipe passes the case both before and after, which is consistent with that reading. Whether lavapipe actually splits at n_q = 1 is UNPROVEN. The commit carries `Co-authored-by: Iggy Jackson <iggy@theiggy.com>`, because `sparse-attn-merge.cu` adapts #219's decode merge (its header comment; the file's commit bb5490b5e carries the same trailer, per `git log --format=%(trailers)`).
- **2.** The fix also covers the CPU and Vulkan backends, which had the same gap (`ggml_vk_sa_select` read `n_kv` from the plan without checking the views), and the plan node. A plan node whose op_params name a sel shape it does not have was supported by HIP before, and the test case `plan: op_params n_kv 2^24, dst sized for 8192` covers it. The HIP compute path reaches `sa_select` through `sa_run`, which aborts with `no HIP kernel for this node: <why>`. An RPC server's `graph_compute` never calls supports_op, so that abort is the server-side guard. Before this change the kernels read past the tensors instead.
- **3.** The refactor keeps every existing pass. The test checks passB's shape and the presence of the 1s passes on every grid point. Whether the compute buffer actually grew past the reserve on such a ubatch before the fix is still UNPROVEN: no model run was possible, because the 104 GB model (`du -sh ~/models/qwen3.8-flash-next-UD-Q4_K_XL`) does not fit next to the stuck process (`free -g`: 92 of 124 GiB used).
- **4.** The claim that the base decoded the ineligible case comes from the review and was not re-run here (UNPROVEN in this step: no model). The `set_causal_attn` reserve request goes through the same `sched_need_reserve` path that `set_embeddings` and others use (`llama-context.cpp:2092-2231` at HEAD), but it has no runtime test (NOT RUN).
- **5 + 8.** On a CPU server the test is vacuous: it reports this itself, and the CPU alloc size is `ggml_nbytes`. The failing case needs a HIP or Vulkan server: `gpu_rerun.sh` `hiprpc_before` / `hiprpc_after` on `build-rpc-hip`, a new HIP + RPC build directory configured with the rule flags plus `-DGGML_RPC=ON`. Each new n_kv per ubatch shape now costs one GET_ALLOC_SIZE round trip. That cost is not measured.
- **6.** `RPC_CMD_SUPPORTS_OP` is inserted before `RPC_CMD_NONE`, so NONE / COUNT shift. NONE is a client-local dispatcher marker (`ggml-rpc.cpp:610-691` at HEAD), not a wire value. The new command stays within 8.1, this branch's own unreleased minor (`git show ea5285ddf:ggml/include/ggml-rpc.h`: 8.0). Only the two QSA ops are asked remotely; every other op still answers true. Case (a) of the review, a CUDA rpc-server, now ends as the QSA init error through `qsa_guard`. That path was not run end to end: it needs a model and a non-HIP CUDA server (NOT RUN). The test covers it with a CPU server and an i32 K node, which the CPU rejects.

### Regression runs on the after tree (no GPU; `F3/regress/`)
| run | result |
|---|---|
| build-cpu `ctest -R "test-qsa\|test-sparse-attn" -j1` | 18/18 passed |
| build-rpc `ctest -R "test-qsa\|test-sparse-attn\|test-rpc" -j1` | 21/21 passed |
| lavapipe build-vk `ctest -R "test-qsa\|test-sparse-attn" -E test-qsa-te-rows -j1` | 17/17 passed. te-rows is excluded because of its known intermittent lavapipe abort (see the previous step) |
| lavapipe `test-sparse-attn-vk` (all) | `OK: 0 failed checks` |
| CPU `test-backend-ops -b CPU -o SPARSE_ATTN` / `-o SPARSE_ATTN_PLAN` | 1402/1402, 185/185 |
| lavapipe `-b Vulkan0 -o SPARSE_ATTN` / `-o SPARSE_ATTN_PLAN` | 742/742 (736 before plus the 6 new G1 nanscore cases), 92/92 |

### Per-commit build (bisectability, CPU + RPC configuration only)
- `F3/bisect.sh` built each commit separately in `~/dev/wt_lab/F3_bisect` (`git archive` + `rsync -c`, then a full `cmake --build -j 16`) and ran `ctest -R "test-qsa|test-sparse-attn|test-rpc" -E test-qsa-te-rows`.
- Results: every commit built with rc=0. ctest passed 15/15, 16/16, 17/17, 18/18, 19/19, 20/20 and 20/20 (`F3/bisect/<sha>.log`).
- The intermediate commits were not built with HIP or Vulkan (UNPROVEN). Those configurations were built only at the final tree.
- The 546 MB bisect tree was deleted afterwards.

### Commits (`git log --format='%h %an | %s' ed4bd3ea5..HEAD`; author Gaetan Puleo <12990773+gaetan-puleo@users.noreply.github.com>, session trailer on each)
- d7e4656d1 `hip : sparse-attention split-K merge keeps a split whose normaliser is NaN` (Co-authored-by: Iggy Jackson <iggy@theiggy.com>)
- 108d6ce00 `ggml : SPARSE_ATTN supports_op and compute check the shape invariants of ggml_sparse_attn`
- 195dadc10 `llama : reserve the one-stream QSA shortcut graph in a split multi-sequence context`
- a22777440 `qwen4exp : the QSA non-compact check applies only where block selection is eligible`
- 458acecf6 `rpc : the SPARSE_ATTN alloc-size cache key holds the node's srcs`
- a82bdb924 `rpc : SUPPORTS_OP for the sparse-attention ops, and reject a server older than 8.1`
- 574689270 `tests : test-qsa-supported-types matches whole list entries`

Shared files were split across commits with `F3/stage.py`, which undoes later commits' edits on the final content, and `F3/commit.sh` (`git hash-object -w` + `update-index --cacheinfo`). After the last commit, `git diff --quiet HEAD` holds and `git status --short` is empty.

### Left for the orchestrator
- `bash ~/dev/wt_rewrite_logs/F3/gpu_rerun.sh` once `~/gpu.lock` is free. Pass criteria:
  - `hip_nanscore_before` FAILs the G1 nanscore cases, and `_after` passes;
  - `hip_shape_before` FAILs, and `_after` is OK;
  - `hiprpc_before` FAILs part (1) on a HIP server (alloc sizes differ), and `_after` is OK;
  - the ROCm0 and RADV SPARSE_ATTN / SPARSE_ATTN_PLAN suites pass;
  - the HIP, HIP + RPC and RADV ctest subsets pass.
- Then delete `~/dev/wt_lab/F3_before` (215 MB).
- Unchanged by this step: `~/dev/wt_pr219v2*`, other worktrees, systemd units, strix-001, and pids 4041224 / 4041225.

## F4: fixes for ten confirmed review bugs (7 commits on top of `574689270`)

All work ran on strix-002 in `~/dev/wt_rewrite` (branch `qsa-rewrite`). Logs are in `strix-002:~/dev/wt_rewrite_logs/F4/`: `before/`, `after/`, `final/`, `bisect/`, and `variants/` (the before / mutation sources and the run scripts). Nothing was pushed or posted.

### Environment constraint: no GPU run was possible
- `~/gpu.lock` was still held by pid 4041224 (`timeout 3600 … build-vk/bin/llama-perplexity …`). Its child, pid 4041225, was in state `D`, with wchan `drm_suballoc_insert`. Its elapsed time was 07:30:34 when this step started and 07:48:53 later in the step. Sources: `lslocks` and `ps -o pid,stat,etime -p 4041225`.
- That process was not started by this step, so it was not touched.
- No GPU device was opened. Every HIP (ROCm0) and RADV result below is **NOT RUN**. The runs are scripted in `~/dev/wt_rewrite_logs/F4/gpu_rerun.sh`, with every job under `flock ~/gpu.lock`.
- Builds ran without the lock, as in F3: a build opens no device.
- The tests ran on:
  - the CPU (`build-cpu`);
  - the CPU + RPC build (`build-rpc`);
  - lavapipe (`~/dev/wt_lab/final_fix/lvp.sh`, a CPU Vulkan ICD that runs the portable tier only and never splits).
- The scalar tier cannot run on lavapipe: `GGML_VK_SA_FORCE=scalar` gives `not supported [Vulkan0]` for every nanscore case.

### Scorecard
The "file:line" column refers to HEAD `e31548e7d`.

| # | bug (review title, short) | commit | fix (file:line) | regression test | before | after |
|---|---|---|---|---|---|---|
| 1 + 9 | -inf score on a row's first attended cell: CPU / portable / fixup / O_CON give 0, G4 / G1 a softmax | `df88ab1bb` | p = 0 for a -inf score: CPU `sparse-attn.cpp:414`, O_CON `sparse-attn-ref.h:209`, HIP portable `sparse-attn-portable.cu:175`, G4 fixup `sparse-attn-attend-g4.cuh:769`, Vulkan portable `sparse_attn_portable.comp:268`, scalar `sparse_attn.comp:196`, cm `m_ref` `sparse_attn_cm.comp:318-323`; contract text `ggml.h:2670-2672` (item 3) | test-backend-ops SPARSE_ATTN `,leadneginf` (15 cases: f16 / q8_0 / bf16; G1 n_q 1 n_sel 2051, n_q 9 n_sel 48, n_q 66 n_sel 257; G4 n_q 4, 66). CPU check: the KV head 0 rows equal O_CON over the rows without cells 0..15, bit for bit | CPU 15/15 FAIL: `leadneginf: query 0 rows of KV head 0 zero; 3072 values of KV head 0 differ …` (`before/A_cpu_lead.csv`). Lavapipe with the CPU fixed but the portable shader not: 9 G1 cases FAIL, e.g. `ERR = 0.943802680` (`before/A_lvp_lead_portable_unfixed.csv`) | CPU 15/15 OK (`after/A_cpu_lead.csv`); lavapipe 9 OK, 6 G4 not supported (`after/A_lvp_lead.csv`); HIP / RADV NOT RUN |
| 8 | G4 / G1 drop a NaN score while the running max is -inf | `c16d2360c` | `mref` / `ref` = 0 when the max is -inf: `sparse-attn-attend-g4.cuh:458,462`, `sparse-attn-attend-g1.cuh:318,322` | `,leadnan` (15 cases, the same shapes): cells 0..15 of query 0 have NaN K rows (KV head 0) and fill its first step. The CPU check requires query 0's KV head 0 rows to be 0 and its KV head 1 rows not | CPU and lavapipe OK both before and after (those tiers were already right; `before/A_cpu_lead.csv`). **HIP NOT RUN** | CPU 15/15, lavapipe 9/9 OK; HIP NOT RUN (`gpu_rerun.sh` `hip_lead_before` / `_after`) |
| 2, 3, 10 | Vulkan cm / scalar epilogues and the split-K reduce write NaN * 0 = NaN for a NaN normaliser | `21f346a6e` | `pos ? O * inv_l : 0` with `pos = sa_l_pos(l)`, an integer test on the bits: `sparse_attn_cm.comp:415`, `sparse_attn.comp:221`, `sparse_attn_reduce.comp:78`; helper `sparse_attn_load.glsl:18`, `sparse_attn_reduce.comp:27` | `,nanscore` now also at n_q 64 (one split on cm and scalar), plus the n_q 1 / 3 nanscore cases (split + reduce) and the lead cases | **RADV NOT RUN**. Lavapipe cannot reach these shaders (portable tier only, no split) | the shaders compile (`final/build_build-vk.log`, rc 0); CPU 6 new cases OK; RADV NOT RUN (`radv_nan_before` / `radv_lead_after` / `_scalar`) |
| 6 | SPARSE_ATTN cases never check their sentinels | `f805cf9a5` | `test_case::whole_graph_checks_sentinels()` (opt-in, `test-backend-ops.cpp:1374`), sentinels appended to the verified nodes (`:1723`), test_sparse_attn opts in (`:13570`); the sentinel callback returns after the byte compare (`:1672`) | mutation (not committed): the CPU op writes a per-call counter into the 4 bytes past dst (`variants/cpu_mut6.cpp`); 18 f16 SPARSE_ATTN cases (`variants/run6.sh`) | old harness with the mutation: 18/18 OK, blind (`before/B6_mut_harness_before.csv`) | new harness with the mutation: 18/18 FAIL `sentinel mismatch: sent_5` (`after/B6_mut_harness_after.csv`); new harness without it: 18/18 OK |
| 7 | the ">= 2 GiB K view" case never reads past byte INT32_MAX | `a5b72e078` | `SA_ROWS_TAIL` (last 4 cells in every row, `test-backend-ops.cpp:13089,13188`); the f16 2^21+4 case uses it and a new q8_0 2^22+4 case is added (`:14575-14581`); `check_reach` fails a case whose storage passes INT32_MAX with no selected row past it (`:13692`, read by verify_engagement `:13677`) | mutation (not committed): the CPU op wraps its K / V byte offsets at 2^31 (`variants/cpu_mut7.cpp`, `variants/run7.sh`) | old case, no check, mutation: OK, blind (`before/B7_orig_mut.csv`). Old rows with the check: FAIL `K storage of 2147487744 bytes, but no selected row reaches past byte INT32_MAX (last selected byte 2146606079)` (`before/B7_check_oldrows.csv`) | mutation: f16 2^21+4 and q8_0 2^22+4 FAIL `CPU op != O_CON reference in 6144 of 6144 values`; no mutation: OK (`after/B7_final*.csv`); HIP NOT RUN |
| 5 | `--cache-type-k-idx iq4_nl` sends the indexer GET_ROWS to the CPU on HIP without any error | `4e08e5b91` | `llm_graph_result::add_qsa_idx_node` (`llama-graph.h:1044`); qwen4exp registers the cpy_k SET_ROWS and the members GET_ROWS (`qwen4exp.cpp:1514,1543`); `qsa_guard_indexer` (`llama-context.cpp:1031`, called at `:1190`) | `test-qsa-guard <model> fake-idx <type>`: a fake GPU that takes the sparse ops but, like HIP, refuses IQ4_NL GET_ROWS when ne[0] % 256 != 0 (`variants/run5.sh`) | guard call removed: iq4_nl `FAIL: the context must not be created …`; q8_0 PASS (`before/B5_guard_*.log`) | iq4_nl PASS, init error `QSA layer 3: the indexer K cache (iq4_nl, on FAKE_GPU) is read by GET_ROWS 'node_537', which device FAKE0 does not take … or a type it takes here: f32, f16, bf16, q8_0, q4_0, q4_1, q5_0, q5_1`; q8_0 / f16 / bf16 / q4_0 / q5_1 / f32 create the context (`final/cpu_guard_fakeidx_*.log`); a real HIP device: NOT RUN |
| 4 | a mid-session reserve takes the QSA route from the live cache (pp reserve on the TE route) | `e31548e7d` | `LLM_GRAPH_QSA_RESERVE_GRAPH` (`llama-graph.h:835`), set by graph_reserve (`llama-context.cpp:3592`); `qsa_graph_compact` / `qsa_compact_reason_empty` (`llama-memory-hybrid-idx.cpp:1881,329`); qwen4exp build and can_reuse decide through it (`qwen4exp.cpp:1463,1293`) | `test-qsa-reserve-view` new part: the graph_reserve pp ubatch (4096 / 512 / 4 tokens, 4 sequences, unified) on an empty cache, with an image in slot 0 next to text in slot 1, and with a seq_cp prefix that ends inside a block | `qsa_graph_compact` returning the live verdict: `6 FAILED` (`reserve graph non-compact` for 2-D-cells-with-another-sequence-in-the-stream and block-shared-by-two-sequence-sets; `before/D4_reserve_view.log`) | OK; the runtime verdict of those layouts stays non-compact (`after/D4_reserve_view.log`). Model run of a mid-session reserve: NOT RUN |

### Notes per bug
- **1 + 9 (one rule).**
  - The rule is that a -inf score has weight 0 wherever the cell lies. Before this change, the CPU applied that rule except at the first cell. The HIP WMMA attends already applied it everywhere.
  - A row whose cells are all -inf has l = 0 and gives 0.
  - With a finite running max, the arithmetic is unchanged on every path: every select differs only when the max is -inf.
  - Proof of bit-exactness:
    - the CPU SPARSE_ATTN suite passes 1439/1439, including the O_CON bit check of every case;
    - O_CON was fixed with the same select;
    - the new check also compares against O_CON over rows with the -inf cells removed, which shares no -inf arithmetic.
  - The Vulkan cm / scalar -inf steps were read from the code, not run. They gave NaN rows: cm through `corr = exp(-inf - -inf)`, scalar through `weight = exp(-inf - -inf)`.
- **8.** The fix keeps the -inf case at p = 0, because `exp2f(-inf) = 0`. A NaN then reaches l and the epilogue / merge writes 0, the merge keeping a NaN split (d7e4656d1). The per-element select became one select per step.
- **2, 3, 10.**
  - `sa_l_pos` is used because the modules have no SignedZeroInfNanPreserve mode (bug report; float-controls patch `ggml-vulkan.cpp:692-751`), so a float `l > 0` may be compiled as if l were never NaN. Whether RADV actually miscompiles a float compare is UNPROVEN either way.
  - The product for rows with l > 0 is unchanged.
  - The partial M / L writes of the splits are unchanged.
- **6.**
  - The opt-in covers test_sparse_attn and its gated / gqa_rep subclasses.
  - The other whole-graph cases of the file keep the old verified set and stay unchecked. That includes test_topk_qsa, test_qsa_score and test_qsa_strip (`test-backend-ops.cpp` fusion_test_nodes overrides). This step did not change them: their GPU runs were impossible.
  - Returning early after the byte compare cannot hide a failure: the bytes are equal. It removes the err() call on a sentinel, which failed every `,overflow` case when the sentinels were first listed (`n_inf == 0 → 1.0`).
- **7.** The new q8_0 case at 2^22 + 4 cells is an op test only. The qwen4exp n_ctx_seq cap stays at 2^21 + 4 (`llama-model.cpp` comment updated). On Vulkan both large cases are over the binding limit and expected unsupported.
- **5.**
  - The guard checks the registered nodes themselves, not their sources. The scheduler had already replaced the GET_ROWS source with its input copy (`CPU#cache_idx_k_l3 (view) (view)#0`), as found while debugging this step.
  - A host cache (`--no-kv-offload`, `-ngl 0`) is not checked.
  - The fake device uses the HIP rule as written in `ggml-cuda.cu:9201-9207`.
  - Whether HIP SET_ROWS takes IQ4_NL: UNPROVEN. The guard reports either way.
- **4.**
  - At init the cache is empty, so the empty-cache verdict equals the live one. The test checks this on the empty layout, so the first reserve is unchanged.
  - The buffer growth of the old behaviour is the review's code-reading estimate; it was not measured.
  - The other live-cache inputs of a reserve graph are not changed by this commit: `qsa_n_kv_window`, `qwen4exp_score_key_limits`. gallocr never shrinks a buffer at a reserve (UNPROVEN in this step: not re-read).

### Build (final tree `e31548e7d`, existing configurations)
- `final/builds.txt`: build-cpu, build-vk, build (HIP, rule flags), build-rpc and build-rpc-hip all rc 0, 0 ` error` lines.
- HIP register usage of the changed TUs on the final tree (`final/spill_after.txt`, from compile_commands.json + `-Rpass-analysis=kernel-resource-usage`):

| TU | kernels | VGPR spills | SGPR spills |
|---|---|---|---|
| g1 | 96 | 0 | 0 |
| g4-{f16,q8_0,bf16,q4_0} | 104 each | 0 | 0 |
| generic | 19 | 0 | 0 |
| portable | 1 | 0 | 0 |

  These kernel counts match the coverage map of `8d1faf602`. That run used the host compiler, not the pinned CI image.
- Against the parent tree (`~/dev/wt_lab/F4_before/build-hip`, `before/spill_before.txt`):
  - the per-TU VGPR maxima are unchanged (138 / 192 / 192 / 192 / 192 / 209 / 54);
  - the VGPR sums move by -48 (g1), -21 (g4-f16, g4-bf16), -20 (g4-q8_0), +15 (g4-q4_0) and 0 (generic, portable);
  - spills are 0 before and after.
  - Throughput: NOT measured (no GPU).

### Regression runs on the final tree (no GPU; `final/regress.txt`)
| run | result |
|---|---|
| build-cpu `ctest -R "test-qsa\|test-sparse-attn" -j1` | 18/18 passed |
| build-rpc `ctest -R "test-qsa\|test-sparse-attn\|test-rpc" -j1` | 21/21 passed |
| lavapipe build-vk ctest (te-rows excluded, as in F3) | 17/17 passed |
| lavapipe `test-sparse-attn-vk` | `OK: 0 failed checks` |
| CPU `test-backend-ops -b CPU -o SPARSE_ATTN` / `SPARSE_ATTN_PLAN` / `GET_ROWS,SET_ROWS` | 1439/1439 (1402 of F3 + 30 lead + 6 nanscore n_q 64 + 1 q8_0 2^22+4), 185/185, 1526/1526 |
| `test-qsa-guard <model> fake` (the old mode) | PASS |
| lavapipe `-b Vulkan0 -o SPARSE_ATTN` / `SPARSE_ATTN_PLAN` / `GET_ROWS,SET_ROWS` | 763/763 (742 of F3 + 18 lead + 3 nanscore n_q 64 G1), 92/92, 1082/1082; 0 FAIL |

### Commits
From `git log --format='%h %an | %s' 574689270..HEAD`. Every commit has the author Gaetan Puleo <12990773+gaetan-puleo@users.noreply.github.com> and the session trailer.
- df88ab1bb `ggml : sparse attention gives a -inf score weight 0 wherever the cell lies` (bugs 1, 9). Co-authored-by: Iggy Jackson <iggy@theiggy.com>, because the touched HIP portable / G4 fixup, Vulkan tiers and O_CON adapt #219 code (commits c0ef4a848, 3b93e6acd, 06d8072de, 4e6328f61 carry the same trailer, per `git log --format=%(trailers)`).
- c16d2360c `hip : sparse-attention G4 / G1 attends keep a NaN score while the running max is -inf` (bug 8). Co-authored-by Iggy Jackson: the online softmax of PR #219's qsa3_attn_kernel and its decode WMMA partial (headers of sparse-attn-attend-g4.cuh / -g1.cuh).
- 21f346a6e `vulkan : sparse-attention cm / scalar epilogues and the split-K reduce write 0 for a NaN normaliser` (bugs 2, 3, 10). Co-authored-by Iggy Jackson (06d8072de).
- f805cf9a5 `tests : SPARSE_ATTN whole-graph cases check their sentinels` (bug 6)
- a5b72e078 `tests : the >= 2 GiB SPARSE_ATTN cases select rows past byte INT32_MAX` (bug 7)
- 4e08e5b91 `llama : QSA guard rejects an indexer K cache op that the device leaves to the CPU` (bug 5)
- e31548e7d `llama : QSA reserve graphs take the empty-cache compact decision` (bug 4)

How the commits were made:
- The shared files were split across commits by `F4/stage/stage.py` (local scratchpad `stage/`). It takes the difflib hunks from HEAD to the final content and assigns each hunk to a commit. Version k is HEAD plus the hunks of commits <= k. In A, the lead case loop holds `{SA_LEAD_NEGINF}` only; B adds `SA_LEAD_NAN`.
- `F4/stage/commit.sh` then wrote the index only (`git hash-object -w` + `update-index --cacheinfo`).
- After the last commit, `git diff --quiet HEAD` holds and `git status --short` is empty.

### Per-commit build (bisectability; CPU and Vulkan configurations, `F4/bisect/summary.txt`)
- `F4/bisect.sh` built each commit in `~/dev/wt_lab/F4_bisect`, incrementally (`git archive` + `rsync -c`).
- Configurations: build-cpu, and build-vk with the SPIRV-Headers flags.
- For each commit it ran the `test-qsa|test-sparse-attn` ctest subset and `test-backend-ops -b CPU -o SPARSE_ATTN -p 'lead|nanscore'`.
- Results: every commit had cpu_build 0, vk_build 0 and ctest 18/18.
- The op cases passed 27/27 (A), 42/42 (B), then 48/48 (C to G). The counts grow as each commit adds its cases.
- The first attempt excluded `common/build-info.cpp.in` (rsync pattern `build-*`), so CMake configure failed. It was rerun with the pattern fixed.
- HIP was built only at the final tree (UNPROVEN per commit).
- The bisect tree was deleted afterwards.

### Left for the orchestrator (GPU, NOT RUN here)
- Run `bash ~/dev/wt_rewrite_logs/F4/gpu_rerun.sh` once `~/gpu.lock` is free. Its logs go to `F4/gpu/`.
- The "before" binaries are in `~/dev/wt_lab/F4_before/{build-hip,build-vk}/bin/test-backend-ops`. They were built by `F4/build_before.sh` (rc 0: `F4/before_build/rc.txt`, `build hip rc=0`, `build vk2 rc=0`) from the parent `574689270` (`git archive`) with this step's final `tests/test-backend-ops.cpp`. The first Vulkan configure lacked the SPIRV-Headers flags of `build-vk/CMakeCache.txt`; the second run used them.
- Pass criteria (also in the script header):
  - `hip_lead_before` FAILs the G1 / G4 `,leadneginf` and `,leadnan` cases;
  - `radv_nan_before` and `radv_nan_before_scalar` FAIL the `,nanscore` n_q 1 / 3 / 64 and `,leadnan` cases with NaN outputs;
  - every `_after` run is OK:
    - the default tiers;
    - `GGML_CUDA_SA_FORCE=portable / fixup / generic`;
    - `GGML_VK_SA_FORCE=scalar`;
    - the 2^21+4 and 2^22+4 cases on ROCm0;
    - the ROCm0 / RADV SPARSE_ATTN and SPARSE_ATTN_PLAN suites;
    - the HIP / RADV ctest subsets;
    - `test-qsa-guard <model> pass q8_0` on ROCm0.
- Afterwards, delete `~/dev/wt_lab/F4_before`.
- Unchanged by this step: `~/dev/wt_pr219v2*`, other worktrees, systemd units, strix-001, and pids 4041224 / 4041225.

## F5: fixes for seven confirmed review bugs (6 commits on top of `e31548e7d`)

All work ran on strix-002 in `~/dev/wt_rewrite` (branch `qsa-rewrite`). Logs are in `strix-002:~/dev/wt_rewrite_logs/F5/`:
- `b1/` … `b6/`: one directory per bug, with `before_summary.txt` where a "before" was run;
- `variants/`: the before sources and run scripts;
- `final/`, `bisect/`;
- `gpu_rerun.sh`.

Nothing was pushed or posted.

### Environment constraint: no GPU run was possible
- `~/gpu.lock` was still held by pid 4041224 (`timeout 3600 … build-vk/bin/llama-perplexity …`). Its child, pid 4041225, was in state `D`, at an elapsed time of 10:56:37 when last checked (`lslocks`, `ps -o pid,stat,etime -p 4041225`). That process was not started by this step and was not touched.
- No GPU device was opened. Every HIP (ROCm0) and RADV result below is **NOT RUN**. The runs are scripted in `~/dev/wt_rewrite_logs/F5/gpu_rerun.sh`, every job under `flock ~/gpu.lock`.
- Builds ran without the lock (a build opens no device).
- Tests ran on:
  - the CPU (`build-cpu`);
  - CPU + RPC (`build-rpc`);
  - lavapipe (`~/dev/wt_lab/final_fix/lvp.sh`, the CPU Vulkan ICD: portable tier, no cm tier, no split).

### Scorecard (file:line at HEAD `4d40d7f6a`)
| # | bug (review title, short) | commit | fix | regression test | before | after |
|---|---|---|---|---|---|---|
| 1 | Vulkan TOPK_QSA never fires on the TE route (`te_select` reads `expanded` twice) | `3eb54536f` | `qwen4exp_te_select(ctx, expanded, vis, width)` gathers the visibility from the mask / per-cell bias as stored, through an element view (`qwen4exp-te.h:46-73`); production passes `blk_bias ? query_mask : bias` (`qwen4exp.cpp:1708`). Also: graph_optimize keeps the TOPK_QSA chain's sources allocated through the TOP_K (`add_pattern_alloc_deps`, `ggml-vulkan.cpp:21225`) | (a) `test-qsa-te-rows`: `fusion_uses` checks, per strip of every layout, the use counts that `ggml_vk_can_fuse_topk_qsa` requires (GET_ROWS / PERMUTE / CONT / RESHAPE / ADD 1, CPY 2); (b) `test-backend-ops` `QSA_TE_STRIP` (eval) and census `TOPK_QSA …,strip=te` at n_tps 1 / 2 and split streams | (a) HEAD te_select: `ADD (expanded) has 2 uses, the fusion needs 1: FAIL` on every case (`b1/before_full_te_rows_cpu.log`). (b) lavapipe census, HEAD: all 3 te cases `MISMATCH expected TOPK_QSA/7=1 got none`; also the base pattern at split streams. te_select fix without the alloc deps: n_tps 2 OK, n_tps 1 and split streams MISMATCH (`b1/before_summary.txt`) | te-rows CPU and lavapipe `RESULT PASS`; census 6/6 OK on lavapipe, one process per case (`final/census_lvp.log`); RADV / ROCm0 NOT RUN |
| 2 | multi-sequence contexts walk every cell's std::set on every decode step | `4fc0dd204` | `llama_kv_cells::seq_pos_n_dup` (`llama-kv-cells.h:334`), kept exact by `seq_pos_inc` / `seq_pos_dec` (`:548-577`, neighbour test); `qsa_seq_has_dup_key` returns from it without a walk unless the sequence is M-RoPE with a repeated position (`prefix.h:62-88`) | `test-qsa-dup-key` (new, ctest): exact (40000 random edits, count == recount, predicate == the old walk), work (2^20 cells shared by 4 sequences: entries visited must be 0) | `4194304 set entries visited per decode-step check (want 0), 359722.835 us per step` (strix-002 CPU), FAIL (`b2/dup_key_before.log`) | `0 set entries visited … 0.006 us per step`, OK (`b2/dup_key_after.log`) |
| 3 | NC0 re-split ignores block-selection eligibility | `10d67d49c` | `qwen4exp_block_selection_eligible` (`qwen4exp-qsa-check.h:19`), used by `qwen4exp_use_block_selection` and the memory: NC0 ratios filtered at construction (`llama-memory-hybrid-idx.cpp:115`); `qsa_nc0` and the `resplit` flag require causal attention and a ratio > 1 that divides top_k (`:168`, `:196`); `llama_context` forwards causal attention at init and on `llama_set_causal_attn` (`llama-context.cpp:571`, `:2254`) | `test-qsa-resplit` ineligible phase (-np 2 unified, image in seq 0, a 3000-token prompt of seq 0); `test-qsa-guard <model> causal-sync`; `test-qsa-noncompact-check` eligibility terms | non-causal r4, r1, r3: `5 re-splits, 5 NC0 evaluations` each, largest ubatch 512 (3 violations); causal-sync: `1 after a non-causal init`, FAIL (`b3/before_summary.txt`) | control (causal r4) 5 re-splits; the other three 0 re-splits, 0 evaluations, one 3000-token ubatch; causal-sync `0 / 1 / 0`, `NC0 ratios: 4`, PASS (`b3/*_after.log`) |
| 4 + 7 | indexer guard SET_ROWS probe has the wrong sources; the branch is dead because the scheduler aborts first | `6a713c0f3` | `qsa_idx_types_taken` probes `ggml_set_rows(ctx, a, t->src[0], t->src[1])` (`llama-context.cpp:1039`); `qsa_guard_indexer_writer` applies the scheduler's rule (some backend takes the buffer type and the op) to the registered writers before the scheduler, in `graph_reserve` (throw → init error) and `process_ubatch` (decode error) (`:1070`, called at `:2458` and `:3669`); the shortcut writer `build_qsa_store_k` is registered too (`qwen4exp.cpp:1400`) | `test-qsa-guard <model> fake-idx-w T`: a fake GPU without SET_ROWS into T | HEAD: abort `ggml-backend.cpp:1175: pre-allocated tensor (cache_idx_k_l3 (view)) in a buffer (FAKE_GPU) that cannot run the operation (SET_ROWS)` for q8_0 / iq4_nl. With the writer check and the old probe: abort `ggml.c:4160: GGML_ASSERT(a->ne[0] == b->ne[0]) failed` (`b4/before_summary.txt`) | init error `QSA layer 3: the indexer K cache (q8_0, on FAKE_GPU) is written by SET_ROWS 'cache_idx_k_l3 (view)', … use --cache-type-k-idx auto or a type it takes here: f32, f16, bf16, q4_0, q4_1, q5_0, q5_1, iq4_nl`; PASS for q8_0 / f16 / q4_0 / iq4_nl (`b4/guard_w_*_after.log`). Real HIP + MXFP4: NOT RUN |
| 5 | CPU / O_CON break item 3 for a -inf cell with a non-finite V | `4d40d7f6a` | contract restated (`ggml.h:2670-2676`): a -inf cell has p = 0 but still enters PV as 0 * v. A +-inf / NaN V element makes that dimension NaN; masked cells (item 4) contribute nothing. Code unchanged. Reason: the WMMA tiers (HIP G1, Vulkan cm) share one PV product per block and cannot drop a cell for one row. Skipping in the CPU op alone would make the reference disagree with them | `test-sparse-attn-neginf-v` (new, ctest): CPU op vs O_CON NaN-aware bitwise; NaN exactly in the V's non-finite dims; other dims bitwise equal to O_CON without the cell; all-(-inf) row 0; plus every GPU device of the build (NaN positions equal, finite dims within 1e-3 x max\|CPU\|) | `--old-text` (the replaced statement): `FAILED (24 failed checks)`, e.g. `q 0 h 0 d 5: -nan, want the softmax over the other cells 0.0378884` (`b5/before_oldtext.log`) | CPU OK (24 NaN dims, 5096 compared); lavapipe default and portable: `0 NaN positions differ, max \|GPU - CPU\| 3.91e-08`; scalar not supported on lavapipe (`b5/after_*.log`); HIP / RADV NOT RUN |
| 6 | HIP prefer_dense ignores whether the device runs FLASH_ATTN_EXT | `233cfd10d` | `ggml_backend_cuda_sparse_attn_prefer_dense` returns dense only when the table says dense **and** the device's supports_op takes a FLASH_ATTN_EXT of the shortcut's shape (`ggml_cuda_sa_dense_fa_supported`, the Vulkan probe; `ggml-cuda.cu:9856`, `:9879`) | `test-qsa-prefer-dense` (new, ctest): on every device with the policy, every dense cell (6 x 6 types x 7 bands x 2 n_kv) must be a supported FLASH_ATTN_EXT | NOT RUN (HIP device needed). Before binary ready: `~/dev/wt_lab/F5_nofa_before/bin` (FA=OFF build, HEAD policy; run with `LD_LIBRARY_PATH` set to that dir, RUNPATH) | lavapipe: `dense 0 of 504 cells`, OK; CPU skipped (no policy); `build-hip-nofa` (-DGGML_CUDA_FA=OFF) and `build`: built rc 0, NOT RUN |

### Notes per bug
- **1.**
  - The rows are now PR219's set `top_k(expanded) ∩ {mask == 0}` by construction. The old gather from expanded relied on the bias being finite at every visible cell (the old header comment). The test still checks that precondition (`visible_with_ninf_bias=0`).
  - The element view of the mask works on a strided strip view. Split streams with several strips are covered: `split2_text_L8193_L8195_strips32` passes on the CPU and lavapipe.
  - Fusion still needs a contiguous mask (`ggml_vk_can_fuse_topk_qsa`). A strided split-stream strip view stays unfused, as in the base.
  - The alloc deps were found while testing the fix (`GGML_VK_FUSION_DEBUG=1`):
    - `TOPK_QSA disabled at node 3: source score (permuted) (cont) (CONT) of fused node 0 overlaps the output node_8`
    - `… source kq_mask (copy) (CPY) of fused node 3 overlaps the output`
    - Cause: the TOP_K is an intermediate in the production graph, and ggml-alloc reused those buffers for its output.
  - Peak memory: the deps keep the score copy and the f32 mask cast alive until the TOP_K. That is no more than the ADD's own working set (its two big inputs plus its output). A measured peak: NOT measured (UNPROVEN).
  - Pre-existing on lavapipe: running more than one TOPK_QSA / QSA_TE_STRIP case in one test-backend-ops process aborts with glibc heap corruption (`corrupted size vs. prev_size`, `realloc(): invalid next size`).
    - It also happens with the F4 before binary `~/dev/wt_lab/F4_before/build-vk/bin/test-backend-ops` (HEAD-era code) and with `GGML_VK_DISABLE_FUSION=1`.
    - Each case passes alone (`b1/tbo_lvp_each_after.log`).
    - Cause UNPROVEN; not addressed here. `gpu_rerun.sh` therefore runs one census case per process.
- **2.**
  - The count is per sequence and covers the whole cache, as the old walk did.
  - Only an M-RoPE sequence with a repeated position is still walked: the (y, x) compare, an image.
  - The O(count) cell loop of `qsa_compact_visibility_stream` is unchanged. The review did not flag it.
  - The 359.7 ms figure is for this test's set (4 x 2^20 entries, built in cell order) on strix-002's CPU. The cost inside a real server decode step is UNPROVEN: no model run.
- **3.**
  - Per-context terms are filtered at construction: no alibi (blk_bias), no soft cap, 256-wide heads, no SWA.
  - The mask-shape terms of blk_bias are assumed true. That over-approximates NC0, which keeps invariant (i): graph-NC ⇒ NC0.
  - The init error about `n_rs_seq` now applies only when an eligible ratio exists.
- **4 + 7.**
  - The post-schedule SET_ROWS branch of `qsa_guard_indexer` stays, with the fixed probe. It is reachable only where a CPU-side backend takes the device buffer type (not the case for the fake device or HIP).
  - The writer check throws from `graph_reserve`. In `llama_init_from_model` that becomes a logged init error (`failed to initialize the context: …`). A mid-session reserve throws as the existing `failed to allocate compute pp buffers` path does.
  - MXFP4 on a real HIP device: NOT RUN.
- **5.** The GPU-tier claim (0 * v on every tier) comes from reading the code (`sparse-attn-attend-g4.cuh:772` fixup, `sparse-attn-portable.cu:191`, `sparse_attn_portable.comp:273`). Only the lavapipe portable tier was run. G1, G4 WMMA, Vulkan cm and scalar: UNPROVEN until `gpu_rerun.sh` b5_*.
- **6.**
  - With FA available, `ggml_cuda_get_best_fattn_kernel` returns NONE for D 256 only for a KV type it does not support, or a mask with ne[2] != 1 (`fattn.cu:554-640`). For the table's f16 / bf16 / q8_0 / q4_0 pairs the policy is therefore unchanged when FA supports the pair (code reading, UNPROVEN on the device).
  - Cost: one small no_alloc ggml context per dense query of the policy (once per layer per graph build).

### Build (final tree `4d40d7f6a`, existing configurations)
- `final/build_*.log`: build-cpu, build-vk, build (HIP, rule flags), build-rpc and build-rpc-hip, all rc 0 with 0 `error:` lines.
- `build-hip-nofa` (new: rule flags + `-DGGML_CUDA_FA=OFF`, target `test-qsa-prefer-dense`): rc 0 (`b6/nofa_build.log`, `b6/nofa_after_build.log`).

### Regression runs on the final tree (no GPU; `final/`)
| run | result |
|---|---|
| build-cpu `ctest -R "test-qsa\|test-sparse-attn" -j1` | 21/21 passed (19 of F4 + test-qsa-dup-key + test-sparse-attn-neginf-v; test-qsa-prefer-dense is not matched by the pattern, run separately: OK, skipped on CPU) |
| build-rpc `ctest -R "test-qsa\|test-sparse-attn\|test-rpc" -j1` | 24/24 passed |
| lavapipe build-vk ctest (te-rows excluded, as in F3 / F4) | 20/20 passed |
| lavapipe `test-qsa-te-rows Vulkan` | RESULT PASS |
| lavapipe census TOPK_QSA, 6 cases, one process each | 6 OK, 0 MISMATCH |
| lavapipe `test-sparse-attn-neginf-v`, `test-qsa-prefer-dense` | OK, OK |
| lavapipe `-o SPARSE_ATTN_PLAN` / `GET_ROWS` | 92/92, 643/643 |
| CPU `test-backend-ops -o SPARSE_ATTN / SPARSE_ATTN_PLAN / GET_ROWS / TOPK_QSA / QSA_TE_STRIP / TOP_K` | 1439/1439, 185/185, 655/655, 4/4, 2/2, 587/587 |
| `test-qsa-guard <model>` fake, fake-idx iq4_nl, fake-idx q8_0, fake-idx-w q8_0, fake-idx-w f16, causal-sync | PASS x 6 |

### Commits (`git log --format='%h %an | %s' e31548e7d..HEAD`; author Gaetan Puleo <12990773+gaetan-puleo@users.noreply.github.com>, session trailer on each)
- 3eb54536f `qwen4exp : TE rows read the mask for the visibility, so the Vulkan TOPK_QSA fusion fires again` (bug 1)
- 4fc0dd204 `llama : the QSA duplicate-key check reads a kept count instead of walking every cell of the sequence` (bug 2)
- 10d67d49c `llama : the QSA NC0 re-split applies only where block selection is eligible` (bug 3)
- 6a713c0f3 `llama : the QSA indexer guard checks the cache writer before the scheduler, with the right probe sources` (bugs 4 + 7)
- 233cfd10d `hip : QSA dense shortcut only where the device runs FLASH_ATTN_EXT` (bug 6)
- 4d40d7f6a `ggml : sparse attention item 3 states what every tier does with a -inf cell's non-finite V` (bug 5)

No Co-authored-by trailer was added: none of the touched code is #219 code. `qwen4exp-te.h` was written in 79308ac04, which carries no trailer. The TOPK_QSA pattern is upstream (daef7b687, f172be756). The HIP policy table is the rewrite's AT-W7PERF table.

### Per-commit build (bisectability; CPU configuration, `F5/bisect/summary.txt`)
- `F5/bisect.sh` built each commit in `~/dev/wt_lab/F5_bisect` (`git archive` + `rsync -c`, incremental build-cpu) and ran the `test-qsa|test-sparse-attn` ctest subset.
- Every commit had build 0 and ctest passing: 18, 19, 19, 19, 20, 21 tests, the count growing as commits add tests. The bisect tree was deleted afterwards.
- Vulkan and HIP were built only at the final tree (UNPROVEN per commit).

### Left for the orchestrator (GPU, NOT RUN here)
- Run `bash ~/dev/wt_rewrite_logs/F5/gpu_rerun.sh` once `~/gpu.lock` is free. Logs go to `F5/gpu/`, and the pass criteria are in the script.
- Afterwards, delete `~/dev/wt_lab/F5_nofa_before`, and `build-hip-nofa` too if it is not wanted.
- Unchanged by this step: `~/dev/wt_pr219v2*`, other worktrees, systemd units, strix-001, and pids 4041224 / 4041225.
