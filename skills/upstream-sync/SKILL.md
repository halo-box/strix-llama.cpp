---
name: upstream-sync
description: Merge upstream ggml-org/llama.cpp master into this fork every 3 days, one step per conflicting upstream PR, without losing fork code, and validate the sensitive steps on a Strix Halo machine. Use when asked to sync, update from, catch up with, or pull upstream / ggml-org master into the fork.
---

# Upstream sync

Bring `upstream/master` (ggml-org/llama.cpp) into `master` (halo-box/strix-llama.cpp) while keeping every fork change. Read `SYNC.md` and `AGENTS.md` first. halo-box/llama.cpp is archived and is not part of this sync.

## Strategy (do not deviate without asking)

- **Merge, never rebase or squash.** Work on a branch `sync/upstream-<YYYY-MM-DD>-<sha>`: the day of the sync and the short sha of the last upstream commit it merges (the end of the last planned step), for example `sync/upstream-2026-10-12-8019dc563`. Then land it through a PR on halo-box/strix-llama.cpp, and merge it with **"Create a merge commit"**. Squash drops the upstream ancestry (see `SYNC.md`).
- **Every 3 days.** At about 25 upstream PRs a day, a sync brings about 75 PRs and a handful of conflicts. A scheduled workflow (`.github/workflows/upstream-sync-report.yml`) posts the plan to an issue as a reminder.
- **One step per conflicting upstream PR.** A step merges all clean upstream PRs up to and including the next PR that conflicts with the fork, so each step has the conflicts of one upstream PR only. A clean run is cut every 100 commits (`MAX_STEP`). All steps of a sync go on the same branch, one merge commit each.
- **Use the content base, not `git merge-base`.** `scripts/upstream-sync.base` holds the upstream commit whose content is really in the fork. After the revert of #155, `git merge-base` claims upstream 2026-10-08 while the content is 2026-09-29, so a plain `git merge` skips 260 upstream commits. `sync.sh merge` merges from the content base and updates the file in each merge commit.
- **Never drop fork code to make a conflict go away.** If the right resolution is unclear (for example upstream rewrote the function the fork patches), stop and ask.
- **When upstream now does what a fork change did, upstream wins**, but only with numbers: if the fork code was a gfx1151 optimization, show `llama-bench` before/after in the PR. Without numbers, keep the fork code.

## Steps

### 1. Report and plan

```sh
git remote add upstream https://github.com/ggml-org/llama.cpp   # once
git config rerere.enabled true                                  # remember resolutions across steps and retries
skills/upstream-sync/sync.sh report
```

It prints:

- the content base and the history base, with a warning if they differ;
- **the plan**: one line per step, with the number of upstream commits, the PR that ends it, and its new conflicted files (`*` = sensitive, see `SENSITIVE` in `sync.sh`). The plan is computed against the fork as it is now, so later steps can shift a little once earlier ones are resolved;
- the next step in detail: for each conflicted file, the fork commits that own our side and the upstream commits on theirs.

Show the plan to the user. The user decides how many steps go in this sync PR. During a catch-up, end each PR after a few sensitive steps, so a regression stays easy to locate.

rerere replays any resolution it recorded, also a throwaway one from a test (the cache is shared by all worktrees). After a test merge, run `git rerere forget <file>` for each file, or clear `.git/rr-cache`. When rerere resolved a file for you, review it with `git rerere diff` before you commit.

### 2. Merge one step

```sh
T=<short sha of the last upstream commit of this sync, from the plan>
git switch -c sync/upstream-$(date +%F)-$T master   # first step only
OLD=$(git rev-parse HEAD)
skills/upstream-sync/sync.sh merge                  # next step of the plan, or: sync.sh merge <upstream-sha>
```

The tree must be clean. The merge stops before the commit, with normal conflict state (`git status`, `git checkout --ours/--theirs`, `git mergetool` and rerere all work). It says if the step has sensitive conflicts.

### 3. Resolve conflicts

For each conflicted file:

1. Read the fork commits that own our side (from the report, or `git log --no-merges HEAD ^upstream/master -- <file>`), and the PRs they came from.
2. Read what upstream changed: `git log -p <base>..MERGE_HEAD -- <file>`.
3. Re-apply the fork behaviour on top of upstream's new code. Do not keep our stale copy of code that upstream refactored.
4. `git checkout --theirs <file>` only when the whole fork diff of that file is now upstream. Otherwise resolve hunk by hunk.

Then edit the merge message (prefilled by `sync.sh merge`): one line per conflicted file, saying how it was resolved and which fork change it keeps.

```sh
git add <files>
git commit
```

### 4. Verify nothing was lost

```sh
skills/upstream-sync/sync.sh verify $OLD
```

- **Files no longer different from upstream:** each one must be explained by upstream now doing the same thing. Anything else is lost fork code: fix it before going on.
- **Files newly different from upstream:** should only be conflict fixups.
- **Fork diff size changed:** inspect with `git diff <target> HEAD -- <file>` and confirm the fork change is intact.
- It fails if HEAD is not a merge of the target, or if conflict markers are left.

### 5. Build (no GPU needed)

The fork supports two GPU backends, HIP (ROCm) and Vulkan, and every sync PR must build and validate both.

After every step, rebuild incrementally the backend the step can break. A clean step can still break fork code (for example a changed signature in `ggml-cuda/common.cuh`), so the rule follows the paths changed by the step, not the conflicts:

| The step changes | HIP build | Vulkan build |
|---|---|---|
| `ggml/src/ggml-cuda/` | yes | no |
| `ggml/src/ggml-vulkan/` | no | yes |
| ggml core (`ggml/include/`, `ggml/src/ggml*.c*`, `ggml/src/ggml-backend*`, `ggml/src/ggml-alloc*`) or `src/` | yes | yes |
| anything else (docs, UI, server, CI) | no | no |

```sh
git diff --name-only HEAD^1 HEAD     # the paths changed by the step
# HIP: ROCm toolchain or a ROCm container. Vulkan: Vulkan headers and glslc. ccache helps both.
cmake -B build-hip -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_HIP_COMPILER="$(hipconfig -l)/clang" -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j
cmake -B build-vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vk -j
```

An incremental build is fast when the step changes a few kernel files, and close to a full build when it changes a header included everywhere (`ggml-cuda/common.cuh` is included by most `.cu` files).

At the end of the sync, before the PR: both GPU builds again, and the CPU tests.

```sh
cmake -B build-check -DCMAKE_BUILD_TYPE=Release
cmake --build build-check -j
ctest --test-dir build-check -L main --output-on-failure -j 8
```

`test-tokenizers-ggml-vocabs` needs `git lfs`. Without it, the test reads LFS pointer files and fails with `invalid magic characters: 'vers'`. That is not a merge failure.

Then go back to step 2 for the next step of this sync, unless a stop rule below applies.

### When to stop a sync

Stop **before** the next step, and open the PR with the steps already merged, when one of these is true:

1. **The resolution is not clear.** For example upstream rewrote a function that the fork patches, or upstream added its own version of a fork feature (MTP, QSA, the UMA ring).
2. **The PR is big enough to review and validate.** About 3 sensitive steps, or about 10 conflicted files in total. Adjust with the user.
3. **A validation fails** and the cause is not found quickly.

An upstream PR cannot be skipped: every later upstream commit contains it. A hard step waits for its own sync PR:

- put it alone in a draft PR that describes the problem;
- ask the authors of the fork code it touches, or the maintainers, which behaviour to keep;
- as a last resort, take the upstream version and port the fork feature onto it, with `llama-bench` and KLD numbers (see "upstream wins" above).

Meanwhile `master` stays at the last merged step: nothing breaks, the fork is only a few days behind.

### 6. Push and open a draft PR on the fork

If the last merged upstream commit is not the one in the branch name (the plan shifted, or the sync stopped early), rename the branch first: `git branch -m sync/upstream-<date>-<sha of the last merged upstream commit>`.

```sh
git push -u origin HEAD
gh pr create --draft --repo halo-box/strix-llama.cpp --base master \
  --title "Merge upstream llama.cpp up to <short-sha> (<date>)" --body-file <file>
```

Always pass `--repo halo-box/strix-llama.cpp`. Never open anything against ggml-org/llama.cpp or halo-box/llama.cpp.

The first part of the body is generated, so it lists every change of the sync:

```sh
OLD0=$(git merge-base HEAD origin/master)            # the fork before the first step
skills/upstream-sync/sync.sh summary $OLD0 > body.md
```

It contains, for each step: the upstream PR that ends it and its upstream commits, the conflicted files (`*` = sensitive) with the resolution line of the merge message, **the conflict zones** (line ranges of the merged file), the resolution diff (automatic merge with markers -> committed), and **every edit outside the conflicts**: a change in a file that did not conflict, or in a conflicted file outside its conflict zones. Nothing marks such edits during the merge, so each one needs an explanation in the PR. The body ends with the diffstat of the whole sync. GitHub limits a PR body to 65536 characters: if `body.md` is longer, keep the overview table in the body and post the `## Steps` part as PR comments, one per step.

Then append the validation part, keep it short, and tick a box only for a check that really ran:

```markdown
## Validation

- [ ] after every step: `sync.sh verify` (no fork code lost, no conflict marker) and the GPU builds the step can break
- [ ] end of sync: HIP (gfx1151) and Vulkan builds, CPU build and `ctest -L main`
- [ ] Strix Halo checklist (step 7), needed if a step is sensitive. Models: <arch (GGUF name)>, one table per model:

| Check | HIP (ROCm0) | Vulkan (Vulkan0) |
|---|---|---|
| `test-backend-ops`: no new failure | [ ] | [ ] |
| `test-llama-archs`, every arch: no new failure | [ ] | [ ] |
| KLD, 4 corpora at `-c 2048` and all at `-c 16384`: within the thresholds | [ ] | [ ] |
| `llama-bench` pp and tg: within noise at every depth (tables below) | [ ] | [ ] |
| server: tool call, same greedy decode, MTP acceptance within 5 points | [ ] | [ ] |
| Status (TARGET PASS / FAIL / INCOMPLETE) | | |

## Requirements

- AI usage disclosure: <NO / ASSISTED / AGENT-AUTHORED, and how>
- What was NOT verified: <every unticked box above, and why>
```

No machine names, home paths or model paths in the PR, the commits or the repo: use placeholders like `<model.gguf>`.

### 7. Validate on a Strix Halo machine

Needed before the PR leaves draft if any step of it had a sensitive conflict, and then on **both backends, HIP (ROCm) and Vulkan**. Give the user the full commands below with the placeholders filled in. Do not shorten them. Run them in clean worktrees, never in a tree with uncommitted work.

Models, as listed by `sync.sh report <last upstream sha of this PR>` under "models to test":

- **every arch**, including all the archs upstream changed: `test-llama-archs` builds a tiny random model per arch and compares each GPU backend with the CPU. No download, a few minutes;
- **a real GGUF** for each arch whose `src/models/<arch>.cpp` both upstream and the fork changed: KLD, bench and server. If the sync also changes shared paths (kernels, ggml core, graph, KV cache), ask the user which real models to add. Check the arch of a GGUF with `python3 gguf-py/gguf/scripts/gguf_dump.py --no-tensors <model.gguf> | grep general.architecture`.

The baseline is the fork before this sync: the first parent of the first merge of the branch (`$OLD` of the first step), not `origin/master`, which can move during the sync.

```sh
# 0. the GPU must be free
pgrep -a -x 'llama-server|llama-cli|llama-bench|llama-perplexity' && echo "stop these first"

# 1. baseline and candidate, one build per backend, same flags
BASE_REV=<first parent of the first sync merge>
git worktree add --detach ../sync-base $BASE_REV
git worktree add --detach ../sync-cand sync/upstream-<date>-<sha>
flags() { case $1 in hip) echo "-DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_HIP_COMPILER=$(hipconfig -l)/clang";; vk) echo "-DGGML_VULKAN=ON";; esac; }
dev()   { case $1 in hip) echo ROCm0;; vk) echo Vulkan0;; esac; }
for be in hip vk; do
  for t in base cand; do
    cmake -S ../sync-$t -B ../sync-$t/build-$be $(flags $be) -DCMAKE_BUILD_TYPE=Release && \
      cmake --build ../sync-$t/build-$be -j --target llama-perplexity llama-bench llama-server test-backend-ops test-llama-archs || break 2
  done
done
MODELS="<arch>=<model.gguf> <arch>=<model.gguf>"   # one GGUF per arch to test
L=../sync-logs/$(date +%F)
C=../sync-cand/tests/corpus
mkdir -p $L/hip $L/vk
cat $C/correctness-prose.txt $C/correctness-code.txt $C/correctness-structured.txt $C/correctness-numeric.txt > $L/all.txt
sha256sum $C/correctness-*.txt $L/all.txt > $L/corpus.sha256

for be in hip vk; do
  bin() { echo ../sync-$1/build-$be/bin; }
  D=$L/$be

  # 2. full test-backend-ops on both builds; only new failures count
  for t in base cand; do $(bin $t)/test-backend-ops -b $(dev $be) > $D/ops-$t.txt 2>&1; done
  diff <(grep FAIL $D/ops-base.txt | sort) <(grep FAIL $D/ops-cand.txt | sort) > $D/ops-new-fail.txt && echo "$be ops: no new failure"

  # 2b. every model arch: tiny random models, GPU vs CPU logits; only new failures count
  for t in base cand; do $(bin $t)/test-llama-archs -b $(dev $be) 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > $D/archs-$t.txt; done
  diff <(grep FAIL $D/archs-base.txt | sort) <(grep FAIL $D/archs-cand.txt | sort) > $D/archs-new-fail.txt && echo "$be archs: no new failure"

  for mm in $MODELS; do
  M=${mm#*=}
  D=$L/$be/${mm%%=*}
  mkdir -p $D

  # 3. logits vs baseline: each corpus at -c 2048, all four together at -c 16384 (several ubatches per chunk)
  run() {  # <name> <file> <args> <reference args>
    local O="-m $M -ngl 99 -fa on -ctk f16 -ctv f16"
    $(bin base)/llama-perplexity $O $3 -f $2 --kl-divergence-base $D/$1.kld > $D/$1-base.txt 2>&1
    $(bin cand)/llama-perplexity $O $3 --kl-divergence-base $D/$1.kld --kl-divergence > $D/$1-cand.txt 2>&1
    $(bin base)/llama-perplexity $O $4 --kl-divergence-base $D/$1.kld --kl-divergence > $D/$1-ref.txt 2>&1
  }
  for n in prose code structured numeric; do
    run $n $C/correctness-$n.txt "-c 2048 -b 2048 -ub 2048" "-c 2048 -b 2048 -ub 512"
  done
  run long $L/all.txt "-c 16384 -b 512 -ub 512" "-c 16384 -b 256 -ub 256"
  grep -H -E 'Mean PPL\(Q\)/PPL\(base\)|Mean +KLD|Same top p' $D/*-cand.txt $D/*-ref.txt

  # 4. performance A/B (AGENTS.md command shape), alternating order
  for t in base cand base cand; do
    $(bin $t)/llama-bench -m $M -p 2048 -d 0,12000,32000,64000 -b 2048 -ub 2048 -n 0 -r 4 -ngl 99 -fa on -ctk f16 -ctv f16 --load-mode none -o md | tee -a $D/bench-pp-$t.md
  done
  for t in base cand; do
    $(bin $t)/llama-bench -m $M -p 0 -n 128 -r 4 -ngl 99 -fa on -ctk f16 -ctv f16 --load-mode none -o md | tee -a $D/bench-tg-$t.md
  done

  # 5. server: tool call, greedy decode and MTP drafting
  for t in base cand; do
    $(bin $t)/llama-server -m $M -ngl 99 -fa on --jinja --port 8080 --spec-type draft-mtp > $D/server-$t.log 2>&1 &
    S=$!
    until curl -sf localhost:8080/health > /dev/null; do sleep 2; done
    curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"What is the weather in Paris?"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get the weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"temperature":0}' > $D/tool-$t.json
    curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Write a Python function that parses ISO 8601 dates, with tests."}],"max_tokens":256,"temperature":0}' > $D/decode-$t.json
    kill $S; wait $S
  done
  for t in base cand; do python3 -c 'import json,sys; c=json.load(open(sys.argv[1]))["choices"][0]; print(sys.argv[1], c["finish_reason"], c["message"].get("tool_calls"))' $D/tool-$t.json; done
  for t in base cand; do python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); t=r.get("timings",{}); print(sys.argv[1], "accepted", t.get("draft_n_accepted"), "of", t.get("draft_n"))' $D/decode-$t.json; done
  cmp <(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["choices"][0]["message"]["content"])' $D/decode-base.json) \
      <(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["choices"][0]["message"]["content"])' $D/decode-cand.json) && echo "$be $mm decode: identical"
  done
done
```

Pass criteria, for each backend and each model:

| Check | Pass |
|---|---|
| 1. build | both builds succeed |
| 2. ops | no FAIL in the candidate that is not in the baseline |
| 2b. archs | no FAIL in `test-llama-archs` of the candidate that is not in the baseline |
| 3. logits, for each of the 5 runs | `Same top p` of the candidate >= the reference minus 0.5 points, `Mean KLD` <= 2 x the reference + 0.0001, `PPL(Q)/PPL(base)` within 1 +/- 0.01. The reference run is the baseline against itself with a smaller ubatch: other kernel paths, both correct |
| 4. performance | pp and tg overlap the baseline within their standard deviations at every depth. Paste the raw tables |
| 5. server | a valid `get_weather` call on both builds; the same decode text, or the first differing token explained by an upstream change in the step range; MTP acceptance rate within 5 points of the baseline (drop `--spec-type draft-mtp` if the model has no MTP head, and say so) |

If a logits check fails, find the step whose merge moves the logits (rebuild at the merge commits of the branch) before going on: it is either an upstream change or a wrong resolution.

Report one status per backend and model in the PR, as `AGENTS.md` asks: `TARGET PASS` (all checks pass), `FAIL` (any check fails), or `INCOMPLETE` (a check was not run, for example no MTP head). The PR leaves draft only when both are `TARGET PASS`. Attach or summarise the logs of `$L` without local paths.

Then the user takes the PR out of draft, and a maintainer merges it with a merge commit.

## Abort

`git merge --abort` (mid-merge), or `git switch master && git branch -D sync/upstream-<date>-<sha>`. `master` is never touched before the PR is merged.
