---
name: upstream-sync
description: Merge upstream ggml-org/llama.cpp master into this fork every 3 days, one step per conflicting upstream PR, without losing fork code, and validate the sensitive steps on a Strix Halo machine. Use when asked to sync, update from, catch up with, or pull upstream / ggml-org master into the fork.
---

# Upstream sync

Bring `upstream/master` (ggml-org/llama.cpp) into `master` (halo-box/strix-llama.cpp) while keeping every fork change. Read `SYNC.md` and `AGENTS.md` first. halo-box/llama.cpp is archived and is not part of this sync.

## Strategy (do not deviate without asking)

- **Merge, never rebase or squash.** Work on a branch `sync/upstream-<YYYY-MM-DD>`, land it through a PR on halo-box/strix-llama.cpp, and merge it with **"Create a merge commit"**. Squash drops the upstream ancestry (see `SYNC.md`).
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
git switch -c sync/upstream-$(date +%F) master      # first step only
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

### 5. Build after every step (no GPU needed)

```sh
cmake -B build-check -DCMAKE_BUILD_TYPE=Release
cmake --build build-check -j
ctest --test-dir build-check -L main --output-on-failure -j 8

# the HIP kernels are the most customized fork code: compile them at every step (ROCm toolchain or a ROCm container, no GPU)
cmake -B build-hip -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_HIP_COMPILER="$(hipconfig -l)/clang" -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j
```

A clean step can still break a fork kernel (for example a changed signature in `ggml-cuda/common.cuh`), so do not skip the HIP build on steps without conflicts.

`test-tokenizers-ggml-vocabs` needs `git lfs`. Without it, the test reads LFS pointer files and fails with `invalid magic characters: 'vers'`. That is not a merge failure.

Then go back to step 2 for the next step of this sync.

### 6. Push and open a draft PR on the fork

```sh
git push -u origin HEAD
gh pr create --draft --repo halo-box/strix-llama.cpp --base master \
  --title "Merge upstream llama.cpp up to <short-sha> (<date>)" --body-file <file>
```

Always pass `--repo halo-box/strix-llama.cpp`. Never open anything against ggml-org/llama.cpp or halo-box/llama.cpp. The body contains:

- the upstream range, the number of steps and commits;
- for each step, the conflicted files with their resolutions;
- the `verify` output and the build results;
- the validation status (below), and what was and was not tested;
- the agent disclosure required by `AGENTS.md`.

No machine names, home paths or model paths in the PR, the commits or the repo: use placeholders like `<model.gguf>`.

### 7. Validate on a Strix Halo machine

Needed before the PR leaves draft if any step of it had a sensitive conflict. Give the user the full commands below with the placeholders filled in. Do not shorten them. Run them in clean worktrees, never in a tree with uncommitted work.

The baseline is the fork before this sync: the first parent of the first merge of the branch (`$OLD` of the first step), not `origin/master`, which can move during the sync.

```sh
# 0. the GPU must be free
pgrep -a -x 'llama-server|llama-cli|llama-bench|llama-perplexity' && echo "stop these first"

# 1. baseline and candidate, same flags
BASE_REV=<first parent of the first sync merge>
git worktree add --detach ../sync-base $BASE_REV
git worktree add --detach ../sync-cand sync/upstream-<date>
HIP="-DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_HIP_COMPILER=$(hipconfig -l)/clang -DCMAKE_BUILD_TYPE=Release"
for t in ../sync-base ../sync-cand; do
  cmake -S $t -B $t/build $HIP && cmake --build $t/build -j --target llama-perplexity llama-bench llama-server test-backend-ops || break
done
M=<model.gguf>
L=../sync-logs/$(date +%F)
mkdir -p $L

# 2. full test-backend-ops on both builds; only new failures count
for t in base cand; do ../sync-$t/build/bin/test-backend-ops -b ROCm0 > $L/ops-$t.txt 2>&1; done
diff <(grep -E 'FAIL' $L/ops-base.txt | sort) <(grep -E 'FAIL' $L/ops-cand.txt | sort) && echo "ops: no new failure"

# 3. logits vs baseline on the target model: each corpus at -c 2048, all four together at -c 16384 (several ubatches per chunk)
#    the corpora are long enough without repetition (tests/corpus/README.md); a repeated text is mostly copied by the model
C=../sync-cand/tests/corpus
cat $C/correctness-prose.txt $C/correctness-code.txt $C/correctness-structured.txt $C/correctness-numeric.txt > $L/all.txt
sha256sum $C/correctness-*.txt $L/all.txt > $L/corpus.sha256
run() {  # <name> <file> <args> <reference args>
  local O="-m $M -ngl 99 -fa on -ctk f16 -ctv f16"
  ../sync-base/build/bin/llama-perplexity $O $3 -f $2 --kl-divergence-base $L/$1.kld > $L/$1-base.txt 2>&1
  ../sync-cand/build/bin/llama-perplexity $O $3 --kl-divergence-base $L/$1.kld --kl-divergence > $L/$1-cand.txt 2>&1
  ../sync-base/build/bin/llama-perplexity $O $4 --kl-divergence-base $L/$1.kld --kl-divergence > $L/$1-ref.txt 2>&1
}
for n in prose code structured numeric; do
  run $n $C/correctness-$n.txt "-c 2048 -b 2048 -ub 2048" "-c 2048 -b 2048 -ub 512"
done
run long $L/all.txt "-c 16384 -b 512 -ub 512" "-c 16384 -b 256 -ub 256"
grep -H -E 'Mean PPL\(Q\)/PPL\(base\)|Mean +KLD|Same top p' $L/*-cand.txt $L/*-ref.txt

# 4. performance A/B (AGENTS.md command shape), alternating order
for t in base cand base cand; do
  ../sync-$t/build/bin/llama-bench -m $M -p 2048 -d 0,12000,32000,64000 -b 2048 -ub 2048 -n 0 -r 4 -ngl 99 -fa on -ctk f16 -ctv f16 --load-mode none -o md | tee -a $L/bench-pp-$t.md
done
for t in base cand; do
  ../sync-$t/build/bin/llama-bench -m $M -p 0 -n 128 -r 4 -ngl 99 -fa on -ctk f16 -ctv f16 --load-mode none -o md | tee -a $L/bench-tg-$t.md
done

# 5. server: tool call, greedy decode and MTP drafting, on both builds
for t in base cand; do
  ../sync-$t/build/bin/llama-server -m $M -ngl 99 -fa on --jinja --port 8080 --spec-type draft-mtp > $L/server-$t.log 2>&1 &
  S=$!
  until curl -sf localhost:8080/health > /dev/null; do sleep 2; done
  curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"What is the weather in Paris?"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get the weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"temperature":0}' > $L/tool-$t.json
  curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Write a Python function that parses ISO 8601 dates, with tests."}],"max_tokens":256,"temperature":0}' > $L/decode-$t.json
  kill $S; wait $S
done
for t in base cand; do python3 -c 'import json,sys; c=json.load(open(sys.argv[1]))["choices"][0]; print(sys.argv[1], c["finish_reason"], c["message"].get("tool_calls"))' $L/tool-$t.json; done
for t in base cand; do python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); t=r.get("timings",{}); print(sys.argv[1], "accepted", t.get("draft_n_accepted"), "of", t.get("draft_n"))' $L/decode-$t.json; done
cmp <(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["choices"][0]["message"]["content"])' $L/decode-base.json) \
    <(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["choices"][0]["message"]["content"])' $L/decode-cand.json) && echo "decode: identical"
```

Pass criteria:

| Check | Pass |
|---|---|
| 1. build | both builds succeed |
| 2. ops | no FAIL in the candidate that is not in the baseline |
| 3. logits, for each of the 5 runs | `Same top p` of the candidate >= the reference minus 0.5 points, `Mean KLD` <= 2 x the reference + 0.0001, `PPL(Q)/PPL(base)` within 1 +/- 0.01. The reference run is the baseline against itself with a smaller ubatch: other kernel paths, both correct |
| 4. performance | pp and tg overlap the baseline within their standard deviations at every depth. Paste the raw tables |
| 5. server | a valid `get_weather` call on both builds; the same decode text, or the first differing token explained by an upstream change in the step range; MTP acceptance rate within 5 points of the baseline (drop `--spec-type draft-mtp` if the model has no MTP head, and say so) |

If a logits check fails, find the step whose merge moves the logits (rebuild at the merge commits of the branch) before going on: it is either an upstream change or a wrong resolution.

Report one status in the PR, as `AGENTS.md` asks: `TARGET PASS` (all checks pass), `FAIL` (any check fails), or `INCOMPLETE` (a check was not run, for example no MTP head). Attach or summarise the logs of `$L` without local paths.

Then the user takes the PR out of draft, and a maintainer merges it with a merge commit.

## Abort

`git merge --abort` (mid-merge), or `git switch master && git branch -D sync/upstream-<date>`. `master` is never touched before the PR is merged.
