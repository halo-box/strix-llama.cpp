---
name: upstream-sync
description: Merge upstream ggml-org/llama.cpp master into this fork in small steps, without losing fork code, and validate each step on a Strix Halo machine. Use when asked to sync, update from, catch up with, or pull upstream / ggml-org master into the fork.
---

# Upstream sync

Bring `upstream/master` (ggml-org/llama.cpp) into `master` (halo-box/strix-llama.cpp) while keeping every fork change. Read `SYNC.md` and `AGENTS.md` first. halo-box/llama.cpp is not part of this sync.

## Strategy (do not deviate without asking)

- **Merge, never rebase or squash.** Work on a branch `sync/upstream-<YYYY-MM-DD>`, land it through a PR on halo-box/strix-llama.cpp, and merge it with **"Create a merge commit"**. Squash drops the upstream ancestry (see `SYNC.md`).
- **Use the content base, not `git merge-base`.** `scripts/upstream-sync.base` holds the upstream commit whose content is really in the fork. After the revert of #155, `git merge-base` claims upstream 2026-10-08 while the content is 2026-09-29, so a plain `git merge` skips 260 upstream commits. `sync.sh merge` merges from the content base and updates the file in the merge commit.
- **Small steps.** One step ends at the first conflicting upstream commit, or after 100 commits (`MAX_STEP`). Several steps can go on the same branch. Build after each step.
- **Never drop fork code to make a conflict go away.** If the right resolution is unclear (for example upstream rewrote the function the fork patches), stop and ask.
- **When upstream now does what a fork change did, upstream wins:** take theirs, remove our copy, and say so in the merge message.

## Steps

### 1. Report

```sh
git remote add upstream https://github.com/ggml-org/llama.cpp   # once
git config rerere.enabled true                                  # remember resolutions across steps and retries
skills/upstream-sync/sync.sh report
```

It prints the content base and the history base (with a warning if they differ), the next step target, and for each conflicted file the fork commits that own our side and the upstream commits on theirs. Summarise this for the user before merging.

### 2. Merge one step

```sh
OLD=$(git rev-parse HEAD)
git switch -c sync/upstream-$(date +%F) master      # or stay on the sync branch for the next step
skills/upstream-sync/sync.sh merge                  # or: sync.sh merge <upstream-sha>
```

The tree must be clean. The merge stops before the commit, with normal conflict state (`git status`, `git checkout --ours/--theirs`, `git mergetool` and rerere all work).

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

### 5. Build where you are

```sh
cmake -B build-check -DCMAKE_BUILD_TYPE=Release
cmake --build build-check -j
ctest --test-dir build-check -L main --output-on-failure -j 8
```

This catches API breaks in shared code (common/, src/, server, ggml core) and tokenizer regressions. It does not compile the HIP kernels. If ROCm is installed, also run the HIP build from step 7.

### 6. Push and open a draft PR on the fork

```sh
git push -u origin HEAD
gh pr create --draft --repo halo-box/strix-llama.cpp --base master \
  --title "Merge upstream llama.cpp up to <short-sha> (<date>)" --body-file <file>
```

Always pass `--repo halo-box/strix-llama.cpp`. Never open anything against ggml-org/llama.cpp or halo-box/llama.cpp. The body contains: upstream range and commit count, conflicted files with their resolutions, the `verify` output, what was and was not tested, the checklist below, and the agent disclosure required by `AGENTS.md`.

### 7. Hand off to the Strix Halo machine

The PR cannot leave draft until these checks pass on a gfx1151 machine. Give the user the **full commands below**, with the date, the op list and the model path filled in. Do not shorten them. Run them in clean worktrees, never in a tree with uncommitted work.

```sh
# 0. the GPU must be free
pgrep -a -x 'llama-server|llama-cli|llama-bench|llama-perplexity' && echo "stop these first"

# 1. baseline (master) and candidate (sync branch), same flags
git fetch origin
git worktree add --detach ../strix-master origin/master 2>/dev/null || git -C ../strix-master checkout --detach origin/master
git worktree add ../strix-sync sync/upstream-<date> 2>/dev/null || git -C ../strix-sync checkout sync/upstream-<date>
HIP="-DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_HIP_COMPILER=$(hipconfig -l)/clang -DCMAKE_BUILD_TYPE=Release"
for t in ../strix-master ../strix-sync; do
  cmake -S $t -B $t/build $HIP && cmake --build $t/build -j --target llama-perplexity llama-bench llama-server test-backend-ops || break
done
M=<model.gguf>

# 2. ops touched by this sync (upstream or fork side): all must be OK. If one fails, run it on master too: only new failures count.
../strix-sync/build/bin/test-backend-ops -b ROCm0 -o <MUL_MAT,MUL_MAT_ID,TOP_K,ARGSORT,GATED_DELTA_NET,FLASH_ATTN_EXT>

# 3. logits vs master on the target model, same bytes for both
for i in $(seq 8); do cat ../strix-sync/tests/corpus/correctness-*.txt; done > /tmp/sync-corpus.txt
sha256sum /tmp/sync-corpus.txt
A="-m $M -c 2048 -b 4096 -ub 4096 -ngl 99 -fa on -ctk f16 -ctv f16"
../strix-master/build/bin/llama-perplexity $A -f /tmp/sync-corpus.txt --kl-divergence-base /tmp/sync-master.kld
../strix-sync/build/bin/llama-perplexity $A --kl-divergence-base /tmp/sync-master.kld --kl-divergence | tee /tmp/sync-kld.txt
# noise reference: master against itself with another ubatch size (other kernel paths, both correct)
../strix-master/build/bin/llama-perplexity ${A/-ub 4096/-ub 512} --kl-divergence-base /tmp/sync-master.kld --kl-divergence | tee /tmp/sync-kld-ref.txt
grep -E 'Mean PPL\(Q\)/PPL\(base\)|Mean +KLD|Same top p' /tmp/sync-kld.txt /tmp/sync-kld-ref.txt

# 4. performance A/B (AGENTS.md command shape), alternating order
for b in ../strix-master ../strix-sync ../strix-master ../strix-sync; do
  $b/build/bin/llama-bench -m $M -p 2048 -d 0,12000,32000,64000 -b 4096 -ub 4096 -n 0 -r 4 -ngl 99 -fa on -ctk f16 -ctv f16 --load-mode none -o md
done
for b in ../strix-master ../strix-sync; do
  $b/build/bin/llama-bench -m $M -p 0 -n 128 -r 4 -ngl 99 -fa on -ctk f16 -ctv f16 --load-mode none -o md
done

# 5. server: tool call and MTP drafting, on master then on the sync build
for b in ../strix-master ../strix-sync; do
  $b/build/bin/llama-server -m $M -ngl 99 -fa on --jinja --port 8080 --spec-type draft-mtp > /tmp/sync-server.log 2>&1 &
  S=$!
  until curl -sf localhost:8080/health > /dev/null; do sleep 2; done
  curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"What is the weather in Paris?"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get the weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"temperature":0}' \
    | python3 -c 'import json,sys; c=json.load(sys.stdin)["choices"][0]; print(c["finish_reason"], c["message"].get("tool_calls"))'
  curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Write a Python function that parses ISO 8601 dates, with tests."}],"max_tokens":300,"temperature":0}' \
    | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"][:300]); print(r.get("timings"))'
  kill $S; wait $S
done
```

Pass criteria:

- Step 1: both builds succeed.
- Step 2: every case OK, or the same failures as master.
- Step 3: `Same top p` and `Mean KLD` of the sync build are no worse than the noise reference (within the printed uncertainty), and `PPL(Q)/PPL(base)` is about 1. If not, find the upstream commit inside the step that moves the logits before going on.
- Step 4: pp and tg overlap master within their standard deviations at every depth. Paste the raw tables in the PR.
- Step 5: `tool_calls` with a valid `get_weather` call on both builds; coherent text, and `draft_n_accepted` > 0 in `timings` (drop `--spec-type draft-mtp` if the model has no MTP head, and say so).

Then the user takes the PR out of draft, and a maintainer merges it with a merge commit. After the merge, the next sync starts again at step 1.

## Abort

`git merge --abort` (mid-merge), or `git switch master && git branch -D sync/upstream-<date>`. `master` is never touched before the PR is merged.
