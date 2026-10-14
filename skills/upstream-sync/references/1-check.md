# 1. Check

Part of the `upstream-sync` skill: read [SKILL.md](../SKILL.md) first for the rules.

## 1.1 Is a sync already open?

```sh
gh pr list --repo halo-box/strix-llama.cpp --state open --search "upstream llama.cpp in:title"
```

If one is open, do not start a second one: two sync PRs at the same time conflict with each other.

- Opened by the same user: continue it. Its branch is the start of your work (`gh pr checkout <number>`).
- Opened by someone else: never push to their branch (`AGENTS.md`). Verify it with the `upstream-sync-verify` skill, or ask its author on the PR, and stop.

## 1.2 Prepare the repo (once per clone)

```sh
git remote add upstream https://github.com/ggml-org/llama.cpp
git config rerere.enabled true
git fetch --multiple origin upstream
```

`origin` must be halo-box/strix-llama.cpp. If it is a personal fork, push the sync branch there and open the PR with `--head <owner>:<branch>` (phase 5).

rerere records each conflict resolution and replays it when the same conflict comes back, which saves work across steps and retries. It also replays a throwaway resolution from a test, and its cache is shared by all worktrees: after a test merge, run `git rerere forget <file>` for each file, or clear `.git/rr-cache`.

## 1.3 Read the plan

```sh
git switch master && git pull
skills/upstream-sync/scripts/sync.sh report
```

It fetches upstream and prints:

- **the divergence**: the fork head, the upstream tip, the content base, and the history base. If they differ, it warns that a plain `git merge` would skip upstream commits; `sync.sh merge` handles it.
- **the plan**: one line per step, with the number of upstream commits, the upstream PR that ends the step, and its conflicted files. `*` marks a sensitive file: kernels, ggml core and quant formats, the llama library, MTP, conversion (see `SENSITIVE` in `sync.sh`). Each step is simulated as resolved by keeping the fork side, so an upstream PR that changes the same fork code again is a step of its own; real resolutions differ, so later steps can still shift a little. It takes about a minute.
- **the models concerned** by the whole pending range (see 3.3).
- **the next step in detail**: for each conflicted file, the fork commits that own our side and the upstream commits on theirs.

Show the plan to the user. The user decides how many steps go in this PR. Use the stop rules of 2.6 to propose a cut: about 3 sensitive steps, or about 10 conflicted files.

**Running alone** (a scheduled run, nobody to ask): merge steps until the first stop rule of 2.6 applies, and open the PR as a draft (`AGENTS.md`: "open the PR as a draft describing the problem"). A `verify: REVIEW` line you cannot explain from the upstream change is such a stop.

`report` also works on the sync branch: it then plans from where the branch is, and details the next step.
