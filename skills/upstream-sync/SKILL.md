---
name: upstream-sync
description: Sync the fork with upstream ggml-org/llama.cpp every 3 days - check for an open sync PR, rebase the fork's commits onto upstream without losing fork code, validate with the benchmarks on a Strix Halo machine, and open the PR with the results. Use when asked to sync, update from, catch up with, or pull upstream / ggml-org master into the fork. To check a sync PR someone else opened, use upstream-sync-verify.
license: MIT
compatibility: Any agent harness that runs shell commands. Needs git 2.43 or later, bash 4.4 or later, gh (logged in), python3 and curl, from the repo root. The benchmark phase needs a Linux Strix Halo (gfx1151) machine with systemd user services (or setsid and nohup), ROCm 10, the Vulkan headers and loader, and glslc (the script downloads the SPIR-V headers).
---

# Upstream sync

Rebase the fork's commits (halo-box/strix-llama.cpp `master`) onto `upstream/master` (ggml-org/llama.cpp) without losing fork code, prove it with the benchmarks, and open the PR. Read `SYNC.md` and `AGENTS.md` first. halo-box/llama.cpp is archived and is not part of this sync.

## Rules (do not deviate without asking)

- **Rebase, never merge or squash.** The fork's commits are replayed onto the upstream tip on a branch, reviewed and validated as a PR, and land by a maintainer's `git push --force-with-lease` of that branch to `master` (see `SYNC.md`). The GitHub merge button must not be used on a sync PR.
- **Fork first.** When a fork commit conflicts, its change wins, adapted to upstream's current code (API, renames, moved code). When upstream rewrote what the fork patches and both cannot live together, keep the fork version of that file or function and list the upstream PRs that are not taken in the PR body.
- **Branch name:** `sync/upstream-<YYYY-MM-DD>-<sha>`, the day of the sync and the short sha of the upstream commit the fork is rebased onto.
- **Every 3 days.** At about 25 upstream PRs a day, a sync brings about 75 upstream PRs. `.github/workflows/upstream-sync-report.yml` posts the plan to an issue as a reminder; its `sync.sh report` still lists where upstream and the fork conflict.
- **Reconcile at the end.** A rebase drops merge commits. The last commit of the sync brings the tree to a 3-way merge of the old `master` and the new upstream tip, from the upstream commit the old `master` sat on (see phase 2), and `forklines.py` must find no fork line lost except the ones the PR body explains.
- **Never drop fork code to make a conflict go away.** Re-apply the fork behaviour on upstream's new code. If the right resolution is unclear, stop and ask.
- **Upstream wins only with numbers.** When upstream now does what a fork change did, take upstream's version only if the fork change was not a gfx1151 optimization, or if `llama-bench` shows upstream is as fast. Otherwise keep the fork code.
- **Explain first, once.** Before the first command, tell the user in a few short lines what this skill will do (the phases, what they change and fetch, from the scripts table below, and roughly how long), then ask "OK?" and wait. Do not ask again at each command: ask again only if the plan changes (more steps, a stop rule, a push or PR that was not announced). A scheduled run with nobody to ask counts as accepted for local work and a draft PR only (see Input).
- **No private data.** No machine names, home paths or model paths in commits, PRs or the repo. Use placeholders like `<model.gguf>`.
- Always pass `--repo halo-box/strix-llama.cpp` to `gh`. Never open anything against ggml-org/llama.cpp or halo-box/llama.cpp.

## Input

The user may give the upstream commit to rebase onto. Without it, show the plan of phase 1 and ask. When nobody can answer (a scheduled run), rebase onto the upstream tip and open the PR as a draft.

## Phases

Do them in order. Each phase has its full instructions in `references/`: read the file when you reach the phase.

1. **Check** ([references/1-check.md](references/1-check.md)): no sync PR already open, the repo prepared (upstream remote, rerere), the conflict plan read and the target upstream commit agreed with the user.
2. **Rebase** ([references/2-steps.md](references/2-steps.md)): replay the fork's commits onto the target, resolving each conflict fork first, then reconcile the tree with the 3-way merge, audit the fork lines, and build CPU, HIP and Vulkan. End with `ctest`.
3. **Write the PR body** ([references/3-pr-body.md](references/3-pr-body.md)): generate it with `summary`, explain every edit outside the conflicts, choose the models from "Where upstream changes fork code" and `models`, fill the models tables and the checklist.
4. **Verify with the benchmarks** ([references/4-benchmarks.md](references/4-benchmarks.md)): on a Strix Halo machine, `validate.sh` compares the fork before the sync with the sync branch on HIP and Vulkan (op tests, every arch `test-llama-archs` supports, a small model in every quant, KLD, `llama-bench` up to 128k depth, server, memory) and gives one status per backend.
5. **Create the PR** ([references/5-create-pr.md](references/5-create-pr.md)): the body with the results, as a draft unless both backends pass. A maintainer lands it by force-push.

## Scripts

| Command | Does | Changes | Network |
|---|---|---|---|
| `sync.sh report` | the step plan and the next step in detail (about a minute) | nothing (only unreferenced git objects for the plan) | fetches `upstream` |
| `forklines.py <base> <old master> <ignore> <target>` | the fork lines (old `master` against the upstream commit it sat on) missing from the working tree | nothing | none |
| `sync.sh models <old>`, `sync.sh owners <old>` | the model archs the sync concerns; where upstream changes fork code (`git blame`) | nothing | none |
| `validate.sh` | builds the fork before the sync and the sync branch for HIP and Vulkan, runs every check, writes `summary.md` (hours) | an output directory (`../sync-validate/<sha>`: the worktrees and the candidate builds) and the baseline builds next to it (`../sync-validate/base-builds`, reused by later runs), about 40 GB the first time; the model cache, the GPU for the whole run; it runs the candidate's code with your environment. With `--pr`: rewrites the results section of the PR body, or posts a comment (`--post comment`) | fetches `origin`; downloads models (Hugging Face, anonymous) and the SPIR-V headers (GitHub) |

All are in `skills/upstream-sync/scripts/` (not the repo's `scripts/`); run them from the repo root. `sync.sh merge`, `verify` and `summary` belong to the merge workflow used until 2026-10-14 and are not used by a rebase sync.

Example of the explanation at the start: "I will fetch upstream and show the plan, rebase the fork's commits onto it on a new branch `sync/upstream-<date>-<sha>` (local only), then on this machine build and benchmark the fork before the sync and the branch (HIP and Vulkan: <duration from 4.2>, the GPU busy, about 40 GB in `../sync-validate/`, models downloaded to the cache), and at the end push the branch and open the PR. OK?"

## Abort

`git rebase --abort` in the middle of the rebase, or `git switch master && git branch -D <sync branch>`. `master` is never touched before the PR is approved and landed.
