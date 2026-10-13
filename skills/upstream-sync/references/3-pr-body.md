# 3. Write the PR body

Part of the `upstream-sync` skill: read [SKILL.md](../SKILL.md) first for the rules.

## 3.1 Generate the changes

```sh
git fetch origin
OLD0=$(git merge-base HEAD origin/master)            # the fork before the first step
# for a PR stacked on another branch, use that branch instead of master
skills/upstream-sync/scripts/sync.sh summary $OLD0 > body.md
```

For each merge commit, `summary` rebuilds the automatic merge from the content base and compares it with the commit. It writes:

- **Overview**: the upstream range, and one row per step with the upstream PR that ends it, its upstream commits, its conflicted files (`*` = sensitive) and its edits outside conflicts;
- **Steps**: for each conflicted file, the resolution line of the merge message, the **conflict zones** (line ranges of the merged file), every **edit outside the conflict zones** of that file, and the resolution diff (automatic merge with markers -> committed); the edits in files that did not conflict; the list of upstream commits;
- **Where upstream changes fork code**: the upstream hunks that overlap the fork's own lines or come within 10 lines of them, kernels included, with the fork commits that own those lines according to `git blame` (fork reverts looked through);
- **All changes**: the diffstat of the sync.

## 3.2 Explain what the merge did not mark

Every **edit outside the conflicts** was made by hand and no conflict marker shows it. Explain each one in the body, next to its step.

## 3.3 Choose the models

The fork does not only change model files: its kernels (MMQ, MMVQ, GDN, flash attention, top-k...) run for many models. **Where upstream changes fork code** is the table that decides what to validate: for each row, which ops and which models run through that fork code.

```sh
skills/upstream-sync/scripts/sync.sh models $OLD0     # target: what the last sync merge took from upstream
```

prints the table of `## Models concerned` (not the heading): every model arch whose model code upstream changes (its file, a base file it builds on such as `delta-net-base.cpp`, or its declaration in `models.h`; `model_archs.py` maps them to arch names) and, when shared paths change, every arch the fork changes or tests in its code (`LLM_ARCH_...`), with the reason. Add the archs that **Where upstream changes fork code** brings in. Fill the Validation column of every row: a real GGUF, listed in **Models to validate**, or `test-llama-archs` only with the reason (no GGUF exists, too large for the machine, an encoder...), and only for an arch the test runs: it skips the archs of `arch_supported()` in `tests/test-llama-archs.cpp` (JAMBA on HIP only) and the draft models EAGLE3 and DFLASH, which need a real GGUF or the reason they are not validated. No arch may be left out.

## 3.4 Add the tables

```markdown
## Models concerned

<sync.sh models table, completed, Validation filled for every row>

## Models to validate

| Arch | Hugging Face repo | File |
|---|---|---|
| <arch> | <org>/<repo> | <path in the repo: file.gguf, or the -00001-of-0000N shard> |

## Checklist

- [ ] after every step: `sync.sh verify` and the GPU builds the step can break
- [ ] end of sync: HIP (gfx1151) and Vulkan builds, CPU build and `ctest -L main`
- [ ] Strix Halo: `validate.sh`, both backends `TARGET PASS` (results below)

## Requirements

- AI usage disclosure: <NO / ASSISTED / AGENT-AUTHORED, and how>
- What was NOT verified: <every unticked box above, and why>
```

Tick a box only for a check that really ran. `validate.sh` reads the **Models to validate** table, so keep its three columns. GitHub limits a PR body to 65536 characters, and the validation results come on top of the body (phase 5). If `body.md` is longer than about 50000 characters, post each step as a PR comment, and keep in the body a short `## Steps` section that lists the steps and says the details are in the comments: `validate.sh` refuses a body without `## Steps`.
