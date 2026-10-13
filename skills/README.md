# Skills

Reusable task workflows for agents, in the [Agent Skills](https://agentskills.io/specification) format, so any harness that supports it can use them (Claude Code, OpenCode, pi...). Each skill is a folder with:

- `SKILL.md`: the frontmatter (`name`, equal to the folder name, and `description`, which tells the agent when to use it), then the rules and the steps;
- `references/`: detailed instructions, read on demand when the agent reaches that step;
- `scripts/`: the scripts the steps run, from the repo root.

| Skill | Use it to |
|---|---|
| [add-new-model](add-new-model/SKILL.md) | add or port a new model architecture |
| [code-review](code-review/SKILL.md) | review a diff, branch or PR against the project conventions before a PR |
| [upstream-sync](upstream-sync/SKILL.md) | sync the fork with upstream ggml-org/llama.cpp: check, merge step by step, benchmark on a Strix Halo machine, open the PR |
| [upstream-sync-verify](upstream-sync-verify/SKILL.md) | verify a sync PR with the same checks and benchmarks, and post the results on it |

## Discovery

Each skill is linked from the folders the harnesses scan, so they list it to the agent and run it when a request matches its description, or on demand (`/<name>` in Claude Code):

| Folder | Read by |
|---|---|
| `.agents/skills/<name>` -> `skills/<name>` | OpenCode, pi, and other harnesses that follow the shared `.agents/skills/` convention |
| `.claude/skills/<name>` -> `skills/<name>` | Claude Code (OpenCode reads it too) |

Agents that read neither find this directory through `AGENTS.md`. The links need `git config core.symlinks true` (the default on Linux and macOS; on Windows, enable Developer Mode or clone from WSL), otherwise git checks them out as small text files and the harnesses do not find the skills.

A new skill goes in `skills/<name>/`, with a row in the table above and both links:

```sh
ln -s ../../skills/<name> .agents/skills/<name>
ln -s ../../skills/<name> .claude/skills/<name>
```

Keep skills portable: only the standard frontmatter fields (`name`, `description`, `license`, `compatibility`, `metadata`), paths relative to the repo root, and no feature of a single harness.
