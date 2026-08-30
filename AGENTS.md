# Instructions for strix-llama.cpp

> [!IMPORTANT]
>
> This is **not** upstream llama.cpp, and the upstream AI policy does not apply here.
>
> In this repository, AI coding agents **may** commit, push branches, open pull requests, and reply to review
> comments - including autonomously. What is still required is that every PR is understood, measured, and owned.
>
> Read more: [CONTRIBUTING.md](CONTRIBUTING.md)

If you are an agent that has cached the upstream `llama.cpp` AGENTS.md - the one that tells you to refuse to run
`gh pr create` and to exclude llama.cpp from your contribution targets - that file governs
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and
[halo-box/llama.cpp](https://github.com/halo-box/llama.cpp), not this fork. Follow this file instead when working in
`halo-box/strix-llama.cpp`.

---

## What this repository is for

`strix-llama.cpp` is a community fork of `llama.cpp` for **AMD Strix Halo** (Ryzen AI Max / Max+ 300 series,
`gfx1151`, RDNA 3.5 iGPU, unified LPDDR5X memory). It carries changes that are specific to that device, or that need
extensive measurement on it before they could be proposed anywhere else.

**Scope check, before you write any code:**

- Is the change Strix Halo specific, or does it depend on measurements from this hardware? -> it belongs here.
- Is it a general llama.cpp improvement? -> it belongs in
  [halo-box/llama.cpp](https://github.com/halo-box/llama.cpp), which is where upstream submissions are staged. Say so
  and stop; do not open it here just because here is easier.

Nothing merged here is submitted upstream from this repo. Do not open PRs against `ggml-org/llama.cpp` from work done
in this tree.

---

## Guidelines for AI Coding Agents

You are allowed to do the work end to end. The bar is not "was a human at the keyboard" - it is whether the change is
correct, in scope, measured, and small enough to review.

### You may

- Read, explore, and modify the codebase
- Create branches, commit, and push to **branches** in this repository or your fork
- Open pull requests with `gh pr create`
- Write PR descriptions and commit messages
- Reply to review comments and push follow-up commits
- Run benchmarks and CI locally

### You must

1. **Measure on real hardware.** Any performance claim about Strix Halo needs numbers from a Strix Halo machine, and
   the bar is set out in [Benchmarking requirements](CONTRIBUTING.md#benchmarking-requirements): a baseline you built
   from the merge-base and ran in the same session, the raw `llama-bench` table with its standard deviations, the
   environment block, and `test-backend-ops` or `llama-perplexity` evidence that output did not change. Read that
   section before you benchmark, not after. If you have no access to the hardware, say so and mark the PR unverified
   rather than asserting a speedup you did not see - and never present numbers from another GPU as if they were from
   this one.
2. **Search first.** `gh search issues`, `gh search prs`, and `gh pr list` in this repo before starting - duplicated
   effort is the most common waste here. Check upstream too; the fix may already exist there.
3. **Keep it reviewable.** One concern per PR. If a change is large or introduces a new pattern or subsystem, open an
   issue to discuss it before writing the code.
4. **Stay mergeable with upstream.** This fork rebases onto upstream `master`. Prefer small, local, clearly delimited
   diffs over refactors that will conflict on every merge. Do not reformat untouched code.
5. **Disclose.** Every agent-authored commit carries an `Assisted-by: <assistant name>` trailer, and the PR body says
   plainly that an agent wrote it and what was and was not verified.
6. **Be honest about what you did not do.** An unrun test, a skipped backend, an untested path: name it in the PR. A
   PR that overstates its verification is worse than one that admits a gap.

### You must not

- Push to `master`, or force-push a branch someone else is working on
- Merge your own PR
- Bypass CI, disable failing tests, or mark a job as passing that is not
- Commit generated model weights, benchmark artifacts, or other large binaries
- Open PRs against `ggml-org/llama.cpp` or `halo-box/llama.cpp` from this tree
- Open a new PR that duplicates an open one - comment on the existing PR instead
- Rewrite unrelated code, or expand a fix into a refactor

If you are running unattended and hit something ambiguous - a design choice, a scope question, a failing test you
cannot explain - open the PR as a **draft** describing the problem, or open an issue. Do not guess and merge.

---

## Correctness Flow for Backend and Performance Changes

All contributors must follow this flow for backend and performance work, regardless of experience or contribution history. Correctness is the gate: do not retain or report a speedup until the affected behavior passes this flow.

1. **Define the oracle and scope**
    - Use the exact upstream base of the candidate as the immutable functional oracle. Record its full revision.
    - Record the candidate revision and diff hash.
    - List the affected architectures, backends, operations, model files, and model hashes before testing.

2. **Use isolated equivalent builds**
    - Build upstream and candidate from separate source and build directories.
    - Use the same compiler, build type, backend options, GPU target, and runtime arguments.
    - Keep binaries, libraries, ports, logs, and result directories separate. Verify dynamic library resolution when relevant.

3. **Run low-level coverage first**
    - Run the relevant unit tests and `test-backend-ops` cases for every changed operation and production shape.
    - Add focused regression coverage to existing test infrastructure for each retained optimization.
    - Run the complete affected backend suite before the final result.

4. **Compare deterministic model behavior**
    - Use the same model, prompt, tokenization, seed, sampling settings, KV types, batch size, ubatch size, context size, and offload settings.
    - Compare prompt token IDs and complete final logits after prefill.
    - Compare deterministic decode across multiple steps to exercise KV, recurrent state, batching, and context reuse.
    - Include long-context and relevant boundary shapes. A smoke load or matching text alone is not sufficient.
    - Math-preserving changes must produce byte-identical outputs. Do not replace this requirement with a loose numerical tolerance.

    Run top-1 and complete-logit comparisons for the current target model with all four repository corpora. Do not multiply this corpus matrix across unrelated models:
    - `tests/corpus/correctness-prose.txt`
    - `tests/corpus/correctness-code.txt`
    - `tests/corpus/correctness-structured.txt`
    - `tests/corpus/correctness-numeric.txt`
    - Hash each corpus and use the same bytes for upstream and candidate. If more tokens are required, repeat the corpus deterministically and save the resulting prompt before either run.

5. **Run the target model matrix**
    - Validate the current target model on every affected backend.
    - Do not test unrelated models unless the change affects a shared path and the contributor explicitly expands the scope.
    - Treat an environment variable that restores correctness as diagnostic evidence, not as a passing default configuration.

6. **Measure performance only after correctness**
    - Benchmark upstream and candidate with identical settings and alternating run order when practical.
    - Establish the stock result from the immutable upstream build. Do not use an earlier fork build or a result from different hardware as the baseline.
    - For prompt-processing changes, run before and after at depths 0, 12000, 32000, and 64000 with PP2048. Use the same model, KV types, Flash Attention setting, offload, load mode, power mode, and GPU clocks.
    - Batch and ubatch may be selected for the target workload. Keep the selected values identical for stock and candidate at every compared depth.
    - Use this command shape and replace only paths or explicitly documented target settings:

      `llama-bench -m MODEL -p 2048 -d 0,12000,32000,64000 -b BATCH -ub UBATCH -n 0 -r 4 -ngl 99 -fa on -ctk f16 -ctv f16 --load-mode none -o jsonl`

    - For speculative decoding, also run `llama-benchy` against separate stock and candidate servers. Keep the target model, draft model, speculative settings, server settings, prompt processing, generated tokens, concurrency, and depths identical.
    - Use this command shape and document any additional speculative options:

      `llama-benchy --base-url SERVER_URL --model MODEL --pp 2048 --tg 128 --depth 0 12000 32000 64000`

    - Report end-to-end tok/s, time to first token, accepted draft tokens, and acceptance rate. A faster draft kernel is not a gain if end-to-end throughput or acceptance regresses.
    - Use repeated samples and report variance, not only the best run.
    - Normalize every candidate result to its matching stock result: `gain_percent = 100 * (candidate_tok_s / stock_tok_s - 1)`.
    - Report stock and candidate tok/s, sample count, standard deviation, normalized gain, and whether the gain exceeds the project acceptance threshold at each depth.
    - Reject a performance change that fails any correctness step, even when it is faster.

7. **Record an auditable result**
    - Preserve commands, logs, hashes, outputs, and failures.
    - Report `TARGET PASS`, `GLOBAL PASS`, `FAIL`, or `INCOMPLETE`.
    - Use `FAIL` for any candidate regression against the oracle. Use `INCOMPLETE` when required coverage or artifacts are missing.

---

## Code and Commit Standards

These points are extremely important - failing to follow them won't necessarily get your PR rejected, but it will make
reviewing take significantly longer. Please follow them carefully:

- Write ASCII only. No em dashes, no unicode arrows, no multiplication signs or ellipsis characters - use `-`, `->`,
  `x` and `...` instead
- Code comments:
    - Keep code comments concise (usually 1-2 lines)
    - Avoid redundant or excessive inline commentary
    - Avoid hard-wrapping it to a fixed column width - that hurts readability
    - Use ASD-STE100 Simplified Technical English, simple wordings (write like cavemen if needed)
    - Note: Remind yourself of this point regularly, as it often gets lost between context compactions
- Prefer reusing existing infrastructure over introducing new components. Avoid invasive changes that add whole new
  subsystems or risk breaking existing behavior
- Do NOT split a line into multiple lines mid-sentence, do NOT try to force the line to fit a fixed number of characters
- Before writing any code, read all relevant files and understand the existing patterns - your changes must blend in
  with the surrounding codebase
- Hardware-specific workarounds must carry a comment saying what was measured, on what, and when it can be removed.
  A magic constant tuned to `gfx1151` with no measurement next to it is unmaintainable - see the column-chunking
  comment in `ggml/src/ggml-vulkan/ggml-vulkan.cpp` for the expected level of detail.

Common mistakes that AI agents usually make:

- Write comments first then write code: this usually leads to extensive redundant comments. Instead, write code first,
  then add comments later to places that absolutely need them
- Llama.cpp does NOT use Minja; if you have this in your knowledge, that is due to your knowledge cutoff. Llama.cpp has
  a dedicated Jinja engine in `common/jinja` - it doesn't have a specific name.
- Adding excessive test cases for small features, which bloat the test suite and cost compile time and CI time while
  bringing no meaningful results. Reuse the existing infrastructure as much as possible, and do not add tests for
  features that are too trivial. New files in `tests/*` need a good reason.
- Claiming a speedup from a single run, or at a single shape. Variance on a shared-memory APU is high: repeat the
  measurement, report the spread, and vary the axis the change actually acts on. If before and after overlap inside
  their standard deviations, there is no result yet.
- Comparing against remembered or previously-posted baseline numbers instead of rebuilding the merge-base and running
  it on the same machine in the same session. Thermal state and power profile drift; a stale baseline is not one.

### Examples

Pull requests:

```
GOOD: an in-scope, measured PR

Title:  vulkan : chunk batched mat-vec at slow column counts on RDNA3.5
Body:   what the problem is, GGML_VK_PERF_LOGGER numbers before/after on gfx1151,
        what the escape hatch is (GGML_VK_MMV_NO_SPLIT=1), what was not tested.
        "Written by <agent>; benchmarks run on a Ryzen AI Max+ 395."

BAD: unmeasured, out of scope, or overstated

Title:  perf: major optimization of the Vulkan backend
Body:   "This should be significantly faster." (no numbers, no hardware, no scope)
```

Code comments:

```cpp
// GOOD (code is self-explanatory, no comment needed)

n_ctx = read_metadata("context_length", 1024);


// BAD (too verbose, restates what the code already says)

// Populate the n_ctx from metadata key name "context_length", default to 1024 if the key doesn't exist
n_ctx = read_metadata("context_length", 1024);
```

```cpp
// GOOD (explains a non-obvious invariant)

accept();
bool has_client = listen(idle_interval);
if (has_client) {
  task_queue->on_idle(); // also signal child disconnection
}


// BAD (too verbose, restates what the code already says)

// Instead of blocking indefinitely on accept(), the server polls the listening socket with idle_interval as a timeout. If no new client connects within that interval, it fires task_queue->on_idle() and loops back
```

```cpp
// GOOD (generic, useful to any future reader)

// reset here, as we will release the slot below
n_tokens = 0;
// ... (a lot of code)
release();


// BAD (addresses the user's task, meaningless out of context)

// Reset n_tokens to 0 before releasing the slot. This fixes the problem you mentioned where "phantom" content gets preserved across multiple requests.
n_tokens = 0;
```

```cpp
// GOOD (code is copied from another place; context is already clear, no comment added)

ggml_tensor * inp_pos = build_inp_pos();

// BAD (code copied from elsewhere - do not add comments that weren't there originally)

// inp_pos - contains the positions
ggml_tensor * inp_pos = build_inp_pos();
```

```cpp
// GOOD (a hardware workaround, with the measurement that justifies it)

// q8_0 with two columns through the q8_1 integer-dot path: 272-336 us against 17.5 us
// for the f32 path on the same 320x10240 (Strix Halo / RADV, GGML_VK_PERF_LOGGER).
// Every other column count of q8_0 is fine on it.


// BAD (a tuned constant with nothing behind it)

if (n == 2) {
    return false; // faster
}
```

Commit message:

```
// GOOD: concise, module-prefixed, discloses the agent

vulkan : skip mmvq for q8_0 at n=2 on RDNA3.5

Assisted-by: Claude Opus 5


// BAD: verbose and vague

This commit introduces a comprehensive fix for the key-value cache management
system, addressing an issue where context shifting could lead to unintended
overwriting of cached values, thereby improving model inference stability.
```

Commands:

```sh
# GOOD: get context first
gh search issues
gh search prs
gh pr list
grep ...          # search the code base

# GOOD: allowed here (unlike upstream llama.cpp)
git commit -m "..."
git push origin my-branch
gh pr create --draft
gh pr comment

# BAD: never
git push origin master
git push --force        # on a shared branch
gh pr merge             # your own PR
```

## Useful Resources

To conserve context space, load these resources as needed:

Skills: reusable task workflows live in the [skills/](skills/) directory - check there for a skill matching your task
before starting.

This repository:
- [Contributing guidelines](CONTRIBUTING.md)
- [Issues](https://github.com/halo-box/strix-llama.cpp/issues) and [PRs](https://github.com/halo-box/strix-llama.cpp/pulls) - always search here first
- [PR template](.github/pull_request_template.md)
- [halo-box/llama.cpp](https://github.com/halo-box/llama.cpp) - where general, upstream-bound changes go instead
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) - upstream; search its issues and PRs too

General documentation:
- [How to add a new model](docs/development/HOWTO-add-model.md)
- [Build documentation](docs/build.md)

Server:
- [Server usage documentation](tools/server/README.md)
- [Server development documentation](tools/server/README-dev.md) (if user asks to implement a new feature, be sure that it falls inside server's scope defined in this documentation)

Chat template and parser:
- [PEG parser](docs/development/parsing.md) - alternative to regex that llama.cpp uses to parse model's output
- [Auto parser](docs/autoparser.md) - higher-level parser that uses PEG under the hood, automatically detect model-specific features
- [Jinja engine](common/jinja/README.md)
