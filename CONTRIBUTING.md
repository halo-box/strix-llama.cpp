# Contributing to strix-llama.cpp

This is a community fork of [`llama.cpp`](https://github.com/ggml-org/llama.cpp) for **AMD Strix Halo**
(Ryzen AI Max / Max+ 300 series, `gfx1151`, RDNA 3.5 iGPU, unified memory). It is small, it is device-specific, and it
runs by its own rules - the sections below replace upstream's where they differ. The coding and naming guidelines
further down are upstream's and are kept as-is, because this fork stays mergeable with upstream.

# Where does my change go?

```
ggml-org/llama.cpp            upstream
  |
  +-- halo-box/llama.cpp      staging fork for changes intended to go upstream
        |
        +-- halo-box/strix-llama.cpp   <-- this repo
```

- **Strix Halo specific, or justified by measurements on Strix Halo** -> open the PR here.
- **A general llama.cpp improvement** -> open it against
  [halo-box/llama.cpp](https://github.com/halo-box/llama.cpp) instead, and follow that repo's (upstream's) rules,
  including its AI policy. Nothing is submitted upstream from this repo.

If you are unsure, open an issue here and ask. Landing something in the wrong repo costs everyone more than asking.

# Contributors

The project differentiates between 3 levels of contributors:

- Contributors: people who have contributed before (no special privileges)
- Collaborators (Triage): people with significant contributions, who may be responsible for some parts of the code, and are expected to maintain and review contributions for the code they own
- Maintainers: responsible for reviewing and merging PRs

Contributions from anyone who owns the hardware are welcome, and so are bug reports and benchmark numbers from people
who do not want to write code. A `llama-bench` run on a configuration nobody here has tested is a real contribution.

# AI Usage Policy

> [!IMPORTANT]
>
> AI-generated code is allowed, and so are **AI-authored pull requests**, including from autonomous agents.
> This is a deliberate difference from upstream `llama.cpp`, which bans automated submissions.
>
> Detailed rules for agents are in [AGENTS.md](AGENTS.md).

What is allowed here that is not allowed upstream:

- An agent may write the code, the commit messages, and the PR description.
- An agent may open the PR itself (`gh pr create`) and reply to review comments.
- An agent may run unattended, as long as it follows [AGENTS.md](AGENTS.md).

What is still required, of humans and agents alike:

1. **Disclose it.** Say in the PR that AI was used and how. Agent-authored commits carry an
   `Assisted-by: <assistant name>` trailer.
2. **Back the claims with measurements.** Performance and correctness claims about this hardware need numbers from
   this hardware - see [Preparing your PR](#preparing-your-pr).
3. **Someone owns the result.** If you submit it, you answer for it: bugs, review feedback, and follow-ups. An agent
   that opens a PR and disappears leaves the maintainers holding it, and such PRs may be closed.
4. **Do not duplicate.** Check for an existing PR or issue covering the same change first, and comment there instead
   of opening a second one.
5. **Do not merge your own work**, and do not push to `master`.

Undisclosed AI usage is the one thing that will get a contributor blocked here. Disclosure is cheap; a reviewer
discovering it later is not.

# Pull requests (for contributors & collaborators)

### Before you start

- Search existing issues and PRs first - duplicates will likely be closed.
- Confirm the change belongs in this repo and not in [halo-box/llama.cpp](https://github.com/halo-box/llama.cpp);
  see [Where does my change go?](#where-does-my-change-go).
- For anything large, or anything that introduces a new pattern or subsystem, open an issue first. Small, measured,
  device-specific changes can go straight to a PR.
- Check whether upstream has already fixed it. Carrying a patch we do not need is a permanent merge cost.

### Preparing your PR

- Test your changes:
  - Build and run on Strix Halo, on the backend you touched (Vulkan and/or ROCm)
  - Produce the numbers required by [Benchmarking requirements](#benchmarking-requirements) - this is the part
    reviewers look at first
  - [Running the CI locally](ci/README.md) is the fastest way to catch what our CI would catch
- Bug fixes should come with a reproducible report and, where practical, a regression test - but reuse the existing
  test infrastructure; do not add new files under `tests/` without a good reason.
- Keep hardware workarounds documented in place: what was measured, on what hardware and driver, and what would let
  the workaround be removed.
- Create separate PRs for each feature or fix; avoid combining unrelated changes.
- Prefer small, local diffs. This fork merges upstream `master` regularly, and a sprawling change conflicts forever.
- Consider allowing write access to your branch for faster reviews, as reviewers can push commits directly.

### After submitting your PR

- Expect requests for modifications. Long-term maintainability matters more here than elsewhere, because every line
  carried in this fork has to survive every upstream merge.
- If your PR becomes stale, rebase it on top of latest `master`.
- Consider adding yourself to [CODEOWNERS](CODEOWNERS) to indicate your availability for fixing related issues and reviewing related PRs.

# Benchmarking requirements

Almost everything in this fork exists because of a measurement, so measurements are the main thing a review here has
to trust. A PR that claims a change is faster, and does not show it on the hardware, is not reviewable and will be
asked for numbers before anything else.

### When numbers are required

Required for any change that touches a compute kernel, a dispatch or scheduling path, memory allocation, batching,
sampling, or speculative decoding - that is, anything that could move throughput, latency or memory use.

Not required for docs, comments, CI configuration, or build-system changes that cannot affect runtime. Say so in the
PR and delete the section from the template.

If you do not have access to Strix Halo hardware, you may still open the PR - mark it clearly as **unverified** and
say what you were unable to run. Do not report numbers from a different GPU as if they applied here.

### Report the environment

Paste this into the PR, filled in. Most disagreements about benchmark results turn out to be a difference in one of
these lines:

```
Device:     Ryzen AI Max+ 395 / <mini-PC or laptop model>
Memory:     128 GB LPDDR5X-8000
Power:      <sustained TDP / platform profile / power governor>
BIOS:       UMA split <n> GB
Kernel:     <uname -r>, amdgpu params: <gttsize / ttm.pages_limit, or none>
Backend:    Vulkan RADV, Mesa <version>   (or: ROCm <version>, HIP_LAUNCH_BLOCKING=<0|1>)
Build:      <cmake flags>
Baseline:   <merge-base commit sha>
Change:     <your branch commit sha>
Model:      <HF repo / file>, <quant>
```

Power and thermals matter more here than on a discrete GPU: the CPU and iGPU share a package power budget and a
memory controller, and the same silicon ships in chassis with very different sustained TDP. Numbers without a stated
power profile are not comparable between contributors.

### Run the baseline yourself

The baseline is a binary you built from the merge-base of your branch, with the same cmake flags, on the same machine,
in the same session as the "after" run. Not numbers from a previous day, another machine, a release build, or memory.
Interleave or alternate the two runs if the machine's thermal state drifts.

### Use llama-bench, and paste the table

```sh
# prompt processing and token generation, 5 repetitions (the default)
./build/bin/llama-bench -m <model.gguf> -p 512 -n 128 -r 5

# add context depth when the change can affect long-context behaviour
./build/bin/llama-bench -m <model.gguf> -p 512 -n 128 -d 0,4096,16384 -r 5
```

- Paste the raw table, including the standard-deviation column. Do not replace it with "about 18% faster".
- Raise `-r` if the spread is wide. Five repetitions is a floor, not a target.
- Vary the axis your change actually acts on. A change to batched mat-vec has to be shown across batch sizes
  (`-ub`), not at one convenient point; a flash-attention change has to be shown with `-fa` both ways.
- If before and after overlap inside their standard deviations, you have not measured a speedup yet.

`llama-bench` is a floor, not a ceiling. Changes to speculative decoding, the server, or anything whose benefit
depends on the content being generated cannot be shown with it - measure end to end against `llama-server` with a
described workload, and say what the workload was.

For per-op attribution on the Vulkan backend, `GGML_VK_PERF_LOGGER=1` gives per-node timings, which is the right
evidence for "this shader variant is slow at this shape" claims.

### Show that output did not change

A speedup that changes results is a bug, so speed numbers alone are not enough:

- Touched `ggml`: run `test-backend-ops` in the default `test` mode for the backend you changed, and narrow with
  `-o <OP>` while iterating.
  ```sh
  ./build/bin/test-backend-ops -b Vulkan0 -o MUL_MAT
  ```
- Touched anything that can alter generated tokens: report `llama-perplexity` before and after on the same file and
  model. State the value for both; a perplexity shift that you cannot explain blocks the PR.
- Say which of these you did **not** run. An honest gap is reviewable; a silent one is not.

# Pull requests (for maintainers)

- Squash-merge PRs
- Use the following format for the squashed commit title: `<module> : <commit title> (#<issue_number>)`. For example: `vulkan : fix mmvq selection on gfx1151 (#1234)`
- Let other maintainers merge their own PRs
- When merging a PR, make sure you have a good understanding of the changes
- Prefer changes that upstream could plausibly accept later, even though we do not submit them from here - it keeps
  merges cheap
- Wait for CI results before merging, including the self-hosted `gfx1151` job where it applies

Maintainers reserve the right to decline review or close pull requests, particularly under any of the following conditions:
- The change belongs in [halo-box/llama.cpp](https://github.com/halo-box/llama.cpp) or upstream instead.
- The pull request duplicates an existing one.
- The contributor fails to adhere to this contributing guide or the AI policy.
- Performance claims are not backed by measurements from the hardware.
- Nobody is available to own the change after it lands.

# Coding guidelines

- Avoid adding third-party dependencies, extra files, extra headers, etc.
- Always consider cross-compatibility with other operating systems and architectures
- Avoid fancy-looking modern STL constructs, use basic `for` loops, avoid templates, keep it simple
- Vertical alignment makes things more readable and easier to batch edit
- Clean-up any trailing whitespaces, use 4 spaces for indentation, brackets on the same line, `void * ptr`, `int & a`
- Use sized integer types such as `int32_t` in the public API, e.g. `size_t` may also be appropriate for allocation sizes or byte offsets
- Declare structs with `struct foo {}` instead of `typedef struct foo {} foo`
    - In C++ code omit optional `struct` and `enum` keyword whenever they are not necessary
    ```cpp
    // OK
    llama_context * ctx;
    const llama_rope_type rope_type;

    // not OK
    struct llama_context * ctx;
    const enum llama_rope_type rope_type;
    ```

    _(NOTE: this guideline is yet to be applied to the `llama.cpp` codebase. New code should follow this guideline.)_

- Try to follow the existing patterns in the code (indentation, spaces, etc.). In case of doubt use `clang-format` (from clang-tools v15+) to format the added code
- For anything not covered in the current guidelines, refer to the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
- Tensors store data in row-major order. We refer to dimension 0 as columns, 1 as rows, 2 as matrices
- Matrix multiplication is unconventional: [`C = ggml_mul_mat(ctx, A, B)`](https://github.com/ggml-org/llama.cpp/blob/880e352277fc017df4d5794f0c21c44e1eae2b84/ggml.h#L1058-L1064) means $C^T = A B^T \Leftrightarrow C = B A^T.$

![matmul](media/matmul.png)

# Naming guidelines

- Use `snake_case` for function, variable and type names
- Naming usually optimizes for longest common prefix (see https://github.com/ggml-org/ggml/pull/302#discussion_r1243240963)

    ```cpp
    // not OK
    int small_number;
    int big_number;

    // OK
    int number_small;
    int number_big;
    ```

- Enum values are always in upper case and prefixed with the enum name

    ```cpp
    enum llama_vocab_type {
        LLAMA_VOCAB_TYPE_NONE = 0,
        LLAMA_VOCAB_TYPE_SPM  = 1,
        LLAMA_VOCAB_TYPE_BPE  = 2,
        LLAMA_VOCAB_TYPE_WPM  = 3,
        LLAMA_VOCAB_TYPE_UGM  = 4,
        LLAMA_VOCAB_TYPE_RWKV = 5,
    };
    ```

- The general naming pattern is `<class>_<method>`, with `<method>` being `<action>_<noun>`

    ```cpp
    llama_model_init();           // class: "llama_model",         method: "init"
    llama_sampler_chain_remove(); // class: "llama_sampler_chain", method: "remove"
    llama_sampler_get_seed();     // class: "llama_sampler",       method: "get_seed"
    llama_set_embeddings();       // class: "llama_context",       method: "set_embeddings"
    llama_n_threads();            // class: "llama_context",       method: "n_threads"
    llama_adapter_lora_free();    // class: "llama_adapter_lora",  method: "free"
    ```

    - The `get` `<action>` can be omitted
    - The `<noun>` can be omitted if not necessary
    - The `_context` suffix of the `<class>` is optional. Use it to disambiguate symbols when needed
    - Use `init`/`free` for constructor/destructor `<action>`

- Use the `_t` suffix when a type is supposed to be opaque to the user - it's not relevant to them if it is a struct or anything else

    ```cpp
    typedef struct llama_context * llama_context_t;

    enum llama_pooling_type llama_pooling_type(const llama_context_t ctx);
    ```

    _(NOTE: this guideline is yet to be applied to the `llama.cpp` codebase. New code should follow this guideline)_

- C/C++ filenames are all lowercase with dashes. Headers use the `.h` extension. Source files use the `.c` or `.cpp` extension
- Python filenames are all lowercase with underscores

- _(TODO: abbreviations usage)_

# Preprocessor directives

- _(TODO: add guidelines with examples and apply them to the codebase)_

    ```cpp
    #ifdef FOO
    #endif // FOO
    ```

# Code maintenance

- Existing code should have designated collaborators and/or maintainers specified in the [CODEOWNERS](CODEOWNERS) file responsible for:
  - Reviewing and merging related PRs
  - Fixing related bugs
  - Providing developer guidance/support

- When adding or modifying a large piece of code:
  - If you are a collaborator, make sure to add yourself to [CODEOWNERS](CODEOWNERS) to indicate your availability for reviewing related PRs
  - If you are a contributor, find an existing collaborator who is willing to review and maintain your code long-term
  - Provide the necessary CI workflow (and hardware) to test your changes (see [ci/README.md](ci/README.md))

- New code should follow the guidelines (coding, naming, etc.) outlined in this document. Exceptions are allowed in isolated, backend-specific parts of the code that do not interface directly with the `ggml` interfaces.
  _(NOTE: for legacy reasons, existing code is not required to follow this guideline)_

- For changes in server, please make sure to refer to the [server development documentation](./tools/server/README-dev.md)

# Documentation

- Documentation is a community effort
- When you need to look into the source code to figure out how to use an API consider adding a short summary to the header file for future reference
- When you notice incorrect or outdated documentation, please update it

# Resources

- [AGENTS.md](AGENTS.md) - rules for AI coding agents working in this repo, including how to open a PR here
- [README.md](README.md) - what this fork is, how to build it, and the Strix Halo specific notes
- [Issues](https://github.com/halo-box/strix-llama.cpp/issues) and [PRs](https://github.com/halo-box/strix-llama.cpp/pulls) in this repo
- [halo-box/llama.cpp](https://github.com/halo-box/llama.cpp) - where upstream-bound changes go instead

Upstream's issues, PRs and discussions remain the best source of background on the codebase itself, and its Github
projects collect the most important of it:

https://github.com/ggml-org/llama.cpp/projects
