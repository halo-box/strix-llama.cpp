# 4. Verify with the benchmarks, on a Strix Halo machine

Part of the `upstream-sync` skill: read [SKILL.md](../SKILL.md) first for the rules.

## 4.1 Prepare the machine

- A Strix Halo (gfx1151) machine with ROCm 10 in `/opt/rocm` (or set `ROCM_PATH`), for Vulkan the Vulkan headers and loader and `glslc` (package `glslc` or `shaderc` depending on the distribution, or the Vulkan SDK; the SPIR-V headers are downloaded by the script to `DEPS_DIR`), `gh` logged in, lingering on (`loginctl enable-linger`) so a job survives the session.
- A clean checkout of the repo with the sync branch, not a tree with uncommitted work.
- About 40 GB of free disk, plus the models. They are downloaded to `MODELS_DIR` (default `~/.cache/strix-llama-sync/models/<org>_<repo>/<path>`); link files you already have there, with the same path as in the Hugging Face repo, to skip the download.
- The GPU free: no `llama-*` process (the script checks it). Nothing else on the GPU while it runs, or the numbers are wrong. One run at a time per `MODELS_DIR`: a second one stops at once (lock file `MODELS_DIR/.validate.lock`).
- **Security**: the script builds and runs the candidate's code with your environment, the `gh` login included. With `--pr`, that is the PR head, possibly from a contributor's fork: run it only on a PR whose diff you have reviewed and trust, or in a container without your credentials. The repo and file names of the models table are checked (plain names, no `..`, the file stays under `MODELS_DIR`) before any download.

## 4.2 Run it

Run it as a job that survives the session, and tell the user how to follow and stop it. `systemd-run` starts the job in a clean environment: pass every variable it needs with `-E NAME` (the value is taken from the current shell; an unset one gives the default). `--collect` removes the unit when it ends, so the same name can be used again.

```sh
systemd-run --user --collect --unit=sync-validate --working-directory="$PWD" \
  -E PATH -E HOME -E ROCM_PATH -E MODELS_DIR -E DEPS_DIR -E BUILD_CACHE \
  -E QUANTS -E MOE_QUANTS -E KV_TYPES -E PP_SIZES -E DEPTHS -E RUN_TIMEOUT -E OPS_JOBS \
  bash -c 'skills/upstream-sync/scripts/validate.sh --cand <sync branch> --body body.md > ~/sync-validate.log 2>&1'
tail -f ~/sync-validate.log                    # follow
systemctl --user is-active sync-validate       # "active" while it runs
systemctl --user stop sync-validate            # stop
```

Without systemd user services (a container, a distribution without them), start the same command with `setsid nohup bash -c '...' > ~/sync-validate.log 2>&1 &`, and stop it with `kill -- -<pid>`.

It has ended when the unit is no longer active and the log ends with the summary. Then read the result (4.4) and go on to phase 5.

**Duration.** On a Strix Halo the four builds take about 15 minutes, less with a kept baseline build. With the defaults, `llama-bench` alone runs, per model and per backend, four processes (base, cand, base, cand) for pp and tg together; each fills the context once per depth (so 4 fills of 32000 tokens and 4 of 128000) and times every test 4 times: count hours per large model. To shorten a run, lower the settings and say so in the PR (the summary prints the scope it ran), for example `DEPTHS=0,32000 PP_SIZES=512 QUANTS="BF16 Q8_0 Q4_K_M Q4_0"`.

| Option or variable | Default | Use |
|---|---|---|
| `--cand <rev>` | the PR head with `--pr` | the sync branch |
| `--body <file>` / `--pr <number>` | | where the models table is read (before / after the PR exists) |
| `--base <rev>` | merge base with `origin/master`, or with the PR's base branch | the fork before the sync |
| `--model <arch>=<gguf>` | the table | a local model instead of the table; repeat it |
| `--small <gguf>` | downloaded | the small model, BF16 or F16 |
| `--small-moe <gguf>` | downloaded (Granite 3.1 1B-A400M, F16) | the small MoE model, BF16 or F16 |
| `--backends` | `hip,vk` | `cpu` only to test the script |
| `--out <dir>` | `../sync-validate/<sha>` | the output directory: new, empty, or one of an earlier run, even an older version's or a stopped one (it keeps its worktrees and builds; its results are deleted first). A directory holding anything validate.sh does not write is refused |
| `ROCM_PATH` | `/opt/rocm` | the ROCm used for HIP |
| `MODELS_DIR` | `~/.cache/strix-llama-sync/models` | the model cache |
| `DEPS_DIR` | `~/.cache/strix-llama-sync/deps` | build dependencies the script downloads (SPIR-V headers, pinned) |
| `BUILD_CACHE` | `base-builds` next to the output directory | one baseline build per commit, configuration and toolchain, with its `test-backend-ops` and `test-llama-archs` results, the importance matrix of the small Qwen (so its quants come out the same) and its KLD reference runs (per model content, text and options) per GPU driver: a rerun, or the next sync from the same `master`, builds and tests only the candidate. The `BUILD_CACHE_KEEP` (6) most recently used are kept, read-only; delete it any time, and after a run on code you do not trust (the candidate can write wherever you can) |
| `MOE_QUANTS` | `MXFP4_MOE` | the quant types of the small MoE (the others are tested on the small Qwen); `none` skips it, download included |
| `KV_TYPES` | `q8_0 q4_0` (`none` to skip) | the KV cache types tested on the small Qwen |
| `QUANTS`, `PP_SIZES`, `DEPTHS` (a sync checks 0, 32000 and 128000; the performance flow of `AGENTS.md` for a kernel PR keeps its own depths) | one type per ggml tensor type (25); 512,2048; 0,32000,128000 | the scope |
| `RUN_TIMEOUT` | `21600` (6 h) | a test or benchmark run that takes longer is killed and counts as a crash: a GPU hang ends the run instead of blocking it |
| `OPS_JOBS` | `4` | `test-backend-ops -j`: parallel workers per process, each with its own device and CPU backend |

`validate.sh --help` prints the same. A model download that fails stops the run at once. Downloads are anonymous, the script never uses a Hugging Face token (it unsets `HF_TOKEN`): for a gated model, download it yourself into `MODELS_DIR`.

## 4.3 What it does

1. Reads `body.md` and stops if it lacks Overview, Steps, Where upstream changes fork code, Models concerned or Models to validate, or if a step is sensitive and no model is listed. It prints the step table and warns when the sync has edits outside the conflicts.
2. Builds two trees in worktrees: the **baseline** (the fork before the sync, the merge base of the branch with `origin/master`) and the **candidate** (the sync branch), each for HIP (ROCm 10) and Vulkan. The baseline build comes from `BUILD_CACHE` when the same commit was built before with the same configuration and compilers; ccache, when installed, speeds up the candidate build.
3. For each backend, baseline against candidate:
   - `test-backend-ops`: every op, GPU against CPU, base and candidate at the same time (it checks results only), each with `OPS_JOBS` parallel workers (`-j`, default 4). On a Strix Halo, 2 x `-j 4` runs both sides of HIP in about 27 minutes, against 44 for `-j 1` side by side or `-j 16` one after the other: a few very large MUL_MAT_ID cases make a tail that leaves the CPU idle, and the two tails overlap. Its stderr goes to `ops-*.err`, as the logs of the other workers would cut the result lines. A failure of the candidate that the baseline does not have runs again alone (`-j 1 -o <test case>`) three times (`ops-rerun.txt`): some kernels fail now and then beside other work (a ROCmFP4 MUL_MAT_VEC_FUSION gave NaN in 2 of 4 parallel runs, never alone), so one that passes alone 3 times is INCOMPLETE, not FAIL. Skipped on a backend when the sync changes no ggml code that backend runs: a change under `ggml/src/ggml-<backend>/` concerns that backend only (`ggml-cuda` and `ggml-hip`: HIP), any other change under `ggml/` (core, `ggml-cpu`, which is the reference), to `tests/test-backend-ops.cpp`, or to the CMake files that set ggml's options or build the test (`CMakeLists.txt`, `cmake/`, `tests/CMakeLists.txt`) both. Its result cannot change otherwise; the summary says it was skipped and why;
   - `test-llama-archs`: every arch the test supports as a tiny random model, GPU against CPU (it skips a fixed list in `tests/test-llama-archs.cpp`: `arch_supported()`, and the draft models EAGLE3 and DFLASH);
   - **Qwen3.5-0.8B-Base** (`ggml-org/Qwen3.5-0.8B-Base-GGUF`, BF16) quantized to every type of `QUANTS` (by default one type per ggml tensor type that `llama-quantize` makes, checked with `--dry-run`: BF16, F16, the Q and K types from Q1_0 to Q8_0, the IQ types from IQ1_S to IQ4_XS (IQ2_M for iq2_s: IQ2_S makes iq2_xs tensors); not the variants that only mix these per tensor, the ternary TQ types, which only BitNet models use, the fork's ROCmFP4 and ROCmFPx types, nor NVFP4, an NVIDIA format), with one importance matrix computed by the baseline on the test texts and given to both quantizers (IQ1 and IQ2 need one; the layers it never runs, such as an MTP head, are put in Q8_0 on both sides, as the very low-bit types refuse a tensor without one): KLD, pp512 / tg128, and whether the quantizer output changed (if so, both files against the source model). The idea comes from upstream `ci/run.sh`, which runs perplexity on every quant of a small Qwen. **Granite 3.1 1B-A400M** (`bartowski/granite-3.1-1b-a400m-instruct-GGUF`, F16), a small MoE, quantized to every type of `MOE_QUANTS` with the same checks: MXFP4_MOE only puts expert tensors in MXFP4, and the dense Qwen has none. Both small-model quant matrices run only when the sync changes ggml code the backend runs (the `test-backend-ops` rule above), the llama library (`src/`, `include/`), `common/`, or the tools that make and measure them (`tools/quantize/`, `tools/imatrix/`, `tools/perplexity/`, `tools/llama-bench/`): otherwise their results cannot change, and the summary says they were skipped and why. The small Qwen also runs with a **quantized KV cache**, K and V of each type of `KV_TYPES` (default q8_0 and q4_0, the two in real use; q4_1, iq4_nl, q5_0, q5_1 and bf16 with `KV_TYPES`): KLD at `-c 2048` and `llama-bench` at depth 8192, so the flash attention paths that read a quantized cache run on a real model. Plus `test-save-load-state`;
   - **each model of the table**: KLD on the four corpora of `tests/corpus/` (taken from the baseline: the candidate does not choose its own test texts) at `-c 2048`, and on all four together at `-c 16384`; `llama-bench` at pp 512 and 2048 with `-b 4096 -ub 4096`, and tg128, at depths 0, 32000 and 128000, two alternating rounds per side; the server: a tool call, a greedy decode compared with the baseline, the MTP acceptance;
   - the **peak RAM** of every run and the **peak GPU memory** (amdgpu VRAM + GTT), sampled every 0.1 s (a peak shorter than that can still be missed). The llama.cpp breakdown per device (model, context, compute) is added only when the tool prints it, which `llama-perplexity` and `llama-server` do not by default.
4. Writes `summary.md` in its output directory (`../sync-validate/<sha>`), with one status per backend.

## 4.4 Read the result

| Check | Passes when |
|---|---|
| builds | both trees build |
| `test-backend-ops`, `test-llama-archs` | the candidate runs to the end (a crash or a timeout is a FAIL; when the baseline running beside it stopped, the baseline runs again alone: if it finishes, the candidate's stop is a FAIL, if not it is INCOMPLETE; when only the candidate stopped beside the baseline, it runs again alone), and has no failure that the baseline does not have. Known failures are compared without their error value only (the op and its parameters stay), so a moving error is not a new failure and a known failure never hides a new one. `test-backend-ops` must have run tests on the GPU device (a skipped device is INCOMPLETE, or a FAIL when the baseline ran it), and fewer backends passed than on the baseline is a FAIL. Fewer tests run on the GPU than on the baseline is INCOMPLETE: ops the device no longer supports are skipped, not failed, and fall back to the CPU. `test-llama-archs` with no test executed is INCOMPLETE, and so is a row (arch, device, dense or MoE) the baseline ran and the candidate skips or no longer has. |
| `test-save-load-state` | it passes on the candidate, or fails on the baseline too (then INCOMPLETE) |
| quantizer | `llama-quantize` of the candidate works for every type (it is a FAIL otherwise; the KLD of the table uses the file of the baseline quantizer). When its file differs from the baseline's, both are measured against the source model with the baseline build, on the first backend: a KLD more than 5 % above the baseline file's (or a worse PPL ratio by more than 0.01) is INCOMPLETE, more than 50 % above, or a file that does not run, a FAIL (a file only the candidate build can read is INCOMPLETE: a format change to explain). Every type of `QUANTS` has a result, a missing one is INCOMPLETE |
| KLD, each run | the candidate did not crash or time out (a FAIL otherwise), every value is a number (`nan` or `inf` is a FAIL), `Same top p` >= the reference minus 0.5 points, `Mean KLD` <= 2 x the reference + 0.0001 (above is a FAIL), `PPL(Q)/PPL(base)` within 1 +/- 0.01 when the reference has it there (on a model this bad, such as a 1-bit 0.8B with a PPL near 1e6, the stored baseline logits no longer give its PPL and the base against itself is far from 1: KLD and top p judge alone). The reference is the baseline against itself with a smaller ubatch: other kernel paths, both correct. A `Mean KLD` between the reference + 0.0001 and that limit is INCOMPLETE: the sync changed the logits more than another correct kernel path does, so the PR must say which upstream change does it. The long text runs at the largest `-c` (up to 16384) that its token count allows with the model's vocab |
| speed | every expected test ran on the candidate, and none is more than 3 % slower than the baseline (A/A runs of the same code move up to about 2 %). A slowdown above 3 % is a FAIL when it is also above the summed standard deviations of both sides, and INCOMPLETE when it is within them: a noisy test cannot tell it from noise, so rerun it (`--cand` alone, or `llama-bench` by hand) or explain it. A test missing on the candidate only is a FAIL (it crashed or stopped there), and so is any round of the candidate that crashed or timed out, even when the other round gave numbers. When a baseline round crashed too, both are INCOMPLETE: a GPU reset (on Vulkan at 128k, the amdgpu compute ring timed out on both sides) cannot be told from the candidate. A crash whose log shows the GPU lost (`vk::DeviceLostError`, `hipErrorLaunchFailure`, `HSA_STATUS_ERROR_*`, `Memory access fault by GPU`) says so, and when the kernel log is readable (`journalctl -k` or `dmesg`, often not without root) the summary gives the number of amdgpu ring timeouts during the run |
| server | a valid tool call (no answer from the candidate while the baseline answered is a FAIL; broken on both sides is INCOMPLETE); the same greedy decode, or a difference explained by an upstream change; MTP acceptance within 5 points. MTP is not applicable when neither side drafts (no MTP head), and a FAIL when the baseline drafts and the candidate does not |
| real models | at least one model of the table ran: the small model alone gives INCOMPLETE |
| memory | no rise above 5 % and 256 MiB, or the rise is explained. `test-backend-ops` is shown without a status: its peak changes from run to run with its parallel workers, and its base and candidate runs share the GPU |

The status of a backend is `TARGET PASS` when every check passes, `FAIL` when one fails, `INCOMPLETE` when one did not run or needs an explanation. A changed quantizer output is not a failure, but say so in the PR.

If a check fails, find the step that causes it: rerun with `--cand <merge commit>` for the steps in between. It is either an upstream change or a wrong resolution (fix it, then rerun). For an upstream change, look first for a later revert or fix in `upstream/master` (`git log <commit>..upstream/master -- <files of the commit>`): a sync stops at an arbitrary point, and upstream often reverts a regression a few days later. If there is one, merge steps up to it, or say in the PR that the regression is known and fixed upstream. Otherwise explain it, or report it upstream with the gfx1151 numbers.
