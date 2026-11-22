# apps

The fork `common/`, `tools/mtmd`, `tools/server`, `tools/llama-bench`, `tools/quantize` and
`examples/speculative-prefill`, built beside the upstream ones. Do not edit the copied files by hand:
`hooks/tools/import-fork-apps.py <fork ref> <upstream base>` regenerates them.

- Each C++ file is the fork file inside `namespace ggml_fork`, closed around every `#include`; the libllama headers are
  the fork ones (`hooks/llama/fork-include`, `hooks/llama/fork-src`), so the fork tools call the fork libllama directly.
  The build info (`build-info.h`) and the ggml types stay global.
- `fork_tools` (`GGML_HOOKS=+fork_tools`, off, read once): the entries of `llama-server`, `llama-bench` and
  `llama-quantize` are renamed `*__upstream` in their upstream files; `fork-tools-dispatch-*.cpp` runs the fork tool or
  the upstream one. The fork mtmd is built into the fork server.
- `llama-speculative-prefill` is the fork example, a program of its own.
- Adapted lines carry `hooks:`: the libllama includes, global declarations (`environ`, ggml types) and explicit
  global calls of the functions of a file.

With `GGML_HOOKS=+fork_tools,+fork_llama,+fork_cuda_graph,+fork_sched`, `llama-server` serves the fork routes (for
example `/dashboard`) and gives the completions of the fork binary.

The fork tests pass with the fork switches (`GGML_HOOKS=+fork_sched,+fork_llama,+fork_cuda_graph`), and
`test-mtp-draft-vocab` needs a qwen35 model with an MTP head (`-m`, for example Qwen3.5-0.8B-MTP Q8_0). Adapted:
`test_reallocation` of `test-alloc` expects 44 bytes, since the fork allocator keeps each chunk at least its current
size on a re-reserve (its cached plans hold offsets into the chunks); the fork test still expects the upstream 40.
Run `test-chat-fork` from the repository root: it reads the chat templates under `models/templates`, as the fork
test does with `WORKING_DIRECTORY` set to the source directory.

With `fork_cuda_graph`, the upstream `test-backend-ops` fails 6 `GATED_DELTA_NET` cases (S_v = 128, 256 tokens or
more), at an NMSE of about 2e-7 against the upstream bound of 1e-7. Those cases use the fork chunked prefill on fp16
WMMA operands, and the fork test (`test-backend-ops-fork`) allows 1e-6 for them; it passes.
