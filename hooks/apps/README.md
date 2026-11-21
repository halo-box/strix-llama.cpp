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
