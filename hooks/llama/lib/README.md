# lib

The fork `src/` (libllama), built into the upstream libllama. `lib-include` is the fork `include/`. Do not edit the
copied files by hand: `hooks/tools/import-llama.py <fork ref> <upstream base>` regenerates them and the
dispatcher.

- Each file is the fork file inside `namespace ggml_fork`, closed around every `#include`. The ggml types it names are
  the global ones (forward declarations at the top of each file), the llama types are the fork ones. The headers of
  `lib-include` have C++ linkage, so the fork API is a separate set of symbols.
- `llama` (`GGML_HOOKS=+llama`, off, read once): `hooks/llama/src/llama-dispatch.cpp` (generated) defines
  every function of `llama.h`; it calls the fork one, or the upstream one renamed `*__upstream` in its file (or
  `*__hooks` for the functions that the libllama hooks define). The argument types have the same layout in both trees
  except `llama_model_params` and `llama_context_params`, converted by field; the fork-only fields keep their defaults.
  `llama-extra-api.h` declares the fork-only functions (`llama_set_eval_callback`), which do nothing without the switch.
- The `llama-ext.h` functions that common and the tools use go through `llama-ext-dispatch.cpp` and
  `ext-bridge.cpp`.
- Adapted lines carry `hooks:`: the EXL3 op checks (`ggml_fork_op()`, `hooks.h`) and the explicit global calls of
  functions that are in the namespace here.

With `GGML_HOOKS=+llama,+cuda_graph,+sched`, gpt-oss-20b gives the perplexity and speed of the fork
binary on gfx1151.

Fixes of the fork code, applied by the import tool and marked `hooks:`:
- `build_rs` in place, with a meta device (multi-device split) only: the fork views the recurrent states through an
  empty copy when every state of the ubatch is already in place, which the meta backend cannot split (the fork aborts
  in `test-llama-archs`). The same rows are viewed from the cache; every other graph is the fork one.
- the incremental QSA cache reads the registry name of each layer device; a meta device has none.
