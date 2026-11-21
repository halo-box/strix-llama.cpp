# hooks

Fork code that changes llama.cpp without editing upstream files. The decision is in `adr/0001-fork-extension-architecture.md`.

```
cmake -B build -DCMAKE_PROJECT_INCLUDE=$PWD/hooks/hooks.cmake ...   # with the hooks
cmake -B build ...                                                  # pure upstream, same tree
```

## How hooks connect to upstream

| Mechanism | Upstream lines | Used for |
|---|---|---|
| CMake: `CMAKE_PROJECT_INCLUDE` + `cmake_language(DEFER)` | 0 | sources, include dirs, link options of upstream targets |
| Run-time wrapping of the backend interfaces | 0 | kernels, fusions, graph passes of the backends, buffer types, device memory |
| Link-time wrapping (`-Wl,--wrap`) | 0 | activation of the backend wrapper, libllama callbacks, sampler creation, server request fields and routes |
| Private member access (`llama-hooks-access.h`) | 0 | reading upstream state without a `friend` line |
| Hook points (`HOOK(name)`, listed in `HOOKS.txt`) | 61 in 19 files | ggml types and scheduler, KV cache events and other points inside libllama, CLI arguments, quantize types, server loop |

Link-time wrapping needs a GNU compatible linker (ld, gold, lld, mold). On MSVC and macOS the hooks are disabled.

## Layout

```
hooks.cmake                 entry point (CMAKE_PROJECT_INCLUDE)
HOOKS.txt                   manifest of the hook points
cmake/hooks-link.cmake      hooks_link (registrar, static builds), hooks_wrap (link-time wrapping)
ggml/                       ggml-base registry, backend wrapper, CPU/CUDA/Vulkan drivers, hook types
  kernels/<backend>/        kernel files, globbed
  fork-cuda/                the fork ggml-cuda tree, namespace ggml_fork, built into ggml-hip
  fork-vulkan/              the fork ggml-vulkan tree and shaders, namespace ggml_fork, built into ggml-vulkan
llama/fork-src, fork-include  the fork libllama (fork_llama)
apps/                       the fork common, tools, example and tests (fork_tools)
python/                     the fork converter
fork-files/                 the fork version of the files that are not built here
  src/fork-sched/           the fork scheduler, graph allocator and sanitizer (fork_sched)
llama/                      libllama hooks (models/, passes/ globbed), common and server wiring
common/, server/            common and server hooks (args/, handlers/ globbed)
tests/                      test-ggml-hooks, test-llama-hooks
tools/                      check-hooks.sh, check-wraps.sh
```

## ggml: kernels and fusions

```cpp
#include "ggml-hooks-cuda.h"

static bool my_supports(const ggml_hook_dev * dev, const ggml_tensor * t) { return t->type == GGML_TYPE_F32; }
static bool my_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) { /* launch on ctx.stream() */ return true; }

GGML_HOOK_REGISTER_KERNEL(ggml_hook_op_kernel("my_rms_norm", GGML_HOOK_BACKEND_CUDA, GGML_OP_RMS_NORM, 0, my_supports, ggml_hook_fn(my_rms_norm)));
```

- Op kernels claim a node in `supports`. Fusions (`ggml_hook_fusion_kernel`) return the number of nodes they compute
  from a `match` function; the wrapper keeps their intermediates allocated until the last node.
- The wrapped `graph_compute` cuts the graph: upstream computes its segments with its own `graph_compute`, the hook
  kernels run between them on the same stream (CUDA/HIP) or command buffer (Vulkan, with a barrier before and after).
- CPU op kernels run from a private extra buffer type of the CPU backend, on every thread of the op.
- Signatures: `ggml-hooks-cpu.h`, `ggml-hooks-cuda.h`, `ggml-hooks-vk.h`. Kernels may call upstream launchers and
  reuse upstream shaders (`ggml_hook_vk_pipeline`).
- A node that a kernel computed early can be skipped with `ggml_hook_mark_computed`.
- `models`: limit a kernel to some GGUF archs. The model tag comes from the libllama hooks.

Registration uses constant descriptors in an ELF section. No code runs when a library is loaded: a `GGML_BACKEND_DL`
CPU variant is loaded to be scored even on CPUs that cannot run it. A generated registrar per library, built without
the ISA flags of the target, registers the descriptors and removes them when the library is unloaded.

Backend extensions (`ggml_hook_backend_ext`, `GGML_HOOK_REGISTER_BACKEND_EXT`): reported memory and properties,
`supports_op`/`supports_buft`/`offload_op` answers, extra buffer types (custom weight layouts), registry procs,
backend init and free, graph passes after the upstream `graph_optimize`, callbacks around `graph_compute`, and
`skip_realloc_sync` (the scheduler sync before a re-reserve, a hook point).

## Fork code beside upstream

Where the fork rewrote a whole upstream subsystem, the fork version is built next to the upstream one and a run-time
switch selects it. Upstream stays the default.

- `fork_sched` (`ggml_hook_switch`, off): the fork scheduler and graph allocator (`ggml/src/fork-sched`). Their public
  functions are compiled as `fork_*`, the upstream ones are renamed `*__upstream`, and `ggml/src/ggml-fork-sched.cpp`
  calls one side. The switch is read once per process.
- HIP: `ggml/fork-cuda` is the fork `ggml/src/ggml-cuda`, each file inside `namespace ggml_fork`, regenerated by
  `hooks/tools/import-fork-cuda.py`. Adapted lines carry `hooks:`. The bridge (`ggml/kernels/cuda/fork-cuda.cu`) gives
  every upstream backend a fork context that computes on the upstream stream.
  - `fork_cuda_node` (on): nodes that upstream refuses and the fork accepts, for example the fork types and the EXL3
    ops, run through the fork `ggml_cuda_compute_forward`.
  - `fork_cuda_graph` (switch, off): the fork `graph_optimize` and `graph_compute` replace the upstream ones.
- Vulkan: `ggml/fork-vulkan` is the fork `ggml/src/ggml-vulkan` with its shaders (`hooks/tools/import-fork-vulkan.py`).
  The fork device uses the upstream instance and logical device, so both read the same buffers and share the queues.
  `fork_vk_node` (on) and `fork_vk_graph` (switch, off) work as on HIP.
- A backend extension can replace `graph_compute` and `graph_optimize` (`replace_graph_compute`,
  `replace_graph_optimize`): the switches above use them, so no allocation dependency is added for the whole graph.
- libllama: `llama/fork-src` is the fork `src/`, `llama/fork-include` the fork `include/`
  (`hooks/tools/import-fork-llama.py`). `fork_llama` (switch, off) sends the public API to the fork libllama; the
  generated `llama/src/llama-fork-dispatch.cpp` calls one side.
- Tools: `apps/` holds the fork common, mtmd, server, llama-bench, quantize and the example speculative-prefill
  (`hooks/tools/import-fork-apps.py`). `fork_tools` (switch, off) runs the fork tools from the upstream programs.
- Tests: `apps/fork-tests` holds the fork tests that are new or changed, built as `test-<name>-fork`
  (`cmake --build build --target hooks-fork-tests`). The upstream tests are unchanged.
- Files that are not built here: `python/` is the fork converter with its `conversion/` and `gguf-py/`
  (`python3 hooks/python/convert_hf_to_gguf.py`); `fork-files/<path>` keeps the fork version of the docs, CI
  workflows, UI sources, scripts and repository files, not active (`hooks/tools/import-fork-files.py`).
- Upstream files that the hooks must reach inside are compiled through a hooks file that includes them
  (`ggml/src/ggml-backend-upstream.cpp`, `ggml/src/ggml-cuda-upstream.cu`); the original is marked as a header.

## ggml types

A type is one entry of `GGML_HOOK_TYPES_LIST` in `ggml/include/ggml-hook-points.h` (id >= 100): name, block, size,
`to_float`, `from_float_ref` and the CPU traits. Quantize and validate are in `ggml/src/ggml-hooks-types.c`.
`GGML_TYPE_COUNT` becomes 160, so every `[GGML_TYPE_COUNT]` table of upstream grows. A device that has no hook kernel
for a node with a hook type does not support the node; the CPU runs `MUL_MAT` with the CPU traits and `GET_ROWS` with a
hook kernel. `q8_0_hook_example` (a copy of q8_0) is the test type.

## libllama

```cpp
#include "llama-hooks.h"

static void my_pass(const llama_model & model, const llm_graph_params & params, llm_graph_result & res) { /* res.get_ctx(), res.get_gf() */ }

LLAMA_HOOK_REGISTER([] {
    llama_hook h;
    h.name       = "my_pass";
    h.models     = "qwen35moe";
    h.graph_pass = my_pass;
    return h;
}());
```

`llama_hook` callbacks connected by link-time wrapping: `model_params`, `context_params`, `load_model`,
`create_model` (model factory), `tensor_created`, `create_memory`, `graph_pass`, `context_init`, `context_free`,
`model_free`, `process_begin`, `process_end`, `memory_op` (public `llama_memory_*` calls).

Callbacks connected by hook points (`llama-hook-points.h`): `kv_event` (every operation of every KV cache, also the
internal ones), `sched_reserve`, `resolve_fused_ops`, `ubatch_done`, `graph_extra_nodes`, `tensor_name` and
`tensor_info` (tensor ids upstream does not know), `kv_type` (indexer caches), `mmap_ranges` (ranges not prefetched),
`quant_type`.

`llama_hooks_context_of(params)` finds the context of a graph, `llama_hooks_current_context()` the context of the
running `llama_process`.

## common and server

- `common_hooks`: `args` (add CLI arguments, or change an upstream one with `common_hooks_find_arg`),
  `sampler_init`, `spec_draft` (change the draft params of each sequence before drafting), `spec_accept`,
  `quant_ftypes` (types of llama-quantize).
- `server_hooks`: `on_request` (read request fields into `hooks_data`), `on_slot_start`, `on_token` (return false to
  stop), `on_slot_release` (before the slot is reset), `register_routes`, `on_load_params`, `on_load`, `on_unload`,
  `on_task_launch` (change the task), `n_ctx_max`, `on_update` (slots as json, on demand), `on_usage`,
  `public_paths` (no API key), `model_hidden` (router presets).
- `hooks/server/handlers/example-counters.cpp` (`LLAMA_SERVER_HOOKS_EXAMPLE=1`) uses each of them.

## Run time switches

- `GGML_HOOKS="none,+my_*,-my_slow,qwen35moe:+moe_*"`: kernels, applied in order over the defaults. `arch:rule` applies
  only while that model runs.
- `LLAMA_HOOKS="none,+name"`: libllama hooks.
- `GGML_HOOKS_LOG=1`: log the first use of each kernel.
- The same `GGML_HOOKS` rules set the switches that are not kernels (`fork_sched`, read with `ggml_hook_switch`).

## Checks

- `hooks/tools/check-hooks.sh <upstream base>`: every changed line outside `hooks/` belongs to a hook point of `HOOKS.txt`.
- `hooks/tools/check-port.py <upstream base> <fork>`: every line the fork added is in this branch, and every
  fork change that is not kept as it is has an entry in `hooks/PORT.txt` (adapted, restored, variant, dropped,
  pending) with its reason. `--files` gives the state of each file.
- `hooks/tools/check-wraps.sh <build dir>`: every internal wrapped symbol is still called from another object file.
  A call from the file that defines the symbol is not wrapped, without any link error.
- `cmake --build build --target test-ggml-hooks test-llama-hooks`: kernels on and off against the reference on each
  backend, a mixed upstream/hook graph, the hook type, every libllama and common callback.
