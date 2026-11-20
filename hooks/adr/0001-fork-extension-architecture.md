# ADR 0001: Fork code outside upstream llama.cpp

- Status: proposed
- Date: 2026-11-20

## Context

A fork that edits upstream files hits merge conflicts on every upstream sync. We want the fork code to live in
`hooks/` so that updating upstream needs no conflict resolution.

We will not upstream our changes. Every line we add to an upstream file stays ours to maintain.

We need to:
- replace and fuse kernels on HIP, Vulkan and CPU;
- use custom weight layouts;
- change sampling;
- change model construction, the KV cache, the server loop and the CLI arguments;
- add ggml types.

Upstream already calls several of its components through tables of function pointers held in writable memory:
- backend devices, backends and registries (`ggml-backend-impl.h`);
- buffer types, and the CPU list of extra buffer types (`ggml_backend_cpu_get_extra_buffer_types()`);
- sampler chains (`llama_sampler_chain_i`).

The build can be extended without editing `CMakeLists.txt` files: `CMAKE_PROJECT_INCLUDE` plus
`cmake_language(DEFER)`. A module can be loaded through `GGML_BACKEND_PATH`. Backend libraries do not use hidden
visibility, so their internals (streams, pools, pipelines, kernel launchers) are reachable from the headers in the repo.

Most other upstream functions are called from another object file than the one that defines them. With a GNU
compatible linker, `-Wl,--wrap=<symbol>` sends those calls to our function, which can call the original. This
covers internal C++ functions (model creation, graph building, memory creation, tensor creation) and the public C API.

ggml types (`ggml_type` enum, const `type_traits` tables), CLI argument parsing and the server loop offer no seam:
the code is a table, or the calls stay inside one file.

## Decision

Fork code uses one hook API in `hooks/`. It connects to upstream in three ways.

**1. Wrapping at run time (no upstream line).** At load time we save upstream's function pointers and install ours in
front of them:

| Interface | Use |
|---|---|
| Device: `supports_op`, `init_backend`, `supports_buft`, `get_memory` | claim nodes for our kernels, adjust reported memory |
| Backend: `graph_compute`, `graph_optimize` | run each graph split as upstream segments plus our kernels on the same stream; reorder the split and add allocation dependencies before allocation |
| Registry: `get_proc_address` | add an extra buffer type to GPU devices, for custom weight layouts |
| CPU extra buffer types | CPU kernels and repacked weights |
| Sampler chains: `apply`, `accept` | read logits and tokens, force or ban tokens, stop a generation |
| Logging callbacks | observe or filter logs |

This covers all compute (HIP, Vulkan, CPU), weights and sampling.

**2. Wrapping at link time (no upstream line).** `-Wl,--wrap` on symbols called from another object file:

| Symbols | Use |
|---|---|
| public functions that list devices and create backends | install the run-time wrapper before the first use, in static, shared and `GGML_BACKEND_DL` builds |
| `llama_model_create`, `llama_model::build_graph`, `llama_model::create_memory`, `llama_model_loader::create_tensor` | model factory, graph passes, memory factory, tensor callbacks |
| public libllama API: model load and free, context init and free, `llama_process`, `llama_memory_*` | params, life time, the model tag, memory events |
| `common_sampler_init`, `common_speculative_draft`/`_accept`, the server request parser, `server_http_context::start`, `server_context::load_model` and its destructor | samplers, draft params, request fields, routes, model load and unload |

A tool checks that each wrapped internal symbol is still called from another object file: a call from its own file
is not wrapped, and that failure has no link error.

**3. Hook points (a few marked lines in upstream files).** Where no seam exists. Each line carries a `HOOK(name)`
marker and is listed in `hooks/HOOKS.txt`, which a tool checks against the upstream diff:

- ggml: the types (enum range, a header include, the two trait tables, quantize and validate) and the scheduler sync
  before a re-reserve;
- libllama: KV cache operations (virtual calls), graph reserve, fused op probes, end of a ubatch, graph node budget,
  tensor names, indexer KV types, mmap prefetch ranges, quantization types;
- CLI arguments and llama-quantize types;
- server: slot start, token, slot release, task launch, update, context size, usage, public paths, hidden presets,
  and a task field for request data.

**Kernels.** A kernel or fusion may call upstream launchers and templates, and may reuse upstream shaders with other
specialization constants. We copy an upstream kernel only to change code inside it. A copy records its source file
and upstream commit, so that the upstream diff of the original can be shown on every update.

**Run-time switches.** Every kernel, fusion and pass can be turned off (`GGML_HOOKS`, `LLAMA_HOOKS`). Results and
speed are always comparable with upstream.

**Fixes.** Problems are fixed in `hooks/`, never in upstream files. This applies to bugs in upstream code as well as to
our own. A fix uses the same seams as a feature. Where no seam exists, it uses one of these hook shapes:

| Shape | Form | Example use |
|---|---|---|
| observe | `hook(args)` | events, stats, cleanup |
| adjust | `hook(params &)` after upstream fills a struct | context, model or sampling params |
| replace | `if (hook(args, &res)) return res;` at the top of a function | our version of a function with a bug, or one that is `static` |
| factory | `if (auto p = hook(args)) return p;` before upstream creates an object | our subclass of a model, memory or other implementation |
| access | hook receives callbacks from the call site | state local to a `.cpp` file (server slots, speculative implementations) |

A replace hook runs our copy of the upstream function, and the copy records its source and upstream commit. When we
change upstream behavior that an upstream test checks, the change is a run-time setting whose default is upstream's,
so upstream tests keep passing unmodified. Our own tests live in separate executables.

Where the fork and upstream have two versions of the same code, upstream's version stays in place, and ours is a
hook selected at run time. A benchmark decides the default.

## Consequences

Positive:
- Compute code needs no upstream line, so upstream changes to backend files cannot conflict with it.
- Our kernels run between complete upstream calls, never in the middle of upstream internals. Upstream fusions do not
  take our nodes away.
- A fusion can cover any set of nodes in a backend split, and the nodes do not need to be consecutive.
- Kernels can reuse upstream launchers and templates, so upstream improvements to them carry over.
- The hook points are 61 lines in 19 files (`hooks/HOOKS.txt`), most of them one line each. Their number is the fixed
  cost of an upstream update.
- Without `CMAKE_PROJECT_INCLUDE`, the same tree builds as pure upstream, for comparisons.

Negative:
- Our code depends on upstream internals: interface layouts and backend context structs. An upstream change there
  breaks the build instead of producing a merge conflict.
- Each cut between upstream segments and our kernels splits the graph. Upstream fusions across a cut are lost, and each
  segment gets its own HIP graph. Measured cost: about 2 us per cut.
- A fusion stays inside one backend split.
- Patterns depend on how upstream builds the graph. A pattern that stops matching falls back to upstream silently, so
  CI must check hit counts per model.
- Copied kernels drift from upstream and need manual updates.
- Hook points remain where no seam exists, and they must be kept small and stable.
- Link-time wrapping uses internal C++ symbol names (Itanium ABI, libstdc++). A changed signature is a link error. A
  call moved into the defining file disables a wrapper without an error, so the wrap check must run in CI.
- MSVC and macOS linkers have no `--wrap`: the hooks are disabled there.

## Evidence

Measured on a Ryzen AI Max+ 395 (gfx1151, ROCm 10.1) with gpt-oss-20b MXFP4, using an example RMS_NORM kernel that
cuts the graph 49 times per token:

| | pp512 tok/s | tg128 tok/s | PPL |
|---|---|---|---|
| upstream | 2118 | 72.49 | 6.0154 |
| wrapper, no kernel | 2104 | 72.40 | 6.0154 |
| fusion off, kernel through hook points | 2072 | 66.69 | 6.0427 |
| fusion off, kernel through wrapper | 2076 | 66.24 | 6.0427 |

- The wrapper with no kernel gives the same perplexity as upstream, and decode is 0.1 % slower.
- With the same kernels, wrapping and hook points give the same perplexity. Wrapping is 0.7 % slower on decode, and
  prefill does not change.
- With upstream fusions on, the kernel connected through a hook point never ran: an upstream fusion took its node.

## Alternatives considered

- **Merge with `rerere`**: conflicts remain.
- **Patch queue rebased on upstream**: conflicts move to the patches.
- **Separate ggml backend**: llama prefers GPU devices and keeps one iGPU registry, each backend switch forces a host
  synchronization, and Vulkan tensor pointers are not real addresses.
- **Hook points everywhere, backends included**: more upstream lines, and our code runs inside upstream internals,
  where upstream optimizations interfere with it.
- **Run-time wrapping only**: it cannot reach model construction, the KV cache, the server loop, CLI arguments or
  ggml types. Link-time wrapping reaches most of them; the rest are the hook points.
