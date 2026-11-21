# fork-vulkan

The fork `ggml/src/ggml-vulkan` tree, built into the upstream `ggml-vulkan` backend. Do not edit the copied files by
hand: `hooks/tools/import-fork-vulkan.py <fork ref> <upstream base>` regenerates them.

- The C++ files are the fork files inside `namespace ggml_fork`, closed around every `#include`. The shaders and the
  shader generator are copied as they are. The generator writes the shader symbols inside `namespace ggml_fork` and the
  header is `ggml-vulkan-fork-shaders.hpp`, so the fork pipelines cannot collide with the upstream ones.
- One VkDevice: the fork code takes the upstream instance and logical device (`ggml-vulkan-upstream.cpp`), with the
  same extensions and features, and the vulkan.hpp dispatcher of upstream. The fork kernels read the upstream buffers
  (the buffer structs are the same in both trees) and submit to the same queues.
- `ggml-impl.h` here is the fork one, for the fork code only.
- Adapted lines carry `hooks:`: the dispatcher, the instance and device, the shader header and the `ggml-vulkan.h`
  functions that the fork files define under other names (argument-dependent lookup would also find the global
  declarations).

Hooks files: `fork-hooks.h` (included at the end of `ggml-vulkan-types.h`), this file.

Known issue: with `fork_vk_graph`, `test-backend-ops -b Vulkan0` faults (GPUVM) or fails in ADD_ID, CONT, IM2COL_3D,
MUL_MAT_ADD, ROPE_SET_ROWS and TOP_K, and GATED_DELTA_NET exceeds its tolerance; the fork binary passes them. Every
other op passes, and a gpt-oss-20b perplexity run matches the fork binary. The default path is not affected.

The bridge is `hooks/ggml/kernels/vulkan/fork-vulkan.cpp`: a fork backend per upstream backend, the `fork_vk_node` kernel
and the `fork_vk_graph` switch. A barrier on the queue puts the fork work after the upstream work before it, and the
synchronize of the upstream backend also waits for the fork backend.
