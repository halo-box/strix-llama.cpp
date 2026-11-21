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

The bridge is `hooks/ggml/kernels/vulkan/fork-vulkan.cpp`.

- `fork_vk_node` (on): a fork backend per upstream backend computes the nodes that upstream refuses. A barrier on the
  queue puts the fork work after the upstream work before it, and the synchronize of the upstream backend also waits
  for the fork backend. The fork `vk_device_struct` has another layout than the upstream one, and the fork code reaches
  the device through the buffers (`buffer->device`): while a fork node is recorded, the upstream buffers it uses point
  at the fork device of the same index, then get their upstream device back.
- `fork_vk_graph` (switch, off): the upstream devices take the fork device functions (`device_install`; the device
  contexts have the same layout), so the buffer types, backends and events are the fork ones and point at the upstream
  device. The fork code only sees fork objects, as in a fork build. Hook kernels do not apply in this mode.
  `test-backend-ops -b Vulkan0` passes in full with the fork test program (49334 cases). The upstream test program
  passes except GATED_DELTA_NET with head size 128 and 64 tokens or more: the fork chunked kernel stays under 5e-7, the
  tolerance the fork test sets for it, above the 1e-7 of the upstream test.
