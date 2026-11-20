#pragma once

// Vulkan hook kernels. Include after ggml-vulkan-common.h, from files built in the backend (kernels/vulkan/).

#include "ggml-hooks.h"

#include <array>
#include <vector>

// kernel signatures for GGML_HOOK_BACKEND_VULKAN (ggml_hook_vk_op_fn, ggml_hook_vk_fusion_fn):
//   op:     bool kernel(ggml_backend_vk_context * ctx, vk_context & subctx, ggml_cgraph * cgraph, int node_idx);  return false to let upstream compute the node
//   fusion: void kernel(ggml_backend_vk_context * ctx, vk_context & subctx, ggml_cgraph * cgraph, int node_idx, int n_nodes);
// The wrapper puts a barrier before and after each kernel. Upstream submits the commands with its next segment.

// pipeline cache for hook shaders, one pipeline per device and name. Uses the descriptor set layout of the device.
// The pipeline is destroyed with the device.
vk_pipeline ggml_hook_vk_pipeline(ggml_backend_vk_context * ctx, const char * name, const void * spv, size_t spv_size, uint32_t parameter_count,
                                  uint32_t push_constant_size, std::array<uint32_t, 3> wg_denoms, const std::vector<uint32_t> & specialization_constants = {},
                                  uint32_t required_subgroup_size = 0);
