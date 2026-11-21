// Vulkan driver for the backend wrapper (see ggml-hooks-impl.h). Built in the backend target.
//
// Upstream ends each segment with a submit. The hook kernels of a run record into the open compute context
// (ggml_vk_get_compute_ctx), and the next upstream segment submits them with its own commands.
// Upstream tracks write/read hazards only inside one graph_compute call, so the driver puts a barrier before and after
// each hook kernel. When nothing follows, the driver submits the context itself.

#include "ggml-vulkan-common.h"

#include "ggml-hooks-impl.h"
#include "ggml-hooks-vk.h"

#include <map>
#include <mutex>
#include <string>

namespace {

thread_local vk_context g_subctx;

void vk_dev_info(ggml_backend_dev_t dev, ggml_hook_dev * info) {
    const auto * dctx = (const ggml_backend_vk_device_context *) dev->context;
    info->device     = (int) dctx->device;
    info->name       = dctx->name.c_str();
    info->integrated = dctx->is_integrated_gpu;
    // without creating the logical device
    const auto devices = vk_instance.instance.enumeratePhysicalDevices();
    const size_t idx = vk_instance.device_indices[dctx->device];
    info->vendor_id = idx < devices.size() ? devices[idx].getProperties().vendorID : 0;
}

ggml_backend_vk_context * vk_ctx(ggml_backend_t backend) {
    return (ggml_backend_vk_context *) backend->context;
}

void vk_begin(ggml_backend_t backend) {
    g_subctx = ggml_vk_get_compute_ctx(vk_ctx(backend));
}

bool vk_run_op(ggml_backend_t backend, const ggml_hook_kernel * k, ggml_cgraph * cgraph, int i) {
    ggml_backend_vk_context * ctx = vk_ctx(backend);
    ggml_vk_sync_buffers(ctx, g_subctx);
    return k->compute.vk_op(ctx, g_subctx, cgraph, i);
}

void vk_run_fusion(ggml_backend_t backend, const ggml_hook_kernel * k, ggml_cgraph * cgraph, int i, int n) {
    ggml_backend_vk_context * ctx = vk_ctx(backend);
    ggml_vk_sync_buffers(ctx, g_subctx);
    k->compute.vk_fusion(ctx, g_subctx, cgraph, i, n);
}

void vk_end(ggml_backend_t backend, bool last) {
    ggml_backend_vk_context * ctx = vk_ctx(backend);
    ggml_vk_sync_buffers(ctx, g_subctx);
    if (last) {
        ggml_vk_ctx_end(g_subctx);
        ctx->compute_ctx.reset();
        ggml_vk_submit(g_subctx, {});
        ctx->submit_pending = true;
        if (!ctx->device->support_async) {
            ggml_vk_synchronize(ctx);
        }
    }
    g_subctx.reset();
}

constexpr ggml_hook_driver driver = {
    /* .backend       = */ GGML_HOOK_BACKEND_VULKAN,
    /* .reg_name      = */ GGML_VK_NAME,
    /* .reg           = */ ggml_backend_vk_reg,
    /* .dev_info      = */ vk_dev_info,
    /* .cut_ops       = */ true,
    /* .begin         = */ vk_begin,
    /* .run_op        = */ vk_run_op,
    /* .run_fusion    = */ vk_run_fusion,
    /* .end           = */ vk_end,
    /* .installed     = */ nullptr,
    /* .private_bufts = */ nullptr,
};

GGML_HOOK_REGISTER_DRIVER(driver);

void create_pipeline(vk_device & device, vk_pipeline & pipeline, const void * spv, size_t spv_size, const std::vector<uint32_t> & spec,
                     uint32_t required_subgroup_size) {
    pipeline->shader_module = device->device.createShaderModule({ {}, spv_size, reinterpret_cast<const uint32_t *>(spv) });

    vk::PushConstantRange pcr(vk::ShaderStageFlagBits::eCompute, 0, pipeline->push_constant_size);
    pipeline->layout = device->device.createPipelineLayout({ {}, device->dsl, pcr });

    std::vector<vk::SpecializationMapEntry> entries(spec.size());
    for (size_t i = 0; i < spec.size(); ++i) {
        entries[i] = vk::SpecializationMapEntry((uint32_t) i, (uint32_t) (i * sizeof(uint32_t)), sizeof(uint32_t));
    }
    vk::SpecializationInfo spec_info(entries.size(), entries.data(), spec.size() * sizeof(uint32_t), spec.data());

    vk::PipelineShaderStageCreateInfo stage({}, vk::ShaderStageFlagBits::eCompute, pipeline->shader_module, "main", &spec_info);
    vk::PipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroup;
    subgroup.requiredSubgroupSize = required_subgroup_size;
    if (device->subgroup_size_control && required_subgroup_size > 0) {
        GGML_ASSERT(device->subgroup_min_size <= required_subgroup_size && required_subgroup_size <= device->subgroup_max_size);
        stage.setPNext(&subgroup);
    }
    pipeline->pipeline = device->device.createComputePipeline(VK_NULL_HANDLE, vk::ComputePipelineCreateInfo({}, stage, pipeline->layout)).value;

    // upstream destroys the pipelines of all_pipelines with the device
    std::lock_guard<std::mutex> guard(device->compile_mutex);
    device->all_pipelines.push_back(pipeline);
    pipeline->compiled = true;
}

} // namespace

vk_pipeline ggml_hook_vk_pipeline(ggml_backend_vk_context * ctx, const char * name, const void * spv, size_t spv_size, uint32_t parameter_count,
                                  uint32_t push_constant_size, std::array<uint32_t, 3> wg_denoms, const std::vector<uint32_t> & specialization_constants,
                                  uint32_t required_subgroup_size) {
    struct cached {
        std::weak_ptr<vk_device_struct> device;
        vk_pipeline                     pipeline;
    };
    static std::mutex mutex;
    static std::map<std::pair<const vk_device_struct *, std::string>, cached> cache;

    std::lock_guard<std::mutex> lock(mutex);
    auto & c = cache[{ ctx->device.get(), name }];
    if (c.pipeline && !c.device.expired()) {
        return c.pipeline;
    }

    vk_pipeline p = std::make_shared<vk_pipeline_struct>();
    p->name               = name;
    p->parameter_count    = parameter_count;
    p->push_constant_size = push_constant_size;
    p->wg_denoms          = wg_denoms;
    p->align              = 1;
    p->initialized        = true;
    create_pipeline(ctx->device, p, spv, spv_size, specialization_constants, required_subgroup_size);

    c = { ctx->device, p };
    return p;
}

// Backends created without the registry (ggml_backend_{name}_init): the upstream function is renamed in its file
// (hooks_rename, ggml-hooks.cmake). This one installs the wrapper of the registry and wraps the new backend.
extern "C" { GGML_BACKEND_API ggml_backend_t ggml_backend_vk_init__upstream(size_t dev_num); }
ggml_backend_t ggml_backend_vk_init(size_t dev_num) {
    ggml_backend_t backend = ggml_backend_vk_init__upstream(dev_num);
    ggml_hook_install(ggml_backend_vk_reg());
    ggml_hook_wrap_backend(backend);
    return backend;
}
