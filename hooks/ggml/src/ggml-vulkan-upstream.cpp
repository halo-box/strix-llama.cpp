// The upstream ggml/src/ggml-vulkan/ggml-vulkan.cpp, compiled through this file (ggml-hooks.cmake marks the original as
// a header): the hooks reach static upstream functions here. The upstream public functions that the hooks replace are
// renamed *__upstream (hooks_rename).

#include "../../../ggml/src/ggml-vulkan/ggml-vulkan.cpp"

// the device answer of upstream, before the backend wrapper
bool ggml_hook_vk_upstream_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    return ggml_backend_vk_device_supports_op(dev, op);
}

// the instance and the logical device of upstream, for the fork code (hooks/ggml/fork-vulkan): one VkDevice
vk::Instance ggml_hook_vk_upstream_instance() {
    ggml_vk_instance_init();
    return vk_instance.instance;
}

vk::Device ggml_hook_vk_upstream_device(size_t idx) {
    return ggml_vk_get_device(idx)->device;
}

// the device index of an upstream Vulkan backend
size_t ggml_hook_vk_backend_index(ggml_backend_t backend) {
    return ((ggml_backend_vk_device_context *) backend->device->context)->device;
}

ggml_backend_dev_t ggml_hook_vk_upstream_dev(size_t device) {
    return ggml_backend_reg_dev_get(ggml_backend_vk_reg(), device);
}

// a device buffer of the upstream backend (not a host buffer)
bool ggml_hook_vk_upstream_buffer_is_vk(ggml_backend_buffer_t buffer) {
    return ggml_backend_buffer_is_vk(buffer);
}
