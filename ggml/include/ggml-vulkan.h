#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_VK_NAME "Vulkan"
#define GGML_VK_MAX_DEVICES 16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_vk_init(size_t dev_num);

GGML_BACKEND_API bool ggml_backend_is_vk(ggml_backend_t backend);
GGML_BACKEND_API int  ggml_backend_vk_get_device_count(void);
GGML_BACKEND_API void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vk_reg(void);

// fused-dispatch engagement counters: number of fused dispatches enqueued under this fusion name (e.g. "RMS_NORM_MUL_CPY",
// "QSA_BLOCK_SELECT") since the last reset. For tests and engagement checks; also reachable through the registry's
// proc-address table ("ggml_backend_vk_fusion_count" / "ggml_backend_vk_fusion_count_reset")
GGML_BACKEND_API uint64_t ggml_backend_vk_fusion_count(const char * fusion_name);
GGML_BACKEND_API void     ggml_backend_vk_fusion_count_reset(void);

#ifdef  __cplusplus
}
#endif
