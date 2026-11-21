// hooks: common/exl3-dir.h of the fork (1fe05fa45, upstream 9048bad8d, new in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
#pragma once
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include "llama.h"
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

} // namespace ggml_fork
#include <cstddef>
#include <memory>
#include <string>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

// direct loading of an exllamav3 EXL3 model directory (config.json, safetensors, tokenizer files)
// the import is the GGUF metadata and tensor set that convert_hf_to_gguf.py writes for the same directory

} // namespace ggml_fork
struct gguf_context; // hooks: global
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md
struct common_exl3_dir;

struct common_exl3_dir_deleter {
    void operator()(common_exl3_dir * dir);
};

typedef std::unique_ptr<common_exl3_dir, common_exl3_dir_deleter> common_exl3_dir_ptr;

// true if path is a directory: -m loads it with the EXL3 importer
bool common_exl3_is_dir(const std::string & path);

// read and check the directory and build the import; throws std::runtime_error if the directory is not supported
common_exl3_dir_ptr common_exl3_dir_open(const std::string & path);

// the GGUF metadata and tensor infos of the import, owned by dir
const gguf_context * common_exl3_dir_gguf(const common_exl3_dir * dir);

// the data of tensor `name` of the import; size must be its byte size; not thread safe
void common_exl3_dir_read(const common_exl3_dir * dir, const char * name, void * dst, size_t size);

// llama_model_init_from_user with the import, or llama_model_load_from_file if path is not a directory
llama_model * common_exl3_load_model(const std::string & path, llama_model_params params);
} // namespace ggml_fork
