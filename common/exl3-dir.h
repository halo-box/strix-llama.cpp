#pragma once

#include "llama.h"

#include <cstddef>
#include <memory>
#include <string>

// direct loading of an exllamav3 EXL3 model directory (config.json, safetensors, tokenizer files)
// the import is the GGUF metadata and tensor set that convert_hf_to_gguf.py writes for the same directory

struct gguf_context;
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
