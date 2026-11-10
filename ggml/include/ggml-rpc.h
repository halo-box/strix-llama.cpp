#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    8
#define RPC_PROTO_MINOR_VERSION    1
#define RPC_PROTO_PATCH_VERSION    0

// The oldest server minor this client talks to. The hello accepts a server of the same major whose minor is in
// [RPC_PROTO_MIN_SERVER_MINOR_VERSION, RPC_PROTO_MINOR_VERSION]; an older server is rejected because this client may
// send what it lacks: 8.1 appends GGML_OP_SPARSE_ATTN_PLAN and GGML_OP_SPARSE_ATTN before GGML_OP_COUNT (an 8.0 server
// has GGML_OP_COUNT == 101 and no case for them) and adds RPC_CMD_SUPPORTS_OP (asked for those two ops). A change of
// the op enum needs a minor bump and this floor raised to it; deserialize rejects an op value >= GGML_OP_COUNT.
#define RPC_PROTO_MIN_SERVER_MINOR_VERSION 1
#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 103, "GGML_OP_COUNT has changed - update RPC_PROTO_MINOR_VERSION");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                                    size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

#ifdef  __cplusplus
}
#endif
