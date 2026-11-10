// The sparse-attention ops through the RPC backend, against an in-process rpc-server on one local device (the first GPU
// of the build, else the CPU):
//  (1) alloc size: the RPC buffer type's size of a SPARSE_ATTN node equals the server device's own for that node. The
//      client cached the size by the dst's own shape and op_params, but the workspace follows the plan's n_kv / n_sel / G
//      and the K / V types: two nodes with one dst shape (a 37-row ubatch at n_kv 256 in the shortcut, then at n_kv 8192 on
//      the selection rows) got the first node's size, and the server's kernels wrote past the allocation.
//  (2) supports_op: the RPC device answers the server device's supports_op for SPARSE_ATTN / SPARSE_ATTN_PLAN. It returned
//      true for every node, so llama's QSA placement guard passed for a server that cannot run the op (a CUDA build aborts
//      in its compute).
// (1) is vacuous on a device whose alloc size does not depend on the plan (the CPU): it then passes trivially.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

static int g_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); ++g_fail; } } while (0)

static bool port_open(int port) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a = {};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t) port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    const bool ok = connect(fd, (sockaddr *) &a, sizeof(a)) == 0;
    close(fd);
    return ok;
}

struct node_shape {
    ggml_type type_k, type_v;
    int64_t n_q, n_kv, n_sel;
    int32_t G;
};

static ggml_tensor * build_attn(ggml_context * ctx, const node_shape & s) {
    const int64_t kv_head = 2, gqa = 12, rows = 4 * ((s.n_kv + 3) / 4);
    ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, s.n_q, kv_head * gqa, 1);
    ggml_tensor * k    = ggml_new_tensor_4d(ctx, s.type_k, 256, rows, kv_head, 1);
    ggml_tensor * v    = ggml_new_tensor_4d(ctx, s.type_v, 256, rows, kv_head, 1);
    ggml_tensor * sel  = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, s.n_sel, s.n_q, 1, 1);
    ggml_tensor * plan = ggml_sparse_attn_plan(ctx, sel, s.n_kv, s.G);
    return ggml_sparse_attn(ctx, q, k, v, plan, 0.0625f, 0, 0);
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count() && dev == nullptr; ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            dev = d;
        }
    }
    if (dev == nullptr) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    if (dev == nullptr) {
        printf("test-rpc-sparse-attn: no device\n");
        return 1;
    }
    const int port = 20000 + (int) (getpid() % 20000);
    const std::string endpoint = "127.0.0.1:" + std::to_string(port);
    printf("server device %s (%s) on %s\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), endpoint.c_str());
    std::thread([dev, endpoint]() {
        ggml_backend_dev_t devs[1] = { dev };
        ggml_backend_rpc_start_server(endpoint.c_str(), nullptr, 4, 1, devs);
    }).detach();
    bool up = false;
    for (int i = 0; i < 200 && !up; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        up = port_open(port);
    }
    if (!up) {
        printf("test-rpc-sparse-attn: the server did not start\n");
        fflush(stdout);
        _exit(1);
    }
    ggml_backend_reg_t reg = ggml_backend_rpc_add_server(endpoint.c_str());
    if (reg == nullptr || ggml_backend_reg_dev_count(reg) != 1) {
        printf("test-rpc-sparse-attn: add_server failed\n");
        fflush(stdout);
        _exit(1);
    }
    ggml_backend_dev_t rdev = ggml_backend_reg_dev_get(reg, 0);
    ggml_backend_buffer_type_t rbuft = ggml_backend_dev_buffer_type(rdev);
    ggml_backend_buffer_type_t lbuft = ggml_backend_dev_buffer_type(dev);

    ggml_init_params ip = { 256 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    // (1) one dst shape [256, 24, n_q, 1] per group, the plan varying: the order matters (the smaller workspace first)
    int n_alloc = 0, n_differ = 0;
    const std::vector<std::vector<node_shape>> groups = {
        { { GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 37, 256, 256, 1 },  { GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 37, 8192, 2051, 1 } },
        { { GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 276, 1536, 1536, 4 }, { GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 276, 16384, 2051, 4 } },
        { { GGML_TYPE_F16, GGML_TYPE_F16, 9, 512, 512, 1 }, { GGML_TYPE_F16, GGML_TYPE_Q8_0, 9, 4096, 2051, 1 },
          { GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, 9, 65536, 2051, 1 } },
    };
    for (const auto & g : groups) {
        size_t first = 0;
        for (size_t j = 0; j < g.size(); ++j) {
            const node_shape & s = g[j];
            ggml_tensor * attn = build_attn(ctx, s);
            if (!ggml_backend_dev_supports_op(dev, attn)) {
                printf("  %s/%s n_q %" PRId64 " n_kv %" PRId64 " G %d: not supported on the server device, skipped\n",
                       ggml_type_name(s.type_k), ggml_type_name(s.type_v), s.n_q, s.n_kv, s.G);
                continue;
            }
            const size_t local  = ggml_backend_buft_get_alloc_size(lbuft, attn);
            const size_t remote = ggml_backend_buft_get_alloc_size(rbuft, attn);
            printf("  %s/%s n_q %" PRId64 " n_kv %" PRId64 " n_sel %" PRId64 " G %d: nbytes %zu, server device %zu, through RPC %zu\n",
                   ggml_type_name(s.type_k), ggml_type_name(s.type_v), s.n_q, s.n_kv, s.n_sel, s.G, ggml_nbytes(attn), local, remote);
            CHECK(remote == local, "RPC alloc size %zu != the server device's %zu", remote, local);
            n_alloc++;
            if (j == 0) {
                first = local;
            } else if (local != first) {
                n_differ++;
            }
        }
    }
    printf("(1) %d alloc sizes compared, %d of a dst shape whose size differs from the first%s\n", n_alloc, n_differ,
           n_differ == 0 ? " (vacuous on this device)" : "");

    // (2) supports_op through RPC == the server device's, for SPARSE_ATTN and its plan
    int n_sup = 0, n_unsup = 0;
    const std::vector<node_shape> sup = {
        { GGML_TYPE_F16,  GGML_TYPE_F16,  9, 4096, 2051, 1 },
        { GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 9, 4096, 2051, 4 },
        { GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, 9, 4096, 2051, 1 },   // outside S: the CPU takes it, HIP and Vulkan do not
        { GGML_TYPE_I32,  GGML_TYPE_I32,  9, 4096, 2051, 1 },   // no to_float: nobody takes it
        { GGML_TYPE_F16,  GGML_TYPE_F16,  9, 4096, 2051, 4 },
    };
    for (const node_shape & s : sup) {
        ggml_tensor * attn = build_attn(ctx, s);
        for (ggml_tensor * t : { attn, attn->src[3] }) {
            const bool local  = ggml_backend_dev_supports_op(dev, t);
            const bool remote = ggml_backend_dev_supports_op(rdev, t);
            printf("  %-17s %s/%s G %d: server device %d, through RPC %d\n", ggml_op_desc(t), ggml_type_name(s.type_k),
                   ggml_type_name(s.type_v), s.G, (int) local, (int) remote);
            CHECK(remote == local, "%s: RPC supports_op %d != the server device's %d", ggml_op_desc(t), (int) remote, (int) local);
            (local ? n_sup : n_unsup)++;
        }
    }
    printf("(2) supports_op compared on %d supported and %d unsupported nodes\n", n_sup, n_unsup);

    ggml_free(ctx);
    printf("test-rpc-sparse-attn: %d failed checks: %s\n", g_fail, g_fail == 0 && n_unsup > 0 ? "OK" : "FAILED");
    fflush(stdout);
    // the server thread blocks in accept(): leave without joining it
    _exit(g_fail == 0 && n_unsup > 0 ? 0 : 1);
}
