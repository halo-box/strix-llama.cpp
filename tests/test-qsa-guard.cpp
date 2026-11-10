// The QSA placement guard of PLAN-v3 §10 (W7 done-when), with the model given on the command line: nothing here allocates the
// weights except the decode case, so the guard runs on the same no_alloc path as the -fit probe.
//   test-qsa-guard <model.gguf> pass [ctk]    a real device: the context is created and every reserve pass inspected
//                                             >= 2 nodes per QSA layer past the shortcut (non-vacuity count)
//   test-qsa-guard <model.gguf> fake          a fake GPU device registered with ggml_backend_device_register that
//                                             rejects SPARSE_ATTN: init error naming the cause and the remedy
//   test-qsa-guard <model.gguf> reject        a real HIP device under the test hook GGML_CUDA_SA_REJECT=1: init error
//   test-qsa-guard <model.gguf> reject-decode the weights are loaded, the hook is set after init: llama_decode returns an
//                                             error from the runtime guard, nothing is computed
//   test-qsa-guard <model.gguf> fake-idx T    a fake GPU device that takes the sparse-attention ops but, as HIP, no GET_ROWS
//                                             of IQ4_NL rows whose width is not a multiple of 256, with --cache-type-k-idx T:
//                                             iq4_nl (128-wide indexer rows) must be an init error naming the indexer cache,
//                                             any other T must create the context
//   test-qsa-guard <model.gguf> causal-sync  on the CPU, n_seq_max 2, unified: the hybrid-idx memory sees the context's
//                                             causal attention (the NC0 re-split applies only under it): false after init
//                                             with LLAMA_ATTENTION_TYPE_NON_CAUSAL, true after llama_set_causal_attn(true);
//                                             and the model's QSA ratios stay NC0 ratios (the eligibility filter)
// Exit status 0 = the expectation held.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "llama.h"
#include "../src/llama-memory-hybrid-idx.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

static std::string g_log;

static void log_capture(ggml_log_level level, const char * text, void *) {
    g_log += text;
    if (level == GGML_LOG_LEVEL_ERROR || strstr(text, "QSA") != nullptr) {
        fputs(text, stderr);
    }
}

// ---- a GPU device that takes everything the CPU takes except the two sparse-attention ops

static ggml_backend_dev_t fake_device();

static const char * fake_buft_name(ggml_backend_buffer_type_t) { return "FAKE_GPU"; }
static ggml_backend_buffer_t fake_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_UNUSED(buft);
    return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
}
static size_t fake_buft_align(ggml_backend_buffer_type_t) { return 64; }

static ggml_backend_buffer_type_t fake_buft() {
    static ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name         = */ fake_buft_name,
            /* .alloc_buffer     = */ fake_buft_alloc,
            /* .alloc_buffer_n   = */ nullptr,
            /* .get_alignment    = */ fake_buft_align,
            /* .get_max_size     = */ nullptr,
            /* .get_alloc_size   = */ nullptr,
            /* .get_alloc_size_n = */ nullptr,
            /* .is_host          = */ nullptr,     // not a host buffer: the K cache is "on the GPU"
        },
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    buft.device = fake_device();
    return &buft;
}

static const char * fake_dev_name(ggml_backend_dev_t) { return "FAKE0"; }
static const char * fake_dev_desc(ggml_backend_dev_t) { return "fake GPU for the QSA guard test"; }
static void fake_dev_memory(ggml_backend_dev_t, size_t * free, size_t * total) { *free = *total = (size_t) 1 << 40; }
static enum ggml_backend_dev_type fake_dev_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }
static void fake_dev_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = fake_dev_name(dev);
    props->description = fake_dev_desc(dev);
    props->type = fake_dev_type(dev);
    fake_dev_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = { false, false, false, false, false };   // async, host_buffer, buffer_from_host_ptr, events, mmap_support
}
static ggml_backend_t fake_dev_init(ggml_backend_dev_t dev, const char *) {
    ggml_backend_t b = ggml_backend_cpu_init();
    b->device = dev;
    return b;
}
static ggml_backend_buffer_type_t fake_dev_buft(ggml_backend_dev_t) { return fake_buft(); }
static bool g_fake_idx = false;   // fake-idx mode: the sparse-attention ops are taken, IQ4_NL GET_ROWS as on HIP

static bool fake_dev_supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    if (!g_fake_idx && (op->op == GGML_OP_SPARSE_ATTN || op->op == GGML_OP_SPARSE_ATTN_PLAN)) {
        return false;
    }
    // ggml-cuda.cu supports_op, GET_ROWS: IQ4_NL only for ne[0] % QK_K == 0
    if (g_fake_idx && op->op == GGML_OP_GET_ROWS && op->src[0]->type == GGML_TYPE_IQ4_NL && op->src[0]->ne[0] % 256 != 0) {
        return false;
    }
    return ggml_backend_dev_supports_op(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), op);
}
static bool fake_dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return buft == fake_buft() || buft == ggml_backend_cpu_buffer_type();
}

static const char * fake_reg_name(ggml_backend_reg_t) { return "FAKE"; }
static size_t fake_reg_count(ggml_backend_reg_t) { return 1; }
static ggml_backend_dev_t fake_reg_dev(ggml_backend_reg_t, size_t) { return fake_device(); }
static void * fake_reg_proc(ggml_backend_reg_t, const char *) { return nullptr; }

static ggml_backend_reg_t fake_reg() {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface = */ { fake_reg_name, fake_reg_count, fake_reg_dev, fake_reg_proc },
        /* .context = */ nullptr,
    };
    return &reg;
}

static ggml_backend_dev_t fake_device() {
    static ggml_backend_device dev = {
        /* .iface = */ {
            /* .get_name             = */ fake_dev_name,
            /* .get_description      = */ fake_dev_desc,
            /* .get_memory           = */ fake_dev_memory,
            /* .get_type             = */ fake_dev_type,
            /* .get_props            = */ fake_dev_props,
            /* .init_backend         = */ fake_dev_init,
            /* .get_buffer_type      = */ fake_dev_buft,
            /* .get_host_buffer_type = */ nullptr,
            /* .buffer_from_host_ptr = */ nullptr,
            /* .supports_op          = */ fake_dev_supports_op,
            /* .supports_buft        = */ fake_dev_supports_buft,
            /* .offload_op           = */ nullptr,
            /* .event_new            = */ nullptr,
            /* .event_free           = */ nullptr,
            /* .event_synchronize    = */ nullptr,
        },
        /* .reg     = */ nullptr,
        /* .context = */ nullptr,
    };
    dev.reg = fake_reg();
    return &dev;
}

static int fail(const char * what) {
    printf("FAIL: %s\n", what);
    return 1;
}

static ggml_type type_of(const char * s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * n = ggml_type_name((ggml_type) t);
        if (n && strcmp(n, s) == 0) {
            return (ggml_type) t;
        }
    }
    return GGML_TYPE_F16;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf pass [ctk] | fake | fake-idx <type> | reject | reject-decode | causal-sync\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[2];
    const ggml_type type = argc > 3 ? type_of(argv[3]) : GGML_TYPE_Q8_0;

    llama_log_set(log_capture, nullptr);
    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    mp.no_alloc = mode != "reject-decode";
    if (mode == "causal-sync") {
        mp.load_mode = LLAMA_LOAD_MODE_NONE;   // no_alloc on the CPU device: nothing is mapped or read
        mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
    }

    ggml_backend_dev_t devs[2] = { nullptr, nullptr };
    g_fake_idx = mode == "fake-idx";
    if (mode == "fake" || mode == "fake-idx") {
        ggml_backend_device_register(fake_device());
        devs[0] = fake_device();
        mp.devices = devs;
    }

    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) {
        return fail("model load");
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = 8192;
    cp.n_batch  = 512;
    cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    cp.type_k = type;
    cp.type_v = type;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (mode == "fake-idx") {
        cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
        cp.type_k_idx = type;
    }

    if (mode == "reject") {
        setenv("GGML_CUDA_SA_REJECT", "1", 1);
    }
    if (mode == "causal-sync") {
        cp.n_seq_max = 2;
        cp.kv_unified = true;
        cp.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
    }

    g_log.clear();
    llama_context * ctx = llama_init_from_model(model, cp);

    int rc = 0;
    if (mode == "pass") {
        if (ctx == nullptr) {
            rc = fail("context creation on a real device");
        } else {
            // the pp and tg lines of the init guard: inspected >= 2 x QSA layers past the shortcut
            int n_lines = 0;
            size_t p = 0;
            while ((p = g_log.find("QSA ctx=main", p)) != std::string::npos) {
                const size_t g = g_log.find("guard=", p);
                const int inspected = atoi(g_log.c_str() + g + 6);
                const int want = atoi(g_log.c_str() + g_log.find('/', g) + 1);
                printf("guard pass %d: inspected %d, 2 x layers = %d\n", n_lines, inspected, want);
                if (inspected < want || want < 2 || inspected == 0) {
                    rc = fail("non-vacuity count");
                }
                n_lines++;
                p = g;
            }
            if (n_lines == 0) {
                rc = fail("no guard line in the init log");
            }
        }
    } else if (mode == "fake" || mode == "reject") {
        const char * want = mode == "fake" ? "no sparse-attention kernel in this backend" : "rejected by the test hook GGML_CUDA_SA_REJECT";
        if (ctx != nullptr) {
            rc = fail("the context must not be created");
        } else if (g_log.find("QSA layer") == std::string::npos || g_log.find(want) == std::string::npos) {
            rc = fail("the init error does not name the layer and the cause");
        } else if (g_log.find("-ngl 0") == std::string::npos) {
            rc = fail("the init error does not name the remedy");
        } else {
            printf("the guard fired: %s\n", want);
        }
    } else if (mode == "fake-idx") {
        if (type == GGML_TYPE_IQ4_NL) {
            if (ctx != nullptr) {
                rc = fail("the context must not be created: the indexer GET_ROWS would run on the CPU");
            } else if (g_log.find("the indexer K cache (iq4_nl") == std::string::npos || g_log.find("GET_ROWS") == std::string::npos ||
                       g_log.find("--cache-type-k-idx auto") == std::string::npos) {
                rc = fail("the init error does not name the indexer cache, the op and the remedy");
            } else {
                printf("the guard fired on the iq4_nl indexer cache\n");
            }
        } else if (ctx == nullptr) {
            rc = fail("context creation with this indexer type");
        } else {
            printf("context created with the %s indexer cache\n", ggml_type_name(type));
        }
    } else if (mode == "causal-sync") {
        const auto * mem = ctx ? dynamic_cast<const llama_memory_hybrid_idx *>(llama_get_memory(ctx)) : nullptr;
        if (mem == nullptr) {
            rc = fail("context creation with a hybrid-idx memory");
        } else {
            const bool at_init = mem->qsa_causal_attn();
            llama_set_causal_attn(ctx, true);
            const bool after_set = mem->qsa_causal_attn();
            llama_set_causal_attn(ctx, false);
            const bool after_unset = mem->qsa_causal_attn();
            printf("memory causal attention: %d after a non-causal init, %d after set(true), %d after set(false)\n", at_init, after_set, after_unset);
            if (at_init || !after_set || after_unset) {
                rc = fail("the memory's causal attention does not follow the context");
            }
            // the model's QSA layers keep their NC0 ratio (Qwen3.8: ratio 4, top-k 2048, head 256): the eligibility filter
            // must not drop a layer block selection runs on
            std::string ratios;
            for (uint32_t r : mem->qsa_nc0_ratios()) {
                ratios += (ratios.empty() ? "" : ", ") + std::to_string(r);
            }
            printf("NC0 ratios: %s\n", ratios.empty() ? "none" : ratios.c_str());
            if (mem->qsa_nc0_ratios().empty()) {
                rc = fail("no NC0 ratio: the re-split would never run for this model");
            }
        }
    } else if (mode == "reject-decode") {
        if (ctx == nullptr) {
            rc = fail("context creation");
        } else {
            setenv("GGML_CUDA_SA_REJECT", "1", 1);
            // 256 queries: the sparse op's n_q band under the shipped prefer_dense policy of q8_0 (AT-W7PERF)
            std::vector<llama_token> tok(256, 1);
            llama_batch b = llama_batch_get_one(tok.data(), (int32_t) tok.size());
            const int ret = llama_decode(ctx, b);
            printf("llama_decode returned %d\n", ret);
            if (ret == 0) {
                rc = fail("llama_decode must return an error");
            } else if (g_log.find("rejected by the test hook") == std::string::npos) {
                rc = fail("the decode error does not name the cause");
            }
        }
    } else {
        rc = fail("unknown mode");
    }

    if (ctx) {
        llama_free(ctx);
    }
    llama_model_free(model);
    llama_backend_free();
    printf("%s\n", rc == 0 ? "PASS" : "FAILED");
    return rc;
}
