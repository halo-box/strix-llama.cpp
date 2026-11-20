// End to end test of the hooks above the backends, on a random "llama" model built in memory:
// - llama model hook: the model is created by a hook
// - graph pass: a pass that scales the logits by 2 changes the output, and only while it is on
// - model tag: a ggml kernel sees the model arch, the `models` filter and the per-model GGML_HOOKS rules apply
//
// The random model comes from the upstream test-llama-archs helpers, included here with its main() renamed.

#define main test_llama_archs_main
#include "test-llama-archs.cpp"
#undef main

#include "common-hooks.h"
#include "ggml-hooks.h"
#include "llama-graph.h"
#include "llama-hooks.h"
#include "llama-hooks-access.h"
#include "llama-context.h"
#include "llama-kv-cache.h"
#include "llama-arch.h"
#include "llama-model-loader.h"
#include "llama-model.h"

#include <cstdlib>

// model hook: create the upstream model, through the hook
static int n_model_hook = 0;
static llama_model * test_model_hook(llama_model_loader & ml, const llama_model_params & params) {
    n_model_hook++;
    return llama_model_create(ml.get_arch(), params);
}

// graph pass: logits * 2. It also records the context it sees.
static int n_pass_with_ctx = 0;
static void test_pass_scale_logits(const llama_model &, const llm_graph_params & params, llm_graph_result & res) {
    ggml_tensor * logits = res.t_logits;
    if (!logits) {
        return;
    }
    if (llama_hooks_current_context() && llama_hooks_context_of(params) == llama_hooks_current_context()) {
        n_pass_with_ctx++;
    }
    ggml_tensor * scaled = ggml_scale(res.get_ctx(), logits, 2.0f);
    llama_hook_graph_replace(res.get_gf(), logits, scaled);
    res.t_logits = scaled;
}

// a pass for another arch must not run
static int n_other_pass = 0;
static void test_pass_other(const llama_model &, const llm_graph_params &, llm_graph_result &) {
    n_other_pass++;
}

// hook points in src/ (llama-hook-points.h)
static int n_kv[8] = {};
static int n_sched_reserve = 0, n_fused = 0, n_ubatch_done = 0, n_extra_nodes = 0, n_mmap = 0, n_quant = 0;
static const llm_tensor TEST_HOOK_TENSOR = (llm_tensor) 100000;
static const llm_tensor_info test_tensor_info = { LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT };

LLAMA_HOOK_REGISTER([] {
    llama_hook h;
    h.name              = "test_points";
    h.kv_event          = [](const llama_kv_cache &, const llama_hook_kv_ev & ev) { n_kv[ev.type]++; };
    h.sched_reserve     = [](llama_context &, llama_memory_context_i *, uint32_t, uint32_t) { n_sched_reserve++; };
    h.resolve_fused_ops = [](llama_context &, const llama_memory_context_i *, uint32_t) { n_fused++; };
    h.ubatch_done       = [](llama_context &, const llama_ubatch &, ggml_status) { n_ubatch_done++; };
    h.graph_extra_nodes = [](const llama_context &) -> uint32_t { n_extra_nodes++; return 16; };
    h.tensor_name       = [](const LLM_TN_IMPL & tn, std::string & name) {
        if (tn.tensor != TEST_HOOK_TENSOR) {
            return false;
        }
        name = "blk." + std::to_string(tn.bid) + ".hook_tensor";
        return true;
    };
    h.tensor_info       = [](llm_tensor t) -> const llm_tensor_info * { return t == TEST_HOOK_TENSOR ? &test_tensor_info : nullptr; };
    h.mmap_ranges       = [](const llama_model_loader &, uint32_t, llama_mmap::ranges &) { n_mmap++; };
    h.quant_type        = [](const llama_model_quantize_params *, const ggml_tensor *, ggml_type type) { n_quant++; return type; };
    return h;
}());

// the other callbacks only count
static int n_tensors = 0, n_memory = 0, n_ctx_init = 0, n_ctx_free = 0, n_model_free = 0, n_begin = 0, n_end = 0, n_mem_clear = 0, n_ctx_params = 0;

LLAMA_HOOK_REGISTER([] {
    llama_hook h;
    h.name         = "test_model";
    h.create_model = test_model_hook;
    return h;
}());

LLAMA_HOOK_REGISTER([] {
    llama_hook h;
    h.name       = "test_scale_logits";
    h.models     = "llama";
    h.graph_pass = test_pass_scale_logits;
    return h;
}());

LLAMA_HOOK_REGISTER([] {
    llama_hook h;
    h.name       = "test_other_arch";
    h.models     = "qwen2";
    h.graph_pass = test_pass_other;
    return h;
}());

LLAMA_HOOK_REGISTER([] {
    llama_hook h;
    h.name           = "test_counters";
    h.tensor_created = [](llama_model_loader &, const LLM_TN_IMPL &, ggml_tensor *, int) { n_tensors++; };
    h.create_memory  = [](const llama_model &, const llama_memory_params &, const llama_cparams &) -> llama_memory_i * { n_memory++; return nullptr; };
    h.context_params = [](const llama_model &, llama_context_params & p) { n_ctx_params++; p.n_threads = 3; };
    h.context_init   = [](llama_context &) { n_ctx_init++; };
    h.context_free   = [](llama_context &) { n_ctx_free++; };
    h.model_free     = [](llama_model &) { n_model_free++; };
    h.process_begin  = [](llama_context &, int) { n_begin++; };
    h.process_end    = [](llama_context &, int, int32_t) { n_end++; };
    h.memory_op      = [](llama_memory_t, const llama_hook_memory_op & op) { n_mem_clear += op.type == LLAMA_HOOK_MEMORY_CLEAR; };
    return h;
}());

// ggml kernels that never take a node: they only record what they see
static std::string seen_arch;
static int n_seen        = 0;
static int n_filtered    = 0;
static int n_rule_off    = 0;

static bool test_tag_supports(const ggml_hook_dev *, const ggml_tensor *) {
    seen_arch = ggml_hook_model_arch();
    n_seen++;
    return false;
}
static bool test_filtered_supports(const ggml_hook_dev *, const ggml_tensor *) {
    n_filtered++;
    return false;
}
static bool test_rule_off_supports(const ggml_hook_dev *, const ggml_tensor *) {
    n_rule_off++;
    return false;
}
struct ggml_compute_params;
static void test_never(const ggml_compute_params *, ggml_tensor *) {}

GGML_HOOK_REGISTER_KERNEL(ggml_hook_op_kernel("test_tag", GGML_HOOK_BACKEND_CPU, GGML_OP_MUL_MAT, 0, test_tag_supports, ggml_hook_fn(test_never)));
GGML_HOOK_REGISTER_KERNEL([] {
    ggml_hook_kernel k = ggml_hook_op_kernel("test_filtered", GGML_HOOK_BACKEND_CPU, GGML_OP_MUL_MAT, 0, test_filtered_supports, ggml_hook_fn(test_never));
    k.models = "qwen2,gemma3";
    return k;
}());
GGML_HOOK_REGISTER_KERNEL(ggml_hook_op_kernel("test_rule_off", GGML_HOOK_BACKEND_CPU, GGML_OP_MUL_MAT, 0, test_rule_off_supports, ggml_hook_fn(test_never)));

// CLI argument hook: the value lives in this file, not in common_params. It also changes an upstream argument.
static int test_arg_value = 0;
static bool test_seed_seen = false;
static int n_sampler_init = 0;
static void (*upstream_seed)(common_params &, const std::string &) = nullptr;
COMMON_HOOK_REGISTER([] {
    common_hooks h;
    h.name = "test_args";
    h.args = [](common_params_context & ctx, const std::function<void(common_arg)> & add) {
        add(common_arg({ "--test-hook-arg" }, "N", "test argument added by a hook", [](common_params &, int value) { test_arg_value = value; })
                .set_examples({ LLAMA_EXAMPLE_COMMON }));
        if (common_arg * seed = common_hooks_find_arg(ctx, "--seed")) {
            // handlers are plain function pointers: the upstream one is kept in a static
            upstream_seed = seed->handler_string;
            seed->handler_string = [](common_params & params, const std::string & value) {
                test_seed_seen = true;
                upstream_seed(params, value);
            };
        }
    };
    h.sampler_init = [](const llama_model *, const common_params_sampling &, common_sampler *) { n_sampler_init++; };
    h.quant_ftypes = [](const common_hooks_add_ftype_fn & add) { add("TEST_HOOK_FTYPE", LLAMA_FTYPE_MOSTLY_Q8_0, "test type added by a hook"); };
    return h;
}());

static int n_fail = 0;

static void check(bool cond, const std::string & what) {
    printf("  %-70s %s\n", what.c_str(), cond ? "OK" : "FAIL");
    n_fail += !cond;
}

static int n_threads_seen = 0;

static std::vector<float> run(const std::vector<llama_token> & tokens) {
    std::vector<ggml_backend_dev_t> devs = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) };
    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_LLAMA, false);
    auto [model, lctx] = get_model_and_ctx(gguf_ctx.get(), nullptr, 42, 0.1f, devs);
    n_threads_seen = llama_n_threads(lctx.get());
    llama_memory_clear(llama_get_memory(lctx.get()), true);
    return get_logits(model.get(), lctx.get(), tokens);
}

// private member access without a friend line (llama-hooks-access.h)
LLAMA_HOOK_ACCESS(test_ctx_cparams, llama_context, cparams, llama_cparams);

int main() {
    // per-model rule, read once at the first lookup
    setenv("GGML_HOOKS", "llama:-test_rule_off", 1);

    common_log_set_verbosity_thold(LOG_LEVEL_ERROR);
    llama_backend_init();
    ggml_backend_load_all();

    init_normal_pool(42, 0.1f); // random weights of the model, as in test-llama-archs main()

    const std::vector<llama_token> tokens = get_tokens(16, 128, 42);

    llama_hooks_set_enabled("test_scale_logits", false);
    const std::vector<float> ref = run(tokens);
    const long passes_off = llama_hooks_hits("test_scale_logits");

    llama_hooks_set_enabled("test_scale_logits", true);
    const std::vector<float> scaled = run(tokens);
    const long passes_on = llama_hooks_hits("test_scale_logits");

    double max_err = 0.0;
    for (size_t i = 0; i < ref.size() && i < scaled.size(); ++i) {
        max_err = std::max(max_err, std::fabs(2.0 * ref[i] - scaled[i]));
    }

    printf("llama hooks\n");
    check(n_model_hook == 2, "model hook created both models (" + std::to_string(n_model_hook) + ")");
    check(passes_off == 0, "graph pass off: not run");
    check(passes_on > 0, "graph pass on: run " + std::to_string(passes_on) + " times");
    check(!ref.empty() && ref.size() == scaled.size() && max_err < 1e-3, "graph pass on: logits x 2 (max err " + std::to_string(max_err) + ")");
    check(n_other_pass == 0, "graph pass for another arch: not run");

    check(n_pass_with_ctx > 0, "graph pass finds its context while llama_process runs (" + std::to_string(n_pass_with_ctx) + ")");
    check(n_tensors > 0, "tensor_created: " + std::to_string(n_tensors) + " tensors");
    check(n_memory == 2, "create_memory asked for each context (" + std::to_string(n_memory) + ")");
    check(n_ctx_params == 2 && n_threads_seen == 3, "context_params changed n_threads to 3 (" + std::to_string(n_threads_seen) + ")");
    check(n_ctx_init == 2 && n_ctx_free == 2 && n_model_free == 2, "context_init, context_free, model_free: 2 each");
    check(n_begin > 0 && n_begin == n_end, "process_begin and process_end around each llama_process (" + std::to_string(n_begin) + ")");
    check(n_mem_clear == 2, "memory_op sees llama_memory_clear (" + std::to_string(n_mem_clear) + ")");
    {
        llama_model_params mp = llama_model_default_params();
        (void) mp;
        std::vector<ggml_backend_dev_t> devs = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) };
        gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_LLAMA, false);
        auto [model, lctx] = get_model_and_ctx(gguf_ctx.get(), nullptr, 42, 0.1f, devs);
        const llama_cparams & cp = llama_hook_get(*lctx, test_ctx_cparams{});
        check(cp.n_threads == 3, "private member read with LLAMA_HOOK_ACCESS (cparams.n_threads = " + std::to_string(cp.n_threads) + ")");
    }

    printf("llama hook points\n");
    check(n_kv[LLAMA_HOOK_KV_CLEAR] >= 2 && n_kv[LLAMA_HOOK_KV_APPLY_UBATCH] > 0,
          "kv events: clear " + std::to_string(n_kv[LLAMA_HOOK_KV_CLEAR]) + ", apply_ubatch " + std::to_string(n_kv[LLAMA_HOOK_KV_APPLY_UBATCH]));
    check(n_sched_reserve > 0 && n_fused > 0 && n_extra_nodes > 0, "sched_reserve, resolve_fused_ops, graph_extra_nodes");
    check(n_ubatch_done > 0, "ubatch_done (" + std::to_string(n_ubatch_done) + ")");
    check(LLM_TN(LLM_ARCH_LLAMA)(TEST_HOOK_TENSOR, 3).str() == "blk.3.hook_tensor" && llm_tensor_info_for(TEST_HOOK_TENSOR).op == GGML_OP_MUL_MAT,
          "tensor name and info of a tensor id upstream does not know");
    {
        std::vector<ggml_backend_dev_t> devs = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) };
        gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_LLAMA, false);
        auto [model, lctx] = get_model_and_ctx(gguf_ctx.get(), nullptr, 42, 0.1f, devs);
        const char * path   = "/tmp/test-llama-hooks.gguf";
        const char * path_q = "/tmp/test-llama-hooks-q8.gguf";
        llama_model_save_to_file(model.get(), path);
        llama_model_params mp = llama_model_default_params();
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        llama_model * m2 = llama_model_load_from_file(path, mp);
        check(m2 && n_mmap > 0, "mmap_ranges when a model is loaded with mmap (" + std::to_string(n_mmap) + ")");
        llama_model_free(m2);
        llama_model_quantize_params qp = llama_model_quantize_default_params();
        qp.ftype = LLAMA_FTYPE_MOSTLY_Q8_0;
        const uint32_t qres = llama_model_quantize(path, path_q, &qp);
        check(qres == 0 && n_quant > 0, "quant_type for each quantized tensor (" + std::to_string(n_quant) + ")");
        remove(path);
        remove(path_q);
    }
    {
        int n_added = 0;
        common_hooks_quant_ftypes([&](const char * name, llama_ftype, const char *) { n_added += std::string(name) == "TEST_HOOK_FTYPE"; });
        check(n_added == 1, "quant_ftypes adds a quantization type");
    }

    printf("ggml hooks with the model tag\n");
    check(n_seen > 0 && seen_arch == "llama", "kernel sees the model arch (\"" + seen_arch + "\", " + std::to_string(n_seen) + " lookups)");
    check(n_filtered == 0, "kernel filtered to other models: never asked");
    check(n_rule_off == 0, "kernel off for llama by GGML_HOOKS=llama:-test_rule_off: never asked");
    check(std::string(ggml_hook_model_arch()).empty(), "model tag cleared after the calls");

    printf("common hooks\n");
    {
        common_params params;
        const char * argv[] = { "test", "-m", "unused.gguf", "--test-hook-arg", "7", "--seed", "5" }; // upstream requires a model
        const bool ok = common_params_parse(7, (char **) argv, params, LLAMA_EXAMPLE_COMMON);
        check(ok && test_arg_value == 7, "CLI argument added by a hook is parsed (" + std::to_string(test_arg_value) + ")");
        check(ok && test_seed_seen && params.sampling.seed == 5, "upstream argument changed by a hook still works (--seed " + std::to_string(params.sampling.seed) + ")");
    }
    {
        std::vector<ggml_backend_dev_t> devs = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) };
        gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_LLAMA, false);
        auto [model, lctx] = get_model_and_ctx(gguf_ctx.get(), nullptr, 42, 0.1f, devs);
        common_params_sampling sp;
        common_sampler * smpl = common_sampler_init(model.get(), sp);
        check(smpl && n_sampler_init == 1, "sampler_init after common_sampler_init (" + std::to_string(n_sampler_init) + ")");
        common_sampler_free(smpl);
    }

    printf("%s\n", n_fail == 0 ? "all tests passed" : "some tests FAILED");
    return n_fail == 0 ? 0 : 1;
}
