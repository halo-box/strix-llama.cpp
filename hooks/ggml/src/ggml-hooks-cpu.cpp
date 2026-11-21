// CPU driver for the backend wrapper (see ggml-hooks-impl.h). Built in every CPU variant.
// Op kernels run from an extra buffer type of the CPU backend: upstream asks every extra buffer type for every node
// before its own kernel (ggml_cpu_extra_compute_forward). The buffer type is private: devices do not report it.

#include "ggml-hooks-cpu.h"
#include "ggml-hooks-impl.h"

#include "ggml-cpu.h"
#include "traits.h"

namespace {

constexpr ggml_hook_dev cpu_dev = { GGML_HOOK_BACKEND_CPU, 0, 0, 0, "CPU", false };

class hook_traits : public ggml::cpu::tensor_traits {
    bool work_size(int n_threads, const ggml_tensor * op, size_t & size) override {
        const ggml_hook_kernel * k = ggml_hook_find_op(&cpu_dev, op);
        if (!k || !k->cpu_work_size) {
            return false; // the op keeps the work size of the upstream kernel
        }
        const size_t n = k->cpu_work_size(n_threads, op);
        if (n == 0) {
            return false;
        }
        size = n;
        return true;
    }

    bool compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        const ggml_hook_kernel * k = ggml_hook_find_op(&cpu_dev, op);
        if (!k) {
            return false;
        }
        if (params->ith == 0) {
            ggml_hook_kernel_used(k, op);
        }
        k->compute.cpu_op(params, op);
        return true;
    }
};

class hook_extra_buffer_type : public ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const ggml_tensor *) override {
        return false; // asked only for nodes with a src in this buffer type, and no tensor is allocated in it
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const ggml_tensor * op) override {
        static hook_traits traits;
        if (!ggml_hook_has_kernels(GGML_HOOK_BACKEND_CPU) || !ggml_hook_find_op(&cpu_dev, op)) {
            return nullptr;
        }
        return &traits;
    }
};

const char * buft_get_name(ggml_backend_buffer_type_t) {
    return "CPU_HOOKS";
}

ggml_backend_buffer_type_t hook_buft() {
    static hook_extra_buffer_type ctx;
    static ggml_backend_buffer_type buft = [] {
        ggml_backend_buffer_type b = *ggml_backend_cpu_buffer_type();
        b.iface.get_name = buft_get_name;
        b.context        = &ctx;
        return b;
    }();
    return &buft;
}

ggml_backend_buffer_type_t * private_bufts() {
    static ggml_backend_buffer_type_t list[2] = { hook_buft(), nullptr };
    return list;
}

void cpu_dev_info(ggml_backend_dev_t, ggml_hook_dev * info) {
    *info = cpu_dev;
}

void cpu_installed(ggml_backend_reg_t) {
    ggml_backend_cpu_get_extra_buffer_types().push_back(hook_buft());
}

constexpr ggml_hook_driver driver = {
    /* .backend       = */ GGML_HOOK_BACKEND_CPU,
    /* .reg_name      = */ "CPU",
    /* .reg           = */ ggml_backend_cpu_reg,
    /* .dev_info      = */ cpu_dev_info,
    /* .cut_ops       = */ false,
    /* .begin         = */ nullptr,
    /* .run_op        = */ nullptr,
    /* .run_fusion    = */ nullptr,
    /* .end           = */ nullptr,
    /* .installed     = */ cpu_installed,
    /* .private_bufts = */ private_bufts,
};

GGML_HOOK_REGISTER_DRIVER(driver);

} // namespace
