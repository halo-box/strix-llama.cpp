// hooks: tools/mtmd/deprecation-warning.cpp of the fork (403dfcb03, upstream 9048bad8d, unchanged in the fork), verbatim inside
// namespace ggml_fork. Adapted lines carry "hooks:".
struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; struct ggml_backend_event; // hooks: global
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <string>
namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/apps/README.md

int main(int argc, char** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string filename = "main";
    if (argc >= 1) {
        filename = argv[0];
    }

    // Get only the program name from the full path
    size_t pos = filename.find_last_of("/\\");
    if (pos != std::string::npos) {
        filename = filename.substr(pos+1);
    }

    fprintf(stdout, "\n");
    fprintf(stdout, "WARNING: The binary '%s' is deprecated.\n", filename.c_str());
    fprintf(stdout, "Please use 'llama-mtmd-cli' instead.\n");
    fprintf(stdout, "\n");

    return EXIT_FAILURE;
}
} // namespace ggml_fork
