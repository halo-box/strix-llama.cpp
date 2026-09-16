#include "build-info.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "host-attestation.h"
#include "dsv41-runtime-receipt.h"
extern "C" {
#include "hash/sha256/sha256.h"
}
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void dsv41_llama_common_runtime_anchor();

#if defined(__linux__)
#include <dlfcn.h>
#include <link.h>
#include <sys/stat.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;

#if defined(__linux__)
static constexpr const char * BUILD_REVISION = DSV41_BUILD_REVISION;
#endif

static std::string sha256_hex(const unsigned char digest[SHA256_DIGEST_SIZE]) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (size_t i = 0; i < SHA256_DIGEST_SIZE; ++i) {
        stream << std::setw(2) << static_cast<unsigned>(digest[i]);
    }
    return stream.str();
}

static std::string sha256_file(const fs::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open for SHA-256: " + path.string());
    }
    sha256_t state;
    sha256_init(&state);
    std::vector<unsigned char> buffer(1024 * 1024);
    while (input) {
        input.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            sha256_update(&state, buffer.data(), static_cast<size_t>(count));
        }
    }
    if (!input.eof()) {
        throw std::runtime_error("failed while hashing: " + path.string());
    }
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&state, digest);
    return sha256_hex(digest);
}

static fs::path canonical_path(const fs::path & path, const char * label) {
    try {
        return fs::canonical(path);
    } catch (const fs::filesystem_error & error) {
        throw std::runtime_error(std::string("cannot resolve ") + label + ": " + error.what());
    }
}

#if defined(__linux__)
static fs::path current_executable_path() {
    return canonical_path("/proc/self/exe", "current executable");
}

template <typename T>
static const void * function_address(T function) {
    return reinterpret_cast<const void *>(reinterpret_cast<uintptr_t>(function));
}

static fs::path module_path(const void * address) {
    Dl_info info = {};
    if (dladdr(address, &info) == 0 || info.dli_fname == nullptr || info.dli_fname[0] == '\0') {
        throw std::runtime_error("cannot identify loaded runtime module");
    }
    return canonical_path(info.dli_fname, "loaded runtime module");
}

static const void * llama_common_runtime_address() {
    dlerror();
    void * address = dlsym(RTLD_DEFAULT, "dsv41_llama_common_runtime_anchor");
    const char * error = dlerror();
    if (error != nullptr || address == nullptr) {
        throw std::runtime_error("cannot resolve llama-common runtime anchor");
    }
    return address;
}

static bool is_project_runtime_library(const fs::path & path) {
    std::string name = path.filename().string();
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return name.rfind("libllama", 0) == 0 || name.rfind("libggml", 0) == 0 ||
        name.rfind("ggml", 0) == 0;
}

static bool is_trusted_system_runtime_path(const fs::path & path) {
    const std::string value = path.string();
    std::error_code error;
    const fs::path rocm_root = fs::canonical("/opt/rocm", error);
    if (!error) {
        const std::string prefix = rocm_root.string() + "/";
        if (value.rfind(prefix, 0) == 0) {
            for (fs::path current = path; current != rocm_root.parent_path(); current = current.parent_path()) {
                struct stat status = {};
                if (::stat(current.c_str(), &status) != 0 || status.st_uid != 0 ||
                        (status.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
                    return false;
                }
                if (current == rocm_root) {
                    return true;
                }
            }
        }
    }
    return value.rfind("/lib/", 0) == 0 ||
        value.rfind("/lib64/", 0) == 0 ||
        value.rfind("/usr/lib/", 0) == 0 ||
        value.rfind("/usr/lib64/", 0) == 0;
}

static std::set<fs::path> loaded_runtime_images(const fs::path & executable) {
    std::set<fs::path> result;
    struct image_context {
        const fs::path * executable;
        std::set<fs::path> * result;
        std::string error;
    } context = {&executable, &result, {}};
    const auto callback = [](dl_phdr_info * info, size_t, void * data) {
        image_context & context = *static_cast<image_context *>(data);
        if (!context.error.empty() || info->dlpi_name == nullptr || info->dlpi_name[0] == '\0') {
            return 0;
        }
        try {
            const fs::path reported_path = info->dlpi_name;
            if (!reported_path.is_absolute() &&
                    (reported_path == "linux-vdso.so.1" || reported_path == "linux-gate.so.1")) {
                return 0;
            }
            const fs::path path = canonical_path(reported_path, "loaded runtime module");
            if (path != *context.executable) {
                context.result->insert(path);
            }
        } catch (const std::exception & error) {
            context.error = error.what();
        }
        return 0;
    };
    if (dl_iterate_phdr(callback, &context) < 0 || !context.error.empty()) {
        throw std::runtime_error(
            context.error.empty() ? "cannot enumerate loaded runtime modules" : context.error);
    }
    return result;
}

static fs::path runtime_library_directory(const fs::path & executable) {
    if (std::string(dsv41_runtime_receipt::profile) != "sibling-lib") {
        throw std::runtime_error("prompt builder requires the sibling-lib runtime profile");
    }
    return executable.parent_path().parent_path() / "lib";
}

static void load_runtime_backends(const fs::path & executable) {
#if defined(GGML_BACKEND_DL)
    const std::string directory = runtime_library_directory(executable).string();
    ggml_backend_load_all_from_path(directory.c_str());
#else
    (void) executable;
    ggml_backend_load_all();
#endif
}

static json runtime_libraries_json(const fs::path & executable) {
    if (dsv41_runtime_receipt::entries.size() != dsv41_runtime_receipt::components.size()) {
        throw std::runtime_error("embedded runtime profile size differs from the receipt");
    }
    const fs::path library_directory =
        canonical_path(runtime_library_directory(executable), "runtime library directory");
    std::map<std::string, const dsv41_runtime_receipt::entry *> receipt_by_component;
    std::map<fs::path, const dsv41_runtime_receipt::entry *> receipt_by_path;
    for (size_t index = 0; index < dsv41_runtime_receipt::entries.size(); ++index) {
        const dsv41_runtime_receipt::entry & entry = dsv41_runtime_receipt::entries[index];
        if (std::string(entry.component) != dsv41_runtime_receipt::components[index]) {
            throw std::runtime_error("embedded runtime profile differs from the receipt");
        }
        const fs::path expected =
            canonical_path(library_directory / entry.filename, "receipt runtime module");
        if (!receipt_by_component.emplace(entry.component, &entry).second ||
                !receipt_by_path.emplace(expected, &entry).second) {
            throw std::runtime_error("embedded runtime receipt is not canonical and unique");
        }
    }
    std::set<fs::path> libraries;
    for (const fs::path & image : loaded_runtime_images(executable)) {
        if (image.parent_path() == library_directory || is_project_runtime_library(image)) {
            libraries.insert(image);
        } else if (!is_trusted_system_runtime_path(image)) {
            throw std::runtime_error("loaded unclassified module outside trusted system roots: " + image.string());
        }
    }
    std::map<std::string, fs::path> loaded_by_component;
    for (const fs::path & library : libraries) {
        const auto receipt = receipt_by_path.find(library);
        if (library.parent_path() != library_directory || receipt == receipt_by_path.end()) {
            throw std::runtime_error("loaded project runtime module is absent from the receipt: " + library.string());
        }
        if (!loaded_by_component.emplace(receipt->second->component, library).second ||
                sha256_file(library) != receipt->second->sha256) {
            throw std::runtime_error("loaded runtime module identity differs from the receipt");
        }
    }
    if (loaded_by_component.size() != receipt_by_component.size()) {
        throw std::runtime_error("loaded runtime component set differs from the receipt");
    }
    const std::array<std::pair<const char *, fs::path>, 3> fixed_roles = {{
        {"llama-common", module_path(llama_common_runtime_address())},
        {"llama", module_path(function_address(&llama_model_load_from_file))},
        {"ggml-base", module_path(function_address(&ggml_init))},
    }};
    for (const auto & item : fixed_roles) {
        const auto loaded = loaded_by_component.find(item.first);
        if (loaded == loaded_by_component.end() || loaded->second != item.second) {
            throw std::runtime_error(
                "runtime symbol provider does not match receipt component " + std::string(item.first));
        }
    }
    json result = json::array();
    for (const fs::path & library : libraries) {
        const dsv41_runtime_receipt::entry & receipt = *receipt_by_path.at(library);
        std::string role = "runtime:" + std::string(receipt.component);
        if (library == fixed_roles[0].second) {
            role = "build-info";
        } else if (library == fixed_roles[1].second) {
            role = "llama";
        } else if (library == fixed_roles[2].second) {
            role = "ggml";
        }
        result.push_back({
            {"component", receipt.component},
            {"filename", receipt.filename},
            {"path", library.string()},
            {"sha256", receipt.sha256},
            {"role", std::move(role)},
            {"revision", receipt.revision[0] == '\0' ? json(nullptr) : json(receipt.revision)},
        });
    }
    return result;
}

static json runtime_build_json(const fs::path & executable) {
    if (std::string(BUILD_REVISION) != llama_commit() || std::string(BUILD_REVISION) != ggml_commit()) {
        throw std::runtime_error("loaded runtime library revision differs from the prompt builder revision");
    }
    const json runtime_libraries = runtime_libraries_json(executable);
    return {
        {"revision", BUILD_REVISION},
        {"path", executable.string()},
        {"sha256", sha256_file(executable)},
        {"runtime_profile", {
            {"name", dsv41_runtime_receipt::profile},
            {"components", dsv41_runtime_receipt::components},
            {"selected_backend_component", "ggml-hip"},
        }},
        {"runtime_receipt_sha256", dsv41_runtime_receipt::sha256},
        {"runtime_libraries", runtime_libraries},
        {"runtime_libraries_post", runtime_libraries},
    };
}
#endif

static std::string read_file(const fs::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

static std::string argument(int argc, char ** argv, const std::string & name) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == name) {
            return argv[index + 1];
        }
    }
    throw std::runtime_error("missing argument: " + name);
}

static bool boolean_argument(int argc, char ** argv, const std::string & name) {
    const std::string value = argument(argc, argv, name);
    if (value == "true") {
        return true;
    }
    if (value == "false") {
        return false;
    }
    throw std::runtime_error(name + " must be true or false");
}

static std::string model_architecture(const llama_model * model) {
    char buffer[128] = {};
    if (llama_model_meta_val_str(model, "general.architecture", buffer, sizeof(buffer)) < 0) {
        throw std::runtime_error("model has no general.architecture metadata");
    }
    return buffer;
}

int main(int argc, char ** argv) {
    try {
#if !defined(__linux__)
        (void) argc;
        (void) argv;
        throw std::runtime_error("prompt builder runtime attestation requires Linux");
#else
        if (argc == 2 && std::string(argv[1]) == "--dsv41-attest-build") {
            const fs::path executable_path = current_executable_path();
            const fs::path invoked_path = canonical_path(fs::absolute(argv[0]), "invoked prompt builder");
            if (invoked_path != executable_path) {
                throw std::runtime_error("invoked prompt builder path does not match the running executable");
            }
            llama_backend_init();
            load_runtime_backends(executable_path);
            std::printf("%s\n", runtime_build_json(executable_path).dump().c_str());
            return 0;
        }
        fs::path model_path = argument(argc, argv, "--model");
        fs::path corpus_path = argument(argc, argv, "--corpus");
        fs::path output_path = argument(argc, argv, "--output");
        const int64_t target_tokens = std::stoll(argument(argc, argv, "--tokens"));
        const bool expected_add_bos = boolean_argument(argc, argv, "--tokenizer-add-bos");
        const bool parse_special = boolean_argument(argc, argv, "--tokenizer-parse-special");
        const bool detokenize_special = boolean_argument(argc, argv, "--tokenizer-detokenize-special");
        const bool remove_leading_bos = boolean_argument(argc, argv, "--tokenizer-remove-leading-bos");
        const bool require_round_trip = boolean_argument(argc, argv, "--tokenizer-require-round-trip");
        if (target_tokens < 2) {
            throw std::runtime_error("--tokens must be at least 2");
        }
        if (!parse_special || !detokenize_special || !require_round_trip ||
                remove_leading_bos != expected_add_bos) {
            throw std::runtime_error("prompt builder requires the exact approved tokenizer policy");
        }
        const fs::path executable_path = current_executable_path();
        const fs::path invoked_path = canonical_path(fs::absolute(argv[0]), "invoked prompt builder");
        if (invoked_path != executable_path) {
            throw std::runtime_error("invoked prompt builder path does not match the running executable");
        }
        llama_backend_init();
        load_runtime_backends(executable_path);
        const json runtime_build = runtime_build_json(executable_path);
        model_path = dsv41::require_nvme_path(model_path, "model").resolved_path;
        corpus_path = dsv41::require_nvme_path(corpus_path, "corpus").resolved_path;
        output_path = dsv41::require_nvme_path(output_path, "prompt output").resolved_path;
        const char * tmpdir_value = std::getenv("TMPDIR");
        if (tmpdir_value == nullptr || *tmpdir_value == '\0') {
            throw std::runtime_error("TMPDIR is required");
        }
        dsv41::require_usable_directory(tmpdir_value, "TMPDIR");
        const dsv41::storage_attestation temporary_storage =
            dsv41::require_nvme_path(tmpdir_value, "temporary directory");
        if (fs::exists(output_path)) {
            throw std::runtime_error("prompt output already exists: " + output_path.string());
        }

        const std::string corpus = read_file(corpus_path);
        if (corpus.empty()) {
            throw std::runtime_error("corpus is empty");
        }

        llama_model_params model_params = llama_model_default_params();
        model_params.vocab_only = true;
        llama_model * model = llama_model_load_from_file(model_path.string().c_str(), model_params);
        if (model == nullptr) {
            throw std::runtime_error("cannot load model vocabulary");
        }
        if (model_architecture(model) != "deepseek41") {
            llama_model_free(model);
            throw std::runtime_error("prompt builder requires general.architecture=deepseek41");
        }
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const bool add_bos = llama_vocab_get_add_bos(vocab);
        if (add_bos != expected_add_bos) {
            llama_model_free(model);
            throw std::runtime_error("model tokenizer add_bos differs from the approved policy");
        }

        std::string repeated = corpus;
        std::vector<llama_token> tokens = common_tokenize(vocab, repeated, add_bos, parse_special);
        while (tokens.size() < static_cast<size_t>(target_tokens)) {
            if (repeated.size() > (size_t(1) << 31)) {
                llama_model_free(model);
                throw std::runtime_error("repeated prompt exceeds 2 GiB");
            }
            repeated += repeated;
            tokens = common_tokenize(vocab, repeated, add_bos, parse_special);
        }
        tokens.resize(static_cast<size_t>(target_tokens));
        std::vector<llama_token> content_tokens = tokens;
        if (remove_leading_bos) {
            if (content_tokens.front() != llama_vocab_bos(vocab)) {
                llama_model_free(model);
                throw std::runtime_error("tokenized prompt does not begin with the configured BOS token");
            }
            content_tokens.erase(content_tokens.begin());
        }
        const std::string prompt = common_detokenize(vocab, content_tokens, detokenize_special);
        const std::vector<llama_token> verified = common_tokenize(vocab, prompt, add_bos, parse_special);
        if (require_round_trip && verified != tokens) {
            llama_model_free(model);
            throw std::runtime_error("constructed prompt does not round-trip to the target token IDs");
        }

        if (!output_path.parent_path().empty()) {
            fs::create_directories(output_path.parent_path());
        }
        std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
        if (!output || !output.write(prompt.data(), static_cast<std::streamsize>(prompt.size()))) {
            llama_model_free(model);
            throw std::runtime_error("cannot write prompt output");
        }
        output.close();
        const json runtime_build_post = runtime_build_json(executable_path);
        if (runtime_build_post != runtime_build) {
            llama_model_free(model);
            throw std::runtime_error("loaded runtime component set changed during prompt construction");
        }
        std::printf("%s\n", json({
            {"target_tokens", target_tokens},
            {"actual_tokens", verified.size()},
            {"byte_count", prompt.size()},
            {"tokenizer", {
                {"add_bos", expected_add_bos},
                {"parse_special", parse_special},
                {"detokenize_special", detokenize_special},
                {"remove_leading_bos_before_detokenize", remove_leading_bos},
                {"require_round_trip", require_round_trip},
            }},
            {"runtime_build", runtime_build_post},
            {"temporary_directory", temporary_storage.resolved_path.string()},
        }).dump().c_str());
        llama_model_free(model);
        return 0;
#endif
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
