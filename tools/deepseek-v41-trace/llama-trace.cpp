#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
extern "C" {
#include "hash/sha256/sha256.h"
}
#include "llama.h"
#include "llama-ext.h"
#include "host-attestation.h"
#include "trace-components.h"
#include "dsv41-runtime-receipt.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

extern "C" void dsv41_llama_common_runtime_anchor();

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <tlhelp32.h>
#include <windows.h>
#else
#include <dlfcn.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#if defined(__linux__)
#include <fcntl.h>
#include <link.h>
#include <poll.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif
#endif

#if !defined(_WIN32)
#include <sys/utsname.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

static constexpr int TRACE_VERSION = 2;
static constexpr const char * BUILD_REVISION = DSV41_BUILD_REVISION;
#if defined(__linux__)
static constexpr const char * WATCHDOG_SCRIPT_SHA256 =
    "799a689198eb873085c98178c62c30cf664b28e59ec6466af0bd89ff8fce68f5";
#endif

static std::string sha256_hex(const unsigned char digest[SHA256_DIGEST_SIZE]) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (size_t i = 0; i < SHA256_DIGEST_SIZE; ++i) {
        stream << std::setw(2) << static_cast<unsigned>(digest[i]);
    }
    return stream.str();
}

static std::string sha256_data(const void * data, size_t size) {
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_hash(digest, static_cast<const unsigned char *>(data), size);
    return sha256_hex(digest);
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

static std::vector<uint8_t> read_file(const fs::path & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open: " + path.string());
    }
    const std::streamsize size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("cannot determine file size: " + path.string());
    }
    std::vector<uint8_t> result(static_cast<size_t>(size));
    input.seekg(0);
    if (size != 0 && !input.read(reinterpret_cast<char *>(result.data()), size)) {
        throw std::runtime_error("cannot read: " + path.string());
    }
    return result;
}

static fs::path canonical_path(const fs::path & path, const char * label) {
    try {
        return fs::canonical(path);
    } catch (const fs::filesystem_error & error) {
        throw std::runtime_error(std::string("cannot resolve ") + label + ": " + error.what());
    }
}

static fs::path current_executable_path() {
#if defined(_WIN32)
    std::vector<wchar_t> buffer(32768);
    const DWORD size = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (size == 0 || size >= buffer.size()) {
        throw std::runtime_error("cannot query current executable path");
    }
    return canonical_path(fs::path(std::wstring(buffer.data(), size)), "current executable");
#elif defined(__APPLE__)
    uint32_t size = 0;
    if (_NSGetExecutablePath(nullptr, &size) != -1 || size == 0) {
        throw std::runtime_error("cannot query current executable path size");
    }
    std::vector<char> buffer(size);
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
        throw std::runtime_error("cannot query current executable path");
    }
    return canonical_path(buffer.data(), "current executable");
#elif defined(__linux__)
    return canonical_path("/proc/self/exe", "current executable");
#else
#error unsupported platform
#endif
}

static fs::path module_path(const void * address) {
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(address),
            &module)) {
        throw std::runtime_error("cannot identify loaded runtime module");
    }
    std::vector<wchar_t> buffer(32768);
    const DWORD size = GetModuleFileNameW(module, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (size == 0 || size >= buffer.size()) {
        throw std::runtime_error("cannot query loaded runtime module path");
    }
    return canonical_path(fs::path(std::wstring(buffer.data(), size)), "loaded runtime module");
#else
    Dl_info info = {};
    if (dladdr(address, &info) == 0 || info.dli_fname == nullptr || info.dli_fname[0] == '\0') {
        throw std::runtime_error("cannot identify loaded runtime module");
    }
    return canonical_path(info.dli_fname, "loaded runtime module");
#endif
}

template <typename T>
static const void * function_address(T function) {
    return reinterpret_cast<const void *>(reinterpret_cast<uintptr_t>(function));
}

static const void * llama_common_runtime_address() {
#if defined(_WIN32)
    return function_address(&dsv41_llama_common_runtime_anchor);
#else
    dlerror();
    void * address = dlsym(RTLD_DEFAULT, "dsv41_llama_common_runtime_anchor");
    const char * error = dlerror();
    if (error != nullptr || address == nullptr) {
        throw std::runtime_error("cannot resolve llama-common runtime anchor");
    }
    return address;
#endif
}

static bool is_project_runtime_library(const fs::path & path) {
    std::string name = path.filename().string();
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
#if defined(_WIN32)
    return name.rfind("llama", 0) == 0 || name.rfind("ggml", 0) == 0 ||
        name.rfind("libllama", 0) == 0 || name.rfind("libggml", 0) == 0;
#else
    return name.rfind("libllama", 0) == 0 || name.rfind("libggml", 0) == 0 ||
        name.rfind("ggml", 0) == 0;
#endif
}

static fs::path loaded_image_path(const fs::path & path) {
    if (fs::exists(path)) {
        return canonical_path(path, "loaded runtime module");
    }
    if (!path.is_absolute()) {
        throw std::runtime_error("loaded runtime module path is not absolute: " + path.string());
    }
    return path.lexically_normal();
}

static bool is_trusted_system_runtime_path(const fs::path & path) {
#if defined(_WIN32)
    std::vector<wchar_t> buffer(32768);
    const UINT size = GetSystemDirectoryW(buffer.data(), static_cast<UINT>(buffer.size()));
    if (size == 0 || size >= buffer.size()) {
        throw std::runtime_error("cannot query the Windows system runtime directory");
    }
    return path.parent_path() == canonical_path(
        fs::path(std::wstring(buffer.data(), size)), "Windows system runtime directory");
#elif defined(__APPLE__)
    return path.string().rfind("/usr/lib/", 0) == 0 ||
        path.string().rfind("/System/Library/", 0) == 0;
#elif defined(__linux__)
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
#else
    (void) path;
    return false;
#endif
}

static fs::path runtime_library_directory(const fs::path & executable);

#if defined(__APPLE__)
static std::array<const mach_header *, 256> protected_interval_images = {};
static std::atomic<size_t> protected_interval_image_count = 0;
static std::atomic<bool> protected_interval_active = false;

static void record_loaded_image(const mach_header * header, intptr_t) {
    if (!protected_interval_active.load(std::memory_order_relaxed)) {
        return;
    }
    const size_t index = protected_interval_image_count.fetch_add(1, std::memory_order_relaxed);
    if (index < protected_interval_images.size()) {
        protected_interval_images[index] = header;
    }
}

static void begin_loader_monitor() {
    static const bool registered = []() {
        _dyld_register_func_for_add_image(record_loaded_image);
        return true;
    }();
    (void) registered;
    protected_interval_image_count.store(0, std::memory_order_relaxed);
    protected_interval_active.store(true, std::memory_order_release);
}

static void end_loader_monitor(const fs::path & executable) {
    protected_interval_active.store(false, std::memory_order_release);
    const size_t count = protected_interval_image_count.load(std::memory_order_relaxed);
    if (count > protected_interval_images.size()) {
        throw std::runtime_error("too many loader image additions during protected trace generation");
    }
    const fs::path library_directory =
        canonical_path(runtime_library_directory(executable), "runtime library directory");
    for (size_t index = 0; index < count; ++index) {
        Dl_info info = {};
        if (dladdr(protected_interval_images[index], &info) == 0 ||
                info.dli_fname == nullptr || info.dli_fname[0] == '\0') {
            throw std::runtime_error("cannot identify loader image added during protected trace generation");
        }
        const fs::path image = loaded_image_path(info.dli_fname);
        if (image.parent_path() == executable.parent_path() ||
                image.parent_path() == library_directory ||
                is_project_runtime_library(image) ||
                !is_trusted_system_runtime_path(image)) {
            throw std::runtime_error(
                "runtime module was added during protected trace generation: " + image.string());
        }
    }
}
#else
static void begin_loader_monitor() {
}

static void end_loader_monitor(const fs::path &) {
}
#endif

static std::set<fs::path> loaded_runtime_images(const fs::path & executable) {
    std::set<fs::path> result;
#if defined(_WIN32)
    const HANDLE snapshot = CreateToolhelp32Snapshot(
        TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32,
        GetCurrentProcessId());
    if (snapshot == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("cannot enumerate loaded runtime modules");
    }
    MODULEENTRY32W entry = {};
    entry.dwSize = sizeof(entry);
    if (!Module32FirstW(snapshot, &entry)) {
        CloseHandle(snapshot);
        throw std::runtime_error("cannot read loaded runtime modules");
    }
    do {
        const fs::path reported_path = entry.szExePath;
        const fs::path path = loaded_image_path(reported_path);
        if (path != executable) {
            result.insert(path);
        }
    } while (Module32NextW(snapshot, &entry));
    CloseHandle(snapshot);
#elif defined(__APPLE__)
    const uint32_t count = _dyld_image_count();
    for (uint32_t index = 0; index < count; ++index) {
        const char * name = _dyld_get_image_name(index);
        if (name != nullptr && name[0] != '\0') {
            const fs::path reported_path = name;
            const fs::path path = loaded_image_path(reported_path);
            if (path != executable) {
                result.insert(path);
            }
        }
    }
#elif defined(__linux__)
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
            const fs::path path = loaded_image_path(reported_path);
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
#endif
    return result;
}

static bool is_lower_hex(const std::string & value, size_t length) {
    return value.size() == length && std::all_of(value.begin(), value.end(), [](unsigned char character) {
        return std::isdigit(character) || (character >= 'a' && character <= 'f');
    });
}

static fs::path runtime_library_directory(const fs::path & executable) {
    const std::string profile = dsv41_runtime_receipt::profile;
    if (profile == "co-located") {
        return executable.parent_path();
    }
    if (profile == "sibling-lib") {
        return executable.parent_path().parent_path() / "lib";
    }
    throw std::runtime_error("embedded runtime profile is invalid");
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

static void reject_loader_overrides() {
    static const std::array<const char *, 13> names = {
        "DYLD_FALLBACK_FRAMEWORK_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_FRAMEWORK_PATH",
        "DYLD_IMAGE_SUFFIX",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "DYLD_ROOT_PATH",
        "DYLD_VERSIONED_FRAMEWORK_PATH",
        "DYLD_VERSIONED_LIBRARY_PATH",
        "GGML_BACKEND_PATH",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
    };
    for (const char * name : names) {
        const char * value = std::getenv(name);
        if (value != nullptr && value[0] != '\0') {
            throw std::runtime_error(std::string("production trace execution forbids loader override: ") + name);
        }
    }
}

static json runtime_libraries_json(
        const fs::path & executable,
        ggml_backend_dev_t selected_device,
        const std::string & revision,
        const std::string & selected_backend_component) {
    if (selected_device == nullptr) {
        throw std::runtime_error("cannot bind a null selected backend device");
    }
    ggml_backend_reg_t selected_backend = ggml_backend_dev_backend_reg(selected_device);
    if (selected_backend == nullptr) {
        throw std::runtime_error("selected backend device has no runtime registry");
    }
    if (dsv41_runtime_receipt::entries.size() != dsv41_runtime_receipt::components.size()) {
        throw std::runtime_error("embedded runtime profile size differs from the receipt");
    }

    const fs::path library_directory =
        canonical_path(runtime_library_directory(executable), "runtime library directory");
    std::map<std::string, const dsv41_runtime_receipt::entry *> receipt_by_component;
    std::map<fs::path, const dsv41_runtime_receipt::entry *> receipt_by_path;
    std::set<std::string> receipt_filenames;
    std::set<std::string> receipt_digests;
    for (size_t index = 0; index < dsv41_runtime_receipt::entries.size(); ++index) {
        const dsv41_runtime_receipt::entry & entry = dsv41_runtime_receipt::entries[index];
        const std::string component = entry.component;
        const std::string filename = entry.filename;
        const std::string digest = entry.sha256;
        const std::string entry_revision = entry.revision;
        if (component.empty() || filename.empty() || fs::path(filename).filename() != filename ||
                !is_lower_hex(digest, 64) ||
                !receipt_by_component.emplace(component, &entry).second ||
                !receipt_filenames.insert(filename).second ||
                !receipt_digests.insert(digest).second) {
            throw std::runtime_error("embedded runtime receipt is not canonical and unique");
        }
        if (component != dsv41_runtime_receipt::components[index]) {
            throw std::runtime_error("embedded runtime profile differs from the receipt");
        }
        const bool revision_bearing = component == "llama-common" || component == "ggml-base";
        if ((revision_bearing && entry_revision != revision) ||
                (!revision_bearing && !entry_revision.empty()) ||
                (!entry_revision.empty() && !is_lower_hex(entry_revision, 40))) {
            throw std::runtime_error("embedded runtime receipt revision is invalid for " + component);
        }
        const fs::path expected_path =
            canonical_path(library_directory / filename, "receipt runtime module");
        if (!receipt_by_path.emplace(expected_path, &entry).second) {
            throw std::runtime_error("embedded runtime receipt path is duplicated");
        }
    }

    const fs::path build_info_module = module_path(llama_common_runtime_address());
    const fs::path llama_module = module_path(function_address(&llama_model_load_from_file));
    const fs::path ggml_module = module_path(function_address(&ggml_init));
    const fs::path selected_backend_module = module_path(selected_backend);
    const std::set<fs::path> images = loaded_runtime_images(executable);
    std::set<fs::path> libraries;
    const fs::path binary_directory = executable.parent_path();
    for (const fs::path & image : images) {
        const bool in_runtime_root =
            image.parent_path() == binary_directory || image.parent_path() == library_directory;
        if (in_runtime_root || is_project_runtime_library(image)) {
            libraries.insert(image);
        } else if (!is_trusted_system_runtime_path(image)) {
            throw std::runtime_error("loaded unclassified module outside trusted system roots: " + image.string());
        }
    }
    std::map<std::string, fs::path> loaded_by_component;
    for (const fs::path & library : libraries) {
        if (library.parent_path() != library_directory) {
            throw std::runtime_error(
                "loaded runtime module is outside the exporter runtime directory for the exact profile: " +
                library.string());
        }
        const auto receipt = receipt_by_path.find(library);
        if (receipt == receipt_by_path.end()) {
            throw std::runtime_error(
                "loaded project runtime module is absent from the receipt: " + library.string());
        }
        const std::string component = receipt->second->component;
        if (!loaded_by_component.emplace(component, library).second) {
            throw std::runtime_error("multiple loaded runtime modules map to receipt component " + component);
        }
        if (sha256_file(library) != receipt->second->sha256) {
            throw std::runtime_error("loaded runtime module SHA-256 differs from the receipt: " + component);
        }
    }
    if (loaded_by_component.size() != receipt_by_component.size()) {
        for (const auto & item : receipt_by_component) {
            if (loaded_by_component.count(item.first) == 0) {
                throw std::runtime_error("receipt runtime component is not loaded: " + item.first);
            }
        }
        throw std::runtime_error("loaded runtime component set differs from the receipt");
    }

    const std::array<std::pair<const char *, fs::path>, 3> fixed_roles = {{
        {"llama-common", build_info_module},
        {"llama", llama_module},
        {"ggml-base", ggml_module},
    }};
    for (const auto & item : fixed_roles) {
        const auto loaded = loaded_by_component.find(item.first);
        if (loaded == loaded_by_component.end() || loaded->second != item.second) {
            throw std::runtime_error(
                "runtime symbol provider does not match receipt component " + std::string(item.first));
        }
    }
    const auto selected = loaded_by_component.find(selected_backend_component);
    if (selected == loaded_by_component.end() || selected->second != selected_backend_module) {
        throw std::runtime_error(
            "selected backend module does not match receipt component " + selected_backend_component);
    }

    json result = json::array();
    for (const fs::path & library : libraries) {
        const dsv41_runtime_receipt::entry & receipt = *receipt_by_path.at(library);
        std::string role = "runtime:" + std::string(receipt.component);
        if (library == build_info_module) {
            role = "build-info";
        }
        if (library == llama_module) {
            role = "llama";
        }
        if (library == ggml_module) {
            role = "ggml";
        }
        if (library == selected_backend_module) {
            role = "selected-backend";
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

static json runtime_build_json(
        const fs::path & executable,
        ggml_backend_dev_t selected_device,
        const std::string & selected_backend_component,
        char ** argv) {
    const fs::path invoked_path = canonical_path(fs::absolute(argv[0]), "invoked exporter");
    if (invoked_path != executable) {
        throw std::runtime_error("invoked exporter path does not match the running executable");
    }
    const std::string revision = BUILD_REVISION;
    if (revision.size() != 40 || !std::all_of(revision.begin(), revision.end(), [](unsigned char value) {
            return std::isdigit(value) || (value >= 'a' && value <= 'f');
        })) {
        throw std::runtime_error("embedded exporter revision is invalid");
    }
    const std::string linked_revision = llama_commit();
    const std::string ggml_revision = ggml_commit();
    if (linked_revision != revision || ggml_revision != revision) {
        throw std::runtime_error(
            "loaded runtime library revision differs from the exporter revision: llama=" +
            linked_revision + ", ggml=" + ggml_revision + ", exporter=" + revision);
    }
#if defined(DSV41_MANIFEST_TEST_HARNESS)
    const std::string build_info = std::string(llama_build_info()) + " [test-only manifest harness]";
#else
    const std::string build_info = llama_build_info();
#endif
    return {
        {"number", llama_build_number()},
        {"info", build_info},
        {"compiler", llama_compiler()},
        {"target", llama_build_target()},
        {"path", executable.string()},
        {"sha256", sha256_file(executable)},
        {"runtime_profile", {
            {"name", dsv41_runtime_receipt::profile},
            {"components", dsv41_runtime_receipt::components},
            {"selected_backend_component", selected_backend_component},
        }},
        {"runtime_receipt_sha256", dsv41_runtime_receipt::sha256},
        {"runtime_libraries", runtime_libraries_json(
            executable, selected_device, revision, selected_backend_component)},
        {"runtime_module_monitor", {
#if defined(__APPLE__)
            {"mechanism", "dyld-add-image"},
#else
            {"mechanism", "pre-post-snapshot"},
#endif
            {"checked_after_trace", false},
            {"project_additions", json::array()},
        }},
    };
}

static std::string required_environment(const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        throw std::runtime_error(std::string("required environment variable is missing: ") + name);
    }
    return value;
}

#if defined(__linux__)
static constexpr uint64_t DSV41_GIB = UINT64_C(1024)*1024*1024;

static int required_descriptor(const char * name) {
    const std::string value = required_environment(name);
    size_t consumed = 0;
    const long parsed = std::stol(value, &consumed);
    if (consumed != value.size() || parsed <= STDERR_FILENO ||
            parsed > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("invalid descriptor environment: ") + name);
    }
    return static_cast<int>(parsed);
}

static std::string sha256_descriptor(int descriptor) {
    sha256_t state;
    sha256_init(&state);
    std::vector<unsigned char> buffer(8 * 1024 * 1024);
    off_t offset = 0;
    while (true) {
        ssize_t count;
        do {
            count = pread(descriptor, buffer.data(), buffer.size(), offset);
        } while (count < 0 && errno == EINTR);
        if (count < 0) {
            throw std::runtime_error("failed while hashing held model descriptor");
        }
        if (count == 0) {
            break;
        }
        sha256_update(&state, buffer.data(), static_cast<size_t>(count));
        offset += count;
    }
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&state, digest);
    return sha256_hex(digest);
}

static json model_descriptor_identity(int descriptor, const fs::path & model_path, bool hash_bytes) {
    struct stat status {};
    const int status_flags = fcntl(descriptor, F_GETFL);
    const int descriptor_flags = fcntl(descriptor, F_GETFD);
    if (fstat(descriptor, &status) != 0 || status_flags < 0 || descriptor_flags < 0) {
        throw std::runtime_error("cannot inspect held model descriptor");
    }
    if (!S_ISREG(status.st_mode) || status.st_nlink < 1 ||
            (status_flags & O_ACCMODE) != O_RDONLY || descriptor_flags != 0) {
        throw std::runtime_error("held model descriptor policy is invalid");
    }
    json result = {
        {"format", "dsv41-model-file-identity"},
        {"version", 1},
        {"path", model_path.string()},
        {"device", static_cast<uint64_t>(status.st_dev)},
        {"inode", static_cast<uint64_t>(status.st_ino)},
        {"owner_uid", static_cast<uint64_t>(status.st_uid)},
        {"owner_gid", static_cast<uint64_t>(status.st_gid)},
        {"mode", static_cast<uint64_t>(status.st_mode & 07777)},
        {"link_count", static_cast<uint64_t>(status.st_nlink)},
        {"byte_count", static_cast<uint64_t>(status.st_size)},
        {"modified_ns",
            static_cast<int64_t>(status.st_mtim.tv_sec)*INT64_C(1000000000) + status.st_mtim.tv_nsec},
        {"changed_ns",
            static_cast<int64_t>(status.st_ctim.tv_sec)*INT64_C(1000000000) + status.st_ctim.tv_nsec},
        {"status_flags", status_flags},
        {"source_descriptor_flags", FD_CLOEXEC},
        {"target_descriptor_flags", 0},
    };
    if (hash_bytes) {
        result["sha256"] = sha256_descriptor(descriptor);
    }
    return result;
}

static json validate_model_descriptor(const fs::path & model_path, bool hash_bytes) {
    const int descriptor = required_descriptor("DSV41_MODEL_DESCRIPTOR");
    json expected;
    try {
        expected = json::parse(required_environment("DSV41_MODEL_DESCRIPTOR_IDENTITY"));
    } catch (const json::exception & error) {
        throw std::runtime_error(std::string("model descriptor identity JSON is invalid: ") + error.what());
    }
    const json observed = model_descriptor_identity(descriptor, model_path, hash_bytes);
    if (expected != observed) {
        throw std::runtime_error("held model descriptor differs from the runner identity");
    }
    return observed;
}

static std::vector<int64_t> namespace_pid_chain() {
    std::ifstream status("/proc/self/status");
    if (!status) {
        throw std::runtime_error("cannot read namespace-local process status");
    }
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("NSpid:", 0) != 0) {
            continue;
        }
        std::istringstream values(line.substr(6));
        std::vector<int64_t> result;
        int64_t value = 0;
        while (values >> value) {
            result.push_back(value);
        }
        if (result.empty() || result.back() != getpid()) {
            throw std::runtime_error("namespace-local NSpid identity is invalid");
        }
        return result;
    }
    throw std::runtime_error("namespace-local NSpid identity is missing");
}

static void require_live_pidfd(int descriptor) {
    if (fcntl(descriptor, F_GETFD) != 0) {
        throw std::runtime_error("watchdog pidfd descriptor flags are invalid");
    }
    std::array<char, 64> target {};
    const std::string descriptor_path = "/proc/self/fd/" + std::to_string(descriptor);
    const ssize_t size = readlink(descriptor_path.c_str(), target.data(), target.size() - 1);
    if (size <= 0 || std::string(target.data(), static_cast<size_t>(size)) != "anon_inode:[pidfd]") {
        throw std::runtime_error("watchdog authority descriptor is not a pidfd");
    }
    pollfd observed {descriptor, POLLIN, 0};
    const int result = poll(&observed, 1, 0);
    if (result < 0 || result != 0 || observed.revents != 0) {
        throw std::runtime_error("watchdog authority pidfd is not live");
    }
}

static json validate_watchdog(const json & data) {
    const int64_t pid = data.value("watchdog_pid", INT64_C(0));
    const int64_t guardian_pid = data.value("guardian_pid", INT64_C(0));
    const int64_t child_pid = data.value("child_pid", INT64_C(0));
    const int64_t child_pgid = data.value("child_process_group_id", INT64_C(0));
    if (data.value("format", "") != "strix-memory-watchdog-lease" || data.value("version", 0) != 2) {
        throw std::runtime_error("watchdog lease format is invalid");
    }
    if (data.value("soft_bytes", UINT64_C(0)) != 116*DSV41_GIB ||
            data.value("emergency_bytes", UINT64_C(0)) != 118*DSV41_GIB ||
            data.value("strict_ceiling_bytes", UINT64_C(0)) != 120*DSV41_GIB ||
            data.value("grace_seconds", 0.0) != 30.0 ||
            data.value("sample_interval_seconds", 0.0) != 1.0 ||
            data.value("procfs_root", "") != "/proc") {
        throw std::runtime_error("watchdog execution policy is invalid");
    }
    if (pid <= 1 || guardian_pid <= 1 || child_pid <= 1 || child_pgid <= 1) {
        throw std::runtime_error("watchdog host identity is invalid");
    }
    const json authority = data.value("namespace_authority", json::object());
    const int authority_descriptor = required_descriptor("DSV41_WATCHDOG_PIDFD");
    if (authority.value("format", "") != "dsv41-watchdog-namespace-authority" ||
            authority.value("version", 0) != 1 ||
            authority.value("mechanism", "") != "inherited-pidfd" ||
            authority.value("descriptor", -1) != authority_descriptor ||
            authority.value("host_procfs_root", "") != "/proc" ||
            authority.value("watchdog_pid", INT64_C(0)) != pid ||
            authority.value("watchdog_start_time_ticks", UINT64_C(0)) !=
                data.value("watchdog_start_time_ticks", UINT64_C(0)) ||
            authority.value("watchdog_executable_path", "") !=
                data.value("watchdog_executable_path", "") ||
            authority.value("watchdog_command_sha256", "") !=
                data.value("watchdog_command_sha256", "") ||
            authority.value("guardian_pid", INT64_C(0)) != guardian_pid ||
            authority.value("child_pid", INT64_C(0)) != child_pid ||
            authority.value("child_process_group_id", INT64_C(0)) != child_pgid) {
        throw std::runtime_error("watchdog namespace authority does not match the host audit");
    }
    require_live_pidfd(authority_descriptor);
    if (getpid() != 2 || getppid() != 1 || getpgrp() != getpid() || getsid(0) != getpid() ||
            canonical_path("/proc/self", "namespace-local process") !=
                canonical_path("/proc/" + std::to_string(getpid()), "namespace-local PID") ||
            !fs::is_directory("/proc/1")) {
        throw std::runtime_error("trace exporter namespace-local process identity is invalid");
    }
    const std::vector<int64_t> namespace_pids = namespace_pid_chain();
    const fs::path script_path = data.value("watchdog_script_path", "");
    if (script_path.empty() || data.value("watchdog_revision", "") !=
    "1568e7903cc07fc83fee1894df491dfcd2f7ba9c" ||
            data.value("watchdog_script_sha256", "") != WATCHDOG_SCRIPT_SHA256 ||
            sha256_file(script_path) != WATCHDOG_SCRIPT_SHA256) {
        throw std::runtime_error("watchdog script identity changed");
    }
    const fs::path heartbeat_path = data.value("heartbeat_path", "");
    const double max_age = data.value("max_heartbeat_age_seconds", 0.0);
    if (heartbeat_path.empty() || max_age <= 0 || max_age > 30) {
        throw std::runtime_error("watchdog heartbeat configuration is invalid");
    }
    if (fs::canonical(required_environment("STRIX_MEMORY_WATCHDOG_LEASE_PATH")) !=
            fs::canonical(fs::path(data.value("lease_path", ""))) ||
            fs::canonical(required_environment("STRIX_MEMORY_WATCHDOG_HEARTBEAT_PATH")) !=
                fs::canonical(heartbeat_path) ||
            fs::canonical(required_environment("STRIX_MEMORY_WATCHDOG_AUDIT_PATH")) !=
                fs::canonical(fs::path(data.value("audit_live_path", ""))) ||
            std::stod(required_environment("STRIX_MEMORY_WATCHDOG_HEARTBEAT_MAX_AGE_SECONDS")) != max_age) {
        throw std::runtime_error("watchdog lease does not match the inherited environment");
    }
    const json child_command = data.value("command", json::array());
    if (!child_command.is_array() || child_command.empty()) {
        throw std::runtime_error("watchdog child command is invalid");
    }
    for (const json & argument : child_command) {
        if (!argument.is_string()) {
            throw std::runtime_error("watchdog child command is invalid");
        }
    }
    const std::string child_command_json = child_command.dump(-1, ' ', true);
    if (sha256_data(
                reinterpret_cast<const uint8_t *>(child_command_json.data()),
                child_command_json.size()) != data.value("child_command_sha256", "")) {
        throw std::runtime_error("watchdog child command SHA-256 is invalid");
    }
    const std::vector<uint8_t> heartbeat_bytes = read_file(heartbeat_path);
    json heartbeat;
    try {
        heartbeat = json::parse(heartbeat_bytes.begin(), heartbeat_bytes.end());
    } catch (const json::exception & error) {
        throw std::runtime_error(std::string("watchdog heartbeat is invalid: ") + error.what());
    }
    if (heartbeat.value("format", "") != "strix-memory-watchdog-heartbeat" ||
            heartbeat.value("version", 0) != 2 ||
            heartbeat.value("lease_id", "") != data.value("lease_id", "") ||
            heartbeat.value("sequence", INT64_C(-1)) < 0 ||
            heartbeat.value("state", "") != "active" ||
            heartbeat.value("updated_at", "").empty() ||
            heartbeat.value("watchdog_pid", INT64_C(0)) != pid ||
            heartbeat.value("watchdog_start_time_ticks", UINT64_C(0)) !=
                data.value("watchdog_start_time_ticks", UINT64_C(0)) ||
            heartbeat.value("child_pid", INT64_C(0)) != child_pid ||
            heartbeat.value("child_process_group_id", INT64_C(0)) != child_pgid) {
        throw std::runtime_error("watchdog heartbeat identity is invalid");
    }
    const json heartbeat_sample = heartbeat.value("sample", json::object());
    const std::string audit_record_sha256 = heartbeat_sample.value("audit_record_sha256", "");
    if (audit_record_sha256.size() != 64) {
        throw std::runtime_error("watchdog heartbeat audit identity is invalid");
    }
    const uint64_t updated_monotonic_ns = heartbeat.value("updated_monotonic_ns", UINT64_C(0));
    struct timespec now;
    if (updated_monotonic_ns == 0 || clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        throw std::runtime_error("watchdog heartbeat monotonic timestamp is invalid");
    }
    const uint64_t now_monotonic_ns =
        static_cast<uint64_t>(now.tv_sec)*UINT64_C(1000000000) + static_cast<uint64_t>(now.tv_nsec);
    const uint64_t max_age_ns = static_cast<uint64_t>(max_age*1000000000.0);
    if (updated_monotonic_ns > now_monotonic_ns ||
            now_monotonic_ns - updated_monotonic_ns > max_age_ns) {
        throw std::runtime_error("watchdog heartbeat is stale");
    }
    const fs::path audit_path = data.value("audit_live_path", "");
    if (audit_path.empty()) {
        throw std::runtime_error("watchdog audit path is invalid");
    }
    struct stat audit_stat;
    const int audit_fd = data.value("audit_fd", -1);
    if (audit_fd < 0 || lstat(audit_path.c_str(), &audit_stat) != 0 ||
            !S_ISREG(audit_stat.st_mode) ||
            static_cast<uint64_t>(audit_stat.st_dev) != data.value("audit_device", UINT64_C(0)) ||
            static_cast<uint64_t>(audit_stat.st_ino) != data.value("audit_inode", UINT64_C(0)) ||
            audit_stat.st_uid != getuid() ||
            (audit_stat.st_mode & 0777) != 0600 ||
            data.value("audit_mode", UINT64_C(0)) != 0600) {
        throw std::runtime_error("watchdog persistent audit identity changed");
    }
    const int local_audit_fd = open(audit_path.c_str(), O_RDONLY | O_NOFOLLOW);
    if (local_audit_fd < 0) {
        throw std::runtime_error("cannot open watchdog persistent audit");
    }
    const int lock_result = flock(local_audit_fd, LOCK_EX | LOCK_NB);
    if (lock_result == 0) {
        flock(local_audit_fd, LOCK_UN);
        close(local_audit_fd);
        throw std::runtime_error("watchdog does not hold the persistent audit lock");
    }
    if (errno != EWOULDBLOCK && errno != EAGAIN) {
        close(local_audit_fd);
        throw std::runtime_error("cannot inspect watchdog persistent audit lock");
    }
    close(local_audit_fd);
    std::ifstream audit_stream(audit_path);
    if (!audit_stream) {
        throw std::runtime_error("cannot read watchdog persistent audit");
    }
    std::string audit_line;
    bool found_audit_record = false;
    while (std::getline(audit_stream, audit_line)) {
        audit_line.push_back('\n');
        if (sha256_data(audit_line.data(), audit_line.size()) == audit_record_sha256) {
            found_audit_record = true;
            break;
        }
    }
    if (!found_audit_record) {
        throw std::runtime_error("watchdog heartbeat audit record is missing");
    }
    require_live_pidfd(authority_descriptor);
    return {
        {"format", "dsv41-watchdog-namespace-binding"},
        {"version", 1},
        {"authority", "inherited-pidfd"},
        {"host_watchdog_pid", pid},
        {"host_watchdog_process_group_id",
            authority.value("watchdog_process_group_id", INT64_C(0))},
        {"host_watchdog_start_time_ticks",
            data.value("watchdog_start_time_ticks", UINT64_C(0))},
        {"local_pid", getpid()},
        {"local_parent_pid", getppid()},
        {"local_process_group_id", getpgrp()},
        {"local_session_id", getsid(0)},
        {"namespace_pids", namespace_pids},
        {"private_procfs", true},
    };
}
#endif

static void bind_memory_audit_metadata(json & result, const json & audit) {
    result["accelerator"] = audit.value("accelerator", json::object());
    result["storage"] = audit.value("storage", json::object());
    result["storage_policy"] = audit.value("storage_policy", json::object());
}

static json audit_reference(const char * environment_name, const char * expected_kind) {
    const fs::path path = required_environment(environment_name);
    dsv41::require_nvme_path(path, "audit");
    const std::vector<uint8_t> bytes = read_file(path);
    json audit;
    try {
        audit = json::parse(bytes.begin(), bytes.end());
    } catch (const json::exception & error) {
        throw std::runtime_error(std::string("invalid audit JSON: ") + error.what());
    }
    if (audit.value("kind", "") != expected_kind) {
        throw std::runtime_error(std::string("audit kind mismatch for ") + expected_kind);
    }
    if (audit.value("environment", json::object()).value("HIP_LAUNCH_BLOCKING", "") != "1") {
        throw std::runtime_error(std::string("audit environment mismatch for ") + expected_kind);
    }
    const int64_t created = audit.value("created_unix", INT64_C(0));
    const int64_t now = static_cast<int64_t>(std::time(nullptr));
    if (created <= 0 || now < created || now - created > 300) {
        throw std::runtime_error(std::string("audit is stale: ") + expected_kind);
    }
    if (std::string(expected_kind) == "swap" && audit["data"].value("enabled", true)) {
        throw std::runtime_error("swap audit reports enabled swap");
    }
    json result = {
        {"path", fs::absolute(path).lexically_normal().string()},
        {"sha256", sha256_data(bytes.data(), bytes.size())},
        {"created_unix", created},
    };
#if defined(__linux__)
    if (std::string(expected_kind) == "watchdog") {
        result["namespace_binding"] = validate_watchdog(audit["data"]);
    }
#endif
    if (std::string(expected_kind) == "watchdog") {
        result["data"] = audit["data"];
    } else if (std::string(expected_kind) == "memory") {
        bind_memory_audit_metadata(result, audit);
    }
    return result;
}

static std::string tensor_dtype(const ggml_tensor * tensor) {
    switch (tensor->type) {
        case GGML_TYPE_F32:  return "f32";
        case GGML_TYPE_BF16: return "bf16";
        case GGML_TYPE_I32:  return "i32";
        case GGML_TYPE_I8:   return "i8";
        default: throw std::runtime_error(
            std::string("unsupported trace tensor type: ") + ggml_type_name(tensor->type));
    }
}

static std::vector<int64_t> tensor_shape(const ggml_tensor * tensor) {
    int rank = GGML_MAX_DIMS;
    while (rank > 2 && tensor->ne[rank - 1] == 1) {
        --rank;
    }
    std::vector<int64_t> result;
    result.reserve(rank);
    for (int i = 0; i < rank; ++i) {
        result.push_back(tensor->ne[i]);
    }
    return result;
}

static void write_manifest_file(const fs::path & path, const json & manifest);

class trace_writer {
public:
    trace_writer(fs::path root, json manifest) :
        root(std::move(root)),
        blobs(this->root / "blobs"),
        manifest(std::move(manifest)) {
        if (fs::exists(this->root) && !fs::is_empty(this->root)) {
            throw std::runtime_error("trace output directory is not empty: " + this->root.string());
        }
        fs::create_directories(blobs);
        events.open(this->root / "events.jsonl", std::ios::binary | std::ios::trunc);
        if (!events) {
            throw std::runtime_error("cannot create events.jsonl");
        }
        this->manifest["trace_format"] = "dsv41-trace";
        this->manifest["trace_version"] = TRACE_VERSION;
    }

    void set_execution(std::string phase, int step, int64_t token_start, int64_t token_count) {
        this->phase = std::move(phase);
        this->step = step;
        this->token_start = token_start;
        this->token_count = token_count;
    }

    void add(
            const std::string & component,
            int layer,
            const std::string & dtype,
            const std::vector<int64_t> & shape,
            const void * data,
            size_t size,
            const char * semantic_id_space = nullptr) {
        if (error_message.size() != 0) {
            return;
        }
        try {
            const std::string digest = sha256_data(data, size);
            const fs::path blob = blobs / (digest + ".bin");
            if (!fs::exists(blob)) {
                const fs::path temp = blob.string() + ".tmp";
                {
                    std::ofstream output(temp, std::ios::binary | std::ios::trunc);
                    if (!output || (size != 0 && !output.write(static_cast<const char *>(data), size))) {
                        throw std::runtime_error("cannot write trace blob");
                    }
                }
                fs::rename(temp, blob);
            }

            json event = {
                {"trace_version", TRACE_VERSION},
                {"component", component},
                {"phase", phase},
                {"step", step},
                {"token_start", token_start},
                {"token_count", token_count},
                {"layer", layer >= 0 ? json(layer) : json(nullptr)},
                {"dtype", dtype},
                {"shape", shape},
                {"byte_order", "little"},
                {"byte_count", size},
                {"sha256", digest},
                {"blob", "blobs/" + digest + ".bin"},
            };
            if (semantic_id_space != nullptr) {
                event["semantic_id_space"] = semantic_id_space;
            }
            events << event.dump() << '\n';
            events.flush();
            if (!events) {
                throw std::runtime_error("cannot append trace event");
            }
            ++event_count;
        } catch (const std::exception & error) {
            error_message = error.what();
        }
    }

    void add_tensor(const ggml_tensor * tensor) {
        const std::string name = tensor->name;
        const auto descriptor = dsv41_trace_parse_name(name);
        if (!descriptor) {
            return;
        }
        const size_t size = ggml_nbytes(tensor);
        buffer.resize(size);
        ggml_backend_tensor_get(tensor, buffer.data(), 0, size);
        add(descriptor->component, descriptor->layer, tensor_dtype(tensor), tensor_shape(tensor),
                buffer.data(), size, descriptor->semantic_id_space);
    }

    bool has_error() const {
        return !error_message.empty();
    }

    const std::string & error() const {
        return error_message;
    }

    void fail(const std::string & message) {
        if (error_message.empty()) {
            error_message = message;
        }
    }

    void bind_runtime_post(json runtime_libraries) {
        if (manifest["build"]["runtime_libraries"] != runtime_libraries) {
            throw std::runtime_error("loaded runtime component set changed during protected trace generation");
        }
        manifest["build"]["runtime_libraries_post"] = std::move(runtime_libraries);
        manifest["build"]["runtime_module_monitor"]["checked_after_trace"] = true;
    }

    void finish() {
        if (has_error()) {
            throw std::runtime_error(error_message);
        }
        events.close();
        manifest["event_count"] = event_count;
        write_manifest_file(root / "manifest.json", manifest);
    }

private:
    fs::path root;
    fs::path blobs;
    std::ofstream events;
    json manifest;
    std::vector<uint8_t> buffer;
    std::string phase = "unknown";
    std::string error_message;
    int step = 0;
    int64_t token_start = 0;
    int64_t token_count = 0;
    uint64_t event_count = 0;
};

static bool trace_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * writer = static_cast<trace_writer *>(user_data);
    try {
        if (ask) {
            return dsv41_trace_select_name(tensor->name);
        }
        writer->add_tensor(tensor);
        return !writer->has_error();
    } catch (const std::exception & error) {
        writer->fail(error.what());
        std::fprintf(stderr, "trace callback failed: %s\n", error.what());
        return false;
    }
}

static std::vector<float> copy_logits(llama_context * ctx, int32_t n_vocab) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        throw std::runtime_error("runtime did not produce final-token logits");
    }
    return std::vector<float>(logits, logits + n_vocab);
}

static int32_t greedy_token(const std::vector<float> & logits) {
    if (logits.empty()) {
        throw std::runtime_error("cannot select from empty logits");
    }
    return static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

static void decode_tokens(
        llama_context * ctx,
        trace_writer & writer,
        const std::vector<llama_token> & tokens,
        int32_t n_ubatch) {
    int64_t offset = 0;
    while (offset < static_cast<int64_t>(tokens.size())) {
        const int32_t count = static_cast<int32_t>(std::min<int64_t>(n_ubatch, tokens.size() - offset));
        llama_batch batch = llama_batch_init(count, 0, 1);
        for (int32_t i = 0; i < count; ++i) {
            const bool logits = offset + i + 1 == static_cast<int64_t>(tokens.size());
            common_batch_add(batch, tokens[offset + i], offset + i, {0}, logits);
        }
        writer.set_execution("prefill", 0, offset, count);
        const int result = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (result != 0) {
            throw std::runtime_error("prefill failed at token " + std::to_string(offset));
        }
        if (writer.has_error()) {
            throw std::runtime_error(writer.error());
        }
        offset += count;
    }
}

static std::string model_architecture(const llama_model * model) {
    std::array<char, 128> buffer = {};
    const int32_t count = llama_model_meta_val_str(
            model, "general.architecture", buffer.data(), buffer.size());
    if (count < 0) {
        throw std::runtime_error("model has no general.architecture metadata");
    }
    return buffer.data();
}

static std::vector<std::string> command_line(int argc, char ** argv) {
    std::vector<std::string> result;
    result.reserve(argc);
    for (int i = 0; i < argc; ++i) {
        result.emplace_back(argv[i]);
    }
    return result;
}

static std::string command_line_json(int argc, char ** argv) {
    return json(command_line(argc, argv)).dump();
}

static bool flash_attention_enabled(enum llama_flash_attn_type value) {
    return value == LLAMA_FLASH_ATTN_TYPE_ENABLED;
}

static std::string runtime_system_info(const common_params & params) {
#if defined(_WIN32)
    const std::string platform = "Windows";
#else
    struct utsname info = {};
    if (uname(&info) != 0) {
        throw std::runtime_error("cannot query operating system identity");
    }
    const std::string platform =
        std::string(info.sysname) + " " + info.release + " " + info.machine;
#endif
    return platform + "; " + common_params_get_system_info(params);
}

static std::string runtime_platform_name() {
#if defined(_WIN32)
    return "windows";
#elif defined(__APPLE__)
    return "darwin";
#elif defined(__linux__)
    return "linux";
#else
    return "unknown";
#endif
}

static json accelerator_json(const dsv41::accelerator_attestation & accelerator) {
    return {
        {"format", "dsv41-accelerator-attestation"},
        {"version", 2},
        {"runtime_kind", "strix-rocm"},
        {"platform", "linux"},
        {"backend", "ROCm"},
        {"backend_device", accelerator.backend_device},
        {"backend_description", accelerator.backend_description},
        {"pci_device_id", accelerator.pci_device_id},
        {"kfd_node", accelerator.kfd_node},
        {"gpu_id", accelerator.gpu_id},
        {"gfx_target_version", accelerator.gfx_target_version},
        {"architecture", accelerator.architecture},
        {"source", "linux-kfd-sysfs"},
    };
}

static json storage_json(const dsv41::storage_attestation & storage) {
    return {
        {"format", "dsv41-storage-attestation"},
        {"version", 2},
        {"runtime_kind", "strix-rocm"},
        {"platform", "linux"},
        {"storage_kind", "linux-nvme"},
        {"resolved_path", storage.resolved_path.string()},
        {"existing_path", storage.existing_path.string()},
        {"mount_point", storage.mount_point},
        {"filesystem_type", storage.filesystem_type},
        {"mount_source", storage.mount_source},
        {"device_number", storage.device_number},
        {"block_device_path", storage.block_device_path.string()},
        {"nvme_device", storage.nvme_device},
        {"rotational", false},
        {"source", "linux-mountinfo-sysfs"},
    };
}

static json storage_policy_json() {
    return {
        {"format", "dsv41-state-storage-policy"},
        {"version", 1},
        {"expert_cache", "memory-resident"},
        {"kv_cache", "memory-resident"},
        {"external_cache_paths", json::array()},
        {"external_state_paths", json::array()},
    };
}

struct native_manifest_evidence {
    json accelerator;
    json paths;
    json config;
};

static json complete_manifest(
        json input,
        native_manifest_evidence evidence,
        const fs::path & executable,
        ggml_backend_dev_t selected_device,
        const std::string & selected_backend_component,
        const std::string & system_info,
        int argc,
        char ** argv) {
    static const std::array<const char *, 5> required = {
        "model", "prompt", "audits", "expected", "event_count",
    };
    if (!input.is_object()) {
        throw std::runtime_error("manifest writer input is not a JSON object");
    }
    for (const char * key : required) {
        if (!input.contains(key) && std::string(key) != "event_count") {
            throw std::runtime_error(std::string("manifest writer input is missing ") + key);
        }
    }
    for (const auto & item : input.items()) {
        if (std::find_if(required.begin(), required.end(), [&](const char * key) {
                return item.key() == key;
            }) == required.end()) {
            throw std::runtime_error("manifest writer input has unexpected field: " + item.key());
        }
    }
    json manifest = {
        {"trace_format", "dsv41-trace"},
        {"trace_version", TRACE_VERSION},
        {"runtime", "llama.cpp"},
        {"revision", BUILD_REVISION},
        {"build", runtime_build_json(executable, selected_device, selected_backend_component, argv)},
        {"model", std::move(input["model"])},
        {"prompt", std::move(input["prompt"])},
        {"accelerator", std::move(evidence.accelerator)},
        {"paths", std::move(evidence.paths)},
        {"storage_policy", storage_policy_json()},
        {"config", std::move(evidence.config)},
        {"comparison", {
            {"tokens", "exact"},
            {"engram_rows", "exact"},
            {"expert_ids", "exact-original-id-space"},
            {"expert_weights", "byte-identical-f32"},
            {"attention_candidates", "exact"},
            {"logits", "byte-identical-f32"},
        }},
        {"environment", {
            {"system_info", system_info},
            {"command", command_line_json(argc, argv)},
        }},
        {"audits", std::move(input["audits"])},
        {"expected", std::move(input["expected"])},
    };
    if (input.contains("event_count")) {
        manifest["event_count"] = std::move(input["event_count"]);
    }
    return manifest;
}

static void write_manifest_file(const fs::path & path, const json & manifest) {
    const fs::path temp = path.string() + ".tmp";
    {
        std::ofstream stream(temp, std::ios::binary | std::ios::trunc);
        stream << manifest.dump() << '\n';
        if (!stream) {
            throw std::runtime_error("cannot write trace manifest");
        }
    }
    fs::rename(temp, path);
}

#if defined(DSV41_MANIFEST_TEST_HARNESS)
static native_manifest_evidence test_manifest_evidence(
        const fs::path & input_path,
        const fs::path & output_path,
        ggml_backend_dev_t device) {
    if (device == nullptr || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        throw std::runtime_error("manifest writer test requires a local CPU device");
    }
    ggml_backend_dev_props properties = {};
    ggml_backend_dev_get_props(device, &properties);
    ggml_backend_reg_t backend = ggml_backend_dev_backend_reg(device);
    if (backend == nullptr) {
        throw std::runtime_error("manifest writer test CPU has no runtime registry");
    }
    const fs::path input = canonical_path(input_path, "manifest writer test input");
    const fs::path output_parent =
        canonical_path(fs::absolute(output_path).parent_path(), "manifest writer test output directory");
    const fs::path output = output_parent / output_path.filename();
    const fs::path repository = canonical_path(DSV41_SOURCE_ROOT, "manifest writer source root");
    return {
        {
            {"format", "dsv41-native-test-accelerator"},
            {"version", 1},
            {"runtime_kind", "native-test"},
            {"platform", runtime_platform_name()},
            {"backend", ggml_backend_reg_name(backend)},
            {"backend_device", properties.name == nullptr ? "" : properties.name},
            {"backend_description", properties.description == nullptr ? "" : properties.description},
            {"device_type", "cpu"},
            {"source", "ggml-runtime"},
            {"test_only", true},
        },
        {
            {"format", "dsv41-native-test-paths"},
            {"version", 1},
            {"input", input.string()},
            {"output", output.string()},
            {"repository", repository.string()},
            {"temporary_directory", output_parent.string()},
            {"test_only", true},
        },
        {
            {"format", "dsv41-native-test-config"},
            {"version", 1},
            {"runtime_kind", "native-test"},
            {"platform", runtime_platform_name()},
            {"device", properties.name == nullptr ? "" : properties.name},
            {"device_description", properties.description == nullptr ? "" : properties.description},
            {"device_type", "cpu"},
            {"build_target", llama_build_target()},
            {"flash_attention", false},
            {"gpu_layers", 0},
            {"test_only", true},
        },
    };
}

static void write_manifest_probe(
        const fs::path & input_path,
        const fs::path & output_path,
        int argc,
        char ** argv) {
    const std::vector<uint8_t> bytes = read_file(input_path);
    json input = json::parse(bytes.begin(), bytes.end());
    static const std::array<const char *, 5> allowed = {
        "model", "prompt", "audits", "expected", "event_count",
    };
    if (!input.is_object()) {
        throw std::runtime_error("manifest writer test input is not a JSON object");
    }
    for (const char * key : allowed) {
        if (!input.contains(key)) {
            throw std::runtime_error(std::string("manifest writer test input is missing ") + key);
        }
    }
    for (const auto & item : input.items()) {
        if (std::find_if(allowed.begin(), allowed.end(), [&](const char * key) {
                return item.key() == key;
            }) == allowed.end()) {
            throw std::runtime_error("manifest writer test input has unexpected field: " + item.key());
        }
    }
    common_init();
    const fs::path executable = current_executable_path();
    load_runtime_backends(executable);
    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    json manifest = complete_manifest(
        std::move(input),
        test_manifest_evidence(input_path, output_path, device),
        executable,
        device,
        "ggml-cpu",
        runtime_platform_name() + " local CPU model-free manifest writer test",
        argc,
        argv);
    begin_loader_monitor();
    end_loader_monitor(executable);
    manifest["build"]["runtime_libraries_post"] = runtime_libraries_json(
        executable, device, BUILD_REVISION, "ggml-cpu");
    if (manifest["build"]["runtime_libraries"] != manifest["build"]["runtime_libraries_post"]) {
        throw std::runtime_error("loaded runtime component set changed during manifest writer test");
    }
    manifest["build"]["runtime_module_monitor"]["checked_after_trace"] = true;
    write_manifest_file(output_path, manifest);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    try {
        if (argc != 4 || std::string(argv[1]) != "--write-test-manifest") {
            throw std::runtime_error("test manifest writer requires input and output paths");
        }
        write_manifest_probe(argv[2], argv[3], argc, argv);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "test-deepseek-v41-manifest: %s\n", error.what());
        return 1;
    }
}
#else
int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    try {
        reject_loader_overrides();
#if defined(__linux__)
        if (argc == 3 && std::string(argv[1]) == "--dsv41-test-watchdog") {
            const std::vector<uint8_t> bytes = read_file(argv[2]);
            std::cout << validate_watchdog(json::parse(bytes.begin(), bytes.end())).dump() << '\n';
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--dsv41-test-model-descriptor") {
            const fs::path model_path = canonical_path(argv[2], "test model");
            std::cout << validate_model_descriptor(model_path, true).dump() << '\n';
            return 0;
        }
#endif
        if (argc == 2 && std::string(argv[1]) == "--version") {
            common_init();
            const fs::path executable = current_executable_path();
            load_runtime_backends(executable);
            const json build = runtime_build_json(
                executable,
                ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU),
                "ggml-cpu",
                argv);
            std::cout << "version: deepseek-v41-trace (build " << build["number"]
                      << ", commit " << BUILD_REVISION << ")\n";
            std::cout << "built with " << build["compiler"].get<std::string>()
                      << " for " << build["target"].get<std::string>() << '\n';
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--dsv41-attest-build") {
            if (std::string(argv[2]) != "ROCm0") {
                throw std::runtime_error("build attestation requires selected execution device ROCm0");
            }
            common_init();
            const fs::path executable = current_executable_path();
            load_runtime_backends(executable);
            ggml_backend_dev_t device = ggml_backend_dev_by_name(argv[2]);
            if (device == nullptr) {
                throw std::runtime_error("cannot find selected execution device ROCm0");
            }
            std::cout << runtime_build_json(executable, device, "ggml-hip", argv).dump() << '\n';
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--dsv41-attest-device") {
            common_init();
            load_runtime_backends(current_executable_path());
            const dsv41::accelerator_attestation accelerator =
                dsv41::require_gfx1151_device(ggml_backend_dev_by_name(argv[2]));
            std::cout << accelerator_json(accelerator).dump() << '\n';
            return 0;
        }

        const uint16_t endian = 1;
        if (*reinterpret_cast<const uint8_t *>(&endian) != 1) {
            throw std::runtime_error("trace writer requires a little-endian host");
        }

        common_params params;
        params.escape = false;
        params.warmup = false;
        params.ctx_shift = false;
        common_init();
        load_runtime_backends(current_executable_path());
        if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_RESULTS)) {
            return 1;
        }
        if (params.model.path.empty() || params.prompt_file.empty() || params.out_file.empty()) {
            throw std::runtime_error("-m, -bf, and -o are required");
        }
        if (params.n_predict < 1) {
            throw std::runtime_error("-n must request at least one deterministic decode step");
        }
        const fs::path executable_path = current_executable_path();

        const dsv41::storage_attestation model_storage =
            dsv41::require_nvme_path(params.model.path, "model");
        const dsv41::storage_attestation prompt_storage =
            dsv41::require_nvme_path(params.prompt_file, "prompt");
        const dsv41::storage_attestation output_storage =
            dsv41::require_nvme_path(params.out_file, "trace output");
        const fs::path temporary_directory = required_environment("TMPDIR");
        dsv41::require_usable_directory(temporary_directory, "TMPDIR");
        const dsv41::storage_attestation temporary_storage =
            dsv41::require_nvme_path(temporary_directory, "temporary directory");
        if (required_environment("HIP_LAUNCH_BLOCKING") != "1") {
            throw std::runtime_error("HIP_LAUNCH_BLOCKING=1 is required for gfx1151 correctness runs");
        }
        const json tokenizer_policy = json::parse(required_environment("DSV41_TOKENIZER_POLICY"));
        if (!tokenizer_policy.is_object() || tokenizer_policy.size() != 5 ||
                !tokenizer_policy.contains("add_bos") ||
                !tokenizer_policy.contains("parse_special") ||
                !tokenizer_policy.contains("detokenize_special") ||
                !tokenizer_policy.contains("remove_leading_bos_before_detokenize") ||
                !tokenizer_policy.contains("require_round_trip") ||
                !tokenizer_policy["add_bos"].is_boolean() ||
                !tokenizer_policy["parse_special"].is_boolean() ||
                !tokenizer_policy["detokenize_special"].is_boolean() ||
                !tokenizer_policy["remove_leading_bos_before_detokenize"].is_boolean() ||
                !tokenizer_policy["require_round_trip"].is_boolean() ||
                !tokenizer_policy["parse_special"].get<bool>() ||
                !tokenizer_policy["detokenize_special"].get<bool>() ||
                !tokenizer_policy["require_round_trip"].get<bool>() ||
                tokenizer_policy["remove_leading_bos_before_detokenize"].get<bool>() !=
                    tokenizer_policy["add_bos"].get<bool>()) {
            throw std::runtime_error("DSV41_TOKENIZER_POLICY is not the exact approved tokenizer policy");
        }
        if (params.devices.size() != 1 || params.devices[0] == nullptr) {
            throw std::runtime_error("trace tool requires exactly one selected execution device");
        }
        const dsv41::accelerator_attestation configured_accelerator =
            dsv41::require_gfx1151_device(params.devices[0]);
        const json memory_audit = audit_reference("DSV41_TRACE_MEMORY_AUDIT", "memory");
        if (memory_audit.value("accelerator", json::object()) != accelerator_json(configured_accelerator)) {
            throw std::runtime_error("preflight accelerator audit does not match the selected execution device");
        }
        if (memory_audit.value("storage_policy", json::object()) != storage_policy_json()) {
            throw std::runtime_error("preflight external cache/state storage policy is invalid");
        }
        const json audited_storage = memory_audit.value("storage", json::object());
        if (audited_storage.value("model", json::object()) != storage_json(model_storage) ||
                audited_storage.value("prompt", json::object()) != storage_json(prompt_storage) ||
                audited_storage.value("output", json::object()) != storage_json(output_storage) ||
                audited_storage.value("temporary_directory", json::object()) != storage_json(temporary_storage)) {
            throw std::runtime_error("preflight storage audit does not match the selected execution paths");
        }
        const json swap_audit = audit_reference("DSV41_TRACE_SWAP_AUDIT", "swap");
        const json watchdog_audit = audit_reference("DSV41_TRACE_WATCHDOG_AUDIT", "watchdog");

        const std::vector<uint8_t> prompt_bytes = read_file(params.prompt_file);
        if (params.prompt.size() != prompt_bytes.size() ||
                !std::equal(prompt_bytes.begin(), prompt_bytes.end(), params.prompt.begin())) {
            throw std::runtime_error("parsed prompt differs from exact prompt file bytes");
        }

        const fs::path model_path = model_storage.resolved_path;
        const fs::path prompt_path = prompt_storage.resolved_path;
        const fs::path output_path = output_storage.resolved_path;
#if !defined(__linux__)
        throw std::runtime_error("trace model descriptor binding requires Linux");
#else
        const int model_descriptor = required_descriptor("DSV41_MODEL_DESCRIPTOR");
        const json model_file_identity = validate_model_descriptor(model_path, true);
        params.model.path = "/proc/self/fd/" + std::to_string(model_descriptor);
#endif
        llama_backend_init();
        llama_numa_init(params.numa);
        common_init_result_ptr init = common_init_from_params(params);
        llama_model * model = init->model();
        llama_context * ctx = init->context();
        if (model == nullptr || ctx == nullptr) {
            throw std::runtime_error("failed to initialize llama.cpp");
        }
        if (model_architecture(model) != "deepseek41") {
            throw std::runtime_error("trace tool requires general.architecture=deepseek41");
        }
        std::vector<std::string> model_devices;
        for (int32_t index = 0; index < llama_model_n_devices(model); ++index) {
            model_devices.emplace_back(ggml_backend_dev_name(llama_model_get_device(model, index)));
        }
        if (model_devices != std::vector<std::string>{"ROCm0"}) {
            throw std::runtime_error("trace tool requires the loaded model to use only ROCm0");
        }
        const dsv41::accelerator_attestation accelerator =
            dsv41::require_gfx1151_device(llama_model_get_device(model, 0));
        if (accelerator_json(accelerator) != accelerator_json(configured_accelerator)) {
            throw std::runtime_error("loaded model device differs from the pre-allocation accelerator attestation");
        }
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const bool add_bos = llama_vocab_get_add_bos(vocab);
        if (add_bos != tokenizer_policy["add_bos"].get<bool>()) {
            throw std::runtime_error("model tokenizer add_bos differs from the approved policy");
        }
        const std::vector<llama_token> tokens = common_tokenize(
            ctx, params.prompt, add_bos, tokenizer_policy["parse_special"].get<bool>());
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        if (tokens.empty()) {
            throw std::runtime_error("prompt tokenization produced no tokens");
        }
        if (tokens.size() > llama_n_ctx(ctx)) {
            throw std::runtime_error("prompt token count exceeds the configured context");
        }
        if (tokens.size() + static_cast<size_t>(params.n_predict) > llama_n_ctx(ctx)) {
            throw std::runtime_error("prompt plus decode steps exceed the configured context");
        }

        std::vector<int32_t> all_layers(40);
        for (int32_t layer = 0; layer < 40; ++layer) {
            all_layers[layer] = layer;
        }

        native_manifest_evidence evidence = {
            accelerator_json(accelerator),
            {
                {"model", model_path.string()},
                {"prompt", prompt_path.string()},
                {"output", output_path.string()},
                {"repository", audited_storage["repository"].value("resolved_path", "")},
                {"temporary_directory", temporary_storage.resolved_path.string()},
            },
            {
                {"context", llama_n_ctx(ctx)},
                {"batch", params.n_batch},
                {"ubatch", params.n_ubatch},
                {"device", model_devices[0]},
                {"device_architecture", accelerator.architecture},
                {"device_pci_id", accelerator.pci_device_id},
                {"decode_steps", params.n_predict},
                {"kv_type_k", ggml_type_name(params.cache_type_k)},
                {"kv_type_v", ggml_type_name(params.cache_type_v)},
                {"flash_attention", flash_attention_enabled(params.flash_attn_type)},
                {"gpu_layers", params.n_gpu_layers},
                {"load_mode", static_cast<int>(params.load_mode)},
                {"expert_cache_slots", params.expert_cache_slots},
                {"expert_cache_bytes", static_cast<uint64_t>(params.expert_cache_mib) << 20},
                {"tokenizer", tokenizer_policy},
#if defined(__linux__)
                {"model_file_identity", model_file_identity},
                {"watchdog_namespace", watchdog_audit["namespace_binding"]},
#endif
                {"deepseek41", {
                    {"layer_count", 40},
                    {"vocab_size", n_vocab},
                    {"engram_layers", {1, 14}},
                    {"engram_rows_per_token", 24},
                    {"expert_count", 384},
                    {"experts_used", 6},
                    {"candidate_source_layer", 20},
                    {"candidate_topk_blocks", 2048},
                    {"candidate_block_size", 8},
                    {"index_top_k", 512},
                    {"raw_attention_layers", {0, 1}},
                    {"raw_attention_width", 128},
                    {"candidate_propagation_layers", {24, 28, 32, 36}},
                }},
            },
        };
        json manifest_input = {
            {"model", {
                {"path", model_path.string()},
                {"architecture", "deepseek41"},
#if defined(__linux__)
                {"byte_count", model_file_identity["byte_count"]},
                {"sha256", model_file_identity["sha256"]},
#else
                {"byte_count", fs::file_size(model_path)},
                {"sha256", sha256_file(model_path)},
#endif
            }},
            {"prompt", {
                {"path", prompt_path.string()},
                {"byte_count", prompt_bytes.size()},
                {"sha256", sha256_data(prompt_bytes.data(), prompt_bytes.size())},
            }},
            {"audits", {
                {"memory", memory_audit},
                {"swap", swap_audit},
                {"watchdog", watchdog_audit},
            }},
            {"expected", {
                {"prompt_tokens", tokens.size()},
                {"decode_steps", params.n_predict},
                {"components", {
                    {"prompt.bytes", {{"layers", nullptr}, {"input", "tokens"}}},
                    {"prompt.tokens", {{"layers", nullptr}, {"input", "tokens"}}},
                    {"engram.row_ids", {{"layers", {1, 14}}, {"prefill", "tokens"}, {"decode", "steps"}}},
                    {"expert.ids", {{"layers", all_layers}, {"prefill", "tokens"}, {"decode", "steps"}}},
                    {"expert.weights", {{"layers", all_layers}, {"prefill", "tokens"}, {"decode", "steps"}}},
                    {"attn.source", {{"layers", all_layers}, {"prefill", "tokens"}, {"decode", "steps"}}},
                    {"attn.candidate_blocks", {{"layers", {20}}, {"prefill", "tokens"}, {"decode", "steps"}}},
                    {"attn.candidates", {{"layers", {24, 28, 32, 36}}, {"prefill", "tokens"}, {"decode", "steps"}}},
                    {"logits.prefill", {{"layers", nullptr}, {"prefill", "final"}}},
                    {"logits.decode", {{"layers", nullptr}, {"decode", "steps"}}},
                    {"decode.greedy_token", {{"layers", nullptr}, {"decode", "steps"}}},
                }},
            }},
        };
        json manifest = complete_manifest(
            std::move(manifest_input),
            std::move(evidence),
            executable_path,
            params.devices[0],
            "ggml-hip",
            runtime_system_info(params),
            argc,
            argv);

        begin_loader_monitor();
        trace_writer writer(output_path, std::move(manifest));
        llama_set_eval_callback(ctx, trace_callback, &writer);

        writer.set_execution("input", 0, 0, tokens.size());
        writer.add("prompt.bytes", -1, "bytes", {static_cast<int64_t>(prompt_bytes.size())},
                prompt_bytes.data(), prompt_bytes.size());
        writer.add("prompt.tokens", -1, "i32", {static_cast<int64_t>(tokens.size())},
                tokens.data(), tokens.size()*sizeof(tokens[0]));

        decode_tokens(ctx, writer, tokens, params.n_ubatch);
        std::vector<float> logits = copy_logits(ctx, n_vocab);
        writer.set_execution("prefill", 0, tokens.size() - 1, 1);
        writer.add("logits.prefill", -1, "f32", {n_vocab}, logits.data(), logits.size()*sizeof(float));

        int64_t position = tokens.size();
        for (int32_t step = 0; step < params.n_predict; ++step) {
            const llama_token token = greedy_token(logits);
            writer.set_execution("decode", step, position, 1);
            writer.add("decode.greedy_token", -1, "i32", {1}, &token, sizeof(token));

            llama_batch batch = llama_batch_init(1, 0, 1);
            common_batch_add(batch, token, position, {0}, true);
            const int result = llama_decode(ctx, batch);
            llama_batch_free(batch);
            if (result != 0) {
                throw std::runtime_error("decode failed at step " + std::to_string(step));
            }
            if (writer.has_error()) {
                throw std::runtime_error(writer.error());
            }
            logits = copy_logits(ctx, n_vocab);
            writer.add("logits.decode", -1, "f32", {n_vocab}, logits.data(), logits.size()*sizeof(float));
            ++position;
        }

#if defined(__linux__)
        if (validate_watchdog(watchdog_audit["data"]) != watchdog_audit["namespace_binding"]) {
            throw std::runtime_error("watchdog namespace binding changed during trace execution");
        }
        if (validate_model_descriptor(model_path, true) != model_file_identity) {
            throw std::runtime_error("held model descriptor changed during trace execution");
        }
#endif
        end_loader_monitor(executable_path);
        writer.bind_runtime_post(runtime_libraries_json(
            executable_path, params.devices[0], BUILD_REVISION, "ggml-hip"));
        writer.finish();
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "llama-deepseek-v41-trace: %s\n", error.what());
        return 1;
    }
}
#endif
