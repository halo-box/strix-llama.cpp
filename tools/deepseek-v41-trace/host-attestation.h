#pragma once

#include "ggml-backend.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace dsv41 {

namespace fs = std::filesystem;

struct storage_attestation {
    fs::path resolved_path;
    fs::path existing_path;
    std::string mount_point;
    std::string filesystem_type;
    std::string mount_source;
    std::string device_number;
    fs::path block_device_path;
    std::string nvme_device;
};

struct accelerator_attestation {
    std::string backend_device;
    std::string backend_description;
    std::string pci_device_id;
    std::string kfd_node;
    uint64_t gpu_id;
    uint64_t gfx_target_version;
    std::string architecture;
};

static inline std::string read_text(const fs::path & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot read " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

static inline std::string trim(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    size_t first = 0;
    while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first]))) {
        ++first;
    }
    return value.substr(first);
}

static inline std::string decode_mount_field(const std::string & value) {
    std::string result;
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '\\' && i + 3 < value.size() &&
                value[i + 1] >= '0' && value[i + 1] <= '7' &&
                value[i + 2] >= '0' && value[i + 2] <= '7' &&
                value[i + 3] >= '0' && value[i + 3] <= '7') {
            const int byte = (value[i + 1] - '0')*64 + (value[i + 2] - '0')*8 + value[i + 3] - '0';
            result.push_back(static_cast<char>(byte));
            i += 3;
        } else {
            result.push_back(value[i]);
        }
    }
    return result;
}

static inline bool path_is_within(const fs::path & path, const fs::path & root) {
    if (path == root) {
        return true;
    }
    const fs::path relative = path.lexically_relative(root);
    return !relative.empty() && *relative.begin() != "..";
}

static inline fs::path existing_ancestor(const fs::path & path) {
    fs::path current = path;
    while (!current.empty() && !fs::exists(current)) {
        const fs::path parent = current.parent_path();
        if (parent == current) {
            break;
        }
        current = parent;
    }
    if (current.empty() || !fs::exists(current)) {
        throw std::runtime_error("path has no existing parent: " + path.string());
    }
    return fs::canonical(current);
}

static inline void reject_symlink_components(const fs::path & path, const char * label) {
    fs::path current;
    for (const fs::path & part : fs::absolute(path)) {
        current /= part;
        std::error_code error;
        const fs::file_status status = fs::symlink_status(current, error);
        if (!error && fs::is_symlink(status)) {
            throw std::runtime_error(std::string(label) + " must not be a symlink or contain symlink components");
        }
        if (error == std::errc::no_such_file_or_directory) {
            break;
        }
        if (error) {
            throw std::runtime_error(
                std::string(label) + " symlink status cannot be read: " + error.message());
        }
    }
}

static inline void require_usable_directory(const fs::path & path, const char * label) {
    reject_symlink_components(path, label);
    if (!fs::is_directory(path)) {
        throw std::runtime_error(std::string(label) + " must be an existing directory");
    }
#if defined(_WIN32)
    if (_access(path.string().c_str(), 6) != 0) {
#else
    if (access(path.string().c_str(), W_OK | X_OK) != 0) {
#endif
        throw std::runtime_error(std::string(label) + " must be writable and searchable");
    }
}

static inline storage_attestation require_nvme_path(
        const fs::path & path,
        const char * label,
        const fs::path & mountinfo_path = "/proc/self/mountinfo",
        const fs::path & sys_dev_block_root = "/sys/dev/block",
        const fs::path & sys_class_block_root = "/sys/class/block",
        const fs::path & forbidden_root = "/mnt/bigspace") {
    const fs::path absolute = fs::absolute(path).lexically_normal();
    const fs::path forbidden = fs::absolute(forbidden_root).lexically_normal();
    if (path_is_within(absolute, forbidden)) {
        throw std::runtime_error(std::string(label) + " must not use /mnt/bigspace");
    }
    reject_symlink_components(path, label);
    const fs::path resolved = fs::weakly_canonical(absolute);
    const fs::path existing = existing_ancestor(resolved);
    if (path_is_within(resolved, forbidden)) {
        throw std::runtime_error(std::string(label) + " must not use /mnt/bigspace");
    }

    struct mount_record {
        fs::path mount_point;
        std::string filesystem_type;
        std::string source;
        std::string device_number;
    };
    std::vector<mount_record> mounts;
    std::istringstream mountinfo(read_text(mountinfo_path));
    std::string line;
    while (std::getline(mountinfo, line)) {
        std::istringstream fields_stream(line);
        std::vector<std::string> fields;
        std::string field;
        while (fields_stream >> field) {
            fields.push_back(field);
        }
        const auto separator = std::find(fields.begin(), fields.end(), "-");
        if (fields.size() < 7 || separator == fields.end() || separator + 2 >= fields.end()) {
            throw std::runtime_error("mountinfo contains an invalid record");
        }
        mounts.push_back({
            decode_mount_field(fields[4]),
            *(separator + 1),
            decode_mount_field(*(separator + 2)),
            fields[2],
        });
    }

    const mount_record * selected = nullptr;
    for (const mount_record & mount : mounts) {
        if (path_is_within(existing, mount.mount_point) &&
                (selected == nullptr || mount.mount_point.string().size() > selected->mount_point.string().size())) {
            selected = &mount;
        }
    }
    if (selected == nullptr) {
        throw std::runtime_error(std::string(label) + " mount cannot be resolved: " + resolved.string());
    }

    std::string device_number = selected->device_number;
    if (device_number.substr(0, device_number.find(':')) == "0") {
        const std::string source = selected->source.substr(0, selected->source.find('['));
        const fs::path source_path = source;
        const fs::path source_dev = sys_class_block_root / source_path.filename() / "dev";
        if (source.rfind("/dev/", 0) != 0 || !fs::is_regular_file(source_dev)) {
            throw std::runtime_error(
                std::string(label) + " is not backed by a local block device: " + resolved.string());
        }
        device_number = trim(read_text(source_dev));
        static const std::regex device_number_pattern("^[0-9]+:[0-9]+$");
        if (!std::regex_match(device_number, device_number_pattern)) {
            throw std::runtime_error(std::string(label) + " backing device identity is invalid");
        }
    }

    const fs::path device_link = sys_dev_block_root / device_number;
    if (!fs::exists(device_link)) {
        throw std::runtime_error(
            std::string(label) + " is not backed by a resolvable block device: " + resolved.string());
    }
    const fs::path block_device = fs::canonical(device_link);
    fs::path rotational_path;
    for (fs::path current = block_device; !current.empty(); current = current.parent_path()) {
        const fs::path candidate = current / "queue" / "rotational";
        if (fs::is_regular_file(candidate)) {
            rotational_path = candidate;
            break;
        }
        if (current == current.parent_path()) {
            break;
        }
    }
    if (rotational_path.empty()) {
        throw std::runtime_error(
            std::string(label) + " block device rotational state cannot be resolved: " + block_device.string());
    }
    if (trim(read_text(rotational_path)) != "0") {
        throw std::runtime_error(std::string(label) + " must use non-rotational storage: " + resolved.string());
    }

    static const std::regex nvme_pattern("^nvme[0-9]+(c[0-9]+)?n[0-9]+$");
    std::string nvme_device;
    for (const fs::path & component : block_device) {
        const std::string name = component.string();
        if (std::regex_match(name, nvme_pattern)) {
            nvme_device = name;
        }
    }
    if (nvme_device.empty()) {
        throw std::runtime_error(
            std::string(label) + " must use an NVMe block device: " + block_device.string());
    }

    return {
        resolved,
        existing,
        selected->mount_point.string(),
        selected->filesystem_type,
        selected->source,
        device_number,
        block_device,
        nvme_device,
    };
}

static inline std::map<std::string, uint64_t> read_kfd_properties(const fs::path & path) {
    std::map<std::string, uint64_t> result;
    std::istringstream input(read_text(path));
    std::string key;
    uint64_t value = 0;
    while (input >> key >> value) {
        result[key] = value;
        std::string rest;
        std::getline(input, rest);
    }
    return result;
}

static inline std::string gfx_architecture(uint64_t version) {
    const uint64_t major = version / 10000;
    const uint64_t minor = (version / 100) % 100;
    const uint64_t stepping = version % 100;
    if (major == 0 || minor > 9 || stepping > 9) {
        throw std::runtime_error("KFD gfx_target_version is invalid");
    }
    return "gfx" + std::to_string(major) + std::to_string(minor) + std::to_string(stepping);
}

static inline accelerator_attestation require_gfx1151_identity(
        const std::string & backend_device,
        const std::string & backend_description,
        const std::string & pci_device_id,
        const fs::path & kfd_nodes_root) {
    if (backend_device != "ROCm0") {
        throw std::runtime_error("selected execution device must be ROCm0");
    }
    if (backend_description.empty()) {
        throw std::runtime_error("selected ROCm device description is missing");
    }

    static const std::regex pci_pattern("^([0-9a-f]{4}):([0-9a-f]{2}):([0-9a-f]{2})\\.([0-7])$");
    std::smatch match;
    if (!std::regex_match(pci_device_id, match, pci_pattern)) {
        throw std::runtime_error("selected ROCm device PCI identity is missing or invalid");
    }
    const uint64_t domain = std::stoull(match[1].str(), nullptr, 16);
    const uint64_t bus = std::stoull(match[2].str(), nullptr, 16);
    const uint64_t slot = std::stoull(match[3].str(), nullptr, 16);
    const uint64_t function = std::stoull(match[4].str(), nullptr, 16);
    const uint64_t location_id = (bus << 8) | (slot << 3) | function;

    std::vector<accelerator_attestation> matches;
    if (!fs::is_directory(kfd_nodes_root)) {
        throw std::runtime_error("KFD topology is unavailable");
    }
    for (const fs::directory_entry & entry : fs::directory_iterator(kfd_nodes_root)) {
        if (!entry.is_directory()) {
            continue;
        }
        const fs::path properties_path = entry.path() / "properties";
        const fs::path gpu_id_path = entry.path() / "gpu_id";
        if (!fs::is_regular_file(properties_path) || !fs::is_regular_file(gpu_id_path)) {
            continue;
        }
        const std::map<std::string, uint64_t> properties = read_kfd_properties(properties_path);
        const auto domain_value = properties.find("domain");
        const auto location_value = properties.find("location_id");
        const auto gfx_value = properties.find("gfx_target_version");
        if (domain_value == properties.end() || location_value == properties.end() ||
                gfx_value == properties.end() || domain_value->second != domain ||
                location_value->second != location_id) {
            continue;
        }
        const uint64_t gpu_id = std::stoull(trim(read_text(gpu_id_path)));
        if (gpu_id == 0) {
            throw std::runtime_error("selected KFD topology node has no GPU identity");
        }
        matches.push_back({
            backend_device,
            backend_description,
            pci_device_id,
            entry.path().filename().string(),
            gpu_id,
            gfx_value->second,
            gfx_architecture(gfx_value->second),
        });
    }
    if (matches.size() != 1) {
        throw std::runtime_error("selected ROCm device does not map to exactly one KFD topology node");
    }
    if (matches[0].architecture != "gfx1151" || matches[0].gfx_target_version != 110501) {
        throw std::runtime_error(
            "selected ROCm device architecture must be gfx1151, found " + matches[0].architecture);
    }
    return matches[0];
}

static inline accelerator_attestation require_gfx1151_device(
        ggml_backend_dev_t device,
        const fs::path & kfd_nodes_root = "/sys/class/kfd/kfd/topology/nodes") {
    if (device == nullptr) {
        throw std::runtime_error("required ROCm device is unavailable");
    }
    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(device, &props);
    return require_gfx1151_identity(
        props.name == nullptr ? "" : props.name,
        props.description == nullptr ? "" : props.description,
        props.device_id == nullptr ? "" : props.device_id,
        kfd_nodes_root);
}

}
