#include "host-attestation.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

static void write_file(const fs::path & path, const std::string & value) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path);
    output << value;
    if (!output) {
        throw std::runtime_error("cannot write test fixture");
    }
}

template <typename Function>
static void require_failure(Function function, const std::string & expected) {
    try {
        function();
    } catch (const std::runtime_error & error) {
        if (std::string(error.what()).find(expected) == std::string::npos) {
            throw;
        }
        return;
    }
    throw std::runtime_error("expected failure containing: " + expected);
}

int main() {
    const fs::path root = fs::canonical(fs::temp_directory_path()) /
        ("dsv41-host-attestation-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        const fs::path xfs = root / "mnt" / "models";
        const fs::path btrfs = root / "home";
        const fs::path rotating = root / "rotating";
        const fs::path ram = root / "ram";
        const fs::path network = root / "network";
        const fs::path missing = root / "missing";
        const fs::path forbidden = root / "forbidden";
        fs::create_directories(xfs);
        fs::create_directories(btrfs);
        fs::create_directories(rotating);
        fs::create_directories(ram);
        fs::create_directories(network);
        fs::create_directories(missing);
        fs::create_directories(forbidden);
        write_file(xfs / "model.gguf", "model");
        write_file(btrfs / "corpus.txt", "corpus");
        write_file(rotating / "data", "data");
        write_file(ram / "data", "data");
        write_file(network / "data", "data");
        write_file(missing / "data", "data");
        fs::create_directory_symlink(ram, xfs / "escape");
        fs::create_directory_symlink(xfs, forbidden / "escape");
        fs::create_directory_symlink(btrfs, root / "tmp-link");
        dsv41::require_usable_directory(btrfs, "TMPDIR");
        require_failure(
            [&]() {
                dsv41::require_usable_directory(root / "tmp-link", "TMPDIR");
            },
            "must not be a symlink");
        require_failure(
            [&]() {
                dsv41::require_usable_directory(fs::path((root / "tmp-link").string() + "/"), "TMPDIR");
            },
            "must not be a symlink");
        require_failure(
            [&]() {
                dsv41::require_usable_directory(root / "tmp-link" / ".", "TMPDIR");
            },
            "must not be a symlink");
        fs::create_directories(btrfs / "child");
        require_failure(
            [&]() {
                dsv41::require_usable_directory(root / "tmp-link" / "child", "TMPDIR");
            },
            "must not be a symlink");

        const fs::path sys = root / "sys";
        const fs::path nvme1 = sys / "devices" / "pci" / "block" / "nvme1n1";
        const fs::path nvme0 = sys / "devices" / "pci" / "block" / "nvme0n1";
        const fs::path nvme0p3 = nvme0 / "nvme0n1p3";
        const fs::path sdb = sys / "devices" / "pci" / "block" / "sdb";
        const fs::path sdb1 = sdb / "sdb1";
        write_file(nvme1 / "queue" / "rotational", "0\n");
        write_file(nvme0 / "queue" / "rotational", "0\n");
        write_file(sdb / "queue" / "rotational", "1\n");
        fs::create_directories(nvme0p3);
        fs::create_directories(sdb1);
        fs::create_directories(sys / "dev" / "block");
        fs::create_directory_symlink(nvme1, sys / "dev" / "block" / "259:0");
        fs::create_directory_symlink(nvme0p3, sys / "dev" / "block" / "259:3");
        fs::create_directory_symlink(sdb1, sys / "dev" / "block" / "8:17");
        write_file(sys / "class" / "block" / "nvme0n1p3" / "dev", "259:3\n");

        const fs::path mountinfo = root / "mountinfo";
        write_file(
            mountinfo,
            "1 0 259:0 / " + fs::canonical(xfs).string() + " rw - xfs /dev/nvme1n1 rw\n" +
            "2 0 0:35 /home " + fs::canonical(btrfs).string() + " rw - btrfs /dev/nvme0n1p3[/home] rw\n" +
            "3 0 8:17 / " + fs::canonical(rotating).string() + " rw - ext4 /dev/sdb1 rw\n" +
            "4 0 0:42 / " + fs::canonical(ram).string() + " rw - tmpfs tmpfs rw\n" +
            "5 0 0:43 / " + fs::canonical(network).string() + " rw - nfs server:/share rw\n" +
            "6 0 240:1 / " + fs::canonical(missing).string() + " rw - ext4 /dev/missing rw\n");

        const dsv41::storage_attestation xfs_attestation =
            dsv41::require_nvme_path(xfs / "model.gguf", "model", mountinfo, sys / "dev" / "block",
                                     sys / "class" / "block");
        if (xfs_attestation.filesystem_type != "xfs" || xfs_attestation.nvme_device != "nvme1n1") {
            throw std::runtime_error("xfs NVMe attestation mismatch");
        }
        const dsv41::storage_attestation btrfs_attestation =
            dsv41::require_nvme_path(btrfs / "new" / "trace", "trace", mountinfo, sys / "dev" / "block",
                                     sys / "class" / "block");
        if (btrfs_attestation.filesystem_type != "btrfs" ||
                btrfs_attestation.device_number != "259:3" ||
                btrfs_attestation.nvme_device != "nvme0n1") {
            throw std::runtime_error("btrfs NVMe partition attestation mismatch");
        }
        require_failure(
            [&]() {
                dsv41::require_nvme_path(
                    root / "tmp-link" / "child", "symlink traversal", mountinfo,
                    sys / "dev" / "block", sys / "class" / "block");
            },
            "must not be a symlink");
        require_failure(
            [&]() {
                dsv41::require_nvme_path(
                    rotating / "data", "rotating", mountinfo, sys / "dev" / "block", sys / "class" / "block");
            },
            "non-rotational");
        require_failure(
            [&]() {
                dsv41::require_nvme_path(
                    ram / "data", "tmpfs", mountinfo, sys / "dev" / "block", sys / "class" / "block");
            },
            "local block device");
        require_failure(
            [&]() {
                dsv41::require_nvme_path(
                    network / "data", "network", mountinfo, sys / "dev" / "block", sys / "class" / "block");
            },
            "local block device");
        require_failure(
            [&]() {
                dsv41::require_nvme_path(
                    missing / "data", "missing", mountinfo, sys / "dev" / "block", sys / "class" / "block");
            },
            "resolvable block device");
        require_failure(
            [&]() {
                dsv41::require_nvme_path(
                    xfs / "escape" / "data", "symlink escape", mountinfo, sys / "dev" / "block",
                    sys / "class" / "block");
            },
            "must not be a symlink");
        require_failure(
            [&]() {
                dsv41::require_nvme_path(
                    "/mnt/bigspace/model.gguf", "forbidden", mountinfo, sys / "dev" / "block",
                    sys / "class" / "block");
            },
            "/mnt/bigspace");
        require_failure(
            [&]() {
                dsv41::require_nvme_path(
                    forbidden / "escape" / "model.gguf", "forbidden symlink", mountinfo,
                    sys / "dev" / "block", sys / "class" / "block", forbidden);
            },
            "must not use");

        const fs::path kfd = root / "kfd";
        write_file(
            kfd / "1" / "properties",
            "domain 0\nlocation_id 50688\ngfx_target_version 110501\n");
        write_file(kfd / "1" / "gpu_id", "1234\n");
        const dsv41::accelerator_attestation accelerator =
            dsv41::require_gfx1151_identity(
                "ROCm0", "AMD Radeon 8060S Graphics", "0000:c6:00.0", kfd);
        if (accelerator.architecture != "gfx1151" || accelerator.gfx_target_version != 110501) {
            throw std::runtime_error("gfx1151 attestation mismatch");
        }
        require_failure(
            [&]() {
                dsv41::require_gfx1151_identity(
                    "ROCm1", "AMD Radeon 8060S Graphics", "0000:c6:00.0", kfd);
            },
            "ROCm0");
        write_file(
            kfd / "1" / "properties",
            "domain 0\nlocation_id 50688\ngfx_target_version 110500\n");
        require_failure(
            [&]() {
                dsv41::require_gfx1151_identity(
                    "ROCm0", "AMD Radeon 8060S Graphics", "0000:c6:00.0", kfd);
            },
            "gfx1151");
        write_file(
            kfd / "1" / "properties",
            "domain 0\nlocation_id 50688\n");
        require_failure(
            [&]() {
                dsv41::require_gfx1151_identity(
                    "ROCm0", "AMD Radeon 8060S Graphics", "0000:c6:00.0", kfd);
            },
            "exactly one KFD");

        fs::remove_all(root);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "test-host-attestation: " << error.what() << '\n';
        std::error_code ec;
        fs::remove_all(root, ec);
        return 1;
    }
}
