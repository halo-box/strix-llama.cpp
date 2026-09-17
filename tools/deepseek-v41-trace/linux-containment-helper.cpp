#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cerrno>
#include <climits>
#include <cstddef>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/audit.h>
#include <linux/capability.h>
#include <linux/filter.h>
#include <linux/magic.h>
#include <linux/sched.h>
#include <linux/seccomp.h>
#include <linux/securebits.h>
#include <sched.h>
#include <poll.h>
#include <sys/mount.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/vfs.h>
#include <sys/wait.h>
#include <unistd.h>

#if !defined(SYS_clone3)
#define SYS_clone3 435
#endif
#if !defined(SYS_pidfd_send_signal)
#define SYS_pidfd_send_signal 424
#endif
#if !defined(SYS_pidfd_open)
#define SYS_pidfd_open 434
#endif
#if !defined(SYS_seccomp)
#define SYS_seccomp 317
#endif
#if !defined(SECCOMP_RET_KILL_PROCESS)
#define SECCOMP_RET_KILL_PROCESS SECCOMP_RET_KILL
#endif
#if !defined(PR_SET_PTRACER)
#define PR_SET_PTRACER 0x59616d61
#endif

extern char ** environ;
#endif

#ifndef DSV41_BUILD_REVISION
#define DSV41_BUILD_REVISION "unknown"
#endif

namespace {

struct options {
    int protocol_fd = -1;
    int expected_parent = -1;
    std::string exec_path;
    std::vector<int> keep_fds;
    std::vector<char *> target_argv;
};

int parse_positive_int(const char * value, const char * label) {
    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed <= 0 ||
            parsed > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("invalid ") + label);
    }
    return static_cast<int>(parsed);
}

options parse_options(int argc, char ** argv) {
    options result;
    int index = 1;
    while (index < argc && std::strcmp(argv[index], "--") != 0) {
        if (std::strcmp(argv[index], "--protocol-fd") == 0 && index + 1 < argc) {
            result.protocol_fd = parse_positive_int(argv[index + 1], "protocol descriptor");
            index += 2;
        } else if (std::strcmp(argv[index], "--expected-parent") == 0 && index + 1 < argc) {
            result.expected_parent = parse_positive_int(argv[index + 1], "expected parent");
            index += 2;
        } else if (std::strcmp(argv[index], "--exec-path") == 0 && index + 1 < argc) {
            result.exec_path = argv[index + 1];
            index += 2;
        } else if (std::strcmp(argv[index], "--keep-fd") == 0 && index + 1 < argc) {
            result.keep_fds.push_back(parse_positive_int(argv[index + 1], "retained descriptor"));
            index += 2;
        } else {
            throw std::runtime_error("unknown containment helper argument");
        }
    }
    if (index >= argc || std::strcmp(argv[index], "--") != 0) {
        throw std::runtime_error("containment helper target separator is missing");
    }
    ++index;
    if (result.protocol_fd < 0 || result.expected_parent < 0 || result.exec_path.empty() ||
            index >= argc) {
        throw std::runtime_error("containment helper arguments are incomplete");
    }
    for (; index < argc; ++index) {
        result.target_argv.push_back(argv[index]);
    }
    result.target_argv.push_back(nullptr);
    return result;
}

#if defined(__linux__)

constexpr uint32_t DIAGNOSTIC_MAGIC = 0x44535634U;
constexpr uint16_t DIAGNOSTIC_VERSION = 1;
constexpr size_t DIAGNOSTIC_STAGE_CAPACITY = 48;

int query_supplementary_group_count(int * error_number) noexcept {
    errno = 0;
    const int count = getgroups(0, nullptr);
    *error_number = count < 0 ? errno : 0;
    return count;
}

void require_zero_supplementary_groups(const char * context) {
    int error_number = 0;
    const int count = query_supplementary_group_count(&error_number);
    if (count < 0) {
        throw std::runtime_error(
            std::string("cannot query ") + context + " supplementary groups: " +
            std::strerror(error_number));
    }
    if (count != 0) {
        throw std::runtime_error(
            std::string(context) + " requires zero supplementary groups; found " +
            std::to_string(count));
    }
}

struct failure_diagnostic {
    uint32_t magic;
    uint16_t version;
    uint16_t stage_size;
    int32_t error_number;
    char stage[DIAGNOSTIC_STAGE_CAPACITY];
};

static_assert(sizeof(failure_diagnostic) <= PIPE_BUF);

void close_checked(int fd, const char * label);

void report_failure(int fd, const char * stage, int error_number) noexcept {
    const int saved_errno = errno;
    failure_diagnostic record {};
    record.magic = DIAGNOSTIC_MAGIC;
    record.version = DIAGNOSTIC_VERSION;
    record.error_number = error_number;
    while (record.stage_size < DIAGNOSTIC_STAGE_CAPACITY &&
            stage[record.stage_size] != '\0') {
        record.stage[record.stage_size] = stage[record.stage_size];
        ++record.stage_size;
    }
    ssize_t written;
    do {
        written = write(fd, &record, sizeof(record));
    } while (written < 0 && errno == EINTR);
    errno = saved_errno;
}

[[noreturn]] void fail_stage(
        int diagnostic_fd,
        const char * stage,
        int error_number,
        int exit_code = 125) noexcept {
    report_failure(diagnostic_fd, stage, error_number);
    _exit(exit_code);
}

std::string receive_failure_diagnostics(int fd) {
    std::string result;
    for (size_t count = 0; count < 8; ++count) {
        failure_diagnostic record {};
        ssize_t size;
        do {
            size = read(fd, &record, sizeof(record));
        } while (size < 0 && errno == EINTR);
        if (size < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            break;
        }
        if (size == 0) {
            break;
        }
        if (size != static_cast<ssize_t>(sizeof(record)) ||
                record.magic != DIAGNOSTIC_MAGIC ||
                record.version != DIAGNOSTIC_VERSION ||
                record.stage_size == 0 ||
                record.stage_size > DIAGNOSTIC_STAGE_CAPACITY) {
            if (!result.empty()) {
                result += "; ";
            }
            result += "invalid setup diagnostic";
            continue;
        }
        if (!result.empty()) {
            result += "; ";
        }
        result += "stage=";
        result.append(record.stage, record.stage_size);
        result += " errno=" + std::to_string(record.error_number);
        if (record.error_number != 0) {
            result += " (";
            result += std::strerror(record.error_number);
            result += ")";
        }
    }
    return result;
}

[[noreturn]] void throw_setup_failure(int diagnostic_fd, const char * fallback) {
    const std::string diagnostic = receive_failure_diagnostics(diagnostic_fd);
    const int close_result = close(diagnostic_fd);
    const int close_error = close_result == 0 ? 0 : errno;
    std::string message = fallback;
    if (!diagnostic.empty()) {
        message += ": " + diagnostic;
    }
    if (close_error != 0) {
        message += "; diagnostics-close errno=" + std::to_string(close_error);
        message += " (";
        message += std::strerror(close_error);
        message += ")";
    }
    throw std::runtime_error(message);
}

bool retained_fd(
        int fd,
        const std::vector<int> & keep_fds,
        int extra_fd,
        int second_extra_fd = -1) {
    if (fd >= 0 && fd <= STDERR_FILENO) {
        return true;
    }
    if (fd == extra_fd || fd == second_extra_fd) {
        return true;
    }
    for (int keep_fd : keep_fds) {
        if (fd == keep_fd) {
            return true;
        }
    }
    return false;
}

void close_checked(int fd, const char * label) {
    if (close(fd) != 0) {
        throw std::runtime_error(std::string("cannot close ") + label);
    }
}

void close_unneeded_fds(
        const std::vector<int> & keep_fds,
        int extra_fd,
        int second_extra_fd = -1) {
    DIR * directory = opendir("/proc/self/fd");
    if (directory == nullptr) {
        throw std::runtime_error("cannot open procfs descriptor directory");
    }
    const int directory_fd = dirfd(directory);
    while (dirent * entry = readdir(directory)) {
        char * end = nullptr;
        const long parsed = std::strtol(entry->d_name, &end, 10);
        if (end == entry->d_name || *end != '\0' || parsed < 0 ||
                parsed > std::numeric_limits<int>::max()) {
            continue;
        }
        const int fd = static_cast<int>(parsed);
        if (fd != directory_fd &&
                !retained_fd(fd, keep_fds, extra_fd, second_extra_fd)) {
            close_checked(fd, "unneeded descriptor");
        }
    }
    if (closedir(directory) != 0) {
        throw std::runtime_error("cannot close procfs descriptor directory");
    }
}

void verify_private_procfs() {
    struct statfs filesystem {};
    if (statfs("/proc", &filesystem) != 0) {
        throw std::runtime_error("cannot inspect private procfs");
    }
    if (static_cast<unsigned long>(filesystem.f_type) !=
            static_cast<unsigned long>(PROC_SUPER_MAGIC)) {
        throw std::runtime_error("private procfs has an unexpected filesystem type");
    }
    char self_target[32] {};
    const ssize_t size = readlink("/proc/self", self_target, sizeof(self_target));
    if (size != 1 || self_target[0] != '1') {
        throw std::runtime_error("private procfs is not bound to the target PID namespace");
    }
}

void require_initial_signal_state() {
    sigset_t mask;
    if (sigprocmask(SIG_SETMASK, nullptr, &mask) != 0) {
        throw std::runtime_error("cannot read containment helper signal mask");
    }
    for (int signal_number = 1; signal_number < NSIG; ++signal_number) {
        if (signal_number == SIGKILL || signal_number == SIGSTOP) {
            continue;
        }
        struct sigaction action {};
        if (sigaction(signal_number, nullptr, &action) != 0) {
            if (errno == EINVAL) {
                continue;
            }
            throw std::runtime_error("cannot read containment helper signal disposition");
        }
        if (sigismember(&mask, signal_number) != 1) {
            throw std::runtime_error("containment helper inherited an unblocked signal");
        }
        if (action.sa_handler != SIG_DFL) {
            throw std::runtime_error("containment helper inherited a signal handler");
        }
    }
}

void unblock_default_signals() {
    sigset_t empty;
    sigemptyset(&empty);
    if (sigprocmask(SIG_SETMASK, &empty, nullptr) != 0) {
        throw std::runtime_error("cannot unblock containment helper signals");
    }
}

void set_parent_death(pid_t expected_parent) {
    if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0) {
        throw std::runtime_error("cannot set containment parent-death signal");
    }
    if (getppid() != expected_parent) {
        throw std::runtime_error("containment parent changed before lifecycle binding");
    }
}

void require_parent_death(pid_t expected_parent) {
    int parent_signal = 0;
    if (prctl(PR_GET_PDEATHSIG, &parent_signal) != 0 ||
            parent_signal != SIGKILL || getppid() != expected_parent) {
        throw std::runtime_error("containment parent-death binding changed");
    }
}

void write_all(int fd, const void * data, size_t size) {
    const char * cursor = static_cast<const char *>(data);
    while (size > 0) {
        const ssize_t written = write(fd, cursor, size);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error("containment protocol write failed");
        }
        cursor += written;
        size -= static_cast<size_t>(written);
    }
}

void write_mapping_file(int process_directory, const char * name, const std::string & content) {
    const int fd = openat(process_directory, name, O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
        throw std::runtime_error(std::string("cannot open namespace ") + name);
    }
    write_all(fd, content.data(), content.size());
    close_checked(fd, "namespace mapping descriptor");
}

std::string receive_packet(int fd) {
    char buffer[128];
    const ssize_t size = recv(fd, buffer, sizeof(buffer), 0);
    if (size <= 0) {
        throw std::runtime_error("containment protocol closed");
    }
    return std::string(buffer, static_cast<size_t>(size));
}

void send_packet(int fd, const char * packet) {
    const size_t size = std::strlen(packet);
    if (send(fd, packet, size, MSG_NOSIGNAL) != static_cast<ssize_t>(size)) {
        throw std::runtime_error("containment protocol send failed");
    }
}

void send_descriptor(int fd, const char * packet, int descriptor) {
    const size_t packet_size = std::strlen(packet);
    iovec vector {const_cast<char *>(packet), packet_size};
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))] {};
    msghdr message {};
    message.msg_iov = &vector;
    message.msg_iovlen = 1;
    message.msg_control = control;
    message.msg_controllen = sizeof(control);
    cmsghdr * header = CMSG_FIRSTHDR(&message);
    header->cmsg_level = SOL_SOCKET;
    header->cmsg_type = SCM_RIGHTS;
    header->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(header), &descriptor, sizeof(descriptor));
    if (sendmsg(fd, &message, MSG_NOSIGNAL) != static_cast<ssize_t>(packet_size)) {
        throw std::runtime_error("cannot send containment descriptor");
    }
}

int receive_descriptor(int fd, const char * expected_packet) {
    char payload[32] {};
    iovec vector {payload, sizeof(payload)};
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))] {};
    msghdr message {};
    message.msg_iov = &vector;
    message.msg_iovlen = 1;
    message.msg_control = control;
    message.msg_controllen = sizeof(control);
    const ssize_t size = recvmsg(fd, &message, 0);
    if (size != static_cast<ssize_t>(std::strlen(expected_packet)) ||
            std::memcmp(payload, expected_packet, static_cast<size_t>(size)) != 0 ||
            (message.msg_flags & (MSG_CTRUNC | MSG_TRUNC)) != 0) {
        throw std::runtime_error("containment descriptor packet is invalid");
    }
    cmsghdr * header = CMSG_FIRSTHDR(&message);
    if (header == nullptr ||
            header->cmsg_level != SOL_SOCKET ||
            header->cmsg_type != SCM_RIGHTS ||
            header->cmsg_len != CMSG_LEN(sizeof(int)) ||
            CMSG_NXTHDR(&message, header) != nullptr) {
        throw std::runtime_error("containment descriptor rights are invalid");
    }
    int descriptor = -1;
    std::memcpy(&descriptor, CMSG_DATA(header), sizeof(descriptor));
    if (descriptor < 0 || fcntl(descriptor, F_SETFD, FD_CLOEXEC) != 0) {
        if (descriptor >= 0) {
            close(descriptor);
        }
        throw std::runtime_error("cannot retain containment descriptor");
    }
    return descriptor;
}

bool pidfd_has_exited(int pidfd) {
    pollfd descriptor {pidfd, POLLIN, 0};
    const int result = poll(&descriptor, 1, 0);
    if (result < 0) {
        throw std::runtime_error("cannot query containment pidfd");
    }
    return result > 0 && (descriptor.revents & (POLLIN | POLLHUP | POLLERR)) != 0;
}

int wait_status_exit_code(int status) {
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return 125;
}

void make_isolated_session() {
    if (setsid() < 0 || getsid(0) != getpid() || getpgrp() != getpid()) {
        throw std::runtime_error("cannot isolate containment session");
    }
}

void protect_namespace_init() {
    make_isolated_session();
    if (prctl(PR_SET_DUMPABLE, 0) != 0 ||
            prctl(PR_SET_PTRACER, 0) != 0 ||
            prctl(PR_GET_DUMPABLE) != 0) {
        throw std::runtime_error("cannot protect target namespace init");
    }
    require_parent_death(0);
}

int read_cap_last_cap() {
    const int fd = open("/proc/sys/kernel/cap_last_cap", O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
        throw std::runtime_error("cannot open kernel capability limit");
    }
    char buffer[32] {};
    const ssize_t size = read(fd, buffer, sizeof(buffer) - 1);
    const int saved_errno = errno;
    close_checked(fd, "kernel capability limit descriptor");
    if (size <= 0) {
        errno = saved_errno;
        throw std::runtime_error("cannot read kernel capability limit");
    }
    char * end = nullptr;
    errno = 0;
    const long result = std::strtol(buffer, &end, 10);
    if (errno != 0 || end == buffer || (*end != '\n' && *end != '\0') ||
            result < 0 || result > 63) {
        throw std::runtime_error("kernel capability limit is invalid");
    }
    return static_cast<int>(result);
}

uint32_t seccomp_audit_arch() {
#if defined(__x86_64__)
    return AUDIT_ARCH_X86_64;
#elif defined(__aarch64__)
    return AUDIT_ARCH_AARCH64;
#else
#error "Unsupported Linux architecture for DeepSeek V4.1 containment"
#endif
}

void filter_statement(std::vector<sock_filter> & filter, uint16_t code, uint32_t value) {
    filter.push_back(sock_filter {code, 0, 0, value});
}

void filter_jump(
        std::vector<sock_filter> & filter,
        uint16_t code,
        uint32_t value,
        uint8_t on_true,
        uint8_t on_false) {
    filter.push_back(sock_filter {code, on_true, on_false, value});
}

void filter_pid_argument(
        std::vector<sock_filter> & filter,
        int syscall_number,
        unsigned argument,
        uint32_t denied_result) {
    filter_statement(filter, BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr));
    filter_jump(filter, BPF_JMP | BPF_JEQ | BPF_K, static_cast<uint32_t>(syscall_number), 0, 4);
    filter_statement(
        filter,
        BPF_LD | BPF_W | BPF_ABS,
        static_cast<uint32_t>(offsetof(seccomp_data, args) + argument * sizeof(uint64_t)));
    filter_jump(filter, BPF_JMP | BPF_JGE | BPF_K, 0x80000000U, 1, 0);
    filter_jump(filter, BPF_JMP | BPF_JGT | BPF_K, 1, 1, 0);
    filter_statement(filter, BPF_RET | BPF_K, denied_result);
}

int install_target_seccomp_listener() {
    std::vector<int> denied_syscalls;
#define DSV41_DENY_SYSCALL(name) denied_syscalls.push_back(__NR_##name)
#if defined(__NR_ptrace)
    DSV41_DENY_SYSCALL(ptrace);
#endif
#if defined(__NR_process_vm_readv)
    DSV41_DENY_SYSCALL(process_vm_readv);
#endif
#if defined(__NR_process_vm_writev)
    DSV41_DENY_SYSCALL(process_vm_writev);
#endif
#if defined(__NR_pidfd_send_signal)
    DSV41_DENY_SYSCALL(pidfd_send_signal);
#endif
#if defined(__NR_setuid)
    DSV41_DENY_SYSCALL(setuid);
#endif
#if defined(__NR_setgid)
    DSV41_DENY_SYSCALL(setgid);
#endif
#if defined(__NR_setreuid)
    DSV41_DENY_SYSCALL(setreuid);
#endif
#if defined(__NR_setregid)
    DSV41_DENY_SYSCALL(setregid);
#endif
#if defined(__NR_setresuid)
    DSV41_DENY_SYSCALL(setresuid);
#endif
#if defined(__NR_setresgid)
    DSV41_DENY_SYSCALL(setresgid);
#endif
#if defined(__NR_setfsuid)
    DSV41_DENY_SYSCALL(setfsuid);
#endif
#if defined(__NR_setfsgid)
    DSV41_DENY_SYSCALL(setfsgid);
#endif
#if defined(__NR_setgroups)
    DSV41_DENY_SYSCALL(setgroups);
#endif
#if defined(__NR_capset)
    DSV41_DENY_SYSCALL(capset);
#endif
#if defined(__NR_setpgid)
    DSV41_DENY_SYSCALL(setpgid);
#endif
#if defined(__NR_setsid)
    DSV41_DENY_SYSCALL(setsid);
#endif
#if defined(__NR_setns)
    DSV41_DENY_SYSCALL(setns);
#endif
#if defined(__NR_unshare)
    DSV41_DENY_SYSCALL(unshare);
#endif
#undef DSV41_DENY_SYSCALL

    const int denied_prctl[] = {
        PR_SET_PDEATHSIG,
        PR_SET_DUMPABLE,
        PR_SET_PTRACER,
        PR_SET_SECUREBITS,
        PR_SET_NO_NEW_PRIVS,
        PR_CAPBSET_DROP,
        PR_CAP_AMBIENT,
    };
    constexpr uint32_t denied_result = SECCOMP_RET_USER_NOTIF;
    std::vector<sock_filter> filter;
    filter_statement(filter, BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, arch));
    filter_jump(filter, BPF_JMP | BPF_JEQ | BPF_K, seccomp_audit_arch(), 1, 0);
    filter_statement(filter, BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS);
#if defined(__NR_kill)
    filter_pid_argument(filter, __NR_kill, 0, denied_result);
#endif
#if defined(__NR_tkill)
    filter_pid_argument(filter, __NR_tkill, 0, denied_result);
#endif
#if defined(__NR_tgkill)
    filter_pid_argument(filter, __NR_tgkill, 0, denied_result);
    filter_pid_argument(filter, __NR_tgkill, 1, denied_result);
#endif
#if defined(__NR_rt_sigqueueinfo)
    filter_pid_argument(filter, __NR_rt_sigqueueinfo, 0, denied_result);
#endif
#if defined(__NR_rt_tgsigqueueinfo)
    filter_pid_argument(filter, __NR_rt_tgsigqueueinfo, 0, denied_result);
    filter_pid_argument(filter, __NR_rt_tgsigqueueinfo, 1, denied_result);
#endif
    filter_statement(filter, BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr));
    filter_jump(
        filter,
        BPF_JMP | BPF_JEQ | BPF_K,
        __NR_prctl,
        0,
        static_cast<uint8_t>(2 + 2 * (sizeof(denied_prctl) / sizeof(denied_prctl[0]))));
    filter_statement(filter, BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args));
    for (int operation : denied_prctl) {
        filter_jump(filter, BPF_JMP | BPF_JEQ | BPF_K, static_cast<uint32_t>(operation), 0, 1);
        filter_statement(filter, BPF_RET | BPF_K, denied_result);
    }
    filter_statement(filter, BPF_RET | BPF_K, SECCOMP_RET_ALLOW);
    filter_statement(filter, BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr));
    for (int syscall_number : denied_syscalls) {
        filter_jump(filter, BPF_JMP | BPF_JEQ | BPF_K, static_cast<uint32_t>(syscall_number), 0, 1);
        filter_statement(filter, BPF_RET | BPF_K, denied_result);
    }
    filter_statement(filter, BPF_RET | BPF_K, SECCOMP_RET_ALLOW);
    if (filter.size() > std::numeric_limits<unsigned short>::max()) {
        throw std::runtime_error("target seccomp filter is too large");
    }
    sock_fprog program {
        static_cast<unsigned short>(filter.size()),
        filter.data(),
    };
    const int listener = static_cast<int>(
        syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, SECCOMP_FILTER_FLAG_NEW_LISTENER, &program));
    if (listener < 0 || fcntl(listener, F_SETFD, FD_CLOEXEC) != 0) {
        if (listener >= 0) {
            close(listener);
        }
        throw std::runtime_error("cannot install target seccomp filter");
    }
    return listener;
}

void require_eperm(long result, const char * label) {
    if (result != -1 || errno != EPERM) {
        throw std::runtime_error(std::string("target isolation did not deny ") + label);
    }
}

int drop_target_privileges() {
    constexpr uid_t target_uid = 65534;
    constexpr gid_t target_gid = 65534;
    constexpr unsigned long securebits =
        SECBIT_NOROOT |
        SECBIT_NOROOT_LOCKED |
        SECBIT_NO_SETUID_FIXUP |
        SECBIT_NO_SETUID_FIXUP_LOCKED |
        SECBIT_KEEP_CAPS_LOCKED |
        SECBIT_NO_CAP_AMBIENT_RAISE |
        SECBIT_NO_CAP_AMBIENT_RAISE_LOCKED;
    const int cap_last_cap = read_cap_last_cap();
    if (prctl(PR_SET_SECUREBITS, securebits) != 0 ||
            setresgid(target_gid, target_gid, target_gid) != 0 ||
            setresuid(target_uid, target_uid, target_uid) != 0) {
        throw std::runtime_error("cannot enter target non-root credentials");
    }
    for (int capability = 0; capability <= cap_last_cap; ++capability) {
        if (prctl(PR_CAPBSET_DROP, capability, 0, 0, 0) != 0) {
            throw std::runtime_error("cannot drop target capability bounding set");
        }
    }
    if (prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL, 0, 0, 0) != 0) {
        throw std::runtime_error("cannot clear target ambient capabilities");
    }
    __user_cap_header_struct header {};
    header.version = _LINUX_CAPABILITY_VERSION_3;
    __user_cap_data_struct data[2] {};
    if (syscall(SYS_capset, &header, data) != 0 ||
            prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        throw std::runtime_error("cannot clear target capabilities");
    }
    __user_cap_data_struct verified[2] {};
    if (syscall(SYS_capget, &header, verified) != 0) {
        throw std::runtime_error("cannot verify target capabilities");
    }
    for (const __user_cap_data_struct & entry : verified) {
        if (entry.effective != 0 || entry.permitted != 0 || entry.inheritable != 0) {
            throw std::runtime_error("target capability sets are not empty");
        }
    }
    for (int capability = 0; capability <= cap_last_cap; ++capability) {
        if (prctl(PR_CAPBSET_READ, capability, 0, 0, 0) != 0 ||
                prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_IS_SET, capability, 0, 0) != 0) {
            throw std::runtime_error("target retained a bounded or ambient capability");
        }
    }
    if (getuid() != target_uid || geteuid() != target_uid ||
            getgid() != target_gid || getegid() != target_gid ||
            getgroups(0, nullptr) != 0 ||
            prctl(PR_GET_SECUREBITS) != static_cast<int>(securebits) ||
            prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) != 1 ||
            getsid(0) != getpid() || getpgrp() != getpid()) {
        throw std::runtime_error("target privilege isolation verification failed");
    }
    const int listener = install_target_seccomp_listener();
    if (prctl(PR_GET_SECCOMP, 0, 0, 0, 0) != SECCOMP_MODE_FILTER) {
        close(listener);
        throw std::runtime_error("target seccomp verification failed");
    }
    return listener;
}

void verify_target_attack_denials() {
    errno = 0;
    require_eperm(syscall(SYS_ptrace, PTRACE_ATTACH, 1, nullptr, nullptr), "ptrace");
#if defined(SYS_process_vm_readv)
    errno = 0;
    require_eperm(syscall(SYS_process_vm_readv, 1, nullptr, 0, nullptr, 0, 0), "process_vm_readv");
#endif
#if defined(SYS_process_vm_writev)
    errno = 0;
    require_eperm(syscall(SYS_process_vm_writev, 1, nullptr, 0, nullptr, 0, 0), "process_vm_writev");
#endif
    errno = 0;
    require_eperm(kill(1, SIGSTOP), "namespace init signaling");
    errno = 0;
    require_eperm(kill(-getpgrp(), 0), "process-group signaling");
    if (kill(getpid(), 0) != 0) {
        throw std::runtime_error("target isolation blocked local signaling");
    }
    errno = 0;
    require_eperm(prctl(PR_SET_PDEATHSIG, 0), "parent-death mutation");
    errno = 0;
    require_eperm(setresuid(0, 0, 0), "UID regain");
    errno = 0;
    require_eperm(setpgid(0, 0), "process-group mutation");
    errno = 0;
    require_eperm(setsid(), "session mutation");
    __user_cap_header_struct header {};
    header.version = _LINUX_CAPABILITY_VERSION_3;
    __user_cap_data_struct data[2] {};
    errno = 0;
    require_eperm(syscall(SYS_capset, &header, data), "capability regain");
}

struct seccomp_notification {
    bool present;
    uint64_t id;
    seccomp_data data;
};

seccomp_notification receive_seccomp_notification(int listener, bool nonblocking) {
    seccomp_notif_sizes sizes {};
    if (syscall(SYS_seccomp, SECCOMP_GET_NOTIF_SIZES, 0, &sizes) != 0 ||
            sizes.seccomp_notif < sizeof(seccomp_notif) ||
            sizes.seccomp_notif_resp < sizeof(seccomp_notif_resp)) {
        throw std::runtime_error("cannot query target seccomp notification sizes");
    }
    std::vector<uint64_t> request_buffer(
        (sizes.seccomp_notif + sizeof(uint64_t) - 1) / sizeof(uint64_t));
    seccomp_notif * request = reinterpret_cast<seccomp_notif *>(request_buffer.data());
    if (ioctl(listener, SECCOMP_IOCTL_NOTIF_RECV, request) != 0) {
        if (nonblocking && (errno == EAGAIN || errno == ENOENT)) {
            return seccomp_notification {false, 0, {}};
        }
        throw std::runtime_error("cannot receive target seccomp notification");
    }
    if (request->flags != 0 ||
            ioctl(listener, SECCOMP_IOCTL_NOTIF_ID_VALID, &request->id) != 0) {
        throw std::runtime_error("target seccomp notification identity is invalid");
    }
    return seccomp_notification {true, request->id, request->data};
}

void deny_seccomp_notification(int listener, const seccomp_notification & notification) {
    seccomp_notif_resp response {};
    response.id = notification.id;
    response.error = -EPERM;
    if (ioctl(listener, SECCOMP_IOCTL_NOTIF_SEND, &response) != 0) {
        throw std::runtime_error("cannot deny target seccomp notification");
    }
}

seccomp_notification expect_seccomp_notification(
        int listener,
        int syscall_number,
        const char * label) {
    const seccomp_notification notification = receive_seccomp_notification(listener, false);
    if (!notification.present || notification.data.nr != syscall_number) {
        throw std::runtime_error(std::string("unexpected target isolation probe: ") + label);
    }
    deny_seccomp_notification(listener, notification);
    return notification;
}

void verify_target_isolation_probes(int listener) {
    expect_seccomp_notification(listener, SYS_ptrace, "ptrace");
#if defined(SYS_process_vm_readv)
    expect_seccomp_notification(listener, SYS_process_vm_readv, "process_vm_readv");
#endif
#if defined(SYS_process_vm_writev)
    expect_seccomp_notification(listener, SYS_process_vm_writev, "process_vm_writev");
#endif
    seccomp_notification notification = expect_seccomp_notification(listener, SYS_kill, "PID 1 signaling");
    if (notification.data.args[0] != 1 || notification.data.args[1] != SIGSTOP) {
        throw std::runtime_error("target PID 1 signal probe is invalid");
    }
    notification = expect_seccomp_notification(listener, SYS_kill, "process-group signaling");
    if (static_cast<int64_t>(notification.data.args[0]) >= 0) {
        throw std::runtime_error("target process-group signal probe is invalid");
    }
    notification = expect_seccomp_notification(listener, SYS_prctl, "parent-death mutation");
    if (notification.data.args[0] != PR_SET_PDEATHSIG) {
        throw std::runtime_error("target parent-death probe is invalid");
    }
    expect_seccomp_notification(listener, SYS_setresuid, "UID regain");
    expect_seccomp_notification(listener, SYS_setpgid, "process-group mutation");
    expect_seccomp_notification(listener, SYS_setsid, "session mutation");
    expect_seccomp_notification(listener, SYS_capset, "capability regain");
}

class namespace_owner {
public:
    namespace_owner(pid_t pid, int pidfd) : pid_(pid), pidfd_(pidfd) {
    }

    ~namespace_owner() {
        if (reaped_) {
            return;
        }
        if (pidfd_ >= 0) {
            syscall(SYS_pidfd_send_signal, pidfd_, SIGKILL, nullptr, 0);
        }
        while (waitpid(pid_, nullptr, 0) < 0 && errno == EINTR) {
        }
        if (pidfd_ >= 0) {
            close(pidfd_);
        }
    }

    int pidfd() const {
        return pidfd_;
    }

    int wait() {
        int status = 0;
        while (waitpid(pid_, &status, 0) < 0) {
            if (errno != EINTR) {
                throw std::runtime_error("cannot reap target PID namespace");
            }
        }
        reaped_ = true;
        return status;
    }

    void close_pidfd() {
        const int descriptor = pidfd_;
        pidfd_ = -1;
        if (close(descriptor) != 0) {
            throw std::runtime_error("cannot close namespace pidfd");
        }
    }

private:
    pid_t pid_;
    int pidfd_;
    bool reaped_ = false;
};

int wait_for_isolated_target(namespace_owner & target, int listener) {
    const int flags = fcntl(listener, F_GETFL);
    if (flags < 0 || fcntl(listener, F_SETFL, flags | O_NONBLOCK) != 0) {
        throw std::runtime_error("cannot configure target seccomp listener");
    }
    pollfd descriptors[2] {
        {listener, POLLIN, 0},
        {target.pidfd(), POLLIN, 0},
    };
    while (true) {
        const int result = poll(descriptors, 2, -1);
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error("cannot monitor isolated target");
        }
        if ((descriptors[0].revents & (POLLIN | POLLHUP | POLLERR)) != 0) {
            const seccomp_notification notification = receive_seccomp_notification(listener, true);
            if (notification.present) {
                throw std::runtime_error("target attempted a forbidden lifecycle operation");
            }
        }
        if ((descriptors[1].revents & (POLLIN | POLLHUP | POLLERR)) != 0) {
            const seccomp_notification notification = receive_seccomp_notification(listener, true);
            if (notification.present) {
                throw std::runtime_error("target attempted a forbidden lifecycle operation");
            }
            return target.wait();
        }
    }
}

[[noreturn]] void run_target_bootstrap(
        int mapping_fd,
        int ready_fd,
        int security_fd,
        int namespace_ready_fd,
        int diagnostic_fd,
        const options & config) {
    const char * stage = "target-parent-death";
    try {
        errno = 0;
        set_parent_death(1);
        stage = "target-session";
        errno = 0;
        make_isolated_session();
        stage = "target-namespace-ready-close";
        errno = 0;
        close_checked(namespace_ready_fd, "target namespace readiness descriptor");
        stage = "target-bound";
        errno = 0;
        write_all(ready_fd, "B", 1);
        stage = "target-mapping-read";
        char mapped = 0;
        ssize_t mapped_size;
        do {
            mapped_size = read(mapping_fd, &mapped, 1);
        } while (mapped_size < 0 && errno == EINTR);
        if (mapped_size != 1 || mapped != 'M') {
            fail_stage(diagnostic_fd, stage, mapped_size < 0 ? errno : 0);
        }
        stage = "target-mapping-close";
        errno = 0;
        close_checked(mapping_fd, "target mapping descriptor");
        stage = "target-groups-verify";
        int groups_error = 0;
        if (query_supplementary_group_count(&groups_error) != 0) {
            fail_stage(diagnostic_fd, stage, groups_error);
        }
        stage = "target-privilege-drop";
        errno = 0;
        const int listener = drop_target_privileges();
        stage = "target-filter-send";
        errno = 0;
        send_descriptor(security_fd, "FILTER", listener);
        stage = "target-listener-close";
        errno = 0;
        close_checked(listener, "target seccomp listener");
        stage = "target-isolation-probes";
        errno = 0;
        verify_target_attack_denials();
        stage = "target-verified-send";
        errno = 0;
        send_packet(security_fd, "VERIFIED");
        stage = "target-go-read";
        errno = 0;
        if (receive_packet(security_fd) != "GO") {
            fail_stage(diagnostic_fd, stage, 0);
        }
        stage = "target-security-close";
        errno = 0;
        close_checked(security_fd, "target security descriptor");
        stage = "target-parent-death-verify";
        errno = 0;
        require_parent_death(1);
        stage = "target-fd-close";
        errno = 0;
        close_unneeded_fds(config.keep_fds, ready_fd, diagnostic_fd);
        stage = "target-isolation-ready";
        errno = 0;
        write_all(ready_fd, "I", 1);
        stage = "target-ready-close";
        errno = 0;
        close_checked(ready_fd, "target readiness descriptor");
        stage = "target-exec";
        errno = 0;
        execve(config.exec_path.c_str(), config.target_argv.data(), environ);
        fail_stage(diagnostic_fd, stage, errno, 127);
    } catch (...) {
        fail_stage(diagnostic_fd, stage, errno);
    }
}

[[noreturn]] void run_namespace_init(
        int release_fd,
        int ready_fd,
        int mapping_fd,
        int diagnostic_fd,
        int protocol_fd,
        int helper_pidfd,
        const options & config) {
    const char * stage = "namespace-parent-death";
    try {
        errno = 0;
        if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0) {
            fail_stage(diagnostic_fd, stage, errno);
        }
        stage = "namespace-parent-identity";
        errno = 0;
        if (getppid() != 0 || pidfd_has_exited(helper_pidfd)) {
            fail_stage(diagnostic_fd, stage, 0);
        }
        stage = "namespace-helper-pidfd-close";
        errno = 0;
        close_checked(helper_pidfd, "namespace helper pidfd");
        stage = "namespace-protocol-close";
        errno = 0;
        close_checked(protocol_fd, "namespace protocol descriptor");
        stage = "namespace-bound";
        errno = 0;
        write_all(ready_fd, "B", 1);
        stage = "namespace-mapping-read";
        char mapped = 0;
        ssize_t mapped_size;
        do {
            mapped_size = read(mapping_fd, &mapped, 1);
        } while (mapped_size < 0 && errno == EINTR);
        if (mapped_size != 1 || mapped != 'M') {
            fail_stage(diagnostic_fd, stage, mapped_size < 0 ? errno : 0);
        }
        stage = "namespace-mapping-close";
        errno = 0;
        close_checked(mapping_fd, "namespace mapping descriptor");
        stage = "namespace-groups-verify";
        int groups_error = 0;
        if (query_supplementary_group_count(&groups_error) != 0) {
            fail_stage(diagnostic_fd, stage, groups_error);
        }
        stage = "namespace-setresgid";
        errno = 0;
        if (setresgid(0, 0, 0) != 0) {
            fail_stage(diagnostic_fd, stage, errno);
        }
        stage = "namespace-setresuid";
        errno = 0;
        if (setresuid(0, 0, 0) != 0) {
            fail_stage(diagnostic_fd, stage, errno);
        }
        stage = "namespace-mount-private";
        errno = 0;
        if (mount(nullptr, "/", nullptr, MS_REC | MS_PRIVATE, nullptr) != 0) {
            fail_stage(diagnostic_fd, stage, errno);
        }
        stage = "namespace-unmount-proc";
        errno = 0;
        if (umount2("/proc", MNT_DETACH) != 0 && errno != EINVAL) {
            fail_stage(diagnostic_fd, stage, errno);
        }
        stage = "namespace-mount-proc";
        errno = 0;
        if (mount("proc", "/proc", "proc", MS_NOSUID | MS_NODEV | MS_NOEXEC, nullptr) != 0) {
            fail_stage(diagnostic_fd, stage, errno);
        }
        stage = "namespace-verify-proc";
        errno = 0;
        verify_private_procfs();
        stage = "namespace-protect-init";
        errno = 0;
        protect_namespace_init();
        stage = "namespace-ready";
        errno = 0;
        write_all(ready_fd, "R", 1);
        stage = "namespace-release-read";
        char release = 0;
        ssize_t received;
        do {
            received = read(release_fd, &release, 1);
        } while (received < 0 && errno == EINTR);
        if (received != 1 || release != 'X') {
            fail_stage(diagnostic_fd, stage, received < 0 ? errno : 0);
        }
        stage = "namespace-release-close";
        errno = 0;
        close_checked(release_fd, "namespace release descriptor");
        int target_mapping_pipe[2] {-1, -1};
        int target_ready_pipe[2] {-1, -1};
        int target_security[2] {-1, -1};
        stage = "namespace-target-mapping-pipe";
        errno = 0;
        if (pipe2(target_mapping_pipe, O_CLOEXEC) != 0) {
            fail_stage(diagnostic_fd, stage, errno);
        }
        stage = "namespace-target-ready-pipe";
        errno = 0;
        if (pipe2(target_ready_pipe, O_CLOEXEC) != 0) {
            fail_stage(diagnostic_fd, stage, errno);
        }
        stage = "namespace-target-security-socket";
        errno = 0;
        if (socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, target_security) != 0) {
            fail_stage(diagnostic_fd, stage, errno);
        }
        int target_pidfd = -1;
        clone_args target_arguments {};
        target_arguments.flags = CLONE_NEWUSER | CLONE_PIDFD;
        target_arguments.pidfd = reinterpret_cast<uintptr_t>(&target_pidfd);
        target_arguments.exit_signal = SIGCHLD;
        stage = "namespace-target-clone";
        errno = 0;
        const pid_t target_pid = static_cast<pid_t>(
            syscall(SYS_clone3, &target_arguments, sizeof(target_arguments)));
        if (target_pid < 0) {
            fail_stage(diagnostic_fd, stage, errno);
        }
        if (target_pid == 0) {
            stage = "target-handoff-close";
            errno = 0;
            close_checked(target_mapping_pipe[1], "target mapping writer");
            close_checked(target_ready_pipe[0], "target readiness reader");
            close_checked(target_security[0], "target security supervisor descriptor");
            run_target_bootstrap(
                target_mapping_pipe[0], target_ready_pipe[1],
                target_security[1], ready_fd, diagnostic_fd, config);
        }
        namespace_owner owned_target(target_pid, target_pidfd);
        stage = "namespace-target-parent-close";
        errno = 0;
        close_checked(target_mapping_pipe[0], "namespace target mapping reader");
        close_checked(target_ready_pipe[1], "namespace target readiness writer");
        close_checked(target_security[1], "namespace target security descriptor");
        stage = "namespace-target-bound-read";
        char target_ready = 0;
        ssize_t target_ready_size;
        do {
            target_ready_size = read(target_ready_pipe[0], &target_ready, 1);
        } while (target_ready_size < 0 && errno == EINTR);
        if (target_ready_size != 1 || target_ready != 'B') {
            fail_stage(diagnostic_fd, stage, target_ready_size < 0 ? errno : 0);
        }
        stage = "namespace-target-process-open";
        errno = 0;
        const std::string target_process_path = "/proc/" + std::to_string(target_pid);
        const int target_process_directory = open(
            target_process_path.c_str(), O_PATH | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
        if (target_process_directory < 0 || pidfd_has_exited(target_pidfd)) {
            fail_stage(
                diagnostic_fd, stage,
                target_process_directory < 0 ? errno : 0);
        }
        stage = "namespace-target-map-setgroups";
        errno = 0;
        write_mapping_file(target_process_directory, "setgroups", "deny\n");
        stage = "namespace-target-map-uid";
        errno = 0;
        write_mapping_file(target_process_directory, "uid_map", "65534 0 1\n");
        stage = "namespace-target-map-gid";
        errno = 0;
        write_mapping_file(target_process_directory, "gid_map", "65534 0 1\n");
        stage = "namespace-target-process-close";
        errno = 0;
        close_checked(target_process_directory, "target process directory");
        stage = "namespace-target-mapping-release";
        errno = 0;
        write_all(target_mapping_pipe[1], "M", 1);
        stage = "namespace-target-mapping-close";
        errno = 0;
        close_checked(target_mapping_pipe[1], "namespace target mapping writer");
        stage = "namespace-target-filter-receive";
        errno = 0;
        const int target_listener = receive_descriptor(target_security[0], "FILTER");
        stage = "namespace-target-probe-verify";
        errno = 0;
        verify_target_isolation_probes(target_listener);
        stage = "namespace-target-verified-read";
        errno = 0;
        if (receive_packet(target_security[0]) != "VERIFIED") {
            fail_stage(diagnostic_fd, stage, 0);
        }
        stage = "namespace-parent-death-verify";
        errno = 0;
        require_parent_death(0);
        stage = "namespace-identity-verify";
        errno = 0;
        if (prctl(PR_GET_DUMPABLE) != 0 ||
                getsid(0) != getpid() || getpgrp() != getpid()) {
            fail_stage(diagnostic_fd, stage, errno);
        }
        stage = "namespace-target-go";
        errno = 0;
        send_packet(target_security[0], "GO");
        stage = "namespace-target-security-close";
        errno = 0;
        close_checked(target_security[0], "namespace target security supervisor descriptor");
        stage = "namespace-target-isolation-read";
        do {
            target_ready_size = read(target_ready_pipe[0], &target_ready, 1);
        } while (target_ready_size < 0 && errno == EINTR);
        stage = "namespace-target-ready-close";
        errno = 0;
        close_checked(target_ready_pipe[0], "namespace target readiness reader");
        stage = "namespace-parent-death-reverify";
        errno = 0;
        require_parent_death(0);
        stage = "namespace-target-isolation-verify";
        errno = 0;
        if (prctl(PR_GET_DUMPABLE) != 0 ||
                getsid(0) != getpid() || getpgrp() != getpid() ||
                target_ready_size != 1 || target_ready != 'I') {
            fail_stage(diagnostic_fd, stage, 0);
        }
        stage = "namespace-isolation-ready";
        errno = 0;
        write_all(ready_fd, "I", 1);
        stage = "namespace-target-wait";
        errno = 0;
        const int target_status = wait_for_isolated_target(owned_target, target_listener);
        stage = "namespace-listener-close";
        errno = 0;
        close_checked(target_listener, "namespace target seccomp listener");
        stage = "namespace-target-pidfd-close";
        errno = 0;
        owned_target.close_pidfd();
        stage = "namespace-descendant-kill";
        errno = 0;
        if (kill(-1, SIGKILL) != 0 && errno != ESRCH) {
            fail_stage(diagnostic_fd, stage, errno);
        }
        stage = "namespace-descendant-reap";
        errno = 0;
        while (true) {
            const pid_t reaped = waitpid(-1, nullptr, 0);
            if (reaped > 0 || (reaped < 0 && errno == EINTR)) {
                continue;
            }
            if (reaped < 0 && errno == ECHILD) {
                break;
            }
            fail_stage(diagnostic_fd, stage, reaped < 0 ? errno : 0);
        }
        stage = "namespace-complete";
        errno = 0;
        write_all(ready_fd, "C", 1);
        stage = "namespace-ready-close";
        errno = 0;
        close_checked(ready_fd, "namespace readiness descriptor");
        _exit(wait_status_exit_code(target_status));
    } catch (...) {
        fail_stage(diagnostic_fd, stage, errno);
    }
}

int run_linux_helper(options config) {
    require_zero_supplementary_groups("containment launcher");
    require_initial_signal_state();
    set_parent_death(config.expected_parent);
    close_unneeded_fds(config.keep_fds, config.protocol_fd);
    unblock_default_signals();
    send_packet(config.protocol_fd, "READY");
    if (receive_packet(config.protocol_fd) != "PREPARE") {
        throw std::runtime_error("containment PREPARE packet is invalid");
    }

    int release_pipe[2] {-1, -1};
    if (pipe2(release_pipe, O_CLOEXEC) != 0) {
        throw std::runtime_error("cannot create namespace release pipe");
    }
    int ready_pipe[2] {-1, -1};
    if (pipe2(ready_pipe, O_CLOEXEC) != 0) {
        close(release_pipe[0]);
        close(release_pipe[1]);
        throw std::runtime_error("cannot create namespace readiness pipe");
    }
    int mapping_pipe[2] {-1, -1};
    if (pipe2(mapping_pipe, O_CLOEXEC) != 0) {
        close(release_pipe[0]);
        close(release_pipe[1]);
        close(ready_pipe[0]);
        close(ready_pipe[1]);
        throw std::runtime_error("cannot create namespace mapping pipe");
    }
    int diagnostic_pipe[2] {-1, -1};
    if (pipe2(diagnostic_pipe, O_CLOEXEC | O_NONBLOCK) != 0) {
        close(release_pipe[0]);
        close(release_pipe[1]);
        close(ready_pipe[0]);
        close(ready_pipe[1]);
        close(mapping_pipe[0]);
        close(mapping_pipe[1]);
        throw std::runtime_error("cannot create namespace diagnostic pipe");
    }
    const int helper_pidfd = static_cast<int>(syscall(SYS_pidfd_open, getpid(), 0));
    if (helper_pidfd < 0) {
        close(release_pipe[0]);
        close(release_pipe[1]);
        close(ready_pipe[0]);
        close(ready_pipe[1]);
        close(mapping_pipe[0]);
        close(mapping_pipe[1]);
        close(diagnostic_pipe[0]);
        close(diagnostic_pipe[1]);
        throw std::runtime_error("cannot open stable helper identity");
    }
    int namespace_pidfd = -1;
    clone_args arguments {};
    arguments.flags = CLONE_NEWUSER | CLONE_NEWPID | CLONE_NEWNS | CLONE_PIDFD;
    arguments.pidfd = reinterpret_cast<uintptr_t>(&namespace_pidfd);
    arguments.exit_signal = SIGCHLD;
    const pid_t namespace_init = static_cast<pid_t>(
        syscall(SYS_clone3, &arguments, sizeof(arguments)));
    if (namespace_init < 0) {
        close(release_pipe[0]);
        close(release_pipe[1]);
        close(ready_pipe[0]);
        close(ready_pipe[1]);
        close(mapping_pipe[0]);
        close(mapping_pipe[1]);
        close(diagnostic_pipe[0]);
        close(diagnostic_pipe[1]);
        close(helper_pidfd);
        throw std::runtime_error(std::string("cannot create target PID namespace: ") + std::strerror(errno));
    }
    if (namespace_init == 0) {
        if (close(release_pipe[1]) != 0) {
            fail_stage(
                diagnostic_pipe[1],
                "namespace-release-writer-close",
                errno);
        }
        if (close(ready_pipe[0]) != 0) {
            fail_stage(
                diagnostic_pipe[1],
                "namespace-ready-reader-close",
                errno);
        }
        if (close(mapping_pipe[1]) != 0) {
            fail_stage(
                diagnostic_pipe[1],
                "namespace-mapping-writer-close",
                errno);
        }
        if (close(diagnostic_pipe[0]) != 0) {
            fail_stage(
                diagnostic_pipe[1],
                "namespace-diagnostic-reader-close",
                errno);
        }
        run_namespace_init(
            release_pipe[0], ready_pipe[1], mapping_pipe[0], diagnostic_pipe[1],
            config.protocol_fd, helper_pidfd, config);
    }
    namespace_owner owned_namespace(namespace_init, namespace_pidfd);
    close_checked(release_pipe[0], "helper release reader");
    close_checked(ready_pipe[1], "helper readiness writer");
    close_checked(mapping_pipe[0], "helper mapping reader");
    close_checked(diagnostic_pipe[1], "helper diagnostic writer");
    char ready = 0;
    ssize_t ready_size;
    do {
        ready_size = read(ready_pipe[0], &ready, 1);
    } while (ready_size < 0 && errno == EINTR);
    if (ready_size != 1 || ready != 'B') {
        throw_setup_failure(
            diagnostic_pipe[0],
            "target PID namespace did not bind helper lifetime");
    }
    const std::string process_path = "/proc/" + std::to_string(namespace_init);
    const int process_directory = open(
        process_path.c_str(), O_PATH | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (process_directory < 0 || pidfd_has_exited(namespace_pidfd)) {
        if (process_directory >= 0) {
            close_checked(process_directory, "namespace process directory");
        }
        throw std::runtime_error("cannot retain target namespace mapping identity");
    }
    write_mapping_file(process_directory, "setgroups", "deny\n");
    write_mapping_file(
        process_directory, "uid_map",
        "0 " + std::to_string(getuid()) + " 1\n");
    write_mapping_file(
        process_directory, "gid_map",
        "0 " + std::to_string(getgid()) + " 1\n");
    close_checked(process_directory, "namespace process directory");
    write_all(mapping_pipe[1], "M", 1);
    close_checked(mapping_pipe[1], "helper mapping writer");
    do {
        ready_size = read(ready_pipe[0], &ready, 1);
    } while (ready_size < 0 && errno == EINTR);
    close_checked(helper_pidfd, "helper self pidfd");
    if (ready_size != 1 || ready != 'R') {
        throw_setup_failure(
            diagnostic_pipe[0],
            "target PID namespace setup did not complete");
    }
    send_descriptor(config.protocol_fd, "PREPARED", owned_namespace.pidfd());
    if (receive_packet(config.protocol_fd) != "EXEC") {
        throw std::runtime_error("containment EXEC packet is invalid");
    }
    write_all(release_pipe[1], "X", 1);
    close_checked(release_pipe[1], "helper release writer");
    do {
        ready_size = read(ready_pipe[0], &ready, 1);
    } while (ready_size < 0 && errno == EINTR);
    if (ready_size != 1 || ready != 'I') {
        throw_setup_failure(
            diagnostic_pipe[0],
            "target privilege isolation did not complete");
    }
    send_packet(config.protocol_fd, "RELEASED");
    const int status = owned_namespace.wait();
    do {
        ready_size = read(ready_pipe[0], &ready, 1);
    } while (ready_size < 0 && errno == EINTR);
    close_checked(ready_pipe[0], "helper readiness reader");
    if (ready_size != 1 || ready != 'C') {
        throw_setup_failure(
            diagnostic_pipe[0],
            "target namespace teardown did not complete");
    }
    close_checked(diagnostic_pipe[0], "helper diagnostics reader");
    owned_namespace.close_pidfd();
    send_packet(config.protocol_fd, "COMPLETE");
    return wait_status_exit_code(status);
}

#endif

}

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::strcmp(argv[1], "--version") == 0) {
            std::cout << "deepseek-v41-containment-helper " << DSV41_BUILD_REVISION << '\n';
            return 0;
        }
#if defined(__linux__)
        if (argc == 2 && std::strcmp(argv[1], "--check-launcher-groups") == 0) {
            require_zero_supplementary_groups("containment launcher");
            std::cout << "supplementary-groups=0\n";
            return 0;
        }
        return run_linux_helper(parse_options(argc, argv));
#else
        (void) argc;
        (void) argv;
        std::cerr << "Linux PID namespace containment is unavailable on this platform\n";
        return 125;
#endif
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 125;
    }
}
