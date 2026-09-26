#!/usr/bin/env python3

import json
import hashlib
import importlib.util
import os
import platform
import plistlib
import re
import stat
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, cast

from trace_format import (
    NO_EXTERNAL_STATE_STORAGE,
    TraceError,
    install_trust_sha256,
    runtime_build_evidence_sha256,
    validate_install_trust_evidence,
    validate_runtime_build_evidence,
    validate_watchdog_event,
)

FORBIDDEN_ROOT = Path("/mnt/bigspace")
SOFT_MEMORY_LIMIT = 116 * 1024 * 1024 * 1024
WATCHDOG_EMERGENCY_LIMIT = 118 * 1024 * 1024 * 1024
STRICT_MEMORY_LIMIT = 120 * 1024 * 1024 * 1024
MAX_WATCHDOG_HEARTBEAT_AGE = 30.0
WATCHDOG_LEASE_FORMAT = "strix-memory-watchdog-lease"
WATCHDOG_HEARTBEAT_FORMAT = "strix-memory-watchdog-heartbeat"
WATCHDOG_VERSION = 2
WATCHDOG_STARTUP_TIMEOUT_SECONDS = 5.0
WATCHDOG_LEASE_ENV = "STRIX_MEMORY_WATCHDOG_LEASE_PATH"
WATCHDOG_HEARTBEAT_ENV = "STRIX_MEMORY_WATCHDOG_HEARTBEAT_PATH"
WATCHDOG_AUDIT_ENV = "STRIX_MEMORY_WATCHDOG_AUDIT_PATH"
WATCHDOG_MAX_AGE_ENV = "STRIX_MEMORY_WATCHDOG_HEARTBEAT_MAX_AGE_SECONDS"
WATCHDOG_REVISION = "1568e7903cc07fc83fee1894df491dfcd2f7ba9c"
WATCHDOG_SCRIPT_SHA256 = "799a689198eb873085c98178c62c30cf664b28e59ec6466af0bd89ff8fce68f5"
APPROVED_WATCHDOGS = {WATCHDOG_SCRIPT_SHA256: WATCHDOG_REVISION}
_WATCHDOG_GUARD = None


class PreflightError(RuntimeError):
    pass


def strict_json_loads(data: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise PreflightError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(data, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as error:
        raise PreflightError(f"invalid JSON: {error}") from error


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def require_no_symlink_components(path: Path, label: str) -> Path:
    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise PreflightError(f"{label} must not be a symlink or contain symlink components")
        if not current.exists():
            break
    return absolute


def reject_forbidden_path(
        path: Path,
        label: str,
        forbidden_root: Path = FORBIDDEN_ROOT) -> Path:
    absolute = path.expanduser().absolute()
    try:
        absolute.relative_to(forbidden_root)
    except ValueError:
        return absolute
    raise PreflightError(f"{label} must not use /mnt/bigspace: {absolute}")


def require_safe_tmpdir_path(path: Path) -> Path:
    if not path.is_absolute():
        raise PreflightError("TMPDIR must use an absolute literal path")
    lexical_path = reject_forbidden_path(path, "TMPDIR")
    try:
        status = os.lstat(path)
    except OSError as error:
        raise PreflightError("TMPDIR must be an existing writable directory at its original lexical path") from error
    if stat.S_ISLNK(status.st_mode):
        raise PreflightError("TMPDIR must not be a symlink or contain symlink components")
    if not stat.S_ISDIR(status.st_mode):
        raise PreflightError("TMPDIR must be an existing writable directory at its original lexical path")
    if not os.access(path, os.W_OK | os.X_OK):
        raise PreflightError("TMPDIR must be an existing writable directory at its original lexical path")
    require_no_symlink_components(lexical_path, "TMPDIR")
    return lexical_path


def _decode_mount_field(value: str) -> str:
    return re.sub(
        r"\\([0-7]{3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
    )


def _existing_ancestor(path: Path) -> Path:
    current = path
    while not current.exists():
        if current == current.parent:
            raise PreflightError(f"path has no existing parent: {path}")
        current = current.parent
    return current.resolve(strict=True)


def storage_attestation(
        path: Path,
        label: str,
        *,
        mountinfo_path: Path = Path("/proc/self/mountinfo"),
        sys_dev_block_root: Path = Path("/sys/dev/block"),
        sys_class_block_root: Path = Path("/sys/class/block"),
        forbidden_root: Path = FORBIDDEN_ROOT) -> dict[str, Any]:
    lexical_path = reject_forbidden_path(path, label, forbidden_root)
    path = resolved(lexical_path)
    try:
        path.relative_to(forbidden_root)
    except ValueError:
        pass
    else:
        raise PreflightError(f"{label} must not use /mnt/bigspace: {path}")

    existing = _existing_ancestor(path)
    mounts: list[tuple[Path, str, str, str]] = []
    for line in read_proc_lines(mountinfo_path):
        fields = line.split()
        try:
            separator = fields.index("-")
            mount = (
                Path(_decode_mount_field(fields[4])),
                fields[separator + 1],
                _decode_mount_field(fields[separator + 2]),
                fields[2],
            )
        except (IndexError, ValueError) as error:
            raise PreflightError(f"invalid mountinfo record: {line}") from error
        try:
            existing.relative_to(mount[0])
        except ValueError:
            continue
        mounts.append(mount)
    if not mounts:
        raise PreflightError(f"{label} mount cannot be resolved: {path}")
    mount_point, filesystem_type, mount_source, device_number = max(
        mounts, key=lambda item: len(item[0].parts))
    if device_number.split(":", 1)[0] == "0":
        source_device = mount_source.split("[", 1)[0]
        source_name = Path(source_device).name
        source_dev_path = sys_class_block_root / source_name / "dev"
        if not source_device.startswith("/dev/") or not source_dev_path.is_file():
            raise PreflightError(f"{label} is not backed by a local block device: {path}")
        try:
            device_number = source_dev_path.read_text(encoding="ascii").strip()
        except OSError as error:
            raise PreflightError(f"{label} backing device identity cannot be read: {error}") from error
        if re.fullmatch(r"[0-9]+:[0-9]+", device_number) is None:
            raise PreflightError(f"{label} backing device identity is invalid: {device_number}")

    device_link = sys_dev_block_root / device_number
    if not device_link.is_symlink():
        raise PreflightError(f"{label} block device cannot be resolved: {path}")
    try:
        block_device = device_link.resolve(strict=True)
    except OSError as error:
        raise PreflightError(f"{label} block device cannot be resolved: {error}") from error
    rotational_path = None
    for candidate in (block_device, *block_device.parents):
        path_candidate = candidate / "queue" / "rotational"
        if path_candidate.is_file():
            rotational_path = path_candidate
            break
    if rotational_path is None:
        raise PreflightError(f"{label} block device rotational state cannot be resolved: {block_device}")
    try:
        rotational = rotational_path.read_text(encoding="ascii").strip()
    except OSError as error:
        raise PreflightError(f"{label} block device rotational state cannot be read: {error}") from error
    if rotational != "0":
        raise PreflightError(f"{label} must use non-rotational storage: {path}")

    nvme_device = next(
        (part for part in reversed(block_device.parts) if re.fullmatch(r"nvme[0-9]+(?:c[0-9]+)?n[0-9]+", part)),
        None,
    )
    if nvme_device is None:
        raise PreflightError(f"{label} must use an NVMe block device: {block_device}")
    return {
        "format": "dsv41-storage-attestation",
        "version": 2,
        "runtime_kind": "strix-rocm",
        "platform": "linux",
        "storage_kind": "linux-nvme",
        "resolved_path": str(path),
        "existing_path": str(existing),
        "mount_point": str(mount_point),
        "filesystem_type": filesystem_type,
        "mount_source": mount_source,
        "device_number": device_number,
        "block_device_path": str(block_device),
        "nvme_device": nvme_device,
        "rotational": False,
        "source": "linux-mountinfo-sysfs",
    }


def require_nvme_path(
        path: Path,
        label: str,
        *,
        mountinfo_path: Path = Path("/proc/self/mountinfo"),
        sys_dev_block_root: Path = Path("/sys/dev/block"),
        sys_class_block_root: Path = Path("/sys/class/block")) -> Path:
    attestation = storage_attestation(
        path,
        label,
        mountinfo_path=mountinfo_path,
        sys_dev_block_root=sys_dev_block_root,
        sys_class_block_root=sys_class_block_root,
    )
    return _attested_resolved_path(attestation, label)


def _diskutil_info(path: Path) -> dict[str, Any]:
    try:
        df = subprocess.check_output(
            ["df", "-P", str(path)],
            text=True,
            stderr=subprocess.STDOUT,
        ).splitlines()
        if len(df) < 2:
            raise PreflightError(f"df cannot resolve a mounted volume for {path}")
        fields = df[-1].split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            raise PreflightError(f"df returned invalid mount evidence for {path}")
        mount_point = fields[5]
        data = subprocess.check_output(
            ["diskutil", "info", "-plist", mount_point],
            stderr=subprocess.STDOUT,
        )
        record = plistlib.loads(data)
    except (OSError, subprocess.CalledProcessError, plistlib.InvalidFileException) as error:
        raise PreflightError(f"diskutil cannot attest storage for {path}: {error}") from error
    if not isinstance(record, dict):
        raise PreflightError(f"diskutil returned invalid storage evidence for {path}")
    record["_dsv41_mount_point"] = mount_point
    return record


def darwin_storage_attestation(
        path: Path,
        label: str,
        *,
        disk_info: Callable[[Path], dict[str, Any]] = _diskutil_info,
        forbidden_root: Path = FORBIDDEN_ROOT) -> dict[str, Any]:
    lexical_path = reject_forbidden_path(path, label, forbidden_root)
    path = resolved(lexical_path)
    try:
        path.relative_to(forbidden_root)
    except ValueError:
        pass
    else:
        raise PreflightError(f"{label} must not use /mnt/bigspace: {path}")
    existing = _existing_ancestor(path)
    record = dict(disk_info(existing))
    mount_point = record.pop("_dsv41_mount_point", record.get("MountPoint"))
    filesystem_type = record.get("FilesystemType")
    device_identifier = record.get("DeviceIdentifier")
    parent_whole_disk = record.get("ParentWholeDisk")
    bus_protocol = record.get("BusProtocol")
    if record.get("Internal") is not True or record.get("SolidState") is not True:
        raise PreflightError(f"{label} must use internal non-rotational storage: {path}")
    if record.get("VolumeNetwork") is True or record.get("DiskImage") is True:
        raise PreflightError(f"{label} must use local storage: {path}")
    for name, value in (
            ("mount point", mount_point),
            ("filesystem type", filesystem_type),
            ("device identifier", device_identifier),
            ("parent whole disk", parent_whole_disk),
            ("bus protocol", bus_protocol)):
        if not isinstance(value, str) or not value:
            raise PreflightError(f"{label} {name} cannot be resolved: {path}")
    if not mount_point.startswith("/"):
        raise PreflightError(f"{label} mount point is invalid: {mount_point}")
    if cast(str, bus_protocol).lower() not in {"nvme", "apple fabric"}:
        raise PreflightError(f"{label} storage is not NVMe-backed: {bus_protocol}")
    try:
        filesystem_device = os.stat(existing).st_dev
        if filesystem_device != os.stat(mount_point).st_dev:
            raise PreflightError(f"{label} filesystem identity differs from its attested mount: {path}")
    except OSError as error:
        raise PreflightError(f"{label} filesystem identity cannot be read: {error}") from error
    return {
        "format": "dsv41-storage-attestation",
        "version": 2,
        "runtime_kind": "apple-metal",
        "platform": "macos",
        "storage_kind": "darwin-local-solid-state",
        "resolved_path": str(path),
        "existing_path": str(existing),
        "mount_point": mount_point,
        "filesystem_type": filesystem_type,
        "device_identifier": device_identifier,
        "parent_whole_disk": parent_whole_disk,
        "bus_protocol": bus_protocol,
        "filesystem_device": filesystem_device,
        "internal": True,
        "solid_state": True,
        "source": "diskutil-info-plist",
    }


def _attested_resolved_path(attestation: dict[str, Any], label: str) -> Path:
    resolved_path = attestation["resolved_path"]
    if not isinstance(resolved_path, str):
        raise PreflightError(f"{label} resolved path evidence is invalid")
    return Path(resolved_path)


def safe_trace_path(root: Path, relative: Path | str) -> Path:
    root = root.expanduser().absolute()
    if root.is_symlink():
        raise PreflightError("trace output root must not be a symlink")
    root = root.resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise PreflightError(f"trace output path is outside the bundle: {relative}")
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise PreflightError(f"trace output path must not use symlinks: {relative}")
    try:
        candidate.resolve().relative_to(root)
    except ValueError as error:
        raise PreflightError(f"trace output path is outside the bundle: {relative}") from error
    return candidate


def read_proc_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="ascii").splitlines()
    except OSError as error:
        raise PreflightError(f"cannot read {path}: {error}") from error


def swap_audit() -> dict[str, Any]:
    lines = read_proc_lines(Path("/proc/swaps"))
    entries = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) >= 5:
            entries.append({
                "path": fields[0],
                "type": fields[1],
                "size_kib": int(fields[2]),
                "used_kib": int(fields[3]),
                "priority": int(fields[4]),
            })
    return {"enabled": bool(entries), "entries": entries}


def memory_audit() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in read_proc_lines(Path("/proc/meminfo")):
        key, value = line.split(":", 1)
        fields = value.split()
        if fields:
            values[key] = int(fields[0]) * 1024
    required = ("MemTotal", "MemAvailable")
    if any(key not in values for key in required):
        raise PreflightError("/proc/meminfo lacks MemTotal or MemAvailable")
    result = {
        "mem_total_bytes": values["MemTotal"],
        "mem_available_bytes": values["MemAvailable"],
        "mem_used_bytes": values["MemTotal"] - values["MemAvailable"],
    }
    if result["mem_used_bytes"] >= SOFT_MEMORY_LIMIT:
        raise PreflightError(
            f"host memory use is at or above the 116 GiB soft limit: {result['mem_used_bytes']}")
    return result


def _command_text(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise PreflightError(f"{' '.join(args)} failed: {error}") from error


def darwin_host_and_memory_audit(
        *,
        command_text: Callable[..., str] = _command_text,
        system: str | None = None,
        machine: str | None = None) -> tuple[dict[str, Any], dict[str, int]]:
    platform_name = platform.system() if system is None else system
    machine_name = platform.machine() if machine is None else machine
    if platform_name != "Darwin" or machine_name != "arm64":
        raise PreflightError("ds4 oracle execution requires macOS on arm64")
    try:
        total = int(command_text("sysctl", "-n", "hw.memsize"))
    except ValueError as error:
        raise PreflightError("Darwin memory size is invalid") from error
    if total < 128 * 1024 * 1024 * 1024:
        raise PreflightError("ds4 oracle host must have at least 128 GiB of memory")
    vm_stat = command_text("vm_stat")
    page_size_match = re.search(r"page size of ([0-9]+) bytes", vm_stat)
    if page_size_match is None:
        raise PreflightError("vm_stat page size is missing")
    page_size = int(page_size_match.group(1))
    pages = {}
    for name, value in re.findall(r"^([^:]+):\s+([0-9]+)\.$", vm_stat, flags=re.MULTILINE):
        pages[name] = int(value)
    available_pages = sum(pages.get(name, 0) for name in (
        "Pages free",
        "Pages inactive",
        "Pages speculative",
    ))
    available = available_pages * page_size
    if available <= 0 or available > total:
        raise PreflightError("Darwin available memory evidence is invalid")
    host = {
        "format": "dsv41-host-attestation",
        "version": 1,
        "runtime_kind": "apple-metal",
        "platform": "macos",
        "machine": "arm64",
        "hardware_model": command_text("sysctl", "-n", "hw.model"),
        "os_version": command_text("sysctl", "-n", "kern.osproductversion"),
        "memory_bytes": total,
        "source": "darwin-sysctl",
    }
    if not host["hardware_model"] or not host["os_version"]:
        raise PreflightError("Darwin host identity is incomplete")
    return host, {
        "mem_total_bytes": total,
        "mem_available_bytes": available,
        "mem_used_bytes": total - available,
    }


def darwin_swap_audit(
        *,
        command_text: Callable[..., str] = _command_text) -> dict[str, Any]:
    value = command_text("sysctl", "-n", "vm.swapusage")
    match = re.fullmatch(
        r"total = ([0-9]+(?:\.[0-9]+)?)M\s+used = ([0-9]+(?:\.[0-9]+)?)M\s+"
        r"free = ([0-9]+(?:\.[0-9]+)?)M(?:\s+\(encrypted\))?",
        value,
    )
    if match is None:
        raise PreflightError("Darwin swap evidence is invalid")
    total, used, free = (int(float(item) * 1024 * 1024) for item in match.groups())
    if total != 0 or used != 0 or free != 0:
        raise PreflightError("swap is enabled; model execution is blocked")
    return {
        "source": "darwin-sysctl-vm.swapusage",
        "total_bytes": total,
        "used_bytes": used,
        "free_bytes": free,
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def proc_start_time_ticks(stat: str) -> int:
    command_end = stat.rfind(")")
    if command_end < 0:
        raise PreflightError("watchdog process stat is invalid")
    fields = stat[command_end + 2:].split()
    if len(fields) < 20:
        raise PreflightError("watchdog process stat is truncated")
    return int(fields[19])


def proc_parent_pid(stat: str) -> int:
    command_end = stat.rfind(")")
    if command_end < 0:
        raise PreflightError("process stat is invalid")
    fields = stat[command_end + 2:].split()
    if len(fields) < 2:
        raise PreflightError("process stat is truncated")
    return int(fields[1])


def proc_process_group_id(stat: str) -> int:
    command_end = stat.rfind(")")
    if command_end < 0:
        raise PreflightError("process stat is invalid")
    fields = stat[command_end + 2:].split()
    if len(fields) < 3:
        raise PreflightError("process stat is truncated")
    return int(fields[2])


def is_descendant(pid: int, ancestor_pid: int, procfs_root: Path) -> bool:
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor_pid:
            return True
        seen.add(pid)
        try:
            pid = proc_parent_pid((procfs_root / str(pid) / "stat").read_text(encoding="ascii"))
        except (OSError, ValueError):
            return False
    return False


def open_watchdog_namespace_authority(
        watchdog: dict[str, Any],
        *,
        procfs_root: Path = Path("/proc"),
        pidfd_open: Callable[[int, int], int] | None = None,
) -> tuple[int, dict[str, Any]]:
    if sys.platform != "linux":
        raise PreflightError("watchdog namespace authority requires Linux pidfds")
    opener = pidfd_open or getattr(os, "pidfd_open", None)
    if opener is None:
        raise PreflightError("watchdog namespace authority requires os.pidfd_open")
    try:
        watchdog_pid = int(watchdog["watchdog_pid"])
        watchdog_start = int(watchdog["watchdog_start_time_ticks"])
        guardian_pid = int(watchdog["guardian_pid"])
        child_pid = int(watchdog["child_pid"])
        child_pgid = int(watchdog["child_process_group_id"])
        executable_path = str(watchdog["watchdog_executable_path"])
        command_sha256 = str(watchdog["watchdog_command_sha256"])
    except (KeyError, TypeError, ValueError) as error:
        raise PreflightError(f"watchdog namespace authority identity is invalid: {error}") from error
    descriptor = -1
    try:
        descriptor = int(opener(watchdog_pid, 0))
        watchdog_stat = (procfs_root / str(watchdog_pid) / "stat").read_text(encoding="ascii")
        guardian_stat = (procfs_root / str(guardian_pid) / "stat").read_text(encoding="ascii")
        child_stat = (procfs_root / str(child_pid) / "stat").read_text(encoding="ascii")
        live_executable = str((procfs_root / str(watchdog_pid) / "exe").resolve(strict=True))
        live_command = (procfs_root / str(watchdog_pid) / "cmdline").read_bytes()
        fdinfo = (procfs_root / "self" / "fdinfo" / str(descriptor)).read_text(encoding="ascii")
        fdinfo_pid = next(
            int(line.split(":", 1)[1].strip())
            for line in fdinfo.splitlines()
            if line.startswith("Pid:")
        )
        if proc_start_time_ticks(watchdog_stat) != watchdog_start:
            raise PreflightError("watchdog namespace authority start time does not match")
        if live_executable != str(Path(executable_path).resolve(strict=True)):
            raise PreflightError("watchdog namespace authority executable does not match")
        if sha256_bytes(live_command) != command_sha256:
            raise PreflightError("watchdog namespace authority command does not match")
        if fdinfo_pid != watchdog_pid:
            raise PreflightError("watchdog namespace authority pidfd does not match")
        watchdog_pgid = proc_process_group_id(watchdog_stat)
        if proc_parent_pid(guardian_stat) != watchdog_pid or (
                proc_process_group_id(guardian_stat) != child_pgid) or (
                guardian_pid != child_pgid) or (
                proc_parent_pid(child_stat) != guardian_pid) or (
                proc_process_group_id(child_stat) != child_pgid):
            raise PreflightError("watchdog namespace authority process tree does not match")
        authority = {
            "format": "dsv41-watchdog-namespace-authority",
            "version": 1,
            "mechanism": "inherited-pidfd",
            "descriptor": descriptor,
            "host_procfs_root": str(procfs_root.resolve(strict=True)),
            "watchdog_pid": watchdog_pid,
            "watchdog_process_group_id": watchdog_pgid,
            "watchdog_start_time_ticks": watchdog_start,
            "watchdog_executable_path": live_executable,
            "watchdog_command_sha256": command_sha256,
            "guardian_pid": guardian_pid,
            "child_pid": child_pid,
            "child_process_group_id": child_pgid,
        }
        return descriptor, authority
    except (OSError, StopIteration, ValueError) as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise PreflightError(f"cannot establish watchdog namespace authority: {error}") from error
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def verify_watchdog_namespace_authority(
        descriptor: int,
        authority: dict[str, Any],
        *,
        procfs_root: Path = Path("/proc"),
) -> None:
    try:
        if authority.get("format") != "dsv41-watchdog-namespace-authority" or (
                authority.get("version") != 1) or authority.get("mechanism") != "inherited-pidfd" or (
                authority.get("descriptor") != descriptor):
            raise PreflightError("watchdog namespace authority receipt is invalid")
        watchdog_pid = int(authority["watchdog_pid"])
        watchdog_start = int(authority["watchdog_start_time_ticks"])
        stat_text = (procfs_root / str(watchdog_pid) / "stat").read_text(encoding="ascii")
        executable = str((procfs_root / str(watchdog_pid) / "exe").resolve(strict=True))
        command = (procfs_root / str(watchdog_pid) / "cmdline").read_bytes()
        fdinfo = (procfs_root / "self" / "fdinfo" / str(descriptor)).read_text(encoding="ascii")
        fdinfo_pid = next(
            int(line.split(":", 1)[1].strip())
            for line in fdinfo.splitlines()
            if line.startswith("Pid:")
        )
        if fdinfo_pid != watchdog_pid or proc_start_time_ticks(stat_text) != watchdog_start or (
                executable != authority.get("watchdog_executable_path")) or (
                sha256_bytes(command) != authority.get("watchdog_command_sha256")):
            raise PreflightError("watchdog namespace authority changed")
    except PreflightError:
        raise
    except (KeyError, OSError, StopIteration, TypeError, ValueError) as error:
        raise PreflightError(f"cannot verify watchdog namespace authority: {error}") from error


def read_heartbeat(
        path: Path,
        max_age_seconds: float,
        *,
        lease_id: str,
        watchdog_pid: int,
        watchdog_start_time_ticks: int,
        child_pid: int,
        child_pgid: int,
        now: int | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns) -> int:
    try:
        record = strict_json_loads(path.read_text(encoding="ascii"))
        updated = datetime.fromisoformat(str(record["updated_at"]).replace("Z", "+00:00"))
        heartbeat = int(updated.timestamp())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise PreflightError(f"watchdog heartbeat is invalid: {error}") from error
    if record.get("format") != WATCHDOG_HEARTBEAT_FORMAT or record.get("version") != WATCHDOG_VERSION:
        raise PreflightError("watchdog heartbeat format is invalid")
    if record.get("lease_id") != lease_id or record.get("state") != "active":
        raise PreflightError("watchdog heartbeat lease identity or state is invalid")
    if type(record.get("sequence")) is not int or record["sequence"] < 0:
        raise PreflightError("watchdog heartbeat sequence is invalid")
    if type(record.get("updated_monotonic_ns")) is not int or record["updated_monotonic_ns"] <= 0:
        raise PreflightError("watchdog heartbeat monotonic timestamp is invalid")
    if record.get("watchdog_pid") != watchdog_pid or (
            record.get("watchdog_start_time_ticks") != watchdog_start_time_ticks):
        raise PreflightError("watchdog heartbeat owner does not match the lease")
    if record.get("child_pid") != child_pid or record.get("child_process_group_id") != child_pgid:
        raise PreflightError("watchdog heartbeat child identity does not match the lease")
    age_ns = monotonic_ns() - record["updated_monotonic_ns"]
    if age_ns < 0 or age_ns > int(max_age_seconds * 1_000_000_000):
        raise PreflightError("watchdog heartbeat is stale")
    now = int(time.time()) if now is None else now
    if heartbeat <= 0 or heartbeat > now:
        raise PreflightError("watchdog heartbeat wall-clock timestamp is invalid")
    return heartbeat


def _read_json_with_retry(
        path: Path,
        *,
        timeout_seconds: float,
        monotonic: Callable[[], float],
        sleeper: Callable[[float], None]) -> dict[str, Any]:
    deadline = monotonic() + timeout_seconds
    last_error: Exception | None = None
    while True:
        try:
            record = strict_json_loads(path.read_text(encoding="ascii"))
            if not isinstance(record, dict):
                raise ValueError("record is not an object")
            return record
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            last_error = error
            if monotonic() >= deadline:
                raise PreflightError(f"watchdog lease did not become ready: {last_error}") from last_error
            sleeper(0.05)


def _read_watchdog_events(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except OSError as error:
        raise PreflightError(f"cannot read watchdog audit: {error}") from error
    events = []
    for line_number, line in enumerate(lines, start=1):
        try:
            event = validate_watchdog_event(strict_json_loads(line))
        except (PreflightError, TraceError) as error:
            raise PreflightError(f"watchdog audit line {line_number} is invalid: {error}") from error
        events.append(event)
    if not events:
        raise PreflightError("watchdog audit is empty")
    return events


def _read_watchdog_startup_events(
        path: Path,
        *,
        timeout_seconds: float,
        monotonic: Callable[[], float],
        sleeper: Callable[[float], None]) -> list[dict[str, Any]]:
    deadline = monotonic() + timeout_seconds
    last_error: Exception | None = None
    while True:
        try:
            events = _read_watchdog_events(path)
            if any(event.get("event") == "preflight" for event in events) and (
                    any(event.get("event") == "child_started" for event in events)):
                return events
            last_error = PreflightError("watchdog audit lacks preflight or child_started evidence")
        except PreflightError as error:
            last_error = error
        if monotonic() >= deadline:
            assert last_error is not None
            raise PreflightError(f"watchdog audit did not become ready: {last_error}") from last_error
        sleeper(0.05)


def _load_watchdog_module(script: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("dsv41_strix_memory_watchdog", script)
    if spec is None or spec.loader is None:
        raise PreflightError(f"cannot load canonical watchdog module: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except (ImportError, OSError, RuntimeError, SyntaxError) as error:
        raise PreflightError(f"cannot load canonical watchdog module: {error}") from error
    return module


def _watchdog_audit_result(
        lease: dict[str, Any],
        *,
        watchdog_revision: str,
        lease_path: Path,
        heartbeat_path: Path,
        audit_path: Path,
        audit_event_count: int,
        procfs_root: Path = Path("/proc")) -> dict[str, Any]:
    try:
        heartbeat_record = strict_json_loads(heartbeat_path.read_text(encoding="ascii"))
        updated = datetime.fromisoformat(str(heartbeat_record["updated_at"]).replace("Z", "+00:00"))
        heartbeat_unix = int(updated.timestamp())
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise PreflightError(f"canonical watchdog heartbeat is invalid: {error}") from error
    try:
        audit_sha256 = sha256_bytes(audit_path.read_bytes())
    except OSError as error:
        raise PreflightError(f"cannot hash canonical watchdog audit: {error}") from error
    watchdog_command = lease.get("watchdog_command")
    if not isinstance(watchdog_command, str) or not watchdog_command:
        try:
            command_bytes = (
                procfs_root / str(lease["watchdog_pid"]) / "cmdline").read_bytes()
        except (OSError, KeyError) as error:
            raise PreflightError(f"cannot read canonical watchdog command: {error}") from error
        watchdog_command = command_bytes.replace(b"\0", b" ").decode("utf-8", "replace").strip()
        if not watchdog_command:
            raise PreflightError("canonical watchdog command is empty")
    result = dict(lease)
    result.update({
        "watchdog_revision": watchdog_revision,
        "lease_path": str(lease_path),
        "watchdog_command": watchdog_command,
        "heartbeat_unix": heartbeat_unix,
        "audit_live_path": str(audit_path),
        "audit_path": str(audit_path),
        "audit_sha256": audit_sha256,
        "audit_event_count": audit_event_count,
    })
    return result


def _canonical_watchdog_audit(
        repo: Path,
        *,
        environment: dict[str, str] | os._Environ[str],
        current_pid: int | None,
        timeout_seconds: float,
        monotonic: Callable[[], float],
        sleeper: Callable[[float], None],
        watchdog_module: object | None) -> dict[str, Any]:
    global _WATCHDOG_GUARD
    try:
        lease_path = require_nvme_path(Path(environment[WATCHDOG_LEASE_ENV]), "watchdog lease")
        heartbeat_path = require_nvme_path(Path(environment[WATCHDOG_HEARTBEAT_ENV]), "watchdog heartbeat")
        audit_path = require_nvme_path(Path(environment[WATCHDOG_AUDIT_ENV]), "watchdog audit")
        max_age_seconds = float(environment[WATCHDOG_MAX_AGE_ENV])
    except (KeyError, ValueError) as error:
        raise PreflightError(f"canonical watchdog environment is incomplete: {error}") from error
    if max_age_seconds <= 0 or max_age_seconds > MAX_WATCHDOG_HEARTBEAT_AGE:
        raise PreflightError(f"watchdog heartbeat age must be within 1..{MAX_WATCHDOG_HEARTBEAT_AGE} seconds")
    expected_script = resolved(repo / "scripts" / "strix_memory_watchdog.py")
    if not expected_script.is_file():
        raise PreflightError(f"canonical watchdog script is missing: {expected_script}")
    script_sha256 = sha256_bytes(expected_script.read_bytes())
    watchdog_revision = APPROVED_WATCHDOGS.get(script_sha256)
    if watchdog_revision is None:
        raise PreflightError(
            "no independently reviewed watchdog revision is approved for correctness execution")
    module = watchdog_module or _load_watchdog_module(expected_script)
    validator = getattr(module, "validate_active_lease", None)
    validation_error = getattr(module, "LeaseValidationError", RuntimeError)
    guard = getattr(module, "start_process_group_lease_guard", None)
    if not callable(validator) or not isinstance(validation_error, type):
        raise PreflightError("canonical watchdog module lacks the lease validation API")
    process_id = os.getpid() if current_pid is None else current_pid
    try:
        lease = validator(
            lease_path,
            expected_script_path=expected_script,
            expected_soft_bytes=SOFT_MEMORY_LIMIT,
            expected_emergency_bytes=WATCHDOG_EMERGENCY_LIMIT,
            expected_procfs_root=Path("/proc"),
            expected_heartbeat_path=heartbeat_path,
            expected_audit_path=audit_path,
            expected_max_heartbeat_age_seconds=max_age_seconds,
            current_process_id=process_id,
            process_procfs_root=Path("/proc"),
        )
        if lease.get("child_pid") == process_id and _WATCHDOG_GUARD is None:
            if not callable(guard):
                raise PreflightError("canonical watchdog module lacks the process-group lease guard")
            _WATCHDOG_GUARD = guard(
                expected_script,
                startup_timeout_seconds=timeout_seconds,
                expected_procfs_root=Path("/proc"),
                process_procfs_root=Path("/proc"),
            )
    except validation_error as error:
        raise PreflightError(f"canonical watchdog lease is invalid: {error}") from error
    if not isinstance(lease, dict):
        raise PreflightError("canonical watchdog lease validator returned invalid data")
    if lease.get("grace_seconds") != 30.0 or lease.get("sample_interval_seconds") != 1.0:
        raise PreflightError("canonical watchdog timing policy is invalid")
    events = _read_watchdog_startup_events(
        audit_path,
        timeout_seconds=timeout_seconds,
        monotonic=monotonic,
        sleeper=sleeper,
    )
    return _watchdog_audit_result(
        lease,
        watchdog_revision=watchdog_revision,
        lease_path=lease_path,
        heartbeat_path=heartbeat_path,
        audit_path=audit_path,
        audit_event_count=len(events),
    )


def watchdog_audit(
        repo: Path,
        *,
        environment: dict[str, str] | os._Environ[str] = os.environ,
        procfs_root: Path = Path("/proc"),
        current_pid: int | None = None,
        current_pgid: int | None = None,
        getpgid: Callable[[int], int] = os.getpgid,
        now: int | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        timeout_seconds: float = WATCHDOG_STARTUP_TIMEOUT_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        watchdog_module: object | None = None) -> dict[str, Any]:
    if watchdog_module is not None or (
            procfs_root == Path("/proc") and current_pid is None and current_pgid is None):
        return _canonical_watchdog_audit(
            repo,
            environment=environment,
            current_pid=current_pid,
            timeout_seconds=timeout_seconds,
            monotonic=monotonic,
            sleeper=sleeper,
            watchdog_module=watchdog_module,
        )
    try:
        pid_file = Path(environment[WATCHDOG_LEASE_ENV])
        environment_heartbeat = require_nvme_path(
            Path(environment[WATCHDOG_HEARTBEAT_ENV]), "watchdog heartbeat")
        environment_audit = require_nvme_path(
            Path(environment[WATCHDOG_AUDIT_ENV]), "watchdog audit")
        environment_max_age = float(environment[WATCHDOG_MAX_AGE_ENV])
    except (KeyError, ValueError) as error:
        raise PreflightError(f"canonical watchdog environment is incomplete: {error}") from error
    pid_file = require_nvme_path(pid_file, "watchdog lease")
    repo = resolved(repo)
    expected_script = resolved(repo / "scripts" / "strix_memory_watchdog.py")
    if not expected_script.is_file():
        raise PreflightError(f"canonical watchdog script is missing: {expected_script}")
    expected_script_sha256 = sha256_bytes(expected_script.read_bytes())
    lease = _read_json_with_retry(
        pid_file,
        timeout_seconds=timeout_seconds,
        monotonic=monotonic,
        sleeper=sleeper,
    )
    try:
        pid = int(lease["watchdog_pid"])
        expected_start = int(lease["watchdog_start_time_ticks"])
        expected_command_sha256 = str(lease["watchdog_command_sha256"])
        child_pid = int(lease["child_pid"])
        child_pgid = int(lease["child_process_group_id"])
        script_path = resolved(Path(str(lease["watchdog_script_path"])))
        script_sha256 = str(lease["watchdog_script_sha256"])
        lease_id = str(lease["lease_id"])
        soft_bytes = int(lease["soft_bytes"])
        emergency_bytes = int(lease["emergency_bytes"])
        strict_bytes = int(lease["strict_ceiling_bytes"])
        procfs_path = str(lease["procfs_root"])
        heartbeat_path = require_nvme_path(Path(lease["heartbeat_path"]), "watchdog heartbeat")
        audit_path = require_nvme_path(Path(lease["audit_path"]), "watchdog audit")
        max_age_seconds = float(lease.get("max_heartbeat_age_seconds", MAX_WATCHDOG_HEARTBEAT_AGE))
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise PreflightError(f"watchdog lease is invalid: {error}") from error
    if lease.get("format") != WATCHDOG_LEASE_FORMAT or lease.get("version") != WATCHDOG_VERSION:
        raise PreflightError("watchdog lease format is invalid")
    if lease.get("state") != "active" or re.fullmatch(r"[0-9a-f]{32,64}", lease_id) is None:
        raise PreflightError("watchdog lease identity or state is invalid")
    if script_path != expected_script or script_sha256 != expected_script_sha256:
        raise PreflightError("watchdog script identity does not match the candidate repository")
    if soft_bytes != SOFT_MEMORY_LIMIT or emergency_bytes != WATCHDOG_EMERGENCY_LIMIT or (
            strict_bytes != STRICT_MEMORY_LIMIT):
        raise PreflightError("watchdog memory thresholds are invalid")
    if procfs_path != "/proc":
        raise PreflightError("watchdog procfs root must be /proc")
    if heartbeat_path != environment_heartbeat or audit_path != environment_audit or (
            max_age_seconds != environment_max_age):
        raise PreflightError("watchdog lease paths or heartbeat age do not match the inherited environment")
    if re.fullmatch(r"[0-9a-f]{64}", expected_command_sha256) is None:
        raise PreflightError("watchdog lease command SHA-256 is invalid")
    if max_age_seconds <= 0 or max_age_seconds > MAX_WATCHDOG_HEARTBEAT_AGE:
        raise PreflightError(f"watchdog heartbeat age must be within 1..{MAX_WATCHDOG_HEARTBEAT_AGE} seconds")
    if pid <= 1 or child_pid <= 1 or child_pgid <= 1:
        raise PreflightError("watchdog or monitored child identity is invalid")
    if current_pgid is None:
        current_pgid = os.getpgrp()
    if current_pid is None:
        current_pid = os.getpid()
    if current_pgid != child_pgid:
        raise PreflightError("current process is outside the watchdog-monitored process group")
    if not is_descendant(current_pid, child_pid, procfs_root):
        raise PreflightError("current process is not a descendant of the watchdog-monitored child")
    try:
        child_parent = proc_parent_pid(
            (procfs_root / str(child_pid) / "stat").read_text(encoding="ascii"))
        if child_parent != pid or getpgid(child_pid) != child_pgid or child_pgid != child_pid:
            raise PreflightError("watchdog child process group does not match the lease")
    except (OSError, ValueError) as error:
        raise PreflightError(f"cannot inspect watchdog child process group: {error}") from error
    if not (procfs_root / str(pid)).exists():
        raise PreflightError(f"watchdog process {pid} is not running")
    try:
        command_bytes = (procfs_root / str(pid) / "cmdline").read_bytes()
        start_time_ticks = proc_start_time_ticks(
            (procfs_root / str(pid) / "stat").read_text(encoding="ascii"))
    except (OSError, ValueError) as error:
        raise PreflightError(f"cannot inspect watchdog process {pid}: {error}") from error
    command_sha256 = sha256_bytes(command_bytes)
    if start_time_ticks != expected_start or command_sha256 != expected_command_sha256:
        raise PreflightError("watchdog process identity does not match its lease")
    command_parts = [part.decode("utf-8", "replace") for part in command_bytes.split(b"\0") if part]
    try:
        watchdog_cwd = (procfs_root / str(pid) / "cwd").resolve()
    except OSError as error:
        raise PreflightError(f"cannot inspect watchdog process working directory: {error}") from error
    script_named = any(
        resolved(Path(argument) if Path(argument).is_absolute() else watchdog_cwd / argument) == expected_script
        for argument in command_parts
    )
    if not script_named:
        raise PreflightError("watchdog command does not execute the candidate repository script")
    child_command = lease.get("command")
    if not isinstance(child_command, list) or not child_command or (
            not all(isinstance(argument, str) for argument in child_command)):
        raise PreflightError("watchdog child command is invalid")
    child_command_sha256 = sha256_bytes(
        json.dumps(child_command, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))
    if child_command_sha256 != lease.get("child_command_sha256"):
        raise PreflightError("watchdog child command does not match the lease")
    heartbeat = read_heartbeat(
        heartbeat_path,
        max_age_seconds,
        lease_id=lease_id,
        watchdog_pid=pid,
        watchdog_start_time_ticks=start_time_ticks,
        child_pid=child_pid,
        child_pgid=child_pgid,
        now=now,
        monotonic_ns=monotonic_ns,
    )
    command = command_bytes.replace(b"\0", b" ").decode("utf-8", "replace").strip()
    if not command:
        raise PreflightError(f"watchdog process {pid} has no command line")
    events = _read_watchdog_startup_events(
        audit_path,
        timeout_seconds=timeout_seconds,
        monotonic=monotonic,
        sleeper=sleeper,
    )
    preflight = next((event for event in events if event.get("event") == "preflight"), None)
    child_started = next((event for event in events if event.get("event") == "child_started"), None)
    if preflight is None or child_started is None:
        raise PreflightError("watchdog audit lacks preflight or child_started evidence")
    if preflight.get("soft_bytes") != SOFT_MEMORY_LIMIT or (
            preflight.get("emergency_bytes") != WATCHDOG_EMERGENCY_LIMIT) or (
            preflight.get("strict_ceiling_bytes") != STRICT_MEMORY_LIMIT):
        raise PreflightError("watchdog audit thresholds are invalid")
    if preflight.get("swap_entries") != 0:
        raise PreflightError("watchdog audit does not report zero swap")
    if child_started.get("child_pid") != child_pid or (
            child_started.get("process_group_id") != child_pgid):
        raise PreflightError("watchdog audit child identity does not match the lease")
    if child_started.get("command") != lease.get("command"):
        raise PreflightError("watchdog audit child command does not match the lease")
    return {
        "format": WATCHDOG_LEASE_FORMAT,
        "version": WATCHDOG_VERSION,
        "lease_id": lease_id,
        "lease_path": str(pid_file),
        "watchdog_pid": pid,
        "watchdog_start_time_ticks": start_time_ticks,
        "watchdog_command": command,
        "watchdog_command_sha256": command_sha256,
        "watchdog_script_path": str(script_path),
        "watchdog_script_sha256": script_sha256,
        "soft_bytes": soft_bytes,
        "emergency_bytes": emergency_bytes,
        "strict_ceiling_bytes": strict_bytes,
        "procfs_root": procfs_path,
        "child_pid": child_pid,
        "child_process_group_id": child_pgid,
        "command": child_command,
        "child_command_sha256": child_command_sha256,
        "heartbeat_path": str(heartbeat_path),
        "heartbeat_unix": heartbeat,
        "max_heartbeat_age_seconds": max_age_seconds,
        "audit_path": str(audit_path),
        "audit_sha256": sha256_bytes(audit_path.read_bytes()),
        "audit_event_count": len(events),
    }


def process_ancestry(pid: int, procfs_root: Path) -> set[int]:
    ancestors = {pid}
    while pid > 1:
        try:
            parent = proc_parent_pid((procfs_root / str(pid) / "stat").read_text(encoding="ascii"))
        except (OSError, UnicodeError, ValueError):
            break
        if parent <= 1 or parent in ancestors:
            break
        ancestors.add(parent)
        pid = parent
    return ancestors


def matching_workloads(
        patterns: list[str],
        *,
        procfs_root: Path = Path("/proc"),
        current_pid: int | None = None) -> list[dict[str, Any]]:
    matches = []
    excluded = process_ancestry(os.getpid() if current_pid is None else current_pid, procfs_root)
    lowered = [pattern.lower() for pattern in patterns if pattern]
    for entry in procfs_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in excluded:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        except OSError:
            continue
        command_lower = command.lower()
        if any(pattern in command_lower for pattern in lowered):
            matches.append({"pid": pid, "command": command})
    return matches


def darwin_matching_workloads(
        patterns: list[str],
        *,
        command_text: Callable[..., str] = _command_text,
        current_pid: int | None = None) -> list[dict[str, Any]]:
    pid_to_parent = {}
    pid_to_command = {}
    for line in command_text("ps", "-axo", "pid=,ppid=,command=").splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3:
            continue
        try:
            pid = int(fields[0])
            parent = int(fields[1])
        except ValueError:
            continue
        pid_to_parent[pid] = parent
        pid_to_command[pid] = fields[2]
    excluded = set()
    pid = os.getpid() if current_pid is None else current_pid
    while pid > 0 and pid not in excluded:
        excluded.add(pid)
        pid = pid_to_parent.get(pid, 0)
    lowered = [pattern.lower() for pattern in patterns if pattern]
    return [
        {"pid": pid, "command": command}
        for pid, command in pid_to_command.items()
        if pid not in excluded and any(pattern in command.lower() for pattern in lowered)
    ]


def run_strix_preflight(
        *,
        model: Path,
        prompt: Path,
        output: Path,
        repo: Path,
        busy_patterns: list[str],
) -> dict[str, Any]:
    model_storage = storage_attestation(model, "model")
    prompt_storage = storage_attestation(prompt, "prompt")
    output_storage = storage_attestation(output, "trace output")
    repo_storage = storage_attestation(repo, "repository")
    tmpdir_value = os.environ.get("TMPDIR")
    if not tmpdir_value:
        raise PreflightError("TMPDIR is required for NVMe-only correctness runs")
    tmpdir_input = require_safe_tmpdir_path(Path(tmpdir_value))
    tmp_storage = storage_attestation(tmpdir_input, "temporary directory")
    tmpdir = _attested_resolved_path(tmp_storage, "temporary directory")
    if not tmpdir.is_dir() or not os.access(tmpdir, os.W_OK | os.X_OK):
        raise PreflightError("TMPDIR must be an existing writable directory")
    model = _attested_resolved_path(model_storage, "model")
    prompt = _attested_resolved_path(prompt_storage, "prompt")
    output = _attested_resolved_path(output_storage, "trace output")
    if not model.is_file():
        raise PreflightError(f"model is not a file: {model}")
    if not prompt.is_file():
        raise PreflightError(f"prompt is not a file: {prompt}")
    if os.environ.get("HIP_LAUNCH_BLOCKING") != "1":
        raise PreflightError("HIP_LAUNCH_BLOCKING=1 is required for gfx1151 correctness runs")
    swap = swap_audit()
    if swap["enabled"]:
        raise PreflightError("swap is enabled; model execution is blocked")
    watchdog = watchdog_audit(repo)
    workloads = matching_workloads(busy_patterns)
    if workloads:
        raise PreflightError("active model workload detected: " + json.dumps(workloads, ensure_ascii=True))
    return {
        "created_unix": int(time.time()),
        "runtime_kind": "strix-rocm",
        "model": str(model),
        "prompt": str(prompt),
        "output": str(output),
        "memory": memory_audit(),
        "swap": swap,
        "watchdog": watchdog,
        "active_workloads": [],
        "environment": {"HIP_LAUNCH_BLOCKING": "1"},
        "storage_policy": dict(NO_EXTERNAL_STATE_STORAGE),
        "storage": {
            "model": model_storage,
            "prompt": prompt_storage,
            "output": output_storage,
            "repository": repo_storage,
            "temporary_directory": tmp_storage,
        },
    }


def run_oracle_preflight(
        *,
        model: Path,
        prompt: Path,
        output: Path,
        repo: Path,
        checkout: Path,
        busy_patterns: list[str],
        accelerator: dict[str, Any],
        runner: dict[str, Any],
        disk_info: Callable[[Path], dict[str, Any]] = _diskutil_info,
        command_text: Callable[..., str] = _command_text,
        system: str | None = None,
        machine: str | None = None) -> dict[str, Any]:
    model_storage = darwin_storage_attestation(model, "model", disk_info=disk_info)
    prompt_storage = darwin_storage_attestation(prompt, "prompt", disk_info=disk_info)
    output_storage = darwin_storage_attestation(output, "trace output", disk_info=disk_info)
    repo_storage = darwin_storage_attestation(repo, "repository", disk_info=disk_info)
    checkout_storage = darwin_storage_attestation(checkout, "ds4 checkout", disk_info=disk_info)
    runner_executable = Path(str(runner.get("runner_executable", "")))
    runner_script = Path(str(runner.get("runner_script", "")))
    exporter = Path(str(runner.get("exporter_path", "")))
    runner_executable_storage = darwin_storage_attestation(
        runner_executable, "runner executable", disk_info=disk_info)
    runner_script_storage = darwin_storage_attestation(runner_script, "runner script", disk_info=disk_info)
    exporter_storage = darwin_storage_attestation(exporter, "trace exporter", disk_info=disk_info)
    tmpdir_value = os.environ.get("TMPDIR")
    if not tmpdir_value:
        raise PreflightError("TMPDIR is required for ds4 oracle correctness runs")
    tmpdir_input = require_safe_tmpdir_path(Path(tmpdir_value))
    tmp_storage = darwin_storage_attestation(tmpdir_input, "temporary directory", disk_info=disk_info)
    tmpdir = _attested_resolved_path(tmp_storage, "temporary directory")
    if not tmpdir.is_dir() or not os.access(tmpdir, os.W_OK | os.X_OK):
        raise PreflightError("TMPDIR must be an existing writable directory")
    model = _attested_resolved_path(model_storage, "model")
    prompt = _attested_resolved_path(prompt_storage, "prompt")
    output = _attested_resolved_path(output_storage, "trace output")
    repo = _attested_resolved_path(repo_storage, "repository")
    checkout = _attested_resolved_path(checkout_storage, "ds4 checkout")
    runner_executable = _attested_resolved_path(runner_executable_storage, "runner executable")
    runner_script = _attested_resolved_path(runner_script_storage, "runner script")
    exporter = _attested_resolved_path(exporter_storage, "trace exporter")
    if not model.is_file():
        raise PreflightError(f"model is not a file: {model}")
    if not prompt.is_file():
        raise PreflightError(f"prompt is not a file: {prompt}")
    if not runner_executable.is_file() or not runner_script.is_file() or not exporter.is_file():
        raise PreflightError("ds4 runner or exporter path is not a file")
    try:
        runner_script.relative_to(repo)
    except ValueError as error:
        raise PreflightError("ds4 runner script is outside the attested repository") from error
    for key, path in (
            ("runner_executable_sha256", runner_executable),
            ("runner_script_sha256", runner_script),
            ("exporter_sha256", exporter)):
        try:
            digest = sha256_bytes(path.read_bytes())
        except OSError as error:
            raise PreflightError(f"cannot hash ds4 {key} path: {error}") from error
        if runner.get(key) != digest:
            raise PreflightError(f"ds4 {key} differs from the executed file")
    if runner.get("checkout_path") != str(checkout):
        raise PreflightError("ds4 runner checkout path differs from the attested checkout")
    host, memory = darwin_host_and_memory_audit(
        command_text=command_text,
        system=system,
        machine=machine,
    )
    swap = darwin_swap_audit(command_text=command_text)
    workloads = darwin_matching_workloads(busy_patterns, command_text=command_text)
    if workloads:
        raise PreflightError("active model workload detected: " + json.dumps(workloads, ensure_ascii=True))
    return {
        "created_unix": int(time.time()),
        "runtime_kind": "apple-metal",
        "model": str(model),
        "prompt": str(prompt),
        "output": str(output),
        "memory": memory,
        "swap": swap,
        "runner": runner,
        "host": host,
        "accelerator": accelerator,
        "active_workloads": [],
        "environment": {},
        "storage_policy": dict(NO_EXTERNAL_STATE_STORAGE),
        "storage": {
            "model": model_storage,
            "prompt": prompt_storage,
            "output": output_storage,
            "repository": repo_storage,
            "runtime_checkout": checkout_storage,
            "temporary_directory": tmp_storage,
            "runner_executable": runner_executable_storage,
            "runner_script": runner_script_storage,
            "exporter": exporter_storage,
        },
    }


def write_audits(root: Path, audit: dict[str, Any]) -> dict[str, str]:
    root = resolved(root)
    if root.exists() and any(root.iterdir()):
        raise PreflightError(f"audit directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    runtime_kind = audit.get("runtime_kind")
    kinds = (
        ("memory", "swap", "watchdog")
        if runtime_kind == "strix-rocm"
        else ("memory", "swap", "runner")
        if runtime_kind == "apple-metal"
        else ()
    )
    if not kinds:
        raise PreflightError("audit runtime kind is invalid")
    for key in kinds:
        path = root / f"{key}.json"
        data = audit[key]
        if key == "watchdog":
            data = dict(data)
            live_audit_path = resolved(Path(str(data["audit_path"])))
            try:
                audit_bytes = live_audit_path.read_bytes()
                audit_text = audit_bytes.decode("ascii")
            except (OSError, UnicodeError) as error:
                raise PreflightError(f"cannot snapshot watchdog audit: {error}") from error
            events = []
            for line_number, line in enumerate(audit_text.splitlines(), start=1):
                try:
                    event = validate_watchdog_event(strict_json_loads(line))
                except (PreflightError, TraceError) as error:
                    raise PreflightError(
                        f"watchdog audit line {line_number} is invalid while snapshotting: {error}") from error
                events.append(event)
            if not events:
                raise PreflightError("watchdog audit snapshot is empty")
            snapshot = root / "watchdog-events.jsonl"
            snapshot.write_bytes(audit_bytes)
            data["audit_live_path"] = str(live_audit_path)
            data["audit_path"] = str(snapshot)
            data["audit_sha256"] = sha256_bytes(audit_bytes)
            data["audit_event_count"] = len(events)
        value = {
            "created_unix": audit["created_unix"],
            "kind": key,
            "data": data,
            "environment": audit["environment"],
        }
        if key == "memory":
            value["storage"] = audit["storage"]
            value["storage_policy"] = audit["storage_policy"]
            value["accelerator"] = audit["accelerator"]
            if "host" in audit:
                value["host"] = audit["host"]
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
        result[key] = str(path)
    summary = root / "preflight.json"
    summary.write_text(json.dumps(audit, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    result["preflight"] = str(summary)
    return result


def seal_audits(audits: dict[str, str]) -> dict[str, str]:
    digests = {}
    paths = []
    try:
        kinds = tuple(kind for kind in ("memory", "swap", "watchdog", "runner") if kind in audits)
        if set(kinds) != set(audits) - {"preflight"}:
            raise PreflightError("audit set has unsupported kinds")
        for kind in kinds:
            path = resolved(Path(audits[kind]))
            paths.append(path)
            data = path.read_bytes()
            digests[kind] = sha256_bytes(data)
            if kind == "watchdog":
                record = strict_json_loads(data.decode("ascii"))
                jsonl_path = resolved(Path(record["data"]["audit_path"]))
                paths.append(jsonl_path)
                digests["watchdog_jsonl"] = sha256_bytes(jsonl_path.read_bytes())
        for path in paths:
            path.chmod(0o444)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise PreflightError(f"cannot seal audit evidence: {error}") from error
    return digests


def verify_sealed_audits(audits: dict[str, str], digests: dict[str, str]) -> None:
    current = seal_audits(audits)
    if current != digests:
        raise PreflightError("preflight audit evidence changed during runtime execution")


def embed_audits(trace_root: Path, phase: str, audits: dict[str, str]) -> dict[str, dict[str, Any]]:
    trace_root = safe_trace_path(trace_root, ".")
    embedded_root = safe_trace_path(trace_root, Path("audits") / phase)
    embedded_root.mkdir(parents=True, exist_ok=True)
    result = {}
    kinds = tuple(kind for kind in ("memory", "swap", "watchdog", "runner") if kind in audits)
    if set(kinds) != set(audits) - {"preflight"}:
        raise PreflightError("audit set has unsupported kinds")
    for kind in kinds:
        source = resolved(Path(audits[kind]))
        data = source.read_bytes()
        try:
            record = strict_json_loads(data.decode("ascii"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise PreflightError(f"cannot embed {kind} audit: {error}") from error
        if kind == "watchdog":
            try:
                jsonl_source = resolved(Path(record["data"].pop("audit_path")))
                jsonl_data = jsonl_source.read_bytes()
            except (OSError, TypeError, KeyError) as error:
                raise PreflightError(f"cannot embed watchdog JSONL audit: {error}") from error
            jsonl_digest = sha256_bytes(jsonl_data)
            if record["data"].get("audit_sha256") != jsonl_digest:
                raise PreflightError("watchdog JSONL audit SHA-256 changed before embedding")
            jsonl_destination = safe_trace_path(
                trace_root, Path("audits") / phase / f"{jsonl_digest}.jsonl")
            if jsonl_destination.exists() and jsonl_destination.read_bytes() != jsonl_data:
                raise PreflightError(f"content-addressed watchdog audit collision: {jsonl_destination}")
            if not jsonl_destination.exists():
                jsonl_destination.write_bytes(jsonl_data)
            record["data"]["audit"] = {
                "path": f"audits/{phase}/{jsonl_digest}.jsonl",
                "sha256": jsonl_digest,
                "event_count": record["data"].pop("audit_event_count"),
            }
            data = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
        digest = sha256_bytes(data)
        destination = safe_trace_path(trace_root, Path("audits") / phase / f"{digest}.json")
        if destination.exists() and destination.read_bytes() != data:
            raise PreflightError(f"content-addressed audit collision: {destination}")
        if not destination.exists():
            destination.write_bytes(data)
        try:
            created = int(record["created_unix"])
        except (ValueError, TypeError, KeyError) as error:
            raise PreflightError(f"cannot embed {kind} audit: {error}") from error
        result[kind] = {
            "path": f"audits/{phase}/{digest}.json",
            "sha256": digest,
            "created_unix": created,
        }
    return result


def bind_embedded_audits(trace_root: Path, audit_sets: dict[str, dict[str, str]]) -> None:
    trace_root = safe_trace_path(trace_root, ".")
    manifest_path = safe_trace_path(trace_root, "manifest.json")
    try:
        manifest = strict_json_loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"cannot bind trace audits: {error}") from error
    if set(audit_sets) != {"pre", "post"}:
        raise PreflightError("trace requires pre and post audit sets")
    manifest["audits"] = {
        phase: embed_audits(trace_root, phase, audits)
        for phase, audits in audit_sets.items()
    }
    temp = manifest_path.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    os.replace(temp, manifest_path)


def validate_prompt_provenance(
    path: Path,
    *,
    prompt: Path,
    corpus_name: str,
    corpus_sha256: str,
    model_sha256: str,
    target_tokens: int,
    context: int,
    decode_steps: int,
    builder_approval_id: str,
    builder_policy: dict[str, Any],
    builder_policy_sha256: str,
    path_resolver: Callable[[Path, str], Path] | None = None,
) -> dict[str, Any]:
    path = (require_nvme_path if path_resolver is None else path_resolver)(path, "prompt provenance")
    try:
        data = path.read_bytes()
        record = strict_json_loads(data.decode("ascii"))
        prompt_path = resolved(prompt)
        prompt_bytes = prompt_path.read_bytes()
        prompt_size = prompt_path.stat().st_size
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"prompt provenance is invalid: {error}") from error
    if not isinstance(record, dict):
        raise PreflightError("prompt provenance must be a JSON object")
    source_root_lexical = Path(str(builder_policy["source_root"]))
    source_root_resolved = resolved(source_root_lexical)
    corpus_resolved = resolved(source_root_resolved / "tests" / "corpus" / corpus_name)
    expected = {
        "format": "dsv41-prompt-provenance",
        "version": 2,
        "corpus_name": corpus_name,
        "corpus_sha256": corpus_sha256,
        "corpus_path": str(corpus_resolved),
        "corpus_resolved_path": str(corpus_resolved),
        "source_root_lexical_path": str(source_root_lexical),
        "source_root_resolved_path": str(source_root_resolved),
        "model_sha256": model_sha256,
        "prompt_sha256": sha256_bytes(prompt_bytes),
        "prompt_byte_count": prompt_size,
        "context": context,
        "decode_steps": decode_steps,
        "target_tokens": target_tokens,
        "actual_tokens": target_tokens,
        "builder_approval_id": builder_approval_id,
        "builder_approval_sha256": builder_policy_sha256,
        "builder_path": builder_policy["executable_path"],
        "builder_sha256": builder_policy["executable_sha256"],
        "builder_revision": builder_policy["revision"],
        "builder_runtime_profile": builder_policy["runtime_profile"],
        "tokenizer": builder_policy["tokenizer"],
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise PreflightError(f"prompt provenance {key} mismatch")
    required = set(expected) | {
        "corpus_lexical_path",
        "builder_runtime_build",
        "builder_runtime_build_sha256",
        "builder_install_trust",
        "builder_install_trust_sha256",
    }
    if set(record) != required:
        raise PreflightError("prompt provenance fields are invalid")
    try:
        if resolved(Path(str(record["corpus_lexical_path"]))) != corpus_resolved:
            raise PreflightError("prompt provenance corpus lexical path resolves outside the approved source")
    except OSError as error:
        raise PreflightError(f"prompt provenance corpus lexical path is invalid: {error}") from error
    try:
        runtime_build = validate_runtime_build_evidence(
            record["builder_runtime_build"], builder_policy, label="prompt builder")
        runtime_build_sha256 = runtime_build_evidence_sha256(
            runtime_build, builder_policy, label="prompt builder")
        trust = validate_install_trust_evidence(record["builder_install_trust"], builder_policy)
        trust_sha256 = install_trust_sha256(trust)
    except TraceError as error:
        raise PreflightError(f"prompt provenance runtime trust is invalid: {error}") from error
    if record["builder_runtime_build_sha256"] != runtime_build_sha256:
        raise PreflightError("prompt provenance runtime build SHA-256 mismatch")
    if record["builder_install_trust_sha256"] != trust_sha256:
        raise PreflightError("prompt provenance install trust SHA-256 mismatch")
    matches = [
        prompt_record for prompt_record in builder_policy["prompts"]
        if prompt_record["corpus_name"] == corpus_name and
        prompt_record["context"] == context and
        prompt_record["decode_steps"] == decode_steps
    ]
    if len(matches) != 1:
        raise PreflightError("prompt provenance configuration is not externally approved")
    for key in (
            "corpus_name", "corpus_sha256", "context", "decode_steps", "target_tokens",
            "prompt_sha256", "prompt_byte_count"):
        if record[key] != matches[0][key]:
            raise PreflightError(f"prompt provenance {key} differs from external approval")
    return {"path": str(path), "bytes": data, "record": record}


def bind_prompt_provenance(trace_root: Path, provenance: dict[str, Any]) -> None:
    trace_root = safe_trace_path(trace_root, ".")
    manifest_path = safe_trace_path(trace_root, "manifest.json")
    try:
        manifest = strict_json_loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"cannot bind prompt provenance: {error}") from error
    prompt = manifest.get("prompt")
    if not isinstance(prompt, dict):
        raise PreflightError("trace manifest prompt is invalid")
    record = provenance["record"]
    data = provenance["bytes"]
    if not isinstance(record, dict) or not isinstance(data, bytes):
        raise PreflightError("validated prompt provenance is invalid")
    digest = sha256_bytes(data)
    provenance_root = safe_trace_path(trace_root, "provenance")
    provenance_root.mkdir(parents=True, exist_ok=True)
    destination = safe_trace_path(trace_root, Path("provenance") / f"{digest}.json")
    if destination.exists() and destination.read_bytes() != data:
        raise PreflightError(f"content-addressed provenance collision: {destination}")
    if not destination.exists():
        destination.write_bytes(data)
    prompt["corpus_name"] = record["corpus_name"]
    prompt["corpus_sha256"] = record["corpus_sha256"]
    prompt["target_tokens"] = record["target_tokens"]
    prompt["provenance"] = {
        "path": f"provenance/{digest}.json",
        "sha256": digest,
    }
    temp = manifest_path.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    os.replace(temp, manifest_path)
