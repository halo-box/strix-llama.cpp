#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import select
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Protocol


GIB = 1024**3
STRICT_CEILING_BYTES = 120 * GIB
DEFAULT_SOFT_BYTES = 116 * GIB
DEFAULT_EMERGENCY_BYTES = 118 * GIB
DEFAULT_GRACE_SECONDS = 30.0
DEFAULT_SAMPLE_INTERVAL_SECONDS = 1.0
DEFAULT_HEARTBEAT_MAX_AGE_SECONDS = 5.0
MAX_GRACE_SECONDS = 30.0
MAX_SAMPLE_INTERVAL_SECONDS = 1.0
MAX_HEARTBEAT_MAX_AGE_SECONDS = 5.0

LEASE_FORMAT = "strix-memory-watchdog-lease"
LEASE_VERSION = 2
HEARTBEAT_FORMAT = "strix-memory-watchdog-heartbeat"
HEARTBEAT_VERSION = 2
PR_SET_PDEATHSIG = 1
LEASE_GUARD_SIGNAL = signal.SIGUSR1

EXIT_PROCFS_ERROR = 2
EXIT_SWAP_ACTIVE = 3
EXIT_SOFT_LIMIT = 4
EXIT_EMERGENCY_LIMIT = 5
EXIT_GRACE_TIMEOUT = 6
EXIT_SIGNAL_ERROR = 7
EXIT_LEASE_ERROR = 8
EXIT_INTERNAL_ERROR = 70
EXIT_LAUNCH_ERROR = 127

MEMINFO_VALUE_RE = re.compile(r"([0-9]+) kB")
SWAPS_HEADER = ["Filename", "Type", "Size", "Used", "Priority"]
PARENT_SIGNALS = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)


class ProcfsError(RuntimeError):
    pass


class ProcessGroupError(RuntimeError):
    pass


class ArtifactError(RuntimeError):
    def __init__(self, component: str, detail: str):
        self.component = component
        super().__init__(detail)


class LeaseValidationError(RuntimeError):
    pass


class ParentSignal(RuntimeError):
    def __init__(self, signal_number: int):
        self.signal_number = signal_number
        super().__init__(signal.Signals(signal_number).name)


class ProcessHandle(Protocol):
    pid: int

    def poll(self) -> int | None:
        ...

    def wait(self, timeout: float | None = None) -> int:
        ...


@dataclass
class GuardianProcess:
    process: subprocess.Popen[bytes]
    payload_pid: int
    pulse_fd: int

    @property
    def pid(self) -> int:
        return self.process.pid

    def poll(self) -> int | None:
        return self.process.poll()

    def wait(self, timeout: float | None = None) -> int:
        return self.process.wait(timeout=timeout)

    def pulse(self) -> None:
        self._write_control(b"P")

    def begin_grace(self) -> None:
        self._write_control(b"G")

    def _write_control(self, value: bytes) -> None:
        try:
            os.write(self.pulse_fd, value)
        except BlockingIOError as exc:
            raise ProcessGroupError(
                "guardian pulse pipe is blocked"
            ) from exc
        except OSError as exc:
            detail = exc.strerror or str(exc)
            raise ProcessGroupError(
                f"cannot pulse guardian: {detail}"
            ) from exc

    def close(self) -> None:
        try:
            os.close(self.pulse_fd)
        except OSError:
            pass


@dataclass(frozen=True)
class HostSnapshot:
    total_bytes: int
    available_bytes: int
    active_swaps: tuple[str, ...]

    @property
    def used_bytes(self) -> int:
        return self.total_bytes - self.available_bytes


@dataclass
class RuntimeState:
    snapshot: HostSnapshot
    peak_used_bytes: int


@dataclass(frozen=True)
class ArtifactPaths:
    lease: Path
    heartbeat: Path
    audit: Path


@dataclass(frozen=True)
class WatchdogConfig:
    command: tuple[str, ...]
    procfs_root: Path = Path("/proc")
    soft_bytes: int = DEFAULT_SOFT_BYTES
    emergency_bytes: int = DEFAULT_EMERGENCY_BYTES
    grace_seconds: float = DEFAULT_GRACE_SECONDS
    sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS
    lease_path: Path | None = None
    heartbeat_path: Path | None = None
    audit_path: Path | None = None
    heartbeat_max_age_seconds: float = DEFAULT_HEARTBEAT_MAX_AGE_SECONDS

    @property
    def lease_enabled(self) -> bool:
        return self.lease_path is not None

    def validate(self) -> ArtifactPaths | None:
        if not self.command:
            raise ValueError("a command is required after --")
        if self.soft_bytes <= 0:
            raise ValueError("soft threshold must be greater than zero")
        if self.emergency_bytes <= self.soft_bytes:
            raise ValueError("emergency threshold must be greater than soft threshold")
        if self.emergency_bytes >= STRICT_CEILING_BYTES:
            raise ValueError("emergency threshold must be below 120 GiB")
        if (
            not math.isfinite(self.grace_seconds)
            or self.grace_seconds <= 0
            or self.grace_seconds > MAX_GRACE_SECONDS
        ):
            raise ValueError(
                "grace period must be greater than zero and at most 30 seconds"
            )
        if (
            not math.isfinite(self.sample_interval_seconds)
            or self.sample_interval_seconds <= 0
            or self.sample_interval_seconds > MAX_SAMPLE_INTERVAL_SECONDS
        ):
            raise ValueError(
                "sample interval must be greater than zero and at most 1 second"
            )
        if (
            not math.isfinite(self.heartbeat_max_age_seconds)
            or self.heartbeat_max_age_seconds
            <= self.sample_interval_seconds
            or self.heartbeat_max_age_seconds
            > MAX_HEARTBEAT_MAX_AGE_SECONDS
        ):
            raise ValueError(
                "heartbeat max age must be greater than sample interval "
                "and at most 5 seconds"
            )
        lease_paths = (
            self.lease_path,
            self.heartbeat_path,
            self.audit_path,
        )
        if any(path is not None for path in lease_paths) and not all(
            path is not None for path in lease_paths
        ):
            raise ValueError(
                "lease, heartbeat, and audit paths must be specified together"
            )
        if self.lease_enabled:
            assert self.lease_path is not None
            assert self.heartbeat_path is not None
            assert self.audit_path is not None
            try:
                paths = ArtifactPaths(
                    self.lease_path.expanduser().resolve(),
                    self.heartbeat_path.expanduser().resolve(),
                    self.audit_path.expanduser().resolve(),
                )
            except (OSError, RuntimeError) as exc:
                raise ValueError(
                    f"cannot resolve watchdog artifact path: {exc}"
                ) from exc
            if len({paths.lease, paths.heartbeat, paths.audit}) != 3:
                raise ValueError(
                    "lease, heartbeat, and audit paths must be distinct"
                )
            return paths
        return None


class ProcfsReader:
    def __init__(self, root: Path):
        self.root = root

    def _read_text(self, name: str) -> str:
        path = self.root / name
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            detail = exc.strerror or str(exc)
            raise ProcfsError(f"cannot read {path}: {detail}") from exc

    def read_snapshot(self) -> HostSnapshot:
        active_swaps = self._parse_swaps(self._read_text("swaps"))
        total_bytes, available_bytes = self._parse_meminfo(
            self._read_text("meminfo")
        )
        return HostSnapshot(total_bytes, available_bytes, active_swaps)

    @staticmethod
    def _parse_meminfo(content: str) -> tuple[int, int]:
        values: dict[str, int] = {}
        required = {"MemTotal", "MemAvailable"}
        for line in content.splitlines():
            key, separator, raw_value = line.partition(":")
            if not separator or key not in required:
                continue
            if key in values:
                raise ProcfsError(f"duplicate {key} in meminfo")
            match = MEMINFO_VALUE_RE.fullmatch(raw_value.strip())
            if match is None:
                raise ProcfsError(f"malformed {key} in meminfo")
            values[key] = int(match.group(1)) * 1024

        missing = sorted(required - values.keys())
        if missing:
            raise ProcfsError(f"missing {', '.join(missing)} in meminfo")
        if values["MemAvailable"] > values["MemTotal"]:
            raise ProcfsError("MemAvailable exceeds MemTotal")
        return values["MemTotal"], values["MemAvailable"]

    @staticmethod
    def _parse_swaps(content: str) -> tuple[str, ...]:
        lines = content.splitlines()
        if not lines or lines[0].split() != SWAPS_HEADER:
            raise ProcfsError("malformed swaps header")

        entries: list[str] = []
        for line in lines[1:]:
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != len(SWAPS_HEADER):
                raise ProcfsError("malformed swaps entry")
            try:
                int(fields[2])
                int(fields[3])
                int(fields[4])
            except ValueError as exc:
                raise ProcfsError("malformed swaps entry") from exc
            entries.append(fields[0])
        return tuple(entries)


def _timestamp_utc(
    wall_clock: Callable[[], datetime] | None = None,
) -> str:
    timestamp = (wall_clock or (
        lambda: datetime.now(timezone.utc)
    ))().astimezone(timezone.utc)
    return timestamp.isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise ArtifactError(
            "lease", f"cannot hash {path}: {detail}"
        ) from exc


def _command_sha256(command: Sequence[str]) -> str:
    encoded = json.dumps(
        list(command),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _set_parent_death_signal(
    signal_number: int, expected_parent_pid: int
) -> None:
    if not sys.platform.startswith("linux"):
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal_number, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    if os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)


def _kill_own_process_group(
    _signal_number: int | None = None,
    _frame: object | None = None,
) -> None:
    try:
        os.killpg(os.getpgrp(), signal.SIGKILL)
    except OSError:
        os._exit(EXIT_SIGNAL_ERROR)


def _guardian_main(
    control_fd: int,
    status_fd: int,
    pulse_timeout_seconds: float,
    grace_timeout_seconds: float,
    command: tuple[str, ...],
) -> int:
    if not sys.platform.startswith("linux"):
        return EXIT_LAUNCH_ERROR
    os.set_inheritable(control_fd, False)
    os.set_inheritable(status_fd, False)
    signal.signal(LEASE_GUARD_SIGNAL, _kill_own_process_group)
    for signal_number in PARENT_SIGNALS:
        signal.signal(signal_number, signal.SIG_IGN)
    _set_parent_death_signal(LEASE_GUARD_SIGNAL, os.getppid())

    def prepare_payload() -> None:
        for signal_number in PARENT_SIGNALS:
            signal.signal(signal_number, signal.SIG_DFL)

    try:
        payload = subprocess.Popen(command, preexec_fn=prepare_payload)
    except (OSError, ValueError) as exc:
        os.write(
            status_fd,
            json.dumps(
                {"error": getattr(exc, "strerror", None) or str(exc)}
            ).encode("utf-8")
            + b"\n",
        )
        os.close(status_fd)
        return EXIT_LAUNCH_ERROR

    os.write(
        status_fd,
        json.dumps({"payload_pid": payload.pid}).encode("utf-8") + b"\n",
    )
    os.close(status_fd)
    poller = select.poll()
    poller.register(
        control_fd,
        select.POLLIN | select.POLLHUP | select.POLLERR,
    )
    current_timeout_seconds = pulse_timeout_seconds
    deadline = time.monotonic() + pulse_timeout_seconds
    while True:
        remaining = max(0.0, deadline - time.monotonic())
        events = poller.poll(max(1, min(50, int(remaining * 1000))))
        for _, event_mask in events:
            if event_mask & (select.POLLHUP | select.POLLERR):
                _kill_own_process_group()
            try:
                pulse = os.read(control_fd, 65536)
            except BlockingIOError:
                pulse = b""
            if not pulse:
                _kill_own_process_group()
            if b"G" in pulse:
                current_timeout_seconds = grace_timeout_seconds
            deadline = time.monotonic() + current_timeout_seconds
        if time.monotonic() >= deadline:
            _kill_own_process_group()
        returncode = payload.poll()
        if returncode is not None:
            if returncode >= 0:
                return returncode
            signal_number = -returncode
            if signal_number not in (signal.SIGKILL, signal.SIGSTOP):
                signal.signal(signal_number, signal.SIG_DFL)
            os.kill(os.getpid(), signal_number)
            return 128 + signal_number


def _read_guardian_status(
    descriptor: int, timeout_seconds: float
) -> int:
    poller = select.poll()
    poller.register(descriptor, select.POLLIN | select.POLLHUP)
    deadline = time.monotonic() + timeout_seconds
    content = b""
    while time.monotonic() < deadline:
        events = poller.poll(
            max(1, int((deadline - time.monotonic()) * 1000))
        )
        if not events:
            continue
        chunk = os.read(descriptor, 4096)
        if not chunk:
            break
        content += chunk
        if b"\n" in content:
            break
    if not content:
        raise OSError("guardian did not report payload startup")
    try:
        status = json.loads(content.splitlines()[0])
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OSError("guardian returned malformed startup status") from exc
    if not isinstance(status, dict):
        raise OSError("guardian returned malformed startup status")
    if "error" in status:
        raise OSError(str(status["error"]))
    payload_pid = status.get("payload_pid")
    if not isinstance(payload_pid, int):
        raise OSError("guardian did not report a payload PID")
    return payload_pid


def _launch_guardian(
    command: tuple[str, ...],
    environment: dict[str, str],
    pulse_timeout_seconds: float,
    grace_timeout_seconds: float,
    launch_mask: set[signal.Signals],
) -> GuardianProcess:
    control_read, control_write = os.pipe()
    os.set_blocking(control_read, False)
    os.set_blocking(control_write, False)
    status_read, status_write = os.pipe()
    parent_pid = os.getpid()

    def prepare_guardian() -> None:
        signal.pthread_sigmask(signal.SIG_SETMASK, launch_mask)
        _set_parent_death_signal(signal.SIGKILL, parent_pid)

    guardian_command = (
        sys.executable,
        str(Path(__file__).resolve()),
        "--internal-guardian",
        str(control_read),
        str(status_write),
        str(pulse_timeout_seconds),
        str(grace_timeout_seconds),
        "--",
        *command,
    )
    try:
        process = subprocess.Popen(
            guardian_command,
            start_new_session=True,
            pass_fds=(control_read, status_write),
            preexec_fn=prepare_guardian,
            env=environment,
        )
    finally:
        os.close(control_read)
        os.close(status_write)
    try:
        payload_pid = _read_guardian_status(status_read, 5.0)
    except OSError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5.0)
        os.close(control_write)
        raise
    finally:
        os.close(status_read)
    guardian = GuardianProcess(process, payload_pid, control_write)
    guardian.pulse()
    return guardian


def _read_proc_bytes(root: Path, process_id: int, name: str) -> bytes:
    path = root / str(process_id) / name
    try:
        return path.read_bytes()
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise LeaseValidationError(
            f"cannot read {path}: {detail}"
        ) from exc


def _parse_proc_stat(content: str) -> tuple[int, int, int]:
    close_paren = content.rfind(")")
    if close_paren < 0:
        raise LeaseValidationError("malformed process stat")
    fields = content[close_paren + 1:].split()
    if len(fields) < 20:
        raise LeaseValidationError("malformed process stat")
    try:
        return int(fields[1]), int(fields[2]), int(fields[19])
    except ValueError as exc:
        raise LeaseValidationError("malformed process stat") from exc


def _read_proc_stat(
    root: Path, process_id: int
) -> tuple[int, int, int]:
    content = _read_proc_bytes(
        root, process_id, "stat"
    ).decode("utf-8")
    return _parse_proc_stat(content)


def _write_json_atomic(
    path: Path,
    value: dict[str, object],
    *,
    create: bool = False,
) -> None:
    parent = path.parent
    temp_path = parent / (
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    try:
        descriptor = os.open(
            temp_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            file_status = os.fstat(stream.fileno())
            record = {
                **value,
                "file_device": file_status.st_dev,
                "file_inode": file_status.st_ino,
                "file_uid": file_status.st_uid,
                "file_mode": stat.S_IMODE(file_status.st_mode),
            }
            payload = (
                json.dumps(
                    record,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if create:
            os.link(temp_path, path)
            temp_path.unlink()
        else:
            os.replace(temp_path, path)
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as exc:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        detail = exc.strerror or str(exc)
        action = "create" if create else "write"
        raise ArtifactError(
            "lease", f"cannot atomically {action} {path}: {detail}"
        ) from exc


class LeaseManager:
    def __init__(
        self,
        config: WatchdogConfig,
        paths: ArtifactPaths,
        *,
        process_procfs_root: Path = Path("/proc"),
        wall_clock: Callable[[], datetime] | None = None,
        monotonic_ns: Callable[[], int] | None = None,
    ):
        self.config = config
        self.lease_path = paths.lease
        self.heartbeat_path = paths.heartbeat
        self.audit_path = paths.audit
        self.process_procfs_root = process_procfs_root
        self.wall_clock = wall_clock
        self.monotonic_ns = monotonic_ns or time.monotonic_ns
        self.lease_id = secrets.token_hex(16)
        self.sequence = 0
        self.lease: dict[str, object] | None = None

    def _watchdog_identity(self) -> dict[str, object]:
        script_path = Path(__file__).resolve()
        cmdline_path = (
            self.process_procfs_root / str(os.getpid()) / "cmdline"
        )
        proc_start_time_ticks: int | None = None
        try:
            cmdline = cmdline_path.read_bytes()
            _, _, proc_start_time_ticks = _read_proc_stat(
                self.process_procfs_root, os.getpid()
            )
            executable_path = (
                self.process_procfs_root
                / str(os.getpid())
                / "exe"
            ).resolve()
        except (OSError, LeaseValidationError):
            if sys.platform.startswith("linux"):
                raise ArtifactError(
                    "lease",
                    "cannot read watchdog process identity from procfs",
                )
            cmdline = b"\0".join(
                os.fsencode(argument) for argument in sys.argv
            )
            executable_path = Path(sys.executable).resolve()
        return {
            "pid": os.getpid(),
            "start_time_utc": _timestamp_utc(self.wall_clock),
            "proc_start_time_ticks": proc_start_time_ticks,
            "cmdline_sha256": _sha256_bytes(cmdline),
            "executable_path": str(executable_path),
            "script_path": str(script_path),
            "script_sha256": _sha256_file(script_path),
        }

    def _heartbeat_record(
        self,
        state: str,
        sample: dict[str, object] | None = None,
    ) -> dict[str, object]:
        assert self.lease is not None
        self.sequence += 1
        record: dict[str, object] = {
            "format": HEARTBEAT_FORMAT,
            "version": HEARTBEAT_VERSION,
            "lease_id": self.lease_id,
            "sequence": self.sequence,
            "state": state,
            "updated_at": _timestamp_utc(self.wall_clock),
            "updated_monotonic_ns": self.monotonic_ns(),
            "watchdog_pid": self.lease["watchdog_pid"],
            "watchdog_start_time_ticks": (
                self.lease["watchdog_start_time_ticks"]
            ),
            "child_pid": self.lease["child_pid"],
            "child_process_group_id": (
                self.lease["child_process_group_id"]
            ),
        }
        if sample is not None:
            record["sample"] = sample
        return record

    def start(
        self, child: ProcessHandle, audit: AuditLogger
    ) -> None:
        watchdog_identity = self._watchdog_identity()
        audit_identity = audit.persistent_identity()
        payload_pid = (
            child.payload_pid
            if isinstance(child, GuardianProcess)
            else child.pid
        )
        self.lease = {
            "format": LEASE_FORMAT,
            "version": LEASE_VERSION,
            "lease_id": self.lease_id,
            "state": "active",
            "watchdog_pid": watchdog_identity["pid"],
            "watchdog_start_time_utc": (
                watchdog_identity["start_time_utc"]
            ),
            "watchdog_start_time_ticks": (
                watchdog_identity["proc_start_time_ticks"]
            ),
            "watchdog_command_sha256": (
                watchdog_identity["cmdline_sha256"]
            ),
            "watchdog_executable_path": (
                watchdog_identity["executable_path"]
            ),
            "watchdog_script_path": watchdog_identity["script_path"],
            "watchdog_script_sha256": (
                watchdog_identity["script_sha256"]
            ),
            "soft_bytes": self.config.soft_bytes,
            "emergency_bytes": self.config.emergency_bytes,
            "strict_ceiling_bytes": STRICT_CEILING_BYTES,
            "grace_seconds": self.config.grace_seconds,
            "sample_interval_seconds": self.config.sample_interval_seconds,
            "guardian_pid": child.pid,
            "child_pid": payload_pid,
            "child_process_group_id": child.pid,
            "command": list(self.config.command),
            "child_command_sha256": _command_sha256(
                self.config.command
            ),
            "heartbeat_path": str(self.heartbeat_path),
            "max_heartbeat_age_seconds": (
                self.config.heartbeat_max_age_seconds
            ),
            "audit_path": str(self.audit_path),
            "audit_device": audit_identity["device"],
            "audit_inode": audit_identity["inode"],
            "audit_uid": audit_identity["uid"],
            "audit_mode": audit_identity["mode"],
            "audit_fd": audit_identity["fd"],
            "procfs_root": str(
                self.config.procfs_root.expanduser().resolve()
            ),
        }
        heartbeat = self._heartbeat_record(
            "active",
            {"audit_record_sha256": audit.last_record_sha256},
        )
        _write_json_atomic(self.heartbeat_path, heartbeat, create=True)
        _write_json_atomic(self.lease_path, self.lease, create=True)

    def update_heartbeat(self, sample: dict[str, object]) -> None:
        heartbeat = self._heartbeat_record("active", sample)
        _write_json_atomic(self.heartbeat_path, heartbeat)

    def finalize(self, final_record: dict[str, object]) -> None:
        if self.lease is None:
            return
        self.lease["state"] = "final"
        self.lease["final"] = final_record
        heartbeat = self._heartbeat_record("final")
        _write_json_atomic(self.heartbeat_path, heartbeat)
        _write_json_atomic(self.lease_path, self.lease)


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            file_status = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(file_status.st_mode)
                or file_status.st_uid != os.getuid()
                or stat.S_IMODE(file_status.st_mode) != 0o600
            ):
                raise LeaseValidationError(
                    f"{path} has unsafe type, owner, or mode"
                )
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LeaseValidationError(
            f"cannot read valid JSON from {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise LeaseValidationError(f"{path} must contain a JSON object")
    if (
        value.get("file_device") != file_status.st_dev
        or value.get("file_inode") != file_status.st_ino
        or value.get("file_uid") != file_status.st_uid
        or value.get("file_mode") != stat.S_IMODE(file_status.st_mode)
    ):
        raise LeaseValidationError(f"{path} identity does not match")
    return value


def _require_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise LeaseValidationError(f"lease field {field} is invalid")
    return value


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise LeaseValidationError(f"lease field {field} is invalid")
    return value


def validate_active_lease(
    lease_path: Path,
    *,
    expected_script_path: Path,
    expected_executable_path: Path | None = None,
    expected_soft_bytes: int = DEFAULT_SOFT_BYTES,
    expected_emergency_bytes: int = DEFAULT_EMERGENCY_BYTES,
    expected_procfs_root: Path = Path("/proc"),
    expected_command: Sequence[str] | None = None,
    expected_heartbeat_path: Path | None = None,
    expected_audit_path: Path | None = None,
    expected_max_heartbeat_age_seconds: float | None = None,
    current_process_id: int | None = None,
    process_procfs_root: Path = Path("/proc"),
    monotonic_ns: Callable[[], int] | None = None,
    pidfd_open: Callable[[int], int] | None = getattr(
        os, "pidfd_open", None
    ),
) -> dict[str, object]:
    lease_path = lease_path.expanduser().resolve()
    lease = _read_json_object(lease_path)
    if (
        lease.get("format") != LEASE_FORMAT
        or lease.get("version") != LEASE_VERSION
        or lease.get("state") != "active"
    ):
        raise LeaseValidationError("lease format, version, or state is invalid")

    script_path = Path(
        _require_string(
            lease.get("watchdog_script_path"),
            "watchdog_script_path",
        )
    ).resolve()
    expected_script_path = expected_script_path.expanduser().resolve()
    if script_path != expected_script_path:
        raise LeaseValidationError("watchdog script path does not match")
    script_sha256 = _require_string(
        lease.get("watchdog_script_sha256"),
        "watchdog_script_sha256",
    )
    if script_sha256 != _sha256_file(expected_script_path):
        raise LeaseValidationError("watchdog script SHA does not match")

    if (
        _require_int(
            lease.get("soft_bytes"), "soft_bytes"
        )
        != expected_soft_bytes
        or _require_int(
            lease.get("emergency_bytes"),
            "emergency_bytes",
        )
        != expected_emergency_bytes
        or _require_int(
            lease.get("strict_ceiling_bytes"),
            "strict_ceiling_bytes",
        )
        != STRICT_CEILING_BYTES
    ):
        raise LeaseValidationError("watchdog thresholds do not match")
    lease_procfs_root = Path(
        _require_string(lease.get("procfs_root"), "procfs_root")
    ).resolve()
    if lease_procfs_root != expected_procfs_root.expanduser().resolve():
        raise LeaseValidationError("watchdog procfs root does not match")

    watchdog_pid = _require_int(
        lease.get("watchdog_pid"), "watchdog_pid"
    )
    pidfd: int | None = None
    if pidfd_open is not None:
        try:
            pidfd = pidfd_open(watchdog_pid)
        except OSError as exc:
            raise LeaseValidationError(
                "cannot open watchdog pidfd"
            ) from exc
    watchdog_start_ticks = _require_int(
        lease.get("watchdog_start_time_ticks"),
        "watchdog_start_time_ticks",
    )
    _, _, live_watchdog_start_ticks = _read_proc_stat(
        process_procfs_root, watchdog_pid
    )
    if live_watchdog_start_ticks != watchdog_start_ticks:
        raise LeaseValidationError("watchdog process start time does not match")
    expected_executable = (
        expected_executable_path or Path(sys.executable)
    ).expanduser().resolve()
    try:
        live_executable = (
            process_procfs_root / str(watchdog_pid) / "exe"
        ).resolve()
    except OSError as exc:
        raise LeaseValidationError(
            "cannot resolve watchdog executable"
        ) from exc
    if (
        live_executable != expected_executable
        or Path(
            _require_string(
                lease.get("watchdog_executable_path"),
                "watchdog_executable_path",
            )
        ).resolve()
        != expected_executable
    ):
        raise LeaseValidationError("watchdog executable does not match")
    live_cmdline = _read_proc_bytes(
        process_procfs_root, watchdog_pid, "cmdline"
    )
    if _sha256_bytes(live_cmdline) != _require_string(
        lease.get("watchdog_command_sha256"),
        "watchdog_command_sha256",
    ):
        raise LeaseValidationError("watchdog command line does not match")
    argv = [
        os.fsdecode(argument)
        for argument in live_cmdline.split(b"\0")
        if argument
    ]
    if len(argv) < 2 or argv[1] in ("-c", "-m"):
        raise LeaseValidationError(
            "watchdog script is not in executable argv position"
        )
    try:
        watchdog_cwd = (
            process_procfs_root / str(watchdog_pid) / "cwd"
        ).resolve()
    except OSError as exc:
        raise LeaseValidationError(
            "cannot resolve watchdog working directory"
        ) from exc
    argv_script = Path(argv[1]).expanduser()
    if not argv_script.is_absolute():
        argv_script = watchdog_cwd / argv_script
    if argv_script.resolve() != expected_script_path:
        raise LeaseValidationError(
            "watchdog script is not in executable argv position"
        )
    try:
        live_config = parse_args(argv[2:])
        live_paths = live_config.validate()
    except (SystemExit, ValueError) as exc:
        raise LeaseValidationError(
            "watchdog command line is invalid"
        ) from exc
    if (
        live_config.soft_bytes != expected_soft_bytes
        or live_config.emergency_bytes != expected_emergency_bytes
        or live_config.procfs_root.expanduser().resolve()
        != expected_procfs_root.expanduser().resolve()
    ):
        raise LeaseValidationError(
            "watchdog command-line policy does not match"
        )
    if (
        lease.get("grace_seconds") != live_config.grace_seconds
        or lease.get("sample_interval_seconds")
        != live_config.sample_interval_seconds
        or lease.get("max_heartbeat_age_seconds")
        != live_config.heartbeat_max_age_seconds
    ):
        raise LeaseValidationError(
            "watchdog lease timing policy does not match"
        )
    if (
        live_paths is None
        or live_paths.lease != lease_path
        or (
            expected_heartbeat_path is not None
            and live_paths.heartbeat
            != expected_heartbeat_path.expanduser().resolve()
        )
        or (
            expected_audit_path is not None
            and live_paths.audit
            != expected_audit_path.expanduser().resolve()
        )
    ):
        raise LeaseValidationError(
            "watchdog command-line artifact paths do not match"
        )
    if expected_command is not None and tuple(
        expected_command
    ) != live_config.command:
        raise LeaseValidationError("monitored command does not match")

    guardian_pid = _require_int(
        lease.get("guardian_pid"), "guardian_pid"
    )
    child_pid = _require_int(lease.get("child_pid"), "child_pid")
    process_group_id = _require_int(
        lease.get("child_process_group_id"),
        "child_process_group_id",
    )
    guardian_parent_pid, guardian_group_id, _ = _read_proc_stat(
        process_procfs_root, guardian_pid
    )
    child_parent_pid, child_group_id, _ = _read_proc_stat(
        process_procfs_root, child_pid
    )
    if (
        guardian_parent_pid != watchdog_pid
        or guardian_group_id != process_group_id
        or guardian_pid != process_group_id
        or child_parent_pid != guardian_pid
        or child_group_id != process_group_id
    ):
        raise LeaseValidationError(
            "watchdog, guardian, child, or process group does not match"
        )
    command = lease.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(argument, str) for argument in command)
    ):
        raise LeaseValidationError("lease field command is invalid")
    command_sha256 = _require_string(
        lease.get("child_command_sha256"),
        "child_command_sha256",
    )
    if command_sha256 != _command_sha256(command):
        raise LeaseValidationError("monitored command SHA is invalid")
    if expected_command is not None and command_sha256 != _command_sha256(
        expected_command
    ):
        raise LeaseValidationError("monitored command SHA does not match")

    process_id = (
        current_process_id
        if current_process_id is not None
        else os.getpid()
    )
    _, current_group_id, _ = _read_proc_stat(
        process_procfs_root, process_id
    )
    if current_group_id != process_group_id:
        raise LeaseValidationError(
            "current process is outside the monitored process group"
        )

    heartbeat_path = Path(
        _require_string(
            lease.get("heartbeat_path"), "heartbeat_path"
        )
    ).resolve()
    if (
        expected_heartbeat_path is not None
        and heartbeat_path
        != expected_heartbeat_path.expanduser().resolve()
    ):
        raise LeaseValidationError("heartbeat path does not match")
    heartbeat_max_age = lease.get("max_heartbeat_age_seconds")
    if (
        not isinstance(heartbeat_max_age, (int, float))
        or isinstance(heartbeat_max_age, bool)
        or not math.isfinite(heartbeat_max_age)
        or heartbeat_max_age <= 0
    ):
        raise LeaseValidationError(
            "lease field max_heartbeat_age_seconds is invalid"
        )
    if (
        expected_max_heartbeat_age_seconds is not None
        and heartbeat_max_age != expected_max_heartbeat_age_seconds
    ):
        raise LeaseValidationError("heartbeat max age does not match")
    heartbeat = _read_json_object(heartbeat_path)
    lease_id = _require_string(lease.get("lease_id"), "lease_id")
    if (
        heartbeat.get("format") != HEARTBEAT_FORMAT
        or heartbeat.get("version") != HEARTBEAT_VERSION
        or heartbeat.get("state") != "active"
        or heartbeat.get("lease_id") != lease_id
        or heartbeat.get("watchdog_pid") != watchdog_pid
        or heartbeat.get("watchdog_start_time_ticks")
        != watchdog_start_ticks
        or heartbeat.get("child_pid") != child_pid
        or heartbeat.get("child_process_group_id") != process_group_id
    ):
        raise LeaseValidationError("heartbeat identity does not match lease")
    updated_monotonic_ns = _require_int(
        heartbeat.get("updated_monotonic_ns"),
        "heartbeat.updated_monotonic_ns",
    )
    _require_int(heartbeat.get("sequence"), "heartbeat.sequence")
    _require_string(heartbeat.get("updated_at"), "heartbeat.updated_at")
    heartbeat_sample = heartbeat.get("sample")
    if not isinstance(heartbeat_sample, dict):
        raise LeaseValidationError("heartbeat sample is invalid")
    audit_record_sha256 = _require_string(
        heartbeat_sample.get("audit_record_sha256"),
        "heartbeat.sample.audit_record_sha256",
    )
    now_monotonic_ns = (monotonic_ns or time.monotonic_ns)()
    age_ns = now_monotonic_ns - updated_monotonic_ns
    if age_ns < 0 or age_ns > int(heartbeat_max_age * 1_000_000_000):
        raise LeaseValidationError("watchdog heartbeat is stale")

    audit_path = Path(
        _require_string(lease.get("audit_path"), "audit_path")
    ).resolve()
    if (
        expected_audit_path is not None
        and audit_path != expected_audit_path.expanduser().resolve()
    ):
        raise LeaseValidationError("persistent audit path does not match")
    audit_fd = _require_int(lease.get("audit_fd"), "audit_fd")
    audit_device = _require_int(
        lease.get("audit_device"), "audit_device"
    )
    audit_inode = _require_int(
        lease.get("audit_inode"), "audit_inode"
    )
    audit_uid = _require_int(lease.get("audit_uid"), "audit_uid")
    audit_mode = _require_int(lease.get("audit_mode"), "audit_mode")
    try:
        audit_status = audit_path.stat(follow_symlinks=False)
        live_audit_status = (
            process_procfs_root
            / str(watchdog_pid)
            / "fd"
            / str(audit_fd)
        ).stat()
        if (
            not stat.S_ISREG(audit_status.st_mode)
            or audit_status.st_dev != audit_device
            or audit_status.st_ino != audit_inode
            or live_audit_status.st_dev != audit_device
            or live_audit_status.st_ino != audit_inode
            or audit_status.st_uid != audit_uid
            or audit_uid != os.getuid()
            or stat.S_IMODE(audit_status.st_mode) != audit_mode
            or audit_mode != 0o600
        ):
            raise LeaseValidationError(
                "persistent audit identity does not match"
            )
        audit_descriptor = os.open(
            audit_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            try:
                fcntl.flock(
                    audit_descriptor,
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError:
                pass
            else:
                fcntl.flock(audit_descriptor, fcntl.LOCK_UN)
                raise LeaseValidationError(
                    "watchdog does not hold the persistent audit lock"
                )
        finally:
            os.close(audit_descriptor)
        audit_lines = [
            line
            for line in audit_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line
        ]
        first_line = next(iter(audit_lines))
        first_record = json.loads(first_line)
        if (
            not isinstance(first_record, dict)
            or not isinstance(first_record.get("event"), str)
            or not isinstance(first_record.get("timestamp"), str)
        ):
            raise LeaseValidationError(
                "persistent audit does not contain watchdog records"
            )
        if not any(
            _sha256_bytes((line + "\n").encode("utf-8"))
            == audit_record_sha256
            for line in audit_lines
        ):
            raise LeaseValidationError(
                "heartbeat audit record does not match persistent audit"
            )
    except StopIteration as exc:
        raise LeaseValidationError("persistent audit is empty") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LeaseValidationError(
            "persistent audit does not contain valid JSONL"
        ) from exc
    except LeaseValidationError:
        raise
    except OSError as exc:
        raise LeaseValidationError(
            f"cannot inspect persistent audit {audit_path}: {exc}"
        ) from exc
    _, _, final_watchdog_start_ticks = _read_proc_stat(
        process_procfs_root, watchdog_pid
    )
    if final_watchdog_start_ticks != watchdog_start_ticks:
        raise LeaseValidationError(
            "watchdog process changed during validation"
        )
    if pidfd is not None:
        os.close(pidfd)
    return lease


def start_process_group_lease_guard(
    expected_script_path: Path,
    *,
    startup_timeout_seconds: float = 5.0,
    expected_procfs_root: Path = Path("/proc"),
    process_procfs_root: Path = Path("/proc"),
) -> threading.Thread:
    try:
        lease_path = Path(
            os.environ["STRIX_MEMORY_WATCHDOG_LEASE_PATH"]
        ).resolve()
        heartbeat_path = Path(
            os.environ["STRIX_MEMORY_WATCHDOG_HEARTBEAT_PATH"]
        ).resolve()
        audit_path = Path(
            os.environ["STRIX_MEMORY_WATCHDOG_AUDIT_PATH"]
        ).resolve()
        max_age_seconds = float(
            os.environ[
                "STRIX_MEMORY_WATCHDOG_HEARTBEAT_MAX_AGE_SECONDS"
            ]
        )
    except (KeyError, ValueError) as exc:
        raise LeaseValidationError(
            "watchdog artifact environment is missing or invalid"
        ) from exc
    current_cmdline = _read_proc_bytes(
        process_procfs_root, os.getpid(), "cmdline"
    )
    expected_command = tuple(
        os.fsdecode(argument)
        for argument in current_cmdline.split(b"\0")
        if argument
    )
    deadline = time.monotonic() + startup_timeout_seconds
    while True:
        try:
            lease = validate_active_lease(
                lease_path,
                expected_script_path=expected_script_path,
                expected_procfs_root=expected_procfs_root,
                expected_command=expected_command,
                expected_heartbeat_path=heartbeat_path,
                expected_audit_path=audit_path,
                expected_max_heartbeat_age_seconds=max_age_seconds,
                process_procfs_root=process_procfs_root,
            )
            break
        except Exception:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)

    guardian_pid = _require_int(
        lease.get("guardian_pid"), "guardian_pid"
    )
    signal.signal(LEASE_GUARD_SIGNAL, _kill_own_process_group)
    _set_parent_death_signal(LEASE_GUARD_SIGNAL, guardian_pid)

    def monitor() -> None:
        interval = min(1.0, max_age_seconds / 3)
        while True:
            time.sleep(interval)
            try:
                validate_active_lease(
                    lease_path,
                    expected_script_path=expected_script_path,
                    expected_procfs_root=expected_procfs_root,
                    expected_command=expected_command,
                    expected_heartbeat_path=heartbeat_path,
                    expected_audit_path=audit_path,
                    expected_max_heartbeat_age_seconds=max_age_seconds,
                    process_procfs_root=process_procfs_root,
                )
            except Exception:
                _kill_own_process_group()

    guard = threading.Thread(
        target=monitor,
        name="strix-watchdog-lease-guard",
        daemon=True,
    )
    guard.start()
    return guard


class AuditLogger:
    def __init__(
        self,
        stream: IO[str],
        wall_clock: Callable[[], datetime] | None = None,
    ):
        self.stream = stream
        self.stream_enabled = True
        self.wall_clock = wall_clock
        self.persistent_stream: IO[str] | None = None
        self.lease_manager: LeaseManager | None = None
        self.finalized = False
        self.final_exit_code = EXIT_INTERNAL_ERROR
        self.last_record_sha256: str | None = None

    def open_persistent(self, path: Path) -> None:
        resolved_path = path.expanduser().resolve()
        try:
            descriptor = os.open(
                resolved_path,
                os.O_CREAT
                | os.O_EXCL
                | os.O_WRONLY
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            fcntl.flock(
                descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB
            )
            self.persistent_stream = os.fdopen(
                descriptor, "w", encoding="utf-8"
            )
        except OSError as exc:
            detail = exc.strerror or str(exc)
            raise ArtifactError(
                "audit",
                f"cannot create persistent audit {resolved_path}: {detail}",
            ) from exc

    def persistent_identity(self) -> dict[str, int]:
        if self.persistent_stream is None:
            raise ArtifactError(
                "audit", "persistent audit is not open"
            )
        file_status = os.fstat(self.persistent_stream.fileno())
        return {
            "device": file_status.st_dev,
            "inode": file_status.st_ino,
            "uid": file_status.st_uid,
            "mode": stat.S_IMODE(file_status.st_mode),
            "fd": self.persistent_stream.fileno(),
        }

    def close(self) -> None:
        persistent_stream = self.persistent_stream
        self.persistent_stream = None
        if persistent_stream is not None:
            try:
                persistent_stream.close()
            except (OSError, ValueError):
                pass

    def disable_component(self, component: str) -> None:
        if component == "audit":
            self.close()
        elif component == "lease":
            self.lease_manager = None
        elif component == "stderr":
            self.stream_enabled = False

    def emit(self, event: str, **fields: object) -> dict[str, object]:
        record = {
            "timestamp": _timestamp_utc(self.wall_clock),
            "event": event,
            **fields,
        }
        line = (
            json.dumps(record, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        if self.stream_enabled:
            try:
                self.stream.write(line)
                self.stream.flush()
            except (OSError, ValueError) as exc:
                detail = getattr(exc, "strerror", None) or str(exc)
                raise ArtifactError(
                    "stderr",
                    f"cannot write standard error audit: {detail}",
                ) from exc
        self.last_record_sha256 = _sha256_bytes(line.encode("utf-8"))
        if self.persistent_stream is not None:
            try:
                self.persistent_stream.write(line)
                self.persistent_stream.flush()
                os.fsync(self.persistent_stream.fileno())
            except OSError as exc:
                detail = exc.strerror or str(exc)
                raise ArtifactError(
                    "audit",
                    f"cannot write persistent audit: {detail}",
                ) from exc
        return record

    def heartbeat(self, sample: dict[str, object]) -> None:
        if self.lease_manager is not None:
            self.lease_manager.update_heartbeat(
                {
                    **sample,
                    "audit_record_sha256": self.last_record_sha256,
                }
            )

    def finalize(self, record: dict[str, object]) -> None:
        if self.lease_manager is not None:
            self.lease_manager.finalize(record)

    def mark_final(self, exit_code: int) -> None:
        self.finalized = True
        self.final_exit_code = exit_code


def _child_status(returncode: int | None, started: bool = True) -> str:
    if not started:
        return "not_started"
    if returncode is None:
        return "running"
    return "signaled" if returncode < 0 else "exited"


def _state_fields(
    snapshot: HostSnapshot | None,
    peak_used_bytes: int | None,
    child: ProcessHandle | None,
    child_returncode: int | None,
    process_group_status: str,
    threshold_reason: str,
) -> dict[str, object]:
    return {
        "total_bytes": snapshot.total_bytes if snapshot else None,
        "available_bytes": snapshot.available_bytes if snapshot else None,
        "used_bytes": snapshot.used_bytes if snapshot else None,
        "swap_entries": len(snapshot.active_swaps) if snapshot else None,
        "peak_used_bytes": peak_used_bytes,
        "child_pid": child.pid if child else None,
        "child_status": _child_status(
            child_returncode, started=child is not None
        ),
        "child_returncode": child_returncode,
        "process_group_id": child.pid if child else None,
        "process_group_status": process_group_status,
        "threshold_reason": threshold_reason,
    }


def _emit_final(
    audit: AuditLogger,
    classification: str,
    exit_code: int,
    reason: str,
    snapshot: HostSnapshot | None,
    peak_used_bytes: int | None,
    child: ProcessHandle | None = None,
    child_returncode: int | None = None,
    process_group_status: str = "not_created",
    error: str | None = None,
    preserve_primary_on_artifact_error: bool = False,
    secondary_errors: Sequence[dict[str, str]] | None = None,
) -> int:
    fields = _state_fields(
        snapshot,
        peak_used_bytes,
        child,
        child_returncode,
        process_group_status,
        reason,
    )
    fields.update(classification=classification, exit_code=exit_code)
    if error:
        fields["error"] = error
    if secondary_errors:
        fields["secondary_errors"] = list(secondary_errors)

    def record_artifact_error(exc: ArtifactError) -> None:
        nonlocal exit_code
        detail = {
            "component": exc.component,
            "detail": str(exc),
        }
        if preserve_primary_on_artifact_error:
            secondary_errors = fields.setdefault(
                "secondary_errors", []
            )
            assert isinstance(secondary_errors, list)
            secondary_errors.append(detail)
        else:
            fields.update(
                classification="lease_error",
                exit_code=EXIT_LEASE_ERROR,
                threshold_reason="watchdog artifact finalization failed",
                error=f"{exc.component}: {exc}",
            )
            exit_code = EXIT_LEASE_ERROR

    def emit_final_record() -> dict[str, object]:
        try:
            return audit.emit("final", **fields)
        except ArtifactError as exc:
            audit.disable_component(exc.component)
            record_artifact_error(exc)
        try:
            return audit.emit("final", **fields)
        except ArtifactError as exc:
            audit.disable_component(exc.component)
            record_artifact_error(exc)
        return audit.emit("final", **fields)

    previous_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK, PARENT_SIGNALS
    )
    try:
        record = emit_final_record()
        try:
            audit.finalize(record)
        except ArtifactError as exc:
            audit.disable_component(exc.component)
            record_artifact_error(exc)
            emit_final_record()
        audit.mark_final(exit_code)
        return exit_code
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _signal_process_group(process_group_id: int, signal_number: int) -> str:
    try:
        os.killpg(process_group_id, signal_number)
    except ProcessLookupError:
        return "missing"
    except OSError as exc:
        name = signal.Signals(signal_number).name
        detail = exc.strerror or str(exc)
        raise ProcessGroupError(
            f"cannot send {name} to process group {process_group_id}: {detail}"
        ) from exc
    return f"{signal.Signals(signal_number).name.lower()}_sent"


def _process_group_alive(process_group_id: int) -> bool:
    if sys.platform.startswith("linux"):
        try:
            process_paths = Path("/proc").iterdir()
            for process_path in process_paths:
                if not process_path.name.isdigit():
                    continue
                try:
                    content = (
                        process_path / "stat"
                    ).read_text(encoding="utf-8")
                    close_paren = content.rfind(")")
                    fields = content[close_paren + 1:].split()
                    if (
                        close_paren >= 0
                        and len(fields) >= 3
                        and fields[0] != "Z"
                        and int(fields[2]) == process_group_id
                    ):
                        return True
                except (OSError, UnicodeError, ValueError):
                    continue
            return False
        except OSError:
            pass
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise ProcessGroupError(
            f"cannot inspect process group {process_group_id}: {detail}"
        ) from exc
    return True


def _raise_parent_signal(signal_number: int, _frame: object) -> None:
    raise ParentSignal(signal_number)


def _set_parent_signal_handlers(
    handler: Any,
) -> dict[int, Any]:
    previous: dict[int, Any] = {}
    for signal_number in PARENT_SIGNALS:
        previous[signal_number] = signal.signal(signal_number, handler)
    return previous


def _restore_parent_signal_handlers(
    previous: dict[int, Any],
) -> None:
    for signal_number, handler in previous.items():
        signal.signal(signal_number, handler)


def _kill_and_finish(
    audit: AuditLogger,
    child: ProcessHandle,
    snapshot: HostSnapshot,
    peak_used_bytes: int,
    classification: str,
    exit_code: int,
    reason: str,
    signal_group: Callable[[int, int], str],
) -> int:
    try:
        group_status = signal_group(child.pid, signal.SIGKILL)
    except ProcessGroupError as exc:
        return _emit_final(
            audit,
            "signal_error",
            EXIT_SIGNAL_ERROR,
            reason,
            snapshot,
            peak_used_bytes,
            child,
            child.poll(),
            "signal_error",
            str(exc),
        )

    artifact_error: ArtifactError | None = None
    try:
        audit.emit(
            "process_group_signal",
            **_state_fields(
                snapshot,
                peak_used_bytes,
                child,
                child.poll(),
                group_status,
                reason,
            ),
            signal="SIGKILL",
        )
    except ArtifactError as exc:
        artifact_error = exc
        audit.disable_component(exc.component)
    try:
        child_returncode = child.wait(timeout=5.0)
    except subprocess.TimeoutExpired as exc:
        return _emit_final(
            audit,
            "termination_timeout",
            EXIT_SIGNAL_ERROR,
            reason,
            snapshot,
            peak_used_bytes,
            child,
            child.poll(),
            "sigkill_timeout",
            str(exc),
        )
    if artifact_error is not None:
        classification = "lease_error"
        exit_code = EXIT_LEASE_ERROR
        error = f"{artifact_error.component}: {artifact_error}"
    else:
        error = None
    return _emit_final(
        audit,
        classification,
        exit_code,
        reason,
        snapshot,
        peak_used_bytes,
        child,
        child_returncode,
        group_status,
        error,
    )


def _graceful_cleanup(
    audit: AuditLogger,
    child: ProcessHandle,
    snapshot: HostSnapshot,
    peak_used_bytes: int,
    classification: str,
    exit_code: int,
    reason: str,
    graceful_signal: int | None,
    grace_seconds: float,
    signal_group: Callable[[int, int], str],
    group_alive: Callable[[int], bool],
    monotonic: Callable[[], float],
    sleeper: Callable[[float], None],
    process_group_status: str = "active",
    escalation_result: tuple[str, int, str] | None = None,
    error: str | None = None,
) -> int:
    escalated = False
    artifact_error: ArtifactError | None = None
    guardian_control_error: ProcessGroupError | None = None
    try:
        if graceful_signal is not None:
            process_group_status = signal_group(
                child.pid, graceful_signal
            )
            try:
                audit.emit(
                    "process_group_signal",
                    **_state_fields(
                        snapshot,
                        peak_used_bytes,
                        child,
                        child.poll(),
                        process_group_status,
                        reason,
                    ),
                    signal=signal.Signals(graceful_signal).name,
                )
            except ArtifactError as exc:
                artifact_error = exc
                audit.disable_component(exc.component)
            if (
                isinstance(child, GuardianProcess)
                and child.poll() is None
            ):
                try:
                    child.begin_grace()
                except ProcessGroupError as exc:
                    guardian_control_error = exc
        deadline = monotonic() + grace_seconds
        while (
            guardian_control_error is None
            and monotonic() < deadline
        ):
            child.poll()
            if not group_alive(child.pid):
                break
            if (
                isinstance(child, GuardianProcess)
                and child.poll() is None
            ):
                try:
                    child.pulse()
                except ProcessGroupError as exc:
                    guardian_control_error = exc
                    break
            sleeper(min(0.05, deadline - monotonic()))
        child.poll()
        if (
            guardian_control_error is not None
            or group_alive(child.pid)
        ):
            escalated = True
            process_group_status = signal_group(
                child.pid, signal.SIGKILL
            )
            if guardian_control_error is not None:
                signal_reason = (
                    "guardian control failed during graceful cleanup"
                )
            else:
                signal_reason = (
                    escalation_result[2]
                    if escalation_result is not None
                    else reason
                )
            try:
                audit.emit(
                    "process_group_signal",
                    **_state_fields(
                        snapshot,
                        peak_used_bytes,
                        child,
                        child.poll(),
                        process_group_status,
                        signal_reason,
                    ),
                    signal="SIGKILL",
                )
            except ArtifactError as exc:
                if artifact_error is None:
                    artifact_error = exc
                audit.disable_component(exc.component)
    except ProcessGroupError as exc:
        return _emit_final(
            audit,
            "signal_error",
            EXIT_SIGNAL_ERROR,
            reason,
            snapshot,
            peak_used_bytes,
            child,
            child.poll(),
            "signal_error",
            str(exc),
        )

    child_returncode = child.poll()
    if child_returncode is None:
        try:
            child_returncode = child.wait(timeout=5.0)
        except subprocess.TimeoutExpired as exc:
            return _emit_final(
                audit,
                "termination_timeout",
                EXIT_SIGNAL_ERROR,
                reason,
                snapshot,
                peak_used_bytes,
                child,
                child.poll(),
                "termination_timeout",
                str(exc),
            )

    if guardian_control_error is not None:
        classification = "signal_error"
        exit_code = EXIT_SIGNAL_ERROR
        reason = "guardian control failed during graceful cleanup"
        error = str(guardian_control_error)
    elif escalated and escalation_result is not None:
        classification, exit_code, reason = escalation_result
    if artifact_error is not None and guardian_control_error is None:
        classification = "lease_error"
        exit_code = EXIT_LEASE_ERROR
        error = f"{artifact_error.component}: {artifact_error}"
    return _emit_final(
        audit,
        classification,
        exit_code,
        reason,
        snapshot,
        peak_used_bytes,
        child,
        child_returncode,
        process_group_status,
        error,
        preserve_primary_on_artifact_error=(
            guardian_control_error is not None
        ),
        secondary_errors=(
            [
                {
                    "component": artifact_error.component,
                    "detail": str(artifact_error),
                }
            ]
            if (
                guardian_control_error is not None
                and artifact_error is not None
            )
            else None
        ),
    )


def _monitor_child(
    config: WatchdogConfig,
    reader: ProcfsReader,
    audit: AuditLogger,
    child: ProcessHandle,
    state: RuntimeState,
    signal_group: Callable[[int, int], str],
    group_alive: Callable[[int], bool],
    pulse_guardian: Callable[[], None],
    monotonic: Callable[[], float],
    sleeper: Callable[[float], None],
) -> int:
    soft_deadline: float | None = None

    while True:
        child_returncode = child.poll()
        if child_returncode is not None:
            soft_stop = soft_deadline is not None
            classification = "soft_limit" if soft_stop else "child_exit"
            exit_code = EXIT_SOFT_LIMIT if soft_stop else (
                128 - child_returncode
                if child_returncode < 0
                else child_returncode
            )
            reason = (
                "child exited during soft-threshold grace period"
                if soft_stop
                else "child exited"
            )
            if group_alive(child.pid):
                grace_seconds = config.grace_seconds
                graceful_signal: int | None = signal.SIGTERM
                group_status = "active"
                if soft_stop:
                    grace_seconds = max(
                        0.0, soft_deadline - monotonic()
                    )
                    graceful_signal = None
                    group_status = "sigterm_sent"
                return _graceful_cleanup(
                    audit,
                    child,
                    state.snapshot,
                    state.peak_used_bytes,
                    classification,
                    exit_code,
                    (
                        f"{reason}; process group members still running"
                    ),
                    graceful_signal,
                    grace_seconds,
                    signal_group,
                    group_alive,
                    monotonic,
                    sleeper,
                    group_status,
                    (
                        (
                            "grace_timeout",
                            EXIT_GRACE_TIMEOUT,
                            "soft-threshold grace period expired with "
                            "process group members still running",
                        )
                        if soft_stop
                        else None
                    ),
                )
            return _emit_final(
                audit,
                classification,
                exit_code,
                reason,
                state.snapshot,
                state.peak_used_bytes,
                child,
                child_returncode,
                "leader_exited",
            )

        now = monotonic()
        if soft_deadline is not None and now >= soft_deadline:
            return _kill_and_finish(
                audit,
                child,
                state.snapshot,
                state.peak_used_bytes,
                "grace_timeout",
                EXIT_GRACE_TIMEOUT,
                "soft-threshold grace period expired",
                signal_group,
            )

        try:
            state.snapshot = reader.read_snapshot()
        except ProcfsError as exc:
            return _kill_and_finish(
                audit,
                child,
                state.snapshot,
                state.peak_used_bytes,
                "procfs_error",
                EXIT_PROCFS_ERROR,
                str(exc),
                signal_group,
            )

        state.peak_used_bytes = max(
            state.peak_used_bytes, state.snapshot.used_bytes
        )
        if state.snapshot.active_swaps:
            return _kill_and_finish(
                audit,
                child,
                state.snapshot,
                state.peak_used_bytes,
                "swap_appeared",
                EXIT_SWAP_ACTIVE,
                "active swap appeared during execution",
                signal_group,
            )
        if state.snapshot.used_bytes >= config.emergency_bytes:
            return _kill_and_finish(
                audit,
                child,
                state.snapshot,
                state.peak_used_bytes,
                "emergency_limit",
                EXIT_EMERGENCY_LIMIT,
                "used_bytes >= emergency_bytes",
                signal_group,
            )
        soft_signal_fields: dict[str, object] | None = None
        if (
            soft_deadline is None
            and state.snapshot.used_bytes >= config.soft_bytes
        ):
            try:
                group_status = signal_group(child.pid, signal.SIGTERM)
            except ProcessGroupError as exc:
                return _emit_final(
                    audit,
                    "signal_error",
                    EXIT_SIGNAL_ERROR,
                    "used_bytes >= soft_bytes",
                    state.snapshot,
                    state.peak_used_bytes,
                    child,
                    child.poll(),
                    "signal_error",
                    str(exc),
                )
            soft_deadline = now + config.grace_seconds
            if isinstance(child, GuardianProcess):
                try:
                    child.begin_grace()
                except ProcessGroupError as exc:
                    return _kill_and_finish(
                        audit,
                        child,
                        state.snapshot,
                        state.peak_used_bytes,
                        "signal_error",
                        EXIT_SIGNAL_ERROR,
                        str(exc),
                        signal_group,
                    )
            soft_signal_fields = {
                **_state_fields(
                    state.snapshot,
                    state.peak_used_bytes,
                    child,
                    child.poll(),
                    group_status,
                    "used_bytes >= soft_bytes",
                ),
                "signal": "SIGTERM",
                "grace_deadline_monotonic": soft_deadline,
            }
        try:
            pulse_guardian()
        except ProcessGroupError as exc:
            if child.poll() is not None:
                continue
            return _kill_and_finish(
                audit,
                child,
                state.snapshot,
                state.peak_used_bytes,
                "signal_error",
                EXIT_SIGNAL_ERROR,
                str(exc),
                signal_group,
            )
        if soft_signal_fields is not None:
            audit.emit(
                "process_group_signal",
                **soft_signal_fields,
            )
        sample_record = audit.emit(
            "sample",
            **_state_fields(
                state.snapshot,
                state.peak_used_bytes,
                child,
                None,
                "active",
                "none",
            )
        )
        audit.heartbeat(sample_record)
        try:
            pulse_guardian()
        except ProcessGroupError as exc:
            if child.poll() is not None:
                continue
            return _kill_and_finish(
                audit,
                child,
                state.snapshot,
                state.peak_used_bytes,
                "signal_error",
                EXIT_SIGNAL_ERROR,
                str(exc),
                signal_group,
            )

        sleep_seconds = config.sample_interval_seconds
        if soft_deadline is not None:
            sleep_seconds = min(
                sleep_seconds,
                max(0.0, soft_deadline - monotonic()),
            )
        sleeper(sleep_seconds)


def run_watchdog(
    config: WatchdogConfig,
    *,
    reader: ProcfsReader | None = None,
    audit: AuditLogger | None = None,
    launcher: Callable[..., ProcessHandle] | None = None,
    signal_group: Callable[[int, int], str] | None = None,
    group_alive: Callable[[int], bool] | None = None,
    monotonic: Callable[[], float] | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> int:
    artifact_paths = config.validate()
    use_guardian = (
        launcher is None and sys.platform.startswith("linux")
    )
    reader = reader or ProcfsReader(config.procfs_root)
    audit = audit or AuditLogger(sys.stderr)
    launcher = launcher or subprocess.Popen
    signal_group = signal_group or _signal_process_group
    group_alive = group_alive or _process_group_alive
    monotonic = monotonic or time.monotonic
    sleeper = sleeper or time.sleep

    if artifact_paths is not None:
        try:
            audit.open_persistent(artifact_paths.audit)
        except ArtifactError as exc:
            return _emit_final(
                audit,
                "lease_error",
                EXIT_LEASE_ERROR,
                "cannot initialize watchdog artifacts",
                None,
                None,
                error=f"{exc.component}: {exc}",
            )

    try:
        snapshot = reader.read_snapshot()
    except ProcfsError as exc:
        return _emit_final(
            audit,
            "procfs_error",
            EXIT_PROCFS_ERROR,
            str(exc),
            None,
            None,
            error=str(exc),
        )

    try:
        audit.emit(
            "preflight",
            **_state_fields(
                snapshot,
                snapshot.used_bytes,
                None,
                None,
                "not_created",
                "none",
            ),
            soft_bytes=config.soft_bytes,
            emergency_bytes=config.emergency_bytes,
            strict_ceiling_bytes=STRICT_CEILING_BYTES,
        )
    except ArtifactError as exc:
        audit.disable_component(exc.component)
        return _emit_final(
            audit,
            "lease_error",
            EXIT_LEASE_ERROR,
            "cannot write watchdog preflight audit",
            snapshot,
            snapshot.used_bytes,
            error=f"{exc.component}: {exc}",
        )

    if snapshot.active_swaps:
        return _emit_final(
            audit,
            "startup_swap_active",
            EXIT_SWAP_ACTIVE,
            "active swap present before command launch",
            snapshot,
            snapshot.used_bytes,
        )
    if snapshot.used_bytes >= config.emergency_bytes:
        return _emit_final(
            audit,
            "startup_emergency_limit",
            EXIT_EMERGENCY_LIMIT,
            "used_bytes >= emergency_bytes before launch",
            snapshot,
            snapshot.used_bytes,
        )
    if snapshot.used_bytes >= config.soft_bytes:
        return _emit_final(
            audit,
            "startup_soft_limit",
            EXIT_SOFT_LIMIT,
            "used_bytes >= soft_bytes before launch",
            snapshot,
            snapshot.used_bytes,
        )

    previous_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK, PARENT_SIGNALS
    )
    mask_restored = False
    previous_handlers: dict[int, Any] = {}
    child: ProcessHandle | None = None
    state = RuntimeState(snapshot, snapshot.used_bytes)
    try:
        launch_mask = previous_mask
        lease_manager = (
            LeaseManager(config, artifact_paths)
            if artifact_paths is not None
            else None
        )

        def restore_child_signal_mask() -> None:
            signal.pthread_sigmask(signal.SIG_SETMASK, launch_mask)

        try:
            child_environment = os.environ.copy()
            if lease_manager is not None:
                child_environment.update(
                    {
                        "STRIX_MEMORY_WATCHDOG_LEASE_PATH": str(
                            lease_manager.lease_path
                        ),
                        "STRIX_MEMORY_WATCHDOG_HEARTBEAT_PATH": str(
                            lease_manager.heartbeat_path
                        ),
                        "STRIX_MEMORY_WATCHDOG_AUDIT_PATH": str(
                            lease_manager.audit_path
                        ),
                        "STRIX_MEMORY_WATCHDOG_HEARTBEAT_MAX_AGE_SECONDS": (
                            str(config.heartbeat_max_age_seconds)
                        ),
                    }
                )
            if use_guardian:
                child = _launch_guardian(
                    config.command,
                    child_environment,
                    config.heartbeat_max_age_seconds,
                    config.grace_seconds + 1.0,
                    launch_mask,
                )
            elif lease_manager is not None:
                child = launcher(
                    config.command,
                    start_new_session=True,
                    preexec_fn=restore_child_signal_mask,
                    env=child_environment,
                )
            else:
                child = launcher(
                    config.command,
                    start_new_session=True,
                    preexec_fn=restore_child_signal_mask,
                )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            detail = getattr(exc, "strerror", None) or str(exc)
            return _emit_final(
                audit,
                "launch_error",
                EXIT_LAUNCH_ERROR,
                "command launch failed",
                snapshot,
                snapshot.used_bytes,
                error=detail,
            )

        previous_handlers = _set_parent_signal_handlers(
            _raise_parent_signal
        )
        if lease_manager is not None:
            lease_manager.start(child, audit)
            audit.lease_manager = lease_manager
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        mask_restored = True
        audit.emit(
            "child_started",
            **_state_fields(
                state.snapshot,
                state.peak_used_bytes,
                child,
                None,
                "active",
                "none",
            ),
            command=list(config.command),
        )
        return _monitor_child(
            config,
            reader,
            audit,
            child,
            state,
            signal_group,
            group_alive,
            child.pulse if isinstance(child, GuardianProcess) else lambda: None,
            monotonic,
            sleeper,
        )
    except ArtifactError as exc:
        _set_parent_signal_handlers(signal.SIG_IGN)
        audit.disable_component(exc.component)
        if child is None:
            return _emit_final(
                audit,
                "lease_error",
                EXIT_LEASE_ERROR,
                "watchdog artifact initialization failed",
                state.snapshot,
                state.peak_used_bytes,
                error=f"{exc.component}: {exc}",
            )
        return _graceful_cleanup(
            audit,
            child,
            state.snapshot,
            state.peak_used_bytes,
            "lease_error",
            EXIT_LEASE_ERROR,
            "watchdog artifact update failed",
            signal.SIGTERM,
            config.grace_seconds,
            signal_group,
            group_alive,
            monotonic,
            sleeper,
            error=f"{exc.component}: {exc}",
        )
    except ParentSignal as exc:
        if audit.finalized:
            return audit.final_exit_code
        _set_parent_signal_handlers(signal.SIG_IGN)
        assert child is not None
        signal_name = signal.Signals(exc.signal_number).name
        return _graceful_cleanup(
            audit,
            child,
            state.snapshot,
            state.peak_used_bytes,
            "parent_signal",
            128 + exc.signal_number,
            f"wrapper received {signal_name}",
            exc.signal_number,
            config.grace_seconds,
            signal_group,
            group_alive,
            monotonic,
            sleeper,
        )
    except Exception as exc:
        if audit.finalized:
            return audit.final_exit_code
        _set_parent_signal_handlers(signal.SIG_IGN)
        if child is None:
            return _emit_final(
                audit,
                "internal_error",
                EXIT_INTERNAL_ERROR,
                "unexpected pre-launch exception",
                state.snapshot,
                state.peak_used_bytes,
                error=f"{type(exc).__name__}: {exc}",
            )
        return _graceful_cleanup(
            audit,
            child,
            state.snapshot,
            state.peak_used_bytes,
            "internal_error",
            EXIT_INTERNAL_ERROR,
            "unexpected post-launch exception",
            signal.SIGTERM,
            config.grace_seconds,
            signal_group,
            group_alive,
            monotonic,
            sleeper,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if not mask_restored:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        if previous_handlers:
            _restore_parent_signal_handlers(previous_handlers)
        if isinstance(child, GuardianProcess):
            child.close()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parse_args(argv: Sequence[str]) -> WatchdogConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Launch a command in a new process group and stop it before "
            "host-wide memory use reaches the 120 GiB Strix validation ceiling."
        )
    )
    parser.add_argument(
        "--procfs-root",
        type=Path,
        default=Path("/proc"),
        help="procfs root containing meminfo and swaps (default: /proc)",
    )
    parser.add_argument(
        "--soft-gib",
        type=_positive_int,
        default=116,
        help="send SIGTERM at this many GiB used (default: 116)",
    )
    parser.add_argument(
        "--emergency-gib",
        type=_positive_int,
        default=118,
        help=(
            "send SIGKILL at this many GiB used (default: 118, leaving "
            "a 2 GiB sampling margin below 120 GiB)"
        ),
    )
    parser.add_argument(
        "--grace-seconds",
        type=_positive_float,
        default=DEFAULT_GRACE_SECONDS,
        help="maximum time after SIGTERM before SIGKILL (default: 30)",
    )
    parser.add_argument(
        "--sample-interval-seconds",
        type=_positive_float,
        default=DEFAULT_SAMPLE_INTERVAL_SECONDS,
        help="procfs sampling interval (default: 1)",
    )
    parser.add_argument(
        "--lease-path",
        type=Path,
        help=(
            "atomically publish the watchdog-owned lease JSON; requires "
            "--heartbeat-path and --audit-path"
        ),
    )
    parser.add_argument(
        "--heartbeat-path",
        type=Path,
        help=(
            "atomically update watchdog heartbeat JSON on every sample; "
            "requires --lease-path and --audit-path"
        ),
    )
    parser.add_argument(
        "--audit-path",
        type=Path,
        help=(
            "create a persistent JSONL audit in addition to standard error; "
            "requires --lease-path and --heartbeat-path"
        ),
    )
    parser.add_argument(
        "--heartbeat-max-age-seconds",
        type=_positive_float,
        default=DEFAULT_HEARTBEAT_MAX_AGE_SECONDS,
        help=(
            "maximum heartbeat age accepted by a matching harness "
            "(default: 5)"
        ),
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command and arguments, preceded by --",
    )
    args = parser.parse_args(argv)
    command = tuple(args.command)
    if command and command[0] == "--":
        command = command[1:]
    return WatchdogConfig(
        command=command,
        procfs_root=args.procfs_root,
        soft_bytes=args.soft_gib * GIB,
        emergency_bytes=args.emergency_gib * GIB,
        grace_seconds=args.grace_seconds,
        sample_interval_seconds=args.sample_interval_seconds,
        lease_path=args.lease_path,
        heartbeat_path=args.heartbeat_path,
        audit_path=args.audit_path,
        heartbeat_max_age_seconds=args.heartbeat_max_age_seconds,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(argv if argv is not None else sys.argv[1:])
    if arguments and arguments[0] == "--internal-guardian":
        if len(arguments) < 7 or arguments[5] != "--":
            return EXIT_LAUNCH_ERROR
        try:
            return _guardian_main(
                int(arguments[1]),
                int(arguments[2]),
                _positive_float(arguments[3]),
                _positive_float(arguments[4]),
                tuple(arguments[6:]),
            )
        except (OSError, ValueError):
            return EXIT_LAUNCH_ERROR
    config = parse_args(arguments)
    audit = AuditLogger(sys.stderr)
    try:
        return run_watchdog(config, audit=audit)
    except ValueError as exc:
        return _emit_final(
            audit,
            "configuration_error",
            EXIT_PROCFS_ERROR,
            "invalid configuration",
            None,
            None,
            error=str(exc),
        )
    except Exception as exc:
        return _emit_final(
            audit,
            "internal_error",
            EXIT_INTERNAL_ERROR,
            "unexpected watchdog error",
            None,
            None,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        audit.close()


if __name__ == "__main__":
    sys.exit(main())
