#!/usr/bin/env python3

import argparse
import array
import ctypes
import hashlib
import json
import math
import os
import re
import select
import selectors
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

TRACE_FORMAT = "dsv41-trace"
TRACE_VERSION = 2
DS4_REVISION = "bd66c402070042bf0a79ad6ece8242de4c93680c"
DS4_REPOSITORY = "antirez/ds4"
APPROVED_TRACE_SIGNERS: dict[str, dict[str, str]] = {}
APPROVED_EXECUTABLE_APPROVERS: dict[str, dict[str, Any]] = {}
APPROVED_CANDIDATE_EXPORTERS: dict[str, dict[str, Any]] = {}
APPROVED_DS4_EXPORTERS: dict[str, dict[str, Any]] = {}
APPROVED_PROMPT_BUILDERS: dict[str, dict[str, Any]] = {}
EXECUTABLE_APPROVAL_FORMAT = "dsv41-executable-approval"
EXECUTABLE_APPROVAL_VERSION = 2
EXECUTABLE_APPROVAL_NAMESPACE = "dsv41-executable-approval-v2"
SEAL_FORMAT = "dsv41-trace-bundle-signature"
SEAL_VERSION = 1
SEAL_NAMESPACE = "dsv41-trace-bundle-v1"
SEAL_DOMAIN_PREFIX = b"dsv41-trace-bundle-v1\n"
SIGNATURE_NAME = "bundle-signature.json"
CANDIDATE_LANE = "strix-llama-candidate-v1"
ORACLE_LANE = "apple-ds4-oracle-v1"
AUTHORIZATION_FORMAT = "dsv41-execution-authorization"
AUTHORIZATION_VERSION = 1
MAX_AUTHORIZATION_LIFETIME_SECONDS = 24 * 60 * 60
MODEL_SHA256 = "1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42"
REPOSITORY = "halo-box/strix-llama.cpp"
SOFT_MEMORY_LIMIT = 116 * 1024 * 1024 * 1024
WATCHDOG_EMERGENCY_LIMIT = 118 * 1024 * 1024 * 1024
STRICT_MEMORY_LIMIT = 120 * 1024 * 1024 * 1024
WATCHDOG_LEASE_FORMAT = "strix-memory-watchdog-lease"
WATCHDOG_VERSION = 2
WATCHDOG_REVISION = "02235de637ae1ffe8aeef9479628962792190b22"
WATCHDOG_SCRIPT_SHA256 = "d2781a25f978dd2bc14fc113079aa2dbf513aa157b44da9d0d51d750daa6c94f"
APPROVED_WATCHDOGS = {WATCHDOG_SCRIPT_SHA256: WATCHDOG_REVISION}
NO_EXTERNAL_STATE_STORAGE = {
    "format": "dsv41-state-storage-policy",
    "version": 1,
    "expert_cache": "memory-resident",
    "kv_cache": "memory-resident",
    "external_cache_paths": [],
    "external_state_paths": [],
}
ADMITTED_UBATCH = 32
ADMITTED_BATCH = 2048
EXPERT_COUNT = 384
EXPERTS_USED = 6
EXPERT_SLOT_BYTES = 398_131_200
REQUIRED_EXPERT_SLOTS = min(EXPERT_COUNT, EXPERTS_USED * ADMITTED_UBATCH)
REQUIRED_EXPERT_CACHE_BYTES = REQUIRED_EXPERT_SLOTS * EXPERT_SLOT_BYTES
REQUIRED_EXPERT_CACHE_MIB = REQUIRED_EXPERT_CACHE_BYTES // (1024 * 1024)
RAW_ATTENTION_LAYERS = (0, 1)
RAW_ATTENTION_WIDTH = 128
CORPUS_SHA256 = {
    "correctness-prose.txt": "2da590a37e3297767336c10b024a0de732d64bee4da5792596f8ddf49ea408d2",
    "correctness-code.txt": "41b4246ef4e6b4e3f9f23a3d02aa8cdab48f495b3af0ebeaccea255679c771f0",
    "correctness-structured.txt": "1278707adea5a953196c4cf5c04de301952813be3eac416a6aac4ff94f42f701",
    "correctness-numeric.txt": "ebd444cf70662cc09289af45ef654af0953b98e967a449d031627d8ea92bc2e0",
}
MANIFEST_NAME = "manifest.json"
EVENTS_NAME = "events.jsonl"
BLOBS_DIR = "blobs"
FORBIDDEN_LOADER_ENVIRONMENT = (
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
)

DTYPE_SIZES = {
    "f32": 4,
    "bf16": 2,
    "i32": 4,
    "u32": 4,
    "i8": 1,
    "u8": 1,
    "bytes": 1,
}

HARD_FAILURE_COMPONENTS = (
    "prompt.bytes",
    "prompt.tokens",
    "engram.row_ids",
    "expert.ids",
    "expert.weights",
    "attn.source",
    "attn.candidate_blocks",
    "attn.candidates",
    "logits.prefill",
    "logits.decode",
    "decode.greedy_token",
)

DEEPSEEK41_EXPECTED_COMPONENTS = {
    "prompt.bytes": {"layers": None, "input": "tokens"},
    "prompt.tokens": {"layers": None, "input": "tokens"},
    "engram.row_ids": {"layers": [1, 14], "prefill": "tokens", "decode": "steps"},
    "expert.ids": {"layers": list(range(40)), "prefill": "tokens", "decode": "steps"},
    "expert.weights": {"layers": list(range(40)), "prefill": "tokens", "decode": "steps"},
    "attn.source": {"layers": list(range(40)), "prefill": "tokens", "decode": "steps"},
    "attn.candidate_blocks": {"layers": [20], "prefill": "tokens", "decode": "steps"},
    "attn.candidates": {"layers": [24, 28, 32, 36], "prefill": "tokens", "decode": "steps"},
    "logits.prefill": {"layers": None, "prefill": "final"},
    "logits.decode": {"layers": None, "decode": "steps"},
    "decode.greedy_token": {"layers": None, "decode": "steps"},
}


class TraceError(RuntimeError):
    pass


PROCESS_TREE_CLEANUP_TIMEOUT_SECONDS = 5
PROCESS_STARTUP_DIAGNOSTIC_MAX_BYTES = 65536
LINUX_PROTOCOL_MAX_RECEIVED_DESCRIPTORS = 8
PROCESS_TREE_TERM_GRACE_SECONDS = 1
WINDOWS_CREATE_SUSPENDED = 0x00000004
WINDOWS_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000


@dataclass(frozen=True)
class _IntegrityFailure:
    component: str
    error: BaseException


class ExecutionIntegrityError(TraceError):
    def __init__(
            self,
            message: str,
            *,
            primary_error: BaseException | None,
            secondary_errors: list[_IntegrityFailure],
            quiescence_proven: bool = False):
        super().__init__(message)
        self.primary_error = primary_error
        self.secondary_errors = tuple(secondary_errors)
        self.quiescence_proven = quiescence_proven


@dataclass
class _ProcessContainment:
    process: Any
    linux_root_pidfd: int | None = None
    linux_namespace_pidfd: int | None = None
    linux_lock_held: bool = False
    linux_exec_released: bool = False
    test_process_group_id: int | None = None
    job_handle: int | None = None
    windows_job_assigned: bool = False
    windows_process_resumed: bool = False


@dataclass
class _ContainedRun:
    result: subprocess.CompletedProcess[bytes] | None
    primary_error: BaseException | None
    integrity_failures: list[_IntegrityFailure]
    containment: _ProcessContainment | None
    quiescence_proven: bool = False
    process_started: bool = False


@dataclass
class _ContainmentCleanup:
    failures: list[_IntegrityFailure]
    quiescence_proven: bool


_LINUX_HELPER_LOCK = threading.Lock()
_LINUX_HELPER_POISONED = False
_LINUX_POISONED_CONTAINMENT: _ProcessContainment | None = None
_TEST_PROCESS_GROUP_CONTAINMENT = threading.local()


@dataclass(frozen=True)
class TraceVerifier:
    principal: str
    trusted_signers: dict[str, dict[str, str]]
    ssh_keygen: Path
    expected_lane: str
    expected_challenge: str
    expected_run_id: str
    verification_unix: int
    candidate_exporter_policies: dict[str, dict[str, Any]]
    ds4_exporter_policies: dict[str, dict[str, Any]]
    prompt_builder_policies: dict[str, dict[str, Any]]
    expected_candidate_exporter_policy_id: str | None
    expected_ds4_exporter_policy_id: str | None
    expected_prompt_builder_policy_id: str
    expected_approval_policy_sha256: str
    expected_verifier_revision: str
    seen_run_ids: set[str] | None = None
    test_only: bool = False

    @classmethod
    def production(
            cls,
            principal: str,
            *,
            expected_lane: str,
            expected_challenge: str,
            expected_run_id: str,
            expected_candidate_exporter_policy_id: str | None,
            expected_ds4_exporter_policy_id: str | None,
            expected_prompt_builder_policy_id: str,
            approval_policy: "ExecutableApprovalPolicy | None" = None,
            verification_unix: int | None,
            seen_run_ids: set[str] | None = None) -> "TraceVerifier":
        return cls(
            principal=principal,
            trusted_signers=APPROVED_TRACE_SIGNERS,
            ssh_keygen=trusted_ssh_keygen_path(),
            expected_lane=expected_lane,
            expected_challenge=expected_challenge,
            expected_run_id=expected_run_id,
            verification_unix=int(time.time()) if verification_unix is None else verification_unix,
            candidate_exporter_policies=(
                approval_policy.candidate_exporters
                if approval_policy is not None else APPROVED_CANDIDATE_EXPORTERS),
            ds4_exporter_policies=(
                approval_policy.ds4_exporters
                if approval_policy is not None else APPROVED_DS4_EXPORTERS),
            prompt_builder_policies=(
                approval_policy.prompt_builders
                if approval_policy is not None else APPROVED_PROMPT_BUILDERS),
            expected_candidate_exporter_policy_id=expected_candidate_exporter_policy_id,
            expected_ds4_exporter_policy_id=expected_ds4_exporter_policy_id,
            expected_prompt_builder_policy_id=expected_prompt_builder_policy_id,
            expected_approval_policy_sha256=(
                approval_policy.sha256 if approval_policy is not None else ""),
            expected_verifier_revision=(
                approval_policy.verifier_revision if approval_policy is not None else ""),
            seen_run_ids=seen_run_ids,
        )

    @classmethod
    def for_tests(
            cls,
            principal: str,
            public_key: str,
            *,
            lane: str,
            runtime: str,
            runtime_profile: str,
            expected_challenge: str,
            expected_run_id: str,
            candidate_exporter_policies: dict[str, dict[str, Any]] | None = None,
            ds4_exporter_policies: dict[str, dict[str, Any]] | None = None,
            prompt_builder_policies: dict[str, dict[str, Any]] | None = None,
            expected_candidate_exporter_policy_id: str | None = None,
            expected_ds4_exporter_policy_id: str | None = None,
            expected_prompt_builder_policy_id: str = "",
            expected_approval_policy_sha256: str = "e" * 64,
            expected_verifier_revision: str = "a" * 40,
            verification_unix: int | None = None,
            ssh_keygen: Path | None = None,
            seen_run_ids: set[str] | None = None) -> "TraceVerifier":
        return cls(
            principal=principal,
            trusted_signers={
                principal: {
                    "public_key": public_key,
                    "lane": lane,
                    "runtime": runtime,
                    "runtime_profile": runtime_profile,
                },
            },
            ssh_keygen=ssh_keygen or trusted_ssh_keygen_path(),
            expected_lane=lane,
            expected_challenge=expected_challenge,
            expected_run_id=expected_run_id,
            verification_unix=verification_unix,
            candidate_exporter_policies=candidate_exporter_policies or {},
            ds4_exporter_policies=ds4_exporter_policies or {},
            prompt_builder_policies=prompt_builder_policies or {},
            expected_candidate_exporter_policy_id=expected_candidate_exporter_policy_id,
            expected_ds4_exporter_policy_id=expected_ds4_exporter_policy_id,
            expected_prompt_builder_policy_id=expected_prompt_builder_policy_id,
            expected_approval_policy_sha256=expected_approval_policy_sha256,
            expected_verifier_revision=expected_verifier_revision,
            seen_run_ids=seen_run_ids,
            test_only=True,
        )


@dataclass(frozen=True)
class BundleFileReceipt:
    device: int
    inode: int
    byte_count: int
    modified_ns: int
    changed_ns: int
    sha256: str


@dataclass(frozen=True)
class ExecutableFileReceipt:
    path: str
    install_root: str
    device: int
    inode: int
    owner_uid: int
    mode: int
    link_count: int
    byte_count: int
    modified_ns: int
    changed_ns: int
    sha256: str
    path_chain: tuple[tuple[str, int, int, int, int], ...]


@dataclass(frozen=True)
class ExecutableApprovalPolicy:
    principal: str
    verifier_revision: str
    candidate_exporters: dict[str, dict[str, Any]]
    ds4_exporters: dict[str, dict[str, Any]]
    prompt_builders: dict[str, dict[str, Any]]
    sha256: str


@dataclass(frozen=True)
class Mismatch:
    classification: str
    component: str
    phase: str
    step: int
    token_start: int
    layer: int | None
    detail: str
    element_index: int | None = None
    byte_offset: int | None = None
    token_index: int | None = None
    component_element_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        result = {
            "classification": self.classification,
            "component": self.component,
            "phase": self.phase,
            "step": self.step,
            "token_start": self.token_start,
            "layer": self.layer,
            "detail": self.detail,
        }
        if self.element_index is not None:
            result["element_index"] = self.element_index
        if self.byte_offset is not None:
            result["byte_offset"] = self.byte_offset
        if self.token_index is not None:
            result["token_index"] = self.token_index
        if self.component_element_index is not None:
            result["component_element_index"] = self.component_element_index
        return result


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _execution_uid() -> int:
    if os.name != "posix" or not hasattr(os, "geteuid"):
        raise TraceError("immutable install paths require a POSIX execution identity")
    return os.geteuid()


def _path_is_writable_by_execution_identity(path: Path) -> bool:
    if _execution_uid() == 0:
        raise TraceError("immutable install paths require an unprivileged POSIX execution identity")
    try:
        return os.access(path, os.W_OK, effective_ids=True)
    except (OSError, TypeError, NotImplementedError) as error:
        raise TraceError(f"cannot verify effective write access for {path}: {error}") from error


def _path_mode_for_trust(_path: Path, mode: int) -> int:
    return stat.S_IMODE(mode)


def _require_distinct_trusted_owner(expected_owner_uid: int) -> None:
    execution_uid = _execution_uid()
    if execution_uid == 0 or expected_owner_uid == execution_uid:
        raise TraceError("approved install tree owner must be distinct from the unprivileged execution identity")


def _has_access_control_entries(path: Path) -> bool:
    if sys.platform.startswith("linux"):
        try:
            attributes = os.listxattr(path, follow_symlinks=False)
        except OSError as error:
            raise TraceError(f"cannot inspect access controls for {path}: {error}") from error
        return any(name in {"system.posix_acl_access", "system.posix_acl_default"} for name in attributes)
    if sys.platform == "darwin":
        tool = Path("/bin/ls")
        try:
            tool_stat = tool.stat(follow_symlinks=False)
        except OSError as error:
            raise TraceError(f"cannot inspect the fixed macOS ACL verifier: {error}") from error
        if not stat.S_ISREG(tool_stat.st_mode) or tool_stat.st_uid != 0 or (
                stat.S_IMODE(tool_stat.st_mode) & 0o022):
            raise TraceError("the fixed macOS ACL verifier is not trusted")
        try:
            result = subprocess.run(
                [str(tool), "-lde", str(path)],
                stdin=subprocess.DEVNULL,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise TraceError(f"cannot inspect access controls for {path}: {error}") from error
        if result.returncode != 0:
            raise TraceError(f"cannot inspect access controls for {path}")
        first_line = result.stdout.splitlines()[0] if result.stdout else ""
        fields = first_line.split()
        if not fields or len(fields[0]) < 10:
            raise TraceError(f"access control output for {path} is invalid")
        return "+" in fields[0]
    raise TraceError("immutable install path access-control verification is unsupported on this platform")


def _immutable_path_chain(
        path: Path,
        *,
        install_root: Path,
        expected_owner_uid: int,
        label: str,
) -> tuple[tuple[str, int, int, int, int], ...]:
    if not path.is_absolute() or not install_root.is_absolute() or (
            path != install_root and install_root not in path.parents):
        raise TraceError(f"{label} path is outside its approved install root")
    try:
        if str(path.resolve(strict=True)) != str(path) or str(install_root.resolve(strict=True)) != str(install_root):
            raise TraceError(f"{label} path must be canonical and must not use aliases")
    except OSError as error:
        raise TraceError(f"cannot resolve {label} path: {error}") from error
    if type(expected_owner_uid) is not int or expected_owner_uid < 0:
        raise TraceError(f"{label} install owner is invalid")
    _require_distinct_trusted_owner(expected_owner_uid)
    result = []
    current = Path(path.anchor)
    candidates = [current]
    for part in path.parent.parts[1:]:
        current /= part
        candidates.append(current)
    install_seen = Path(path.anchor) == install_root
    for current in candidates:
        try:
            current_stat = current.stat(follow_symlinks=False)
        except OSError as error:
            raise TraceError(f"cannot inspect {label} path: {error}") from error
        if stat.S_ISLNK(current_stat.st_mode):
            raise TraceError(f"{label} path must not use symlinks")
        if not stat.S_ISDIR(current_stat.st_mode):
            raise TraceError(f"{label} parent path is not a directory")
        if current == install_root:
            install_seen = True
        if current_stat.st_uid not in {0, expected_owner_uid}:
            raise TraceError(f"{label} path is not trusted-owned")
        if install_seen and current_stat.st_uid != expected_owner_uid:
            raise TraceError(f"{label} install tree owner differs from external approval")
        path_mode = _path_mode_for_trust(current, current_stat.st_mode)
        if path_mode & 0o022 or (
                _path_is_writable_by_execution_identity(current)) or _has_access_control_entries(current):
            raise TraceError(f"{label} path is mutable by the execution identity or an untrusted group")
        result.append((
            str(current),
            current_stat.st_dev,
            current_stat.st_ino,
            current_stat.st_uid,
            path_mode,
        ))
    if not install_seen:
        raise TraceError(f"{label} path does not traverse its approved install root")
    return tuple(result)


def _approved_file_identity(
        path: Path,
        *,
        install_root: Path,
        expected_owner_uid: int,
        expected_path: str,
        expected_sha256: str,
        label: str,
        executable: bool,
) -> tuple[ExecutableFileReceipt, int]:
    if not path.is_absolute() or str(path) != expected_path or not install_root.is_absolute():
        raise TraceError(f"{label} path differs from external approval")
    path_chain = _immutable_path_chain(
        path,
        install_root=install_root,
        expected_owner_uid=expected_owner_uid,
        label=label,
    )
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
    except OSError as error:
        raise TraceError(f"cannot open {label}: {error}") from error
    if not stat.S_ISREG(before.st_mode) or before.st_uid != expected_owner_uid or before.st_nlink != 1 or (
            stat.S_IMODE(before.st_mode) & 0o222) or (
            _path_is_writable_by_execution_identity(path)) or _has_access_control_entries(path) or (
            executable and not (stat.S_IMODE(before.st_mode) & 0o111)):
        os.close(descriptor)
        raise TraceError(f"{label} is not an immutable trusted-owned one-link regular file")
    try:
        digest = hashlib.sha256()
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        digest_value = digest.hexdigest()
        after = os.fstat(descriptor)
        path_after = path.stat(follow_symlinks=False)
    except OSError as error:
        os.close(descriptor)
        raise TraceError(f"cannot recheck {label}: {error}") from error
    identity = (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    if identity != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) or identity != (
            path_after.st_dev, path_after.st_ino, path_after.st_size,
            path_after.st_mtime_ns, path_after.st_ctime_ns):
        os.close(descriptor)
        raise TraceError(f"{label} changed while hashing")
    if digest_value != expected_sha256:
        os.close(descriptor)
        raise TraceError(f"{label} SHA-256 differs from external approval")
    return ExecutableFileReceipt(
        path=str(path),
        install_root=str(install_root),
        device=after.st_dev,
        inode=after.st_ino,
        owner_uid=after.st_uid,
        mode=stat.S_IMODE(after.st_mode),
        link_count=after.st_nlink,
        byte_count=after.st_size,
        modified_ns=after.st_mtime_ns,
        changed_ns=after.st_ctime_ns,
        sha256=digest_value,
        path_chain=path_chain,
    ), descriptor


def approved_executable_identity(
        path: Path,
        *,
        install_root: str,
        expected_owner_uid: int,
        expected_path: str,
        expected_sha256: str,
        label: str,
) -> ExecutableFileReceipt:
    identity, descriptor = _approved_file_identity(
        path,
        install_root=Path(install_root),
        expected_owner_uid=expected_owner_uid,
        expected_path=expected_path,
        expected_sha256=expected_sha256,
        label=label,
        executable=True,
    )
    os.close(descriptor)
    return identity


def verify_approved_executable_identity(
        path: Path,
        expected: ExecutableFileReceipt,
        *,
        label: str,
) -> None:
    observed = approved_executable_identity(
        path,
        install_root=expected.install_root,
        expected_owner_uid=expected.owner_uid,
        expected_path=expected.path,
        expected_sha256=expected.sha256,
        label=label,
    )
    if observed != expected:
        raise TraceError(f"{label} identity changed after approval")


def approved_runtime_file_identities(
        policy: dict[str, Any],
        *,
        label: str,
) -> list[ExecutableFileReceipt]:
    install_root = Path(policy["install_root"])
    identities = []
    for component in policy["runtime_receipt"]["components"]:
        path = install_root / "lib" / component["filename"]
        identity, descriptor = _approved_file_identity(
            path,
            install_root=install_root,
            expected_owner_uid=policy["install_owner_uid"],
            expected_path=str(path),
            expected_sha256=component["sha256"],
            label=f"{label} runtime component {component['component']}",
            executable=False,
        )
        os.close(descriptor)
        identities.append(identity)
    return identities


def verify_approved_runtime_file_identities(
        identities: list[ExecutableFileReceipt],
        *,
        label: str,
) -> None:
    for identity in identities:
        observed, descriptor = _approved_file_identity(
            Path(identity.path),
            install_root=Path(identity.path).parent.parent,
            expected_owner_uid=identity.owner_uid,
            expected_path=identity.path,
            expected_sha256=identity.sha256,
            label=f"{label} runtime component",
            executable=False,
        )
        os.close(descriptor)
        if observed != identity:
            raise TraceError(f"{label} runtime component identity changed after approval")


def _windows_error(message: str) -> TraceError:
    return TraceError(f"{message}: Windows error {ctypes.get_last_error()}")


def _windows_kernel32() -> Any:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateJobObject.restype = wintypes.BOOL
    kernel32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
    kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Thread32First.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    kernel32.Thread32First.restype = wintypes.BOOL
    kernel32.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    kernel32.Thread32Next.restype = wintypes.BOOL
    kernel32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenThread.restype = wintypes.HANDLE
    kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel32.ResumeThread.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


def _create_windows_kill_job() -> int:
    from ctypes import wintypes

    class BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimitInformation),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = _windows_kernel32()
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise _windows_error("cannot create process containment job")
    limits = ExtendedLimitInformation()
    limits.BasicLimitInformation.LimitFlags = WINDOWS_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
            job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        primary_error = _windows_error("cannot configure process containment job")
        if not kernel32.CloseHandle(job):
            close_error = _windows_error("cannot close unconfigured process containment job")
            raise ExecutionIntegrityError(
                f"Windows job configuration primary failure "
                f"[{type(primary_error).__name__}: {primary_error}]; "
                f"secondary integrity failures: windows-job-handle-close "
                f"[{type(close_error).__name__}: {close_error}]",
                primary_error=primary_error,
                secondary_errors=[_IntegrityFailure("windows-job-handle-close", close_error)],
            ) from primary_error
        raise primary_error
    return int(job)


def _open_windows_process_thread(process_id: int) -> int:
    from ctypes import wintypes

    class ThreadEntry32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ThreadID", wintypes.DWORD),
            ("th32OwnerProcessID", wintypes.DWORD),
            ("tpBasePri", wintypes.LONG),
            ("tpDeltaPri", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
        ]

    kernel32 = _windows_kernel32()
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000004, 0)
    if not snapshot or int(snapshot) == ctypes.c_void_p(-1).value:
        raise _windows_error("cannot enumerate suspended process threads")
    entry = ThreadEntry32()
    entry.dwSize = ctypes.sizeof(entry)
    thread = None
    try:
        found = kernel32.Thread32First(snapshot, ctypes.byref(entry))
        while found:
            if entry.th32OwnerProcessID == process_id:
                thread = kernel32.OpenThread(0x0002 | 0x00100000, False, entry.th32ThreadID)
                if not thread:
                    raise _windows_error("cannot open suspended process thread")
                break
            found = kernel32.Thread32Next(snapshot, ctypes.byref(entry))
    except BaseException as primary_error:
        if not kernel32.CloseHandle(snapshot):
            close_error = _windows_error("cannot close process thread snapshot")
            raise ExecutionIntegrityError(
                f"Windows thread enumeration primary failure "
                f"[{type(primary_error).__name__}: {primary_error}]; "
                f"secondary integrity failures: windows-snapshot-handle-close "
                f"[{type(close_error).__name__}: {close_error}]",
                primary_error=primary_error,
                secondary_errors=[_IntegrityFailure("windows-snapshot-handle-close", close_error)],
            ) from primary_error
        raise
    if not kernel32.CloseHandle(snapshot):
        close_error = _windows_error("cannot close process thread snapshot")
        failures = [_IntegrityFailure("windows-snapshot-handle-close", close_error)]
        if thread:
            if not kernel32.CloseHandle(thread):
                failures.append(_IntegrityFailure(
                    "windows-thread-handle-close",
                    _windows_error("cannot close suspended process thread")))
        if len(failures) > 1:
            raise ExecutionIntegrityError(
                f"Windows thread enumeration integrity failures: "
                f"{_format_integrity_failures(failures)}",
                primary_error=None,
                secondary_errors=failures,
            ) from close_error
        raise close_error
    if not thread:
        raise TraceError("cannot find suspended process thread")
    return int(thread)


def _start_windows_job_process(command: list[str], launch: dict[str, Any]) -> _ProcessContainment:
    creationflags = int(launch.pop("creationflags", 0)) | WINDOWS_CREATE_SUSPENDED
    process = subprocess.Popen(command, creationflags=creationflags, **launch)
    job = None
    thread = None
    job_assigned = False
    process_resumed = False
    try:
        job = _create_windows_kill_job()
        kernel32 = _windows_kernel32()
        process_handle = int(process._handle)
        if not kernel32.AssignProcessToJobObject(job, process_handle):
            raise _windows_error("cannot assign suspended process to containment job")
        job_assigned = True
        thread = _open_windows_process_thread(process.pid)
        if kernel32.ResumeThread(thread) == 0xFFFFFFFF:
            raise _windows_error("cannot resume contained process")
        process_resumed = True
        thread_closed = kernel32.CloseHandle(thread)
        thread = None
        if not thread_closed:
            raise _windows_error("cannot close resumed process thread")
        return _ProcessContainment(
            process=process,
            job_handle=job,
            windows_job_assigned=job_assigned,
            windows_process_resumed=process_resumed,
        )
    except BaseException as primary_error:
        kernel32 = _windows_kernel32()
        failures = []
        if thread is not None:
            if not kernel32.CloseHandle(thread):
                failures.append(_IntegrityFailure(
                    "windows-thread-handle-close",
                    _windows_error("cannot close suspended process thread")))
        if job_assigned and job is not None:
            if not kernel32.TerminateJobObject(job, 1):
                failures.append(_IntegrityFailure(
                    "windows-job-termination",
                    _windows_error("cannot terminate failed process containment job")))
        else:
            try:
                process.kill()
            except BaseException as error:
                failures.append(_IntegrityFailure("windows-process-termination", error))
        try:
            process.wait(timeout=PROCESS_TREE_CLEANUP_TIMEOUT_SECONDS)
        except BaseException as error:
            failures.append(_IntegrityFailure("windows-process-reap", error))
        try:
            _close_windows_process_handle(process)
        except BaseException as error:
            failures.append(_IntegrityFailure("windows-process-handle-close", error))
        if job is not None and not kernel32.CloseHandle(job):
            failures.append(_IntegrityFailure(
                "windows-job-handle-close",
                _windows_error("cannot close failed process containment job")))
        if failures:
            raise ExecutionIntegrityError(
                f"Windows containment startup primary failure "
                f"[{type(primary_error).__name__}: {primary_error}]; "
                f"secondary integrity failures: {_format_integrity_failures(failures)}",
                primary_error=primary_error,
                secondary_errors=failures,
                quiescence_proven=False,
            ) from primary_error
        raise ExecutionIntegrityError(
            f"Windows containment startup primary failure "
            f"[{type(primary_error).__name__}: {primary_error}]",
            primary_error=primary_error,
            secondary_errors=[],
            quiescence_proven=False,
        ) from primary_error


@contextmanager
def _test_only_process_group_containment() -> Iterable[None]:
    previous = getattr(_TEST_PROCESS_GROUP_CONTAINMENT, "enabled", False)
    _TEST_PROCESS_GROUP_CONTAINMENT.enabled = True
    try:
        yield
    finally:
        _TEST_PROCESS_GROUP_CONTAINMENT.enabled = previous


def _linux_fd_is_close_on_exec(descriptor: int) -> bool:
    import fcntl

    return bool(fcntl.fcntl(descriptor, fcntl.F_GETFD) & fcntl.FD_CLOEXEC)


class _LinuxNativeHelperProcess:
    def __init__(
            self,
            command: list[str],
            pid: int,
            *,
            protocol_socket: socket.socket,
            stdin_fd: int | None,
            stdout_fd: int | None,
            stderr_fd: int | None):
        self.args = command
        self.pid = pid
        self.returncode = None
        self.root_pidfd = None
        self.namespace_pidfd = None
        self.completion_proven = False
        self.launch_primary_error = None
        self.launch_integrity_failures = []
        self._protocol_socket = protocol_socket
        self._stdin_fd = stdin_fd
        self._stdout_fd = stdout_fd
        self._stderr_fd = stderr_fd

    def _receive_protocol(self, expected: bytes, *, receive_pidfd: bool = False) -> None:
        item_size = array.array("i").itemsize
        requested_flags = getattr(socket, "MSG_CMSG_CLOEXEC", 0)
        data, ancillary, flags, _address = self._protocol_socket.recvmsg(
            128,
            socket.CMSG_SPACE(item_size * LINUX_PROTOCOL_MAX_RECEIVED_DESCRIPTORS),
            requested_flags,
        )
        received = []
        rights_records = 0
        ancillary_error = None
        for level, kind, content in ancillary:
            if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
                ancillary_error = "Linux containment helper sent unexpected ancillary data"
                continue
            rights_records += 1
            complete_size = len(content) - len(content) % item_size
            descriptor_bytes = array.array("i")
            descriptor_bytes.frombytes(content[:complete_size])
            received.extend(descriptor_bytes)
            if len(content) == 0 or complete_size != len(content):
                ancillary_error = "Linux containment helper sent malformed descriptor data"

        def reject(message: str) -> None:
            failures = []
            for descriptor in received:
                try:
                    os.close(descriptor)
                except BaseException as error:
                    failures.append(_IntegrityFailure(
                        "linux-helper-received-fd-close", error))
            primary = TraceError(message)
            if failures:
                raise ExecutionIntegrityError(
                    f"{message}; secondary integrity failures: "
                    f"{_format_integrity_failures(failures)}",
                    primary_error=primary,
                    secondary_errors=failures,
                    quiescence_proven=False,
                ) from primary
            raise primary

        allowed_flags = {0, requested_flags}
        if flags not in allowed_flags:
            reject(
                f"Linux containment helper protocol returned unexpected flags {flags}")
        if data != expected:
            reject(
                f"Linux containment helper protocol expected {expected.decode('ascii')}")
        if ancillary_error is not None:
            reject(ancillary_error)
        if rights_records > 1:
            reject("Linux containment helper sent multiple descriptor records")
        if receive_pidfd:
            if len(received) != 1:
                reject("Linux containment helper did not provide one namespace pidfd")
            descriptor = received[0]
            try:
                close_on_exec = _linux_fd_is_close_on_exec(descriptor)
            except BaseException as error:
                failures = []
                try:
                    os.close(descriptor)
                except BaseException as close_error:
                    failures.append(_IntegrityFailure(
                        "linux-helper-received-fd-close", close_error))
                raise ExecutionIntegrityError(
                    f"Linux containment helper namespace pidfd validation failed "
                    f"[{type(error).__name__}: {error}]"
                    + (f"; secondary integrity failures: "
                       f"{_format_integrity_failures(failures)}" if failures else ""),
                    primary_error=error,
                    secondary_errors=failures,
                    quiescence_proven=False,
                ) from error
            if not close_on_exec:
                reject("Linux containment helper namespace pidfd is not close-on-exec")
            self.namespace_pidfd = descriptor
        elif received:
            reject("Linux containment helper sent an unexpected descriptor")

    def release_exec(self) -> None:
        self._receive_protocol(b"READY")
        self._protocol_socket.sendall(b"PREPARE")
        self._receive_protocol(b"PREPARED", receive_pidfd=True)
        self._protocol_socket.sendall(b"EXEC")
        self._receive_protocol(b"RELEASED")

    def abort_blocked(self) -> _ContainmentCleanup:
        failures = []
        try:
            self._protocol_socket.shutdown(socket.SHUT_RDWR)
        except BaseException as error:
            failures.append(_IntegrityFailure("linux-helper-protocol-shutdown", error))
            try:
                self._protocol_socket.close()
            except BaseException as close_error:
                failures.append(_IntegrityFailure(
                    "linux-process-fd-close:protocol", close_error))
            self._protocol_socket = None
        try:
            self.wait(timeout=PROCESS_TREE_CLEANUP_TIMEOUT_SECONDS)
        except BaseException as error:
            failures.append(_IntegrityFailure("linux-blocked-child-reap", error))
            if self._protocol_socket is not None:
                try:
                    self._protocol_socket.close()
                except BaseException as close_error:
                    failures.append(_IntegrityFailure(
                        "linux-process-fd-close:protocol", close_error))
                self._protocol_socket = None
            try:
                self.wait(timeout=PROCESS_TREE_CLEANUP_TIMEOUT_SECONDS)
            except BaseException as retry_error:
                failures.append(_IntegrityFailure("linux-blocked-child-reap-retry", retry_error))
        return _ContainmentCleanup(failures, not failures and self.returncode is not None)

    def close_streams(self) -> list[_IntegrityFailure]:
        failures = []
        if self._protocol_socket is not None:
            try:
                self._protocol_socket.close()
            except BaseException as error:
                failures.append(_IntegrityFailure("linux-process-fd-close:protocol", error))
            self._protocol_socket = None
        for attribute in ("_stdin_fd", "_stdout_fd", "_stderr_fd"):
            descriptor = getattr(self, attribute)
            if descriptor is None:
                continue
            try:
                os.close(descriptor)
            except BaseException as error:
                failures.append(_IntegrityFailure(
                    f"linux-process-fd-close:{attribute.removeprefix('_').removesuffix('_fd')}",
                    error,
                ))
            setattr(self, attribute, None)
        return failures

    def collect_startup_stderr(self) -> tuple[bytes, list[_IntegrityFailure]]:
        if self._stderr_fd is None:
            return b"", []
        descriptor = self._stderr_fd
        self._stderr_fd = None
        data = bytearray()
        failures = []
        try:
            while len(data) <= PROCESS_STARTUP_DIAGNOSTIC_MAX_BYTES:
                chunk = os.read(
                    descriptor,
                    PROCESS_STARTUP_DIAGNOSTIC_MAX_BYTES + 1 - len(data),
                )
                if not chunk:
                    break
                data.extend(chunk)
            if len(data) > PROCESS_STARTUP_DIAGNOSTIC_MAX_BYTES:
                failures.append(_IntegrityFailure(
                    "linux-helper-stderr-bounds",
                    TraceError("Linux helper startup stderr exceeded its bound"),
                ))
        except BaseException as error:
            failures.append(_IntegrityFailure("linux-helper-stderr-read", error))
        try:
            os.close(descriptor)
        except BaseException as error:
            failures.append(_IntegrityFailure("linux-process-fd-close:stderr", error))
        return bytes(data[:PROCESS_STARTUP_DIAGNOSTIC_MAX_BYTES]), failures

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        try:
            waited_pid, status = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError as error:
            if self.returncode is None:
                raise TraceError("owned Linux child identity was lost before reap") from error
            return self.returncode
        if waited_pid == 0:
            return None
        self.returncode = os.waitstatus_to_exitcode(status)
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(self.args, timeout)
            time.sleep(0.01)
        return int(self.returncode)

    def kill(self) -> None:
        if self.poll() is None:
            raise TraceError("owned Linux helper termination requires its stable pidfd")

    def communicate(
            self,
            input: bytes | None = None,
            timeout: float | None = None,
    ) -> tuple[bytes | None, bytes | None]:
        if input is not None and self._stdin_fd is None:
            raise ValueError("stdin is not a pipe")
        output = bytearray()
        errors = bytearray()
        had_stdout = self._stdout_fd is not None
        had_stderr = self._stderr_fd is not None
        selector = selectors.DefaultSelector()
        streams = {}
        input_view = memoryview(input or b"")
        input_offset = 0
        if self._stdin_fd is not None:
            if input is None:
                os.close(self._stdin_fd)
                self._stdin_fd = None
            else:
                os.set_blocking(self._stdin_fd, False)
                selector.register(self._stdin_fd, selectors.EVENT_WRITE)
        for fd, buffer in ((self._stdout_fd, output), (self._stderr_fd, errors)):
            if fd is not None:
                os.set_blocking(fd, False)
                selector.register(fd, selectors.EVENT_READ)
                streams[fd] = buffer
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            while selector.get_map() or self.poll() is None:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(
                            self.args,
                            timeout,
                            output=bytes(output) if self._stdout_fd is not None else None,
                            stderr=bytes(errors) if self._stderr_fd is not None else None,
                        )
                else:
                    remaining = 0.05
                for key, events in selector.select(min(remaining, 0.05)):
                    if key.fd == self._stdin_fd and events & selectors.EVENT_WRITE:
                        try:
                            written = os.write(
                                key.fd, input_view[input_offset:input_offset + 65536])
                            input_offset += written
                        except BrokenPipeError:
                            input_offset = len(input_view)
                        if input_offset == len(input_view):
                            selector.unregister(key.fd)
                            os.close(key.fd)
                            self._stdin_fd = None
                        continue
                    chunk = os.read(key.fd, 65536)
                    if chunk:
                        streams[key.fd].extend(chunk)
                    else:
                        selector.unregister(key.fd)
                        os.close(key.fd)
                        if key.fd == self._stdout_fd:
                            self._stdout_fd = None
                        if key.fd == self._stderr_fd:
                            self._stderr_fd = None
                if not selector.get_map() and self.poll() is None:
                    time.sleep(0.01)
            self.wait(timeout=0)
            self._receive_protocol(b"COMPLETE")
            self.completion_proven = True
        finally:
            selector.close()
        self._stdout_fd = None
        self._stderr_fd = None
        return (
            bytes(output) if had_stdout else None,
            bytes(errors) if had_stderr else None,
        )


def _linux_child_file_descriptors(
        mode: Any,
        target_fd: int,
) -> tuple[int | None, int | None]:
    if mode is None:
        return None, None
    if mode == subprocess.PIPE:
        read_fd, write_fd = os.pipe()
        if target_fd == 0:
            return write_fd, read_fd
        return read_fd, write_fd
    if mode == subprocess.DEVNULL:
        flags = os.O_RDONLY if target_fd == 0 else os.O_WRONLY
        descriptor = os.open(os.devnull, flags)
        return None, descriptor
    if mode == subprocess.STDOUT and target_fd == 2:
        return None, subprocess.STDOUT
    raise TraceError("Linux native helper containment received unsupported stream controls")


def _all_catchable_signals() -> set[int]:
    return {
        int(member) for member in signal.valid_signals()
        if int(member) not in {signal.SIGKILL, signal.SIGSTOP}
    }


def _require_zero_supplementary_groups() -> None:
    try:
        groups = os.getgroups()
    except OSError as error:
        raise TraceError(
            f"cannot query Linux containment launcher supplementary groups: {error}") from error
    if groups:
        raise TraceError(
            f"Linux containment launcher requires zero supplementary groups; found {len(groups)}")


def _start_linux_native_helper_process(
        command: list[str],
        launch: dict[str, Any],
) -> _LinuxNativeHelperProcess:
    controls = dict(launch)
    helper_path = str(controls.pop("_containment_helper_path"))
    helper_descriptor = int(controls.pop("_containment_helper_descriptor"))
    target_executable = str(controls.pop("executable", command[0]))
    pass_fds = tuple(int(fd) for fd in controls.pop("pass_fds", ()))
    environment = controls.pop("env", None)
    working_directory = controls.pop("cwd", None)
    if working_directory is not None:
        raise TraceError("Linux native containment helper does not support cwd")
    unknown_controls = set(controls) - {"stderr", "stdin", "stdout"}
    if unknown_controls:
        raise TraceError(
            f"Linux native containment helper received unsupported controls: {sorted(unknown_controls)}")
    parent_socket = None
    child_socket = None
    inheritable_before = {}
    stdin_parent = None
    stdin_child = None
    stdout_parent = None
    stdout_child = None
    stderr_parent = None
    stderr_child = None
    process = None
    primary_error = None
    integrity_failures = []
    try:
        stdin_parent, stdin_child = _linux_child_file_descriptors(controls.pop("stdin", None), 0)
        stdout_parent, stdout_child = _linux_child_file_descriptors(controls.pop("stdout", None), 1)
        stderr_parent, stderr_child = _linux_child_file_descriptors(controls.pop("stderr", None), 2)
        parent_socket, child_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        parent_socket.settimeout(PROCESS_TREE_CLEANUP_TIMEOUT_SECONDS)
        child_descriptors = [
            descriptor for descriptor in (stdin_child, stdout_child, stderr_child)
            if descriptor is not None and descriptor != subprocess.STDOUT]
        if any(descriptor <= 2 for descriptor in child_descriptors):
            raise TraceError("Linux native containment helper requires intact standard descriptors")
        inherited = {child_socket.fileno(), *pass_fds}
        for descriptor in inherited:
            inheritable_before[descriptor] = os.get_inheritable(descriptor)
            os.set_inheritable(descriptor, True)
        helper_argv = [
            helper_path,
            "--protocol-fd", str(child_socket.fileno()),
            "--expected-parent", str(os.getpid()),
            "--exec-path", target_executable,
        ]
        for descriptor in pass_fds:
            helper_argv.extend(("--keep-fd", str(descriptor)))
        helper_argv.append("--")
        helper_argv.extend(command)
        file_actions = []
        for child_fd, target_fd in (
                (stdin_child, 0), (stdout_child, 1), (stderr_child, 2)):
            if child_fd == subprocess.STDOUT:
                file_actions.append((os.POSIX_SPAWN_DUP2, 1, 2))
            elif child_fd is not None:
                file_actions.append((os.POSIX_SPAWN_DUP2, child_fd, target_fd))
        helper_exec_path = f"/proc/self/fd/{helper_descriptor}"
        process_id = os.posix_spawn(
            helper_exec_path,
            helper_argv,
            os.environ if environment is None else environment,
            file_actions=file_actions,
            setsid=True,
            setsigmask=_all_catchable_signals(),
            setsigdef=_all_catchable_signals(),
        )
        process = _LinuxNativeHelperProcess(
            command,
            process_id,
            protocol_socket=parent_socket,
            stdin_fd=stdin_parent,
            stdout_fd=stdout_parent,
            stderr_fd=stderr_parent,
        )
        parent_socket = None
        stdin_parent = None
        stdout_parent = None
        stderr_parent = None
        process.root_pidfd = _linux_open_pidfd(process_id)
    except BaseException as error:
        primary_error = error
    for descriptor, previous in inheritable_before.items():
        try:
            os.set_inheritable(descriptor, previous)
        except BaseException as error:
            integrity_failures.append(_IntegrityFailure(
                "linux-launch-descriptor-inheritability-restore", error))
    if child_socket is not None:
        try:
            child_socket.close()
        except BaseException as error:
            integrity_failures.append(_IntegrityFailure(
                "linux-helper-child-protocol-close", error))
    for component, descriptor in (
            ("stdin", stdin_child),
            ("stdout", stdout_child),
            ("stderr", stderr_child)):
        if descriptor is not None and descriptor != subprocess.STDOUT:
            try:
                os.close(descriptor)
            except BaseException as error:
                integrity_failures.append(_IntegrityFailure(
                    f"linux-helper-child-{component}-close", error))
    if process is not None:
        process.launch_primary_error = primary_error
        process.launch_integrity_failures = integrity_failures
        return process
    if parent_socket is not None:
        try:
            parent_socket.close()
        except BaseException as error:
            integrity_failures.append(_IntegrityFailure(
                "linux-helper-parent-protocol-close", error))
    for component, descriptor in (
            ("stdin", stdin_parent),
            ("stdout", stdout_parent),
            ("stderr", stderr_parent)):
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException as error:
                integrity_failures.append(_IntegrityFailure(
                    f"linux-helper-parent-{component}-close", error))
    if primary_error is None and not integrity_failures:
        raise TraceError("Linux native helper launch did not return a process")
    if primary_error is None:
        primary_error = integrity_failures.pop(0).error
    if integrity_failures:
        raise ExecutionIntegrityError(
            f"Linux helper launch primary failure "
            f"[{type(primary_error).__name__}: {primary_error}]"
            + (f"; secondary integrity failures: "
               f"{_format_integrity_failures(integrity_failures)}" if integrity_failures else ""),
            primary_error=primary_error,
            secondary_errors=integrity_failures,
            quiescence_proven=False,
        ) from primary_error
    raise primary_error


def _linux_task_ids() -> set[int]:
    task_root = Path("/proc/self/task")
    if not task_root.is_dir():
        raise TraceError("Linux native containment requires procfs task identities")
    try:
        return {int(task.name) for task in task_root.iterdir()}
    except (OSError, ValueError) as error:
        raise TraceError(f"cannot read Linux native containment task identities: {error}") from error


def _linux_open_pidfd(process_id: int) -> int:
    opener = getattr(os, "pidfd_open", None)
    sender = getattr(signal, "pidfd_send_signal", None)
    if opener is None or sender is None:
        raise TraceError("Linux native containment requires pidfd signaling")
    try:
        return int(opener(process_id, 0))
    except ProcessLookupError:
        raise
    except OSError as error:
        raise TraceError(f"cannot open stable Linux process identity: {error}") from error


def _linux_require_pidfd_support() -> None:
    if getattr(os, "pidfd_open", None) is None or getattr(signal, "pidfd_send_signal", None) is None:
        raise TraceError("Linux native containment requires pidfd signaling")


def _linux_signal_pidfd(pidfd: int, requested_signal: int) -> None:
    sender = getattr(signal, "pidfd_send_signal", None)
    if sender is None:
        raise TraceError("Linux native containment requires pidfd signaling")
    sender(pidfd, requested_signal)


def _linux_pidfd_has_exited(pidfd: int) -> bool:
    poller = select.poll()
    poller.register(pidfd, select.POLLIN)
    return bool(poller.poll(0))


def _linux_signal_owned_children(
        containment: _ProcessContainment,
        requested_signal: int,
) -> list[_IntegrityFailure]:
    failures = []
    for component, pidfd in (
            ("linux-namespace-termination", containment.linux_namespace_pidfd),
            ("linux-helper-termination", containment.linux_root_pidfd)):
        if pidfd is None:
            continue
        try:
            _linux_signal_pidfd(pidfd, requested_signal)
        except ProcessLookupError:
            pass
        except BaseException as error:
            failures.append(_IntegrityFailure(component, error))
    return failures


def _start_linux_native_helper(command: list[str], launch: dict[str, Any]) -> _ProcessContainment:
    global _LINUX_HELPER_POISONED
    if not _LINUX_HELPER_LOCK.acquire(blocking=False):
        raise TraceError("Linux native containment helper is already active")
    if _LINUX_HELPER_POISONED:
        _LINUX_HELPER_LOCK.release()
        raise TraceError("Linux native containment helper supervisor is not reusable")
    process = None
    pidfd = None
    exec_released = False
    try:
        _require_zero_supplementary_groups()
        _linux_require_pidfd_support()
        if len(_linux_task_ids()) != 1:
            raise TraceError("Linux native containment requires a single-threaded Python supervisor")
        process = _start_linux_native_helper_process(command, launch)
        pidfd = process.root_pidfd
        containment = _ProcessContainment(
            process=process,
            linux_root_pidfd=pidfd,
            linux_lock_held=True,
        )
        if process.launch_primary_error is not None or process.launch_integrity_failures:
            launch_primary = process.launch_primary_error
            launch_failures = list(process.launch_integrity_failures)
            if launch_primary is None:
                launch_primary = launch_failures.pop(0).error
            raise ExecutionIntegrityError(
                f"Linux helper launch primary failure "
                f"[{type(launch_primary).__name__}: {launch_primary}]"
                + (f"; secondary integrity failures: "
                   f"{_format_integrity_failures(launch_failures)}" if launch_failures else ""),
                primary_error=launch_primary,
                secondary_errors=launch_failures,
                quiescence_proven=False,
            ) from launch_primary
        if pidfd is None:
            raise TraceError("Linux native containment helper identity is unavailable")
        process.release_exec()
        if process.namespace_pidfd is None or _linux_pidfd_has_exited(process.namespace_pidfd):
            raise TraceError("Linux target PID namespace authority is unavailable before exec")
        containment.linux_namespace_pidfd = process.namespace_pidfd
        exec_released = True
        containment.linux_exec_released = True
        return containment
    except BaseException as primary_error:
        failures = []
        startup_stderr = b""
        if isinstance(primary_error, ExecutionIntegrityError):
            failures.extend(primary_error.secondary_errors)
            primary_error = primary_error.primary_error or primary_error
        if process is not None:
            failed_containment = _ProcessContainment(
                process=process,
                linux_root_pidfd=pidfd,
                linux_namespace_pidfd=process.namespace_pidfd,
                linux_lock_held=True,
                linux_exec_released=exec_released,
            )
            if process.namespace_pidfd is not None or exec_released:
                cleanup = _terminate_process_tree(failed_containment)
            else:
                cleanup = process.abort_blocked()
            failures.extend(cleanup.failures)
            if cleanup.quiescence_proven:
                startup_stderr, stderr_failures = process.collect_startup_stderr()
                failures.extend(stderr_failures)
            close_failures = _close_process_containment(
                failed_containment,
                quiescence_proven=cleanup.quiescence_proven,
            )
            failures.extend(close_failures)
        else:
            _LINUX_HELPER_LOCK.release()
        if failures:
            raise ExecutionIntegrityError(
                f"Linux containment startup primary failure "
                f"[{type(primary_error).__name__}: {primary_error}]"
                + (f"; helper stderr {startup_stderr!r}" if startup_stderr else "")
                + "; "
                f"secondary integrity failures: {_format_integrity_failures(failures)}",
                primary_error=primary_error,
                secondary_errors=failures,
                quiescence_proven=False,
            ) from primary_error
        if process is not None:
            raise ExecutionIntegrityError(
                f"Linux containment startup primary failure "
                f"[{type(primary_error).__name__}: {primary_error}]"
                + (f"; helper stderr {startup_stderr!r}" if startup_stderr else ""),
                primary_error=primary_error,
                secondary_errors=[],
                quiescence_proven=False,
            ) from primary_error
        raise


def _start_contained_process(command: list[str], launch: dict[str, Any]) -> _ProcessContainment:
    if sys.platform == "win32":
        return _start_windows_job_process(command, launch)
    if sys.platform in {"linux", "darwin"} and getattr(
            _TEST_PROCESS_GROUP_CONTAINMENT, "enabled", False):
        launch.pop("_containment_helper_path", None)
        launch.pop("_containment_helper_descriptor", None)
        launch["start_new_session"] = True
        process = subprocess.Popen(command, **launch)
        return _ProcessContainment(process=process, test_process_group_id=process.pid)
    if sys.platform == "linux":
        return _start_linux_native_helper(command, launch)
    raise TraceError("proven process containment is unavailable on this platform")


def _test_process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _windows_job_active_processes(job_handle: int) -> int:
    from ctypes import wintypes

    class BasicAccountingInformation(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_int64),
            ("TotalKernelTime", ctypes.c_int64),
            ("ThisPeriodTotalUserTime", ctypes.c_int64),
            ("ThisPeriodTotalKernelTime", ctypes.c_int64),
            ("TotalPageFaultCount", wintypes.DWORD),
            ("TotalProcesses", wintypes.DWORD),
            ("ActiveProcesses", wintypes.DWORD),
            ("TotalTerminatedProcesses", wintypes.DWORD),
        ]

    accounting = BasicAccountingInformation()
    kernel32 = _windows_kernel32()
    if not kernel32.QueryInformationJobObject(
            job_handle, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None):
        raise _windows_error("cannot query process containment job")
    return int(accounting.ActiveProcesses)


def _close_windows_process_handle(process: subprocess.Popen[bytes]) -> None:
    process_handle = getattr(process, "_handle", None)
    if process_handle is None:
        return
    close = getattr(process_handle, "Close", None)
    if not callable(close):
        raise TraceError("subprocess process handle does not expose owned Close()")
    close()
    process._handle = None


def _process_tree_is_quiescent(containment: _ProcessContainment) -> bool:
    if containment.job_handle is not None:
        return _windows_job_active_processes(containment.job_handle) == 0
    if containment.linux_lock_held:
        helper_exited = containment.process.poll() is not None
        if containment.linux_namespace_pidfd is None:
            return helper_exited and not containment.linux_exec_released
        return helper_exited and _linux_pidfd_has_exited(containment.linux_namespace_pidfd)
    if containment.test_process_group_id is not None:
        return not _test_process_group_exists(containment.test_process_group_id)
    raise TraceError("process containment identity is missing")


def _wait_for_process_tree_quiescence(containment: _ProcessContainment, deadline: float) -> None:
    while not _process_tree_is_quiescent(containment):
        if time.monotonic() >= deadline:
            raise TraceError("process tree did not become quiescent before the cleanup deadline")
        time.sleep(0.01)


def _terminate_process_tree(containment: _ProcessContainment) -> _ContainmentCleanup:
    failures = []
    deadline = time.monotonic() + PROCESS_TREE_CLEANUP_TIMEOUT_SECONDS
    process = containment.process
    if containment.job_handle is not None:
        try:
            if not _windows_kernel32().TerminateJobObject(containment.job_handle, 1):
                raise _windows_error("cannot terminate process containment job")
        except BaseException as error:
            failures.append(_IntegrityFailure("windows-job-termination", error))
        try:
            process.communicate(timeout=max(0.01, deadline - time.monotonic()))
        except BaseException as error:
            failures.append(_IntegrityFailure("windows-process-reap", error))
    elif containment.linux_lock_held:
        failures.extend(_linux_signal_owned_children(containment, signal.SIGTERM))
        try:
            process.communicate(timeout=PROCESS_TREE_TERM_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            failures.extend(_linux_signal_owned_children(containment, signal.SIGKILL))
            try:
                process.communicate(timeout=max(0.01, deadline - time.monotonic()))
            except BaseException as error:
                failures.append(_IntegrityFailure("direct-child-reap", error))
        except BaseException as error:
            failures.append(_IntegrityFailure("direct-child-reap", error))
        failures.extend(_linux_signal_owned_children(containment, signal.SIGKILL))
    elif containment.test_process_group_id is not None:
        try:
            if _test_process_group_exists(containment.test_process_group_id):
                try:
                    os.killpg(containment.test_process_group_id, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                term_deadline = min(deadline, time.monotonic() + PROCESS_TREE_TERM_GRACE_SECONDS)
                try:
                    _wait_for_process_tree_quiescence(containment, term_deadline)
                except TraceError:
                    if _test_process_group_exists(containment.test_process_group_id):
                        try:
                            os.killpg(containment.test_process_group_id, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            pass
        except BaseException as error:
            failures.append(_IntegrityFailure("process-tree-termination", error))
        try:
            process.communicate(timeout=max(0.01, deadline - time.monotonic()))
        except BaseException as error:
            failures.append(_IntegrityFailure("direct-child-reap", error))
    else:
        failures.append(_IntegrityFailure(
            "process-tree-termination", TraceError("process containment identity is missing")))
    try:
        _wait_for_process_tree_quiescence(containment, deadline)
    except BaseException as error:
        failures.append(_IntegrityFailure("process-tree-quiescence", error))
    if containment.linux_exec_released and isinstance(
            process, _LinuxNativeHelperProcess) and not process.completion_proven:
        failures.append(_IntegrityFailure(
            "linux-helper-completion",
            TraceError("Linux native helper did not prove containment teardown"),
        ))
    quiescence_proven = not failures
    if quiescence_proven:
        try:
            quiescence_proven = _process_tree_is_quiescent(containment)
        except BaseException as error:
            failures.append(_IntegrityFailure("process-tree-quiescence", error))
            quiescence_proven = False
    return _ContainmentCleanup(failures, quiescence_proven)


def _run_contained_process(
        command: list[str],
        *,
        label: str,
        timeout: float | None,
        input_data: bytes | None,
        launch: dict[str, Any],
) -> _ContainedRun:
    containment = None
    try:
        containment = _start_contained_process(command, launch)
    except ExecutionIntegrityError as error:
        return _ContainedRun(
            None,
            error.primary_error or error,
            list(error.secondary_errors),
            None,
            error.quiescence_proven,
            True,
        )
    except BaseException as error:
        return _ContainedRun(None, error, [], None, False, False)
    try:
        stdout, stderr = containment.process.communicate(input=input_data, timeout=timeout)
        result = subprocess.CompletedProcess(
            command, containment.process.returncode, stdout, stderr)
    except BaseException as error:
        cleanup = _terminate_process_tree(containment)
        return _ContainedRun(
            None, error, cleanup.failures, containment, cleanup.quiescence_proven, True)
    try:
        if _process_tree_is_quiescent(containment):
            return _ContainedRun(result, None, [], containment, True, True)
    except BaseException as error:
        failures = [_IntegrityFailure("process-tree-quiescence", error)]
    else:
        error = TraceError(f"{label} process tree remained active after direct child exit")
        failures = []
    cleanup = _terminate_process_tree(containment)
    failures.extend(cleanup.failures)
    return _ContainedRun(
        None, error, failures, containment, cleanup.quiescence_proven, True)


def _close_process_containment(
        containment: _ProcessContainment | None,
        *,
        quiescence_proven: bool,
) -> list[_IntegrityFailure]:
    global _LINUX_HELPER_POISONED
    global _LINUX_POISONED_CONTAINMENT
    if containment is None:
        return []
    failures = []
    if containment.linux_lock_held and not quiescence_proven:
        failures.append(_IntegrityFailure(
            "linux-helper-teardown",
            TraceError("Linux native helper cannot be released before full tree quiescence")))
        _LINUX_HELPER_POISONED = True
        _LINUX_POISONED_CONTAINMENT = containment
        return failures
    if containment.job_handle is not None:
        try:
            if not _windows_kernel32().CloseHandle(containment.job_handle):
                raise _windows_error("cannot close process containment job")
            containment.job_handle = None
        except BaseException as error:
            failures.append(_IntegrityFailure("containment-handle-close", error))
        try:
            _close_windows_process_handle(containment.process)
        except BaseException as error:
            failures.append(_IntegrityFailure("windows-process-handle-close", error))
    linux_failure_count = len(failures)
    if containment.linux_root_pidfd is not None:
        try:
            os.close(containment.linux_root_pidfd)
            containment.linux_root_pidfd = None
            if isinstance(containment.process, _LinuxNativeHelperProcess):
                containment.process.root_pidfd = None
        except BaseException as error:
            failures.append(_IntegrityFailure("linux-root-pidfd-close", error))
    if containment.linux_namespace_pidfd is not None:
        try:
            os.close(containment.linux_namespace_pidfd)
            containment.linux_namespace_pidfd = None
        except BaseException as error:
            failures.append(_IntegrityFailure("linux-namespace-pidfd-close", error))
    if isinstance(containment.process, _LinuxNativeHelperProcess):
        failures.extend(containment.process.close_streams())
    if containment.linux_lock_held:
        if len(failures) != linux_failure_count:
            _LINUX_HELPER_POISONED = True
            _LINUX_POISONED_CONTAINMENT = containment
        containment.linux_lock_held = False
        _LINUX_HELPER_LOCK.release()
    return failures


def _format_integrity_failures(failures: list[_IntegrityFailure]) -> str:
    return "; ".join(
        f"{failure.component} [{type(failure.error).__name__}: {failure.error}]"
        for failure in failures)


def _raise_execution_integrity_failures(
        *,
        label: str,
        primary_error: BaseException | None,
        integrity_failures: list[_IntegrityFailure],
        containment_started: bool,
        quiescence_proven: bool,
) -> None:
    if primary_error is not None and containment_started:
        suffix = ""
        if integrity_failures:
            suffix = f"; secondary integrity failures: {_format_integrity_failures(integrity_failures)}"
        raise ExecutionIntegrityError(
            f"{label} primary failure [{type(primary_error).__name__}: {primary_error}]{suffix}",
            primary_error=primary_error,
            secondary_errors=integrity_failures,
            quiescence_proven=quiescence_proven,
        ) from primary_error
    if primary_error is not None:
        raise primary_error
    if integrity_failures:
        raise ExecutionIntegrityError(
            f"{label} integrity failures: {_format_integrity_failures(integrity_failures)}",
            primary_error=None,
            secondary_errors=integrity_failures,
            quiescence_proven=quiescence_proven,
        ) from integrity_failures[0].error


def _decode_subprocess_stream(
        stream: bytes | None,
        *,
        text: bool,
        encoding: str | None,
        errors: str | None,
) -> bytes | str | None:
    if stream is None or not text:
        return stream
    return stream.decode(encoding or "utf-8", errors or "strict")


def run_approved_executable(
        command: list[str],
        *,
        path: Path,
        runtime_policy: dict[str, Any],
        expected_path: str,
        expected_sha256: str,
        label: str,
        retained_fds: tuple[int, ...] = (),
        **kwargs: Any,
) -> tuple[subprocess.CompletedProcess[Any], ExecutableFileReceipt]:
    if sys.platform not in {"linux", "darwin", "win32"}:
        raise TraceError(f"{label} immutable execution is unsupported on this platform")
    if not command or command[0] != str(path):
        raise TraceError(f"{label} command path differs from external approval")
    protected_controls = {
        "creationflags", "executable", "pass_fds", "preexec_fn", "process_group", "start_new_session"}
    if protected_controls & kwargs.keys():
        raise TraceError(f"{label} execution parameters may not override immutable launch controls")
    if sys.platform == "win32" and retained_fds:
        raise TraceError(f"{label} retained descriptors are unsupported on Windows")
    if any(type(descriptor) is not int or descriptor <= 2 for descriptor in retained_fds) or (
            len(set(retained_fds)) != len(retained_fds)):
        raise TraceError(f"{label} retained descriptors are invalid")
    try:
        for retained_descriptor in retained_fds:
            os.fstat(retained_descriptor)
    except OSError as error:
        raise TraceError(f"{label} retained descriptor is invalid: {error}") from error
    timeout = kwargs.pop("timeout", None)
    check = bool(kwargs.pop("check", False))
    capture_output = bool(kwargs.pop("capture_output", False))
    text = bool(kwargs.pop("text", kwargs.pop("universal_newlines", False)))
    encoding = kwargs.pop("encoding", None)
    errors = kwargs.pop("errors", None)
    input_value = kwargs.pop("input", None)
    if encoding is not None or errors is not None:
        text = True
    if input_value is not None and "stdin" in kwargs:
        raise TraceError(f"{label} execution input conflicts with stdin")
    if capture_output and ("stdout" in kwargs or "stderr" in kwargs):
        raise TraceError(f"{label} capture_output conflicts with stdout or stderr")
    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    if input_value is not None:
        kwargs["stdin"] = subprocess.PIPE
    if isinstance(input_value, str):
        input_data = input_value.encode(encoding or "utf-8", errors or "strict")
    elif input_value is None or isinstance(input_value, bytes):
        input_data = input_value
    else:
        raise TraceError(f"{label} execution input is invalid")
    identity, descriptor = _approved_file_identity(
        path,
        install_root=Path(runtime_policy["install_root"]),
        expected_owner_uid=runtime_policy["install_owner_uid"],
        expected_path=expected_path,
        expected_sha256=expected_sha256,
        label=label,
        executable=True,
    )
    containment_helper: tuple[ExecutableFileReceipt, int] | None = None
    if sys.platform == "linux":
        helper_policy = _validate_containment_helper_policy(
            runtime_policy.get("containment_helper"),
            install_root=runtime_policy["install_root"],
            revision=runtime_policy["revision"],
            label=label,
        )
        containment_helper = _approved_file_identity(
            Path(helper_policy["path"]),
            install_root=Path(runtime_policy["install_root"]),
            expected_owner_uid=runtime_policy["install_owner_uid"],
            expected_path=helper_policy["path"],
            expected_sha256=helper_policy["sha256"],
            label=f"{label} containment helper",
            executable=True,
        )
    runtime_files: list[tuple[ExecutableFileReceipt, int]] = []
    containment = None
    result = None
    decoded_result = None
    primary_error = None
    integrity_failures: list[_IntegrityFailure] = []
    containment_started = False
    quiescence_proven = False
    try:
        for component in runtime_policy["runtime_receipt"]["components"]:
            runtime_path = Path(runtime_policy["install_root"]) / "lib" / component["filename"]
            runtime_files.append(_approved_file_identity(
                runtime_path,
                install_root=Path(runtime_policy["install_root"]),
                expected_owner_uid=runtime_policy["install_owner_uid"],
                expected_path=str(runtime_path),
                expected_sha256=component["sha256"],
                label=f"{label} runtime component {component['component']}",
                executable=False,
            ))
        verify_approved_executable_identity(path, identity, label=label)
        if containment_helper is not None:
            verify_approved_executable_identity(
                Path(containment_helper[0].path),
                containment_helper[0],
                label=f"{label} containment helper",
            )
        for runtime_identity, _runtime_descriptor in runtime_files:
            verify_approved_executable_identity(
                Path(runtime_identity.path),
                runtime_identity,
                label=f"{label} runtime component",
            )
        retained_descriptors = (
            descriptor,
            *(item[1] for item in runtime_files),
            *retained_fds,
        )
        launch = dict(kwargs)
        if sys.platform != "win32":
            launch["pass_fds"] = retained_descriptors
        if sys.platform == "linux":
            launch["executable"] = f"/proc/self/fd/{descriptor}"
            launch["_containment_helper_path"] = containment_helper[0].path
            launch["_containment_helper_descriptor"] = containment_helper[1]
        contained = _run_contained_process(
            command,
            label=label,
            timeout=timeout,
            input_data=input_data,
            launch=launch,
        )
        containment = contained.containment
        containment_started = contained.process_started
        quiescence_proven = contained.quiescence_proven
        result = contained.result
        primary_error = contained.primary_error
        integrity_failures.extend(contained.integrity_failures)
        if result is not None and check and result.returncode != 0 and primary_error is None:
            primary_error = subprocess.CalledProcessError(
                result.returncode,
                result.args,
                output=result.stdout,
                stderr=result.stderr,
            )
        try:
            descriptor_after = os.fstat(descriptor)
            if (
                    descriptor_after.st_dev,
                    descriptor_after.st_ino,
                    descriptor_after.st_size,
                    descriptor_after.st_mtime_ns,
                    descriptor_after.st_ctime_ns,
            ) != (
                    identity.device,
                    identity.inode,
                    identity.byte_count,
                    identity.modified_ns,
                    identity.changed_ns,
            ):
                raise TraceError(f"{label} descriptor identity changed during execution")
        except BaseException as error:
            integrity_failures.append(_IntegrityFailure("executable-descriptor", error))
        try:
            verify_approved_executable_identity(path, identity, label=label)
        except BaseException as error:
            integrity_failures.append(_IntegrityFailure("executable-path-root", error))
        if containment_helper is not None:
            helper_identity, helper_descriptor = containment_helper
            try:
                helper_after = os.fstat(helper_descriptor)
                if (
                        helper_after.st_dev,
                        helper_after.st_ino,
                        helper_after.st_size,
                        helper_after.st_mtime_ns,
                        helper_after.st_ctime_ns,
                ) != (
                        helper_identity.device,
                        helper_identity.inode,
                        helper_identity.byte_count,
                        helper_identity.modified_ns,
                        helper_identity.changed_ns,
                ):
                    raise TraceError(f"{label} containment helper descriptor changed during execution")
            except BaseException as error:
                integrity_failures.append(_IntegrityFailure("containment-helper-descriptor", error))
            try:
                verify_approved_executable_identity(
                    Path(helper_identity.path),
                    helper_identity,
                    label=f"{label} containment helper",
                )
            except BaseException as error:
                integrity_failures.append(_IntegrityFailure("containment-helper-path-root", error))
        for runtime_identity, runtime_descriptor in runtime_files:
            try:
                runtime_after = os.fstat(runtime_descriptor)
                if (
                        runtime_after.st_dev,
                        runtime_after.st_ino,
                        runtime_after.st_uid,
                        stat.S_IMODE(runtime_after.st_mode),
                        runtime_after.st_nlink,
                        runtime_after.st_size,
                        runtime_after.st_mtime_ns,
                        runtime_after.st_ctime_ns,
                ) != (
                        runtime_identity.device,
                        runtime_identity.inode,
                        runtime_identity.owner_uid,
                        runtime_identity.mode,
                        runtime_identity.link_count,
                        runtime_identity.byte_count,
                        runtime_identity.modified_ns,
                        runtime_identity.changed_ns,
                ):
                    raise TraceError(f"{label} runtime component descriptor changed during execution")
            except BaseException as error:
                integrity_failures.append(_IntegrityFailure(
                    f"runtime-descriptor:{runtime_identity.path}", error))
            try:
                verify_approved_executable_identity(
                    Path(runtime_identity.path),
                    runtime_identity,
                    label=f"{label} runtime component",
                )
            except BaseException as error:
                integrity_failures.append(_IntegrityFailure(
                    f"runtime-path-root:{runtime_identity.path}", error))
    except BaseException as error:
        if primary_error is None:
            primary_error = error
        else:
            integrity_failures.append(_IntegrityFailure("launcher-orchestration", error))
    finally:
        teardown_failures = []
        for _runtime_identity, runtime_descriptor in runtime_files:
            try:
                os.close(runtime_descriptor)
            except BaseException as error:
                teardown_failures.append(_IntegrityFailure("runtime-descriptor-close", error))
        if containment_helper is not None:
            try:
                os.close(containment_helper[1])
            except BaseException as error:
                teardown_failures.append(_IntegrityFailure("containment-helper-descriptor-close", error))
        try:
            os.close(descriptor)
        except BaseException as error:
            teardown_failures.append(_IntegrityFailure("executable-descriptor-close", error))
        containment_failures = _close_process_containment(
            containment,
            quiescence_proven=quiescence_proven,
        )
        teardown_failures.extend(containment_failures)
        integrity_failures.extend(teardown_failures)
        if teardown_failures:
            quiescence_proven = False
    if result is not None and primary_error is None:
        try:
            stdout = _decode_subprocess_stream(
                result.stdout, text=text, encoding=encoding, errors=errors)
            stderr = _decode_subprocess_stream(
                result.stderr, text=text, encoding=encoding, errors=errors)
            decoded_result = subprocess.CompletedProcess(
                result.args, result.returncode, stdout, stderr)
        except BaseException as error:
            primary_error = error
    _raise_execution_integrity_failures(
        label=label,
        primary_error=primary_error,
        integrity_failures=integrity_failures,
        containment_started=containment_started,
        quiescence_proven=quiescence_proven,
    )
    if decoded_result is None:
        raise TraceError(f"{label} execution did not return a result")
    return decoded_result, identity


def install_trust_evidence(
        executable: ExecutableFileReceipt,
        runtime_files: list[ExecutableFileReceipt],
        additional_files: tuple[ExecutableFileReceipt, ...] = (),
) -> dict[str, Any]:
    files = [executable, *runtime_files, *additional_files]
    if any(item.install_root != executable.install_root or item.owner_uid != executable.owner_uid for item in files):
        raise TraceError("approved install trust evidence spans multiple roots or owners")
    directories: dict[str, tuple[str, int, int, int, int]] = {}
    for item in files:
        for directory in item.path_chain:
            existing = directories.get(directory[0])
            if existing is not None and existing != directory:
                raise TraceError("approved install directory identity is inconsistent")
            directories[directory[0]] = directory
    return {
        "format": "dsv41-install-trust",
        "version": 1,
        "install_root": executable.install_root,
        "owner_uid": executable.owner_uid,
        "execution_uid": _execution_uid(),
        "directories": [
            {
                "path": item[0],
                "device": item[1],
                "inode": item[2],
                "owner_uid": item[3],
                "mode": item[4],
                "effective_write_access": False,
                "acl_entries": False,
            }
            for item in sorted(directories.values())
        ],
        "files": [
            {
                "path": item.path,
                "device": item.device,
                "inode": item.inode,
                "owner_uid": item.owner_uid,
                "mode": item.mode,
                "link_count": item.link_count,
                "byte_count": item.byte_count,
                "modified_ns": item.modified_ns,
                "changed_ns": item.changed_ns,
                "sha256": item.sha256,
                "effective_write_access": False,
                "acl_entries": False,
            }
            for item in sorted(files, key=lambda value: value.path)
        ],
    }


def install_trust_sha256(record: dict[str, Any]) -> str:
    validate_install_trust_evidence(record)
    return sha256_bytes(canonical_json(record).encode("ascii"))


def validate_install_trust_evidence(
        record: object,
        policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TraceError("install trust evidence is invalid")
    _require_exact_keys(
        record,
        {"format", "version", "install_root", "owner_uid", "execution_uid", "directories", "files"},
        "install trust evidence",
    )
    if record["format"] != "dsv41-install-trust" or record["version"] != 1:
        raise TraceError("install trust evidence version is invalid")
    install_root = _approval_path(record["install_root"], "install trust root")
    owner_uid = record["owner_uid"]
    execution_uid = record["execution_uid"]
    if type(owner_uid) is not int or owner_uid < 0 or type(execution_uid) is not int or (
            execution_uid <= 0) or execution_uid == owner_uid:
        raise TraceError("install trust owner or execution identity is invalid")
    directories = record["directories"]
    files = record["files"]
    if not isinstance(directories, list) or not directories or not isinstance(files, list) or not files:
        raise TraceError("install trust evidence is incomplete")
    previous_path = None
    directory_paths = set()
    for directory in directories:
        if not isinstance(directory, dict):
            raise TraceError("install trust directory evidence is invalid")
        _require_exact_keys(
            directory,
            {
                "path", "device", "inode", "owner_uid", "mode",
                "effective_write_access", "acl_entries",
            },
            "install trust directory evidence",
        )
        path = _approval_path(directory["path"], "install trust directory")
        if path in directory_paths or (previous_path is not None and path <= previous_path):
            raise TraceError("install trust directories are duplicated or unsorted")
        directory_paths.add(path)
        previous_path = path
        if any(type(directory[key]) is not int or directory[key] < 0 for key in (
                "device", "inode", "owner_uid", "mode")) or (
                directory["mode"] & 0o022) or directory["effective_write_access"] is not False or (
                directory["acl_entries"] is not False):
            raise TraceError("install trust directory is mutable or malformed")
        if directory["owner_uid"] not in {0, owner_uid}:
            raise TraceError("install trust directory owner is not trusted")
        if (path == install_root or PurePosixPath(install_root) in PurePosixPath(path).parents) and (
                directory["owner_uid"] != owner_uid):
            raise TraceError("install trust tree owner differs from approval")
    previous_path = None
    file_paths = set()
    for file_record in files:
        if not isinstance(file_record, dict):
            raise TraceError("install trust file evidence is invalid")
        _require_exact_keys(
            file_record,
            {
                "path", "device", "inode", "owner_uid", "mode", "link_count", "byte_count",
                "modified_ns", "changed_ns", "sha256", "effective_write_access", "acl_entries",
            },
            "install trust file evidence",
        )
        path = _approval_path(file_record["path"], "install trust file")
        if path in file_paths or (previous_path is not None and path <= previous_path):
            raise TraceError("install trust files are duplicated or unsorted")
        file_paths.add(path)
        previous_path = path
        if any(type(file_record[key]) is not int or file_record[key] < 0 for key in (
                "device", "inode", "owner_uid", "mode", "link_count", "byte_count",
                "modified_ns", "changed_ns")) or file_record["owner_uid"] != owner_uid or (
                file_record["mode"] & 0o222) or file_record["link_count"] != 1 or (
                file_record["effective_write_access"] is not False) or file_record["acl_entries"] is not False or (
                re.fullmatch(r"[0-9a-f]{64}", file_record.get("sha256", "")) is None):
            raise TraceError("install trust file is mutable or malformed")
    if policy is not None:
        if install_root != policy["install_root"] or owner_uid != policy["install_owner_uid"]:
            raise TraceError("install trust root differs from external approval")
        expected_files = {
            policy["executable_path"]: policy["executable_sha256"],
            **{
                f"{install_root}/lib/{component['filename']}": component["sha256"]
                for component in policy["runtime_receipt"]["components"]
            },
        }
        if "containment_helper" in policy:
            helper = _validate_containment_helper_policy(
                policy["containment_helper"],
                install_root=install_root,
                revision=policy["revision"],
                label="external approval",
            )
            expected_files[helper["path"]] = helper["sha256"]
        observed_files = {item["path"]: item["sha256"] for item in files}
        if observed_files != expected_files:
            raise TraceError("install trust files differ from external approval")
        expected_directories = set()
        for path in expected_files:
            current = PurePosixPath(path).parent
            while True:
                expected_directories.add(str(current))
                if str(current) == "/":
                    break
                current = current.parent
        if directory_paths != expected_directories:
            raise TraceError("install trust directory coverage is incomplete")
    return record


def validate_runtime_build_evidence(
        record: object,
        policy: dict[str, Any],
        *,
        label: str,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TraceError(f"{label} runtime build evidence is invalid")
    _require_exact_keys(
        record,
        {
            "revision",
            "path",
            "sha256",
            "runtime_profile",
            "runtime_receipt_sha256",
            "runtime_libraries",
            "runtime_libraries_post",
        },
        f"{label} runtime build evidence",
    )
    receipt_sha256 = sha256_bytes(canonical_json(policy["runtime_receipt"]).encode("ascii"))
    expected = {
        "revision": policy["revision"],
        "path": policy["executable_path"],
        "sha256": policy["executable_sha256"],
        "runtime_profile": policy["runtime_profile"],
        "runtime_receipt_sha256": receipt_sha256,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise TraceError(f"{label} runtime build {key} differs from external approval")
    libraries = record["runtime_libraries"]
    if libraries != record["runtime_libraries_post"] or not isinstance(libraries, list):
        raise TraceError(f"{label} loaded runtime closure changed during execution")
    expected_libraries = []
    for component in policy["runtime_receipt"]["components"]:
        name = component["component"]
        expected_libraries.append({
            "component": name,
            "filename": component["filename"],
            "path": f"{policy['install_root']}/lib/{component['filename']}",
            "sha256": component["sha256"],
            "role": {
                "llama-common": "build-info",
                "llama": "llama",
                "ggml-base": "ggml",
            }.get(name, f"runtime:{name}"),
            "revision": component["revision"],
        })
    expected_libraries.sort(key=lambda item: item["path"])
    if libraries != expected_libraries:
        raise TraceError(f"{label} loaded runtime libraries differ from external approval")
    return record


def runtime_build_evidence_sha256(record: object, policy: dict[str, Any], *, label: str) -> str:
    validated = validate_runtime_build_evidence(record, policy, label=label)
    return sha256_bytes(canonical_json(validated).encode("ascii"))


def reject_loader_overrides(environment: dict[str, str] | None = None) -> None:
    values = os.environ if environment is None else environment
    active = sorted(name for name in FORBIDDEN_LOADER_ENVIRONMENT if values.get(name))
    if active:
        raise TraceError("production trace execution forbids loader overrides: " + ", ".join(active))


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def strict_json_loads(data: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise TraceError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise TraceError(f"invalid JSON constant: {value}")

    try:
        return json.loads(
            data,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise TraceError(f"invalid JSON: {error}") from error


def trusted_ssh_keygen_path() -> Path:
    if sys.platform == "win32":
        return Path(r"C:\Windows\System32\OpenSSH\ssh-keygen.exe")
    if sys.platform in ("darwin", "linux"):
        return Path("/usr/bin/ssh-keygen")
    raise TraceError(f"unsupported platform for trace signature verification: {sys.platform}")


def _validate_ssh_keygen(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise TraceError("trusted ssh-keygen path must be an absolute non-symlink")
    try:
        record = path.stat(follow_symlinks=False)
    except OSError as error:
        raise TraceError(f"cannot inspect trusted ssh-keygen: {error}") from error
    _immutable_path_chain(
        path,
        install_root=path.parent.parent,
        expected_owner_uid=0,
        label="trusted ssh-keygen",
    )
    if not stat.S_ISREG(record.st_mode) or record.st_uid != 0 or record.st_nlink != 1 or (
            stat.S_IMODE(record.st_mode) & 0o022) or _path_is_writable_by_execution_identity(path) or (
            _has_access_control_entries(path)) or not (stat.S_IMODE(record.st_mode) & 0o111):
        raise TraceError("trusted ssh-keygen is not an executable regular file")
    try:
        result = subprocess.run(
            [str(path), "-Y", "verify"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise TraceError(f"cannot probe trusted ssh-keygen: {error}") from error
    diagnostic = (result.stdout + result.stderr).lower()
    if result.returncode == 0 or any(
            marker in diagnostic for marker in ("unknown option", "illegal option", "unknown operation")):
        raise TraceError("trusted ssh-keygen lacks required -Y signature support")
    return path


def _ssh_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("SSH_AUTH_SOCK", None)
    environment.pop("SSH_AGENT_PID", None)
    return environment


def _validate_principal(principal: str) -> None:
    if not isinstance(principal, str) or re.fullmatch(r"[A-Za-z0-9._@+-]{1,128}", principal) is None:
        raise TraceError("trace signer principal is invalid")


def _signer_policy(
        trusted_signers: dict[str, dict[str, str]],
        principal: str) -> dict[str, str]:
    _validate_principal(principal)
    policy = trusted_signers.get(principal)
    if not isinstance(policy, dict) or set(policy) != {
            "public_key", "lane", "runtime", "runtime_profile"}:
        raise TraceError(f"trace signer principal is not approved: {principal}")
    expected = {
        CANDIDATE_LANE: ("llama.cpp", "sibling-lib"),
        ORACLE_LANE: ("ds4", "apple-metal"),
    }.get(policy.get("lane"))
    if expected is None or (policy.get("runtime"), policy.get("runtime_profile")) != expected:
        raise TraceError("trace signer policy is invalid")
    _normalize_public_key(policy.get("public_key", ""))
    return policy


def _normalize_public_key(public_key: str, *, allow_comment: bool = False) -> str:
    fields = public_key.strip().split()
    expected_fields = len(fields) >= 2 if allow_comment else len(fields) == 2
    if not expected_fields or fields[0] != "ssh-ed25519" or re.fullmatch(
            r"[A-Za-z0-9+/]+={0,2}", fields[1]) is None:
        raise TraceError("trace signer public key must be an OpenSSH Ed25519 key")
    return " ".join(fields[:2])


def _approval_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("/") or value.startswith("//") or (
            ".." in PurePosixPath(value).parts or str(PurePosixPath(value)) != value):
        raise TraceError(f"{label} is not an absolute canonical path")
    return value


def _approval_digest(kind: str, approval_id: str, policy: dict[str, Any]) -> str:
    if re.fullmatch(r"[A-Za-z0-9._-]{1,128}", approval_id) is None:
        raise TraceError(f"{kind} approval ID is invalid")
    record = {
        "format": EXECUTABLE_APPROVAL_FORMAT,
        "version": EXECUTABLE_APPROVAL_VERSION,
        "kind": kind,
        "id": approval_id,
        "policy": policy,
    }
    return sha256_bytes(canonical_json(record).encode("ascii"))


def _validate_containment_helper_policy(
        record: object,
        *,
        install_root: str,
        revision: str,
        label: str,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TraceError(f"{label} containment helper receipt is missing")
    _require_exact_keys(
        record,
        {
            "format",
            "version",
            "revision",
            "filename",
            "sha256",
            "launcher_policy",
            "supplementary_groups",
        },
        f"{label} containment helper receipt",
    )
    if record["format"] != "dsv41-containment-helper" or record["version"] != 2 or (
            record["revision"] != revision) or (
            record["filename"] != "llama-deepseek-v41-containment-helper") or re.fullmatch(
                r"[0-9a-f]{64}", record.get("sha256", "")) is None or (
            record["launcher_policy"] != "zero-supplementary-groups-v1") or (
            record["supplementary_groups"] != []):
        raise TraceError(f"{label} containment helper receipt is invalid")
    result = dict(record)
    result["path"] = f"{install_root}/bin/{record['filename']}"
    return result


def approved_containment_helper_identity(
        policy: dict[str, Any],
        *,
        label: str,
) -> ExecutableFileReceipt:
    helper = _validate_containment_helper_policy(
        policy.get("containment_helper"),
        install_root=policy["install_root"],
        revision=policy["revision"],
        label=label,
    )
    return approved_executable_identity(
        Path(helper["path"]),
        install_root=policy["install_root"],
        expected_owner_uid=policy["install_owner_uid"],
        expected_path=helper["path"],
        expected_sha256=helper["sha256"],
        label=f"{label} containment helper",
    )


def candidate_exporter_approval(
        approval_id: str,
        *,
        policies: dict[str, dict[str, Any]] = APPROVED_CANDIDATE_EXPORTERS,
) -> tuple[dict[str, Any], str]:
    policy = policies.get(approval_id)
    if not isinstance(policy, dict):
        raise TraceError(f"candidate exporter approval is not trusted: {approval_id}")
    _require_exact_keys(
        policy,
        {
            "runtime",
            "repository",
            "revision",
            "base_revision",
            "diff_sha256",
            "install_root",
            "install_owner_uid",
            "executable_path",
            "executable_sha256",
            "containment_helper",
            "runtime_profile",
            "runtime_receipt",
        },
        "candidate exporter approval",
    )
    if policy["runtime"] != "llama.cpp" or policy["repository"] != REPOSITORY:
        raise TraceError("candidate exporter approval runtime identity is invalid")
    for key in ("revision", "base_revision"):
        if re.fullmatch(r"[0-9a-f]{40}", policy.get(key, "")) is None:
            raise TraceError(f"candidate exporter approval {key} is invalid")
    for key in ("diff_sha256", "executable_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", policy.get(key, "")) is None:
            raise TraceError(f"candidate exporter approval {key} is invalid")
    install_root = _approval_path(policy["install_root"], "candidate exporter approval install root")
    if type(policy["install_owner_uid"]) is not int or policy["install_owner_uid"] < 0:
        raise TraceError("candidate exporter approval install owner is invalid")
    executable_path = _approval_path(
        policy["executable_path"], "candidate exporter approval executable path")
    if executable_path != f"{install_root}/bin/llama-deepseek-v41-trace":
        raise TraceError("candidate exporter approval executable path is outside its install policy")
    _validate_containment_helper_policy(
        policy["containment_helper"],
        install_root=install_root,
        revision=policy["revision"],
        label="candidate exporter approval",
    )
    profile = policy["runtime_profile"]
    if not isinstance(profile, dict):
        raise TraceError("candidate exporter approval runtime profile is invalid")
    _require_exact_keys(
        profile, {"name", "components", "selected_backend_component"},
        "candidate exporter approval runtime profile")
    if profile["name"] != "sibling-lib" or profile["selected_backend_component"] != "ggml-hip":
        raise TraceError("candidate exporter approval runtime profile is invalid")
    components = profile["components"]
    if not isinstance(components, list) or components != sorted(components) or (
            len(components) != len(set(components))) or any(
                not isinstance(component, str) or re.fullmatch(r"[a-z0-9-]+", component) is None
                for component in components):
        raise TraceError("candidate exporter approval components are invalid")
    if not {"llama-common", "llama", "ggml", "ggml-base", "ggml-hip"}.issubset(set(components)):
        raise TraceError("candidate exporter approval is missing required components")
    receipt = policy["runtime_receipt"]
    if not isinstance(receipt, dict):
        raise TraceError("candidate exporter approval runtime receipt is invalid")
    _require_exact_keys(
        receipt, {"format", "version", "revision", "profile", "components"},
        "candidate exporter approval runtime receipt")
    if receipt["format"] != "dsv41-runtime-receipt" or receipt["version"] != 1 or (
            receipt["revision"] != policy["revision"]) or receipt["profile"] != profile["name"]:
        raise TraceError("candidate exporter approval runtime receipt identity is invalid")
    receipt_components = receipt["components"]
    if not isinstance(receipt_components, list) or receipt_components != sorted(
            receipt_components, key=lambda item: item.get("component", "") if isinstance(item, dict) else ""):
        raise TraceError("candidate exporter approval runtime receipt components are not canonical")
    seen_components = set()
    seen_filenames = set()
    seen_digests = set()
    for component in receipt_components:
        if not isinstance(component, dict):
            raise TraceError("candidate exporter approval runtime receipt component is invalid")
        _require_exact_keys(
            component, {"component", "filename", "sha256", "revision"},
            "candidate exporter approval runtime receipt component")
        name = component["component"]
        filename = component["filename"]
        digest = component["sha256"]
        revision = component["revision"]
        if name not in components or name in seen_components:
            raise TraceError("candidate exporter approval runtime receipt component name is invalid")
        if not isinstance(filename, str) or re.fullmatch(r"[A-Za-z0-9._+-]+", filename) is None or (
                filename in seen_filenames):
            raise TraceError("candidate exporter approval runtime receipt filename is invalid")
        if re.fullmatch(r"[0-9a-f]{64}", digest or "") is None or digest in seen_digests:
            raise TraceError("candidate exporter approval runtime receipt digest is invalid")
        revision_bearing = name in {"llama-common", "ggml-base"}
        if (revision_bearing and revision != policy["revision"]) or (
                not revision_bearing and revision is not None):
            raise TraceError("candidate exporter approval runtime receipt revision is invalid")
        seen_components.add(name)
        seen_filenames.add(filename)
        seen_digests.add(digest)
    if seen_components != set(components):
        raise TraceError("candidate exporter approval receipt differs from its runtime profile")
    return policy, _approval_digest("candidate-exporter", approval_id, policy)


def ds4_exporter_approval(
        approval_id: str,
        *,
        policies: dict[str, dict[str, Any]] = APPROVED_DS4_EXPORTERS,
) -> tuple[dict[str, Any], str]:
    policy = policies.get(approval_id)
    if not isinstance(policy, dict):
        raise TraceError(f"ds4 exporter approval is not trusted: {approval_id}")
    _require_exact_keys(
        policy,
        {
            "runtime",
            "repository",
            "revision",
            "install_root",
            "install_owner_uid",
            "executable_path",
            "executable_sha256",
            "containment_helper",
            "runtime_profile",
            "runtime_receipt",
        },
        "ds4 exporter approval",
    )
    if policy["runtime"] != "ds4" or policy["repository"] != DS4_REPOSITORY or (
            policy["revision"] != DS4_REVISION):
        raise TraceError("ds4 exporter approval runtime identity is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", policy.get("executable_sha256", "")) is None:
        raise TraceError("ds4 exporter approval executable SHA-256 is invalid")
    install_root = _approval_path(policy["install_root"], "ds4 exporter approval install root")
    if type(policy["install_owner_uid"]) is not int or policy["install_owner_uid"] < 0:
        raise TraceError("ds4 exporter approval install owner is invalid")
    executable_path = _approval_path(
        policy["executable_path"], "ds4 exporter approval executable path")
    executable = PurePosixPath(executable_path)
    if executable.parent != PurePosixPath(install_root) / "bin":
        raise TraceError("ds4 exporter approval executable path is outside its install policy")
    _validate_containment_helper_policy(
        policy["containment_helper"],
        install_root=install_root,
        revision=policy["revision"],
        label="ds4 exporter approval",
    )
    profile = policy["runtime_profile"]
    if not isinstance(profile, dict):
        raise TraceError("ds4 exporter approval runtime profile is invalid")
    _require_exact_keys(
        profile, {"name", "components", "selected_backend_component"},
        "ds4 exporter approval runtime profile")
    components = profile["components"]
    selected_backend = profile["selected_backend_component"]
    if profile["name"] not in {"co-located", "sibling-lib"} or not isinstance(components, list) or (
            components != sorted(components)) or len(components) != len(set(components)) or not components or any(
                not isinstance(component, str) or re.fullmatch(r"[a-z0-9-]+", component) is None
                for component in components) or not isinstance(selected_backend, str) or (
                selected_backend not in components):
        raise TraceError("ds4 exporter approval runtime profile is invalid")
    receipt = policy["runtime_receipt"]
    if not isinstance(receipt, dict):
        raise TraceError("ds4 exporter approval runtime receipt is invalid")
    _require_exact_keys(
        receipt, {"format", "version", "revision", "profile", "components"},
        "ds4 exporter approval runtime receipt")
    if receipt["format"] != "dsv41-runtime-receipt" or receipt["version"] != 1 or (
            receipt["revision"] != DS4_REVISION) or receipt["profile"] != profile["name"]:
        raise TraceError("ds4 exporter approval runtime receipt identity is invalid")
    receipt_components = receipt["components"]
    if not isinstance(receipt_components, list) or receipt_components != sorted(
            receipt_components, key=lambda item: item.get("component", "") if isinstance(item, dict) else ""):
        raise TraceError("ds4 exporter approval runtime receipt components are not canonical")
    seen_components = set()
    seen_filenames = set()
    seen_digests = set()
    revision_bearing = 0
    for component in receipt_components:
        if not isinstance(component, dict):
            raise TraceError("ds4 exporter approval runtime receipt component is invalid")
        _require_exact_keys(
            component, {"component", "filename", "sha256", "revision"},
            "ds4 exporter approval runtime receipt component")
        name = component["component"]
        filename = component["filename"]
        digest = component["sha256"]
        revision = component["revision"]
        if name not in components or name in seen_components:
            raise TraceError("ds4 exporter approval runtime receipt component name is invalid")
        if not isinstance(filename, str) or re.fullmatch(r"[A-Za-z0-9._+-]+", filename) is None or (
                filename in seen_filenames):
            raise TraceError("ds4 exporter approval runtime receipt filename is invalid")
        if re.fullmatch(r"[0-9a-f]{64}", digest or "") is None or digest in seen_digests:
            raise TraceError("ds4 exporter approval runtime receipt digest is invalid")
        if revision not in {None, DS4_REVISION}:
            raise TraceError("ds4 exporter approval runtime receipt revision is invalid")
        revision_bearing += revision == DS4_REVISION
        seen_components.add(name)
        seen_filenames.add(filename)
        seen_digests.add(digest)
    if seen_components != set(components) or revision_bearing == 0:
        raise TraceError("ds4 exporter approval receipt differs from its runtime profile")
    return policy, _approval_digest("ds4-exporter", approval_id, policy)


def validate_tokenizer_policy(record: object) -> dict[str, bool]:
    if not isinstance(record, dict):
        raise TraceError("tokenizer policy is missing")
    _require_exact_keys(
        record,
        {
            "add_bos",
            "parse_special",
            "detokenize_special",
            "remove_leading_bos_before_detokenize",
            "require_round_trip",
        },
        "tokenizer policy",
    )
    if any(type(value) is not bool for value in record.values()):
        raise TraceError("tokenizer policy values must be explicit booleans")
    if record["parse_special"] is not True or record["detokenize_special"] is not True or (
            record["require_round_trip"] is not True) or (
            record["remove_leading_bos_before_detokenize"] != record["add_bos"]):
        raise TraceError("tokenizer policy is not the exact prompt construction policy")
    return dict(record)


def tokenizer_policy_sha256(record: object) -> str:
    return sha256_bytes(canonical_json(validate_tokenizer_policy(record)).encode("ascii"))


def prompt_builder_approval(
        approval_id: str,
        *,
        policies: dict[str, dict[str, Any]] = APPROVED_PROMPT_BUILDERS,
) -> tuple[dict[str, Any], str]:
    policy = policies.get(approval_id)
    if not isinstance(policy, dict):
        raise TraceError(f"prompt builder approval is not trusted: {approval_id}")
    _require_exact_keys(
        policy,
        {
            "runtime",
            "runtime_profile",
            "repository",
            "revision",
            "install_root",
            "install_owner_uid",
            "executable_path",
            "executable_sha256",
            "containment_helper",
            "source_root",
            "runtime_receipt",
            "model_sha256",
            "corpora",
            "tokenizer",
            "prompts",
        },
        "prompt builder approval",
    )
    if policy["runtime"] != "llama.cpp" or (
            policy["repository"] != REPOSITORY) or re.fullmatch(
                r"[0-9a-f]{40}", policy.get("revision", "")) is None:
        raise TraceError("prompt builder approval runtime identity is invalid")
    if policy["model_sha256"] != MODEL_SHA256 or re.fullmatch(
            r"[0-9a-f]{64}", policy.get("executable_sha256", "")) is None:
        raise TraceError("prompt builder approval executable or model identity is invalid")
    install_root = _approval_path(policy["install_root"], "prompt builder approval install root")
    if type(policy["install_owner_uid"]) is not int or policy["install_owner_uid"] < 0:
        raise TraceError("prompt builder approval install owner is invalid")
    executable_path = _approval_path(
        policy["executable_path"], "prompt builder approval executable path")
    source_root = _approval_path(policy["source_root"], "prompt builder approval source root")
    if executable_path != f"{install_root}/bin/llama-deepseek-v41-prompt-builder":
        raise TraceError("prompt builder approval executable path is outside its install policy")
    _validate_containment_helper_policy(
        policy["containment_helper"],
        install_root=install_root,
        revision=policy["revision"],
        label="prompt builder approval",
    )
    profile = policy["runtime_profile"]
    if not isinstance(profile, dict):
        raise TraceError("prompt builder approval runtime profile is invalid")
    _require_exact_keys(
        profile, {"name", "components", "selected_backend_component"},
        "prompt builder approval runtime profile")
    if profile["name"] != "sibling-lib" or not isinstance(profile["components"], list) or (
            profile["components"] != sorted(profile["components"])) or len(
                profile["components"]) != len(set(profile["components"])) or not {
                    "llama-common", "llama", "ggml", "ggml-base"
                }.issubset(set(profile["components"])) or (
                profile["selected_backend_component"] != "ggml-hip") or (
                profile["selected_backend_component"] not in profile["components"]):
        raise TraceError("prompt builder approval runtime profile is invalid")
    receipt = policy["runtime_receipt"]
    if not isinstance(receipt, dict):
        raise TraceError("prompt builder approval runtime receipt is invalid")
    _require_exact_keys(
        receipt, {"format", "version", "revision", "profile", "components"},
        "prompt builder approval runtime receipt")
    if receipt["format"] != "dsv41-runtime-receipt" or receipt["version"] != 1 or (
            receipt["revision"] != policy["revision"]) or receipt["profile"] != profile["name"]:
        raise TraceError("prompt builder approval runtime receipt identity is invalid")
    receipt_components = receipt["components"]
    if not isinstance(receipt_components, list) or receipt_components != sorted(
            receipt_components, key=lambda item: item.get("component", "") if isinstance(item, dict) else ""):
        raise TraceError("prompt builder approval runtime receipt components are not canonical")
    seen_components = set()
    seen_filenames = set()
    seen_digests = set()
    for component in receipt_components:
        if not isinstance(component, dict):
            raise TraceError("prompt builder approval runtime receipt component is invalid")
        _require_exact_keys(
            component, {"component", "filename", "sha256", "revision"},
            "prompt builder approval runtime receipt component")
        name = component["component"]
        filename = component["filename"]
        digest = component["sha256"]
        revision = component["revision"]
        if name not in profile["components"] or name in seen_components:
            raise TraceError("prompt builder approval runtime receipt component name is invalid")
        if not isinstance(filename, str) or re.fullmatch(r"[A-Za-z0-9._+-]+", filename) is None or (
                filename in seen_filenames):
            raise TraceError("prompt builder approval runtime receipt filename is invalid")
        if re.fullmatch(r"[0-9a-f]{64}", digest or "") is None or digest in seen_digests:
            raise TraceError("prompt builder approval runtime receipt digest is invalid")
        revision_bearing = name in {"llama-common", "ggml-base"}
        if (revision_bearing and revision != policy["revision"]) or (
                not revision_bearing and revision is not None):
            raise TraceError("prompt builder approval runtime receipt revision is invalid")
        seen_components.add(name)
        seen_filenames.add(filename)
        seen_digests.add(digest)
    if seen_components != set(profile["components"]):
        raise TraceError("prompt builder approval receipt differs from its runtime profile")
    if policy["corpora"] != CORPUS_SHA256:
        raise TraceError("prompt builder approval corpus policy is invalid")
    validate_tokenizer_policy(policy["tokenizer"])
    prompts = policy["prompts"]
    if not isinstance(prompts, list) or not prompts:
        raise TraceError("prompt builder approval prompt policy is empty")
    previous_key = None
    seen_keys = set()
    for prompt in prompts:
        if not isinstance(prompt, dict):
            raise TraceError("prompt builder approval prompt record is invalid")
        _require_exact_keys(
            prompt,
            {
                "corpus_name", "corpus_sha256", "context", "decode_steps", "target_tokens",
                "prompt_sha256", "prompt_byte_count",
            },
            "prompt builder approval prompt record",
        )
        corpus_name = prompt["corpus_name"]
        if corpus_name not in CORPUS_SHA256 or prompt["corpus_sha256"] != CORPUS_SHA256[corpus_name]:
            raise TraceError("prompt builder approval prompt corpus identity is invalid")
        context = prompt["context"]
        decode_steps = prompt["decode_steps"]
        target_tokens = prompt["target_tokens"]
        if type(context) is not int or context < 2 or type(decode_steps) is not int or decode_steps < 1 or (
                target_tokens != context - decode_steps):
            raise TraceError("prompt builder approval prompt configuration is invalid")
        if re.fullmatch(r"[0-9a-f]{64}", prompt.get("prompt_sha256", "")) is None or (
                type(prompt.get("prompt_byte_count")) is not int or prompt["prompt_byte_count"] <= 0):
            raise TraceError("prompt builder approval prompt output identity is invalid")
        key = (corpus_name, context, decode_steps)
        if key in seen_keys or (previous_key is not None and key <= previous_key):
            raise TraceError("prompt builder approval prompt records are duplicated or unsorted")
        seen_keys.add(key)
        previous_key = key
    return policy, _approval_digest("prompt-builder", approval_id, policy)


def approved_prompt_record(
        policy: dict[str, Any],
        *,
        corpus_name: str,
        context: int,
        decode_steps: int,
) -> dict[str, Any]:
    matches = [
        prompt for prompt in policy["prompts"]
        if prompt["corpus_name"] == corpus_name and
        prompt["context"] == context and
        prompt["decode_steps"] == decode_steps
    ]
    if len(matches) != 1:
        raise TraceError("prompt builder approval does not contain the requested prompt configuration")
    return matches[0]


def _read_external_regular_file(
        path: Path,
        label: str,
        *,
        trusted_root: Path,
        expected_owner_uid: int,
        test_only_trust: bool = False,
) -> bytes:
    if not path.is_absolute() or str(path.resolve()) != str(path):
        raise TraceError(f"{label} path must be absolute, canonical, and non-symlinked")
    if path != trusted_root and trusted_root not in path.parents:
        raise TraceError(f"{label} is outside its trusted root")
    if not test_only_trust:
        _immutable_path_chain(
            path,
            install_root=trusted_root,
            expected_owner_uid=expected_owner_uid,
            label=label,
        )
    try:
        before = path.stat(follow_symlinks=False)
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
    except OSError as error:
        raise TraceError(f"cannot inspect {label}: {error}") from error
    identity = (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    if identity != (
            opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) or (
            not stat.S_ISREG(opened.st_mode)) or opened.st_nlink != 1 or (
            not test_only_trust and (
                opened.st_uid != expected_owner_uid or stat.S_IMODE(opened.st_mode) & 0o222 or
                _path_is_writable_by_execution_identity(path) or _has_access_control_entries(path))):
        os.close(descriptor)
        raise TraceError(f"{label} must be an immutable trusted-owned one-link regular file")
    try:
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            data = stream.read()
        after = os.fstat(descriptor)
        path_after = path.stat(follow_symlinks=False)
    except OSError as error:
        os.close(descriptor)
        raise TraceError(f"cannot read {label}: {error}") from error
    if identity != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) or (
            identity != (
                path_after.st_dev, path_after.st_ino, path_after.st_size,
                path_after.st_mtime_ns, path_after.st_ctime_ns)) or len(data) != before.st_size:
        os.close(descriptor)
        raise TraceError(f"{label} changed while reading")
    os.close(descriptor)
    return data


def load_executable_approval_policy(
        policy_path: Path,
        signature_path: Path,
        *,
        expected_principal: str,
        trusted_approvers: dict[str, dict[str, Any]] = APPROVED_EXECUTABLE_APPROVERS,
        ssh_keygen: Path | None = None,
        forbidden_roots: Iterable[Path] = (),
        test_only_trust: bool = False,
) -> ExecutableApprovalPolicy:
    _validate_principal(expected_principal)
    resolved_policy = policy_path.resolve()
    resolved_signature = signature_path.resolve()
    for forbidden_root in forbidden_roots:
        root = forbidden_root.resolve()
        if resolved_policy == root or root in resolved_policy.parents or (
                resolved_signature == root) or root in resolved_signature.parents:
            raise TraceError("executable approval policy and signature must be outside protected output roots")
    approver = trusted_approvers.get(expected_principal)
    if not isinstance(approver, dict) or set(approver) != {
            "public_key", "policy_root", "owner_uid"}:
        raise TraceError(f"executable approval principal is not trusted: {expected_principal}")
    policy_root = Path(_approval_path(approver["policy_root"], "executable approval trusted root"))
    owner_uid = approver["owner_uid"]
    if type(owner_uid) is not int or owner_uid < 0:
        raise TraceError("executable approval trusted owner is invalid")
    approved_key = _normalize_public_key(approver["public_key"])
    policy_bytes = _read_external_regular_file(
        policy_path,
        "executable approval policy",
        trusted_root=policy_root,
        expected_owner_uid=owner_uid,
        test_only_trust=test_only_trust,
    )
    signature_bytes = _read_external_regular_file(
        signature_path,
        "executable approval signature",
        trusted_root=policy_root,
        expected_owner_uid=owner_uid,
        test_only_trust=test_only_trust,
    )
    try:
        policy = strict_json_loads(policy_bytes.decode("ascii"))
        signature = signature_bytes.decode("ascii")
    except UnicodeError as error:
        raise TraceError("executable approval policy and signature must be ASCII") from error
    if policy_bytes != _canonical_json_bytes(policy):
        raise TraceError("executable approval policy is not canonical")
    if not isinstance(policy, dict):
        raise TraceError("executable approval policy must be a JSON object")
    _require_exact_keys(
        policy,
        {
            "format", "version", "principal", "verifier_repository", "verifier_revision",
            "candidate_exporters", "ds4_exporters", "prompt_builders",
        },
        "executable approval policy",
    )
    if policy["format"] != EXECUTABLE_APPROVAL_FORMAT or (
            policy["version"] != EXECUTABLE_APPROVAL_VERSION) or (
            policy["principal"] != expected_principal) or (
            policy["verifier_repository"] != REPOSITORY) or re.fullmatch(
                r"[0-9a-f]{40}", policy.get("verifier_revision", "")) is None:
        raise TraceError("executable approval policy identity is invalid")
    candidate_exporters = policy["candidate_exporters"]
    ds4_exporters = policy["ds4_exporters"]
    prompt_builders = policy["prompt_builders"]
    if not isinstance(candidate_exporters, dict) or not isinstance(ds4_exporters, dict) or (
            not isinstance(prompt_builders, dict)):
        raise TraceError("executable approval policy maps are invalid")
    for approval_id in sorted(candidate_exporters):
        candidate_exporter_approval(approval_id, policies=candidate_exporters)
    for approval_id in sorted(ds4_exporters):
        ds4_policy, _digest = ds4_exporter_approval(approval_id, policies=ds4_exporters)
        if ds4_policy["revision"] == policy["verifier_revision"]:
            raise TraceError("ds4 exporter producer revision must differ from the verifier revision")
    for approval_id in sorted(prompt_builders):
        prompt_builder_approval(approval_id, policies=prompt_builders)
    if not signature.startswith("-----BEGIN SSH SIGNATURE-----\n") or not signature.endswith(
            "-----END SSH SIGNATURE-----\n"):
        raise TraceError("executable approval signature is invalid")
    executable = _validate_ssh_keygen(ssh_keygen or trusted_ssh_keygen_path())
    with tempfile.TemporaryDirectory(prefix="dsv41-approval-verify-") as temp:
        temporary = Path(temp)
        allowed_signers = temporary / "allowed_signers"
        signature_file = temporary / "signature"
        allowed_signers.write_text(f"{expected_principal} {approved_key}\n", encoding="ascii")
        signature_file.write_text(signature, encoding="ascii")
        try:
            result = subprocess.run(
                [
                    str(executable),
                    "-Y", "verify",
                    "-f", str(allowed_signers),
                    "-I", expected_principal,
                    "-n", EXECUTABLE_APPROVAL_NAMESPACE,
                    "-s", str(signature_file),
                ],
                input=policy_bytes,
                check=False,
                capture_output=True,
                timeout=30,
                env=_ssh_environment(),
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise TraceError(f"cannot verify executable approval signature: {error}") from error
    if result.returncode != 0:
        raise TraceError("executable approval signature verification failed")
    return ExecutableApprovalPolicy(
        principal=expected_principal,
        verifier_revision=policy["verifier_revision"],
        candidate_exporters=candidate_exporters,
        ds4_exporters=ds4_exporters,
        prompt_builders=prompt_builders,
        sha256=sha256_bytes(policy_bytes),
    )


def approval_binding(
        kind: str,
        approval_id: str,
        digest: str,
        trust_sha256: str,
) -> dict[str, str]:
    if kind not in {"candidate_exporter", "ds4_exporter", "prompt_builder"} or re.fullmatch(
            r"[A-Za-z0-9._-]{1,128}", approval_id) is None or re.fullmatch(
                r"[0-9a-f]{64}", digest) is None or re.fullmatch(
                r"[0-9a-f]{64}", trust_sha256) is None:
        raise TraceError("execution approval binding is invalid")
    return {"id": approval_id, "sha256": digest, "install_trust_sha256": trust_sha256}


def validate_execution_authorization(
        manifest: dict[str, Any],
        *,
        policy: dict[str, str],
        expected_lane: str,
        expected_challenge: str,
        expected_run_id: str,
        verification_unix: int,
        candidate_exporter_policies: dict[str, dict[str, Any]],
        ds4_exporter_policies: dict[str, dict[str, Any]],
        prompt_builder_policies: dict[str, dict[str, Any]],
        expected_candidate_exporter_policy_id: str | None,
        expected_ds4_exporter_policy_id: str | None,
        expected_prompt_builder_policy_id: str,
        expected_approval_policy_sha256: str,
        expected_verifier_revision: str,
        seen_run_ids: set[str] | None = None) -> None:
    if expected_lane not in {CANDIDATE_LANE, ORACLE_LANE}:
        raise TraceError("externally expected execution lane is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", expected_challenge) is None:
        raise TraceError("externally expected execution challenge is invalid")
    run_prefix = "strix-llama-" if expected_lane == CANDIDATE_LANE else "apple-ds4-"
    if re.fullmatch(re.escape(run_prefix) + r"[A-Za-z0-9._-]{1,96}", expected_run_id) is None:
        raise TraceError("externally expected lane run ID is invalid")
    if type(verification_unix) is not int or verification_unix <= 0:
        raise TraceError("trace verification time is invalid")
    authorization = manifest.get("authorization")
    if not isinstance(authorization, dict):
        raise TraceError("manifest execution authorization is missing")
    _require_exact_keys(
        authorization,
        {
            "format", "version", "lane", "challenge", "run_id", "issued_unix",
            "expires_unix", "approval_policy_sha256", "verifier_revision",
            "tokenizer_policy_sha256", "approvals",
        },
        "manifest execution authorization",
    )
    if authorization.get("format") != AUTHORIZATION_FORMAT or (
            authorization.get("version") != AUTHORIZATION_VERSION):
        raise TraceError("manifest execution authorization version is invalid")
    if authorization.get("lane") != expected_lane or policy["lane"] != expected_lane:
        raise TraceError("trace signer is not approved for the expected execution lane")
    if authorization.get("challenge") != expected_challenge:
        raise TraceError("manifest execution challenge differs from the external challenge")
    if authorization.get("run_id") != expected_run_id:
        raise TraceError("manifest lane run ID differs from the external run ID")
    if re.fullmatch(r"[0-9a-f]{64}", expected_approval_policy_sha256) is None or (
            authorization.get("approval_policy_sha256") != expected_approval_policy_sha256):
        raise TraceError("manifest executable approval policy differs from the external policy")
    if re.fullmatch(r"[0-9a-f]{40}", expected_verifier_revision) is None or (
            authorization.get("verifier_revision") != expected_verifier_revision):
        raise TraceError("manifest verifier revision differs from the external approval policy")
    issued_unix = authorization.get("issued_unix")
    expires_unix = authorization.get("expires_unix")
    if type(issued_unix) is not int or type(expires_unix) is not int or (
            issued_unix <= 0 or expires_unix <= issued_unix or
            expires_unix - issued_unix > MAX_AUTHORIZATION_LIFETIME_SECONDS):
        raise TraceError("manifest execution authorization validity window is invalid")
    if verification_unix < issued_unix or verification_unix > expires_unix:
        raise TraceError("manifest execution authorization is expired or not yet valid")
    runtime = manifest.get("runtime")
    if runtime != policy["runtime"]:
        raise TraceError("trace signer runtime role does not match the signed manifest")
    runtime_profile = (
        manifest.get("build", {}).get("runtime_profile", {}).get("name")
        if runtime == "llama.cpp"
        else manifest.get("accelerator", {}).get("runtime_kind")
    )
    if runtime_profile != policy["runtime_profile"]:
        raise TraceError("trace signer runtime profile does not match the signed manifest")
    _prompt_policy, prompt_digest = prompt_builder_approval(
        expected_prompt_builder_policy_id, policies=prompt_builder_policies)
    if authorization.get("tokenizer_policy_sha256") != tokenizer_policy_sha256(_prompt_policy["tokenizer"]):
        raise TraceError("manifest tokenizer policy differs from external prompt approval")
    approvals = authorization["approvals"]
    required_approvals = {"prompt_builder"}
    if expected_lane == CANDIDATE_LANE:
        required_approvals.add("candidate_exporter")
    else:
        required_approvals.add("ds4_exporter")
    if not isinstance(approvals, dict):
        raise TraceError("manifest execution approval bindings are invalid")
    _require_exact_keys(approvals, required_approvals, "manifest execution approval bindings")
    prompt_binding = approvals["prompt_builder"]
    if not isinstance(prompt_binding, dict) or set(prompt_binding) != {
            "id", "sha256", "install_trust_sha256"} or prompt_binding.get("id") != (
            expected_prompt_builder_policy_id) or prompt_binding.get("sha256") != prompt_digest or re.fullmatch(
            r"[0-9a-f]{64}", prompt_binding.get("install_trust_sha256", "")) is None:
        raise TraceError("manifest prompt builder approval differs from external policy")
    if expected_lane == CANDIDATE_LANE:
        if expected_candidate_exporter_policy_id is None:
            raise TraceError("external candidate exporter approval ID is required")
        if expected_ds4_exporter_policy_id is not None:
            raise TraceError("candidate verification must not specify a ds4 exporter approval")
        _candidate_policy, candidate_digest = candidate_exporter_approval(
            expected_candidate_exporter_policy_id, policies=candidate_exporter_policies)
        candidate_binding = approvals["candidate_exporter"]
        if not isinstance(candidate_binding, dict) or set(candidate_binding) != {
                "id", "sha256", "install_trust_sha256"} or candidate_binding.get("id") != (
                expected_candidate_exporter_policy_id) or candidate_binding.get("sha256") != (
                candidate_digest) or re.fullmatch(
                r"[0-9a-f]{64}", candidate_binding.get("install_trust_sha256", "")) is None:
            raise TraceError("manifest candidate exporter approval differs from external policy")
    else:
        if expected_candidate_exporter_policy_id is not None:
            raise TraceError("oracle verification must not specify a candidate exporter approval")
        if expected_ds4_exporter_policy_id is None:
            raise TraceError("external ds4 exporter approval ID is required")
        _ds4_policy, ds4_digest = ds4_exporter_approval(
            expected_ds4_exporter_policy_id, policies=ds4_exporter_policies)
        ds4_binding = approvals["ds4_exporter"]
        if not isinstance(ds4_binding, dict) or set(ds4_binding) != {
                "id", "sha256", "install_trust_sha256"} or ds4_binding.get("id") != (
                expected_ds4_exporter_policy_id) or ds4_binding.get("sha256") != (
                ds4_digest) or re.fullmatch(
                r"[0-9a-f]{64}", ds4_binding.get("install_trust_sha256", "")) is None:
            raise TraceError("manifest ds4 exporter approval differs from external policy")
    if seen_run_ids is not None:
        if expected_run_id in seen_run_ids:
            raise TraceError("trace lane run ID was reused")
        seen_run_ids.add(expected_run_id)


def execution_authorization(
        *,
        lane: str,
        challenge: str,
        run_id: str,
        issued_unix: int,
        expires_unix: int,
        approval_policy_sha256: str,
        verifier_revision: str,
        tokenizer_policy_sha256_value: str,
        approvals: dict[str, dict[str, str]]) -> dict[str, Any]:
    if lane not in {CANDIDATE_LANE, ORACLE_LANE}:
        raise TraceError("execution authorization lane is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", challenge) is None:
        raise TraceError("execution authorization challenge is invalid")
    run_prefix = "strix-llama-" if lane == CANDIDATE_LANE else "apple-ds4-"
    if re.fullmatch(re.escape(run_prefix) + r"[A-Za-z0-9._-]{1,96}", run_id) is None:
        raise TraceError("execution authorization run ID is invalid")
    if type(issued_unix) is not int or type(expires_unix) is not int or (
            issued_unix <= 0 or expires_unix <= issued_unix or
            expires_unix - issued_unix > MAX_AUTHORIZATION_LIFETIME_SECONDS):
        raise TraceError("execution authorization validity window is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", approval_policy_sha256) is None or re.fullmatch(
            r"[0-9a-f]{40}", verifier_revision) is None:
        raise TraceError("execution authorization approval policy identity is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", tokenizer_policy_sha256_value) is None:
        raise TraceError("execution authorization tokenizer policy identity is invalid")
    required_approvals = {"prompt_builder"}
    if lane == CANDIDATE_LANE:
        required_approvals.add("candidate_exporter")
    else:
        required_approvals.add("ds4_exporter")
    if not isinstance(approvals, dict):
        raise TraceError("execution authorization approvals are invalid")
    _require_exact_keys(approvals, required_approvals, "execution authorization approvals")
    for kind, binding in approvals.items():
        if not isinstance(binding, dict):
            raise TraceError("execution authorization approval binding is invalid")
        if binding != approval_binding(
                kind,
                binding.get("id", ""),
                binding.get("sha256", ""),
                binding.get("install_trust_sha256", ""),
        ):
            raise TraceError("execution authorization approval binding is invalid")
    authorization = {
        "format": AUTHORIZATION_FORMAT,
        "version": AUTHORIZATION_VERSION,
        "lane": lane,
        "challenge": challenge,
        "run_id": run_id,
        "issued_unix": issued_unix,
        "expires_unix": expires_unix,
        "approval_policy_sha256": approval_policy_sha256,
        "verifier_revision": verifier_revision,
        "tokenizer_policy_sha256": tokenizer_policy_sha256_value,
        "approvals": approvals,
    }
    return authorization


def bind_execution_authorization(root: Path, authorization: dict[str, Any]) -> None:
    manifest_path = root / MANIFEST_NAME
    try:
        manifest = strict_json_loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, TraceError) as error:
        raise TraceError(f"cannot bind execution authorization: {error}") from error
    if not isinstance(manifest, dict):
        raise TraceError("cannot bind execution authorization to a non-object manifest")
    if "authorization" in manifest:
        raise TraceError("manifest execution authorization is already present")
    manifest["authorization"] = authorization
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_bytes(_canonical_json_bytes(manifest))
    os.replace(temporary, manifest_path)


def validate_signing_identity(
        private_key: Path,
        principal: str,
        *,
        trusted_signers: dict[str, dict[str, str]] = APPROVED_TRACE_SIGNERS,
        ssh_keygen: Path | None = None,
        forbidden_root: Path | None = None) -> tuple[Path, str]:
    policy = _signer_policy(trusted_signers, principal)
    expected_key = policy["public_key"]
    executable = _validate_ssh_keygen(ssh_keygen or trusted_ssh_keygen_path())
    key_path = private_key.resolve()
    if private_key.is_symlink() or not key_path.is_file():
        raise TraceError("trace signing key must be a regular non-symlink file")
    key_stat = key_path.stat()
    if os.name != "nt":
        if key_stat.st_uid != os.getuid():
            raise TraceError("trace signing key is not owned by the current user")
        if key_stat.st_mode & 0o077:
            raise TraceError("trace signing key permissions are too broad")
    if forbidden_root is not None:
        try:
            key_path.relative_to(forbidden_root.resolve())
        except ValueError:
            pass
        else:
            raise TraceError("trace signing key must be outside the bundle")
    try:
        result = subprocess.run(
            [str(executable), "-y", "-f", str(key_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env=_ssh_environment(),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise TraceError(f"cannot derive trace signing public key: {error}") from error
    if result.returncode != 0:
        raise TraceError("cannot derive trace signing public key")
    derived_key = _normalize_public_key(result.stdout, allow_comment=True)
    if derived_key != _normalize_public_key(expected_key):
        raise TraceError("trace signing key does not match the approved signer")
    return executable, derived_key


def _canonical_json_bytes(data: Any) -> bytes:
    return (canonical_json(data) + "\n").encode("ascii")


def _bundle_path_parts(relative: str) -> tuple[str, ...]:
    if not isinstance(relative, str) or not relative or any(ord(character) > 0x7f for character in relative):
        raise TraceError(f"trace path is not portable ASCII: {relative!r}")
    if "\\" in relative or "%" in relative or relative.startswith("/") or relative.startswith("//") or (
            re.match(r"^[A-Za-z]:", relative) is not None):
        raise TraceError(f"trace path is outside the bundle: {relative}")
    parts = relative.split("/")
    if any(
            not part or part in {".", ".."} or re.fullmatch(r"[A-Za-z0-9._-]+", part) is None
            for part in parts):
        raise TraceError(f"trace path is not canonical: {relative}")
    if PurePosixPath(*parts).as_posix() != relative:
        raise TraceError(f"trace path is not canonical: {relative}")
    return tuple(parts)


def _safe_bundle_file(root: Path, relative: str) -> Path:
    parts = _bundle_path_parts(relative)
    candidate = root
    for part in parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise TraceError(f"trace path must not use symlinks: {relative}")
    try:
        candidate.resolve().relative_to(root)
    except ValueError as error:
        raise TraceError(f"trace path is outside the bundle: {relative}") from error
    if not candidate.is_file():
        raise TraceError(f"trace bundle file is missing or not regular: {relative}")
    return candidate


def _read_bundle_file(
        root: Path,
        relative: str,
        *,
        retain: bool) -> tuple[BundleFileReceipt, bytes | None]:
    path = _safe_bundle_file(root, relative)
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise TraceError(f"cannot open trace bundle file {relative}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise TraceError(f"trace bundle file is not regular: {relative}")
        if getattr(before, "st_nlink", 1) != 1:
            raise TraceError(f"trace bundle file must not be hard linked: {relative}")
        digest = hashlib.sha256()
        chunks = [] if retain else None
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
        path_stat = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise TraceError(f"cannot read trace bundle file {relative}: {error}") from error
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    identity_path = (
        path_stat.st_dev,
        path_stat.st_ino,
        path_stat.st_size,
        path_stat.st_mtime_ns,
        path_stat.st_ctime_ns,
    )
    if identity_before != identity_after or identity_after != identity_path or byte_count != after.st_size:
        raise TraceError(f"trace bundle file changed while reading: {relative}")
    receipt = BundleFileReceipt(
        device=after.st_dev,
        inode=after.st_ino,
        byte_count=byte_count,
        modified_ns=after.st_mtime_ns,
        changed_ns=after.st_ctime_ns,
        sha256=digest.hexdigest(),
    )
    return receipt, b"".join(chunks) if chunks is not None else None


def _read_canonical_json(
        root: Path,
        relative: str) -> tuple[dict[str, Any], bytes, BundleFileReceipt]:
    receipt, retained = _read_bundle_file(root, relative, retain=True)
    assert retained is not None
    try:
        data = retained
        text = data.decode("ascii")
        record = strict_json_loads(text)
    except (UnicodeError, TraceError) as error:
        raise TraceError(f"cannot read canonical JSON {relative}: {error}") from error
    if not isinstance(record, dict) or data != _canonical_json_bytes(record):
        raise TraceError(f"trace JSON is not canonical: {relative}")
    return record, data, receipt


def _read_canonical_jsonl(
        root: Path,
        relative: str) -> tuple[list[dict[str, Any]], bytes, BundleFileReceipt]:
    receipt, retained = _read_bundle_file(root, relative, retain=True)
    assert retained is not None
    data = retained
    if not data or not data.endswith(b"\n"):
        raise TraceError(f"trace JSONL is empty or truncated: {relative}")
    records = []
    for line_number, raw in enumerate(data.splitlines(keepends=True), 1):
        try:
            text = raw.decode("ascii")
            record = strict_json_loads(text)
        except (UnicodeError, TraceError) as error:
            raise TraceError(f"invalid JSONL at {relative}:{line_number}: {error}") from error
        if not isinstance(record, dict) or raw != _canonical_json_bytes(record):
            raise TraceError(f"trace JSONL is not canonical: {relative}:{line_number}")
        records.append(record)
    return records, data, receipt


def _bundle_domain(
        root: Path,
) -> tuple[
        bytes,
        dict[str, Any],
        list[dict[str, Any]],
        dict[str, BundleFileReceipt],
        dict[str, bytes],
]:
    if root.is_symlink():
        raise TraceError("trace root must not be a symlink")
    try:
        canonical_root = root.resolve(strict=True)
    except OSError as error:
        raise TraceError(f"cannot resolve trace root: {error}") from error
    if not canonical_root.is_dir():
        raise TraceError("trace root is not a directory")

    manifest, manifest_bytes, manifest_receipt = _read_canonical_json(canonical_root, MANIFEST_NAME)
    events, events_bytes, events_receipt = _read_canonical_jsonl(canonical_root, EVENTS_NAME)
    expected_paths = {MANIFEST_NAME, EVENTS_NAME}
    receipts = {
        MANIFEST_NAME: manifest_receipt,
        EVENTS_NAME: events_receipt,
    }
    contents = {
        MANIFEST_NAME: manifest_bytes,
        EVENTS_NAME: events_bytes,
    }

    for event in events:
        blob = event.get("blob")
        if not isinstance(blob, str):
            raise TraceError("event blob reference is invalid")
        expected_paths.add(blob)

    prompt = manifest.get("prompt")
    provenance = prompt.get("provenance") if isinstance(prompt, dict) else None
    if not isinstance(provenance, dict) or not isinstance(provenance.get("path"), str):
        raise TraceError("prompt provenance reference is missing")
    expected_paths.add(provenance["path"])

    audits = manifest.get("audits")
    if not isinstance(audits, dict):
        raise TraceError("manifest audit envelope is invalid")
    audit_paths = []
    referenced_metadata_paths = {provenance["path"]}
    for phase in ("pre", "post"):
        phase_audits = audits.get(phase)
        if not isinstance(phase_audits, dict):
            raise TraceError(f"manifest {phase} audit set is invalid")
        for reference in phase_audits.values():
            if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
                raise TraceError(f"manifest {phase} audit reference is invalid")
            if reference["path"] in referenced_metadata_paths:
                raise TraceError(f"trace metadata path is referenced more than once: {reference['path']}")
            referenced_metadata_paths.add(reference["path"])
            audit_paths.append(reference["path"])
            expected_paths.add(reference["path"])

    for relative in audit_paths:
        record, data, receipt = _read_canonical_json(canonical_root, relative)
        receipts[relative] = receipt
        contents[relative] = data
        audit = record.get("data", {}).get("audit")
        if isinstance(audit, dict) and isinstance(audit.get("path"), str):
            audit_jsonl = audit["path"]
            if audit_jsonl in referenced_metadata_paths:
                raise TraceError(f"trace metadata path is referenced more than once: {audit_jsonl}")
            referenced_metadata_paths.add(audit_jsonl)
            expected_paths.add(audit_jsonl)
            _records, jsonl_data, jsonl_receipt = _read_canonical_jsonl(canonical_root, audit_jsonl)
            receipts[audit_jsonl] = jsonl_receipt
            contents[audit_jsonl] = jsonl_data

    _provenance, provenance_data, provenance_receipt = _read_canonical_json(
        canonical_root,
        provenance["path"],
    )
    receipts[provenance["path"]] = provenance_receipt
    contents[provenance["path"]] = provenance_data

    actual_paths = set()
    try:
        for path in canonical_root.rglob("*"):
            relative = path.relative_to(canonical_root).as_posix()
            if path.is_symlink():
                raise TraceError(f"trace bundle contains a symlink: {relative}")
            if path.is_file():
                if relative != SIGNATURE_NAME:
                    actual_paths.add(relative)
            elif not path.is_dir():
                raise TraceError(f"trace bundle contains a nonregular entry: {relative}")
    except OSError as error:
        raise TraceError(f"cannot enumerate trace bundle: {error}") from error
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unexpected " + ", ".join(extra))
        raise TraceError(f"trace signed file set is invalid: {'; '.join(detail)}")

    records = []
    for relative in sorted(expected_paths, key=lambda value: value.encode("ascii")):
        _bundle_path_parts(relative)
        data = contents.get(relative)
        if data is not None:
            byte_count = len(data)
            digest = sha256_bytes(data)
        else:
            receipt, _retained = _read_bundle_file(canonical_root, relative, retain=False)
            receipts[relative] = receipt
            byte_count = receipt.byte_count
            digest = receipt.sha256
        records.append({
            "path": relative,
            "byte_count": byte_count,
            "sha256": digest,
        })
    domain = SEAL_DOMAIN_PREFIX + _canonical_json_bytes(records)
    return domain, manifest, events, receipts, contents


def _validate_unsealed_bundle(
        root: Path,
        manifest: dict[str, Any],
        events: list[dict[str, Any]],
        receipts: dict[str, BundleFileReceipt],
        contents: dict[str, bytes],
        verifier: TraceVerifier) -> None:
    bundle = object.__new__(_TRACE_BUNDLE_TYPE)
    bundle.root = root
    bundle.signer_principal = verifier.principal
    bundle.verifier = verifier
    bundle.manifest = manifest
    bundle._file_receipts = receipts
    bundle._retained_files = contents
    bundle._validate_manifest()
    bundle.events = bundle._validate_sealed_events(events, True)
    if bundle.manifest.get("event_count") != len(bundle.events):
        raise TraceError("manifest event_count mismatch")
    bundle._validate_coverage()


def seal_bundle(
        root: Path,
        *,
        private_key: Path,
        principal: str,
        expected_lane: str,
        expected_challenge: str,
        expected_run_id: str,
        candidate_exporter_policies: dict[str, dict[str, Any]] = APPROVED_CANDIDATE_EXPORTERS,
        ds4_exporter_policies: dict[str, dict[str, Any]] = APPROVED_DS4_EXPORTERS,
        prompt_builder_policies: dict[str, dict[str, Any]] = APPROVED_PROMPT_BUILDERS,
        expected_candidate_exporter_policy_id: str | None = None,
        expected_ds4_exporter_policy_id: str | None = None,
        expected_prompt_builder_policy_id: str = "",
        expected_approval_policy_sha256: str = "",
        expected_verifier_revision: str = "",
        verification_unix: int | None = None,
        trusted_signers: dict[str, dict[str, str]] = APPROVED_TRACE_SIGNERS,
        ssh_keygen: Path | None = None) -> str:
    root = root.resolve()
    executable, _public_key = validate_signing_identity(
        private_key,
        principal,
        trusted_signers=trusted_signers,
        ssh_keygen=ssh_keygen,
        forbidden_root=root,
    )
    key_path = private_key.resolve()
    signature_path = root / SIGNATURE_NAME
    if signature_path.exists() or signature_path.is_symlink():
        raise TraceError("trace signature envelope already exists")
    domain, manifest, events, receipts, contents = _bundle_domain(root)
    verification_time = int(time.time()) if verification_unix is None else verification_unix
    policy = _signer_policy(trusted_signers, principal)
    validate_execution_authorization(
        manifest,
        policy=policy,
        expected_lane=expected_lane,
        expected_challenge=expected_challenge,
        expected_run_id=expected_run_id,
        verification_unix=verification_time,
        candidate_exporter_policies=candidate_exporter_policies,
        ds4_exporter_policies=ds4_exporter_policies,
        prompt_builder_policies=prompt_builder_policies,
        expected_candidate_exporter_policy_id=expected_candidate_exporter_policy_id,
        expected_ds4_exporter_policy_id=expected_ds4_exporter_policy_id,
        expected_prompt_builder_policy_id=expected_prompt_builder_policy_id,
        expected_approval_policy_sha256=expected_approval_policy_sha256,
        expected_verifier_revision=expected_verifier_revision,
    )
    _validate_unsealed_bundle(
        root,
        manifest,
        events,
        receipts,
        contents,
        TraceVerifier(
            principal=principal,
            trusted_signers=trusted_signers,
            ssh_keygen=executable,
            expected_lane=expected_lane,
            expected_challenge=expected_challenge,
            expected_run_id=expected_run_id,
            verification_unix=verification_time,
            candidate_exporter_policies=candidate_exporter_policies,
            ds4_exporter_policies=ds4_exporter_policies,
            prompt_builder_policies=prompt_builder_policies,
            expected_candidate_exporter_policy_id=expected_candidate_exporter_policy_id,
            expected_ds4_exporter_policy_id=expected_ds4_exporter_policy_id,
            expected_prompt_builder_policy_id=expected_prompt_builder_policy_id,
            expected_approval_policy_sha256=expected_approval_policy_sha256,
            expected_verifier_revision=expected_verifier_revision,
        ),
    )
    try:
        with tempfile.TemporaryDirectory(prefix="dsv41-trace-sign-") as signing_temp:
            signing_root = Path(signing_temp).resolve()
            if os.name != "nt":
                signing_root.chmod(0o700)
            domain_path = signing_root / "bundle-domain"
            signature_file = signing_root / "bundle-domain.sig"
            with domain_path.open("xb") as stream:
                stream.write(domain)
                stream.flush()
                os.fsync(stream.fileno())
            if os.name != "nt":
                domain_path.chmod(0o600)
            if signature_file.exists() or signature_file.is_symlink():
                raise TraceError("temporary trace signature output already exists")
            result = subprocess.run(
                [
                    str(executable),
                    "-Y", "sign",
                    "-f", str(key_path),
                    "-n", SEAL_NAMESPACE,
                    str(domain_path),
                ],
                stdin=subprocess.DEVNULL,
                check=False,
                capture_output=True,
                timeout=30,
                env=_ssh_environment(),
            )
            if result.returncode != 0:
                raise TraceError("trusted ssh-keygen failed to sign trace bundle")
            signature_receipt, signature_bytes = _read_bundle_file(
                signing_root,
                signature_file.name,
                retain=True,
            )
            if signature_receipt.byte_count == 0 or signature_receipt.sha256 != sha256_bytes(signature_bytes or b""):
                raise TraceError("trusted ssh-keygen did not create a stable signature")
            assert signature_bytes is not None
    except (OSError, subprocess.SubprocessError) as error:
        raise TraceError(f"cannot sign trace bundle: {error}") from error
    try:
        signature = signature_bytes.decode("ascii")
    except UnicodeError as error:
        raise TraceError("trace signature is not ASCII") from error
    if not signature.startswith("-----BEGIN SSH SIGNATURE-----\n") or not signature.endswith(
            "-----END SSH SIGNATURE-----\n"):
        raise TraceError("trusted ssh-keygen returned an invalid signature")
    envelope = {
        "format": SEAL_FORMAT,
        "version": SEAL_VERSION,
        "namespace": SEAL_NAMESPACE,
        "principal": principal,
        "domain_sha256": sha256_bytes(domain),
        "signature": signature,
    }
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=root,
                prefix=".bundle-signature.",
                delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(_canonical_json_bytes(envelope))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, signature_path)
    except FileExistsError as error:
        raise TraceError("trace signature envelope already exists") from error
    except OSError as error:
        raise TraceError(f"cannot install trace signature envelope: {error}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return envelope["domain_sha256"]


def verify_bundle_seal(
        root: Path,
        verifier: TraceVerifier,
) -> tuple[
        dict[str, Any],
        list[dict[str, Any]],
        str,
        dict[str, BundleFileReceipt],
        dict[str, bytes],
]:
    policy = _signer_policy(verifier.trusted_signers, verifier.principal)
    approved_key = _normalize_public_key(policy["public_key"])
    executable = _validate_ssh_keygen(verifier.ssh_keygen)
    root = root.resolve()
    envelope, _envelope_bytes, _envelope_receipt = _read_canonical_json(root, SIGNATURE_NAME)
    _require_exact_keys(
        envelope,
        {"format", "version", "namespace", "principal", "domain_sha256", "signature"},
        "trace signature envelope",
    )
    if envelope.get("format") != SEAL_FORMAT or envelope.get("version") != SEAL_VERSION or (
            envelope.get("namespace") != SEAL_NAMESPACE):
        raise TraceError("trace signature envelope version is invalid")
    if envelope.get("principal") != verifier.principal:
        raise TraceError("trace signature principal differs from the externally expected signer")
    signature = envelope.get("signature")
    if not isinstance(signature, str) or not signature.startswith("-----BEGIN SSH SIGNATURE-----\n") or (
            not signature.endswith("-----END SSH SIGNATURE-----\n")):
        raise TraceError("trace signature envelope contains an invalid signature")
    domain, manifest, events, receipts, contents = _bundle_domain(root)
    validate_execution_authorization(
        manifest,
        policy=policy,
        expected_lane=verifier.expected_lane,
        expected_challenge=verifier.expected_challenge,
        expected_run_id=verifier.expected_run_id,
        verification_unix=verifier.verification_unix,
        candidate_exporter_policies=verifier.candidate_exporter_policies,
        ds4_exporter_policies=verifier.ds4_exporter_policies,
        prompt_builder_policies=verifier.prompt_builder_policies,
        expected_candidate_exporter_policy_id=verifier.expected_candidate_exporter_policy_id,
        expected_ds4_exporter_policy_id=verifier.expected_ds4_exporter_policy_id,
        expected_prompt_builder_policy_id=verifier.expected_prompt_builder_policy_id,
        expected_approval_policy_sha256=verifier.expected_approval_policy_sha256,
        expected_verifier_revision=verifier.expected_verifier_revision,
        seen_run_ids=verifier.seen_run_ids,
    )
    domain_sha256 = sha256_bytes(domain)
    if envelope.get("domain_sha256") != domain_sha256:
        raise TraceError("trace signed-domain SHA-256 mismatch")
    with tempfile.TemporaryDirectory(prefix="dsv41-trace-verify-") as temp:
        temporary = Path(temp)
        allowed_signers = temporary / "allowed_signers"
        signature_file = temporary / "signature"
        allowed_signers.write_text(
            f"{verifier.principal} {approved_key}\n",
            encoding="ascii",
        )
        signature_file.write_text(signature, encoding="ascii")
        try:
            result = subprocess.run(
                [
                    str(executable),
                    "-Y", "verify",
                    "-f", str(allowed_signers),
                    "-I", verifier.principal,
                    "-n", SEAL_NAMESPACE,
                    "-s", str(signature_file),
                ],
                input=domain,
                check=False,
                capture_output=True,
                timeout=30,
                env=_ssh_environment(),
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise TraceError(f"cannot verify trace bundle signature: {error}") from error
    if result.returncode != 0:
        raise TraceError("trace bundle signature verification failed")
    return manifest, events, domain_sha256, receipts, contents


def _require_exact_keys(record: dict[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - set(record))
    extra = sorted(set(record) - keys)
    if missing or extra:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unexpected " + ", ".join(extra))
        raise TraceError(f"{label} fields are invalid: {'; '.join(detail)}")


def validate_accelerator_attestation(runtime: str, accelerator: Any) -> dict[str, Any]:
    if not isinstance(accelerator, dict):
        raise TraceError("manifest accelerator attestation is invalid")
    common = {
        "format",
        "version",
        "runtime_kind",
        "platform",
        "backend",
        "backend_device",
        "backend_description",
        "architecture",
        "source",
    }
    if runtime == "llama.cpp":
        _require_exact_keys(
            accelerator,
            common | {"pci_device_id", "kfd_node", "gpu_id", "gfx_target_version"},
            "llama.cpp accelerator attestation",
        )
        expected = {
            "format": "dsv41-accelerator-attestation",
            "version": 2,
            "runtime_kind": "strix-rocm",
            "platform": "linux",
            "backend": "ROCm",
            "backend_device": "ROCm0",
            "architecture": "gfx1151",
            "gfx_target_version": 110501,
            "source": "linux-kfd-sysfs",
        }
        for key, value in expected.items():
            if accelerator.get(key) != value:
                raise TraceError(f"llama.cpp accelerator {key} mismatch")
        if re.fullmatch(
                r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]",
                accelerator.get("pci_device_id", "")) is None:
            raise TraceError("llama.cpp accelerator PCI identity is invalid")
        if not isinstance(accelerator.get("kfd_node"), str) or not accelerator["kfd_node"].isdigit():
            raise TraceError("llama.cpp accelerator KFD node is invalid")
        if type(accelerator.get("gpu_id")) is not int or accelerator["gpu_id"] <= 0:
            raise TraceError("llama.cpp accelerator GPU identity is invalid")
    elif runtime == "ds4":
        _require_exact_keys(
            accelerator,
            common | {
                "metal_registry_id",
                "recommended_max_working_set_bytes",
                "unified_memory",
            },
            "ds4 accelerator attestation",
        )
        expected = {
            "format": "dsv41-accelerator-attestation",
            "version": 2,
            "runtime_kind": "apple-metal",
            "platform": "macos",
            "backend": "Metal",
            "source": "metal-device-query",
            "unified_memory": True,
        }
        for key, value in expected.items():
            if accelerator.get(key) != value:
                raise TraceError(f"ds4 accelerator {key} mismatch")
        if type(accelerator.get("unified_memory")) is not bool:
            raise TraceError("ds4 accelerator unified-memory identity is invalid")
        registry_id = accelerator.get("metal_registry_id")
        if type(registry_id) is not int or registry_id <= 0:
            raise TraceError("ds4 accelerator Metal registry identity is invalid")
        working_set = accelerator.get("recommended_max_working_set_bytes")
        if type(working_set) is not int or working_set <= 0:
            raise TraceError("ds4 accelerator working-set identity is invalid")
    else:
        raise TraceError(f"unsupported runtime accelerator attestation: {runtime}")
    for key in ("backend_device", "backend_description", "architecture"):
        if not isinstance(accelerator.get(key), str) or not accelerator[key]:
            raise TraceError(f"manifest accelerator {key} is invalid")
    return accelerator


def validate_storage_attestation(runtime: str, item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise TraceError("storage attestation is invalid")
    common = {
        "format",
        "version",
        "runtime_kind",
        "platform",
        "storage_kind",
        "resolved_path",
        "existing_path",
        "mount_point",
        "filesystem_type",
        "source",
    }
    if runtime == "llama.cpp":
        _require_exact_keys(
            item,
            common | {
                "mount_source",
                "device_number",
                "block_device_path",
                "nvme_device",
                "rotational",
            },
            "llama.cpp storage attestation",
        )
        expected = {
            "format": "dsv41-storage-attestation",
            "version": 2,
            "runtime_kind": "strix-rocm",
            "platform": "linux",
            "storage_kind": "linux-nvme",
            "source": "linux-mountinfo-sysfs",
            "rotational": False,
        }
        for key, value in expected.items():
            if item.get(key) != value:
                raise TraceError(f"llama.cpp storage {key} mismatch")
        if type(item.get("rotational")) is not bool:
            raise TraceError("llama.cpp storage rotational identity is invalid")
        if not isinstance(item.get("nvme_device"), str) or re.fullmatch(
                r"nvme[0-9]+(?:c[0-9]+)?n[0-9]+", item["nvme_device"]) is None:
            raise TraceError("llama.cpp storage NVMe device identity is invalid")
        if not isinstance(item.get("mount_source"), str) or not item["mount_source"].startswith("/dev/"):
            raise TraceError("llama.cpp storage mount source is not a local block device")
        if re.fullmatch(r"[0-9]+:[0-9]+", item.get("device_number", "")) is None:
            raise TraceError("llama.cpp storage device number is invalid")
        block_device_path = item.get("block_device_path")
        if not isinstance(block_device_path, str) or item["nvme_device"] not in Path(block_device_path).parts:
            raise TraceError("llama.cpp storage block device ancestry is invalid")
    elif runtime == "ds4":
        _require_exact_keys(
            item,
            common | {
                "device_identifier",
                "parent_whole_disk",
                "bus_protocol",
                "filesystem_device",
                "internal",
                "solid_state",
            },
            "ds4 storage attestation",
        )
        expected = {
            "format": "dsv41-storage-attestation",
            "version": 2,
            "runtime_kind": "apple-metal",
            "platform": "macos",
            "storage_kind": "darwin-local-solid-state",
            "source": "diskutil-info-plist",
            "internal": True,
            "solid_state": True,
        }
        for key, value in expected.items():
            if item.get(key) != value:
                raise TraceError(f"ds4 storage {key} mismatch")
        if type(item.get("internal")) is not bool or type(item.get("solid_state")) is not bool:
            raise TraceError("ds4 storage media identity is invalid")
        for key in ("device_identifier", "parent_whole_disk", "bus_protocol"):
            if not isinstance(item.get(key), str) or not item[key]:
                raise TraceError(f"ds4 storage {key} is invalid")
        if item["bus_protocol"].lower() not in {"nvme", "apple fabric"}:
            raise TraceError("ds4 storage is not NVMe-backed")
        if type(item.get("filesystem_device")) is not int or item["filesystem_device"] < 0:
            raise TraceError("ds4 storage filesystem device identity is invalid")
    else:
        raise TraceError(f"unsupported runtime storage attestation: {runtime}")
    for path_key in ("resolved_path", "existing_path", "mount_point"):
        value = item.get(path_key)
        if not isinstance(value, str) or not value.startswith("/"):
            raise TraceError(f"storage {path_key} is invalid")
    if not isinstance(item.get("filesystem_type"), str) or not item["filesystem_type"]:
        raise TraceError("storage filesystem type is invalid")
    try:
        Path(item["resolved_path"]).relative_to(Path(item["mount_point"]))
        Path(item["existing_path"]).relative_to(Path(item["mount_point"]))
    except ValueError as error:
        raise TraceError("storage mount ancestry is invalid") from error
    return item


def validate_host_attestation(host: Any) -> dict[str, Any]:
    if not isinstance(host, dict):
        raise TraceError("ds4 host attestation is invalid")
    _require_exact_keys(
        host,
        {
            "format",
            "version",
            "runtime_kind",
            "platform",
            "machine",
            "hardware_model",
            "os_version",
            "memory_bytes",
            "source",
        },
        "ds4 host attestation",
    )
    expected = {
        "format": "dsv41-host-attestation",
        "version": 1,
        "runtime_kind": "apple-metal",
        "platform": "macos",
        "machine": "arm64",
        "source": "darwin-sysctl",
    }
    for key, value in expected.items():
        if host.get(key) != value:
            raise TraceError(f"ds4 host {key} mismatch")
    for key in ("hardware_model", "os_version"):
        if not isinstance(host.get(key), str) or not host[key]:
            raise TraceError(f"ds4 host {key} is invalid")
    if type(host.get("memory_bytes")) is not int or host["memory_bytes"] < 128 * 1024 * 1024 * 1024:
        raise TraceError("ds4 host memory is below 128 GiB")
    return host


WATCHDOG_STATE_KEYS = {
    "total_bytes",
    "available_bytes",
    "used_bytes",
    "swap_entries",
    "peak_used_bytes",
    "child_pid",
    "child_status",
    "child_returncode",
    "process_group_id",
    "process_group_status",
    "threshold_reason",
}
WATCHDOG_REQUIRED_ERROR_CLASSIFICATIONS = {
    "configuration_error",
    "internal_error",
    "launch_error",
    "lease_error",
    "termination_timeout",
}
WATCHDOG_OPTIONAL_ERROR_CLASSIFICATIONS = {
    "procfs_error",
    "signal_error",
}


def validate_watchdog_event(event: Any) -> dict[str, Any]:
    if not isinstance(event, dict):
        raise TraceError("watchdog JSONL event is not an object")
    event_name = event.get("event")
    required = {"timestamp", "event"} | WATCHDOG_STATE_KEYS
    optional: set[str] = set()
    if event_name == "preflight":
        required |= {"soft_bytes", "emergency_bytes", "strict_ceiling_bytes"}
    elif event_name == "child_started":
        required.add("command")
    elif event_name == "sample":
        pass
    elif event_name == "process_group_signal":
        required.add("signal")
        optional.add("grace_deadline_monotonic")
    elif event_name == "final":
        required |= {"classification", "exit_code"}
        optional |= {"error", "secondary_errors"}
    else:
        raise TraceError(f"watchdog JSONL event name is invalid: {event_name}")
    missing = sorted(required - set(event))
    unexpected = sorted(set(event) - required - optional)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise TraceError("watchdog JSONL event fields are invalid: " + "; ".join(details))
    if not isinstance(event["timestamp"], str) or re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z",
            event["timestamp"]) is None:
        raise TraceError("watchdog JSONL timestamp is invalid")
    for key in ("total_bytes", "available_bytes", "used_bytes", "swap_entries", "peak_used_bytes"):
        if event[key] is not None and (type(event[key]) is not int or event[key] < 0):
            raise TraceError(f"watchdog JSONL {key} is invalid")
    for key in ("child_pid", "process_group_id"):
        if event[key] is not None and (type(event[key]) is not int or event[key] <= 0):
            raise TraceError(f"watchdog JSONL {key} is invalid")
    if event["child_returncode"] is not None and type(event["child_returncode"]) is not int:
        raise TraceError("watchdog JSONL child_returncode is invalid")
    if event["child_status"] not in {"not_started", "running", "signaled", "exited"}:
        raise TraceError("watchdog JSONL child_status is invalid")
    if event["process_group_status"] not in {
            "not_created", "active", "leader_exited", "signal_error",
            "termination_timeout", "missing", "sighup_sent", "sigint_sent",
            "sigterm_sent", "sigkill_sent", "sigkill_timeout"}:
        raise TraceError("watchdog JSONL process_group_status is invalid")
    if not isinstance(event["threshold_reason"], str) or not event["threshold_reason"]:
        raise TraceError("watchdog JSONL threshold_reason is invalid")
    if event_name == "preflight":
        for key in ("soft_bytes", "emergency_bytes", "strict_ceiling_bytes"):
            if type(event[key]) is not int or event[key] <= 0:
                raise TraceError(f"watchdog JSONL {key} is invalid")
    elif event_name == "child_started":
        if not isinstance(event["command"], list) or not event["command"] or (
                not all(isinstance(value, str) and value for value in event["command"])):
            raise TraceError("watchdog JSONL command is invalid")
    elif event_name == "process_group_signal":
        if event["signal"] not in {"SIGHUP", "SIGINT", "SIGTERM", "SIGKILL"}:
            raise TraceError("watchdog JSONL signal is invalid")
        deadline = event.get("grace_deadline_monotonic")
        if deadline is not None and (
                event["signal"] != "SIGTERM" or
                not isinstance(deadline, (int, float)) or isinstance(deadline, bool) or
                not math.isfinite(deadline)):
            raise TraceError("watchdog JSONL grace deadline is invalid")
    elif event_name == "final":
        if event["classification"] not in {
                "child_exit", "soft_limit", "grace_timeout", "swap_appeared",
                "emergency_limit", "procfs_error", "signal_error", "termination_timeout",
                "lease_error", "parent_signal", "internal_error", "launch_error",
                "configuration_error", "startup_swap_active", "startup_emergency_limit",
                "startup_soft_limit"}:
            raise TraceError("watchdog JSONL classification is invalid")
        if type(event["exit_code"]) is not int:
            raise TraceError("watchdog JSONL exit code is invalid")
        classification = event["classification"]
        requires_error = classification in WATCHDOG_REQUIRED_ERROR_CLASSIFICATIONS
        allows_error = requires_error or classification in WATCHDOG_OPTIONAL_ERROR_CLASSIFICATIONS
        if requires_error and "error" not in event:
            raise TraceError("watchdog JSONL error presence does not match classification")
        if not allows_error and "error" in event:
            raise TraceError("watchdog JSONL error presence does not match classification")
        if "error" in event and (not isinstance(event["error"], str) or not event["error"]):
            raise TraceError("watchdog JSONL error is invalid")
        if "secondary_errors" in event:
            secondary_errors = event["secondary_errors"]
            if classification != "signal_error" or "error" not in event:
                raise TraceError("watchdog JSONL secondary errors require a primary signal error")
            if not isinstance(secondary_errors, list) or not secondary_errors:
                raise TraceError("watchdog JSONL secondary errors are invalid")
            for secondary_error in secondary_errors:
                _require_exact_keys(
                    secondary_error, {"component", "detail"}, "watchdog JSONL secondary error")
                if secondary_error["component"] not in {"audit", "lease", "stderr"} or (
                        not isinstance(secondary_error["detail"], str) or not secondary_error["detail"]):
                    raise TraceError("watchdog JSONL secondary error is invalid")
    return event


def element_count(shape: Iterable[int]) -> int:
    count = 1
    for dim in shape:
        if type(dim) is not int or dim <= 0:
            raise TraceError(f"shape dimension must be a nonzero positive integer: {dim!r}")
        count *= dim
    return count


def expected_bytes(event: dict[str, Any]) -> int:
    dtype = event.get("dtype")
    if dtype not in DTYPE_SIZES:
        raise TraceError(f"unsupported dtype: {dtype!r}")
    shape = event.get("shape")
    if not isinstance(shape, list):
        raise TraceError("event shape must be an array")
    return element_count(shape) * DTYPE_SIZES[dtype]


def event_key(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        event.get("phase", ""),
        int(event.get("step", -1)),
        int(event.get("token_start", -1)),
        event.get("layer"),
        event.get("component", ""),
    )


def event_order(key: tuple[Any, ...]) -> tuple[Any, ...]:
    phase_order = {"input": 0, "prefill": 1, "decode": 2}
    component_order = {
        "prompt.bytes": 0,
        "prompt.tokens": 1,
        "engram.row_ids": 2,
        "expert.ids": 3,
        "expert.weights": 4,
        "attn.source": 5,
        "attn.candidate_blocks": 6,
        "attn.candidates": 7,
        "decode.greedy_token": 8,
        "logits.prefill": 9,
        "logits.decode": 10,
    }
    return (
        phase_order.get(key[0], 99),
        key[2],
        key[1],
        -1 if key[3] is None else key[3],
        component_order.get(key[4], 99),
        key[4],
    )


def classify(component: str) -> str:
    if component == "prompt.tokens":
        return "tokenizer"
    if component == "engram.row_ids":
        return "engram_row"
    if component == "expert.ids":
        return "routing_original_expert"
    if component == "expert.weights":
        return "routing_weight"
    if component.startswith("attn.candidate"):
        return "attention_candidate"
    if component == "attn.source":
        return "attention_source"
    if component == "logits.prefill":
        return "prefill_logits"
    if component == "logits.decode":
        return "decode_logits"
    if component == "decode.greedy_token":
        return "decode_token"
    return "trace_data"


def validate_event(event: dict[str, Any]) -> None:
    required = {
        "trace_version",
        "component",
        "phase",
        "step",
        "token_start",
        "token_count",
        "layer",
        "dtype",
        "shape",
        "byte_order",
        "byte_count",
        "sha256",
        "blob",
    }
    missing = sorted(required - event.keys())
    if missing:
        raise TraceError(f"event is missing fields: {', '.join(missing)}")
    allowed = set(required)
    if event.get("component") == "expert.ids":
        allowed.add("semantic_id_space")
    extra = sorted(set(event) - allowed)
    if extra:
        raise TraceError(f"event has unexpected fields: {', '.join(extra)}")
    if event["trace_version"] != TRACE_VERSION:
        raise TraceError(f"unsupported event version: {event['trace_version']!r}")
    if event["byte_order"] != "little":
        raise TraceError("trace blobs must use little-endian byte order")
    if event["phase"] not in ("input", "prefill", "decode"):
        raise TraceError("event phase is invalid")
    if type(event["step"]) is not int or event["step"] < 0:
        raise TraceError("event step is invalid")
    if type(event["token_start"]) is not int or event["token_start"] < 0:
        raise TraceError("event token_start is invalid")
    if type(event["token_count"]) is not int or event["token_count"] <= 0:
        raise TraceError("event token_count is invalid")
    if event["layer"] is not None and (type(event["layer"]) is not int or event["layer"] < 0):
        raise TraceError("event layer is invalid")
    if not isinstance(event["shape"], list) or not event["shape"] or any(dim <= 0 for dim in event["shape"]):
        raise TraceError("event shape dimensions must be nonzero")
    if event["byte_count"] != expected_bytes(event):
        raise TraceError("event byte_count does not match dtype and shape")
    digest = event["sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise TraceError("event sha256 is invalid")
    if event["blob"] != f"{BLOBS_DIR}/{digest}.bin":
        raise TraceError("event blob path is not content addressed")
    component = event["component"]
    if component == "expert.ids":
        if event.get("semantic_id_space") != "original":
            raise TraceError("expert.ids must contain original expert IDs, not cache slot IDs")
    if "slot" in component or event.get("semantic_id_space") == "cache_slot":
        raise TraceError("cache slot IDs are forbidden in correctness traces")


class TraceBundleWriter:
    def __init__(self, root: Path, manifest: dict[str, Any]):
        self.root = root
        self.blobs = root / BLOBS_DIR
        if root.exists() and any(root.iterdir()):
            raise TraceError(f"trace output directory is not empty: {root}")
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.events = (root / EVENTS_NAME).open("x", encoding="ascii", newline="\n")
        self.manifest = dict(manifest)
        self.manifest["trace_format"] = TRACE_FORMAT
        self.manifest["trace_version"] = TRACE_VERSION
        self.event_count = 0

    def add_event(
        self,
        *,
        component: str,
        phase: str,
        step: int,
        token_start: int,
        token_count: int,
        layer: int | None,
        dtype: str,
        shape: list[int],
        data: bytes,
        semantic_id_space: str | None = None,
    ) -> dict[str, Any]:
        digest = sha256_bytes(data)
        blob_rel = f"{BLOBS_DIR}/{digest}.bin"
        event = {
            "trace_version": TRACE_VERSION,
            "component": component,
            "phase": phase,
            "step": step,
            "token_start": token_start,
            "token_count": token_count,
            "layer": layer,
            "dtype": dtype,
            "shape": shape,
            "byte_order": "little",
            "byte_count": len(data),
            "sha256": digest,
            "blob": blob_rel,
        }
        if semantic_id_space is not None:
            event["semantic_id_space"] = semantic_id_space
        validate_event(event)
        blob_path = self.root / blob_rel
        if not blob_path.exists():
            temp = blob_path.with_suffix(".tmp")
            temp.write_bytes(data)
            os.replace(temp, blob_path)
        self.events.write(canonical_json(event) + "\n")
        self.events.flush()
        self.event_count += 1
        return event

    def close(self) -> None:
        if self.events.closed:
            return
        self.events.close()
        self.manifest["event_count"] = self.event_count
        manifest_path = self.root / MANIFEST_NAME
        temp = manifest_path.with_suffix(".tmp")
        temp.write_text(canonical_json(self.manifest) + "\n", encoding="ascii")
        os.replace(temp, manifest_path)

    def __enter__(self) -> "TraceBundleWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.events.close()
        if exc_type is None:
            self.manifest["event_count"] = self.event_count
            manifest_path = self.root / MANIFEST_NAME
            temp = manifest_path.with_suffix(".tmp")
            temp.write_text(canonical_json(self.manifest) + "\n", encoding="ascii")
            os.replace(temp, manifest_path)


class TraceBundle:
    def __init__(
            self,
            root: Path,
            verify_blobs: bool = True,
            *,
            verifier: TraceVerifier | None = None,
            signer_principal: str | None = None,
            expected_lane: str | None = None,
            expected_challenge: str | None = None,
            expected_run_id: str | None = None,
            expected_candidate_exporter_policy_id: str | None = None,
            expected_ds4_exporter_policy_id: str | None = None,
            expected_prompt_builder_policy_id: str | None = None,
            verification_unix: int | None = None,
            seen_run_ids: set[str] | None = None):
        if root.is_symlink():
            raise TraceError("trace root must not be a symlink")
        self.root = root.resolve()
        if verifier is None:
            if None in (
                    signer_principal, expected_lane, expected_challenge, expected_run_id,
                    expected_prompt_builder_policy_id):
                raise TraceError(
                    "external signer, lane, challenge, run ID, and prompt builder approval are required")
            if expected_lane == CANDIDATE_LANE and expected_candidate_exporter_policy_id is None:
                raise TraceError("external candidate exporter approval is required")
            if expected_lane == ORACLE_LANE and expected_ds4_exporter_policy_id is None:
                raise TraceError("external ds4 exporter approval is required")
            verifier = TraceVerifier.production(
                signer_principal,
                expected_lane=expected_lane,
                expected_challenge=expected_challenge,
                expected_run_id=expected_run_id,
                expected_candidate_exporter_policy_id=expected_candidate_exporter_policy_id,
                expected_ds4_exporter_policy_id=expected_ds4_exporter_policy_id,
                expected_prompt_builder_policy_id=expected_prompt_builder_policy_id,
                verification_unix=verification_unix,
                seen_run_ids=seen_run_ids,
            )
        elif any(value is not None for value in (
                signer_principal, expected_lane, expected_challenge, expected_run_id,
                expected_candidate_exporter_policy_id, expected_ds4_exporter_policy_id,
                expected_prompt_builder_policy_id,
                verification_unix, seen_run_ids)):
            raise TraceError("trace verifier cannot be combined with separate verification inputs")
        self.signer_principal = verifier.principal
        self.verifier = verifier
        (
            self.manifest,
            sealed_events,
            self.seal_sha256,
            self._file_receipts,
            self._retained_files,
        ) = verify_bundle_seal(self.root, verifier)
        if self.manifest.get("trace_format") != TRACE_FORMAT:
            raise TraceError("manifest trace_format mismatch")
        if self.manifest.get("trace_version") != TRACE_VERSION:
            raise TraceError("manifest trace_version mismatch")
        self._validate_manifest()
        self.events = self._validate_sealed_events(sealed_events, verify_blobs)
        if self.manifest.get("event_count") != len(self.events):
            raise TraceError("manifest event_count mismatch")
        self._validate_coverage()

    def _validate_sealed_events(
            self,
            sealed_events: list[dict[str, Any]],
            verify_blobs: bool) -> list[dict[str, Any]]:
        result = []
        for line_number, event in enumerate(sealed_events, 1):
            validate_event(event)
            if verify_blobs:
                data = self.read_blob(event)
                if len(data) != event["byte_count"]:
                    raise TraceError(f"truncated blob for event line {line_number}")
                if sha256_bytes(data) != event["sha256"]:
                    raise TraceError(f"corrupt blob for event line {line_number}")
            result.append(event)
        return result

    def _validate_manifest(self) -> None:
        if not isinstance(self.manifest.get("runtime"), str) or not self.manifest["runtime"]:
            raise TraceError("manifest runtime is invalid")
        if self.manifest["runtime"] not in ("ds4", "llama.cpp"):
            raise TraceError("manifest runtime must be ds4 or llama.cpp")
        top_level = {
            "trace_format",
            "trace_version",
            "event_count",
            "runtime",
            "revision",
            "build",
            "model",
            "prompt",
            "accelerator",
            "config",
            "comparison",
            "environment",
            "paths",
            "storage_policy",
            "authorization",
            "audits",
            "expected",
        }
        if self.manifest["runtime"] == "llama.cpp":
            top_level.add("candidate")
        else:
            top_level.update({"host", "oracle"})
        _require_exact_keys(self.manifest, top_level, f"{self.manifest['runtime']} manifest")
        if not isinstance(self.manifest["revision"], str) or not self.manifest["revision"]:
            raise TraceError("manifest revision is invalid")
        if self.manifest["runtime"] == "ds4" and self.manifest["revision"] != DS4_REVISION:
            raise TraceError(f"ds4 revision must be {DS4_REVISION}")
        if self.manifest["runtime"] == "llama.cpp" and re.fullmatch(
                r"[0-9a-f]{40}", self.manifest["revision"]) is None:
            raise TraceError("llama.cpp revision must be the exact full Git revision")
        if not isinstance(self.manifest["build"], dict):
            raise TraceError("manifest build is invalid")
        build_keys = (
            {
                "number", "info", "compiler", "target", "path", "sha256",
                "runtime_profile", "runtime_receipt_sha256",
                "runtime_libraries", "runtime_libraries_post", "runtime_module_monitor",
            }
            if self.manifest["runtime"] == "llama.cpp"
            else {
                "compiler", "target", "path", "sha256",
                "runtime_profile", "runtime_receipt_sha256",
                "runtime_libraries", "runtime_libraries_post", "runtime_module_monitor",
            }
        )
        _require_exact_keys(self.manifest["build"], build_keys, "manifest build")
        build_sha256 = self.manifest["build"].get("sha256", "")
        if not isinstance(build_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", build_sha256) is None:
            raise TraceError("manifest build SHA-256 is invalid")
        for key in build_keys - {
                "sha256", "number", "runtime_profile", "runtime_receipt_sha256",
                "runtime_libraries", "runtime_libraries_post", "runtime_module_monitor"}:
            value = self.manifest["build"].get(key)
            if not isinstance(value, str) or not value:
                raise TraceError(f"manifest build {key} is invalid")
        if "number" in build_keys and type(self.manifest["build"]["number"]) is not int:
            raise TraceError("manifest build number is invalid")
        build_path = self.manifest["build"]["path"]
        if not build_path.startswith("/") or ".." in PurePosixPath(build_path).parts or (
                str(PurePosixPath(build_path)) != build_path):
            raise TraceError("manifest build path is not canonical")
        if self.manifest["runtime"] == "llama.cpp":
            libraries = self.manifest["build"]["runtime_libraries"]
            if not isinstance(libraries, list) or not libraries:
                raise TraceError("manifest runtime library identities are invalid")
            post_libraries = self.manifest["build"]["runtime_libraries_post"]
            if post_libraries != libraries:
                raise TraceError("manifest runtime library closure changed during trace generation")
            module_monitor = self.manifest["build"]["runtime_module_monitor"]
            if not isinstance(module_monitor, dict):
                raise TraceError("manifest runtime module monitor is invalid")
            _require_exact_keys(
                module_monitor,
                {"mechanism", "checked_after_trace", "project_additions"},
                "manifest runtime module monitor",
            )
            if module_monitor["mechanism"] not in {"dyld-add-image", "pre-post-snapshot"}:
                raise TraceError("manifest runtime module monitor mechanism is invalid")
            if module_monitor["checked_after_trace"] is not True:
                raise TraceError("manifest runtime module monitor did not complete")
            if module_monitor["project_additions"] != []:
                raise TraceError("manifest records a runtime module addition during trace generation")
            profile = self.manifest["build"]["runtime_profile"]
            if not isinstance(profile, dict):
                raise TraceError("manifest runtime profile is invalid")
            _require_exact_keys(
                profile,
                {"name", "components", "selected_backend_component"},
                "manifest runtime profile",
            )
            profile_name = profile.get("name")
            components = profile.get("components")
            selected_backend_component = profile.get("selected_backend_component")
            if profile_name not in {"co-located", "sibling-lib"}:
                raise TraceError("manifest runtime profile name is invalid")
            if not isinstance(components, list) or components != sorted(components) or (
                    len(components) != len(set(components))) or any(
                        not isinstance(component, str) or re.fullmatch(r"[a-z0-9-]+", component) is None
                        for component in components):
                raise TraceError("manifest runtime profile components are invalid")
            if not {"llama-common", "llama", "ggml", "ggml-base"}.issubset(set(components)):
                raise TraceError("manifest runtime profile is missing core components")
            if not isinstance(selected_backend_component, str) or (
                    selected_backend_component not in components or
                    selected_backend_component in {"ggml", "ggml-base"} or
                    not selected_backend_component.startswith("ggml-")):
                raise TraceError("manifest selected backend component is invalid")
            roles = set()
            paths = set()
            library_components = set()
            previous_path = None
            executable_path = PurePosixPath(build_path)
            binary_directory = executable_path.parent
            library_directory = binary_directory.parent / "lib"
            for library in libraries:
                _require_exact_keys(
                    library,
                    {"component", "filename", "path", "sha256", "role", "revision"},
                    "manifest runtime library",
                )
                component = library.get("component")
                filename = library.get("filename")
                path = library.get("path")
                digest = library.get("sha256")
                role = library.get("role")
                revision = library.get("revision")
                if component not in components or component in library_components:
                    raise TraceError("manifest runtime library component is invalid")
                library_components.add(component)
                if not isinstance(filename, str) or PurePosixPath(filename).name != filename:
                    raise TraceError("manifest runtime library filename is invalid")
                expected_role = {
                    "llama-common": "build-info",
                    "llama": "llama",
                    "ggml-base": "ggml",
                    selected_backend_component: "selected-backend",
                }.get(component, f"runtime:{component}")
                if role != expected_role or role in roles:
                    raise TraceError("manifest runtime library role is invalid")
                roles.add(role)
                if not isinstance(path, str) or not path.startswith("/") or ".." in PurePosixPath(path).parts or (
                        str(PurePosixPath(path)) != path):
                    raise TraceError("manifest runtime library path is not canonical")
                if path in paths:
                    raise TraceError("manifest runtime library path is duplicated")
                if previous_path is not None and path <= previous_path:
                    raise TraceError("manifest runtime library paths are not sorted")
                runtime_path = PurePosixPath(path)
                try:
                    runtime_path.relative_to(library_directory)
                    in_library_directory = True
                except ValueError:
                    in_library_directory = False
                if runtime_path != executable_path and runtime_path.parent != binary_directory and (
                        not in_library_directory):
                    raise TraceError("manifest runtime library is outside the exporter runtime directory")
                paths.add(path)
                previous_path = path
                if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                    raise TraceError("manifest runtime library SHA-256 is invalid")
                if component in {"llama-common", "ggml-base"}:
                    if revision != self.manifest["revision"]:
                        raise TraceError("manifest runtime library revision is invalid")
                elif revision is not None:
                    raise TraceError("manifest runtime library revision is unexpected")
                expected_directory = binary_directory if profile_name == "co-located" else library_directory
                if runtime_path.parent != expected_directory or runtime_path.name != filename:
                    raise TraceError("manifest runtime library path differs from the runtime profile")
            if library_components != set(components):
                raise TraceError("manifest runtime library set differs from the runtime profile")
            receipt = {
                "format": "dsv41-runtime-receipt",
                "version": 1,
                "revision": self.manifest["revision"],
                "profile": profile_name,
                "components": sorted(
                    [
                        {
                            "component": library["component"],
                            "filename": library["filename"],
                            "sha256": library["sha256"],
                            "revision": library["revision"],
                        }
                        for library in libraries
                    ],
                    key=lambda item: item["component"],
                ),
            }
            receipt_sha256 = self.manifest["build"].get("runtime_receipt_sha256")
            if not isinstance(receipt_sha256, str) or receipt_sha256 != sha256_bytes(
                    canonical_json(receipt).encode("ascii")):
                raise TraceError("manifest runtime receipt SHA-256 is invalid")
        else:
            ds4_policy, _ds4_policy_sha256 = ds4_exporter_approval(
                self.verifier.expected_ds4_exporter_policy_id or "",
                policies=self.verifier.ds4_exporter_policies,
            )
            build_evidence = {
                key: self.manifest["build"][key]
                for key in (
                    "revision",
                    "path",
                    "sha256",
                    "runtime_profile",
                    "runtime_receipt_sha256",
                    "runtime_libraries",
                    "runtime_libraries_post",
                )
                if key in self.manifest["build"]
            }
            build_evidence["revision"] = self.manifest["revision"]
            validate_runtime_build_evidence(
                build_evidence,
                ds4_policy,
                label="ds4 exporter",
            )
            module_monitor = self.manifest["build"]["runtime_module_monitor"]
            if not isinstance(module_monitor, dict):
                raise TraceError("ds4 manifest runtime module monitor is invalid")
            _require_exact_keys(
                module_monitor,
                {"mechanism", "checked_after_trace", "project_additions"},
                "ds4 manifest runtime module monitor",
            )
            if module_monitor["mechanism"] != "dyld-add-image" or (
                    module_monitor["checked_after_trace"] is not True) or (
                    module_monitor["project_additions"] != []):
                raise TraceError("ds4 manifest runtime module monitor is invalid")
            receipt = ds4_policy["runtime_receipt"]
        for section in ("model", "prompt"):
            if not isinstance(self.manifest[section], dict):
                raise TraceError(f"manifest {section} is invalid")
        _require_exact_keys(
            self.manifest["model"],
            {"path", "sha256", "byte_count", "architecture"},
            "manifest model",
        )
        _require_exact_keys(
            self.manifest["prompt"],
            {
                "path",
                "sha256",
                "byte_count",
                "corpus_name",
                "corpus_sha256",
                "target_tokens",
                "provenance",
            },
            "manifest prompt",
        )
        for section in ("model", "prompt"):
            digest = self.manifest[section].get("sha256")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise TraceError(f"manifest {section} SHA-256 is invalid")
            if type(self.manifest[section].get("byte_count")) is not int or self.manifest[section]["byte_count"] <= 0:
                raise TraceError(f"manifest {section} byte_count is invalid")
        for section in ("config", "comparison", "environment", "audits"):
            if not isinstance(self.manifest[section], dict):
                raise TraceError(f"manifest {section} is invalid")
        _require_exact_keys(
            self.manifest["environment"],
            {"system_info", "command"},
            "manifest environment",
        )
        if not all(isinstance(value, str) and value for value in self.manifest["environment"].values()):
            raise TraceError("manifest environment values are invalid")
        system_info = self.manifest["environment"]["system_info"].lower()
        if self.manifest["runtime"] == "llama.cpp" and "linux" not in system_info:
            raise TraceError("llama.cpp environment is not Linux")
        if self.manifest["runtime"] == "ds4" and not any(
                name in system_info for name in ("darwin", "macos")):
            raise TraceError("ds4 environment is not macOS")
        context = self.manifest["config"].get("context")
        decode_steps = self.manifest["config"].get("decode_steps")
        if type(context) is not int or type(decode_steps) is not int or (
                decode_steps <= 0 or context <= decode_steps):
            raise TraceError("manifest context or decode_steps is invalid")
        if self.manifest["model"]["sha256"] != MODEL_SHA256:
            raise TraceError(f"model SHA-256 must be {MODEL_SHA256}")
        if self.manifest["model"].get("architecture") != "deepseek41":
            raise TraceError("model architecture must be deepseek41")
        if self.manifest["storage_policy"] != NO_EXTERNAL_STATE_STORAGE:
            raise TraceError("manifest external cache/state storage policy is invalid")
        accelerator = validate_accelerator_attestation(self.manifest["runtime"], self.manifest["accelerator"])
        if self.manifest["runtime"] == "ds4":
            if "host" not in self.manifest:
                raise TraceError("ds4 host attestation is missing")
            validate_host_attestation(self.manifest["host"])
        else:
            if "host" in self.manifest:
                raise TraceError("llama.cpp manifest must not contain Apple host attestation")
        required_paths = {
            "model", "prompt", "output", "repository", "temporary_directory",
        }
        if self.manifest["runtime"] == "ds4":
            required_paths.update({"runtime_checkout", "runner_executable", "runner_script", "exporter"})
        paths = self.manifest["paths"]
        if not isinstance(paths, dict) or set(paths) != required_paths:
            raise TraceError("manifest execution paths are invalid")
        for label, value in paths.items():
            if not isinstance(label, str) or not isinstance(value, str) or not value.startswith("/"):
                raise TraceError("manifest execution path is invalid")
        if self.manifest["model"].get("path") != paths["model"]:
            raise TraceError("manifest model path is not bound to execution paths")
        if self.manifest["prompt"].get("path") != paths["prompt"]:
            raise TraceError("manifest prompt path is not bound to execution paths")
        if self.manifest["runtime"] == "ds4" and self.manifest["build"]["path"] != paths["exporter"]:
            raise TraceError("ds4 build path is not bound to the executed exporter")
        corpus_name = self.manifest["prompt"].get("corpus_name")
        if corpus_name not in CORPUS_SHA256:
            raise TraceError("prompt corpus is not in the fixed correctness corpus set")
        if self.manifest["prompt"].get("corpus_sha256") != CORPUS_SHA256[corpus_name]:
            raise TraceError(f"prompt corpus SHA-256 is invalid for {corpus_name}")
        provenance = self.manifest["prompt"].get("provenance")
        if not isinstance(provenance, dict):
            raise TraceError("prompt provenance reference is missing")
        _require_exact_keys(provenance, {"path", "sha256"}, "prompt provenance reference")
        provenance_sha256 = provenance.get("sha256", "")
        if not isinstance(provenance_sha256, str) or re.fullmatch(
                r"[0-9a-f]{64}", provenance_sha256) is None:
            raise TraceError("prompt provenance SHA-256 is invalid")
        if provenance.get("path") != f"provenance/{provenance_sha256}.json":
            raise TraceError("prompt provenance path is not content addressed")
        try:
            provenance_bytes = self._read_verified_file(provenance["path"])
            provenance_record = strict_json_loads(provenance_bytes.decode("ascii"))
        except (OSError, UnicodeError, TraceError) as error:
            raise TraceError(f"cannot read prompt provenance: {error}") from error
        if sha256_bytes(provenance_bytes) != provenance_sha256:
            raise TraceError("prompt provenance SHA-256 mismatch")
        expected_target = context - decode_steps
        prompt_policy, prompt_policy_sha256 = prompt_builder_approval(
            self.verifier.expected_prompt_builder_policy_id,
            policies=self.verifier.prompt_builder_policies,
        )
        source_root_lexical = Path(prompt_policy["source_root"])
        source_root_resolved = source_root_lexical.resolve(strict=False)
        corpus_resolved = (
            source_root_resolved / "tests" / "corpus" / corpus_name).resolve(strict=False)
        provenance_checks = {
            "format": "dsv41-prompt-provenance",
            "version": 2,
            "corpus_name": corpus_name,
            "corpus_sha256": self.manifest["prompt"]["corpus_sha256"],
            "corpus_path": str(corpus_resolved),
            "corpus_resolved_path": str(corpus_resolved),
            "source_root_lexical_path": str(source_root_lexical),
            "source_root_resolved_path": str(source_root_resolved),
            "model_sha256": self.manifest["model"]["sha256"],
            "prompt_sha256": self.manifest["prompt"]["sha256"],
            "prompt_byte_count": self.manifest["prompt"]["byte_count"],
            "context": context,
            "decode_steps": decode_steps,
            "target_tokens": expected_target,
            "actual_tokens": expected_target,
        }
        for key, value in provenance_checks.items():
            if provenance_record.get(key) != value:
                raise TraceError(f"prompt provenance {key} mismatch")
        try:
            if Path(str(provenance_record.get("corpus_lexical_path"))).resolve(strict=False) != corpus_resolved:
                raise TraceError("prompt provenance corpus lexical path resolves outside the approved source")
        except OSError as error:
            raise TraceError(f"prompt provenance corpus lexical path is invalid: {error}") from error
        expected_prompt = approved_prompt_record(
            prompt_policy,
            corpus_name=corpus_name,
            context=context,
            decode_steps=decode_steps,
        )
        builder_checks = {
            "builder_approval_id": self.verifier.expected_prompt_builder_policy_id,
            "builder_approval_sha256": prompt_policy_sha256,
            "builder_path": prompt_policy["executable_path"],
            "builder_sha256": prompt_policy["executable_sha256"],
            "builder_revision": prompt_policy["revision"],
            "builder_runtime_profile": prompt_policy["runtime_profile"],
            "tokenizer": prompt_policy["tokenizer"],
        }
        for key, value in builder_checks.items():
            if provenance_record.get(key) != value:
                raise TraceError(f"prompt provenance {key} differs from external approval")
        builder_runtime_build = validate_runtime_build_evidence(
            provenance_record.get("builder_runtime_build"),
            prompt_policy,
            label="prompt builder",
        )
        builder_runtime_build_sha256 = runtime_build_evidence_sha256(
            builder_runtime_build,
            prompt_policy,
            label="prompt builder",
        )
        if provenance_record.get("builder_runtime_build_sha256") != builder_runtime_build_sha256:
            raise TraceError("prompt provenance runtime build SHA-256 mismatch")
        builder_trust = validate_install_trust_evidence(
            provenance_record.get("builder_install_trust"), prompt_policy)
        builder_trust_sha256 = install_trust_sha256(builder_trust)
        if provenance_record.get("builder_install_trust_sha256") != builder_trust_sha256:
            raise TraceError("prompt provenance install trust SHA-256 mismatch")
        if self.manifest["authorization"]["approvals"]["prompt_builder"][
                "install_trust_sha256"] != builder_trust_sha256:
            raise TraceError("manifest prompt builder trust differs from signed provenance")
        for key in ("corpus_name", "corpus_sha256", "context", "decode_steps", "target_tokens",
                    "prompt_sha256", "prompt_byte_count"):
            if provenance_record[key] != expected_prompt[key]:
                raise TraceError(f"prompt provenance {key} differs from approved prompt output")
        _require_exact_keys(
            provenance_record,
            set(provenance_checks) | set(builder_checks) | {
                "corpus_lexical_path",
                "builder_runtime_build", "builder_runtime_build_sha256",
                "builder_install_trust", "builder_install_trust_sha256"},
            "prompt provenance",
        )
        if self.manifest["prompt"].get("target_tokens") != expected_target:
            raise TraceError("prompt target token count does not fill the configured context")
        if self.manifest["runtime"] == "llama.cpp":
            candidate = self.manifest.get("candidate")
            if not isinstance(candidate, dict):
                raise TraceError("llama.cpp candidate attestation is missing")
            _require_exact_keys(
                candidate,
                {
                    "repository",
                    "revision",
                    "base_revision",
                    "diff_sha256",
                    "executable_path",
                    "executable_sha256",
                    "runtime_libraries_sha256",
                    "runtime_receipt_sha256",
                    "exporter_approval_id",
                    "exporter_approval_sha256",
                    "install_trust",
                    "install_trust_sha256",
                },
                "llama.cpp candidate attestation",
            )
            if candidate.get("repository") != REPOSITORY:
                raise TraceError(f"candidate repository must be {REPOSITORY}")
            for key in (
                    "revision",
                    "base_revision",
                    "diff_sha256",
                    "executable_sha256",
                    "runtime_libraries_sha256",
                    "runtime_receipt_sha256",
                    "exporter_approval_sha256"):
                value = candidate.get(key, "")
                if not isinstance(value, str) or re.fullmatch(
                        r"[0-9a-f]{40}" if "revision" in key else r"[0-9a-f]{64}", value) is None:
                    raise TraceError(f"candidate {key} is invalid")
            if re.fullmatch(
                    r"[A-Za-z0-9._-]{1,128}", candidate.get("exporter_approval_id", "")) is None:
                raise TraceError("candidate exporter approval ID is invalid")
            executable_path = candidate.get("executable_path")
            if not isinstance(executable_path, str) or not executable_path.startswith("/") or (
                    ".." in PurePosixPath(executable_path).parts or
                    str(PurePosixPath(executable_path)) != executable_path):
                raise TraceError("candidate executable path is invalid")
            if candidate["revision"] != self.manifest["revision"]:
                raise TraceError("candidate revision does not match the exporter build revision")
            if candidate.get("executable_path") != self.manifest["build"]["path"]:
                raise TraceError("candidate executable path does not match the trace build")
            if candidate["executable_sha256"] != self.manifest["build"]["sha256"]:
                raise TraceError("candidate executable SHA-256 does not match the trace build")
            runtime_libraries_sha256 = sha256_bytes(
                canonical_json({
                    "pre": self.manifest["build"]["runtime_libraries"],
                    "post": self.manifest["build"]["runtime_libraries_post"],
                }).encode("ascii"))
            if candidate["runtime_libraries_sha256"] != runtime_libraries_sha256:
                raise TraceError("candidate runtime library identities do not match the trace build")
            if candidate["runtime_receipt_sha256"] != self.manifest["build"]["runtime_receipt_sha256"]:
                raise TraceError("candidate runtime receipt does not match the trace build")
            candidate_policy, candidate_policy_sha256 = candidate_exporter_approval(
                self.verifier.expected_candidate_exporter_policy_id or "",
                policies=self.verifier.candidate_exporter_policies,
            )
            if candidate["exporter_approval_id"] != self.verifier.expected_candidate_exporter_policy_id or (
                    candidate["exporter_approval_sha256"] != candidate_policy_sha256):
                raise TraceError("candidate exporter approval differs from external policy")
            policy_checks = {
                "repository": candidate["repository"],
                "revision": candidate["revision"],
                "base_revision": candidate["base_revision"],
                "diff_sha256": candidate["diff_sha256"],
                "executable_path": candidate["executable_path"],
                "executable_sha256": candidate["executable_sha256"],
                "runtime_profile": self.manifest["build"]["runtime_profile"],
                "runtime_receipt": receipt,
            }
            for key, value in policy_checks.items():
                if candidate_policy[key] != value:
                    raise TraceError(f"candidate {key} differs from external exporter approval")
            candidate_trust = validate_install_trust_evidence(
                candidate["install_trust"], candidate_policy)
            candidate_trust_sha256 = install_trust_sha256(candidate_trust)
            if candidate["install_trust_sha256"] != candidate_trust_sha256 or (
                    self.manifest["authorization"]["approvals"]["candidate_exporter"][
                        "install_trust_sha256"] != candidate_trust_sha256):
                raise TraceError("candidate install trust evidence is not bound to authorization")
        else:
            oracle = self.manifest.get("oracle")
            if not isinstance(oracle, dict):
                raise TraceError("ds4 oracle attestation is missing")
            _require_exact_keys(
                oracle,
                {
                    "repository",
                    "revision",
                    "verifier_revision",
                    "executable_path",
                    "executable_sha256",
                    "runtime_profile",
                    "runtime_build_sha256",
                    "runtime_libraries_sha256",
                    "runtime_receipt_sha256",
                    "exporter_approval_id",
                    "exporter_approval_sha256",
                    "install_trust",
                    "install_trust_sha256",
                },
                "ds4 oracle attestation",
            )
            for key in (
                    "revision",
                    "verifier_revision",
                    "executable_sha256",
                    "runtime_libraries_sha256",
                    "runtime_build_sha256",
                    "runtime_receipt_sha256",
                    "exporter_approval_sha256",
                    "install_trust_sha256"):
                value = oracle.get(key, "")
                if not isinstance(value, str) or re.fullmatch(
                        r"[0-9a-f]{40}" if "revision" in key else r"[0-9a-f]{64}", value) is None:
                    raise TraceError(f"ds4 oracle {key} is invalid")
            if oracle["repository"] != DS4_REPOSITORY or oracle["revision"] != DS4_REVISION or (
                    oracle["revision"] == oracle["verifier_revision"]):
                raise TraceError("ds4 oracle producer/verifier identity is invalid")
            if re.fullmatch(
                    r"[A-Za-z0-9._-]{1,128}", oracle.get("exporter_approval_id", "")) is None:
                raise TraceError("ds4 exporter approval ID is invalid")
            if oracle["executable_path"] != self.manifest["build"]["path"] or (
                    oracle["executable_sha256"] != self.manifest["build"]["sha256"]):
                raise TraceError("ds4 oracle executable identity does not match the trace build")
            if oracle["runtime_profile"] != self.manifest["build"]["runtime_profile"]:
                raise TraceError("ds4 oracle runtime profile does not match the trace build")
            ds4_policy, ds4_policy_sha256 = ds4_exporter_approval(
                self.verifier.expected_ds4_exporter_policy_id or "",
                policies=self.verifier.ds4_exporter_policies,
            )
            build_evidence = {
                "revision": self.manifest["revision"],
                "path": self.manifest["build"]["path"],
                "sha256": self.manifest["build"]["sha256"],
                "runtime_profile": self.manifest["build"]["runtime_profile"],
                "runtime_receipt_sha256": self.manifest["build"]["runtime_receipt_sha256"],
                "runtime_libraries": self.manifest["build"]["runtime_libraries"],
                "runtime_libraries_post": self.manifest["build"]["runtime_libraries_post"],
            }
            if oracle["runtime_build_sha256"] != runtime_build_evidence_sha256(
                    build_evidence, ds4_policy, label="ds4 exporter"):
                raise TraceError("ds4 oracle runtime build SHA-256 does not match the trace build")
            runtime_libraries_sha256 = sha256_bytes(
                canonical_json({
                    "pre": self.manifest["build"]["runtime_libraries"],
                    "post": self.manifest["build"]["runtime_libraries_post"],
                }).encode("ascii"))
            if oracle["runtime_libraries_sha256"] != runtime_libraries_sha256 or (
                    oracle["runtime_receipt_sha256"] != self.manifest["build"]["runtime_receipt_sha256"]):
                raise TraceError("ds4 oracle runtime evidence does not match the trace build")
            if oracle["exporter_approval_id"] != self.verifier.expected_ds4_exporter_policy_id or (
                    oracle["exporter_approval_sha256"] != ds4_policy_sha256):
                raise TraceError("ds4 exporter approval differs from external policy")
            policy_checks = {
                "repository": oracle["repository"],
                "revision": oracle["revision"],
                "executable_path": oracle["executable_path"],
                "executable_sha256": oracle["executable_sha256"],
                "runtime_profile": oracle["runtime_profile"],
                "runtime_receipt": receipt,
            }
            for key, value in policy_checks.items():
                if ds4_policy[key] != value:
                    raise TraceError(f"ds4 oracle {key} differs from external exporter approval")
            if oracle["verifier_revision"] != self.verifier.expected_verifier_revision:
                raise TraceError("ds4 oracle verifier revision differs from external approval")
            oracle_trust = validate_install_trust_evidence(
                oracle["install_trust"], ds4_policy)
            oracle_trust_sha256 = install_trust_sha256(oracle_trust)
            if oracle["install_trust_sha256"] != oracle_trust_sha256 or (
                    self.manifest["authorization"]["approvals"]["ds4_exporter"][
                        "install_trust_sha256"] != oracle_trust_sha256):
                raise TraceError("ds4 exporter install trust evidence is not bound to authorization")
        expected_config = {
            "layer_count": 40,
            "vocab_size": 129280,
            "engram_layers": [1, 14],
            "engram_rows_per_token": 24,
            "expert_count": 384,
            "experts_used": 6,
            "candidate_source_layer": 20,
            "candidate_topk_blocks": 2048,
            "candidate_block_size": 8,
            "index_top_k": 512,
            "raw_attention_layers": list(RAW_ATTENTION_LAYERS),
            "raw_attention_width": RAW_ATTENTION_WIDTH,
            "candidate_propagation_layers": [24, 28, 32, 36],
        }
        if self.manifest["config"].get("deepseek41") != expected_config:
            raise TraceError("DeepSeek V4.1 configuration is invalid")
        expected_comparison = {
            "tokens": "exact",
            "engram_rows": "exact",
            "expert_ids": "exact-original-id-space",
            "expert_weights": "byte-identical-f32",
            "attention_candidates": "exact",
            "logits": "byte-identical-f32",
        }
        if self.manifest["comparison"] != expected_comparison:
            raise TraceError("trace comparison policy is invalid")
        config = self.manifest["config"]
        if self.manifest["runtime"] == "llama.cpp":
            _require_exact_keys(
                config,
                {
                    "context",
                    "batch",
                    "ubatch",
                    "device",
                    "device_architecture",
                    "device_pci_id",
                    "decode_steps",
                    "kv_type_k",
                    "kv_type_v",
                    "flash_attention",
                    "gpu_layers",
                    "load_mode",
                    "expert_cache_slots",
                    "expert_cache_bytes",
                    "tokenizer",
                    "model_file_identity",
                    "watchdog_namespace",
                    "deepseek41",
                },
                "llama.cpp config",
            )
            if accelerator["backend_device"] != "ROCm0":
                raise TraceError("llama.cpp accelerator backend device must be ROCm0")
            if config.get("batch") != ADMITTED_BATCH or config.get("ubatch") != ADMITTED_UBATCH:
                raise TraceError("llama.cpp trace does not use the admitted batch and ubatch")
            if config.get("expert_cache_slots") != REQUIRED_EXPERT_SLOTS or (
                    config.get("expert_cache_bytes") != REQUIRED_EXPERT_CACHE_BYTES):
                raise TraceError("llama.cpp trace does not use the admitted expert cache")
            if config.get("device") != "ROCm0" or config.get("gpu_layers") != 99:
                raise TraceError("llama.cpp trace does not use the required ROCm0 offload")
            if config.get("device_architecture") != accelerator["architecture"] or (
                    config.get("device_pci_id") != accelerator["pci_device_id"]):
                raise TraceError("llama.cpp trace device identity is not bound to the accelerator attestation")
            if config.get("kv_type_k") != "f16" or config.get("kv_type_v") != "f16" or (
                    config.get("flash_attention") is not True) or config.get("load_mode") != 0:
                raise TraceError("llama.cpp trace inference configuration is invalid")
            if validate_tokenizer_policy(config.get("tokenizer")) != prompt_policy["tokenizer"]:
                raise TraceError("llama.cpp tokenizer policy differs from external prompt approval")
            model_file_identity = config.get("model_file_identity")
            if not isinstance(model_file_identity, dict):
                raise TraceError("llama.cpp model file identity is invalid")
            _require_exact_keys(
                model_file_identity,
                {
                    "format",
                    "version",
                    "path",
                    "device",
                    "inode",
                    "owner_uid",
                    "owner_gid",
                    "mode",
                    "link_count",
                    "byte_count",
                    "modified_ns",
                    "changed_ns",
                    "status_flags",
                    "source_descriptor_flags",
                    "target_descriptor_flags",
                    "sha256",
                },
                "llama.cpp model file identity",
            )
            if model_file_identity["format"] != "dsv41-model-file-identity" or (
                    model_file_identity["version"] != 1) or (
                    model_file_identity["path"] != self.manifest["model"]["path"]) or (
                    model_file_identity["byte_count"] != self.manifest["model"]["byte_count"]) or (
                    model_file_identity["sha256"] != self.manifest["model"]["sha256"]):
                raise TraceError("llama.cpp model file identity does not match the manifest model")
            for key in (
                    "device", "inode", "owner_uid", "owner_gid", "mode", "link_count",
                    "byte_count", "modified_ns", "changed_ns", "status_flags",
                    "source_descriptor_flags", "target_descriptor_flags"):
                if type(model_file_identity[key]) is not int or model_file_identity[key] < 0:
                    raise TraceError(f"llama.cpp model file identity {key} is invalid")
            if model_file_identity["link_count"] < 1 or model_file_identity["mode"] > 0o7777 or (
                    model_file_identity["source_descriptor_flags"] != 1) or (
                    model_file_identity["target_descriptor_flags"] != 0):
                raise TraceError("llama.cpp model descriptor policy is invalid")
            watchdog_namespace = config.get("watchdog_namespace")
            if not isinstance(watchdog_namespace, dict):
                raise TraceError("llama.cpp watchdog namespace binding is invalid")
            _require_exact_keys(
                watchdog_namespace,
                {
                    "format",
                    "version",
                    "authority",
                    "host_watchdog_pid",
                    "host_watchdog_process_group_id",
                    "host_watchdog_start_time_ticks",
                    "local_pid",
                    "local_parent_pid",
                    "local_process_group_id",
                    "local_session_id",
                    "namespace_pids",
                    "private_procfs",
                },
                "llama.cpp watchdog namespace binding",
            )
            if watchdog_namespace["format"] != "dsv41-watchdog-namespace-binding" or (
                    watchdog_namespace["version"] != 1) or (
                    watchdog_namespace["authority"] != "inherited-pidfd") or (
                    watchdog_namespace["private_procfs"] is not True):
                raise TraceError("llama.cpp watchdog namespace binding policy is invalid")
            for key in (
                    "host_watchdog_pid", "host_watchdog_process_group_id",
                    "host_watchdog_start_time_ticks", "local_pid",
                    "local_process_group_id", "local_session_id"):
                if type(watchdog_namespace[key]) is not int or watchdog_namespace[key] <= 1:
                    raise TraceError(f"llama.cpp watchdog namespace {key} is invalid")
            if watchdog_namespace["local_parent_pid"] != 1 or (
                    watchdog_namespace["local_pid"] != 2) or (
                    watchdog_namespace["local_pid"] != watchdog_namespace["local_process_group_id"]) or (
                    watchdog_namespace["local_pid"] != watchdog_namespace["local_session_id"]) or (
                    not isinstance(watchdog_namespace["namespace_pids"], list)) or (
                    not watchdog_namespace["namespace_pids"]) or (
                    watchdog_namespace["namespace_pids"][-1] != watchdog_namespace["local_pid"]):
                raise TraceError("llama.cpp watchdog namespace-local identity is invalid")
        if self.manifest["runtime"] == "ds4":
            _require_exact_keys(
                config,
                {
                    "context",
                    "decode_steps",
                    "prefill_chunk",
                    "device_backend",
                    "device_registry_id",
                    "deepseek41",
                },
                "ds4 config",
            )
            if config.get("prefill_chunk") != ADMITTED_UBATCH:
                raise TraceError("ds4 trace does not use the admitted prefill chunk")
            if config.get("device_backend") != "Metal" or (
                    config.get("device_registry_id") != accelerator["metal_registry_id"]):
                raise TraceError("ds4 trace device identity is not bound to the accelerator attestation")
        _require_exact_keys(self.manifest["audits"], {"pre", "post"}, "manifest audit envelope")
        for audit_phase in ("pre", "post"):
            phase_audits = self.manifest["audits"].get(audit_phase)
            if not isinstance(phase_audits, dict):
                raise TraceError(f"manifest {audit_phase} audit set is invalid")
            expected_kinds = (
                ("memory", "swap", "watchdog")
                if self.manifest["runtime"] == "llama.cpp"
                else ("memory", "swap", "runner")
            )
            if set(phase_audits) != set(expected_kinds):
                raise TraceError(f"manifest {audit_phase} audit kinds are invalid")
            for kind in expected_kinds:
                self._validate_audit_reference(audit_phase, kind, phase_audits.get(kind))

    def _validate_audit_reference(self, phase: str, kind: str, audit: Any) -> None:
        if not isinstance(audit, dict):
            raise TraceError(f"manifest {phase} {kind} audit reference is invalid")
        _require_exact_keys(
            audit,
            {"path", "sha256", "created_unix"},
            f"manifest {phase} {kind} audit reference",
        )
        audit_path = audit.get("path")
        if not isinstance(audit_path, str) or not audit_path:
            raise TraceError(f"manifest {phase} {kind} audit path is invalid")
        digest = audit.get("sha256", "")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise TraceError(f"manifest {phase} {kind} audit SHA-256 is invalid")
        if type(audit.get("created_unix")) is not int or audit["created_unix"] <= 0:
            raise TraceError(f"manifest {phase} {kind} audit timestamp is invalid")
        expected_path = f"audits/{phase}/{digest}.json"
        if audit_path != expected_path:
            raise TraceError(f"manifest {phase} {kind} audit path is not content addressed")
        try:
            evidence = self._read_verified_file(audit_path)
        except TraceError as error:
            raise TraceError(f"cannot read {phase} {kind} audit evidence: {error}") from error
        if sha256_bytes(evidence) != digest:
            raise TraceError(f"{phase} {kind} audit evidence SHA-256 mismatch")
        try:
            record = strict_json_loads(evidence.decode("ascii"))
        except (UnicodeError, TraceError) as error:
            raise TraceError(f"{phase} {kind} audit evidence is invalid: {error}") from error
        if record.get("kind") != kind or record.get("created_unix") != audit["created_unix"]:
            raise TraceError(f"{phase} {kind} audit evidence metadata mismatch")
        record_keys = {"created_unix", "kind", "environment", "data"}
        if kind == "memory":
            record_keys |= {"storage", "storage_policy", "accelerator"}
            if self.manifest["runtime"] == "ds4":
                record_keys.add("host")
        _require_exact_keys(record, record_keys, f"{phase} {kind} audit evidence")
        expected_environment = (
            {"HIP_LAUNCH_BLOCKING": "1"}
            if self.manifest["runtime"] == "llama.cpp"
            else {}
        )
        if record.get("environment") != expected_environment:
            raise TraceError(f"{phase} {kind} audit environment is invalid")
        if not isinstance(record.get("data"), dict):
            raise TraceError(f"{phase} {kind} audit evidence data is invalid")
        if kind == "memory":
            _require_exact_keys(
                record["data"],
                {"mem_total_bytes", "mem_available_bytes", "mem_used_bytes"},
                f"{phase} memory audit data",
            )
            used = record["data"].get("mem_used_bytes")
            total = record["data"].get("mem_total_bytes")
            available = record["data"].get("mem_available_bytes")
            if type(used) is not int or used < 0:
                raise TraceError(f"{phase} memory audit evidence is invalid")
            if self.manifest["runtime"] == "llama.cpp" and used >= SOFT_MEMORY_LIMIT:
                raise TraceError(f"{phase} memory audit evidence is invalid")
            if self.manifest["runtime"] == "ds4" and (
                    type(total) is not int or total < 128 * 1024 * 1024 * 1024 or
                    type(available) is not int or available <= 0 or available > total or used > total):
                raise TraceError(f"{phase} ds4 memory audit evidence is invalid")
            storage = record.get("storage")
            if record.get("storage_policy") != self.manifest["storage_policy"]:
                raise TraceError(f"{phase} memory audit storage policy mismatch")
            required_storage = {
                "model", "prompt", "output", "repository", "temporary_directory",
            }
            if self.manifest["runtime"] == "ds4":
                required_storage.update({"runtime_checkout", "runner_executable", "runner_script", "exporter"})
            if not isinstance(storage, dict) or set(storage) != required_storage:
                raise TraceError(f"{phase} memory audit storage evidence is missing")
            for label, item in storage.items():
                if not isinstance(label, str):
                    raise TraceError(f"{phase} memory audit storage evidence is invalid")
                validate_storage_attestation(self.manifest["runtime"], item)
                if item["resolved_path"] != self.manifest["paths"][label]:
                    raise TraceError(f"{phase} memory audit {label} path differs from the manifest")
            if record.get("accelerator") != self.manifest["accelerator"]:
                raise TraceError(f"{phase} memory audit accelerator evidence mismatch")
            if self.manifest["runtime"] == "ds4":
                host = validate_host_attestation(record.get("host"))
                if host != self.manifest["host"]:
                    raise TraceError(f"{phase} ds4 host evidence mismatch")
                if host["memory_bytes"] != total:
                    raise TraceError(f"{phase} ds4 host memory differs from the memory audit")
        if kind == "swap":
            if self.manifest["runtime"] == "llama.cpp":
                _require_exact_keys(record["data"], {"enabled", "entries"}, f"{phase} swap audit data")
                if record["data"].get("enabled") is not False or record["data"].get("entries") != []:
                    raise TraceError(f"{phase} swap audit evidence does not report zero configured swap")
            elif record["data"] != {
                    "source": "darwin-sysctl-vm.swapusage",
                    "total_bytes": 0,
                    "used_bytes": 0,
                    "free_bytes": 0,
            }:
                raise TraceError(f"{phase} ds4 swap audit evidence does not report zero swap")
        if kind == "runner":
            data = record["data"]
            _require_exact_keys(
                data,
                {
                    "format",
                    "version",
                    "runtime_kind",
                    "source",
                    "runner_pid",
                    "runner_parent_pid",
                    "runner_uid",
                    "runner_executable",
                    "runner_executable_sha256",
                    "runner_script",
                    "runner_script_sha256",
                    "exporter_path",
                    "exporter_sha256",
                    "exporter_approval_id",
                    "exporter_approval_sha256",
                    "exporter_install_trust_sha256",
                    "exporter_runtime_build_sha256",
                    "exporter_runtime_profile",
                    "exporter_runtime_receipt_sha256",
                    "producer_revision",
                    "verifier_revision",
                    "checkout_path",
                    "checkout_revision",
                    "command_sha256",
                },
                f"{phase} ds4 runner audit",
            )
            expected = {
                "format": "dsv41-runner-ownership",
                "version": 1,
                "runtime_kind": "apple-metal",
                "source": "python-subprocess",
                "exporter_sha256": self.manifest["build"]["sha256"],
                "exporter_approval_id": self.manifest["oracle"]["exporter_approval_id"],
                "exporter_approval_sha256": self.manifest["oracle"]["exporter_approval_sha256"],
                "exporter_install_trust_sha256": self.manifest["oracle"]["install_trust_sha256"],
                "exporter_runtime_build_sha256": self.manifest["oracle"]["runtime_build_sha256"],
                "exporter_runtime_profile": self.manifest["oracle"]["runtime_profile"],
                "exporter_runtime_receipt_sha256": self.manifest["oracle"]["runtime_receipt_sha256"],
                "producer_revision": self.manifest["oracle"]["revision"],
                "verifier_revision": self.manifest["oracle"]["verifier_revision"],
                "checkout_revision": DS4_REVISION,
                "checkout_path": self.manifest["paths"]["runtime_checkout"],
                "runner_executable": self.manifest["paths"]["runner_executable"],
                "runner_script": self.manifest["paths"]["runner_script"],
                "exporter_path": self.manifest["paths"]["exporter"],
            }
            for key, value in expected.items():
                if data.get(key) != value:
                    raise TraceError(f"{phase} ds4 runner {key} mismatch")
            for key in ("runner_pid", "runner_parent_pid"):
                if type(data.get(key)) is not int or data[key] <= 0:
                    raise TraceError(f"{phase} ds4 runner {key} is invalid")
            if type(data.get("runner_uid")) is not int or data["runner_uid"] < 0:
                raise TraceError(f"{phase} ds4 runner UID is invalid")
            for key in (
                    "runner_executable_sha256",
                    "runner_script_sha256",
                    "exporter_sha256",
                    "exporter_approval_sha256",
                    "exporter_install_trust_sha256",
                    "exporter_runtime_build_sha256",
                    "exporter_runtime_receipt_sha256",
                    "command_sha256"):
                if re.fullmatch(r"[0-9a-f]{64}", data.get(key, "")) is None:
                    raise TraceError(f"{phase} ds4 runner {key} is invalid")
            if re.fullmatch(
                    r"[A-Za-z0-9._-]{1,128}", data.get("exporter_approval_id", "")) is None:
                raise TraceError(f"{phase} ds4 runner exporter approval ID is invalid")
            for key in ("producer_revision", "verifier_revision"):
                if re.fullmatch(r"[0-9a-f]{40}", data.get(key, "")) is None:
                    raise TraceError(f"{phase} ds4 runner {key} is invalid")
            if not isinstance(data.get("exporter_runtime_profile"), dict):
                raise TraceError(f"{phase} ds4 runner runtime profile is invalid")
        if kind == "watchdog":
            required = (
                "format",
                "version",
                "lease_id",
                "state",
                "file_device",
                "file_inode",
                "file_uid",
                "file_mode",
                "lease_path",
                "watchdog_pid",
                "watchdog_start_time_utc",
                "watchdog_start_time_ticks",
                "watchdog_command",
                "watchdog_command_sha256",
                "watchdog_executable_path",
                "watchdog_script_path",
                "watchdog_script_sha256",
                "watchdog_revision",
                "soft_bytes",
                "emergency_bytes",
                "strict_ceiling_bytes",
                "grace_seconds",
                "sample_interval_seconds",
                "procfs_root",
                "guardian_pid",
                "child_pid",
                "child_process_group_id",
                "command",
                "child_command_sha256",
                "heartbeat_path",
                "heartbeat_unix",
                "max_heartbeat_age_seconds",
                "audit_live_path",
                "audit_device",
                "audit_inode",
                "audit_uid",
                "audit_mode",
                "audit_fd",
                "audit_sha256",
                "audit",
                "namespace_authority",
            )
            _require_exact_keys(record["data"], set(required), f"{phase} watchdog audit evidence")
            data = record["data"]
            if data["format"] != WATCHDOG_LEASE_FORMAT or data["version"] != WATCHDOG_VERSION:
                raise TraceError(f"{phase} watchdog audit format is invalid")
            if data["state"] != "active":
                raise TraceError(f"{phase} watchdog audit state is invalid")
            for key in ("file_device", "file_inode", "file_uid"):
                if type(data[key]) is not int or data[key] < 0:
                    raise TraceError(f"{phase} watchdog lease {key} is invalid")
            if data["file_mode"] != 0o600:
                raise TraceError(f"{phase} watchdog lease file mode is invalid")
            if not isinstance(data["lease_id"], str) or re.fullmatch(r"[0-9a-f]{32,64}", data["lease_id"]) is None:
                raise TraceError(f"{phase} watchdog audit lease ID is invalid")
            if type(data["watchdog_pid"]) is not int or data["watchdog_pid"] <= 1:
                raise TraceError(f"{phase} watchdog audit PID is invalid")
            if type(data["watchdog_start_time_ticks"]) is not int or data["watchdog_start_time_ticks"] <= 0:
                raise TraceError(f"{phase} watchdog audit start time is invalid")
            if not isinstance(data["watchdog_start_time_utc"], str) or re.fullmatch(
                    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z",
                    data["watchdog_start_time_utc"]) is None:
                raise TraceError(f"{phase} watchdog audit UTC start time is invalid")
            for key in ("watchdog_command_sha256", "watchdog_script_sha256"):
                if not isinstance(data[key], str) or re.fullmatch(r"[0-9a-f]{64}", data[key]) is None:
                    raise TraceError(f"{phase} watchdog audit {key} is invalid")
            for key in ("lease_path", "watchdog_command", "watchdog_script_path", "heartbeat_path", "audit_live_path"):
                if not isinstance(data[key], str) or not data[key]:
                    raise TraceError(f"{phase} watchdog audit {key} is invalid")
            if APPROVED_WATCHDOGS.get(data["watchdog_script_sha256"]) != data["watchdog_revision"]:
                raise TraceError(f"{phase} watchdog audit revision is invalid")
            if not isinstance(data["watchdog_executable_path"], str) or not data["watchdog_executable_path"]:
                raise TraceError(f"{phase} watchdog executable path is invalid")
            if data["soft_bytes"] != SOFT_MEMORY_LIMIT or (
                    data["emergency_bytes"] != WATCHDOG_EMERGENCY_LIMIT) or (
                    data["strict_ceiling_bytes"] != STRICT_MEMORY_LIMIT):
                raise TraceError(f"{phase} watchdog audit thresholds are invalid")
            if data["grace_seconds"] != 30.0 or data["sample_interval_seconds"] != 1.0:
                raise TraceError(f"{phase} watchdog timing policy is invalid")
            if data["procfs_root"] != "/proc":
                raise TraceError(f"{phase} watchdog procfs root is invalid")
            if type(data["guardian_pid"]) is not int or data["guardian_pid"] <= 1 or (
                    type(data["child_pid"]) is not int or data["child_pid"] <= 1) or (
                    type(data["child_process_group_id"]) is not int or data["child_process_group_id"] <= 1):
                raise TraceError(f"{phase} watchdog child identity is invalid")
            for key in ("audit_device", "audit_inode", "audit_uid", "audit_fd"):
                if type(data[key]) is not int or data[key] < 0:
                    raise TraceError(f"{phase} watchdog audit {key} is invalid")
            if data["audit_mode"] != 0o600:
                raise TraceError(f"{phase} watchdog audit mode is invalid")
            if not isinstance(data["command"], list) or not data["command"] or (
                    not all(isinstance(argument, str) for argument in data["command"])):
                raise TraceError(f"{phase} watchdog child command is invalid")
            canonical_command = json.dumps(
                data["command"], ensure_ascii=True, separators=(",", ":")).encode("utf-8")
            if data["child_command_sha256"] != sha256_bytes(canonical_command):
                raise TraceError(f"{phase} watchdog child command SHA-256 is invalid")
            if not isinstance(data["watchdog_command_sha256"], str) or re.fullmatch(
                    r"[0-9a-f]{64}", data["watchdog_command_sha256"]) is None:
                raise TraceError(f"{phase} watchdog audit command SHA-256 is invalid")
            if type(data["heartbeat_unix"]) is not int or data["heartbeat_unix"] <= 0:
                raise TraceError(f"{phase} watchdog audit heartbeat timestamp is invalid")
            namespace_authority = data["namespace_authority"]
            if not isinstance(namespace_authority, dict):
                raise TraceError(f"{phase} watchdog namespace authority is invalid")
            _require_exact_keys(
                namespace_authority,
                {
                    "format",
                    "version",
                    "mechanism",
                    "descriptor",
                    "host_procfs_root",
                    "watchdog_pid",
                    "watchdog_process_group_id",
                    "watchdog_start_time_ticks",
                    "watchdog_executable_path",
                    "watchdog_command_sha256",
                    "guardian_pid",
                    "child_pid",
                    "child_process_group_id",
                },
                f"{phase} watchdog namespace authority",
            )
            if namespace_authority["format"] != "dsv41-watchdog-namespace-authority" or (
                    namespace_authority["version"] != 1) or (
                    namespace_authority["mechanism"] != "inherited-pidfd") or (
                    namespace_authority["host_procfs_root"] != "/proc"):
                raise TraceError(f"{phase} watchdog namespace authority policy is invalid")
            authority_checks = {
                "watchdog_pid": data["watchdog_pid"],
                "watchdog_start_time_ticks": data["watchdog_start_time_ticks"],
                "watchdog_executable_path": data["watchdog_executable_path"],
                "watchdog_command_sha256": data["watchdog_command_sha256"],
                "guardian_pid": data["guardian_pid"],
                "child_pid": data["child_pid"],
                "child_process_group_id": data["child_process_group_id"],
            }
            for key, value in authority_checks.items():
                if namespace_authority.get(key) != value:
                    raise TraceError(f"{phase} watchdog namespace authority {key} mismatch")
            for key in ("descriptor", "watchdog_process_group_id"):
                if type(namespace_authority[key]) is not int or namespace_authority[key] <= 2:
                    raise TraceError(f"{phase} watchdog namespace authority {key} is invalid")
            max_age = data.get("max_heartbeat_age_seconds")
            if not isinstance(max_age, (int, float)) or isinstance(max_age, bool) or (
                    max_age <= 0 or max_age > 30):
                raise TraceError(f"{phase} watchdog audit heartbeat age is invalid")
            if data["heartbeat_unix"] > record["created_unix"] or (
                    record["created_unix"] - data["heartbeat_unix"] > max_age):
                raise TraceError(f"{phase} watchdog audit heartbeat was stale when captured")
            audit_jsonl = data["audit"]
            if not isinstance(audit_jsonl, dict):
                raise TraceError(f"{phase} watchdog JSONL reference is invalid")
            _require_exact_keys(
                audit_jsonl,
                {"path", "sha256", "event_count"},
                f"{phase} watchdog JSONL reference",
            )
            jsonl_digest = audit_jsonl.get("sha256", "")
            if not isinstance(jsonl_digest, str) or re.fullmatch(r"[0-9a-f]{64}", jsonl_digest) is None:
                raise TraceError(f"{phase} watchdog JSONL SHA-256 is invalid")
            if data["audit_sha256"] != jsonl_digest:
                raise TraceError(f"{phase} watchdog live and embedded audit SHA-256 differ")
            if audit_jsonl.get("path") != f"audits/{phase}/{jsonl_digest}.jsonl":
                raise TraceError(f"{phase} watchdog JSONL path is invalid")
            if type(audit_jsonl.get("event_count")) is not int or audit_jsonl["event_count"] < 2:
                raise TraceError(f"{phase} watchdog JSONL event count is invalid")
            try:
                jsonl_bytes = self._read_verified_file(audit_jsonl["path"])
            except TraceError as error:
                raise TraceError(f"cannot read {phase} watchdog JSONL audit: {error}") from error
            if sha256_bytes(jsonl_bytes) != jsonl_digest:
                raise TraceError(f"{phase} watchdog JSONL SHA-256 mismatch")
            lines = jsonl_bytes.decode("ascii").splitlines()
            if len(lines) != audit_jsonl["event_count"]:
                raise TraceError(f"{phase} watchdog JSONL event count mismatch")
            try:
                events = [validate_watchdog_event(strict_json_loads(line)) for line in lines]
            except TraceError as error:
                raise TraceError(f"{phase} watchdog JSONL is invalid: {error}") from error
            if not any(event.get("event") == "preflight" for event in events) or (
                    not any(event.get("event") == "child_started" for event in events)):
                raise TraceError(f"{phase} watchdog JSONL lacks startup evidence")

    def read_blob(self, event: dict[str, Any]) -> bytes:
        return self._read_verified_file(event["blob"])

    def _read_verified_file(self, relative: str) -> bytes:
        retained = self._retained_files.get(relative)
        if retained is not None:
            return retained
        expected = self._file_receipts.get(relative)
        if expected is None:
            raise TraceError(f"trace file is outside the signed domain: {relative}")
        receipt, data = _read_bundle_file(self.root, relative, retain=True)
        if receipt != expected:
            raise TraceError(f"trace bundle file changed after signature verification: {relative}")
        assert data is not None
        return data

    def _path(self, relative: str) -> Path:
        parts = _bundle_path_parts(relative)
        candidate = self.root
        for part in parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise TraceError(f"trace path must not use symlinks: {relative}")
        try:
            candidate.resolve().relative_to(self.root)
        except ValueError as error:
            raise TraceError(f"trace path is outside the bundle: {relative}") from error
        return candidate

    def _validate_coverage(self) -> None:
        expected = self.manifest.get("expected")
        if not isinstance(expected, dict):
            raise TraceError("manifest expected coverage is missing")
        _require_exact_keys(
            expected,
            {"prompt_tokens", "decode_steps", "components"},
            "manifest expected coverage",
        )
        prompt_tokens = expected.get("prompt_tokens")
        decode_steps = expected.get("decode_steps")
        components = expected.get("components")
        if type(prompt_tokens) is not int or prompt_tokens <= 0:
            raise TraceError("expected prompt_tokens is invalid")
        if type(decode_steps) is not int or decode_steps <= 0:
            raise TraceError("expected decode_steps is invalid")
        if self.manifest.get("config", {}).get("decode_steps") != decode_steps:
            raise TraceError("expected decode_steps does not match config")
        if prompt_tokens != self.manifest.get("prompt", {}).get("target_tokens"):
            raise TraceError("expected prompt_tokens does not match prompt provenance")
        for event in self.events:
            self._validate_component_schema(event)
        raw_attention = {
            (event["phase"], event["step"], event["token_start"], event["layer"]): event
            for event in self.events
            if event["component"] == "attn.source" and event["layer"] in RAW_ATTENTION_LAYERS
        }
        raw_coordinates = {
            (phase, step, token_start)
            for phase, step, token_start, _layer in raw_attention
        }
        for phase, step, token_start in raw_coordinates:
            layer0 = raw_attention.get((phase, step, token_start, 0))
            layer1 = raw_attention.get((phase, step, token_start, 1))
            if layer0 is None or layer1 is None:
                continue
            if layer0["shape"] != layer1["shape"] or self.read_blob(layer0) != self.read_blob(layer1):
                raise TraceError(
                    f"raw attn.source differs between layers 0 and 1 at "
                    f"{phase} step {step} token {token_start}")
        if not isinstance(components, dict):
            raise TraceError("expected components are invalid")
        if self.manifest.get("model", {}).get("architecture") == "deepseek41":
            if components != DEEPSEEK41_EXPECTED_COMPONENTS:
                raise TraceError("DeepSeek V4.1 expected component coverage contract is invalid")

        by_component: dict[str, list[dict[str, Any]]] = {}
        for event in self.events:
            by_component.setdefault(event["component"], []).append(event)
        for component in HARD_FAILURE_COMPONENTS:
            if component not in components:
                raise TraceError(f"expected coverage lacks {component}")
            rules = components[component]
            events = by_component.get(component, [])
            if not events:
                raise TraceError(f"trace lacks required component: {component}")
            expected_layers = rules.get("layers")
            if expected_layers is not None:
                actual_layers = sorted({event["layer"] for event in events})
                if actual_layers != expected_layers:
                    raise TraceError(
                        f"{component} layer coverage mismatch: expected {expected_layers}, found {actual_layers}")
            input_coverage = rules.get("input")
            prefill_coverage = rules.get("prefill")
            decode_coverage = rules.get("decode")
            if input_coverage == "tokens":
                input_events = [event for event in events if event["phase"] == "input"]
                if len(input_events) != 1 or input_events[0]["token_start"] != 0 or (
                        input_events[0]["token_count"] != prompt_tokens):
                    raise TraceError(f"{component} input coverage is incomplete")
            for layer in expected_layers or [None]:
                layer_events = [event for event in events if event["layer"] == layer]
                if prefill_coverage == "tokens":
                    prefill = sorted(
                        (event for event in layer_events if event["phase"] == "prefill"),
                        key=lambda event: event["token_start"],
                    )
                    frontier = 0
                    for event in prefill:
                        if event["token_start"] != frontier or event["token_count"] <= 0:
                            raise TraceError(f"{component} prefill coverage has a gap or overlap at token {frontier}")
                        frontier += event["token_count"]
                    if frontier != prompt_tokens:
                        raise TraceError(
                            f"{component} prefill coverage ends at {frontier}, expected {prompt_tokens}")
                elif prefill_coverage == "final":
                    prefill = [event for event in layer_events if event["phase"] == "prefill"]
                    if len(prefill) != 1 or prefill[0]["token_start"] != prompt_tokens - 1 or prefill[0]["token_count"] != 1:
                        raise TraceError(f"{component} final prefill coverage is invalid")
                if decode_coverage == "steps":
                    decode = sorted(
                        (event for event in layer_events if event["phase"] == "decode"),
                        key=lambda event: event["step"],
                    )
                    if [event["step"] for event in decode] != list(range(decode_steps)):
                        raise TraceError(f"{component} decode step coverage is incomplete")
                    for event in decode:
                        if event["token_start"] != prompt_tokens + event["step"] or event["token_count"] != 1:
                            raise TraceError(f"{component} decode token coordinates are invalid")

    def _validate_component_schema(self, event: dict[str, Any]) -> None:
        component = event["component"]
        phase = event["phase"]
        layer = event["layer"]
        dtype = event["dtype"]
        shape = event["shape"]
        token_count = event["token_count"]
        config = self.manifest["config"].get("deepseek41")
        if not isinstance(config, dict):
            raise TraceError("DeepSeek V4.1 component schema configuration is missing")

        if component == "prompt.bytes":
            if phase != "input" or layer is not None or dtype != "bytes" or len(shape) != 1:
                raise TraceError("prompt.bytes schema is invalid")
            if shape[0] != self.manifest["prompt"]["byte_count"]:
                raise TraceError("prompt.bytes length does not match the manifest")
            if event["sha256"] != self.manifest["prompt"]["sha256"]:
                raise TraceError("prompt.bytes SHA-256 does not match the manifest")
            return
        if component == "prompt.tokens":
            if phase != "input" or layer is not None or dtype != "i32" or shape != [token_count]:
                raise TraceError("prompt.tokens schema is invalid")
            return
        if component in ("logits.prefill", "logits.decode"):
            if phase not in ("prefill", "decode") or layer is not None or dtype != "f32":
                raise TraceError(f"{component} schema is invalid")
            if token_count != 1 or shape != [config.get("vocab_size")]:
                raise TraceError(f"{component} must contain one complete vocabulary-sized logit vector")
            return
        if component == "decode.greedy_token":
            if phase != "decode" or layer is not None or dtype != "i32" or token_count != 1 or shape != [1]:
                raise TraceError("decode.greedy_token schema is invalid")
            return

        if phase not in ("prefill", "decode") or layer is None:
            raise TraceError(f"{component} phase or layer is invalid")
        if len(shape) != 2 or shape[1] != token_count:
            raise TraceError(f"{component} second dimension must equal token_count")
        widths = {
            "engram.row_ids": ("i32", config.get("engram_rows_per_token")),
            "expert.ids": ("i32", config.get("experts_used")),
            "expert.weights": ("f32", config.get("experts_used")),
            "attn.source": ("i32", None),
            "attn.candidate_blocks": ("i32", None),
            "attn.candidates": ("i32", None),
        }
        if component not in widths:
            raise TraceError(f"unsupported trace component: {component}")
        expected_dtype, width = widths[component]
        if dtype != expected_dtype:
            raise TraceError(f"{component} dtype must be {expected_dtype}")
        if width is not None and shape[0] != width:
            raise TraceError(f"{component} shape must be [{width},token_count]")
        if component == "attn.source" and layer in RAW_ATTENTION_LAYERS:
            if shape[0] != RAW_ATTENTION_WIDTH:
                raise TraceError(f"raw attn.source shape must be [{RAW_ATTENTION_WIDTH},token_count]")
            values = list(struct.iter_unpack("<i", self.read_blob(event)))
            sentinel = RAW_ATTENTION_WIDTH + token_count
            for token in range(token_count):
                for row in range(RAW_ATTENTION_WIDTH):
                    value = values[token*RAW_ATTENTION_WIDTH + row][0]
                    if 0 <= value < RAW_ATTENTION_WIDTH or (
                            RAW_ATTENTION_WIDTH <= value <= RAW_ATTENTION_WIDTH + token) or value == sentinel:
                        continue
                    raise TraceError(
                        f"raw attn.source contains invalid row {value} for token {token}; "
                        f"expected physical row, visible ubatch row, or sentinel {sentinel}")
        if component == "attn.candidate_blocks" and shape[0] > config.get("candidate_topk_blocks", 0):
            raise TraceError("attn.candidate_blocks width exceeds candidate_topk_blocks")
        if component == "attn.source" and layer not in RAW_ATTENTION_LAYERS and (
                shape[0] > config.get("index_top_k", 0)):
            raise TraceError(f"{component} width exceeds index_top_k")
        if component == "attn.candidates" and shape[0] > config.get("index_top_k", 0):
            raise TraceError(f"{component} width exceeds index_top_k")
        if component == "expert.ids":
            values = struct.iter_unpack("<i", self.read_blob(event))
            if any(value < 0 or value >= config.get("expert_count", 0) for value, in values):
                raise TraceError("expert.ids contains an out-of-range original expert ID")


_TRACE_BUNDLE_TYPE = TraceBundle


def first_byte_difference(left: bytes, right: bytes) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def compare_manifests(left: TraceBundle, right: TraceBundle) -> Mismatch | None:
    checks = (
        ("model.sha256", "model_identity"),
        ("model.architecture", "model_identity"),
        ("prompt.sha256", "prompt_identity"),
        ("prompt.byte_count", "prompt_identity"),
        ("expected.prompt_tokens", "tokenizer"),
        ("config.context", "configuration"),
        ("config.decode_steps", "configuration"),
        ("config.deepseek41", "configuration"),
        ("comparison.logits", "comparison_policy"),
    )
    for dotted, classification in checks:
        left_value: Any = left.manifest
        right_value: Any = right.manifest
        for key in dotted.split("."):
            left_value = left_value.get(key) if isinstance(left_value, dict) else None
            right_value = right_value.get(key) if isinstance(right_value, dict) else None
        if left_value != right_value:
            return Mismatch(
                classification,
                "manifest",
                "metadata",
                -1,
                -1,
                None,
                f"{dotted} differs: {left_value!r} != {right_value!r}",
            )
    return None


def compare_bundles(
        left: TraceBundle,
        right: TraceBundle,
        *,
        runtime_roles: tuple[str, str] = ("ds4", "llama.cpp")) -> Mismatch | None:
    if left.root.resolve() == right.root.resolve():
        return Mismatch(
            "artifact_identity",
            "manifest",
            "metadata",
            -1,
            -1,
            None,
            "cannot compare a trace bundle with itself",
        )
    if left.manifest["runtime"] != runtime_roles[0] or right.manifest["runtime"] != runtime_roles[1]:
        return Mismatch(
            "runtime_role",
            "manifest",
            "metadata",
            -1,
            -1,
            None,
            f"left trace must be {runtime_roles[0]} and right trace must be {runtime_roles[1]}",
        )
    mismatch = compare_manifests(left, right)
    if mismatch is not None:
        return mismatch

    for bundle_name, bundle in (("left", left), ("right", right)):
        components = {event["component"] for event in bundle.events}
        missing = sorted(set(HARD_FAILURE_COMPONENTS) - components)
        if missing:
            return Mismatch(
                "artifact_missing",
                "manifest",
                "metadata",
                -1,
                -1,
                None,
                f"{bundle_name} trace is missing required components: {', '.join(missing)}",
            )

    left_map = {event_key(event): event for event in left.events}
    right_map = {event_key(event): event for event in right.events}
    if len(left_map) != len(left.events) or len(right_map) != len(right.events):
        return Mismatch(
            "artifact_duplicate",
            "events",
            "metadata",
            -1,
            -1,
            None,
            "a trace contains duplicate phase/step/token/layer/component coordinates",
        )
    mismatches = []
    all_keys = sorted(set(left_map) | set(right_map), key=event_order)
    for key in all_keys:
        left_event = left_map.get(key)
        right_event = right_map.get(key)
        template = left_event or right_event
        assert template is not None
        if left_event is None or right_event is None:
            mismatches.append(Mismatch(
                "artifact_missing",
                template["component"],
                template["phase"],
                template["step"],
                template["token_start"],
                template["layer"],
                "event is missing from " + ("left" if left_event is None else "right") + " trace",
                token_index=template["token_start"],
            ))
            continue
        shape_mismatch = False
        for field in ("dtype", "shape", "token_count", "semantic_id_space"):
            if left_event.get(field) != right_event.get(field):
                mismatches.append(Mismatch(
                    "artifact_shape",
                    template["component"],
                    template["phase"],
                    template["step"],
                    template["token_start"],
                    template["layer"],
                    f"{field} differs: {left_event.get(field)!r} != {right_event.get(field)!r}",
                    token_index=template["token_start"],
                ))
                shape_mismatch = True
                break
        if shape_mismatch:
            continue
        if left_event["sha256"] == right_event["sha256"]:
            continue
        left_data = left.read_blob(left_event)
        right_data = right.read_blob(right_event)
        byte_offset = first_byte_difference(left_data, right_data)
        assert byte_offset is not None
        item_size = DTYPE_SIZES[left_event["dtype"]]
        flat_element_index = byte_offset // item_size
        token_index = None
        component_element_index = None
        token_count = left_event["token_count"]
        elements = element_count(left_event["shape"])
        if token_count > 0 and elements % token_count == 0:
            elements_per_token = elements // token_count
            token_index = left_event["token_start"] + flat_element_index // elements_per_token
            component_element_index = flat_element_index % elements_per_token
        detail = f"first byte mismatch at {byte_offset}"
        item_offset = byte_offset - byte_offset % item_size
        if item_offset + item_size <= min(len(left_data), len(right_data)):
            left_item = left_data[item_offset:item_offset + item_size]
            right_item = right_data[item_offset:item_offset + item_size]
            detail += f"; left=0x{left_item.hex()} right=0x{right_item.hex()}"
        mismatches.append(Mismatch(
            classify(template["component"]),
            template["component"],
            template["phase"],
            template["step"],
            template["token_start"],
            template["layer"],
            detail,
            element_index=flat_element_index,
            byte_offset=byte_offset,
            token_index=token_index,
            component_element_index=component_element_index,
        ))
    if not mismatches:
        return None
    phase_order = {"input": 0, "prefill": 1, "decode": 2, "metadata": -1}
    component_order = {component: index for index, component in enumerate(HARD_FAILURE_COMPONENTS)}
    return min(mismatches, key=lambda item: (
        phase_order.get(item.phase, 99),
        item.token_index if item.token_index is not None else item.token_start,
        item.step,
        -1 if item.layer is None else item.layer,
        component_order.get(item.component, 99),
        item.component,
    ))


def report(
        left: TraceBundle,
        right: TraceBundle,
        *,
        runtime_roles: tuple[str, str] = ("ds4", "llama.cpp"),
        success_status: str = "TARGET PASS") -> dict[str, Any]:
    mismatch = compare_bundles(left, right, runtime_roles=runtime_roles)
    if mismatch is None:
        return {
            "status": success_status,
            "trace_version": TRACE_VERSION,
            "left_runtime": left.manifest.get("runtime"),
            "right_runtime": right.manifest.get("runtime"),
            "events_compared": len(left.events),
            "left_seal_sha256": left.seal_sha256,
            "right_seal_sha256": right.seal_sha256,
            "left_signer_principal": left.signer_principal,
            "right_signer_principal": right.signer_principal,
            "first_divergence": None,
        }
    return {
        "status": "FAIL",
        "trace_version": TRACE_VERSION,
        "left_runtime": left.manifest.get("runtime"),
        "right_runtime": right.manifest.get("runtime"),
        "events_compared": 0,
        "left_seal_sha256": left.seal_sha256,
        "right_seal_sha256": right.seal_sha256,
        "left_signer_principal": left.signer_principal,
        "right_signer_principal": right.signer_principal,
        "first_divergence": mismatch.as_dict(),
    }


def command_validate(args: argparse.Namespace) -> int:
    if hasattr(args, "approval_policy"):
        approval_policy = load_executable_approval_policy(
            args.approval_policy,
            args.approval_signature,
            expected_principal=args.approval_principal,
            forbidden_roots=(args.bundle,),
        )
        bundle = TraceBundle(
            args.bundle,
            verifier=TraceVerifier.production(
                args.signer_principal,
                expected_lane=args.lane,
                expected_challenge=args.execution_challenge,
                expected_run_id=args.run_id,
                expected_candidate_exporter_policy_id=args.candidate_exporter_policy_id,
                expected_ds4_exporter_policy_id=args.ds4_exporter_policy_id,
                expected_prompt_builder_policy_id=args.prompt_builder_policy_id,
                approval_policy=approval_policy,
                verification_unix=None,
            ),
        )
    else:
        bundle = TraceBundle(args.bundle)
    print(canonical_json({
        "status": "valid",
        "runtime": bundle.manifest.get("runtime"),
        "events": len(bundle.events),
    }))
    return 0


def command_compare(args: argparse.Namespace) -> int:
    left_principal = getattr(args, "left_signer_principal", None)
    right_principal = getattr(args, "right_signer_principal", None)
    challenge = getattr(args, "execution_challenge", None)
    left_run_id = getattr(args, "left_run_id", None)
    right_run_id = getattr(args, "right_run_id", None)
    approval_policy = (
        load_executable_approval_policy(
            args.approval_policy,
            args.approval_signature,
            expected_principal=args.approval_principal,
            forbidden_roots=(args.left, args.right),
        )
        if hasattr(args, "approval_policy") else None
    )
    seen_run_ids: set[str] = set()
    if approval_policy is None:
        left_bundle = TraceBundle(
            args.left,
            signer_principal=left_principal,
            expected_lane=ORACLE_LANE,
            expected_challenge=challenge,
            expected_run_id=left_run_id,
            expected_candidate_exporter_policy_id=None,
            expected_ds4_exporter_policy_id=getattr(args, "left_ds4_exporter_policy_id", None),
            expected_prompt_builder_policy_id=getattr(args, "prompt_builder_policy_id", None),
            seen_run_ids=seen_run_ids,
        )
        right_bundle = TraceBundle(
            args.right,
            signer_principal=right_principal,
            expected_lane=CANDIDATE_LANE,
            expected_challenge=challenge,
            expected_run_id=right_run_id,
            expected_candidate_exporter_policy_id=getattr(
                args, "right_candidate_exporter_policy_id", None),
            expected_ds4_exporter_policy_id=None,
            expected_prompt_builder_policy_id=getattr(args, "prompt_builder_policy_id", None),
            seen_run_ids=seen_run_ids,
        )
    else:
        left_bundle = TraceBundle(
            args.left,
            verifier=TraceVerifier.production(
                left_principal,
                expected_lane=ORACLE_LANE,
                expected_challenge=challenge,
                expected_run_id=left_run_id,
                expected_candidate_exporter_policy_id=None,
                expected_ds4_exporter_policy_id=args.left_ds4_exporter_policy_id,
                expected_prompt_builder_policy_id=args.prompt_builder_policy_id,
                approval_policy=approval_policy,
                verification_unix=None,
                seen_run_ids=seen_run_ids,
            ),
        )
        right_bundle = TraceBundle(
            args.right,
            verifier=TraceVerifier.production(
                right_principal,
                expected_lane=CANDIDATE_LANE,
                expected_challenge=challenge,
                expected_run_id=right_run_id,
                expected_candidate_exporter_policy_id=args.right_candidate_exporter_policy_id,
                expected_ds4_exporter_policy_id=None,
                expected_prompt_builder_policy_id=args.prompt_builder_policy_id,
                approval_policy=approval_policy,
                verification_unix=None,
                seen_run_ids=seen_run_ids,
            ),
        )
    result = report(
        left_bundle,
        right_bundle,
    )
    text = canonical_json(result) + "\n"
    if args.report:
        args.report.write_text(text, encoding="ascii")
    sys.stdout.write(text)
    return 0 if result["status"] == "TARGET PASS" else 1


def local_report(left: TraceBundle, right: TraceBundle, mode: str) -> dict[str, Any]:
    result = report(
        left,
        right,
        runtime_roles=("llama.cpp", "llama.cpp"),
        success_status="BRINGUP PASS",
    )
    result["mode"] = mode
    result["cross_runtime_status"] = "INCOMPLETE"
    result["cross_runtime_requirement"] = (
        "Run the pinned ds4 exporter and trace_format.py compare before reporting TARGET PASS.")
    if mode == "self-consistency":
        if left.manifest.get("candidate") != right.manifest.get("candidate"):
            result = {
                **result,
                "status": "FAIL",
                "first_divergence": Mismatch(
                    "candidate_identity",
                    "manifest",
                    "metadata",
                    -1,
                    -1,
                    None,
                    "self-consistency traces use different candidate attestations",
                ).as_dict(),
            }
    else:
        base_revision = left.manifest.get("candidate", {}).get("revision")
        integrated_base = right.manifest.get("candidate", {}).get("base_revision")
        if base_revision != integrated_base:
            result = {
                **result,
                "status": "FAIL",
                "first_divergence": Mismatch(
                    "candidate_identity",
                    "manifest",
                    "metadata",
                    -1,
                    -1,
                    None,
                    f"base trace revision {base_revision!r} != integrated oracle {integrated_base!r}",
                ).as_dict(),
            }
    return result


def command_compare_local(args: argparse.Namespace) -> int:
    left_principal = getattr(args, "left_signer_principal", None)
    right_principal = getattr(args, "right_signer_principal", None)
    challenge = getattr(args, "execution_challenge", None)
    left_run_id = getattr(args, "left_run_id", None)
    right_run_id = getattr(args, "right_run_id", None)
    approval_policy = (
        load_executable_approval_policy(
            args.approval_policy,
            args.approval_signature,
            expected_principal=args.approval_principal,
            forbidden_roots=(args.left, args.right),
        )
        if hasattr(args, "approval_policy") else None
    )
    seen_run_ids: set[str] = set()
    if approval_policy is None:
        left_bundle = TraceBundle(
            args.left,
            signer_principal=left_principal,
            expected_lane=CANDIDATE_LANE,
            expected_challenge=challenge,
            expected_run_id=left_run_id,
            expected_candidate_exporter_policy_id=getattr(
                args, "left_candidate_exporter_policy_id", None),
            expected_ds4_exporter_policy_id=None,
            expected_prompt_builder_policy_id=getattr(
                args, "left_prompt_builder_policy_id", None),
            seen_run_ids=seen_run_ids,
        )
        right_bundle = TraceBundle(
            args.right,
            signer_principal=right_principal,
            expected_lane=CANDIDATE_LANE,
            expected_challenge=challenge,
            expected_run_id=right_run_id,
            expected_candidate_exporter_policy_id=getattr(
                args, "right_candidate_exporter_policy_id", None),
            expected_ds4_exporter_policy_id=None,
            expected_prompt_builder_policy_id=getattr(
                args, "right_prompt_builder_policy_id", None),
            seen_run_ids=seen_run_ids,
        )
    else:
        left_bundle = TraceBundle(
            args.left,
            verifier=TraceVerifier.production(
                left_principal,
                expected_lane=CANDIDATE_LANE,
                expected_challenge=challenge,
                expected_run_id=left_run_id,
                expected_candidate_exporter_policy_id=args.left_candidate_exporter_policy_id,
                expected_ds4_exporter_policy_id=None,
                expected_prompt_builder_policy_id=args.left_prompt_builder_policy_id,
                approval_policy=approval_policy,
                verification_unix=None,
                seen_run_ids=seen_run_ids,
            ),
        )
        right_bundle = TraceBundle(
            args.right,
            verifier=TraceVerifier.production(
                right_principal,
                expected_lane=CANDIDATE_LANE,
                expected_challenge=challenge,
                expected_run_id=right_run_id,
                expected_candidate_exporter_policy_id=args.right_candidate_exporter_policy_id,
                expected_ds4_exporter_policy_id=None,
                expected_prompt_builder_policy_id=args.right_prompt_builder_policy_id,
                approval_policy=approval_policy,
                verification_unix=None,
                seen_run_ids=seen_run_ids,
            ),
        )
    result = local_report(
        left_bundle,
        right_bundle,
        args.mode,
    )
    text = canonical_json(result) + "\n"
    if args.report:
        args.report.write_text(text, encoding="ascii")
    sys.stdout.write(text)
    return 0 if result["status"] == "BRINGUP PASS" else 1


def add_executable_approval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--approval-policy", type=Path, required=True)
    parser.add_argument("--approval-signature", type=Path, required=True)
    parser.add_argument("--approval-principal", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and compare DeepSeek V4.1 correctness traces")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("bundle", type=Path)
    validate_parser.add_argument("--signer-principal", required=True)
    validate_parser.add_argument("--lane", choices=(CANDIDATE_LANE, ORACLE_LANE), required=True)
    validate_parser.add_argument("--execution-challenge", required=True)
    validate_parser.add_argument("--run-id", required=True)
    validate_parser.add_argument("--candidate-exporter-policy-id")
    validate_parser.add_argument("--ds4-exporter-policy-id")
    validate_parser.add_argument("--prompt-builder-policy-id", required=True)
    add_executable_approval_arguments(validate_parser)
    validate_parser.set_defaults(func=command_validate)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("left", type=Path)
    compare_parser.add_argument("right", type=Path)
    compare_parser.add_argument("--left-signer-principal", required=True)
    compare_parser.add_argument("--right-signer-principal", required=True)
    compare_parser.add_argument("--execution-challenge", required=True)
    compare_parser.add_argument("--left-run-id", required=True)
    compare_parser.add_argument("--right-run-id", required=True)
    compare_parser.add_argument("--left-ds4-exporter-policy-id", required=True)
    compare_parser.add_argument("--right-candidate-exporter-policy-id", required=True)
    compare_parser.add_argument("--prompt-builder-policy-id", required=True)
    add_executable_approval_arguments(compare_parser)
    compare_parser.add_argument("--report", type=Path)
    compare_parser.set_defaults(func=command_compare)
    local_parser = subparsers.add_parser("compare-local")
    local_parser.add_argument("mode", choices=("self-consistency", "base-regression"))
    local_parser.add_argument("left", type=Path)
    local_parser.add_argument("right", type=Path)
    local_parser.add_argument("--left-signer-principal", required=True)
    local_parser.add_argument("--right-signer-principal", required=True)
    local_parser.add_argument("--execution-challenge", required=True)
    local_parser.add_argument("--left-run-id", required=True)
    local_parser.add_argument("--right-run-id", required=True)
    local_parser.add_argument("--left-candidate-exporter-policy-id", required=True)
    local_parser.add_argument("--right-candidate-exporter-policy-id", required=True)
    local_parser.add_argument("--left-prompt-builder-policy-id", required=True)
    local_parser.add_argument("--right-prompt-builder-policy-id", required=True)
    add_executable_approval_arguments(local_parser)
    local_parser.add_argument("--report", type=Path)
    local_parser.set_defaults(func=command_compare_local)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
