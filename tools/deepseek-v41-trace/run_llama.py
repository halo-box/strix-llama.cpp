#!/usr/bin/env python3

import argparse
import copy
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
from pathlib import Path

from preflight import (
    PreflightError,
    bind_embedded_audits,
    bind_prompt_provenance,
    open_watchdog_namespace_authority,
    resolved,
    run_strix_preflight,
    safe_trace_path,
    seal_audits,
    validate_prompt_provenance,
    verify_watchdog_namespace_authority,
    verify_sealed_audits,
    write_audits,
)
from trace_format import (
    ADMITTED_BATCH,
    ADMITTED_UBATCH,
    APPROVED_CANDIDATE_EXPORTERS,
    APPROVED_PROMPT_BUILDERS,
    APPROVED_TRACE_SIGNERS,
    CANDIDATE_LANE,
    CORPUS_SHA256,
    MODEL_SHA256,
    REPOSITORY,
    REQUIRED_EXPERT_CACHE_BYTES,
    REQUIRED_EXPERT_CACHE_MIB,
    REQUIRED_EXPERT_SLOTS,
    TraceBundle,
    TraceError,
    TraceVerifier,
    approval_binding,
    approved_containment_helper_identity,
    approved_executable_identity,
    approved_runtime_file_identities,
    bind_execution_authorization,
    candidate_exporter_approval,
    canonical_json,
    execution_authorization,
    install_trust_evidence,
    install_trust_sha256,
    load_executable_approval_policy,
    reject_loader_overrides,
    run_approved_executable,
    seal_bundle,
    sha256_bytes,
    sha256_file,
    strict_json_loads,
    tokenizer_policy_sha256,
    prompt_builder_approval,
    validate_signing_identity,
    validate_tokenizer_policy,
    verify_approved_executable_identity,
    verify_approved_runtime_file_identities,
)


def approved_source_root(prompt_policy: dict[str, object]) -> Path:
    try:
        return Path(str(prompt_policy["source_root"])).expanduser().resolve(strict=True)
    except (KeyError, OSError) as error:
        raise PreflightError(f"prompt builder approved source root is invalid: {error}") from error


def git_output(repo: Path, *args: str) -> bytes:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as error:
        raise PreflightError(f"git {' '.join(args)} failed: {error}") from error


def sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    try:
        while True:
            chunk = os.pread(descriptor, 8 * 1024 * 1024, offset)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)
            offset += len(chunk)
    except AttributeError as error:
        raise PreflightError("descriptor hashing requires os.pread") from error
    except OSError as error:
        raise PreflightError(f"cannot hash held model descriptor: {error}") from error


def model_descriptor_identity(
        descriptor: int,
        path: Path,
        *,
        hash_bytes: bool,
) -> dict[str, object]:
    try:
        import fcntl

        record = os.fstat(descriptor)
        status_flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
        descriptor_flags = fcntl.fcntl(descriptor, fcntl.F_GETFD)
    except (ImportError, OSError) as error:
        raise PreflightError(f"cannot inspect held model descriptor: {error}") from error
    if not stat.S_ISREG(record.st_mode):
        raise PreflightError("model descriptor is not a regular file")
    if record.st_nlink < 1:
        raise PreflightError("model descriptor has no linked pathname")
    if (status_flags & os.O_ACCMODE) != os.O_RDONLY:
        raise PreflightError("model descriptor is not read-only")
    identity = {
        "format": "dsv41-model-file-identity",
        "version": 1,
        "path": str(path),
        "device": record.st_dev,
        "inode": record.st_ino,
        "owner_uid": record.st_uid,
        "owner_gid": record.st_gid,
        "mode": stat.S_IMODE(record.st_mode),
        "link_count": record.st_nlink,
        "byte_count": record.st_size,
        "modified_ns": record.st_mtime_ns,
        "changed_ns": record.st_ctime_ns,
        "status_flags": status_flags,
        "source_descriptor_flags": descriptor_flags,
        "target_descriptor_flags": 0,
    }
    if hash_bytes:
        identity["sha256"] = sha256_descriptor(descriptor)
    return identity


def open_model_descriptor(path: Path) -> tuple[int, dict[str, object]]:
    lexical_path = path.expanduser()
    if not lexical_path.is_absolute():
        lexical_path = Path.cwd() / lexical_path
    try:
        canonical_path = lexical_path.resolve(strict=True)
    except OSError as error:
        raise PreflightError(f"cannot resolve model path: {error}") from error
    if canonical_path != lexical_path:
        raise PreflightError("model path must not contain lexical or symbolic-link aliases")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(canonical_path, flags)
        before = model_descriptor_identity(descriptor, canonical_path, hash_bytes=False)
        identity = model_descriptor_identity(descriptor, canonical_path, hash_bytes=True)
        after = model_descriptor_identity(descriptor, canonical_path, hash_bytes=False)
        if before != after:
            raise PreflightError("model descriptor identity changed while hashing")
        path_record = canonical_path.stat(follow_symlinks=False)
        if (
                path_record.st_dev,
                path_record.st_ino,
                path_record.st_uid,
                path_record.st_gid,
                stat.S_IMODE(path_record.st_mode),
                path_record.st_nlink,
                path_record.st_size,
                path_record.st_mtime_ns,
                path_record.st_ctime_ns,
        ) != (
                identity["device"],
                identity["inode"],
                identity["owner_uid"],
                identity["owner_gid"],
                identity["mode"],
                identity["link_count"],
                identity["byte_count"],
                identity["modified_ns"],
                identity["changed_ns"],
        ):
            raise PreflightError("model pathname does not identify the held descriptor")
        return descriptor, identity
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def verify_model_descriptor(
        descriptor: int,
        identity: dict[str, object],
) -> None:
    path = Path(str(identity["path"]))
    observed = model_descriptor_identity(descriptor, path, hash_bytes=True)
    if observed != identity:
        raise PreflightError("model descriptor identity or bytes changed during execution")
    try:
        path_record = path.stat(follow_symlinks=False)
    except OSError as error:
        raise PreflightError(f"cannot revalidate model pathname: {error}") from error
    if (
            path_record.st_dev,
            path_record.st_ino,
            path_record.st_uid,
            path_record.st_gid,
            stat.S_IMODE(path_record.st_mode),
            path_record.st_nlink,
            path_record.st_size,
            path_record.st_mtime_ns,
            path_record.st_ctime_ns,
    ) != (
            identity["device"],
            identity["inode"],
            identity["owner_uid"],
            identity["owner_gid"],
            identity["mode"],
            identity["link_count"],
            identity["byte_count"],
            identity["modified_ns"],
            identity["changed_ns"],
    ):
        raise PreflightError("model pathname identity changed during execution")


def candidate_attestation(
        args: argparse.Namespace,
        exporter: Path,
        exporter_sha256: str,
        approval_id: str,
        approval_sha256: str,
        approval: dict[str, object],
        verifier_revision: str,
        install_trust: dict[str, object]) -> dict[str, object]:
    repo = resolved(args.repo)
    exporter = resolved(exporter)
    observed_verifier_revision = git_output(repo, "rev-parse", "HEAD").decode("ascii").strip()
    revision = git_output(repo, "rev-parse", args.candidate_revision).decode("ascii").strip()
    base_revision = git_output(repo, "rev-parse", args.base_revision).decode("ascii").strip()
    if observed_verifier_revision != verifier_revision:
        raise PreflightError(
            f"verifier revision mismatch: expected {verifier_revision}, found {observed_verifier_revision}")
    if revision != args.candidate_revision:
        raise PreflightError(
            f"candidate revision mismatch: expected {args.candidate_revision}, found {revision}")
    if base_revision != args.base_revision:
        raise PreflightError(
            f"base revision mismatch: expected {args.base_revision}, found {base_revision}")
    try:
        subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", base_revision, revision],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        subprocess.run(
            ["git", "-C", str(repo), "diff", "--quiet"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        subprocess.run(
            ["git", "-C", str(repo), "diff", "--cached", "--quiet"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise PreflightError(f"candidate repository is not cleanly based on {base_revision}: {error}") from error
    status = git_output(repo, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise PreflightError("candidate repository has tracked or untracked changes")
    diff = git_output(repo, "diff", "--binary", "--no-ext-diff", base_revision, revision, "--")
    diff_sha256 = hashlib.sha256(diff).hexdigest()
    if diff_sha256 != args.candidate_diff_sha256:
        raise PreflightError(
            f"candidate diff SHA-256 mismatch: expected {args.candidate_diff_sha256}, found {diff_sha256}")
    expected = {
        "repository": REPOSITORY,
        "revision": revision,
        "base_revision": base_revision,
        "diff_sha256": diff_sha256,
        "executable_path": str(exporter),
        "executable_sha256": exporter_sha256,
    }
    for key, value in expected.items():
        if approval.get(key) != value:
            raise PreflightError(f"candidate {key} differs from external exporter approval")
    return {
        **expected,
        "exporter_approval_id": approval_id,
        "exporter_approval_sha256": approval_sha256,
        "install_trust": install_trust,
        "install_trust_sha256": install_trust_sha256(install_trust),
    }


def validate_runtime_build(
        manifest: dict[str, object],
        *,
        exporter: Path,
        exporter_sha256: str,
        candidate_revision: str,
        approval: dict[str, object]) -> tuple[str, str]:
    build = manifest.get("build")
    if not isinstance(build, dict):
        raise PreflightError("llama trace build identity is missing")
    exporter = resolved(exporter)
    if build.get("path") != str(exporter):
        raise PreflightError("llama trace build path does not match the executed exporter")
    if build.get("sha256") != exporter_sha256:
        raise PreflightError("llama trace build SHA-256 does not match the executed exporter")
    if manifest.get("revision") != candidate_revision:
        raise PreflightError("llama trace build revision does not match the exact candidate revision")
    if "test-only manifest harness" in str(build.get("info", "")):
        raise PreflightError("llama trace was produced by the test-only manifest harness")
    libraries = build.get("runtime_libraries")
    if not isinstance(libraries, list) or not libraries:
        raise PreflightError("llama trace runtime library identities are missing")
    post_libraries = build.get("runtime_libraries_post")
    if post_libraries != libraries:
        raise PreflightError("llama trace runtime library closure changed during trace generation")
    module_monitor = build.get("runtime_module_monitor")
    if not isinstance(module_monitor, dict) or set(module_monitor) != {
            "mechanism", "checked_after_trace", "project_additions"}:
        raise PreflightError("llama trace runtime module monitor is invalid")
    if module_monitor["mechanism"] not in {"dyld-add-image", "pre-post-snapshot"}:
        raise PreflightError("llama trace runtime module monitor mechanism is invalid")
    if module_monitor["checked_after_trace"] is not True:
        raise PreflightError("llama trace runtime module monitor did not complete")
    if module_monitor["project_additions"] != []:
        raise PreflightError("llama trace records a runtime module addition during trace generation")
    profile = build.get("runtime_profile")
    if not isinstance(profile, dict) or set(profile) != {
            "name", "components", "selected_backend_component"}:
        raise PreflightError("llama trace runtime profile is invalid")
    if profile.get("name") != "sibling-lib":
        raise PreflightError("llama trace runtime profile is not the Linux sibling-lib profile")
    components = profile.get("components")
    if not isinstance(components, list) or components != sorted(components) or (
            len(components) != len(set(components))):
        raise PreflightError("llama trace runtime profile components are invalid")
    if not {"llama-common", "llama", "ggml", "ggml-base", "ggml-hip"}.issubset(set(components)):
        raise PreflightError("llama trace runtime profile is missing required ROCm components")
    selected_backend_component = profile.get("selected_backend_component")
    if selected_backend_component != "ggml-hip":
        raise PreflightError("llama trace selected backend component is not ggml-hip")
    roles = set()
    paths = set()
    found_components = set()
    previous_path = None
    library_directory = resolved(exporter.parent.parent / "lib")
    for library in libraries:
        if not isinstance(library, dict) or set(library) != {
                "component", "filename", "path", "sha256", "role", "revision"}:
            raise PreflightError("llama trace runtime library identity is invalid")
        component = library.get("component")
        filename = library.get("filename")
        path_value = library.get("path")
        digest = library.get("sha256")
        role = library.get("role")
        revision = library.get("revision")
        if component not in components or component in found_components:
            raise PreflightError("llama trace runtime library component is invalid")
        found_components.add(component)
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise PreflightError("llama trace runtime library filename is invalid")
        expected_role = {
            "llama-common": "build-info",
            "llama": "llama",
            "ggml-base": "ggml",
            selected_backend_component: "selected-backend",
        }.get(component, f"runtime:{component}")
        if role != expected_role or role in roles:
            raise PreflightError("llama trace runtime library role is invalid")
        roles.add(role)
        if component in {"llama-common", "ggml-base"}:
            if revision != candidate_revision:
                raise PreflightError("llama trace runtime library revision mismatch")
        elif revision is not None:
            raise PreflightError("llama trace runtime library revision is unexpected")
        if not isinstance(path_value, str):
            raise PreflightError("llama trace runtime library path is invalid")
        path = resolved(Path(path_value))
        if path_value != str(path):
            raise PreflightError("llama trace runtime library path is not canonical")
        if path in paths or (previous_path is not None and str(path) <= str(previous_path)):
            raise PreflightError("llama trace runtime library paths are duplicated or unsorted")
        paths.add(path)
        previous_path = path
        if path.parent != library_directory or path.name != filename:
            raise PreflightError("llama trace runtime library path differs from the exact runtime profile")
        if not path.is_file() or not isinstance(digest, str) or sha256_file(path) != digest:
            raise PreflightError("llama trace runtime library SHA-256 mismatch")
    if found_components != set(components):
        raise PreflightError("llama trace runtime library set differs from the runtime profile")
    receipt = {
        "format": "dsv41-runtime-receipt",
        "version": 1,
        "revision": candidate_revision,
        "profile": profile["name"],
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
    receipt_sha256 = sha256_bytes(canonical_json(receipt).encode("ascii"))
    if build.get("runtime_receipt_sha256") != receipt_sha256:
        raise PreflightError("llama trace runtime receipt SHA-256 mismatch")
    if approval.get("runtime_profile") != profile or approval.get("runtime_receipt") != receipt:
        raise PreflightError("llama trace runtime receipt differs from external exporter approval")
    closure = {"pre": libraries, "post": post_libraries}
    return sha256_bytes(canonical_json(closure).encode("ascii")), receipt_sha256


def query_runtime_build_attestation(
        exporter: Path,
        device: str,
        *,
        exporter_sha256: str,
        candidate_revision: str,
        approval: dict[str, object]) -> dict[str, object]:
    result, _identity = run_approved_executable(
        [str(exporter), "--dsv41-attest-build", device],
        path=exporter,
        runtime_policy=approval,
        expected_path=approval["executable_path"],
        expected_sha256=approval["executable_sha256"],
        label="candidate exporter",
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise PreflightError(f"candidate exporter build attestation failed: {result.stderr.strip()}")
    try:
        build = strict_json_loads(result.stdout)
    except TraceError as error:
        raise PreflightError(f"candidate exporter build attestation is invalid: {error}") from error
    if not isinstance(build, dict):
        raise PreflightError("candidate exporter build attestation is not an object")
    manifest = {"revision": candidate_revision, "build": copy.deepcopy(build)}
    libraries = manifest["build"].get("runtime_libraries")
    if not isinstance(libraries, list):
        raise PreflightError("candidate exporter build attestation has no runtime libraries")
    manifest["build"]["runtime_libraries_post"] = copy.deepcopy(libraries)
    monitor = manifest["build"].get("runtime_module_monitor")
    if not isinstance(monitor, dict):
        raise PreflightError("candidate exporter build attestation has no runtime monitor")
    monitor["checked_after_trace"] = True
    validate_runtime_build(
        manifest,
        exporter=exporter,
        exporter_sha256=exporter_sha256,
        candidate_revision=candidate_revision,
        approval=approval,
    )
    return build


def bind_candidate_attestation(
        output: Path,
        attestation: dict[str, str],
        accelerator: dict[str, object],
        exporter: Path,
        exporter_sha256: str,
        approval: dict[str, object]) -> None:
    manifest_path = safe_trace_path(output, "manifest.json")
    try:
        manifest = strict_json_loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, TraceError) as error:
        raise PreflightError(f"cannot bind candidate attestation: {error}") from error
    if manifest.get("accelerator") != accelerator:
        raise PreflightError("llama trace accelerator attestation differs from the preflight query")
    bound_attestation = dict(attestation)
    libraries_sha256, receipt_sha256 = validate_runtime_build(
        manifest,
        exporter=exporter,
        exporter_sha256=exporter_sha256,
        candidate_revision=attestation["revision"],
        approval=approval,
    )
    bound_attestation["runtime_libraries_sha256"] = libraries_sha256
    bound_attestation["runtime_receipt_sha256"] = receipt_sha256
    manifest["candidate"] = bound_attestation
    temp = manifest_path.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    os.replace(temp, manifest_path)


def validate_accelerator_attestation(
        record: object,
        *,
        expected_device: str = "ROCm0") -> dict[str, object]:
    if not isinstance(record, dict):
        raise PreflightError("accelerator attestation is not an object")
    required_keys = {
        "format",
        "version",
        "runtime_kind",
        "platform",
        "backend",
        "backend_device",
        "backend_description",
        "pci_device_id",
        "kfd_node",
        "gpu_id",
        "gfx_target_version",
        "architecture",
        "source",
    }
    if set(record) != required_keys:
        raise PreflightError("accelerator attestation fields are invalid")
    expected = {
        "format": "dsv41-accelerator-attestation",
        "version": 2,
        "runtime_kind": "strix-rocm",
        "platform": "linux",
        "backend": "ROCm",
        "backend_device": expected_device,
        "architecture": "gfx1151",
        "gfx_target_version": 110501,
        "source": "linux-kfd-sysfs",
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise PreflightError(f"accelerator attestation {key} mismatch")
    if not isinstance(record.get("backend_description"), str) or not record["backend_description"]:
        raise PreflightError("accelerator attestation backend description is missing")
    pci_device_id = record.get("pci_device_id")
    if not isinstance(pci_device_id, str) or re.fullmatch(
            r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", pci_device_id) is None:
        raise PreflightError("accelerator attestation PCI identity is invalid")
    if not isinstance(record.get("kfd_node"), str) or not record["kfd_node"].isdigit():
        raise PreflightError("accelerator attestation KFD node is invalid")
    if type(record.get("gpu_id")) is not int or record["gpu_id"] <= 0:
        raise PreflightError("accelerator attestation GPU identity is invalid")
    return dict(record)


def query_accelerator_attestation(
        exporter: Path,
        device: str,
        approval: dict[str, object] | None = None) -> dict[str, object]:
    try:
        if approval is None:
            result = subprocess.run(
                [str(exporter), "--dsv41-attest-device", device],
                check=False,
                capture_output=True,
                text=True,
            )
        else:
            result, _identity = run_approved_executable(
                [str(exporter), "--dsv41-attest-device", device],
                path=exporter,
                runtime_policy=approval,
                expected_path=approval["executable_path"],
                expected_sha256=approval["executable_sha256"],
                label="candidate exporter",
                check=False,
                capture_output=True,
                text=True,
            )
    except OSError as error:
        raise PreflightError(f"cannot query selected accelerator: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit {result.returncode}"
        raise PreflightError(f"selected accelerator query failed: {detail}")
    try:
        record = strict_json_loads(result.stdout)
    except TraceError as error:
        raise PreflightError(f"selected accelerator query returned invalid JSON: {error}") from error
    return validate_accelerator_attestation(record, expected_device=device)


def build_command(args: argparse.Namespace, exporter: Path, output: Path) -> list[str]:
    return [
        str(exporter),
        "-m", str(resolved(args.model)),
        "-bf", str(resolved(args.prompt)),
        "-o", str(output),
        "-c", str(args.context),
        "-n", str(args.decode_steps),
        "-b", str(args.batch),
        "-ub", str(args.ubatch),
        "--device", args.device,
        "-ngl", str(args.gpu_layers),
        "-fa", "on",
        "-ctk", "f16",
        "-ctv", "f16",
        "--load-mode", "none",
        "--expert-cache-slots", str(args.expert_cache_slots),
        "--expert-cache-mib", str(args.expert_cache_mib),
    ]


def validate_runtime_config(args: argparse.Namespace) -> None:
    if args.batch != ADMITTED_BATCH:
        raise PreflightError(
            f"DeepSeek V4.1 correctness runs require batch {ADMITTED_BATCH}, found {args.batch}")
    if args.ubatch != ADMITTED_UBATCH:
        raise PreflightError(
            f"DeepSeek V4.1 correctness runs require admitted ubatch {ADMITTED_UBATCH}, found {args.ubatch}")
    if args.expert_cache_slots != REQUIRED_EXPERT_SLOTS:
        raise PreflightError(
            f"DeepSeek V4.1 correctness runs require {REQUIRED_EXPERT_SLOTS} expert cache slots, "
            f"found {args.expert_cache_slots}")
    if args.expert_cache_mib != REQUIRED_EXPERT_CACHE_MIB:
        raise PreflightError(
            f"DeepSeek V4.1 correctness runs require {REQUIRED_EXPERT_CACHE_BYTES} expert cache bytes "
            f"({REQUIRED_EXPERT_CACHE_MIB} MiB), found {args.expert_cache_mib} MiB")
    if args.device != "ROCm0":
        raise PreflightError(f"DeepSeek V4.1 correctness runs require device ROCm0, found {args.device}")
    if args.gpu_layers != 99:
        raise PreflightError(f"DeepSeek V4.1 correctness runs require 99 GPU layers, found {args.gpu_layers}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed launcher for llama.cpp DeepSeek V4.1 traces")
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--candidate-diff-sha256", required=True)
    parser.add_argument("--candidate-exporter-policy-id", required=True)
    parser.add_argument("--prompt-builder-policy-id", required=True)
    parser.add_argument("--approval-policy", type=Path, required=True)
    parser.add_argument("--approval-signature", type=Path, required=True)
    parser.add_argument("--approval-principal", required=True)
    parser.add_argument("--corpus-name", choices=sorted(CORPUS_SHA256), required=True)
    parser.add_argument("--corpus-sha256", required=True)
    parser.add_argument("--prompt-provenance", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--busy-pattern", action="append", default=["ds4-v41", "DeepSeek-V4.1"])
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--batch", type=int, default=ADMITTED_BATCH)
    parser.add_argument("--ubatch", type=int, default=ADMITTED_UBATCH)
    parser.add_argument("--device", default="ROCm0")
    parser.add_argument("--expert-cache-slots", type=int, default=REQUIRED_EXPERT_SLOTS)
    parser.add_argument("--expert-cache-mib", type=int, default=REQUIRED_EXPERT_CACHE_MIB)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--signer-principal", required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--execution-challenge", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--authorization-issued-unix", type=int, required=True)
    parser.add_argument("--authorization-expires-unix", type=int, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    model_descriptor = -1
    watchdog_descriptor = -1
    try:
        validate_runtime_config(args)
        reject_loader_overrides()
        output = resolved(args.output)
        approval_policy = load_executable_approval_policy(
            args.approval_policy,
            args.approval_signature,
            expected_principal=args.approval_principal,
            forbidden_roots=(output,),
        )
        candidate_policy, candidate_policy_sha256 = candidate_exporter_approval(
            args.candidate_exporter_policy_id,
            policies=approval_policy.candidate_exporters,
        )
        prompt_policy, prompt_policy_sha256 = prompt_builder_approval(
            args.prompt_builder_policy_id,
            policies=approval_policy.prompt_builders,
        )
        if args.corpus_sha256 != CORPUS_SHA256[args.corpus_name]:
            raise PreflightError(f"corpus SHA-256 mismatch for {args.corpus_name}")
        validate_signing_identity(
            args.signing_key,
            args.signer_principal,
            trusted_signers=APPROVED_TRACE_SIGNERS,
            forbidden_root=output,
        )
        exporter = args.exporter
        exporter_identity = approved_executable_identity(
            exporter,
            install_root=candidate_policy["install_root"],
            expected_owner_uid=candidate_policy["install_owner_uid"],
            expected_path=candidate_policy["executable_path"],
            expected_sha256=candidate_policy["executable_sha256"],
            label="candidate exporter",
        )
        runtime_identities = approved_runtime_file_identities(
            candidate_policy, label="candidate exporter")
        helper_identity = approved_containment_helper_identity(
            candidate_policy, label="candidate exporter")
        candidate_trust = install_trust_evidence(
            exporter_identity, runtime_identities, (helper_identity,))
        exporter_sha256 = exporter_identity.sha256
        if args.candidate_revision != candidate_policy["revision"] or (
                args.base_revision != candidate_policy["base_revision"]) or (
                args.candidate_diff_sha256 != candidate_policy["diff_sha256"]):
            raise PreflightError("candidate arguments differ from external exporter approval")
        repo = resolved(args.repo)
        if git_output(repo, "rev-parse", "HEAD").decode("ascii").strip() != (
                approval_policy.verifier_revision):
            raise PreflightError("candidate verifier checkout differs from the external approval policy")
        if repo != approved_source_root(prompt_policy) or candidate_policy["revision"] != prompt_policy["revision"]:
            raise PreflightError("candidate repository or revision differs from prompt builder approval")
        model_descriptor, model_identity = open_model_descriptor(args.model)
        model_sha256 = str(model_identity["sha256"])
        if model_sha256 != MODEL_SHA256:
            raise PreflightError(f"published model SHA-256 mismatch: expected {MODEL_SHA256}, found {model_sha256}")
        provenance = validate_prompt_provenance(
            args.prompt_provenance,
            prompt=args.prompt,
            corpus_name=args.corpus_name,
            corpus_sha256=args.corpus_sha256,
            model_sha256=model_sha256,
            target_tokens=args.context - args.decode_steps,
            context=args.context,
            decode_steps=args.decode_steps,
            builder_approval_id=args.prompt_builder_policy_id,
            builder_policy=prompt_policy,
            builder_policy_sha256=prompt_policy_sha256,
        )
        prompt_trust_sha256 = provenance["record"]["builder_install_trust_sha256"]
        authorization = execution_authorization(
            lane=CANDIDATE_LANE,
            challenge=args.execution_challenge,
            run_id=args.run_id,
            issued_unix=args.authorization_issued_unix,
            expires_unix=args.authorization_expires_unix,
            approval_policy_sha256=approval_policy.sha256,
            verifier_revision=approval_policy.verifier_revision,
            tokenizer_policy_sha256_value=tokenizer_policy_sha256(prompt_policy["tokenizer"]),
            approvals={
                "candidate_exporter": approval_binding(
                    "candidate_exporter",
                    args.candidate_exporter_policy_id,
                    candidate_policy_sha256,
                    install_trust_sha256(candidate_trust),
                ),
                "prompt_builder": approval_binding(
                    "prompt_builder",
                    args.prompt_builder_policy_id,
                    prompt_policy_sha256,
                    prompt_trust_sha256,
                ),
            },
        )
        verify_approved_runtime_file_identities(
            runtime_identities, label="candidate exporter")
        pre_runtime_build = query_runtime_build_attestation(
            exporter,
            args.device,
            exporter_sha256=exporter_sha256,
            candidate_revision=args.candidate_revision,
            approval=candidate_policy,
        )
        verify_approved_executable_identity(exporter, exporter_identity, label="candidate exporter")
        verify_approved_runtime_file_identities(
            runtime_identities, label="candidate exporter")
        accelerator = query_accelerator_attestation(exporter, args.device, candidate_policy)
        verify_approved_executable_identity(exporter, exporter_identity, label="candidate exporter")
        verify_approved_runtime_file_identities(
            runtime_identities, label="candidate exporter")
        if args.preflight_only:
            audit = run_strix_preflight(
                model=args.model,
                prompt=args.prompt,
                output=args.output,
                repo=args.repo,
                busy_patterns=args.busy_pattern,
            )
            audit["accelerator"] = accelerator
            print(json.dumps(audit, sort_keys=True, separators=(",", ":")))
            return 0

        attestation = candidate_attestation(
            args,
            exporter,
            exporter_sha256,
            args.candidate_exporter_policy_id,
            candidate_policy_sha256,
            candidate_policy,
            approval_policy.verifier_revision,
            candidate_trust,
        )
        if output.exists() and any(output.iterdir()):
            raise PreflightError(f"trace output directory is not empty: {output}")
        preflight_audit = run_strix_preflight(
            model=args.model,
            prompt=args.prompt,
            output=args.output,
            repo=args.repo,
            busy_patterns=args.busy_pattern,
        )
        preflight_audit["runtime"] = "llama.cpp"
        preflight_audit["accelerator"] = accelerator
        preflight_audit["config"] = {
            "context": args.context,
            "decode_steps": args.decode_steps,
            "batch": args.batch,
            "ubatch": args.ubatch,
            "device": args.device,
            "device_architecture": accelerator["architecture"],
            "device_pci_id": accelerator["pci_device_id"],
            "expert_cache_slots": args.expert_cache_slots,
            "expert_cache_mib": args.expert_cache_mib,
            "gpu_layers": args.gpu_layers,
        }
        watchdog_descriptor, watchdog_authority = open_watchdog_namespace_authority(
            preflight_audit["watchdog"])
        preflight_audit["watchdog"]["namespace_authority"] = watchdog_authority
        pre_audits = write_audits(Path(str(output) + ".audit") / "pre", preflight_audit)
        pre_audit_digests = seal_audits(pre_audits)
        environment = os.environ.copy()
        environment["DSV41_TRACE_MEMORY_AUDIT"] = pre_audits["memory"]
        environment["DSV41_TRACE_SWAP_AUDIT"] = pre_audits["swap"]
        environment["DSV41_TRACE_WATCHDOG_AUDIT"] = pre_audits["watchdog"]
        environment["DSV41_TOKENIZER_POLICY"] = canonical_json(prompt_policy["tokenizer"])
        environment["DSV41_MODEL_DESCRIPTOR"] = str(model_descriptor)
        environment["DSV41_MODEL_DESCRIPTOR_IDENTITY"] = canonical_json(model_identity)
        environment["DSV41_WATCHDOG_PIDFD"] = str(watchdog_descriptor)
        command = build_command(args, exporter, output)
        print("exec:", shlex.join(command), file=sys.stderr)
        verify_approved_executable_identity(exporter, exporter_identity, label="candidate exporter")
        verify_approved_runtime_file_identities(
            runtime_identities, label="candidate exporter")
        result, executed_identity = run_approved_executable(
            command,
            path=exporter,
            runtime_policy=candidate_policy,
            expected_path=candidate_policy["executable_path"],
            expected_sha256=candidate_policy["executable_sha256"],
            label="candidate exporter",
            env=environment,
            check=False,
            retained_fds=(model_descriptor, watchdog_descriptor),
        )
        verify_model_descriptor(model_descriptor, model_identity)
        verify_watchdog_namespace_authority(watchdog_descriptor, watchdog_authority)
        if executed_identity != exporter_identity:
            raise PreflightError("candidate exporter execution identity differs from external approval")
        if result.returncode != 0:
            return result.returncode
        verify_approved_executable_identity(exporter, exporter_identity, label="candidate exporter")
        verify_approved_runtime_file_identities(
            runtime_identities, label="candidate exporter")
        post_runtime_build = query_runtime_build_attestation(
            exporter,
            args.device,
            exporter_sha256=exporter_sha256,
            candidate_revision=args.candidate_revision,
            approval=candidate_policy,
        )
        if post_runtime_build != pre_runtime_build:
            raise PreflightError("candidate exporter build identity changed during trace execution")
        verify_approved_executable_identity(exporter, exporter_identity, label="candidate exporter")
        verify_approved_runtime_file_identities(
            runtime_identities, label="candidate exporter")
        verify_sealed_audits(pre_audits, pre_audit_digests)
        postflight_audit = run_strix_preflight(
            model=args.model,
            prompt=args.prompt,
            output=args.output,
            repo=args.repo,
            busy_patterns=args.busy_pattern,
        )
        verify_watchdog_namespace_authority(watchdog_descriptor, watchdog_authority)
        postflight_audit["watchdog"]["namespace_authority"] = watchdog_authority
        verify_approved_runtime_file_identities(
            runtime_identities, label="candidate exporter")
        post_accelerator = query_accelerator_attestation(
            exporter, args.device, candidate_policy)
        if post_accelerator != accelerator:
            raise PreflightError("selected accelerator identity changed during trace execution")
        verify_approved_runtime_file_identities(
            runtime_identities, label="candidate exporter")
        postflight_audit["runtime"] = "llama.cpp"
        postflight_audit["accelerator"] = post_accelerator
        post_audits = write_audits(Path(str(output) + ".audit") / "post", postflight_audit)
        bind_embedded_audits(output, {"pre": pre_audits, "post": post_audits})
        bind_prompt_provenance(output, provenance)
        bind_candidate_attestation(
            output, attestation, accelerator, exporter, exporter_sha256, candidate_policy)
        bind_execution_authorization(output, authorization)
        unsealed_manifest = strict_json_loads(
            safe_trace_path(output, "manifest.json").read_text(encoding="ascii"))
        if not isinstance(unsealed_manifest, dict) or validate_tokenizer_policy(
                unsealed_manifest.get("config", {}).get("tokenizer")) != prompt_policy["tokenizer"]:
            raise PreflightError("candidate manifest tokenizer policy differs from external approval")
        seal_bundle(
            output,
            private_key=args.signing_key,
            principal=args.signer_principal,
            expected_lane=CANDIDATE_LANE,
            expected_challenge=args.execution_challenge,
            expected_run_id=args.run_id,
            candidate_exporter_policies=approval_policy.candidate_exporters,
            ds4_exporter_policies={},
            prompt_builder_policies=approval_policy.prompt_builders,
            expected_candidate_exporter_policy_id=args.candidate_exporter_policy_id,
            expected_ds4_exporter_policy_id=None,
            expected_prompt_builder_policy_id=args.prompt_builder_policy_id,
            expected_approval_policy_sha256=approval_policy.sha256,
            expected_verifier_revision=approval_policy.verifier_revision,
            trusted_signers=APPROVED_TRACE_SIGNERS,
        )
        bundle = TraceBundle(
            output,
            verifier=TraceVerifier.production(
                args.signer_principal,
                expected_lane=CANDIDATE_LANE,
                expected_challenge=args.execution_challenge,
                expected_run_id=args.run_id,
                expected_candidate_exporter_policy_id=args.candidate_exporter_policy_id,
                expected_ds4_exporter_policy_id=None,
                expected_prompt_builder_policy_id=args.prompt_builder_policy_id,
                approval_policy=approval_policy,
                verification_unix=None,
            ),
        )
        if bundle.manifest.get("runtime") != "llama.cpp":
            raise PreflightError("llama exporter wrote a non-llama.cpp trace")
        if bundle.manifest.get("build", {}).get("sha256") != exporter_sha256:
            raise PreflightError("llama trace build SHA-256 does not match the executed exporter")
        if bundle.manifest.get("model", {}).get("sha256") != MODEL_SHA256:
            raise PreflightError("llama trace model SHA-256 does not match the published GGUF")
        return 0
    except (PreflightError, TraceError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        for descriptor in (watchdog_descriptor, model_descriptor):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError as error:
                    print(f"error: cannot close retained descriptor {descriptor}: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
