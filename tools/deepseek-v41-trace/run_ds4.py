#!/usr/bin/env python3

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from preflight import (
    PreflightError,
    bind_embedded_audits,
    bind_prompt_provenance,
    darwin_storage_attestation,
    resolved,
    run_oracle_preflight,
    seal_audits,
    validate_prompt_provenance,
    verify_sealed_audits,
    write_audits,
)
from trace_format import (
    ADMITTED_UBATCH,
    APPROVED_PROMPT_BUILDERS,
    APPROVED_TRACE_SIGNERS,
    CORPUS_SHA256,
    DS4_REPOSITORY,
    DS4_REVISION,
    ExecutionIntegrityError,
    ExecutableFileReceipt,
    MODEL_SHA256,
    NO_EXTERNAL_STATE_STORAGE,
    ORACLE_LANE,
    TraceBundle,
    TraceError,
    TraceVerifier,
    approval_binding,
    approved_containment_helper_identity,
    approved_executable_identity,
    approved_runtime_file_identities,
    bind_execution_authorization,
    canonical_json,
    ds4_exporter_approval,
    execution_authorization,
    install_trust_evidence,
    install_trust_sha256,
    load_executable_approval_policy,
    prompt_builder_approval,
    reject_loader_overrides,
    run_approved_executable,
    runtime_build_evidence_sha256,
    seal_bundle,
    sha256_bytes,
    sha256_file,
    strict_json_loads,
    tokenizer_policy_sha256,
    validate_runtime_build_evidence,
    validate_signing_identity,
    verify_approved_executable_identity,
    verify_approved_runtime_file_identities,
)

EXPORTER_ATTESTATION_TIMEOUT_SECONDS = 60
EXPORTER_TRACE_TIMEOUT_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class InvocationSecondaryFailure:
    component: str
    error: BaseException


class InvocationIntegrityError(PreflightError):
    def __init__(
            self,
            message: str,
            *,
            primary_error: BaseException,
            secondary_errors: list[InvocationSecondaryFailure]):
        super().__init__(message)
        self.primary_error = primary_error
        self.secondary_errors = tuple(secondary_errors)


def exporter_install_trust_evidence(
        exporter_identity: ExecutableFileReceipt,
        runtime_identities: list[ExecutableFileReceipt],
        exporter_policy: dict[str, Any]) -> dict[str, Any]:
    helper_identity = approved_containment_helper_identity(
        exporter_policy, label="ds4 exporter")
    return install_trust_evidence(
        exporter_identity, runtime_identities, (helper_identity,))


def git_output(checkout: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(checkout), *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise PreflightError(f"ds4 git {' '.join(args)} failed: {error}") from error


def verify_checkout(checkout: Path) -> str:
    revision = git_output(checkout, "rev-parse", "HEAD")
    status = git_output(checkout, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise PreflightError("ds4 checkout has tracked or untracked changes")
    return revision


def validate_accelerator_attestation(
        record: object,
        *,
        expected_device: str = "Metal0") -> dict[str, Any]:
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
        "architecture",
        "metal_registry_id",
        "recommended_max_working_set_bytes",
        "unified_memory",
        "source",
    }
    if set(record) != required_keys:
        raise PreflightError("accelerator attestation fields are invalid")
    expected = {
        "format": "dsv41-accelerator-attestation",
        "version": 2,
        "runtime_kind": "apple-metal",
        "platform": "macos",
        "backend": "Metal",
        "backend_device": expected_device,
        "unified_memory": True,
        "source": "metal-device-query",
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise PreflightError(f"accelerator attestation {key} mismatch")
    if type(record.get("unified_memory")) is not bool:
        raise PreflightError("accelerator attestation unified-memory identity is invalid")
    for key in ("backend_description", "architecture"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise PreflightError(f"accelerator attestation {key} is missing")
    if type(record.get("metal_registry_id")) is not int or record["metal_registry_id"] <= 0:
        raise PreflightError("accelerator attestation Metal registry identity is invalid")
    if type(record.get("recommended_max_working_set_bytes")) is not int or (
            record["recommended_max_working_set_bytes"] <= 0):
        raise PreflightError("accelerator attestation working-set identity is invalid")
    return dict(record)


def run_exporter_command(
        command: list[str],
        *,
        exporter: Path,
        exporter_identity: ExecutableFileReceipt,
        exporter_policy: dict[str, Any],
        timeout_seconds: int | None = None,
        **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    if type(timeout_seconds) is not int or timeout_seconds <= 0 or (
            {"timeout", "text", "encoding", "errors", "universal_newlines"} & kwargs.keys()):
        raise PreflightError("ds4 exporter timeout is invalid")
    result, executed_identity = run_approved_executable(
        command,
        path=exporter,
        runtime_policy=exporter_policy,
        expected_path=exporter_policy["executable_path"],
        expected_sha256=exporter_policy["executable_sha256"],
        label="ds4 exporter",
        timeout=timeout_seconds,
        **kwargs,
    )
    if executed_identity != exporter_identity:
        raise PreflightError("ds4 exporter execution identity differs from external approval")
    return result


def decode_exporter_output(value: bytes | None, *, label: str) -> str:
    if not isinstance(value, bytes):
        raise PreflightError(f"{label} bytes are missing")
    try:
        return value.decode("utf-8", "strict")
    except UnicodeError as error:
        raise PreflightError(f"{label} is not valid UTF-8: {error}") from error


def query_runtime_build_attestation(
        exporter: Path,
        *,
        exporter_identity: ExecutableFileReceipt,
        exporter_policy: dict[str, Any]) -> dict[str, Any]:
    result = run_exporter_command(
        [str(exporter), "--dsv41-attest-build"],
        exporter=exporter,
        exporter_identity=exporter_identity,
        exporter_policy=exporter_policy,
        timeout_seconds=EXPORTER_ATTESTATION_TIMEOUT_SECONDS,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = decode_exporter_output(
            result.stderr, label="ds4 exporter build attestation stderr").strip()
        raise PreflightError(f"ds4 exporter build attestation failed: {detail}")
    try:
        record = strict_json_loads(decode_exporter_output(
            result.stdout, label="ds4 exporter build attestation stdout"))
        return validate_runtime_build_evidence(record, exporter_policy, label="ds4 exporter")
    except TraceError as error:
        raise PreflightError(f"ds4 exporter build attestation is invalid: {error}") from error


def run_exporter_with_post_attestation(
        command: list[str],
        *,
        operation: str,
        exporter: Path,
        exporter_identity: ExecutableFileReceipt,
        exporter_policy: dict[str, Any],
        expected_runtime_build: dict[str, Any],
        timeout_seconds: int,
        decode_stdout_label: str | None = None,
        decode_stderr_label: str | None = None,
        **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    result = None
    primary_error = None
    try:
        result = run_exporter_command(
            command,
            exporter=exporter,
            exporter_identity=exporter_identity,
            exporter_policy=exporter_policy,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )
    except BaseException as error:
        primary_error = error
    if primary_error is not None and (
            not isinstance(primary_error, ExecutionIntegrityError)
            or not primary_error.quiescence_proven):
        raise primary_error
    if result is not None and primary_error is None:
        try:
            stdout = (
                decode_exporter_output(result.stdout, label=decode_stdout_label)
                if decode_stdout_label is not None else result.stdout)
            stderr = (
                decode_exporter_output(result.stderr, label=decode_stderr_label)
                if decode_stderr_label is not None else result.stderr)
            result = subprocess.CompletedProcess(
                result.args, result.returncode, stdout, stderr)
        except BaseException as error:
            primary_error = error
    nonzero_error = None
    if result is not None and result.returncode != 0:
        nonzero_error = PreflightError(f"{operation} failed: exit {result.returncode}")
    secondary_error = None
    try:
        post_runtime_build = query_runtime_build_attestation(
            exporter,
            exporter_identity=exporter_identity,
            exporter_policy=exporter_policy,
        )
        if post_runtime_build != expected_runtime_build:
            raise PreflightError(f"ds4 exporter build identity changed during {operation}")
    except BaseException as error:
        secondary_error = error
    reported_primary = primary_error or nonzero_error
    if reported_primary is not None:
        if secondary_error is not None:
            raise InvocationIntegrityError(
                f"{operation} primary failure [{type(reported_primary).__name__}: {reported_primary}]; "
                f"secondary post-invocation runtime-build attestation failure "
                f"[{type(secondary_error).__name__}: {secondary_error}]",
                primary_error=reported_primary,
                secondary_errors=[InvocationSecondaryFailure(
                    "post-invocation-runtime-build-attestation", secondary_error)],
            ) from reported_primary
        if primary_error is not None:
            raise primary_error
    if secondary_error is not None:
        raise secondary_error
    if result is None:
        raise PreflightError(f"{operation} did not return a result")
    return result


def query_accelerator_attestation(
        exporter: Path,
        device: str,
        *,
        exporter_identity: ExecutableFileReceipt,
        exporter_policy: dict[str, Any],
        expected_runtime_build: dict[str, Any]) -> dict[str, Any]:
    result = run_exporter_with_post_attestation(
        [str(exporter), "--dsv41-attest-device", device],
        operation="selected accelerator query",
        exporter=exporter,
        exporter_identity=exporter_identity,
        exporter_policy=exporter_policy,
        expected_runtime_build=expected_runtime_build,
        timeout_seconds=EXPORTER_ATTESTATION_TIMEOUT_SECONDS,
        check=False,
        capture_output=True,
        decode_stdout_label="selected accelerator query stdout",
        decode_stderr_label="selected accelerator query stderr",
    )
    validation_error = None
    attestation = None
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit {result.returncode}"
        validation_error = PreflightError(f"selected accelerator query failed: {detail}")
    else:
        try:
            record = strict_json_loads(result.stdout)
            attestation = validate_accelerator_attestation(record, expected_device=device)
        except (TraceError, PreflightError) as error:
            validation_error = PreflightError(
                f"selected accelerator query returned invalid attestation: {error}")
    if validation_error is not None:
        raise validation_error
    if attestation is None:
        raise PreflightError("selected accelerator attestation is missing")
    return attestation


def runner_attestation(
        *,
        exporter: Path,
        exporter_sha256: str,
        exporter_approval_id: str,
        exporter_approval_sha256: str,
        exporter_install_trust_sha256: str,
        exporter_runtime_build_sha256: str,
        exporter_runtime_profile: dict[str, Any],
        exporter_runtime_receipt_sha256: str,
        verifier_revision: str,
        checkout: Path,
        command: list[str]) -> dict[str, Any]:
    runner_executable = resolved(Path(sys.executable))
    runner_script = resolved(Path(__file__))
    return {
        "format": "dsv41-runner-ownership",
        "version": 1,
        "runtime_kind": "apple-metal",
        "source": "python-subprocess",
        "runner_pid": os.getpid(),
        "runner_parent_pid": os.getppid(),
        "runner_uid": os.getuid(),
        "runner_executable": str(runner_executable),
        "runner_executable_sha256": sha256_file(runner_executable),
        "runner_script": str(runner_script),
        "runner_script_sha256": sha256_file(runner_script),
        "exporter_path": str(exporter),
        "exporter_sha256": exporter_sha256,
        "exporter_approval_id": exporter_approval_id,
        "exporter_approval_sha256": exporter_approval_sha256,
        "exporter_install_trust_sha256": exporter_install_trust_sha256,
        "exporter_runtime_build_sha256": exporter_runtime_build_sha256,
        "exporter_runtime_profile": exporter_runtime_profile,
        "exporter_runtime_receipt_sha256": exporter_runtime_receipt_sha256,
        "producer_revision": DS4_REVISION,
        "verifier_revision": verifier_revision,
        "checkout_path": str(checkout),
        "checkout_revision": DS4_REVISION,
        "command_sha256": sha256_bytes(canonical_json(command).encode("ascii")),
    }


def bind_oracle_attestation(
        output: Path,
        audit: dict[str, Any],
        accelerator: dict[str, Any],
        command: list[str],
        *,
        exporter_policy: dict[str, Any],
        exporter_approval_id: str,
        exporter_approval_sha256: str,
        exporter_install_trust: dict[str, Any],
        runtime_build: dict[str, Any],
        verifier_revision: str) -> None:
    manifest_path = output / "manifest.json"
    try:
        manifest = strict_json_loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, TraceError) as error:
        raise PreflightError(f"cannot bind ds4 runtime attestation: {error}") from error
    if manifest.get("accelerator") != accelerator:
        raise PreflightError("ds4 trace accelerator attestation differs from the preflight query")
    storage = audit.get("storage")
    if not isinstance(storage, dict):
        raise PreflightError("ds4 storage attestation is missing")
    paths = {}
    for label, record in storage.items():
        if not isinstance(record, dict) or not isinstance(record.get("resolved_path"), str):
            raise PreflightError(f"ds4 storage attestation is invalid for {label}")
        paths[label] = record["resolved_path"]
    if manifest.get("model", {}).get("path") != paths["model"]:
        raise PreflightError("ds4 trace model path differs from the attested path")
    if manifest.get("prompt", {}).get("path") != paths["prompt"]:
        raise PreflightError("ds4 trace prompt path differs from the attested path")
    if "paths" in manifest and manifest["paths"] != paths:
        raise PreflightError("ds4 trace execution paths differ from preflight")
    manifest["paths"] = paths
    build = manifest.get("build")
    if not isinstance(build, dict) or build.get("path") != paths["exporter"]:
        raise PreflightError("ds4 trace build path differs from the executed exporter")
    runner = audit.get("runner")
    if not isinstance(runner, dict) or build.get("sha256") != runner.get("exporter_sha256"):
        raise PreflightError("ds4 trace build SHA-256 differs from the executed exporter")
    build_evidence = {
        "revision": manifest.get("revision"),
        "path": build.get("path"),
        "sha256": build.get("sha256"),
        "runtime_profile": build.get("runtime_profile"),
        "runtime_receipt_sha256": build.get("runtime_receipt_sha256"),
        "runtime_libraries": build.get("runtime_libraries"),
        "runtime_libraries_post": build.get("runtime_libraries_post"),
    }
    try:
        validated_build = validate_runtime_build_evidence(
            build_evidence, exporter_policy, label="ds4 exporter")
    except TraceError as error:
        raise PreflightError(f"ds4 trace runtime build differs from external approval: {error}") from error
    if validated_build != runtime_build:
        raise PreflightError("ds4 trace runtime build differs from measured exporter attestation")
    runtime_build_sha256 = runtime_build_evidence_sha256(
        validated_build, exporter_policy, label="ds4 exporter")
    runtime_libraries_sha256 = sha256_bytes(canonical_json({
        "pre": validated_build["runtime_libraries"],
        "post": validated_build["runtime_libraries_post"],
    }).encode("ascii"))
    runtime_receipt_sha256 = sha256_bytes(
        canonical_json(exporter_policy["runtime_receipt"]).encode("ascii"))
    trust_sha256 = install_trust_sha256(exporter_install_trust)
    manifest["oracle"] = {
        "repository": DS4_REPOSITORY,
        "revision": DS4_REVISION,
        "verifier_revision": verifier_revision,
        "executable_path": exporter_policy["executable_path"],
        "executable_sha256": exporter_policy["executable_sha256"],
        "runtime_profile": exporter_policy["runtime_profile"],
        "runtime_build_sha256": runtime_build_sha256,
        "runtime_libraries_sha256": runtime_libraries_sha256,
        "runtime_receipt_sha256": runtime_receipt_sha256,
        "exporter_approval_id": exporter_approval_id,
        "exporter_approval_sha256": exporter_approval_sha256,
        "install_trust": exporter_install_trust,
        "install_trust_sha256": trust_sha256,
    }
    storage_policy = audit.get("storage_policy")
    if storage_policy != NO_EXTERNAL_STATE_STORAGE:
        raise PreflightError("ds4 external cache/state storage policy is invalid")
    if "storage_policy" in manifest and manifest["storage_policy"] != storage_policy:
        raise PreflightError("ds4 trace external cache/state storage policy differs from preflight")
    manifest["storage_policy"] = storage_policy
    host = audit.get("host")
    if not isinstance(host, dict):
        raise PreflightError("ds4 host attestation is missing")
    if "host" in manifest and manifest["host"] != host:
        raise PreflightError("ds4 trace host identity differs from preflight")
    manifest["host"] = host
    manifest["environment"] = {
        "system_info": f"macOS {host['os_version']} arm64 {host['hardware_model']}",
        "command": shlex.join(command),
    }
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise PreflightError("ds4 trace config is invalid")
    for key, value in (
            ("device_backend", "Metal"),
            ("device_registry_id", accelerator["metal_registry_id"])):
        if key in config and config[key] != value:
            raise PreflightError(f"ds4 trace {key} differs from preflight")
        config[key] = value
    temp = manifest_path.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    os.replace(temp, manifest_path)


def preflight(
        args: argparse.Namespace,
        *,
        accelerator: dict[str, Any],
        runner: dict[str, Any]) -> dict[str, Any]:
    checkout = resolved(args.checkout)
    revision = verify_checkout(checkout)
    if revision != DS4_REVISION:
        raise PreflightError(f"ds4 revision mismatch: expected {DS4_REVISION}, found {revision}")
    result = run_oracle_preflight(
        model=args.model,
        prompt=args.prompt,
        output=args.output,
        repo=args.repo,
        checkout=checkout,
        busy_patterns=args.busy_pattern,
        accelerator=accelerator,
        runner=runner,
    )
    result.update({
        "runtime": "ds4",
        "ds4_revision": revision,
        "checkout": str(checkout),
        "config": {
            "context": args.context,
            "decode_steps": args.decode_steps,
            "prefill_chunk": args.prefill_chunk,
            "device_backend": "Metal",
            "device_registry_id": accelerator["metal_registry_id"],
        },
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed launcher for the pinned ds4 trace exporter")
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--busy-pattern", action="append", default=["ds4-v41", "DeepSeek-V4.1"])
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--exporter-sha256", required=True)
    parser.add_argument("--corpus-name", choices=sorted(CORPUS_SHA256), required=True)
    parser.add_argument("--corpus-sha256", required=True)
    parser.add_argument("--prompt-provenance", type=Path, required=True)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--prefill-chunk", type=int, default=ADMITTED_UBATCH)
    parser.add_argument("--device", default="Metal0")
    parser.add_argument("--signer-principal", required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--execution-challenge", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--authorization-issued-unix", type=int, required=True)
    parser.add_argument("--authorization-expires-unix", type=int, required=True)
    parser.add_argument("--ds4-exporter-policy-id", required=True)
    parser.add_argument("--prompt-builder-policy-id", required=True)
    parser.add_argument("--approval-policy", type=Path, required=True)
    parser.add_argument("--approval-signature", type=Path, required=True)
    parser.add_argument("--approval-principal", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    try:
        if args.prefill_chunk != ADMITTED_UBATCH:
            raise PreflightError(
                f"DeepSeek V4.1 correctness runs require admitted prefill chunk {ADMITTED_UBATCH}, "
                f"found {args.prefill_chunk}")
        reject_loader_overrides()
        output = resolved(args.output)
        approval_policy = load_executable_approval_policy(
            args.approval_policy,
            args.approval_signature,
            expected_principal=args.approval_principal,
            forbidden_roots=(output,),
        )
        prompt_policy, prompt_policy_sha256 = prompt_builder_approval(
            args.prompt_builder_policy_id,
            policies=approval_policy.prompt_builders,
        )
        exporter_policy, exporter_policy_sha256 = ds4_exporter_approval(
            args.ds4_exporter_policy_id,
            policies=approval_policy.ds4_exporters,
        )
        harness_repo = resolved(args.repo)
        harness_revision = subprocess.check_output(
            ["git", "-C", str(harness_repo), "rev-parse", "HEAD"],
            stderr=subprocess.STDOUT,
        ).decode("ascii").strip()
        if harness_revision != approval_policy.verifier_revision:
            raise PreflightError("ds4 verifier checkout differs from the external approval policy")
        checkout = resolved(args.checkout)
        checkout_revision = verify_checkout(checkout)
        if checkout_revision != DS4_REVISION:
            raise PreflightError(
                f"ds4 revision mismatch: expected {DS4_REVISION}, found {checkout_revision}")
        if not args.exporter.is_absolute() or str(args.exporter) != exporter_policy["executable_path"]:
            raise PreflightError("ds4 exporter path or caller digest differs from external approval")
        exporter = resolved(args.exporter)
        if str(exporter) != str(args.exporter) or (
                args.exporter_sha256 != exporter_policy["executable_sha256"]):
            raise PreflightError("ds4 exporter path or caller digest differs from external approval")
        exporter_identity = approved_executable_identity(
            exporter,
            install_root=exporter_policy["install_root"],
            expected_owner_uid=exporter_policy["install_owner_uid"],
            expected_path=exporter_policy["executable_path"],
            expected_sha256=exporter_policy["executable_sha256"],
            label="ds4 exporter",
        )
        runtime_identities = approved_runtime_file_identities(
            exporter_policy, label="ds4 exporter")
        exporter_install_trust = exporter_install_trust_evidence(
            exporter_identity, runtime_identities, exporter_policy)
        exporter_install_trust_sha256 = install_trust_sha256(exporter_install_trust)
        pre_runtime_build = query_runtime_build_attestation(
            exporter,
            exporter_identity=exporter_identity,
            exporter_policy=exporter_policy,
        )
        runtime_build_sha256 = runtime_build_evidence_sha256(
            pre_runtime_build, exporter_policy, label="ds4 exporter")
        runtime_receipt_sha256 = sha256_bytes(
            canonical_json(exporter_policy["runtime_receipt"]).encode("ascii"))
        if args.corpus_sha256 != CORPUS_SHA256[args.corpus_name]:
            raise PreflightError(f"corpus SHA-256 mismatch for {args.corpus_name}")
        model_sha256 = sha256_file(resolved(args.model))
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
            path_resolver=lambda path, label: Path(
                str(darwin_storage_attestation(path, label)["resolved_path"])),
        )
        authorization = execution_authorization(
            lane=ORACLE_LANE,
            challenge=args.execution_challenge,
            run_id=args.run_id,
            issued_unix=args.authorization_issued_unix,
            expires_unix=args.authorization_expires_unix,
            approval_policy_sha256=approval_policy.sha256,
            verifier_revision=approval_policy.verifier_revision,
            tokenizer_policy_sha256_value=tokenizer_policy_sha256(prompt_policy["tokenizer"]),
            approvals={
                "ds4_exporter": approval_binding(
                    "ds4_exporter",
                    args.ds4_exporter_policy_id,
                    exporter_policy_sha256,
                    exporter_install_trust_sha256,
                ),
                "prompt_builder": approval_binding(
                    "prompt_builder",
                    args.prompt_builder_policy_id,
                    prompt_policy_sha256,
                    provenance["record"]["builder_install_trust_sha256"],
                ),
            },
        )
        exporter_sha256 = exporter_identity.sha256
        validate_signing_identity(
            args.signing_key,
            args.signer_principal,
            trusted_signers=APPROVED_TRACE_SIGNERS,
            forbidden_root=output,
        )
        command = [
            str(exporter),
            "--model", str(resolved(args.model)),
            "--prompt-file", str(resolved(args.prompt)),
            "--output", str(output),
            "--context", str(args.context),
            "--decode-steps", str(args.decode_steps),
            "--prefill-chunk", str(args.prefill_chunk),
            "--device", args.device,
        ]
        accelerator = query_accelerator_attestation(
            exporter,
            args.device,
            exporter_identity=exporter_identity,
            exporter_policy=exporter_policy,
            expected_runtime_build=pre_runtime_build,
        )
        runner = runner_attestation(
            exporter=exporter,
            exporter_sha256=exporter_sha256,
            exporter_approval_id=args.ds4_exporter_policy_id,
            exporter_approval_sha256=exporter_policy_sha256,
            exporter_install_trust_sha256=exporter_install_trust_sha256,
            exporter_runtime_build_sha256=runtime_build_sha256,
            exporter_runtime_profile=exporter_policy["runtime_profile"],
            exporter_runtime_receipt_sha256=runtime_receipt_sha256,
            verifier_revision=approval_policy.verifier_revision,
            checkout=checkout,
            command=command,
        )
        if args.preflight_only:
            audit = preflight(args, accelerator=accelerator, runner=runner)
            print(json.dumps(audit, sort_keys=True, separators=(",", ":")))
            return 0

        if output.exists() and any(output.iterdir()):
            raise PreflightError(f"trace output directory is not empty: {output}")
        preflight_audit = preflight(args, accelerator=accelerator, runner=runner)
        preflight_audit["exporter"] = {
            "path": str(exporter),
            "sha256": exporter_sha256,
            "approval_id": args.ds4_exporter_policy_id,
            "approval_sha256": exporter_policy_sha256,
            "install_trust_sha256": exporter_install_trust_sha256,
            "runtime_build_sha256": runtime_build_sha256,
            "runtime_receipt_sha256": runtime_receipt_sha256,
        }
        pre_audits = write_audits(Path(str(output) + ".audit") / "pre", preflight_audit)
        pre_audit_digests = seal_audits(pre_audits)
        print("exec:", shlex.join(command), file=sys.stderr)
        result = run_exporter_with_post_attestation(
            command,
            operation="ds4 trace execution",
            exporter=exporter,
            exporter_identity=exporter_identity,
            exporter_policy=exporter_policy,
            expected_runtime_build=pre_runtime_build,
            timeout_seconds=EXPORTER_TRACE_TIMEOUT_SECONDS,
            cwd=checkout,
            check=False,
        )
        verify_approved_executable_identity(
            exporter, exporter_identity, label="ds4 exporter")
        verify_approved_runtime_file_identities(
            runtime_identities, label="ds4 exporter")
        if result.returncode != 0:
            return result.returncode
        verify_sealed_audits(pre_audits, pre_audit_digests)
        post_accelerator = query_accelerator_attestation(
            exporter,
            args.device,
            exporter_identity=exporter_identity,
            exporter_policy=exporter_policy,
            expected_runtime_build=pre_runtime_build,
        )
        if post_accelerator != accelerator:
            raise PreflightError("selected accelerator identity changed during trace execution")
        postflight_audit = preflight(args, accelerator=post_accelerator, runner=runner)
        if postflight_audit.get("host") != preflight_audit.get("host"):
            raise PreflightError("ds4 host identity changed during trace execution")
        post_audits = write_audits(Path(str(output) + ".audit") / "post", postflight_audit)
        bind_embedded_audits(output, {"pre": pre_audits, "post": post_audits})
        bind_prompt_provenance(output, provenance)
        bind_oracle_attestation(
            output,
            preflight_audit,
            accelerator,
            command,
            exporter_policy=exporter_policy,
            exporter_approval_id=args.ds4_exporter_policy_id,
            exporter_approval_sha256=exporter_policy_sha256,
            exporter_install_trust=exporter_install_trust,
            runtime_build=pre_runtime_build,
            verifier_revision=approval_policy.verifier_revision,
        )
        bind_execution_authorization(output, authorization)
        seal_bundle(
            output,
            private_key=args.signing_key,
            principal=args.signer_principal,
            expected_lane=ORACLE_LANE,
            expected_challenge=args.execution_challenge,
            expected_run_id=args.run_id,
            candidate_exporter_policies={},
            ds4_exporter_policies=approval_policy.ds4_exporters,
            prompt_builder_policies=approval_policy.prompt_builders,
            expected_candidate_exporter_policy_id=None,
            expected_ds4_exporter_policy_id=args.ds4_exporter_policy_id,
            expected_prompt_builder_policy_id=args.prompt_builder_policy_id,
            expected_approval_policy_sha256=approval_policy.sha256,
            expected_verifier_revision=approval_policy.verifier_revision,
            trusted_signers=APPROVED_TRACE_SIGNERS,
        )
        bundle = TraceBundle(
            output,
            verifier=TraceVerifier.production(
                args.signer_principal,
                expected_lane=ORACLE_LANE,
                expected_challenge=args.execution_challenge,
                expected_run_id=args.run_id,
                expected_candidate_exporter_policy_id=None,
                expected_ds4_exporter_policy_id=args.ds4_exporter_policy_id,
                expected_prompt_builder_policy_id=args.prompt_builder_policy_id,
                approval_policy=approval_policy,
                verification_unix=None,
            ),
        )
        if bundle.manifest.get("runtime") != "ds4":
            raise PreflightError("ds4 exporter wrote a non-ds4 trace")
        if bundle.manifest.get("revision") != DS4_REVISION:
            raise PreflightError(
                f"ds4 trace revision mismatch: expected {DS4_REVISION}, found {bundle.manifest.get('revision')}")
        if bundle.manifest.get("build", {}).get("sha256") != exporter_sha256:
            raise PreflightError("ds4 trace build SHA-256 does not match the executed exporter")
        if bundle.manifest.get("model", {}).get("sha256") != MODEL_SHA256:
            raise PreflightError("ds4 trace model SHA-256 does not match the published GGUF")
        if bundle.manifest.get("accelerator") != accelerator:
            raise PreflightError("ds4 trace accelerator attestation differs from the measured Metal device")
        return 0
    except (PreflightError, TraceError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
