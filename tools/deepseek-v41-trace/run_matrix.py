#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from preflight import PreflightError, require_nvme_path, resolved, run_strix_preflight
from trace_format import (
    ADMITTED_BATCH,
    ADMITTED_UBATCH,
    APPROVED_CANDIDATE_EXPORTERS,
    APPROVED_PROMPT_BUILDERS,
    APPROVED_TRACE_SIGNERS,
    CANDIDATE_LANE,
    CORPUS_SHA256,
    MODEL_SHA256,
    REQUIRED_EXPERT_CACHE_MIB,
    REQUIRED_EXPERT_SLOTS,
    TraceError,
    approval_binding,
    approved_containment_helper_identity,
    approved_executable_identity,
    approved_prompt_record,
    approved_runtime_file_identities,
    candidate_exporter_approval,
    execution_authorization,
    install_trust_evidence,
    install_trust_sha256,
    load_executable_approval_policy,
    prompt_builder_approval,
    reject_loader_overrides,
    runtime_build_evidence_sha256,
    run_approved_executable,
    sha256_file,
    strict_json_loads,
    tokenizer_policy_sha256,
    validate_signing_identity,
    validate_runtime_build_evidence,
    validate_tokenizer_policy,
    verify_approved_executable_identity,
    verify_approved_runtime_file_identities,
)

CORPORA = (
    "correctness-prose.txt",
    "correctness-code.txt",
    "correctness-structured.txt",
    "correctness-numeric.txt",
)


def approved_source_root(builder_policy: dict[str, object]) -> Path:
    try:
        return Path(str(builder_policy["source_root"])).expanduser().resolve(strict=True)
    except (KeyError, OSError) as error:
        raise PreflightError(f"prompt builder approved source root is invalid: {error}") from error


def query_prompt_builder_runtime_build(
        builder: Path,
        builder_policy: dict[str, object]) -> dict[str, object]:
    result, _identity = run_approved_executable(
        [str(builder), "--dsv41-attest-build"],
        path=builder,
        runtime_policy=builder_policy,
        expected_path=builder_policy["executable_path"],
        expected_sha256=builder_policy["executable_sha256"],
        label="prompt builder",
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"prompt builder build attestation failed: {result.stderr.strip()}")
    try:
        record = strict_json_loads(result.stdout)
        return validate_runtime_build_evidence(record, builder_policy, label="prompt builder")
    except TraceError as error:
        raise RuntimeError(f"prompt builder build attestation is invalid: {error}") from error


def run(command: list[str]) -> None:
    displayed = list(command)
    if "--signing-key" in displayed:
        index = displayed.index("--signing-key")
        if index + 1 < len(displayed):
            displayed[index + 1] = "<redacted>"
    print("exec:", " ".join(displayed), file=sys.stderr)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with status {result.returncode}")


def file_identity(path: Path) -> tuple[int, int, int, int, int]:
    record = path.stat()
    return (
        record.st_dev,
        record.st_ino,
        record.st_size,
        record.st_mtime_ns,
        record.st_ctime_ns,
    )


def prepare_prompt(
    *,
    builder: Path,
    builder_approval_id: str,
    builder_policy: dict[str, object],
    builder_policy_sha256: str,
    model: Path,
    corpus: Path,
    source_corpus: Path,
    corpus_name: str,
    corpus_sha256: str,
    output: Path,
    context: int,
    decode_steps: int,
) -> dict[str, object]:
    target_tokens = context - decode_steps
    expected_prompt = approved_prompt_record(
        builder_policy,
        corpus_name=corpus_name,
        context=context,
        decode_steps=decode_steps,
    )
    builder_identity = approved_executable_identity(
        builder,
        install_root=builder_policy["install_root"],
        expected_owner_uid=builder_policy["install_owner_uid"],
        expected_path=builder_policy["executable_path"],
        expected_sha256=builder_policy["executable_sha256"],
        label="prompt builder",
    )
    runtime_identities = approved_runtime_file_identities(
        builder_policy, label="prompt builder")
    helper_identity = approved_containment_helper_identity(
        builder_policy, label="prompt builder")
    source_root_lexical = Path(builder_policy["source_root"])
    source_root_resolved = approved_source_root(builder_policy)
    source_corpus_lexical = source_corpus
    source_corpus_resolved = source_corpus_lexical.resolve(strict=True)
    expected_source = (
        source_root_resolved / "tests" / "corpus" / corpus_name).resolve(strict=True)
    if source_corpus_resolved != expected_source or sha256_file(source_corpus_resolved) != corpus_sha256 or (
            sha256_file(corpus) != corpus_sha256):
        raise RuntimeError("prompt builder corpus path or bytes differ from external approval")
    source_identity = file_identity(source_corpus_resolved)
    corpus_identity = file_identity(corpus)
    verify_approved_runtime_file_identities(runtime_identities, label="prompt builder")
    tokenizer = validate_tokenizer_policy(builder_policy["tokenizer"])
    pre_runtime_build = query_prompt_builder_runtime_build(builder, builder_policy)
    command = [
        str(builder),
        "--model", str(model),
        "--corpus", str(corpus),
        "--output", str(output),
        "--tokens", str(target_tokens),
        "--tokenizer-add-bos", str(tokenizer["add_bos"]).lower(),
        "--tokenizer-parse-special", str(tokenizer["parse_special"]).lower(),
        "--tokenizer-detokenize-special", str(tokenizer["detokenize_special"]).lower(),
        "--tokenizer-remove-leading-bos", str(
            tokenizer["remove_leading_bos_before_detokenize"]).lower(),
        "--tokenizer-require-round-trip", str(tokenizer["require_round_trip"]).lower(),
    ]
    print("exec:", " ".join(command), file=sys.stderr)
    result, executed_identity = run_approved_executable(
        command,
        path=builder,
        runtime_policy=builder_policy,
        expected_path=builder_policy["executable_path"],
        expected_sha256=builder_policy["executable_sha256"],
        label="prompt builder",
        check=False,
        capture_output=True,
        text=True,
    )
    if executed_identity != builder_identity:
        raise RuntimeError("prompt builder execution identity differs from external approval")
    verify_approved_executable_identity(builder, builder_identity, label="prompt builder")
    verify_approved_runtime_file_identities(runtime_identities, label="prompt builder")
    if file_identity(source_corpus_resolved) != source_identity or file_identity(corpus) != corpus_identity or (
            sha256_file(source_corpus_resolved) != corpus_sha256) or sha256_file(corpus) != corpus_sha256:
        raise RuntimeError("prompt builder corpus changed during execution")
    if result.returncode != 0:
        raise RuntimeError(f"prompt builder failed: {result.stderr.strip()}")
    try:
        native_record = strict_json_loads(result.stdout)
    except TraceError as error:
        raise RuntimeError(f"prompt builder returned invalid JSON: {error}") from error
    if not isinstance(native_record, dict) or set(native_record) != {
            "target_tokens", "actual_tokens", "byte_count", "tokenizer",
            "runtime_build", "temporary_directory"}:
        raise RuntimeError("prompt builder returned an invalid result schema")
    if native_record.get("target_tokens") != target_tokens or native_record.get("actual_tokens") != target_tokens:
        raise RuntimeError("prompt builder did not produce the requested token count")
    if type(native_record.get("byte_count")) is not int or native_record["byte_count"] != output.stat().st_size:
        raise RuntimeError("prompt builder byte count does not match its output")
    if validate_tokenizer_policy(native_record.get("tokenizer")) != tokenizer:
        raise RuntimeError("prompt builder tokenizer policy differs from external approval")
    runtime_build = validate_runtime_build_evidence(
        native_record.get("runtime_build"), builder_policy, label="prompt builder")
    if runtime_build != pre_runtime_build:
        raise RuntimeError("prompt builder runtime build changed during prompt construction")
    temporary_directory = native_record["temporary_directory"]
    expected_temporary_directory = os.environ.get("TMPDIR")
    if not expected_temporary_directory or temporary_directory != str(resolved(Path(expected_temporary_directory))):
        raise RuntimeError("prompt builder did not attest the selected temporary directory")
    prompt_sha256 = sha256_file(output)
    prompt_byte_count = output.stat().st_size
    if prompt_sha256 != expected_prompt["prompt_sha256"] or (
            prompt_byte_count != expected_prompt["prompt_byte_count"]):
        raise RuntimeError("prompt builder output differs from external approval")
    trust_evidence = install_trust_evidence(
        builder_identity, runtime_identities, (helper_identity,))
    record = {
        "format": "dsv41-prompt-provenance",
        "version": 2,
        "corpus_name": corpus_name,
        "corpus_sha256": corpus_sha256,
        "corpus_path": str(source_corpus_resolved),
        "corpus_lexical_path": str(source_corpus_lexical),
        "corpus_resolved_path": str(source_corpus_resolved),
        "source_root_lexical_path": str(source_root_lexical),
        "source_root_resolved_path": str(source_root_resolved),
        "model_sha256": MODEL_SHA256,
        "prompt_sha256": prompt_sha256,
        "prompt_byte_count": prompt_byte_count,
        "context": context,
        "decode_steps": decode_steps,
        "builder_approval_id": builder_approval_id,
        "builder_approval_sha256": builder_policy_sha256,
        "builder_path": str(builder),
        "builder_sha256": builder_identity.sha256,
        "builder_revision": builder_policy["revision"],
        "builder_runtime_profile": builder_policy["runtime_profile"],
        "tokenizer": tokenizer,
        "builder_runtime_build": runtime_build,
        "builder_runtime_build_sha256": runtime_build_evidence_sha256(
            runtime_build, builder_policy, label="prompt builder"),
        "builder_install_trust": trust_evidence,
        "builder_install_trust_sha256": install_trust_sha256(trust_evidence),
        "target_tokens": target_tokens,
        "actual_tokens": target_tokens,
    }
    provenance_path = output.with_suffix(output.suffix + ".provenance.json")
    provenance_path.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    record.update({
        "path": str(output),
        "provenance_path": str(provenance_path),
        "provenance_sha256": sha256_file(provenance_path),
    })
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture the DeepSeek V4.1 llama.cpp corpus matrix")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llama-runner", type=Path, required=True)
    parser.add_argument("--llama-exporter", type=Path, required=True)
    parser.add_argument("--llama-prompt-builder", type=Path, required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--candidate-diff-sha256", required=True)
    parser.add_argument("--candidate-exporter-policy-id", required=True)
    parser.add_argument("--prompt-builder-policy-id", required=True)
    parser.add_argument("--approval-policy", type=Path, required=True)
    parser.add_argument("--approval-signature", type=Path, required=True)
    parser.add_argument("--approval-principal", required=True)
    parser.add_argument("--llama-only", action="store_true")
    parser.add_argument("--contexts", type=int, nargs="+", default=[32768])
    parser.add_argument("--ubatches", type=int, nargs="+", default=[ADMITTED_UBATCH])
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--batch", type=int, default=ADMITTED_BATCH)
    parser.add_argument("--device", default="ROCm0")
    parser.add_argument("--expert-cache-slots", type=int, default=REQUIRED_EXPERT_SLOTS)
    parser.add_argument("--expert-cache-mib", type=int, default=REQUIRED_EXPERT_CACHE_MIB)
    parser.add_argument("--busy-pattern", action="append", default=["ds4-v41", "DeepSeek-V4.1"])
    parser.add_argument("--signer-principal", required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--execution-challenge", required=True)
    parser.add_argument("--run-id-prefix", required=True)
    parser.add_argument("--authorization-issued-unix", type=int, required=True)
    parser.add_argument("--authorization-expires-unix", type=int, required=True)
    args = parser.parse_args()

    try:
        if args.ubatches != [ADMITTED_UBATCH]:
            raise PreflightError(
                f"DeepSeek V4.1 correctness matrix requires admitted ubatch [{ADMITTED_UBATCH}]")
        if args.batch != ADMITTED_BATCH:
            raise PreflightError(f"DeepSeek V4.1 correctness matrix requires batch {ADMITTED_BATCH}")
        if args.device != "ROCm0":
            raise PreflightError("DeepSeek V4.1 correctness matrix requires device ROCm0")
        if args.expert_cache_slots != REQUIRED_EXPERT_SLOTS:
            raise PreflightError(
                f"DeepSeek V4.1 correctness matrix requires {REQUIRED_EXPERT_SLOTS} expert cache slots")
        if args.expert_cache_mib != REQUIRED_EXPERT_CACHE_MIB:
            raise PreflightError(
                f"DeepSeek V4.1 correctness matrix requires {REQUIRED_EXPERT_CACHE_MIB} MiB expert cache")
        reject_loader_overrides()
        output_candidate = resolved(args.output)
        approval_policy = load_executable_approval_policy(
            args.approval_policy,
            args.approval_signature,
            expected_principal=args.approval_principal,
            forbidden_roots=(output_candidate,),
        )
        candidate_policy, candidate_policy_sha256 = candidate_exporter_approval(
            args.candidate_exporter_policy_id,
            policies=approval_policy.candidate_exporters,
        )
        prompt_policy, prompt_policy_sha256 = prompt_builder_approval(
            args.prompt_builder_policy_id,
            policies=approval_policy.prompt_builders,
        )
        candidate_identity = approved_executable_identity(
            args.llama_exporter,
            install_root=candidate_policy["install_root"],
            expected_owner_uid=candidate_policy["install_owner_uid"],
            expected_path=candidate_policy["executable_path"],
            expected_sha256=candidate_policy["executable_sha256"],
            label="candidate exporter",
        )
        candidate_runtime_identities = approved_runtime_file_identities(
            candidate_policy, label="candidate exporter")
        candidate_helper_identity = approved_containment_helper_identity(
            candidate_policy, label="candidate exporter")
        candidate_trust = install_trust_evidence(
            candidate_identity, candidate_runtime_identities, (candidate_helper_identity,))
        prompt_builder = args.llama_prompt_builder
        prompt_identity = approved_executable_identity(
            prompt_builder,
            install_root=prompt_policy["install_root"],
            expected_owner_uid=prompt_policy["install_owner_uid"],
            expected_path=prompt_policy["executable_path"],
            expected_sha256=prompt_policy["executable_sha256"],
            label="prompt builder",
        )
        prompt_runtime_identities = approved_runtime_file_identities(
            prompt_policy, label="prompt builder")
        prompt_helper_identity = approved_containment_helper_identity(
            prompt_policy, label="prompt builder")
        prompt_trust = install_trust_evidence(
            prompt_identity, prompt_runtime_identities, (prompt_helper_identity,))
        if args.candidate_revision != candidate_policy["revision"] or (
                args.base_revision != candidate_policy["base_revision"]) or (
                args.candidate_diff_sha256 != candidate_policy["diff_sha256"]):
            raise PreflightError("matrix candidate identity differs from external exporter approval")
        execution_authorization(
            lane=CANDIDATE_LANE,
            challenge=args.execution_challenge,
            run_id=f"{args.run_id_prefix}-preflight",
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
                    install_trust_sha256(prompt_trust),
                ),
            },
        )
        if not args.llama_only:
            raise PreflightError(
                "cross-runtime capture must run on separate Strix and Apple hosts; "
                "use --llama-only here and compare completed bundles with trace_format.py")
        validate_signing_identity(
            args.signing_key,
            args.signer_principal,
            trusted_signers=APPROVED_TRACE_SIGNERS,
            forbidden_root=output_candidate,
        )
        repo = resolved(args.repo)
        revision = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.STDOUT,
        ).decode("ascii").strip()
        if revision != approval_policy.verifier_revision:
            raise PreflightError("matrix verifier checkout differs from the external approval policy")
        output = require_nvme_path(output_candidate, "matrix output")
        model = require_nvme_path(args.model, "model")
        if not model.is_file():
            raise PreflightError(f"model is not a file: {model}")
        model_sha256 = sha256_file(model)
        if model_sha256 != MODEL_SHA256:
            raise PreflightError(f"published model SHA-256 mismatch: expected {MODEL_SHA256}, found {model_sha256}")
        initial_corpus = require_nvme_path(
            repo / "tests" / "corpus" / CORPORA[0],
            "repository corpus",
        )
        run_strix_preflight(
            model=model,
            prompt=initial_corpus,
            output=output,
            repo=repo,
            busy_patterns=args.busy_pattern,
        )
        if repo != approved_source_root(prompt_policy) or args.candidate_revision != prompt_policy["revision"]:
            raise PreflightError("matrix repository or revision differs from prompt builder approval")
        if output.exists() and any(output.iterdir()):
            raise PreflightError(f"matrix output directory is not empty: {output}")
        inputs = output / "inputs"
        sources = inputs / "sources"
        prompts = inputs / "prompts"
        sources.mkdir(parents=True, exist_ok=True)
        prompts.mkdir(parents=True, exist_ok=True)
        corpus_records = []
        for name in CORPORA:
            source = require_nvme_path(repo / "tests" / "corpus" / name, "repository corpus")
            if not source.is_file():
                raise PreflightError(f"repository corpus is missing: {source}")
            destination = sources / name
            shutil.copyfile(source, destination)
            source_sha256 = sha256_file(destination)
            if source_sha256 != CORPUS_SHA256[name]:
                raise PreflightError(
                    f"repository corpus SHA-256 mismatch for {name}: expected {CORPUS_SHA256[name]}, found {source_sha256}")
            corpus_records.append({
                "name": name,
                "source": str(source),
                "path": str(destination),
                "byte_count": destination.stat().st_size,
                "sha256": source_sha256,
            })

        results = []
        prompt_records = []
        for context in args.contexts:
            if context < 32768 or context > 131072:
                raise PreflightError(f"context is outside the supported 32768..131072 matrix: {context}")
            target_tokens = context - args.decode_steps
            if target_tokens < 1:
                raise PreflightError("decode steps leave no room for prompt tokens")
            prepared_prompts = {}
            for corpus in corpus_records:
                stem = Path(corpus["name"]).stem
                prompt = prompts / f"{stem}-c{context}.txt"
                run_strix_preflight(
                    model=model,
                    prompt=Path(corpus["path"]),
                    output=prompt,
                    repo=repo,
                    busy_patterns=args.busy_pattern,
                )
                prepared = prepare_prompt(
                    builder=resolved(args.llama_prompt_builder),
                    builder_approval_id=args.prompt_builder_policy_id,
                    builder_policy=prompt_policy,
                    builder_policy_sha256=prompt_policy_sha256,
                    model=model,
                    corpus=Path(corpus["path"]),
                    source_corpus=Path(corpus["source"]),
                    corpus_name=corpus["name"],
                    corpus_sha256=corpus["sha256"],
                    output=prompt,
                    context=context,
                    decode_steps=args.decode_steps,
                )
                prepared.update({"corpus": corpus["name"], "context": context})
                prepared_prompts[corpus["name"]] = prepared
                prompt_records.append(prepared)
            for ubatch in args.ubatches:
                for corpus in corpus_records:
                    stem = Path(corpus["name"]).stem
                    case = f"{stem}-c{context}-ub{ubatch}"
                    run_id = f"{args.run_id_prefix}-{case}"
                    llama_output = output / "llama" / case
                    prompt = prepared_prompts[corpus["name"]]["path"]
                    provenance = prepared_prompts[corpus["name"]]["provenance_path"]
                    common = [
                        "--model", str(model),
                        "--prompt", prompt,
                        "--prompt-provenance", provenance,
                        "--corpus-name", corpus["name"],
                        "--corpus-sha256", corpus["sha256"],
                        "--context", str(context),
                        "--decode-steps", str(args.decode_steps),
                    ]
                    for pattern in args.busy_pattern:
                        common.extend(["--busy-pattern", pattern])
                    run([
                        sys.executable,
                        str(resolved(args.llama_runner)),
                        "--exporter", str(resolved(args.llama_exporter)),
                        "--repo", str(repo),
                        "--candidate-revision", args.candidate_revision,
                        "--base-revision", args.base_revision,
                        "--candidate-diff-sha256", args.candidate_diff_sha256,
                        "--candidate-exporter-policy-id", args.candidate_exporter_policy_id,
                        "--prompt-builder-policy-id", args.prompt_builder_policy_id,
                        "--approval-policy", str(resolved(args.approval_policy)),
                        "--approval-signature", str(resolved(args.approval_signature)),
                        "--approval-principal", args.approval_principal,
                        "--output", str(llama_output),
                        "--batch", str(args.batch),
                        "--ubatch", str(ubatch),
                        "--device", args.device,
                        "--expert-cache-slots", str(args.expert_cache_slots),
                        "--expert-cache-mib", str(args.expert_cache_mib),
                        "--signer-principal", args.signer_principal,
                        "--signing-key", str(resolved(args.signing_key)),
                        "--execution-challenge", args.execution_challenge,
                        "--run-id", run_id,
                        "--authorization-issued-unix", str(args.authorization_issued_unix),
                        "--authorization-expires-unix", str(args.authorization_expires_unix),
                        *common,
                    ])
                    results.append({
                        "case": case,
                        "status": "BRINGUP TRACE CAPTURED",
                        "cross_runtime_status": "INCOMPLETE",
                        "trace": str(llama_output),
                        "run_id": run_id,
                    })

        summary = {
            "status": "BRINGUP TRACE CAPTURED",
            "mode": "llama-only",
            "cross_runtime_status": "INCOMPLETE",
            "model": str(model),
            "model_sha256": model_sha256,
            "candidate_revision": args.candidate_revision,
            "base_revision": args.base_revision,
            "candidate_diff_sha256": args.candidate_diff_sha256,
            "signer_principal": args.signer_principal,
            "execution_challenge": args.execution_challenge,
            "run_id_prefix": args.run_id_prefix,
            "corpora": corpus_records,
            "prompts": prompt_records,
            "contexts": args.contexts,
            "ubatches": args.ubatches,
            "decode_steps": args.decode_steps,
            "target_prompt_tokens": {
                str(context): context - args.decode_steps for context in args.contexts
            },
            "cases": results,
        }
        (output / "summary.json").write_text(
            json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="ascii",
        )
        return 0
    except (PreflightError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
