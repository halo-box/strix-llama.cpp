#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


def quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--profile", required=True, choices=("co-located", "sibling-lib"))
    parser.add_argument("--entry", action="append", nargs=3, default=[], metavar=("COMPONENT", "PATH", "REVISION"))
    args = parser.parse_args()

    if re.fullmatch(r"[0-9a-f]{40}", args.revision) is None:
        parser.error("revision must be an exact full Git revision")
    if not args.entry:
        parser.error("at least one receipt entry is required")

    entries = []
    components = set()
    filenames = set()
    digests = set()
    for component, path_text, revision in args.entry:
        path = Path(path_text).resolve(strict=True)
        digest = sha256_file(path)
        filename = path.name
        if re.fullmatch(r"[a-z0-9-]+", component) is None:
            parser.error(f"invalid component: {component}")
        if re.fullmatch(r"[A-Za-z0-9._+-]+", filename) is None:
            parser.error(f"invalid filename: {filename}")
        if component in components:
            parser.error(f"duplicate component: {component}")
        if filename in filenames:
            parser.error(f"duplicate filename: {filename}")
        if digest in digests:
            parser.error(f"duplicate SHA-256: {digest}")
        if revision != "-" and revision != args.revision:
            parser.error(f"invalid revision for {component}")
        components.add(component)
        filenames.add(filename)
        digests.add(digest)
        entries.append((component, filename, digest, "" if revision == "-" else revision))

    entries.sort()
    receipt = {
        "format": "dsv41-runtime-receipt",
        "version": 1,
        "revision": args.revision,
        "profile": args.profile,
        "components": [
            {
                "component": component,
                "filename": filename,
                "sha256": digest,
                "revision": revision or None,
            }
            for component, filename, digest, revision in entries
        ],
    }
    receipt_sha256 = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    lines = [
        "#pragma once",
        "",
        "#include <array>",
        "#include <cstddef>",
        "",
        "namespace dsv41_runtime_receipt {",
        "",
        "struct entry {",
        "    const char * component;",
        "    const char * filename;",
        "    const char * sha256;",
        "    const char * revision;",
        "};",
        "",
        f"inline constexpr const char * profile = {quote(args.profile)};",
        f"inline constexpr const char * sha256 = {quote(receipt_sha256)};",
        f"inline constexpr std::array<entry, {len(entries)}> entries = {{{{",
    ]
    for component, filename, digest, revision in entries:
        lines.append(
            "    {"
            + ", ".join((quote(component), quote(filename), quote(digest), quote(revision)))
            + "},"
        )
    lines.extend([
        "}};",
        f"inline constexpr std::array<const char *, {len(entries)}> components = {{{{",
    ])
    for component, _, _, _ in entries:
        lines.append(f"    {quote(component)},")
    lines.extend([
        "}};",
        "",
        "}",
        "",
    ])
    content = "\n".join(lines).encode("ascii")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, args.output)
    receipt_content = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    receipt_temporary = args.json_output.with_suffix(args.json_output.suffix + ".tmp")
    receipt_temporary.write_bytes(receipt_content)
    os.replace(receipt_temporary, args.json_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
