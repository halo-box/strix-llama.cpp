#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
from pathlib import Path


def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--install-root", required=True, type=Path)
    args = parser.parse_args()

    receipt_bytes = args.receipt.read_bytes()
    try:
        receipt = json.loads(receipt_bytes, object_pairs_hook=strict_object)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        parser.error(f"runtime receipt JSON is invalid: {error}")
    canonical = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    if receipt_bytes != canonical:
        parser.error("runtime receipt JSON is not canonical")
    if set(receipt) != {"format", "version", "revision", "profile", "components"} or (
            receipt["format"] != "dsv41-runtime-receipt" or receipt["version"] != 1 or
            re.fullmatch(r"[0-9a-f]{40}", receipt["revision"]) is None or
            receipt["profile"] not in {"co-located", "sibling-lib"}):
        parser.error("runtime receipt is invalid")
    directory = args.install_root / ("bin" if receipt["profile"] == "co-located" else "lib")
    components = receipt["components"]
    if not isinstance(components, list) or not components:
        parser.error("runtime receipt component list is invalid")
    if components != sorted(components, key=lambda item: item.get("component", "") if isinstance(item, dict) else ""):
        parser.error("runtime receipt component list is not canonical")
    names = set()
    filenames = set()
    digests = set()
    for component in components:
        if not isinstance(component, dict) or set(component) != {
                "component", "filename", "sha256", "revision"}:
            parser.error("runtime receipt component is invalid")
        filename = component["filename"]
        name = component["component"]
        digest = component["sha256"]
        revision = component["revision"]
        if not isinstance(name, str) or re.fullmatch(r"[a-z0-9-]+", name) is None or name in names:
            parser.error("runtime receipt component name is invalid")
        if not isinstance(filename, str) or Path(filename).name != filename or (
                re.fullmatch(r"[A-Za-z0-9._+-]+", filename) is None) or filename in filenames:
            parser.error("runtime receipt filename is invalid")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None or digest in digests:
            parser.error("runtime receipt component SHA-256 is invalid")
        revision_bearing = name in {"llama-common", "ggml-base"}
        if (revision_bearing and revision != receipt["revision"]) or (
                not revision_bearing and revision is not None):
            parser.error("runtime receipt component revision is invalid")
        names.add(name)
        filenames.add(filename)
        digests.add(digest)
        installed = directory / filename
        if installed.is_symlink() or not installed.is_file():
            parser.error(f"installed runtime component is missing or not regular: {filename}")
        if sha256_file(installed) != digest:
            parser.error(f"installed runtime component SHA-256 mismatch: {name}")
    if not {"llama-common", "llama", "ggml", "ggml-base"}.issubset(names):
        parser.error("runtime receipt is missing core components")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
