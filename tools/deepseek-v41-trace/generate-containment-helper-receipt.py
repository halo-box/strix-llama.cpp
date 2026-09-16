#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--helper", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.revision) is None:
        parser.error("revision must be an exact full Git revision")
    helper = args.helper.resolve(strict=True)
    receipt = {
        "format": "dsv41-containment-helper",
        "version": 2,
        "revision": args.revision,
        "filename": helper.name,
        "sha256": sha256_file(helper),
        "launcher_policy": "zero-supplementary-groups-v1",
        "supplementary_groups": [],
    }
    content = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
