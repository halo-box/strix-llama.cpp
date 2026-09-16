#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path

from trace_format import DS4_REVISION, sha256_file


ANCHORS = {
    "tests/test-vectors/README.md":
        "0e59b2f2832bed8af0a91e6ff20962debf964cd2d1d141c086e52cfcc995a1c3",
    "tests/test-vectors/flash-0731/manifest.json":
        "ebf237a5660a6851fb8085e77f532901a9d758208b25d7ed5af0b7af4b28f91b",
    "tests/test-vectors/flash-0731/official.vec":
        "77ae699889bfaf1348768dcbe7ea2c72279ae86abb10470d3e1b08cd1fd82a83",
    "tests/test-vectors/flash-0731/local-golden.vec":
        "23d942ff3b9bb2a3f82927d11aa3ed1461e1f302071e788d0d95a5c165e47d3b",
    "tests/test_engram.c":
        "198a561d981f62518a9d28035480a7e220b99c156cde6d248b8baabd684cc74b",
    "ds4_engram.c":
        "2b6ca468510ebf45ee298a905525bc7234dacad9a384bf2011eba19ba2c0bdf7",
    "ds4_engram.h":
        "f84a264e0fe199d23a6f0c56fbbd19e222adc7009eed13c185af6403f68f7f5c",
    "tests/test_deepseek41_metal.c":
        "9197c2f9d65b380ce25be5334991e4bfaf40e82e6708e64f2b9329552412d28c",
    "tests/test_deepseek41_graph.c":
        "6dc786f831c93ae7f5aa56e7518f67657125f7c0fd35cead3c646eff3d3f9e09",
    "tests/test_deepseek41_prefill.c":
        "452774b9332d393822d84288eeb2d71ca1fb25f1ba30d20a49160d1787de734f",
    "tests/test_deepseek41_manifest.py":
        "2d7aa1fc93805d9c97839c5de9eccc856628f13e6913c3bbc3f0c99b0827aeae",
    "tests/test_deepseek41_conversion.py":
        "a40b83062a9b91338773addd77fe62650296f4de607057cb4fd1329287f347b5c",
    "tests/test_deepseek41_gguf.c":
        "8f41e049d5ec179c38a1a00ac61712db0306902f0ef74bcdb989d8993316ff35",
    "gguf-tools/deepseek41_metadata.py":
        "39300bbd504165b97de017edd72563377f50b8a5511ca7478252a86f31a0009b",
    "ds4.c":
        "1776dbfed177ea14f3ce6cac1d8d0b1c1b44dfff2c2663769a9a5634aeec34e7",
}


class AnchorError(RuntimeError):
    pass


def git_output(checkout: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(checkout), *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise AnchorError(f"git {' '.join(args)} failed: {error}") from error


def verify(
        checkout: Path,
        *,
        anchors: dict[str, str] = ANCHORS,
        expected_revision: str = DS4_REVISION) -> dict[str, object]:
    checkout = checkout.expanduser().resolve()
    revision = git_output(checkout, "rev-parse", "HEAD")
    if revision != expected_revision:
        raise AnchorError(f"ds4 revision mismatch: expected {expected_revision}, found {revision}")
    if git_output(checkout, "diff", "--name-only") or git_output(checkout, "diff", "--cached", "--name-only"):
        raise AnchorError("ds4 tracked source differs from the pinned revision")
    files = []
    for relative, expected in anchors.items():
        path = checkout / relative
        if not path.is_file():
            raise AnchorError(f"pinned ds4 anchor is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise AnchorError(
                f"pinned ds4 anchor SHA-256 mismatch for {relative}: expected {expected}, found {actual}")
        files.append({"path": relative, "sha256": actual})
    return {
        "status": "ANCHORS VERIFIED",
        "revision": revision,
        "files": files,
        "limitations": [
            "official.vec contains selected-token and top-logprob slices, not complete logits",
            "local-golden.vec is a tolerant top-64 drift anchor",
            "the Metal source fixture is not executable evidence on Strix",
            "cross-runtime TARGET PASS still requires the pinned ds4 trace exporter",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify pinned ds4 bring-up evidence anchors")
    parser.add_argument(
        "--checkout",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.checkout), sort_keys=True, separators=(",", ":")))
        return 0
    except AnchorError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
