#!/usr/bin/env python3
# From lemonade-sdk/llamacpp-rocm (MIT): lists the ROCm libraries a build links against, see build-llamacpp-rocm.yml

import subprocess
import os
import shutil
import argparse
import logging

logger = logging.getLogger("gather_required_libs")


def find_lib_in_rocm(libname, rocm_dir):
    # Simple linear search for libname inside rocm_dir
    for root, _, files in os.walk(rocm_dir):
        if libname in files:
            return os.path.join(root, libname)

    # Exit if the library is not found in rocm_dir
    raise RuntimeError(f"Could not find {libname} in {rocm_dir}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Gather required libraries for llama-server"
    )
    parser.add_argument(
        "--rocm-dir",
        default="/opt/rocm",
        help="Path to ROCm installation directory (default: /opt/rocm)",
    )
    parser.add_argument(
        "--dest-dir",
        default=os.path.expanduser("~/llama.cpp/build/bin"),
        help="Destination directory for libraries and binary (default: ~/llama.cpp/build/bin)",
    )

    args = parser.parse_args()

    rocm_dir = args.rocm_dir
    dest_dir = args.dest_dir
    binary = os.path.join(dest_dir, "llama-server")

    # Create the destination directory and run the binary
    os.makedirs(dest_dir, exist_ok=True)
    result = subprocess.run([binary], capture_output=True, text=True)
    logger.info(f"Error Found: {result.stderr}")

    # Copy the missing libraries to the destination directory
    while "error while loading shared libraries" in result.stderr:
        so_file = result.stderr.split("shared libraries: ")[1].split(": ")[0]
        so_file_path = find_lib_in_rocm(so_file, rocm_dir)
        shutil.copy2(so_file_path, dest_dir)
        logger.info(f"Copied {so_file_path} -> {dest_dir}")
        result = subprocess.run([binary], capture_output=True, text=True)
        logger.info(f"Error Found: {result.stderr}")


if __name__ == "__main__":
    main()
