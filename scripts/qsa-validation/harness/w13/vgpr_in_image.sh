#!/bin/bash
# Runs inside rocm/dev-ubuntu-26.04@sha256:4f5ed1bf... (the pinned image of scripts/hip/gfx1151-vgpr-baseline.json).
# /src = a git-archive export of the commit under test (read-only), /out = the build directory and logs.
# The two build steps of .github/workflows/hip-quality-check.yml, in one clean build directory.
set -o pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update > /out/apt.log 2>&1 && apt-get install -y build-essential git cmake amdrocm-blas-dev10.1 libssl-dev python3 jq >> /out/apt.log 2>&1
echo apt_rc=$? > /out/rc.txt
"$(hipconfig -l)/clang" --version | head -n 1 > /out/compiler_version.txt
cd /src
cmake -B /out/build -S . \
  -DCMAKE_HIP_COMPILER="$(hipconfig -l)/clang" \
  -DGPU_TARGETS=gfx1151 \
  -DGGML_HIP=ON \
  -DGGML_HIP_EXPORT_METRICS=Off \
  -DCMAKE_HIP_FLAGS="-Werror -Wno-tautological-compare" \
  -DCMAKE_BUILD_TYPE=Release > /out/werror_cfg.log 2>&1
echo werror_cfg_rc=$? >> /out/rc.txt
(cd /out/build && make -j 16 > /out/werror_build.log 2>&1)
echo werror_build_rc=$? >> /out/rc.txt
python3 scripts/hip/test_vgpr_check.py > /out/test_vgpr_check.log 2>&1
echo test_vgpr_check_rc=$? >> /out/rc.txt
cmake -B /out/build -S . \
  -DCMAKE_HIP_COMPILER="$(hipconfig -l)/clang" \
  -DGPU_TARGETS=gfx1151 \
  -DGGML_HIP=ON \
  -DGGML_HIP_EXPORT_METRICS=On \
  -DCMAKE_HIP_FLAGS="" \
  -DCMAKE_BUILD_TYPE=Release > /out/metrics_cfg.log 2>&1
echo metrics_cfg_rc=$? >> /out/rc.txt
(cd /out/build && make -j 16 2>&1 | tee /out/metrics.log | grep -v 'Rpass-analysis=kernel-resource-usage\|remark:\|^$' > /out/metrics_build.log)
echo metrics_build_rc=$? >> /out/rc.txt
echo DONE >> /out/rc.txt
