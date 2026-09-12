#!/usr/bin/env bash

set -euo pipefail

sd=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tmpdir=$(mktemp -d)
output="${tmpdir}/output.log"

function cleanup {
    rm -f "${tmpdir}/bin/hipconfig" "${tmpdir}/bin/rocminfo"
    rm -f "${tmpdir}/hip-only/hipconfig" "${tmpdir}/rocm-only/rocminfo" "${output}"
    rmdir "${tmpdir}/results" "${tmpdir}/mnt" 2>/dev/null || true
    rmdir "${tmpdir}/bin" "${tmpdir}/hip-only" "${tmpdir}/rocm-only" "${tmpdir}"
}

trap cleanup EXIT

mkdir "${tmpdir}/bin" "${tmpdir}/hip-only" "${tmpdir}/rocm-only"
cat > "${tmpdir}/bin/hipconfig" <<'EOF'
#!/bin/bash
printf '%s\n' "${TEST_HIPCONFIG_OUTPUT:-7.1.52802-9999}"
exit "${TEST_HIPCONFIG_STATUS:-0}"
EOF
cat > "${tmpdir}/bin/rocminfo" <<'EOF'
#!/bin/bash
printf '%s\n' "${TEST_ROCMINFO_OUTPUT:-}"
exit "${TEST_ROCMINFO_STATUS:-0}"
EOF
chmod +x "${tmpdir}/bin/hipconfig" "${tmpdir}/bin/rocminfo"
cp "${tmpdir}/bin/hipconfig" "${tmpdir}/hip-only/hipconfig"
cp "${tmpdir}/bin/rocminfo" "${tmpdir}/rocm-only/rocminfo"

rocminfo_gfx1151='Runtime Version:         1.1
  Name:                    AMD RYZEN AI MAX+ 395 w/ Radeon 8060S
  Marketing Name:          AMD RYZEN AI MAX+ 395 w/ Radeon 8060S
  Name:                    gfx1151
  Marketing Name:          AMD Radeon 8060S Graphics'

rocminfo_other='Runtime Version:         1.1
  Name:                    AMD Ryzen Processor
  Name:                    gfx1100'

function expect_pass {
    local name=$1
    shift

    if ! "$@" > "${output}" 2>&1; then
        echo "FAIL: ${name}"
        cat "${output}"
        exit 1
    fi

    if ! grep -Fq "ROCm preflight passed" "${output}"; then
        echo "FAIL: ${name} did not report success"
        cat "${output}"
        exit 1
    fi
}

function expect_fail {
    local name=$1
    local message=$2
    shift 2

    if "$@" > "${output}" 2>&1; then
        echo "FAIL: ${name} unexpectedly passed"
        cat "${output}"
        exit 1
    fi

    if ! grep -Fq "${message}" "${output}"; then
        echo "FAIL: ${name} did not report '${message}'"
        cat "${output}"
        exit 1
    fi
}

preflight=(/bin/bash "${sd}/run.sh" --rocm-preflight)
stub_env=(env -i PATH="${tmpdir}/bin:/usr/bin:/bin")

expect_pass "valid gfx1151 environment" \
    "${stub_env[@]}" HIP_LAUNCH_BLOCKING=1 GG_BUILD_AMDGPU_TARGETS='gfx1100;gfx1151' TEST_ROCMINFO_OUTPUT="${rocminfo_gfx1151}" \
    "${preflight[@]}"

expect_fail "invalid HIP_LAUNCH_BLOCKING" "HIP_LAUNCH_BLOCKING must be exactly 1" \
    "${stub_env[@]}" HIP_LAUNCH_BLOCKING=0 GG_BUILD_AMDGPU_TARGETS=gfx1151 TEST_ROCMINFO_OUTPUT="${rocminfo_gfx1151}" \
    "${preflight[@]}"

expect_fail "missing gfx1151 target" "GG_BUILD_AMDGPU_TARGETS must include gfx1151" \
    "${stub_env[@]}" HIP_LAUNCH_BLOCKING=1 GG_BUILD_AMDGPU_TARGETS=gfx11510 TEST_ROCMINFO_OUTPUT="${rocminfo_gfx1151}" \
    "${preflight[@]}"

expect_fail "missing gfx1151 agent" "rocminfo did not report a gfx1151 agent" \
    "${stub_env[@]}" HIP_LAUNCH_BLOCKING=1 GG_BUILD_AMDGPU_TARGETS=gfx1151 TEST_ROCMINFO_OUTPUT="${rocminfo_other}" \
    "${preflight[@]}"

expect_fail "missing hipconfig command" "hipconfig was not found in PATH" \
    env -i PATH="${tmpdir}/rocm-only" HIP_LAUNCH_BLOCKING=1 GG_BUILD_AMDGPU_TARGETS=gfx1151 \
    /bin/bash "${sd}/run.sh" --rocm-preflight

expect_fail "missing rocminfo command" "rocminfo was not found in PATH" \
    env -i PATH="${tmpdir}/hip-only" HIP_LAUNCH_BLOCKING=1 GG_BUILD_AMDGPU_TARGETS=gfx1151 \
    /bin/bash "${sd}/run.sh" --rocm-preflight

expect_fail "hipconfig command failure" "hipconfig --version exited with status 6" \
    "${stub_env[@]}" HIP_LAUNCH_BLOCKING=1 GG_BUILD_AMDGPU_TARGETS=gfx1151 TEST_HIPCONFIG_OUTPUT="HIP runtime unavailable" TEST_HIPCONFIG_STATUS=6 \
    "${preflight[@]}"

expect_fail "rocminfo command failure" "rocminfo exited with status 7" \
    "${stub_env[@]}" HIP_LAUNCH_BLOCKING=1 GG_BUILD_AMDGPU_TARGETS=gfx1151 TEST_ROCMINFO_OUTPUT="ROCk module is not loaded" TEST_ROCMINFO_STATUS=7 \
    "${preflight[@]}"

expect_fail "generic ROCm target validation" "Missing GG_BUILD_AMDGPU_TARGETS" \
    "${stub_env[@]}" GG_BUILD_ROCM=1 \
    /bin/bash "${sd}/run.sh" "${tmpdir}/results" "${tmpdir}/mnt"

expect_fail "self-hosted opt-in before build" "HIP_LAUNCH_BLOCKING must be exactly 1" \
    "${stub_env[@]}" GG_BUILD_ROCM=1 GG_BUILD_ROCM_STRIX_PREFLIGHT=1 HIP_LAUNCH_BLOCKING=0 GG_BUILD_AMDGPU_TARGETS=gfx1151 \
    /bin/bash "${sd}/run.sh" "${tmpdir}/results" "${tmpdir}/mnt"

echo "ROCm preflight self-test passed (10 cases)"
