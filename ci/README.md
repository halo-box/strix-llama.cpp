# CI

This CI implements heavy-duty workflows that run on self-hosted runners. Typically the purpose of these workflows is to
cover hardware configurations that are not available from Github-hosted runners and/or require more computational
resource than normally available.

It is a good practice, before publishing changes to execute the full CI locally on your machine. For example:

```bash
mkdir tmp

# CPU-only build
bash ./ci/run.sh ./tmp/results ./tmp/mnt

# with CUDA support
GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

# with ROCm support
GG_BUILD_ROCM=1 GG_BUILD_AMDGPU_TARGETS=gfx1100 bash ./ci/run.sh ./tmp/results ./tmp/mnt

# with SYCL support
source /opt/intel/oneapi/setvars.sh
GG_BUILD_SYCL=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

# with MUSA support
GG_BUILD_MUSA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

# etc.
```

The Strix Halo self-hosted job also sets `GG_BUILD_ROCM_STRIX_PREFLIGHT=1`. This opt-in requires `HIP_LAUNCH_BLOCKING=1`, a `gfx1151` build target, and a working `gfx1151` agent reported by `rocminfo` before the ROCm build starts. Other ROCm runs do not use this device-specific check.

# Adding self-hosted runners

- Add a self-hosted `ggml-ci` workflow to [[.github/workflows/build.yml]] with an appropriate label
- Request a runner token from `ggml-org` (for example, via a comment in the PR or email)
- Set-up a machine using the received token ([docs](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners))
- Optionally update [ci/run.sh](https://github.com/ggml-org/llama.cpp/blob/master/ci/run.sh) to build and run on the target platform by gating the implementation with a `GG_BUILD_...` env
