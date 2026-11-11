# Workflow scope: CPU, ROCm and Vulkan only

This fork targets Strix Halo (gfx1151, RDNA 3.5), so its CI builds and tests only the backends the
device can run: the CPU backend, HIP/ROCm and Vulkan. Every other platform lane is disabled, not
deleted. GitHub reads only `*.yml` and `*.yaml` under `.github/workflows/`, so `<name>.yml.disabled`
never runs and does not appear in the Actions tab either. The recipe is kept for a manual run: copy
it back to `<name>.yml` on a throwaway branch.

Disabled 2026-11-11:

| file | platform |
| --- | --- |
| build-cuda-ubuntu.yml, build-cuda-windows.yml, ci-self-hosted-cuda.yml | CUDA |
| build-apple.yml, ci-self-hosted-metal.yml | Metal / Apple |
| build-sycl.yml | SYCL (Intel) |
| build-cann.yml | CANN (Ascend) |
| build-opencl.yml | OpenCL |
| build-openvino.yml, ci-self-hosted-openvino.yml | OpenVINO |
| build-webgpu.yml, ci-self-hosted-webgpu.yml, build-wasm.yml | WebGPU / WebAssembly |
| build-virtgpu.yml | VirtGPU (Venus) |
| build-and-test-snapdragon.yml | Hexagon (Snapdragon) |
| build-riscv.yml | RISC-V |
| build-ibm.yml | ppc64le, s390x |
| build-cross.yml | SpacemiT, LoongArch |
| build-android.yml | Android |
| build-msys.yml | Windows MSYS2 |
| ci-self-hosted-kleidiai.yml | KleidiAI (ARM) |
| build-3rd-party.yml | third-party libs, ARM64 runner only |
| server-self-hosted.yml | server tests on CUDA and macOS runners |

`models-check.yml` keeps only its in-scope jobs: `rocm`, `vulkan-nvidia` and `vulkan-amd`. Its `cuda`,
`metal` and `webgpu-nvidia` jobs are removed.

Kept: build-cpu, ci-self-hosted-cpu, build-vulkan, ci-self-hosted-vulkan, build-llamacpp-rocm,
test-llamacpp-rocm, hip-quality-check, models-check, bench-gfx1151 (three lanes), server,
server-sanitize, build-sanitize, build-cmake-pkg, build-cache, ui (five lanes), and the
platform-neutral checks (code-style, editorconfig, python-*, update-ops-docs, pre-tokenizer-hashes,
check-vendor, labeler, pr-draft-label, close-issue, ai-issues, copilot-setup-steps, make-release,
gguf-publish, nightly-failure-alert, upstream-sync-report). `bench.yml.disabled` was already disabled
before this change.

Still multi-platform on purpose: `docker.yml` publishes container images (cpu/arm64/s390x/cuda/vulkan/
rocm/openvino); it is a publish pipeline, not a test. `winget.yml` stops itself with
`if: github.repository_owner == 'ggml-org'`.

Two things to know:

- A rebase or merge onto upstream brings the disabled files back. Resolve by keeping them disabled
  (`git rm` the resurrected `<name>.yml`).
- If a disabled lane is listed as a required status check in branch protection, pull requests wait
  for it forever. Remove it under Settings -> Branches.
