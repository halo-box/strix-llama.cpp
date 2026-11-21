# ROCm and Vulkan release builds

`.github/workflows/build-llamacpp-rocm.yml` publishes prebuilt ROCm binaries of this fork. It copies the
pipeline of [lemonade-sdk/llamacpp-rocm](https://github.com/lemonade-sdk/llamacpp-rocm) (MIT) and uses the same
artifact layout, so tools that consume their releases can consume these ones too.

## What a release contains

- One zip per GPU target and OS: `llama-bNNNN-ubuntu-rocm-{target}-x64.zip` and
  `llama-bNNNN-windows-rocm-{target}-x64.zip`. Windows builds are best effort: a failed Windows job leaves its zip
  out of the release instead of blocking it, and the release notes list only the OSes that were built.
- Target: `gfx1151` (Strix Halo) only.
- One Linux and one Windows Vulkan zip per release: `llama-bNNNN-ubuntu-vulkan-x64.zip` and
  `llama-bNNNN-windows-vulkan-x64.zip`. Each is built from the same commit with
  `GGML_VULKAN=ON`, loads the CPU variants and backends at runtime (`GGML_BACKEND_DL`,
  `GGML_CPU_ALL_VARIANTS`), sets RPATH to `$ORIGIN`, and uses the system Vulkan loader and driver
  (Mesa RADV on Linux, AMD's Adrenalin driver on Windows). The Windows zip sets no RPATH; its DLLs sit next to
  the binaries. A manual run can turn both off with the `vulkan` input.
- Each ROCm zip holds the llama.cpp binaries and shared libraries, built with `GGML_HIP=ON`, `GGML_RPC=ON` and
  `BUILD_SHARED_LIBS=ON`.
- Each zip also holds the ROCm runtime libraries it needs (hipBLAS, rocBLAS, hipBLASLt and their kernel
  libraries). On Linux, RPATH is set to `$ORIGIN`, so the zip runs without a system ROCm install.
- ROCm is pinned to `10.1.0a20260910`, the last 10.1 nightly on `nightly.repo.amd.com`, because halo-box HIP work
  is built and tested on 10.1 and there is no stable 10.1 tarball yet. A manual run with `rocm_version=latest` uses
  the newest nightly instead. If AMD removes the pinned tarball from the index, the ROCm jobs fail until the pin
  moves.
- Releases are tagged `b1000`, `b1001` and so on, counting up. This numbering does not collide with the `b1xxxx`
  tags carried over from upstream llama.cpp.
- The release notes record the build number, targets, ROCm version, source commit (5 characters) and build date.

## When it runs

- On every merge to `master`. This builds the merged commit and publishes a release. A newer merge cancels a
  release that is still building.
- On demand with `workflow_dispatch`. Inputs: OS list (`windows,ubuntu` by default), targets,
  ROCm version, branch or tag, and whether to publish.
- On pull requests that change the pipeline itself. These runs build but do not publish.

If a release build for a merge fails, `nightly-failure-alert.yml` opens or updates an issue labelled `nightly-failure`.

## Hardware tests

`test-stx-halo` runs the built zip on real hardware. It is skipped unless the repository variable
`STX_HALO_RUNNERS` is set to `true`, because jobs whose self-hosted runners are missing sit in the queue until
they time out. To enable it:

1. Register runners with the labels `stx-halo` + `Windows` / `Linux`.
2. Set `STX_HALO_RUNNERS` to `true`.

`test-llamacpp-rocm.yml` tests an already published release on the same runners.

## Tokens

The workflow uses the `GH_TOKEN` secret when it exists. Otherwise it uses the job token with `contents: write`.
