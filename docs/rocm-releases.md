# ROCm release builds

`.github/workflows/build-llamacpp-rocm.yml` publishes prebuilt ROCm binaries of this fork. It copies the
pipeline of [lemonade-sdk/llamacpp-rocm](https://github.com/lemonade-sdk/llamacpp-rocm) (MIT) and uses the same
artifact layout, so tools that consume their releases can consume these ones too.

## What a release contains

- One zip per OS and GPU target: `llama-bNNNN-{windows,ubuntu}-rocm-{target}-x64.zip`.
- Targets: `gfx1151`, `gfx1150`, `gfx120X`, `gfx110X`, `gfx103X`, `gfx90a`, `gfx908`.
- Each zip holds the llama.cpp binaries and shared libraries, built with `GGML_HIP=ON`, `GGML_RPC=ON` and
  `BUILD_SHARED_LIBS=ON`.
- Each zip also holds the ROCm runtime libraries it needs (hipBLAS, rocBLAS, hipBLASLt and their kernel
  libraries). On Linux, RPATH is set to `$ORIGIN`, so the zip runs without a system ROCm install.
- ROCm comes from the newest TheRock nightly tarball on `nightly.repo.amd.com` unless one is pinned.
- Releases are tagged `b1000`, `b1001` and so on, counting up. This numbering does not collide with the `b1xxxx`
  tags carried over from upstream llama.cpp.
- The release notes record the build number, targets, ROCm version, source commit (5 characters) and build date.

## When it runs

- Nightly at 13:00 UTC, two hours after TheRock publishes its nightly. This builds `master` and publishes a release.
- On demand with `workflow_dispatch`. Inputs: OS list, targets, ROCm version, branch or tag, and whether to
  publish.
- On pull requests that change the pipeline itself. These runs build but do not publish.

If a nightly run fails, `nightly-failure-alert.yml` opens or updates an issue labelled `nightly-failure`.

## Hardware tests

`test-stx-halo` and `test-stx` run the built zip on real hardware. They are skipped unless the repository
variables `STX_HALO_RUNNERS` and `STX_RUNNERS` are set to `true`, because jobs whose self-hosted runners are
missing sit in the queue until they time out. To enable them:

1. Register runners with the labels `stx-halo` + `Windows` / `Linux` (and `stx` + `Windows` for gfx1150).
2. Set the matching variable to `true`.

`test-llamacpp-rocm.yml` tests an already published release on the same runners.

## Tokens

The workflow uses the `GH_TOKEN` secret when it exists. Otherwise it uses the job token with `contents: write`.
