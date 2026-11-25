# cuda

The fork `ggml/src/ggml-cuda` tree, built into the upstream `ggml-hip` backend. Do not edit the copied files by hand:
`hooks/tools/import-cuda.py <fork ref> <upstream base>` regenerates them.

- Each file is the fork file inside `namespace ggml_fork`. The namespace is closed around every `#include`, so system,
  ggml and HIP headers stay global, and the fork templates and kernels cannot collide with the upstream ones.
- The includes of `ggml-cuda/` headers resolve to the copies here. `ggml-impl.h` here is the fork one, for the fork
  code only (its fusions use 32-node subgraphs).
- Adapted lines carry `hooks:`: include paths, the EXL3 ops (custom nodes, `ggml_fork_op()` in `hooks.cuh`, their
  parameter in op_params slot 6), and the `ggml-cuda.h` functions that the fork `ggml-cuda.cu` defines under other
  names (argument-dependent lookup would also find the global declarations).

Hooks files: `hooks.cuh` (included at the end of `common.cuh`), `entry.inc` (the entry points of the bridge,
included at the end of `ggml-cuda.cu`), this file.

The bridge is `hooks/ggml/kernels/cuda/cuda-bridge.cu`: a fork context per upstream backend, on the upstream stream, and
the `cuda_node` and `cuda_graph` kernels.
