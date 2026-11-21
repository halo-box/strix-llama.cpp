# Equivalence with the fork

Programs built against the fork master and against this branch, whose outputs must be identical.

- `types-equiv.cpp`: each fork type, quantized bytes, dequantized values, validation and a CPU MUL_MAT.
- `cpu-equiv.cpp`: the CPU ops the fork changes (MUL_MAT with an F16 src1, MUL_MAT_ID, binary ops with f16 sources, ADD /
  OUT_PROD / GET_ROWS on the fork types). Run it also with `GGML_HOOKS=+fork_cpu_*` (every variant).

```
g++ -O2 -std=c++17 -I <tree>/ggml/include types-equiv.cpp -L <build>/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,<build>/bin -o equiv
```
