# Equivalence with the fork

Programs built against the fork master and against this branch, whose outputs must be identical.

- `types-equiv.cpp`: each fork type, quantized bytes, dequantized values, validation and a CPU MUL_MAT.

```
g++ -O2 -std=c++17 -I <tree>/ggml/include types-equiv.cpp -L <build>/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,<build>/bin -o equiv
```
