# Correctness corpora

Input text for the logit comparisons of the Correctness Flow in `AGENTS.md`. Each file is long enough for one `llama-perplexity` run at `-c 2048` (at least 2 x 2048 tokens) without repetition, and the four together cover `-c 16384` (2 x 16384 tokens): 43516 tokens with the `qwen35` vocab, 39559 with `llama-bpe`. A vocab with still more bytes per token gives fewer: use the largest `-c` that half the token count allows, as `validate.sh` does. Counts with the `qwen35` vocab (`models/ggml-vocab-qwen35.gguf`):

| File | Tokens | Content |
|---|---|---|
| `correctness-prose.txt` | 12273 | natural English prose, three styles |
| `correctness-code.txt` | 10229 | C++, Python, GLSL, TypeScript, CMake, shell, CUDA |
| `correctness-structured.txt` | 8300 | JSON, YAML, CSV, TOML, a Markdown table, GBNF |
| `correctness-numeric.txt` | 12714 | physical constants with uncertainties and units, atomic masses and isotopic compositions |

Do not edit these files: results are only comparable across runs with the same bytes. Record their sha256 with each run.

## Sources

The text is real, not generated (by a model or a script): generated text is too easy to predict and hides small numeric errors.

- **prose**: public-domain books from Project Gutenberg, with the Project Gutenberg header and license removed, cut at paragraph boundaries and converted to ASCII:
  - #10849, O. F. Walton, *Saved at Sea: A Lighthouse Story* (1887), from chapter IV;
  - #52817, F. A. Talbot, *Lightships and Lighthouses* (1913), from chapter VII;
  - #70176, J. Kennedy, *The History of Steam Navigation* (1903), from chapter X.

  Each excerpt runs on from where it starts, without illustration captions and footnotes.
- **code**: excerpts of this repository (MIT): `src/llama-sampler.cpp`, `gguf-py/gguf/gguf_reader.py`, `ggml/src/ggml-vulkan/vulkan-shaders/soft_max.comp`, `tools/ui/src/lib/utils/api-fetch.ts`, `ggml/src/ggml-hip/CMakeLists.txt`, `scripts/compare-commits.sh`, `ggml/src/ggml-cuda/softmax.cu`. They are a frozen copy: later changes to these files do not change the corpus.
- **structured**: excerpts of this repository (MIT): `CMakePresets.json`, `tools/ui/package.json`, `.github/workflows/build-cpu.yml`, `docs/ops/CUDA.csv`, `pyproject.toml`, the table of `docs/ops.md` (symbols replaced by `yes`, `no`, `partial`), `grammars/json.gbnf`, `grammars/c.gbnf`.
- **numeric**: data published by NIST (works of the US government, not subject to copyright in the US), as published under a one-line title each, trailing spaces removed:
  - the first 168 constants of the 2022 CODATA *Fundamental Physical Constants, Complete Listing* (`https://physics.nist.gov/cuu/Constants/Table/allascii.txt`): values and uncertainties with digit groups, exponents, units;
  - the first 70 isotopes of *Atomic Weights and Isotopic Compositions for All Elements* (`https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&ascii=ascii2&isotype=some`, the text of its `<pre>` block): relative atomic masses, isotopic compositions and standard atomic weights with their uncertainties.
