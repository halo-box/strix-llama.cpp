#!/bin/bash
# Apply the measurement patches, in order, to a tree checked out at a commit of the qsa-rewrite branch that still has
# the PR219 QSA graph (W0 + W12), e.g. the W0-lab arm: git worktree add --detach <dir> <W12 commit>; apply.sh <dir>.
# Every hook is inert unless its environment variable is set (QPIN, LAB_DEF, GGML_QSA_TRACE, GGML_QSA_MASKED_DENSE,
# GGML_QSA_DECODE_MAXQ, LLAMA_QSA_ALLVIS, QPIN_IDX_TYPE).
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
T=${1:?tree}
cd "$T"
for p in 01-qpin 02-r1e-qsa-chunk 03-r1q8-defect-hook 04-c1-lab; do
  git apply --check "$D/$p.patch"
  git apply "$D/$p.patch"
  echo "applied $p"
done
