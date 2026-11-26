#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
    echo "Usage: $0 BIN_DIRECTORY LOG_DIRECTORY" >&2
    exit 2
fi
bin=$(cd "$1" && pwd)
mkdir -p "$2"
log=$(cd "$2" && pwd)

"$bin/llama-server" --list-devices 2>&1 | tee "$log/devices.log"
grep -Eq '^[[:space:]]*HRX0:' "$log/devices.log"
"$bin/test-llama-archs" --hrx-devices 2>&1 | tee "$log/placement.log"
"$bin/test-backend-ops" test -b HRX0 --backend-params trace -o GATED_UPDATE -p 'n_embd=2304,n_tokens=128,v=7' 2>&1 | tee "$log/gate.log"
awk -F '\t' '$4 == "ggml_inkling_ctrl_apply_native" { ran = 1 } END { exit !ran }' "$log/gate.log"

"$bin/test-flash-attn-indexed-bias" --wide-hrx 2>&1 | tee "$log/wide-attention.log"
