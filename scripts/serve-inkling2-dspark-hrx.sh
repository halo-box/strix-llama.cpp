#!/usr/bin/env bash
set -eu

usage() {
    echo "Usage: $0 TARGET.gguf DSPARK.gguf [llama-server options...]"
    echo "Set LLAMA_SERVER to the path of the HRX-enabled llama-server."
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if (( $# < 2 )); then
    usage >&2
    exit 2
fi
target=$1
draft=$2
shift 2

for model in "$target" "$draft"; do
    if [[ ! -f $model || ! -r $model ]]; then
        echo "Cannot read model: $model" >&2
        exit 2
    fi
done
server=${LLAMA_SERVER:-"$(dirname "${BASH_SOURCE[0]}")/../build/bin/llama-server"}
exec "$server" -m "$target" -md "$draft" --spec-type draft-dspark \
    --device HRX0 --device-draft HRX0 -ngl 99 -ngld 99 \
    -ot 'token_embd\.weight=HRX0' -otd 'mask_embd\.weight=HRX0' \
    --spec-draft-n-max 3 --backend-sampling --temp 0 \
    -fa on -c 40960 -b 4096 -ub 4096 -np 1 --jinja "$@"
