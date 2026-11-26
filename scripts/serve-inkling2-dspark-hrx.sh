#!/usr/bin/env bash
set -eu

usage() {
    echo "Usage: $0 TARGET.gguf DSPARK.gguf [llama-server options...]"
    echo "Use --server FILE before the model paths to select the executable."
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
server="$root/bin/llama-server"
[[ -x $server ]] || server="$root/build/bin/llama-server"
if [[ ${1:-} == --server && $# -ge 2 ]]; then
    server=$2
    shift 2
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
exec "$server" -m "$target" -md "$draft" --spec-type draft-dspark \
    -ngl 99 -ngld 99 \
    --spec-draft-n-max 3 --backend-sampling --temp 0 \
    -fa on -c 40960 -b 4096 -ub 4096 -np 1 --jinja "$@"
