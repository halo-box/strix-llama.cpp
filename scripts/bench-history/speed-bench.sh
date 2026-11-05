#!/bin/bash
# SPEED-Bench decode measurement of bench-gfx1151-history.yml: start llama-server on one model with
# speculative decoding, run the client in tools/server/bench/speed-bench against it once per bench
# config, then stop the server. One server, so the model is loaded once for every split.
#
# usage: speed-bench.sh <llama-server> <client dir> <python> <name> <model path> <out prefix> [server flags...]
#        writes <out prefix>-<bench>.json and <out prefix>-<bench>.log per bench
#
# Settings come from the environment, so the workflow records them once for the whole lane:
#   SPEED_BENCH_BENCHES SPEED-Bench configs to run, comma-separated, default throughput_1k,throughput_32k
#   SPEED_BENCH_BENCH   single-config fallback, used when SPEED_BENCH_BENCHES is unset
#   SPEED_BENCH_LIMIT   samples per category, default 2
#   SPEED_BENCH_OSL     output tokens per request, default 4096
#   SPEED_BENCH_CTX     server context, default 40960, so the 32k split's prompt plus its output fits
#   SPEED_BENCH_BATCH   server -b, default 4096
#   SPEED_BENCH_UBATCH  server -ub, default 4096
#   SPEED_BENCH_SPEC    server --spec-type, default draft-mtp
#   SPEED_MTP_DRAFT     path of the MTP draft head passed with -md; unset means no draft
#   SPEED_BENCH_PORT    server port, default 8199
#
# Writes command lines to commands.tsv and one result row per bench to results.tsv next to the output.

set -uo pipefail
server=$1
client=$2
python=$3
name=$4
path=$5
prefix=$6
shift 6
extra=("$@")
logs=$(dirname "$prefix")

benches=${SPEED_BENCH_BENCHES:-${SPEED_BENCH_BENCH:-throughput_1k,throughput_32k}}
limit=${SPEED_BENCH_LIMIT:-2}
osl=${SPEED_BENCH_OSL:-4096}
ctx=${SPEED_BENCH_CTX:-40960}
batch=${SPEED_BENCH_BATCH:-4096}
ubatch=${SPEED_BENCH_UBATCH:-4096}
spec=${SPEED_BENCH_SPEC:-draft-mtp}
draft=${SPEED_MTP_DRAFT:-}
port=${SPEED_BENCH_PORT:-8199}

# shell-quote for the record
quote() {
    local a out=""
    for a in "$@"; do
        case "$a" in
            *[!A-Za-z0-9_./:=,+@-]* | '') a="'${a//\'/\'\\\'\'}'" ;;
        esac
        out="${out:+$out }$a"
    done
    printf '%s' "$out"
}

srv_log="$logs/server-$name.log"
cmd=("$server" -m "$path" -c "$ctx" -b "$batch" -ub "$ubatch" -ngl 99 -fa on -ctk f16 -ctv f16
     -np 1 -lzm on-direct --spec-type "$spec" --host 127.0.0.1 --port "$port" --no-webui)
# shellcheck disable=SC2206
[ ${#extra[@]} -eq 0 ] || cmd+=("${extra[@]}")
# the target GGUF carries no MTP head, so the head arrives as a sidecar
if [ -n "$draft" ]; then
    if [ ! -s "$draft" ]; then
        echo "::error::MTP draft head is missing: $draft" >&2
        exit 1
    fi
    cmd+=(-md "$draft")
fi
printf 'speed-bench-server-%s\t%s\n' "$name" "$(quote "${cmd[@]}")" >> "$logs/commands.tsv"

echo "=== llama-server $name (--spec-type $spec, -c $ctx, -ub $ubatch, draft ${draft:-none}) ==="
"${cmd[@]}" > "$srv_log" 2>&1 &
srv_pid=$!
trap 'kill -9 "$srv_pid" 2> /dev/null' EXIT

# the server answers /health with 503 while it loads the model and 200 once it can serve
rc=0
ready=0
for _ in $(seq 1 900); do
    if ! kill -0 "$srv_pid" 2> /dev/null; then
        echo "::error::llama-server exited before it was ready" >&2
        tail -40 "$srv_log" >&2
        rc=1
        break
    fi
    if curl -fsS "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 2
done
if [ "$rc" -eq 0 ] && [ "$ready" -ne 1 ]; then
    echo "::error::llama-server did not become ready within 30 min" >&2
    tail -40 "$srv_log" >&2
    rc=1
fi

if [ "$rc" -eq 0 ]; then
    echo "server ready, MTP and device lines:"
    grep -E 'speculative|draft|MTP|ggml_(hip|cuda)|using device' "$srv_log" | head -20
    for bench in ${benches//,/ }; do
        out="$prefix-$bench.json"
        cli=("$python" "$client/speed_bench.py" --url "127.0.0.1:$port" --bench "$bench" --category all
             --limit "$limit" --osl "$osl" --concurrency 1 --output "$out")
        printf 'speed-bench-%s-%s\t%s\n' "$name" "$bench" "$(quote "${cli[@]}")" >> "$logs/commands.tsv"
        echo "=== speed_bench.py $name $bench limit $limit osl $osl ==="
        t0=$(date +%s.%N)
        timeout 2700 "${cli[@]}" 2>&1 | tee "$logs/speed-bench-$name-$bench.log"
        bench_rc=${PIPESTATUS[0]}
        wall=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }')
        echo "$name $bench: exit $bench_rc, $wall s"
        if [ "$bench_rc" -ne 0 ] || [ ! -s "$out" ]; then
            echo "::error::SPEED-Bench $bench failed for $name (exit $bench_rc)" >&2
            tail -20 "$srv_log" >&2
            bench_rc=1
            rc=1
        fi
        printf 'speed-bench\t%s\t%s\t%s\t%s\n' "$name" "$bench" "$bench_rc" "$wall" >> "$logs/results.tsv"
    done
fi

# a server that never came up still gets a row per bench, so the run page shows what was attempted
if [ "$rc" -ne 0 ] && [ "$ready" -ne 1 ]; then
    for bench in ${benches//,/ }; do
        printf 'speed-bench\t%s\t%s\t%s\t%s\n' "$name" "$bench" "$rc" 0 >> "$logs/results.tsv"
    done
fi

kill -TERM "$srv_pid" 2> /dev/null
for _ in $(seq 1 30); do
    kill -0 "$srv_pid" 2> /dev/null || break
    sleep 1
done
kill -9 "$srv_pid" 2> /dev/null
wait "$srv_pid" 2> /dev/null
trap - EXIT

exit "$rc"
