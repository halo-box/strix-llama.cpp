#!/bin/bash
# Model files and llama-bench runs of bench-gfx1151-history.yml, shared by the HIP and the Vulkan
# step so both backends use the same pinned files, the same checks and the same flags.
#
# usage: model.sh fetch <name>       download and verify into $RUNNER_TEMP/models, print the path
#        model.sh fits <name>...     exit 0 if the named models still fit on the disk
#        model.sh bench <kind> <llama-bench> <name> <path>
#                                    kind is llama-bench (ROCm0) or llama-bench-vulkan (Vulkan0)
# fetch exits 2 for too little memory, 3 for too little disk, 4 for a failed download and 5 for a
# size or sha256 mismatch. Records go to models.tsv, commands.tsv and results.tsv in $RUNNER_TEMP/logs.

set -uo pipefail
logs="$RUNNER_TEMP/logs"
model_root="$RUNNER_TEMP/models"
# held back for KV cache and compute buffers when checking that a model fits in memory
margin_bytes=$(( 16 * 1024 * 1024 * 1024 ))

# repo|revision|file|bytes|sha256, read from the Hugging Face API on 2026-10-20
spec() {
    case "$1" in
        qwen36-35b-a3b)
            echo 'bartowski/Qwen_Qwen3.6-35B-A3B-GGUF|5c2410d71524f4f72b023ce8daf7a80528226d5f|Qwen_Qwen3.6-35B-A3B-Q8_0.gguf|37812647552|bc57de096ec30e45773aa86103f35a8a32cea01f2193a53a3c636d618e4434e9'
            ;;
        qwen38-27b)
            echo 'unsloth/Qwen3.8-27B-GGUF|4ca720788d1e01f1bff70c033e0d0028fd02e502|Qwen3.8-27B-Q8_0.gguf|29047086048|a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348'
            ;;
        *) return 1 ;;
    esac
}

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

mem_fits() {
    local bytes=$1 what=$2 avail
    avail=$(awk '/^MemAvailable:/ {print $2 * 1024; exit}' /proc/meminfo)
    if [ $(( bytes + margin_bytes )) -gt "$avail" ]; then
        echo "::error::SKIP $what: needs $(( (bytes + margin_bytes) >> 30 )) GiB of memory, $(( avail >> 30 )) GiB available" >&2
        return 1
    fi
}

fetch() {
    local name=$1 repo rev file bytes sha dest out free got_bytes got_sha
    IFS='|' read -r repo rev file bytes sha <<< "$(spec "$name")"
    dest="$model_root/$name"
    out="$dest/$file"
    note() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$repo" "$rev" "$file" "$bytes" "$sha" "$1" "$2" >> "$logs/models.tsv"; }

    mem_fits "$bytes" "$name" || { note "" skipped-memory; return 2; }
    mkdir -p "$dest"
    free=$(df -B1 --output=avail "$dest" | tail -1)
    if [ "$free" -lt $(( bytes + bytes / 5 )) ]; then
        echo "::error::SKIP $name: needs $(( (bytes + bytes / 5) >> 30 )) GiB of disk, $(( free >> 30 )) GiB free" >&2
        note "" skipped-disk
        return 3
    fi
    local url="https://huggingface.co/$repo/resolve/$rev/$file" auth=()
    echo "fetching $url" >&2
    [ -z "${HF_TOKEN:-}" ] || auth=(-H "Authorization: Bearer $HF_TOKEN")
    if ! curl -fsSL --retry 5 --retry-delay 5 --retry-all-errors -C - "${auth[@]}" -o "$out" "$url"; then
        echo "::error::download failed: $url" >&2
        note "" download-failed
        rm -rf "$dest"
        return 4
    fi
    got_bytes=$(stat -c %s "$out")
    got_sha=$(sha256sum "$out" | cut -d' ' -f1)
    if [ "$got_bytes" != "$bytes" ] || [ "$got_sha" != "$sha" ]; then
        echo "::error::$name: got $got_bytes bytes sha256 $got_sha, pinned $bytes bytes sha256 $sha" >&2
        note "$got_sha" hash-mismatch
        rm -rf "$dest"
        return 5
    fi
    note "$got_sha" verified
    printf '%s\n' "$out"
}

fits() {
    local need=0 name bytes free
    for name in "$@"; do
        IFS='|' read -r _ _ _ bytes _ <<< "$(spec "$name")"
        need=$(( need + bytes + bytes / 5 ))
    done
    mkdir -p "$model_root"
    free=$(df -B1 --output=avail "$model_root" | tail -1)
    [ "$free" -ge "$need" ]
}

bench() {
    local kind=$1 bin=$2 name=$3 path=$4 bytes rc rows
    IFS='|' read -r _ _ _ bytes _ <<< "$(spec "$name")"
    if ! mem_fits "$bytes" "$kind $name"; then
        printf '%s\t%s\tskipped-memory\t0\n' "$kind" "$name" >> "$logs/results.tsv"
        return 1
    fi
    # -fa and the KV types are spelled out so the numbers do not move with the auto defaults.
    # HIP_LAUNCH_BLOCKING is not set: it serialises every kernel launch.
    # 3 h per model: the 27B HIP run took 1 h 53 min in run 39469034755.
    local cmd=(timeout 10800 "$bin"
        -m "$path"
        -p "$BENCH_N_PROMPT" -n "$BENCH_N_GEN"
        -b "$BENCH_N_BATCH" -ub "$BENCH_UBATCH"
        -d "$BENCH_DEPTHS"
        -lzm on-direct
        -ngl 99 -fa on -ctk f16 -ctv f16
        -r "$BENCH_REPS" -o jsonl --progress)
    printf '%s-%s\t%s\n' "$kind" "$name" "$(quote "${cmd[@]}")" >> "$logs/commands.tsv"
    echo "=== $kind $name ==="
    "${cmd[@]}" > "$logs/$kind-$name.jsonl" 2> "$logs/$kind-$name.log"
    rc=$?
    cat "$logs/$kind-$name.log"
    rows=$(wc -l < "$logs/$kind-$name.jsonl")
    printf '%s\t%s\t%s\t%s\n' "$kind" "$name" "$rc" "$rows" >> "$logs/results.tsv"
    echo "$kind $name: exit $rc, $rows jsonl rows"
    if [ "$rc" -ne 0 ] || [ "$rows" -eq 0 ]; then
        echo "::error::$kind failed for $name (exit $rc, $rows rows)"
        return 1
    fi
}

action=$1
shift
case "$action" in
    fetch | fits | bench) "$action" "$@" ;;
    *) echo "usage: model.sh fetch|fits|bench ..." >&2; exit 2 ;;
esac
