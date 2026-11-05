#!/bin/bash
# Model files and llama-bench runs of bench-gfx1151-history.yml, shared by the HIP and the Vulkan
# step so both backends use the same pinned files, the same checks and the same flags.
#
# usage: model.sh fetch <name>       download and verify into $RUNNER_TEMP/models, print the path
#        model.sh fits <name>...     exit 0 if the named models still fit on the disk
#        model.sh have <name>        print the first shard of an already fetched model, exit 1 if absent
#        model.sh flags <name>       print the model's extra llama flags
#        model.sh speed <name>       print the SPEED-Bench configs of a model that gets a SPEED-Bench run, exit 1 if none
#        model.sh draft <name>       download and verify the model's MTP draft head, print its path, exit 1 if none
#        model.sh bench <kind> <llama-bench> <name> <path>
#                                    kind is llama-bench (ROCm0) or llama-bench-vulkan (Vulkan0)
# fetch exits 2 for too little memory, 3 for too little disk, 4 for a failed download and 5 for a
# size or sha256 mismatch. Records go to models.tsv, commands.tsv and results.tsv in $RUNNER_TEMP/logs.

set -uo pipefail
logs="$RUNNER_TEMP/logs"
model_root="$RUNNER_TEMP/models"
# held back for KV cache and compute buffers when checking that a model fits in memory
margin_bytes=$(( 16 * 1024 * 1024 * 1024 ))

# repo|revision|total bytes|manifest|extra llama flags, read from the Hugging Face API
# A manifest entry is file:bytes:sha256. A split model lists one entry per shard and the total
# covers all of them; llama.cpp takes the first shard and finds the rest beside it.
spec() {
    case "$1" in
    qwen36-35b-a3b) echo 'bartowski/Qwen_Qwen3.6-35B-A3B-GGUF|5c2410d71524f4f72b023ce8daf7a80528226d5f|37812647552|Qwen_Qwen3.6-35B-A3B-Q8_0.gguf:37812647552:bc57de096ec30e45773aa86103f35a8a32cea01f2193a53a3c636d618e4434e9|' ;;
    qwen38-27b) echo 'unsloth/Qwen3.8-27B-GGUF|4ca720788d1e01f1bff70c033e0d0028fd02e502|29047086048|Qwen3.8-27B-Q8_0.gguf:29047086048:a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348|' ;;
    # the second shard is the PLE table alone and -lzm on-direct never allocates it, so 36.5 GiB of
    # this 63.3 GiB file is resident; -lm dio keeps a 68 GB mmap out of the load path
    flash-next-iq2xs) echo 'ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF|ed59f92082b1e93c0e96d60a8b11aab089b52f09|68026093024|IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf:39225954592:92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7 IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf:28800138432:316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113|-lm dio' ;;
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
    local name=$1 repo rev bytes manifest extra dest free first entry file want out url got_bytes got_sha
    IFS='|' read -r repo rev bytes manifest extra <<< "$(spec "$name")"
    dest="$model_root/$name"
    first=${manifest%% *}
    note() {
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$repo" "$rev" "${first%%:*}" "$bytes" "${first##*:}" "$1" "$2" >> "$logs/models.tsv"
    }
    mem_fits "$bytes" "$name" || { note "" skipped-memory; return 2; }
    mkdir -p "$dest"
    free=$(df -B1 --output=avail "$dest" | tail -1)
    if [ "$free" -lt $(( bytes + bytes / 5 )) ]; then
        echo "::error::SKIP $name: needs $(( (bytes + bytes / 5) >> 30 )) GiB of disk, $(( free >> 30 )) GiB free" >&2
        note "" skipped-disk
        return 3
    fi
    local auth=()
    [ -z "${HF_TOKEN:-}" ] || auth=(-H "Authorization: Bearer $HF_TOKEN")
    for entry in $manifest; do
        file=${entry%%:*}
        want=${entry#*:}
        out="$dest/$file"
        url="https://huggingface.co/$repo/resolve/$rev/$file"
        mkdir -p "$(dirname "$out")"
        echo "fetching $url" >&2
        if ! curl -fsSL --retry 5 --retry-delay 5 --retry-all-errors -C - "${auth[@]}" -o "$out" "$url"; then
            echo "::error::download failed: $url" >&2
            note "" download-failed
            rm -rf "$dest"
            return 4
        fi
        got_bytes=$(stat -c %s "$out")
        got_sha=$(sha256sum "$out" | cut -d' ' -f1)
        if [ "$got_bytes" != "${want%%:*}" ] || [ "$got_sha" != "${want##*:}" ]; then
            echo "::error::$name $file: got $got_bytes bytes sha256 $got_sha, pinned ${want%%:*} bytes sha256 ${want##*:}" >&2
            note "$got_sha" hash-mismatch
            rm -rf "$dest"
            return 5
        fi
    done
    note "$got_sha" verified
    printf '%s\n' "$dest/${first%%:*}"
}

fits() {
    local need=0 name bytes
    for name in "$@"; do
        IFS='|' read -r _ _ bytes _ <<< "$(spec "$name")"
        need=$(( need + bytes + bytes / 5 ))
    done
    mkdir -p "$model_root"
    free=$(df -B1 --output=avail "$model_root" | tail -1)
    [ "$free" -ge "$need" ]
}

have() {
    local name=$1 repo rev bytes manifest first dest
    IFS='|' read -r repo rev bytes manifest _ <<< "$(spec "$name")"
    first=${manifest%% *}
    dest="$model_root/$name/${first%%:*}"
    [ -s "$dest" ] || return 1
    printf '%s\n' "$dest"
}

flags() {
    local extra
    IFS='|' read -r _ _ _ _ extra <<< "$(spec "$1")"
    printf '%s\n' "$extra"
}

# The models that get a SPEED-Bench run. Prints the SPEED-Bench configs, comma-separated. These are not
# the models that carry an MTP head: see draft_spec() below for the head they need.
speed() {
    case "$1" in
    flash-next-iq2xs) echo 'throughput_1k,throughput_32k' ;;
    *) return 1 ;;
    esac
}

# The MTP draft head a model needs, repo|revision|total bytes|file:bytes:sha256. The lane's GGUFs carry
# no MTP tensor (IQ2_XS checked 2026-10-30), so llama-server takes a sidecar with -md. A self-contained
# head carries its own embeddings and LM head; a -shared- one borrows them from the target, which this
# tree supports (src/llama-context.cpp). unsloth calls shared-Q8_0 the fastest of the set: swapping is
# this one line, plus its bytes and sha256.
draft_spec() {
    case "$1" in
    flash-next-iq2xs) echo 'unsloth/Qwen3.8-Flash-Next-GGUF|766911a6b7369840a91dbcd95f9f997acaab6cd6|2786204800|MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf:2786204800:8087dbb39fc73f79ac069b1debe4069a44b73ada900c8bd761569d43a5a90231' ;;
    *) return 1 ;;
    esac
}

# download and verify the draft head into $RUNNER_TEMP/models/<name>-mtp, print its path
# exits 3 for too little disk, 4 for a failed download and 5 for a size or sha256 mismatch
draft() {
    local name=$1 repo rev bytes manifest file want dest out free got_bytes got_sha
    local auth=()
    IFS='|' read -r repo rev bytes manifest <<< "$(draft_spec "$name")" || return 1
    file=${manifest%%:*}
    want=${manifest#*:}
    dest="$model_root/$name-mtp"
    out="$dest/$(basename "$file")"
    # key, repo, revision, file, bytes, pinned sha256, actual sha256, status
    note() {
        printf 'draft\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$name" "$repo" "$rev" "$file" "$bytes" "${want##*:}" "$1" "$2" >> "$logs/drafts.tsv"
    }
    mkdir -p "$dest"
    free=$(df -B1 --output=avail "$dest" | tail -1)
    if [ "$free" -lt "$bytes" ]; then
        echo "::error::SKIP draft $name: needs $(( bytes >> 20 )) MiB of disk, $(( free >> 20 )) MiB free" >&2
        note "" skipped-disk
        return 3
    fi
    [ -z "${HF_TOKEN:-}" ] || auth=(-H "Authorization: Bearer $HF_TOKEN")
    echo "fetching draft https://huggingface.co/$repo/resolve/$rev/$file" >&2
    if ! curl -fsSL --retry 5 --retry-delay 5 --retry-all-errors -C - "${auth[@]}" -o "$out" \
        "https://huggingface.co/$repo/resolve/$rev/$file"; then
        echo "::error::download failed: draft $name" >&2
        note "" download-failed
        rm -rf "$dest"
        return 4
    fi
    got_bytes=$(stat -c %s "$out")
    got_sha=$(sha256sum "$out" | cut -d' ' -f1)
    if [ "$got_bytes" != "${want%%:*}" ] || [ "$got_sha" != "${want##*:}" ]; then
        echo "::error::draft $name: got $got_bytes bytes sha256 $got_sha, pinned ${want%%:*} bytes sha256 ${want##*:}" >&2
        note "$got_sha" hash-mismatch
        rm -rf "$dest"
        return 5
    fi
    note "$got_sha" verified
    printf '%s\n' "$out"
}

bench() {
    local kind=$1 bin=$2 name=$3 path=$4 bytes rc rows extra
    IFS='|' read -r _ _ bytes _ extra <<< "$(spec "$name")"
    if ! mem_fits "$bytes" "$kind $name"; then
        printf '%s\t%s\tskipped-memory\t0\n' "$kind" "$name" >> "$logs/results.tsv"
        return 1
    fi
    # -fa and the KV types are spelled out so the numbers do not move with the auto defaults.
    # HIP_LAUNCH_BLOCKING is not set: it serialises every kernel launch.
    # 3 h per model: the 27B HIP run took 1 h 53 min in run 39469034755.
    local cmd=(timeout 10800 "$bin" -m "$path" -p "$BENCH_N_PROMPT" -n "$BENCH_N_GEN" -b "$BENCH_N_BATCH" -ub "$BENCH_UBATCH" -d "$BENCH_DEPTHS" -lzm on-direct -ngl 99 -fa on -ctk f16 -ctv f16 -r "$BENCH_REPS" -o jsonl --progress)
    # shellcheck disable=SC2206
    [ -z "$extra" ] || cmd+=($extra)
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
    fetch | fits | have | flags | speed | draft | bench) "$action" "$@" ;;
    *) echo "usage: model.sh fetch|fits|have|flags|speed|draft|bench ..." >&2; exit 2 ;;
esac
