#!/usr/bin/env bash
# Model-level correctness on Strix Halo (gfx1151, ROCm): this fork vs the upstream ggml-org/llama.cpp commit it is
# based on (merge-base of HEAD and upstream master, i.e. the latest upstream master commit contained in the fork).
#
# For every model in $MODELS_DIR:
#   1. upstream llama-perplexity saves its logits (--kl-divergence-base) and reports its own final PPL
#   2. floor: upstream again, with a benign runtime change ($FLOOR_ARGS, default -ub 256), against those logits.
#      How far upstream drifts from itself is the noise floor for KLD and top-1: a different ubatch changes the
#      GEMM shapes and with them upstream's own kernel choice (e.g. MMQ vs dequant + hipBLAS) and rounding.
#   3. this fork against the same logits (--kl-divergence): PPL(Q), mean KLD, same top p
#
# The fork's PPL(Q) is compared against the upstream run's own final PPL, NOT against the "Mean PPL(base)"
# the fork prints: that one is recomputed from the saved base log-probs, which the KL base format stores as
# uint16 logits inside a 16-nat window (tools/perplexity/perplexity.cpp), so it reads low for some models
# (e.g. gemma) even when the fork matches upstream. PPL(Q) and the upstream final PPL are both computed over
# the same tokens (the second half of every chunk).
#
# A model fails if any of these holds:
#   - PPL(fork)/PPL(upstream) - 1 > $PPL_TOL                                 (one-sided: a lower PPL never fails)
#   - % tokens whose top-1 differs from upstream > max($TOP_TOL, $FLOOR_K x floor)
#   - mean KLD vs upstream                       > max($KLD_TOL, $FLOOR_K x floor)
#   - a run did not offload every layer to the GPU, or its output could not be parsed
#
# Meant to run inside the ROCm container (see .github/workflows/strix-halo-correctness.yml), but works anywhere
# with ROCm, cmake and ninja:
#   SRC=. MODELS_DIR=~/models/ci TEXT=~/wikitext-2-raw/wiki.test.raw CACHE=~/ci-cache ci/strix-halo/correctness.sh
set -euo pipefail

SRC=${SRC:-$(pwd)}
MODELS_DIR=${MODELS_DIR:-/models}
TEXT=${TEXT:-/data/wiki.test.raw}
CACHE=${CACHE:-/cache}
CHUNKS=${CHUNKS:-16}
CTX=${CTX:-512}
PPL_TOL=${PPL_TOL:-0.01}    # max relative PPL increase vs upstream
TOP_TOL=${TOP_TOL:-3}       # min allowed limit, % of tokens whose top-1 token differs from upstream
KLD_TOL=${KLD_TOL:-0.005}   # min allowed limit, mean KL divergence vs upstream
FLOOR_K=${FLOOR_K:-2}       # the fork may drift from upstream up to FLOOR_K times as much as upstream drifts from itself
read -r -a FLOOR_ARGS <<< "${FLOOR_ARGS:--ub 256}"
UPSTREAM_URL=${UPSTREAM_URL:-https://github.com/ggml-org/llama.cpp.git}
OUT=${OUT:-$SRC/correctness}

mkdir -p "$CACHE" "$OUT"

build() { # <src dir> <build dir>
    cmake -S "$1" -B "$2" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 \
        -DCMAKE_HIP_COMPILER="$(hipconfig -l)/clang" \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_BUILD_TOOLS=ON -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF > "$2.cmake.log"
    cmake --build "$2" --target llama-perplexity -j "$(nproc)" > "$2.build.log" 2>&1 || { tail -50 "$2.build.log"; exit 1; }
}

ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null || hipconfig --version 2>/dev/null || echo unknown)
HIPCC_VERSION=$(hipconfig -l >/dev/null 2>&1 && "$(hipconfig -l)/clang" --version 2>/dev/null | head -1 || echo unknown)
echo "== environment"
echo "ROCm $ROCM_VERSION, $HIPCC_VERSION, image ${IMAGE:-unknown} ${IMAGE_ID:-}"

echo "== upstream llama.cpp: latest upstream master commit contained in this fork"
# needs the fork's history (checkout with fetch-depth: 0)
git -C "$SRC" fetch -q --filter=blob:none "$UPSTREAM_URL" master
BASE=$(git -C "$SRC" merge-base HEAD FETCH_HEAD)
UP=$CACHE/upstream
[[ -d $UP/.git ]] || git init -q "$UP"
git -C "$UP" fetch -q --depth 1 "$UPSTREAM_URL" "$BASE"
git -C "$UP" checkout -q --force FETCH_HEAD
UP_SHA=$(git -C "$UP" rev-parse --short HEAD)
echo "upstream $UP_SHA ($(git -C "$UP" log -1 --format=%cs))"
build "$UP" "$CACHE/build-upstream"

echo "== fork"
FORK_SHA=$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)
echo "fork $FORK_SHA"
build "$SRC" "$SRC/build-correctness"

# -lv 4: keep llama's info messages (GPU device, layer offload) in the logs
PPL_ARGS=(-f "$TEXT" -c "$CTX" --chunks "$CHUNKS" -ngl 99 -fa on --seed 1 -lv 4)

SUMMARY=$OUT/summary.md
GPU=unknown

fail=0
shopt -s nullglob
models=("$MODELS_DIR"/*.gguf)
[[ ${#models[@]} -gt 0 ]] || { echo "no models in $MODELS_DIR"; exit 1; }

# <log> <pattern>: the value after the last ':' of the first matching line, '' if there is none
val() { grep -m1 "$2" "$1" | sed -E 's/.*: *//; s/ *±.*//; s/ *%//' | tr -d ' ' || true; }
# <log>: "ok" if every layer was offloaded to the GPU, else what was found
offload() {
    local l
    l=$(grep -m1 -oE 'offloaded [0-9]+/[0-9]+ layers to GPU' "$1" | grep -oE '[0-9]+/[0-9]+' || true)
    if [[ -z $l ]]; then echo "no offload line"; elif [[ ${l%/*} == "${l#*/}" ]]; then echo ok; else echo "offloaded $l"; fi
}

rows=()
for m in "${models[@]}"; do
    # split models: only pass the first shard, llama.cpp finds the rest next to it
    [[ $m =~ -[0-9]{5}-of-[0-9]{5}\.gguf$ && ! $m =~ -00001-of-[0-9]{5}\.gguf$ ]] && continue
    name=$(basename "$m" .gguf)
    name=${name%-00001-of-*}
    echo "== $name"
    base=$CACHE/kld-$name.bin
    up_log=$OUT/$name.upstream.log
    fl_log=$OUT/$name.floor.log
    fk_log=$OUT/$name.fork.log
    rm -f "$base"

    if ! "$CACHE/build-upstream/bin/llama-perplexity" -m "$m" "${PPL_ARGS[@]}" --kl-divergence-base "$base" > "$up_log" 2>&1; then
        echo "upstream failed on $name"; tail -20 "$up_log"
        rows+=("| $name | error | | | | | | FAIL (upstream run) |"); fail=1; rm -f "$base"; continue
    fi
    [[ $GPU == unknown ]] && GPU=$(grep -m1 -oE 'ROCm0 \([^)]*\)' "$up_log" || true)
    [[ -n $GPU ]] || GPU=unknown

    floor_ok=1
    if ! "$CACHE/build-upstream/bin/llama-perplexity" -m "$m" "${PPL_ARGS[@]}" "${FLOOR_ARGS[@]}" \
            --kl-divergence-base "$base" --kl-divergence > "$fl_log" 2>&1; then
        echo "floor run failed on $name, using the fixed limits"; tail -20 "$fl_log"
        floor_ok=0
    fi

    if ! "$SRC/build-correctness/bin/llama-perplexity" -m "$m" "${PPL_ARGS[@]}" \
            --kl-divergence-base "$base" --kl-divergence > "$fk_log" 2>&1; then
        echo "fork failed on $name"; tail -20 "$fk_log"
        rows+=("| $name | | error | | | | | FAIL (fork run) |"); fail=1; rm -f "$base"; continue
    fi
    rm -f "$base"

    ppl_b=$(grep -m1 'Final estimate: PPL' "$up_log" | sed -E 's/.*PPL = *//; s/ *\+.*//; s/ //g' || true)
    ppl_q=$(val "$fk_log" 'Mean PPL(Q)  ')
    kld=$(val "$fk_log" 'Mean    KLD')
    kld_max=$(val "$fk_log" 'Maximum KLD')
    top=$(val "$fk_log" 'Same top p:')
    kld_fl=""; top_fl=""
    if [[ $floor_ok == 1 ]]; then
        kld_fl=$(val "$fl_log" 'Mean    KLD')
        top_fl=$(val "$fl_log" 'Same top p:')
    fi

    off=""
    logs=("$up_log" "$fk_log")
    [[ $floor_ok == 1 ]] && logs+=("$fl_log")
    for l in "${logs[@]}"; do
        o=$(offload "$l")
        [[ $o == ok ]] || off="$off $(basename "$l" .log | sed 's/.*\.//'): $o;"
    done

    # prints: ratio kld_limit top_limit reasons
    read -r ratio kld_lim top_lim why < <(LC_ALL=C awk \
        -v q="$ppl_q" -v b="$ppl_b" -v t="$PPL_TOL" \
        -v top="$top" -v tt="$TOP_TOL" -v topf="$top_fl" \
        -v k="$kld" -v kt="$KLD_TOL" -v kf="$kld_fl" -v f="$FLOOR_K" 'BEGIN {
        if (q == "" || b == "" || top == "" || k == "") { print "- - - parse_error"; exit }
        r = q / b
        kl = kt; if (kf != "" && f * kf > kl) kl = f * kf
        tl = tt; if (topf != "" && f * (100 - topf) > tl) tl = f * (100 - topf)
        w = ""
        if (r - 1 > t)       w = w ",PPL"
        if (100 - top > tl)  w = w ",top-1"
        if (k > kl)          w = w ",KLD"
        if (w == "") w = ",-"
        printf "%.6f %.6f %.3f %s\n", r, kl, 100 - tl, substr(w, 2) }')
    [[ $why == - ]] && why=""
    [[ -n $off ]] && why="${why:+$why,}offload"
    if [[ -z $why ]]; then
        res=OK
    else
        why=${why//_/ }
        res="FAIL (${why//,/, })"; fail=1
    fi
    [[ -n $off ]] && echo "offload:$off"
    echo "PPL upstream $ppl_b  fork $ppl_q  ratio $ratio  KLD $kld (floor ${kld_fl:-n/a}, limit $kld_lim, max $kld_max)  same top p $top% (floor ${top_fl:-n/a}%, limit $top_lim%)  -> $res"
    pct() { [[ -n $1 && $1 != - ]] && echo "$1 %" || echo "${2:-}"; }
    rows+=("| $name | $ppl_b | $ppl_q | $ratio | $kld | ${kld_fl:-n/a} | $kld_lim | $(pct "$top") | $(pct "$top_fl" n/a) | $(pct "$top_lim" -) | $kld_max | $res |")
done

{
    echo "### Model correctness: fork \`$FORK_SHA\` vs upstream llama.cpp \`$UP_SHA\` (gfx1151, ROCm)"
    echo
    echo "GPU: $GPU. ROCm $ROCM_VERSION, $HIPCC_VERSION, image \`${IMAGE:-unknown}\`${IMAGE_ID:+ (\`${IMAGE_ID:0:19}\`)}."
    echo
    echo "wikitext-2 test, ctx $CTX, $CHUNKS chunks. PPL upstream is upstream's own final estimate."
    echo "Floor: upstream vs upstream with \`${FLOOR_ARGS[*]}\`."
    echo "Fails if PPL(fork)/PPL(upstream) - 1 > $PPL_TOL (a lower PPL never fails), if mean KLD > max($KLD_TOL, $FLOOR_K x floor),"
    echo "if the share of tokens with a different top-1 > max($TOP_TOL %, $FLOOR_K x floor), or if a run did not offload every layer to the GPU."
    echo
    echo "| model | PPL upstream | PPL fork | ratio | mean KLD | KLD floor | KLD limit | same top p | top p floor | top p limit | max KLD | result |"
    echo "|---|---|---|---|---|---|---|---|---|---|---|---|"
    printf '%s\n' "${rows[@]}"
} > "$SUMMARY"

echo
cat "$SUMMARY"
exit $fail
