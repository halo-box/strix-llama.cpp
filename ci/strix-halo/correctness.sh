#!/usr/bin/env bash
# Model-level correctness on Strix Halo (gfx1151, ROCm): this fork vs the upstream ggml-org/llama.cpp commit it is
# based on (merge-base of HEAD and upstream master, i.e. the latest upstream master commit contained in the fork).
#
# For every model in $MODELS_DIR:
#   1. upstream llama-perplexity saves its logits (--kl-divergence-base) and reports its own final PPL
#   2. this fork's llama-perplexity reads them (--kl-divergence) and reports PPL(Q), KLD, same-top-p
# The fork's PPL(Q) is compared against the upstream run's own final PPL, NOT against the "Mean PPL(base)"
# the fork prints: that one is recomputed from the saved base log-probs, which the KL base format stores as
# uint16 logits inside a 16-nat window (tools/perplexity/perplexity.cpp), so it reads low for models with
# final logit softcapping (e.g. gemma) even when the fork matches upstream. PPL(Q) and the upstream final PPL
# are both computed over the same tokens (the second half of every chunk).
# Fails if |PPL(fork)/PPL(upstream) - 1| > $PPL_TOL or if the top-1 token differs from upstream on more than $TOP_TOL %
# of the tokens (same top p < 100 - $TOP_TOL), or if the mean KLD vs upstream > $KLD_TOL, for any model.
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
PPL_TOL=${PPL_TOL:-0.01}
TOP_TOL=${TOP_TOL:-3}   # max % of tokens whose top-1 token differs from upstream
KLD_TOL=${KLD_TOL:-0.005}   # max mean KL divergence vs upstream
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

PPL_ARGS=(-f "$TEXT" -c "$CTX" --chunks "$CHUNKS" -ngl 99 -fa on --seed 1)

SUMMARY=$OUT/summary.md
{
    echo "### Model correctness: fork \`$FORK_SHA\` vs upstream llama.cpp \`$UP_SHA\` (gfx1151, ROCm)"
    echo
    echo "wikitext-2 test, ctx $CTX, $CHUNKS chunks. PPL upstream is upstream's own final estimate. Fails if |PPL ratio - 1| > $PPL_TOL or same top p < $((100 - TOP_TOL)) % or mean KLD > $KLD_TOL."
    echo
    echo "| model | PPL upstream | PPL fork | ratio | mean KLD | max KLD | same top p | result |"
    echo "|---|---|---|---|---|---|---|---|"
} > "$SUMMARY"

fail=0
shopt -s nullglob
models=("$MODELS_DIR"/*.gguf)
[[ ${#models[@]} -gt 0 ]] || { echo "no models in $MODELS_DIR"; exit 1; }

for m in "${models[@]}"; do
    # split models: only pass the first shard, llama.cpp finds the rest next to it
    [[ $m =~ -[0-9]{5}-of-[0-9]{5}\.gguf$ && ! $m =~ -00001-of-[0-9]{5}\.gguf$ ]] && continue
    name=$(basename "$m" .gguf)
    name=${name%-00001-of-*}
    echo "== $name"
    base=$CACHE/kld-$name.bin
    up_log=$OUT/$name.upstream.log
    fk_log=$OUT/$name.fork.log

    "$CACHE/build-upstream/bin/llama-perplexity" -m "$m" "${PPL_ARGS[@]}" --kl-divergence-base "$base" > "$up_log" 2>&1 \
        || { echo "upstream failed on $name"; tail -20 "$up_log"; echo "| $name | error | | | | | | FAIL |" >> "$SUMMARY"; fail=1; continue; }
    "$SRC/build-correctness/bin/llama-perplexity" -m "$m" "${PPL_ARGS[@]}" --kl-divergence-base "$base" --kl-divergence > "$fk_log" 2>&1 \
        || { echo "fork failed on $name"; tail -20 "$fk_log"; echo "| $name | | error | | | | | FAIL |" >> "$SUMMARY"; fail=1; rm -f "$base"; continue; }
    rm -f "$base"

    val() { grep -m1 "$1" "$fk_log" | sed -E 's/.*: *//; s/ *±.*//; s/ *%//' | tr -d ' '; }
    ppl_q=$(val 'Mean PPL(Q)  ')
    # upstream's own final PPL; do NOT use the fork's "Mean PPL(base)" (see the note at the top of this file)
    ppl_b=$(grep -m1 'Final estimate: PPL' "$up_log" | sed -E 's/.*PPL = *//; s/ *\+.*//; s/ //g' || true)
    if [[ -n $ppl_q && -n $ppl_b ]]; then
        ratio=$(LC_ALL=C awk -v q="$ppl_q" -v b="$ppl_b" 'BEGIN { printf "%.6f", q/b }')
    else
        ratio=""
    fi
    kld=$(val 'Mean    KLD')
    kld_max=$(val 'Maximum KLD')
    top=$(val 'Same top p:')

    why=$(LC_ALL=C awk -v r="$ratio" -v t="$PPL_TOL" -v top="$top" -v tt="$TOP_TOL" -v k="$kld" -v kt="$KLD_TOL" 'BEGIN {
        if (r == "" || top == "" || k == "") { print "parse error"; exit }
        d = r - 1; if (d < 0) d = -d
        w = ""
        if (d > t)        w = w " PPL"
        if (100 - top > tt) w = w " top-1"
        if (k > kt)       w = w " KLD"
        print w }')
    if [[ -z $why ]]; then
        res=OK
    else
        res="FAIL ($(echo $why | sed 's/ /, /g'))"; fail=1
    fi
    echo "PPL upstream $ppl_b  fork $ppl_q  ratio $ratio  KLD $kld (max $kld_max)  same top p $top%  -> $res"
    echo "| $name | $ppl_b | $ppl_q | $ratio | $kld | $kld_max | $top % | $res |" >> "$SUMMARY"
done

echo
cat "$SUMMARY"
exit $fail
