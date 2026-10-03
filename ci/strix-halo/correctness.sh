#!/usr/bin/env bash
# Model-level correctness on Strix Halo (gfx1151, ROCm): this fork vs the upstream ggml-org/llama.cpp commit it is
# based on (merge-base of HEAD and upstream master, i.e. the latest upstream master commit contained in the fork).
#
# For every model in $MODELS_DIR (wikitext-2 as one token stream, $CHUNKS consecutive chunks of $CTX tokens):
#   1. upstream llama-perplexity saves its logits (--kl-divergence-base) and reports its own final PPL
#   2. floor: upstream again, once per runtime change in $FLOOR_ARGS (';'-separated, default "-ub 256;-fa off"),
#      against those logits. How far upstream drifts from itself is the noise floor: a different ubatch changes the
#      GEMM shapes and with them upstream's own kernel choice (e.g. MMQ vs dequant + hipBLAS), -fa off takes the
#      other attention path. Every floor value is the worst over these runs.
#   3. this fork against the same logits (--kl-divergence): PPL(Q), mean KLD, 99th percentile KLD, same top p
#   4. only for a model that failed: the fork again with its own optimizations switched off ($STRICT_ENV). Not a
#      gate, a hint: if the failure goes away, it comes from one of the switched-off code paths.
#
# The fork's PPL(Q) is compared against the upstream run's own final PPL, NOT against the "Mean PPL(base)"
# the fork prints: that one is recomputed from the saved base log-probs, which the KL base format stores as
# uint16 logits inside a 16-nat window (tools/perplexity/perplexity.cpp), so it reads low for some models
# (e.g. gemma) even when the fork matches upstream. PPL(Q) and the upstream final PPL are both computed over
# the same tokens (the second half of every chunk).
#
# A model fails if any of these holds (floor = worst upstream-vs-upstream value):
#   - PPL(fork)/PPL(upstream) - 1        > max($PPL_TOL, $FLOOR_K x floor)    (one-sided: a lower PPL never fails)
#   - % tokens whose top-1 differs       > max($TOP_TOL, $FLOOR_K x floor)
#   - mean KLD vs upstream               > max($KLD_TOL, $FLOOR_K x floor)
#   - 99th percentile KLD vs upstream    > max($TAIL_TOL, $FLOOR_K x floor)   (the worst 1 % of the tokens)
#     (not the 99.9th percentile or the max: those rest on a handful of tokens, and upstream alone already shows
#     single tokens with KLD > 8 on long chunks when only its attention path changes)
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
CTX=${CTX:-2048}            # tokens per chunk; only the second half of every chunk is scored (>= CTX/2 tokens of context)
PPL_TOL=${PPL_TOL:-0.01}    # min allowed limit, relative PPL increase vs upstream
TOP_TOL=${TOP_TOL:-3}       # min allowed limit, % of tokens whose top-1 token differs from upstream
KLD_TOL=${KLD_TOL:-0.005}   # min allowed limit, mean KL divergence vs upstream
TAIL_TOL=${TAIL_TOL:-0.05}  # min allowed limit, 99th percentile of the per-token KL divergence vs upstream
FLOOR_K=${FLOOR_K:-2}       # the fork may drift from upstream up to FLOOR_K times as much as upstream drifts from itself
IFS=';' read -r -a FLOOR_SETS <<< "${FLOOR_ARGS:--ub 256;-fa off}"
# the fork's own code paths, off (unknown variables are ignored, so this may name switches a fork build lacks)
STRICT_ENV=${STRICT_ENV:-GGML_CUDA_DISABLE_MMB=1 GGML_CUDA_DISABLE_MMQ_TUNE=1 GGML_GDN_CHUNK=0 GGML_CUDA_DISABLE_SGMA=1 GGML_CUDA_DISABLE_MOE_SGMA=1 GGML_CUDA_DISABLE_GDN_QKNORM=1 GGML_CUDA_DISABLE_GDN_CONV=1 GGML_CUDA_DISABLE_GDN_OUTNORM=1 GGML_CUDA_DISABLE_GDN_GATE=1 GGML_CUDA_DISABLE_WEIGHTED_DOWN=1 GGML_CUDA_DISABLE_F32_DUAL=1 GGML_CUDA_DISABLE_MMID_512=1 GGML_CUDA_DISABLE_MMV_GROUP=1 GGML_CUDA_DISABLE_IDX_GEMM=1}
read -r -a STRICT <<< "$STRICT_ENV"
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

UP_BIN=$CACHE/build-upstream/bin/llama-perplexity
FK_BIN=$SRC/build-correctness/bin/llama-perplexity
# -lv 4: keep llama's info messages (GPU device, layer offload) in the logs
PPL_ARGS=(-f "$TEXT" -c "$CTX" --chunks "$CHUNKS" -ngl 99 -fa on --seed 1 -lv 4)

SUMMARY=$OUT/summary.md
GPU=unknown
rows=()
diags=()

# rewritten after every model, so a cancelled or crashed run still leaves the models done so far
write_summary() {
    {
        echo "### Model correctness: fork \`$FORK_SHA\` vs upstream llama.cpp \`$UP_SHA\` (gfx1151, ROCm)"
        echo
        echo "GPU: $GPU. ROCm $ROCM_VERSION, $HIPCC_VERSION, image \`${IMAGE:-unknown}\`${IMAGE_ID:+ (\`${IMAGE_ID:0:19}\`)}."
        echo
        echo "wikitext-2 test, ctx $CTX, $CHUNKS chunks. PPL upstream is upstream's own final estimate."
        echo "Floor: the worst of upstream vs upstream with$(printf ' `%s`,' "${FLOOR_SETS[@]}" | sed 's/,$//')."
        echo "Each cell: fork value (floor / limit). Limit = max(fixed limit, $FLOOR_K x floor); fixed limits: PPL ratio - 1 $PPL_TOL"
        echo "(a lower PPL never fails), top-1 differs $TOP_TOL %, mean KLD $KLD_TOL, 99 % KLD $TAIL_TOL. 99.9 % and max KLD are shown, not gated."
        echo "Also fails if a run did not offload every layer to the GPU."
        echo
        echo "| model | PPL upstream | PPL fork | PPL ratio - 1 | top-1 differs % | mean KLD | 99 % KLD | 99.9 % KLD | max KLD | result |"
        echo "|---|---|---|---|---|---|---|---|---|---|"
        printf '%s\n' "${rows[@]}"
        if [[ ${#diags[@]} -gt 0 ]]; then
            echo
            echo "Failed models, fork rerun with its own optimizations off (\`$STRICT_ENV\`). Not a gate: if the failure"
            echo "goes away, it comes from one of the switched-off code paths."
            echo
            echo "| model | top-1 differs % | mean KLD | 99 % KLD | 99.9 % KLD | max KLD | reading |"
            echo "|---|---|---|---|---|---|---|"
            printf '%s\n' "${diags[@]}"
        fi
    } > "$SUMMARY"
}

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
# <log>: "PPL(Q) same-top-p mean-KLD p99-KLD p99.9-KLD max-KLD" of a --kl-divergence run, '-' for a missing value
stats() {
    local v out=""
    for p in 'Mean PPL(Q)  ' 'Same top p:' 'Mean    KLD' '99.0%   KLD' '99.9%   KLD' 'Maximum KLD'; do
        v=$(val "$1" "$p"); out="$out ${v:--}"
    done
    echo "$out"
}

for m in "${models[@]}"; do
    # split models: only pass the first shard, llama.cpp finds the rest next to it
    [[ $m =~ -[0-9]{5}-of-[0-9]{5}\.gguf$ && ! $m =~ -00001-of-[0-9]{5}\.gguf$ ]] && continue
    name=$(basename "$m" .gguf)
    name=${name%-00001-of-*}
    echo "== $name"
    base=$CACHE/kld-$name.bin
    up_log=$OUT/$name.upstream.log
    fk_log=$OUT/$name.fork.log
    st_log=$OUT/$name.strict.log
    rm -f "$base"

    if ! "$UP_BIN" -m "$m" "${PPL_ARGS[@]}" --kl-divergence-base "$base" > "$up_log" 2>&1; then
        echo "upstream failed on $name"; tail -20 "$up_log"
        rows+=("| $name | error | | | | | | | | FAIL (upstream run) |"); fail=1; rm -f "$base"; write_summary; continue
    fi
    [[ $GPU == unknown ]] && GPU=$(grep -m1 -oE 'ROCm0 \([^)]*\)' "$up_log" || true)
    [[ -n $GPU ]] || GPU=unknown

    # floor runs: "ppl top kld p99 p999 max" per run that worked, ';'-separated
    floors=""; logs=("$up_log")
    for i in "${!FLOOR_SETS[@]}"; do
        read -r -a fa <<< "${FLOOR_SETS[$i]}"
        fl_log=$OUT/$name.floor$((i + 1)).log
        if "$UP_BIN" -m "$m" "${PPL_ARGS[@]}" "${fa[@]}" --kl-divergence-base "$base" --kl-divergence > "$fl_log" 2>&1; then
            s=$(stats "$fl_log")
            echo "floor ${FLOOR_SETS[$i]}: PPL top KLD p99 p99.9 max =$s"
            floors="$floors;$s"; logs+=("$fl_log")
        else
            echo "floor run '${FLOOR_SETS[$i]}' failed on $name, left out"; tail -20 "$fl_log"
        fi
    done

    if ! "$FK_BIN" -m "$m" "${PPL_ARGS[@]}" --kl-divergence-base "$base" --kl-divergence > "$fk_log" 2>&1; then
        echo "fork failed on $name"; tail -20 "$fk_log"
        rows+=("| $name | | error | | | | | | | FAIL (fork run) |"); fail=1; rm -f "$base"; write_summary; continue
    fi
    logs+=("$fk_log")

    ppl_b=$(grep -m1 'Final estimate: PPL' "$up_log" | sed -E 's/.*PPL = *//; s/ *\+.*//; s/ //g' || true)
    read -r ppl_q top kld tail p999 kmax <<< "$(stats "$fk_log")"

    off=""
    for l in "${logs[@]}"; do
        o=$(offload "$l")
        [[ $o == ok ]] || off="$off $(basename "$l" .log | sed 's/.*\.//'): $o;"
    done

    # <ppl_q> <top> <kld> <tail>: "cells... reasons", cells are "value (floor / limit)" with '_' for spaces
    gate() {
        LC_ALL=C awk -v b="$ppl_b" -v q="$1" -v top="$2" -v k="$3" -v p="$4" -v fl="$floors" -v f="$FLOOR_K" \
            -v t="$PPL_TOL" -v tt="$TOP_TOL" -v kt="$KLD_TOL" -v pt="$TAIL_TOL" '
        function mx(a, c) { return a > c ? a : c }
        function cell(v, fv, lim, fmt) {
            return sprintf(fmt, v) "_(" (fv == "" ? "n/a" : sprintf(fmt, fv)) "_/_" sprintf(fmt, lim) ")" }
        BEGIN {
            if (b == "" || q == "-" || top == "-" || k == "-" || p == "-") { print "- - - - parse_error"; exit }
            # worst floor values over the floor runs that parsed
            n = split(fl, runs, ";"); rf = df = kf = pf = ""
            for (i = 1; i <= n; i++) {
                if (split(runs[i], s, " ") < 4 || s[1] == "-" || s[2] == "-" || s[3] == "-" || s[4] == "-") continue
                d = s[1] / b - 1; if (d < 0) d = -d
                rf = rf == "" ? d : mx(rf, d)
                df = df == "" ? 100 - s[2] : mx(df, 100 - s[2])
                kf = kf == "" ? s[3] : mx(kf, s[3])
                pf = pf == "" ? s[4] : mx(pf, s[4])
            }
            rl = mx(t, f * rf); dl = mx(tt, f * df); kl = mx(kt, f * kf); pl = mx(pt, f * pf)
            r = q / b - 1; d = 100 - top
            w = ""
            if (r > rl) w = w ",PPL"
            if (d > dl) w = w ",top-1"
            if (k > kl) w = w ",KLD"
            if (p > pl) w = w ",KLD_tail"
            if (w == "") w = ",-"
            printf "%s %s %s %s %s\n", cell(r, rf, rl, "%+.4f"), cell(d, df, dl, "%.2f"),
                cell(k, kf, kl, "%.6f"), cell(p, pf, pl, "%.4f"), substr(w, 2) }'
    }

    read -r c_ppl c_top c_kld c_tail why <<< "$(gate "$ppl_q" "$top" "$kld" "$tail")"
    [[ $why == - ]] && why=""
    gated=$why
    [[ -n $off ]] && why="${why:+$why,}offload"
    if [[ -z $why ]]; then
        res=OK
    else
        why=${why//_/ }
        res="FAIL (${why//,/, })"; fail=1
    fi
    [[ -n $off ]] && echo "offload:$off"
    sp() { echo "${1//_/ }"; }
    echo "PPL upstream $ppl_b fork $ppl_q | ratio-1 $(sp "$c_ppl") | top-1 differs % $(sp "$c_top") | KLD $(sp "$c_kld") | p99 $(sp "$c_tail") | p99.9 $p999 | max $kmax -> $res"
    rows+=("| $name | $ppl_b | $ppl_q | $(sp "$c_ppl") | $(sp "$c_top") | $(sp "$c_kld") | $(sp "$c_tail") | $p999 | $kmax | $res |")

    # failed on the numbers (not on offload or parsing): rerun the fork with its own code paths off
    if [[ -n $gated && $gated != parse_error && ${#STRICT[@]} -gt 0 ]]; then
        if env "${STRICT[@]}" "$FK_BIN" -m "$m" "${PPL_ARGS[@]}" --kl-divergence-base "$base" --kl-divergence > "$st_log" 2>&1; then
            read -r s_ppl s_top s_kld s_tail s_p999 s_max <<< "$(stats "$st_log")"
            read -r _ _ _ _ s_why <<< "$(gate "$s_ppl" "$s_top" "$s_kld" "$s_tail")"
            case $s_why in
                -)           reading="passes: the failure comes from a switched-off fork path" ;;
                parse_error) reading="could not be parsed" ;;
                *)           reading="still fails (${s_why//,/, }): not (only) the switched-off paths"; reading=${reading//_/ } ;;
            esac
            s_d=$(LC_ALL=C awk -v t="$s_top" 'BEGIN { if (t == "-") print "-"; else printf "%.2f", 100 - t }')
            echo "strict: top-1 differs $s_d %, KLD $s_kld, p99 $s_tail, p99.9 $s_p999, max $s_max -> $reading"
            diags+=("| $name | $s_d | $s_kld | $s_tail | $s_p999 | $s_max | $reading |")
        else
            echo "strict run failed on $name"; tail -20 "$st_log"
            diags+=("| $name | | | | | | strict run failed |")
        fi
    fi
    rm -f "$base"
    write_summary
done

write_summary
echo
cat "$SUMMARY"
exit $fail
