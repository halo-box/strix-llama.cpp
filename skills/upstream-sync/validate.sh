#!/usr/bin/env bash
# Validation phase of an upstream sync PR, run by hand on a Strix Halo (gfx1151) machine. See SKILL.md, step 7.
# Builds the fork before the sync (base) and the sync branch (cand) in worktrees, runs every check on each GPU
# backend, and writes summary.md with one status per backend: TARGET PASS, FAIL or INCOMPLETE.
#
#   validate.sh --pr <number> [--cand <rev>] [--base <rev>] [--backends hip,vk] [--out <dir>]
#   validate.sh --cand <rev> [--model <arch>=<gguf>[:short|long]]... [--small <gguf>] ...
#
#   --pr       read the sync PR: its head is the candidate, and its "## Models to validate" table gives the models
#              (Hugging Face repo and file, downloaded to MODELS_DIR) and their context profile, short or long
#   --model    a local model instead of the PR table; repeat the option
#   --small    Qwen3.5-0.8B-Base in F16 or BF16, quantized here to every type of QUANTS (the idea comes from
#              upstream ci/run.sh). Default: downloaded from ggml-org/Qwen3.5-0.8B-Base-GGUF
#   --base     default: the merge base of the candidate with the branch the PR targets (origin/master without --pr)
#   --backends hip, vk, or cpu (cpu only to test this script)
set -euo pipefail
export LC_ALL=C
cd "$(git rev-parse --show-toplevel)"
HERE=$(cd "$(dirname "$0")" && pwd)

QUANTS=${QUANTS:-"BF16 F16 Q8_0 Q6_K Q5_K_M Q5_1 Q5_0 Q4_K_M Q4_1 Q4_0 Q3_K_M Q2_K IQ4_NL IQ4_XS"}
MODELS_DIR=${MODELS_DIR:-$HOME/.cache/strix-llama-sync/models}
# bench: every pp size at every depth of the context profile, -b 4096 -ub 4096
PP_SIZES=${PP_SIZES:-512,1024,2048,4096}
DEPTHS_SHORT=${DEPTHS_SHORT:-0,4096,16384}
DEPTHS_LONG=${DEPTHS_LONG:-0,12000,32000,64000}
CAND="" BASE="" SMALL="" BACKENDS="hip,vk" OUT="" PR=""
MODELS=()
while [ $# -gt 0 ]; do
    case $1 in
        --pr) PR=$2; shift 2 ;;
        --cand) CAND=$2; shift 2 ;;
        --base) BASE=$2; shift 2 ;;
        --small) SMALL=$2; shift 2 ;;
        --model) MODELS+=("$2"); shift 2 ;;
        --backends) BACKENDS=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

die() { echo "error: $*" >&2; exit 1; }

# download <repo> <file> (and every shard of a split gguf) to MODELS_DIR, print the local path of <file>
fetch() {
    local d=$MODELS_DIR/${1//\//_} f=$2 files=$2 n i x
    mkdir -p "$d"
    if [[ $f =~ -00001-of-([0-9]{5})\.gguf$ ]]; then
        n=$((10#${BASH_REMATCH[1]})) files=""
        for i in $(seq -f %05g 1 $n); do files="$files ${f/-00001-of-/-$i-of-}"; done
    fi
    for x in $files; do
        [ -s "$d/$x" ] && continue
        echo "download $1/$x" >&2
        curl -fL --retry 3 ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} -o "$d/$x.part" "https://huggingface.co/$1/resolve/main/$x" >&2 && mv "$d/$x.part" "$d/$x"
    done
    echo "$d/$f"
}

PR_BODY=$(mktemp)
trap 'rm -f "$PR_BODY"' EXIT
section() { awk -v h="## $1" '$0 == h {f=1; next} /^## /{f=0} f' "$PR_BODY"; }

# 0. read the PR first: what the sync changed, and whether the PR says enough to validate it
if [ -n "$PR" ]; then
    gh pr view "$PR" --repo halo-box/strix-llama.cpp --json title,url,headRefName,baseRefName,body > "$PR_BODY.json" || die "cannot read PR $PR"
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["body"])' "$PR_BODY.json" | tr -d '\r' > "$PR_BODY"
    PR_URL=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["url"])' "$PR_BODY.json")
    head=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["headRefName"])' "$PR_BODY.json")
    PR_BASE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["baseRefName"])' "$PR_BODY.json")
    rm -f "$PR_BODY.json"
    for h in "Overview" "Steps" "Models to validate"; do
        [ -n "$(section "$h")" ] || die "PR $PR has no '## $h' section: generate the body with 'sync.sh summary' and fill the models table (SKILL.md, step 6)"
    done
    echo "== $PR_URL"
    section Overview | grep -E '^\|' || true
    if grep -q -F '**edit outside the conflict zones**' "$PR_BODY" || section Overview | grep -E '^\| [0-9]+ ' | grep -qvE '\| none \|$'; then
        echo "note: this sync has edits outside the conflicts, read them in the PR before trusting the numbers"
    fi
    [ -n "$CAND" ] || { git fetch -q origin "$head" && CAND=FETCH_HEAD; }
    # rows of the "## Models to validate" table: | arch | repo | file | short or long |
    while IFS='|' read -r _ a r f c _; do
        a=$(echo $a) r=$(echo $r) f=$(echo $f) c=$(echo $c)
        case $a in ""|Arch|---*|"<"*) continue ;; esac
        MODELS+=("$a=$(fetch "$r" "$f"):${c:-short}")
    done < <(section "Models to validate" | grep -E '^\|')
    if [ "${#MODELS[@]}" -eq 0 ]; then
        if section Overview | grep -q '`\*'; then
            die "the PR has sensitive steps but no model to validate: fill the 'Models to validate' table"
        fi
        echo "warning: no model in the PR table, only the small model and the op tests run" >&2
    fi
fi
[ -n "$CAND" ] || die "--pr <number> or --cand <rev> is required"
[ -n "$SMALL" ] || SMALL=$(fetch ggml-org/Qwen3.5-0.8B-Base-GGUF Qwen3.5-0.8B-Base-BF16.gguf)
[ -f "$SMALL" ] || die "small model not found: $SMALL"
SMALL_TYPE=$(basename "$SMALL" | grep -oE 'BF16|F16' | head -1)
[ -n "$SMALL_TYPE" ] || die "--small must be an F16 or BF16 gguf (the type must be in the file name)"
CAND=$(git rev-parse --verify "$CAND^{commit}")
# the fork before the sync: where the candidate leaves the branch the PR targets (master, or a stacked PR branch)
target=origin/master
if [ -n "${PR_BASE:-}" ]; then git fetch -q origin "$PR_BASE"; target=FETCH_HEAD; fi
BASE=$(git rev-parse --verify "${BASE:-$(git merge-base "$CAND" "$target")}^{commit}")
OUT=${OUT:-../sync-validate/$(git rev-parse --short=9 "$CAND")}
mkdir -p "$OUT/logs"
OUT=$(cd "$OUT" && pwd)
SMALL=$(realpath "$SMALL")
[ -z "$PR" ] || cp "$PR_BODY" "$OUT/pr-body.md"

# match the command line: pgrep -x only sees the first 15 characters of a name (llama-perplexity has 16)
if busy=$(pgrep -af '(^|/)llama-(server|cli|bench|perplexity|completion)( |$)'); then die "the GPU is busy: $busy"; fi

flags() { case $1 in hip) echo "-DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_HIP_COMPILER=$(hipconfig -l)/clang" ;; vk) echo "-DGGML_VULKAN=ON" ;; cpu) echo "" ;; esac; }
dev()   { case $1 in hip) echo ROCm0 ;; vk) echo Vulkan0 ;; cpu) echo CPU ;; esac; }
ngl()   { if [ "$1" = cpu ]; then echo 0; else echo 99; fi; }
bin()   { echo "$OUT/build-$1-$2/bin"; }
# GPU memory in use on the amdgpu devices, VRAM + GTT in bytes (GTT matters on an APU); 0 without the counters
gpu_used() { { cat /sys/class/drm/card*/device/mem_info_vram_used /sys/class/drm/card*/device/mem_info_gtt_used 2> /dev/null || true; } | awk '{s+=$1} END {print s+0}'; }
# sample a process until it exits: peak RSS of it and its children, peak GPU memory above <gpu bytes before the start>
peak() {  # <pid> <gpu before>
    local rp=0 gp=$2 r g
    while kill -0 "$1" 2> /dev/null; do
        # the process can exit between two samples: ps then fails, which must not stop the script
        r=$({ ps -o rss= --ppid "$1" -p "$1" 2> /dev/null || true; } | awk '{s+=$1} END {print s+0}')
        g=$(gpu_used)
        [ "$r" -gt "$rp" ] && rp=$r
        [ "$g" -gt "$gp" ] && gp=$g
        sleep 0.5
    done
    echo "PEAK_RSS_MIB=$((rp / 1024))"
    [ "$2" -eq 0 ] || echo "PEAK_GPU_MIB=$(((gp - $2) / 1048576))"
}
# run a command, keep its output and its memory peaks, never stop the whole validation on one failure
run() {
    local log=$1 g0 pid rc=0
    shift
    g0=$(gpu_used)
    "$@" > "$log" 2>&1 &
    pid=$!
    peak $pid "$g0" > "$log.mem"
    wait $pid || rc=$?
    [ $rc = 0 ] || echo "EXIT=$rc" >> "$log"
    cat "$log.mem" >> "$log"
    rm -f "$log.mem"
}

echo "base $(git log -1 --format='%h %s' "$BASE")"
echo "cand $(git log -1 --format='%h %s' "$CAND")"
echo "out  $OUT"
{
    echo "pr=${PR_URL:-}"
    echo "base=$BASE"
    echo "cand=$CAND"
    echo "small=$(basename "$SMALL") $(sha256sum "$SMALL" | cut -c1-16)"
    for mm in "${MODELS[@]}"; do x=${mm#*=}; echo "model=${mm%%=*} $(basename "${x%:*}") ${x##*:}"; done
    echo "backends=$BACKENDS"
} > "$OUT/manifest.txt"

# 1. sources and builds
for side in base cand; do
    rev=$BASE; [ $side = cand ] && rev=$CAND
    [ -d "$OUT/src-$side" ] || git worktree add -q --detach "$OUT/src-$side" "$rev"
    git -C "$OUT/src-$side" checkout -q --detach "$rev"
done
TARGETS="llama-perplexity llama-bench llama-server llama-quantize test-backend-ops test-llama-archs test-save-load-state"
for be in ${BACKENDS//,/ }; do
    for side in base cand; do
        echo "build $side $be"
        { cmake -S "$OUT/src-$side" -B "$OUT/build-$side-$be" $(flags $be) -DCMAKE_BUILD_TYPE=Release && \
          cmake --build "$OUT/build-$side-$be" -j --target $TARGETS; } > "$OUT/logs/build-$side-$be.log" 2>&1 \
          && echo ok > "$OUT/logs/build-$side-$be.status" || echo failed > "$OUT/logs/build-$side-$be.status"
    done
done

C=$OUT/src-cand/tests/corpus
cat "$C/correctness-prose.txt" "$C/correctness-code.txt" "$C/correctness-structured.txt" "$C/correctness-numeric.txt" > "$OUT/all.txt"
sha256sum "$C"/correctness-*.txt "$OUT/all.txt" > "$OUT/corpus.sha256"

# logits of base and cand on the same text, plus the reference: base against itself with a smaller ubatch
kld() {  # <be> <label> <model> <text> <args> <reference args>
    local be=$1 d=$OUT/$1/kld/$2 k=$OUT/$1/kld/$2.kld
    local o="-m $3 -ngl $(ngl $1) -fa on -ctk f16 -ctv f16"
    run "$d-base.txt" "$(bin base $be)/llama-perplexity" $o $5 -f "$4" --kl-divergence-base "$k"
    run "$d-cand.txt" "$(bin cand $be)/llama-perplexity" $o $5 --kl-divergence-base "$k" --kl-divergence
    run "$d-ref.txt"  "$(bin base $be)/llama-perplexity" $o $6 --kl-divergence-base "$k" --kl-divergence
    rm -f "$k"   # about 0.5 MB per scored token with a 248k vocab
}

bench() {  # <be> <label> <model> <args>: alternating order, two rounds per side, one log per round
    local t i=0
    for t in base cand base cand; do
        i=$((i + 1))
        run "$OUT/$1/bench/$2-$t-$i.log" "$(bin $t $1)/llama-bench" -m "$3" $4 -o jsonl
    done
}

server() {  # <be> <arch> <model>: tool call, greedy decode, MTP drafting
    local be=$1 a=$2 m=$3 t s ok spec g0 sp
    for t in base cand; do
        for spec in "--spec-type draft-mtp" ""; do
            g0=$(gpu_used)
            "$(bin $t $be)/llama-server" -m "$m" -ngl $(ngl $be) -fa on --jinja --port 8099 $spec > "$OUT/$be/server/$a-$t.log" 2>&1 &
            s=$! ok=0
            peak $s "$g0" > "$OUT/$be/server/$a-$t.mem" &
            sp=$!
            for _ in $(seq 300); do
                if curl -sf localhost:8099/health > /dev/null; then ok=1; break; fi
                kill -0 $s 2> /dev/null || break
                sleep 2
            done
            if [ $ok = 1 ]; then
                echo "${spec:-no-mtp}" > "$OUT/$be/server/$a-$t.spec"
                curl -s localhost:8099/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"What is the weather in Paris?"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get the weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"temperature":0}' > "$OUT/$be/server/$a-tool-$t.json" || true
                curl -s localhost:8099/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Write a Python function that parses ISO 8601 dates, with tests."}],"max_tokens":256,"temperature":0}' > "$OUT/$be/server/$a-decode-$t.json" || true
            fi
            kill $s 2> /dev/null || true
            wait $s 2> /dev/null || true
            wait $sp 2> /dev/null || true
            cat "$OUT/$be/server/$a-$t.mem" >> "$OUT/$be/server/$a-$t.log"
            [ $ok = 1 ] && break
        done
    done
}

for be in ${BACKENDS//,/ }; do
    [ "$(cat "$OUT/logs/build-base-$be.status")" = ok ] && [ "$(cat "$OUT/logs/build-cand-$be.status")" = ok ] || { echo "skip $be: build failed"; continue; }
    mkdir -p "$OUT/$be/kld" "$OUT/$be/bench" "$OUT/$be/server" "$OUT/$be/quant"

    # 2. every op and every arch, GPU against CPU
    echo "$be: test-backend-ops, test-llama-archs"
    for t in base cand; do
        run "$OUT/$be/ops-$t.txt" "$(bin $t $be)/test-backend-ops" -b "$(dev $be)"
        run "$OUT/$be/archs-$t.txt" "$(bin $t $be)/test-llama-archs" -b "$(dev $be)"
        sed -i 's/\x1b\[[0-9;]*m//g' "$OUT/$be/archs-$t.txt"
    done

    # 3. the small model in every quant type: logits, speed, and whether the quantizer output changed
    echo "$be: Qwen3.5-0.8B quant matrix"
    for q in $QUANTS; do
        f=$OUT/quant/small-$q.gguf
        mkdir -p "$OUT/quant"
        if [ $q = "$SMALL_TYPE" ]; then
            f=$SMALL
        elif [ ! -f "$f" ]; then
            run "$OUT/quant/$q-base.log" "$(bin base $be)/llama-quantize" "$SMALL" "$f" $q
            run "$OUT/quant/$q-cand.log" "$(bin cand $be)/llama-quantize" "$SMALL" "$OUT/quant/cand-$q.gguf" $q
            echo "$(sha256sum < "$f" | cut -c1-16) $(sha256sum < "$OUT/quant/cand-$q.gguf" | cut -c1-16)" > "$OUT/quant/$q.sha"
            rm -f "$OUT/quant/cand-$q.gguf"
        fi
        kld "$be" "small-$q" "$f" "$OUT/all.txt" "-c 512 -b 512 -ub 512 --chunks 8" "-c 512 -b 512 -ub 128 --chunks 8"
        bench "$be" "small-$q" "$f" "-p 512 -n 128 -r 3 -ngl $(ngl $be) -fa on"
    done
    run "$OUT/$be/sls-base.txt" "$(bin base $be)/test-save-load-state" --model "$OUT/quant/small-Q4_0.gguf" -ngl $(ngl $be) -c 1024 -fa on
    run "$OUT/$be/sls-cand.txt" "$(bin cand $be)/test-save-load-state" --model "$OUT/quant/small-Q4_0.gguf" -ngl $(ngl $be) -c 1024 -fa on

    # 4. the real models of the archs this sync touches
    for mm in "${MODELS[@]}"; do
        a=${mm%%=*} m=${mm#*=} ctx=short
        case $m in *:short|*:long) ctx=${m##*:} m=${m%:*} ;; esac
        depths=$DEPTHS_SHORT; [ $ctx = long ] && depths=$DEPTHS_LONG
        echo "$be: $a ($ctx context)"
        for n in prose code structured numeric; do
            kld "$be" "$a-$n" "$m" "$C/correctness-$n.txt" "-c 2048 -b 2048 -ub 2048" "-c 2048 -b 2048 -ub 512"
        done
        if [ $ctx = long ]; then
            kld "$be" "$a-long" "$m" "$OUT/all.txt" "-c 16384 -b 512 -ub 512" "-c 16384 -b 256 -ub 256"
        fi
        bench "$be" "$a-pp" "$m" "-p $PP_SIZES -d $depths -b 4096 -ub 4096 -n 0 -r 4 -ngl $(ngl $be) -fa on -ctk f16 -ctv f16 --load-mode none"
        bench "$be" "$a-tg" "$m" "-p 0 -n 128 -d $depths -b 4096 -ub 4096 -r 4 -ngl $(ngl $be) -fa on -ctk f16 -ctv f16 --load-mode none"
        server "$be" "$a" "$m"
    done
done

python3 "$HERE/validate_summary.py" "$OUT" > "$OUT/summary.md"
echo
cat "$OUT/summary.md"
