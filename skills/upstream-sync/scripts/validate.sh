#!/usr/bin/env bash
# Validation of an upstream sync, run on a Strix Halo (gfx1151) machine: by the upstream-sync skill before the PR (--body),
# and by the upstream-sync-verify skill on an open PR (--pr --post comment).
# Builds the fork before the sync (base) and the sync branch (cand) in worktrees, runs every check on each GPU
# backend, and writes summary.md with one status per backend: TARGET PASS, FAIL or INCOMPLETE.
#
#   validate.sh --pr <number> [--cand <rev>] [--base <rev>] [--backends hip,vk] [--out <dir>]
#   validate.sh --cand <rev> [--model <arch>=<gguf>]... [--small <gguf>] ...
#
#   --body     read a PR body from a file instead, before the PR exists (with --cand); same checks
#   --post     with --pr: "body" (default) writes the results into the PR body, "comment" posts them as a comment
#   --pr       read the sync PR: its head is the candidate, and its "## Models to validate" table gives the models
#              (Hugging Face repo and file, downloaded to MODELS_DIR; link existing files there to skip the download)
#   --model    a local model instead of the PR table; repeat the option
#   --small    Qwen3.5-0.8B-Base in F16 or BF16, quantized here to every type of QUANTS (the idea comes from
#              upstream ci/run.sh). Default: downloaded from ggml-org/Qwen3.5-0.8B-Base-GGUF
#   --small-moe a small MoE in F16 or BF16, quantized to every type of MOE_QUANTS (MXFP4_MOE only changes expert
#              tensors, which a dense model has none of). Default: Granite 3.1 1B-A400M from bartowski/granite-3.1-1b-a400m-instruct-GGUF
#   --base     default: the merge base of the candidate with the branch the PR targets (origin/master without --pr)
#   --backends hip, vk, or cpu (cpu only to test this script)
#
#   --out      the output directory, default ../sync-validate/<candidate sha>
#
# Environment: ROCM_PATH (default /opt/rocm), MODELS_DIR (default ~/.cache/strix-llama-sync/models), QUANTS, MOE_QUANTS, KV_TYPES,
# PP_SIZES (default 512,2048), DEPTHS (default 0,32000,128000), RUN_TIMEOUT (default 21600 s:
# a single test or benchmark run that takes longer is killed and counts as a crash, so a GPU hang ends the run), OPS_JOBS
# (default 4: test-backend-ops parallel workers per process).
# Downloads are anonymous: no Hugging Face token is ever used. For a gated model, download it yourself into MODELS_DIR.
#
# HIP builds use the ROCm of ROCM_PATH (default /opt/rocm, ROCm 10), not the first hipconfig in PATH.
# Vulkan builds need glslc and the SPIR-V headers: the headers are downloaded once (a pinned release, checked by sha256)
# to DEPS_DIR (default ~/.cache/strix-llama-sync/deps), so no distribution package is needed.
# The baseline build, its test-backend-ops / test-llama-archs results, its importance matrix of the small model and its
# KLD reference runs are kept in BUILD_CACHE (default base-builds next to the output directory) and reused for the same
# commit, configuration, compilers and GPU driver.
# With --pr, the results are written to the PR body after each backend and at the end.
#
# Security: this builds and runs the code of the candidate (with --pr, the PR head, possibly from a contributor's fork)
# with your environment, the gh login included. Only run it on a reviewed PR you trust, or in a container.
# The candidate can also write wherever you can, BUILD_CACHE included: the kept baseline builds are made read-only,
# but after a run on code you do not trust, delete BUILD_CACHE (and the output directory).
# One run at a time per MODELS_DIR (lock file MODELS_DIR/.validate.lock).
set -euo pipefail
export LC_ALL=C
# never a Hugging Face token: the code under test runs with this environment
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
HERE=$(cd "$(dirname "$0")" && pwd)
START=$(date +%s)
cd "$(git rev-parse --show-toplevel)"

# one quant type per ggml tensor type that llama-quantize makes, checked with llama-quantize --dry-run on this model:
# the other types only mix these per tensor (K_S / K_L, IQ3_M / IQ3_XS), and IQ2_S makes iq2_xs tensors, so iq2_s
# comes from IQ2_M. Not the ternary TQ (BitNet models only), MXFP4_MOE (the small MoE below), the fork's ROCmFP4 and
# ROCmFPx types, nor NVFP4 (an NVIDIA format, no llama-quantize type). QUANTS="..." for others.
QUANTS=${QUANTS:-"BF16 F16 Q8_0 Q6_K Q5_K_M Q5_1 Q5_0 Q4_K_M Q4_1 Q4_0 Q3_K_M Q2_K Q2_0 Q1_0
 IQ4_NL IQ4_XS IQ3_S IQ3_XXS IQ2_M IQ2_XS IQ2_XXS IQ1_M IQ1_S"}
QUANTS=$(echo $QUANTS)   # one line: the manifest holds one value per line
# empty means the default (systemd-run -E passes an unset variable as empty): "none" turns a matrix off
MOE_QUANTS=${MOE_QUANTS:-MXFP4_MOE}   # the other types are tested on the small Qwen
# the quantized KV cache types in real use: the flash attention paths that read a quantized K and V (the others:
# q4_1 iq4_nl q5_0 q5_1 bf16, with KV_TYPES)
KV_TYPES=${KV_TYPES:-"q8_0 q4_0"}
[ "$MOE_QUANTS" != none ] || MOE_QUANTS=""
[ "$KV_TYPES" != none ] || KV_TYPES=""
MODELS_DIR=${MODELS_DIR:-$HOME/.cache/strix-llama-sync/models}
DEPS_DIR=${DEPS_DIR:-$HOME/.cache/strix-llama-sync/deps}
BUILD_CACHE=${BUILD_CACHE:-}   # default: base-builds next to the output directory
BUILD_CACHE_KEEP=${BUILD_CACHE_KEEP:-6}   # baseline builds kept, the most recently used first
# SPIR-V headers for the Vulkan build (upstream ggml-vulkan: find_package(SPIRV-Headers CONFIG REQUIRED))
SPIRV_TAG=vulkan-sdk-1.4.341.0
SPIRV_SHA256=cab0a654c4917e16367483296b44cdb1d614e3120c721beafcd37e3a8580486c
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
# bench: every pp size at every depth, -b 4096 -ub 4096
PP_SIZES=${PP_SIZES:-512,2048}
DEPTHS=${DEPTHS:-0,32000,128000}
RUN_TIMEOUT=${RUN_TIMEOUT:-21600}
OPS_JOBS=${OPS_JOBS:-4}   # test-backend-ops -j: parallel workers per process
CAND="" BASE="" SMALL="" SMALL_MOE="" BACKENDS="hip,vk" OUT="" PR="" BODY="" POST=body
MODELS=()
while [ $# -gt 0 ]; do
    case $1 in
        --pr) PR=$2; shift 2 ;;
        --body) BODY=$2; shift 2 ;;
        --post) POST=$2; shift 2 ;;
        --cand) CAND=$2; shift 2 ;;
        --base) BASE=$2; shift 2 ;;
        --small) SMALL=$2; shift 2 ;;
        --small-moe) SMALL_MOE=$2; shift 2 ;;
        --model) MODELS+=("$2"); shift 2 ;;
        --backends) BACKENDS=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        -h|--help) sed -n '2,/^set -euo/p' "$HERE/$(basename "$0")" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

die() { echo "error: $*" >&2; exit 1; }
case $POST in body|comment) ;; *) die "--post must be body or comment" ;; esac
for be in ${BACKENDS//,/ }; do case $be in hip|vk|cpu) ;; *) die "unknown backend: $be (hip, vk or cpu)" ;; esac; done
[ -n "${BACKENDS//,/}" ] || die "--backends is empty"

# one validation at a time: two runs would share the GPU, the models and the numbers
mkdir -p "$MODELS_DIR"
exec 9> "$MODELS_DIR/.validate.lock"
flock -n 9 || die "another validate.sh is running (lock $MODELS_DIR/.validate.lock)"

# download <repo> <file> (and every shard of a split gguf) to MODELS_DIR, print the local path of <file>
# repo and file come from a PR body: only plain names, and the result must stay under MODELS_DIR
fetch() {
    local d=$MODELS_DIR/${1//\//_} f=$2 files=$2 n i x
    [[ $1 =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] && [[ $1 != *..* ]] || die "bad Hugging Face repo: $1"
    [[ $f =~ ^[A-Za-z0-9._/-]+\.gguf$ ]] && [[ $f != /* ]] && [[ $f != *..* ]] || die "bad file name: $f"
    case $(realpath -m "$d/$f") in "$(realpath -m "$MODELS_DIR")"/*) ;; *) die "file outside MODELS_DIR: $1/$f" ;; esac
    mkdir -p "$d/$(dirname "$f")"
    if [[ $f =~ -00001-of-([0-9]{5})\.gguf$ ]]; then
        n=$((10#${BASH_REMATCH[1]})) files=""
        for i in $(seq -f %05g 1 $n); do files="$files ${f/-00001-of-/-$i-of-}"; done
    fi
    for x in $files; do
        [ -s "$d/$x" ] && continue
        echo "download $1/$x" >&2
        code=$(curl -sSL --retry 3 -w '%{http_code}' -o "$d/$x.part" "https://huggingface.co/$1/resolve/main/$x") || code=000
        case $code in
            200) mv "$d/$x.part" "$d/$x" ;;
            401|403) rm -f "$d/$x.part"; die "$1/$x is gated or private (HTTP $code): download it yourself to $d/$x" ;;
            *) rm -f "$d/$x.part"; die "download failed: $1/$x (HTTP $code)" ;;
        esac
    done
    for x in $files; do [ -s "$d/$x" ] || die "download failed: $1/$x"; done
    echo "$d/$f"
}

PR_BODY=$(mktemp)
# a stopped run must not leave a server, a sampler or a test running on the GPU
# a background run is a subshell with timeout and the test below it, and timeout gives the test its own process group:
# killing the job alone would leave the test on the GPU, so kill each tree from the leaves up
kill_tree() { local c; for c in $(pgrep -P "$1" 2> /dev/null || true); do kill_tree "$c"; done; kill "$1" 2> /dev/null || true; }
cleanup() {
    local j
    for j in $(jobs -p); do kill_tree "$j"; done
    rm -f "$PR_BODY"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
section() { awk -v h="## $1" '$0 == h {f=1; next} /^## /{f=0} f' "$PR_BODY"; }

# 0. read the PR (or the body file) first: what the sync changed, and whether it says enough to validate it
if [ -n "$PR" ]; then
    gh pr view "$PR" --repo halo-box/strix-llama.cpp --json title,url,headRefName,baseRefName,body > "$PR_BODY.json" || die "cannot read PR $PR"
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["body"])' "$PR_BODY.json" | tr -d '\r' > "$PR_BODY"
    PR_URL=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["url"])' "$PR_BODY.json")
    head=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["headRefName"])' "$PR_BODY.json")
    PR_BASE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["baseRefName"])' "$PR_BODY.json")
    rm -f "$PR_BODY.json"
elif [ -n "$BODY" ]; then
    [ -n "$CAND" ] || die "--body needs --cand <rev>"
    tr -d '\r' < "$BODY" > "$PR_BODY"
fi
if [ -n "$PR" ] || [ -n "$BODY" ]; then
    for h in "Overview" "Steps" "Where upstream changes fork code" "Models concerned" "Models to validate"; do
        [ -n "$(section "$h")" ] || die "the PR body has no '## $h' section: generate it with 'sync.sh summary' and fill the models tables (skill upstream-sync)"
    done
    # the placeholder sync.sh writes in every row: a table pasted as generated says nothing about how each arch is validated
    if section "Models concerned" | grep -F '<real GGUF' > /dev/null; then
        die "the 'Models concerned' table still has the placeholder of sync.sh: fill the Validation column of every row"
    fi
    echo "== ${PR_URL:-$BODY}"
    section Overview | grep -E '^\|' || true
    if grep -F '**edit outside the conflict zones**' "$PR_BODY" > /dev/null || section Overview | grep -E '^\| [0-9]+ ' | grep -vE '\| none \|$' > /dev/null; then
        echo "note: this sync has edits outside the conflicts, read them in the PR before trusting the numbers"
    fi
    # pull/<n>/head works for a PR from a contributor's fork too, where the branch is not on origin
    [ -n "$CAND" ] || [ -z "$PR" ] || { git fetch -q origin "pull/$PR/head" && CAND=FETCH_HEAD; } || die "cannot fetch the head of PR $PR"
    # rows of the "## Models to validate" table: | arch | Hugging Face repo | file |
    trim() { local x=${1//\`/}; x=${x#"${x%%[![:space:]]*}"}; echo "${x%"${x##*[![:space:]]}"}"; }
    while IFS='|' read -r _ a r f _; do
        a=$(trim "$a") r=$(trim "$r") f=$(trim "$f")
        case $a in Arch|:-*|-*|"<"*) continue ;; esac
        [ -n "$a" ] || continue
        [ -n "$r" ] && [ -n "$f" ] || die "the 'Models to validate' row of $a needs an arch, a Hugging Face repo and a file"
        [[ $a =~ ^[A-Za-z0-9._-]+$ ]] || die "bad arch in 'Models to validate': $a"
        # in its own assignment: a die inside $(...) only leaves the subshell
        p=$(fetch "$r" "$f") || exit 1
        MODELS+=("$a=$p")
    done < <(section "Models to validate" | grep -E '^\|' || true)
    if [ "${#MODELS[@]}" -eq 0 ]; then
        if section Overview | grep '`\*' > /dev/null; then
            die "the PR has sensitive steps but no model to validate: fill the 'Models to validate' table"
        fi
        echo "warning: no model in the PR table, only the small model and the op tests run" >&2
    fi
fi
[ -n "$CAND" ] || die "--pr <number> or --cand <rev> is required"
[ -n "$SMALL" ] || SMALL=$(fetch ggml-org/Qwen3.5-0.8B-Base-GGUF Qwen3.5-0.8B-Base-BF16.gguf)
[ -f "$SMALL" ] || die "small model not found: $SMALL"
# the small MoE only when it has a quant type to test (MOE_QUANTS=none skips it, download included)
if [ -n "${MOE_QUANTS// /}" ]; then
    [ -n "$SMALL_MOE" ] || SMALL_MOE=$(fetch bartowski/granite-3.1-1b-a400m-instruct-GGUF granite-3.1-1b-a400m-instruct-f16.gguf)
    [ -f "$SMALL_MOE" ] || die "small MoE model not found: $SMALL_MOE"
fi
# two models of the same arch get their own labels (qwen35, qwen35-2, ...): every result file is named after the label
declare -A seen=()
uniq=()
for mm in "${MODELS[@]}"; do
    a=${mm%%=*}
    seen[$a]=$(( ${seen[$a]:-0} + 1 ))
    [ "${seen[$a]}" -eq 1 ] || a="$a-${seen[$a]}"
    uniq+=("$a=${mm#*=}")
done
MODELS=("${uniq[@]}")
# the source type of a small model, from its file name (any case): it is its own row, not quantized again
src_type() { basename "$1" | tr '[:lower:]' '[:upper:]' | grep -oE 'BF16|F16' | head -1 || true; }
SMALL_TYPE=$(src_type "$SMALL") MOE_TYPE=
[ -z "$SMALL_MOE" ] || MOE_TYPE=$(src_type "$SMALL_MOE")
[ -n "$SMALL_TYPE" ] || die "--small must be an F16 or BF16 gguf (the type must be in the file name)"
[ -z "$SMALL_MOE" ] || [ -n "$MOE_TYPE" ] || die "--small-moe must be an F16 or BF16 gguf (the type must be in the file name)"
CAND=$(git rev-parse --verify "$CAND^{commit}")
# the fork before the sync: where the candidate leaves the branch the PR targets (master, or a stacked PR branch)
if [ -n "${PR_BASE:-}" ]; then git fetch -q origin "$PR_BASE" || die "cannot fetch $PR_BASE"; target=FETCH_HEAD
else git fetch -q origin master || die "cannot fetch origin master"; target=origin/master; fi
BASE=$(git rev-parse --verify "${BASE:-$(git merge-base "$CAND" "$target")}^{commit}")
OUT=${OUT:-../sync-validate/$(git rev-parse --short=9 "$CAND")}
# the cleanup below deletes subdirectories: only in an output directory of this script (even one of an older version
# or a run stopped early), never in --out ~. Such a directory holds only what this script writes.
if [ -d "$OUT" ]; then
    for e in "$OUT"/* "$OUT"/.[!.]*; do
        [ -e "$e" ] || continue
        case ${e##*/} in
            .validate-out|logs|src-base|src-cand|build-*|hip|vk|cpu|quant|all.txt|corpus.sha256|manifest.txt|summary.md|results-block.md|gpu-resets.txt|pr-body*.md) ;;
            *) die "$OUT holds ${e##*/}, which validate.sh does not write: give a new or empty --out" ;;
        esac
    done
fi
mkdir -p "$OUT/logs"
OUT=$(cd "$OUT" && pwd)
touch "$OUT/.validate-out"
BUILD_CACHE=${BUILD_CACHE:-$(dirname "$OUT")/base-builds}
[[ $BUILD_CACHE_KEEP =~ ^[1-9][0-9]*$ ]] || die "BUILD_CACHE_KEEP must be a number above 0"
mkdir -p "$BUILD_CACHE"
# one run at a time per cache too (two runs with different MODELS_DIR can share it)
exec 8> "$BUILD_CACHE/.lock"
flock -n 8 || die "another validate.sh is using $BUILD_CACHE"
# a reused --out keeps its worktrees and builds, never the results of an earlier run
rm -rf "$OUT/hip" "$OUT/vk" "$OUT/cpu" "$OUT/quant" "$OUT/logs"/*.status
rm -f "$OUT/summary.md" "$OUT/results-block.md" "$OUT/pr-body-now.md" "$OUT/pr-body-new.md"
SMALL=$(realpath "$SMALL")
[ -z "$SMALL_MOE" ] || SMALL_MOE=$(realpath "$SMALL_MOE")
[ -z "$PR$BODY" ] || cp "$PR_BODY" "$OUT/pr-body.md"

# match the command line: pgrep -x only sees the first 15 characters of a name (llama-perplexity has 16)
if busy=$(pgrep -af '(^|/)(llama-(server|cli|bench|perplexity|completion|imatrix|quantize)|test-backend-ops|test-llama-archs)( |$)'); then die "the GPU is busy: $busy"; fi

# one cmake argument per line: a ROCm or deps path with spaces stays one argument
flags() { case $1 in hip) printf '%s\n' -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 "-DCMAKE_HIP_COMPILER=$("$ROCM_PATH/bin/hipconfig" -l)/clang" "-DCMAKE_PREFIX_PATH=$("$ROCM_PATH/bin/hipconfig" -R)" ;; vk) printf '%s\n' -DGGML_VULKAN=ON "-DCMAKE_PREFIX_PATH=$SPIRV_PREFIX" ;; cpu) ;; esac; }
inc()   { if [ "$1" = vk ]; then echo "$SPIRV_PREFIX/include${CPATH:+:$CPATH}"; else echo "${CPATH:-}"; fi; }
# the SPIR-V headers in DEPS_DIR, installed on first use: the same on every Linux distribution, no root needed
spirv_headers() {
    local p=$DEPS_DIR/spirv-headers-$SPIRV_TAG t
    if [ ! -f "$p/share/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake" ]; then
        echo "install SPIR-V headers $SPIRV_TAG to $p" >&2
        t=$(mktemp -d)
        curl -fsSL --retry 3 -o "$t/s.tgz" "https://github.com/KhronosGroup/SPIRV-Headers/archive/refs/tags/$SPIRV_TAG.tar.gz" || die "cannot download the SPIR-V headers"
        echo "$SPIRV_SHA256  $t/s.tgz" | sha256sum -c --quiet > /dev/null 2>&1 || die "SPIR-V headers: sha256 mismatch"
        tar -xzf "$t/s.tgz" -C "$t"
        { cmake -S "$t/SPIRV-Headers-$SPIRV_TAG" -B "$t/b" -DCMAKE_INSTALL_PREFIX="$p" -DSPIRV_HEADERS_ENABLE_TESTS=OFF && cmake --install "$t/b"; } > "$t/log" 2>&1 \
            || { cat "$t/log" >&2; die "cannot install the SPIR-V headers"; }
        rm -rf "$t"
    fi
    echo "$p"
}
dev()   { case $1 in hip) echo ROCm0 ;; vk) echo Vulkan0 ;; cpu) echo CPU ;; esac; }
ngl()   { if [ "$1" = cpu ]; then echo 0; else echo 99; fi; }
bin()   { echo "$OUT/build-$1-$2/bin"; }
sha()   { if [ -s "$1" ]; then sha256sum < "$1" | cut -c1-16; else echo missing; fi; }
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
        # compute buffers can live a fraction of a second (pp512 of a small model): at 0.5 s one round in two missed 385 MiB
        sleep 0.1
    done
    echo "PEAK_RSS_MIB=$((rp / 1024))"
    [ "$2" -eq 0 ] || echo "PEAK_GPU_MIB=$(((gp - $2) / 1048576))"
}
# run a command, keep its output and its memory peaks, never stop the whole validation on one failure.
# RUN_ERR=<file>: stderr there instead of the log
run() {
    local log=$1 g0 pid rc=0
    shift
    g0=$(gpu_used)
    # a hung GPU must not stop the validation forever: killed after RUN_TIMEOUT, it exits 124 (137 if it ignores TERM)
    if [ -n "${RUN_ERR:-}" ]; then
        timeout -k 60 "$RUN_TIMEOUT" "$@" > "$log" 2> "$RUN_ERR" &
    else
        timeout -k 60 "$RUN_TIMEOUT" "$@" > "$log" 2>&1 &
    fi
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
    echo "small_type=$SMALL_TYPE"
    [ -z "$SMALL_MOE" ] || echo "small_moe=$(basename "$SMALL_MOE") $(sha256sum "$SMALL_MOE" | cut -c1-16)"
    echo "moe_type=$MOE_TYPE"
    echo "moe_quants=$MOE_QUANTS"
    echo "kv_types=$KV_TYPES"
    echo "quants=$QUANTS"
    echo "pp_sizes=$PP_SIZES"
    echo "depths=$DEPTHS"
    echo "ops_jobs=$OPS_JOBS"
    for mm in "${MODELS[@]}"; do echo "model=${mm%%=*} $(basename "${mm#*=}")"; done
    echo "backends=$BACKENDS"
    case ,$BACKENDS, in *,hip,*) echo "rocm=$(cat "$ROCM_PATH/.info/version" 2> /dev/null || basename "$("$ROCM_PATH/bin/hipconfig" -R)") (HIP $("$ROCM_PATH/bin/hipconfig" --version))" ;; esac
} > "$OUT/manifest.txt"

# 1. sources and builds
SPIRV_PREFIX=""
case ,$BACKENDS, in
    *,vk,*)
        command -v glslc > /dev/null || die "the Vulkan build needs glslc (package glslc or shaderc, or the Vulkan SDK)"
        SPIRV_PREFIX=$(spirv_headers) || exit 1 ;;
esac
git worktree prune
for side in base cand; do
    rev=$BASE; [ $side = cand ] && rev=$CAND
    [ -d "$OUT/src-$side" ] || git worktree add -q --detach "$OUT/src-$side" "$rev"
    git -C "$OUT/src-$side" checkout -q --detach "$rev"
done
TARGETS="llama-perplexity llama-tokenize llama-imatrix llama-bench llama-server llama-quantize test-backend-ops test-llama-archs test-save-load-state"
# ccache, when installed, for every compiler (the HIP one included): the candidate shares most objects with the baseline
cc=()
command -v ccache > /dev/null && cc=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_HIP_COMPILER_LAUNCHER=ccache)
# toolchain identity: a cached baseline build is reused only with the same compilers
toolchain() {
    case $1 in
        hip) "$ROCM_PATH/bin/hipconfig" --version 2> /dev/null || true ;;
        vk) glslc --version 2> /dev/null | head -1 || true
            grep -h -m1 '#define VK_HEADER_VERSION ' /usr/include/vulkan/vulkan_core.h "${VULKAN_SDK:-/nonexistent}"/include/vulkan/vulkan_core.h 2> /dev/null || true ;;
    esac
    c++ --version 2> /dev/null | head -1 || true
    cmake --version 2> /dev/null | head -1 || true
    # cmake takes the compilers and flags from the environment too
    env | grep -E '^(CC|CXX|CFLAGS|CXXFLAGS|LDFLAGS|HIPFLAGS|HIPCXX|CPATH)=' | sort || true
}
build() {  # <be> <source dir> <build dir> <log>
    { HIP_PATH=$("$ROCM_PATH/bin/hipconfig" -R 2> /dev/null || true) cmake -S "$2" -B "$3" "${fl[@]}" "${cc[@]}" -DCMAKE_BUILD_TYPE=Release && \
      CPATH=$(inc $1) cmake --build "$3" -j --target $TARGETS; } > "$4" 2>&1
}
declare -A BASE_CACHE   # <be> -> the cache directory of its baseline build, when it has one
for be in ${BACKENDS//,/ }; do
    mapfile -t fl < <(flags $be)
    for side in base cand; do
        echo "build $side $be"
        st=$OUT/logs/build-$side-$be.status
        # the baseline is the same commit for every step and every rerun of a sync, and often for the next sync: one
        # build per commit, configuration and toolchain, in BUILD_CACHE, linked from the output directory. An output
        # directory that already holds its own baseline build (an older run) keeps it.
        if [ $side = base ] && { [ ! -e "$OUT/build-base-$be" ] || [ -L "$OUT/build-base-$be" ]; }; then
            key=$(git rev-parse --short=12 "$BASE")-$be-$({ printf '%s\n' "${fl[@]}" "${cc[@]}" $TARGETS; toolchain $be; } | sha256sum | cut -c1-12)
            cdir=$BUILD_CACHE/$key
            if [ "$(cat "$cdir/status" 2> /dev/null)" != ok ]; then
                mkdir -p "$BUILD_CACHE"
                [ -d "$cdir/src" ] || git worktree add -q --detach "$cdir/src" "$BASE"
                git -C "$cdir/src" checkout -q --detach "$BASE"
                if build $be "$cdir/src" "$cdir/build" "$cdir/build.log"; then
                    echo ok > "$cdir/status"
                    # read-only once done: nothing writes to a kept build by mistake (its results go to results-*)
                    chmod -R a-w "$cdir/build"
                else
                    echo failed > "$cdir/status"
                fi
            else
                echo "  reused $cdir"
            fi
            cp "$cdir/build.log" "$OUT/logs/build-base-$be.log"
            ln -sfn "$cdir/build" "$OUT/build-base-$be"
            cp "$cdir/status" "$st"
            touch "$cdir"   # most recently used, for the eviction below
            BASE_CACHE[$be]=$cdir
            continue
        fi
        # CPATH: upstream finds SPIRV-Headers but does not link its target, so its include directory never reaches the compiler
        if build $be "$OUT/src-$side" "$OUT/build-$side-$be" "$OUT/logs/build-$side-$be.log"; then echo ok > "$st"; else echo failed > "$st"; fi
    done
done

# keep the BUILD_CACHE_KEEP most recently used baseline builds, never one this run uses. Only entries this script
# made (named <sha>-<backend>-<hash>, with a status file): BUILD_CACHE may point to a shared directory by mistake
mapfile -t entries < <(find "$BUILD_CACHE" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended \
    -regex '.*/[0-9a-f]{12}-(hip|vk|cpu)-[0-9a-f]{12}' -printf '%T@ %p\n' | sort -rn | cut -d' ' -f2-)
for d in "${entries[@]:$BUILD_CACHE_KEEP}"; do
    [ -f "$d/status" ] || continue
    used=0
    for u in "${BASE_CACHE[@]}"; do [ "$u" = "$d" ] && used=1; done
    [ $used = 1 ] && continue
    chmod -R u+w "$d" && rm -rf "$d" && echo "removed the old baseline build $d"
done
git worktree prune

# the test texts come from the baseline (the fork before the sync): the candidate must not choose its own test data
C=$OUT/src-base/tests/corpus
for n in prose code structured numeric; do [ -s "$C/correctness-$n.txt" ] || die "the baseline has no tests/corpus/correctness-$n.txt"; done
cat "$C/correctness-prose.txt" "$C/correctness-code.txt" "$C/correctness-structured.txt" "$C/correctness-numeric.txt" > "$OUT/all.txt"
sha256sum "$C"/correctness-*.txt "$OUT/all.txt" > "$OUT/corpus.sha256"

# logits of base and cand on the same text, plus the reference: base against itself with a smaller ubatch
model_id() {
    local n
    n=$(stat -c '%s' "$1")
    echo "$n"
    if [ "$n" -lt 4294967296 ]; then sha256sum < "$1"; else { head -c 1048576 "$1"; tail -c 1048576 "$1"; } | sha256sum; fi
}
kld() {  # <be> <label> <model> <text> <args> <reference args>
    local be=$1 d=$OUT/$1/kld/$2 k=$OUT/$1/kld/$2.kld
    # an array: a model path with spaces stays one argument (<args> are options only, split on purpose)
    # --load-mode none as in bench(): with mmap a model of 80 GB is resident twice on a 128 GB UMA machine (page cache
    # and GPU copy), and the long-context run is OOM-killed
    local o=(-m "$3" -ngl "$(ngl $1)" -fa on -ctk f16 -ctv f16 --load-mode none)
    run "$d-base.txt" "$(bin base $be)/llama-perplexity" "${o[@]}" $5 -f "$4" --kl-divergence-base "$k"
    run "$d-cand.txt" "$(bin cand $be)/llama-perplexity" "${o[@]}" $5 --kl-divergence-base "$k" --kl-divergence
    # the reference (the baseline against itself, another ubatch) depends only on the baseline build, the model, the
    # text and the options: kept next to the cached baseline build, so a rerun skips it. Only a finished run is kept
    local rd ref=""
    if rd=$(results_dir "$be"); then
        # the model by its content, not its date: the small model quants are written again by every run. All of it
        # below 4 GiB (every small model); above, its size and first and last MiB, to stay fast on a large model
        ref=$rd/kldref-$({ echo "$2 $5 $6"; model_id "$3"; sha256sum < "$4"; } | sha256sum | cut -c1-16).txt
    fi
    if [ -n "$ref" ] && [ -s "$ref" ] && grep -F 'Same top p' "$ref" > /dev/null && ! grep -x 'EXIT=[0-9]*' "$ref" > /dev/null; then
        cp "$ref" "$d-ref.txt"
    else
        run "$d-ref.txt"  "$(bin base $be)/llama-perplexity" "${o[@]}" $6 --kl-divergence-base "$k" --kl-divergence
        if [ -n "$ref" ] && grep -F 'Same top p' "$d-ref.txt" > /dev/null && ! grep -x 'EXIT=[0-9]*' "$d-ref.txt" > /dev/null; then
            mkdir -p "$rd" && cp "$d-ref.txt" "$ref"
        fi
    fi
    rm -f "$k"   # about 0.5 MB per scored token with a 248k vocab
}

bench() {  # <be> <label> <model> <args> <expected tests "p g d" ...>: alternating order, two rounds per side, one log per round
    local t i=0
    printf '%s\n' "${@:5}" | tr ':' ' ' > "$OUT/$1/bench/$2.expect"
    for t in base cand base cand; do
        i=$((i + 1))
        run "$OUT/$1/bench/$2-$t-$i.log" "$(bin $t $1)/llama-bench" -m "$3" $4 -o jsonl
    done
}

fails() {  # <test-backend-ops log>: the failing test cases it lists at the end, one per line, sorted
    sed -n '/^Failing tests:/,/^  Backend /p' "$1" 2> /dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -E '^  [A-Z0-9_]+\(.*\)$' | sed 's/^  //' | sort -u
}

# test-backend-ops compares the GPU with the CPU inside one build: its result can only change when the sync changes
# ggml. The reason to skip it on <be>, or nothing to run it: a change under ggml/src/ggml-<backend>/ concerns that
# backend only (ggml-cuda and ggml-hip: HIP), any other change under ggml/ (core, ggml-cpu, the reference) concerns all.
ops_skip() {  # <be>
    local f hit=""
    while IFS= read -r f; do
        case $f in
            ggml/src/ggml-cuda/*|ggml/src/ggml-hip/*) [ "$1" = hip ] && hit=1 ;;
            ggml/src/ggml-vulkan/*) [ "$1" = vk ] && hit=1 ;;
            ggml/src/ggml-cpu/*) hit=1 ;;
            ggml/src/ggml-*/*) ;;
            ggml/*) hit=1 ;;
        esac
    done < <(git diff --name-only "$BASE" "$CAND" -- ggml/)
    # new or changed test cases from upstream run too, and the CMake files read before ggml/ (default GGML_* options,
    # global compile flags) or that build the test
    git diff --quiet "$BASE" "$CAND" -- tests/test-backend-ops.cpp tests/CMakeLists.txt CMakeLists.txt cmake/ || hit=1
    [ -n "$hit" ] || echo "the sync changes no ggml code that $1 runs, nor test-backend-ops or the CMake files"
}

# results of the baseline build kept next to it: the same build gives the same test-backend-ops and test-llama-archs
# results, so a rerun of a sync, or the next sync from the same master, runs only the candidate. Kept only when the
# run finished (<done> printed, no EXIT=), and only with the GPU driver the machine runs now: the kernel (amdgpu, and
# ROCm is in the build key) and, for Vulkan, the RADV library of Mesa, which updates without a kernel change.
driver_id() {
    { uname -r
      # the amdgpu firmware (MES, SMU...) updates on its own package
      ls -l --time-style=+%s /lib/firmware/amdgpu 2> /dev/null || true
      if [ "$1" = vk ]; then
          # the driver the loader picks: RADV or AMDVLK from the system, or the ICD files named in the environment
          local icd="${VK_ICD_FILENAMES:-} ${VK_DRIVER_FILES:-}" j l
          echo "$icd"
          cat /usr/lib*/libvulkan_radeon.so /usr/lib*/*/libvulkan_radeon.so /usr/lib*/amdvlk64.so /usr/lib*/*/amdvlk64.so 2> /dev/null || true
          # and the library each AMD ICD file names: a Mesa or AMDVLK in /usr/local or /opt is found through it only
          for j in ${icd//:/ } /usr/share/vulkan/icd.d/*{radeon,amd}*.json /usr/local/share/vulkan/icd.d/*{radeon,amd}*.json \
                   /etc/vulkan/icd.d/*{radeon,amd}*.json; do
              [ -f "$j" ] || continue
              cat "$j"
              l=$(sed -n 's/.*"library_path" *: *"\([^"]*\)".*/\1/p' "$j" | head -1)
              case $l in
                  /*) cat "$l" 2> /dev/null || true ;;
                  */*) cat "$(dirname "$j")/$l" 2> /dev/null || true ;;   # relative to the ICD file; a bare name is in /usr/lib*
              esac
          done
          env | grep -E '^(AMD_VULKAN_ICD|RADV_[A-Z_]*|ACO_[A-Z_]*)=' | sort || true
      else
          env | grep -E '^(HSA_[A-Z_]*|HIP_[A-Z_]*|AMD_LOG_LEVEL|AMD_SERIALIZE_[A-Z_]*)=' | sort || true
      fi
      # the backend options read at run time
      env | grep -E '^GGML_[A-Z0-9_]*=' | sort || true; } | sha256sum | cut -c1-12
}
results_dir() { [ -n "${BASE_CACHE[$1]:-}" ] && echo "${BASE_CACHE[$1]}/results-$(driver_id "$1")"; }
cached() {  # <be> <file> <done>: copies a kept result into the output directory
    local d
    d=$(results_dir "$1") || return 1
    [ -s "$d/$2" ] && grep -F -- "$3" "$d/$2" > /dev/null && ! grep -x 'EXIT=[0-9]*' "$d/$2" > /dev/null || return 1
    cp "$d/$2" "$OUT/$1/$2"
    echo "$1: $2 from $d"
}
keep() {  # <be> <file> <done>
    local d
    d=$(results_dir "$1") || return 0
    [ -s "$OUT/$1/$2" ] && grep -F -- "$3" "$OUT/$1/$2" > /dev/null && ! grep -x 'EXIT=[0-9]*' "$OUT/$1/$2" > /dev/null || return 0
    mkdir -p "$d" && cp "$OUT/$1/$2" "$d/$2"
}

# the small models in every quant type run the quantizer, the quant kernels and the whole inference path on a real
# model: the reason to skip them on <be>, or nothing to run them. Their results depend on any ggml code that <be> runs
# (the ops_skip rule), on the llama library (src/, include/: the graph, the MoE FFN, the context, the model files), on
# common/ and on the tools that make and measure them. Only a sync that changes none of these skips them.
quants_skip() {  # <be>
    git diff --quiet "$BASE" "$CAND" -- src/ include/ common/ tools/quantize/ tools/imatrix/ tools/perplexity/ tools/llama-bench/ || return 0
    [ -z "$(ops_skip "$1")" ] || echo "the sync changes no ggml code that $1 runs, nor the llama library, common/ or the quantize, imatrix, perplexity and bench tools"
}

# a quantizer that writes another file: both files against the logits of the source model, with the baseline build,
# so only the file differs. Without it a broken candidate quantizer would pass: the KLD of quants() uses the base file
srckld() {  # <be> <name> <source model> <type> <base file> <cand file>
    local d=$OUT/quant/$2-$4 k=$OUT/quant/$2-src.kld o
    o=(-ngl "$(ngl $1)" -fa on -ctk f16 -ctv f16 -c 512 -b 512 -ub 512 --chunks 8)
    if [ ! -s "$k" ] || grep -x 'EXIT=[0-9]*' "$OUT/quant/$2-src.txt" > /dev/null; then
        run "$OUT/quant/$2-src.txt" "$(bin base $1)/llama-perplexity" -m "$3" "${o[@]}" -f "$OUT/all.txt" --kl-divergence-base "$k"
    fi
    run "$d-src-base.txt" "$(bin base $1)/llama-perplexity" -m "$5" "${o[@]}" --kl-divergence-base "$k" --kl-divergence
    run "$d-src-cand.txt" "$(bin base $1)/llama-perplexity" -m "$6" "${o[@]}" --kl-divergence-base "$k" --kl-divergence
    # the baseline cannot read it: a broken file, or a format only the new code knows. The candidate tells them apart
    if grep -x 'EXIT=[0-9]*' "$d-src-cand.txt" > /dev/null; then
        run "$d-src-candbin.txt" "$(bin cand $1)/llama-perplexity" -m "$6" "${o[@]}" --kl-divergence-base "$k" --kl-divergence
    fi
}

# a small model in every quant type of <types>: both quantizers (same file or not), then KLD and speed on the file of the
# baseline quantizer. <name>: small or moe, the prefix of every file; the rest of the arguments go to llama-quantize.
quants() {  # <be> <name> <model> <source type> <types> [quantize options...]
    local be=$1 n=$2 m=$3 st=$4 types=$5 q f c sb sc
    shift 5
    for q in $types; do
        f=$OUT/quant/$n-$q.gguf c=$OUT/quant/cand-$n-$q.gguf
        if [ "$q" = "$st" ]; then
            f=$m
        elif [ ! -f "$OUT/quant/$n-$q.sha" ]; then
            # to .part first: a quantizer killed half-way must not leave a file that looks done
            run "$OUT/quant/$n-$q-base.log" "$(bin base $be)/llama-quantize" "$@" "$m" "$f.part" "$q"
            run "$OUT/quant/$n-$q-cand.log" "$(bin cand $be)/llama-quantize" "$@" "$m" "$c.part" "$q"
            grep -x 'EXIT=[0-9]*' "$OUT/quant/$n-$q-base.log" > /dev/null || { [ -s "$f.part" ] && mv "$f.part" "$f"; } || true
            grep -x 'EXIT=[0-9]*' "$OUT/quant/$n-$q-cand.log" > /dev/null || { [ -s "$c.part" ] && mv "$c.part" "$c"; } || true
            # the summary fails the quant when the candidate file is missing
            echo "$(sha "$f") $(sha "$c")" > "$OUT/quant/$n-$q.sha"
            # on the first backend only, with the files: the second reuses them, and this compares files, not kernels
            read -r sb sc < "$OUT/quant/$n-$q.sha"
            [ "$sb" = missing ] || [ "$sc" = missing ] || [ "$sb" = "$sc" ] || srckld "$be" "$n" "$m" "$q" "$f" "$c"
            rm -f "$f.part" "$c" "$c.part"
        fi
        [ -f "$f" ] || { echo "$be: no $n $q file from the base quantizer, skip its KLD and bench"; continue; }
        kld "$be" "$n-$q" "$f" "$OUT/all.txt" "-c 512 -b 512 -ub 512 --chunks 8" "-c 512 -b 512 -ub 128 --chunks 8"
        bench "$be" "$n-$q" "$f" "-p 512 -n 128 -r 3 -ngl $(ngl $be) -fa on" "512 0 0" "0 128 0"
    done
    rm -f "$OUT/quant/$n-src.kld"   # the logits of srckld(): about 1 GB
}

# a free port: another listener on a fixed port would answer for a server that never started
free_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }

server() {  # <be> <arch> <model>: tool call, greedy decode, MTP drafting
    local be=$1 a=$2 m=$3 t s ok spec g0 sp port
    for t in base cand; do
        for spec in "--spec-type draft-mtp" ""; do
            g0=$(gpu_used)
            port=$(free_port)
            # under timeout as run(): llama-server only asks itself to stop on the first TERM, so a stuck one would never
            # exit; a TERM to timeout is passed on and followed by a KILL 30 s later (also when this script is stopped)
            timeout -k 30 "$RUN_TIMEOUT" "$(bin $t $be)/llama-server" -m "$m" -ngl $(ngl $be) -fa on --jinja --host 127.0.0.1 --port "$port" $spec > "$OUT/$be/server/$a-$t.log" 2>&1 &
            s=$! ok=0
            peak $s "$g0" > "$OUT/$be/server/$a-$t.mem" &
            sp=$!
            for _ in $(seq 300); do
                # our server must still be alive: only then is the answer its own
                kill -0 $s 2> /dev/null || break
                if curl -sf --max-time 10 "localhost:$port/health" > /dev/null; then ok=1; break; fi
                sleep 2
            done
            if [ $ok = 1 ]; then
                echo "${spec:-no-mtp}" > "$OUT/$be/server/$a-$t.spec"
                # bounded: a candidate that loops at temperature 0 must not run for hours
                curl -s --max-time 600 "localhost:$port/v1/chat/completions" -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"What is the weather in Paris?"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get the weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"max_tokens":512,"temperature":0}' > "$OUT/$be/server/$a-tool-$t.json" || true
                curl -s --max-time 600 "localhost:$port/v1/chat/completions" -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Write a Python function that parses ISO 8601 dates, with tests."}],"max_tokens":256,"temperature":0}' > "$OUT/$be/server/$a-decode-$t.json" || true
            fi
            kill $s 2> /dev/null || true
            # bounded: a process stuck in the GPU driver outlives even the KILL, and a bare wait would never return
            for _ in $(seq 60); do kill -0 $s 2> /dev/null || break; sleep 1; done
            if kill -0 $s 2> /dev/null; then
                echo "$be: the $t llama-server did not exit 60 s after it was stopped, left behind" | tee -a "$OUT/$be/server/$a-$t.log"
                kill $sp 2> /dev/null || true
            else
                wait $s 2> /dev/null || true
            fi
            wait $sp 2> /dev/null || true
            cat "$OUT/$be/server/$a-$t.mem" >> "$OUT/$be/server/$a-$t.log"
            [ $ok = 1 ] && break
            # the try without MTP writes the same log: keep the one that says why the MTP server did not start
            [ -z "$spec" ] || mv "$OUT/$be/server/$a-$t.log" "$OUT/$be/server/$a-$t-mtp-failed.log"
        done
    done
}

# write summary.md and, with --pr, put it in the PR body between markers (replaced at every update)
report_pr() {
    # GPU resets during the run, when the kernel log is readable (often it is not without root): nothing otherwise
    local since n
    since=$(date -d "@$START" '+%F %T')
    n=$({ journalctl -k -q --no-pager --since "$since" 2> /dev/null || dmesg --since "$since" 2> /dev/null || true; } | grep -c -E 'amdgpu.*ring .* timeout' || true)
    if [ "${n:-0}" -gt 0 ]; then echo "$n" > "$OUT/gpu-resets.txt"; else rm -f "$OUT/gpu-resets.txt"; fi
    python3 "$HERE/validate_summary.py" "$OUT" > "$OUT/summary.md"
    # the same results between markers, for a PR body: a later run replaces this block instead of adding a second one
    {
        echo "<!-- validate.sh results -->"
        [ -z "${1:-}" ] || printf '%s\n\n' "$1"
        sed '0,/^## Validation$/s//## Validation results/' "$OUT/summary.md"
        echo "<!-- /validate.sh results -->"
    } > "$OUT/results-block.md"
    [ -n "$PR" ] || return 0
    if [ "$POST" = comment ]; then
        # a reviewer does not edit the PR body: post the final result as a comment
        [ -z "${1:-}" ] || return 0
        if [ "$(wc -c < "$OUT/summary.md")" -gt 65000 ]; then
            echo "ERROR: the results pass the GitHub limit of 65536 characters for a comment, they are in $OUT/summary.md" >&2
            return 0
        fi
        gh api "repos/halo-box/strix-llama.cpp/issues/$PR/comments" -F body=@"$OUT/summary.md" > /dev/null \
            && echo "PR $PR: results posted as a comment" || echo "ERROR: cannot comment on PR $PR, the results are in $OUT/summary.md" >&2
        return 0
    fi
    gh pr view "$PR" --repo halo-box/strix-llama.cpp --json body --jq .body > "$OUT/pr-body-now.md" || { echo "ERROR: cannot read PR $PR" >&2; return 0; }
    python3 - "$OUT/pr-body-now.md" "$OUT/results-block.md" > "$OUT/pr-body-new.md" <<'PY'
import re, sys
body, block = open(sys.argv[1]).read().rstrip("\n"), open(sys.argv[2]).read().strip()
# whole lines only, begin before end: a marker quoted in the text must not cut the body
m = re.search(r"^<!-- validate\.sh results -->$.*?^<!-- /validate\.sh results -->$", body, re.M | re.S)
if m:
    body = body[:m.start()] + block + body[m.end():]
else:
    body = body + "\n\n" + block
print(body)
PY
    if [ "$(wc -c < "$OUT/pr-body-new.md")" -gt 65000 ]; then
        echo "ERROR: with the results the PR body would pass the GitHub limit of 65536 characters: posting them as a comment" >&2
        [ "$(wc -c < "$OUT/summary.md")" -le 65000 ] || { echo "ERROR: too long for a comment too, the results are in $OUT/summary.md" >&2; return 0; }
        gh api "repos/halo-box/strix-llama.cpp/issues/$PR/comments" -F body=@"$OUT/summary.md" > /dev/null || echo "ERROR: cannot comment on PR $PR either" >&2
        return 0
    fi
    # REST, not gh pr edit: gh pr edit fails on this repo with a GraphQL error about classic Projects
    gh api -X PATCH "repos/halo-box/strix-llama.cpp/pulls/$PR" -F body=@"$OUT/pr-body-new.md" > /dev/null \
        && echo "PR $PR updated" || echo "ERROR: cannot update PR $PR, the results are in $OUT/summary.md" >&2
}

for be in ${BACKENDS//,/ }; do
    [ "$(cat "$OUT/logs/build-base-$be.status" 2> /dev/null)" = ok ] && [ "$(cat "$OUT/logs/build-cand-$be.status" 2> /dev/null)" = ok ] || { echo "skip $be: build failed"; continue; }
    mkdir -p "$OUT/$be/kld" "$OUT/$be/bench" "$OUT/$be/server" "$OUT/$be/quant"

    # 2. every op and every arch, GPU against CPU
    echo "$be: test-backend-ops, test-llama-archs"
    # test-backend-ops checks the GPU against the CPU, results only: base and candidate at the same time, OPS_JOBS
    # workers each (upstream #23637: each worker has its own device and CPU backend). On a Strix Halo, 2 x -j 4 runs
    # both in about 27 min, against 44 min for -j 1 side by side or -j 16 one after the other (the largest
    # MUL_MAT_ID cases make a tail that leaves the CPU idle, so the two tails overlap)
    why=$(ops_skip "$be")
    if [ -n "$why" ]; then
        echo "$be: test-backend-ops skipped: $why"
        echo "$why" > "$OUT/$be/ops-skipped.txt"
    else
        ops() {  # <side>: stderr apart, the results are printed whole but the logs of the other workers cut their lines
            RUN_ERR="$OUT/$be/ops-$1.err" run "$OUT/$be/ops-$1.txt" "$(bin $1 $be)/test-backend-ops" -b "$(dev $be)" -j "$OPS_JOBS"
        }
        j1=""
        cached "$be" ops-base.txt "backends passed" || { ops base & j1=$!; }
        ops cand &
        j2=$!
        wait $j1 $j2 || true
        # the baseline did not finish: one GPU hang or reset stops both, or the candidate beside it took its memory.
        # Again, alone: if it does not finish, the machine did it (INCOMPLETE); the candidate is judged below
        if [ -n "$j1" ] && ! grep -F 'backends passed' "$OUT/$be/ops-base.txt" > /dev/null 2>&1; then
            echo "$be: the baseline test-backend-ops stopped, again alone"
            ops base
        fi
        # the candidate alone stopped while running beside the baseline: again alone, so a stop caused by the other
        # run is not blamed on it
        if [ -n "$j1" ] && grep -F 'backends passed' "$OUT/$be/ops-base.txt" > /dev/null 2>&1 \
           && ! grep -F 'backends passed' "$OUT/$be/ops-cand.txt" > /dev/null 2>&1; then
            echo "$be: the candidate test-backend-ops stopped beside the baseline, again alone"
            ops cand
        fi
        # a failure of the candidate that the baseline does not have: alone (-j 1, nothing else on the GPU) three
        # times. Some kernels fail only now and then beside other work (a ROCmFP4 MUL_MAT_VEC_FUSION gave NaN in
        # 2 of 4 parallel runs, never alone): the summary tells such a failure from a new one
        : > "$OUT/$be/ops-rerun.txt"
        comm -13 <(fails "$OUT/$be/ops-base.txt") <(fails "$OUT/$be/ops-cand.txt") | head -20 | while IFS= read -r tc; do
            n=0
            for i in 1 2 3; do
                timeout -k 60 "$RUN_TIMEOUT" "$(bin cand $be)/test-backend-ops" -b "$(dev $be)" -j 1 -o "$tc" 2> /dev/null \
                    | grep -q -E '^ *1/1 tests passed' && n=$((n + 1))
            done
            printf '%s\t%d/3\n' "$tc" "$n" >> "$OUT/$be/ops-rerun.txt"
        done
        keep "$be" ops-base.txt "backends passed"
    fi
    for t in base cand; do
        [ $t = cand ] || ! cached "$be" archs-base.txt "Summary:" || continue
        run "$OUT/$be/archs-$t.txt" "$(bin $t $be)/test-llama-archs" -b "$(dev $be)"
        sed -i 's/\x1b\[[0-9;]*m//g' "$OUT/$be/archs-$t.txt"
    done
    keep "$be" archs-base.txt "Summary:"

    # 3. the small model in every quant type: logits, speed, and whether the quantizer output changed
    why=$(quants_skip "$be")
    if [ -n "$why" ]; then
        echo "$be: quant matrices skipped: $why"
        echo "$why" > "$OUT/$be/quants-skipped.txt"
    else
        echo "$be: Qwen3.5-0.8B quant matrix"
        mkdir -p "$OUT/quant"
        # IQ1 and IQ2 need an importance matrix: one, from the baseline build, for every type and both quantizers, so the
        # quantizers are the only difference. Computed on the test texts: it makes no side better than the other.
        # Kept next to the cached baseline build: every quant names it (quantize.imatrix.file and .dataset), so the same
        # path and bytes give the same quant files on the next run, and kld() finds their references. One for both
        # backends, as the second one reuses the quant files of the first; if it failed, the second computes its own
        if [ ! -s "${im:-}" ]; then
            im=$OUT/quant/small-imatrix.gguf
            if rd=$(results_dir "$be"); then
                im=$rd/imatrix-$({ echo "-c 512 --chunks 64 unused"; model_id "$SMALL"; sha256sum < "$OUT/all.txt"; } | sha256sum | cut -c1-16).gguf
            fi
            if [ ! -s "$im" ]; then
                # to .part first: llama-imatrix saves every 10 chunks, a run killed half-way must not leave a file that looks done
                mkdir -p "$(dirname "$im")"
                run "$OUT/quant/imatrix.log" "$(bin base $be)/llama-imatrix" -m "$SMALL" -f "$OUT/all.txt" -o "$im.part" -ngl "$(ngl $be)" -c 512 --chunks 64
                # the layers it never runs (an MTP head): no importance data, kept next to it for the overrides below
                grep -o 'unused tensor blk\.[0-9]*\.' "$OUT/quant/imatrix.log" | grep -o '[0-9]*' | sort -un | paste -sd '|' > "$im.unused" || true
                grep -x 'EXIT=[0-9]*' "$OUT/quant/imatrix.log" > /dev/null || { [ -s "$im.part" ] && mv "$im.part" "$im"; } || true
                rm -f "$im.part"
            fi
        fi
        imo=()
        if [ -s "$im" ]; then imo=(--imatrix "$im"); else echo "$be: no importance matrix, the IQ1 / IQ2 types will fail on both sides"; fi
        # the very low-bit types refuse a tensor without importance data: those layers in q8_0, the same on both sides
        if [ -s "$im" ] && [ -s "$im.unused" ]; then imo+=(--tensor-type "blk\.($(cat "$im.unused"))\.=q8_0"); fi
        quants "$be" small "$SMALL" "$SMALL_TYPE" "$QUANTS" "${imo[@]}"
        echo "$be: Granite 1B-A400M (MoE) quant matrix"
        quants "$be" moe "$SMALL_MOE" "$MOE_TYPE" "$MOE_QUANTS"
    fi

    # the small model with a quantized KV cache: K and V of the same type (the later -ctk / -ctv win over kld's f16)
    kvm=$SMALL
    echo "$be: KV cache types"
    for t in $KV_TYPES; do
        kld "$be" "kv-$t" "$kvm" "$OUT/all.txt" "-c 2048 -b 2048 -ub 2048 --chunks 4 -ctk $t -ctv $t" "-c 2048 -b 2048 -ub 512 --chunks 4 -ctk $t -ctv $t"
        bench "$be" "kv-$t" "$kvm" "-p 512 -n 128 -d 8192 -r 3 -ngl $(ngl $be) -fa on -ctk $t -ctv $t" "512 0 8192" "0 128 8192"
    done
    sls=$OUT/quant/small-Q4_0.gguf
    [ -f "$sls" ] || sls=$(ls "$OUT"/quant/small-*.gguf 2> /dev/null | grep -v -- '-imatrix\.gguf$' | head -1 || true)
    [ -n "$sls" ] || sls=$SMALL
    run "$OUT/$be/sls-base.txt" "$(bin base $be)/test-save-load-state" --model "$sls" -ngl $(ngl $be) -c 1024 -fa on
    run "$OUT/$be/sls-cand.txt" "$(bin cand $be)/test-save-load-state" --model "$sls" -ngl $(ngl $be) -c 1024 -fa on

    # 4. the real models of the archs this sync touches
    for mm in "${MODELS[@]}"; do
        a=${mm%%=*} m=${mm#*=}
        echo "$be: $a"
        for n in prose code structured numeric; do
            kld "$be" "$a-$n" "$m" "$C/correctness-$n.txt" "-c 2048 -b 2048 -ub 2048" "-c 2048 -b 2048 -ub 512"
        done
        # the four texts together at the largest context their token count allows, up to 16384: llama-perplexity needs
        # 2 x n_ctx tokens: about 43.5k with Qwen 3.5, 39.6k with Llama 3, but a vocab with more bytes per token can give
        # fewer than 32768
        nt=$("$(bin base $be)/llama-tokenize" -m "$m" -f "$OUT/all.txt" --ids --show-count 2> /dev/null | sed -n 's/^Total number of tokens: //p' || true)
        lc=$(( ${nt:-32768} / 2 / 512 * 512 ))
        [ "$lc" -le 16384 ] || lc=16384
        # llama-perplexity keeps n_ctx x n_vocab logits in RAM (16 GB at 16384 x 248k): on a 128 GB UMA machine a model
        # that already takes more than LONG_CTX_BIG_GPU_MIB of GPU memory at 2048 is OOM-killed, so its long run is shorter
        gpu=$(sed -n 's/^PEAK_GPU_MIB=//p' "$OUT/$be/kld/$a-numeric-base.txt" 2> /dev/null | tail -1)
        if [ "${gpu:-0}" -gt "${LONG_CTX_BIG_GPU_MIB:-65536}" ] && [ "$lc" -gt "${LONG_CTX_BIG:-4096}" ]; then
            lc=${LONG_CTX_BIG:-4096}
        fi
        echo "$lc" > "$OUT/$be/kld/$a-long.ctx"
        kld "$be" "$a-long" "$m" "$OUT/all.txt" "-c $lc -b 512 -ub 512" "-c $lc -b 256 -ub 256"
        # pp and tg in one process: llama-bench fills the context once per depth and restores it for the next tests
        # at that depth, so the fill (most of the time at 128k) is not done again for tg
        bench "$be" "$a-bench" "$m" "-p $PP_SIZES -n 128 -d $DEPTHS -b 4096 -ub 4096 -r 4 -ngl $(ngl $be) -fa on -ctk f16 -ctv f16 --load-mode none" \
            $(for d in ${DEPTHS//,/ }; do for p in ${PP_SIZES//,/ }; do echo "$p:0:$d"; done; echo "0:128:$d"; done)
        server "$be" "$a" "$m"
    done
    report_pr "Running: $be done, $(date -u '+%F %H:%M') UTC."
done

report_pr
echo
cat "$OUT/summary.md"
