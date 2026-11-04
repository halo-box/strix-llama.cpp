#!/usr/bin/env bash
#
# Runs test-backend-ops on a Vulkan device with every fusion engagement check on.
#
# The engagement asserts in tests/test-backend-ops.cpp (GGML_VK_TEST_REQUIRE_*) are opt-in: a default run compares the
# results but does not check that the fused path ran, so a fusion that silently stops engaging still passes. This script
# runs one test-backend-ops invocation per REQUIRE knob with its op filter, the census mode, and one negative control
# that must fail (it proves the asserts are live, i.e. the backend exports the dispatch census hooks).
#
# Usage:
#   scripts/vulkan-engagement-tests.sh [-b Vulkan0] [-k REGEX] [-x REGEX] [-l LOGDIR] [-n] path/to/test-backend-ops
#
#   -b  backend name passed to test-backend-ops -b (default Vulkan0)
#   -k  only run the knobs whose name matches REGEX (the census and the negative control always run unless -n)
#   -x  skip the knobs whose name matches REGEX
#   -l  directory for one log per invocation (default: a temporary directory, kept and printed)
#   -n  skip the census mode and the negative control
#
# A knob that the binary does not contain (an older or newer tree) is reported as n/a, not as a pass. A knob whose
# filter ran no case, or whose backend line never ran, fails. Known failures are listed in XFAIL below with the reason;
# they are reported as XFAIL and do not fail the run, and an XFAIL entry that passes is reported as XPASS.
#
# Exit status: 0 when every invocation passed or is an expected failure, 1 otherwise, 2 on bad usage.

set -u

backend=Vulkan0
knob_re=
skip_re=
logdir=
controls=1

usage() {
    sed -n '/^# Usage:/,/^# Exit status/p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}

while getopts "b:k:x:l:nh" opt; do
    case "$opt" in
        b) backend=$OPTARG ;;
        k) knob_re=$OPTARG ;;
        x) skip_re=$OPTARG ;;
        l) logdir=$OPTARG ;;
        n) controls=0 ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))
[ $# -eq 1 ] || usage
tbo=$1
[ -x "$tbo" ] || { echo "error: $tbo is not executable" >&2; exit 2; }

if [ -z "$logdir" ]; then
    logdir=$(mktemp -d "${TMPDIR:-/tmp}/vk-engage.XXXXXX")
fi
mkdir -p "$logdir"

# knob | -o op filter | -p params regex (may be empty) | extra environment (space separated, may be empty)
#
# Fusions that are on by default run with the default configuration. The opt-in ones get their switch, else the assert
# only checks that the path stays off; the ragged split runs with GGML_VK_RAGGED_FORCE=1 (no wave model), where its assert
# requires it on every eligible shape.
KNOBS=(
    "GGML_VK_TEST_REQUIRE_DOWN_STREAM|MUL_MAT_ID_MUL_CPY16,MUL_MAT_ID_CPY16||"
    "GGML_VK_TEST_REQUIRE_MMID_F16OUT|MUL_MAT_ID_MUL_CPY16,MUL_MAT_ID_CPY16||"
    "GGML_VK_TEST_REQUIRE_REGA|MUL_MAT_ID|n_mats=320,|"
    "GGML_VK_TEST_REQUIRE_PAIR_GLU|MOE_GLU||"
    "GGML_VK_TEST_REQUIRE_MOE_GLU_GEMV|MOE_GLU_GEMV||GGML_VK_MOE_GLU_GEMV=1"
    "GGML_VK_TEST_REQUIRE_F16B_NATIVE|MUL_MAT_F16B||"
    "GGML_VK_TEST_REQUIRE_ROWSTRIDE|MUL_MAT_F16B||"
    "GGML_VK_TEST_REQUIRE_DQ8|MUL_MAT_F16B||"
    "GGML_VK_TEST_REQUIRE_DQ8_F32B|MUL_MAT_DQ8_F32B||"
    "GGML_VK_TEST_REQUIRE_HC_W8A8|MUL_MAT|k=[0-9]+,pad=[0-9]+$|"
    "GGML_VK_TEST_REQUIRE_RAGGED|MUL_MAT|pad=[0-9]+,bs=[0-9]+$|GGML_VK_RAGGED_FORCE=1"
    "GGML_VK_TEST_REQUIRE_MMV_GLU|MUL_MAT_VEC_GLU||"
    "GGML_VK_TEST_REQUIRE_GET_ROWS_NARROW|GET_ROWS||"
    "GGML_VK_TEST_REQUIRE_HC_POST_NORM|HC_POST_NORM||"
    "GGML_VK_TEST_REQUIRE_HC_INJECT|HC_POST_NORM||"
    "GGML_VK_TEST_REQUIRE_HC_POST_NORM_ADD|HC_POST_NORM_ADD||"
    "GGML_VK_TEST_REQUIRE_HC_UP_MIX|HC_UP_MIX_F16||"
    "GGML_VK_TEST_REQUIRE_HC_DQ8_T5|HC_UP_MIX_F16||GGML_VK_HC_DQ8_TILE=5"
    "GGML_VK_TEST_REQUIRE_HC_MOE_EPI|HC_MOE_EPI||"
    "GGML_VK_TEST_REQUIRE_HC_DEFER|HC_LAYER||"
    "GGML_VK_TEST_REQUIRE_HC_BF16RES|HC_LAYER2||"
    "GGML_VK_TEST_REQUIRE_HC_BOUNDARY|DSV4_HC_BOUNDARY||"
    "GGML_VK_TEST_REQUIRE_FN_DECODE_HC|FN_DECODE_HC||"
    "GGML_VK_TEST_REQUIRE_GDN_NORM|GDN_NORM_F16||"
    "GGML_VK_TEST_REQUIRE_GDN_QKNORM|GDN_CONV_PREFILL,GDN_QK_ROOT||"
    "GGML_VK_TEST_REQUIRE_GDN_CONVFOLD|GDN_CONV_PREFILL||"
    "GGML_VK_TEST_REQUIRE_GDN_ZHM|GDN_OUT_NORM_PREFILL||GGML_VK_GDN_ZHM=1 GGML_VK_DENSE_DQ8=2"
    "GGML_VK_TEST_REQUIRE_GDN_STATE_DIRECT|GDN_STATE_DIRECT||"
    "GGML_VK_TEST_REQUIRE_QSA_SCORE|QSA_SCORE,QSA_STRIP||"
    "GGML_VK_TEST_REQUIRE_QSA_EXPAND|QSA_EXPAND||"
    "GGML_VK_TEST_REQUIRE_QSA_RADIX_SELECT|TOP_K||"
    "GGML_VK_TEST_REQUIRE_QSA_PREFILL_SELECT|TOP_K||"
    "GGML_VK_TEST_REQUIRE_FA_OUT_GATE|FA_OUT_GATE||"
    "GGML_VK_TEST_REQUIRE_FA_OUT_GATE_EPI|FA_OUT_GATE||GGML_VK_TEST_REQUIRE_FA_OUT_GATE=1"
    "GGML_VK_TEST_REQUIRE_FA_OUT_GATE_SEL|FA_OUT_GATE_SEL||"
    "GGML_VK_TEST_REQUIRE_FA_SEL_LEAN|QSA_PREFILL_MASKLESS||"
)

# knob | reason
XFAIL=()

xfail_reason() {
    local e
    for e in "${XFAIL[@]}"; do
        [ "${e%%|*}" = "$1" ] && { echo "${e#*|}"; return 0; }
    done
    return 1
}

# runs one invocation; prints "<status> <passed>/<total>" where status is ok, fail, norun or empty
run_one() {
    local log=$1 ops=$2 params=$3; shift 3
    local args=(test -b "$backend" -o "$ops")
    [ -n "$params" ] && args+=(-p "$params")
    # test-backend-ops colours its OK / FAIL; keep the log plain
    env "$@" "$tbo" "${args[@]}" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > "$log"
    local rc=${PIPESTATUS[0]}
    # `-b` with a name that matches no device exits 0, so prove the device ran
    if ! awk -v dev="$backend" '
            $0 ~ "^Backend [0-9]+/[0-9]+: " dev "$" { seen = 1; if ((getline nxt) > 0 && nxt !~ /^  Skipping/) ran = 1 }
            END { exit !(seen && ran) }' "$log"; then
        echo "norun 0/0"
        return
    fi
    local summary
    summary=$(grep -oE '[0-9]+/[0-9]+ tests passed' "$log" | tail -1 | cut -d' ' -f1)
    if [ -z "$summary" ] || [ "${summary#*/}" = 0 ]; then
        echo "empty ${summary:-0/0}"
    elif [ $rc -eq 0 ] && ! grep -qE '^  .*: FAIL' "$log"; then
        echo "ok $summary"
    else
        echo "fail $summary"
    fi
}

n_bad=0
printf '%-40s %-42s %-12s %s\n' "knob" "filter" "cases" "result"
for entry in "${KNOBS[@]}"; do
    IFS='|' read -r knob ops params extra <<< "$entry"
    if [ -n "$knob_re" ] && ! [[ $knob =~ $knob_re ]]; then
        continue
    fi
    if [ -n "$skip_re" ] && [[ $knob =~ $skip_re ]]; then
        continue
    fi
    filter="-o $ops${params:+ -p '$params'}"
    if ! grep -qaF "$knob" "$tbo"; then
        printf '%-40s %-42s %-12s %s\n' "$knob" "$filter" "-" "n/a (not in this binary)"
        continue
    fi
    log="$logdir/${knob#GGML_VK_TEST_REQUIRE_}.log"
    # shellcheck disable=SC2086
    read -r status counts < <(run_one "$log" "$ops" "$params" "$knob=1" $extra)
    result=$status
    reason=
    if reason=$(xfail_reason "$knob"); then
        case "$status" in
            fail) result="XFAIL ($reason)" ;;
            ok)   result="XPASS (listed as expected to fail: $reason)" ;;
            *)    result="FAIL ($status)"; n_bad=$((n_bad + 1)) ;;
        esac
    else
        case "$status" in
            ok) result=PASS ;;
            *)  result="FAIL ($status)"; n_bad=$((n_bad + 1)) ;;
        esac
    fi
    [ -n "$extra" ] && result="$result [$extra]"
    printf '%-40s %-42s %-12s %s\n' "$knob" "$filter" "$counts" "$result"
    if [[ $status = fail ]]; then
        grep -E '^  .*: FAIL' "$log" | head -5 | sed 's/^ */    /'
    fi
done

if [ $controls -eq 1 ]; then
    # census mode: the fused dispatches of whole graphs through the scheduler against the expected census
    log="$logdir/census.log"
    "$tbo" census -b "$backend" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > "$log"
    rc=${PIPESTATUS[0]}
    line=$(grep -E '^  CENSUS [0-9]+/[0-9]+ OK' "$log" | tail -1 | sed 's/^  CENSUS //')
    if grep -q 'CENSUS skipped' "$log" || [ -z "$line" ]; then
        result="FAIL (no census ran: $(grep -m1 'CENSUS skipped' "$log" | sed 's/^ *//'))"
        n_bad=$((n_bad + 1))
    elif [ $rc -ne 0 ] || grep -q ' MISMATCH' "$log"; then
        result="FAIL"
        n_bad=$((n_bad + 1))
    else
        result=PASS
    fi
    printf '%-40s %-42s %-12s %s\n' "census" "census" "${line%% *}" "$result"
    grep ' MISMATCH' "$log" | head -5 | sed 's/^ */    /'

    # negative control: the q5_1 streaming down turned off under its assert must fail
    log="$logdir/negative-control.log"
    if grep -qaF GGML_VK_TEST_REQUIRE_DOWN_STREAM "$tbo"; then
        read -r status counts < <(run_one "$log" MUL_MAT_ID_MUL_CPY16,MUL_MAT_ID_CPY16 "type_a=q5_1,type_b=f32,n_mats=16," \
            GGML_VK_TEST_REQUIRE_DOWN_STREAM=1 GGML_VK_MMID_DOWN_STREAM=0)
        if [ "$status" = fail ]; then
            result="PASS (failed as it must)"
        else
            result="FAIL ($status: the engagement asserts are not live)"
            n_bad=$((n_bad + 1))
        fi
    else
        counts=-
        result="n/a (not in this binary)"
    fi
    printf '%-40s %-42s %-12s %s\n' "negative control" "DOWN_STREAM=0 under its assert" "$counts" "$result"
fi

echo "logs: $logdir"
if [ $n_bad -ne 0 ]; then
    echo "$n_bad invocation(s) failed"
    exit 1
fi
echo "all engagement checks passed"
