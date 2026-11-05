#!/bin/bash
# The test-backend-ops perf subset of bench-gfx1151-ops-perf.yml on one device, appended to one log:
# MUL_MAT and MUL_MAT_ID for every type (selected with -p, so a new type is picked up without an
# edit), then the fixed case list for the other ops. The full suite takes more than 150 min per
# device on gfx1151, mostly in the CPU-side weight setup of the trellis MUL_MAT_ID cases.
#
# usage: ops-perf.sh <test-backend-ops> <device> <log> <case list>
# Writes command lines to commands.tsv and results to results.tsv next to the log.
#
# A case that fails allocation or compute prints no `OP(params):` line, only "failed to allocate
# tensors" or "graph_compute failed", and test-backend-ops still exits 0. Such failures are counted.

set -uo pipefail
bin=$1
dev=$2
log=$3
list=$4
logs=$(dirname "$log")

mul_mat='^type_a=[A-Za-z0-9_]+,type_b=f32,m=4096,n=(1|8|512),k=14336,bs=\[1,1\],'
mul_mat_id='^type_a=[A-Za-z0-9_]+,type_b=f32,n_mats=32,n_used=8,b=0,m=512,n=(1|8|512),k=2048,'

# shell-quote for the record; single quotes keep the -p patterns readable
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
rows() { grep -c 'us/run' "$log"; }
cases() { grep -cE '^  [A-Za-z0-9_]+\(.*\): ' "$log"; }

: > "$log"
rc_all=0
group_rc=0

# run_group <label> <command as recorded> <test-backend-ops args>...
run_group() {
    local label=$1 shown=$2 before rc
    shift 2
    before=$(rows)
    printf 'ops-perf-%s-%s\t%s\n' "$dev" "$label" "$shown" >> "$logs/commands.tsv"
    echo "=== test-backend-ops perf $dev $label ==="
    "$bin" perf -b "$dev" "$@" 2>&1 | tee -a "$log"
    rc=${PIPESTATUS[0]}
    printf 'ops-perf-group\t%s\t%s\t%s\t%s\n' "$dev" "$label" "$rc" "$(( $(rows) - before ))" >> "$logs/results.tsv"
    echo "$dev $label: exit $rc, $(( $(rows) - before )) rows"
    group_rc=$rc
    [ "$rc_all" -ne 0 ] || rc_all=$rc
}

run_group mul_mat "$(quote "$bin" perf -b "$dev" -o MUL_MAT -p "$mul_mat")" -o MUL_MAT -p "$mul_mat"

before=$(cases)
run_group mul_mat_id "$(quote "$bin" perf -b "$dev" -o MUL_MAT_ID -p "$mul_mat_id")" -o MUL_MAT_ID -p "$mul_mat_id"
if [ "$group_rc" -eq 0 ] && [ "$(cases)" -eq "$before" ]; then
    echo "::warning::$dev: this build has no all-types MUL_MAT_ID perf cases (n_mats=32, m=512, k=2048)"
fi

listed=$(grep -c -v -e '^#' -e '^$' "$list")
lines=$(wc -l < "$log")
run_group list "$(quote "$bin" perf -b "$dev") -o \"\$(grep -v -e '^#' -e '^\$' $(quote "$list") | paste -sd, -)\"" \
    -o "$(grep -v -e '^#' -e '^$' "$list" | paste -sd, -)"
failures=$(grep -cE 'failed to allocate|graph_compute failed' "$log")
# distinct names: a few cases appear twice in the perf list and run twice
seen=$(tail -n +"$(( lines + 1 ))" "$log" | grep -E '^  [A-Za-z0-9_]+\(.*\): ' | sed -E 's/\): .*$/)/' | sort -u | wc -l)
if [ "$group_rc" -eq 0 ] && [ "$seen" -lt "$listed" ]; then
    echo "::warning::$dev: $(( listed - seen )) of $listed listed perf cases printed no result (not in this build, or among the $failures failed cases)"
fi
if [ "$failures" -gt 0 ]; then
    echo "::warning::$dev: $failures perf cases failed to allocate or compute"
fi

# `-b` with a name that matches no device exits 0, so prove the device ran
final=$rc_all
total=$(rows)
if ! awk -v dev="$dev" '
    $0 ~ "^Backend [0-9]+/[0-9]+: " dev "$" {
        seen = 1
        if ((getline nxt) > 0 && nxt !~ /^  Skipping/) ran = 1
    }
    END { exit !(seen && ran) }
' "$log"; then
    echo "::error::$dev never ran" >&2
    final=1
elif [ "$total" -eq 0 ]; then
    echo "::error::no perf rows on $dev" >&2
    final=1
fi
printf 'ops-perf\t%s\t%s\t%s\t%s\n' "$dev" "$final" "$total" "$failures" >> "$logs/results.tsv"
exit "$final"
