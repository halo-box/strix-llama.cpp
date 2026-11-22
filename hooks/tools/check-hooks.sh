#!/usr/bin/env bash
# Checks that the fork changes upstream files only at the hook points of hooks/HOOKS.txt.
#
#   hooks/tools/check-hooks.sh [base]
#
# base: upstream commit to compare with (default: git merge-base HEAD upstream/master)
#
# Each changed hunk outside hooks/ must hold HOOK(name) markers of HOOKS.txt entries for its file, and exactly the
# number of changed lines of those entries. Each entry must be found once. Compares the working tree.
# Only files that exist in base count: new files cannot conflict.
set -euo pipefail
export LC_ALL=C

cd "$(git rev-parse --show-toplevel)"

base=${1:-$(git merge-base HEAD upstream/master)}
manifest=hooks/HOOKS.txt

declare -A want_file want_lines seen
while read -r name file lines _; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    want_file[$name]=$file
    want_lines[$name]=$lines
done < "$manifest"

# the repository files whose fork version is active at their path (no build input): ACTIVE_FORK_PATHS of
# hooks/tools/import-fork-files.py
active_fork_paths=(README.md AGENTS.md CONTRIBUTING.md .editorconfig .github/ .agents/ .claude/ docs/ .devops/
                   scripts/hip/gcn-cdna-vgpr-check.py gguf-py/README.md tools/cli/README.md tools/completion/README.md tools/llama-bench/README.md tools/server/README.md ui/apps/chat/README.md tests/.gitignore)
is_active() {
    local a
    for a in "${active_fork_paths[@]}"; do
        [[ "$1" == "$a" || ( "$a" == */ && "$1" == "$a"* ) ]] && return 0
    done
    return 1
}

fail=0
total=0

while IFS= read -r file; do
    is_active "$file" && continue
    # one hunk per line: "<start> <changed lines> <names>"
    while read -r start n names; do
        total=$((total + n))
        if [ -z "$names" ]; then
            echo "  NOT A HOOK  $file:$start ($n lines)"
            fail=1
            continue
        fi
        expect=0
        for name in $names; do
            if [ -z "${want_file[$name]:-}" ]; then
                echo "  UNKNOWN     $file:$start HOOK($name): not in $manifest"
                fail=1
                continue
            fi
            if [ "${want_file[$name]}" != "$file" ]; then
                echo "  WRONG FILE  $file:$start HOOK($name): $manifest says ${want_file[$name]}"
                fail=1
            fi
            seen[$name]=$(( ${seen[$name]:-0} + 1 ))
            expect=$((expect + want_lines[$name]))
        done
        if [ "$n" -ne "$expect" ]; then
            echo "  SIZE        $file:$start: $n changed lines, $manifest says $expect for: $names"
            fail=1
        fi
    done < <(git diff -U0 "$base" -- "$file" | awk '
        function flush() {
            if (n > 0) { print start, n, names }
            n = 0; names = ""
        }
        /^@@/ { flush(); split($3, a, ","); start = substr(a[1], 2); next }
        /^\+\+\+|^---/ { next }
        /^[+-]/ {
            n++
            line = $0
            while (match(line, /HOOK\([a-z0-9-]+\)/)) {
                names = names " " substr(line, RSTART + 5, RLENGTH - 6)
                line = substr(line, RSTART + RLENGTH)
            }
        }
        END { flush() }
    ')
done < <(git diff --name-only --diff-filter=MDRT "$base" -- . ':!hooks')

for name in "${!want_file[@]}"; do
    if [ "${seen[$name]:-0}" -ne 1 ]; then
        echo "  MISSING     HOOK($name) found ${seen[$name]:-0} times in ${want_file[$name]}, expected once"
        fail=1
    fi
done

# new files outside hooks/ (docs, scripts of the fork): they cannot conflict with an upstream file that exists,
# only with one that upstream adds later at the same path
n_new=$(git diff --name-only --diff-filter=A "$base" -- . ':!hooks' | wc -l)
echo "hook points: ${#want_file[@]}, changed lines in upstream files: $total (base $(git rev-parse --short "$base"))"
echo "new files outside hooks/: $n_new" 
if [ "$fail" != 0 ]; then
    echo "FAIL: move the change to hooks/, or add the hook point to $manifest"
    exit 1
fi
echo "OK"
