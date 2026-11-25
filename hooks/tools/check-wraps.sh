#!/usr/bin/env bash
# Checks that every internal link-time wrapped symbol is still called from another object file of upstream.
# Public API symbols (hooks_wrap API) are entry points for the users and are not checked.
# A call from the object file that defines the symbol is not wrapped: if upstream moves the callers there, or inlines
# the function, the wrapper stops running without a link error.
#
#   hooks/tools/check-wraps.sh <build dir>
#
# The list of wrapped symbols is written by cmake (hooks_wrap) to <build dir>/hooks-wraps.txt.
set -euo pipefail
export LC_ALL=C

build=${1:?usage: check-wraps.sh <build dir>}
list=$build/hooks-wraps.txt
[ -f "$list" ] || { echo "no $list: configure with -DCMAKE_PROJECT_INCLUDE=<repo>/hooks/hooks.cmake"; exit 1; }

# undefined symbols of the upstream objects (the objects of hooks/ files are excluded)
undef=$(mktemp)
trap 'rm -f "$undef"' EXIT
find "$build" -name '*.o' -not -path '*/hooks/*' -not -path '*hooks-registrar*' -print0 |
    xargs -0 -r nm -u --format=just-symbols 2>/dev/null | sort -u > "$undef"

fail=0
n=0
while read -r kind target sym; do
    [ "$kind" = internal ] || continue # api and rename: the link checks them
    n=$((n + 1))
    if ! grep -qxF "$sym" "$undef"; then
        echo "  NOT CALLED  $sym ($target): no upstream object calls it from another file, the wrapper never runs"
        fail=1
    fi
done < "$list"

echo "internal wrapped symbols: $n"
if [ "$fail" != 0 ]; then
    echo "FAIL"
    exit 1
fi
echo "OK"
