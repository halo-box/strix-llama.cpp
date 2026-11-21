#!/usr/bin/env python3
"""Pick the test-backend-ops perf cases that bench-gfx1151-ops-perf runs besides MUL_MAT and MUL_MAT_ID.

Reads the full perf case list, one `OP(params)` string per line, and keeps an evenly spaced sample of
each op's cases (first, last and evenly in between): at least 3 per op, or 1 in 20 for large ops.
MUL_MAT and MUL_MAT_ID are left out; the workflow selects them for every type with -p patterns.

usage: select_perf_cases.py <case list> > perf-cases.txt
"""

import math
import re
import sys

CASE = re.compile(r'^([A-Z0-9_]+)\((.*)\)$')
SKIP = {'MUL_MAT', 'MUL_MAT_ID'}


def main() -> int:
    by_op: dict[str, list[str]] = {}
    seen = set()
    with open(sys.argv[1]) as f:
        for line in f:
            line = line.strip()
            m = CASE.match(line)
            if not m or m.group(1) in SKIP or line in seen:
                continue
            seen.add(line)
            by_op.setdefault(m.group(1), []).append(line)
    out = sys.stdout
    out.write('# test-backend-ops perf cases run by .github/workflows/bench-gfx1151-ops-perf.yml besides the\n'
              '# all-types MUL_MAT and MUL_MAT_ID cases; written by scripts/bench-history/select_perf_cases.py\n')
    for op, cases in by_op.items():
        k = min(len(cases), max(3, math.ceil(len(cases) / 20)))
        picks = sorted({round(j * (len(cases) - 1) / max(1, k - 1)) for j in range(k)})
        for i in picks:
            out.write(cases[i] + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
