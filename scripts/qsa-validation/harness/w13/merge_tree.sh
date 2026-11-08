#!/bin/bash
# git merge-tree of HEAD against master and the open PR heads named by PLAN-v3 §0 (no checkout, no merge):
# conflicted files, the git conflict messages, and per text conflict the region of HEAD's side (first line of each hunk)
cd ~/dev/wt_rewrite
echo "HEAD $(git rev-parse HEAD)"
for r in origin/master pr/234 pr/225 pr/247 pr/248 pr/230; do
  out=$(git merge-tree --write-tree --name-only HEAD $r 2>&1); rc=$?
  tree=$(echo "$out" | head -1)
  echo "== $r $(git rev-parse --short=10 $r) merge-base $(git merge-base HEAD $r | cut -c1-10) rc=$rc"
  echo "$out" | grep '^CONFLICT' | sed 's/^/   /'
  for f in $(echo "$out" | sed -n '2,/^$/p' | grep -v '^$'); do
    n=$(git show $tree:$f 2>/dev/null | grep -c '^<<<<<<<')
    echo "   file $f text-hunks=$n $(git show $tree:$f 2>/dev/null | grep -n '^<<<<<<<' | cut -d: -f1 | tr '\n' ' ')"
  done
done
