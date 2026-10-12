#!/usr/bin/env bash
# Helper for the upstream-sync skill (skills/upstream-sync/SKILL.md, SYNC.md).
#   sync.sh report [target]        the step plan (one step per conflicting upstream PR), the models to test, the next step in detail
#   sync.sh merge  [target]        start the merge of target on the current branch, from the content base
#   sync.sh verify <old> [target]  after the merge commit: did every fork change survive?
#   sync.sh models <old> [target]  model archs to validate for a sync PR started at <old>, up to target (default HEAD^2)
#   sync.sh summary <old> [branch] markdown for the sync PR: every step, its upstream PRs, conflicts, conflict zones, resolutions and edits outside conflicts
# target defaults to the next step: the first conflicting upstream commit, or at most MAX_STEP commits.
set -euo pipefail
export LC_ALL=C

cd "$(git rev-parse --show-toplevel)"

UPSTREAM=${UPSTREAM:-upstream/master}
MAX_STEP=${MAX_STEP:-100}
# conflicts in these paths need a run of the validation checklist on a Strix Halo machine
SENSITIVE=${SENSITIVE:-'^(ggml/src/ggml-cuda/|ggml/src/ggml-vulkan/|ggml/src/ggml-backend|ggml/src/ggml-alloc|src/models/|src/llama-graph|src/llama-context|src/llama-kv-cache|src/llama-memory|src/llama-arch|src/llama-model|src/llama-vocab|conversion/|gguf-py/gguf/constants)'}
# a change in these paths can affect every model
SHARED=${SHARED:-'^(ggml/|src/llama-(graph|context|kv-cache|memory|model|arch|vocab|batch|adapter|cparams|hparams))'}
BASE_FILE=scripts/upstream-sync.base

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

die() { echo "error: $*" >&2; exit 1; }

fetch() {
    git remote get-url upstream >/dev/null 2>&1 || die "no 'upstream' remote (git remote add upstream https://github.com/ggml-org/llama.cpp)"
    git fetch --quiet --no-tags upstream
}

# Upstream commit whose content is in <rev>. After a reverted sync, git merge-base is newer than this.
content_base() {
    local b
    b=$(git show "$1:$BASE_FILE" 2>/dev/null | tr -d '[:space:]') || true
    if [ -z "$b" ]; then
        git merge-base "$1" "$UPSTREAM"
        return
    fi
    git rev-parse --verify --quiet "$b^{commit}" || die "$BASE_FILE in $1 names an unknown commit: $b"
}

# exit 0 if <theirs> merges cleanly into <ours> from <base>, 1 if it conflicts; result in $TMP/mt
try_merge() {
    local rc=0
    git merge-tree --write-tree --name-only --merge-base="$1" "$2" "$3" > "$TMP/mt" 2>/dev/null || rc=$?
    [ "$rc" -le 1 ] || die "git merge-tree failed ($rc)"
    return "$rc"
}

conflict_files() { sed -n '2,/^$/p' "$TMP/mt" | sed '/^$/d'; }

# First conflicting upstream commit (bisected on first-parent history), capped at MAX_STEP commits.
next_target() {
    local base=$1 ours=$2 lo=-1 hi mid
    local -a c
    mapfile -t c < <(git rev-list --first-parent --reverse "$base..$UPSTREAM")
    [ "${#c[@]}" -gt 0 ] || die "nothing to merge"
    # hi = index of the first conflicting commit, or the count if the tip merges cleanly
    hi=${#c[@]}
    if ! try_merge "$base" "$ours" "${c[-1]}"; then
        hi=$((${#c[@]} - 1))
        while [ $((hi - lo)) -gt 1 ]; do
            mid=$(((lo + hi) / 2))
            if try_merge "$base" "$ours" "${c[$mid]}"; then lo=$mid; else hi=$mid; fi
        done
    fi
    if [ "$hi" -ge "$MAX_STEP" ]; then
        echo "${c[$((MAX_STEP - 1))]}"
    elif [ "$hi" -eq "${#c[@]}" ]; then
        echo "${c[-1]}"
    else
        echo "${c[$hi]}"
    fi
}

resolve_target() {  # <base> <ours> [target]
    local t
    if [ -n "${3:-}" ]; then t=$3; else t=$(next_target "$1" "$2") || exit 1; fi
    t=$(git rev-parse --verify --quiet "$t^{commit}") || die "unknown target: $3"
    git merge-base --is-ancestor "$1" "$t" || die "content base $(git rev-parse --short "$1") is not an ancestor of $t"
    echo "$t"
}

oneline() { git log -1 --format='%h %cs %s' "$1"; }

# One step per upstream commit that adds a conflicted file, plus a cut every MAX_STEP clean commits.
# Computed against the fork as it is now, so later steps can shift once earlier ones are resolved.
plan() {
    local base=$1 ours=$2 c n=0 last=0 steps=0 sens=0 prev="" cur new mark
    printf '%-4s %-7s %-35s %s\n' step commits "ends at" "new conflicts (* = sensitive)"
    for c in $(git rev-list --first-parent --reverse "$base..$UPSTREAM"); do
        n=$((n + 1))
        if try_merge "$base" "$ours" "$c"; then cur=""; else cur=$(conflict_files | sort); fi
        new=$(comm -13 <(echo "$prev" | sed '/^$/d') <(echo "$cur" | sed '/^$/d') | paste -sd, -)
        if [ -n "$new" ] || [ $((n - last)) -ge "$MAX_STEP" ]; then
            steps=$((steps + 1))
            mark=""
            if echo "$new" | tr ',' '\n' | grep -qE "$SENSITIVE"; then mark="*"; sens=$((sens + 1)); fi
            printf '%-4s %-7s %-35s %s%s\n' "$steps" "$((n - last))" "$(git log -1 --format='%h %cs' "$c") $(git log -1 --format=%s "$c" | grep -oE '#[0-9]+\)$' | tr -d ')')" "$mark" "${new:-(none)}"
            last=$n
        fi
        prev=$cur
    done
    if [ "$n" -gt "$last" ]; then
        steps=$((steps + 1))
        printf '%-4s %-7s %-35s %s\n' "$steps" "$((n - last))" "$(git log -1 --format='%h %cs' "$c")" "(none)"
    fi
    echo "$steps steps, $sens with sensitive conflicts"
}

# Model archs to test: src/models/<arch>.cpp changed by upstream in the range and by the fork since the base.
# <fork>: the fork before the sync (default HEAD); after merges, HEAD already holds upstream changes.
# An arch only upstream changed comes in as upstream wrote it.
models() {  # <base> <target> [fork]
    local up fork both
    up=$(git diff --name-only "$1" "$2" -- src/models/ | sed -n 's#^src/models/\(.*\)\.cpp$#\1#p' | sort -u)
    fork=$(git diff --name-only "$1" "${3:-HEAD}" -- src/models/ | sed -n 's#^src/models/\(.*\)\.cpp$#\1#p' | sort -u)
    both=$(comm -12 <(echo "$up") <(echo "$fork") | sed '/^$/d')
    echo "  changed by upstream, checked with test-llama-archs: $(echo "$up" | grep -c .) archs"
    echo "$up" | paste -sd' ' | fold -s -w 110 | sed 's/^/    /'
    echo "  changed on both sides, checked with a real GGUF:"
    if [ -n "$both" ]; then
        echo "$both" | sed 's/^/    /'
    else
        echo "    none"
    fi
    if git diff --name-only "$1" "$2" | grep -qE "$SHARED"; then
        echo "  shared paths changed too (kernels, ggml core, graph, KV cache): ask the user which real models to add"
    fi
}

report() {
    fetch
    local ours base hist target left
    ours=$(git rev-parse HEAD)
    base=$(content_base HEAD)
    hist=$(git merge-base HEAD "$UPSTREAM")

    echo "== divergence"
    echo "HEAD:         $(oneline HEAD)"
    echo "upstream:     $(oneline "$UPSTREAM")"
    echo "content base: $(oneline "$base")"
    if [ "$hist" != "$base" ]; then
        echo "history base: $(oneline "$hist")"
        echo "warning: the bases differ (a sync was reverted?). A plain 'git merge' would skip $(git rev-list --count --first-parent "$base..$hist") upstream commits. Use 'sync.sh merge'."
    fi
    left=$(git rev-list --count --first-parent "$base..$UPSTREAM")
    echo "upstream commits not in the fork: $left"
    if [ "$left" -eq 0 ]; then
        echo "nothing to merge"
        return 0
    fi

    echo
    echo "== plan"
    plan "$base" "$ours"

    echo
    echo "== models to test, upstream $(git rev-parse --short "$base")..$(git rev-parse --short "${1:-$UPSTREAM}")"
    models "$base" "$(git rev-parse "${1:-$UPSTREAM}")"

    target=$(resolve_target "$base" "$ours" "${1:-}")
    echo
    echo "== next step"
    echo "target: $(oneline "$target")"
    echo "upstream commits in this step: $(git rev-list --count --first-parent "$base..$target"), left after it: $(git rev-list --count --first-parent "$target..$UPSTREAM")"
    if try_merge "$base" "$ours" "$target"; then
        echo "merges cleanly"
        return 0
    fi

    echo
    echo "== conflicts: fork commits that own our side, upstream commits on theirs"
    conflict_files | while IFS= read -r f; do
        echo "-- $f"
        echo "   fork:"
        git log -n 15 --no-merges --format='     %h %cs %s' "$ours" ^"$UPSTREAM" ^"$base" -- "$f"
        echo "   upstream:"
        git log -n 15 --format='     %h %cs %s' "$base..$target" -- "$f"
    done
}

merge() {
    [ -z "$(git status --porcelain --untracked-files=no)" ] || die "working tree has uncommitted changes"
    [ ! -e "$(git rev-parse --git-path MERGE_HEAD)" ] || die "a merge is already in progress"
    [ "$(git rev-parse --abbrev-ref HEAD)" != master ] || die "run on a sync branch: git switch -c sync/upstream-$(date +%F)-<upstream sha> master"
    fetch

    local ours base hist target rc=0 zero n
    ours=$(git rev-parse HEAD)
    base=$(content_base HEAD)
    hist=$(git merge-base HEAD "$UPSTREAM")
    target=$(resolve_target "$base" "$ours" "${1:-}")
    n=$(git rev-list --count --first-parent "$base..$target")

    if [ "$hist" = "$base" ]; then
        git merge --no-ff --no-commit "$target" || rc=$?
    else
        # git merge cannot take a merge base, so build the same merge state from merge-tree
        git merge-tree --write-tree --merge-base="$base" HEAD "$target" > "$TMP/mt" || rc=$?
        [ "$rc" -le 1 ] || die "git merge-tree failed ($rc)"
        git read-tree -u -m HEAD "$(head -1 "$TMP/mt")"
        # conflicted entries are "<mode> <object> <stage>\t<path>"; put them in the index as unmerged stages
        sed -n '2,/^$/p' "$TMP/mt" | sed '/^$/d' > "$TMP/stages"
        if [ -s "$TMP/stages" ]; then
            zero=$(git hash-object --stdin < /dev/null | tr '0-9a-f' '0')
            cut -f2- "$TMP/stages" | sort -u | sed "s/^/0 $zero\t/" > "$TMP/remove"
            cat "$TMP/remove" "$TMP/stages" | git update-index --index-info
        fi
        echo "$target" > "$(git rev-parse --git-path MERGE_HEAD)"
        # exactly "no-ff", no newline: otherwise git commit drops a parent that is already an ancestor
        printf 'no-ff' > "$(git rev-parse --git-path MERGE_MODE)"
        git rerere >/dev/null 2>&1 || true
    fi

    echo "$target" > "$BASE_FILE"
    git add "$BASE_FILE"

    {
        echo "Merge upstream llama.cpp up to $(git rev-parse --short=9 "$target") ($(git log -1 --format=%cs "$target"))"
        echo
        echo "Upstream: ggml-org/llama.cpp $(git rev-parse --short=9 "$base")..$(git rev-parse --short=9 "$target") ($n commits)"
        if [ "$rc" -ne 0 ]; then
            echo
            echo "Conflicts:"
            git diff --name-only --diff-filter=U | sed 's/$/: <how it was resolved>/; s/^/- /'
        fi
    } > "$(git rev-parse --git-path MERGE_MSG)"

    echo "merging $n upstream commits up to $(oneline "$target")"
    if [ "$rc" -ne 0 ]; then
        echo "conflicts to resolve:"
        git diff --name-only --diff-filter=U | sed 's/^/  /'
        if git diff --name-only --diff-filter=U | grep -qE "$SENSITIVE"; then
            echo "sensitive: run the validation checklist on a Strix Halo machine before this sync PR leaves draft"
        fi
    else
        echo "no conflicts"
    fi
    echo "then: git add <files> && git commit (edit the message: one line per conflict), and run: $0 verify $ours"
}

verify() {
    local old=${1:?usage: sync.sh verify <old-head> [target]} target old_base before after markers
    target=${2:-$(git show "HEAD:$BASE_FILE" 2>/dev/null | tr -d '[:space:]')}
    target=$(git rev-parse --verify --quiet "$target^{commit}") || die "no target: pass it, or commit $BASE_FILE"
    # after a reverted sync the target can already be an ancestor, so check the merge parent itself
    [ "$(git rev-parse --verify --quiet HEAD^2)" = "$target" ] || die "HEAD is not a merge of $(git rev-parse --short "$target") (second parent: $(git rev-parse --short --verify --quiet HEAD^2 || echo none))"
    [ "$(git show "HEAD:$BASE_FILE" 2>/dev/null | tr -d '[:space:]')" = "$target" ] || echo "warning: $BASE_FILE in HEAD is not $target"
    old_base=$(content_base "$old")

    echo "== fork delta before (vs $(git rev-parse --short "$old_base")) and after (vs $(git rev-parse --short "$target"))"
    before=$(git diff --numstat "$old_base" "$old" -- . ":!$BASE_FILE" | awk '{print $3"\t+"$1" -"$2}' | sort)
    after=$(git diff --numstat "$target" HEAD -- . ":!$BASE_FILE" | awk '{print $3"\t+"$1" -"$2}' | sort)

    echo "-- files no longer different from upstream (each must be explained: upstream took the change, or it was lost in a conflict)"
    comm -23 <(cut -f1 <<< "$before") <(cut -f1 <<< "$after") | sed 's/^/  /' | { grep . || echo "  (none)"; }
    echo "-- files newly different from upstream (should only be conflict fixups)"
    comm -13 <(cut -f1 <<< "$before") <(cut -f1 <<< "$after") | sed 's/^/  /' | { grep . || echo "  (none)"; }
    echo "-- files whose fork diff size changed (inspect: git diff $(git rev-parse --short "$target") HEAD -- <file>)"
    join -t $'\t' <(echo "$before") <(echo "$after") | awk -F'\t' '$2 != $3 {printf "  %-60s before %s   after %s\n", $1, $2, $3}' | { grep . || echo "  (none)"; }

    echo
    echo "== conflict markers left in files changed by the merge"
    markers=$(git diff --name-only "$old" HEAD | xargs -r git grep -n -E '^(<<<<<<<|>>>>>>>)( |$)' HEAD -- 2>/dev/null || true)
    if [ -n "$markers" ]; then
        echo "$markers" | sed 's/^/  /'
        return 1
    fi
    echo "  (none)"
}

# Conflict zones of a file with conflict markers: "start-end" line ranges, one per zone.
zones() { awk '/^<<<<<<< /{s=NR} /^>>>>>>> /{if (s) print s "-" NR; s=0}' "$1"; }

summary() {
    local old=${1:?usage: sync.sh summary <old-head> [branch]} br=${2:-HEAD} m n=0 p1 p2 base auto files f z first last total edits
    local -a merges
    mapfile -t merges < <(git rev-list --first-parent --reverse --merges "$old..$br")
    [ "${#merges[@]}" -gt 0 ] || die "no merge commit in $old..$br"
    first=$(content_base "$old")
    last=$(git rev-parse "${merges[-1]}^2")
    total=$(git rev-list --count --first-parent "$first..$last")

    echo "## Overview"
    echo
    echo "Merge upstream ggml-org/llama.cpp $(git rev-parse --short=9 "$first")..$(git rev-parse --short=9 "$last") ($(git log -1 --format=%cs "$first") to $(git log -1 --format=%cs "$last")): $total upstream commits in ${#merges[@]} steps."
    echo
    echo "| Step | Ends at | Upstream commits | Conflicts (* = sensitive) | Edits outside conflicts |"
    echo "|---|---|---|---|---|"
    for m in "${merges[@]}"; do
        n=$((n + 1)); p1=$(git rev-parse "$m^1"); p2=$(git rev-parse "$m^2"); base=$(content_base "$p1")
        try_merge "$base" "$p1" "$p2" || true
        files=$(conflict_files | while IFS= read -r f; do if echo "$f" | grep -qE "$SENSITIVE"; then echo "\`$f\`*"; else echo "\`$f\`"; fi; done | paste -sd' ' -)
        auto=$(head -1 "$TMP/mt")
        edits=$(git diff --name-only "$auto" "$m" -- . ":!$BASE_FILE" | { grep -vxF -f <(conflict_files; echo "$BASE_FILE") || true; } | sed 's/.*/`&`/' | paste -sd' ' -)
        printf '| %s | %s %s | %s | %s | %s |\n' "$n" "$(git rev-parse --short=9 "$p2")" "$(git log -1 --format=%s "$p2" | grep -oE '\(#[0-9]+\)$' | tr -d '()')" \
            "$(git rev-list --count --first-parent "$base..$p2")" "${files:-none}" "${edits:-none}"
    done
    echo
    echo "Edits outside conflicts are changes the merge commit makes that neither the automatic merge nor a conflict asked for. Review each one."
    echo

    echo "## Steps"
    n=0
    for m in "${merges[@]}"; do
        n=$((n + 1)); p1=$(git rev-parse "$m^1"); p2=$(git rev-parse "$m^2"); base=$(content_base "$p1")
        echo
        echo "### Step $n: up to $(git log -1 --format='%h %s' "$p2")"
        echo
        if try_merge "$base" "$p1" "$p2"; then echo "No conflict."; fi
        auto=$(head -1 "$TMP/mt")
        conflict_files > "$TMP/cf"
        while IFS= read -r f; do
            echo
            echo "**\`$f\`**$(echo "$f" | grep -qE "$SENSITIVE" && echo ' (sensitive)')"
            git log -1 --format=%B "$m" | sed -n "s#^- $f: ##p" | sed 's/^/Resolution: /'
            git show "$auto:$f" > "$TMP/conflicted" 2>/dev/null || : > "$TMP/conflicted"
            zones "$TMP/conflicted" > "$TMP/zones"
            while IFS= read -r z; do
                echo "- conflict zone at lines $z of the merged file (fork side $(git log -1 --format=%h "$p1"), upstream side $(git log -1 --format=%h "$p2"))"
            done < "$TMP/zones"
            # hunks of the resolution that touch no conflict zone: manual edits that no marker asked for
            git diff -U0 "$auto" "$m" -- "$f" | sed -n 's/^@@ -\([0-9]*\)\(,\([0-9]*\)\)\? .*/\1 \3/p' | while read -r a b; do
                [ -n "$b" ] || b=1
                e=$((a + (b > 0 ? b - 1 : 0)))
                if ! awk -F- -v a="$a" -v e="$e" '$1 <= e && $2 >= a {hit=1} END {exit !hit}' "$TMP/zones"; then
                    echo "- **edit outside the conflict zones** at lines $a-$e of the merged file"
                fi
            done
            echo
            echo "<details><summary>Resolution diff (automatic merge with markers -> committed)</summary>"
            echo
            echo '```diff'
            git diff "$auto" "$m" -- "$f"
            echo '```'
            echo "</details>"
        done < "$TMP/cf"
        edits=$(git diff --name-only "$auto" "$m" -- . ":!$BASE_FILE" | grep -vxF -f <(cat "$TMP/cf"; echo "$BASE_FILE") || true)
        if [ -n "$edits" ]; then
            echo
            echo "**Edits outside conflicts:**"
            echo
            echo '```diff'
            git diff "$auto" "$m" -- $edits
            echo '```'
        fi
        echo
        echo "<details><summary>Upstream commits in this step ($(git rev-list --count --first-parent "$base..$p2"))</summary>"
        echo
        git log --first-parent --format='- %h %s' "$base..$p2"
        echo
        echo "</details>"
    done

    echo
    echo "## All changes"
    echo
    echo "<details><summary>Diffstat against the fork before the sync ($(git rev-parse --short "$old"))</summary>"
    echo
    echo '```'
    git diff --stat=120 "$old" "$br" | tail -40
    echo '```'
    echo "</details>"
}

case "${1:-}" in
    report) shift; report "$@" ;;
    merge)  shift; merge "$@" ;;
    verify) shift; verify "$@" ;;
    summary) shift; summary "$@" ;;
    models) shift; models "$(content_base "$1")" "$(git rev-parse "${2:-HEAD^2}")" "$1" ;;
    *) echo "usage: $0 report [target] | merge [target] | verify <old-head> [target] | summary <old-head> [branch] | models <old-head> [target]" >&2; exit 2 ;;
esac
