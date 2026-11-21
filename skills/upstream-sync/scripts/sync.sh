#!/usr/bin/env bash
# Helper for the upstream-sync skill (skills/upstream-sync/SKILL.md, SYNC.md).
#   sync.sh report [target]        the step plan (one step per conflicting upstream PR), the models to test, the next step in detail
#   sync.sh merge  [target]        start the merge of target on the current branch, from the content base
#   sync.sh verify <old> [target]  after the merge commit: did every fork change survive?
#   sync.sh models <old> [target]  table of every model arch a sync PR started at <old> concerns, up to target (default: the last sync merge)
#   sync.sh owners <old> [target]  where upstream changes the fork's own code, with the fork commits that own it (git blame)
#   sync.sh summary <old> [branch] markdown for the sync PR: every step, its upstream PRs, conflicts, conflict zones, resolutions and edits outside conflicts
# target defaults to the next step: the first conflicting upstream commit, or at most MAX_STEP commits.
set -euo pipefail
export LC_ALL=C

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$(git rev-parse --show-toplevel)"
# paths with spaces or non-ASCII characters come out as they are, not quoted
git() { command git -c core.quotePath=false "$@"; }

UPSTREAM=${UPSTREAM:-upstream/master}
MAX_STEP=${MAX_STEP:-100}
# conflicts in these paths need a run of the validation checklist on a Strix Halo machine
# (kernels, ggml core and the fork's quant formats, the llama library, MTP, conversion)
SENSITIVE=${SENSITIVE:-'^(ggml/include/|ggml/src/ggml\.c|ggml/src/ggml-(quants|common|impl|backend|alloc)|ggml/src/ggml-(cuda|vulkan|hip|cpu)/|ggml/rocmfp|include/llama\.h|src/models/|src/llama-|common/speculative|conversion/|gguf-py/gguf/(constants|quants))'}
# a change in these paths can affect every model
SHARED=${SHARED:-'^(ggml/|src/llama-(graph|context|kv-cache|memory|model|arch|vocab|batch|adapter|cparams|hparams))'}
BASE_FILE=scripts/upstream-sync.base

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

die() { echo "error: $*" >&2; exit 1; }

fetch() {
    git remote get-url upstream >/dev/null 2>&1 || die "no 'upstream' remote (git remote add upstream https://github.com/ggml-org/llama.cpp)"
    git fetch --quiet --no-tags upstream || die "cannot fetch upstream"
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

# First conflicting upstream commit on first-parent history, capped at MAX_STEP commits.
# A linear scan, not a bisection: a later commit can merge cleanly again (upstream reverts), which would hide the first conflict.
next_target() {
    local base=$1 ours=$2 i n
    local -a c
    mapfile -t c < <(git rev-list --first-parent --reverse "$base..$UPSTREAM")
    [ "${#c[@]}" -gt 0 ] || die "nothing to merge"
    n=${#c[@]}
    [ "$n" -le "$MAX_STEP" ] || n=$MAX_STEP
    for ((i = 0; i < n; i++)); do
        try_merge "$base" "$ours" "${c[$i]}" || { echo "${c[$i]}"; return 0; }
    done
    echo "${c[$((n - 1))]}"
}

resolve_target() {  # <base> <ours> [target]
    local t
    if [ -n "${3:-}" ]; then t=$3; else t=$(next_target "$1" "$2") || exit 1; fi
    t=$(git rev-parse --verify --quiet "$t^{commit}") || die "unknown target: $t"
    git merge-base --is-ancestor "$1" "$t" || die "content base $(git rev-parse --short "$1") is not an ancestor of $t"
    echo "$t"
}

oneline() { git log -1 --format='%h %cs %s' "$1"; }

# One step per conflicting upstream commit, plus a cut every MAX_STEP clean commits: the same cut as `merge`.
# Each step is simulated as resolved by keeping the fork side of its conflicts (merge-tree -X ours, objects only, no ref
# changes), so a later upstream commit that changes the same code again is a step of its own, as in the real sync.
# Real resolutions differ from that, so later steps can still shift a little once earlier ones are resolved.
plan() {
    local base=$1 ours=$2 c n=0 last=0 steps=0 sens=0 cur mark tree
    printf '%-4s %-7s %-35s %s\n' step commits "ends at" "conflicts (* = sensitive)"
    for c in $(git rev-list --first-parent --reverse "$base..$UPSTREAM"); do
        n=$((n + 1))
        if try_merge "$base" "$ours" "$c"; then cur=""; else cur=$(conflict_files | sort | paste -sd, -); fi
        [ -n "$cur" ] || [ $((n - last)) -ge "$MAX_STEP" ] || continue
        steps=$((steps + 1))
        mark=""
        if echo "$cur" | tr ',' '\n' | grep -E "$SENSITIVE" > /dev/null; then mark="*"; sens=$((sens + 1)); fi
        printf '%-4s %-7s %-35s %s%s\n' "$steps" "$((n - last))" "$(git log -1 --format='%h %cs' "$c") $(git log -1 --format=%s "$c" | { grep -oE '#[0-9]+\)$' || true; } | tr -d ')')" "$mark" "${cur:-(none)}"
        # the fork after this step: upstream up to <c>, the fork side kept in the conflicts
        # exit 1 is fine: -X ours cannot settle a file deleted on one side, the tree keeps the fork's copy
        tree=$({ git merge-tree --write-tree -X ours --merge-base="$base" "$ours" "$c" || [ $? = 1 ] || die "git merge-tree failed"; } | head -1)
        # a fixed identity: these objects are never referenced, and a CI runner has no git identity
        ours=$(git -c user.name=upstream-sync-plan -c user.email=plan@upstream-sync.invalid commit-tree "$tree" -p "$ours" -p "$c" -m plan)
        base=$c last=$n
    done
    if [ "$n" -gt "$last" ]; then
        steps=$((steps + 1))
        printf '%-4s %-7s %-35s %s\n' "$steps" "$((n - last))" "$(git log -1 --format='%h %cs' "$c") $(git log -1 --format=%s "$c" | { grep -oE '#[0-9]+\)$' || true; } | tr -d ')')" "(none)"
    fi
    echo "$steps steps, $sens with sensitive conflicts"
}

# Every model arch the sync concerns, as a markdown table for the PR: archs whose model code upstream changes (in
# src/models/, a base file or models.h included: model_archs.py), and, when shared paths change, every arch the fork
# changes or tests in its code (LLM_ARCH_...). A changed model file that maps to no arch gets a row of its own.
# <fork>: the fork before the sync (default HEAD); after merges, HEAD already holds upstream changes.
models() {  # <base> <target> [fork]
    local fork=${3:-HEAD} upr up forkm gates shared=no a why
    upr=$(python3 "$HERE/model_archs.py" "$1" "$2")
    up=$(cut -f1 <<< "$upr" | { grep -vxF -e - -e '' || true; } | sort -u)
    forkm=$(python3 "$HERE/model_archs.py" "$1" "$fork" | cut -f1 | { grep -vxF -e - -e '' || true; } | sort -u)
    gates=$(git diff "$1" "$fork" -- src ggml common tools | { grep -E '^\+' || true; } | { grep -oE 'LLM_ARCH_[A-Z0-9_]+' || true; } | sort -u | \
        while read -r e; do git show "$fork:src/llama-arch.cpp" | sed -n "s/.*{ *$e *, *\"\([^\"]*\)\".*/\1/p"; done | sort -u)
    if git diff --name-only "$1" "$2" | grep -E "$SHARED" > /dev/null; then shared=yes; fi
    echo "| Arch | Why it is concerned | Validation |"
    echo "|---|---|---|"
    for a in $({ echo "$up"; if [ $shared = yes ]; then echo "$forkm"; echo "$gates"; fi; } | sed '/^$/d' | sort -u); do
        why=""
        if grep -xF "$a" <<< "$up" > /dev/null; then why="upstream changes $(awk -F'\t' -v a="$a" '$1 == a {print $2}' <<< "$upr" | paste -sd, - | sed 's/,/, /g')"; fi
        if [ $shared = yes ] && grep -xF "$a" <<< "$forkm" > /dev/null; then why="${why:+$why, }the fork changes it and shared paths changed"; fi
        if [ $shared = yes ] && grep -xF "$a" <<< "$gates" > /dev/null && ! grep -xF "$a" <<< "$forkm" > /dev/null; then why="${why:+$why, }fork code tests it and shared paths changed"; fi
        echo "| $a | $why | <real GGUF (see Models to validate), or test-llama-archs only and why> |"
    done
    awk -F'\t' '$1 == "-" {print $2}' <<< "$upr" | while read -r f; do
        echo "| $(basename "$f" .cpp) | upstream changes $f, which defines no model class: check which archs use it | <real GGUF (see Models to validate), or test-llama-archs only and why> |"
    done
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
    echo "== models concerned, upstream $(git rev-parse --short "$base")..$(git rev-parse --short "${1:-$UPSTREAM}")"
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
        # a failure without a merge in progress is not a conflict: git refused to start
        [ "$rc" -eq 0 ] || [ -e "$(git rev-parse --git-path MERGE_HEAD)" ] || die "git merge failed ($rc) without starting a merge"
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

    # only while the file exists: once the content base passes the reverted sync it is deleted for good
    if git cat-file -e "HEAD:$BASE_FILE" 2> /dev/null; then
        echo "$target" > "$BASE_FILE"
        git add "$BASE_FILE"
    fi

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
        if git diff --name-only --diff-filter=U | grep -E "$SENSITIVE" > /dev/null; then
            echo "sensitive: run the validation checklist on a Strix Halo machine before this sync PR leaves draft"
        fi
    else
        echo "no conflicts"
    fi
    echo "then: git add <files> && git commit (edit the message: one line per conflict), and run: $0 verify $ours"
}

# first-parent merges of <range> that merge upstream history (oldest first): a refresh merge of master is not a sync step
sync_merges() {
    local m
    for m in $(git rev-list --first-parent --reverse --merges "$@"); do
        git merge-base --is-ancestor "$m^2" "$UPSTREAM" && echo "$m"
    done
    return 0
}

# the last sync merge since <old>
last_merge() { sync_merges "$1..${2:-HEAD}" | tail -1; }

verify() {
    local old=${1:?usage: sync.sh verify <old-head> [target]} target old_base before after markers last gone lost sized rc=0 review=0
    # the last sync merge since <old>: fixup commits may follow it
    last=$(last_merge "$old")
    [ -n "$last" ] || die "no merge of upstream in $(git rev-parse --short "$old")..HEAD"
    # no base file (deleted once the history base is right again): the target is what the last merge took
    target=${2:-$({ git show "HEAD:$BASE_FILE" 2> /dev/null || true; } | tr -d '[:space:]')}
    target=${target:-$(git rev-parse "$last^2")}
    target=$(git rev-parse --verify --quiet "$target^{commit}") || die "unknown target: $target"
    # after a reverted sync the target can already be an ancestor, so check the merge parent itself
    [ "$(git rev-parse "$last^2")" = "$target" ] || die "the last merge $(git rev-parse --short "$last") is not a merge of $(git rev-parse --short "$target") (second parent: $(git rev-parse --short "$last^2"))"
    if [ "$last" != "$(git rev-parse HEAD)" ]; then
        echo "note: $(git rev-list --count "$last..HEAD") commit(s) after the last merge, included below"
    fi
    if git cat-file -e "HEAD:$BASE_FILE" 2> /dev/null && [ "$(git show "HEAD:$BASE_FILE" | tr -d '[:space:]')" != "$target" ]; then
        echo "warning: $BASE_FILE in HEAD is not $target"
    fi
    old_base=$(content_base "$old")

    echo "== fork delta before (vs $(git rev-parse --short "$old_base")) and after (vs $(git rev-parse --short "$target"))"
    # no rename detection: a renamed path prints as "old => new" and would never match the other side
    before=$(git diff --numstat --no-renames "$old_base" "$old" -- . ":!$BASE_FILE" | awk -F'\t' '{print $3"\t+"$1" -"$2}' | sort)
    after=$(git diff --numstat --no-renames "$target" HEAD -- . ":!$BASE_FILE" | awk -F'\t' '{print $3"\t+"$1" -"$2}' | sort)

    echo "-- files no longer different from upstream (each must be explained: upstream took the change, or it was lost in a conflict)"
    gone=$(comm -23 <(cut -f1 <<< "$before") <(cut -f1 <<< "$after") | grep . || true)
    if [ -n "$gone" ]; then sed 's/^/  /' <<< "$gone"; review=1; else echo "  (none)"; fi
    echo "-- files newly different from upstream (should only be conflict fixups)"
    # grep . before the indent: comm prints an empty line when one side is empty
    comm -13 <(cut -f1 <<< "$before") <(cut -f1 <<< "$after") | { grep . || true; } | sed 's/^/  /' | { grep . || echo "  (none)"; }
    echo "-- files whose fork diff size changed (inspect: git diff $(git rev-parse --short "$target") HEAD -- <file>)"
    # a REVIEW too: a lost fork line that upstream also added elsewhere in the file keeps every count, only the size moves
    sized=$(join -t $'\t' <(echo "$before") <(echo "$after") | awk -F'\t' '$2 != $3 {printf "  %-60s before %s   after %s\n", $1, $2, $3}')
    if [ -n "$sized" ]; then echo "$sized"; review=1; else echo "  (none)"; fi
    echo "-- fork lines no longer in their file (each must be explained: moved, or upstream changed that code)"
    lost=$(python3 "$HERE/forklines.py" "$old_base" "$old" "$BASE_FILE" "$target")
    if [ -n "$lost" ]; then echo "$lost"; review=1; else echo "  (none)"; fi

    echo
    echo "== conflict markers left in files changed by the merge"
    markers=$(git diff --name-only "$old" HEAD | xargs -r -d '\n' git -c core.quotePath=false grep -n -E '^(<<<<<<<|>>>>>>>)( |$)' HEAD -- 2>/dev/null || true)
    if [ -n "$markers" ]; then
        echo "$markers" | sed 's/^/  /'
        rc=1
    else
        echo "  (none)"
    fi

    echo
    echo "== merge messages still holding the placeholder"
    markers=$(git log --format='%h %s' --grep='<how it was resolved>' "$old..HEAD")
    if [ -n "$markers" ]; then
        echo "$markers" | sed 's/^/  /'
        echo "  fill them: git commit --amend for the last one, or redo the step"
        rc=1
    else
        echo "  (none)"
    fi
    echo
    if [ $rc != 0 ]; then
        echo "verify: FAIL"
    elif [ $review != 0 ]; then
        # exit 2: not a failure by itself, but nothing may go on until a person explains every listed file and line
        echo "verify: REVIEW, fork code is gone from the files or lines listed above: explain each one in the merge message, or fix it"
        rc=2
    else
        echo "verify: OK, every fork line is still there"
    fi
    return $rc
}

# Conflict zones of a file with conflict markers: "start-end" line ranges, one per zone.
zones() { awk '/^<<<<<<< /{s=NR} /^>>>>>>> /{if (s) print s "-" NR; s=0}' "$1"; }

summary() {
    local old=${1:?usage: sync.sh summary <old-head> [branch]} br=${2:-HEAD} m n=0 p1 p2 base auto files f z first last total edits
    local -a merges
    mapfile -t merges < <(sync_merges "$old..$br")
    [ "${#merges[@]}" -gt 0 ] || die "no merge of upstream in $old..$br"
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
        files=$(conflict_files | while IFS= read -r f; do if grep -E "$SENSITIVE" <<< "$f" > /dev/null; then echo "\`$f\`*"; else echo "\`$f\`"; fi; done | paste -sd' ' -)
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
            echo "**\`$f\`**$(grep -E "$SENSITIVE" <<< "$f" > /dev/null && echo ' (sensitive)')"
            # the path as a plain string, not a sed pattern
            git log -1 --format=%B "$m" | awk -v p="- $f: " 'index($0, p) == 1 {print "Resolution: " substr($0, length(p) + 1)}'
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
        mapfile -t edits < <(git diff --name-only "$auto" "$m" -- . ":!$BASE_FILE" | grep -vxF -f <(cat "$TMP/cf"; echo "$BASE_FILE") || true)
        if [ "${#edits[@]}" -gt 0 ]; then
            echo
            echo "**Edits outside conflicts:**"
            echo
            echo '```diff'
            git diff "$auto" "$m" -- "${edits[@]}"
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
    echo "## Where upstream changes fork code"
    echo
    echo "Upstream hunks that overlap the fork's own lines or come within 10 lines of them, and the fork commits that own those lines (git blame, reverts looked through). Validate the ops and models that run through these fork changes."
    echo
    python3 "$HERE/owners.py" "$first" "$last" "$old" "$UPSTREAM"

    # fixups after the last merge are part of the PR and show in no step
    m=$(git rev-list --first-parent "${merges[-1]}..$br" | wc -l)
    if [ "$m" -gt 0 ]; then
        echo
        echo "## Commits after the last merge"
        echo
        echo "Not part of any step: read each one."
        echo
        git log --first-parent --format='- `%h` %s' "${merges[-1]}..$br"
    fi

    echo
    echo "## All changes"
    echo
    echo "<details><summary>Diffstat against the fork before the sync ($(git rev-parse --short "$old"))</summary>"
    echo
    echo '```'
    # every file up to 300, then only per directory: a PR body holds 65536 characters
    if [ "$(git diff --name-only "$old" "$br" | wc -l)" -le 300 ]; then git diff --stat=120 "$old" "$br"
    else git diff --dirstat=files,1 "$old" "$br"; git diff --shortstat "$old" "$br"; fi
    echo '```'
    echo "</details>"
}

case "${1:-}" in
    report) shift; report "$@" ;;
    merge)  shift; merge "$@" ;;
    verify) shift; verify "$@" ;;
    summary) shift; summary "$@" ;;
    models|owners)
        cmd=$1; shift
        [ -n "${1:-}" ] || die "usage: sync.sh $cmd <old-head> [target]"
        # default target: what the last sync merge took from upstream (fixup commits may follow it)
        t=${2:-}
        if [ -z "$t" ]; then
            t=$(last_merge "$1")
            [ -n "$t" ] || die "no merge of upstream in $1..HEAD: give the target"
            t="$t^2"
        fi
        t=$(git rev-parse --verify --quiet "$t^{commit}") || die "unknown target: $t"
        if [ "$cmd" = models ]; then models "$(content_base "$1")" "$t" "$1"
        else python3 "$HERE/owners.py" "$(content_base "$1")" "$t" "$1" "$UPSTREAM"; fi ;;
    *) echo "usage: $0 report [target] | merge [target] | verify <old-head> [target] | summary <old-head> [branch] | models <old-head> [target] | owners <old-head> [target]" >&2; exit 2 ;;
esac
