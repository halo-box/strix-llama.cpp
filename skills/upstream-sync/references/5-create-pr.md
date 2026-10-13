# 5. Create the PR

Part of the `upstream-sync` skill: read [SKILL.md](../SKILL.md) first for the rules.

Append the results to `body.md` between their markers (`results-block.md`, written next to `summary.md`), so that a later `validate.sh --pr` replaces them instead of adding a second block. Check that the body holds no private data, then push and open the PR:

```sh
cat <output dir>/results-block.md >> body.md
# home and model paths, and this machine's name: nothing is pushed while one is left
if grep -n -i -E "/home/|/Users/|/root/|/mnt/|/media/|/data/|/srv/|~/|$(hostname -s)" body.md; then
  echo "remove the private data above first"
else
  git push -u origin HEAD &&
  gh pr create --repo halo-box/strix-llama.cpp --base master \
    --title "Merge upstream llama.cpp up to <sha> (<date>)" --body-file body.md
fi
```

If `origin` is a personal fork, add `--head <owner>:<branch>`.

Open it as a draft (`--draft`) unless both backends are `TARGET PASS`, and say in the body what is missing. To refresh the results after a fix, rerun the validation on the PR itself: `validate.sh --pr <number>` replaces the marked section of the PR body (and updates it after each backend while it runs). If the body would pass the GitHub limit, it posts the results as a comment instead, and says so. To update the body by hand, use `gh api -X PATCH repos/halo-box/strix-llama.cpp/pulls/<number> -F body=@body.md`: `gh pr edit` fails on this repo with a GraphQL error about classic Projects. A maintainer merges the PR with a merge commit, after `upstream-sync-verify` if someone runs it.
