#!/usr/bin/env bash
# Merge develop into the main branch, dropping the working documentation.
#   bash scripts/sync_main.sh [--push]
set -euo pipefail
cd "$(dirname "$0")/.."

KEEP='^(README\.md|docs/locales/README_[a-z]{2}\.md)$'
SOURCE=develop
TARGET=main

[ -z "$(git status --porcelain)" ] || { echo "working tree is dirty; commit or stash first" >&2; exit 1; }
START=$(git rev-parse --abbrev-ref HEAD)
trap 'git checkout -q "$START" 2>/dev/null || true' EXIT

git checkout -q "$TARGET"
git merge --no-commit --no-ff "$SOURCE" || true   # conflicts are resolved below

mapfile -t CONFLICTS < <(git diff --name-only --diff-filter=U)
for f in "${CONFLICTS[@]:-}"; do
    [ -n "$f" ] || continue
    git checkout --theirs -- "$f" 2>/dev/null || true
    git add -- "$f"
done

mapfile -t DROP < <(git ls-files '*.md' | grep -Ev "$KEEP" || true)
if [ "${#DROP[@]}" -gt 0 ] && [ -n "${DROP[0]}" ]; then
    git rm -rqf --ignore-unmatch -- "${DROP[@]}"
    printf 'dropped %d documentation file(s)\n' "${#DROP[@]}"
fi

if git diff --cached --quiet && git diff --quiet; then
    echo "release already matches $SOURCE; nothing to commit"
else
    git commit -q -m "Merge $SOURCE into $TARGET

The working documentation stays on $SOURCE: release ships the code and
the README pair."
    echo "merged $SOURCE -> $TARGET at $(git rev-parse --short HEAD)"
fi

git ls-files '*.md'

if [ "${1:-}" = "--push" ]; then
    git push origin "$TARGET"
fi
