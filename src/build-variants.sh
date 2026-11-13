#!/usr/bin/env bash

#######################################################
# Render each screen shape onto its own branch
#######################################################
#
# For every shape below, renders the wallpapers and records a commit on that
# shape's branch whose tree is `main` with backgrounds/ swapped out. The commit
# has two parents (the branch's previous tip and main), so the branches only
# ever move forward and `git pull` on an installed theme stays a fast-forward.
#
#   src/build-variants.sh            build every shape from main
#   src/build-variants.sh 21x9 3x2   build some
#   src/build-variants.sh --push     build every shape, then push the branches

set -euo pipefail

# branch  size
VARIANTS=(
  "16x10 3840x2400"
  "3x2   3840x2560"
  "21x9  5120x2160"
  "32x9  7680x2160"
)

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

push=false
wanted=()
for arg in "$@"; do
  case "$arg" in
    --push) push=true ;;
    *) wanted+=("$arg") ;;
  esac
done

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "❌ Commit your changes on main first; variants are built from main's tree." >&2
  exit 1
fi

main_rev="$(git rev-parse main)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

built=()
for variant in "${VARIANTS[@]}"; do
  read -r branch size <<<"$variant"
  if ((${#wanted[@]})) && [[ ! " ${wanted[*]} " == *" $branch "* ]]; then
    continue
  fi

  echo "🎨 $branch ($size)"
  out="$work/$branch"
  # Render from main's committed source, not whatever is checked out.
  mkdir -p "$work/src-$branch"
  git archive "$main_rev" src | tar -x -C "$work/src-$branch"
  uv run -q "$work/src-$branch/src/generate.py" --size "$size" --out "$out" --backgrounds-only

  # main's tree, with backgrounds/ replaced by this shape's renders
  export GIT_INDEX_FILE="$work/index-$branch"
  git read-tree "$main_rev"
  git ls-files -z backgrounds | xargs -0 git update-index --force-remove
  for f in "$out"/backgrounds/*; do
    blob="$(git hash-object -w "$f")"
    git update-index --add --cacheinfo "100644,$blob,backgrounds/$(basename "$f")"
  done
  tree="$(git write-tree)"
  unset GIT_INDEX_FILE

  parents=(-p "$main_rev")
  if prev="$(git rev-parse --verify --quiet "refs/heads/$branch")"; then
    if [[ "$(git rev-parse "$prev^{tree}")" == "$tree" ]]; then
      echo "   unchanged"
      continue
    fi
    parents=(-p "$prev" -p "$main_rev")
  fi

  commit="$(git commit-tree "$tree" "${parents[@]}" \
    -m "$branch wallpapers ($size) for $(git rev-parse --short "$main_rev")")"
  git update-ref "refs/heads/$branch" "$commit"
  built+=("$branch")
  echo "   ✅ $branch → $(git rev-parse --short "$commit")"
done

if $push && ((${#built[@]})); then
  git push origin "${built[@]}"
fi
