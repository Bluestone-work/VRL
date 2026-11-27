#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
upstream=${MR_NAV_ROOT:-external/mr-nav}
rev=c129da1447d9854e31b1d7b970b7f48b8a553ea7
if [[ ! -e "$upstream" ]]; then
  git clone --no-checkout https://github.com/yinghansun/mr-nav.git "$upstream"
  git -C "$upstream" checkout --detach "$rev"
fi
if [[ "$(git -C "$upstream" rev-parse HEAD)" != "$rev" ]] || [[ -n "$(git -C "$upstream" status --porcelain --untracked-files=no)" ]]; then
  echo "Expected a clean mr-nav checkout at $rev in $upstream" >&2
  exit 1
fi
PYTHONPATH="$(pwd):$(cd "$upstream" && pwd)${PYTHONPATH:+:$PYTHONPATH}" \
  python scripts/paired_mr_nav.py --upstream "$upstream" "$@"
