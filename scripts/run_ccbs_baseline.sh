#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
src=${CCBS_ROOT:-external/Continuous-CBS}
rev=b8f45e166cf39ee7e8e06f0bfe4fbab0dee68932
if [[ ! -e "$src" ]]; then
  git clone --no-checkout https://github.com/PathPlanning/Continuous-CBS.git "$src"
  git -C "$src" checkout --detach "$rev"
fi
if [[ "$(git -C "$src" rev-parse HEAD)" != "$rev" ]] || [[ -n "$(git -C "$src" status --porcelain --untracked-files=no)" ]]; then
  echo "Expected a clean Continuous-CBS checkout at $rev in $src" >&2
  exit 1
fi
cmake -S "$src" -B "$src/build-vrl" -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$src/build-vrl" --target CCBS --parallel 2
python scripts/paired_ccbs.py --binary "$src/build-vrl/CCBS" --config "$src/Examples/config.xml" "$@"
