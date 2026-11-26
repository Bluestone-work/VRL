#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
upstream=${MR_NAV_ROOT:-external/mr-nav}
if [[ ! -d "$upstream" ]]; then
  echo "SKIP: mr-nav upstream checkout is missing at $upstream" >&2
  echo "Clone https://github.com/yinghansun/mr-nav there, record its commit and license, then add the adapter." >&2
  exit 2
fi
echo "SKIP: upstream checkout exists but the VRL observation/action adapter is not implemented" >&2
exit 2
