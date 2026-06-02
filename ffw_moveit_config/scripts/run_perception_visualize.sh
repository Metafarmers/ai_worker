#!/usr/bin/env bash
# Run perception_visualize_node from SOURCE (no colcon rebuild required).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! python3 -c "import ultralytics" 2>/dev/null; then
  echo "[perception] ultralytics not installed — running install_perception_deps.sh ..."
  bash "${SCRIPT_DIR}/install_perception_deps.sh"
fi

exec python3 "${SCRIPT_DIR}/perception_visualize_node.py" "$@"
