#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! python3 -c "import ultralytics" 2>/dev/null; then
  bash "${SCRIPT_DIR}/install_perception_deps.sh"
fi
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
exec python3 "${SCRIPT_DIR}/perception_tf_node.py" "$@"
