#!/usr/bin/env bash
# Docker: ~/ros2_ws/src/ai_worker/ffw_moveit_config/scripts/run_obstacle_node.sh
# Runs obstacle_node from SOURCE (skips stale install/ copy until you colcon build).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/obstacle_node.py" "$@"
