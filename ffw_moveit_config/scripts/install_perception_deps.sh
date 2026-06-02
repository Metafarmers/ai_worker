#!/usr/bin/env bash
# Install Python deps for perception_visualize_node.py (YOLO / OpenCV).
# Does NOT upgrade pip (Debian/apt pip cannot be replaced safely).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="${SCRIPT_DIR}/requirements-perception.txt"

# Jetson/arm images use /opt/venv; amd64 Docker uses system python3.
if [ -x /opt/venv/bin/python3 ]; then
  PY=/opt/venv/bin/python3
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "ERROR: python3 not found" >&2
  exit 1
fi

PIP_EXTRA=()
if "${PY}" -m pip install --help 2>/dev/null | grep -q 'break-system-packages'; then
  # Ubuntu 24.04 / PEP 668 — required when using apt python3 in ROS Jazzy images
  PIP_EXTRA+=(--break-system-packages)
fi

echo "[perception] Installing from ${REQ} using ${PY} ..."
"${PY}" -m pip install --no-cache-dir "${PIP_EXTRA[@]}" -r "${REQ}"
"${PY}" -c "from ultralytics import YOLO; print('[perception] ultralytics OK')"
