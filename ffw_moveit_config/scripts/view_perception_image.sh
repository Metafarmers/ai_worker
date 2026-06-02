#!/usr/bin/env bash
# View /perception/visualization/image (requires a display or X11 forward).
set -euo pipefail

TOPIC="${1:-/perception/visualization/image}"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"

if ros2 pkg executables image_tools 2>/dev/null | grep -q showimage; then
  exec ros2 run image_tools showimage --ros-args -r "image:=${TOPIC}"
fi

if ros2 pkg executables rqt_image_view 2>/dev/null | grep -q rqt_image_view; then
  exec ros2 run rqt_image_view rqt_image_view
fi

echo "No image viewer found. Install one of:" >&2
echo "  apt-get install -y ros-${ROS_DISTRO}-image-tools" >&2
echo "  apt-get install -y ros-${ROS_DISTRO}-rqt-image-view" >&2
echo "" >&2
echo "Then run:" >&2
echo "  ros2 run image_tools showimage --ros-args -r image:=${TOPIC}" >&2
echo "  # or" >&2
echo "  ros2 run rqt_image_view rqt_image_view   # pick topic in GUI" >&2
exit 1
