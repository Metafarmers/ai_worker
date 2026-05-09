################################################################################
# Set minimum required version of cmake, project name and compile options
################################################################################
cmake_minimum_required(VERSION 3.22)
project(ffw_moveit_config)

################################################################################
# Find and load build settings from external packages
################################################################################
find_package(ament_cmake REQUIRED)

################################################################################
# Install
################################################################################
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}
  PATTERN "setup_assistant.launch" EXCLUDE)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
  scripts/moveit_control_example.py
  scripts/moveit_planning_scene_obstacles.py
  scripts/moveit_plan_avoid_obstacles.py
  scripts/obstacle_node.py
  scripts/pathplanning_node.py
  DESTINATION lib/${PROJECT_NAME}
)

################################################################################
# Macro for ament package
################################################################################
ament_package()


