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
file(CHMOD
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/curobo_pathplanning_node.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/curobo_obstacle_utils.py"
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}
  PATTERN "setup_assistant.launch" EXCLUDE)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
  scripts/moveit_control_example.py
  scripts/moveit_planning_scene_obstacles.py
  scripts/moveit_plan_avoid_obstacles.py
  scripts/obstacle_node.py
  scripts/pathplanning_node.py
  scripts/curobo_obstacle_utils.py
  scripts/curobo_pathplanning_node.py
  DESTINATION lib/${PROJECT_NAME}
)

################################################################################
# Macro for ament package
################################################################################
ament_package()


