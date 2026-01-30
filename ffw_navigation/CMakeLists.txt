cmake_minimum_required(VERSION 3.8)
project(ffw_navigation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(behaviortree_cpp REQUIRED) # Jazzy (v4)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)

# 라이브러리 추가
add_library(is_path_length_under_plugin SHARED
  src/is_path_length_under.cpp
)

# Include 디렉토리 설정
target_include_directories(is_path_length_under_plugin PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# ★ 변경 추천: ament_target_dependencies 사용
# 이렇게 하면 include 경로와 라이브러리 링킹을 자동으로 처리해줍니다.
# nav_msgs__rosidl_typesupport_cpp 처럼 길게 안 써도 됩니다.
ament_target_dependencies(is_path_length_under_plugin
  behaviortree_cpp
  nav_msgs
  geometry_msgs
  rclcpp
)

# Install 규칙 (라이브러리)
install(TARGETS is_path_length_under_plugin
  EXPORT export_is_path_length_under_plugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install 규칙 (디렉토리)
# launch, config 등의 폴더가 실제로 존재하는지 꼭 확인하세요.
install(DIRECTORY
  launch
  rviz
  config
  maps
  DESTINATION share/${PROJECT_NAME}
)

target_compile_definitions(is_path_length_under_plugin PRIVATE BT_PLUGIN_EXPORT)

# Export 설정
ament_export_libraries(is_path_length_under_plugin)
ament_export_dependencies(behaviortree_cpp nav_msgs geometry_msgs rclcpp)

ament_package()