#include "ffw_navigation/is_path_length_under.hpp"
#include "behaviortree_cpp/bt_factory.h"
#include <cmath> 
#include <limits> // std::numeric_limits 사용을 위해 추가

namespace nav2_behavior_tree
{

// 생성자 구현
IsPathLengthUnder::IsPathLengthUnder(
  const std::string & name, const BT::NodeConfiguration & conf)
: BT::ConditionNode(name, conf)
{
}

// 포트 정의 구현
BT::PortsList IsPathLengthUnder::providedPorts()
{
  return {
    BT::InputPort<nav_msgs::msg::Path>("path", "Path to calculate distance for"),
    BT::InputPort<double>("distance_threshold", 1.0, "Distance threshold in meters")
  };
}

// Tick 구현
BT::NodeStatus IsPathLengthUnder::tick()
{
  nav_msgs::msg::Path path;
  double threshold;

  // 1. Input Port 데이터 가져오기
  if (!getInput("path", path)) {
    return BT::NodeStatus::FAILURE;
  }
  if (!getInput("distance_threshold", threshold)) {
    return BT::NodeStatus::FAILURE;
  }

  // 2. 경로가 비어있으면 도착한 것으로 간주 (혹은 짧음)
  if (path.poses.empty()) {
    return BT::NodeStatus::SUCCESS;
  }

  // 3. 경로 전체 길이 계산
  double path_length = 0.0;
  for (size_t i = 0; i + 1 < path.poses.size(); ++i) {
    const auto & p1 = path.poses[i].pose.position;
    const auto & p2 = path.poses[i+1].pose.position;
    path_length += std::hypot(p2.x - p1.x, p2.y - p1.y);
  }

  // 5. 결과 반환
  if (path_length <= threshold) {
    return BT::NodeStatus::SUCCESS; // 짧은 거리 (Crab 주행)
  }

  return BT::NodeStatus::FAILURE; // 긴 거리 (회전 주행)
}

}  // namespace nav2_behavior_tree

// Plugin 등록 매크로
BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<nav2_behavior_tree::IsPathLengthUnder>("IsPathLengthUnder");
}
