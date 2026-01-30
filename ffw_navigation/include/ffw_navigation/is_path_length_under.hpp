#ifndef FFW_NAVIGATION__IS_PATH_LENGTH_UNDER_HPP_
#define FFW_NAVIGATION__IS_PATH_LENGTH_UNDER_HPP_

#include <string>
#include <memory>
#include <cmath>

#include "behaviortree_cpp/condition_node.h"
#include "nav_msgs/msg/path.hpp"

namespace nav2_behavior_tree
{

/**
 * @brief 주어진 Path의 전체 길이가 특정 임계값(Threshold) 이하인지 확인하는 Condition Node
 */
class IsPathLengthUnder : public BT::ConditionNode
{
public:
  // 생성자
  IsPathLengthUnder(const std::string & name, const BT::NodeConfiguration & conf);

  // 입력 포트 정의
  static BT::PortsList providedPorts();

  // 실행 로직 (Tick)
  BT::NodeStatus tick() override;
};

}  // namespace nav2_behavior_tree

#endif  // FFW_NAVIGATION__IS_PATH_LENGTH_UNDER_HPP_
