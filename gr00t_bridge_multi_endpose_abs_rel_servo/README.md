# GR00T End-Pose Relative Action Inference — MoveIt Servo

MoveIt Servo PoseCommand 기반의 GR00T 듀얼암 추론 브릿지.

기존 `/compute_ik` blocking 방식을 MoveIt Servo로 교체하여 실시간 제어를 실현합니다.

## MoveIt Servo 장점

- **비동기 실시간 제어**: blocking IK 서비스 대기 없이 PoseStamped publish만으로 동작
- **내장 모션 스무딩**: Butterworth / AccelerationLimited / Ruckig 필터
- **특이점 자동 감속**: 특이점 근접 시 속도 자동 스케일 다운
- **충돌 방지**: self-collision + scene collision 실시간 체크
- **조인트 제한 자동 적용**: URDF + joint_limits.yaml 기반

## 폴더 구조

```
gr00t_bridge_multi_endpose_abs_rel_servo/
├── __init__.py
├── setup.py
├── ffw_sg2_endpose_rel_action_inference_config.py   # 모델 데이터 config (변경 없음)
├── gr00t_endpose_rel_servo_node.py                  # 핵심: Servo PoseCommand 노드
├── config/
│   ├── servo_params_left.yaml                       # arm_l Servo 파라미터
│   └── servo_params_right.yaml                      # arm_r Servo 파라미터
├── launch/
│   └── servo_dual_arm.launch.py                     # 양팔 Servo 노드 런치
├── scripts/
│   └── run_gr00t_endpose_rel_servo_inference.py     # 실행 스크립트
└── README.md
```

## 실행 순서

```bash
# 1. ai_worker Docker: 로봇 bringup (기존)
ros2 launch ffw_bringup ffw_sg2_follower_ai.launch.py

# 2. ai_worker Docker: end-pose 퍼블리셔 (기존)
python3 inference_time_end_pose.py --rate 50

# 3. ai_worker Docker: MoveIt Servo 듀얼암 런치 (신규)
ros2 launch gr00t_bridge_multi_endpose_abs_rel_servo servo_dual_arm.launch.py

# 4. 4090 Docker: GR00T 추론 실행 (dry-run 테스트)
python3 scripts/run_gr00t_endpose_rel_servo_inference.py --dry-run

# 4'. 4090 Docker: GR00T 추론 실행 (실제 동작)
python3 scripts/run_gr00t_endpose_rel_servo_inference.py --publish
```

## 아키텍처

```
Model (relative delta) → absolute PoseStamped → MoveIt Servo
                                                    ├── IK solving
                                                    ├── Motion smoothing
                                                    ├── Singularity check
                                                    ├── Collision check
                                                    └── JointTrajectory → controller

Gripper → JointTrajectory (별도, allow_partial_joints_goal=true)
```

## 환경 전제

- **ai_worker Docker**: ROS2 Jazzy
- `ros-jazzy-moveit-servo` 설치 필요
- `allow_partial_joints_goal: true` 설정 (기존 hardware_controller.yaml에 이미 설정됨)
- SRDF에 end-effector 정의 완비: `end_effector_l_link`, `end_effector_r_link`

## 기존 대비 변경점

| 항목 | 기존 (IK) | 변경 (Servo) |
|------|----------|-------------|
| IK 방식 | `/compute_ik` blocking 서비스 | Servo 내장 IK (non-blocking) |
| 스무딩 | 없음 | Butterworth 필터 |
| 특이점 | 미검사 | 자동 감속/정지 |
| 충돌 검사 | 없음 | self + scene collision |
| 제어 주기 | inference rate 의존 | Servo 50Hz 독립 |
| 코드량 | ~1200줄 | ~650줄 (IK 관련 ~200줄 삭제) |

## 주요 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--servo-left-ns` | `servo_left` | 왼팔 Servo 네임스페이스 |
| `--servo-right-ns` | `servo_right` | 오른팔 Servo 네임스페이스 |
| `--rate` | `20.0` | 추론 주기 (Hz) |
| `--action-horizon` | `16` | 전체 action horizon |
| `--action-horizon-execute` | `4` | 실행할 스텝 수 |
| `--dry-run` | (flag) | 동작 없이 추론만 수행 |
| `--publish` | (flag) | 실제 Servo에 publish |
