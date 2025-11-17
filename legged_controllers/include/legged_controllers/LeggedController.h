//
// Created by qiayuan on 2022/6/24.
//

#pragma once

#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/imu_sensor_interface.h>
#include <legged_common/hardware_interface/ContactSensorInterface.h>

#include <ocs2_centroidal_model/CentroidalModelRbdConversions.h>
#include <ocs2_core/misc/Benchmark.h>
#include <ocs2_legged_robot_ros/visualization/LeggedRobotVisualizer.h>
#include <ocs2_mpc/MPC_MRT_Interface.h>

#include <legged_estimation/StateEstimateBase.h>
#include <legged_interface/LeggedInterface.h>
#include <legged_wbc/WbcBase.h>
#include <sensor_msgs/Joy.h>
#include <std_msgs/Float64MultiArray.h>


#include "legged_controllers/SafetyChecker.h"
#include "legged_controllers/visualization/LeggedSelfCollisionVisualization.h"

namespace legged {
using namespace ocs2;
using namespace legged_robot;

class LeggedController : public controller_interface::MultiInterfaceController<HybridJointInterface, hardware_interface::ImuSensorInterface,
                                                                               ContactSensorInterface> {
 public:
  LeggedController() = default;
  ~LeggedController() override;
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& controller_nh) override;
  void update(const ros::Time& time, const ros::Duration& period) override;
  void starting(const ros::Time& time) override;
  void stopping(const ros::Time& /*time*/) override { mpcRunning_ = false; }

 protected:
  virtual void updateStateEstimation(const ros::Time& time, const ros::Duration& period);

  virtual void setupLeggedInterface(const std::string& taskFile, const std::string& urdfFile, const std::string& referenceFile,
                                    bool verbose);
  virtual void setupMpc();
  virtual void setupMrt();
  virtual void setupStateEstimate(const std::string& taskFile, bool verbose);

  // Interface
  std::shared_ptr<LeggedInterface> leggedInterface_;
  std::shared_ptr<PinocchioEndEffectorKinematics> eeKinematicsPtr_;
  std::vector<HybridJointHandle> hybridJointHandles_;
  std::vector<ContactSensorHandle> contactHandles_;
  hardware_interface::ImuSensorHandle imuSensorHandle_;

  // State Estimation
  SystemObservation currentObservation_;
  vector_t measuredRbdState_;
  std::shared_ptr<StateEstimateBase> stateEstimate_;
  std::shared_ptr<CentroidalModelRbdConversions> rbdConversions_;

  // Whole Body Control
  std::shared_ptr<WbcBase> wbc_;
  std::shared_ptr<SafetyChecker> safetyChecker_;

  // Nonlinear MPC
  std::shared_ptr<MPC_BASE> mpc_;
  std::shared_ptr<MPC_MRT_Interface> mpcMrtInterface_;

  // Visualization
  std::shared_ptr<LeggedRobotVisualizer> robotVisualizer_;
  std::shared_ptr<LeggedSelfCollisionVisualization> selfCollisionVisualization_;
  ros::Publisher observationPublisher_;

 private:
  std::thread mpcThread_;
  std::atomic_bool controllerRunning_{}, mpcRunning_{};
  benchmark::RepeatedTimer mpcTimer_;
  benchmark::RepeatedTimer wbcTimer_;
  scalar_t controller_time;
 const size_t dof = 12;
 // ========= 模式切换 =========
 bool useRlControl_ = false;
 bool lastSwitchButtonState_ = false;
 int rlSwitchButtonIndex_ = 0;
 // ========= RL 通信 =========
 ros::Publisher  rlObsPub_;           // 发布 45 维 obs
 ros::Subscriber rlPosSub_;           // 订阅 12 维 pos



 std::vector<double> lastRlPos_;      // 最近一次 RL 输出的 12 维力矩
 bool rlPosReceived_ = false;

 // prev_actions = 上一帧 RL 力矩（45 维里的一部分）
 std::vector<double> prevRlPos_;   // 12 维

 // ========= 组成 45 维所需量 =========
 ros::Subscriber joySub_;
 ros::Subscriber targetVelSub_;       // 期望速度（来自 joy->cmd_vel）
 ros::Subscriber stageBufSub_;        // 3 维阶段编码（你自己算后发过来）

 double targetVels_[3] = {0.0, 0.0, 0.0};   // vx, vy, wz
 double stageBuf_[3]   = {0.0, 0.0, 0.0};   // 3 维状态标识

 Eigen::Vector3d baseRPY_{0.0, 0.0, 0.0};   // roll, pitch, yaw

 // 最近一次关节位置/速度（给 RL obs 用）
 vector_t lastJointPos_;
 vector_t lastJointVel_;

 // ========= 回调 =========
 void joyCallback(const sensor_msgs::Joy::ConstPtr& msg);
 void rlPosCallback(const std_msgs::Float64MultiArray::ConstPtr& msg);
 void targetVelCallback(const geometry_msgs::Twist::ConstPtr& msg);
 void stageBufCallback(const std_msgs::Float64MultiArray::ConstPtr& msg);

 // 构造并发布 45 维 obs
 void publishRlObservation();

 // RL 模式下用的 PD 增益（可选，如果你想在 torque 上再做点东西）
 double rlKp_ = 0.0;
 double rlKd_ = 0.0;
};

class LeggedCheaterController : public LeggedController {
 protected:
  void setupStateEstimate(const std::string& taskFile, bool verbose) override;
};

}  // namespace legged
