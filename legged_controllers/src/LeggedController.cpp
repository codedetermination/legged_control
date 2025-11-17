//
// Created by qiayuan on 2022/6/24.
//

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "legged_controllers/LeggedController.h"

#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_core/thread_support/ExecuteAndSleep.h>
#include <ocs2_core/thread_support/SetThreadPriority.h>
#include <ocs2_legged_robot_ros/gait/GaitReceiver.h>
#include <ocs2_msgs/mpc_observation.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_ros_interfaces/common/RosMsgConversions.h>
#include <ocs2_ros_interfaces/synchronized_module/RosReferenceManager.h>
#include <ocs2_sqp/SqpMpc.h>

#include <angles/angles.h>
#include <legged_estimation/FromTopiceEstimate.h>
#include <legged_estimation/LinearKalmanFilter.h>
#include <legged_wbc/HierarchicalWbc.h>
#include <legged_wbc/WeightedWbc.h>
#include <pluginlib/class_list_macros.hpp>

namespace legged {
bool LeggedController::init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& controller_nh) {
  // Initialize OCS2
  std::string urdfFile;
  std::string taskFile;
  std::string referenceFile;
  controller_nh.getParam("/urdfFile", urdfFile);
  controller_nh.getParam("/taskFile", taskFile);
  controller_nh.getParam("/referenceFile", referenceFile);
  bool verbose = false;
  loadData::loadCppDataType(taskFile, "legged_robot_interface.verbose", verbose);

  setupLeggedInterface(taskFile, urdfFile, referenceFile, verbose);
  setupMpc();
  setupMrt();
  // Visualization
  ros::NodeHandle nh;
  CentroidalModelPinocchioMapping pinocchioMapping(leggedInterface_->getCentroidalModelInfo());
  eeKinematicsPtr_ = std::make_shared<PinocchioEndEffectorKinematics>(leggedInterface_->getPinocchioInterface(), pinocchioMapping,
                                                                      leggedInterface_->modelSettings().contactNames3DoF);
  robotVisualizer_ = std::make_shared<LeggedRobotVisualizer>(leggedInterface_->getPinocchioInterface(),
                                                             leggedInterface_->getCentroidalModelInfo(), *eeKinematicsPtr_, nh);
  selfCollisionVisualization_.reset(new LeggedSelfCollisionVisualization(leggedInterface_->getPinocchioInterface(),
                                                                         leggedInterface_->getGeometryInterface(), pinocchioMapping, nh));

  // Hardware interface
  auto* hybridJointInterface = robot_hw->get<HybridJointInterface>();
  std::vector<std::string> joint_names{"LF_HAA", "LF_HFE", "LF_KFE", "LH_HAA", "LH_HFE", "LH_KFE",
                                       "RF_HAA", "RF_HFE", "RF_KFE", "RH_HAA", "RH_HFE", "RH_KFE"};
  for (const auto& joint_name : joint_names) {
    hybridJointHandles_.push_back(hybridJointInterface->getHandle(joint_name));
  }
  auto* contactInterface = robot_hw->get<ContactSensorInterface>();
  for (const auto& name : leggedInterface_->modelSettings().contactNames3DoF) {
    contactHandles_.push_back(contactInterface->getHandle(name));
  }
  imuSensorHandle_ = robot_hw->get<hardware_interface::ImuSensorInterface>()->getHandle("unitree_imu");

  // State estimation
  setupStateEstimate(taskFile, verbose);

  // Whole body control
  wbc_ = std::make_shared<WeightedWbc>(leggedInterface_->getPinocchioInterface(), leggedInterface_->getCentroidalModelInfo(),
                                       *eeKinematicsPtr_);
  wbc_->loadTasksSetting(taskFile, verbose);

  // Safety Checker
  safetyChecker_ = std::make_shared<SafetyChecker>(leggedInterface_->getCentroidalModelInfo());

  // RL 切换按钮 & PD
  controller_nh.param("rl_switch_button_index", rlSwitchButtonIndex_, 0);
  controller_nh.param("rl_kp", rlKp_, 0.0);
  controller_nh.param("rl_kd", rlKd_, 0.0);

  joySub_ = nh.subscribe("/joy", 1, &LeggedController::joyCallback, this);

  // RL 两个 topic： 45 in / 12 out
  rlObsPub_ = nh.advertise<std_msgs::Float64MultiArray>("rl_obs", 1);
  rlPosSub_ = nh.subscribe("rl_pos", 1, &LeggedController::rlPosCallback, this);

  // 期望速度：上游节点把 /joy 映射成 /target_vels (geometry_msgs/Twist)
  targetVelSub_ = nh.subscribe("target_vels", 1, &LeggedController::targetVelCallback, this);

  // 阶段编码：你自己算好发到 rl_stage_buf (Float64MultiArray, 长度 3)
  stageBufSub_ = nh.subscribe("rl_stage_buf", 1, &LeggedController::stageBufCallback, this);

  lastRlPos_.assign(12, 0.0);
  prevRlPos_.assign(12, 0.0);

  // 初始化关节状态缓存
  lastJointPos_.setZero(hybridJointHandles_.size());
  lastJointVel_.setZero(hybridJointHandles_.size());

  ROS_INFO_STREAM("[LeggedController] RL topic interface ready. "
                  "Button " << rlSwitchButtonIndex_ << " toggles RL/normal.");
  return true;
}

void LeggedController::joyCallback(const sensor_msgs::Joy::ConstPtr& msg) {
  if (rlSwitchButtonIndex_ < 0 ||
      rlSwitchButtonIndex_ >= static_cast<int>(msg->buttons.size())) {
    return;
      }
  bool pressed = (msg->buttons[rlSwitchButtonIndex_] != 0);

  if (pressed && !lastSwitchButtonState_) {
    useRlControl_ = !useRlControl_;
    ROS_WARN_STREAM("[LeggedController] Switch to "
                    << (useRlControl_ ? "RL control." : "normal MPC+WBC control."));
  }
  lastSwitchButtonState_ = pressed;
}

void LeggedController::rlPosCallback(const std_msgs::Float64MultiArray::ConstPtr& msg) {
  if (msg->data.size() != 12) {
    ROS_ERROR_THROTTLE(1.0, "[LeggedController] rl_tau size %zu != 12", msg->data.size());
    return;
  }
  lastRlPos_ = msg->data;
  rlPosReceived_ = true;

  // 作为下一帧 obs 里的 prev_actions
  prevRlPos_ = lastRlPos_;
}


void LeggedController::targetVelCallback(const geometry_msgs::Twist::ConstPtr& msg) {
  targetVels_[0] = msg->linear.x;
  targetVels_[1] = msg->linear.y;
  targetVels_[2] = msg->angular.z;
}

void LeggedController::stageBufCallback(const std_msgs::Float64MultiArray::ConstPtr& msg) {
  if (msg->data.size() != 3) {
    ROS_ERROR_THROTTLE(1.0, "[LeggedController] rl_stage_buf size %zu != 3", msg->data.size());
    return;
  }
  for (int i = 0; i < 3; ++i) {
    stageBuf_[i] = msg->data[i];
  }
}

void LeggedController::starting(const ros::Time& time) {
  // Initial state
  currentObservation_.state.setZero(leggedInterface_->getCentroidalModelInfo().stateDim);
  updateStateEstimation(time, ros::Duration(0.002));
  controller_time=time.toSec();
  currentObservation_.input.setZero(leggedInterface_->getCentroidalModelInfo().inputDim);
  currentObservation_.mode = ModeNumber::STANCE;

  TargetTrajectories target_trajectories({currentObservation_.time}, {currentObservation_.state}, {currentObservation_.input});

  // Set the first observation and command and wait for optimization to finish
  mpcMrtInterface_->setCurrentObservation(currentObservation_);
  mpcMrtInterface_->getReferenceManager().setTargetTrajectories(target_trajectories);
  ROS_INFO_STREAM("Waiting for the initial policy ...");
  while (!mpcMrtInterface_->initialPolicyReceived() && ros::ok()) {
    mpcMrtInterface_->advanceMpc();
    ros::WallRate(leggedInterface_->mpcSettings().mrtDesiredFrequency_).sleep();
  }
  ROS_INFO_STREAM("Initial policy has been received.");

  mpcRunning_ = true;
}

void LeggedController::publishRlObservation() {
  std_msgs::Float64MultiArray msg;
  msg.data.resize(45);

  // 0~2: target_vels
  msg.data[0] = targetVels_[0];
  msg.data[1] = targetVels_[1];
  msg.data[2] = targetVels_[2];

  // 3~5: base RPY
  msg.data[3] = baseRPY_[0];
  msg.data[4] = baseRPY_[1];
  msg.data[5] = baseRPY_[2];

  // 6~17: 12 维关节位置
  for (size_t j = 0; j < 12; ++j) {
    msg.data[6 + j] = lastJointPos_(j);
  }

  // 18~29: 12 维关节速度
  for (size_t j = 0; j < 12; ++j) {
    msg.data[18 + j] = lastJointVel_(j);
  }

  // 30~41: prev_actions（上一帧 RL 位置）
  for (size_t j = 0; j < 12; ++j) {
    msg.data[30 + j] = prevRlPos_[j];
  }

  // 42~44: stage_buf
  msg.data[42] = stageBuf_[0];
  msg.data[43] = stageBuf_[1];
  msg.data[44] = stageBuf_[2];

  rlObsPub_.publish(msg);
}


void LeggedController::update(const ros::Time& time, const ros::Duration& period) {
  // State Estimate
  updateStateEstimation(time, period);

  // Update the current state of the system
  mpcMrtInterface_->setCurrentObservation(currentObservation_);

  // Rl相关的controller状态发布

  // Load the latest MPC policy

  mpcMrtInterface_->updatePolicy();

  vector_t torque(dof);
  vector_t posDes(dof);
  vector_t velDes(dof);

  // Evaluate the current policy
  if (!useRlControl_) {
    vector_t optimizedState, optimizedInput;

    size_t plannedMode = 0;  // The mode that is active at the time the policy is evaluated at.
    mpcMrtInterface_->evaluatePolicy(currentObservation_.time, currentObservation_.state, optimizedState, optimizedInput, plannedMode);

    // Whole body control
    currentObservation_.input = optimizedInput;

    wbcTimer_.startTimer();
    vector_t x = wbc_->update(optimizedState, optimizedInput, measuredRbdState_, plannedMode, period.toSec());
    wbcTimer_.endTimer();

    torque = x.tail(12);

    posDes = centroidal_model::getJointAngles(optimizedState, leggedInterface_->getCentroidalModelInfo());
    velDes = centroidal_model::getJointVelocities(optimizedInput, leggedInterface_->getCentroidalModelInfo());

    // Safety check, if failed, stop the controller
    if (!safetyChecker_->check(currentObservation_, optimizedState, optimizedInput)) {
      ROS_ERROR_STREAM("[Legged Controller] Safety check failed, stopping the controller.");
      stopRequest(time);
    }

    for (size_t j = 0; j < leggedInterface_->getCentroidalModelInfo().actuatedDofNum; ++j) {
      hybridJointHandles_[j].setCommand(posDes(j), velDes(j), 0, 1, torque(j));
    }
  }
  else {
    publishRlObservation();
    if (!rlPosReceived_ || lastRlPos_.size() != dof) {
      // 没有有效 RL 动作时，简单保持当前姿态 + 零力矩
      double stiffness = 20.0;
      double damping = 0.5;
      ROS_WARN_THROTTLE(1.0, "[LeggedController] RL action invalid, holding current joint positions.");
      std::vector<double> joint_init_pos{
          0.1,   // LF_HAA  <- FL_hip_joint
          0.8,   // LF_HFE  <- FL_thigh_joint
         -1.5,   // LF_KFE  <- FL_calf_joint

          0.1,   // LH_HAA  <- RL_hip_joint
          1.0,   // LH_HFE  <- RL_thigh_joint
         -1.5,   // LH_KFE  <- RL_calf_joint

         -0.1,   // RF_HAA  <- FR_hip_joint
          0.8,   // RF_HFE  <- FR_thigh_joint
         -1.5,   // RF_KFE  <- FR_calf_joint

         -0.1,   // RH_HAA  <- RR_hip_joint
          1.0,   // RH_HFE  <- RR_thigh_joint
         -1.5    // RH_KFE  <- RR_calf_joint
      };

      Eigen::VectorXd currentPos(dof);
      Eigen::VectorXd currentVel(dof);

      for (size_t j = 0; j < dof; ++j) {
        currentPos(j) = hybridJointHandles_[j].getPosition();
        currentVel(j) = hybridJointHandles_[j].getVelocity();
      }

      // 2) 按照 PD 控制计算力矩：
      //    torques = stiffness * (joint_targets - (current_dof_positions + motor_offsets))
      //             - damping * current_dof_velocities
      for (size_t j = 0; j < dof; ++j) {
        double posWithOffset = currentPos(j) ;
        double q_des = joint_init_pos[j];          // 期望位置，相当于 joint_targets[j]
        double dq = currentVel(j);

        double tau = stiffness * (q_des - posWithOffset)
                   - damping   * dq;

        // 对应：torques = torques * motor_strengths
        tau *= 0.8;

        double limit = 45;
        if (tau >  limit) tau =  limit;
        if (tau < -limit) tau = -limit;

        torque(j) = tau;
      }

      for (size_t j = 0; j < leggedInterface_->getCentroidalModelInfo().actuatedDofNum; ++j) {
        hybridJointHandles_[j].setCommand(0,0, 0,0, torque(j));
      }
      // for (size_t j = 0; j < dof; ++j) {
      //   posDes(j) = hybridJointHandles_[j].getPosition();
      //   velDes(j) = 0.0;
      //   torque(j) = 0.0;
      // }
    } else {
      // ====== RL 控制分支：直接用 RL 输出的 12 维 torque ======
      if (!rlPosReceived_ || lastRlPos_.size() != dof) {
        ROS_WARN_THROTTLE(1.0, "[LeggedController] RL torque not ready, use zero torque.");
        torque.setZero();
      } else {
        double stiffness = 20.0;
        double damping = 0.5;
        ROS_WARN_THROTTLE(1.0, "[LeggedController] RL action invalid, holding current joint positions.");
        Eigen::VectorXd currentPos(dof);
        Eigen::VectorXd currentVel(dof);

        for (size_t j = 0; j < dof; ++j) {
          currentPos(j) = hybridJointHandles_[j].getPosition();
          currentVel(j) = hybridJointHandles_[j].getVelocity();
        }

        // 2) 按照 PD 控制计算力矩：
        //    torques = stiffness * (joint_targets - (current_dof_positions + motor_offsets))
        //             - damping * current_dof_velocities
        for (size_t j = 0; j < dof; ++j) {
          double posWithOffset = currentPos(j) ;
          double q_des = lastRlPos_[j];          // 期望位置，相当于 joint_targets[j]
          double dq = currentVel(j);

          double tau = stiffness * (q_des - posWithOffset)
                     - damping   * dq;

          // 对应：torques = torques * motor_strengths
          tau *= 0.8;

          double limit = 24;
          if (tau >  limit) tau =  limit;
          if (tau < -limit) tau = -limit;

          torque(j) = tau;
        }

        for (size_t j = 0; j < leggedInterface_->getCentroidalModelInfo().actuatedDofNum; ++j) {
          hybridJointHandles_[j].setCommand(0,0, 0,0, torque(j));
        }
        prevRlPos_ = lastRlPos_;
      }
      // pos / vel 指令：可以设为当前关节 + 0 速度
      // for (size_t j = 0; j < dof; ++j) {
      //     torque(j) = ;
      // }
      //todo 这里要加安全检查 增加pd控制
    }
  }
 

  // Visualization
  robotVisualizer_->update(currentObservation_, mpcMrtInterface_->getPolicy(), mpcMrtInterface_->getCommand());
  selfCollisionVisualization_->update(currentObservation_);

  // Publish the observation. Only needed for the command interface
  observationPublisher_.publish(ros_msg_conversions::createObservationMsg(currentObservation_));
}

void LeggedController::updateStateEstimation(const ros::Time& time, const ros::Duration& period) {
  vector_t jointPos(hybridJointHandles_.size()), jointVel(hybridJointHandles_.size());
  contact_flag_t contacts;
  Eigen::Quaternion<scalar_t> quat;
  contact_flag_t contactFlag;
  vector3_t angularVel, linearAccel;
  matrix3_t orientationCovariance, angularVelCovariance, linearAccelCovariance;

  for (size_t i = 0; i < hybridJointHandles_.size(); ++i) {
    jointPos(i) = hybridJointHandles_[i].getPosition();
    jointVel(i) = hybridJointHandles_[i].getVelocity();
  }

  for (size_t i = 0; i < contacts.size(); ++i) {

    // contactFlag[i] = contactHandles_[i].isContact();
    contactFlag[i] = leggedInterface_->getSwitchedModelReferenceManagerPtr()->getContactFlags(time.toSec()-controller_time)[i];

  }
  for (size_t i = 0; i < 4; ++i) {
    quat.coeffs()(i) = imuSensorHandle_.getOrientation()[i];
  }
  for (size_t i = 0; i < 3; ++i) {
    angularVel(i) = imuSensorHandle_.getAngularVelocity()[i];
    linearAccel(i) = imuSensorHandle_.getLinearAcceleration()[i];
  }
  for (size_t i = 0; i < 9; ++i) {
    orientationCovariance(i) = imuSensorHandle_.getOrientationCovariance()[i];
    angularVelCovariance(i) = imuSensorHandle_.getAngularVelocityCovariance()[i];
    linearAccelCovariance(i) = imuSensorHandle_.getLinearAccelerationCovariance()[i];
  }

  stateEstimate_->updateJointStates(jointPos, jointVel);
  stateEstimate_->updateContact(contactFlag);
  stateEstimate_->updateImu(quat, angularVel, linearAccel, orientationCovariance, angularVelCovariance, linearAccelCovariance);
  measuredRbdState_ = stateEstimate_->update(time, period);
  currentObservation_.time += period.toSec();
  scalar_t yawLast = currentObservation_.state(9);
  currentObservation_.state = rbdConversions_->computeCentroidalStateFromRbdModel(measuredRbdState_);
  currentObservation_.state(9) = yawLast + angles::shortest_angular_distance(yawLast, currentObservation_.state(9));
  currentObservation_.mode = stateEstimate_->getMode();


  // === 新增：把 jointPos / jointVel 存一下给 RL 用 ===
  lastJointPos_ = jointPos;
  lastJointVel_ = jointVel;

  // === 新增：把四元数转成 RPY 给 RL 用 ===
  Eigen::Quaterniond q_eig(quat.w(), quat.x(), quat.y(), quat.z());
  Eigen::Vector3d rpy = q_eig.toRotationMatrix().eulerAngles(0, 1, 2); // roll, pitch, yaw
  baseRPY_ = rpy;

}

LeggedController::~LeggedController() {
  controllerRunning_ = false;
  if (mpcThread_.joinable()) {
    mpcThread_.join();
  }
  std::cerr << "########################################################################";
  std::cerr << "\n### MPC Benchmarking";
  std::cerr << "\n###   Maximum : " << mpcTimer_.getMaxIntervalInMilliseconds() << "[ms].";
  std::cerr << "\n###   Average : " << mpcTimer_.getAverageInMilliseconds() << "[ms]." << std::endl;
  std::cerr << "########################################################################";
  std::cerr << "\n### WBC Benchmarking";
  std::cerr << "\n###   Maximum : " << wbcTimer_.getMaxIntervalInMilliseconds() << "[ms].";
  std::cerr << "\n###   Average : " << wbcTimer_.getAverageInMilliseconds() << "[ms].";
}

void LeggedController::setupLeggedInterface(const std::string& taskFile, const std::string& urdfFile, const std::string& referenceFile,
                                            bool verbose) {
  leggedInterface_ = std::make_shared<LeggedInterface>(taskFile, urdfFile, referenceFile);
  leggedInterface_->setupOptimalControlProblem(taskFile, urdfFile, referenceFile, verbose);
}

void LeggedController::setupMpc() {
  mpc_ = std::make_shared<SqpMpc>(leggedInterface_->mpcSettings(), leggedInterface_->sqpSettings(),
                                  leggedInterface_->getOptimalControlProblem(), leggedInterface_->getInitializer());
  rbdConversions_ = std::make_shared<CentroidalModelRbdConversions>(leggedInterface_->getPinocchioInterface(),
                                                                    leggedInterface_->getCentroidalModelInfo());

  const std::string robotName = "legged_robot";
  ros::NodeHandle nh;
  // Gait receiver
  auto gaitReceiverPtr =
      std::make_shared<GaitReceiver>(nh, leggedInterface_->getSwitchedModelReferenceManagerPtr()->getGaitSchedule(), robotName);
  // ROS ReferenceManager
  auto rosReferenceManagerPtr = std::make_shared<RosReferenceManager>(robotName, leggedInterface_->getReferenceManagerPtr());
  rosReferenceManagerPtr->subscribe(nh);
  mpc_->getSolverPtr()->addSynchronizedModule(gaitReceiverPtr);
  mpc_->getSolverPtr()->setReferenceManager(rosReferenceManagerPtr);
  observationPublisher_ = nh.advertise<ocs2_msgs::mpc_observation>(robotName + "_mpc_observation", 1);
}

void LeggedController::setupMrt() {
  mpcMrtInterface_ = std::make_shared<MPC_MRT_Interface>(*mpc_);
  mpcMrtInterface_->initRollout(&leggedInterface_->getRollout());
  mpcTimer_.reset();

  controllerRunning_ = true;
  mpcThread_ = std::thread([&]() {
    while (controllerRunning_) {
      try {
        executeAndSleep(
            [&]() {
              if (mpcRunning_) {
                mpcTimer_.startTimer();
                mpcMrtInterface_->advanceMpc();
                mpcTimer_.endTimer();
              }
            },
            leggedInterface_->mpcSettings().mpcDesiredFrequency_);
      } catch (const std::exception& e) {
        controllerRunning_ = false;
        ROS_ERROR_STREAM("[Ocs2 MPC thread] Error : " << e.what());
        stopRequest(ros::Time());
      }
    }
  });
  setThreadPriority(leggedInterface_->sqpSettings().threadPriority, mpcThread_);
}

void LeggedController::setupStateEstimate(const std::string& taskFile, bool verbose) {
  stateEstimate_ = std::make_shared<KalmanFilterEstimate>(leggedInterface_->getPinocchioInterface(),
                                                          leggedInterface_->getCentroidalModelInfo(), *eeKinematicsPtr_);
  dynamic_cast<KalmanFilterEstimate&>(*stateEstimate_).loadSettings(taskFile, verbose);
  currentObservation_.time = 0;
}

void LeggedCheaterController::setupStateEstimate(const std::string& /*taskFile*/, bool /*verbose*/) {
  stateEstimate_ = std::make_shared<FromTopicStateEstimate>(leggedInterface_->getPinocchioInterface(),
                                                            leggedInterface_->getCentroidalModelInfo(), *eeKinematicsPtr_);
}

}  // namespace legged

PLUGINLIB_EXPORT_CLASS(legged::LeggedController, controller_interface::ControllerBase)
PLUGINLIB_EXPORT_CLASS(legged::LeggedCheaterController, controller_interface::ControllerBase)
