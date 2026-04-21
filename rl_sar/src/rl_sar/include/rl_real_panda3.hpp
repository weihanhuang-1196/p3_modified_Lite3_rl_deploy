/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RL_REAL_HPP
#define RL_REAL_HPP

// #define PLOT
// #define CSV_LOGGER

#include "rl_sdk.hpp"
#include "observation_buffer.hpp"
#include "inference_runtime.hpp"
#include "loop.hpp"
#include "fsm_all.hpp"

#include <csignal>
#include <vector>
#include <string>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>


#include "robot_msgs/msg/robot_command.hpp"
#include "robot_msgs/msg/robot_state.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_srvs/srv/empty.hpp>
#include <rcl_interfaces/srv/get_parameters.hpp>

#include "matplotlibcpp.h"

#define ROS_BAG_RECORDER

#ifdef ROS_BAG_RECORDER
#include "ros_bag_recorder.hpp"
#endif

#include "joystick_base.hpp"
#include "joystick_all.hpp"

#include "ldelay_monitor_macros.hpp"



namespace plt = matplotlibcpp;

// 话题信息基类，包含超时机制和回调扩展
struct TopicInfoBase {
    double timeout_sec; // 超时时间
    std::atomic<std::chrono::steady_clock::time_point> last_time;
    std::function<void()> extra_callback; // 回调扩展
    virtual ~TopicInfoBase() = default;
};

template<typename MsgT>
struct TopicInfo : public TopicInfoBase {
    std::shared_ptr<MsgT> latest_msg{nullptr};
};




class RL_Real : public RL
{
public:
    RL_Real(int argc, char **argv);
    ~RL_Real();

    std::shared_ptr<rclcpp::Node> ros2_node;

private:

    // rl functions
    std::vector<float> Forward() override;
    void GetState(RobotState<float> *state) override;
    void SetCommand(const RobotCommand<float> *command) override;
    void RunModel();
    void RobotControl();

    // loop
    std::shared_ptr<LoopFunc> loop_keyboard;
    std::shared_ptr<LoopFunc> loop_control;
    std::shared_ptr<LoopFunc> loop_rl;
    std::shared_ptr<LoopFunc> loop_plot;

    // plot
    const int plot_size = 100;
    std::vector<int> plot_t;
    std::vector<std::vector<float>> plot_real_joint_pos, plot_target_joint_pos;
    void Plot();

    // ros interface
    std::string ros_namespace;
    geometry_msgs::msg::Twist cmd_vel;
    sensor_msgs::msg::Joy joy_msg;
    robot_msgs::msg::RobotCommand robot_command_publisher_msg;
    robot_msgs::msg::RobotState robot_state_subscriber_msg;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscriber;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_subscriber;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscriber;
    rclcpp::Publisher<robot_msgs::msg::RobotCommand>::SharedPtr robot_command_publisher;
    rclcpp::Publisher<robot_msgs::msg::RobotCommand>::SharedPtr robot_real_command_publisher;
    rclcpp::Subscription<robot_msgs::msg::RobotState>::SharedPtr robot_state_subscriber;
    rclcpp::Client<rcl_interfaces::srv::GetParameters>::SharedPtr param_client;
    void CmdvelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void RobotStateCallback(const robot_msgs::msg::RobotState::SharedPtr msg);
    void JoyCallback(const sensor_msgs::msg::Joy::SharedPtr msg);

    // others
    std::string gazebo_model_name;
    std::map<std::string, float> joint_positions;
    std::map<std::string, float> joint_velocities;
    std::map<std::string, float> joint_efforts;

    sensor_msgs::msg::Imu imu_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;


#ifdef ROS_BAG_RECORDER
    std::unique_ptr<RosbagRecorder> rosbag_recorder;
#endif



    template<typename MsgT>
    void GenericCallback(const std::string& topic_name, const std::shared_ptr<MsgT>& msg) {
        auto info = std::static_pointer_cast<TopicInfo<MsgT>>(topics[topic_name]);

        // 1️⃣ 更新最新消息（线程安全）
        // std::atomic_store(&info->latest_msg, msg);
        std::atomic_store_explicit(&info->latest_msg, msg, std::memory_order_release);

        // 2️⃣ 更新时间戳
        info->last_time.store(std::chrono::steady_clock::now(), std::memory_order_release);

        // 3️⃣ 执行扩展操作（额外逻辑）
        if (info->extra_callback) {
            info->extra_callback();
        }
    };

    void CheckTimeouts();
    void InitTopics();
    std::unordered_map<std::string, std::shared_ptr<TopicInfoBase>> topics;

    std::string cmd_topic_name;
    std::string joy_topic_name;
    std::string imu_topic_name;
    std::string robot_state_topic_name;
    std::string image_topic_name;

    std::shared_ptr<joystick_base> joystick; // joystick pointer





};

#endif // RL_REAL_HPP
