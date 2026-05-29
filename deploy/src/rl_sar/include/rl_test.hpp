/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RL_Test_HPP
#define RL_Test_HPP

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

#define ROS_BAG_RECORDER

#ifdef ROS_BAG_RECORDER
#include "ros_bag_recorder.hpp"
#endif
#include <tf2_ros/transform_broadcaster.h>


#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

class RL_Test : public RL
{
public:
    RL_Test(int argc, char **argv);
    ~RL_Test();

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


    sensor_msgs::msg::Imu imu_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    void publishMapToBase(double x, double y, double yaw, double qx, double qy, double qz, double qw);

#ifdef ROS_BAG_RECORDER
    std::unique_ptr<RosbagRecorder> rosbag_recorder;
#endif

    // others
    std::string gazebo_model_name;
    std::map<std::string, float> joint_positions;
    std::map<std::string, float> joint_velocities;
    std::map<std::string, float> joint_efforts;


    float percent_pre_getup = 0.0f;
    float percent_getup = 0.0f;
    std::vector<float> pre_running_pos = {
        0.87, 1.36, -2.65,
        -0.87, 1.36, -2.65,
        0.87, 1.36, -2.65,
        -0.87, 1.36, -2.65,
        0.00, 0.00, 0.00, 0.00
    };
    float max_duration = 0.20;
    float min_duration = 0.02;
    float current_duration = 0.20;
    float gain = 0.005;


public:
    bool Interpolate(float& percent,
        const std::vector<float>& start_pos,
        const std::vector<float>& target_pos,
        float duration_seconds,
        std::vector<float> &output_dof_pos,
        bool use_fixed_gains = true
    );

};

#endif // RL_Test_HPP
