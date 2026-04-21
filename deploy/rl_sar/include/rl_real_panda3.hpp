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


#include <sensor_msgs/image_encodings.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>


#include <csignal>
#include <vector>
#include <string>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <atomic>
#include <functional>
#include <chrono>
#include <memory>
#include <unordered_map>


#include "robot_msgs/msg/robot_command.hpp"
#include "robot_msgs/msg/robot_state.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_srvs/srv/empty.hpp>
#include <rcl_interfaces/srv/get_parameters.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include "matplotlibcpp.h"

// #define ROS_BAG_RECORDER

#ifdef ROS_BAG_RECORDER
#include "ros_bag_recorder.hpp"
#endif

#include "joystick_base.hpp"
#include "joystick_all.hpp"

#include "robot_msgs/msg/controller_state.hpp"



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
    // rclcpp::Publisher<robot_msgs::msg::RobotCommand>::SharedPtr robot_command_publisher;
    rclcpp::Publisher<robot_msgs::msg::RobotCommand>::SharedPtr robot_real_command_publisher;
    rclcpp::Subscription<robot_msgs::msg::RobotState>::SharedPtr robot_state_subscriber;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscriber;
    rclcpp::Client<rcl_interfaces::srv::GetParameters>::SharedPtr param_client;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_image_normalized_publisher;


    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr depth_array_subscriber;

    void CmdvelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void RobotStateCallback(const robot_msgs::msg::RobotState::SharedPtr msg);
    // void JoyCallback(const sensor_msgs::msg::Joy::SharedPtr msg);

    void InitJointStateMsg();

    rclcpp::Publisher<robot_msgs::msg::ControllerState>::SharedPtr controller_state_publisher;


    sensor_msgs::msg::JointState joint_state_msg;

    template<typename MsgT>
    void GenericCallback(const std::string& topic_name, const std::shared_ptr<MsgT>& msg) {
        auto info = std::static_pointer_cast<TopicInfo<MsgT>>(topics[topic_name]);

        // 1️⃣ 更新最新消息（线程安全）
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
    std::string depth_array_topic_name;




    // others
    std::string gazebo_model_name;
    std::map<std::string, float> joint_positions;
    std::map<std::string, float> joint_velocities;
    std::map<std::string, float> joint_efforts;

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    void publishMapToBase(double x, double y, double yaw, double qx, double qy, double qz, double qw);


#ifdef ROS_BAG_RECORDER
    std::unique_ptr<RosbagRecorder> rosbag_recorder;
#endif

    std::shared_ptr<joystick_base> joystick; // joystick pointer

    std::vector<float> depth_image_to_vector(const std::vector<uint8_t>& data,int src_width, int src_height, int width, int height);

    void printDepthAsciiNormalized(const float* depth,int width,int height,int stride_x = 1,int stride_y = 2);
    sensor_msgs::msg::Image depthVectorToGrayImage(const std::vector<float>& depth_vec,int width,int height);

    std::vector<float> rotate180FloatImage(const std::vector<float>& src,int width,int height);

};

#endif // RL_REAL_HPP
