/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_real_panda3.hpp"

RL_Real::RL_Real(int argc, char **argv)
{
    ros2_node = std::make_shared<rclcpp::Node>("rl_sim_node");
    this->ang_vel_axis = "body";
    this->ros_namespace = ros2_node->get_namespace();
    // get params from param_node
    param_client = ros2_node->create_client<rcl_interfaces::srv::GetParameters>("/param_node/get_parameters");
    while (!param_client->wait_for_service(std::chrono::seconds(1)))
    {
        if (!rclcpp::ok()) {
            std::cout << LOGGER::ERROR << "Interrupted while waiting for param_node service. Exiting." << std::endl;
            return;
        }
        std::cout << LOGGER::WARNING << "Waiting for param_node service to be available..." << std::endl;
    }
    auto request = std::make_shared<rcl_interfaces::srv::GetParameters::Request>();
    request->names = {"robot_name", "gazebo_model_name"};
    // Use a timeout for the future
    auto future = param_client->async_send_request(request);
    auto status = rclcpp::spin_until_future_complete(ros2_node->get_node_base_interface(), future, std::chrono::seconds(5));
    if (status == rclcpp::FutureReturnCode::SUCCESS)
    {
        auto result = future.get();
        if (result->values.size() < 2)
        {
            std::cout << LOGGER::ERROR << "Failed to get all parameters from param_node" << std::endl;
        }
        else
        {
            this->robot_name = result->values[0].string_value;
            this->gazebo_model_name = result->values[1].string_value;
            std::cout << LOGGER::INFO << "Get param robot_name: " << this->robot_name << std::endl;
            std::cout << LOGGER::INFO << "Get param gazebo_model_name: " << this->gazebo_model_name << std::endl;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "Failed to call param_node service" << std::endl;
    }

    // read params from yaml
    this->ReadYaml(this->robot_name, "base.yaml");

    this->config_name = this->params.Get<std::string>("algorithm");

        // init joystick
    this->joystick = JoystickManager::GetInstance().CreateJoystick(
        this->params.Get<std::string>("joystick_type"),
        *this
    );
    if(!this->joystick)
    {
        std::cout << LOGGER::ERROR << "[Joystick] No joystick registered for type: " 
                  << this->params.Get<std::string>("joystick_type") << std::endl;
        return;
    }


    // auto load FSM by robot_name
    if (FSMManager::GetInstance().IsTypeSupported(this->robot_name))
    {
        auto fsm_ptr = FSMManager::GetInstance().CreateFSM(this->robot_name, this);
        if (fsm_ptr)
        {
            this->fsm = *fsm_ptr;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "[FSM] No FSM registered for robot: " << this->robot_name << std::endl;
    }

    // init robot

    this->robot_command_publisher_msg.motor_command.resize(this->params.Get<int>("num_of_dofs"));
    this->robot_state_subscriber_msg.motor_state.resize(this->params.Get<int>("num_of_dofs"));
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();
    this->InitJointStateMsg();

    // publisher
    // this->robot_command_publisher = ros2_node->create_publisher<robot_msgs::msg::RobotCommand>(
    //     this->ros_namespace + "robot_joint_controller/command", rclcpp::SystemDefaultsQoS());

    InitTopics();

    // subscriber
    // this->cmd_vel_subscriber = ros2_node->create_subscription<geometry_msgs::msg::Twist>(
    //     "/cmd_vel", rclcpp::SystemDefaultsQoS(),
    //     [this] (const geometry_msgs::msg::Twist::SharedPtr msg) {this->CmdvelCallback(msg);}
    // );
    // this->joy_subscriber = ros2_node->create_subscription<sensor_msgs::msg::Joy>(
    //     "/Devices/joy", rclcpp::SystemDefaultsQoS(),
    //     [this] (const sensor_msgs::msg::Joy::SharedPtr msg) {this->joystick->JoyCallback(msg);}
    // );
    // this->robot_state_subscriber = ros2_node->create_subscription<robot_msgs::msg::RobotState>(
    //     this->ros_namespace + "rl_sar/Robot_State", rclcpp::SystemDefaultsQoS(),
    //     [this] (const robot_msgs::msg::RobotState::SharedPtr msg) {this->RobotStateCallback(msg);}
    // );


    // loop
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("control_dt"), std::bind(&RL_Real::RobotControl, this), std::vector<int>{5});
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Real::RunModel, this),std::vector<int>{4});
    this->loop_control->start();
    this->loop_rl->start();

    // keyboard
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Real::KeyboardInterface, this));
    this->loop_keyboard->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    this->plot_target_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.001, std::bind(&RL_Real::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif

    std::cout << LOGGER::INFO << "RL_Real start" << std::endl;

#ifdef ROS_BAG_RECORDER
    rosbag_recorder = std::make_unique<RosbagRecorder>(
        this->params.Get<std::string>("rosbag_save_path"),   // 保存路径
        this->params.Get<std::string>("rosbag_save_name")       // 包名前缀
    );
#endif

}

RL_Real::~RL_Real()
{
    this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Real exit" << std::endl;
}

void RL_Real::publishMapToBase(double x, double y, double yaw, double qx, double qy, double qz, double qw)
{
    geometry_msgs::msg::TransformStamped tf;

    tf.header.stamp = ros2_node->now();
    tf.header.frame_id = "map";
    tf.child_frame_id = "base";

    tf.transform.translation.x = x;
    tf.transform.translation.y = y;
    tf.transform.translation.z = yaw;

    // tf2::Quaternion q;
    // q.setRPY(0.0, 0.0, yaw);
    tf.transform.rotation.x = qx;
    tf.transform.rotation.y = qy;
    tf.transform.rotation.z = qz;
    tf.transform.rotation.w = qw;

    tf_broadcaster_->sendTransform(tf);
}

void RL_Real::InitJointStateMsg()
{
    auto dofs = this->params.Get<int>("num_of_dofs");
    this->joint_state_msg.name.resize(dofs);
    this->joint_state_msg.position.resize(dofs);
    this->joint_state_msg.velocity.resize(dofs);
    this->joint_state_msg.effort.resize(dofs);
    for(int i=0;i<dofs;i++){
        this->joint_state_msg.name[i] = this->params.Get<std::vector<std::string>>("joint_names")[i];
    }
}


void RL_Real::GetState(RobotState<float> *state)
{

    auto info2 = std::static_pointer_cast<TopicInfo<robot_msgs::msg::RobotState>>(topics[robot_state_topic_name.c_str()]);
    auto robot_state_msg = std::atomic_load_explicit(&info2->latest_msg, std::memory_order_acquire);

    const auto orientation = robot_state_msg->imu.quaternion;
    const auto angular_velocity = robot_state_msg->imu.gyroscope;

    state->imu.quaternion[0] = orientation[0];
    state->imu.quaternion[1] = orientation[1];
    state->imu.quaternion[2] = orientation[2];
    state->imu.quaternion[3] = orientation[3];

    state->imu.gyroscope[0] = angular_velocity[0];
    state->imu.gyroscope[1] = angular_velocity[1];
    state->imu.gyroscope[2] = angular_velocity[2];
    
    auto dofs = this->params.Get<int>("num_of_dofs");
    auto joint_map = this->params.Get<std::vector<int>>("joint_mapping");

    
    for (int i = 0; i < dofs; ++i)
    {
        state->motor_state.q[i] = robot_state_msg->motor_state[joint_map[i]].q;
        state->motor_state.dq[i] = robot_state_msg->motor_state[joint_map[i]].dq;
        state->motor_state.tau_est[i] = robot_state_msg->motor_state[joint_map[i]].tau_est;
        state->motor_state.status_word[i] =  robot_state_msg->motor_state[joint_map[i]].status_word;

        this->joint_state_msg.position[joint_map[i]] = state->motor_state.q[i];
        this->joint_state_msg.velocity[joint_map[i]] = state->motor_state.dq[i];
        this->joint_state_msg.effort[joint_map[i]] = state->motor_state.tau_est[i];
    }

    this->joint_state_msg.header.stamp = this->ros2_node->now();
    joint_state_publisher_->publish(this->joint_state_msg);

}

void RL_Real::SetCommand(const RobotCommand<float> *command)
{
    int motor_enable = this->motor_enabled ? 1 : 0;
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].q = command->motor_command.q[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq = command->motor_command.dq[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].kp = command->motor_command.kp[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].kd = command->motor_command.kd[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau = command->motor_command.tau[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].mode = motor_enable;
    }
    this->robot_command_publisher_msg.t_id += 0.001f;

    // this->robot_command_publisher->publish(this->robot_command_publisher_msg);
    this->robot_real_command_publisher->publish(this->robot_command_publisher_msg);
}

void RL_Real::RobotControl()
{
    CheckTimeouts();
    {
        auto info = std::static_pointer_cast<TopicInfo<robot_msgs::msg::RobotState>>(topics[robot_state_topic_name.c_str()]);
        auto robot_state_msg = std::atomic_load_explicit(&info->latest_msg, std::memory_order_acquire);
        if(!robot_state_msg) return;
    }

    this->GetState(&this->robot_state);
    this->publishMapToBase(0,0,0, 
                            this->robot_state.imu.quaternion[1], 
                            this->robot_state.imu.quaternion[2], 
                            this->robot_state.imu.quaternion[3], 
                            this->robot_state.imu.quaternion[0]);

    this->StateController(&this->robot_state, &this->robot_command);

    robot_msgs::msg::ControllerState controller_state_msg;
    controller_state_msg.header.stamp = ros2_node->now();
    controller_state_msg.fsm_state = this->fsm.current_state_->GetStateName();
    controller_state_msg.control_mode = this->control.navigation_mode;
    controller_state_msg.enable = this->motor_enabled;
    this->controller_state_publisher->publish(controller_state_msg);

    this->control.ClearInput();

    this->SetCommand(&this->robot_command);
}


void RL_Real::CmdvelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
    this->cmd_vel = *msg;
}

void RL_Real::InitTopics() {

    
    this->robot_real_command_publisher = ros2_node->create_publisher<robot_msgs::msg::RobotCommand>(
    this->ros_namespace + "rl_sar/Robot_Command", rclcpp::SystemDefaultsQoS());


    this->joint_state_publisher_ = ros2_node->create_publisher<sensor_msgs::msg::JointState>(
        "/joint_states", rclcpp::SystemDefaultsQoS());

    this->tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(ros2_node);

    this->controller_state_publisher = this->ros2_node->create_publisher<robot_msgs::msg::ControllerState>(
        this->ros_namespace + "controller_state", rclcpp::SystemDefaultsQoS());


   this->depth_image_normalized_publisher = this->ros2_node->create_publisher<sensor_msgs::msg::Image>(
        this->ros_namespace + "depth_image_normalized", rclcpp::SystemDefaultsQoS());




    auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
    qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);

    // 初始化 /cmd_vel 话题信息
    this->cmd_topic_name = "/cmd_vel";
    auto cmd_vel_info = std::make_shared<TopicInfo<geometry_msgs::msg::Twist>>();
    cmd_vel_info->timeout_sec = 2.0;
    cmd_vel_info->last_time.store(std::chrono::steady_clock::now(), std::memory_order_release);
    topics[this->cmd_topic_name] = cmd_vel_info;

    cmd_vel_subscriber = ros2_node->create_subscription<geometry_msgs::msg::Twist>(
        this->cmd_topic_name, qos,
        [this](const geometry_msgs::msg::Twist::SharedPtr msg){ GenericCallback(this->cmd_topic_name, msg); }
    );


    // 初始化 /joy 话题信息
    this->joy_topic_name = "/Devices/joy";
    auto joy_info = std::make_shared<TopicInfo<sensor_msgs::msg::Joy>>();
    joy_info->timeout_sec = 0.5;
    joy_info->last_time.store(std::chrono::steady_clock::now(), std::memory_order_release);
    // 可选：设置额外回调
    joy_info->extra_callback = [this, joy_info] () {
        auto msg = std::atomic_load_explicit(&joy_info->latest_msg, std::memory_order_acquire);
        this->joystick->JoyCallback(msg);
    };
    topics[this->joy_topic_name] = joy_info;
    joy_subscriber = ros2_node->create_subscription<sensor_msgs::msg::Joy>(
        this->joy_topic_name, rclcpp::SystemDefaultsQoS(),
        [this](const sensor_msgs::msg::Joy::SharedPtr msg){ GenericCallback(this->joy_topic_name, msg); }
    );


    // 初始化 /imu 话题信息
    // this->imu_topic_name = "/imu";
    // auto imu_info = std::make_shared<TopicInfo<sensor_msgs::msg::Imu>>();
    // imu_info->timeout_sec = 0.5;
    // imu_info->last_time.store(std::chrono::steady_clock::now(), memory_order_release);
    // topics[this->imu_topic_name] = imu_info;
    // gazebo_imu_subscriber = ros2_node->create_subscription<sensor_msgs::msg::Imu>(
    //     this->imu_topic_name, rclcpp::SystemDefaultsQoS(),
    //     [this](const sensor_msgs::msg::Imu::SharedPtr msg){ GenericCallback(this->imu_topic_name, msg); }
    // );


    // 初始化 /robot_joint_controller/state 话题信息
    this->robot_state_topic_name = this->ros_namespace + "rl_sar/Robot_State";
    auto robot_state_info = std::make_shared<TopicInfo<robot_msgs::msg::RobotState>>();
    robot_state_info->timeout_sec = 0.05;
    robot_state_info->last_time.store(std::chrono::steady_clock::now(), std::memory_order_release);
    topics[this->robot_state_topic_name] = robot_state_info;
    robot_state_subscriber = ros2_node->create_subscription<robot_msgs::msg::RobotState>(
        this->robot_state_topic_name, rclcpp::SystemDefaultsQoS(),
        [this](const robot_msgs::msg::RobotState::SharedPtr msg){ GenericCallback(this->robot_state_topic_name, msg); }
    );

    // this->image_topic_name = "/Devices/camera/depth/image_rect_raw";
    // auto image_info = std::make_shared<TopicInfo<sensor_msgs::msg::Image>>();
    // image_info->timeout_sec = 1.0;
    // image_info->last_time.store(std::chrono::steady_clock::now(), std::memory_order_relaxed);
    // image_info->extra_callback = [this, image_info] () {
    //     auto msg = std::atomic_load_explicit(&image_info->latest_msg, std::memory_order_acquire);
    //     auto depth_image = depth_image_to_vector(msg->data,msg->width,msg->height, this->image_width, this->image_height);
    //     sensor_msgs::msg::Image depth_image_ros = depthVectorToGrayImage(depth_image, this->image_width, this->image_height);
    //     this->depth_image_normalized_publisher->publish(depth_image_ros);
    //     std::atomic_store_explicit(&this->depth_image_ptr, std::make_shared<std::vector<float>>(std::move(depth_image)), std::memory_order_release);
    // };
    // topics[this->image_topic_name] = image_info;
    // image_subscriber = ros2_node->create_subscription<sensor_msgs::msg::Image>(
    //     this->image_topic_name, rclcpp::SystemDefaultsQoS(),
    //     [this](const sensor_msgs::msg::Image::SharedPtr msg){ GenericCallback(this->image_topic_name, msg); }
    // );


    this->depth_array_topic_name = "/forward_depth_image";
    auto depth_info = std::make_shared<TopicInfo<std_msgs::msg::Float32MultiArray>>();
    depth_info->timeout_sec = 1.0;
    depth_info->last_time.store(std::chrono::steady_clock::now(), std::memory_order_relaxed);
    depth_info->extra_callback = [this, depth_info] () {
        auto msg = std::atomic_load_explicit(&depth_info->latest_msg, std::memory_order_acquire);
        // std::vector<float> rotated_data = msg->data;
        // std::reverse(rotated_data.begin(), rotated_data.end());
        // std::vector<float> rotated_data = rotate180FloatImage(msg->data,this->image_width,this->image_height);
        // auto depth_image = depth_image_to_vector(msg->data,msg->width,msg->height, this->image_width, this->image_height);
        sensor_msgs::msg::Image depth_image_ros = depthVectorToGrayImage(msg->data, this->image_width, this->image_height);
        this->depth_image_normalized_publisher->publish(depth_image_ros);
        std::atomic_store_explicit(&this->depth_image_ptr, std::make_shared<std::vector<float>>(std::move(msg->data)), std::memory_order_release);
    };
    topics[this->depth_array_topic_name] = depth_info;
    depth_array_subscriber = ros2_node->create_subscription<std_msgs::msg::Float32MultiArray>(
        this->depth_array_topic_name, rclcpp::SystemDefaultsQoS(),
        [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg){ GenericCallback(this->depth_array_topic_name, msg); }
    );

    



}

std::vector<float> RL_Real::rotate180FloatImage(const std::vector<float>& src,int width,int height)
{
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Invalid image size");
    }

    if (static_cast<int>(src.size()) != width * height) {
        throw std::runtime_error("Input data size does not match width * height");
    }

    std::vector<float> dst(src.size());

    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int src_idx = r * width + c;
            int dst_idx = (height - 1 - r) * width + (width - 1 - c);
            dst[dst_idx] = src[src_idx];
        }
    }

    return dst;
}

void RL_Real::CheckTimeouts() {
    auto now = std::chrono::steady_clock::now();

    for (auto& kv : topics) {
        auto& info = kv.second;
        auto dt = std::chrono::duration<double>(now - info->last_time.load(std::memory_order_acquire)).count();
        if (dt > info->timeout_sec) {
            // std::cout << LOGGER::INFO << "RL_Sim start" << std::endl;
            // RCLCPP_WARN(node->get_logger(), "Topic %s timeout %.3f s", kv.first.c_str(), dt);
            // std::cout << LOGGER::INFO  << kv.first << " timeout " << dt << " s" << std::endl;

            // 超时处理，例如停机
            if (kv.first == "/cmd_vel") {
                geometry_msgs::msg::Twist stop_cmd{};
                auto twist_info = std::static_pointer_cast<TopicInfo<geometry_msgs::msg::Twist>>(info);
                std::atomic_store_explicit(&twist_info->latest_msg, std::make_shared<geometry_msgs::msg::Twist>(stop_cmd), std::memory_order_release);
            }
            if (kv.first == "/joy") {
                // 处理 joystick 超时，例如清除输入状态
                // this->control.x = 0.0f;
                // this->control.y = 0.0f;
                // this->control.yaw = 0.0f;
                // this->joystick->ClearInput();
            }
            if (kv.first == "/imu") {
                // 处理 IMU 超时，例如设置默认状态
                // this->gazebo_imu = sensor_msgs::msg::Imu();
            }
             if (kv.first == this->robot_state_topic_name) {
                // 处理 robot state 超时，例如设置默认状态
                // this->robot_state_subscriber_msg = robot_msgs::msg::RobotState();
                auto info = std::static_pointer_cast<TopicInfo<robot_msgs::msg::RobotState>>(topics[robot_state_topic_name.c_str()]);
                auto robot_state_msg = std::atomic_load_explicit(&info->latest_msg, std::memory_order_acquire);
                if(robot_state_msg)
                {
                    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
                    {
                          robot_state_msg->motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].status_word = 0; // 设置状态字为0，表示无效或未连接
                    }
                    std::atomic_store_explicit(&info->latest_msg, robot_state_msg, std::memory_order_release);
                }

            }
            if (kv.first == this->image_topic_name ){

            }
            
        }
    }
}



void RL_Real::RobotStateCallback(const robot_msgs::msg::RobotState::SharedPtr msg)
{
    this->robot_state_subscriber_msg = *msg;

}


sensor_msgs::msg::Image RL_Real::depthVectorToGrayImage(
    const std::vector<float>& depth_vec,
    int width,
    int height)
{

    cv::Mat depth_mat(height, width, CV_32FC1, const_cast<float*>(depth_vec.data()));
    cv::Mat depth_display;
    depth_mat.convertTo(depth_display, CV_8UC1, 255.0, 127); 

    sensor_msgs::msg::Image img;

    img.header.stamp = this->ros2_node->now();
    img.header.frame_id = "camera_frame";

    img.height = height;
    img.width  = width;
    img.encoding = "mono8"; 
    img.is_bigendian = false;
    img.step = width;

    img.data.resize(width * height);

    std::memcpy(img.data.data(), depth_display.data, img.data.size());

    return img;
}



std::vector<float> RL_Real::depth_image_to_vector(const std::vector<uint8_t>& data,int src_width, int src_height, int dst_width, int dst_height)
{
    const float min_depth = this->znear;
    const float max_depth = this->zfar;


    const uint16_t* depth_data = reinterpret_cast<const uint16_t*>(data.data());
    cv::Mat src_depth(src_height, src_width, CV_16UC1, const_cast<uint16_t*>(depth_data));


        // 原始 FOV
    float FOV_h_src = 85.2f;
    float FOV_v_src = 58.0f;
    float FOV_h_dst = 58.0f;
    float FOV_v_dst = 58.0f;

   // 计算裁剪后像素
    int new_width = int(src_width * tan(FOV_h_dst * M_PI/360.0f) / tan(FOV_h_src * M_PI/360.0f));
    int new_height = src_height; // 垂直保持

    int crop_x = (src_width - new_width)/2;
    int crop_y = 0;


    cv::Rect roi(crop_x, crop_y, new_width, new_height);
    cv::Mat cropped = src_depth(roi).clone();


    // 2. Resize 到目标大小
    cv::Mat resized_depth;
    cv::resize(cropped, resized_depth, cv::Size(dst_width, dst_height), 0, 0, cv::INTER_NEAREST);

    // 3. 转换为 float，单位：米
    cv::Mat depth_meters;
    resized_depth.convertTo(depth_meters, CV_32F, 0.001f); // 16-bit毫米 -> 米

    for (int i = 0; i < depth_meters.rows; i++) {
        float* row_ptr = depth_meters.ptr<float>(i);
        for (int j = 0; j < depth_meters.cols; j++) {
            float& d = row_ptr[j];
            if (std::isnan(d)) d = max_depth;               // 无效/未命中
            else if (d == std::numeric_limits<float>::infinity()) d = max_depth; 
            else if (d == -std::numeric_limits<float>::infinity()) d = min_depth; 
            else if (d <= 0.0f) d = min_depth;              // 认为非正值代表太近
            else d = std::clamp(d, min_depth, max_depth);
        }
    }


    // 4. 裁剪视深 [0, 2.0]
    cv::threshold(depth_meters, depth_meters, max_depth, max_depth, cv::THRESH_TRUNC); // 上限2.0m
    cv::threshold(depth_meters, depth_meters, min_depth, min_depth, cv::THRESH_TOZERO); // 下限0m

 
    cv::Mat depth_filtered;
    cv::medianBlur(depth_meters, depth_filtered, 5); 

    cv::Mat morph_depth;

    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(3,3)
    );

    cv::morphologyEx(
        depth_filtered,
        morph_depth,
        cv::MORPH_CLOSE,
        kernel
    );



    // 5. 归一化到 [-0.5, 0.5]
    depth_filtered = depth_filtered / 2.0f - 0.5f;

    // 6. 转成 std::vector<float>
    std::vector<float> depth_vec;
    depth_vec.assign((float*)depth_filtered.datastart, (float*)depth_filtered.dataend);

    return depth_vec;
}


void RL_Real::printDepthAsciiNormalized(
    const float* depth,
    int width,
    int height,
    int stride_x,
    int stride_y
)
{
    const std::string levels = " .:-=+*#%@"; // 10级灰度

    const float min_d = -0.5f;
    const float max_d =  0.5f;
    const float range = max_d - min_d;

    for (int y = 0; y < height; y += stride_y)
    {
        for (int x = 0; x < width; x += stride_x)
        {
            float d = depth[y * width + x];

            // clamp 防止越界（很重要）
            d = std::clamp(d, min_d, max_d);

            // 归一化到 [0,1]
            float norm = (d - min_d) / range;

            int idx = static_cast<int>(norm * (levels.size() - 1));
            idx = std::clamp(idx, 0, (int)levels.size() - 1);

            std::cout << levels[idx];
        }
        std::cout << "\n";
    }

    std::cout << std::flush;
}






void RL_Real::RunModel()
{
    if (this->rl_init_done)
    {
        this->episode_length_buf += 1;
        this->obs.ang_vel = this->robot_state.imu.gyroscope;
        if(this->config_name == "np3o")
        {
            if(this->current_rl_fsm_name.compare("RLFSMStateRLStand") == 0)
                this->obs.commands = {0.0f, 0.0f, 0.0f, 0.0f,this->control.stand, 0.0f,0.0f,0.0f,0.0f,0.0f};
            else if(this->current_rl_fsm_name.compare("RLFSMStateRLCrouch") == 0)
                this->obs.commands = {(float)this->control.x, (float)this->control.y, (float)this->control.yaw, this->control.height, 0.0f, 0.0f,0.0f,0.0f,0.0f,0.0f};
            else
                this->obs.commands = {(float)this->control.x, (float)this->control.y, (float)this->control.yaw,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
        }
        else
        {
            if(this->current_rl_fsm_name.compare("RLFSMStateRLStand") == 0)
                this->obs.commands = {0.0f, 0.0f, 0.0f, 0.0f,this->control.stand};
            else
                this->obs.commands = {this->control.x, this->control.y, this->control.yaw};
        }
        if (this->control.navigation_mode)
        {
            auto info = std::static_pointer_cast<TopicInfo<geometry_msgs::msg::Twist>>(topics[cmd_topic_name.c_str()]);
            auto cmd = std::atomic_load_explicit(&info->latest_msg, std::memory_order_acquire);
            if(cmd)
            {
                if(this->config_name == "np3o")
                {
                    if(this->current_rl_fsm_name.compare("RLFSMStateRLStand") == 0)
                        this->obs.commands = {0.0f, 0.0f, 0.0f, 0.0f,this->control.stand, 0.0f,0.0f,0.0f,0.0f,0.0f};
                    else if(this->current_rl_fsm_name.compare("RLFSMStateRLCrouch") == 0)
                        this->obs.commands = {(float)cmd->linear.x, (float)cmd->linear.y, (float)cmd->angular.z, 0.0f,0.0f, 0.0f,0.0f,0.0f,0.0f,0.0f};
                    else
                        this->obs.commands = {(float)cmd->linear.x, (float)cmd->linear.y, (float)cmd->angular.z,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
                }
                else
                    this->obs.commands = {(float)cmd->linear.x, (float)cmd->linear.y, (float)cmd->angular.z};
            }else
                this->obs.commands = {0.0f, 0.0f, 0.0f};

        }
        this->obs.base_quat = this->robot_state.imu.quaternion;
        this->obs.dof_pos = this->robot_state.motor_state.q;
        this->obs.dof_vel = this->robot_state.motor_state.dq;

        this->obs.actions = this->Forward();
        this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);
        if (!this->output_dof_pos.empty())
        {
            output_dof_pos_queue.push(this->output_dof_pos);
        }
        if (!this->output_dof_vel.empty())
        {
            output_dof_vel_queue.push(this->output_dof_vel);
        }
        if (!this->output_dof_tau.empty())
        {
            output_dof_tau_queue.push(this->output_dof_tau);
        }

        // this->TorqueProtect(this->output_dof_tau);
        // this->AttitudeProtect(this->robot_state.imu.quaternion, 75.0f, 75.0f);

#ifdef CSV_LOGGER
        std::vector<float> tau_est(this->params.Get<int>("num_of_dofs"), 0.0f);
        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            // tau_est[i] = this->joint_efforts[this->params.Get<std::vector<std::string>>("joint_controller_names")[i]];
            tau_est[i] = this->robot_state_subscriber_msg.motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau_est;
        }
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

std::vector<float> RL_Real::Forward()
{
    std::unique_lock<std::mutex> lock(this->model_mutex, std::try_to_lock);

    // If model is being reinitialized, return previous actions to avoid blocking
    if (!lock.owns_lock())
    {
        std::cout << LOGGER::WARNING << "Model is being reinitialized, using previous actions" << std::endl;
        return this->obs.actions;
    }

    std::vector<float> clamped_obs = this->ComputeObservation();
    std::vector<float> world_obs;

    if(this->config_name == "wmp")
    {
        world_obs = this->ComputeWorldObservation();
    }

    std::vector<float> actions;
    if (this->params.Get<std::vector<int>>("observations_history").size() != 0)
    {
        if(this->config_name == "np3o")
        {
            
            this->history_obs = this->history_obs_buf.get_obs_vec(this->params.Get<std::vector<int>>("observations_history"));
            actions = this->model->forward({clamped_obs,this->history_obs});
            this->history_obs_buf.insert(clamped_obs);
            
        }
        else if(this->config_name == "himloco")
        {
            this->history_obs_buf.insert(clamped_obs);
            this->history_obs = this->history_obs_buf.get_obs_vec(this->params.Get<std::vector<int>>("observations_history"));
            // auto start = std::chrono::steady_clock::now();
            actions = this->model->forward({this->history_obs});
            // auto end = std::chrono::steady_clock::now();
            // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            // std::cout << LOGGER::DEBUG << "Policy model forward time: " << duration.count() << " us" << std::endl;
        }
        else if(this->config_name == "wmp")
        {

            this->wm_action_history.erase(this->wm_action_history.begin(), this->wm_action_history.begin() + this->wm_action.size());
            this->wm_action_history.insert(this->wm_action_history.end(), this->wm_action.begin(), this->wm_action.end());
            this->wm_action = this->wm_action_history;

            if(global_counter % visual_update_interval == 0)
            {

                std::fill(this->input_image.begin(), this->input_image.end(), 0.0f);
                auto depth_image = std::atomic_load_explicit(&this->depth_image_ptr, std::memory_order_acquire);
                if(depth_image)
                {
                    this->wm_input_image = *depth_image;
                }else
                {
                    this->wm_input_image = std::vector<float>(this->image_width * this->image_height, 0.0f); // 如果没有图像数据，使用全零输入
                }

                if(this->pre_wm_image.empty())
                    this->input_image = this->wm_input_image; // 初始化前一帧图像
                else
                    this->input_image = this->pre_wm_image; // 使用上一帧图像作为输入

                // sensor_msgs::msg::Image image_mono8;
                // image_mono8.header.stamp = this->ros2_node->now();
                // image_mono8.header.frame_id = "camera_frame";  // 自定义
                // image_mono8.height = this->image_height;
                // image_mono8.width  = this->image_width;
                // image_mono8.encoding = "mono8";   // 灰度
                // image_mono8.is_bigendian = false;
                // image_mono8.step = this->image_width * sizeof(float);
                // image_mono8.data.resize(input_image.size() * sizeof(float));
                // std::memcpy(image_mono8.data.data(), input_image.data(), image_mono8.data.size());
                // this->depth_image_normalized_publisher->publish(image_mono8);
                // printDepthAsciiNormalized(input_image.data(), input_image.size(), input_image.size());
                // auto start = std::chrono::steady_clock::now();
                auto world_model_output = this->world_model->forward_world({world_obs, this->input_image, this->wm_logit, this->wm_stoch, this->wm_deter, this->wm_action, this->wm_is_first});
                // auto end = std::chrono::steady_clock::now();
                // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                // std::cout << LOGGER::DEBUG << "World model forward time: " << duration.count() << " us" << std::endl;
                this->wm_logit = std::move(world_model_output[0]);
                this->wm_stoch = std::move(world_model_output[1]);
                this->wm_deter = std::move(world_model_output[2]);
                this->wm_feature = std::move(world_model_output[3]);

                this->pre_wm_image = std::move(this->wm_input_image);
            }
            this->wm_is_first[0] = 0;

            this->history_obs_buf.insert(clamped_obs);
            this->history_obs = this->history_obs_buf.get_obs_vec(this->params.Get<std::vector<int>>("observations_history"));
            // auto start = std::chrono::steady_clock::now();
            actions = this->model->forward({this->obs.commands, this->history_obs, this->wm_feature});
            // auto end = std::chrono::steady_clock::now();
            // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            // std::cout << LOGGER::DEBUG << "Policy model forward time: " << duration.count() << " us" << std::endl;
            this->wm_action = actions;



            
        }
        
    }
    else
    {
        actions = this->model->forward({clamped_obs});
    }

    global_counter += 1;

    if (!this->params.Get<std::vector<float>>("clip_actions_upper").empty() && !this->params.Get<std::vector<float>>("clip_actions_lower").empty())
    {
        return clamp(actions, this->params.Get<std::vector<float>>("clip_actions_lower"), this->params.Get<std::vector<float>>("clip_actions_upper"));
    }
    else
    {
        return actions;
    }
}

void RL_Real::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
        this->plot_real_joint_pos[i].push_back(this->robot_state_subscriber_msg.motor_state[i].q);
        this->plot_target_joint_pos[i].push_back(this->robot_command_publisher_msg.motor_command[i].q);
        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.01);
}


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto rl_sar = std::make_shared<RL_Real>(argc, argv);
    rclcpp::spin(rl_sar->ros2_node);
    rclcpp::shutdown();
    return 0;
}
