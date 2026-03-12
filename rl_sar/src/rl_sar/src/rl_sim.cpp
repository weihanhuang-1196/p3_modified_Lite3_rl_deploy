/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sim.hpp"

RL_Sim::RL_Sim(int argc, char **argv)
{
#if defined(USE_ROS1)
    this->ang_vel_axis = "world";
    ros::NodeHandle nh;
    nh.param<std::string>("ros_namespace", this->ros_namespace, "");
    nh.param<std::string>("robot_name", this->robot_name, "");
#elif defined(USE_ROS2)



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
#endif

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
#if defined(USE_ROS1)
    this->joint_publishers_commands.resize(this->params.Get<int>("num_of_dofs"));
#elif defined(USE_ROS2)
    this->robot_command_publisher_msg.motor_command.resize(this->params.Get<int>("num_of_dofs"));
    // this->robot_state_subscriber_msg.motor_state.resize(this->params.Get<int>("num_of_dofs"));
#endif
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();

#if defined(USE_ROS1)
    auto joint_controller_names_vec = this->params.Get<std::vector<std::string>>("joint_controller_names");  // avoid dangling reference
    this->StartJointController(this->ros_namespace, joint_controller_names_vec);
    // publisher
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        const std::string &joint_controller_name = joint_controller_names_vec[i];
        const std::string topic_name = this->ros_namespace + joint_controller_name + "/command";
        this->joint_publishers[joint_controller_name] =
            nh.advertise<robot_msgs::MotorCommand>(topic_name, 10);
    }

    // subscriber
    this->cmd_vel_subscriber = nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 10, &RL_Sim::CmdvelCallback, this);
    // this->joy_subscriber = nh.subscribe<sensor_msgs::Joy>("/joy", 10, &RL_Sim::JoyCallback, this);
    this->model_state_subscriber = nh.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states", 10, &RL_Sim::ModelStatesCallback, this);
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        const std::string &joint_controller_name = joint_controller_names_vec[i];
        const std::string topic_name = this->ros_namespace + joint_controller_name + "/state";
        this->joint_subscribers[joint_controller_name] =
            nh.subscribe<robot_msgs::MotorState>(topic_name, 10,
                [this, joint_controller_name](const robot_msgs::MotorState::ConstPtr &msg)
                {
                    this->JointStatesCallback(msg, joint_controller_name);
                }
            );
        this->joint_positions[joint_controller_name] = 0.0f;
        this->joint_velocities[joint_controller_name] = 0.0f;
        this->joint_efforts[joint_controller_name] = 0.0f;
    }

    // service
    nh.param<std::string>("gazebo_model_name", this->gazebo_model_name, "");
    this->gazebo_pause_physics_client = nh.serviceClient<std_srvs::Empty>("/gazebo/pause_physics");
    this->gazebo_unpause_physics_client = nh.serviceClient<std_srvs::Empty>("/gazebo/unpause_physics");
    this->gazebo_reset_world_client = nh.serviceClient<std_srvs::Empty>("/gazebo/reset_world");
#elif defined(USE_ROS2)
    this->StartJointController(this->ros_namespace, this->params.Get<std::vector<std::string>>("joint_names"));
    // publisher
    this->robot_command_publisher = ros2_node->create_publisher<robot_msgs::msg::RobotCommand>(
        this->ros_namespace + "robot_joint_controller/command", rclcpp::SystemDefaultsQoS());

    this->actions_publisher = ros2_node->create_publisher<robot_msgs::msg::Actions>(
        this->ros_namespace + "actions", rclcpp::SystemDefaultsQoS());

    this->grid_publisher = this->ros2_node->create_publisher<nav_msgs::msg::OccupancyGrid>("grid", 1);
    this->tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(ros2_node);
    this->marker_publisher = this->ros2_node->create_publisher<visualization_msgs::msg::Marker>("visualization_marker", 10);
    this->fsm_state_publisher = this->ros2_node->create_publisher<std_msgs::msg::String>(
        this->ros_namespace + "fsm_state", rclcpp::SystemDefaultsQoS());

    this->controller_state_publisher = this->ros2_node->create_publisher<robot_msgs::msg::ControllerState>(
        this->ros_namespace + "controller_state", rclcpp::SystemDefaultsQoS());


    InitTopics();

    // service
    this->gazebo_pause_physics_client = ros2_node->create_client<std_srvs::srv::Empty>("/pause_physics");
    this->gazebo_unpause_physics_client = ros2_node->create_client<std_srvs::srv::Empty>("/unpause_physics");
    this->gazebo_reset_world_client = ros2_node->create_client<std_srvs::srv::Empty>("/reset_world");

    auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
    auto result = this->gazebo_reset_world_client->async_send_request(empty_request);
#endif

    // loop
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Sim::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Sim::RunModel, this));
    this->loop_control->start();
    this->loop_rl->start();

    // keyboard
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Sim::KeyboardInterface, this));
    this->loop_keyboard->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    this->plot_target_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.001, std::bind(&RL_Sim::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif


#ifdef MOTOR_POLICY
    try
    {
        this->InitMotorPolicy();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
#endif


    std::cout << LOGGER::INFO << "RL_Sim start" << std::endl;

#ifdef ROS_BAG_RECORDER
    rosbag_recorder = std::make_unique<RosbagRecorder>(
        this->params.Get<std::string>("rosbag_save_path"),   // 保存路径
        this->params.Get<std::string>("rosbag_save_name")       // 包名前缀
    );
#endif

    // legged_odom_ptr = std::make_unique<odom_utils::legged_odom>(this->ros2_node);
    
    
}

#ifdef MOTOR_POLICY
void RL_Sim::InitMotorPolicy()
{
    std::string model_path = std::string(POLICY_DIR) + "/panda3/motor.onnx";
    loop_motor_policy = InferenceRuntime::ModelFactory::load_model(model_path);
    if (!this->model)
    {
        throw std::runtime_error("Failed to load model from: " + model_path);
    }
}
#endif


void RL_Sim::quatToRotMatrix(double qx, double qy, double qz, double qw, double R[3][3])
{

    R[0][0] = 1 - 2*qy*qy - 2*qz*qz;
    R[0][1] = 2*qx*qy - 2*qz*qw;
    R[0][2] = 2*qx*qz + 2*qy*qw;

    R[1][0] = 2*qx*qy + 2*qz*qw;
    R[1][1] = 1 - 2*qx*qx - 2*qz*qz;
    R[1][2] = 2*qy*qz - 2*qx*qw;

    R[2][0] = 2*qx*qz - 2*qy*qw;
    R[2][1] = 2*qy*qz + 2*qx*qw;
    R[2][2] = 1 - 2*qx*qx - 2*qy*qy;
}

void RL_Sim::updatePosition(RobotState<float> *state)
{
    rclcpp::Time now = ros2_node->now();
    double dt = 0.0;
     if (!first_call) {
        dt = (now - last_time).seconds();
    } else {
        first_call = false; // 第一次调用，不积分
    }
    last_time = now;

    double R[3][3];
    quatToRotMatrix(state->imu.quaternion[1],state->imu.quaternion[2],state->imu.quaternion[3],state->imu.quaternion[1], R);

    // 矩阵乘向量
    std::array<double,3> delta = {0.0, 0.0, 0.0};
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            delta[i] += R[i][j] * state->imu.accelerometer[j] * dt;
        }
    }

    // 更新世界坐标
    for(int i=0;i<3;i++)
        position_world[i] += delta[i];
}



void RL_Sim::publishMapToBase(double x, double y, double yaw, double qx, double qy, double qz, double qw)
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


void RL_Sim::publishMarker()
{
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = this->ros2_node->now();
    marker.ns = "ground";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = 0.0;
    marker.pose.position.y = 0.0;
    marker.pose.position.z = -0.01;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 10.0;
    marker.scale.y = 10.0;
    marker.scale.z = 0.02;
    marker.color.r = 0.5;
    marker.color.g = 0.5;
    marker.color.b = 0.5;
    marker.color.a = 1.0;  // 完全不透明

    this->marker_publisher->publish(marker);
}


void RL_Sim::publishGrid()
{
    
    nav_msgs::msg::OccupancyGrid grid;
    grid.header.frame_id = "map";
    grid.header.stamp = this->ros2_node->now();

    grid.info.resolution = 0.5;   // 0.5 m per cell
    grid.info.width = 20;         // 10 m
    grid.info.height = 20;
    grid.info.origin.position.x = -5.0;
    grid.info.origin.position.y = -5.0;
    grid.info.origin.orientation.w = 1.0;

    grid.data.resize(grid.info.width * grid.info.height, 255); 

    grid_publisher->publish(grid);
}





RL_Sim::~RL_Sim()
{
    this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;
}

void RL_Sim::StartJointController(const std::string& ros_namespace, const std::vector<std::string>& names)
{
#if defined(USE_ROS1)
    pid_t pid0 = fork();
    if (pid0 == 0)
    {
        std::string cmd = "rosrun controller_manager spawner joint_state_controller ";
        for (const auto& name : names)
        {
            cmd += name + " ";
        }
        cmd += "__ns:=" + ros_namespace;
        // cmd += " > /dev/null 2>&1";  // Comment this line to see the output
        execlp("sh", "sh", "-c", cmd.c_str(), nullptr);
        exit(1);
    }
#elif defined(USE_ROS2)
    const char* ros_distro = std::getenv("ROS_DISTRO");
    std::string spawner = (ros_distro && std::string(ros_distro) == "foxy") ? "spawner.py" : "spawner";

    std::filesystem::path tmp_path = std::filesystem::temp_directory_path() / "robot_joint_controller_params.yaml";
    {
        std::ofstream tmp_file(tmp_path);
        if (!tmp_file)
        {
            throw std::runtime_error("Failed to create temporary parameter file");
        }

        tmp_file << "/robot_joint_controller:\n";
        tmp_file << "    ros__parameters:\n";
        tmp_file << "        joints:\n";
        for (const auto& name : names)
        {
            tmp_file << "            - " << name << "\n";
        }
    }

    pid_t pid = fork();
    if (pid == 0)
    {
        std::string cmd = "ros2 run controller_manager " + spawner + " robot_joint_controller ";
        cmd += "-p " + tmp_path.string() + " ";
        // cmd += " > /dev/null 2>&1";  // Comment this line to see the output
        execlp("sh", "sh", "-c", cmd.c_str(), nullptr);
        exit(1);
    }
    else if (pid > 0)
    {
        int status;
        waitpid(pid, &status, 0);

        if (WIFEXITED(status) && WEXITSTATUS(status) != 0)
        {
            throw std::runtime_error("Failed to start joint controller");
        }

        std::filesystem::remove(tmp_path);
    }
    else
    {
        throw std::runtime_error("fork() failed");
    }
#endif
}

void RL_Sim::GetState(RobotState<float> *state)
{
#if defined(USE_ROS1)
    const auto &orientation = this->pose.orientation;
    const auto &angular_velocity = this->vel.angular;
#elif defined(USE_ROS2)
    auto info = std::static_pointer_cast<TopicInfo<sensor_msgs::msg::Imu>>(topics[imu_topic_name.c_str()]);
    auto gazebo_imu = std::atomic_load(&info->latest_msg);

    const auto &orientation = gazebo_imu->orientation;
    const auto &angular_velocity = gazebo_imu->angular_velocity;
    const auto &linear_acceleration = gazebo_imu->linear_acceleration;

    auto info2 = std::static_pointer_cast<TopicInfo<robot_msgs::msg::RobotState>>(topics[robot_state_topic_name.c_str()]);
    auto robot_state_msg = std::atomic_load(&info2->latest_msg);
#endif

    state->imu.quaternion[0] = orientation.w;
    state->imu.quaternion[1] = orientation.x;
    state->imu.quaternion[2] = orientation.y;
    state->imu.quaternion[3] = orientation.z;

    state->imu.gyroscope[0] = angular_velocity.x;
    state->imu.gyroscope[1] = angular_velocity.y;
    state->imu.gyroscope[2] = angular_velocity.z;

    state->imu.accelerometer[0] = linear_acceleration.x;
    state->imu.accelerometer[1] = linear_acceleration.y;
    state->imu.accelerometer[2] = linear_acceleration.z;


    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
#if defined(USE_ROS1)
        state->motor_state.q[i] = this->joint_positions[this->params.Get<std::vector<std::string>>("joint_controller_names")[this->params.Get<std::vector<int>>("joint_mapping")[i]]];
        state->motor_state.dq[i] = this->joint_velocities[this->params.Get<std::vector<std::string>>("joint_controller_names")[this->params.Get<std::vector<int>>("joint_mapping")[i]]];
        state->motor_state.tau_est[i] = this->joint_efforts[this->params.Get<std::vector<std::string>>("joint_controller_names")[this->params.Get<std::vector<int>>("joint_mapping")[i]]];
#elif defined(USE_ROS2)
        // state->motor_state.q[i] = this->robot_state_subscriber_msg.motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].q;
        // state->motor_state.dq[i] = this->robot_state_subscriber_msg.motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq;
        // state->motor_state.tau_est[i] = this->robot_state_subscriber_msg.motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau_est;
        state->motor_state.q[i] = robot_state_msg->motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].q;
        state->motor_state.dq[i] = robot_state_msg->motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq;
        state->motor_state.tau_est[i] = robot_state_msg->motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau_est;
        state->motor_state.status_word[i] = robot_state_msg->motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].status_word;
#endif
    }
}

void RL_Sim::SetCommand(const RobotCommand<float> *command)
{
    int motor_enable = this->motor_enabled ? 1 : 0;
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
#if defined(USE_ROS1)
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].q = command->motor_command.q[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq = command->motor_command.dq[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].kp = command->motor_command.kp[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].kd = command->motor_command.kd[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau = command->motor_command.tau[i];
        this->joint_publishers_commands[this->params.Get<std::vector<int>>("joint_mapping")[i]].mode = motor_enable;
#elif defined(USE_ROS2)
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].q = command->motor_command.q[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq = command->motor_command.dq[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].kp = command->motor_command.kp[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].kd = command->motor_command.kd[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau = command->motor_command.tau[i];
        this->robot_command_publisher_msg.motor_command[this->params.Get<std::vector<int>>("joint_mapping")[i]].mode = motor_enable;
#endif
    }

#if defined(USE_ROS1)
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->joint_publishers[this->params.Get<std::vector<std::string>>("joint_controller_names")[i]].publish(this->joint_publishers_commands[i]);
    }
#elif defined(USE_ROS2)
    this->robot_command_publisher->publish(this->robot_command_publisher_msg);
#endif
}

void RL_Sim::RobotControl()
{

    CheckTimeouts();
    {
        auto info = std::static_pointer_cast<TopicInfo<robot_msgs::msg::RobotState>>(topics[robot_state_topic_name.c_str()]);
        auto robot_state_msg = std::atomic_load(&info->latest_msg);
        if(!robot_state_msg) return;
    }

    
    this->GetState(&this->robot_state);
    // this->updatePosition(&this->robot_state);

    this->publishMapToBase(0,0,0, 
                            this->robot_state.imu.quaternion[1], 
                            this->robot_state.imu.quaternion[2], 
                            this->robot_state.imu.quaternion[3], 
                            this->robot_state.imu.quaternion[0]);
    this->publishGrid(); //调试地图
    // this->publishMarker();

    this->StateController(&this->robot_state, &this->robot_command);

    // std_msgs::msg::String fsm_state_msg;
    // fsm_state_msg.data = this->fsm.current_state_->GetStateName();
    // this->fsm_state_publisher->publish(fsm_state_msg);

    robot_msgs::msg::ControllerState controller_state_msg;
    controller_state_msg.fsm_state = this->fsm.current_state_->GetStateName();
    controller_state_msg.control_mode = this->control.navigation_mode;
    controller_state_msg.enable = this->motor_enabled;
    this->controller_state_publisher->publish(controller_state_msg);

    if (this->control.current_keyboard == Input::Keyboard::R || this->control.current_gamepad == Input::Gamepad::RB_Y)
    {
#if defined(USE_ROS1)
        std_srvs::Empty empty;
        this->gazebo_reset_world_client.call(empty);
#elif defined(USE_ROS2)
        auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
        auto result = this->gazebo_reset_world_client->async_send_request(empty_request);
#endif
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::Enter || this->control.current_gamepad == Input::Gamepad::RB_X)
    {
        if (simulation_running)
        {
#if defined(USE_ROS1)
            std_srvs::Empty empty;
            this->gazebo_pause_physics_client.call(empty);
#elif defined(USE_ROS2)
            auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
            auto result = this->gazebo_pause_physics_client->async_send_request(empty_request);
#endif
            std::cout << std::endl << LOGGER::INFO << "Simulation Stop" << std::endl;
        }
        else
        {
#if defined(USE_ROS1)
            std_srvs::Empty empty;
            this->gazebo_unpause_physics_client.call(empty);
#elif defined(USE_ROS2)
            auto empty_request = std::make_shared<std_srvs::srv::Empty::Request>();
            auto result = this->gazebo_unpause_physics_client->async_send_request(empty_request);
#endif
            std::cout << std::endl << LOGGER::INFO << "Simulation Start" << std::endl;
        }
        simulation_running = !simulation_running;
        this->control.current_keyboard = this->control.last_keyboard;
    }

    this->control.ClearInput();

    this->SetCommand(&this->robot_command);

    std::vector<float> actions;
    if (this->output_actions_queue.try_pop(actions))
    {
        if(actions.empty())return;
        robot_msgs::msg::Actions actions_msg;
        actions_msg.actions.resize(actions.size());
        for (size_t i = 0; i < 12; i++)
        {
            actions_msg.actions[i] = actions[i];
        }
        this->actions_publisher->publish(actions_msg);
    }

}

#if defined(USE_ROS1)
void RL_Sim::ModelStatesCallback(const gazebo_msgs::ModelStates::ConstPtr &msg)
{
    this->vel = msg->twist[2];
    this->pose = msg->pose[2];
}
#elif defined(USE_ROS2)
void RL_Sim::GazeboImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    this->gazebo_imu = *msg;
    // legged_odom_ptr->imu_wz_ = this->gazebo_imu.angular_velocity.z;
}
#endif

void RL_Sim::CmdvelCallback(
#if defined(USE_ROS1)
    const geometry_msgs::Twist::ConstPtr &msg
#elif defined(USE_ROS2)
    const geometry_msgs::msg::Twist::SharedPtr msg
#endif
)
{
    this->cmd_vel = *msg;
}


#if defined(USE_ROS2)
void RL_Sim::InitTopics() {

    
    // 初始化 /cmd_vel 话题信息
    this->cmd_topic_name = "/cmd_vel";
    auto cmd_vel_info = std::make_shared<TopicInfo<geometry_msgs::msg::Twist>>();
    cmd_vel_info->timeout_sec = 0.5;
    cmd_vel_info->last_time.store(std::chrono::steady_clock::now());
    topics[this->cmd_topic_name] = cmd_vel_info;

    cmd_vel_subscriber = ros2_node->create_subscription<geometry_msgs::msg::Twist>(
        this->cmd_topic_name, rclcpp::SystemDefaultsQoS(),
        [this](const geometry_msgs::msg::Twist::SharedPtr msg){ GenericCallback(this->cmd_topic_name, msg); }
    );


    // 初始化 /joy 话题信息
    this->joy_topic_name = "/joy";
    auto joy_info = std::make_shared<TopicInfo<sensor_msgs::msg::Joy>>();
    joy_info->timeout_sec = 0.5;
    joy_info->last_time.store(std::chrono::steady_clock::now());
    // 可选：设置额外回调
    joy_info->extra_callback = [this, joy_info] () {
        auto msg = std::atomic_load(&joy_info->latest_msg);
        this->joystick->JoyCallback(msg);
    };
    topics[this->joy_topic_name] = joy_info;
    joy_subscriber = ros2_node->create_subscription<sensor_msgs::msg::Joy>(
        this->joy_topic_name, rclcpp::SystemDefaultsQoS(),
        [this](const sensor_msgs::msg::Joy::SharedPtr msg){ GenericCallback(this->joy_topic_name, msg); }
    );


    // 初始化 /imu 话题信息
    this->imu_topic_name = "/imu";
    auto imu_info = std::make_shared<TopicInfo<sensor_msgs::msg::Imu>>();
    imu_info->timeout_sec = 0.5;
    imu_info->last_time.store(std::chrono::steady_clock::now());
    topics[this->imu_topic_name] = imu_info;
    gazebo_imu_subscriber = ros2_node->create_subscription<sensor_msgs::msg::Imu>(
        this->imu_topic_name, rclcpp::SystemDefaultsQoS(),
        [this](const sensor_msgs::msg::Imu::SharedPtr msg){ GenericCallback(this->imu_topic_name, msg); }
    );


    // 初始化 /robot_joint_controller/state 话题信息
    this->robot_state_topic_name = this->ros_namespace + "robot_joint_controller/state";
    auto robot_state_info = std::make_shared<TopicInfo<robot_msgs::msg::RobotState>>();
    robot_state_info->timeout_sec = 0.5;
    robot_state_info->last_time.store(std::chrono::steady_clock::now());
    topics[this->robot_state_topic_name] = robot_state_info;
    robot_state_subscriber = ros2_node->create_subscription<robot_msgs::msg::RobotState>(
        this->robot_state_topic_name, rclcpp::SystemDefaultsQoS(),
        [this](const robot_msgs::msg::RobotState::SharedPtr msg){ GenericCallback(this->robot_state_topic_name, msg); }
    );

    this->image_topic_name = "/depth/depth_camera/depth/image_raw";
    auto image_info = std::make_shared<TopicInfo<sensor_msgs::msg::Image>>();
    image_info->timeout_sec = 1.0;
    image_info->last_time.store(std::chrono::steady_clock::now());
    image_info->extra_callback = [this, image_info] () {
        auto msg = std::atomic_load(&image_info->latest_msg);
        auto depth_image = depth_image_to_vector(msg->data, this->image_width, this->image_height);
        std::atomic_store(&this->depth_image_ptr, std::make_shared<std::vector<float>>(std::move(depth_image)));
    };
    topics[this->image_topic_name] = image_info;
    image_subscriber = ros2_node->create_subscription<sensor_msgs::msg::Image>(
        this->image_topic_name, rclcpp::SystemDefaultsQoS(),
        [this](const sensor_msgs::msg::Image::SharedPtr msg){ GenericCallback(this->image_topic_name, msg); }
    );

}

void RL_Sim::CheckTimeouts() {
    auto now = std::chrono::steady_clock::now();

    for (auto& kv : topics) {
        auto& info = kv.second;
        auto dt = std::chrono::duration<double>(now - info->last_time.load()).count();
        if (dt > info->timeout_sec) {
            // std::cout << LOGGER::INFO << "RL_Sim start" << std::endl;
            // RCLCPP_WARN(node->get_logger(), "Topic %s timeout %.3f s", kv.first.c_str(), dt);
            // std::cout << LOGGER::INFO  << kv.first << " timeout " << dt << " s" << std::endl;

            // 超时处理，例如停机
            if (kv.first == "/cmd_vel") {
                // geometry_msgs::msg::Twist stop_cmd{};
                // auto twist_info = std::static_pointer_cast<TopicInfo<geometry_msgs::msg::Twist>>(info);
                // std::atomic_store(&twist_info->latest_msg, std::make_shared<geometry_msgs::msg::Twist>(stop_cmd));
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
                auto robot_state_msg = std::atomic_load(&info->latest_msg);
                if(robot_state_msg)
                {
                    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
                    {
                          robot_state_msg->motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].status_word = 0; // 设置状态字为0，表示无效或未连接
                    }
                    std::atomic_store(&info->latest_msg, robot_state_msg);
                }

            }
        }
    }
}


#endif


std::vector<float> RL_Sim::depth_image_to_vector(const std::vector<uint8_t>& data, int width, int height)
{
    const float min_depth = 0.05f;
    const float max_depth = 2.0f;

    std::vector<float> depth_vec;
    depth_vec.reserve(width * height);

    // reinterpret_cast 指向原始 float 数据
    const float* data_ptr = reinterpret_cast<const float*>(data.data());

    for (size_t i = 0; i < width * height; ++i)
    {
        float d = data_ptr[i];

        // nan / inf -> clamp
        if (std::isnan(d)) d = 0.0f;
        else if (std::isinf(d)) d = (d > 0 ? max_depth : min_depth);

        // clamp
        d = std::clamp(d, min_depth, max_depth);

        // normalize [-0.5, 0.5]
        d = (d - min_depth) / (max_depth - min_depth) - 0.5f;

        depth_vec.push_back(d);
    }

    return depth_vec; // 扁平化 vector
}



#if defined(USE_ROS1)
void RL_Sim::JointStatesCallback(const robot_msgs::MotorState::ConstPtr &msg, const std::string &joint_controller_name)
{
    this->joint_positions[joint_controller_name] = msg->q;
    this->joint_velocities[joint_controller_name] = msg->dq;
    this->joint_efforts[joint_controller_name] = msg->tau_est;
}
#elif defined(USE_ROS2)
void RL_Sim::RobotStateCallback(const robot_msgs::msg::RobotState::SharedPtr msg)
{
    std::atomic_store(&this->robot_state_subscriber_msg, msg);
    // this->robot_state_subscriber_msg = *msg;
}
#endif

void RL_Sim::RunModel()
{
    if (this->rl_init_done && simulation_running)
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
            this->obs.commands = {this->control.x, this->control.y, this->control.yaw};
        if (this->control.navigation_mode)
        {
            auto info = std::static_pointer_cast<TopicInfo<geometry_msgs::msg::Twist>>(topics[cmd_topic_name.c_str()]);
            auto cmd = std::atomic_load(&info->latest_msg);
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



#ifdef MOTOR_POLICY
        if (qd_history.size() >= 3) {
            qd_history.pop_front();  // 丢最旧的
        }
        qd_history.push_back(this->output_dof_pos);

        if(errs_history.size() >= 3) {
            errs_history.pop_front();  // 丢最旧的
        }
        auto err_pos = this->output_dof_pos - this->obs.dof_pos;
        errs_history.push_back(err_pos);

        std::vector<float> input(12 * 6, 0.0f);
        int hist_size = std::min(3, (int)qd_history.size());
        for (size_t i = 0; i < 12; i++)
        {
            int base = i * 6;
            for(int h = 0; h < hist_size; ++h)
            {
                // 最新在 back()，训练时是 t, t-1, t-2
                input[base + 0 + h] = errs_history[errs_history.size() - 1 - h][i];
            }
            for(int h = 0; h < hist_size; ++h)
            {
                input[base + 3 + h] = qd_history[qd_history.size() - 1 - h][i];
            }
        }

        auto taus = this->loop_motor_policy->forward_motor_policy(input);
        // output_dof_tau = this->params.Get<std::vector<float>>("rl_kp") * (all_actions_scaled + this->params.Get<std::vector<float>>("default_dof_pos") - this->obs.dof_pos) - this->params.Get<std::vector<float>>("rl_kd") * this->obs.dof_vel;
        auto q = (taus + this->params.Get<std::vector<float>>("rl_kd") * this->obs.dof_vel) / this->params.Get<std::vector<float>>("rl_kp") + this->obs.dof_pos;
        

#endif



        // output_actions_queue.push(this->obs.actions); // for logging

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
        auto info = std::static_pointer_cast<TopicInfo<robot_msgs::msg::RobotState>>(topics[robot_state_topic_name.c_str()]);
        auto robot_state_msg = std::atomic_load(&info->latest_msg);
        if(!robot_state_msg) return;
        
        std::vector<float> tau_est(this->params.Get<int>("num_of_dofs"), 0.0f);
        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            // tau_est[i] = this->joint_efforts[this->params.Get<std::vector<std::string>>("joint_controller_names")[i]];
            // tau_est[i] = this->robot_state_subscriber_msg.motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau_est;
            tau_est[i] = robot_state_msg->motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau_est;
        }
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

std::vector<float> RL_Sim::Forward()
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
            actions = this->model->forward({this->history_obs});
        }
        else if(this->config_name == "wmp")
        {

            this->wm_action_history.erase(this->wm_action_history.begin(), this->wm_action_history.begin() + this->wm_action.size());
            this->wm_action_history.insert(this->wm_action_history.end(), this->wm_action.begin(), this->wm_action.end());
            this->wm_action = this->wm_action_history;

            std::vector<float> input_image(this->image_width * this->image_height, 0.0f);
            auto depth_image = std::atomic_load(&this->depth_image_ptr);
            if(depth_image)
            {
                this->wm_input_image = *depth_image;
            }else
            {
                this->wm_input_image = std::vector<float>(this->image_width * this->image_height, 0.0f); // 如果没有图像数据，使用全零输入
            }
            if(global_counter % visual_update_interval == 0)
            {
                
                if(this->pre_wm_image.empty())
                    input_image = this->wm_input_image; // 初始化前一帧图像
                else
                    input_image = this->pre_wm_image; // 使用上一帧图像作为输入
                auto start = std::chrono::steady_clock::now();
                auto world_model_output = this->world_model->forward_world({world_obs, input_image, this->wm_logit, this->wm_stoch, this->wm_deter, this->wm_action, this->wm_is_first});
                auto end = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                std::cout << LOGGER::DEBUG << "World model forward time: " << duration.count() << " us" << std::endl;
                this->wm_logit = std::move(world_model_output[0]);
                this->wm_stoch = std::move(world_model_output[1]);
                this->wm_deter = std::move(world_model_output[2]);
                this->wm_feature = std::move(world_model_output[3]);
            }
            this->wm_is_first[0] = 0;

            this->history_obs_buf.insert(clamped_obs);
            this->history_obs = this->history_obs_buf.get_obs_vec(this->params.Get<std::vector<int>>("observations_history"));
            auto start = std::chrono::steady_clock::now();
            actions = this->model->forward({this->obs.commands, this->history_obs, this->wm_feature});
            auto end = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << LOGGER::DEBUG << "Policy model forward time: " << duration.count() << " us" << std::endl;
            this->wm_action = actions;



            this->pre_wm_image = std::move(this->wm_input_image);
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

void RL_Sim::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    auto info = std::static_pointer_cast<TopicInfo<robot_msgs::msg::RobotState>>(topics[robot_state_topic_name.c_str()]);
    auto robot_state_msg = std::atomic_load(&info->latest_msg);
    if(!robot_state_msg) return;
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
#if defined(USE_ROS1)
        this->plot_real_joint_pos[i].push_back(this->joint_positions[this->params.Get<std::vector<std::string>>("joint_controller_names")[i]]);
        this->plot_target_joint_pos[i].push_back(this->joint_publishers_commands[i].q);
#elif defined(USE_ROS2)
        // this->plot_real_joint_pos[i].push_back(this->robot_state_subscriber_msg.motor_state[i].q);
        this->plot_real_joint_pos[i].push_back(robot_state_msg->motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].q);
        this->plot_target_joint_pos[i].push_back(this->robot_command_publisher_msg.motor_command[i].q);
#endif
        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.01);
}

#if defined(USE_ROS1)
void signalHandler(int signum)
{
    ros::shutdown();
    exit(0);
}
#endif

int main(int argc, char **argv)
{
#if defined(USE_ROS1)
    signal(SIGINT, signalHandler);
    ros::init(argc, argv, "rl_sar");
    RL_Sim rl_sar(argc, argv);
    ros::spin();
#elif defined(USE_ROS2)
    rclcpp::init(argc, argv);
    auto rl_sar = std::make_shared<RL_Sim>(argc, argv);
    rclcpp::spin(rl_sar->ros2_node);
    rclcpp::shutdown();
#endif
    return 0;
}
