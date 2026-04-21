/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_test.hpp"

RL_Test::RL_Test(int argc, char **argv)
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

    // publisher
    this->robot_command_publisher = ros2_node->create_publisher<robot_msgs::msg::RobotCommand>(
        this->ros_namespace + "robot_joint_controller/command", rclcpp::SystemDefaultsQoS());
    this->robot_real_command_publisher = ros2_node->create_publisher<robot_msgs::msg::RobotCommand>(
        this->ros_namespace + "rl_sar/Robot_Command", rclcpp::SystemDefaultsQoS());

    this->imu_publisher_ = ros2_node->create_publisher<sensor_msgs::msg::Imu>(
        "/imu", rclcpp::SystemDefaultsQoS());

    this->joint_state_publisher_ = ros2_node->create_publisher<sensor_msgs::msg::JointState>(
        "/joint_states", rclcpp::SystemDefaultsQoS());


    // subscriber
    this->cmd_vel_subscriber = ros2_node->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", rclcpp::SystemDefaultsQoS(),
        [this] (const geometry_msgs::msg::Twist::SharedPtr msg) {this->CmdvelCallback(msg);}
    );
    this->joy_subscriber = ros2_node->create_subscription<sensor_msgs::msg::Joy>(
        "/joy", rclcpp::SystemDefaultsQoS(),
        [this] (const sensor_msgs::msg::Joy::SharedPtr msg) {this->JoyCallback(msg);}
    );
    this->robot_state_subscriber = ros2_node->create_subscription<robot_msgs::msg::RobotState>(
        this->ros_namespace + "rl_sar/Robot_State", rclcpp::SystemDefaultsQoS(),
        [this] (const robot_msgs::msg::RobotState::SharedPtr msg) {this->RobotStateCallback(msg);}
    );

    this->tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(ros2_node);


    // loop
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("control_dt"), std::bind(&RL_Test::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Test::RunModel, this));
    this->loop_control->start();
    this->loop_rl->start();

    // keyboard
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Test::KeyboardInterface, this));
    this->loop_keyboard->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    this->plot_target_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.001, std::bind(&RL_Test::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif

    std::cout << LOGGER::INFO << "RL_Test start" << std::endl;

#ifdef ROS_BAG_RECORDER
    rosbag_recorder = std::make_unique<RosbagRecorder>(
        this->params.Get<std::string>("rosbag_save_path"),   // 保存路径
        this->params.Get<std::string>("rosbag_save_name")       // 包名前缀
    );
#endif

}

RL_Test::~RL_Test()
{
    this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Test exit" << std::endl;


}


void RL_Test::publishMapToBase(double x, double y, double yaw, double qx, double qy, double qz, double qw)
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


void RL_Test::GetState(RobotState<float> *state)
{
    const auto orientation = this->robot_state_subscriber_msg.imu.quaternion;
    const auto angular_velocity = this->robot_state_subscriber_msg.imu.gyroscope;

    this->imu_.orientation.x = orientation[1];
    this->imu_.orientation.y = orientation[2];
    this->imu_.orientation.z = orientation[3];
    this->imu_.orientation.w = orientation[0];
    
    this->imu_.angular_velocity.x = angular_velocity[0];
    this->imu_.angular_velocity.y = angular_velocity[1];
    this->imu_.angular_velocity.z = angular_velocity[2];
    
    this->imu_.linear_acceleration.x = this->robot_state_subscriber_msg.imu.accelerometer[0];
    this->imu_.linear_acceleration.y = this->robot_state_subscriber_msg.imu.accelerometer[1];
    this->imu_.linear_acceleration.z = this->robot_state_subscriber_msg.imu.accelerometer[2];


    this->imu_publisher_->publish(this->imu_);


    state->imu.quaternion[0] = orientation[0];
    state->imu.quaternion[1] = orientation[1];
    state->imu.quaternion[2] = orientation[2];
    state->imu.quaternion[3] = orientation[3];

    state->imu.gyroscope[0] = angular_velocity[0];
    state->imu.gyroscope[1] = angular_velocity[1];
    state->imu.gyroscope[2] = angular_velocity[2];


    sensor_msgs::msg::JointState joint_state_msg;
    joint_state_msg.header.stamp = this->ros2_node->now();
    joint_state_msg.name = std::vector<std::string>(this->params.Get<int>("num_of_dofs"),"");
    joint_state_msg.position = std::vector<double>(this->params.Get<int>("num_of_dofs"),0.0f);
    joint_state_msg.velocity  = std::vector<double>(this->params.Get<int>("num_of_dofs"),0.0f);
    joint_state_msg.effort  = std::vector<double>(this->params.Get<int>("num_of_dofs"),0.0f);


    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        state->motor_state.q[i] = this->robot_state_subscriber_msg.motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].q;
        state->motor_state.dq[i] = this->robot_state_subscriber_msg.motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq;
        state->motor_state.tau_est[i] = this->robot_state_subscriber_msg.motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau_est;

        joint_state_msg.name[this->params.Get<std::vector<int>>("joint_mapping")[i]] = this->params.Get<std::vector<std::string>>("joint_names")[i];
        joint_state_msg.position[this->params.Get<std::vector<int>>("joint_mapping")[i]] = state->motor_state.q[i];
        joint_state_msg.velocity[this->params.Get<std::vector<int>>("joint_mapping")[i]] = state->motor_state.dq[i];
        joint_state_msg.effort[this->params.Get<std::vector<int>>("joint_mapping")[i]] = state->motor_state.tau_est[i];
    }

    joint_state_publisher_->publish(joint_state_msg);
}

void RL_Test::SetCommand(const RobotCommand<float> *command)
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

    this->robot_command_publisher->publish(this->robot_command_publisher_msg);
    this->robot_real_command_publisher->publish(this->robot_command_publisher_msg);
}

void RL_Test::RobotControl()
{
    this->GetState(&this->robot_state);
    this->publishMapToBase(0,0,0, 
                            this->robot_state.imu.quaternion[1], 
                            this->robot_state.imu.quaternion[2], 
                            this->robot_state.imu.quaternion[3], 
                            this->robot_state.imu.quaternion[0]);

    this->StateController(&this->robot_state, &this->robot_command);

    this->control.ClearInput();

    this->SetCommand(&this->robot_command);
}


void RL_Test::CmdvelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
    this->cmd_vel = *msg;
}

void RL_Test::JoyCallback(const sensor_msgs::msg::Joy::SharedPtr msg)
{
    this->joy_msg = *msg;

    // joystick control
    // Description of buttons and axes(F710):
    // |__ buttons[]: A=0, B=1, X=2, Y=3, LB=4, RB=5, back=6, start=7, power=8, stickL=9, stickR=10
    // |__ axes[]: Lx=0, Ly=1, Rx=3, Ry=4, LT=2, RT=5, DPadX=6, DPadY=7

    if (this->joy_msg.buttons[0]) this->control.SetGamepad(Input::Gamepad::A);
    if (this->joy_msg.buttons[1]) this->control.SetGamepad(Input::Gamepad::B);
    if (this->joy_msg.buttons[2]) this->control.SetGamepad(Input::Gamepad::X);
    if (this->joy_msg.buttons[3]) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->joy_msg.buttons[4]) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->joy_msg.buttons[5]) this->control.SetGamepad(Input::Gamepad::RB);
    if (this->joy_msg.buttons[9]) this->control.SetGamepad(Input::Gamepad::LStick);
    if (this->joy_msg.buttons[10]) this->control.SetGamepad(Input::Gamepad::RStick);
    if (this->joy_msg.axes[7] > 0) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (this->joy_msg.axes[7] < 0) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (this->joy_msg.axes[6] < 0) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (this->joy_msg.axes[6] > 0) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if (this->joy_msg.axes[7] > 0 && this->joy_msg.buttons[2]) this->control.SetGamepad(Input::Gamepad::X);
    if (this->joy_msg.axes[7] < 0 && this->joy_msg.buttons[2]) this->control.SetGamepad(Input::Gamepad::X);
    if (this->joy_msg.axes[7] > 0 && this->joy_msg.buttons[3]) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->joy_msg.axes[7] < 0 && this->joy_msg.buttons[3]) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[0]) this->control.SetGamepad(Input::Gamepad::LB_A);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[1]) this->control.SetGamepad(Input::Gamepad::LB_B);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[2]) this->control.SetGamepad(Input::Gamepad::LB_X);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[3]) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[9]) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[10]) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if (this->joy_msg.buttons[4] && this->joy_msg.axes[7] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if (this->joy_msg.buttons[4] && this->joy_msg.axes[7] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if (this->joy_msg.buttons[4] && this->joy_msg.axes[6] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if (this->joy_msg.buttons[4] && this->joy_msg.axes[6] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[0]) this->control.SetGamepad(Input::Gamepad::RB_A);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[1]) this->control.SetGamepad(Input::Gamepad::RB_B);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[2]) this->control.SetGamepad(Input::Gamepad::RB_X);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[3]) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[9]) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if (this->joy_msg.buttons[5] && this->joy_msg.buttons[10]) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if (this->joy_msg.buttons[5] && this->joy_msg.axes[7] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if (this->joy_msg.buttons[5] && this->joy_msg.axes[7] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if (this->joy_msg.buttons[5] && this->joy_msg.axes[6] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if (this->joy_msg.buttons[5] && this->joy_msg.axes[6] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if (this->joy_msg.buttons[4] && this->joy_msg.buttons[5]) this->control.SetGamepad(Input::Gamepad::LB_RB);

    this->control.x = this->joy_msg.axes[1]; // LY
    this->control.y = this->joy_msg.axes[0]; // LX
    this->control.yaw = this->joy_msg.axes[3]; // RX
    this->control.stand = (this->joy_msg.axes[7] == 1 ? 1.0f : 0.0f);
    this->control.height = this->joy_msg.axes[4];

    if(this->joy_msg.buttons[7] == 1)
    {
        motor_enable_changed = true;
    }

    if (this->joy_msg.buttons[7] == 0 && motor_enable_changed)
    {
        motor_enable_changed = false;
        this->motor_enabled = !this->motor_enabled;
    }



}


void RL_Test::RobotStateCallback(const robot_msgs::msg::RobotState::SharedPtr msg)
{
    this->robot_state_subscriber_msg = *msg;
}

void RL_Test::RunModel()
{
    if (this->rl_init_done)
    {
        this->episode_length_buf += 1;
        this->obs.ang_vel = this->robot_state.imu.gyroscope;
        if(this->current_rl_fsm_name.compare("RLFSMStateRLStand") == 0)
            this->obs.commands = {0.0f, 0.0f, 0.0f, 0.0f,this->control.stand, 0.0f,0.0f,0.0f,0.0f,0.0f};
        else if(this->current_rl_fsm_name.compare("RLFSMStateRLCrouch") == 0)
            this->obs.commands = {(float)this->control.x, (float)this->control.y, (float)this->control.yaw, this->control.height, 0.0f, 0.0f,0.0f,0.0f,0.0f,0.0f};
        else
            this->obs.commands = {(float)this->control.x, (float)this->control.y, (float)this->control.yaw,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
        if (this->control.navigation_mode)
        {
            if(this->current_rl_fsm_name.compare("RLFSMStateRLStand") == 0)
                this->obs.commands = {0.0f, 0.0f, 0.0f, 0.0f,this->control.stand, 0.0f,0.0f,0.0f,0.0f,0.0f};
            else if(this->current_rl_fsm_name.compare("RLFSMStateRLCrouch") == 0)
                this->obs.commands = {(float)this->cmd_vel.linear.x, (float)this->cmd_vel.linear.y, (float)this->cmd_vel.angular.z, 0.0f,0.0f, 0.0f,0.0f,0.0f,0.0f,0.0f};
            else
                this->obs.commands = {(float)this->cmd_vel.linear.x, (float)this->cmd_vel.linear.y, (float)this->cmd_vel.angular.z,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
        }
        this->obs.base_quat = this->robot_state.imu.quaternion;
        this->obs.dof_pos = this->robot_state.motor_state.q;
        this->obs.dof_vel = this->robot_state.motor_state.dq;

        this->obs.actions = this->Forward();
        // this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);


        if(!Interpolate(percent_pre_getup, this->params.Get<std::vector<float>>("default_dof_pos"), pre_running_pos, current_duration,this->output_dof_pos, true))
            if(!Interpolate(percent_getup, pre_running_pos, this->params.Get<std::vector<float>>("default_dof_pos"), current_duration,this->output_dof_pos, true))
                {
                    percent_pre_getup = 0.0f;
                    percent_getup = 0.0f;
                    if(current_duration > max_duration)
                    {
                        current_duration = max_duration;
                        gain *= -1.0;
                    }
                    else if(current_duration < min_duration)
                    {
                        current_duration = min_duration;
                        gain *= -1.0;
                    }
                    current_duration += gain;
                    std::cout<< current_duration << std::endl;
                }


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


bool RL_Test::Interpolate(
        float& percent,
        const std::vector<float>& start_pos,
        const std::vector<float>& target_pos,
        float duration_seconds,
        std::vector<float> &output_dof_pos,
        bool use_fixed_gains
    )
{
    if (percent >= 1.0f)
    {
        return false;
    }

    if (percent == 0.0f)
    {
        float max_diff = 0.0f;
        for (size_t i = 0; i < start_pos.size() && i < target_pos.size(); ++i)
        {
            max_diff = std::max(max_diff, std::abs(start_pos[i] - target_pos[i]));
        }

        if (max_diff < 0.1f)
        {
            percent = 1.0f;
        }
    }

    int required_frames = std::max(1, static_cast<int>(std::ceil(duration_seconds / this->params.Get<float>("dt"))));
    float step = 1.0f / required_frames;

    percent += step;
    percent = std::min(percent, 1.0f);

    auto kp = use_fixed_gains ? this->params.Get<std::vector<float>>("fixed_kp") : this->params.Get<std::vector<float>>("rl_kp");
    auto kd = use_fixed_gains ? this->params.Get<std::vector<float>>("fixed_kd") : this->params.Get<std::vector<float>>("rl_kd");

    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        output_dof_pos[i] = (1 - percent) * start_pos[i] + percent * target_pos[i];
    }


    if (percent >= 1.0f)
    {
        return false;
    }

    return true;

}

std::vector<float> RL_Test::Forward()
{
    std::unique_lock<std::mutex> lock(this->model_mutex, std::try_to_lock);

    // If model is being reinitialized, return previous actions to avoid blocking
    if (!lock.owns_lock())
    {
        std::cout << LOGGER::WARNING << "Model is being reinitialized, using previous actions" << std::endl;
        return this->obs.actions;
    }

    std::vector<float> clamped_obs = this->ComputeObservation();

    std::vector<float> actions;
    if (this->params.Get<std::vector<int>>("observations_history").size() != 0)
    {

        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.Get<std::vector<int>>("observations_history"));
        actions = this->model->forward({clamped_obs,this->history_obs});
        this->history_obs_buf.insert(clamped_obs);
        
    }
    else
    {
        actions = this->model->forward({clamped_obs});
    }

    if (!this->params.Get<std::vector<float>>("clip_actions_upper").empty() && !this->params.Get<std::vector<float>>("clip_actions_lower").empty())
    {
        return clamp(actions, this->params.Get<std::vector<float>>("clip_actions_lower"), this->params.Get<std::vector<float>>("clip_actions_upper"));
    }
    else
    {
        return actions;
    }
}

void RL_Test::Plot()
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
    auto rl_sar = std::make_shared<RL_Test>(argc, argv);
    rclcpp::spin(rl_sar->ros2_node);
    rclcpp::shutdown();
    return 0;
}
