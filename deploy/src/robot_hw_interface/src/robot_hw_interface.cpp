#include "robot_hw_interface.hpp"

#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <iostream>
namespace robot_hw_interface
{


  RobotHWInterface::RobotHWInterface(){

  }



hardware_interface::CallbackReturn RobotHWInterface::on_init(const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) != hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  // Initialize hardware commands and states
  hw_commands_.resize(info_.joints.size(), 0.0);
  hw_states_positions_.resize(info_.joints.size(), 0.0);
  hw_states_velocities_.resize(info_.joints.size(), 0.0);
  hw_states_efforts_.resize(info_.joints.size(), 0.0);

  node_ = rclcpp::Node::make_shared("robot_hw_interface");
  rt_robot_command_publisher_ = std::make_shared<realtime_tools::RealtimePublisher<robot_msgs::msg::RobotCommand>>(
      node_->create_publisher<robot_msgs::msg::RobotCommand>("rl_sar/Robot_Command", rclcpp::SystemDefaultsQoS()));

  imu_publisher_ = node_->create_publisher<sensor_msgs::msg::Imu>(
        "/imu", rclcpp::SystemDefaultsQoS());
  
  robot_state_subscriber_ = node_->create_subscription<robot_msgs::msg::RobotState>(
    "rl_sar/Robot_State", rclcpp::SystemDefaultsQoS(),
    std::bind(&RobotHWInterface::robot_state_callback, this, std::placeholders::_1));


    joints_command_subscriber_ = node_->create_subscription<robot_msgs::msg::RobotCommand>(
        "robot_joint_controller/command", rclcpp::SystemDefaultsQoS(), 
        std::bind(&RobotHWInterface::set_command_callback, this, std::placeholders::_1));


    std::thread t4([this]() {             
        rclcpp::spin(node_);
    });
    t4.detach();


  std::cout << "----------- RobotHWInterface initialized successfully! ---------------" << std::endl;
  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> RobotHWInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;
  for (uint i = 0; i < info_.joints.size(); i++)
  {
    state_interfaces.emplace_back(hardware_interface::StateInterface(
      info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_states_positions_[i]));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
      info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_states_velocities_[i]));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
      info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &hw_states_efforts_[i]));
  }

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> RobotHWInterface::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  for (uint i = 0; i < info_.joints.size(); i++)
  {
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
      info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &hw_commands_[i]));
  }
  return command_interfaces;
}

hardware_interface::return_type RobotHWInterface::read(const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  // std::cout << "Reading motor states from robot..." << std::endl;
  
  auto robot_state = rt_robot_state_ptr_.readFromRT();

  if(!robot_state)
    return hardware_interface::return_type::OK;

  if(robot_state->motor_state.size() == 0)
    return hardware_interface::return_type::OK;


    for (uint i = 0; i < hw_states_positions_.size(); i++)
  {
    hw_states_positions_[i] = robot_state->motor_state[i].q;
    hw_states_velocities_[i] = robot_state->motor_state[i].dq;
    hw_states_efforts_[i] = robot_state->motor_state[i].tau_est;
    // std::cout<<"motor "<<i<<" pos: "<<hw_states_positions_[i]<<" vel: "<<hw_states_velocities_[i]<<" effort: "<<hw_states_efforts_[i]<<std::endl;
  }


  return hardware_interface::return_type::OK;
}

hardware_interface::return_type RobotHWInterface::write(const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  // std::cout << "write motor commands to robot..." << std::endl;
  robot_msgs::msg::RobotCommand command_msg;
  command_msg.motor_command.resize(hw_commands_.size());
  auto joint_commands = rt_command_ptr_.readFromRT();

  // std::vector<int> FLR_leg_motors = {3, 4, 5, 0, 1, 2};
  // std::vector<int> RLR_leg_motors = {9, 10, 11,6, 7, 8};
  // std::vector<int> new_order;
  //   // 先放右腿电机，再放左腿电机
  // new_order.insert(new_order.end(), FLR_leg_motors.begin(), FLR_leg_motors.end());
  // new_order.insert(new_order.end(), RLR_leg_motors.begin(), RLR_leg_motors.end());



  for (uint i = 0; i < hw_commands_.size(); i++)
  {
    // This is a simplified mapping. In a real robot, you would have a more complex mapping
    // between the joint command and the motor command.
    command_msg.motor_command[i].tau = hw_commands_[i];
    if (joint_commands && (joint_commands->motor_command.size() == hw_commands_.size()))
    {
        command_msg.motor_command[i].mode = joint_commands->motor_command[i].mode;
        command_msg.motor_command[i].q = joint_commands->motor_command[i].q;
        command_msg.motor_command[i].dq = joint_commands->motor_command[i].dq;
        command_msg.motor_command[i].kp = joint_commands->motor_command[i].kp;
        command_msg.motor_command[i].kd = joint_commands->motor_command[i].kd;
    }
  }
  if(rt_robot_command_publisher_ && rt_robot_command_publisher_->trylock())
  {
      command_msg.header.stamp = node_->get_clock()->now();
      rt_robot_command_publisher_->msg_ = command_msg;
      rt_robot_command_publisher_->unlockAndPublish();
  }
  return hardware_interface::return_type::OK;
}

void RobotHWInterface::robot_state_callback(const robot_msgs::msg::RobotState::SharedPtr msg)
{
  last_robot_state_ = *msg;
  rt_robot_state_ptr_.writeFromNonRT(last_robot_state_);


  this->imu_.orientation.x = msg->imu.quaternion[1];
  this->imu_.orientation.y = msg->imu.quaternion[2];
  this->imu_.orientation.z = msg->imu.quaternion[3];
  this->imu_.orientation.w = msg->imu.quaternion[0];
  
  this->imu_.angular_velocity.x = msg->imu.gyroscope[0];
  this->imu_.angular_velocity.y = msg->imu.gyroscope[1];
  this->imu_.angular_velocity.z = msg->imu.gyroscope[2];
  
  this->imu_.linear_acceleration.x = msg->imu.accelerometer[0];
  this->imu_.linear_acceleration.y = msg->imu.accelerometer[1];
  this->imu_.linear_acceleration.z = msg->imu.accelerometer[2];

  // std::cout<<"publishing imu data"<<std::endl;

  this->imu_publisher_->publish(this->imu_);

}


void RobotHWInterface::set_command_callback(const robot_msgs::msg::RobotCommand::SharedPtr msg)
{
    last_command_ = *msg;
    rt_command_ptr_.writeFromNonRT(last_command_);
}


}  // namespace robot_hw_interface

PLUGINLIB_EXPORT_CLASS(
  robot_hw_interface::RobotHWInterface, hardware_interface::SystemInterface)
