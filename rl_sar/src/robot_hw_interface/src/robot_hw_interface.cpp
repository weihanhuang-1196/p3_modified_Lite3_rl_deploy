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
  robot_command_publisher_ = node_->create_publisher<robot_msgs::msg::RobotCommand>(
    "rl_sar/Robot_Command", rclcpp::SystemDefaultsQoS());

  imu_publisher_ = node_->create_publisher<sensor_msgs::msg::Imu>(
        "/imu", rclcpp::SystemDefaultsQoS());
  
  robot_state_subscriber_ = node_->create_subscription<robot_msgs::msg::RobotState>(
    "rl_sar/Robot_state", rclcpp::SystemDefaultsQoS(),
    std::bind(&RobotHWInterface::robot_state_callback, this, std::placeholders::_1));


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
  rclcpp::spin_some(node_);

  std::lock_guard<std::mutex> lock(mutex_);
  if(this->robot_state_.motor_state.size() == 0){
    return hardware_interface::return_type::OK;
  }


    for (uint i = 0; i < hw_states_positions_.size(); i++)
  {
    hw_states_positions_[i] = this->robot_state_.motor_state[i].q;
    hw_states_velocities_[i] = this->robot_state_.motor_state[i].dq;
    hw_states_efforts_[i] = this->robot_state_.motor_state[i].tau_est;
  }


  return hardware_interface::return_type::OK;
}

hardware_interface::return_type RobotHWInterface::write(const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  // std::cout << "write motor commands to robot..." << std::endl;
  robot_msgs::msg::RobotCommand command_msg;
  command_msg.motor_command.resize(hw_commands_.size());
  for (uint i = 0; i < hw_commands_.size(); i++)
  {
    // This is a simplified mapping. In a real robot, you would have a more complex mapping
    // between the joint command and the motor command.
    command_msg.motor_command[i].tau = hw_commands_[i];
  }
  robot_command_publisher_->publish(command_msg);
  return hardware_interface::return_type::OK;
}

void RobotHWInterface::robot_state_callback(const robot_msgs::msg::RobotState::SharedPtr msg)
{

  std::lock_guard<std::mutex> lock(mutex_);
  this->robot_state_ = *msg;

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


}  // namespace robot_hw_interface

PLUGINLIB_EXPORT_CLASS(
  robot_hw_interface::RobotHWInterface, hardware_interface::SystemInterface)
