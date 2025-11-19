#ifndef ROBOT_HW_INTERFACE_HPP_
#define ROBOT_HW_INTERFACE_HPP_

#include <hardware_interface/system_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <vector>
#include <string>
#include <mutex>

#include "robot_msgs/msg/motor_command.hpp"
#include "robot_msgs/msg/motor_state.hpp"
#include "robot_msgs/msg/robot_state.hpp"
#include "robot_msgs/msg/robot_command.hpp"
#include "sensor_msgs/msg/imu.hpp"

namespace robot_hw_interface
{
class RobotHWInterface : public hardware_interface::SystemInterface
{
public:
  // RCLCPP_SHARED_PTR_DEFINITIONS(RobotHWInterface)
  RobotHWInterface();

  hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override;

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;

  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::return_type read(const rclcpp::Time & time, const rclcpp::Duration & period) override;

  hardware_interface::return_type write(const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  rclcpp::Node::SharedPtr node_;
  std::vector<double> hw_commands_;
  std::vector<double> hw_states_positions_;
  std::vector<double> hw_states_velocities_;
  std::vector<double> hw_states_efforts_;


  rclcpp::Publisher<robot_msgs::msg::RobotCommand>::SharedPtr robot_command_publisher_;
  rclcpp::Subscription<robot_msgs::msg::RobotState>::SharedPtr robot_state_subscriber_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
  robot_msgs::msg::MotorState motor_state_;
  robot_msgs::msg::RobotState robot_state_;
  sensor_msgs::msg::Imu imu_;

  std::mutex mutex_;

  void robot_state_callback(const robot_msgs::msg::RobotState::SharedPtr msg);
};
}  // namespace robot_hw_interface

#endif  // ROBOT_HW_INTERFACE_HPP_
