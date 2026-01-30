#ifndef RL_SAR_LIBRARY_CORE_ODOM_LEGGED_ODOM_HPP_
#define RL_SAR_LIBRARY_CORE_ODOM_LEGGED_ODOM_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2/LinearMath/Quaternion.h>

#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>

namespace odom_utils
{

struct Vec3
{
  double x;
  double y;
  double z;

};

inline std::ostream& operator<<(std::ostream& os, const Vec3& v)
{
  os << "[" << v.x << ", " << v.y << ", " << v.z << "]";
  return os;
}

class legged_odom
{
private:

    double x_ = 0.0;
    double y_ = 0.0;
    double yaw_ = 0.0;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;

public:

    legged_odom(std::shared_ptr<rclcpp::Node> node): node_(node) {
        joint_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", 50,
        std::bind(&legged_odom::jointCallback, this, std::placeholders::_1));
        odom_pub_ = node_->create_publisher<nav_msgs::msg::Odometry>("/odom", 50);
        last_time_ = node_->now();

    };
    ~legged_odom(){};

    double imu_wz_ = 0.0;
    std::unordered_map<std::string, double> joint_pos_;
    std::vector<Vec3> last_foot_pos_;
    rclcpp::Time last_time_;
    std::shared_ptr<rclcpp::Node> node_;


    void jointCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
      for (size_t i = 0; i < msg->name.size(); i++)
        joint_pos_[msg->name[i]] = msg->position[i];

      updateOdometry();
    }



    inline Vec3 rotX(const Vec3& v, double a)
    {
      return {
        v.x,
        std::cos(a) * v.y - std::sin(a) * v.z,
        std::sin(a) * v.y + std::cos(a) * v.z
      };
    }

    inline Vec3 rotY(const Vec3& v, double a)
    {
      return {
         std::cos(a) * v.x + std::sin(a) * v.z,
         v.y,
        -std::sin(a) * v.x + std::cos(a) * v.z
      };
    }



    Vec3 footFK(const std::string& leg,
            double q_hip,
            double q_thigh,
            double q_calf)
    {
      double hx = 0.286;
      double hy = 0.082;

      if (leg == "FR") hy = -hy;
      if (leg == "RL") hx = -hx;
      if (leg == "RR") { hx = -hx; hy = -hy; }

      Vec3 p{0.0, 0.0, 0.0};

      p.z -= 0.26851;
      p.z -= 0.26;
      p = rotY(p, q_calf);

      p.y += 0.1055;
      p = rotY(p, q_thigh);

      p.x += hx;
      p.y += hy;
      p = rotX(p, q_hip);
    //   std::cout << "Foot FK for leg " << leg << ": " << p << std::endl;

      return p;
    }


    void publishOdom(const rclcpp::Time &stamp)
    {
      nav_msgs::msg::Odometry odom;
      odom.header.stamp = stamp;
      odom.header.frame_id = "odom";
      odom.child_frame_id = "base";

      odom.pose.pose.position.x = x_;
      odom.pose.pose.position.y = y_;
      odom.pose.pose.position.z = 0.0;

      tf2::Quaternion q;
      q.setRPY(0.0, 0.0, yaw_);

      odom.pose.pose.orientation.x = q.x();
      odom.pose.pose.orientation.y = q.y();
      odom.pose.pose.orientation.z = q.z();
      odom.pose.pose.orientation.w = q.w();

      odom_pub_->publish(odom);
    }






    void updateOdometry()
    {
      rclcpp::Time now_time = node_->now();
      double dt = (now_time - last_time_).seconds();
      if (dt <= 0.0)
        return;

      // ---------- 1. IMU 预测 yaw ----------
      yaw_ += imu_wz_ * dt;

      // ---------- 2. 计算足端位置 ----------
      std::vector<std::string> legs = {"FL", "FR", "RL", "RR"};
      std::vector<Vec3> foot_pos;

      for (auto &leg : legs)
        foot_pos.push_back(footFK(leg,
                                  joint_pos_[leg + "_hip_joint"],
                                  joint_pos_[leg + "_thigh_joint"],
                                  joint_pos_[leg + "_calf_joint"]));

      if (last_foot_pos_.empty())
      {
        last_foot_pos_ = foot_pos;
        last_time_ = now_time;
        return;
      }

         // ---------- 3. 生成 contact_ 推断 ----------
        std::vector<bool> contact_;
        for (size_t i = 0; i < foot_pos.size(); i++) {
            double vz = (foot_pos[i].z - last_foot_pos_[i].z) / dt;
            // 默认 z 高度约为0（地面），阈值可调
            std::cout<<"Foot " << legs[i] << " pos z: " << foot_pos[i].z << ", vz: " << vz << std::endl;
            bool is_support = (std::abs(foot_pos[i].z) > 0.35) && (std::abs(vz) < 0.15);
            contact_.push_back(is_support);
        }

        // ---------- 4. 计算支撑腿平均位移 ----------
        Vec3 delta{0.0, 0.0, 0.0};
        int support_count = 0;
        for (size_t i = 0; i < foot_pos.size(); i++) {
            if (!contact_[i]) continue;  // 只考虑支撑腿
            delta.x += foot_pos[i].x - last_foot_pos_[i].x;
            delta.y += foot_pos[i].y - last_foot_pos_[i].y;
            delta.z += foot_pos[i].z - last_foot_pos_[i].z;
            support_count++;
        }

        if (support_count > 0) {
            delta.x /= support_count;
            delta.y /= support_count;
            delta.z /= support_count;
        } else {
            delta = {0,0,0};
        }

        // 支撑腿假设
        delta.x = -delta.x;
        delta.y = -delta.y;

        last_foot_pos_ = foot_pos;

        // ---------- 5. 转到世界坐标 ----------
        double dx_world = std::cos(yaw_) * delta.x - std::sin(yaw_) * delta.y;
        double dy_world = std::sin(yaw_) * delta.x + std::cos(yaw_) * delta.y;

        x_ += dx_world;
        y_ += dy_world;

        publishOdom(now_time);
        last_time_ = now_time;
    }


};


};  // namespace odom_utils


#endif