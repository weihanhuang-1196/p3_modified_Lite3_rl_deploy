/**
 * @file lite3_socket_policy_runner.hpp
 * @brief Lite3 策略运行器：通过 Unix Socket 接入 RKNN 推理服务
 */

#pragma once

#include "policy_runner_base.hpp"

#include <Eigen/Dense>

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

using namespace types;

/**
 * @brief 通过 Unix Domain Socket 与 `lite3_rknn_service.py` 通信的策略运行器
 */
class Lite3SocketPolicyRunner : public PolicyRunnerBase {
private:
    std::string socket_path_;
    int socket_fd_ = -1;

    const int obs_dim_ = 45;
    const int act_dim_ = 12;

    VecXf current_obs_;
    VecXf joint_pos_rl_ = VecXf(act_dim_);
    VecXf joint_vel_rl_ = VecXf(act_dim_);

    VecXf last_action_;
    VecXf tmp_action_;
    VecXf action_;

    VecXf dof_pos_default_robot_;
    VecXf dof_pos_default_policy_;
    VecXf kp_;
    VecXf kd_;
    Vec3f max_cmd_vel_;
    Vec3f gravity_direction_ = Vec3f(0.f, 0.f, -1.f);

    std::vector<int> robot2policy_idx_;
    std::vector<int> policy2robot_idx_;

    float omega_scale_ = 0.25f;
    float dof_vel_scale_ = 0.05f;

    std::vector<std::string> robot_order_ = {
        "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
        "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
        "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
        "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint"};

    std::vector<std::string> policy_order_ = {
        "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
        "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
        "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
        "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint"};

    std::vector<float> action_scale_robot_ = {0.125f, 0.25f, 0.25f,
                                              0.125f, 0.25f, 0.25f,
                                              0.125f, 0.25f, 0.25f,
                                              0.125f, 0.25f, 0.25f};

    RobotAction ra_;

    static bool SendAll(int fd, const void* data, size_t size) {
        const char* buffer = static_cast<const char*>(data);
        size_t sent = 0;
        while (sent < size) {
            ssize_t ret = ::send(fd, buffer + sent, size - sent, 0);
            if (ret <= 0) {
                return false;
            }
            sent += static_cast<size_t>(ret);
        }
        return true;
    }

    static bool RecvAll(int fd, void* data, size_t size) {
        char* buffer = static_cast<char*>(data);
        size_t received = 0;
        while (received < size) {
            ssize_t ret = ::recv(fd, buffer + received, size - received, 0);
            if (ret <= 0) {
                return false;
            }
            received += static_cast<size_t>(ret);
        }
        return true;
    }

    void CloseSocket() {
        if (socket_fd_ >= 0) {
            ::close(socket_fd_);
            socket_fd_ = -1;
        }
    }

    void EnsureConnected() {
        if (socket_fd_ >= 0) {
            return;
        }

        socket_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (socket_fd_ < 0) {
            throw std::runtime_error("[RKNN SOCKET] 创建 socket 失败");
        }

        sockaddr_un addr;
        std::memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        if (socket_path_.size() >= sizeof(addr.sun_path)) {
            CloseSocket();
            throw std::runtime_error("[RKNN SOCKET] 路径长度超出上限");
        }
        std::strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

        if (::connect(socket_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            CloseSocket();
            throw std::runtime_error("[RKNN SOCKET] 无法连接到推理服务: " + socket_path_);
        }
    }

    std::vector<int> GeneratePermutation(
        const std::vector<std::string>& from,
        const std::vector<std::string>& to,
        int default_index = 0) {
        std::unordered_map<std::string, int> idx_map;
        for (int i = 0; i < static_cast<int>(from.size()); ++i) {
            idx_map[from[i]] = i;
        }

        std::vector<int> perm;
        perm.reserve(to.size());
        for (const auto& name : to) {
            auto it = idx_map.find(name);
            if (it != idx_map.end()) {
                perm.push_back(it->second);
            } else {
                perm.push_back(default_index);
            }
        }
        return perm;
    }

    VecXf BuildObservation(const RobotBasicState& ro) {
        Vec3f base_omega = ro.base_omega * omega_scale_;
        Vec3f projected_gravity = ro.base_rot_mat.inverse() * gravity_direction_;
        Vec3f cmd_vel = ro.cmd_vel_normlized.cwiseProduct(max_cmd_vel_);

        for (int i = 0; i < act_dim_; ++i) {
            joint_pos_rl_(i) = ro.joint_pos(robot2policy_idx_[i]);
            joint_vel_rl_(i) = ro.joint_vel(robot2policy_idx_[i]) * dof_vel_scale_;
        }
        joint_pos_rl_ -= dof_pos_default_policy_;

        current_obs_.setZero(obs_dim_);
        current_obs_ << base_omega,
                        projected_gravity,
                        cmd_vel,
                        joint_pos_rl_,
                        joint_vel_rl_,
                        last_action_;
        return current_obs_;
    }

    VecXf DecodeAction(const VecXf& raw_action) {
        VecXf reordered = VecXf(act_dim_);
        for (int i = 0; i < act_dim_; ++i) {
            reordered(i) = raw_action(policy2robot_idx_[i]);
            reordered(i) *= action_scale_robot_[i];
        }
        reordered += dof_pos_default_robot_;
        return reordered;
    }

public:
    explicit Lite3SocketPolicyRunner(std::string policy_name)
        : PolicyRunnerBase(std::move(policy_name)) {
        const char* socket_env = std::getenv("LITE3_RKNN_SOCKET");
        socket_path_ = socket_env ? socket_env : "/tmp/lite3_rknn.sock";

        current_obs_ = VecXf(obs_dim_);
        last_action_ = VecXf::Zero(act_dim_);
        tmp_action_ = VecXf(act_dim_);
        action_ = VecXf(act_dim_);

        dof_pos_default_robot_ = VecXf::Zero(act_dim_);
        dof_pos_default_policy_ = VecXf::Zero(act_dim_);

        dof_pos_default_policy_ <<  0.0000f, -0.8000f, 1.6000f,
                                    0.0000f, -0.8000f, 1.6000f,
                                    0.0000f, -0.8000f, 1.6000f,
                                    0.0000f, -0.8000f, 1.6000f;

        dof_pos_default_robot_ = dof_pos_default_policy_;

        kp_ = 30.f * VecXf::Ones(act_dim_);
        kd_ = 1.f * VecXf::Ones(act_dim_);
        max_cmd_vel_ << 0.8f, 0.8f, 0.8f;

        ra_.goal_joint_pos = VecXf::Zero(act_dim_);
        ra_.goal_joint_vel = VecXf::Zero(act_dim_);
        ra_.tau_ff = VecXf::Zero(act_dim_);
        ra_.kp = kp_;
        ra_.kd = kd_;

        robot2policy_idx_ = GeneratePermutation(robot_order_, policy_order_);
        policy2robot_idx_ = GeneratePermutation(policy_order_, robot_order_);

        decimation_ = 12;
    }

    ~Lite3SocketPolicyRunner() override {
        CloseSocket();
    }

    void DisplayPolicyInfo() override {
        std::cout << "RKNN socket policy: " << policy_name_ << "\n";
        std::cout << "socket: " << socket_path_ << "\n";
        std::cout << "obs_dim: " << obs_dim_ << ", action_dim: " << act_dim_ << "\n";
    }

    void OnEnter() override {
        run_cnt_ = 0;
        last_action_.setZero();
        EnsureConnected();
        std::cout << "[RKNN SOCKET] Connected to service at " << socket_path_ << std::endl;
    }

    RobotAction GetRobotAction(const RobotBasicState& ro) override {
        EnsureConnected();

        VecXf obs = BuildObservation(ro);
        const uint32_t obs_len = static_cast<uint32_t>(obs_dim_);

        if (!SendAll(socket_fd_, &obs_len, sizeof(obs_len)) ||
            !SendAll(socket_fd_, obs.data(), sizeof(float) * obs_dim_)) {
            CloseSocket();
            throw std::runtime_error("[RKNN SOCKET] 发送观测失败，连接已断开");
        }

        uint32_t act_len = 0;
        if (!RecvAll(socket_fd_, &act_len, sizeof(act_len))) {
            CloseSocket();
            throw std::runtime_error("[RKNN SOCKET] 接收动作长度失败，连接已断开");
        }

        if (act_len != static_cast<uint32_t>(act_dim_)) {
            CloseSocket();
            throw std::runtime_error("[RKNN SOCKET] 推理服务返回的动作维度不匹配");
        }

        std::vector<float> action_buffer(act_dim_);
        if (!RecvAll(socket_fd_, action_buffer.data(), sizeof(float) * act_dim_)) {
            CloseSocket();
            throw std::runtime_error("[RKNN SOCKET] 接收动作数据失败，连接已断开");
        }

        action_ = Eigen::Map<VecXf>(action_buffer.data(), act_dim_);
        last_action_ = action_;

        tmp_action_ = DecodeAction(action_);

        ra_.goal_joint_pos = tmp_action_;
        ra_.goal_joint_vel = VecXf::Zero(act_dim_);
        ra_.tau_ff = VecXf::Zero(act_dim_);
        ra_.kp = kp_;
        ra_.kd = kd_;

        ++run_cnt_;
        return ra_;
    }
};

