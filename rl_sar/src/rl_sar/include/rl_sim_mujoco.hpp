/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RL_SIM_HPP
#define RL_SIM_HPP

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
#include <memory>
#include <opencv2/opencv.hpp>

#include <mujoco/mujoco.h>
#include "joystick.hh"
#include "mujoco_utils.hpp"

#include "matplotlibcpp.h"
#include <algorithm>

#include "ldelay_monitor_macros.hpp"

#include "register_policies.hpp"

#ifndef NDEBUG

static LatencyStats world_stats("world_model", 500);
static LatencyStats policy_stats("policy_model", 500);
static LatencyStats forward_stats("forward", 50);
static LatencyStats run_model_stats("run_model", 50);

#else
#endif


namespace plt = matplotlibcpp;

class Button
{
public:
    Button() {}

    void update(bool state)
    {
        on_press = state ? state != pressed : false;
        on_release = state ? false : state != pressed;
        pressed = state;
    }

    bool pressed = false;
    bool on_press = false;
    bool on_release = false;
};

class RL_Sim : public RL
{
public:
    RL_Sim(int argc, char **argv);
    ~RL_Sim();

    std::unique_ptr<mj::Simulate> sim;
    static RL_Sim* instance;

private:
    // rl functions
    std::vector<float> Forward() override;
    void GetState(RobotState<float> *state) override;
    void SetCommand(const RobotCommand<float> *command) override;
    void RunModel();
    void RobotControl();

    // loop
    std::shared_ptr<LoopFunc> loop_keyboard;
    std::shared_ptr<LoopFunc> loop_joystick;
    std::shared_ptr<LoopFunc> loop_control;
    std::shared_ptr<LoopFunc> loop_rl;
    std::shared_ptr<LoopFunc> loop_plot;

    // plot
    const int plot_size = 100;
    std::vector<int> plot_t;
    std::vector<std::vector<float>> plot_real_joint_pos, plot_target_joint_pos;
    void Plot();

    // mujoco
    mjData *mj_data;
    mjModel *mj_model;
    std::string scene_name;

    // joystick
    std::unique_ptr<Joystick> sys_js;
    JoystickEvent sys_js_event;

    Button sys_js_button[20];
    int sys_js_axis[10] = {0};
    bool sys_js_active = false;
    float axis_deadzone = 0.05f;
    int sys_js_max_value = (1 << (16 - 1));
    void SetupSysJoystick(const std::string& device, int bits);
    void GetSysJoystick();

    // others
    std::string gazebo_model_name;
    std::map<std::string, float> joint_positions;
    std::map<std::string, float> joint_velocities;
    std::map<std::string, float> joint_efforts;
    void StartJointController(const std::string& ros_namespace, const std::vector<std::string>& names);

    void InitDepthCamera();
    std::vector<float> GetDepthImage();
    std::vector<float> depth_image_to_vector(const std::vector<float>& data, int width, int height);
    void show_depth_image(const std::vector<float>& depth_vec, int width, int height);

    mjvCamera depth_cam;      // 深度相机，只需初始化一次
    mjvOption depth_opt;      // 渲染选项
    mjvPerturb depth_pert;    // 用户扰动，通常不用动
    mjvScene depth_scene;     // 场景结构
    mjrContext depth_con;     // 渲染上下文
    std::vector<float> depth_buffer; // 深度缓冲
    int depth_width = 64;
    int depth_height = 64;

    bool init_camera_done = false;


};

#endif // RL_SIM_HPP
