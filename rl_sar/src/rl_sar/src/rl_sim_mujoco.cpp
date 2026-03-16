/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sim_mujoco.hpp"



#define TIME_START auto __start = std::chrono::high_resolution_clock::now();
#define TIME_END(msg) \
    auto __end = std::chrono::high_resolution_clock::now(); \
    std::cout << LOGGER::WARNING << msg << " took " \
              << std::chrono::duration_cast<std::chrono::microseconds>(__end - __start).count() \
              << " us\n";


RL_Sim* RL_Sim::instance = nullptr;

RL_Sim::RL_Sim(int argc, char **argv)
{
    // Set static instance pointer early for signal handler
    instance = this;

    if (argc < 3)
    {
        std::cout << LOGGER::ERROR << "Usage: " << argv[0] << " robot_name scene_name" << std::endl;
        throw std::runtime_error("Invalid arguments");
    }
    else
    {
        this->robot_name = argv[1];
        this->scene_name = argv[2];
    }

    this->ang_vel_axis = "body";

    // now launch mujoco
    std::cout << LOGGER::INFO << "[MuJoCo] Launching..." << std::endl;

    // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
    if (rosetta_error_msg)
    {
        DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
        std::exit(1);
    }
#endif

    // print version, check compatibility
    std::cout << LOGGER::INFO << "[MuJoCo] Version: " << mj_versionString() << std::endl;
    if (mjVERSION_HEADER != mj_version())
    {
        mju_error("Headers and library have different versions");
    }

    // scan for libraries in the plugin directory to load additional plugins
    scanPluginLibraries();

    mjvCamera cam;
    mjv_defaultCamera(&cam);

    mjvOption opt;
    mjv_defaultOption(&opt);

    mjvPerturb pert;
    mjv_defaultPerturb(&pert);

    // simulate object encapsulates the UI
    sim = std::make_unique<mj::Simulate>(
        std::make_unique<mj::GlfwAdapter>(),
        &cam, &opt, &pert, /* is_passive = */ false);

    std::string filename = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/../rl_sar_zoo/" + this->robot_name + "_description/mjcf/" + this->scene_name + ".xml";

    // start physics thread
    std::thread physicsthreadhandle(&PhysicsThread, sim.get(), filename.c_str());
    physicsthreadhandle.detach();

    while (1)
    {
        if (d)
        {
            std::cout << LOGGER::INFO << "[MuJoCo] Data prepared" << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    this->mj_model = m; // mujoco 模型结构，包含了物理参数等信息，后续会频繁访问
    this->mj_data = d; //mujoco 数据结构，包含了当前的物理状态等信息，后续会频繁访问



    this->SetupSysJoystick("/dev/input/js0", 16); // 16 bits joystick

    // read params from yaml
    this->ReadYaml(this->robot_name, "base.yaml");
    this->config_name = this->params.Get<std::string>("algorithm");

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
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();

    // loop
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Sim::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Sim::RunModel, this));
    this->loop_control->start();
    this->loop_rl->start();

    // keyboard
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Sim::KeyboardInterface, this));
    this->loop_keyboard->start();

    // joystick
    this->loop_joystick = std::make_shared<LoopFunc>("loop_joystick", 0.01, std::bind(&RL_Sim::GetSysJoystick, this));
    this->loop_joystick->start();

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

    std::cout << LOGGER::INFO << "RL_Sim start" << std::endl;

    // start simulation UI loop (blocking call)
    sim->RenderLoop();
}

RL_Sim::~RL_Sim()
{
    // Clear static instance pointer
    instance = nullptr;

    this->loop_keyboard->shutdown();
    this->loop_joystick->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;
}


void RL_Sim::InitDepthCamera()
{

    glfwInit();                 // 初始化GLFW库
    GLFWwindow* window = glfwCreateWindow(640, 480, "", NULL, NULL); // 创建OpenGL context

    glfwMakeContextCurrent(window); // 激活context

    // camera 初始化
    mjv_defaultCamera(&this->depth_cam);
    this->depth_cam.type = mjCAMERA_FIXED;
    this->depth_cam.fixedcamid = mj_name2id(this->mj_model, mjOBJ_CAMERA, "depth_cam"); // depth_cam 名字
    this->depth_cam.lookat[0] = 0.37;
    this->depth_cam.lookat[1] = 0.0;
    this->depth_cam.lookat[2] = 0.3;
    this->depth_cam.distance = 0.5;

    // option & perturb
    mjv_defaultOption(&this->depth_opt);
    mjv_defaultPerturb(&this->depth_pert);

    // scene, 注意 3.2.7 需要 maxgeom 参数
    mjv_defaultScene(&this->depth_scene);
    mjv_makeScene(this->mj_model, &this->depth_scene, 2000);
    this->depth_scene.scale = 1.0;

    mjr_defaultContext(&this->depth_con);
    // context
    mjr_makeContext(this->mj_model, &this->depth_con, mjFONTSCALE_150);

    mjr_setBuffer(mjFB_OFFSCREEN, &this->depth_con);

    // depth buffer
    this->depth_buffer.resize(this->depth_width * this->depth_height, 0.0f);

    
}

std::vector<float> RL_Sim::GetDepthImage()
{
    // 检查模型和数据
    if (!mj_model || !mj_data)
    {
        std::cerr << LOGGER::ERROR << "mj_model or mj_data is null!" << std::endl;
        return std::vector<float>(this->depth_width * this->depth_height, 0.0f);
    }
    float old_znear = this->mj_model->vis.map.znear;
    float old_zfar  = this->mj_model->vis.map.zfar;

    mj_model->vis.map.znear = this->znear;
    mj_model->vis.map.zfar = this->zfar;

    // 更新场景
    mjv_updateScene(
        mj_model,
        mj_data,
        &depth_opt,
        &depth_pert,
        &depth_cam,
        mjCAT_ALL, // 更新所有类别
        &depth_scene
    );

    // 渲染到 GPU buffer
    mjrRect viewport = {0, 0, depth_width, depth_height};
    try
    {
        mjr_render(viewport, &depth_scene, &depth_con);
    }
    catch (...)
    {
        std::cerr << LOGGER::ERROR << "mjr_render failed!" << std::endl;
        return std::vector<float>(depth_width * depth_height, 0.0f);
    }

    // 读取深度缓冲
    try
    {
        mjr_readPixels(nullptr, depth_buffer.data(), viewport, &depth_con);
    }
    catch (...)
    {
        std::cerr << LOGGER::ERROR << "mjr_readPixels failed!" << std::endl;
        return std::vector<float>(depth_width * depth_height, 0.0f);
    }

    this->mj_model->vis.map.znear = old_znear;
    this->mj_model->vis.map.zfar  = old_zfar;

    // 转换为真实深度
    const float znear = this->znear;
    const float zfar  = this->zfar;
    for (auto &d : depth_buffer)
    {
        if (std::isnan(d)) d = 0.0f;
        else if (std::isinf(d)) d = (d > 0 ? 1.0f : 0.0f); // depth buffer 范围是 0~1

        // 线性化
        d = znear * zfar / (zfar - d * (zfar - znear));

        // clamp 到有效范围
        d = std::clamp(d, znear, zfar);
    }

    // 扁平化处理
    return depth_image_to_vector(depth_buffer, depth_width, depth_height);
}

std::vector<float> RL_Sim::depth_image_to_vector(const std::vector<float>& data, int width, int height)
{
    const float min_depth = this->znear;
    const float max_depth = 2.0f;

    auto maxv = *std::max_element(data.begin(), data.end());
    auto minv = *std::min_element(data.begin(), data.end());
    std::vector<float> depth_vec;
    depth_vec.reserve(width * height);

    // reinterpret_cast 指向原始 float 数据
    const float* data_ptr = data.data();

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

    auto maxv_depth = *std::max_element(depth_vec.begin(), depth_vec.end());
    auto minv_depth = *std::min_element(depth_vec.begin(), depth_vec.end());

    // // 显示深度图
    // cv::Mat depth_display;
    // depth_mat.convertTo(depth_display, CV_8UC1, 255.0, 127); // x*255 + 127

    //     // 3. 放大（插值）
    // cv::resize(depth_display, depth_mat, cv::Size(width*6, height*6), 0, 0, cv::INTER_LINEAR);

    // // cv::applyColorMap(depth_up, depth_display, cv::COLORMAP_JET); // 彩色深度图
    // cv::imshow("Depth Image", depth_mat);
    // cv::waitKey(1);
    // show_depth_image(depth_vec, width, height);

    return depth_vec; // 扁平化 vector
}


void RL_Sim::show_depth_image(const std::vector<float>& depth_vec, int width, int height)
{
    cv::Mat depth_mat(height, width, CV_32FC1, const_cast<float*>(depth_vec.data()));
    cv::Mat depth_display;
    depth_mat.convertTo(depth_display, CV_8UC1, 255.0, 127); // x*255 + 127

            // 3. 放大（插值）
    cv::Mat depth_up;
    cv::resize(depth_display, depth_up, cv::Size(width*6, height*6), 0, 0, cv::INTER_LINEAR);

    cv::imshow("Depth Image", depth_up);
    cv::waitKey(1);
}


void RL_Sim::GetState(RobotState<float> *state)
{
    if (mj_data)
    {
        state->imu.quaternion[0] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 0];
        state->imu.quaternion[1] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 1];
        state->imu.quaternion[2] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 2];
        state->imu.quaternion[3] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 3];

        state->imu.gyroscope[0] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 4];
        state->imu.gyroscope[1] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 5];
        state->imu.gyroscope[2] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 6];
        // std::cout << LOGGER::DEBUG << "Gyro: "
        //           << state->imu.gyroscope[0] << ", "
        //           << state->imu.gyroscope[1] << ", "
        //           << state->imu.gyroscope[2] << std::endl;

        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            state->motor_state.q[i] = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i]];
            state->motor_state.dq[i] = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + this->params.Get<int>("num_of_dofs")];
            state->motor_state.tau_est[i] = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + 2 * this->params.Get<int>("num_of_dofs")];
        }

       std::atomic_store_explicit(&this->depth_image_ptr, std::make_shared<std::vector<float>>(std::move(GetDepthImage())), std::memory_order_release);


    }
}

void RL_Sim::SetCommand(const RobotCommand<float> *command)
{
    if (mj_data)
    {
#if 0    // 使用上次位置计算速度
        static std::vector<double> last_q = std::vector<double>(this->params.Get<int>("num_of_dofs"), 0.0);
        double current_q, current_dq;
        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            current_q = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i]];
            current_dq = (current_q - last_q[i]) / (double)this->params.Get<float>("dt");
            mj_data->ctrl[this->params.Get<std::vector<int>>("joint_mapping")[i]] =
                command->motor_command.tau[i] +
                command->motor_command.kp[i] * (command->motor_command.q[i] - current_q) +
                // command->motor_command.kd[i] * (command->motor_command.dq[i] - mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + this->params.Get<int>("num_of_dofs")]);
                command->motor_command.kd[i] *(command->motor_command.dq[i] - current_dq);
            last_q[i] = current_q;
        }
#else   // 使用传感器速度
        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            mj_data->ctrl[this->params.Get<std::vector<int>>("joint_mapping")[i]] =
                // command->motor_command.tau[i] +
                command->motor_command.kp[i] * (command->motor_command.q[i] - mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i]]) +
                command->motor_command.kd[i] * (command->motor_command.dq[i] - mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + this->params.Get<int>("num_of_dofs")]);
        }
#endif
    }


}

void RL_Sim::RobotControl()
{

    if(!init_camera_done)
    {
        InitDepthCamera();
        init_camera_done = true;
    }

    // Lock the sim mutex once for the entire control cycle to prevent race conditions
    const std::lock_guard<std::recursive_mutex> lock(sim->mtx);

    this->GetState(&this->robot_state);

    this->StateController(&this->robot_state, &this->robot_command);

    if (this->control.current_keyboard == Input::Keyboard::R || this->control.current_gamepad == Input::Gamepad::RB_Y)
    {
        if (this->mj_model && this->mj_data)
        {
            mj_resetData(this->mj_model, this->mj_data);
            mj_forward(this->mj_model, this->mj_data);
        }
    }
    if (this->control.current_keyboard == Input::Keyboard::Enter || this->control.current_gamepad == Input::Gamepad::RB_X)
    {
        if (simulation_running)
        {
            sim->run = 0;
            std::cout << std::endl << LOGGER::INFO << "Simulation Stop" << std::endl;
        }
        else
        {
            sim->run = 1;
            std::cout << std::endl << LOGGER::INFO << "Simulation Start" << std::endl;
        }
        simulation_running = !simulation_running;
    }

    this->control.ClearInput();

    this->SetCommand(&this->robot_command);
}

void RL_Sim::SetupSysJoystick(const std::string& device, int bits)
{
    this->sys_js = std::make_unique<Joystick>(device);
    if (!this->sys_js->isFound())
    {
        std::cout << LOGGER::ERROR << "Joystick [" << device << "] open failed." << std::endl;
        // exit(1);
    }

    this->sys_js_max_value = (1 << (bits - 1));
}

void RL_Sim::GetSysJoystick()
{
    // Clear all button event states
    for (int i = 0; i < 20; ++i)
    {
        this->sys_js_button[i].on_press = false;
        this->sys_js_button[i].on_release = false;
    }

    // Check if joystick is valid before using
    if (!this->sys_js)
    {
        return;
    }

    while (this->sys_js->sample(&this->sys_js_event))
    {
        if (this->sys_js_event.isButton())
        {
            this->sys_js_button[this->sys_js_event.number].update(this->sys_js_event.value);
        }
        else if (this->sys_js_event.isAxis())
        {
            double normalized = double(this->sys_js_event.value) / this->sys_js_max_value;
            if (std::abs(normalized) < this->axis_deadzone)
            {
                this->sys_js_axis[this->sys_js_event.number] = 0;
            }
            else
            {
                this->sys_js_axis[this->sys_js_event.number] = this->sys_js_event.value;
            }
        }
    }

    if (this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::A);
    if (this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::B);
    if (this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::X);
    if (this->sys_js_button[4].on_press) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->sys_js_button[6].on_press) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->sys_js_button[7].on_press) this->control.SetGamepad(Input::Gamepad::RB);
    // if (this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::LStick);
    // if (this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::RStick);
    // if (this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::DPadUp);
    // if (this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::DPadDown);
    // if (this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    // if (this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::DPadRight);
    // if (this->sys_js_axis[7] > 0 && this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::X);
    // if (this->sys_js_axis[7] < 0 && this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::X);
    // if (this->sys_js_axis[7] > 0 && this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::Y);
    // if (this->sys_js_axis[7] < 0 && this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::Y);
    // if (this->sys_js_button[4].pressed && this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::LB_A);
    // if (this->sys_js_button[4].pressed && this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::LB_B);
    // if (this->sys_js_button[4].pressed && this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::LB_X);
    // if (this->sys_js_button[4].pressed && this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::LB_Y);
    // if (this->sys_js_button[4].pressed && this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    // if (this->sys_js_button[4].pressed && this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    // if (this->sys_js_button[4].pressed && this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    // if (this->sys_js_button[4].pressed && this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    // if (this->sys_js_button[4].pressed && this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    // if (this->sys_js_button[4].pressed && this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    // if (this->sys_js_button[5].pressed && this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::RB_A);
    // if (this->sys_js_button[5].pressed && this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::RB_B);
    // if (this->sys_js_button[5].pressed && this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::RB_X);
    // if (this->sys_js_button[5].pressed && this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::RB_Y);
    // if (this->sys_js_button[5].pressed && this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    // if (this->sys_js_button[5].pressed && this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    // if (this->sys_js_button[5].pressed && this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    // if (this->sys_js_button[5].pressed && this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    // if (this->sys_js_button[5].pressed && this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    // if (this->sys_js_button[5].pressed && this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    // if (this->sys_js_button[4].pressed && this->sys_js_button[5].on_press) this->control.SetGamepad(Input::Gamepad::LB_RB);

    float ly = -float(this->sys_js_axis[1]) / float(this->sys_js_max_value);
    float lx = -float(this->sys_js_axis[0]) / float(this->sys_js_max_value);
    float rx = -float(this->sys_js_axis[3]) / float(this->sys_js_max_value);

    bool has_input = (ly != 0.0f || lx != 0.0f || rx != 0.0f);

    if (has_input)
    {
        this->control.x = ly;
        this->control.y = lx;
        this->control.yaw = rx;
        this->sys_js_active = true;
    }
    else if (this->sys_js_active)
    {
        this->control.x = 0.0f;
        this->control.y = 0.0f;
        this->control.yaw = 0.0f;
        this->sys_js_active = false;
    }
    // this->control.stand = (this->sys_js_axis[7] < 0 ? 1.0f : 0.0f);
    // this->control.height = -float(this->sys_js_axis[4]) / float(this->sys_js_max_value);



    if(this->sys_js_button[11].pressed)
    {
        motor_enable_changed = true;
    }

    if (this->sys_js_button[11].pressed && motor_enable_changed)
    {
        motor_enable_changed = false;
        this->motor_enabled = !this->motor_enabled;
    }



}

void RL_Sim::RunModel()
{
    // TIME_START
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
        //not currently available for non-ros mujoco version
        // if (this->control.navigation_mode)
        // {
        //     this->obs.commands = {(float)this->cmd_vel.linear.x, (float)this->cmd_vel.linear.y, (float)this->cmd_vel.angular.z};
        // }
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
            tau_est[i] = this->joint_efforts[this->params.Get<std::vector<std::string>>("joint_controller_names")[i]];
        }
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }

    // TIME_END("RunModel")
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
        }else if(this->config_name == "himloco")
        {
            this->history_obs_buf.insert(clamped_obs);
            this->history_obs = this->history_obs_buf.get_obs_vec(this->params.Get<std::vector<int>>("observations_history"));
            actions = this->model->forward({this->history_obs});
        }else if(this->config_name == "wmp")
        {
            this->wm_action_history.erase(this->wm_action_history.begin(), this->wm_action_history.begin() + this->wm_action.size());
            this->wm_action_history.insert(this->wm_action_history.end(), this->wm_action.begin(), this->wm_action.end());
            this->wm_action = this->wm_action_history;

            std::vector<float> input_image(this->image_width * this->image_height, 0.0f);
            auto depth_image = std::atomic_load_explicit(&this->depth_image_ptr, std::memory_order_acquire);
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
                show_depth_image(input_image, this->image_width, this->image_height);
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
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
        this->plot_real_joint_pos[i].push_back(mj_data->sensordata[i]);
        this->plot_target_joint_pos[i].push_back(this->robot_command.motor_command.q[i]);  // TODO
        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.01);
}

// Signal handler for Ctrl+C
void signalHandler(int signum)
{
    std::cout << LOGGER::INFO << "Received signal " << signum << ", exiting..." << std::endl;
    if (RL_Sim::instance && RL_Sim::instance->sim)
    {
        RL_Sim::instance->sim->exitrequest.store(1);
    }
}

int main(int argc, char **argv)
{
    std::string robot_name(argc > 1 ? argv[1] : "");
    signal(SIGINT, signalHandler);
    RL_Sim rl_sar(argc, argv);
    return 0;
}
