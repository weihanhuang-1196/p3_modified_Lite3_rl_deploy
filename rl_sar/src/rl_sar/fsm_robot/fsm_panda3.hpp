/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PANDA3_FSM_HPP
#define PANDA3_FSM_HPP

#include "fsm.hpp"
#include "rl_sdk.hpp"

namespace panda3_fsm
{

class RLFSMStatePassive : public RLFSMState
{
public:
    RLFSMStatePassive(RL *rl) : RLFSMState(*rl, "RLFSMStatePassive") {}

    void Enter() override
    {
        std::cout << LOGGER::NOTE << "Entered passive mode. Press '0' (Keyboard) or 'A' (Gamepad) to switch to RLFSMStateGetUp." << std::endl;
    }

    void Run() override
    {
        for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
        {
            fsm_command->motor_command.q[i] = fsm_state->motor_state.q[i];
            fsm_command->motor_command.dq[i] = 0;
            fsm_command->motor_command.kp[i] = 0;
            fsm_command->motor_command.kd[i] = rl.params.Get<double>("damping_kd");
            fsm_command->motor_command.tau[i] = 0;
        }
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.motor_enabled && (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B))
        {
            return "RLFSMStateGetUp";
        }
        else if (rl.motor_enabled && (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A))
        {
            return "RLFSMStateRLStand";
        }
        return state_name_;
    }
};

class RLFSMStateGetUp : public RLFSMState
{
public:
    RLFSMStateGetUp(RL *rl) : RLFSMState(*rl, "RLFSMStateGetUp") {}

    float percent_pre_getup = 0.0f;
    float percent_getup = 0.0f;
    std::vector<float> pre_running_pos = {
        0.00, 1.36, -2.65,
        0.00, 1.36, -2.65,
        0.00, 1.36, -2.65,
        0.00, 1.36, -2.65,
        0.00, 0.00, 0.00, 0.00
    };
    bool stand_from_passive = true;

    void Enter() override
    {
        rl.ReadYaml(rl.robot_name, "base.yaml");
        percent_pre_getup = 0.0f;
        percent_getup = 0.0f;
        if (rl.fsm.previous_state_->GetStateName() == "RLFSMStatePassive")
        {
            stand_from_passive = true;
        }
        else
        {
            stand_from_passive = false;
        }
        rl.now_state = *fsm_state;
        rl.start_state = rl.now_state;
    }

    void Run() override
    {
        if(stand_from_passive)
        {

            if (Interpolate(percent_pre_getup, rl.now_state.motor_state.q, pre_running_pos, 1.0f, "Pre Getting up", true)) return;
            if (Interpolate(percent_getup, pre_running_pos, rl.params.Get<std::vector<float>>("default_dof_pos"), 2.0f, "Getting up", true)) return;
        }
        else
        {
            if (Interpolate(percent_getup, rl.now_state.motor_state.q, rl.params.Get<std::vector<float>>("default_dof_pos"), 1.0f, "Getting up", true)) return;
        }
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        if (percent_getup >= 1.0f)
        {
            if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::X)
            {
                return "RLFSMStateRLWalk";
            }
            else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
            {
                return "RLFSMStateRLStand";
            }
            else if (rl.control.current_keyboard == Input::Keyboard::Num6 || rl.control.current_gamepad == Input::Gamepad::Y)
            {
                return "RLFSMStateRLCrouch";
            }
            else if (rl.control.current_keyboard == Input::Keyboard::Num5)
            {
                return "RLFSMStateGetDown";
            }
        }
        return state_name_;
    }
};

class RLFSMStateGetDown : public RLFSMState
{
public:
    RLFSMStateGetDown(RL *rl) : RLFSMState(*rl, "RLFSMStateGetDown") {}

    std::vector<float> down_pos = {
        0.00, 2.34, -2.42,
        0.00, 2.34, -2.42,
        0.00, 2.34, -2.42,
        0.00, 2.34, -2.42,
        0.00, 0.00, 0.00, 0.00
    };

    std::vector<float> pre_running_pos = {
        0.00, 1.36, -2.65,
        0.00, 1.36, -2.65,
        0.00, 1.36, -2.65,
        0.00, 1.36, -2.65,
        0.00, 0.00, 0.00, 0.00
    };


    float percent_getdown = 0.0f;
    float percent_pre_getdown = 0.0f;

    void Enter() override
    {
        percent_getdown = 0.0f;
        percent_pre_getdown = 0.0f;
        rl.now_state = *fsm_state;
        rl.ReadYaml(rl.robot_name, "base.yaml");

    }

    void Run() override
    {

        if (Interpolate(percent_pre_getdown, rl.now_state.motor_state.q, pre_running_pos, 3.0f, "Pre Getting down", true)) return;
        if (Interpolate(percent_getdown, pre_running_pos, down_pos, 2.0f, "Getting down", true)) return;
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X || percent_getdown >= 1.0)
        {
            return "RLFSMStatePassive";
        }
        return state_name_;
    }
};

class RLFSMStateRLStand : public RLFSMState
{
public:
    RLFSMStateRLStand(RL *rl) : RLFSMState(*rl, "RLFSMStateRLStand") {}
    float percent_transition = 0.0f;
    void Enter() override
    {
        percent_transition = 0.0f;
        rl.episode_length_buf = 0;

        // read params from yaml
        rl.config_name = "himloco";
        std::string robot_config_path = rl.robot_name + "/" + rl.config_name + "/stand";
        try
        {
            rl.InitRL(robot_config_path, "RLFSMStateRLStand");
            if(fsm_state == nullptr)
                return;
            rl.now_state = *fsm_state;
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.fsm.RequestStateChange("RLFSMStatePassive");
        }
    }

    void Run() override
    {
        // position transition from last default_dof_pos to current default_dof_pos
        // if (Interpolate(percent_transition, rl.now_state.motor_state.q, rl.params.Get<std::vector<float>>("default_dof_pos"), 0.5f, "Policy transition", true)) return;

        if (!rl.rl_init_done) rl.rl_init_done = true;

        // std::cout << "\r\033[K" << std::flush << LOGGER::INFO << "RL Controller [" << rl.config_name << "] stand:" << rl.control.stand << std::flush;
        RLControl();
    }

    void Exit() override
    {
        rl.rl_init_done = false;
    }

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
        {
            return "RLFSMStateGetDown";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateRLStand";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::X)
        {
            return "RLFSMStateRLWalk";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num6 || rl.control.current_gamepad == Input::Gamepad::Y)
        {
            return "RLFSMStateRLCrouch";
        }
        return state_name_;
    }
    
};



class RLFSMStateRLWalk : public RLFSMState
{
public:
    RLFSMStateRLWalk(RL *rl) : RLFSMState(*rl, "RLFSMStateRLWalk") {}

    float percent_transition = 0.0f;

    void Enter() override
    {
        percent_transition = 0.0f;
        rl.episode_length_buf = 0;

        // read params from yaml
        rl.config_name = "himloco";
        std::string robot_config_path = rl.robot_name + "/" + rl.config_name + "/walk";
        try
        {
            rl.InitRL(robot_config_path, "RLFSMStateRLWalk");
            rl.now_state = *fsm_state;
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.fsm.RequestStateChange("RLFSMStatePassive");
        }
    }

    void Run() override
    {
        // position transition from last default_dof_pos to current default_dof_pos
        // if (Interpolate(percent_transition, rl.now_state.motor_state.q, rl.params.Get<std::vector<float>>("default_dof_pos"), 0.5f, "Policy transition", true)) return;

        if (!rl.rl_init_done) rl.rl_init_done = true;

        // std::cout << "\r\033[K" << std::flush << LOGGER::INFO << "RL Controller [" << rl.config_name << "] x:" << rl.control.x << " y:" << rl.control.y << " yaw:" << rl.control.yaw << std::flush;
        RLControl();
    }

    void Exit() override
    {
        rl.rl_init_done = false;
    }

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
        {
            return "RLFSMStateGetDown";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateRLStand";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::X)
        {
            return "RLFSMStateRLWalk";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num6 || rl.control.current_gamepad == Input::Gamepad::Y)
        {
            return "RLFSMStateRLCrouch";
        }
        
        return state_name_;
    }
};


class RLFSMStateRLCrouch : public RLFSMState
{
public:
    RLFSMStateRLCrouch(RL *rl) : RLFSMState(*rl, "RLFSMStateRLCrouch") {}

    float percent_transition = 0.0f;

    void Enter() override
    {
        percent_transition = 0.0f;
        rl.episode_length_buf = 0;

        // read params from yaml
        rl.config_name = "himloco";
        std::string robot_config_path = rl.robot_name + "/" + rl.config_name + "/crouch";
        try
        {
            rl.InitRL(robot_config_path, "RLFSMStateRLCrouch");
            rl.now_state = *fsm_state;
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.fsm.RequestStateChange("RLFSMStatePassive");
        }
    }

    void Run() override
    {
        // position transition from last default_dof_pos to current default_dof_pos
        // if (Interpolate(percent_transition, rl.now_state.motor_state.q, rl.params.Get<std::vector<float>>("default_dof_pos"), 0.5f, "Policy transition", true)) return;

        if (!rl.rl_init_done) rl.rl_init_done = true;

        // std::cout << "\r\033[K" << std::flush << LOGGER::INFO << "RL Controller [" << rl.config_name << "] x:" << rl.control.x << " y:" << rl.control.y << " yaw:" << rl.control.yaw << " height:" << rl.control.height << std::flush;
        RLControl();
    }

    void Exit() override
    {
        rl.rl_init_done = false;
    }

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
        {
            return "RLFSMStateGetDown";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateRLStand";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::X)
        {
            return "RLFSMStateRLWalk";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num6 || rl.control.current_gamepad == Input::Gamepad::Y)
        {
            return "RLFSMStateRLCrouch";
        }
        
        return state_name_;
    }
};







} // namespace panda3_fsm

class Panda3FSMFactory : public FSMFactory
{
public:
    Panda3FSMFactory(const std::string& initial) : initial_state_(initial) {}
    std::shared_ptr<FSMState> CreateState(void *context, const std::string &state_name) override
    {
        RL *rl = static_cast<RL *>(context);
        if (state_name == "RLFSMStatePassive")
            return std::make_shared<panda3_fsm::RLFSMStatePassive>(rl);
        else if (state_name == "RLFSMStateGetUp")
            return std::make_shared<panda3_fsm::RLFSMStateGetUp>(rl);
        else if (state_name == "RLFSMStateGetDown")
            return std::make_shared<panda3_fsm::RLFSMStateGetDown>(rl);
        else if (state_name == "RLFSMStateRLWalk")
            return std::make_shared<panda3_fsm::RLFSMStateRLWalk>(rl);
        else if (state_name == "RLFSMStateRLStand")
            return std::make_shared<panda3_fsm::RLFSMStateRLStand>(rl);
        else if (state_name == "RLFSMStateRLCrouch")
            return std::make_shared<panda3_fsm::RLFSMStateRLCrouch>(rl);
        return nullptr;
    }
    std::string GetType() const override { return "panda3"; }
    std::vector<std::string> GetSupportedStates() const override
    {
        return {
            "RLFSMStatePassive",
            "RLFSMStateGetUp",
            "RLFSMStateGetDown",
            "RLFSMStateRLWalk",
            "RLFSMStateRLStand",
            "RLFSMStateRLCrouch"
        };
    }
    std::string GetInitialState() const override { return initial_state_; }
private:
    std::string initial_state_;
};

REGISTER_FSM_FACTORY(Panda3FSMFactory, "RLFSMStatePassive")

#endif // PANDA3_FSM_HPP
