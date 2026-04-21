#ifndef JOYSTICK_BEITONG_H
#define JOYSTICK_BEITONG_H

#include "joystick_base.hpp"

#if defined(USE_ROS1)
#include <sensor_msgs/Joy.h>
#elif defined(USE_ROS2)
#include <sensor_msgs/msg/joy.hpp>
#endif


class joystick_beitong: public joystick_base
{
private:
    /* data */
public:
    joystick_beitong(RL& user_ref, std::string name) : joystick_base(user_ref, name){};
    virtual ~joystick_beitong(){};


    void JoyCallback(
        #if defined(USE_ROS1)
        const sensor_msgs::Joy::ConstPtr& msg
        #elif defined(USE_ROS2)
        const sensor_msgs::msg::Joy::SharedPtr msg
        #endif
    ) override
    {
        this->joy_msg = *msg;
        if (this->joy_msg.buttons[0]) this->user.control.SetGamepad(Input::Gamepad::A);
        if (this->joy_msg.buttons[1]) this->user.control.SetGamepad(Input::Gamepad::B);
        if (this->joy_msg.buttons[3]) this->user.control.SetGamepad(Input::Gamepad::Y);
        if (this->joy_msg.buttons[4]) this->user.control.SetGamepad(Input::Gamepad::X);
        if (this->joy_msg.buttons[6]) this->user.control.SetGamepad(Input::Gamepad::LB_X);
        if (this->joy_msg.buttons[7]) this->user.control.SetGamepad(Input::Gamepad::RB);
        // if (this->joy_msg.buttons[9]) this->user.control.SetGamepad(Input::Gamepad::LStick);
        // if (this->joy_msg.buttons[10]) this->user.control.SetGamepad(Input::Gamepad::RStick);
        // if (this->joy_msg.axes[7] > 0) this->user.control.SetGamepad(Input::Gamepad::DPadUp);
        // if (this->joy_msg.axes[7] < 0) this->user.control.SetGamepad(Input::Gamepad::DPadDown);
        // if (this->joy_msg.axes[6] < 0) this->user.control.SetGamepad(Input::Gamepad::DPadLeft);
        // if (this->joy_msg.axes[6] > 0) this->user.control.SetGamepad(Input::Gamepad::DPadRight);
        // if (this->joy_msg.axes[7] > 0 && this->joy_msg.buttons[2]) this->user.control.SetGamepad(Input::Gamepad::X);
        // if (this->joy_msg.axes[7] < 0 && this->joy_msg.buttons[2]) this->user.control.SetGamepad(Input::Gamepad::X);
        // if (this->joy_msg.axes[7] > 0 && this->joy_msg.buttons[3]) this->user.control.SetGamepad(Input::Gamepad::Y);
        // if (this->joy_msg.axes[7] < 0 && this->joy_msg.buttons[3]) this->user.control.SetGamepad(Input::Gamepad::Y);
        // if (this->joy_msg.buttons[4] && this->joy_msg.buttons[0]) this->user.control.SetGamepad(Input::Gamepad::LB_A);
        // if (this->joy_msg.buttons[4] && this->joy_msg.buttons[1]) this->user.control.SetGamepad(Input::Gamepad::LB_B);
        // if (this->joy_msg.buttons[4] && this->joy_msg.buttons[3]) this->user.control.SetGamepad(Input::Gamepad::LB_X);
        // if (this->joy_msg.buttons[4] && this->joy_msg.buttons[2]) this->user.control.SetGamepad(Input::Gamepad::LB_Y);
        // if (this->joy_msg.buttons[4] && this->joy_msg.buttons[9]) this->user.control.SetGamepad(Input::Gamepad::LB_LStick);
        // if (this->joy_msg.buttons[4] && this->joy_msg.buttons[10]) this->user.control.SetGamepad(Input::Gamepad::LB_RStick);
        // if (this->joy_msg.buttons[4] && this->joy_msg.axes[7] > 0) this->user.control.SetGamepad(Input::Gamepad::LB_DPadUp);
        // if (this->joy_msg.buttons[4] && this->joy_msg.axes[7] < 0) this->user.control.SetGamepad(Input::Gamepad::LB_DPadDown);
        // if (this->joy_msg.buttons[4] && this->joy_msg.axes[6] < 0) this->user.control.SetGamepad(Input::Gamepad::LB_DPadRight);
        // if (this->joy_msg.buttons[4] && this->joy_msg.axes[6] > 0) this->user.control.SetGamepad(Input::Gamepad::LB_DPadLeft);
        // if (this->joy_msg.buttons[5] && this->joy_msg.buttons[0]) this->user.control.SetGamepad(Input::Gamepad::RB_A);
        // if (this->joy_msg.buttons[5] && this->joy_msg.buttons[1]) this->user.control.SetGamepad(Input::Gamepad::RB_B);
        // if (this->joy_msg.buttons[5] && this->joy_msg.buttons[2]) this->user.control.SetGamepad(Input::Gamepad::RB_X);
        // if (this->joy_msg.buttons[5] && this->joy_msg.buttons[3]) this->user.control.SetGamepad(Input::Gamepad::RB_Y);
        // if (this->joy_msg.buttons[5] && this->joy_msg.buttons[9]) this->user.control.SetGamepad(Input::Gamepad::RB_LStick);
        // if (this->joy_msg.buttons[5] && this->joy_msg.buttons[10]) this->user.control.SetGamepad(Input::Gamepad::RB_RStick);
        // if (this->joy_msg.buttons[5] && this->joy_msg.axes[7] > 0) this->user.control.SetGamepad(Input::Gamepad::RB_DPadUp);
        // if (this->joy_msg.buttons[5] && this->joy_msg.axes[7] < 0) this->user.control.SetGamepad(Input::Gamepad::RB_DPadDown);
        // if (this->joy_msg.buttons[5] && this->joy_msg.axes[6] < 0) this->user.control.SetGamepad(Input::Gamepad::RB_DPadRight);
        // if (this->joy_msg.buttons[5] && this->joy_msg.axes[6] > 0) this->user.control.SetGamepad(Input::Gamepad::RB_DPadLeft);
        // if (this->joy_msg.buttons[4] && this->joy_msg.buttons[5]) this->user.control.SetGamepad(Input::Gamepad::LB_RB);

        this->user.control.x = this->joy_msg.axes[1]; // LY
        this->user.control.y = this->joy_msg.axes[0]; // LX
        this->user.control.yaw = this->joy_msg.axes[2]; // RX
        this->user.control.stand = (this->joy_msg.axes[7] == 1 ? 1.0f : 0.0f);
        this->user.control.height = this->joy_msg.axes[4];

        if(this->joy_msg.buttons[11] == 1)
        {
            this->user.motor_enable_changed = true;  
        }

        if (this->joy_msg.buttons[11] == 0 && this->user.motor_enable_changed)
        {
            this->user.motor_enable_changed = false;
            this->user.motor_enabled = !this->user.motor_enabled;
        }
    

    };

};


class beitongFactory : public JoystickFactory
{
public:
    beitongFactory(const std::string& name) : name_(name) {}
    std::shared_ptr<joystick_base> CreateJoystick(RL &context, const std::string &state_name) override
    {
        return std::make_shared<joystick_beitong>(context, state_name);
    }
    std::string GetType() const override { return "beitong"; }

public:
    std::string name_;
 
};


REGISTER_JOYSTICK_FACTORY(beitongFactory, "beitong")



#endif  // JOYSTICK_BEITONG_H