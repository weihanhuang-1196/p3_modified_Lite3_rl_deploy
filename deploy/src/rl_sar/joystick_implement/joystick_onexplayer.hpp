#ifndef JOYSTICK_ONEPLAYER_H
#define JOYSTICK_ONEPLAYER_H


#include "joystick_base.hpp"

#if defined(USE_ROS1)
#include <sensor_msgs/Joy.h>
#elif defined(USE_ROS2)
#include <sensor_msgs/msg/joy.hpp>
#endif

class joystick_onexplayer: public joystick_base
{
private:
    /* data */
public:
    joystick_onexplayer(RL& user_ref, std::string name) : joystick_base(user_ref, name){};
    virtual ~joystick_onexplayer(){};


    void JoyCallback(
        #if defined(USE_ROS1)
        const sensor_msgs::Joy::ConstPtr& msg
        #elif defined(USE_ROS2)
        const sensor_msgs::msg::Joy::SharedPtr msg
        #endif
    ) override
    {
        // this->joy_msg.buttons[0] 上使能
        // this->joy_msg.buttons[13] 下使能
        // this->joy_msg.buttons[3] B 力控站立
        // this->joy_msg.buttons[7] LB 力控站立
        // this->joy_msg.buttons[2] A  行走模型
        // this->joy_msg.buttons[8] RB 力控蹲下



        this->joy_msg = *msg;
        if ( (this->joy_msg.buttons[3]) || (this->joy_msg.buttons[7]) ) this->user.control.SetGamepad(Input::Gamepad::B);
        if (this->joy_msg.buttons[2]) this->user.control.SetGamepad(Input::Gamepad::X);
        if (this->joy_msg.buttons[8]) this->user.control.SetGamepad(Input::Gamepad::RB);

        if(this->joy_msg.buttons[0] == 1) 
        {
            this->user.control.SetGamepad(Input::Gamepad::LB_X);
            this->user.motor_enabled = true;
        }

        if (this->joy_msg.buttons[13] == 1) this->user.motor_enabled = false;

        this->user.control.x = this->joy_msg.axes[1]; // LY
        this->user.control.y = this->joy_msg.axes[0]; // LX
        this->user.control.yaw = this->joy_msg.axes[3]; // RX


    

    };

};


class OnexplayerFactory : public JoystickFactory
{
public:
    OnexplayerFactory(const std::string& name) : name_(name) {}
    std::shared_ptr<joystick_base> CreateJoystick(RL &context, const std::string &state_name) override
    {
        return std::make_shared<joystick_onexplayer>(context, state_name);
    }
    std::string GetType() const override { return "onexplayer"; }

public:
    std::string name_;
 
};


REGISTER_JOYSTICK_FACTORY(OnexplayerFactory, "onexplayer")





#endif // JOYSTICK_ONEPLAYER_H