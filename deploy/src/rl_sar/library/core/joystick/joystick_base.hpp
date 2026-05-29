#ifndef JOYSTICK_BASE_H
#define JOYSTICK_BASE_H


#include <iostream>
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>
#include "logger.hpp"

#include "rl_sdk.hpp"
#if defined(USE_ROS1)
#include <sensor_msgs/Joy.h>
#elif defined(USE_ROS2)
#include <sensor_msgs/msg/joy.hpp>
#endif
class joystick_base
{ 
public:
    joystick_base(RL& user_ref, std::string name) : user(user_ref), state_name_(std::move(name)) {};
    virtual ~joystick_base(){};

    void virtual JoyCallback(
        #if defined(USE_ROS1)
        const sensor_msgs::Joy::ConstPtr& msg
        #elif defined(USE_ROS2)
        const sensor_msgs::msg::Joy::SharedPtr msg
        #endif
    ) = 0;


    std::string GetJoystickName() const{
        return state_name_;
    };

protected:
    std::string state_name_;
    RL& user;

public:
#if defined(USE_ROS1)
    sensor_msgs::Joy joy_msg;
#elif defined(USE_ROS2)
    sensor_msgs::msg::Joy joy_msg;
#endif
};



class JoystickFactory
{
public:
    virtual ~JoystickFactory() = default;
    virtual std::shared_ptr<joystick_base> CreateJoystick(RL &context, const std::string &state_name) = 0;
    virtual std::string GetType() const = 0;
};


class JoystickManager
{
    public:
        static JoystickManager& GetInstance()
        {
            static JoystickManager instance;
            return instance;
        }


        void RegisterFactory(std::shared_ptr<JoystickFactory> factory)
        {
            if (factory)
            {
                std::string type = factory->GetType();
                factories_[type] = factory;
                std::cout << LOGGER::INFO << "[JoystickManager] Registered type: " << type << std::endl;
            }
        }

        std::shared_ptr<joystick_base> CreateJoystick(const std::string &type, RL &context)
        {
            auto it = factories_.find(type);
            if (it == factories_.end())
            {
                std::cout << LOGGER::ERROR << "[JoystickManager] Error: Unsupported type: " << type << std::endl;
                return nullptr;
            }
            auto factory = it->second;
            auto state = factory->CreateJoystick(context, it->first);
            std::cout << LOGGER::INFO << "[JoystickManager] use joystick type: " << it->first << std::endl;
            return state;
        }

    private:
        JoystickManager() = default;
        std::unordered_map<std::string, std::shared_ptr<JoystickFactory>> factories_;
};



#define CONCATENATE_DETAIL(x, y) x##y
#define CONCATENATE(x, y) CONCATENATE_DETAIL(x, y)
#define REGISTER_JOYSTICK_FACTORY(FactoryClass, initialStateName) \
    namespace { \
        const bool CONCATENATE(registered_joystick_factory_, __COUNTER__) = []() { \
            JoystickManager::GetInstance().RegisterFactory(std::make_shared<FactoryClass>(initialStateName)); \
            return true; \
        }(); \
    }



#endif // JOYSTICK_BASE_H