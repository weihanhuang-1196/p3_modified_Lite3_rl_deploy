#ifndef POLICY_CONTEXT_HPP
#define POLICY_CONTEXT_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <stdexcept>

namespace rl_policy {

struct Tensor
{
    std::vector<float> data;
    std::vector<int64_t> shape;
};

struct RobotState
{
    std::vector<float> lin_vel;
    std::vector<float> ang_vel;
    std::vector<float> gravity_vec;
    std::vector<float> base_quat;
};

struct JointState
{
    std::vector<float> dof_pos;
    std::vector<float> dof_vel;
};

struct CommandState
{
    std::vector<float> velocity;
    std::vector<float> gait;
};

struct PolicyContext
{
    RobotState robot;
    JointState joints;
    CommandState command;

    std::vector<float> last_actions;

    std::unordered_map<std::string, Tensor> tensors;

    double timestamp = 0.0;

    bool HasTensor(const std::string& name) const
    {
        return tensors.find(name) != tensors.end();
    }

    const Tensor& GetTensor(const std::string& name) const
    {
        auto it = tensors.find(name);

        if (it == tensors.end())
        {
            throw std::runtime_error("Missing tensor in PolicyContext: " + name);
        }

        return it->second;
    }
};

}  // namespace rl_policy



#endif