#include "policy_context_builder.hpp"
namespace rl_policy {

PolicyContext PolicyContextBuilder::Build()
{
    return context_;
}

void PolicyContextBuilder::SetRobotState(
    const std::vector<float>& lin_vel,
    const std::vector<float>& ang_vel,
    const std::vector<float>& gravity_vec,
    const std::vector<float>& base_quat)
{
    context_.robot.lin_vel = lin_vel;
    context_.robot.ang_vel = ang_vel;
    context_.robot.gravity_vec = gravity_vec;
    context_.robot.base_quat = base_quat;
}

void PolicyContextBuilder::SetJointState(
    const std::vector<float>& dof_pos,
    const std::vector<float>& dof_vel)
{
    context_.joints.dof_pos = dof_pos;
    context_.joints.dof_vel = dof_vel;
}

void PolicyContextBuilder::SetCommand(
    const std::vector<float>& velocity_command)
{
    context_.command.velocity = velocity_command;
}

void PolicyContextBuilder::SetLastActions(
    const std::vector<float>& last_actions)
{
    context_.last_actions = last_actions;
}

// void PolicyContextBuilder::SetDepthImage(
//     const std::vector<float>& depth_image,
//     const std::vector<int64_t>& shape)
// {
//     context_.tensors[tensor_keys::DEPTH_IMAGE] = Tensor{
//         depth_image,
//         shape
//     };
// }

}  // namespace rl_policy