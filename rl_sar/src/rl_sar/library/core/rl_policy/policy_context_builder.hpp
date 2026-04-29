#ifndef POLICY_CONTEXT_BUILDER_HPP
#define POLICY_CONTEXT_BUILDER_HPP

#include "policy_context.hpp"

namespace rl_policy {

class PolicyContextBuilder
{
public:
    PolicyContext Build();

    void SetRobotState(
        const std::vector<float>& lin_vel,
        const std::vector<float>& ang_vel,
        const std::vector<float>& gravity_vec,
        const std::vector<float>& base_quat);

    void SetJointState(
        const std::vector<float>& dof_pos,
        const std::vector<float>& dof_vel);

    void SetCommand(
        const std::vector<float>& velocity_command);

    void SetLastActions(
        const std::vector<float>& last_actions);

    // void SetDepthImage(
    //     const std::vector<float>& depth_image,
    //     const std::vector<int64_t>& shape);

private:
    PolicyContext context_;
};

}  // namespace rl_policy

#endif