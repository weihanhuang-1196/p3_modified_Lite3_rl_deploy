#ifndef POLICY_OUTPUT_HPP
#define POLICY_OUTPUT_HPP

#include <vector>

namespace rl_policy {

struct PolicyOutput
{
    std::vector<float> target_dof_pos;
    std::vector<float> target_dof_vel;
    std::vector<float> target_dof_tau;

    std::vector<float> raw_actions;

    bool valid = true;
    double inference_time_ms = 0.0;
};

}  // namespace rl_policy

#endif