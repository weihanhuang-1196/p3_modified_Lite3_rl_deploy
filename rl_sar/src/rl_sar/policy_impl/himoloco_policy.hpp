#ifndef HIMOLOCO_POLICY_HPP
#define HIMOLOCO_POLICY_HPP

#include "policy_base.hpp"

namespace rl_policy{

class HimolocoPolicy : public PolicyBase
{
private:
    /* data */
public:
    HimolocoPolicy(/* args */);
    ~HimolocoPolicy();

    HimolocoPolicy(const HimolocoPolicy&) = delete;
    HimolocoPolicy& operator=(const HimolocoPolicy&) = delete;

protected:
    void ComputeOutput(const std::vector<float>& actions, std::vector<float> &output_dof_pos, std::vector<float> &output_dof_vel, std::vector<float> &output_dof_tau) override;
    void onInit() override;
    void LoadModel() override;
    std::vector<float> BuildObservation(const PolicyContext& context) override;
    std::vector<float> ProcessObservation(const std::vector<float>& obs) override;
    std::vector<float> RunInference(const std::vector<float>& model_input) override;
    PolicyOutput ComputeOutput(const std::vector<float>& actions, const PolicyContext& context) override;


};



}   //namespace rl_policy


#endif