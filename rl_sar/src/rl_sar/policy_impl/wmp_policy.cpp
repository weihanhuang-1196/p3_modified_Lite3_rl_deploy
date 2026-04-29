#include "wmp_policy.hpp"


namespace rl_policy{

void WMPPolicy::OnInit()
{

}

void WMPPolicy::OnReset()
{

}

void WMPPolicy::LoadModel(const std::string & policy_dir)
{

}


void WMPPolicy::BuildObservation(const PolicyContext& context)
{

}

std::vector<float>& WMPPolicy::ProcessObservation()
{

}


std::vector<float> WMPPolicy::RunInference(const std::vector<float>& model_input)
{

}

PolicyOutput& WMPPolicy::ComputeOutput(const std::vector<float>& actions, const PolicyContext& context)
{

}


std::vector<float> WMPPolicy::ComputeObservation(const PolicyContext& context)
{

}




} //namespace rl_policy