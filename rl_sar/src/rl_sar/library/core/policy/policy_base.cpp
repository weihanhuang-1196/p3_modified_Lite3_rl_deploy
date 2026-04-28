#include "policy_base.hpp"



void PolicyBase::init(const YAML::Node& config_node, const std::string& policy_dir)
{
    if (initialized_)
    {
        return;
    }
    _params.config_node = config_node;
    _num_of_dofs = _params.Get<int>("num_of_dofs", 12);
    _lin_vel_scale = _params.Get<int>("lin_vel_scale", 1);
    _ang_vel_scale = _params.Get<float>("ang_vel_scale", 1.0f);
    _dof_pos_scale = _params.Get<int>("dof_pos_scale", 1);
    _dof_vel_scale = _params.Get<float>("dof_vel_scale", 1.0f);
    _action_scale = _params.Get<std::vector<float>>("action_scale", std::vector<float>(_num_of_dofs, 1.0));
    _kp = _params.Get<std::vector<float>>("rl_kp", std::vector<float>(_num_of_dofs, 1.0));
    _kd = _params.Get<std::vector<float>>("rl_kd", std::vector<float>(_num_of_dofs, 1.0));
    _num_observations = _params.Get<int>("num_observations", 1);
    _default_dof_pos_ = _params.Get<std::vector<float>>("default_dof_pos");


    OnInit();

    LoadModel();

}


PolicyOutput& PolicyBase::Forward(const PolicyContext& context)
{
    if (!initialized_)
    {
        throw std::runtime_error("Policy is not initialized: " + name_);
    }

    std::vector<float> obs = BuildObservation(context);
    std::vector<float> model_input = ProcessObservation(obs);
    std::vector<float> actions = RunInference(model_input);
    PolicyOutput output = ComputeOutput(actions, context);
    output.raw_actions = actions;
    // last_actions_ = actions;

    return output;

}


void PolicyBase::Reset()
{

}
