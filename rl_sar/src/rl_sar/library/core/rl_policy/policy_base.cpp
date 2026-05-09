#include "policy_base.hpp"

#include <algorithm>

namespace rl_policy{

void PolicyBase::Init(const YAML::Node config_node, const std::string& policy_dir)
{
    if (_initialized)
    {
        return;
    }
    if(!config_node[policy_dir])
    {
        throw;
    }
    _params.config_node = config_node[policy_dir];
    _num_of_dofs = _params.Get<int>("num_of_dofs", 12);
    _lin_vel_scale = _params.Get<float>("lin_vel_scale", 1.0f);
    _ang_vel_scale = _params.Get<float>("ang_vel_scale", 1.0f);
    _dof_pos_scale = _params.Get<float>("dof_pos_scale", 1.0f);
    _dof_vel_scale = _params.Get<float>("dof_vel_scale", 1.0f);
    _action_scale = _params.Get<std::vector<float>>("action_scale", std::vector<float>(_num_of_dofs, 1.0));
    _kp = _params.Get<std::vector<float>>("rl_kp", std::vector<float>(_num_of_dofs, 1.0));
    _kd = _params.Get<std::vector<float>>("rl_kd", std::vector<float>(_num_of_dofs, 1.0));
    _num_observations = _params.Get<int>("num_observations", 1);
    _default_dof_pos = _params.Get<std::vector<float>>("default_dof_pos");
    _observations = _params.Get<std::vector<std::string>>("observations");
    _commands_scale = _params.Get<std::vector<float>>("commands_scale");
    _actions = std::vector<float>(_num_of_dofs, 0.0f);
    _torque_limits = _params.Get<std::vector<float>>("torque_limits");
    _clip_actions_upper = _params.Get<std::vector<float>>("clip_actions_upper");
    _clip_actions_lower = _params.Get<std::vector<float>>("clip_actions_lower");
    _ang_vel_axis = _params.Get<std::string>("ang_vel_axis");
    _observations_history = _params.Get<std::vector<int>>("observations_history");
    _observations_history_priority = _params.Get<std::string>("observations_history_priority");

    InitObservations();


    OnInit();
    try
    {
        LoadModel(policy_dir);
    }
    catch(const std::exception& e)
    {
        throw;
    }

    _initialized = true;
    
    

}

void PolicyBase::InitObservations()
{
    this->obs.lin_vel = {0.0f, 0.0f, 0.0f};
    this->obs.ang_vel = {0.0f, 0.0f, 0.0f};
    this->obs.gravity_vec = {0.0f, 0.0f, -1.0f};

    
    auto commands = this->_params.Get<std::vector<float>>("commands_scale").size();
    this->obs.commands = std::vector<float>(commands, 0.0f);

    this->obs.base_quat = {0.0f, 0.0f, 0.0f, 1.0f};
    this->obs.dof_pos = this->_params.Get<std::vector<float>>("default_dof_pos");
    this->obs.dof_vel.clear();
    this->obs.dof_vel.resize(this->_params.Get<int>("num_of_dofs"), 0.0f);
    this->obs.actions.clear();
    this->obs.actions.resize(this->_params.Get<int>("num_of_dofs"), 0.0f);

}


const YamlParams& PolicyBase::getConfig() const
{
    return _params;
}



PolicyOutput& PolicyBase::Forward(PolicyContext& context)
{
    if (!_initialized)
    {
        throw std::runtime_error("Policy is not initialized: " + _name);
    }

    BuildObservation(context);
    auto model_input = ProcessObservation();
    _actions = RunInference(model_input);
    context.last_actions = _actions;
    output = ComputeOutput(_actions, context);
    output.raw_actions = _actions;

    return output;

}


void PolicyBase::Reset()
{
    InitObservations();
    OnReset();
}


} // rl_policy