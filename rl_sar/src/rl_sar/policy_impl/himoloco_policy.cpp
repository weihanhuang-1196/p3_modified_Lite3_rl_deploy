#include "himoloco_policy.hpp"
#include "policy_context_builder.hpp"

namespace rl_policy{




void HimolocoPolicy::OnInit()
{

    PolicyContextBuilder context_builder;
    std::vector<float> lin_vel = {0.0f, 0.0f, 0.0f};
    std::vector<float> ang_vel = {0.0f, 0.0f, 0.0f};
    std::vector<float> gravity_vec = {0.0f, 0.0f, -1.0f};
    std::vector<float> base_quat = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> dof_pos = this->_params.Get<std::vector<float>>("default_dof_pos");
    std::vector<float> dof_vel(this->_params.Get<int>("num_of_dofs"), 0.0f);
    std::vector<float> command(this->_params.Get<int>("num_of_command"), 0.0f);
    context_builder.SetRobotState(
        lin_vel,
        ang_vel,
        gravity_vec,
        base_quat
    );
    context_builder.SetJointState(
        dof_pos,
        dof_vel
    );
    context_builder.SetCommand(command);
    std::vector<float> last_actions(12, 0.0f);
    context_builder.SetLastActions(last_actions);
    PolicyContext ctx = context_builder.Build();


    this->obs = ComputeObservation(ctx);
    const auto& observations_history = this->_params.Get<std::vector<int>>("observations_history");  // avoid dangling reference
    if (!observations_history.empty())
    {
        int history_length = *std::max_element(observations_history.begin(), observations_history.end()) + 1;
        this->history_obs_buf = ObservationBuffer(1, this->obs_dims, history_length, this->_params.Get<std::string>("observations_history_priority"));
    }

}


void HimolocoPolicy::LoadModel(const std::string & policy_dir)
{
    std::string model_path = std::string(POLICY_DIR) + "/" + policy_dir + "/" + this->_params.Get<std::string>("model_name");
    this->model = InferenceRuntime::ModelFactory::load_model(model_path);
    if (!this->model)
    {
        throw std::runtime_error("Failed to load model from: " + model_path);
    }
}


void HimolocoPolicy::BuildObservation(const PolicyContext& context)
{
    this->obs = ComputeObservation(context);
    
}


std::vector<float>& HimolocoPolicy::ProcessObservation()
{
    this->history_obs_buf.insert(this->obs);
    this->history_obs = this->history_obs_buf.get_obs_vec(this->_params.Get<std::vector<int>>("observations_history"));
    return this->history_obs;
}


std::vector<float> HimolocoPolicy::RunInference(const std::vector<float>& model_input)
{
    std::vector<float> actions = this->model->forward({model_input});
    if (!_clip_actions_upper.empty() && !_clip_actions_lower.empty())
        return clamp(actions, _clip_actions_lower, _clip_actions_upper);
    else
        return actions;
}

PolicyOutput& HimolocoPolicy::ComputeOutput(const std::vector<float>& actions, const PolicyContext& context)
{
    std::vector<float> actions_scaled = actions * _action_scale;
    std::vector<float> pos_actions_scaled = actions_scaled;
    std::vector<float> vel_actions_scaled(actions.size(), 0.0f);
    for (int i : this->_params.Get<std::vector<int>>("wheel_indices"))
    {
        pos_actions_scaled[i] = 0.0f;
        vel_actions_scaled[i] = actions_scaled[i];
    }
    std::vector<float> all_actions_scaled = pos_actions_scaled + vel_actions_scaled;

    output.target_dof_pos = pos_actions_scaled + _default_dof_pos;
    output.target_dof_vel = vel_actions_scaled;
    output.target_dof_tau = _kp * (all_actions_scaled + _default_dof_pos - context.joints.dof_pos) - _kd * context.joints.dof_vel;
    output.target_dof_tau = clamp( output.target_dof_tau, -_torque_limits, _torque_limits);
    return output;

}


std::vector<float> HimolocoPolicy::ComputeObservation(const PolicyContext& context)
{
    std::vector<std::vector<float>> obs_list;

    for (const std::string &observation : _observations)
    {
        // ============= Base Observations =============
        if (observation == "lin_vel")
        {
            obs_list.push_back(context.robot.lin_vel * _lin_vel_scale);
        }
        else if (observation == "ang_vel")
        {
            // In ROS1 Gazebo, the coordinate system for angular velocity is in the world coordinate system.
            // In ROS2 Gazebo, mujoco and real robot, the coordinate system for angular velocity is in the body coordinate system.
            if (this->_ang_vel_axis == "body")
            {
                obs_list.push_back(context.robot.ang_vel * _ang_vel_scale);
            }
            else if (this->_ang_vel_axis == "world")
            {
                obs_list.push_back(QuatRotateInverse(context.robot.base_quat, context.robot.ang_vel) * _ang_vel_scale);
            }
        }
        else if (observation == "gravity_vec")
        {
            obs_list.push_back(QuatRotateInverse(context.robot.base_quat, context.robot.gravity_vec));
        }
        else if (observation == "commands")
        {
            obs_list.push_back(context.command.velocity * _commands_scale);
        }
        else if (observation == "dof_pos")
        {
            std::vector<float> dof_pos_rel = context.joints.dof_pos - _default_dof_pos;
            for (int i : this->_params.Get<std::vector<int>>("wheel_indices"))
            {
                dof_pos_rel[i] = 0.0f;
            }
            obs_list.push_back(dof_pos_rel * _dof_pos_scale);
        }
        else if (observation == "dof_vel")
        {
            obs_list.push_back(context.joints.dof_vel * _dof_vel_scale);
        }
        else if (observation == "actions")
        {
            obs_list.push_back(context.last_actions);
        }
    }

    this->obs_dims.clear();
    for (const auto& obs : obs_list)
    {
       this->obs_dims.push_back(obs.size());
    }

    std::vector<float> obs;
    for (const auto& obs_vec : obs_list)
    {
        obs.insert(obs.end(), obs_vec.begin(), obs_vec.end());
    }
    std::vector<float> clamped_obs = clamp(obs, -this->_params.Get<float>("clip_obs"), this->_params.Get<float>("clip_obs"));
    return clamped_obs;
}


void HimolocoPolicy::OnReset()
{
    const auto& observations_history = this->_params.Get<std::vector<int>>("observations_history");  // avoid dangling reference
    if (!observations_history.empty())
    {
        int history_length = *std::max_element(observations_history.begin(), observations_history.end()) + 1;
        this->history_obs_buf = ObservationBuffer(1, this->obs_dims, history_length, this->_params.Get<std::string>("observations_history_priority"));
    }
}


} // namespace rl_policy