#include "wmp_policy.hpp"


namespace rl_policy{

void WMPPolicy::OnInit()
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

void WMPPolicy::OnReset()
{

}

void WMPPolicy::LoadModel(const std::string & policy_dir)
{
    std::string model_path = std::string(POLICY_DIR) + "/" + policy_dir + "/" + this->_params.Get<std::string>("model_name");
    this->model = InferenceRuntime::ModelFactory::load_model(model_path);
    if (!this->model)
    {
        throw std::runtime_error("Failed to load model from: " + model_path);
    }
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