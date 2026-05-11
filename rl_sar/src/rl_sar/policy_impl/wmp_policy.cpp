#include "wmp_policy.hpp"
#include "policy_context.hpp"
#include "policy_context_builder.hpp"

namespace rl_policy{

void WMPPolicy::OnInit()
{
    PolicyContextBuilder context_builder;
    std::vector<float> lin_vel = {0.0f, 0.0f, 0.0f};
    std::vector<float> ang_vel = {0.0f, 0.0f, 0.0f};
    std::vector<float> gravity_vec = {0.0f, 0.0f, -1.0f};
    std::vector<float> base_quat = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> dof_pos = this->_params.Get<std::vector<float>>("default_dof_pos");
    std::vector<float> dof_vel(_num_of_dofs, 0.0f);
    std::vector<float> command(_num_of_command, 0.0f);
    _policy_model_name = this->_params.Get<std::string>("model_name");
    _world_model_name = this->_params.Get<std::string>("world_name");
    _world_observations = this->_params.Get<std::vector<std::string>>("world_observations");
    _global_counter = 0;
    _visual_update_interval = 5;
    _image_width = this->_params.Get<int>("image_width");
    _image_height = this->_params.Get<int>("image_height");


    _pre_wm_image.reserve(_image_width*_image_height);
    _wm_logit = std::vector<float>(1*32*32, 0.0f);
    _wm_stoch = std::vector<float>(1*32*32, 0.0f);
    _wm_deter = std::vector<float>(1*512, 0.0f);
    _wm_feature = std::vector<float>(1*512, 0.0f);
    _wm_action = std::vector<float>(1*5*12, 0.0f);
    _wm_is_first = std::vector<float>(1, 1.0f);
    _wm_prop = std::vector<float>(33, 0.0f);
    _wm_action_history = std::vector<float>(1*5*12, 0.0f);
    _wm_input_image.reserve(_image_width*_image_height);

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


    this->_obs = clamp(ComputeObservation(ctx, _observations), -_clip_obs, _clip_obs);
    if (!_observations_history.empty())
    {
        int history_length = *std::max_element(_observations_history.begin(), _observations_history.end()) + 1;
        this->_history_obs_buf = ObservationBuffer(1, this->_obs_dims, history_length, _observations_history_priority);
    }


}

void WMPPolicy::OnReset()
{
    _global_counter = 0;
    if (!_observations_history.empty())
    {
        int history_length = *std::max_element(_observations_history.begin(), _observations_history.end()) + 1;
        this->_history_obs_buf = ObservationBuffer(1, this->_obs_dims, history_length, _observations_history_priority);
    }
    std::fill(_wm_logit.begin(), _wm_logit.end(), 0.0f);
    std::fill(_wm_stoch.begin(), _wm_stoch.end(), 0.0f);
    std::fill(_wm_deter.begin(), _wm_deter.end(), 0.0f);
    std::fill(_wm_feature.begin(), _wm_feature.end(), 0.0f);
    std::fill(_wm_is_first.begin(), _wm_is_first.end(), 0.0f);
    std::fill(_wm_action.begin(), _wm_action.end(), 0.0f);
    std::fill(_wm_prop.begin(), _wm_prop.end(), 0.0f);
    std::fill(_wm_action_history.begin(), _wm_action_history.end(), 0.0f);
    std::fill(_pre_wm_image.begin(), _pre_wm_image.end(), 0.0f);
    std::fill(_wm_input_image.begin(), _wm_input_image.end(), 0.0f); 

}

void WMPPolicy::LoadModel(const std::string & policy_dir)
{
    std::string model_path = std::string(POLICY_DIR) + "/" + policy_dir + "/" + _policy_model_name;
    _model = InferenceRuntime::ModelFactory::load_model(model_path);
    if (!_model)
    {
        throw std::runtime_error("Failed to load model from: " + model_path);
    }

    std::string world_model_path = std::string(POLICY_DIR) + "/" + policy_dir + "/" + _world_model_name;
    _world_model = InferenceRuntime::ModelFactory::load_model(world_model_path);
    if (!_world_model)
    {
        throw std::runtime_error("Failed to load world model from: " + world_model_path);
    }
}


void WMPPolicy::BuildObservation(const PolicyContext& context)
{
    _world_obs = ComputeObservation(context, _world_observations);
    _obs = clamp(ComputeObservation(context, _observations), -_clip_obs, _clip_obs);
    


    _current_command =  context.command.velocity;
    if(context.HasTensor("depth_image"))
    {
        rl_policy::Tensor depth_image = context.GetTensor("depth_image");
        if(!depth_image.data.empty())
        {
            
            _wm_input_image = std::move(depth_image.data);
        }
        else
            std::fill(_wm_input_image.begin(), _wm_input_image.end(), 0.0f);
    }
    else
    {
        std::fill(_wm_input_image.begin(), _wm_input_image.end(), 0.0f);
    }
}

void WMPPolicy::show_depth_image(const std::vector<float>& depth_vec, int width, int height)
{
#ifndef NDEBUG
    cv::Mat depth_mat(height, width, CV_32FC1, const_cast<float*>(depth_vec.data()));
    cv::Mat depth_display;
    depth_mat.convertTo(depth_display, CV_8UC1, 255.0, 127); // x*255 + 127

    //         // 3. 放大（插值）
    // cv::Mat depth_up;
    // cv::resize(depth_display, depth_up, cv::Size(width*6, height*6), 0, 0, cv::INTER_LINEAR);

    cv::namedWindow("Depth Image", cv::WINDOW_NORMAL);
    cv::imshow("Depth Image", depth_display);
    cv::waitKey(1);
#endif
}


std::vector<std::vector<float>> WMPPolicy::ProcessObservation()
{
    _wm_action_history.erase(_wm_action_history.begin(), _wm_action_history.begin() + _wm_action.size());
    _wm_action_history.insert(_wm_action_history.end(), _wm_action.begin(), _wm_action.end());
    _wm_action = _wm_action_history;

    _history_obs_buf.insert(_obs);
    _history_obs = _history_obs_buf.get_obs_vec(_observations_history);
    return {_current_command, _history_obs};

}


std::vector<float> WMPPolicy::RunInference(std::vector<std::vector<float>>& model_input)
{
    if(_global_counter % _visual_update_interval == 0)
    {
        std::vector<float> input_image(_image_width * _image_height, 0.0f);
        if(_pre_wm_image.empty())
            input_image = _wm_input_image;
        else
            input_image = _pre_wm_image;

        // show_depth_image(input_image,64,64);
        auto world_model_output = _world_model->forward_world({_world_obs, input_image, _wm_logit, _wm_stoch, _wm_deter, _wm_action, _wm_is_first});
        _wm_logit = std::move(world_model_output[0]);
        _wm_stoch = std::move(world_model_output[1]);
        _wm_deter = std::move(world_model_output[2]);
        _wm_feature = std::move(world_model_output[3]);

        _pre_wm_image = std::move(_wm_input_image);
    }
    _global_counter += 1;
    _wm_is_first[0] = 0;
    model_input.push_back(_wm_feature);
    std::vector<float> actions = _model->forward(model_input);
    _wm_action = actions;
    if (!_clip_actions_upper.empty() && !_clip_actions_lower.empty())
        return clamp(actions, _clip_actions_lower, _clip_actions_upper);
    else
        return actions;

    

}

PolicyOutput& WMPPolicy::ComputeOutput(const std::vector<float>& actions, const PolicyContext& context)
{
    std::vector<float> actions_scaled = actions * _action_scale;
    std::vector<float> pos_actions_scaled = actions_scaled;
    std::vector<float> vel_actions_scaled(actions.size(), 0.0f);
    std::vector<float> all_actions_scaled = pos_actions_scaled + vel_actions_scaled;
    output.target_dof_pos = pos_actions_scaled + _default_dof_pos;
    output.target_dof_vel = vel_actions_scaled;
    output.target_dof_tau = _kp * (all_actions_scaled + _default_dof_pos - context.joints.dof_pos) - _kd * context.joints.dof_vel;
    output.target_dof_tau = clamp( output.target_dof_tau, -_torque_limits, _torque_limits);
    return output;
}


std::vector<float> WMPPolicy::ComputeObservation(const PolicyContext& context, const std::vector<std::string>& observations)
{
    std::vector<std::vector<float>> obs_list;

    for (const std::string &observation : observations)
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

    this->_obs_dims.clear();
    for (const auto& obs : obs_list)
    {
       this->_obs_dims.push_back(obs.size());
    }

    std::vector<float> obs;
    for (const auto& obs_vec : obs_list)
    {
        obs.insert(obs.end(), obs_vec.begin(), obs_vec.end());
    }
    return obs;
}



} //namespace rl_policy