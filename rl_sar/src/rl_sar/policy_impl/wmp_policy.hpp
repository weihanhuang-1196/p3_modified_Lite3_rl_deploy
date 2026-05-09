#ifndef WMP_POLICY_HPP
#define WMP_POLICY_HPP

#include "policy_base.hpp"
#include <yaml-cpp/yaml.h>
#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "vector_math.hpp"
#include "observation_buffer.hpp"
#include "inference_runtime.hpp"



namespace rl_policy{


class WMPPolicy : public PolicyBase
{
private:
    /* data */
public:
    WMPPolicy(std::string name):PolicyBase(name){};
    ~WMPPolicy() = default;

    WMPPolicy(const WMPPolicy&) = delete;
    WMPPolicy& operator=(const WMPPolicy&) = delete;

protected:
    void OnInit() override;
    void OnReset() override;
    void LoadModel(const std::string & policy_dir) override;
    void BuildObservation(const PolicyContext& context) override;
    std::vector<std::vector<float>> ProcessObservation() override;
    std::vector<float> RunInference(std::vector<std::vector<float>>& model_input) override;
    PolicyOutput& ComputeOutput(const std::vector<float>& actions, const PolicyContext& context) override;

private:
    std::vector<float> ComputeObservation(const PolicyContext& context, const std::vector<std::string>& observations);
    std::vector<int> _obs_dims;
    ObservationBuffer _history_obs_buf;
    std::vector<float> _history_obs;
    std::unique_ptr<InferenceRuntime::Model> _model;
    std::unique_ptr<InferenceRuntime::Model> _world_model;

    std::vector<float> _obs;
    std::vector<float> _world_obs;


    std::string _policy_model_name;
    std::string _world_model_name;
    std::vector<std::string> _world_observations;


    std::vector<float> _pre_wm_image;
    std::vector<float> _wm_logit;
    std::vector<float> _wm_stoch;
    std::vector<float> _wm_deter;
    std::vector<float> _wm_feature;
    std::vector<float> _wm_action;
    std::vector<float> _wm_is_first;
    std::vector<float> _wm_prop;
    std::vector<float> _wm_action_history; 
    std::vector<float> _wm_input_image;
    int _image_width;
    int _image_height;

    int _global_counter;
    int _visual_update_interval;

    std::vector<float> _current_command;



};


} //namespace rl_policy



#endif