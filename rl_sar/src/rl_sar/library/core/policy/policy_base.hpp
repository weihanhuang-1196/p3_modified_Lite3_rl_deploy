#ifndef POLICY_HPP
#define POLICY_HPP

#include "observation_buffer.hpp"
#include "inference_runtime.hpp"
#include "policy_output.hpp"
#include "policy_context.hpp"
#include <yaml-cpp/yaml.h>
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace rl_policy{



struct YamlParams
{
    YAML::Node config_node;

    // Get config value by key
    // WARNING: For vectors/containers, store result in a variable before using iterators/references:
    //   ✓ auto vec = params.Get<std::vector<int>>("key"); vec.begin()
    //   ✗ params.Get<std::vector<int>>("key").begin()  // dangling reference!
    template<typename T>
    T Get(const std::string& key, const T& default_value = T()) const
    {
        if (config_node[key])
        {
            return config_node[key].as<T>();
        }
        return default_value;
    }

    bool Has(const std::string& key) const
    {
        return config_node[key].IsDefined();
    }
};



class PolicyBase
{

public:
    explicit PolicyBase(std::string name):_name(std::move(name)){};
    virtual ~PolicyBase() = default;

    PolicyBase(const PolicyBase&) = delete;
    PolicyBase& operator=(const PolicyBase&) = delete;

    PolicyOutput& Forward(const PolicyContext& context);

    void init(const YAML::Node& config_node, const std::string& policy_dir);

    virtual void InitObservations() = 0;
    

protected:
    virtual void ComputeOutput(const std::vector<float>& actions, std::vector<float> &output_dof_pos, std::vector<float> &output_dof_vel, std::vector<float> &output_dof_tau) = 0;
    virtual void onInit() = 0;
    virtual void LoadModel() = 0;
    virtual std::vector<float> BuildObservation(const PolicyContext& context) = 0;
    virtual std::vector<float> ProcessObservation(const std::vector<float>& obs) = 0;
    virtual std::vector<float> RunInference(const std::vector<float>& model_input) = 0;
    virtual PolicyOutput ComputeOutput(const std::vector<float>& actions, const PolicyContext& context) = 0;

protected:
    YamlParams _params;
    std::string _name;

    bool _initialized = false;

    // rl model
    std::unique_ptr<InferenceRuntime::Model> _model;

    int _num_of_dofs;
    std::vector<float> _action_scale;
    int _lin_vel_scale;
    int _ang_vel_scale;
    int _dof_pos_scale;
    int _dof_vel_scale;
    std::vector<float> _default_dof_pos;
    float _clip_obs;
    std::vector<float> _clip_actions;
    std::vector<float> _kp;
    std::vector<float> _kd;
    int _num_observations;



};



    
}






#endif //POLICY_HPP