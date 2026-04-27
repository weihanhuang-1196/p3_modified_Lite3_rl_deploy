#ifndef POLICY_HPP
#define POLICY_HPP

#include "observation_buffer.hpp"
#include "inference_runtime.hpp"
#include <yaml-cpp/yaml.h>
#include <string>

namespace rl_policy{


template <typename T>
struct Observations
{
    std::vector<T> lin_vel;
    std::vector<T> ang_vel;
    std::vector<T> gravity_vec;
    std::vector<T> commands;
    std::vector<T> base_quat;
    std::vector<T> dof_pos;
    std::vector<T> dof_vel;
    std::vector<T> actions;
};


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



class policBase
{

public:
    Observations<float> obs;


public:
    policBase(std::string name):_name(name) = default;
    virtual ~policBase() = default;

    policBase(const policBase&) = delete;
    policBase& operator=(const policBase&) = delete;


    void init(const std::string& file_path, const std::string& file_name)
    {
        if (_initialized) return;

        ReadYaml(file_path, file_name);
        InitObservations();
        initPolicy(file_path);

        _initialized = true;
    }

    virtual void Forward(std::vector<float> &output_dof_pos, std::vector<float> &output_dof_vel, std::vector<float> &output_dof_tau) = 0;
    virtual void InitObservations() = 0;

protected:
    virtual void ComputeOutput(const std::vector<float>& actions, std::vector<float> &output_dof_pos, std::vector<float> &output_dof_vel, std::vector<float> &output_dof_tau) = 0;
    


private:
    void initPolicy(const std::string& file_path){
        // init obs history
        const auto& observations_history = this->_params.Get<std::vector<int>>("observations_history");  // avoid dangling reference
        if (!observations_history.empty())
        {
            int history_length = *std::max_element(observations_history.begin(), observations_history.end()) + 1;
            this->_history_obs_buf = std::make_unique<ObservationBuffer>(1, this->_obs_dims, history_length, this->_params.Get<std::string>("observations_history_priority"));
        }
        // init model
        std::string model_path = std::string(POLICY_DIR) + "/" + file_path + "/" + this->_params.Get<std::string>("model_name");
        this->_model = InferenceRuntime::ModelFactory::load_model(model_path);
        if (!this->_model)
        {
            throw std::runtime_error("Failed to load model from: " + model_path);
        }

    }
    void ReadYaml(const std::string& file_path, const std::string& file_name)
    {
        std::string config_path = std::string(POLICY_DIR) + "/" + file_path + "/" + file_name;
        YAML::Node config;
        try
        {
            config = YAML::LoadFile(config_path)[file_path];
        }
        catch (YAML::BadFile &e)
        {
            std::cout << LOGGER::ERROR << "The file '" << config_path << "' does not exist" << std::endl;
            throw std::runtime_error("YAML load failed: " + config_path);
        }
    
        for (auto it = config.begin(); it != config.end(); ++it)
        {
            std::string key = it->first.as<std::string>();
            this->_params.config_node[key] = it->second;
        }
    }

private:
    YamlParams _params;
    std::string _name;
    // history buffer
    std::unique_ptr<ObservationBuffer> _history_obs_buf;
    std::vector<float> _history_obs;
    std::vector<int> _obs_dims;

    bool _initialized = false;

    // rl model
    std::unique_ptr<InferenceRuntime::Model> _model;

};



    
}






#endif //POLICY_HPP