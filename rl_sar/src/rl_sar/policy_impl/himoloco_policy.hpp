#ifndef HIMOLOCO_POLICY_HPP
#define HIMOLOCO_POLICY_HPP


#include <yaml-cpp/yaml.h>
#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "policy_base.hpp"
#include "vector_math.hpp"
#include "observation_buffer.hpp"
#include "inference_runtime.hpp"

namespace rl_policy{

class HimolocoPolicy : public PolicyBase
{
private:
    /* data */
public:
    explicit HimolocoPolicy(std::string name):PolicyBase(name){};
    ~HimolocoPolicy() = default;

    HimolocoPolicy(const HimolocoPolicy&) = delete;
    HimolocoPolicy& operator=(const HimolocoPolicy&) = delete;

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

    std::vector<float> _obs;
    std::string _policy_model_name;

    


};



}   //namespace rl_policy


#endif