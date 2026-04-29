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
    std::vector<float>& ProcessObservation() override;
    std::vector<float> RunInference(const std::vector<float>& model_input) override;
    PolicyOutput& ComputeOutput(const std::vector<float>& actions, const PolicyContext& context) override;

private:
    std::vector<float> ComputeObservation(const PolicyContext& context);
    std::vector<int> obs_dims;
    ObservationBuffer history_obs_buf;
    std::vector<float> history_obs;
    std::unique_ptr<InferenceRuntime::Model> model;

    std::vector<float> obs;

    


};



}   //namespace rl_policy


#endif