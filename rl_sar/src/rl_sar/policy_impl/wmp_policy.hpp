#ifndef WMP_POLICY_HPP
#define WMP_POLICY_HPP

#include "policy_base.hpp"


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


} //namespace rl_policy



#endif