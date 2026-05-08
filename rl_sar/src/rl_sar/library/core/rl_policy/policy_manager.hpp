#ifndef POLICY_MANAGER_HPP
#define POLICY_MANAGER_HPP

#include "policy_base.hpp"

#include <yaml-cpp/yaml.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace rl_policy {

class PolicyManager
{

    using PolicyPtr = std::unique_ptr<PolicyBase>;
    using PolicyMap = std::unordered_map<std::string,PolicyPtr>;
    using PolicyGroupMap = std::unordered_map<std::string,PolicyMap>;
public:
    void LoadFromYaml(
        const std::string& yaml_path,
        const std::string& policy_dir);

    void SwitchPolicy(const std::string& fsm_name, std::string policy_name = "");
    void SwitchNextPolicy();
    void SwitchPrevPolicy();

    PolicyOutput Forward(PolicyContext& context);

    const std::string& ActivePolicyName() const;
    const YamlParams& getActivePolicyConfig() const;


private:
    PolicyGroupMap policies_;
    YAML::Node root;


    PolicyBase* active_policy_ = nullptr;
    std::string active_policy_name_;
    std::string active_fsm_name_;
    size_t active_policy_index_;

    std::vector<std::string> fsm_order_;
    std::unordered_map<std::string,std::vector<std::string>> policy_order_;

};

}  // namespace rl_policy


#endif