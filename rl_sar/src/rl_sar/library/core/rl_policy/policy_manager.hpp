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
public:
    void LoadFromYaml(
        const std::string& yaml_path,
        const std::string& policy_dir);

    void SwitchPolicy(const std::string& policy_name);

    PolicyOutput Forward(PolicyContext& context);

    const std::string& ActivePolicyName() const;


private:
    std::unordered_map<std::string, std::unique_ptr<PolicyBase>> policies_;

    PolicyBase* active_policy_ = nullptr;
    std::string active_policy_name_;
};

}  // namespace rl_policy


#endif