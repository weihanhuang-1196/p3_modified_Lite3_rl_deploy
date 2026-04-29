#include "policy_manager.hpp"
#include "policy_factory.hpp"

#include <stdexcept>

namespace rl_policy {

void PolicyManager::LoadFromYaml(
    const std::string& yaml_path,
    const std::string& policy_dir)
{
    YAML::Node root = YAML::LoadFile(yaml_path);

    if (!root["policies"])
    {
        throw std::runtime_error("Missing policies node in yaml: " + yaml_path);
    }

    YAML::Node policies_node = root["policies"];

    for (auto it = policies_node.begin(); it != policies_node.end(); ++it)
    {
        std::string policy_name = it->first.as<std::string>();
        YAML::Node policy_config = it->second;

        std::string type = policy_config["type"].as<std::string>();
        std::string policy_config_path = std::string(POLICY_DIR) + "/" + policy_config["policy_config_path"].as<std::string>();
        YAML::Node policy_yaml = YAML::LoadFile(policy_config_path);
        if(!policy_yaml["policy_name"])
        {
            throw std::runtime_error("Missing policy configuration in yaml: " + policy_config_path);
        }

        auto policy = PolicyFactory::Instance().Create(type);

        policy->Init(policy_yaml, policy_dir);

        policies_[policy_name] = std::move(policy);
    }

    if (root["active_policy"])
    {
        SwitchPolicy(root["active_policy"].as<std::string>());
    }
}

void PolicyManager::SwitchPolicy(const std::string& policy_name)
{
    auto it = policies_.find(policy_name);

    if (it == policies_.end())
    {
        throw std::runtime_error("Policy not loaded: " + policy_name);
    }

    active_policy_ = it->second.get();
    active_policy_name_ = policy_name;

    active_policy_->Reset();
}

PolicyOutput PolicyManager::Forward(const PolicyContext& context)
{
    if (!active_policy_)
    {
        throw std::runtime_error("No active policy selected");
    }

    return active_policy_->Forward(context);
}

const std::string& PolicyManager::ActivePolicyName() const
{
    return active_policy_name_;
}

}  // namespace rl_policy