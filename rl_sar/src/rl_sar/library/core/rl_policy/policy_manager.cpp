#include "policy_manager.hpp"
#include "policy_factory.hpp"

#include <stdexcept>

namespace rl_policy {

void PolicyManager::LoadFromYaml(
    const std::string& file_path,
    const std::string& file_name)
{


    std::string config_path = std::string(POLICY_DIR) + "/" + file_path + "/" + file_name;
    root = YAML::LoadFile(config_path);

    if (!root["policies"])
    {
        throw std::runtime_error("Missing policies node in yaml: " + config_path);
    }

    YAML::Node policies_node = root["policies"];

    for (auto it = policies_node.begin(); it != policies_node.end(); ++it)
    {
        std::string fsm_name = it->first.as<std::string>();
        YAML::Node fsm_config = it->second;
        fsm_order_.push_back(fsm_name);

        std::string type = fsm_config["type"].as<std::string>();

        YAML::Node policy_list_node = fsm_config["policy_list"];
        for (auto i = policy_list_node.begin(); i != policy_list_node.end(); ++i)
        {
            std::string policy_name = i->first.as<std::string>();
            YAML::Node policy_config = i->second;
            std::string policy_config_path = std::string(POLICY_DIR) + "/" + file_path + "/" + policy_config["policy_config_path"].as<std::string>();
            std::string policy_dir =  file_path + "/" + type;
            YAML::Node policy_yaml = YAML::LoadFile(policy_config_path);
            auto policy = PolicyFactory::Instance().Create(type);
            try
            {
                policy->Init(policy_yaml, policy_dir);
            }
            catch(const std::exception& e)
            {
                std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
                throw;
            }
            
            policies_[fsm_name][policy_name] = std::move(policy);
            policy_order_[fsm_name].push_back(policy_name);

        }

    }

    
}

void PolicyManager::SwitchPolicy(const std::string& fsm_name, std::string policy_name)
{

    if(active_fsm_name_.compare(fsm_name) != 0)
    {
        active_policy_index_ = 0;
    }

    auto policy_map = policies_.find(fsm_name);

    if (policy_map == policies_.end())
    {
        throw std::runtime_error("fsm not loaded: " + fsm_name);
    }

    if (policy_name.empty())
    {
        policy_name = root["policies"][fsm_name]["selected"].as<std::string>();
    }
    
    auto it = policy_map->second.find(policy_name);
    if(it == policy_map->second.end())
    {
        throw std::runtime_error("policies not loaded: " + policy_name);
    }

    active_policy_ = it->second.get();
    active_policy_name_ = policy_name;
    active_fsm_name_ = fsm_name;

    active_policy_->Reset();
    std::cout << LOGGER::INFO << "switch policy:  " << policy_name  << std::endl;
}

void PolicyManager::SwitchNextPolicy()
{
    auto& order = policy_order_.at(active_fsm_name_);

    if (order.empty())
    {
        throw std::runtime_error("empty policy order for fsm: " + active_fsm_name_);
    }

    active_policy_index_ =
        (active_policy_index_ + 1) % order.size();

    SwitchPolicy(
        active_fsm_name_,
        order[active_policy_index_]
    );
}

void PolicyManager::SwitchPrevPolicy()
{
    auto& order = policy_order_.at(active_fsm_name_);

    if (order.empty())
    {
        throw std::runtime_error("empty policy order for fsm: " + active_fsm_name_);
    }

    active_policy_index_ =
        (active_policy_index_ + order.size() - 1) % order.size();

    SwitchPolicy(
        active_fsm_name_,
        order[active_policy_index_]
    );
}


PolicyOutput PolicyManager::Forward(PolicyContext& context)
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


const YamlParams& PolicyManager::getActivePolicyConfig() const
{
    if (!active_policy_)
    {
        throw std::runtime_error("No active policy selected");
    }
    return active_policy_->getConfig();
}

}  // namespace rl_policy