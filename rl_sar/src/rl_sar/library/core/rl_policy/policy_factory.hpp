#ifndef POLICY_FACTORY_HPP
#define POLICY_FACTORY_HPP

#include "policy_base.hpp"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace rl_policy {

class PolicyFactory
{
public:
    using Creator = std::function<std::unique_ptr<PolicyBase>()>;

    static PolicyFactory& Instance();

    void Register(const std::string& type, Creator creator);

    std::unique_ptr<PolicyBase> Create(const std::string& type) const;

private:
    std::unordered_map<std::string, Creator> creators_;
};

}  // namespace rl_policy


#endif