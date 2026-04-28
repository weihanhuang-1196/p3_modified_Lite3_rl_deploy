#include "policy_factory.hpp"

#include <stdexcept>

namespace rl_policy {

PolicyFactory& PolicyFactory::Instance()
{
    static PolicyFactory instance;
    return instance;
}

void PolicyFactory::Register(const std::string& type, Creator creator)
{
    creators_[type] = std::move(creator);
}

std::unique_ptr<PolicyBase> PolicyFactory::Create(const std::string& type) const
{
    auto it = creators_.find(type);

    if (it == creators_.end())
    {
        throw std::runtime_error("Unknown policy type: " + type);
    }

    return it->second();
}

}  // namespace rl_policy