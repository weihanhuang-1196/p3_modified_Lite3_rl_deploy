
#ifndef REGISTER_POLICIES_HPP
#define REGISTER_POLICIES_HPP
#include "rl_policy/policy_factory.hpp"
#include "himoloco_policy.hpp"
#include "wmp_policy.hpp"

namespace rl_policy {

// 例子
PolicyFactory::Instance().Register(
    "himoloco_walk_ppo",
    []()
    {
        return std::make_unique<HimolocoPolicy>();
    }
);

PolicyFactory::Instance().Register(
    "wmp_ppo",
    []()
    {
        return std::make_unique<WMPPolicy>();
    }
);




void RegisterAllPolicies()
{
    static bool registered = false;

    if (registered)
    {
        return;
    }

    registered = true;
}

}  // namespace rl_policy


#endif