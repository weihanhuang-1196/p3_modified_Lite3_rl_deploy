#include "register_policies.hpp"

#include "policy_factory.hpp"
#include "himoloco_policy.hpp"
#include "wmp_policy.hpp"

#include <memory>
#include <mutex>

namespace rl_policy {

void RegisterAllPolicies()
{
    static std::once_flag flag;

    std::call_once(flag, []()
    {
        PolicyFactory::Instance().Register(
            "himoloco_walk_ppo",
            []()
            {
                return std::make_unique<HimolocoPolicy>("himoloco_walk_ppo");
            }
        );

        PolicyFactory::Instance().Register(
            "wmp_ppo",
            []()
            {
                return std::make_unique<WMPPolicy>("wmp_ppo");
            }
        );
    });
}

}  // namespace rl_policy