#ifndef DELAY_MONITOR_MACROS_HPP
#define DELAY_MONITOR_MACROS_HPP
#include "delay_monitor.hpp"

// Debug 模式启用
#ifndef NDEBUG

    #define LAT_STATS_DECLARE(name, print_every) \
        static LatencyStats name(#name, print_every)

    #define LAT_STATS_SCOPE(name) \
        ScopedTimer scoped_timer_##__LINE__(name)

    #define LAT_STATS_ADD(name, value_ms) \
        name.add(value_ms)

    #define LAT_STATS_PRINT(name) \
        name.print()

#else

    // Release 模式全部编译为空
    #define LAT_STATS_DECLARE(name, print_every)
    #define LAT_STATS_SCOPE(name)
    #define LAT_STATS_ADD(name, value_ms) ((void)0)
    #define LAT_STATS_PRINT(name) ((void)0)

#endif


#endif