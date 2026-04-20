#ifndef DELAY_MONITOR_HPP
#define DELAY_MONITOR_HPP

#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>

class LatencyStats
{
public:
    explicit LatencyStats(std::string name,
                          size_t print_every = 1000,
                          bool auto_reset = true)
        : name_(std::move(name)),
          print_every_(print_every),
          auto_reset_(auto_reset)
    {
        data_.reserve(print_every_);
    }

    // 添加一次测量（单位：ms）
    inline void add(double ms)
    {
        data_.push_back(ms);

        if (data_.size() >= print_every_)
        {
            print();
            if (auto_reset_) reset();
        }
    }

    void reset()
    {
        data_.clear();
    }

    void print()
    {
        if (data_.empty()) return;

        std::vector<double> v = data_;
        std::sort(v.begin(), v.end());

        auto percentile = [&](double p)
        {
            double idx = p * (v.size() - 1);
            size_t lo = static_cast<size_t>(idx);
            size_t hi = std::min(lo + 1, v.size() - 1);
            double frac = idx - lo;
            return v[lo] * (1.0 - frac) + v[hi] * frac;
        };

        double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double min  = v.front();
        double max  = v.back();

        double p50 = percentile(0.50);
        double p90 = percentile(0.90);
        double p99 = percentile(0.99);

        std::cout << std::fixed << std::setprecision(3)
                  << "[LatencyStats][" << name_ << "] "
                  << "N=" << v.size()
                  << " | mean=" << mean << " ms"
                  << ", min=" << min
                  << ", p50=" << p50
                  << ", p90=" << p90
                  << ", p99=" << p99
                  << ", max=" << max
                  << std::endl;
    }

private:
    std::string name_;
    size_t print_every_;
    bool auto_reset_;
    std::vector<double> data_;
};



/**
 * RAII计时
 * 
 */
class ScopedTimer
{
public:
    explicit ScopedTimer(LatencyStats& stats)
        : stats_(stats),
          start_(Clock::now())
    {}

    ~ScopedTimer()
    {
        auto end = Clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count() / 1000.0;
        stats_.add(ms);
    }

private:
    using Clock = std::chrono::steady_clock;

    LatencyStats& stats_;
    std::chrono::time_point<Clock> start_;
};


/**
 * 手动计时
 * 
 */
class ManualTimer
{
public:
    void tic()
    {
        start_ = Clock::now();
    }

    double toc_ms()
    {
        auto end = Clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count() / 1000.0;
    }

private:
    using Clock = std::chrono::steady_clock;
    std::chrono::time_point<Clock> start_;
};




#endif DELAY_MONITOR_HPP