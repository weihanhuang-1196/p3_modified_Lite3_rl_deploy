#include <thread>
#include <atomic>
#include <chrono>
#include <functional>

class OneShotAsyncTimer
{
public:
    OneShotAsyncTimer(double delay_sec, std::function<void()> cb)
        : delay_(delay_sec), cb_(std::move(cb)) {}

    void start()
    {
        bool expected = false;
        if (!running_.compare_exchange_strong(expected, true))
        {
            // 已经在跑，直接忽略
            return;
        }

        cancelled_.store(false, std::memory_order_release);

        std::thread([this]() {
            std::this_thread::sleep_for(delay_);

            if (!cancelled_.load(std::memory_order_acquire))
                cb_();

            running_.store(false, std::memory_order_release);
        }).detach();
    }

    void cancel()
    {
        cancelled_.store(true, std::memory_order_release);
    }

    bool running() const
    {
        return running_.load(std::memory_order_acquire);
    }

private:
    std::chrono::duration<double> delay_;
    std::function<void()> cb_;

    std::atomic<bool> cancelled_{false};
    std::atomic<bool> running_{false};
};
