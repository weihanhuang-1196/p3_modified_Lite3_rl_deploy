#ifndef ONE_SHOT_ASYNC_TIMER_HPP
#define ONE_SHOT_ASYNC_TIMER_HPP

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <chrono>
#include <atomic>
#include <unordered_map>

class TimerQueue
{
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    using TaskId = uint64_t;

    TimerQueue()
    {
        worker_ = std::thread([this]() { this->run(); });
    }

    ~TimerQueue()
    {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_ = true;
            cv_.notify_all();
        }
        if (worker_.joinable())
            worker_.join();
    }

    // 添加 one-shot 定时任务
    TaskId schedule_after(std::chrono::milliseconds delay, std::function<void()> cb)
    {
        auto id = ++id_gen_;
        TimePoint tp = Clock::now() + delay;

        {
            std::lock_guard<std::mutex> lk(mtx_);
            tasks_.push(Task{tp, id, std::move(cb)});
            cancelled_[id] = false;
        }
        cv_.notify_one();
        return id;
    }

    // 取消任务
    void cancel(TaskId id)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = cancelled_.find(id);
        if (it != cancelled_.end())
            it->second = true;
        cv_.notify_one();
    }

private:
    struct Task
    {
        TimePoint tp;
        TaskId id;
        std::function<void()> cb;
    };

    struct Cmp
    {
        bool operator()(const Task& a, const Task& b) const
        {
            return a.tp > b.tp; // 最早到期的在 top
        }
    };

    void run()
    {
        std::unique_lock<std::mutex> lk(mtx_);
        while (!stop_)
        {
            if (tasks_.empty())
            {
                cv_.wait(lk);
                continue;
            }

            auto now = Clock::now();
            auto next = tasks_.top();

            if (now < next.tp)
            {
                cv_.wait_until(lk, next.tp);
                continue; // 可能被新任务 / cancel 唤醒
            }

            // 到期了
            tasks_.pop();

            // 检查是否被取消
            if (cancelled_[next.id])
            {
                cancelled_.erase(next.id);
                continue;
            }

            // 执行回调（注意：先解锁，避免回调里再 schedule 死锁）
            auto cb = next.cb;
            cancelled_.erase(next.id);

            lk.unlock();
            cb();
            lk.lock();
        }
    }

private:
    std::thread worker_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool stop_{false};

    std::priority_queue<Task, std::vector<Task>, Cmp> tasks_;
    std::unordered_map<TaskId, bool> cancelled_;
    std::atomic<TaskId> id_gen_{0};
};


#endif // ONE_SHOT_ASYNC_TIMER_HPP


//****
/*
/*
/*int main()
/*{
/*    TimerQueue tq;
/*
/*    auto id1 = tq.schedule_after(std::chrono::seconds(2), [] {
/*        printf("2 seconds!\n");
/*    });
/*
/*    auto id2 = tq.schedule_after(std::chrono::seconds(5), [] {
/*        printf("5 seconds!\n");
/*    });
/*
/*    std::this_thread::sleep_for(std::chrono::seconds(1));
/*    tq.cancel(id2); // 取消 5 秒那个
/*
/*    auto foo = std::make_shared<Foo>();
/*    std::weak_ptr<Foo> wfoo = foo;
/*    
/*    tq.schedule_after(std::chrono::seconds(1),
/*                      [wfoo]() {
/*                          if (auto f = wfoo.lock()) {
/*                              f->onTimeout();
/*                          }
/*                          // else: 对象已销毁，安全忽略
/*                      });
/*      
/*
/*
/*
/*    std::this_thread::sleep_for(std::chrono::seconds(5));
/*}
/*
/*
/*
*/
