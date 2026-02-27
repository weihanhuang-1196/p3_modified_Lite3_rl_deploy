#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>

#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
#include <vector>

class RosbagRecorder
{
public:
    RosbagRecorder(const std::string& output_dir,
                   const std::string& bag_prefix)
        : output_dir_(output_dir), bag_prefix_(bag_prefix)
    {
        start();
    }

    ~RosbagRecorder()
    {
        stop();
    }

private:
    void start()
    {
        // 时间戳
        auto t = std::time(nullptr);
        std::tm tm;
        localtime_r(&t, &tm);

        std::ostringstream oss;
        oss << bag_prefix_ << "_"
            << std::put_time(&tm, "%Y%m%d_%H%M%S");

        bag_name_ = oss.str();

        std::ostringstream cmd;
        cmd << "ros2 bag record --storage mcap "
            << "--max-bag-size 1024 ";
        for (const auto& topic : topics_)
            cmd << topic << " ";
        cmd << "-o " << output_dir_ << "/" << bag_name_;

        command_ = cmd.str();

        pid_ = fork();
        if (pid_ == 0)
        {
            // 子进程
            execl("/bin/sh", "sh", "-c", command_.c_str(), (char*)nullptr);
            _exit(EXIT_FAILURE);
        }
    }

    void stop()
    {
        if (pid_ > 0)
        {
            // 给 ros2 bag record 发送 SIGINT（等价 Ctrl+C）
            kill(pid_, SIGINT);
        }
    }

private:
    std::string output_dir_;
    std::string bag_prefix_;
    std::string bag_name_;
    std::string command_;
    pid_t pid_{-1};
    std::vector<std::string> topics_ = {
            "--all"
            // "/cmd_vel",
            // "/Devices/joy",
            // "/controller_state",
            // "/rl_sar/Robot_State",
            // "/rl_sar/Robot_Command",
            // "/joint_states",
            // "/tf",
            // "/tf_static",
            // "/robot_description",
            // "/parameter_events"
        };
};
