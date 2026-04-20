#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import json
import math
import os



class MotionRecorderROS2(Node):

    def __init__(self):
        super().__init__('motion_recorder')

        self.frames = []
        self.latest_odom = None
        self.latest_joint = None
        self.prev_position = None

        # ================= QoS 配置（关键） =================
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 订阅
        self.create_subscription(
            Odometry,
            '/base_odom',
            self.odom_callback,
            qos
        )

        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            qos
        )

        # 采样周期 0.021s (~50Hz)
        self.timer = self.create_timer(0.021, self.record_frame)

        # 位置变化阈值
        self.position_thresh = 0.005

        # 文件路径
        self.filename = os.path.join(os.getcwd(), 'motion.json')

    # ================= 回调 =================

    def odom_callback(self, msg):
        self.latest_odom = msg

    def joint_callback(self, msg):
        self.latest_joint = msg

    # ================= 核心记录逻辑 =================

    def record_frame(self):
        if self.latest_odom is None or self.latest_joint is None:
            return

        odom = self.latest_odom
        joint = self.latest_joint

        # base position
        pos = odom.pose.pose.position
        curr_position = (pos.x, pos.y, pos.z)

        # 判断是否静止
        if self.prev_position is not None:
            dx = curr_position[0] - self.prev_position[0]
            dy = curr_position[1] - self.prev_position[1]
            dz = curr_position[2] - self.prev_position[2]
            dist = math.sqrt(dx**2 + dy**2 + dz**2)

            if dist < self.position_thresh:
                return

        self.prev_position = curr_position

        frame = []


        target_order = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]

        name_to_index = {name: i for i, name in enumerate(joint.name)}

        ordered_pos = []
        ordered_vel = []

        for name in target_order:
            if name in name_to_index:
                idx = name_to_index[name]
                ordered_pos.append(joint.position[idx])
                ordered_vel.append(joint.velocity[idx])
            else:
                ordered_pos.append(0.0)
                ordered_vel.append(0.0)

        


        # base xyz
        frame.extend([pos.x, pos.y, pos.z])

        # quaternion
        ori = odom.pose.pose.orientation
        frame.extend([ori.x, ori.y, ori.z, ori.w])

        # joint pos
        # frame.extend(joint.position[:12])
        frame.extend(ordered_pos)

        # linear vel
        lin = odom.twist.twist.linear
        frame.extend([lin.x, lin.y, lin.z])

        # angular vel
        ang = odom.twist.twist.angular
        frame.extend([ang.x, ang.y, ang.z])

        # joint vel
        # frame.extend(joint.velocity[:12])
        frame.extend(ordered_vel)
        

        self.frames.append(frame)

    # ================= 保存 =================

    def save_json(self):

        with open(self.filename, 'w') as f:
            f.write('{\n')
            f.write('  "LoopMode": "Wrap",\n')
            f.write('  "FrameDuration": 0.021,\n')
            f.write('  "EnableCycleOffsetPosition": true,\n')
            f.write('  "EnableCycleOffsetRotation": true,\n')
            f.write('  "MotionWeight": 0.5,\n')
            f.write('  "Frames": [\n')

            for i, frame in enumerate(self.frames):
                line = ', '.join('{:.5f}'.format(x) for x in frame)
                if i < len(self.frames) - 1:
                    f.write('    [' + line + '],\n')
                else:
                    f.write('    [' + line + ']\n')

            f.write('  ]\n')
            f.write('}\n')

        self.get_logger().info(
            f"Saved {len(self.frames)} frames to {self.filename}"
        )


# ================= 主函数 =================

def main(args=None):
    rclpy.init(args=args)

    node = MotionRecorderROS2()

    try:
        node.get_logger().info("Motion recorder started...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_json()
        node.destroy_node()
        rclpy.shutdown()
2

if __name__ == '__main__':
    main()