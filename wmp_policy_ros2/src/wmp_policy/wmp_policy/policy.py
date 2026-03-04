#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import Image, Imu
from wmp_policy.dreamer import *
import torch
import cv2
import numpy as np

import threading


class PolicyNode(Node):

    dynamics = None
    encoder = None
    device = "cuda"
    depth_lock = threading.Lock()
    proprio_lock = threading.Lock()

    dof_names = [
        'FL_hip_joint', 'FL_thigh_joint' , 'FL_calf_joint',
          'FR_hip_joint','FR_thigh_joint','FR_calf_joint',
          'RL_hip_joint', 'RL_thigh_joint','RL_calf_joint',
          'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
    ]

    default_joint_angles = { # = target angles [rad] when action = 0.0
        'FL_hip_joint': 0.0,   # [rad]
        'FL_thigh_joint': 0.71,     # [rad]
        'FL_calf_joint': -1.5,   # [rad]

        'RL_hip_joint': 0.0,   # [rad]
        'RL_thigh_joint': 0.71,   # [rad]
        'RL_calf_joint': -1.5,    # [rad]

        'FR_hip_joint': -0.0 ,  # [rad]
        'FR_thigh_joint': 0.71,     # [rad]
        'FR_calf_joint': -1.5,  # [rad]

        'RR_hip_joint': -0.0,   # [rad]
        'RR_thigh_joint': 0.71,   # [rad]
        'RR_calf_joint': -1.5,    # [rad]
        
    }
    dof_pos = []
    dof_vel = []
    output_dof_pos = []
    action_scale = 0.25
    stiffness = {'joint': 55.}  # [N*m/rad]
    damping = {'joint': 1.8}  # [N*m*s/rad]
    num_dof = 12
    num_actions = 12
    clip_actions = 7.0
    depth_data = None
    n_hist_len = 5
    visual_update_interval = 5
    world_model = None
    policy_model = None
    actions = None
    latest_depth_tensor = None
    num_obs = 42

    def __init__(self, device="cuda:0"):
        super().__init__('policy_node')
        self.device = device
        self.config()
        self.init_rostopic()
        


    def config(self):
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.output_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.trajectory_history = torch.zeros(size=(1, self.n_hist_len, 42), device = self.device)
        self.global_counter = 0
        self.wm_is_first = torch.ones(1, device=self.device)
        self.wm_action = torch.zeros(1, 60, device=self.device)
        self.latest_proprio = None 
        self.latest_depth_tensor = torch.zeros((1, 64, 64, 1), device=self.device)
        self.policy_prop = np.zeros(self.num_obs, dtype=np.float32)
        self.wm_prop = np.zeros(33, dtype=np.float32)
        self.obs_history = np.zeros((5, self.num_obs), dtype=np.float32)

        self.cmd = np.zeros(3, dtype=np.float32) 

        self.wm_action_history = torch.zeros(
            size=(1, 5, self.num_actions),
            device=self.device
        )
        self.actions = torch.zeros(1, self.num_actions, device=self.device)
        self.action = np.zeros(self.num_actions, dtype=np.float32)

        self.wm_logit = torch.zeros(1, 32, 32, device=self.device)
        self.wm_stoch = torch.zeros(1, 32, 32, device=self.device)
        self.wm_deter = torch.zeros(1, 512, device=self.device)
        self.wm_feature = torch.zeros(1, 512, device=self.device)

        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.stiffness[dof_name]
                    self.d_gains[i] = self.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)


    def init_rostopic(self):
        # self.depth_sub = self.create_subscription(
        #     Image,
        #     '/depth/depth_camera/depth/image_raw',  # 你的深度图话题名
        #     self.depth_callback,
        #     10
        # )

        # 订阅本体感知信息
        self.proprio_sub = self.create_subscription(
            Float32MultiArray,
            '/proprio', 
            self.proprio_callback,
            10
        )

        # 发布 12 个动作
        self.output_dof_pos_pub = self.create_publisher(
            Float32MultiArray,
            '/policy/output_dof_pos',
            10
        )

    def register_models(self, policy):
        self.policy = policy

    def start_main_loop_timer(self, duration):
        self.main_loop_timer = self.create_timer(
            duration, # in sec
            self.main_loop,
        )

    def send_output_dof_pos(self, dof_pos):
        msg = Float32MultiArray()
        msg.data = dof_pos.detach().cpu().numpy().flatten().tolist()
        self.output_dof_pos_pub.publish(msg)
    


    def build_observation(self):
        latest_proprio = None
        with self.proprio_lock:
            latest_proprio = self.latest_proprio

        qj = latest_proprio.data[9:21]
        dqj = latest_proprio.data[21:33]
        omega = latest_proprio.data[0:3]
        gravity = latest_proprio.data[3:6]

        # 使用手柄输入修改 self.cmd
        # self.cmd[0] = self.latest_proprio.data[6]      # 第一个元素
        # self.cmd[1] = self.latest_proprio.data[7]      # 第二个元素  
        # self.cmd[2] = self.latest_proprio.data[8]    # 第三个元素
        self.cmd[0] = 0.0     # 第一个元素
        self.cmd[1] = 0.0      # 第二个 元素 
        self.cmd[2] = 0.0    # 第三个

        self.policy_prop[:3] = omega
        self.policy_prop[3:6] = gravity
        self.policy_prop[6:18] = qj
        self.policy_prop[18:30] = dqj
        self.policy_prop[30:42] = self.action

        self.wm_prop[:3] = omega
        self.wm_prop[3:6] = gravity
        self.wm_prop[6:9] = self.cmd[:3]
        self.wm_prop[9:21] = qj
        self.wm_prop[21:33] = dqj

    def main_loop(self):
        start_time = time.monotonic()
        with self.proprio_lock:
            if self.latest_proprio is None:
                return None
        self.build_observation()
        get_pro_time = time.monotonic()
        get_hist_pro_time = time.monotonic()

        input_wm_prop = torch.tensor(self.wm_prop, dtype=torch.float32, device=self.device).unsqueeze(0)
        history_tensor = torch.tensor(self.obs_history, dtype=torch.float32, device=self.device).unsqueeze(0)
        cmd_tensor = torch.tensor(self.cmd[:3], dtype=torch.float32, device=self.device).unsqueeze(0)

        if self.global_counter % self.visual_update_interval == 0:
            self.wm_logit, self.wm_stoch, self.wm_deter, self.wm_feature = self.world_model(
                input_wm_prop,
                self.latest_depth_tensor,
                self.wm_logit,
                self.wm_stoch,
                self.wm_deter,
                self.wm_action,
                self.wm_is_first,
            )
            self.wm_is_first[:] = 0
        get_obs_time = time.monotonic()

        turn_obs_time = time.monotonic()

        # ===== Policy 推理 =====
        history_flat = history_tensor.flatten(1)
        action = self.policy_model(cmd_tensor.detach(),history_flat.detach(),self.wm_feature.detach())
        action = action.squeeze(0)

        # ===== 更新 WM action history =====
        action_tensor = action.unsqueeze(0).unsqueeze(0)
        self.wm_action_history = torch.cat((self.wm_action_history[:, 1:], action_tensor), dim=1)
        self.wm_action = self.wm_action_history.flatten(1)


        self.obs_history[:-1] = self.obs_history[1:]
        self.obs_history[-1] = self.policy_prop

        policy_time = time.monotonic()
        self.actions = action.to(self.device)
        actions_scaled = self.actions * self.action_scale
        self.output_dof_pos = self.default_dof_pos + actions_scaled

        self.send_output_dof_pos(self.output_dof_pos)

        publish_time = time.monotonic()
        # print(
        #     "get proprio time: {:.5f}".format(get_pro_time - start_time),
        #     "get hist pro time: {:.5f}".format(get_hist_pro_time - get_pro_time),
        #     "get_depth time: {:.5f}".format(get_obs_time - get_hist_pro_time),
        #     "get obs time: {:.5f}".format(get_obs_time - start_time),
        #     "turn_obs_time: {:.5f}".format(turn_obs_time - get_obs_time),
        #     "policy_time: {:.5f}".format(policy_time - turn_obs_time),
        #     "publish_time: {:.5f}".format(publish_time - policy_time),
        #     "total time: {:.5f}".format(publish_time - start_time)
        # )
        self.global_counter += 1


    def depth_callback(self, msg: Image):
        if msg.encoding == "16UC1":
            dtype = torch.uint16
            depth = torch.frombuffer(msg.data, dtype=dtype)
            depth = depth.view(msg.height, msg.width)
            depth = depth.float() / 1000.0  # mm → m

        elif msg.encoding == "32FC1":
            depth = torch.frombuffer(msg.data, dtype=torch.float32)
            depth = depth.view(msg.height, msg.width)

        else:
            self.get_logger().error(f"Unsupported encoding {msg.encoding}")
            return

        # 加 batch + channel
        depth = depth.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        depth = torch.nn.functional.interpolate(
            depth,
            size=(64,64),
            mode='bilinear',
            align_corners=False
        )

        # ===== 归一化部分 =====
        min_depth = 0.0
        max_depth = 2.0

        depth = torch.clamp(depth, min_depth, max_depth)
        depth = (depth - min_depth) / (max_depth - min_depth)
        depth = depth - 0.5
        # ======================


        depth = depth.permute(0,2,3,1)  # NHWC

        with self.depth_lock:
            self.latest_depth_tensor = depth.to(self.device)

        depth_show = depth.squeeze().cpu().numpy()  # (64,64)

        # 从 [-0.5,0.5] → [0,1]
        depth_vis = depth_show + 0.5

        # clamp 防止数值溢出
        depth_vis = np.clip(depth_vis, 0.0, 1.0)

        # 转成 0~255 uint8
        depth_vis = (depth_vis * 255).astype(np.uint8)

        # 伪彩色
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        cv2.imshow("Depth", depth_color)
        cv2.waitKey(1)



    def proprio_callback(self, msg: Float32MultiArray):
        # 保存最新本体感知信息
        with self.proprio_lock:
            self.latest_proprio = msg
        # self.get_logger().debug('Received proprio data.')
    




class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.count = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello {self.count}'
        self.publisher_.publish(msg)
        # self.get_logger().info(f'Publishing: "{msg.data}"')
        self.count += 1

def main(args=None):
    rclpy.init(args=args)


    device = "cuda:0"
    duration = 0.02
    policy_node = PolicyNode(device=device)

    export_path = "/home/yong/yang/p3_modified_Lite3_rl_deploy/wmp_policy_ros2/src/wmp_policy/policy"
    world_model = torch.jit.load(os.path.join(export_path, "world_model.pt"),map_location=device).to(device)
    policy_model = torch.jit.load(os.path.join(export_path, "policy.pt"),map_location=device).to(device)
    world_model.eval()
    policy_model.eval()

    policy_node.world_model = world_model
    policy_node.policy_model = policy_model



    policy_node.get_logger().info('Model and Policy are ready')
    policy_node.start_main_loop_timer(duration)
    rclpy.spin(policy_node)

    rclpy.shutdown()

if __name__ == '__main__':
    main()