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
import torchvision.transforms.functional as TF
from rclpy.time import Time
import ros2_numpy as rnp
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.qos import qos_profile_sensor_data
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


import time
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} 耗时: {(end-start)*1000:.3f} ms")
        return result
    return wrapper


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

    # default_joint_angles = { # = target angles [rad] when action = 0.0
    #     'FL_hip_joint': 0.1,   # [rad]
    #     'FL_thigh_joint': 0.8,     # [rad]
    #     'FL_calf_joint': -1.5,   # [rad]

    #     'RL_hip_joint': -0.1,   # [rad]
    #     'RL_thigh_joint': 0.8,   # [rad]
    #     'RL_calf_joint': -1.5,    # [rad]

    #     'FR_hip_joint': 0.1,  # [rad]
    #     'FR_thigh_joint': 1.0,     # [rad]
    #     'FR_calf_joint': -1.5,  # [rad]

    #     'RR_hip_joint': -0.1,   # [rad]
    #     'RR_thigh_joint': 1.0,   # [rad]
    #     'RR_calf_joint': -1.5,    # [rad]
        
    # }

    dof_pos = []
    dof_vel = []
    output_dof_pos = []
    action_scale = 0.25
    stiffness = {'joint': 55.}  # [N*m/rad]
    damping = {'joint': 1.8}  # [N*m*s/rad]
    num_dof = 12
    num_actions = 12
    clip_actions = 100
    depth_data = None
    n_hist_len = 5
    visual_update_interval = 5
    world_model = None
    policy_model = None
    actions = None
    latest_depth_tensor = None
    num_obs = 45
    wm_latent = None

    def __init__(self, device="cuda:0"):
        super().__init__('policy_node')
        self.device = device

        self._last_depth_sim_time = None
        self._last_depth_recv_time = None
        

        self.config()
        self.init_rostopic()
    

    def load_model(self, export_path):
        # export_path = "/home/yong/yang/p3_modified_Lite3_rl_deploy/wmp_policy_ros2/src/wmp_policy/policy"
        # world_model = torch.jit.load(os.path.join(export_path, "world_model_int8.pt"),map_location=device).to(device)
        # policy_model = torch.jit.load(os.path.join(export_path, "policy_int8.pt"),map_location=device).to(device)



        self.wm_encoder = torch.jit.load(os.path.join(export_path, "wm_enc.pt"),map_location=self.device).to(self.device)
        self.hist_encoder = torch.jit.load(os.path.join(export_path, "hist_enc.pt"),map_location=self.device).to(self.device)
        self.actor = torch.jit.load(os.path.join(export_path, "actor.pt"),map_location=self.device).to(self.device)

        self.hist_encoder.eval()
        self.actor.eval()

        self.wm_model = torch.load(os.path.join(export_path, "model.pt"), map_location=self.device)
        full_dict = self.wm_model['world_model_dict']

        self.obs_shape = {'prop': (33,), 'image': (64,64,1)}
        self.encoder = networks.MultiEncoder(self.obs_shape,'.*','image','SiLU',True,32,4,4,5,1024,True,True)
        self.dynamics = networks.RSSM(32,512,512,1,32,'SiLU',True,'none','sigmoid2',0.1,0.01,'learned',60,5120,self.device)

        encoder_dict = {
            k.replace('encoder.', '', 1): v
            for k, v in full_dict.items()
            if k.startswith('encoder.')
        }

        dynamics_dict = {
            k.replace('dynamics.', '', 1): v
            for k, v in full_dict.items()
            if k.startswith('dynamics.')
        }

        self.encoder.load_state_dict(encoder_dict, strict=False)
        self.dynamics.load_state_dict(dynamics_dict, strict=False)

        self.encoder.to(self.device)
        self.dynamics.to(self.device)
        self.encoder.eval()
        self.dynamics.eval()




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
        self.policy_prop = torch.zeros(size=(1, self.num_obs), device=self.device)
        self.wm_prop = np.zeros(33, dtype=np.float32)
        self.obs_history = np.zeros((5, self.num_obs), dtype=np.float32)
        self.prev_wm_image = None
        self.action_tensor = torch.zeros(1, 1, self.num_actions, device=self.device)
        self.prev_wm_image = None


        self.wm_obs = {
            "prop": torch.zeros(1,33,device=self.device, dtype=torch.float),
            "is_first": self.wm_is_first,
        }

        self.wm_obs["image"] = torch.zeros(((1,) + (64,64) + (1,)),
                                      device=self.device)


        self.input_wm_prop_tensor = torch.zeros(1,33,device=self.device)
        self.history_tensor = torch.zeros(1,5,self.num_obs,device=self.device)
        self.cmd_tensor = torch.zeros(1,3,device=self.device)

        # self.cmd = torch.zeros(3, device=self.device)

        self.wm_action_history = torch.zeros(
            size=(1, 5, self.num_actions),
            device=self.device
        )
        self.actions = torch.zeros(self.num_actions, device=self.device)
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
        self.depth_input_pub = self.create_publisher(
            Image,
            "/forward_depth",
            1,
        )


        self.depth_sub = self.create_subscription(
            Image,
            '/depth/depth_camera/depth/image_raw',  # 你的深度图话题名
            self.depth_callback,
            qos_profile_sensor_data
        )

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
            latest_proprio = torch.tensor(self.latest_proprio.data, device=self.device)

        qj = latest_proprio[9:21]
        dqj = latest_proprio[21:33]
        omega = latest_proprio[0:3]
        gravity = latest_proprio[3:6]

        # 使用手柄输入修改 self.cmd
        self.cmd_tensor[0,0] = latest_proprio[6]      # 第一个元素
        self.cmd_tensor[0,1] = latest_proprio[7]      # 第二个元素  
        self.cmd_tensor[0,2] = latest_proprio[8]    # 第三个元素
        # self.cmd[0] = 0.0     # 第一个元素
        # self.cmd[1] = 0.0      # 第二个 元素 
        # self.cmd[2] = 0.0    # 第三个

        self.policy_prop[0,:3] = omega
        self.policy_prop[0,3:6] = gravity
        self.policy_prop[0,6:9] = self.cmd_tensor[0,:3]
        self.policy_prop[0,9:21] = qj
        self.policy_prop[0,21:33] = dqj
        self.policy_prop[0,33:45] = self.actions

        # self.wm_prop[:3] = omega
        # self.wm_prop[3:6] = gravity
        # self.wm_prop[6:9] = self.cmd[:3]
        # self.wm_prop[9:21] = qj
        # self.wm_prop[21:33] = dqj


        obs_without_command = torch.concat((self.policy_prop[:,:6],
                                        self.policy_prop[:, 9:]), dim=1)
        self.trajectory_history = torch.concat((self.trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)


        # self.wm_action_history = torch.cat((self.wm_action_history[:, 1:], self.action_tensor), dim=1)
        # # self.wm_action_history = torch.roll(self.wm_action_history,-1,dim=1)
        # # self.wm_action_history[:,-1] = self.action_tensor
        # self.wm_action = self.wm_action_history.flatten(1)
        # self.obs_history[:-1] = self.obs_history[1:]
        # self.obs_history[-1] = self.policy_prop


    @torch.inference_mode()
    def encode_depth(self,wm_obs,wm_action,wm_latent):
        wm_embed = self.encoder(wm_obs)
        wm_latent, _ = self.dynamics.obs_step(wm_latent, wm_action, wm_embed, wm_obs["is_first"], sample=False)
        wm_feature = self.dynamics.get_deter_feat(wm_latent)
        return wm_feature,wm_latent

    @torch.inference_mode()
    def actor_model(self, command, history, wm_feature):
        latent_vector = self.hist_encoder(history)
        wm_latent_vector = self.wm_encoder(wm_feature)
        concat_observations = torch.concat((latent_vector, command, wm_latent_vector),
                                           dim=-1)
        action = self.actor(concat_observations)
        return action

    @timer
    @torch.inference_mode()
    def main_loop(self):
        # start_time = time.monotonic()
        with self.proprio_lock:
            if self.latest_proprio is None:
                return None
        self.build_observation()

        self.wm_obs["image"] = torch.zeros(((1,) + (64,64) + (1,)),
                                          device=self.device)

        self.wm_obs =  {
            "prop": self.policy_prop[:,:33],
            "is_first": self.wm_is_first,
        }
        
        if self.global_counter % self.visual_update_interval == 0:

            with self.depth_lock:
                depth_image = self.latest_depth_tensor.clone()
            self.wm_obs["image"] =  depth_image
            # self.wm_obs["image"] =  torch.zeros(((1,) + (64,64) + (1,)),
            #                               device=self.device)
            self.wm_feature,self.wm_latent = self.encode_depth(self.wm_obs, self.wm_action,self.wm_latent)
            self.wm_is_first[:] = 0

            if isinstance(self.wm_latent, dict):
                self.wm_latent = {k: v.detach() for k, v in self.wm_latent.items()}
            elif self.wm_latent is not None:
                self.wm_latent = self.wm_latent.detach()

            self.wm_feature = self.wm_feature.detach()
            
            self.prev_wm_image = self.latest_depth_tensor.clone()


        history = self.trajectory_history.flatten(1).to(self.device)
        action = self.actor_model(self.cmd_tensor, history, self.wm_feature)
        self.wm_action_history = torch.concat(
        (self.wm_action_history[:, 1:], action.unsqueeze(1)), dim=1)
        self.wm_action = self.wm_action_history.flatten(1)
        # action = self.actions_sim[self.sim_ite, :]
        self.send_action(action)

        self.global_counter += 1
        # self.sim_ite += 1


    def send_action(self, actions):
        if isinstance(actions, list):
            actions = torch.tensor(actions, device=self.device).unsqueeze(0)
        self.actions = actions.detach()

        clipped_scaled_action = torch.clip(actions, -self.clip_actions, self.clip_actions) * self.action_scale
        # clipped_scaled_action = actions * self.action_scale
        robot_coordinates_action = clipped_scaled_action + self.default_dof_pos.unsqueeze(0)
        self.send_output_dof_pos(robot_coordinates_action[0])

    def _monitor_depth_latency(self, msg: Image):
        """监测深度图延迟和抖动的辅助方法"""
        # 获取当前接收时间
        recv_time = self.get_clock().now()
        
        # 提取仿真时间戳
        sim_time_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # 第一次调用，只记录不计算
        if self._last_depth_recv_time is None:
            self._last_depth_sim_time = sim_time_sec
            self._last_depth_recv_time = recv_time
            return
        
        # 计算时间间隔
        sim_interval_ms = (sim_time_sec - self._last_depth_sim_time) * 1000
        recv_interval = recv_time - self._last_depth_recv_time
        recv_interval_ms = recv_interval.nanoseconds / 1e6
        
        # 期望间隔：20Hz = 50ms
        expected_ms = 100.0
        jitter_ms = abs(recv_interval_ms - expected_ms)
        
        # 打印监测结果
        self.get_logger().info(
            f'[Depth性能] '
            f'仿真间隔: {sim_interval_ms:.2f}ms | '
            f'实际间隔: {recv_interval_ms:.2f}ms | '
            f'抖动: {jitter_ms:.2f}ms | '
            f'状态: {"⚠️ 异常" if jitter_ms > 5 else "✓ 正常"}',
            throttle_duration_sec=2.0  # 2秒打印一次，避免刷屏
        )
        
        # 更新记录
        self._last_depth_sim_time = sim_time_sec
        self._last_depth_recv_time = recv_time


    def fill_neg_inf_by_inpaint(depth, min_d, max_d):
        depth = depth.copy()

        # 1. 构造 mask（-inf 区域）
        mask = (depth == -np.inf).astype(np.uint8)

        # 2. 先把 inf/nan 统一成合法范围（避免污染插值）
        depth = np.nan_to_num(depth, nan=max_d, posinf=max_d, neginf=min_d)

        # 3. inpaint（基于周围像素做渐变填补）
        depth_filled = cv2.inpaint(
            depth.astype(np.float32),
            mask,
            3,                      # 邻域半径
            cv2.INPAINT_NS          # Navier-Stokes，更平滑
        )

        return depth_filled


    def depth_callback(self, msg: Image):
        self._monitor_depth_latency(msg)  # 监测深度图性能
        start_time = self.get_clock().now()

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

        depth_np = depth.cpu().numpy()
        neg_inf_mask = np.isneginf(depth_np).astype(np.uint8)

        depth_np = np.nan_to_num(
            depth_np,
            nan=2.0,
            posinf=2.0,
            neginf=0.0
        )

        if np.any(neg_inf_mask):
            depth_np = cv2.inpaint(
                depth_np.astype(np.float32),
                neg_inf_mask,
                3,                      # 邻域半径（可调 3~5）
                cv2.INPAINT_NS          # 平滑效果更好
            )

        depth = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)

        # 加 batch + channel
        # depth = depth.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        # depth = torch.nn.functional.interpolate(
        #     depth,
        #     size=(64,64),
        #     mode='bilinear',
        #     align_corners=False
        # )

        # ===== 归一化部分 =====
        min_depth = 0.0
        max_depth = 2.0

        depth = torch.clamp(depth, min_depth, max_depth)
        # depth = TF.gaussian_blur(depth, kernel_size=[5, 5], sigma=[1.0, 1.0])
        depth = (depth - min_depth) / (max_depth - min_depth) - 0.5
        # ======================


        depth = depth.permute(0,2,3,1)  # NHWC
        end_time = self.get_clock().now()
        print((end_time - start_time).nanoseconds / 1e6)

        # depth_np = depth.squeeze().detach().squeeze().clone().numpy()
        # depth_input_msg = rnp.msgify(Image, depth_np.astype(np.float32), encoding= "32FC1")
        # depth_input_msg.header.stamp = self.get_clock().now().to_msg()
        # depth_input_msg.header.frame_id = "d435_sim_depth_link"
        # self.depth_input_pub.publish(depth_input_msg)

        with self.depth_lock:
            self.latest_depth_tensor = depth.to(self.device)




        depth_show = depth.squeeze().cpu().numpy()  # (64,64)

        # 从 [-0.5,0.5] → [0,1]
        depth_vis = depth_show + 0.5

        # clamp 防止数值溢出
        # depth_vis = np.clip(depth_vis, 0.0, 1.0)


        # depth_vis_big = np.repeat(
        #     np.repeat(depth_vis, 6, axis=0),
        #     6,
        #     axis=1
        # )


        # # 转成 0~255 uint8
        # depth_vis = (depth_vis * 255).astype(np.uint8)

        # # 伪彩色
        # depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.namedWindow("depth window", cv2.WINDOW_NORMAL)

        cv2.imshow("depth window", depth_vis)
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


    device = "cpu"
    duration = 0.02
    policy_node = PolicyNode(device=device)

    export_path = "/home/yong/yang/p3_modified_Lite3_rl_deploy/wmp_policy_ros2/src/wmp_policy/policy/yuxuan-4-17"
    # world_model = torch.jit.load(os.path.join(export_path, "world_model_int8.pt"),map_location=device).to(device)
    # policy_model = torch.jit.load(os.path.join(export_path, "policy_int8.pt"),map_location=device).to(device)


    policy_node.load_model(export_path)



    policy_node.get_logger().info('Model and Policy are ready')
    policy_node.start_main_loop_timer(duration)
    rclpy.spin(policy_node)

    rclpy.shutdown()

if __name__ == '__main__':
    main()