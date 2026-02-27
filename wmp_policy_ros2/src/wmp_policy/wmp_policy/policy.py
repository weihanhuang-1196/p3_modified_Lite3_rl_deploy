#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import Image, Imu
from wmp_policy.dreamer import *
import torch

import threading


class PolicyNode(Node):

    dynamics = None
    encoder = None
    device = "cuda"
    depth_lock = threading.Lock()
    proprio_lock = threading.Lock()

    dof_names = [
        'FL_hip_joint', 'RL_hip_joint', 'FR_hip_joint', 'RR_hip_joint',
        'FL_thigh_joint', 'RL_thigh_joint', 'FR_thigh_joint', 'RR_thigh_joint',
        'FL_calf_joint', 'RL_calf_joint', 'FR_calf_joint', 'RR_calf_joint'
    ]

    default_joint_angles = { # = target angles [rad] when action = 0.0
        'FL_hip_joint': 0.0,   # [rad]
        'RL_hip_joint': 0.0,   # [rad]
        'FR_hip_joint': -0.0 ,  # [rad]
        'RR_hip_joint': -0.0,   # [rad]

        'FL_thigh_joint': 0.71,     # [rad]
        'RL_thigh_joint': 0.71,   # [rad]
        'FR_thigh_joint': 0.71,     # [rad]
        'RR_thigh_joint': 0.71,   # [rad]

        'FL_calf_joint': -1.5,   # [rad]
        'RL_calf_joint': -1.5,    # [rad]
        'FR_calf_joint': -1.5,  # [rad]
        'RR_calf_joint': -1.5,    # [rad]
    }
    dof_pos = []
    dof_vel = []
    output_dof_pos = []
    action_scale = 0.25
    stiffness = {'joint': 55.}  # [N*m/rad]
    damping = {'joint': 1.8}  # [N*m*s/rad]
    num_dof = 12
    clip_actions = 7.0
    depth_data = None
    n_hist_len = 5

    def __init__(self, device="cuda"):
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
        self.depth_sub = self.create_subscription(
            Image,
            '/depth/depth_camera/depth/image_raw',  # 你的深度图话题名
            self.depth_callback,
            10
        )

        # 订阅本体感知信息
        self.imu_sub = self.create_subscription(
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
        msg.data = dof_pos.cpu().numpy().tolist()
        self.output_dof_pos_pub.publish(msg)

    def get_proprio(self):
        # 将最新的本体感知消息转换为 numpy 数组
        proprio_data = None
        with self.proprio_lock:
            proprio_data = np.array(self.latest_proprio.data, dtype=np.float32)
        torch.tensor(proprio_data, dtype=torch.float32, device=self.device)
        obs_without_command = torch.concat((proprio_data[:,:6],
                                        proprio_data[:, 9:]), dim=1)
        self.trajectory_history = torch.concat((self.trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)
        return proprio_data

    def main_loop(self):
        start_time = time.monotonic()

        proprio = self.get_proprio()
        get_pro_time = time.monotonic()
        proprio_history = self._get_history_proprio()
        get_hist_pro_time = time.monotonic()
        # print('proprioception: ', proprio)
        # print('history proprioception: ', proprio_history)
        self.wm_obs =  {
            "prop": proprio[:,33],
            "is_first": self.wm_is_first,
        }
        if self.global_counter % self.visual_update_interval == 0:
            with self.depth_lock:
                if self.latest_depth_tensor is not None:
                    self.wm_obs["image"] = self.latest_depth_tensor
                else:
                    return
            self.wm_embed = self.encoder(self.wm_obs)
            self.wm_latent, _ = self.dynamics.obs_step(self.wm_latent, self.wm_action, self.wm_embed, self.wm_obs["is_first"], sample=True)
            self.wm_feature = self.dynamics.get_deter_feat(self.wm_latent)
            self.wm_is_first[:] = 0
            # self.depth_latent_yaw = self.depth_encode(self.last_depth_image, proprio)
            # print('depth latent: ', self.depth_latent_yaw)
        get_obs_time = time.monotonic()

            # obs = self.turn_obs(proprio, self.depth_latent_yaw, proprio_history, self.n_proprio, self.n_depth_latent, self.n_hist_len)

        turn_obs_time = time.monotonic()
        history = self.trajectory_history.flatten(1).to(self.model_device)
        action = self.policy(self._get_commands_obs, history, self.wm_feature)
        policy_time = time.monotonic()
        # print('action before clip and normalize: ', action)
        # action = self.actions_sim[self.sim_ite, :]

        action = torch.clip(action, -self.clip_actions, self.clip_actions).to(self.device)
        actions_scaled = action * self.action_scale
        self.output_dof_pos = self.default_dof_pos + actions_scaled

        self.send_output_dof_pos(self.output_dof_pos)
        # print('action: ', action)

        publish_time = time.monotonic()
        print(
            "get proprio time: {:.5f}".format(get_pro_time - start_time),
            "get hist pro time: {:.5f}".format(get_hist_pro_time - get_pro_time),
            "get_depth time: {:.5f}".format(get_obs_time - get_hist_pro_time),
            "get obs time: {:.5f}".format(get_obs_time - start_time),
            "turn_obs_time: {:.5f}".format(turn_obs_time - get_obs_time),
            "policy_time: {:.5f}".format(policy_time - turn_obs_time),
            "publish_time: {:.5f}".format(publish_time - policy_time),
            "total time: {:.5f}".format(publish_time - start_time)
        )
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

        depth = depth.permute(0,2,3,1)  # NHWC

        with self.depth_lock:
            self.latest_depth_tensor = depth.to(self.device)

    def proprio_callback(self, msg: Float32MultiArray):
        # 保存最新本体感知信息
        self.latest_proprio = msg
        self.get_logger().debug('Received proprio data.')
    




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
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.count += 1

def main(args=None):
    rclpy.init(args=args)
    # node = MinimalPublisher()
    # rclpy.spin(node)
    # node.destroy_node()


    device = "cuda"
    duration = 0.02
    policy_node = PolicyNode(device=device)

    wm_encoder = torch.jit.load('/home/yong/yang/p3_modified_Lite3_rl_deploy/wmp_policy_ros2/src/wmp_policy/policy/wm_enc.pt', map_location=device)
    hist_encoder = torch.jit.load('/home/yong/yang/p3_modified_Lite3_rl_deploy/wmp_policy_ros2/src/wmp_policy/policy/hist_enc.pt', map_location=device)
    actor = torch.jit.load('/home/yong/yang/p3_modified_Lite3_rl_deploy/wmp_policy_ros2/src/wmp_policy/policy/actor.pt', map_location=device)
    wm_encoder.eval()
    hist_encoder.eval()
    actor.eval()


    obs_shape = {'prop': (33,), 'image': (64,64,1)}
    encoder = networks.MultiEncoder(obs_shape,'.*','image','SiLU',True,32,4,4,5,1024,True,True)
    dynamics = networks.RSSM(32,512,512,1,32,'SiLU',True,'none','sigmoid2',0.1,0.01,'learned',60,5120,device)
    # 原始 state_dict
    wm_model = torch.load('/home/yong/yang/p3_modified_Lite3_rl_deploy/wmp_policy_ros2/src/wmp_policy/policy/model_12000.pt', map_location=device)
    full_dict = wm_model['world_model_dict']

    # 过滤 encoder 和 dynamics 的参数（去掉前缀以匹配子模块）
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

    # 加载参数到各自模块
    encoder.load_state_dict(encoder_dict, strict=False)
    dynamics.load_state_dict(dynamics_dict, strict=False)

    encoder.to(device)
    dynamics.to(device)
    encoder.eval()
    dynamics.eval()

    policy_node.dynamics = dynamics
    policy_node.encoder = encoder


    def actor_model(command, history, wm_feature):
        latent_vector = hist_encoder(history)
        wm_latent_vector = wm_encoder(wm_feature)
        concat_observations = torch.concat((latent_vector, command, wm_latent_vector),
                                           dim=-1)
        action = actor(concat_observations)
        return action

    policy_node.register_models(policy=actor_model)



    policy_node.get_logger().info('Model and Policy are ready')
    policy_node.start_main_loop_timer(duration)
    rclpy.spin(policy_node)

    rclpy.shutdown()

if __name__ == '__main__':
    main()