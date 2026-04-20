import rclpy
from rclpy.node import Node
from unitree_ros2_real import UnitreeRos2Real, get_euler_xyz

import os
import ast
import os.path as osp
import json
import time
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import sys
sys.path.append("/home/unitree/WMP")  # 添加父目录到Python路径
sys.path.append('/home/unitree/WMP/rsl_rl')

import cv2

import time
import sys
import threading

from dreamer import *  # 现在可以正常导入

from sport_api_constants import *

class Go2Node(UnitreeRos2Real):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_class_name= "Go2", **kwargs)
        self.global_counter = 0
        self.visual_update_interval = 5
        self.is_recording = False
        self.recorded_data = []
        self.last_b_pressed = False  # 添加这一行，记录上一次B键的状态
        # self.actions_sim = torch.from_numpy(np.load('Action_sim_335-11_flat.npy')).to(self.model_device)

        self.sim_ite = 3
 
        self.use_stand_policy = False
        self.use_parkour_policy = False
        self.use_sport_mode = True
        self.NUM_DOF = 12  # 或直接写 12 if 确定是 12 DOF
        self.enable_monitor = False

        if self.enable_monitor:
            from collections import deque
            import matplotlib.pyplot as plt
            import time

            self.monitor_plot_size = 2000  # 200Hz * 10s
            self.monitor_time = deque(maxlen=self.monitor_plot_size)
            self.monitor_real_joint_pos = [deque(maxlen=self.monitor_plot_size) for _ in range(self.NUM_DOF)]
            self.monitor_target_joint_pos = [deque(maxlen=self.monitor_plot_size) for _ in range(self.NUM_DOF)]
            self.monitor_last_save_time = time.monotonic()
            self.monitor_motion_time = 0
            self.monitor_timer = self.create_timer(0.005, self.monitor_timer_callback)

    # This warm up is useful in my experiment on Go2
    # The first two iterations are very slow, but the rest is fast
    def warm_up(self):
        for _ in range(2):
            start_time = time.monotonic()

            proprio = self.get_proprio()
            get_pro_time = time.monotonic()
            get_hist_pro_time = time.monotonic()

            history = self.trajectory_history.flatten(1).to(self.model_device)
            self.wm_obs =  {
                "prop": proprio[:,:33],
                "is_first": self.wm_is_first,
            }
            if self.global_counter % self.visual_update_interval == 0:
                depth_image = self._get_depth_image()
                self.wm_obs["image"] = depth_image
                self.wm_feature,self.wm_latent = self.depth_encode(self.wm_obs, self.wm_action,self.wm_latent)


            get_obs_time = time.monotonic()

            action = self.policy(self._get_commands_obs(), history, self.wm_feature)


            policy_time = time.monotonic()

            publish_time = time.monotonic()
            print("warm up: ",
                "get proprio time: {:.5f}".format(get_pro_time - start_time),
                "get hist pro time: {:.5f}".format(get_hist_pro_time - get_pro_time),
                "get_depth time: {:.5f}".format(get_obs_time - get_hist_pro_time),
                "get obs time: {:.5f}".format(get_obs_time - start_time),
                "policy_time: {:.5f}".format(policy_time),
                "publish_time: {:.5f}".format(publish_time - policy_time),
                "total time: {:.5f}".format(publish_time - start_time)
            )

    def monitor_timer_callback(self):
        if self.low_state_buffer is None or self.low_cmd_buffer is None:
            return

        real_joint_data = [self.low_state_buffer.motor_state[i].q for i in range(self.NUM_DOF)]
        target_joint_data = [self.low_cmd_buffer.motor_cmd[i].q for i in range(self.NUM_DOF)]

        self.monitor_motion_time += 1
        self.monitor_time.append(self.monitor_motion_time)

        for i in range(self.NUM_DOF):
            self.monitor_real_joint_pos[i].append(real_joint_data[i])
            self.monitor_target_joint_pos[i].append(target_joint_data[i])

        now = time.monotonic()
        if now - self.monitor_last_save_time >= 10.0:
            import matplotlib.pyplot as plt
            plt.clf()
            for i in range(self.NUM_DOF):
                plt.subplot(4, 3, i + 1)
                plt.plot(self.monitor_time, self.monitor_real_joint_pos[i], 'r', label='real')
                plt.plot(self.monitor_time, self.monitor_target_joint_pos[i], 'b', label='target')
                plt.legend()
            filename = f"joint_plot_{int(now)}.png"
            plt.savefig(filename)
            self.get_logger().info(f"Joint plot saved to {filename}")
            self.monitor_last_save_time = now



    def register_models(self, depth_encode, policy):
        self.depth_encode = depth_encode
        self.policy = policy

    def start_main_loop_timer(self, duration):
        self.main_loop_timer = self.create_timer(
            duration, # in sec
            self.main_loop,
        )
        
    def main_loop(self):
        if self.use_sport_mode:
            if (self.joy_stick_buffer.keys & self.WirelessButtons.R1):
                self.get_logger().info("In the sport mode, R1 pressed, robot will stand up.")
                self._sport_mode_change(ROBOT_SPORT_API_ID_STANDUP)
            if (self.joy_stick_buffer.keys & self.WirelessButtons.R2):
                self.get_logger().info("In the sport mode, R2 pressed, robot will sit down.")
                self._sport_mode_change(ROBOT_SPORT_API_ID_STANDDOWN)

            if (self.joy_stick_buffer.keys & self.WirelessButtons.X):
                self.get_logger().info("In the sport mode, X pressed, robot will balance stand.")
                self._sport_mode_change(ROBOT_SPORT_API_ID_BALANCESTAND)

            if (self.joy_stick_buffer.keys & self.WirelessButtons.L1):
                self.get_logger().info("Exist the sport mode. Switch to stand policy.")
                self.use_sport_mode = False
                self._sport_state_change(0)
                self.use_stand_policy = True
                self.use_parkour_policy = False
        
        if self.use_stand_policy:
            stand_action = self.get_stand_action()
            self.send_stand_action(stand_action)
        
        if (self.joy_stick_buffer.keys & self.WirelessButtons.Y):
            self.get_logger().info("Y pressed, use the parkour policy")
            self.use_stand_policy = False
            self.use_parkour_policy = True
            self.use_sport_mode = False
            self.global_counter = 2

        if self.use_parkour_policy:
            self.use_stand_policy = False
            self.use_sport_mode = False
                        # 检查 B 键状态
            b_pressed = bool(self.joy_stick_buffer.keys & self.WirelessButtons.B)
            if b_pressed and not self.last_b_pressed:  # 只在B键从未按下状态变为按下状态时触发
                if not self.is_recording:
                    self.get_logger().info("B pressed, start recording data")
                    self.is_recording = True
                    self.recorded_data = []  # 使用单个列表存储所有数据
                else:
                    self.get_logger().info("B pressed, stop recording and save data")
                    self.is_recording = False
                    if self.recorded_data:
                        timestamp = int(time.time())
                        filename = f'parkour_data_{timestamp}.npy'
                        
                        # 转换为结构化数组并保存
                        dtype = [
                            ('timestamp', 'f8'),  # 时间戳
                            ('wm_obs_prop', 'f4', (33,)),  # proprioception数据
                            ('wm_obs_image', 'f4', (64, 64, 1)),  # 深度图像
                            ('wm_obs_is_first', 'f4', (1,)),  # is_first标志
                            ('command', 'f4', (self._get_commands_obs().shape[-1],)),  # 命令数据
                            ('wm_action', 'f4', (60,)),  # 新增动作数据
                        ]
                        structured_data = np.array(self.recorded_data, dtype=dtype)
                        np.save(filename, structured_data)
                        structured_data = np.array(self.recorded_data, dtype=dtype)
                        np.save(filename, structured_data)
                        self.get_logger().info(f"Saved recorded data to {filename}")
                        self.recorded_data = []
            self.last_b_pressed = b_pressed  # 更新按键状态
            
            start_time = time.monotonic()
            time1 = time.monotonic()
            time2 = time.monotonic()
            proprio = self.get_proprio()
            get_pro_time = time.monotonic()
            get_hist_pro_time = time.monotonic()
            # print('proprioception: ', proprio)
            # print('history proprioception: ', proprio_history)

            self.wm_obs["image"] = torch.zeros(((1,) + (64,64) + (1,)),
                                          device=self.model_device)
            self.wm_obs =  {
                "prop": proprio[:,:33],
                "is_first": self.wm_is_first,
            }
            if self.global_counter % self.visual_update_interval == 0:
                time1 = time.monotonic()
                depth_image = self._get_depth_image()
                time2 = time.monotonic()
                self.wm_obs["image"] = depth_image
                self.wm_feature,self.wm_latent = self.depth_encode(self.wm_obs, self.wm_action,self.wm_latent)
                self.wm_is_first[:] = 0

            get_obs_time = time.monotonic()

            turn_obs_time = time.monotonic()
            history = self.trajectory_history.flatten(1).to(self.model_device)
            action = self.policy(self._get_commands_obs(), history, self.wm_feature)
            if self.is_recording and self.global_counter % self.visual_update_interval == 0:
                current_data = (
                    time.monotonic(),  # 时间戳
                    self.wm_obs['prop'].cpu().numpy()[0],  # proprioception
                    self.wm_obs['image'].cpu().numpy()[0] if 'image' in self.wm_obs else np.zeros((64,64,1)),  # 深度图像
                    self.wm_obs['is_first'].cpu().numpy()[0],  # is_first
                    self._get_commands_obs().cpu().numpy()[0],
                    self.wm_action.cpu().numpy()[0]  # 命令
                )
                self.recorded_data.append(current_data)
            self.wm_action_history = torch.concat(
            (self.wm_action_history[:, 1:], action.unsqueeze(1)), dim=1)
            self.wm_action = self.wm_action_history.flatten(1)

            policy_time = time.monotonic()
            # action = self.actions_sim[self.sim_ite, :]
            self.send_action(action)
            self.sim_ite += 1

        
            publish_time = time.monotonic()
            print(
                "get proprio time: {:.5f}".format(get_pro_time - start_time),
                "get hist pro time: {:.5f}".format(get_hist_pro_time - get_pro_time),
                "get_depth time: {:.5f}".format(get_obs_time - get_hist_pro_time),
                "get obs time: {:.5f}".format(get_obs_time - start_time),
                "policy_time: {:.5f}".format(policy_time - turn_obs_time),
                "publish_time: {:.5f}".format(publish_time - policy_time),
                "img_time: {:.5f}".format(time2 - time1),
                "total time: {:.5f}".format(publish_time - start_time)
            )
            self.global_counter += 1

        if (self.joy_stick_buffer.keys & self.WirelessButtons.L2):
            self.get_logger().info("L2 pressed, stop using parkour policy, switch to sport mode.")
            self.use_stand_policy = False
            self.use_parkour_policy = False
            self.use_sport_mode = True
            self.reset_obs()
            self._sport_state_change(1)
            self._sport_mode_change(ROBOT_SPORT_API_ID_BALANCESTAND)


@torch.inference_mode()
def main(args):
    rclpy.init()

    assert args.logdir is not None, "Please provide a logdir"
    with open(osp.join(args.logdir, "config.json"), "r") as f:
        config_dict = json.load(f, object_pairs_hook= OrderedDict)
    
    config_dict["control"]["computer_clip_torque"] = True
    
    # duration = config_dict["sim"]["dt"] * config_dict["control"]["decimation"] # different from parkour
    device = "cuda"
    duration = 0.02

    env_node = Go2Node(
        "go2",
        cfg= config_dict,
        model_device= device,
        dryrun= not args.nodryrun,
        mode = args.mode,
    )

    env_node.get_logger().info("Model loaded from: {}".format(osp.join(args.logdir)))
    env_node.get_logger().info("Control Duration: {} sec".format(duration))
    env_node.get_logger().info("Motor Stiffness (kp): {}".format(env_node.p_gains))
    env_node.get_logger().info("Motor Damping (kd): {}".format(env_node.d_gains))



    wm_encoder = torch.jit.load('/home/unitree/WMP/deploy/traced/wm_enc.pt', map_location=device)
    hist_encoder = torch.jit.load('/home/unitree/WMP/deploy/traced/hist_enc.pt', map_location=device)
    actor = torch.jit.load('/home/unitree/WMP/deploy/traced/actor.pt', map_location=device)

    # vel_enc = torch.jit.load('/home/unitree/WMP/deploy/traced/vel_enc3.pt', map_location=device)
    # vel_enc.eval()

    wm_encoder.eval()
    hist_encoder.eval()
    actor.eval()

    obs_shape = {'prop': (33,), 'image': (64,64,1)}

    # -------------
    # encoder = networks.MultiEncoder(obs_shape,'.*','image','SiLU',True,64,4,8,5,1024,True,True)
    # dynamics = networks.RSSM(32,512,512,1,32,'SiLU',True,'none','sigmoid2',0.1,0.01,'learned',60,1280,device)
    
    encoder = networks.MultiEncoder(obs_shape,'.*','image','SiLU',True,32,4,4,5,1024,True,True)
    dynamics = networks.RSSM(32,512,512,1,32,'SiLU',True,'none','sigmoid2',0.1,0.01,'learned',60,5120,device)
    # # 原始 state_dict

    wm_model = torch.load('/home/unitree/WMP/deploy/traced/model_10000.pt', map_location=device)
    # 过滤 encoder 和 dynamics 的参数（去掉前缀以匹配子模块）
    full_dict = wm_model['world_model_dict']
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


    def encode_depth(wm_obs,wm_action,wm_latent):
        wm_embed = encoder(wm_obs)
        wm_latent, _ = dynamics.obs_step(wm_latent, wm_action, wm_embed, wm_obs["is_first"], sample=False)
        wm_feature = dynamics.get_deter_feat(wm_latent)
        return wm_feature,wm_latent
    
    def actor_model(command, history, wm_feature):
        latent_vector = hist_encoder(history)
        wm_latent_vector = wm_encoder(wm_feature)
        concat_observations = torch.concat((latent_vector, command, wm_latent_vector),
                                           dim=-1)
        action = actor(concat_observations)
        return action

    # def actor_model(command, history, wm_feature):
    #     latent_vector = hist_encoder(history)
    #     vel = vel_enc(latent_vector)
    #     wm_latent_vector = wm_encoder(wm_feature)
    #     concat_observations = torch.concat((latent_vector, vel, command, wm_latent_vector),
    #                                        dim=-1)
    #     action = actor(concat_observations)
    #     return action

    env_node.register_models(depth_encode=encode_depth, policy=actor_model)


    env_node.start_ros_handlers()
    env_node.warm_up()

    if args.loop_mode == "while":
        rclpy.spin_once(env_node, timeout_sec= 0.)
        env_node.get_logger().info("Model and Policy are ready")
        while rclpy.ok():
            main_loop_time = time.monotonic()
            env_node.main_loop()
            rclpy.spin_once(env_node, timeout_sec= 0.)
            time.sleep(max(0, duration - (time.monotonic() - main_loop_time)))
    elif args.loop_mode == "timer":
        env_node.get_logger().info('Model and Policy are ready')
        env_node.start_main_loop_timer(duration)
        rclpy.spin(env_node)

    rclpy.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--logdir", type= str, default= '/home/unitree/WMP/deploy/traced', help= "The directory which contains the config.json and model_*.pt files")
    parser.add_argument("--nodryrun", action= "store_true", default= False, help= "Disable dryrun mode")
    parser.add_argument("--loop_mode", type= str, default= "timer",
        choices= ["while", "timer"],
        help= "Select which mode to run the main policy control iteration",
    )
    parser.add_argument("--mode", type= str, default= "parkour", choices=["parkour", "walk"])
    args = parser.parse_args()
    
    main(args)
