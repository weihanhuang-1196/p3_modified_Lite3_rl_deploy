import json
import time
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch
import math
import sys
import glob


def quaternion_to_euler_xyz(w, x, y, z):
    # Roll (X-axis)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (Y-axis)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(np.pi / 2, sinp)  # 处理90°边界
    else:
        pitch = math.asin(sinp)

    # Yaw (Z-axis)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.degrees([roll, pitch, yaw])  # 转换为角度制

def standardize_quaternion(q):
  """Returns a quaternion where q.w >= 0 to remove redundancy due to q = -q.

  Args:
    q: A quaternion to be standardized.

  Returns:
    A quaternion with q.w >= 0.

  """
  if q[-1] < 0:
    q = -q
  return q

def QuaternionNormalize(q):
  """Normalizes the quaternion to length 1.

  Divides the quaternion by its magnitude.  If the magnitude is too
  small, returns the quaternion identity value (1.0).

  Args:
    q: A quaternion to be normalized.

  Raises:
    ValueError: If input quaternion has length near zero.

  Returns:
    A quaternion with magnitude 1 in a numpy array [x, y, z, w].

  """
  q_norm = np.linalg.norm(q)
  if np.isclose(q_norm, 0.0):
    raise ValueError(
        'Quaternion may not be zero in QuaternionNormalize: |q| = %f, q = %s' %
        (q_norm, q))
  return q / q_norm


def euler_to_quaternion(roll, pitch, yaw):
    # 转换为弧度
    roll_rad = math.radians(roll)
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)

        # 计算半角
    cy = math.cos(yaw_rad * 0.5)
    sy = math.sin(yaw_rad * 0.5)
    cp = math.cos(pitch_rad * 0.5)
    sp = math.sin(pitch_rad * 0.5)
    cr = math.cos(roll_rad * 0.5)
    sr = math.sin(roll_rad * 0.5)

        # 计算四元数分量
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])  # 返回 (w, x, y, z)


def reorder_from_pybullet_to_isaac(motion_data):
    """Convert from PyBullet ordering to Isaac ordering.

    Rearranges leg and joint order from PyBullet [FR, FL, RR, RL] to
    IsaacGym order [FL, FR, RL, RR].
    """
    def get_batch_slice(data, start, end):
        return data[:, start:end]

    # Index constants
    POS_SIZE = 3
    ROT_SIZE = 4
    JOINT_POS_SIZE = 12
    TAR_TOE_POS_LOCAL_SIZE = 12
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3

    JOINT_VEL_SIZE = 12
    TAR_TOE_VEL_LOCAL_SIZE = 12

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE
    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE
    JOINT_POSE_START_IDX = ROOT_ROT_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE
    TAR_TOE_POS_LOCAL_START_IDX = JOINT_POSE_END_IDX
    TAR_TOE_POS_LOCAL_END_IDX = TAR_TOE_POS_LOCAL_START_IDX + TAR_TOE_POS_LOCAL_SIZE
    LINEAR_VEL_START_IDX = TAR_TOE_POS_LOCAL_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE
    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE
    JOINT_VEL_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE
    TAR_TOE_VEL_LOCAL_START_IDX = JOINT_VEL_END_IDX
    TAR_TOE_VEL_LOCAL_END_IDX = TAR_TOE_VEL_LOCAL_START_IDX + TAR_TOE_VEL_LOCAL_SIZE

    root_pos = get_batch_slice(motion_data, ROOT_POS_START_IDX, ROOT_POS_END_IDX)
    root_rot = get_batch_slice(motion_data, ROOT_ROT_START_IDX, ROOT_ROT_END_IDX)

    jp = get_batch_slice(motion_data, JOINT_POSE_START_IDX, JOINT_POSE_END_IDX)
    joint_pos = np.hstack(np.split(jp, 4, axis=1)[1::-1] + np.split(jp, 4, axis=1)[3:1:-1])  # FL, FR, RL, RR

    fp = get_batch_slice(motion_data, TAR_TOE_POS_LOCAL_START_IDX, TAR_TOE_POS_LOCAL_END_IDX)
    foot_pos = np.hstack(np.split(fp, 4, axis=1)[1::-1] + np.split(fp, 4, axis=1)[3:1:-1])

    lv = get_batch_slice(motion_data, LINEAR_VEL_START_IDX, LINEAR_VEL_END_IDX)
    av = get_batch_slice(motion_data, ANGULAR_VEL_START_IDX, ANGULAR_VEL_END_IDX)

    jv = get_batch_slice(motion_data, JOINT_VEL_START_IDX, JOINT_VEL_END_IDX)
    joint_vel = np.hstack(np.split(jv, 4, axis=1)[1::-1] + np.split(jv, 4, axis=1)[3:1:-1])

    fv = get_batch_slice(motion_data, TAR_TOE_VEL_LOCAL_START_IDX, TAR_TOE_VEL_LOCAL_END_IDX)
    foot_vel = np.hstack(np.split(fv, 4, axis=1)[1::-1] + np.split(fv, 4, axis=1)[3:1:-1])

    return np.hstack([
        root_pos, root_rot, joint_pos, foot_pos,
        lv, av, joint_vel, foot_vel
    ])



# ====== 初始化 Isaac Gym ======
gym = gymapi.acquire_gym()



# 设置白色背景（可选）
# gym.set_viewer_background_color(viewer, gymapi.Vec3(1, 1, 1))
args = gymutil.parse_arguments()
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim = gym.create_sim(2, 2, gymapi.SIM_PHYSX, sim_params)
# sys.path.append("/home/liuyuxuan/WMP/rsl_rl/datasets")



if sim is None:
    print("Failed to create sim")
    quit()

# 创建 viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())

# ===== 添加地面 =====
ground_plane = gymapi.PlaneParams()
ground_plane.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, ground_plane)

# 创建环境
env = gym.create_env(sim, gymapi.Vec3(-2, 0, 0), gymapi.Vec3(2, 0, 0), 1)

# 加载 robot asset
# asset_root = "/home/liuyuxuan/HIMLoco/legged_gym/resources/robots/a1/urdf"
# asset_file = "a1.urdf"

# asset_root = "/home/liuyuxuan/disk/IsaacgymLoco/legged_gym/resources/robots/go2/urdf"
# asset_file = "go2.urdf"

asset_root = "/home/yong/yang/WMP_Locomotion_modified/resources/robots/panda3/urdf"
asset_file = "panda3.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.flip_visual_attachments = False
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)


# 添加 actor
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 0.3)
actor_handle = gym.create_actor(env, asset, pose, "robot", 0, 1)

# 获取状态 tensor 句柄
gym.prepare_sim(sim)
root_tensor = gym.acquire_actor_root_state_tensor(sim)
dof_tensor = gym.acquire_dof_state_tensor(sim)

root_states = gymtorch.wrap_tensor(root_tensor)
dof_states = gymtorch.wrap_tensor(dof_tensor)

# 获取 DOF 数
dof_props = gym.get_asset_dof_properties(asset)
num_dofs = len(dof_props)

# ===== 读取轨迹数据 =====
# with open("/home/liuyuxuan/WMP/datasets/mocap_motions/hop1.txt") as f:

with open("/home/yong/yang/p3_modified_Lite3_rl_deploy/wmp_policy_ros2/jump.json") as f:
    motion_json = json.load(f)
    motion_data = np.array(motion_json["Frames"])             # 原始 frames
    reordered_motion = reorder_from_pybullet_to_isaac(motion_data)  # ndarray, (N, D)


frames = reordered_motion                                     # ndarray
frame_dt = motion_json["FrameDuration"]      


# ===== 播放轨迹 =====
frame_count = len(frames)
i = 0


print("Starting playback...")
while not gym.query_viewer_has_closed(viewer):
    frame = frames[i % frame_count]
    frame = np.array(frame)

    # 设置 base pose
    base_pos = frame[0:3]
    base_quat = frame[3:7]
    roll, pitch, yaw = quaternion_to_euler_xyz(base_quat[3],base_quat[0],base_quat[1],base_quat[2])
    print(f"Roll: {roll:.2f}°, Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°")
    w, x, y, z = euler_to_quaternion(yaw ,pitch ,roll)
    roll, pitch, yaw = quaternion_to_euler_xyz(w,x,y,z)
    print(f"Roll: {roll:.2f}°, Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°")

    root_states[0, 0:3] = torch.tensor(base_pos)
    root_states[0, 3:7] = torch.tensor(base_quat)
    # root_states[0, 3:7] = torch.tensor([x, y, z, w], dtype=torch.float32)

    root_states[0, 7:13] = 0  # linear + angular vel = 0

    # 设置关节位置（关节速度设为0）
    joint_pos = frame[7:7+num_dofs]
    dof_states[0:num_dofs, 0] = torch.tensor(joint_pos)
    dof_states[0:num_dofs, 1] = 0

    # 更新状态
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_states))

    # 仿真步进
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    time.sleep(frame_dt)  # 控制播放速度
    i += 1

# ===== 清理资源 =====
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
