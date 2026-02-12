import time
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml

# Joint order mapping
# MuJoCo:    [FL_h, FL_t, FL_c, FR_h, FR_t, FR_c, RL_h, RL_t, RL_c, RR_h, RR_t, RR_c]
# Isaac Lab: [FL_h, FR_h, RL_h, RR_h, FL_t, FR_t, RL_t, RR_t, FL_c, FR_c, RL_c, RR_c]

# MuJoCo index -> Isaac Lab index
MUJOCO_TO_ISAACLAB = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
# Isaac Lab index -> MuJoCo index  
ISAACLAB_TO_MUJOCO = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]

def quat_rotate_inverse(q, v):
    """Rotate vector by inverse of quaternion (world to body frame)."""
    q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]
    t = 2.0 * np.cross(np.array([q_x, q_y, q_z]), v)
    return v - q_w * t + np.cross(np.array([q_x, q_y, q_z]), t)

def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name")
    args = parser.parse_args()
    
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{args.config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        cmd = np.array(config["cmd_init"], dtype=np.float32)
        lin_vel_scale = 2.0

    # Reorder default angles to Isaac Lab order
    default_angles_isaaclab = default_angles[MUJOCO_TO_ISAACLAB]

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    m = mujoco.MjModel.from_xml_path(os.path.expanduser(xml_path))
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    
    d.qpos[0:3] = [0, 0, 0.35]
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[7:] = default_angles
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    print("Initial pose set")
    print(f"Joint mapping: MuJoCo -> Isaac Lab: {MUJOCO_TO_ISAACLAB}")

    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[0:3] = tau[3:6]
            d.ctrl[3:6] = tau[0:3]
            d.ctrl[6:9] = tau[9:12]
            d.ctrl[9:12] = tau[6:9]
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                qj = d.qpos[7:]  # MuJoCo order
                dqj = d.qvel[6:]  # MuJoCo order
                quat = d.qpos[3:7]
                
                # World to body frame
                world_lin_vel = d.qvel[0:3]
                world_ang_vel = d.qvel[3:6]
                base_lin_vel = quat_rotate_inverse(quat, world_lin_vel)
                base_ang_vel = quat_rotate_inverse(quat, world_ang_vel)
                
                gravity_orientation = get_gravity_orientation(quat)
                
                # Reorder joint data: MuJoCo -> Isaac Lab
                qj_isaaclab = qj[MUJOCO_TO_ISAACLAB]
                dqj_isaaclab = dqj[MUJOCO_TO_ISAACLAB]
                
                # Build observation (Isaac Lab order)
                obs[0:3] = base_lin_vel * lin_vel_scale
                obs[3:6] = base_ang_vel * ang_vel_scale
                obs[6:9] = gravity_orientation
                obs[9:12] = cmd * cmd_scale
                obs[12:24] = (qj_isaaclab - default_angles_isaaclab) * dof_pos_scale
                obs[24:36] = dqj_isaaclab * dof_vel_scale
                obs[36:48] = action  # previous action (already in Isaac Lab order)
                
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                action = policy(obs_tensor).detach().numpy().squeeze()  # Isaac Lab order
                
                # Reorder action: Isaac Lab -> MuJoCo
                action_mujoco = action[ISAACLAB_TO_MUJOCO]
                target_dof_pos = action_mujoco * action_scale + default_angles

            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
