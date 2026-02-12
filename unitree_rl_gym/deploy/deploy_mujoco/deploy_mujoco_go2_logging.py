import time
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import os
from datetime import datetime

def quat_rotate_inverse(q, v):
    """Rotate vector v by inverse of quaternion q (wxyz format)"""
    q_w, q_x, q_y, q_z = q
    # Conjugate of quaternion
    q_conj = np.array([q_w, -q_x, -q_y, -q_z])
    # v as quaternion [0, vx, vy, vz]
    v_quat = np.array([0, v[0], v[1], v[2]])
    # q_conj * v * q
    # First: q_conj * v
    t = 2 * np.cross(q_conj[1:], v)
    v_rotated = v + q_conj[0] * t + np.cross(q_conj[1:], t)
    return v_rotated

def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("--log_dir", type=str, default="logs/sim2sim")
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()

    LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    PROJECT_ROOT = os.path.dirname(LEGGED_GYM_ROOT_DIR)

    config_path = os.path.join(LEGGED_GYM_ROOT_DIR, "deploy/deploy_mujoco/configs", args.config_file)
    
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        if "unitree_mujoco" in xml_path:
            xml_path = os.path.join(PROJECT_ROOT, "unitree_mujoco", xml_path.split("unitree_mujoco/")[-1])
        
        if "/home/drl-68/Sim-to-Sim_Policy_Transfer_for_Learned-Legged-Locomotion/" in policy_path:
            policy_path = policy_path.replace("/home/drl-68/Sim-to-Sim_Policy_Transfer_for_Learned-Legged-Locomotion/", PROJECT_ROOT + "/")
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

    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"mujoco_baseline_{timestamp}.npz")

    log_data = {
        'time': [],
        'base_pos': [],
        'base_quat': [],
        'base_lin_vel': [],
        'base_ang_vel': [],
        'joint_pos': [],
        'joint_vel': [],
        'actions': [],
        'torques': [],
        'cmd': [],
    }

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    
    d.qpos[0:3] = [0, 0, 0.35]
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[7:] = default_angles
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    print(f"Starting baseline logging for {args.duration}s")
    print(f"Command: vx={cmd[0]}, vy={cmd[1]}, wz={cmd[2]}")

    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        sim_time = 0.0
        
        while viewer.is_running() and sim_time < args.duration:
            step_start = time.time()
            
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[0:3] = tau[3:6]
            d.ctrl[3:6] = tau[0:3]
            d.ctrl[6:9] = tau[9:12]
            d.ctrl[9:12] = tau[6:9]
            mujoco.mj_step(m, d)
            sim_time += simulation_dt

            counter += 1
            if counter % control_decimation == 0:
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]  # MuJoCo: wxyz
                
                # World frame velocities
                world_lin_vel = d.qvel[0:3]
                world_ang_vel = d.qvel[3:6]
                
                # Transform to body frame
                base_lin_vel = quat_rotate_inverse(quat, world_lin_vel)
                base_ang_vel = quat_rotate_inverse(quat, world_ang_vel)
                
                gravity_orientation = get_gravity_orientation(quat)
                
                # Log (body frame velocities)
                log_data['time'].append(sim_time)
                log_data['base_pos'].append(d.qpos[0:3].copy())
                log_data['base_quat'].append(quat.copy())
                log_data['base_lin_vel'].append(base_lin_vel.copy())
                log_data['base_ang_vel'].append(base_ang_vel.copy())
                log_data['joint_pos'].append(qj.copy())
                log_data['joint_vel'].append(dqj.copy())
                log_data['actions'].append(action.copy())
                log_data['torques'].append(tau.copy())
                log_data['cmd'].append(cmd.copy())

                # Build observation with body frame velocities
                obs[0:3] = base_lin_vel * lin_vel_scale
                obs[3:6] = base_ang_vel * ang_vel_scale
                obs[6:9] = gravity_orientation
                obs[9:12] = cmd * cmd_scale
                obs[12:24] = (qj - default_angles) * dof_pos_scale
                obs[24:36] = dqj * dof_vel_scale
                obs[36:48] = action
                
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()
                target_dof_pos = action * action_scale + default_angles

            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    for key in log_data:
        log_data[key] = np.array(log_data[key])
    
    np.savez(log_file, **log_data, cmd_init=cmd, simulation_dt=simulation_dt)
    print(f"\nSaved log to: {log_file}")
    
    print("\n=== MuJoCo Baseline Metrics (Body Frame) ===")
    print(f"Duration: {log_data['time'][-1]:.2f}s")
    print(f"Command: vx={cmd[0]:.2f}, vy={cmd[1]:.2f}, wz={cmd[2]:.2f}")
    
    actual_vx = np.mean(log_data['base_lin_vel'][:, 0])
    actual_vy = np.mean(log_data['base_lin_vel'][:, 1])
    actual_wz = np.mean(log_data['base_ang_vel'][:, 2])
    
    print(f"\nActual velocity (mean, body frame):")
    print(f"  vx: {actual_vx:.3f} m/s (cmd: {cmd[0]:.2f}, error: {abs(actual_vx - cmd[0]):.3f})")
    print(f"  vy: {actual_vy:.3f} m/s (cmd: {cmd[1]:.2f}, error: {abs(actual_vy - cmd[1]):.3f})")
    print(f"  wz: {actual_wz:.3f} rad/s (cmd: {cmd[2]:.2f}, error: {abs(actual_wz - cmd[2]):.3f})")
    
    vx_rmse = np.sqrt(np.mean((log_data['base_lin_vel'][:, 0] - cmd[0])**2))
    vy_rmse = np.sqrt(np.mean((log_data['base_lin_vel'][:, 1] - cmd[1])**2))
    wz_rmse = np.sqrt(np.mean((log_data['base_ang_vel'][:, 2] - cmd[2])**2))
    
    print(f"\nTracking RMSE:")
    print(f"  vx: {vx_rmse:.4f}")
    print(f"  vy: {vy_rmse:.4f}")
    print(f"  wz: {wz_rmse:.4f}")
    
    print(f"\nTorque stats:")
    print(f"  Mean abs: {np.mean(np.abs(log_data['torques'])):.3f} N·m")
    print(f"  Max: {np.max(np.abs(log_data['torques'])):.3f} N·m")