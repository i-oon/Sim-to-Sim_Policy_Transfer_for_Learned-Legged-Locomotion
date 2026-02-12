"""
Simple fix: PD + Explicit Torque Clipping (no ML)
"""
import torch
import numpy as np
import mujoco
import mujoco.viewer
import yaml
import os
import time
import argparse

def quat_rotate_inverse(q, v):
    q_w, q_x, q_y, q_z = q
    q_conj = np.array([q_w, -q_x, -q_y, -q_z])
    t = 2 * np.cross(q_conj[1:], v)
    return v + q_conj[0] * t + np.cross(q_conj[1:], t)

def get_gravity_orientation(quat):
    qw, qx, qy, qz = quat
    grav = np.zeros(3)
    grav[0] = 2 * (-qz * qx + qw * qy)
    grav[1] = -2 * (qz * qy + qw * qx)
    grav[2] = 1 - 2 * (qw * qw + qz * qz)
    return grav

def run_test(config_file, use_clipping=True, torque_limit=30.0, duration=10.0, cmd=[0.5, 0.0, 0.0]):
    # Dynamic Path Resolution
    CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_FILE_DIR))
    PROJECT_ROOT = os.path.dirname(LEGGED_GYM_ROOT_DIR)

    config_path = os.path.join(LEGGED_GYM_ROOT_DIR, "deploy/deploy_mujoco/configs", config_file)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    # Fix for parallel unitree_mujoco folder
    if "unitree_mujoco" in xml_path:
        xml_path = os.path.join(PROJECT_ROOT, "unitree_mujoco", xml_path.split("unitree_mujoco/")[-1])
    
    # Clean hardcoded paths
    if "/home/drl-68/Sim-to-Sim_Policy_Transfer_for_Learned-Legged-Locomotion/" in policy_path:
        policy_path = policy_path.replace("/home/drl-68/Sim-to-Sim_Policy_Transfer_for_Learned-Legged-Locomotion/", PROJECT_ROOT + "/")

    policy = torch.jit.load(policy_path)
    policy.eval()
    
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = config["simulation_dt"]
    
    kps = np.array(config["kps"])
    kds = np.array(config["kds"])
    
    controller_name = f"PD + Clipping (±{torque_limit})" if use_clipping else "PD Only"
    print(f"\n=== Running with {controller_name} ===")
    print(f"Command: vx={cmd[0]}, vy={cmd[1]}, wz={cmd[2]}")
    
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    d.qpos[0:3] = [0, 0, 0.35]
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[7:] = default_angles
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    
    action_scale = config["action_scale"]
    decimation = config["control_decimation"]
    dt = config["simulation_dt"]
    cmd = np.array(cmd, dtype=np.float32)
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    
    action = np.zeros(12, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    
    log = {'time': [], 'vx': [], 'vy': [], 'wz': [], 'torque_max': [], 'clipped_count': []}
    
    counter = 0
    sim_time = 0.0
    fallen = False
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running() and sim_time < duration:
            step_start = time.time()
            
            # PD control
            tau_pd = kps * (target_dof_pos - d.qpos[7:]) + kds * (0 - d.qvel[6:])
            
            # Apply clipping if enabled
            if use_clipping:
                tau = np.clip(tau_pd, -torque_limit, torque_limit)
                clipped = np.sum(np.abs(tau_pd) > torque_limit)
            else:
                tau = tau_pd
                clipped = 0
            
            # Remap and apply
            d.ctrl[0:3] = tau[3:6]
            d.ctrl[3:6] = tau[0:3]
            d.ctrl[6:9] = tau[9:12]
            d.ctrl[9:12] = tau[6:9]
            
            mujoco.mj_step(m, d)
            sim_time += dt
            counter += 1
            
            # Check fallen
            quat = d.qpos[3:7]
            grav = get_gravity_orientation(quat)
            pitch = np.arcsin(np.clip(-grav[0], -1, 1))
            roll = np.arcsin(np.clip(grav[1], -1, 1))
            
            if abs(pitch) > 1.0 or abs(roll) > 1.0 or d.qpos[2] < 0.15:
                print(f"FALLEN at t={sim_time:.3f}s!")
                fallen = True
                break
            
            if counter % decimation == 0:
                base_lin_vel = quat_rotate_inverse(quat, d.qvel[0:3])
                base_ang_vel = quat_rotate_inverse(quat, d.qvel[3:6])
                gravity = get_gravity_orientation(quat)
                
                obs = np.zeros(config["num_obs"], dtype=np.float32)
                obs[0:3] = base_lin_vel * 2.0
                obs[3:6] = base_ang_vel * config["ang_vel_scale"]
                obs[6:9] = gravity
                obs[9:12] = cmd * cmd_scale
                obs[12:24] = (d.qpos[7:] - default_angles) * config["dof_pos_scale"]
                obs[24:36] = d.qvel[6:] * config["dof_vel_scale"]
                obs[36:48] = action
                
                action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
                target_dof_pos = action * action_scale + default_angles
                
                log['time'].append(sim_time)
                log['vx'].append(base_lin_vel[0])
                log['vy'].append(base_lin_vel[1])
                log['wz'].append(base_ang_vel[2])
                log['torque_max'].append(np.max(np.abs(tau)))
                log['clipped_count'].append(clipped)
            
            viewer.sync()
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    print(f"\n=== Results ({controller_name}) ===")
    print(f"Fallen: {fallen}")
    if not fallen and len(log['vx']) > 0:
        print(f"vx mean: {np.mean(log['vx']):.3f} m/s (cmd: {cmd[0]}, error: {abs(np.mean(log['vx'])-cmd[0]):.3f})")
        print(f"wz mean: {np.mean(log['wz']):.3f} rad/s (cmd: {cmd[2]})")
        print(f"Torque max: {np.max(log['torque_max']):.2f} N·m")
        if use_clipping:
            print(f"Clipped samples: {np.sum(log['clipped_count'])} / {len(log['clipped_count'])}")
    
    return log, fallen

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--no-clip", action="store_true", help="Disable clipping")
    parser.add_argument("--limit", type=float, default=30.0, help="Torque limit")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0])
    args = parser.parse_args()
    
    run_test(args.config, use_clipping=not args.no_clip, torque_limit=args.limit,
             duration=args.duration, cmd=args.cmd)
