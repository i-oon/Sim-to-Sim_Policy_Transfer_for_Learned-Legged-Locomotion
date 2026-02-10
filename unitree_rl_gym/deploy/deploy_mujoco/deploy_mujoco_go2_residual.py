"""
Deploy policy to MuJoCo using PD + Residual Learning
τ = τ_pd + Δτ_learned
"""
import torch
import numpy as np
import mujoco
import mujoco.viewer
import yaml
import os
import pickle
import time
import argparse

LEGGED_GYM_ROOT_DIR = os.path.expanduser("~/6619_ws/unitree_rl_gym")

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

class ResidualController:
    """PD Control + Learned Residual"""
    
    def __init__(self, kps, kds, residual_model_path, scaler_path):
        self.kps = np.array(kps)
        self.kds = np.array(kds)
        
        # Load residual model
        self.residual_net = torch.jit.load(residual_model_path)
        self.residual_net.eval()
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"Loaded residual_net from: {residual_model_path}")
    
    def compute_torque(self, desired_pos, current_pos, velocity):
        # PD torque
        tau_pd = self.kps * (desired_pos - current_pos) + self.kds * (0 - velocity)
        
        # Residual prediction for each joint
        residuals = np.zeros(12)
        for i in range(12):
            pos_error = desired_pos[i] - current_pos[i]
            features = np.array([[pos_error, velocity[i]]])
            features_scaled = self.scaler.transform(features)
            
            with torch.no_grad():
                delta_tau = self.residual_net(torch.FloatTensor(features_scaled)).item()
            
            residuals[i] = delta_tau
        
        # Final torque = PD + residual
        tau = tau_pd + residuals
        return tau, tau_pd, residuals

class PDController:
    """Standard PD control"""
    
    def __init__(self, kps, kds):
        self.kps = np.array(kps)
        self.kds = np.array(kds)
    
    def compute_torque(self, desired_pos, current_pos, velocity):
        tau = self.kps * (desired_pos - current_pos) + self.kds * (0 - velocity)
        return tau, tau, np.zeros(12)

def run_deployment(config_file, use_residual=True, duration=10.0, cmd=[0.5, 0.0, 0.0]):
    config_path = os.path.join(LEGGED_GYM_ROOT_DIR, "deploy/deploy_mujoco/configs", config_file)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path = config["xml_path"]
    
    policy = torch.jit.load(policy_path)
    policy.eval()
    
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = config["simulation_dt"]
    
    if use_residual:
        controller = ResidualController(
            config["kps"], config["kds"],
            os.path.join(LEGGED_GYM_ROOT_DIR, "logs/residual_net.pt"),
            os.path.join(LEGGED_GYM_ROOT_DIR, "logs/residual_scaler.pkl")
        )
        controller_name = "PD + Residual"
    else:
        controller = PDController(config["kps"], config["kds"])
        controller_name = "PD Only"
    
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
    
    lin_vel_scale = 2.0
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    num_obs = config["num_obs"]
    
    action = np.zeros(12, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    
    log = {'time': [], 'vx': [], 'vy': [], 'wz': [], 
           'torque_mean': [], 'torque_max': [], 'residual_mean': []}
    
    counter = 0
    sim_time = 0.0
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running() and sim_time < duration:
            step_start = time.time()
            
            tau, tau_pd, residuals = controller.compute_torque(
                target_dof_pos, d.qpos[7:], d.qvel[6:]
            )
            
            # Remap to MuJoCo ctrl order
            d.ctrl[0:3] = tau[3:6]
            d.ctrl[3:6] = tau[0:3]
            d.ctrl[6:9] = tau[9:12]
            d.ctrl[9:12] = tau[6:9]
            
            mujoco.mj_step(m, d)
            sim_time += dt
            counter += 1
            
            if counter % decimation == 0:
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                
                world_lin_vel = d.qvel[0:3]
                world_ang_vel = d.qvel[3:6]
                base_lin_vel = quat_rotate_inverse(quat, world_lin_vel)
                base_ang_vel = quat_rotate_inverse(quat, world_ang_vel)
                gravity = get_gravity_orientation(quat)
                
                obs = np.zeros(num_obs, dtype=np.float32)
                obs[0:3] = base_lin_vel * lin_vel_scale
                obs[3:6] = base_ang_vel * ang_vel_scale
                obs[6:9] = gravity
                obs[9:12] = cmd * cmd_scale
                obs[12:24] = (qj - default_angles) * dof_pos_scale
                obs[24:36] = dqj * dof_vel_scale
                obs[36:48] = action
                
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()
                target_dof_pos = action * action_scale + default_angles
                
                log['time'].append(sim_time)
                log['vx'].append(base_lin_vel[0])
                log['vy'].append(base_lin_vel[1])
                log['wz'].append(base_ang_vel[2])
                log['torque_mean'].append(np.mean(np.abs(tau)))
                log['torque_max'].append(np.max(np.abs(tau)))
                log['residual_mean'].append(np.mean(np.abs(residuals)))
            
            viewer.sync()
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    print(f"\n=== Results ({controller_name}) ===")
    print(f"vx mean: {np.mean(log['vx']):.3f} m/s (cmd: {cmd[0]}, error: {abs(np.mean(log['vx'])-cmd[0]):.3f})")
    print(f"vy mean: {np.mean(log['vy']):.3f} m/s (cmd: {cmd[1]})")
    print(f"wz mean: {np.mean(log['wz']):.3f} rad/s (cmd: {cmd[2]})")
    print(f"Torque mean: {np.mean(log['torque_mean']):.3f} N·m")
    print(f"Torque max: {np.max(log['torque_max']):.3f} N·m")
    print(f"Residual mean: {np.mean(log['residual_mean']):.3f} N·m")
    
    return log

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--pd", action="store_true", help="Use PD only (no residual)")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0])
    args = parser.parse_args()
    
    run_deployment(args.config, use_residual=not args.pd, duration=args.duration, cmd=args.cmd)
