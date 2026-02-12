"""
Deploy with ActuatorNet V2 (trained on excitation data)
"""
import torch
import numpy as np
import mujoco
import mujoco.viewer
import yaml
import os
import time
import pickle

import os

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "../../../"))
LEGGED_GYM_ROOT_DIR = os.path.join(PROJECT_ROOT, "unitree_rl_gym")

# Default angles for Go2
DEFAULT_ANGLES = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])
ACTION_SCALE = 0.25

class ActuatorNetV2Controller:
    def __init__(self, model_path, scaler_X_path, scaler_y_path):
        self.model = torch.jit.load(model_path)
        self.model.eval()
        
        with open(scaler_X_path, 'rb') as f:
            self.scaler_X = pickle.load(f)
        with open(scaler_y_path, 'rb') as f:
            self.scaler_y = pickle.load(f)
        
        print(f"Loaded ActuatorNet V2")
    
    def compute_torque(self, actions, dof_pos, dof_vel):
        """Compute torque using ActuatorNet V2"""
        torques = np.zeros(12)
        
        for i in range(12):
            target_pos = actions[i] * ACTION_SCALE + DEFAULT_ANGLES[i]
            pos_error = target_pos - dof_pos[i]
            vel = dof_vel[i]
            
            # Features: [pos_error, vel, pos_error * vel]
            features = np.array([[pos_error, vel, pos_error * vel]])
            features_scaled = self.scaler_X.transform(features)
            
            with torch.no_grad():
                torque_scaled = self.model(torch.FloatTensor(features_scaled)).numpy()
            
            torques[i] = self.scaler_y.inverse_transform(torque_scaled.reshape(1, -1))[0, 0]
        
        return torques

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

def run_test(config_file, use_actuator_net=True, duration=10.0, cmd=[0.5, 0.0, 0.0]):
    config_path = os.path.join(LEGGED_GYM_ROOT_DIR, "deploy/deploy_mujoco/configs", config_file)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    policy = torch.jit.load(policy_path)
    policy.eval()
    
    m = mujoco.MjModel.from_xml_path(os.path.expanduser(config["xml_path"]))
    d = mujoco.MjData(m)
    m.opt.timestep = config["simulation_dt"]
    
    kps = np.array(config["kps"])
    kds = np.array(config["kds"])
    
    # Load ActuatorNet V2
    if use_actuator_net:
        actuator_net = ActuatorNetV2Controller(
            os.path.join(LEGGED_GYM_ROOT_DIR, "logs/actuator_net_v2.pt"),
            os.path.join(LEGGED_GYM_ROOT_DIR, "logs/actuator_net_v2_scaler_X.pkl"),
            os.path.join(LEGGED_GYM_ROOT_DIR, "logs/actuator_net_v2_scaler_y.pkl")
        )
        controller_name = "ActuatorNet V2"
    else:
        actuator_net = None
        controller_name = "PD Control"
    
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
    
    log = {'time': [], 'vx': [], 'vy': [], 'wz': [], 'torque_max': [], 'torque_mean': []}
    
    counter = 0
    sim_time = 0.0
    fallen = False
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running() and sim_time < duration:
            step_start = time.time()
            
            if use_actuator_net:
                # Use ActuatorNet V2
                tau = actuator_net.compute_torque(action, d.qpos[7:], d.qvel[6:])
            else:
                # PD control
                tau = kps * (target_dof_pos - d.qpos[7:]) + kds * (0 - d.qvel[6:])
            
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
                log['torque_mean'].append(np.mean(np.abs(tau)))
            
            viewer.sync()
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    # Results
    print(f"\n=== Results ({controller_name}) ===")
    print(f"Fallen: {fallen}")
    if not fallen and len(log['vx']) > 0:
        vx_mean = np.mean(log['vx'])
        print(f"vx mean: {vx_mean:.3f} m/s (cmd: {cmd[0]}, error: {abs(vx_mean - cmd[0]):.3f})")
        print(f"vy mean: {np.mean(log['vy']):.3f} m/s (cmd: {cmd[1]})")
        print(f"wz mean: {np.mean(log['wz']):.3f} rad/s (cmd: {cmd[2]})")
        print(f"Torque mean: {np.mean(log['torque_mean']):.3f} N·m")
        print(f"Torque max: {np.max(log['torque_max']):.3f} N·m")
    
    return log, fallen

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--pd", action="store_true", help="Use PD control instead")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0])
    args = parser.parse_args()
    
    run_test(args.config, use_actuator_net=not args.pd, duration=args.duration, cmd=args.cmd)
