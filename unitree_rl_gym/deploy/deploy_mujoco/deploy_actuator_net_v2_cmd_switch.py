"""
Test S2 Command Switch with ActuatorNet V2
"""
import torch
import numpy as np
import mujoco
import mujoco.viewer
import yaml
import os
import time
import pickle

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "../../../"))
LEGGED_GYM_ROOT_DIR = os.path.join(PROJECT_ROOT, "unitree_rl_gym")

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
    
    def compute_torque(self, actions, dof_pos, dof_vel):
        torques = np.zeros(12)
        for i in range(12):
            target_pos = actions[i] * ACTION_SCALE + DEFAULT_ANGLES[i]
            pos_error = target_pos - dof_pos[i]
            vel = dof_vel[i]
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

def run_cmd_switch(use_actuator_net=True, duration=6.0):
    config_path = os.path.join(LEGGED_GYM_ROOT_DIR, "deploy/deploy_mujoco/configs/go2.yaml")
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
    
    if use_actuator_net:
        actuator_net = ActuatorNetV2Controller(
            os.path.join(LEGGED_GYM_ROOT_DIR, "logs/actuator_net_v2.pt"),
            os.path.join(LEGGED_GYM_ROOT_DIR, "logs/actuator_net_v2_scaler_X.pkl"),
            os.path.join(LEGGED_GYM_ROOT_DIR, "logs/actuator_net_v2_scaler_y.pkl")
        )
        controller_name = "ActuatorNet V2"
    else:
        actuator_net = None
        controller_name = "PD Only"
    
    print(f"\n=== S2 Command Switch Test ===")
    print(f"Controller: {controller_name}")
    print(f"t < 3s: cmd = [0.4, 0, 0]")
    print(f"t >= 3s: cmd = [0.4, 0, 1.0]")
    
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    d.qpos[0:3] = [0, 0, 0.35]
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[7:] = default_angles
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    
    action_scale = config["action_scale"]
    decimation = config["control_decimation"]
    dt = config["simulation_dt"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    
    action = np.zeros(12, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    
    log = {'time': [], 'vx': [], 'wz': [], 'torque_max': [], 'pitch': [], 'roll': []}
    
    counter = 0
    sim_time = 0.0
    fallen = False
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running() and sim_time < duration:
            step_start = time.time()
            
            if sim_time < 3.0:
                cmd = np.array([0.4, 0.0, 0.0], dtype=np.float32)
            else:
                cmd = np.array([0.4, 0.0, 1.0], dtype=np.float32)
            
            if use_actuator_net:
                tau = actuator_net.compute_torque(action, d.qpos[7:], d.qvel[6:])
            else:
                tau = kps * (target_dof_pos - d.qpos[7:]) + kds * (0 - d.qvel[6:])
            
            d.ctrl[0:3] = tau[3:6]
            d.ctrl[3:6] = tau[0:3]
            d.ctrl[6:9] = tau[9:12]
            d.ctrl[9:12] = tau[6:9]
            
            mujoco.mj_step(m, d)
            sim_time += dt
            counter += 1
            
            quat = d.qpos[3:7]
            grav = get_gravity_orientation(quat)
            pitch = np.degrees(np.arcsin(np.clip(-grav[0], -1, 1)))
            roll = np.degrees(np.arcsin(np.clip(grav[1], -1, 1)))
            
            if abs(pitch) > 60 or abs(roll) > 60 or d.qpos[2] < 0.15:
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
                log['wz'].append(base_ang_vel[2])
                log['torque_max'].append(np.max(np.abs(tau)))
                log['pitch'].append(pitch)
                log['roll'].append(roll)
            
            viewer.sync()
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    print(f"\n=== Results ({controller_name}) ===")
    print(f"Fallen: {fallen}")
    
    if not fallen and len(log['time']) > 0:
        idx_before = [i for i, t in enumerate(log['time']) if t < 3.0]
        idx_after = [i for i, t in enumerate(log['time']) if t >= 3.0]
        
        vx_before = np.mean([log['vx'][i] for i in idx_before]) if idx_before else 0
        vx_after = np.mean([log['vx'][i] for i in idx_after]) if idx_after else 0
        wz_after = np.mean([log['wz'][i] for i in idx_after]) if idx_after else 0
        
        print(f"vx (t<3s): {vx_before:.3f} m/s (cmd: 0.4, error: {abs(vx_before-0.4):.3f})")
        print(f"vx (t>=3s): {vx_after:.3f} m/s (cmd: 0.4, error: {abs(vx_after-0.4):.3f})")
        print(f"wz (t>=3s): {wz_after:.3f} rad/s (cmd: 1.0, error: {abs(wz_after-1.0):.3f})")
        print(f"Torque max: {np.max(log['torque_max']):.2f} N·m")
        print(f"Max pitch: {np.max(np.abs(log['pitch'])):.1f}°")
        print(f"Max roll: {np.max(np.abs(log['roll'])):.1f}°")
    
    return log, fallen

if __name__ == "__main__":
    import sys
    use_actuator_net = "--pd" not in sys.argv
    run_cmd_switch(use_actuator_net=use_actuator_net)
