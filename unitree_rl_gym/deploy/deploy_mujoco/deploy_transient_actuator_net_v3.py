"""
Transient Response Analysis - ActuatorNet V3 (Hwangbo-style)
Input: [pos_error, pos_error_t-1, pos_error_t-2, vel, vel_t-1, vel_t-2]
"""
import torch
import numpy as np
import mujoco
import mujoco.viewer
import yaml
import os
import time
import json
import pickle

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "../../../"))
LEGGED_GYM_ROOT_DIR = os.path.join(PROJECT_ROOT, "unitree_rl_gym")
ACTUATOR_NET_DIR = os.path.join(PROJECT_ROOT, "actuator_net/app/resources")

class ActuatorNetV3Controller:
    """ActuatorNet V3 with 6-feature input and history buffer"""
    
    def __init__(self):
        # Load model
        self.model = torch.load(os.path.join(ACTUATOR_NET_DIR, "actuator.pth"))
        self.model.eval()
        
        # Load scaler
        with open(os.path.join(ACTUATOR_NET_DIR, "scaler.pkl"), 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
        
        # History buffers for each joint
        self.pos_error_hist = np.zeros((12, 3))  # [joint, time] 
        self.vel_hist = np.zeros((12, 3))
        
        print(f"Loaded ActuatorNet V3 (Hwangbo-style)")
        print(f"  Input features: {self.scaler.n_features_in_}")
    
    def reset(self):
        """Reset history buffers"""
        self.pos_error_hist = np.zeros((12, 3))
        self.vel_hist = np.zeros((12, 3))
    
    def compute_torque(self, pos_error, vel):
        """Compute torques using ActuatorNet V3 with history"""
        torques = np.zeros(12)
        
        for i in range(12):
            # Update history (shift and add new)
            self.pos_error_hist[i] = np.roll(self.pos_error_hist[i], 1)
            self.pos_error_hist[i, 0] = pos_error[i]
            
            self.vel_hist[i] = np.roll(self.vel_hist[i], 1)
            self.vel_hist[i, 0] = vel[i]
            
            # Build 6-feature input
            features = np.array([[
                self.pos_error_hist[i, 0],  # pos_error
                self.pos_error_hist[i, 1],  # pos_error_t-1
                self.pos_error_hist[i, 2],  # pos_error_t-2
                self.vel_hist[i, 0],        # vel
                self.vel_hist[i, 1],        # vel_t-1
                self.vel_hist[i, 2],        # vel_t-2
            ]])
            
            # Scale and predict
            features_scaled = features  # No scaling needed
            with torch.no_grad():
                torque = self.model(torch.FloatTensor(features_scaled)).numpy()[0, 0]
            
            torques[i] = torque
        
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


def compute_transient_metrics(log, switch_time=3.0, target_before=None, target_after=None):
    """Compute transient response metrics"""
    t = np.array(log['time'])
    metrics = {}
    
    for var_name, target_b, target_a in [('vx', target_before[0], target_after[0]),
                                          ('wz', target_before[2], target_after[2])]:
        data = np.array(log[var_name])
        switch_idx = np.argmin(np.abs(t - switch_time))
        
        before_data = data[:switch_idx]
        before_mean = np.mean(before_data[-50:]) if len(before_data) > 50 else np.mean(before_data)
        before_std = np.std(before_data[-50:]) if len(before_data) > 50 else np.std(before_data)
        
        after_data = data[switch_idx:]
        after_time = t[switch_idx:] - switch_time
        
        if len(after_data) < 10:
            continue
        
        if target_a > target_b:
            peak_idx = np.argmax(after_data)
        else:
            peak_idx = np.argmin(after_data)
        peak_val = after_data[peak_idx]
        time_to_peak = after_time[peak_idx]
        
        final_val = np.mean(after_data[-50:]) if len(after_data) > 50 else np.mean(after_data[-10:])
        
        if abs(target_a - before_mean) > 0.01:
            overshoot = (peak_val - target_a) / (target_a - before_mean) * 100
        else:
            overshoot = 0
        
        rise_time = None
        if abs(final_val - before_mean) > 0.01:
            val_10 = before_mean + 0.1 * (final_val - before_mean)
            val_90 = before_mean + 0.9 * (final_val - before_mean)
            try:
                if final_val > before_mean:
                    idx_10 = np.where(after_data >= val_10)[0][0]
                    idx_90 = np.where(after_data >= val_90)[0][0]
                else:
                    idx_10 = np.where(after_data <= val_10)[0][0]
                    idx_90 = np.where(after_data <= val_90)[0][0]
                rise_time = after_time[idx_90] - after_time[idx_10]
            except:
                pass
        
        settling_time = None
        tolerance = 0.05 * abs(final_val - before_mean) if abs(final_val - before_mean) > 0.01 else 0.01
        settled = np.abs(after_data - final_val) < tolerance
        if np.any(settled):
            outside = np.where(~settled)[0]
            if len(outside) > 0 and outside[-1] < len(after_time) - 1:
                settling_time = after_time[outside[-1]]
        
        metrics[var_name] = {
            'before_mean': float(before_mean),
            'before_std': float(before_std),
            'peak_value': float(peak_val),
            'time_to_peak': float(time_to_peak),
            'overshoot_pct': float(overshoot),
            'rise_time': float(rise_time) if rise_time else None,
            'settling_time': float(settling_time) if settling_time else None,
            'final_value': float(final_val),
            'ss_error': float(abs(final_val - target_a)),
            'target': float(target_a)
        }
    
    # Torque metrics
    torque_data = np.array(log['torque_max'])
    switch_idx = np.argmin(np.abs(t - switch_time))
    after_torque = torque_data[switch_idx:]
    metrics['torque'] = {
        'peak': float(np.max(after_torque)),
        'mean_transient': float(np.mean(after_torque[:50])) if len(after_torque) > 50 else float(np.mean(after_torque)),
        'mean_steady': float(np.mean(after_torque[-50:])) if len(after_torque) > 50 else float(np.mean(after_torque))
    }
    
    # Pitch/Roll
    for angle_name in ['pitch', 'roll']:
        angle_data = np.array(log[angle_name])
        after_angle = angle_data[switch_idx:]
        metrics[angle_name] = {
            'peak': float(np.max(np.abs(after_angle))),
            'mean_transient': float(np.mean(np.abs(after_angle[:50]))) if len(after_angle) > 50 else float(np.mean(np.abs(after_angle)))
        }
    
    return metrics


def run_transient_test(scenario='S2', duration=6.0, headless=False):
    """Run transient test with ActuatorNet V3"""
    
    scenarios = {
        'S1': {'before': [0.6, 0.0, 0.0], 'after': [0.0, 0.0, 0.0], 'name': 'Stop'},
        'S2': {'before': [0.4, 0.0, 0.0], 'after': [0.4, 0.0, 1.0], 'name': 'Turn'},
        'S3': {'before': [0.3, 0.3, 0.0], 'after': [0.3, -0.3, 0.0], 'name': 'Lateral'}
    }
    
    scn = scenarios[scenario]
    cmd_before = np.array(scn['before'], dtype=np.float32)
    cmd_after = np.array(scn['after'], dtype=np.float32)
    
    config_path = os.path.join(LEGGED_GYM_ROOT_DIR, "deploy/deploy_mujoco/configs/go2.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    policy = torch.jit.load(policy_path)
    policy.eval()
    
    m = mujoco.MjModel.from_xml_path(config["xml_path"])
    d = mujoco.MjData(m)
    m.opt.timestep = config["simulation_dt"]
    
    actuator_net = ActuatorNetV3Controller()
    
    print(f"\n=== {scenario} {scn['name']} Transient Analysis ===")
    print(f"Controller: ActuatorNet V3 (Hwangbo-style)")
    print(f"Before t=3s: cmd = {scn['before']}")
    print(f"After t>=3s: cmd = {scn['after']}")
    
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
    
    log = {
        'time': [], 'vx': [], 'vy': [], 'wz': [],
        'torque_mean': [], 'torque_max': [],
        'pitch': [], 'roll': [], 'height': [],
        'cmd_vx': [], 'cmd_wz': []
    }
    
    counter = 0
    sim_time = 0.0
    fallen = False
    switch_time = 3.0
    
    def step_sim():
        nonlocal sim_time, counter, fallen, action, target_dof_pos
        
        if sim_time < switch_time:
            cmd = cmd_before.copy()
        else:
            cmd = cmd_after.copy()
        
        pos_error = target_dof_pos - d.qpos[7:]
        vel = d.qvel[6:]
        tau = actuator_net.compute_torque(pos_error, vel)
        
        # Remap to MuJoCo
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
            return False
        
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
            log['torque_mean'].append(np.mean(np.abs(tau)))
            log['torque_max'].append(np.max(np.abs(tau)))
            log['pitch'].append(pitch)
            log['roll'].append(roll)
            log['height'].append(d.qpos[2])
            log['cmd_vx'].append(cmd[0])
            log['cmd_wz'].append(cmd[2])
        
        return True
    
    if headless:
        while sim_time < duration and not fallen:
            if not step_sim():
                break
    else:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running() and sim_time < duration and not fallen:
                step_start = time.time()
                if not step_sim():
                    break
                viewer.sync()
                elapsed = time.time() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
    
    if not fallen:
        metrics = compute_transient_metrics(log, switch_time, scn['before'], scn['after'])
        metrics['fallen'] = False
    else:
        metrics = {'fallen': True, 'fall_time': sim_time}
    
    metrics['scenario'] = scenario
    metrics['controller'] = 'ActuatorNet V3'
    
    return log, metrics


def print_report(metrics):
    print(f"\n{'='*60}")
    print(f"TRANSIENT RESPONSE - {metrics['scenario']} {metrics['controller']}")
    print(f"{'='*60}")
    
    if metrics.get('fallen'):
        print(f"❌ FALLEN at t={metrics.get('fall_time', 'N/A'):.3f}s")
        return
    
    for var in ['vx', 'wz']:
        if var in metrics:
            m = metrics[var]
            print(f"\n{var.upper()} Response:")
            print(f"  Target: {m['target']:.3f}")
            print(f"  Before: {m['before_mean']:.3f} ± {m['before_std']:.3f}")
            print(f"  Final:  {m['final_value']:.3f} (error: {m['ss_error']:.3f})")
            print(f"  Overshoot: {m['overshoot_pct']:.1f}%")
            if m['rise_time']:
                print(f"  Rise time: {m['rise_time']*1000:.1f} ms")
            if m['settling_time']:
                print(f"  Settling time: {m['settling_time']*1000:.1f} ms")
    
    print(f"\nTorque:")
    print(f"  Peak: {metrics['torque']['peak']:.2f} N·m")
    
    print(f"\nStability:")
    print(f"  Peak pitch: {metrics['pitch']['peak']:.1f}°")
    print(f"  Peak roll: {metrics['roll']['peak']:.1f}°")


if __name__ == "__main__":
    import sys
    
    scenario = sys.argv[1] if len(sys.argv) > 1 else 'S1'
    headless = '--headless' in sys.argv
    
    log, metrics = run_transient_test(scenario, headless=headless)
    print_report(metrics)
    
    # Save
    save_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs/transient_analysis")
    os.makedirs(save_dir, exist_ok=True)
    
    np.savez(os.path.join(save_dir, f"{scenario}_go2_actuatornetv3_log.npz"), **log)
    with open(os.path.join(save_dir, f"{scenario}_go2_actuatornetv3_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nSaved to: {save_dir}")
