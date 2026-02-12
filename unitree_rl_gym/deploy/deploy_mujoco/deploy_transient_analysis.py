"""
Transient Response Analysis - Collect detailed time-series data
"""
import torch
import numpy as np
import mujoco
import mujoco.viewer
import yaml
import os
import time
import json

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "../../../"))
LEGGED_GYM_ROOT_DIR = os.path.join(PROJECT_ROOT, "unitree_rl_gym")

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
        
        # Find switch index
        switch_idx = np.argmin(np.abs(t - switch_time))
        
        # Before switch (steady-state)
        before_data = data[:switch_idx]
        before_mean = np.mean(before_data[-50:]) if len(before_data) > 50 else np.mean(before_data)
        before_std = np.std(before_data[-50:]) if len(before_data) > 50 else np.std(before_data)
        
        # After switch
        after_data = data[switch_idx:]
        after_time = t[switch_idx:] - switch_time
        
        if len(after_data) < 10:
            continue
            
        # Peak value and time to peak
        if target_a > target_b:  # Increasing
            peak_idx = np.argmax(after_data)
            peak_val = after_data[peak_idx]
        else:  # Decreasing
            peak_idx = np.argmin(after_data)
            peak_val = after_data[peak_idx]
        
        time_to_peak = after_time[peak_idx]
        
        # Overshoot
        final_val = np.mean(after_data[-50:]) if len(after_data) > 50 else np.mean(after_data[-10:])
        if abs(target_a - before_mean) > 0.01:
            overshoot = (peak_val - target_a) / (target_a - before_mean) * 100
        else:
            overshoot = 0
        
        # Rise time (10% to 90% of final value)
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
                rise_time = np.nan
        else:
            rise_time = np.nan
        
        # Settling time (within 5% of final value)
        tolerance = 0.05 * abs(final_val - before_mean) if abs(final_val - before_mean) > 0.01 else 0.01
        settled = np.abs(after_data - final_val) < tolerance
        if np.any(settled):
            # Find last time it went outside tolerance
            outside = np.where(~settled)[0]
            if len(outside) > 0:
                settling_time = after_time[outside[-1]] if outside[-1] < len(after_time)-1 else np.nan
            else:
                settling_time = 0
        else:
            settling_time = np.nan
        
        # Steady-state error
        ss_error = abs(final_val - target_a)
        
        metrics[var_name] = {
            'before_mean': float(before_mean),
            'before_std': float(before_std),
            'peak_value': float(peak_val),
            'time_to_peak': float(time_to_peak),
            'overshoot_pct': float(overshoot),
            'rise_time': float(rise_time) if not np.isnan(rise_time) else None,
            'settling_time': float(settling_time) if not np.isnan(settling_time) else None,
            'final_value': float(final_val),
            'ss_error': float(ss_error),
            'target': float(target_a)
        }
    
    # Torque metrics during transient
    torque_data = np.array(log['torque_max'])
    after_torque = torque_data[switch_idx:]
    metrics['torque'] = {
        'peak': float(np.max(after_torque)),
        'mean_transient': float(np.mean(after_torque[:50])) if len(after_torque) > 50 else float(np.mean(after_torque)),
        'mean_steady': float(np.mean(after_torque[-50:])) if len(after_torque) > 50 else float(np.mean(after_torque))
    }
    
    # Pitch/Roll metrics
    for angle_name in ['pitch', 'roll']:
        angle_data = np.array(log[angle_name])
        after_angle = angle_data[switch_idx:]
        metrics[angle_name] = {
            'peak': float(np.max(np.abs(after_angle))),
            'mean_transient': float(np.mean(np.abs(after_angle[:50]))) if len(after_angle) > 50 else float(np.mean(np.abs(after_angle))),
            'settling_value': float(np.mean(np.abs(after_angle[-50:]))) if len(after_angle) > 50 else float(np.mean(np.abs(after_angle)))
        }
    
    return metrics

def run_transient_test(config_name, scenario='S2', duration=6.0, use_residual=False):
    """Run command switch test and collect detailed data"""
    
    # Scenario definitions
    scenarios = {
        'S1': {'before': [0.6, 0.0, 0.0], 'after': [0.0, 0.0, 0.0], 'name': 'Stop'},
        'S2': {'before': [0.4, 0.0, 0.0], 'after': [0.4, 0.0, 1.0], 'name': 'Turn'},
        'S3': {'before': [0.3, 0.3, 0.0], 'after': [0.3, -0.3, 0.0], 'name': 'Lateral'}
    }
    
    scn = scenarios[scenario]
    cmd_before = np.array(scn['before'], dtype=np.float32)
    cmd_after = np.array(scn['after'], dtype=np.float32)
    
    config_path = os.path.join(LEGGED_GYM_ROOT_DIR, f"deploy/deploy_mujoco/configs/{config_name}")
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
    
    # Load residual net if needed
    if use_residual:
        import pickle
        residual_net = torch.jit.load(os.path.join(LEGGED_GYM_ROOT_DIR, "logs/residual_net.pt"))
        residual_net.eval()
        with open(os.path.join(LEGGED_GYM_ROOT_DIR, "logs/residual_scaler.pkl"), 'rb') as f:
            residual_scaler = pickle.load(f)
        controller_name = "PD + Residual"
    else:
        residual_net = None
        residual_scaler = None
        controller_name = "PD Only"
    
    print(f"\n=== {scenario} {scn['name']} Transient Analysis ===")
    print(f"Controller: {controller_name}")
    print(f"Config: {config_name}")
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
    
    # Detailed logging
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
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running() and sim_time < duration:
            step_start = time.time()
            
            # Command switch
            if sim_time < switch_time:
                cmd = cmd_before.copy()
            else:
                cmd = cmd_after.copy()
            
            # PD control
            pos_error = target_dof_pos - d.qpos[7:]
            vel = d.qvel[6:]
            tau = kps * pos_error + kds * (0 - vel)
            
            # Add residual if enabled
            if use_residual and residual_net is not None:
                for i in range(12):
                    features = np.array([[pos_error[i], vel[i]]])
                    features_scaled = residual_scaler.transform(features)
                    with torch.no_grad():
                        tau[i] += residual_net(torch.FloatTensor(features_scaled)).item()
            
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
                
                # Log everything
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
            
            viewer.sync()
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    # Compute transient metrics
    if not fallen:
        metrics = compute_transient_metrics(log, switch_time, scn['before'], scn['after'])
        metrics['fallen'] = False
    else:
        metrics = {'fallen': True, 'fall_time': sim_time}
    
    metrics['scenario'] = scenario
    metrics['controller'] = controller_name
    metrics['config'] = config_name
    
    return log, metrics

def print_transient_report(metrics):
    """Print formatted transient analysis report"""
    print(f"\n{'='*60}")
    print(f"TRANSIENT RESPONSE ANALYSIS - {metrics['scenario']} {metrics['controller']}")
    print(f"{'='*60}")
    
    if metrics['fallen']:
        print(f"❌ ROBOT FALLEN at t={metrics.get('fall_time', 'N/A'):.3f}s")
        return
    
    for var in ['vx', 'wz']:
        if var in metrics:
            m = metrics[var]
            print(f"\n{var.upper()} Response:")
            print(f"  Target: {m['target']:.3f}")
            print(f"  Before: {m['before_mean']:.3f} ± {m['before_std']:.3f}")
            print(f"  Final:  {m['final_value']:.3f} (error: {m['ss_error']:.3f})")
            print(f"  Peak:   {m['peak_value']:.3f} at t={m['time_to_peak']:.3f}s")
            print(f"  Overshoot: {m['overshoot_pct']:.1f}%")
            if m['rise_time']:
                print(f"  Rise time (10-90%): {m['rise_time']*1000:.1f} ms")
            if m['settling_time']:
                print(f"  Settling time (5%): {m['settling_time']*1000:.1f} ms")
    
    print(f"\nTorque:")
    print(f"  Peak: {metrics['torque']['peak']:.2f} N·m")
    print(f"  Transient mean: {metrics['torque']['mean_transient']:.2f} N·m")
    print(f"  Steady mean: {metrics['torque']['mean_steady']:.2f} N·m")
    
    print(f"\nBody Stability:")
    print(f"  Peak pitch: {metrics['pitch']['peak']:.1f}°")
    print(f"  Peak roll: {metrics['roll']['peak']:.1f}°")

if __name__ == "__main__":
    import sys
    
    scenario = sys.argv[1] if len(sys.argv) > 1 else 'S2'
    config = sys.argv[2] if len(sys.argv) > 2 else 'go2.yaml'
    use_residual = '--residual' in sys.argv
    
    log, metrics = run_transient_test(config, scenario, use_residual=use_residual)
    print_transient_report(metrics)
    
    # Save results
    save_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs/transient_analysis")
    os.makedirs(save_dir, exist_ok=True)
    
    suffix = "residual" if use_residual else "pd"
    config_name = config.replace('.yaml', '')
    
    np.savez(os.path.join(save_dir, f"{scenario}_{config_name}_{suffix}_log.npz"), **log)
    with open(os.path.join(save_dir, f"{scenario}_{config_name}_{suffix}_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nSaved to: {save_dir}")
