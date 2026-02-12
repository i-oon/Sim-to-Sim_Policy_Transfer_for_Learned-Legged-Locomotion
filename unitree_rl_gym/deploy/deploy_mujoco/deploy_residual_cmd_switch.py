import torch
import numpy as np
import mujoco
import mujoco.viewer
import yaml
import os
import pickle
import time
import argparse
from datetime import datetime

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "../../../"))
LEGGED_GYM_ROOT_DIR = os.path.join(PROJECT_ROOT, "unitree_rl_gym")

# Import transient metrics calculator
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from transient_metrics import analyze_scenario_transients, print_transient_summary

SCENARIOS = {
    'S1_stop': {
        'name': 'Stop Shock',
        'cmd_before': [0.6, 0.0, 0.0],
        'cmd_after': [0.0, 0.0, 0.0],
    },
    'S2_turn': {
        'name': 'Turn Shock', 
        'cmd_before': [0.4, 0.0, 0.0],
        'cmd_after': [0.4, 0.0, 1.0],
    },
    'S3_lateral': {
        'name': 'Lateral Flip',
        'cmd_before': [0.3, 0.3, 0.0],
        'cmd_after': [0.3, -0.3, 0.0],
    },
}

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

def run_cmd_switch(scenario_key='S2_turn', use_residual=True, duration=6.0, switch_time=3.0, 
                   save_log=True, no_viewer=False):
    # Load config
    config_path = os.path.join(LEGGED_GYM_ROOT_DIR, "deploy/deploy_mujoco/configs/go2.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load Policy
    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    policy = torch.jit.load(policy_path)
    policy.eval()
    
    # Setup Mujoco
    m = mujoco.MjModel.from_xml_path(config["xml_path"])
    d = mujoco.MjData(m)
    m.opt.timestep = config["simulation_dt"]
    
    kps = np.array(config["kps"])
    kds = np.array(config["kds"])
    
    # Load residual net
    residual_net, scaler = None, None
    controller_name = "PD Only"
    if use_residual:
        res_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs/residual_net.pt")
        scaler_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs/residual_scaler.pkl")
        if os.path.exists(res_path) and os.path.exists(scaler_path):
            residual_net = torch.jit.load(res_path)
            residual_net.eval()
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            controller_name = "PD + Residual"
        else:
            print("Warning: Residual files not found. Falling back to PD Only.")
    
    # Get Scenario Info
    scenario = SCENARIOS.get(scenario_key, SCENARIOS['S2_turn'])
    cmd_before = np.array(scenario['cmd_before'], dtype=np.float32)
    cmd_after = np.array(scenario['cmd_after'], dtype=np.float32)
    
    print(f"\n=== {scenario['name']} with {controller_name} ===")
    print(f"t < {switch_time}s: cmd = {cmd_before}")
    print(f"t >= {switch_time}s: cmd = {cmd_after}")
    
    # Init robot state
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
    residuals = np.zeros(12, dtype=np.float32)
    
    # Enhanced logging
    log = {
        'time': [], 'vx': [], 'vy': [], 'wz': [], 
        'torque_max': [], 'pitch': [], 'roll': [],
        'base_pos': [], 'base_quat': [], 
        'base_lin_vel': [], 'base_ang_vel': [],
        'joint_pos': [], 'joint_vel': [], 'torques': [],
        'actions': [], 'cmd': []
    }
    
    counter = 0
    sim_time = 0.0
    fallen = False
    fall_time = None
    
    def run_step():
        nonlocal action, target_dof_pos, residuals, counter, sim_time, fallen, fall_time
        
        # 1. Command switch logic
        current_cmd = cmd_before if sim_time < switch_time else cmd_after
        
        # 2. Compute torque (PD + Residual)
        pos_error = target_dof_pos - d.qpos[7:]
        vel = d.qvel[6:]
        tau_pd = kps * pos_error + kds * (0 - vel)
        tau = tau_pd + residuals if use_residual else tau_pd
        
        # 3. Apply to Mujoco (Remap Isaac -> Go2 MuJoCo)
        d.ctrl[0:3] = tau[3:6]
        d.ctrl[3:6] = tau[0:3]
        d.ctrl[6:9] = tau[9:12]
        d.ctrl[9:12] = tau[6:9]
        
        mujoco.mj_step(m, d)
        sim_time = d.time
        
        # 4. Check stability
        quat = d.qpos[3:7]
        grav = get_gravity_orientation(quat)
        pitch = np.arcsin(np.clip(-grav[0], -1.0, 1.0))
        roll = np.arcsin(np.clip(grav[1], -1.0, 1.0))
        
        if not fallen and (abs(pitch) > 1.0 or abs(roll) > 1.0 or d.qpos[2] < 0.15):
            fallen = True
            fall_time = sim_time
            print(f"FALLEN at t={sim_time:.3f}s!")
        
        # 5. Policy & Residual Update (at decimation frequency)
        if counter % decimation == 0:
            world_lin_vel = d.qvel[0:3]
            world_ang_vel = d.qvel[3:6]
            base_lin_vel = quat_rotate_inverse(quat, world_lin_vel)
            base_ang_vel = quat_rotate_inverse(quat, world_ang_vel)
            
            # Update Residuals if used
            if use_residual and residual_net is not None:
                feat_list = []
                for i in range(12):
                    feat_list.append([pos_error[i], vel[i]])
                feats_scaled = scaler.transform(np.array(feat_list))
                with torch.no_grad():
                    residuals = residual_net(torch.FloatTensor(feats_scaled)).numpy().flatten()

            # Policy Inference
            obs = np.zeros(config["num_obs"], dtype=np.float32)
            obs[0:3] = base_lin_vel * 2.0
            obs[3:6] = base_ang_vel * config["ang_vel_scale"]
            obs[6:9] = grav
            obs[9:12] = current_cmd * cmd_scale
            obs[12:24] = (d.qpos[7:] - default_angles) * config["dof_pos_scale"]
            obs[24:36] = d.qvel[6:] * config["dof_vel_scale"]
            obs[36:48] = action
            
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            with torch.no_grad():
                action = policy(obs_tensor).numpy().squeeze()
            target_dof_pos = action * action_scale + default_angles
            
            # Enhanced logging
            log['time'].append(sim_time)
            log['vx'].append(base_lin_vel[0])
            log['vy'].append(base_lin_vel[1])
            log['wz'].append(base_ang_vel[2])
            log['torque_max'].append(np.max(np.abs(tau)))
            log['pitch'].append(np.degrees(pitch))
            log['roll'].append(np.degrees(roll))
            log['base_pos'].append(d.qpos[0:3].copy())
            log['base_quat'].append(quat.copy())
            log['base_lin_vel'].append(base_lin_vel.copy())
            log['base_ang_vel'].append(base_ang_vel.copy())
            log['joint_pos'].append(d.qpos[7:].copy())
            log['joint_vel'].append(vel.copy())
            log['torques'].append(tau.copy())
            log['actions'].append(action.copy())
            log['cmd'].append(current_cmd.copy())
        
        counter += 1
    
    if no_viewer:
        while sim_time < duration and not fallen:
            run_step()
    else:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running() and sim_time < duration and not fallen:
                step_start = time.time()
                run_step()
                viewer.sync()
                elapsed = time.time() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
    
    # Convert to numpy arrays
    for key in log:
        log[key] = np.array(log[key])
    
    # Save log file
    if save_log:
        log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs/sim2sim/cmd_switch")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        controller_tag = "residual" if use_residual else "pd"
        log_file = os.path.join(log_dir, f"{controller_tag}_{scenario_key}_{timestamp}.npz")
        
        np.savez(log_file, **log,
                 scenario=scenario_key, switch_time=switch_time,
                 cmd_before=cmd_before, cmd_after=cmd_after,
                 controller=controller_name,
                 fallen=fallen, fall_time=fall_time if fall_time else -1)
        print(f"\nSaved log to: {log_file}")
    
    # Compute and print transient metrics
    transient_results = analyze_scenario_transients(log, scenario_key, switch_time)
    print_transient_summary(transient_results, scenario_key)
    
    # Print final results
    print_results(log, fallen, scenario, controller_name, switch_time)
    
    return log, fallen, transient_results

def print_results(log, fallen, scenario, controller_name, switch_time):
    print(f"\n=== Final Results ({controller_name}) ===")
    print(f"Scenario: {scenario['name']}")
    print(f"Fallen: {fallen}")
    if not fallen and len(log['time']) > 0:
        times = np.array(log['time'])
        idx_after = np.where(times >= switch_time)[0]
        if len(idx_after) > 0:
            vx_final = np.mean(np.array(log['vx'])[idx_after])
            vy_final = np.mean(np.array(log['vy'])[idx_after])
            wz_final = np.mean(np.array(log['wz'])[idx_after])
            print(f"Final vx: {vx_final:.3f} m/s (Target: {scenario['cmd_after'][0]})")
            print(f"Final vy: {vy_final:.3f} m/s (Target: {scenario['cmd_after'][1]})")
            print(f"Final wz: {wz_final:.3f} rad/s (Target: {scenario['cmd_after'][2]})")
        print(f"Max Torque: {np.max(log['torque_max']):.2f} N·m")
        print(f"Max Pitch/Roll: {np.max(np.abs(log['pitch'])):.1f}° / {np.max(np.abs(log['roll'])):.1f}°")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='S2_turn', choices=list(SCENARIOS.keys()))
    parser.add_argument('--pd', action='store_true', help='Use PD Only (disable residual)')
    parser.add_argument('--no_viewer', action='store_true', help='Run headless')
    parser.add_argument('--duration', type=float, default=6.0)
    parser.add_argument('--switch_time', type=float, default=3.0)
    args = parser.parse_args()
    
    run_cmd_switch(scenario_key=args.scenario, use_residual=not args.pd, 
                   duration=args.duration, switch_time=args.switch_time,
                   no_viewer=args.no_viewer)