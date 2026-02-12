"""
Test Command Switch with RMA/DR-trained Policy
Supports multiple scenarios: Stop, Turn, and Lateral movements.
Includes transient metrics calculation and logging.
"""
import torch
import numpy as np
import mujoco
import mujoco.viewer
import yaml
import os
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

# --- SCENARIOS Dictionary ---
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

def run_cmd_switch(config_name, scenario_key='S2_turn', duration=6.0, switch_time=3.0,
                   save_log=True, no_viewer=False):
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
    
    # Get Scenario Details
    scenario = SCENARIOS.get(scenario_key, SCENARIOS['S2_turn'])
    cmd_before = np.array(scenario['cmd_before'], dtype=np.float32)
    cmd_after = np.array(scenario['cmd_after'], dtype=np.float32)

    print(f"\n=== RMA Command Switch Test: {scenario['name']} ===")
    print(f"Config: {config_name}")
    print(f"t < {switch_time}s: cmd = {cmd_before}")
    print(f"t >= {switch_time}s: cmd = {cmd_after}")
    
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
        nonlocal action, target_dof_pos, counter, sim_time, fallen, fall_time
        
        # Command switch logic
        cmd = cmd_before if sim_time < switch_time else cmd_after
        
        # PD control
        tau = kps * (target_dof_pos - d.qpos[7:]) + kds * (0 - d.qvel[6:])
        
        # Remap and apply (Isaac FL,FR,RL,RR -> Go2 MuJoCo FR,FL,RR,RL)
        d.ctrl[0:3] = tau[3:6]
        d.ctrl[3:6] = tau[0:3]
        d.ctrl[6:9] = tau[9:12]
        d.ctrl[9:12] = tau[6:9]
        
        mujoco.mj_step(m, d)
        sim_time = d.time
        counter += 1
        
        # Check fallen
        quat = d.qpos[3:7]
        grav = get_gravity_orientation(quat)
        pitch = np.arcsin(np.clip(-grav[0], -1.0, 1.0))
        roll = np.arcsin(np.clip(grav[1], -1.0, 1.0))
        
        if not fallen and (abs(pitch) > 1.0 or abs(roll) > 1.0 or d.qpos[2] < 0.15):
            print(f"FALLEN at t={sim_time:.3f}s!")
            fallen = True
            fall_time = sim_time
        
        if counter % decimation == 0:
            base_lin_vel = quat_rotate_inverse(quat, d.qvel[0:3])
            base_ang_vel = quat_rotate_inverse(quat, d.qvel[3:6])
            
            obs = np.zeros(config["num_obs"], dtype=np.float32)
            obs[0:3] = base_lin_vel * 2.0
            obs[3:6] = base_ang_vel * config["ang_vel_scale"]
            obs[6:9] = grav
            obs[9:12] = cmd * cmd_scale
            obs[12:24] = (d.qpos[7:] - default_angles) * config["dof_pos_scale"]
            obs[24:36] = d.qvel[6:] * config["dof_vel_scale"]
            obs[36:48] = action
            
            with torch.no_grad():
                action = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
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
            log['joint_vel'].append(d.qvel[6:].copy())
            log['torques'].append(tau.copy())
            log['actions'].append(action.copy())
            log['cmd'].append(cmd.copy())
    
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
        log_file = os.path.join(log_dir, f"rma_{scenario_key}_{timestamp}.npz")
        
        np.savez(log_file, **log,
                 scenario=scenario_key, switch_time=switch_time,
                 cmd_before=cmd_before, cmd_after=cmd_after,
                 controller="DR-trained",
                 fallen=fallen, fall_time=fall_time if fall_time else -1)
        print(f"\nSaved log to: {log_file}")
    
    # Compute and print transient metrics
    transient_results = analyze_scenario_transients(log, scenario_key, switch_time)
    print_transient_summary(transient_results, scenario_key)
    
    # Statistics
    print(f"\n=== Final Results: {scenario['name']} ===")
    print(f"Fallen: {fallen}")
    if not fallen and len(log['time']) > 0:
        idx_after = [i for i, t in enumerate(log['time']) if t >= switch_time]
        if idx_after:
            vx_final = np.mean([log['vx'][i] for i in idx_after])
            vy_final = np.mean([log['vy'][i] for i in idx_after])
            wz_final = np.mean([log['wz'][i] for i in idx_after])
            print(f"Final vx: {vx_final:.3f} (Target: {cmd_after[0]})")
            print(f"Final vy: {vy_final:.3f} (Target: {cmd_after[1]})")
            print(f"Final wz: {wz_final:.3f} (Target: {cmd_after[2]})")
            print(f"Max Torque: {np.max(log['torque_max']):.2f} N·m")
            print(f"Max Pitch/Roll: {np.max(np.abs(log['pitch'])):.1f}° / {np.max(np.abs(log['roll'])):.1f}°")
    
    return log, fallen, transient_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="go2_rma.yaml")
    parser.add_argument('--scenario', type=str, default="S2_turn", choices=list(SCENARIOS.keys()))
    parser.add_argument('--no_viewer', action='store_true', help='Run headless')
    parser.add_argument('--duration', type=float, default=6.0)
    parser.add_argument('--switch_time', type=float, default=3.0)
    args = parser.parse_args()
    
    run_cmd_switch(args.config, scenario_key=args.scenario, 
                   duration=args.duration, switch_time=args.switch_time,
                   no_viewer=args.no_viewer)