import time
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import os
from datetime import datetime
import argparse

def quat_rotate_inverse(q, v):
    q_w, q_x, q_y, q_z = q
    q_conj = np.array([q_w, -q_x, -q_y, -q_z])
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

def quat_to_euler(quat):
    """Convert quaternion (wxyz) to roll, pitch, yaw"""
    w, x, y, z = quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    return roll, pitch

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

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

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("--scenario", type=str, default="S1_stop", choices=SCENARIOS.keys())
    parser.add_argument("--switch_time", type=float, default=3.0)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--log_dir", type=str, default="logs/sim2sim/cmd_switch")
    parser.add_argument("--no_viewer", action="store_true")
    args = parser.parse_args()
    
    LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    PROJECT_ROOT = os.path.dirname(LEGGED_GYM_ROOT_DIR)

    scenario = SCENARIOS[args.scenario]
    cmd_before = np.array(scenario['cmd_before'], dtype=np.float32)
    cmd_after = np.array(scenario['cmd_after'], dtype=np.float32)
    switch_time = args.switch_time
    
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
        lin_vel_scale = 2.0

    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"mujoco_{args.scenario}_{timestamp}.npz")

    log_data = {
        'time': [], 'base_pos': [], 'base_quat': [], 'base_lin_vel': [],
        'base_ang_vel': [], 'joint_pos': [], 'joint_vel': [], 'actions': [],
        'torques': [], 'cmd': [], 'roll': [], 'pitch': [],
    }

    m = mujoco.MjModel.from_xml_path(os.path.expanduser(xml_path))
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    
    d.qpos[0:3] = [0, 0, 0.35]
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[7:] = default_angles
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)

    policy = torch.jit.load(policy_path)
    
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0
    fallen = False
    fall_time = None
    sim_time = 0.0

    def run_step():
        global action, target_dof_pos, obs, counter, fallen, fall_time, sim_time
        
        if sim_time >= switch_time:
            cmd = cmd_after
        else:
            cmd = cmd_before
        
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
            quat = d.qpos[3:7]
            
            world_lin_vel = d.qvel[0:3]
            world_ang_vel = d.qvel[3:6]
            base_lin_vel = quat_rotate_inverse(quat, world_lin_vel)
            base_ang_vel = quat_rotate_inverse(quat, world_ang_vel)
            gravity_orientation = get_gravity_orientation(quat)
            
            # Use correct euler conversion
            roll, pitch = quat_to_euler(quat)
            
            # Check if fallen using height and euler angles
            if not fallen and (abs(roll) > 1.0 or abs(pitch) > 1.0 or d.qpos[2] < 0.15):
                fallen = True
                fall_time = sim_time
                print(f"FALLEN at t={sim_time:.3f}s (roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°, h={d.qpos[2]:.3f})")
            
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
            log_data['roll'].append(roll)
            log_data['pitch'].append(pitch)

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

    if args.no_viewer:
        while sim_time < args.duration:
            run_step()
    else:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running() and sim_time < args.duration:
                step_start = time.time()
                run_step()
                viewer.sync()
                elapsed = time.time() - step_start
                if elapsed < simulation_dt:
                    time.sleep(simulation_dt - elapsed)

    for key in log_data:
        log_data[key] = np.array(log_data[key])
    
    np.savez(log_file, **log_data, 
             scenario=args.scenario, switch_time=switch_time,
             cmd_before=cmd_before, cmd_after=cmd_after,
             fallen=fallen, fall_time=fall_time if fall_time else -1)
    print(f"\nSaved log to: {log_file}")

    print(f"\n=== {scenario['name']} Metrics ===")
    
    times = log_data['time']
    vx = log_data['base_lin_vel'][:, 0]
    vy = log_data['base_lin_vel'][:, 1]
    wz = log_data['base_ang_vel'][:, 2]
    
    ss_mask = (times >= switch_time - 0.5) & (times < switch_time)
    tr_mask = (times >= switch_time) & (times <= switch_time + 1.5)
    
    print(f"\n--- Steady-State (t={switch_time-0.5:.1f} to {switch_time:.1f}s) ---")
    print(f"  vx: {np.mean(vx[ss_mask]):.3f} m/s (cmd: {cmd_before[0]:.2f})")
    print(f"  vy: {np.mean(vy[ss_mask]):.3f} m/s (cmd: {cmd_before[1]:.2f})")
    print(f"  wz: {np.mean(wz[ss_mask]):.3f} rad/s (cmd: {cmd_before[2]:.2f})")
    
    print(f"\n--- Transient Response (t={switch_time:.1f} to {switch_time+1.5:.1f}s) ---")
    print(f"  vx mean: {np.mean(vx[tr_mask]):.3f} m/s (target: {cmd_after[0]:.2f})")
    print(f"  vy mean: {np.mean(vy[tr_mask]):.3f} m/s (target: {cmd_after[1]:.2f})")
    print(f"  wz mean: {np.mean(wz[tr_mask]):.3f} rad/s (target: {cmd_after[2]:.2f})")
    
    if args.scenario == 'S1_stop':
        print(f"  vx min (undershoot): {np.min(vx[tr_mask]):.3f} m/s")
    elif args.scenario == 'S2_turn':
        print(f"  wz max: {np.max(wz[tr_mask]):.3f} rad/s (overshoot: {np.max(wz[tr_mask]) - cmd_after[2]:.3f})")
    elif args.scenario == 'S3_lateral':
        print(f"  vy min: {np.min(vy[tr_mask]):.3f} m/s (overshoot: {np.min(vy[tr_mask]) - cmd_after[1]:.3f})")
    
    print(f"\n--- Stability ---")
    print(f"  Max |roll|: {np.degrees(np.max(np.abs(log_data['roll']))):.1f}°")
    print(f"  Max |pitch|: {np.degrees(np.max(np.abs(log_data['pitch']))):.1f}°")
    print(f"  Fallen: {fallen}" + (f" at t={fall_time:.3f}s" if fallen else ""))
    
    print(f"\n--- Torque ---")
    print(f"  Mean |τ|: {np.mean(np.abs(log_data['torques'])):.3f} N·m")
    print(f"  Max |τ|: {np.max(np.abs(log_data['torques'])):.3f} N·m")
