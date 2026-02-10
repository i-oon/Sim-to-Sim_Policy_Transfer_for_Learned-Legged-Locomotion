from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from isaacgym.torch_utils import quat_apply

import numpy as np
import os
from datetime import datetime
import torch

# Scenarios now use heading for angular velocity control
# For S2_turn: we set target_heading_delta to control turning
SCENARIOS = {
    'S1_stop': {'cmd_before': [0.6, 0.0, 0.0], 'cmd_after': [0.0, 0.0, 0.0], 'heading_delta': 0.0},
    'S2_turn': {'cmd_before': [0.4, 0.0, 0.0], 'cmd_after': [0.4, 0.0, 0.0], 'heading_delta': np.pi},  # Turn 180 degrees
    'S3_lateral': {'cmd_before': [0.3, 0.3, 0.0], 'cmd_after': [0.3, -0.3, 0.0], 'heading_delta': 0.0},
}

def get_current_heading(env):
    """Get current robot heading from base orientation"""
    forward = quat_apply(env.base_quat, env.forward_vec)
    heading = torch.atan2(forward[:, 1], forward[:, 0])
    return heading

def play_cmd_switch(args, scenario_name, switch_time=3.0, duration=6.0):
    scenario = SCENARIOS[scenario_name]
    cmd_before = torch.tensor([scenario['cmd_before']], device='cuda')
    cmd_after = torch.tensor([scenario['cmd_after']], device='cuda')
    heading_delta = scenario['heading_delta']
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    # Check if heading_command is enabled
    use_heading_cmd = env_cfg.commands.heading_command
    print(f"heading_command enabled: {use_heading_cmd}")
    
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    dt = env_cfg.sim.dt * env_cfg.control.decimation
    num_steps = int(duration / dt)
    switch_step = int(switch_time / dt)
    
    log_data = {
        'time': [], 'base_lin_vel': [], 'base_ang_vel': [],
        'torques': [], 'cmd': [], 'cmd_wz_computed': [],
        'roll': [], 'pitch': [], 'heading': [],
    }
    
    print(f"=== Isaac Gym: {scenario_name} ===")
    print(f"Before t={switch_time}s: cmd = {scenario['cmd_before']}")
    print(f"After  t={switch_time}s: cmd = {scenario['cmd_after']}, heading_delta = {np.degrees(heading_delta):.0f}°")
    
    # Get initial heading for reference
    initial_heading = None
    target_heading = None
    
    for i in range(num_steps):
        current_heading = get_current_heading(env)
        
        if i == 0:
            initial_heading = current_heading.clone()
            # Set initial target heading = current heading (no turning)
            target_heading = initial_heading.clone()
        
        # Switch command at switch_time
        if i < switch_step:
            cmd = cmd_before
            # Before switch: maintain current heading
            if use_heading_cmd:
                env.commands[:, 3] = current_heading  # Target = current (no turn)
        else:
            cmd = cmd_after
            # After switch: set target heading with delta
            if use_heading_cmd and heading_delta != 0.0:
                if i == switch_step:
                    # Set target heading at switch moment
                    target_heading = current_heading + heading_delta
                env.commands[:, 3] = target_heading
        
        # Set linear velocity commands
        env.commands[:, 0] = cmd[0, 0]  # vx
        env.commands[:, 1] = cmd[0, 1]  # vy
        
        # If not using heading command, set wz directly (for S1, S3)
        if not use_heading_cmd or heading_delta == 0.0:
            env.commands[:, 2] = cmd[0, 2]  # wz
        
        actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())
        
        # Log
        log_data['time'].append(i * dt)
        log_data['base_lin_vel'].append(env.base_lin_vel[0].detach().cpu().numpy())
        log_data['base_ang_vel'].append(env.base_ang_vel[0].detach().cpu().numpy())
        log_data['torques'].append(env.torques[0].detach().cpu().numpy())
        log_data['cmd'].append(np.array([env.commands[0, 0].item(), 
                                          env.commands[0, 1].item(), 
                                          env.commands[0, 2].item()]))
        log_data['cmd_wz_computed'].append(env.commands[0, 2].item())
        log_data['heading'].append(current_heading[0].item())
        
        # Roll/pitch from projected gravity
        grav = env.projected_gravity[0].detach().cpu().numpy()
        roll = np.arctan2(grav[1], grav[2])
        pitch = np.arctan2(-grav[0], np.sqrt(grav[1]**2 + grav[2]**2))
        log_data['roll'].append(roll)
        log_data['pitch'].append(pitch)
    
    # Save
    for key in log_data:
        log_data[key] = np.array(log_data[key])
    
    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs/sim2sim/cmd_switch")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"isaacgym_{scenario_name}_{timestamp}.npz")
    
    np.savez(log_file, **log_data, switch_time=switch_time,
             cmd_before=scenario['cmd_before'], cmd_after=scenario['cmd_after'],
             heading_delta=heading_delta)
    print(f"Saved: {log_file}")
    
    # Metrics
    times = log_data['time']
    vx = log_data['base_lin_vel'][:, 0]
    wz = log_data['base_ang_vel'][:, 2]
    torques = log_data['torques']
    
    ss_mask = (times >= switch_time - 0.5) & (times < switch_time)
    tr_mask = (times >= switch_time) & (times <= switch_time + 1.5)
    
    print(f"\n--- Steady-State (before switch) ---")
    print(f"  vx: {np.mean(vx[ss_mask]):.3f} m/s")
    print(f"  wz: {np.mean(wz[ss_mask]):.3f} rad/s")
    
    print(f"\n--- Transient (after switch) ---")
    print(f"  vx mean: {np.mean(vx[tr_mask]):.3f} m/s")
    print(f"  wz mean: {np.mean(wz[tr_mask]):.3f} rad/s")
    print(f"  Peak |torque|: {np.max(np.abs(torques[tr_mask])):.3f} N·m")
    
    print(f"  Max |roll|: {np.degrees(np.max(np.abs(log_data['roll'][tr_mask]))):.1f}°")
    print(f"  Max |pitch|: {np.degrees(np.max(np.abs(log_data['pitch'][tr_mask]))):.1f}°")
    
    # Heading change
    heading_before = np.mean(log_data['heading'][ss_mask])
    heading_after = log_data['heading'][-1]
    print(f"\n--- Heading ---")
    print(f"  Before: {np.degrees(heading_before):.1f}°")
    print(f"  After: {np.degrees(heading_after):.1f}°")
    print(f"  Change: {np.degrees(heading_after - heading_before):.1f}°")

if __name__ == '__main__':
    import sys
    
    # Extract --scenario before get_args() parses
    scenario = "S1_stop"
    new_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--scenario":
            scenario = sys.argv[i+1]
            i += 2
        else:
            new_argv.append(sys.argv[i])
            i += 1
    sys.argv = new_argv
    
    args = get_args()
    play_cmd_switch(args, scenario)