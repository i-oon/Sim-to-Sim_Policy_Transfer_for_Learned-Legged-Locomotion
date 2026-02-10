"""
Collect actuator data from Isaac Gym in actuator_net format
Format: motorStatePos_X, motorStateVel_X, motorStateCur_X, motorAction_X
"""
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import torch
import numpy as np
import pandas as pd
import os
from datetime import datetime

def collect_actuator_data(args, duration=60.0):
    """
    Collect actuator data in actuator_net format
    """
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    dt = env_cfg.sim.dt * env_cfg.control.decimation
    num_steps = int(duration / dt)
    
    # Get default angles
    default_angles = torch.zeros(12, device=env.device)
    for i, name in enumerate(env.dof_names):
        default_angles[i] = env_cfg.init_state.default_joint_angles[name]
    
    action_scale = env_cfg.control.action_scale
    
    # Create column names for actuator_net format
    columns = ['index']
    for i in range(12):
        columns.extend([f'motorStatePos_{i}', f'motorStateVel_{i}', f'motorStateCur_{i}'])
    columns.append('time')
    columns.append('diff')
    columns.append('index2')  # duplicate index in original format
    for i in range(12):
        columns.append(f'motorAction_{i}')
    columns.extend(['time2', 'diff2'])
    
    data_rows = []
    
    # Vary commands to get diverse data
    commands_list = [
        [0.5, 0.0, 0.0],   # Forward
        [0.3, 0.3, 0.0],   # Forward + lateral
        [0.4, 0.0, 0.5],   # Forward + turn
        [0.0, 0.0, 1.0],   # Turn in place
        [0.6, 0.0, 0.0],   # Fast forward
        [0.3, -0.3, 0.0],  # Forward + lateral opposite
        [0.4, 0.0, -0.5],  # Forward + turn opposite
        [0.0, 0.0, 0.0],   # Stand still
    ]
    
    print(f"Collecting actuator data for {duration}s ({num_steps} steps)...")
    print(f"action_scale={action_scale}")
    
    for step in range(num_steps):
        # Change command every 5 seconds
        cmd_idx = (step // int(5.0 / dt)) % len(commands_list)
        cmd = torch.tensor([commands_list[cmd_idx]], device=env.device)
        env.commands[:, :3] = cmd
        
        # Get action from policy
        actions = policy(obs.detach())
        
        # Compute desired position (target for PD controller)
        desired_pos = (actions * action_scale + default_angles)[0].detach().cpu().numpy()
        
        # Get current state
        current_pos = env.dof_pos[0].detach().cpu().numpy()
        current_vel = env.dof_vel[0].detach().cpu().numpy()
        
        # Step environment
        obs, _, _, _, _ = env.step(actions.detach())
        
        # Get actual torque (after step)
        torque = env.torques[0].detach().cpu().numpy()
        
        # Build row in actuator_net format
        timestamp = int(step * dt * 1e9)  # nanoseconds
        row = [step]
        for i in range(12):
            row.extend([current_pos[i], current_vel[i], torque[i]])
        row.extend([timestamp, 0, step])
        for i in range(12):
            row.append(desired_pos[i])
        row.extend([timestamp, 0])
        
        data_rows.append(row)
        
        if step % 500 == 0:
            print(f"  Step {step}/{num_steps}, cmd={commands_list[cmd_idx]}")
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=columns)
    
    # Save to actuator_net resources folder
    save_path = os.path.expanduser("~/6619_ws/actuator_net/app/resources/actuator_data.csv")
    df.to_csv(save_path, index=False)
    
    print(f"\nSaved {len(df)} samples to: {save_path}")
    print(f"Columns: {len(columns)}")
    print(f"\nTorque stats:")
    for i in range(12):
        col = f'motorStateCur_{i}'
        print(f"  Motor {i}: [{df[col].min():.2f}, {df[col].max():.2f}] N·m, mean={df[col].mean():.2f}")
    
    return save_path

if __name__ == "__main__":
    args = get_args()
    collect_actuator_data(args, duration=600.0)