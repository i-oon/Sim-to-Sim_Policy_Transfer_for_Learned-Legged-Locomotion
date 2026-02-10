"""
Collect data to learn residual torque: Δτ = τ_isaac - τ_pd
"""
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import torch
import numpy as np
import pandas as pd
import os

def collect_residual_data(args, duration=300.0):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    dt = env_cfg.sim.dt * env_cfg.control.decimation
    num_steps = int(duration / dt)
    
    # PD gains (same as MuJoCo)
    Kp = 20.0  # MuJoCo default
    Kd = 0.5
    
    default_angles = torch.zeros(12, device=env.device)
    for i, name in enumerate(env.dof_names):
        default_angles[i] = env_cfg.init_state.default_joint_angles[name]
    
    action_scale = env_cfg.control.action_scale
    
    data_rows = []
    commands_list = [
        [0.5, 0.0, 0.0], [0.3, 0.3, 0.0], [0.4, 0.0, 0.5],
        [0.0, 0.0, 1.0], [0.6, 0.0, 0.0], [0.3, -0.3, 0.0],
    ]
    
    print(f"Collecting residual data for {duration}s...")
    
    for step in range(num_steps):
        cmd_idx = (step // int(5.0 / dt)) % len(commands_list)
        cmd = torch.tensor([commands_list[cmd_idx]], device=env.device)
        env.commands[:, :3] = cmd
        
        actions = policy(obs.detach())
        desired_pos = (actions * action_scale + default_angles)[0].detach().cpu().numpy()
        current_pos = env.dof_pos[0].detach().cpu().numpy()
        current_vel = env.dof_vel[0].detach().cpu().numpy()
        
        obs, _, _, _, _ = env.step(actions.detach())
        
        # Isaac Gym actual torque
        tau_isaac = env.torques[0].detach().cpu().numpy()
        
        # What PD control would give
        tau_pd = Kp * (desired_pos - current_pos) + Kd * (0 - current_vel)
        
        # Residual
        tau_residual = tau_isaac - tau_pd
        
        for i in range(12):
            data_rows.append({
                'motor': i,
                'pos_error': desired_pos[i] - current_pos[i],
                'velocity': current_vel[i],
                'tau_isaac': tau_isaac[i],
                'tau_pd': tau_pd[i],
                'tau_residual': tau_residual[i],
            })
        
        if step % 1000 == 0:
            print(f"  Step {step}/{num_steps}")
    
    df = pd.DataFrame(data_rows)
    save_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs/residual_data.csv")
    df.to_csv(save_path, index=False)
    
    print(f"\nSaved {len(df)} samples to: {save_path}")
    print(f"\nResidual torque stats:")
    print(f"  Mean: {df['tau_residual'].mean():.3f} N·m")
    print(f"  Std: {df['tau_residual'].std():.3f} N·m")
    print(f"  Range: [{df['tau_residual'].min():.2f}, {df['tau_residual'].max():.2f}]")
    
    return save_path

if __name__ == "__main__":
    args = get_args()
    collect_residual_data(args, duration=300.0)