"""
Collect Excitation Trajectories for ActuatorNet
Based on Hwangbo et al. recommendations:
- Random joint excitations
- High velocity motions
- Near-limit torques
- Various command scenarios
"""
import isaacgym
import torch
import numpy as np
import pandas as pd
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
import os

def collect_excitation_data():
    args = get_args()
    args.task = "go2"
    args.num_envs = 100  # More envs for diversity
    args.headless = True
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = args.num_envs
    
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # Load trained policy
    policy_path = os.path.expanduser("~/6619_ws/unitree_rl_gym/logs/rough_go2/exported/policies/policy_1.pt")
    policy = torch.jit.load(policy_path).to(env.device)
    policy.eval()
    
    print("=== Collecting Excitation Data ===")
    print("Phase 1: Normal walking with varied commands")
    print("Phase 2: Random joint perturbations")
    print("Phase 3: High-speed commands")
    print("Phase 4: Recovery scenarios")
    
    all_data = []
    
    # Phase 1: Normal walking with varied commands (10000 steps)
    print("\n[Phase 1] Normal walking...")
    obs = env.get_observations()
    for step in range(2500):
        # Vary commands
        if step % 250 == 0:
            env.commands[:, 0] = torch.rand(args.num_envs, device=env.device) * 1.0 - 0.2  # vx: -0.2 to 0.8
            env.commands[:, 1] = torch.rand(args.num_envs, device=env.device) * 0.6 - 0.3  # vy: -0.3 to 0.3
            env.commands[:, 2] = torch.rand(args.num_envs, device=env.device) * 2.0 - 1.0  # wz: -1.0 to 1.0
        
        with torch.no_grad():
            actions = policy(obs)
        
        # Record before step
        pos_before = env.dof_pos.clone()
        vel_before = env.dof_vel.clone()
        
        obs, _, _, _, _ = env.step(actions)
        
        # Record after step
        torques = env.torques.clone()
        
        # Store data
        for e in range(min(10, args.num_envs)):  # Sample 10 envs
            for j in range(12):
                all_data.append({
                    'phase': 'normal',
                    'motor': j,
                    'pos': pos_before[e, j].item(),
                    'vel': vel_before[e, j].item(),
                    'action': actions[e, j].item(),
                    'torque': torques[e, j].item()
                })
        
        if step % 500 == 0:
            print(f"  Step {step}/2500")
    
    # Phase 2: Random joint perturbations (5000 steps)
    print("\n[Phase 2] Random perturbations...")
    for step in range(1250):
        # Add random perturbations to actions
        with torch.no_grad():
            actions = policy(obs)
        
        # Add noise to actions (excitation)
        if step % 50 < 25:  # 50% of time add perturbations
            noise = torch.randn_like(actions) * 0.5  # Large noise
            actions = actions + noise
        
        pos_before = env.dof_pos.clone()
        vel_before = env.dof_vel.clone()
        
        obs, _, _, _, _ = env.step(actions)
        torques = env.torques.clone()
        
        for e in range(min(10, args.num_envs)):
            for j in range(12):
                all_data.append({
                    'phase': 'perturbation',
                    'motor': j,
                    'pos': pos_before[e, j].item(),
                    'vel': vel_before[e, j].item(),
                    'action': actions[e, j].item(),
                    'torque': torques[e, j].item()
                })
        
        if step % 250 == 0:
            print(f"  Step {step}/1250")
    
    # Phase 3: High-speed commands (5000 steps)
    print("\n[Phase 3] High-speed commands...")
    for step in range(1250):
        # Aggressive commands
        if step % 100 == 0:
            env.commands[:, 0] = torch.rand(args.num_envs, device=env.device) * 1.5  # vx: 0 to 1.5
            env.commands[:, 2] = torch.rand(args.num_envs, device=env.device) * 3.0 - 1.5  # wz: -1.5 to 1.5
        
        with torch.no_grad():
            actions = policy(obs)
        
        pos_before = env.dof_pos.clone()
        vel_before = env.dof_vel.clone()
        
        obs, _, _, _, _ = env.step(actions)
        torques = env.torques.clone()
        
        for e in range(min(10, args.num_envs)):
            for j in range(12):
                all_data.append({
                    'phase': 'high_speed',
                    'motor': j,
                    'pos': pos_before[e, j].item(),
                    'vel': vel_before[e, j].item(),
                    'action': actions[e, j].item(),
                    'torque': torques[e, j].item()
                })
        
        if step % 250 == 0:
            print(f"  Step {step}/1250")
    
    # Phase 4: Sudden command switches (5000 steps)
    print("\n[Phase 4] Command switches...")
    for step in range(1250):
        # Sudden command changes
        if step % 50 == 0:
            # Random sudden switch
            switch_type = np.random.randint(3)
            if switch_type == 0:  # Stop
                env.commands[:, :] = 0
            elif switch_type == 1:  # Turn
                env.commands[:, 0] = 0.4
                env.commands[:, 2] = torch.rand(args.num_envs, device=env.device) * 2.0 - 1.0
            else:  # Lateral
                env.commands[:, 1] = torch.rand(args.num_envs, device=env.device) * 0.6 - 0.3
        
        with torch.no_grad():
            actions = policy(obs)
        
        pos_before = env.dof_pos.clone()
        vel_before = env.dof_vel.clone()
        
        obs, _, _, _, _ = env.step(actions)
        torques = env.torques.clone()
        
        for e in range(min(10, args.num_envs)):
            for j in range(12):
                all_data.append({
                    'phase': 'switch',
                    'motor': j,
                    'pos': pos_before[e, j].item(),
                    'vel': vel_before[e, j].item(),
                    'action': actions[e, j].item(),
                    'torque': torques[e, j].item()
                })
        
        if step % 250 == 0:
            print(f"  Step {step}/1250")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    print(f"\n=== Data Collection Complete ===")
    print(f"Total samples: {len(df)}")
    print(f"\nPhase breakdown:")
    print(df['phase'].value_counts())
    
    print(f"\nData ranges:")
    print(f"Position: [{df['pos'].min():.3f}, {df['pos'].max():.3f}]")
    print(f"Velocity: [{df['vel'].min():.3f}, {df['vel'].max():.3f}]")
    print(f"Torque: [{df['torque'].min():.3f}, {df['torque'].max():.3f}]")
    
    # Save
    save_path = os.path.expanduser("~/actuator_net/app/resources/excitation_data.csv")
    df.to_csv(save_path, index=False)
    print(f"\nSaved to: {save_path}")
    
    return df

if __name__ == "__main__":
    collect_excitation_data()
