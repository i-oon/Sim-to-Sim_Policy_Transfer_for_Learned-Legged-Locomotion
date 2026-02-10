import isaacgym
import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime
import glob

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from rsl_rl.modules.actor_critic_rma import ActorCriticRMA

def collect_data(args):
    # =========================================================================
    # 1. Setup environment (Force High Parallelization)
    # =========================================================================
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = args.num_envs # ใช้ 4096 เพื่อความหลากหลายของข้อมูล
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    
    # =========================================================================
    # 2. Load trained policy (Phase 1 Teacher)
    # =========================================================================
    log_root = os.path.join('logs', 'go2_full_rma')
    run_path = os.path.join(log_root, args.load_run)
    ckpts = sorted(glob.glob(os.path.join(run_path, 'model_*.pt')))
    ckpt_path = ckpts[-1]
    
    print(f"Loading Teacher Policy: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=args.device)
    
    # ดึงค่า Config ให้ตรงกับตอนเทรน Phase 1
    base_obs_dim = getattr(train_cfg.policy, 'base_obs_dim', 48)
    num_env_params = getattr(train_cfg.policy, 'num_env_params', 10)
    env_encoding_dim = getattr(train_cfg.policy, 'env_encoding_dim', 8)
    
    actor_critic = ActorCriticRMA(
        num_actor_obs=env.num_obs,
        num_critic_obs=env.num_privileged_obs,
        num_actions=env.num_actions,
        base_obs_dim=base_obs_dim,
        num_env_params=num_env_params,
        env_encoding_dim=env_encoding_dim,
    ).to(args.device)
    
    actor_critic.load_state_dict(ckpt['model_state_dict'])
    actor_critic.eval()
    
    # =========================================================================
    # 3. Enhanced Collection Loop (Fixing R2 & Turning)
    # =========================================================================
    obs_history_len = 50
    all_obs_histories = []
    all_env_encodings = []
    
    obs = env.get_observations()
    collect_after = 100 # รอให้ History เต็มก่อนเริ่มบันทึก
    
    print(f"\nCollecting: {args.num_steps} steps | Targeting R2 > 0.8")

    for step in range(args.num_steps):
        # --- FIXED 1: Aggressive Command Randomization ---
        # บังคับให้หุ่นเลี้ยวซ้าย-ขวาสลับไปมาอย่างรุนแรง เพื่อให้เกิดแรงเหวี่ยงหนีศูนย์กลาง
        if step % 50 == 0:
            new_cmds = torch.zeros((env.num_envs, 3), device=args.device)
            # vx: สุ่มเดินหน้า/ถอยหลัง (-0.5 ถึง 1.2 m/s)
            new_cmds[:, 0] = torch.rand(env.num_envs, device=args.device) * 1.7 - 0.5 
            # wz: สุ่ม Yaw Rate รุนแรง (-1.5 ถึง 1.5 rad/s) เพื่อแก้ปัญหา S2 Turn
            new_cmds[:, 2] = torch.rand(env.num_envs, device=args.device) * 3.0 - 1.5 
            env.commands[:, :3] = new_cmds
            
        # --- FIXED 2: Online Domain Randomization ---
        # สุ่ม Friction/Mass ใหม่ระหว่างที่หุ่นกำลังก้าวขา (RMA Core Principle)
        if step % 100 == 0:
            if hasattr(env, 'randomize_rigid_body_properties'):
                env.randomize_rigid_body_properties()

        with torch.no_grad():
            actions = actor_critic.act_inference(obs)
        
        obs, _, _, _, _ = env.step(actions)
        
        if step >= collect_after:
            # ดึง Obs History และ Encoding (Ground Truth)
            obs_history = env.get_obs_history().clone()
            env_params = env.get_env_params()
            env_encoding = actor_critic.get_env_encoding(env_params)
            
            # Flatten: [num_envs, 2400]
            obs_history_flat = obs_history.reshape(env.num_envs, -1)
            
            all_obs_histories.append(obs_history_flat.cpu())
            all_env_encodings.append(env_encoding.cpu())
        
        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}/{args.num_steps} | Samples: {len(all_obs_histories)*args.num_envs:,}")

    # =========================================================================
    # 4. Save Dataset
    # =========================================================================
    obs_histories = torch.cat(all_obs_histories, dim=0)
    env_encodings = torch.cat(all_env_encodings, dim=0)
    
    save_dir = os.path.join(run_path, 'adaptation_data')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'rma_adaptation_data.pt')
    
    torch.save({
        'obs_histories': obs_histories,
        'env_encodings': env_encodings,
        'obs_history_len': obs_history_len,
        'base_obs_dim': base_obs_dim,
        'env_encoding_dim': env_encoding_dim,
    }, save_path)
    
    print(f"\nSuccess! Dataset for RMA Phase 2 saved to: {save_path}")

if __name__ == '__main__':
    args = get_args()
    parser = argparse.ArgumentParser(description="RMA Collector", add_help=False)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--num_envs', type=int, default=4096)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    custom_args, _ = parser.parse_known_args()
    args.num_steps = custom_args.num_steps
    args.num_envs = custom_args.num_envs
    args.device = custom_args.device
    args.task = 'go2_full_rma'

    collect_data(args)