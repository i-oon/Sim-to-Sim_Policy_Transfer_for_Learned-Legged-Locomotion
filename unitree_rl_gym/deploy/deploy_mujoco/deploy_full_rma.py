"""
Phase 2 - Step 3: Deploy Full RMA Policy in MuJoCo

Uses:
  - Trained Actor (from Phase 1)
  - Trained Adaptation Module (from Phase 2)
  - No env_params needed — adaptation module estimates from obs history

Usage:
  cd ~/6619_ws/unitree_rl_gym
  python deploy/deploy_mujoco/deploy_full_rma.py \
      --policy_path=logs/go2_full_rma/<run>/model_5000.pt \
      --adaptation_path=logs/go2_full_rma/<run>/adaptation_data/adaptation_module.pt \
      --scenario S2_turn --duration 10
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import time

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from rsl_rl.modules.actor_critic_rma import ActorCriticRMA, AdaptationModule


# =========================================================================
# MuJoCo Environment (adapt from your existing deploy scripts)
# =========================================================================

def load_mujoco_env(cfg_path='deploy/deploy_mujoco/configs/go2.yaml'):
    """Load MuJoCo environment from config."""
    import yaml
    import mujoco
    
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    model = mujoco.MjModel.from_xml_path(cfg['mujoco_model_path'])
    data = mujoco.MjData(model)
    
    return model, data, cfg


class FullRMADeployer:
    """Deploy Full RMA policy in MuJoCo with online adaptation."""
    
    def __init__(self, policy_path, adaptation_path, device='cpu'):
        self.device = torch.device(device)
        
        # ----- Load Adaptation Module -----
        print(f"Loading adaptation module: {adaptation_path}")
        adapt_ckpt = torch.load(adaptation_path, map_location=self.device)
        
        self.obs_dim = adapt_ckpt['obs_dim']
        self.history_len = adapt_ckpt['history_len']
        self.encoding_dim = adapt_ckpt['encoding_dim']
        hidden_dims = adapt_ckpt['hidden_dims']
        
        self.adaptation_module = AdaptationModule(
            obs_dim=self.obs_dim,
            history_len=self.history_len,
            encoding_dim=self.encoding_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.adaptation_module.load_state_dict(adapt_ckpt['model_state_dict'])
        self.adaptation_module.eval()
        
        print(f"  R² score: {adapt_ckpt.get('r2_score', 'N/A')}")
        
        # ----- Load Policy (Actor only) -----
        print(f"Loading policy: {policy_path}")
        policy_ckpt = torch.load(policy_path, map_location=self.device)
        state_dict = policy_ckpt['model_state_dict']
        
        # Create ActorCriticRMA
        self.actor_critic = ActorCriticRMA(
            num_actor_obs=self.obs_dim + 10,  # 58 (obs + env_params packed)
            num_critic_obs=self.obs_dim + 10,
            num_actions=12,
            base_obs_dim=self.obs_dim,
            num_env_params=10,
            env_encoding_dim=self.encoding_dim,
        ).to(self.device)
        self.actor_critic.load_state_dict(state_dict)
        self.actor_critic.eval()
        
        # Enable adaptation mode
        self.actor_critic.use_adaptation = True
        self.actor_critic.adaptation_module = self.adaptation_module
        
        # ----- Observation History Buffer -----
        self.obs_history = deque(maxlen=self.history_len)
        # Initialize with zeros
        for _ in range(self.history_len):
            self.obs_history.append(torch.zeros(self.obs_dim))
        
        print(f"  History buffer: {self.history_len} x {self.obs_dim}")
        print("Ready for deployment!")
    
    def get_action(self, obs_48):
        """Get action from Full RMA pipeline.
        
        Args:
            obs_48: numpy array or tensor, 48-dim base observations
                    [lin_vel(3), ang_vel(3), gravity(3), commands(3),
                     dof_pos(12), dof_vel(12), actions(12)]
        
        Returns:
            action: numpy array, 12-dim joint position targets
        """
        if isinstance(obs_48, np.ndarray):
            obs_48 = torch.from_numpy(obs_48).float()
        
        obs_48 = obs_48.to(self.device)
        
        # Update history
        self.obs_history.append(obs_48.cpu().clone())
        
        # Compute adaptation encoding
        with torch.no_grad():
            # Flatten history: [1, history_len * obs_dim]
            history_flat = torch.cat(list(self.obs_history)).unsqueeze(0).to(self.device)
            env_encoding = self.adaptation_module(history_flat)  # [1, 8]
            
            # Cache encoding for actor
            self.actor_critic.set_adaptation_encoding(env_encoding)
            
            # Actor input: [obs(48), dummy_env_params(10)] = 58
            # ActorCriticRMA will use cached encoding instead of encoder
            dummy_params = torch.zeros(1, 10, device=self.device)
            full_obs = torch.cat([obs_48.unsqueeze(0), dummy_params], dim=-1)
            
            action = self.actor_critic.act_inference(full_obs)
        
        return action.cpu().numpy().flatten()
    
    def reset_history(self):
        """Clear observation history (call on episode reset)."""
        self.obs_history.clear()
        for _ in range(self.history_len):
            self.obs_history.append(torch.zeros(self.obs_dim))


# =========================================================================
# Command Switching Scenarios (same as existing deploy scripts)
# =========================================================================

SCENARIOS = {
    'baseline': {
        'description': 'Constant forward velocity',
        'commands': lambda t: [0.5, 0.0, 0.0],
    },
    'S1_forward': {
        'description': 'Forward velocity: 0 -> 0.5 at t=3s',
        'commands': lambda t: [0.5 if t >= 3.0 else 0.0, 0.0, 0.0],
    },
    'S2_turn': {
        'description': 'Turn: wz 0 -> 1.0 at t=3s, vx=0.5 constant',
        'commands': lambda t: [0.5, 0.0, 1.0 if t >= 3.0 else 0.0],
    },
    'S3_stop': {
        'description': 'Stop: vx 0.5 -> 0.0 at t=3s',
        'commands': lambda t: [0.0 if t >= 3.0 else 0.5, 0.0, 0.0],
    },
    'constant_turn': {
        'description': 'Constant turn: vx=0.5, wz=1.0',
        'commands': lambda t: [0.5, 0.0, 1.0],
    },
}


def main():
    parser = argparse.ArgumentParser(description='Deploy Full RMA in MuJoCo')
    parser.add_argument('--policy_path', type=str, required=True)
    parser.add_argument('--adaptation_path', type=str, required=True)
    parser.add_argument('--scenario', type=str, default='S2_turn', choices=list(SCENARIOS.keys()))
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--dt', type=float, default=0.005, help='Simulation dt')
    parser.add_argument('--decimation', type=int, default=4, help='Control decimation')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_log', action='store_true', help='Save trajectory log')
    parser.add_argument('--output_dir', type=str, default='logs/sim2sim/full_rma')
    args = parser.parse_args()
    
    # Create deployer
    deployer = FullRMADeployer(
        policy_path=args.policy_path,
        adaptation_path=args.adaptation_path,
        device=args.device,
    )
    
    scenario = SCENARIOS[args.scenario]
    print(f"\nScenario: {args.scenario} - {scenario['description']}")
    print(f"Duration: {args.duration}s")
    
    # ----- NOTE: Integration with your MuJoCo deploy pipeline -----
    # Replace the section below with your existing MuJoCo simulation loop.
    # The key change is: instead of calling policy(obs) directly,
    # call deployer.get_action(obs_48) which handles history + adaptation.
    #
    # Example integration with your existing deploy_mujoco_go2.py:
    #
    # BEFORE (DR policy):
    #   obs = get_mujoco_obs()           # 48-dim
    #   action = policy(obs)              # plain actor
    #
    # AFTER (Full RMA):
    #   obs = get_mujoco_obs()           # 48-dim (same)
    #   action = deployer.get_action(obs) # adaptation + actor
    # ---------------------------------------------------------------
    
    print("\n" + "=" * 60)
    print("INTEGRATION NOTE:")
    print("=" * 60)
    print("Replace your existing MuJoCo deploy loop's action computation:")
    print("  OLD: action = policy(obs)")
    print("  NEW: action = deployer.get_action(obs_48)")
    print("")
    print("The deployer handles:")
    print("  1. Maintaining observation history buffer")
    print("  2. Running adaptation module on history")  
    print("  3. Feeding encoding to actor for adapted action")
    print("=" * 60)


if __name__ == '__main__':
    main()
