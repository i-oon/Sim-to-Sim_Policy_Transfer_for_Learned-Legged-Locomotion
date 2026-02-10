"""
Go2 RMA Environment - with privileged observations
"""
from legged_gym.envs.base.legged_robot import LeggedRobot
import torch

class Go2RMAEnv(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # Initialize environment parameters
        self.friction_coeffs = torch.ones(self.num_envs, 1, device=self.device)
        self.mass_offsets = torch.zeros(self.num_envs, 1, device=self.device)
        self.kp_factors = torch.ones(self.num_envs, 1, device=self.device)
        self.kd_factors = torch.ones(self.num_envs, 1, device=self.device)
        
        # Initialize privileged obs buffer
        self.privileged_obs_buf = torch.zeros(
            self.num_envs, self.cfg.env.num_privileged_obs, 
            device=self.device, dtype=torch.float
        )
    
    def compute_observations(self):
        """Compute both regular and privileged observations"""
        # Regular observations (same as before)
        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)
        
        # Privileged observations = obs + environment parameters (10 extra)
        env_params = torch.cat((
            self.friction_coeffs,           # 1
            self.mass_offsets / 2.0,        # 1 (normalized)
            self.kp_factors - 1.0,          # 1 (centered at 0)
            self.kd_factors - 1.0,          # 1 (centered at 0)
            self.base_lin_vel,              # 3 (ground truth, no noise)
            self.base_ang_vel,              # 3 (ground truth, no noise)
        ), dim=-1)  # Total: 10 extra
        
        self.privileged_obs_buf = torch.cat((self.obs_buf, env_params), dim=-1)
        
        # Add noise to regular obs only
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def reset_idx(self, env_ids):
        """Reset environments and randomize parameters"""
        super().reset_idx(env_ids)
        
        # Randomize env params for reset envs
        if len(env_ids) > 0:
            self._randomize_env_params_for_ids(env_ids)
    
    def _randomize_env_params_for_ids(self, env_ids):
        """Randomize parameters for specific environment IDs"""
        n = len(env_ids)
        
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch.empty(n, 1, device=self.device).uniform_(
                self.cfg.domain_rand.friction_range[0],
                self.cfg.domain_rand.friction_range[1]
            )
            
        if self.cfg.domain_rand.randomize_base_mass:
            self.mass_offsets[env_ids] = torch.empty(n, 1, device=self.device).uniform_(
                self.cfg.domain_rand.added_mass_range[0],
                self.cfg.domain_rand.added_mass_range[1]
            )
            
        if self.cfg.domain_rand.randomize_gains:
            self.kp_factors[env_ids] = torch.empty(n, 1, device=self.device).uniform_(
                self.cfg.domain_rand.stiffness_multiplier_range[0],
                self.cfg.domain_rand.stiffness_multiplier_range[1]
            )
            self.kd_factors[env_ids] = torch.empty(n, 1, device=self.device).uniform_(
                self.cfg.domain_rand.damping_multiplier_range[0],
                self.cfg.domain_rand.damping_multiplier_range[1]
            )
