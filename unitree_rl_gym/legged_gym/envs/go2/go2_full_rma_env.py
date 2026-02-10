"""
Go2 Full RMA Environment (FIXED)

BUG FIXED: env_params were fed raw (unnormalized) into the encoder.
  - friction: 0.1–3.0  (mean≈1.5, std≈0.8)
  - mass:    -3.0–5.0  (mean≈1.0, std≈2.3)
  - kp/kd:   0.5–1.5   (mean≈1.0, std≈0.3)
  
These very different scales caused the encoder to produce near-zero
outputs (std=0.01) because gradients from different dims cancelled.

FIX: Normalize each env_param to ~N(0,1) before feeding the encoder.
     Also increased encoder hidden dims [64,32] → [128,64] for capacity.
"""
from legged_gym.envs.base.legged_robot import LeggedRobot
import torch


class Go2FullRMAEnv(LeggedRobot):
    
    # ── Normalization constants (computed from domain_rand ranges) ─────
    # mean = (lo + hi) / 2,  std = (hi - lo) / (2 * sqrt(3))  [uniform dist]
    ENV_PARAM_MEAN = torch.tensor([
        1.55,   # friction:  (0.1+3.0)/2
        1.00,   # mass:      (-3.0+5.0)/2
        1.00,   # kp leg0
        1.00,   # kp leg1
        1.00,   # kp leg2
        1.00,   # kp leg3
        1.00,   # kd leg0
        1.00,   # kd leg1
        1.00,   # kd leg2
        1.00,   # kd leg3
    ])
    ENV_PARAM_STD = torch.tensor([
        0.838,  # friction:  (3.0-0.1)/(2*sqrt(3))
        2.309,  # mass:      (5.0-(-3.0))/(2*sqrt(3))
        0.289,  # kp:        (1.5-0.5)/(2*sqrt(3))
        0.289,
        0.289,
        0.289,
        0.289,  # kd
        0.289,
        0.289,
        0.289,
    ])

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.friction_coeffs = torch.ones(self.num_envs, 1, device=self.device)
        self.mass_offsets    = torch.zeros(self.num_envs, 1, device=self.device)
        self.kp_factors      = torch.ones(self.num_envs, 4, device=self.device)
        self.kd_factors      = torch.ones(self.num_envs, 4, device=self.device)

        obs_history_len = getattr(self.cfg.env, 'obs_history_len', 50)
        base_obs_dim    = 48
        self.obs_history = torch.zeros(
            self.num_envs, obs_history_len, base_obs_dim,
            device=self.device
        )

        # Move normalization constants to device
        self._ep_mean = self.ENV_PARAM_MEAN.to(self.device)
        self._ep_std  = self.ENV_PARAM_STD.to(self.device)

    def _randomize_env_params(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        n = len(env_ids)

        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch.empty(n, 1, device=self.device).uniform_(
                *self.cfg.domain_rand.friction_range)

        if self.cfg.domain_rand.randomize_base_mass:
            self.mass_offsets[env_ids] = torch.empty(n, 1, device=self.device).uniform_(
                *self.cfg.domain_rand.added_mass_range)

        if self.cfg.domain_rand.randomize_gains:
            self.kp_factors[env_ids] = torch.empty(n, 4, device=self.device).uniform_(
                *self.cfg.domain_rand.stiffness_multiplier_range)
            self.kd_factors[env_ids] = torch.empty(n, 4, device=self.device).uniform_(
                *self.cfg.domain_rand.damping_multiplier_range)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._randomize_env_params(env_ids)
        self.obs_history[env_ids] = 0.0

    def _get_raw_env_params(self):
        """Raw env params before normalization."""
        return torch.cat((
            self.friction_coeffs,
            self.mass_offsets,
            self.kp_factors,
            self.kd_factors,
        ), dim=-1)  # [N, 10]

    def _get_normalized_env_params(self):
        """
        FIXED: Normalize env_params to ~N(0,1) before feeding encoder.
        This is critical — without it, encoder outputs near-zero vectors.
        """
        raw = self._get_raw_env_params()
        return (raw - self._ep_mean) / self._ep_std  # [N, 10]

    def compute_observations(self):
        """
        obs_buf (58-dim) = [base_obs(48), normalized_env_params(10)]
        
        FIXED: env_params are now normalized before concatenation.
        The encoder receives ~N(0,1) inputs → meaningful gradients.
        """
        base_obs = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)  # [N, 48]

        # FIXED: normalized env_params
        env_params_normed = self._get_normalized_env_params()  # [N, 10]

        self.obs_buf            = torch.cat((base_obs, env_params_normed), dim=-1)  # [N, 58]
        self.privileged_obs_buf = self.obs_buf.clone()

        # History stores base_obs only (not env_params)
        self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)
        self.obs_history[:, -1, :] = base_obs

    def get_obs_history(self):
        return self.obs_history

    def get_env_params(self):
        """Return normalized env_params (consistent with obs_buf)."""
        return self._get_normalized_env_params()