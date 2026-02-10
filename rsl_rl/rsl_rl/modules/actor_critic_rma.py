"""
ActorCriticRMA (FIXED)

Bug found: EnvironmentEncoder was receiving RAW (unnormalized) env_params
  AND had hidden_dims=[64,32] which is too small for 10→8 compression.
  Result: encoder collapsed to near-zero outputs (std=0.01, norm=0.027).
  
Fixes applied:
  1. env_encoder_hidden_dims default: [64,32] → [256,128]
  2. Added encoding_variance_loss in evaluate():
       L_enc = -lambda * mean(Var(encoding, dim=0))
     This penalizes near-zero variance encodings, forcing the encoder
     to produce diverse, informative outputs across environments.
  3. No changes to rsl_rl interface.

Note: env_params MUST be normalized (~N(0,1)) in the environment before
      being passed here. See go2_full_rma_env_fixed.py for normalization.

Reference: Kumar et al. "RMA: Rapid Motor Adaptation" (RSS 2021)
"""
import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic import ActorCritic, get_activation


class EnvironmentEncoder(nn.Module):
    """Encodes (normalized) environment parameters → compact latent vector."""
    def __init__(self, num_env_params=10, encoding_dim=8,
                 hidden_dims=[256, 128],   # FIXED: was [64, 32]
                 activation='elu'):
        super().__init__()
        activation_fn = get_activation(activation)

        layers = []
        prev_dim = num_env_params
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(activation_fn)
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, encoding_dim))

        self.encoder = nn.Sequential(*layers)

        # Log encoding stats every N forward passes (for monitoring)
        self._call_count = 0

    def forward(self, env_params):
        enc = self.encoder(env_params)

        # Debug logging every 500 calls during training
        self._call_count += 1
        if self._call_count % 500 == 0:
            with torch.no_grad():
                std  = enc.std(dim=0).mean().item()
                norm = enc.norm(dim=-1).mean().item()
            # Will show in Isaac Gym training output
            print(f"[EnvEncoder] step={self._call_count:6d}  "
                  f"enc_std={std:.4f}  enc_norm={norm:.4f}  "
                  f"(target: std>0.3, norm>1.0)")
        return enc


class AdaptationModule(nn.Module):
    """Estimates environment encoding from observation history (Phase 2)."""
    def __init__(self, obs_dim=48, history_len=50, encoding_dim=8,
                 hidden_dims=[256, 128]):
        super().__init__()
        input_dim = obs_dim * history_len

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ELU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, encoding_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, obs_history_flat):
        return self.net(obs_history_flat)


class ActorCriticRMA(ActorCritic):
    """Actor-Critic with Environment Encoder.

    Receives 58-dim obs from env, splits internally:
      base_obs   = obs[:, :48]      (proprioception)
      env_params = obs[:, 48:]      (NORMALIZED physics params)
      actor_in   = cat(base_obs, encoder(env_params))  = 56-dim
      critic_in  = full obs                             = 58-dim
    """

    is_recurrent = False

    def __init__(self,
                 num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 base_obs_dim=48,
                 num_env_params=10,
                 env_encoding_dim=8,
                 env_encoder_hidden_dims=[256, 128],
                 actor_hidden_dims=[512, 256, 128],
                 critic_hidden_dims=[512, 256, 128],
                 activation='elu',
                 init_noise_std=1.0,
                 # Loss 1: maximize encoding variance across batch
                 # scale relative to reward (~25) → need coef ~5.0
                 encoding_loss_coef=5.0,
                 # Loss 2: penalize near-zero encoding norm
                 # loss += coef * max(0, 1.0 - mean_norm)^2
                 encoding_l2_coef=1.0,
                 **kwargs):

        self.base_obs_dim      = base_obs_dim
        self.num_env_params    = num_env_params
        self.env_encoding_dim  = env_encoding_dim
        self.encoding_loss_coef = encoding_loss_coef
        self.encoding_l2_coef   = encoding_l2_coef

        actual_actor_input  = base_obs_dim + env_encoding_dim  # 56
        actual_critic_input = num_critic_obs                   # 58

        super().__init__(
            num_actor_obs=actual_actor_input,
            num_critic_obs=actual_critic_input,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            **kwargs
        )

        self.env_encoder = EnvironmentEncoder(
            num_env_params=num_env_params,
            encoding_dim=env_encoding_dim,
            hidden_dims=env_encoder_hidden_dims,
            activation=activation,
        )

        self.adaptation_module = None
        self.use_adaptation    = False
        self._cached_encoding  = None

        print(f"[ActorCriticRMA] Initialized:")
        print(f"  base_obs_dim={base_obs_dim}, env_params={num_env_params}")
        print(f"  env_encoding_dim={env_encoding_dim}")
        print(f"  encoder hidden: {env_encoder_hidden_dims}  "
              f"(was [64,32], FIXED to {env_encoder_hidden_dims})")
        print(f"  encoding_loss_coef={encoding_loss_coef}")
        print(f"  Actor input: {actual_actor_input}  "
              f"(obs {base_obs_dim} + encoding {env_encoding_dim})")
        print(f"  Critic input: {actual_critic_input}")
        print(f"  Actor: obs(48) + encoding(8) = 56 -> 12")

    # ── obs splitting ─────────────────────────────────────────────────
    def _split_obs(self, observations):
        base_obs   = observations[:, :self.base_obs_dim]
        env_params = observations[:, self.base_obs_dim:]
        return base_obs, env_params

    def _get_actor_input(self, observations):
        base_obs, env_params = self._split_obs(observations)

        if self.use_adaptation and self._cached_encoding is not None:
            env_encoding = self._cached_encoding
        else:
            env_encoding = self.env_encoder(env_params)

        return torch.cat([base_obs, env_encoding], dim=-1)

    # ── rsl_rl interface ──────────────────────────────────────────────
    def act(self, observations, **kwargs):
        actor_input = self._get_actor_input(observations)
        return super().act(actor_input, **kwargs)

    def act_inference(self, observations):
        actor_input = self._get_actor_input(observations)
        return super().act_inference(actor_input)

    def evaluate(self, critic_observations, **kwargs):
        """
        Critic forward + two encoding losses to prevent collapse:

        Loss 1 — Variance loss (encoding_loss_coef):
          -mean(Var(encoding, dim=0))
          Maximizes diversity across the batch.
          coef=5.0 needed because reward~25 dominates otherwise.

        Loss 2 — L2 norm loss (encoding_l2_coef):
          max(0, 1.0 - mean_norm)^2
          Penalizes when encoder outputs near-zero vectors.
          Pulls mean encoding norm toward >=1.0.
        """
        values = super().evaluate(critic_observations, **kwargs)

        if self.training and (self.encoding_loss_coef > 0.0
                              or self.encoding_l2_coef > 0.0):
            _, env_params = self._split_obs(critic_observations)
            encoding = self.env_encoder(env_params)   # [batch, 8]

            total_enc_loss = torch.tensor(0.0, device=encoding.device)

            # Loss 1: variance — maximize spread across environments
            if self.encoding_loss_coef > 0.0:
                enc_var  = encoding.var(dim=0).mean()        # scalar
                var_loss = -enc_var                          # negate to maximize
                total_enc_loss = total_enc_loss + self.encoding_loss_coef * var_loss

            # Loss 2: norm — penalize near-zero encoding magnitude
            if self.encoding_l2_coef > 0.0:
                mean_norm  = encoding.norm(dim=-1).mean()    # scalar
                norm_loss  = torch.relu(1.0 - mean_norm) ** 2
                total_enc_loss = total_enc_loss + self.encoding_l2_coef * norm_loss

            values = values + total_enc_loss

        return values

    # ── helper methods ────────────────────────────────────────────────
    def get_env_encoding(self, env_params):
        """Get encoder output for Phase 2 data collection."""
        with torch.no_grad():
            return self.env_encoder(env_params)

    def set_adaptation_encoding(self, encoding):
        """Cache encoding from adaptation module (Phase 2 inference)."""
        self._cached_encoding = encoding

    def enable_adaptation(self, obs_dim=48, history_len=50,
                          hidden_dims=[256, 128]):
        """Initialize adaptation module for Phase 2 deployment."""
        self.adaptation_module = AdaptationModule(
            obs_dim=obs_dim,
            history_len=history_len,
            encoding_dim=self.env_encoding_dim,
            hidden_dims=hidden_dims,
        ).to(next(self.parameters()).device)
        self.use_adaptation = True
        print(f"[ActorCriticRMA] Adaptation module enabled:")
        print(f"  {obs_dim} x {history_len} → {self.env_encoding_dim}")
        return self.adaptation_module