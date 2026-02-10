"""
Go2 Full RMA Config (FIXED v2)

v2 changes:
  - encoding_loss_coef: 0.5 → 5.0  (reward~25, ต้องใหญ่กว่านี้)
  - เพิ่ม encoding_l2_coef = 1.0   (force encoder output norm ให้ใหญ่ขึ้น)
"""
from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO


class Go2FullRMACfg(GO2RoughCfg):
    class env(GO2RoughCfg.env):
        num_observations     = 58
        num_privileged_obs   = 58
        base_obs_dim         = 48
        num_env_params       = 10
        env_encoding_dim     = 8
        obs_history_len      = 50

    class domain_rand(GO2RoughCfg.domain_rand):
        randomize_friction            = True
        friction_range                = [0.1, 3.0]
        randomize_base_mass           = True
        added_mass_range              = [-3.0, 5.0]
        randomize_gains               = True
        stiffness_multiplier_range    = [0.5, 1.5]
        damping_multiplier_range      = [0.5, 1.5]
        push_robots                   = True
        push_interval_s               = 15
        max_push_vel_xy               = 1.0

    class control(GO2RoughCfg.control):
        control_type  = 'P'
        stiffness     = {'joint': 20.}
        damping       = {'joint': 0.5}
        action_scale  = 0.25
        decimation    = 4


class Go2FullRMACfgPPO(GO2RoughCfgPPO):
    class algorithm(GO2RoughCfgPPO.algorithm):
        entropy_coef = 0.01

    class policy(GO2RoughCfgPPO.policy):
        policy_class_name       = 'ActorCriticRMA'
        base_obs_dim            = 48
        num_env_params          = 10
        env_encoding_dim        = 8
        env_encoder_hidden_dims = [256, 128]
        actor_hidden_dims       = [512, 256, 128]
        critic_hidden_dims      = [512, 256, 128]
        activation              = 'elu'

        # v2: เพิ่มจาก 0.5 → 5.0 (reward~25 ต้องการ coef ที่ใหญ่กว่า)
        encoding_loss_coef  = 5.0

        # v2 NEW: L2 norm penalty บังคับ encoder output ให้ norm~1.0
        # loss += coef * max(0, 1.0 - mean_norm)²
        encoding_l2_coef    = 1.0

    class runner(GO2RoughCfgPPO.runner):
        experiment_name    = 'go2_full_rma'
        policy_class_name  = 'ActorCriticRMA'
        max_iterations     = 5000
        save_interval      = 500