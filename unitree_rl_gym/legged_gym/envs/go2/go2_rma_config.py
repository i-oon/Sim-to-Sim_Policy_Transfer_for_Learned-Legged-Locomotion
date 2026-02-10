"""
Go2 RMA Config - with privileged observations for adaptation
"""
from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO

class Go2RMACfg(GO2RoughCfg):
    class env(GO2RoughCfg.env):
        num_observations = 48  # Same as before
        num_privileged_obs = 48 + 10  # obs + env_params (must match compute_observations)
        # env_params: friction(1) + mass(1) + kp(1) + kd(1) + lin_vel(3) + ang_vel(3) = 10
        
    class domain_rand(GO2RoughCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.3, 1.5]
        
        randomize_base_mass = True
        added_mass_range = [-1.0, 2.0]  # kg
        
        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]  # Kp multiplier
        damping_multiplier_range = [0.8, 1.2]    # Kd multiplier
        
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0

class Go2RMACfgPPO(GO2RoughCfgPPO):
    class algorithm(GO2RoughCfgPPO.algorithm):
        pass
        
    class runner(GO2RoughCfgPPO.runner):
        experiment_name = 'go2_rma'
        max_iterations = 3000
