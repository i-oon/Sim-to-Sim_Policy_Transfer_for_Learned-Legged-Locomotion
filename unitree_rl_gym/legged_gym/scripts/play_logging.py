# IMPORTANT: Import isaacgym FIRST before torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import numpy as np
import os
from datetime import datetime
import torch


def play_and_log(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # --- Number of agents ---
    num_agents = args.num_envs if hasattr(args, 'num_envs') and args.num_envs else 4
    env_cfg.env.num_envs = num_agents

    # Override for logging
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # Set fixed command for all agents
    cmd = torch.tensor([0.5, 0.0, 0.0], device=env.device)
    env.commands[:, :3] = cmd.unsqueeze(0).expand(num_agents, -1)

    # Logging — store all agents: shape will be (num_steps, num_agents, dim)
    log_data = {
        'time': [],
        'base_pos': [],
        'base_quat': [],
        'base_lin_vel': [],
        'base_ang_vel': [],
        'joint_pos': [],
        'joint_vel': [],
        'actions': [],
        'torques': [],
        'cmd': [],
    }

    duration = 30.0
    dt = env_cfg.sim.dt * env_cfg.control.decimation
    num_steps = int(duration / dt)

    print(f"Running Isaac Gym with {num_agents} agents for {duration}s ({num_steps} steps)")
    print(f"Command: vx={cmd[0]:.2f}, vy={cmd[1]:.2f}, wz={cmd[2]:.2f}")

    for i in range(num_steps):
        actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())

        # Re-apply command (env may reset and randomize commands)
        env.commands[:, :3] = cmd.unsqueeze(0).expand(num_agents, -1)

        # Log all agents
        log_data['time'].append(i * dt)
        log_data['base_pos'].append(env.root_states[:, :3].detach().cpu().numpy())
        log_data['base_quat'].append(env.root_states[:, 3:7].detach().cpu().numpy())
        log_data['base_lin_vel'].append(env.base_lin_vel.detach().cpu().numpy())
        log_data['base_ang_vel'].append(env.base_ang_vel.detach().cpu().numpy())
        log_data['joint_pos'].append(env.dof_pos.detach().cpu().numpy())
        log_data['joint_vel'].append(env.dof_vel.detach().cpu().numpy())
        log_data['actions'].append(actions.detach().cpu().numpy())
        log_data['torques'].append(env.torques.detach().cpu().numpy())
        log_data['cmd'].append(
            cmd.unsqueeze(0).expand(num_agents, -1).detach().cpu().numpy()
        )

    # Convert to numpy arrays
    # time: (num_steps,)
    # others: (num_steps, num_agents, dim)
    for key in log_data:
        log_data[key] = np.array(log_data[key])

    # Save
    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs/sim2sim")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
        log_dir, f"isaacgym_baseline_{num_agents}agents_{timestamp}.npz"
    )

    np.savez(log_file, **log_data, simulation_dt=dt, num_agents=num_agents)
    print(f"\nSaved log to: {log_file}")
    print(f"Shapes: time={log_data['time'].shape}, "
          f"base_lin_vel={log_data['base_lin_vel'].shape}")

    # --- Print per-agent metrics ---
    print(f"\n{'='*60}")
    print(f"  Isaac Gym Baseline — {num_agents} Agents")
    print(f"{'='*60}")
    print(f"Duration: {log_data['time'][-1]:.2f}s\n")

    # Per-agent stats
    header = f"{'Agent':>6} | {'vx mean':>8} | {'vx err':>7} | {'vy err':>7} | {'wz err':>7} | {'τ max':>7}"
    print(header)
    print("-" * len(header))

    for a in range(num_agents):
        vx_mean = np.mean(log_data['base_lin_vel'][:, a, 0])
        vx_err = abs(vx_mean - 0.5)
        vy_err = np.mean(np.abs(log_data['base_lin_vel'][:, a, 1]))
        wz_err = np.mean(np.abs(log_data['base_ang_vel'][:, a, 2]))
        tau_max = np.max(np.abs(log_data['torques'][:, a, :]))
        print(f"  {a:>4d} | {vx_mean:>8.3f} | {vx_err:>7.3f} | {vy_err:>7.3f} | {wz_err:>7.3f} | {tau_max:>7.2f}")

    # Aggregate stats (mean ± std across agents)
    print(f"\n{'='*60}")
    print("  Aggregate (mean ± std across agents)")
    print(f"{'='*60}")

    vx_means = np.mean(log_data['base_lin_vel'][:, :, 0], axis=0)  # (num_agents,)
    vy_means = np.mean(np.abs(log_data['base_lin_vel'][:, :, 1]), axis=0)
    wz_means = np.mean(np.abs(log_data['base_ang_vel'][:, :, 2]), axis=0)
    tau_maxs = np.max(np.abs(log_data['torques']).reshape(len(log_data['time']), num_agents, -1), axis=(0, 2))

    print(f"  vx mean:  {np.mean(vx_means):.3f} ± {np.std(vx_means):.3f} m/s (cmd: 0.50)")
    print(f"  vy |err|: {np.mean(vy_means):.4f} ± {np.std(vy_means):.4f} m/s")
    print(f"  wz |err|: {np.mean(wz_means):.4f} ± {np.std(wz_means):.4f} rad/s")
    print(f"  τ max:    {np.mean(tau_maxs):.2f} ± {np.std(tau_maxs):.2f} N·m")

    # RMSE across all agents
    vx_rmse = np.sqrt(np.mean((log_data['base_lin_vel'][:, :, 0] - 0.5) ** 2))
    vy_rmse = np.sqrt(np.mean(log_data['base_lin_vel'][:, :, 1] ** 2))
    wz_rmse = np.sqrt(np.mean(log_data['base_ang_vel'][:, :, 2] ** 2))

    print(f"\n  Tracking RMSE (all agents pooled):")
    print(f"    vx: {vx_rmse:.4f}")
    print(f"    vy: {vy_rmse:.4f}")
    print(f"    wz: {wz_rmse:.4f}")


if __name__ == '__main__':
    args = get_args()
    play_and_log(args)