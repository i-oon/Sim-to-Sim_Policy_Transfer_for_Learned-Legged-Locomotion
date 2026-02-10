"""
Hwangbo-style Excitation Data Collection
- Adds simulated actuator dynamics (delay, friction, saturation)
- Makes the problem non-trivial for neural network to learn
"""

from isaacgym import gymapi, gymtorch

import torch
import numpy as np
import os
import sys

LEGGED_GYM_ROOT_DIR = os.path.expanduser("~/6619_ws/unitree_rl_gym")
sys.path.append(LEGGED_GYM_ROOT_DIR)

from legged_gym import LEGGED_GYM_ROOT_DIR as LG_ROOT

class ActuatorDynamics:
    """Simulated actuator dynamics to make the problem non-trivial"""
    def __init__(self, n_joints=12):
        self.n_joints = n_joints
        # Torque delay buffer (1-2 steps)
        self.torque_buffer = [np.zeros(n_joints) for _ in range(2)]
        # Velocity-dependent friction
        self.viscous_friction = 0.1  # N·m/(rad/s)
        self.coulomb_friction = 0.5  # N·m
        # Motor dynamics (first-order lag)
        self.motor_time_constant = 0.02  # seconds
        self.prev_torque = np.zeros(n_joints)
        
    def apply_dynamics(self, torque_cmd, velocity, dt):
        """Apply actuator dynamics to commanded torque"""
        # 1. First-order motor dynamics (low-pass filter)
        alpha = dt / (self.motor_time_constant + dt)
        torque_filtered = alpha * torque_cmd + (1 - alpha) * self.prev_torque
        self.prev_torque = torque_filtered.copy()
        
        # 2. Velocity-dependent friction loss
        friction_torque = (self.viscous_friction * velocity + 
                         self.coulomb_friction * np.sign(velocity))
        
        # 3. Torque after friction (motor must overcome friction)
        torque_effective = torque_filtered - friction_torque
        
        # 4. Saturation with soft limits
        torque_limit = 30.0
        torque_effective = np.clip(torque_effective, -torque_limit, torque_limit)
        
        # 5. Add small noise (measurement/modeling error)
        noise = np.random.normal(0, 0.1, self.n_joints)
        torque_effective += noise
        
        return torque_effective


class HwangboExcitation:
    def __init__(self, dt=0.02):
        self.dt = dt
        
    def sinusoidal(self, t, freq, amp, phase=0):
        return amp * np.sin(2 * np.pi * freq * t + phase)
    
    def chirp(self, t, f0, f1, duration, amp):
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
        return amp * np.sin(phase)
    
    def multi_sine(self, t, freqs, amps):
        result = 0
        for freq, amp in zip(freqs, amps):
            result += self.sinusoidal(t, freq, amp)
        return result
    
    def generate_excitation_sequence(self, duration_per_phase=100.0):
        phases = []
        phases.append({'name': 'low_freq', 'duration': duration_per_phase,
            'freqs': [0.5, 1.0, 1.5, 2.0], 'amps': [0.5, 0.4, 0.3, 0.3], 'type': 'sinusoidal'})
        phases.append({'name': 'mid_freq', 'duration': duration_per_phase,
            'freqs': [2.0, 3.0, 4.0, 5.0], 'amps': [0.3, 0.25, 0.2, 0.15], 'type': 'sinusoidal'})
        phases.append({'name': 'high_freq', 'duration': duration_per_phase,
            'freqs': [5.0, 7.0, 10.0], 'amps': [0.15, 0.1, 0.08], 'type': 'sinusoidal'})
        phases.append({'name': 'chirp', 'duration': duration_per_phase * 2,
            'f0': 0.5, 'f1': 10.0, 'amp': 0.3, 'type': 'chirp'})
        phases.append({'name': 'saturation', 'duration': duration_per_phase,
            'freqs': [0.5, 1.0], 'amps': [0.8, 0.6], 'type': 'sinusoidal'})
        phases.append({'name': 'multi_sine', 'duration': duration_per_phase,
            'freqs': [0.7, 2.3, 5.1, 8.7], 'amps': [0.25, 0.2, 0.15, 0.1], 'type': 'multi_sine'})
        return phases


def collect_excitation_data():
    gym = gymapi.acquire_gym()
    
    sim_params = gymapi.SimParams()
    sim_params.dt = 0.005
    sim_params.substeps = 1
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    sim_params.use_gpu_pipeline = True
    
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    
    asset_root = os.path.join(LG_ROOT, "resources/robots/go2")
    asset_file = "urdf/go2.urdf"
    
    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
    asset_options.collapse_fixed_joints = True
    asset_options.replace_cylinder_with_capsule = True
    asset_options.flip_visual_attachments = True
    asset_options.density = 0.001
    asset_options.angular_damping = 0.0
    asset_options.linear_damping = 0.0
    asset_options.max_angular_velocity = 1000.0
    asset_options.max_linear_velocity = 1000.0
    asset_options.armature = 0.0
    asset_options.thickness = 0.01
    asset_options.disable_gravity = False
    
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = 1.0
    plane_params.dynamic_friction = 1.0
    gym.add_ground(sim, plane_params)
    
    env_lower = gymapi.Vec3(-1.0, -1.0, 0.0)
    env_upper = gymapi.Vec3(1.0, 1.0, 2.0)
    env = gym.create_env(sim, env_lower, env_upper, 1)
    
    start_pose = gymapi.Transform()
    start_pose.p = gymapi.Vec3(0.0, 0.0, 0.35)
    start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    
    actor = gym.create_actor(env, robot_asset, start_pose, "go2", 0, 0)
    
    props = gym.get_actor_dof_properties(env, actor)
    props['driveMode'].fill(gymapi.DOF_MODE_EFFORT)
    props['stiffness'].fill(0.0)
    props['damping'].fill(0.0)
    gym.set_actor_dof_properties(env, actor, props)
    
    default_angles = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 
                               0.1, 1.0, -1.5, -0.1, 1.0, -1.5], dtype=np.float32)
    
    Kp = 20.0
    Kd = 0.5
    
    gym.prepare_sim(sim)
    
    dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    dof_state = gymtorch.wrap_tensor(dof_state_tensor)
    
    # Actuator dynamics model
    actuator = ActuatorDynamics(n_joints=12)
    
    excitation = HwangboExcitation(dt=0.02)
    phases = excitation.generate_excitation_sequence(duration_per_phase=100.0)
    
    all_rows = []
    
    decimation = 4
    dt = 0.005
    policy_dt = dt * decimation
    
    print("="*60)
    print("Hwangbo-style Excitation Data Collection")
    print("With simulated actuator dynamics (friction, delay, noise)")
    print("="*60)
    
    total_duration = sum(p['duration'] for p in phases)
    print(f"Total duration: {total_duration}s ({total_duration/60:.1f} min)")
    print()
    
    # Reset
    dof_state[:, 0] = torch.from_numpy(default_angles).cuda()
    dof_state[:, 1] = 0.0
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))
    
    sim_time = 0.0
    phase_idx = 0
    phase_start_time = 0.0
    step_count = 0
    
    while phase_idx < len(phases):
        phase = phases[phase_idx]
        phase_time = sim_time - phase_start_time
        
        if phase_time >= phase['duration']:
            print(f"  Phase '{phase['name']}' complete: {len(all_rows):,} rows")
            phase_idx += 1
            phase_start_time = sim_time
            continue
        
        # Generate target positions
        target_offsets = np.zeros(12)
        for joint_idx in range(12):
            joint_phase = joint_idx * np.pi / 6
            
            if phase['type'] == 'sinusoidal':
                freq_idx = joint_idx % len(phase['freqs'])
                target_offsets[joint_idx] = excitation.sinusoidal(
                    phase_time, phase['freqs'][freq_idx], 
                    phase['amps'][freq_idx], joint_phase
                )
            elif phase['type'] == 'chirp':
                target_offsets[joint_idx] = excitation.chirp(
                    phase_time, phase['f0'], phase['f1'], 
                    phase['duration'], phase['amp']
                )
            elif phase['type'] == 'multi_sine':
                target_offsets[joint_idx] = excitation.multi_sine(
                    phase_time, phase['freqs'], phase['amps']
                )
        
        target_pos = default_angles + target_offsets.astype(np.float32)
        
        # Get current state
        gym.refresh_dof_state_tensor(sim)
        current_pos = dof_state[:, 0].cpu().numpy()
        current_vel = dof_state[:, 1].cpu().numpy()
        
        # Compute ideal PD torque
        pos_error = target_pos - current_pos
        torque_cmd = Kp * pos_error - Kd * current_vel
        
        # Apply actuator dynamics (friction, delay, noise)
        torque_actual = actuator.apply_dynamics(torque_cmd, current_vel, policy_dt)
        
        # Apply to simulation
        gym.set_dof_actuation_force_tensor(
            sim,
            gymtorch.unwrap_tensor(torch.from_numpy(torque_actual).float().cuda())
        )
        
        # Step simulation
        for _ in range(decimation):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
        
        sim_time += policy_dt
        step_count += 1
        
        # Save data
        row = {'time': sim_time, 'phase': phase['name']}
        for j in range(12):
            row[f'motorStatePos_{j}'] = current_pos[j]
            row[f'motorStateVel_{j}'] = current_vel[j]
            row[f'motorAction_{j}'] = target_pos[j]
            row[f'motorStateCur_{j}'] = torque_actual[j]  # Actual torque with dynamics
        all_rows.append(row)
        
        if step_count % 5000 == 0:
            print(f"    t={sim_time:.1f}s, rows={len(all_rows):,}")
    
    import pandas as pd
    df = pd.DataFrame(all_rows)
    
    save_path = os.path.expanduser("~/6619_ws/actuator_net/app/resources/hwangbo_excitation_data.csv")
    df.to_csv(save_path, index=False)
    
    print()
    print("="*60)
    print(f"Data collection complete!")
    print(f"Total rows: {len(df):,}")
    print(f"Saved to: {save_path}")
    print()
    print("Data statistics (joint 0):")
    print(f"  motorStatePos_0: [{df['motorStatePos_0'].min():.3f}, {df['motorStatePos_0'].max():.3f}]")
    print(f"  motorStateVel_0: [{df['motorStateVel_0'].min():.3f}, {df['motorStateVel_0'].max():.3f}]")
    print(f"  motorAction_0:   [{df['motorAction_0'].min():.3f}, {df['motorAction_0'].max():.3f}]")
    print(f"  motorStateCur_0: [{df['motorStateCur_0'].min():.3f}, {df['motorStateCur_0'].max():.3f}]")
    
    # Verify no simple leakage
    print()
    print("=== Verifying Actuator Dynamics Effect ===")
    pos_error = df['motorAction_0'].values - df['motorStatePos_0'].values
    vel = df['motorStateVel_0'].values
    torque = df['motorStateCur_0'].values
    
    computed_ideal = 20.0 * pos_error - 0.5 * vel
    diff = torque - computed_ideal
    print(f"  Ideal PD vs Actual diff: mean={diff.mean():.4f}, std={diff.std():.4f}")
    if diff.std() > 0.5:
        print("  ✓ Actuator dynamics adds complexity!")
    else:
        print("  ⚠ Dynamics effect may be too small")
    print("="*60)
    
    gym.destroy_sim(sim)
    return df

if __name__ == "__main__":
    collect_excitation_data()
