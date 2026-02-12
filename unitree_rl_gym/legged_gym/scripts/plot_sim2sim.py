import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

LOG_DIR = "/home/drl-68/Sim-to-Sim_Policy_Transfer_for_Learned-Legged-Locomotion/unitree_rl_gym/logs/sim2sim"
PLOT_DIR = "/home/drl-68/Sim-to-Sim_Policy_Transfer_for_Learned-Legged-Locomotion/unitree_rl_gym/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Transient period: t=3-4.5s (idx 150-225 for 50Hz)
TR_START, TR_END = 150, 225

def load_latest_log(pattern, prefer_fixed=True):
    """Load most recent log matching pattern"""
    files = sorted(glob(os.path.join(LOG_DIR, pattern)))
    if not files:
        files = sorted(glob(os.path.join(LOG_DIR, "cmd_switch", pattern)))
    
    if not files:
        return None
    
    if prefer_fixed and 'isaacgym' in pattern:
        fixed_files = [f for f in files if '20260201' in f]
        if fixed_files:
            return np.load(fixed_files[-1])
    
    return np.load(files[-1])

# ============================================================
# Plot 1: S2 Turn Comparison
# ============================================================
def plot_s2_comparison():
    print("Loading S2 Turn data...")
    isaac = load_latest_log("isaacgym_S2_turn*.npz")
    mujoco = load_latest_log("mujoco_S2_turn*.npz")
    
    if isaac is None or mujoco is None:
        print("Missing S2 Turn logs")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # vx
    axes[0].plot(isaac['time'], isaac['base_lin_vel'][:, 0], 'b-', label='Isaac Gym', linewidth=2)
    axes[0].plot(mujoco['time'], mujoco['base_lin_vel'][:, 0], 'r--', label='MuJoCo', linewidth=2)
    axes[0].axvline(x=3.0, color='gray', linestyle=':', label='Command Switch')
    axes[0].axhline(y=0.4, color='green', linestyle='--', alpha=0.5, label='Target vx')
    axes[0].set_ylabel('vx (m/s)')
    axes[0].legend(loc='upper right')
    axes[0].set_title('S2 Turn: Isaac Gym vs MuJoCo (Fixed - Heading Command Mode)')
    axes[0].set_ylim(-0.3, 0.7)
    
    # wz - NO sign flip
    axes[1].plot(isaac['time'], isaac['base_ang_vel'][:, 2], 'b-', label='Isaac Gym', linewidth=2)
    axes[1].plot(mujoco['time'], mujoco['base_ang_vel'][:, 2], 'r--', label='MuJoCo', linewidth=2)
    axes[1].axvline(x=3.0, color='gray', linestyle=':')
    axes[1].axhline(y=0.0, color='green', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('wz (rad/s)')
    axes[1].legend(loc='upper right')
    axes[1].set_title('Angular Velocity (Both responding to heading command)')
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].annotate('Isaac: CCW (-)', xy=(4.5, -0.7), fontsize=10, color='blue')
    axes[1].annotate('MuJoCo: CW (+)', xy=(4.5, 0.7), fontsize=10, color='red')
    
    # Torque
    isaac_torque = np.mean(np.abs(isaac['torques']), axis=1)
    mujoco_torque = np.mean(np.abs(mujoco['torques']), axis=1)
    axes[2].plot(isaac['time'], isaac_torque, 'b-', label='Isaac Gym', linewidth=2)
    axes[2].plot(mujoco['time'], mujoco_torque, 'r--', label='MuJoCo', linewidth=2)
    axes[2].axvline(x=3.0, color='gray', linestyle=':')
    axes[2].set_ylabel('Mean |τ| (N·m)')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'S2_turn_comparison.png'), dpi=150)
    plt.close()
    print("Saved: S2_turn_comparison.png")

# ============================================================
# Plot 2: Kp Ablation
# ============================================================
def plot_kp_ablation():
    kp_values = [10, 20, 30, 40]
    vx_means = [0.101, 0.413, 0.428, 0.407]
    vx_errors = [0.399, 0.087, 0.072, 0.093]
    isaac_vx = 0.453
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(kp_values))
    bars = ax.bar(x, vx_means, color=['#ff7f7f', '#7fbfff', '#7fff7f', '#ffbf7f'], edgecolor='black')
    ax.axhline(y=isaac_vx, color='blue', linestyle='--', linewidth=2, label=f'Isaac Gym (Kp=20): {isaac_vx} m/s')
    
    ax.set_xlabel('MuJoCo Kp (N·m/rad)')
    ax.set_ylabel('vx mean (m/s)')
    ax.set_title('Kp Ablation: Finding Equivalent Stiffness')
    ax.set_xticks(x)
    ax.set_xticklabels(kp_values)
    ax.legend()
    
    for i, (bar, err) in enumerate(zip(bars, vx_errors)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'err: {err:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'kp_ablation.png'), dpi=150)
    plt.close()
    print("Saved: kp_ablation.png")

# ============================================================
# Plot 3: Foot Friction Ablation
# ============================================================
def plot_foot_friction_ablation():
    friction_values = ['μ=0.2', 'μ=0.4\n(baseline)', 'μ=0.8']
    peak_torque = [19.66, 18.58, 14.62]
    max_roll = [17.9, 18.3, 7.0]
    max_pitch = [9.8, 13.5, 4.6]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(friction_values))
    colors = ['#ff7f7f', '#7fbfff', '#7fff7f']
    
    axes[0].bar(x, peak_torque, color=colors, edgecolor='black')
    axes[0].set_ylabel('Peak Torque (N·m)')
    axes[0].set_title('Peak Torque vs Foot Friction')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(friction_values)
    
    axes[1].bar(x, max_roll, color=colors, edgecolor='black')
    axes[1].set_ylabel('Max Roll (°)')
    axes[1].set_title('Max Roll vs Foot Friction')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(friction_values)
    
    axes[2].bar(x, max_pitch, color=colors, edgecolor='black')
    axes[2].set_ylabel('Max Pitch (°)')
    axes[2].set_title('Max Pitch vs Foot Friction')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(friction_values)
    
    plt.suptitle('S2 Turn: Foot Friction Ablation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'foot_friction_ablation.png'), dpi=150)
    plt.close()
    print("Saved: foot_friction_ablation.png")

# ============================================================
# Plot 4: Observation Delay Effect
# ============================================================
def plot_delay_ablation():
    delays = ['0 ms', '20 ms\n(FALL)', '40 ms']
    peak_torque = [15.32, 27.57, 26.21]
    wz_overshoot = [0.161, 1.348, 0.974]
    max_pitch = [5.0, 35.1, 28.4]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(delays))
    colors = ['#7fff7f', '#ff7f7f', '#ffbf7f']
    
    axes[0].bar(x, peak_torque, color=colors, edgecolor='black')
    axes[0].set_ylabel('Peak Torque (N·m)')
    axes[0].set_title('Peak Torque vs Delay')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(delays)
    
    axes[1].bar(x, wz_overshoot, color=colors, edgecolor='black')
    axes[1].set_ylabel('wz Overshoot (rad/s)')
    axes[1].set_title('Yaw Overshoot vs Delay')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(delays)
    
    axes[2].bar(x, max_pitch, color=colors, edgecolor='black')
    axes[2].set_ylabel('Max Pitch (°)')
    axes[2].set_title('Max Pitch vs Delay')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(delays)
    
    plt.suptitle('S2 Turn: Observation Delay Sensitivity', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'delay_ablation.png'), dpi=150)
    plt.close()
    print("Saved: delay_ablation.png")

# ============================================================
# Plot 5: Scenario Comparison - Using TRANSIENT period only
# ============================================================
def plot_scenario_comparison():
    print("\nLoading all scenarios for comparison...")
    
    scenarios = ['S1_stop', 'S2_turn', 'S3_lateral']
    labels = ['S1 Stop', 'S2 Turn', 'S3 Lateral']
    
    peak_torque_gap = []
    wz_gap = []
    
    for scn in scenarios:
        isaac = load_latest_log(f"isaacgym_{scn}*.npz")
        mujoco = load_latest_log(f"mujoco_{scn}*.npz")
        
        if isaac is None or mujoco is None:
            peak_torque_gap.append(0)
            wz_gap.append(0)
            continue
        
        # Use TRANSIENT period only (t=3-4.5s)
        isaac_torque = np.max(np.abs(isaac['torques'][TR_START:TR_END]))
        mujoco_torque = np.max(np.abs(mujoco['torques'][TR_START:TR_END]))
        
        isaac_wz = np.abs(isaac['base_ang_vel'][TR_START:TR_END, 2].mean())
        mujoco_wz = np.abs(mujoco['base_ang_vel'][TR_START:TR_END, 2].mean())
        
        peak_torque_gap.append(mujoco_torque - isaac_torque)
        wz_gap.append(mujoco_wz - isaac_wz)
        
        print(f"  {scn}: Isaac={isaac_torque:.2f}, MuJoCo={mujoco_torque:.2f}, Gap={mujoco_torque-isaac_torque:.2f} N·m")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(labels))
    
    # Peak Torque Gap
    colors = ['#7fbfff' if abs(v) < 4 else '#ff7f7f' for v in peak_torque_gap]
    axes[0].bar(x, peak_torque_gap, color=colors, edgecolor='black')
    axes[0].set_ylabel('Δ Peak Torque (N·m)')
    axes[0].set_title('Peak Torque Gap (MuJoCo - Isaac Gym)\nTransient Period: t=3-4.5s')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    for i, v in enumerate(peak_torque_gap):
        axes[0].text(i, v + 0.2 if v >= 0 else v - 0.5, f'{v:.1f}', ha='center', fontsize=10)
    
    # wz Gap
    colors = ['#7fbfff' if abs(v) < 0.2 else '#ffbf7f' for v in wz_gap]
    axes[1].bar(x, wz_gap, color=colors, edgecolor='black')
    axes[1].set_ylabel('Δ |wz| (rad/s)')
    axes[1].set_title('Angular Velocity Gap (|MuJoCo| - |Isaac|)\nTransient Period: t=3-4.5s')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    for i, v in enumerate(wz_gap):
        axes[1].text(i, v + 0.02 if v >= 0 else v - 0.05, f'{v:.2f}', ha='center', fontsize=10)
    
    plt.suptitle('Sim-to-Sim Gap Across Command Switching Scenarios (Fixed)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'scenario_comparison.png'), dpi=150)
    plt.close()
    print("Saved: scenario_comparison.png")

# ============================================================
# Plot 6: All Scenarios Overview
# ============================================================
def plot_all_scenarios_overview():
    print("\nGenerating all scenarios overview...")
    
    scenarios = [
        ('S1_stop', 'S1 Stop (vx: 0.6→0)'),
        ('S2_turn', 'S2 Turn (heading +180°)'),
        ('S3_lateral', 'S3 Lateral (vy: +0.3→-0.3)')
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Sim-to-Sim Comparison: Isaac Gym vs MuJoCo (Fixed)', fontsize=14, fontweight='bold')
    
    for row, (scn_key, scn_title) in enumerate(scenarios):
        isaac = load_latest_log(f"isaacgym_{scn_key}*.npz")
        mujoco = load_latest_log(f"mujoco_{scn_key}*.npz")
        
        if isaac is None or mujoco is None:
            continue
        
        # vx
        ax = axes[row, 0]
        ax.plot(isaac['time'], isaac['base_lin_vel'][:, 0], 'b-', label='Isaac Gym', linewidth=1.5)
        ax.plot(mujoco['time'], mujoco['base_lin_vel'][:, 0], 'r--', label='MuJoCo', linewidth=1.5)
        ax.axvline(x=3.0, color='gray', linestyle=':', alpha=0.7)
        ax.axvspan(3.0, 4.5, alpha=0.1, color='yellow', label='Transient')
        ax.set_ylabel(f'{scn_title}\nvx (m/s)')
        if row == 0:
            ax.set_title('Forward Velocity')
            ax.legend(loc='best', fontsize=8)
        if row == 2:
            ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
        
        # wz
        ax = axes[row, 1]
        ax.plot(isaac['time'], isaac['base_ang_vel'][:, 2], 'b-', label='Isaac Gym', linewidth=1.5)
        ax.plot(mujoco['time'], mujoco['base_ang_vel'][:, 2], 'r--', label='MuJoCo', linewidth=1.5)
        ax.axvline(x=3.0, color='gray', linestyle=':', alpha=0.7)
        ax.axvspan(3.0, 4.5, alpha=0.1, color='yellow')
        ax.set_ylabel('wz (rad/s)')
        if row == 0:
            ax.set_title('Angular Velocity')
        if row == 2:
            ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
        
        # Torque
        ax = axes[row, 2]
        isaac_torque = np.mean(np.abs(isaac['torques']), axis=1)
        mujoco_torque = np.mean(np.abs(mujoco['torques']), axis=1)
        ax.plot(isaac['time'], isaac_torque, 'b-', label='Isaac Gym', linewidth=1.5)
        ax.plot(mujoco['time'], mujoco_torque, 'r--', label='MuJoCo', linewidth=1.5)
        ax.axvline(x=3.0, color='gray', linestyle=':', alpha=0.7)
        ax.axvspan(3.0, 4.5, alpha=0.1, color='yellow')
        ax.set_ylabel('Mean |τ| (N·m)')
        if row == 0:
            ax.set_title('Torque')
        if row == 2:
            ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'all_scenarios_overview.png'), dpi=150)
    plt.close()
    print("Saved: all_scenarios_overview.png")

if __name__ == "__main__":
    print("=" * 60)
    print("Generating Sim-to-Sim Plots (FIXED - Transient Period)")
    print("=" * 60)
    
    plot_s2_comparison()
    plot_kp_ablation()
    plot_foot_friction_ablation()
    plot_delay_ablation()
    plot_scenario_comparison()
    plot_all_scenarios_overview()
    
    print("\n" + "=" * 60)
    print(f"All plots saved to: {PLOT_DIR}")
    print("=" * 60)