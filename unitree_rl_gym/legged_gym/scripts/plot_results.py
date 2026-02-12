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

def load_latest_log(pattern):
    """Load most recent log matching pattern"""
    files = sorted(glob(os.path.join(LOG_DIR, pattern)))
    if not files:
        files = sorted(glob(os.path.join(LOG_DIR, "cmd_switch", pattern)))
    if files:
        return np.load(files[-1])
    return None

# ============================================================
# Plot 1: Command Switching Time Series (S2 Turn)
# ============================================================
def plot_s2_comparison():
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
    axes[0].set_title('S2 Turn Shock: Isaac Gym vs MuJoCo')
    
    # wz
    axes[1].plot(isaac['time'], -isaac['base_ang_vel'][:, 2], 'b-', label='Isaac Gym', linewidth=2)
    axes[1].plot(mujoco['time'], mujoco['base_ang_vel'][:, 2], 'r--', label='MuJoCo', linewidth=2)
    axes[1].axvline(x=3.0, color='gray', linestyle=':')
    axes[1].axhline(y=0.0, color='green', linestyle='--', alpha=0.5)
    axes[1].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Target wz')
    axes[1].set_ylabel('wz (rad/s)')
    axes[1].legend(loc='upper right')
    axes[1].set_title('wz (Note: Isaac Gym sign flipped for comparison)')
    
    # Torque (mean across joints)
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
# Plot 2: Kp Ablation Bar Chart
# ============================================================
def plot_kp_ablation():
    kp_values = [10, 20, 30, 40]
    vx_means = [0.101, 0.413, 0.428, 0.407]
    vx_errors = [0.399, 0.087, 0.072, 0.093]
    isaac_vx = 0.453
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(kp_values))
    bars = ax.bar(x, vx_means, color=['#ff7f7f', '#7fbfff', '#7fff7f', '#ffbf7f'], edgecolor='black')
    
    # Isaac Gym reference line
    ax.axhline(y=isaac_vx, color='blue', linestyle='--', linewidth=2, label=f'Isaac Gym (Kp=20): {isaac_vx} m/s')
    
    ax.set_xlabel('MuJoCo Kp (N·m/rad)')
    ax.set_ylabel('vx mean (m/s)')
    ax.set_title('Kp Ablation: Finding Equivalent Stiffness')
    ax.set_xticks(x)
    ax.set_xticklabels(kp_values)
    ax.legend()
    
    # Annotate bars
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
    
    # S2 Turn metrics
    peak_torque = [19.66, 18.58, 14.62]
    max_roll = [17.9, 18.3, 7.0]
    max_pitch = [9.8, 13.5, 4.6]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    x = np.arange(len(friction_values))
    colors = ['#ff7f7f', '#7fbfff', '#7fff7f']
    
    # Peak Torque
    axes[0].bar(x, peak_torque, color=colors, edgecolor='black')
    axes[0].set_ylabel('Peak Torque (N·m)')
    axes[0].set_title('Peak Torque vs Foot Friction')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(friction_values)
    
    # Max Roll
    axes[1].bar(x, max_roll, color=colors, edgecolor='black')
    axes[1].set_ylabel('Max Roll (°)')
    axes[1].set_title('Max Roll vs Foot Friction')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(friction_values)
    
    # Max Pitch
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
    
    # New values from experiment
    peak_torque = [15.32, 27.57, 26.21]  # 20ms caused fall, torque before fall
    wz_overshoot = [0.161, 1.348, 0.974]
    max_pitch = [5.0, 35.1, 28.4]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    x = np.arange(len(delays))
    colors = ['#7fff7f', '#ffbf7f', '#ff7f7f']
    
    # Peak Torque
    axes[0].bar(x, peak_torque, color=colors, edgecolor='black')
    axes[0].set_ylabel('Peak Torque (N·m)')
    axes[0].set_title('Peak Torque vs Delay')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(delays)
    for i, v in enumerate(peak_torque):
        axes[0].text(i, v + 0.5, f'+{(v/peak_torque[0]-1)*100:.0f}%' if i > 0 else '', ha='center')
    
    # wz Overshoot
    axes[1].bar(x, wz_overshoot, color=colors, edgecolor='black')
    axes[1].set_ylabel('wz Overshoot (rad/s)')
    axes[1].set_title('Yaw Overshoot vs Delay')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(delays)
    
    # Max Pitch
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
# Plot 5: Summary Comparison (All Scenarios)
# ============================================================
def plot_scenario_comparison():
    scenarios = ['S1 Stop', 'S2 Turn', 'S3 Lateral']
    
    # Isaac Gym vs MuJoCo gaps
    peak_torque_gap = [0.72, 7.18, 0.90]
    max_pitch_gap = [-0.6, 7.2, -1.9]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(scenarios))
    
    # Peak Torque Gap
    colors = ['#7fbfff' if v < 2 else '#ff7f7f' for v in peak_torque_gap]
    axes[0].bar(x, peak_torque_gap, color=colors, edgecolor='black')
    axes[0].set_ylabel('Δ Peak Torque (N·m)')
    axes[0].set_title('Peak Torque Gap (MuJoCo - Isaac Gym)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(scenarios)
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Max Pitch Gap
    colors = ['#7fbfff' if abs(v) < 2 else '#ff7f7f' for v in max_pitch_gap]
    axes[1].bar(x, max_pitch_gap, color=colors, edgecolor='black')
    axes[1].set_ylabel('Δ Max Pitch (°)')
    axes[1].set_title('Max Pitch Gap (MuJoCo - Isaac Gym)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scenarios)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.suptitle('Sim-to-Sim Gap Across Command Switching Scenarios', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'scenario_comparison.png'), dpi=150)
    plt.close()
    print("Saved: scenario_comparison.png")

# ============================================================
# Plot 6: Kd Ablation Bar Chart
# ============================================================
def plot_kd_ablation():
    kd_values = ['0.3', '0.5\n(baseline)', '0.8', '1.0']
    wz_overshoot = [0.167, 0.161, 0.079, 0.090]  # absolute values
    max_roll = [6.3, 5.1, 6.9, 5.1]
    peak_torque = [14.82, 15.32, 15.25, 15.22]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    x = np.arange(len(kd_values))
    colors = ['#ffbf7f', '#7fbfff', '#7fff7f', '#ff7f7f']

    # wz Overshoot
    axes[0].bar(x, wz_overshoot, color=colors, edgecolor='black')
    axes[0].set_ylabel('|wz overshoot| (rad/s)')
    axes[0].set_title('Yaw Overshoot vs Kd')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(kd_values)

    # Max Roll
    axes[1].bar(x, max_roll, color=colors, edgecolor='black')
    axes[1].set_ylabel('Max Roll (°)')
    axes[1].set_title('Max Roll vs Kd')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(kd_values)

    # Peak Torque
    axes[2].bar(x, peak_torque, color=colors, edgecolor='black')
    axes[2].set_ylabel('Peak Torque (N·m)')
    axes[2].set_title('Peak Torque vs Kd')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(kd_values)

    plt.suptitle('S2 Turn: Kd (Damping) Ablation — Minimal Effect', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'kd_ablation.png'), dpi=150)
    plt.close()
    print("Saved: kd_ablation.png")

# ============================================================
# Plot 7: Mass Perturbation Bar Chart
# ============================================================
def plot_mass_ablation():
    mass_labels = ['-1 kg\n(5.92)', 'Baseline\n(6.92)', '+1 kg\n(7.92)', '+2 kg\n(8.92)']
    vx_steady = [0.363, 0.353, 0.358, 0.237]
    max_pitch = [3.8, 5.0, 15.5, 11.2]
    peak_torque = [13.63, 15.32, 23.11, 21.56]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    x = np.arange(len(mass_labels))
    colors = ['#7fff7f', '#7fbfff', '#ff7f7f', '#ff7f7f']

    # vx Steady-State
    axes[0].bar(x, vx_steady, color=colors, edgecolor='black')
    axes[0].axhline(y=0.4, color='green', linestyle='--', alpha=0.5, label='Target vx')
    axes[0].set_ylabel('vx (m/s)')
    axes[0].set_title('Steady-State Velocity')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(mass_labels)
    axes[0].legend(fontsize=9)
    for i, v in enumerate(vx_steady):
        if i >= 2:
            pct = (v / vx_steady[1] - 1) * 100
            axes[0].text(i, v + 0.01, f'{pct:+.0f}%', ha='center', fontsize=10)

    # Max Pitch
    axes[1].bar(x, max_pitch, color=colors, edgecolor='black')
    axes[1].set_ylabel('Max Pitch (°)')
    axes[1].set_title('Max Pitch (Transient)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(mass_labels)
    for i, v in enumerate(max_pitch):
        if v > 10:
            axes[1].text(i, v + 0.5, '⚠️', ha='center', fontsize=14)

    # Peak Torque
    axes[2].bar(x, peak_torque, color=colors, edgecolor='black')
    axes[2].set_ylabel('Peak Torque (N·m)')
    axes[2].set_title('Peak Torque')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(mass_labels)
    for i, v in enumerate(peak_torque):
        if i >= 2:
            pct = (v / peak_torque[1] - 1) * 100
            axes[2].text(i, v + 0.5, f'+{pct:.0f}%', ha='center', fontsize=10)

    plt.suptitle('S2 Turn: Mass Perturbation — Asymmetric Sensitivity', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'mass_ablation.png'), dpi=150)
    plt.close()
    print("Saved: mass_ablation.png")

# ============================================================
# Plot 8: Joint Friction Ablation Bar Chart
# ============================================================
def plot_joint_friction_ablation():
    friction_labels = ['No friction\n(0.0 / 0.0)', 'Baseline\n(0.1 / 0.2)', 'High friction\n(0.3 / 0.5)']
    vx_steady = [0.361, 0.353, 0.326]
    max_roll = [9.3, 5.1, 3.4]
    wz_overshoot = [0.147, 0.161, 0.054]  # absolute values

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    x = np.arange(len(friction_labels))
    colors = ['#ff7f7f', '#7fbfff', '#7fff7f']

    # vx Steady-State
    axes[0].bar(x, vx_steady, color=colors, edgecolor='black')
    axes[0].set_ylabel('vx (m/s)')
    axes[0].set_title('Steady-State Velocity')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(friction_labels, fontsize=9)

    # Max Roll
    axes[1].bar(x, max_roll, color=colors, edgecolor='black')
    axes[1].set_ylabel('Max Roll (°)')
    axes[1].set_title('Max Roll (Transient)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(friction_labels, fontsize=9)
    axes[1].annotate('+82%', xy=(0, 9.3), xytext=(0, 10.5),
                     ha='center', fontsize=10, color='red',
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    axes[1].annotate('-33%', xy=(2, 3.4), xytext=(2, 4.6),
                     ha='center', fontsize=10, color='green',
                     arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    # wz Overshoot
    axes[2].bar(x, wz_overshoot, color=colors, edgecolor='black')
    axes[2].set_ylabel('|wz overshoot| (rad/s)')
    axes[2].set_title('Yaw Overshoot')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(friction_labels, fontsize=9)

    plt.suptitle('S2 Turn: Joint Friction Ablation (damping / frictionloss)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'joint_friction_ablation.png'), dpi=150)
    plt.close()
    print("Saved: joint_friction_ablation.png")

# ============================================================
# Plot 9: Motor Command Delay vs Observation Delay Comparison
# ============================================================
def plot_motor_delay_ablation():
    delay_labels = ['0 ms', '20 ms', '40 ms']

    # Motor delay data
    motor_pitch = [5.0, 6.1, 18.9]
    motor_torque = [15.32, 19.44, 15.32]  # 40ms exploded, cap at baseline for visual
    motor_wz_overshoot = [0.161, 0.241, 0.246]
    motor_fallen = [False, False, True]

    # Observation delay data (Stage 3)
    obs_pitch = [5.0, 35.1, 28.4]
    obs_torque = [15.32, 27.57, 26.21]
    obs_wz_overshoot = [0.161, 1.348, 0.974]
    obs_fallen = [False, True, False]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.arange(len(delay_labels))
    width = 0.35

    # Max Pitch comparison
    axes[0].bar(x - width/2, obs_pitch, width, label='Obs delay (input)', color='#ff7f7f', edgecolor='black')
    axes[0].bar(x + width/2, motor_pitch, width, label='Motor delay (output)', color='#7fbfff', edgecolor='black')
    axes[0].set_ylabel('Max Pitch (°)')
    axes[0].set_title('Max Pitch')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(delay_labels)
    axes[0].legend(fontsize=9)
    # Mark falls
    for i, fallen in enumerate(obs_fallen):
        if fallen:
            axes[0].text(i - width/2, obs_pitch[i] + 1, 'FELL', ha='center', fontsize=9, color='red', fontweight='bold')
    for i, fallen in enumerate(motor_fallen):
        if fallen:
            axes[0].text(i + width/2, motor_pitch[i] + 1, 'FELL', ha='center', fontsize=9, color='red', fontweight='bold')

    # Peak Torque comparison
    axes[1].bar(x - width/2, obs_torque, width, label='Obs delay (input)', color='#ff7f7f', edgecolor='black')
    axes[1].bar(x + width/2, motor_torque, width, label='Motor delay (output)', color='#7fbfff', edgecolor='black')
    axes[1].set_ylabel('Peak Torque (N·m)')
    axes[1].set_title('Peak Torque')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(delay_labels)
    axes[1].legend(fontsize=9)
    # Note for 40ms motor (exploded)
    axes[1].annotate('exploded\n(capped)', xy=(2 + width/2, motor_torque[2]),
                     xytext=(2 + width/2, motor_torque[2] + 3),
                     ha='center', fontsize=8, color='gray',
                     arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    # wz Overshoot comparison
    axes[2].bar(x - width/2, obs_wz_overshoot, width, label='Obs delay (input)', color='#ff7f7f', edgecolor='black')
    axes[2].bar(x + width/2, motor_wz_overshoot, width, label='Motor delay (output)', color='#7fbfff', edgecolor='black')
    axes[2].set_ylabel('|wz overshoot| (rad/s)')
    axes[2].set_title('Yaw Overshoot')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(delay_labels)
    axes[2].legend(fontsize=9)

    plt.suptitle('S2 Turn: Observation Delay (Stage 3) vs Motor Delay (Stage 3.5)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'motor_delay_ablation.png'), dpi=150)
    plt.close()
    print("Saved: motor_delay_ablation.png")

if __name__ == "__main__":
    print("Generating plots...")
    plot_s2_comparison()
    plot_kp_ablation()
    plot_foot_friction_ablation()
    plot_delay_ablation()
    plot_scenario_comparison()
    plot_kd_ablation()
    plot_mass_ablation()
    plot_joint_friction_ablation()
    plot_motor_delay_ablation()
    print(f"\nAll plots saved to: {PLOT_DIR}")