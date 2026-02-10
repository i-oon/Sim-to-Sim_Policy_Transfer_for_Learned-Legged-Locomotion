"""
Plot Transient Response Analysis - Including ActuatorNet V1/V2/V3 Comparison
Updated with evolution plots showing V2 instability and V3 fix
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import json

LEGGED_GYM_ROOT_DIR = os.path.expanduser("~/6619_ws/unitree_rl_gym")
LOG_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, "logs/transient_analysis")
PLOT_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'PD Only': '#1f77b4', 
    'PD + Residual': '#2ca02c', 
    'DR-trained Policy': '#d62728',
    'ActuatorNet v3': '#9467bd',
    'V1': '#ff7f0e',  # Orange
    'V2': '#ff1744',  # Red  
    'V3': '#9467bd',  # Purple
    'PD': '#1f77b4'
}

def load_data(scenario, config, suffix):
    """Load log data"""
    config_name = config.replace('.yaml', '')
    filepath = os.path.join(LOG_DIR, f"{scenario}_{config_name}_{suffix}_log.npz")
    if os.path.exists(filepath):
        return dict(np.load(filepath))
    return None

def load_actuatornet_v3_data(scenario):
    """Load ActuatorNet v3 data"""
    filepath = os.path.join(LOG_DIR, f"{scenario}_go2_actuatornetv3_log.npz")
    if os.path.exists(filepath):
        return dict(np.load(filepath))
    return None

def load_actuatornet_versions_data(scenario):
    """Load ActuatorNet V1, V2, V3 data for comparison"""
    versions_data = {}
    
    patterns = {
        'V1': f"{scenario}_go2_actuatornetv1_log.npz",
        'V2': f"{scenario}_go2_actuatornetv2_log.npz",
        'V3': f"{scenario}_go2_actuatornetv3_log.npz"
    }
    
    for version, filename in patterns.items():
        filepath = os.path.join(LOG_DIR, filename)
        if os.path.exists(filepath):
            versions_data[version] = dict(np.load(filepath))
    
    return versions_data

def load_metrics_from_json():
    """Load all metrics from JSON files"""
    metrics = {}
    
    controllers = [
        ('pd', 'PD Only'),
        ('residual', 'PD + Residual'),
        ('DR-trained Policy_pd', 'DR-trained Policy'),
        ('actuatornetv3', 'ActuatorNet v3')
    ]
    
    for scenario in ['S1', 'S2', 'S3']:
        metrics[scenario] = {}
        for ctrl_key, ctrl_name in controllers:
            filepath = os.path.join(LOG_DIR, f"{scenario}_go2_{ctrl_key}_metrics.json")
            if os.path.exists(filepath):
                with open(filepath) as f:
                    metrics[scenario][ctrl_name] = json.load(f)
    
    return metrics

def plot_scenario_comparison(scenario, title_suffix):
    """Plot comparison for one scenario"""
    
    data = {
        'PD Only': load_data(scenario, 'go2', 'pd'),
        'PD + Residual': load_data(scenario, 'go2', 'residual'),
        'DR-trained Policy': load_data(scenario, 'go2_rma', 'pd'),
        'ActuatorNet v3': load_actuatornet_v3_data(scenario)
    }
    
    available = {k: v for k, v in data.items() if v is not None}
    if not available:
        print(f"No data found for {scenario}")
        return
    
    print(f"Plotting {scenario}: {list(available.keys())}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{scenario} Command Switch: Transient Response Comparison\n{title_suffix}', 
                 fontsize=14, fontweight='bold')
    
    switch_time = 3.0
    
    # Plot 1: Velocity vx
    ax = axes[0, 0]
    for name, d in available.items():
        t = np.array(d['time']) - switch_time
        ax.plot(t, d['vx'], label=name, color=COLORS[name], linewidth=2)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7, label='Command Switch')
    last_data = list(available.values())[-1]
    ax.axhline(last_data['cmd_vx'][-1], color='black', linestyle=':', alpha=0.5, label='Target')
    ax.set_xlabel('Time relative to switch (s)')
    ax.set_ylabel('vx (m/s)')
    ax.set_title('Forward Velocity Response')
    ax.legend(loc='best', fontsize=8)
    ax.set_xlim(-1, 3)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Angular velocity wz
    ax = axes[0, 1]
    for name, d in available.items():
        t = np.array(d['time']) - switch_time
        ax.plot(t, d['wz'], label=name, color=COLORS[name], linewidth=2)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(last_data['cmd_wz'][-1], color='black', linestyle=':', alpha=0.5, label='Target')
    ax.set_xlabel('Time relative to switch (s)')
    ax.set_ylabel('wz (rad/s)')
    ax.set_title('Yaw Rate Response')
    ax.legend(loc='best', fontsize=8)
    ax.set_xlim(-1, 3)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Torque
    ax = axes[1, 0]
    for name, d in available.items():
        t = np.array(d['time']) - switch_time
        ax.plot(t, d['torque_max'], label=name, color=COLORS[name], linewidth=2)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Time relative to switch (s)')
    ax.set_ylabel('Max Joint Torque (N·m)')
    ax.set_title('Peak Torque During Transient')
    ax.legend(loc='best', fontsize=8)
    ax.set_xlim(-1, 3)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Body angles
    ax = axes[1, 1]
    for name, d in available.items():
        t = np.array(d['time']) - switch_time
        ax.plot(t, np.abs(d['pitch']), label=f'{name} pitch', color=COLORS[name], 
                linewidth=2, linestyle='-')
        ax.plot(t, np.abs(d['roll']), label=f'{name} roll', color=COLORS[name], 
                linewidth=2, linestyle='--', alpha=0.7)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Time relative to switch (s)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Body Stability (Pitch & Roll)')
    ax.legend(loc='best', ncol=2, fontsize=7)
    ax.set_xlim(-1, 3)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(PLOT_DIR, f'transient_{scenario}_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_metrics_comparison():
    """Plot bar chart comparing metrics - load from JSON"""
    
    all_metrics = load_metrics_from_json()
    controllers = ['PD Only', 'PD + Residual', 'DR-trained Policy', 'ActuatorNet v3']
    
    scenarios_display = {
        'S1': 'S1 Stop',
        'S2': 'S2 Turn', 
        'S3': 'S3 Lateral'
    }
    
    for scenario, display_name in scenarios_display.items():
        if scenario not in all_metrics:
            continue
        
        scn_metrics = all_metrics[scenario]
        
        # Extract metrics
        metrics_to_plot = {
            'Peak Torque (N·m)': {},
            'Peak Pitch (°)': {},
            'Peak Roll (°)': {}
        }
        
        for ctrl in controllers:
            if ctrl in scn_metrics:
                m = scn_metrics[ctrl]
                metrics_to_plot['Peak Torque (N·m)'][ctrl] = m.get('torque', {}).get('peak', 0)
                metrics_to_plot['Peak Pitch (°)'][ctrl] = m.get('pitch', {}).get('peak', 0)
                metrics_to_plot['Peak Roll (°)'][ctrl] = m.get('roll', {}).get('peak', 0)
        
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        fig.suptitle(f'{display_name}: Transient Metrics Comparison', fontsize=14, fontweight='bold')
        
        for ax, (metric_name, values) in zip(axes, metrics_to_plot.items()):
            avail_ctrls = [c for c in controllers if c in values and values[c] > 0]
            x = np.arange(len(avail_ctrls))
            bars = ax.bar(x, [values[c] for c in avail_ctrls], 
                         color=[COLORS[c] for c in avail_ctrls])
            
            ax.set_xticks(x)
            ax.set_xticklabels(avail_ctrls, rotation=20, ha='right', fontsize=9)
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            
            for bar, val in zip(bars, [values[c] for c in avail_ctrls]):
                height = bar.get_height()
                ax.annotate(f'{val:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
            
            # Highlight best (lowest)
            vals_list = [values[c] for c in avail_ctrls]
            if vals_list:
                best_idx = np.argmin(vals_list)
                bars[best_idx].set_edgecolor('gold')
                bars[best_idx].set_linewidth(3)
        
        plt.tight_layout()
        
        save_path = os.path.join(PLOT_DIR, f'transient_{scenario}_metrics.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()

def plot_all_scenarios_overview():
    """Create overview plot with all scenarios"""
    
    scenarios = ['S1', 'S2', 'S3']
    titles = {
        'S1': 'S1 Stop (vx: 0.6→0)',
        'S2': 'S2 Turn (wz: 0→1.0)',
        'S3': 'S3 Lateral (vy: +0.3→-0.3)'
    }
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('Transient Response Overview: All Scenarios & Controllers', fontsize=16, fontweight='bold')
    
    switch_time = 3.0
    
    for row, scenario in enumerate(scenarios):
        data = {
            'PD Only': load_data(scenario, 'go2', 'pd'),
            'PD + Residual': load_data(scenario, 'go2', 'residual'),
            'DR-trained Policy': load_data(scenario, 'go2_rma', 'pd'),
            'ActuatorNet v3': load_actuatornet_v3_data(scenario)
        }
        available = {k: v for k, v in data.items() if v is not None}
        
        if not available:
            continue
        
        # Column 0: Velocity
        ax = axes[row, 0]
        for name, d in available.items():
            t = np.array(d['time']) - switch_time
            ax.plot(t, d['vx'], label=name, color=COLORS[name], linewidth=1.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylabel(f'{titles[scenario]}\nvx (m/s)')
        if row == 0:
            ax.set_title('Forward Velocity')
            ax.legend(loc='best', fontsize=7)
        if row == 2:
            ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
        
        # Column 1: Torque
        ax = axes[row, 1]
        for name, d in available.items():
            t = np.array(d['time']) - switch_time
            ax.plot(t, d['torque_max'], label=name, color=COLORS[name], linewidth=1.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylabel('Torque (N·m)')
        if row == 0:
            ax.set_title('Peak Torque')
        if row == 2:
            ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
        
        # Column 2: Pitch
        ax = axes[row, 2]
        for name, d in available.items():
            t = np.array(d['time']) - switch_time
            ax.plot(t, np.abs(d['pitch']), label=name, color=COLORS[name], linewidth=1.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylabel('|Pitch| (°)')
        if row == 0:
            ax.set_title('Body Pitch')
        if row == 2:
            ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(PLOT_DIR, 'transient_overview_all.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_actuatornet_evolution():
    """Plot ActuatorNet V1 → V2 → V3 evolution across scenarios"""
    
    scenarios = ['S1', 'S2', 'S3']
    scenario_names = {
        'S1': 'S1 Stop (vx: 0.6→0)',
        'S2': 'S2 Turn (wz: 0→1.0)',
        'S3': 'S3 Lateral (vy: ±0.3)'
    }
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('ActuatorNet Evolution: V1 → V2 → V3\nData Diversity Fixes S2 Instability', 
                 fontsize=16, fontweight='bold')
    
    switch_time = 3.0
    
    for row, scenario in enumerate(scenarios):
        versions_data = load_actuatornet_versions_data(scenario)
        
        if not versions_data:
            print(f"Warning: No ActuatorNet data found for {scenario}")
            continue
        
        # Column 0: Torque
        ax = axes[row, 0]
        for version in ['V1', 'V2', 'V3']:
            if version in versions_data:
                d = versions_data[version]
                t = np.array(d['time']) - switch_time
                label = f"{version}"
                if version == 'V2' and scenario == 'S2':
                    label += " (Unstable)"
                ax.plot(t, d['torque_max'], label=label, 
                       color=COLORS[version], linewidth=2)
        
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(30, color='red', linestyle=':', alpha=0.3, label='Limit (30 N·m)')
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylabel(f'{scenario_names[scenario]}\nTorque (N·m)')
        if row == 0:
            ax.set_title('Peak Torque', fontweight='bold')
            ax.legend(loc='best', fontsize=9)
        if row == 2:
            ax.set_xlabel('Time relative to switch (s)')
        ax.grid(True, alpha=0.3)
        
        # Column 1: Pitch
        ax = axes[row, 1]
        for version in ['V1', 'V2', 'V3']:
            if version in versions_data:
                d = versions_data[version]
                t = np.array(d['time']) - switch_time
                ax.plot(t, np.abs(d['pitch']), label=version, 
                       color=COLORS[version], linewidth=2)
        
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        if scenario == 'S2':
            ax.axhline(15, color='red', linestyle=':', alpha=0.3, label='Unstable threshold')
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylabel('|Pitch| (°)')
        if row == 0:
            ax.set_title('Body Pitch', fontweight='bold')
            ax.legend(loc='best', fontsize=9)
        if row == 2:
            ax.set_xlabel('Time relative to switch (s)')
        ax.grid(True, alpha=0.3)
        
        # Column 2: Roll
        ax = axes[row, 2]
        for version in ['V1', 'V2', 'V3']:
            if version in versions_data:
                d = versions_data[version]
                t = np.array(d['time']) - switch_time
                ax.plot(t, np.abs(d['roll']), label=version, 
                       color=COLORS[version], linewidth=2)
        
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylabel('|Roll| (°)')
        if row == 0:
            ax.set_title('Body Roll', fontweight='bold')
            ax.legend(loc='best', fontsize=9)
        if row == 2:
            ax.set_xlabel('Time relative to switch (s)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(PLOT_DIR, 'actuatornet_v1_v2_v3_evolution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_actuatornet_s2_focus():
    """Focused plot on S2 Turn showing V2 instability and V3 fix"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('S2 Turn: ActuatorNet V2 Instability → V3 Fix\n' + 
                 'V2: Policy-driven data (94.55% R²) | V3: Hwangbo excitation (99.80% R²)',
                 fontsize=14, fontweight='bold')
    
    switch_time = 3.0
    
    # Load data
    versions_data = load_actuatornet_versions_data('S2')
    pd_data = load_data('S2', 'go2', 'pd')
    
    if pd_data:
        versions_data['PD'] = pd_data
    
    # Plot 1: vx
    ax = axes[0, 0]
    for version in ['PD', 'V1', 'V2', 'V3']:
        if version in versions_data:
            d = versions_data[version]
            t = np.array(d['time']) - switch_time
            label = f"ActuatorNet {version}" if version != 'PD' else 'PD Only (Baseline)'
            ax.plot(t, d['vx'], label=label, color=COLORS[version], linewidth=2)
    
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.4, color='black', linestyle=':', alpha=0.5, label='Target')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('vx (m/s)')
    ax.set_title('Forward Velocity')
    ax.legend(loc='best', fontsize=8)
    ax.set_xlim(-0.5, 2.5)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Torque
    ax = axes[0, 1]
    for version in ['PD', 'V2', 'V3']:
        if version in versions_data:
            d = versions_data[version]
            t = np.array(d['time']) - switch_time
            label = version
            if version == 'V2':
                label += " (28.49 N·m peak!)"
            ax.plot(t, d['torque_max'], label=label, color=COLORS[version], linewidth=2)
    
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(30, color='red', linestyle=':', alpha=0.3, label='Saturation limit')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Peak Torque (N·m)')
    ax.set_title('Peak Joint Torque')
    ax.legend(loc='best', fontsize=8)
    ax.set_xlim(-0.5, 2.5)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Pitch
    ax = axes[1, 0]
    for version in ['PD', 'V2', 'V3']:
        if version in versions_data:
            d = versions_data[version]
            t = np.array(d['time']) - switch_time
            label = version
            if version == 'V2':
                label += " (17.8° peak!)"
            ax.plot(t, np.abs(d['pitch']), label=label, color=COLORS[version], linewidth=2)
    
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(15, color='red', linestyle=':', alpha=0.3, label='Instability threshold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|Pitch| (°)')
    ax.set_title('Body Pitch (V2 Unstable!)')
    ax.legend(loc='best', fontsize=8)
    ax.set_xlim(-0.5, 2.5)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Roll
    ax = axes[1, 1]
    for version in ['PD', 'V2', 'V3']:
        if version in versions_data:
            d = versions_data[version]
            t = np.array(d['time']) - switch_time
            ax.plot(t, np.abs(d['roll']), label=version, color=COLORS[version], linewidth=2)
    
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|Roll| (°)')
    ax.set_title('Body Roll')
    ax.legend(loc='best', fontsize=8)
    ax.set_xlim(-0.5, 2.5)
    ax.grid(True, alpha=0.3)
    
    # Add text box explaining the fix
    textstr = 'V2→V3 Improvements:\n' + \
              '• Pitch: 17.8° → 2.8° (84% ↓)\n' + \
              '• Torque: 28.49 → 13.34 N·m (53% ↓)\n' + \
              '• Data: Policy-driven → Hwangbo excitation'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    fig.text(0.98, 0.02, textstr, fontsize=10, verticalalignment='bottom',
             horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    save_path = os.path.join(PLOT_DIR, 'actuatornet_s2_v2_v3_fix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    print("=" * 60)
    print("Generating Transient Analysis Plots")
    print("=" * 60)
    
    # Original plots
    plot_scenario_comparison('S1', 'vx: 0.6 → 0.0 m/s')
    plot_scenario_comparison('S2', 'wz: 0.0 → 1.0 rad/s')
    plot_scenario_comparison('S3', 'vy: +0.3 → -0.3 m/s')
    
    plot_metrics_comparison()
    plot_all_scenarios_overview()
    
    # New ActuatorNet evolution plots
    print("\n" + "=" * 60)
    print("Generating ActuatorNet V1/V2/V3 Evolution Plots")
    print("=" * 60)
    
    plot_actuatornet_evolution()
    plot_actuatornet_s2_focus()
    
    print("\n" + "=" * 60)
    print(f"All plots saved to: {PLOT_DIR}")
    print("=" * 60)
    print("\nNew plots added:")
    print("  - actuatornet_v1_v2_v3_evolution.png (3x3 grid)")
    print("  - actuatornet_s2_v2_v3_fix.png (S2 focused)")