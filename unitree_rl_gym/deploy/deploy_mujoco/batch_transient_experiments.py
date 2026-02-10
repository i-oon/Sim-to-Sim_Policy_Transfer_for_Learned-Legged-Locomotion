"""
Batch run all transient experiments and generate comparison tables
"""
import os
import sys
import subprocess
import numpy as np
from datetime import datetime

LEGGED_GYM_ROOT_DIR = os.path.expanduser("~/6619_ws/unitree_rl_gym")
sys.path.append(os.path.join(LEGGED_GYM_ROOT_DIR, "deploy/deploy_mujoco"))

SCENARIOS = ['S1_stop', 'S2_turn', 'S3_lateral']

def run_experiment(script_name, scenario, extra_args=""):
    """Run a single experiment"""
    script_path = os.path.join(LEGGED_GYM_ROOT_DIR, "deploy/deploy_mujoco", script_name)
    cmd = f"python {script_path} --scenario {scenario} --no_viewer {extra_args}"
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

def run_all_experiments():
    """Run all experiment combinations"""
    results = {
        'pd': {},
        'residual': {},
        'rma': {}
    }
    
    for scenario in SCENARIOS:
        print(f"\n\n{'#'*60}")
        print(f"# SCENARIO: {scenario}")
        print(f"{'#'*60}\n")
        
        # 1. PD Only (Baseline)
        print(f"\n--- PD Only ---")
        success = run_experiment("deploy_residual_cmd_switch.py", scenario, "--pd")
        results['pd'][scenario] = success
        
        # 2. PD + Residual
        print(f"\n--- PD + Residual ---")
        success = run_experiment("deploy_residual_cmd_switch.py", scenario, "")
        results['residual'][scenario] = success
        
        # 3. DR-trained (RMA)
        print(f"\n--- DR-trained ---")
        success = run_experiment("deploy_rma_cmd_switch.py", scenario, "")
        results['rma'][scenario] = success
    
    # Summary
    print(f"\n\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    for controller in ['pd', 'residual', 'rma']:
        print(f"\n{controller.upper()}:")
        for scenario in SCENARIOS:
            status = "✓" if results[controller][scenario] else "✗"
            print(f"  {scenario}: {status}")

def load_latest_log(controller, scenario):
    """Load the most recent log file for a given controller and scenario"""
    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs/sim2sim/cmd_switch")
    
    # Find matching files
    if controller == 'pd':
        pattern = f"pd_{scenario}_"
    elif controller == 'residual':
        pattern = f"residual_{scenario}_"
    elif controller == 'rma':
        pattern = f"rma_{scenario}_"
    elif controller == 'mujoco':
        pattern = f"mujoco_{scenario}_"
    else:
        return None
    
    files = [f for f in os.listdir(log_dir) if f.startswith(pattern) and f.endswith('.npz')]
    
    if not files:
        return None
    
    # Get most recent
    files.sort(reverse=True)
    log_path = os.path.join(log_dir, files[0])
    
    return np.load(log_path, allow_pickle=True)

def analyze_transient_from_log(log_data, scenario_key, switch_time=3.0):
    """Extract transient metrics from saved log"""
    from transient_metrics import analyze_scenario_transients
    return analyze_scenario_transients(log_data, scenario_key, switch_time)

def generate_comparison_table():
    """Generate markdown table comparing all methods"""
    print(f"\n\n{'='*60}")
    print("GENERATING COMPARISON TABLES")
    print(f"{'='*60}\n")
    
    for scenario in SCENARIOS:
        print(f"\n### {scenario.upper()} Comparison\n")
        
        # Load all logs
        logs = {}
        for controller in ['pd', 'residual', 'rma']:
            log = load_latest_log(controller, scenario)
            if log:
                logs[controller] = log
        
        if not logs:
            print(f"No logs found for {scenario}")
            continue
        
        # Extract metrics
        print("| Metric | PD Only | PD + Residual | DR-trained |")
        print("|--------|---------|---------------|------------|")
        
        # Max torque
        row = "| **Peak torque** |"
        for controller in ['pd', 'residual', 'rma']:
            if controller in logs:
                val = np.max(np.abs(logs[controller]['torques']))
                row += f" {val:.2f} N·m |"
            else:
                row += " N/A |"
        print(row)
        
        # Max pitch
        row = "| **Peak pitch** |"
        for controller in ['pd', 'residual', 'rma']:
            if controller in logs:
                pitch = logs[controller]['pitch']
                val = np.max(np.abs(pitch))
                row += f" {val:.1f}° |"
            else:
                row += " N/A |"
        print(row)
        
        # Max roll
        row = "| **Peak roll** |"
        for controller in ['pd', 'residual', 'rma']:
            if controller in logs:
                roll = logs[controller]['roll']
                val = np.max(np.abs(roll))
                row += f" {val:.1f}° |"
            else:
                row += " N/A |"
        print(row)
        
        # Transient metrics
        for controller in ['pd', 'residual', 'rma']:
            if controller in logs:
                trans = analyze_transient_from_log(logs[controller], scenario)
                
                # Only print primary signal for each scenario
                if scenario == 'S1_stop' and 'vx' in trans:
                    if trans['vx']['rise_time_ms']:
                        print(f"| {controller} vx rise time | {trans['vx']['rise_time_ms']:.0f} ms |")
                    if trans['vx']['settling_time_ms']:
                        print(f"| {controller} vx settling | {trans['vx']['settling_time_ms']:.0f} ms |")
                
                elif scenario == 'S2_turn' and 'wz' in trans:
                    if trans['wz']['rise_time_ms']:
                        print(f"| {controller} wz rise time | {trans['wz']['rise_time_ms']:.0f} ms |")
                    if trans['wz']['overshoot_percent']:
                        print(f"| {controller} wz overshoot | {trans['wz']['overshoot_percent']:.1f}% |")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='Run all experiments')
    parser.add_argument('--analyze', action='store_true', help='Analyze existing logs')
    args = parser.parse_args()
    
    if args.run:
        run_all_experiments()
    
    if args.analyze or not args.run:
        generate_comparison_table()