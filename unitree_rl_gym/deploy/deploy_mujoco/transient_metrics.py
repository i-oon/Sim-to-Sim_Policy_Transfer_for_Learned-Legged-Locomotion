"""
Transient Response Metrics Calculation
Computes rise time, settling time, overshoot for command switching scenarios
"""
import numpy as np

def compute_rise_time(time, signal, target, start_time, percent_range=(10, 90)):
    """
    Compute rise time (10-90% by default)
    
    Args:
        time: array of timestamps
        signal: array of signal values
        target: target value
        start_time: time when command switches
        percent_range: tuple of (low%, high%) for rise time calculation
    
    Returns:
        rise_time in milliseconds, or None if cannot compute
    """
    mask = time >= start_time
    if not np.any(mask):
        return None
    
    t_seg = time[mask]
    s_seg = signal[mask]
    
    if len(s_seg) < 2:
        return None
    
    initial_val = s_seg[0]
    change = target - initial_val
    
    if abs(change) < 1e-6:
        return 0.0  # Already at target
    
    low_thresh = initial_val + change * (percent_range[0] / 100.0)
    high_thresh = initial_val + change * (percent_range[1] / 100.0)
    
    # Find crossing indices
    if change > 0:
        low_cross = np.where(s_seg >= low_thresh)[0]
        high_cross = np.where(s_seg >= high_thresh)[0]
    else:
        low_cross = np.where(s_seg <= low_thresh)[0]
        high_cross = np.where(s_seg <= high_thresh)[0]
    
    if len(low_cross) == 0 or len(high_cross) == 0:
        return None
    
    t_low = t_seg[low_cross[0]]
    t_high = t_seg[high_cross[0]]
    
    rise_time_ms = (t_high - t_low) * 1000  # Convert to ms
    return rise_time_ms

def compute_settling_time(time, signal, target, start_time, tolerance_percent=5.0, window=0.1):
    """
    Compute settling time (time to stay within tolerance%)
    
    Args:
        time: array of timestamps
        signal: array of signal values
        target: target value
        start_time: time when command switches
        tolerance_percent: percentage tolerance band
        window: time window (seconds) to verify settled
    
    Returns:
        settling_time in milliseconds, or None if never settles
    """
    mask = time >= start_time
    if not np.any(mask):
        return None
    
    t_seg = time[mask]
    s_seg = signal[mask]
    
    if len(s_seg) < 2:
        return None
    
    tolerance = abs(target) * (tolerance_percent / 100.0) if abs(target) > 1e-6 else tolerance_percent / 100.0
    
    # Find last time signal exits the tolerance band
    in_band = np.abs(s_seg - target) <= tolerance
    
    if not np.any(in_band):
        return None  # Never enters band
    
    # Find the last exit from the band
    exit_indices = np.where(~in_band)[0]
    
    if len(exit_indices) == 0:
        # Always in band
        settling_time_ms = 0.0
    else:
        last_exit_idx = exit_indices[-1]
        
        # Check if it stays in band after last exit for at least 'window' duration
        remaining_time = t_seg[-1] - t_seg[last_exit_idx]
        if remaining_time < window:
            return None  # Not enough data to confirm settling
        
        settling_time_ms = (t_seg[last_exit_idx] - t_seg[0]) * 1000
    
    return settling_time_ms

def compute_overshoot(time, signal, target, start_time, window=1.5):
    """
    Compute overshoot as percentage
    
    Args:
        time: array of timestamps
        signal: array of signal values
        target: target value
        start_time: time when command switches
        window: time window to search for overshoot
    
    Returns:
        overshoot_percent, peak_value
    """
    mask = (time >= start_time) & (time <= start_time + window)
    if not np.any(mask):
        return None, None
    
    s_seg = signal[mask]
    
    if len(s_seg) < 2:
        return None, None
    
    initial_val = s_seg[0]
    change = target - initial_val
    
    if abs(change) < 1e-6:
        return 0.0, s_seg[0]
    
    # Find peak in the direction of change
    if change > 0:
        peak_val = np.max(s_seg)
        overshoot = peak_val - target
    else:
        peak_val = np.min(s_seg)
        overshoot = peak_val - target
    
    overshoot_percent = (overshoot / abs(change)) * 100.0 if abs(change) > 1e-6 else 0.0
    
    return overshoot_percent, peak_val

def compute_all_transient_metrics(time, signal, target, start_time, scenario_type='default'):
    """
    Compute all transient metrics for a signal
    
    Args:
        time: array of timestamps
        signal: array of signal values
        target: target value after command switch
        start_time: time when command switches
        scenario_type: 'stop', 'turn', 'lateral', or 'default'
    
    Returns:
        dict with rise_time, settling_time, overshoot, peak_value
    """
    metrics = {}
    
    rise_time = compute_rise_time(time, signal, target, start_time)
    settling_time = compute_settling_time(time, signal, target, start_time)
    overshoot, peak_val = compute_overshoot(time, signal, target, start_time)
    
    metrics['rise_time_ms'] = rise_time
    metrics['settling_time_ms'] = settling_time
    metrics['overshoot_percent'] = overshoot
    metrics['peak_value'] = peak_val
    
    return metrics

def analyze_scenario_transients(log_data, scenario_key, switch_time=3.0):
    """
    Analyze transient response for a specific scenario
    
    Args:
        log_data: dict with 'time', 'base_lin_vel', 'base_ang_vel', etc.
        scenario_key: 'S1_stop', 'S2_turn', or 'S3_lateral'
        switch_time: time when command switches
    
    Returns:
        dict with transient metrics for relevant signals
    """
    time = np.array(log_data['time'])
    
    if 'base_lin_vel' in log_data and len(log_data['base_lin_vel'].shape) == 2:
        vx = log_data['base_lin_vel'][:, 0]
        vy = log_data['base_lin_vel'][:, 1]
    else:
        vx = np.array(log_data.get('vx', []))
        vy = np.array(log_data.get('vy', []))
    
    if 'base_ang_vel' in log_data and len(log_data['base_ang_vel'].shape) == 2:
        wz = log_data['base_ang_vel'][:, 2]
    else:
        wz = np.array(log_data.get('wz', []))
    
    results = {}
    
    if scenario_key == 'S1_stop':
        # vx: 0.6 -> 0.0
        results['vx'] = compute_all_transient_metrics(time, vx, target=0.0, start_time=switch_time)
        results['wz'] = compute_all_transient_metrics(time, wz, target=0.0, start_time=switch_time)
        
    elif scenario_key == 'S2_turn':
        # vx: 0.4 -> 0.4 (should stay constant)
        # wz: 0.0 -> 1.0
        results['vx'] = compute_all_transient_metrics(time, vx, target=0.4, start_time=switch_time)
        results['wz'] = compute_all_transient_metrics(time, wz, target=1.0, start_time=switch_time)
        
    elif scenario_key == 'S3_lateral':
        # vx: 0.3 -> 0.3 (should stay constant)
        # vy: 0.3 -> -0.3
        # wz: should stay near 0
        results['vx'] = compute_all_transient_metrics(time, vx, target=0.3, start_time=switch_time)
        results['vy'] = compute_all_transient_metrics(time, vy, target=-0.3, start_time=switch_time)
        results['wz'] = compute_all_transient_metrics(time, wz, target=0.0, start_time=switch_time)
    
    return results

def print_transient_summary(results, scenario_key):
    """
    Print formatted transient metrics
    """
    print(f"\n=== Transient Metrics ({scenario_key}) ===")
    
    for signal_name, metrics in results.items():
        print(f"\n{signal_name.upper()}:")
        if metrics['rise_time_ms'] is not None:
            print(f"  Rise time (10-90%): {metrics['rise_time_ms']:.0f} ms")
        else:
            print(f"  Rise time: N/A")
            
        if metrics['settling_time_ms'] is not None:
            print(f"  Settling time (5%): {metrics['settling_time_ms']:.0f} ms")
        else:
            print(f"  Settling time: N/A (did not settle)")
            
        if metrics['overshoot_percent'] is not None:
            print(f"  Overshoot: {metrics['overshoot_percent']:.1f}%")
            print(f"  Peak value: {metrics['peak_value']:.3f}")
        else:
            print(f"  Overshoot: N/A")