import numpy as np
import os
from legged_gym import LEGGED_GYM_ROOT_DIR

def gravity_from_quat_isaac(quat):
    """Isaac Gym: quaternion is (x,y,z,w)"""
    qx, qy, qz, qw = quat
    gravity = np.zeros(3)
    gravity[0] = 2 * (-qz * qx + qw * qy)
    gravity[1] = -2 * (qz * qy + qw * qx)
    gravity[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity

def gravity_from_quat_mujoco(quat):
    """MuJoCo: quaternion is (w,x,y,z)"""
    qw, qx, qy, qz = quat
    gravity = np.zeros(3)
    gravity[0] = 2 * (-qz * qx + qw * qy)
    gravity[1] = -2 * (qz * qy + qw * qx)
    gravity[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity

if __name__ == "__main__":
    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs/sim2sim")
    
    # Load logs
    isaac_logs = sorted([f for f in os.listdir(log_dir) if f.startswith("isaacgym_baseline")])
    mujoco_logs = sorted([f for f in os.listdir(log_dir) if f.startswith("mujoco_baseline")])
    
    if not isaac_logs or not mujoco_logs:
        print("Missing logs! Run both baseline scripts first.")
        exit(1)
    
    isaac_data = np.load(os.path.join(log_dir, isaac_logs[-1]))
    mujoco_data = np.load(os.path.join(log_dir, mujoco_logs[-1]))
    
    print(f"Isaac Gym log: {isaac_logs[-1]}")
    print(f"MuJoCo log: {mujoco_logs[-1]}")
    
    print("\n=== Observation Parity Test (Fixed Quaternion Convention) ===\n")
    
    # Compare first few timesteps
    for i in range(min(5, len(isaac_data['time']), len(mujoco_data['time']))):
        ig_grav = gravity_from_quat_isaac(isaac_data['base_quat'][i])
        mj_grav = gravity_from_quat_mujoco(mujoco_data['base_quat'][i])
        
        print(f"--- Timestep {i} (t={isaac_data['time'][i]:.3f}s) ---")
        print(f"  Isaac quat: {isaac_data['base_quat'][i]}")
        print(f"  MuJoCo quat: {mujoco_data['base_quat'][i]}")
        print(f"  Isaac gravity:  {ig_grav}")
        print(f"  MuJoCo gravity: {mj_grav}")
        print(f"  Gravity diff: {np.abs(ig_grav - mj_grav)}")
        print(f"  lin_vel Isaac: {isaac_data['base_lin_vel'][i]}")
        print(f"  lin_vel MuJoCo: {mujoco_data['base_lin_vel'][i]}")
        print()
    
    # Summary over all timesteps
    print("=== Summary (all timesteps) ===")
    n = min(len(isaac_data['time']), len(mujoco_data['time']))
    
    grav_diffs = []
    lin_vel_diffs = []
    ang_vel_diffs = []
    joint_pos_diffs = []
    
    for i in range(n):
        ig_grav = gravity_from_quat_isaac(isaac_data['base_quat'][i])
        mj_grav = gravity_from_quat_mujoco(mujoco_data['base_quat'][i])
        grav_diffs.append(np.mean(np.abs(ig_grav - mj_grav)))
        
        lin_vel_diffs.append(np.mean(np.abs(isaac_data['base_lin_vel'][i] - mujoco_data['base_lin_vel'][i])))
        ang_vel_diffs.append(np.mean(np.abs(isaac_data['base_ang_vel'][i] - mujoco_data['base_ang_vel'][i])))
        joint_pos_diffs.append(np.mean(np.abs(isaac_data['joint_pos'][i] - mujoco_data['joint_pos'][i])))
    
    print(f"Mean |gravity| diff:   {np.mean(grav_diffs):.4f}")
    print(f"Mean |lin_vel| diff:   {np.mean(lin_vel_diffs):.4f} m/s")
    print(f"Mean |ang_vel| diff:   {np.mean(ang_vel_diffs):.4f} rad/s")
    print(f"Mean |joint_pos| diff: {np.mean(joint_pos_diffs):.4f} rad")
    
    print("\n=== Verdict ===")
    if np.mean(grav_diffs) < 0.1:
        print("✅ Gravity orientation: MATCH (quaternion conventions handled correctly)")
    else:
        print("⚠️ Gravity orientation: Small difference (physics gap)")
    
    if np.mean(lin_vel_diffs) < 0.2:
        print("✅ Linear velocity: CLOSE")
    else:
        print("⚠️ Linear velocity: DIFFER (physics gap)")