import time
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import yaml

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def quat_to_euler(quat):
    """Convert quaternion (wxyz) to roll, pitch, yaw"""
    w, x, y, z = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

if __name__ == "__main__":
    config_file = "go2.yaml"
    
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)

    m = mujoco.MjModel.from_xml_path(os.path.expanduser(xml_path))
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[7:] = default_angles
    else:
        d.qpos[0:3] = [0, 0, 0.35]
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[7:] = default_angles
    
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    
    print(f"=== Zero-Action Stability Test (MuJoCo) ===")
    print(f"Initial height: {d.qpos[2]:.4f} m")
    print(f"Initial quat: {d.qpos[3:7]}")
    print("Running for 5 seconds with action=0...")
    
    target_dof_pos = default_angles.copy()
    
    counter = 0
    duration = 5.0
    
    heights = []
    rolls = []
    pitches = []
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        sim_time = 0.0
        
        while viewer.is_running() and sim_time < duration:
            step_start = time.time()
            
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[0:3] = tau[3:6]
            d.ctrl[3:6] = tau[0:3]
            d.ctrl[6:9] = tau[9:12]
            d.ctrl[9:12] = tau[6:9]
            mujoco.mj_step(m, d)
            sim_time += simulation_dt
            
            counter += 1
            if counter % control_decimation == 0:
                quat = d.qpos[3:7]
                roll, pitch, yaw = quat_to_euler(quat)
                
                heights.append(d.qpos[2])
                rolls.append(roll)
                pitches.append(pitch)
                
                if len(heights) <= 3:
                    print(f"  t={sim_time:.2f}s: h={d.qpos[2]:.3f}, roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°")
            
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    heights = np.array(heights)
    rolls = np.array(rolls)
    pitches = np.array(pitches)
    
    print(f"\n=== Results ===")
    print(f"Final height: {heights[-1]:.4f} m")
    print(f"Height range: [{heights.min():.4f}, {heights.max():.4f}] m")
    print(f"Max |roll|: {np.degrees(np.max(np.abs(rolls))):.2f}°")
    print(f"Max |pitch|: {np.degrees(np.max(np.abs(pitches))):.2f}°")
    
    stable = heights[-1] > 0.15 and np.max(np.abs(rolls)) < 0.5 and np.max(np.abs(pitches)) < 0.5
    print(f"\n{'✅ STABLE' if stable else '❌ UNSTABLE'}")