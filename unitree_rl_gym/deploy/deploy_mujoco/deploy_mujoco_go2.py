import time
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name")
    args = parser.parse_args()
    
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{args.config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        cmd = np.array(config["cmd_init"], dtype=np.float32)
        lin_vel_scale = 2.0  # from obs_scales.lin_vel

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    
    # Set initial pose
    d.qpos[0:3] = [0, 0, 0.35]
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[7:] = default_angles
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    print("Initial pose set")

    policy = torch.jit.load(policy_path)

    debug_counter = 0
    prev_sim_time = 0.0
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # === Detect reset (sim time jumped back) ===
            if d.time < prev_sim_time:
                print("Reset detected! Re-initializing...")
                action = np.zeros(num_actions, dtype=np.float32)
                target_dof_pos = default_angles.copy()
                obs = np.zeros(num_obs, dtype=np.float32)
                counter = 0
                debug_counter = 0
                start = time.time()  # reset wall-clock timer
                # Re-set initial pose
                d.qpos[0:3] = [0, 0, 0.35]
                d.qpos[3:7] = [1, 0, 0, 0]
                d.qpos[7:] = default_angles
                d.qvel[:] = 0
                mujoco.mj_forward(m, d)
            prev_sim_time = d.time
            
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[0:3] = tau[3:6]
            d.ctrl[3:6] = tau[0:3]
            d.ctrl[6:9] = tau[9:12]
            d.ctrl[9:12] = tau[6:9]
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Get states
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                
                # Base velocities in body frame
                base_lin_vel = d.qvel[0:3]  # world frame
                base_ang_vel = d.qvel[3:6]  # world frame
                
                gravity_orientation = get_gravity_orientation(quat)
                
                # Build observation (Isaac Gym order):
                # [lin_vel(3), ang_vel(3), gravity(3), cmd(3), dof_pos(12), dof_vel(12), actions(12)] = 48
                obs[0:3] = base_lin_vel * lin_vel_scale
                obs[3:6] = base_ang_vel * ang_vel_scale
                obs[6:9] = gravity_orientation
                obs[9:12] = cmd * cmd_scale
                obs[12:24] = (qj - default_angles) * dof_pos_scale
                obs[24:36] = dqj * dof_vel_scale
                obs[36:48] = action
                
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()
                target_dof_pos = action * action_scale + default_angles

                debug_counter += 1
                if debug_counter <= 3:
                    print(f"\n=== Step {debug_counter} ===")
                    print(f"lin_vel: {base_lin_vel}")
                    print(f"ang_vel: {base_ang_vel}")
                    print(f"gravity: {gravity_orientation}")
                    print(f"action (first 6): {action[:6]}")

            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)