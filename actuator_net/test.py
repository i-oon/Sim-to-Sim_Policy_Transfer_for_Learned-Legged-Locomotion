import torch
import pandas as pd
import os
import os.path as osp
import pickle as pkl
from train import ActuatorNet
from sklearn.metrics import r2_score


def load_model( model_path, scaler, use_scale=False, device="cpu", num_envs=1, num_actions=12):

    if "pth" == model_path[-3:]:
        actuator_net = torch.jit.load(model_path, map_location=torch.device(device))
    else:
        actuator_net = ActuatorNet(in_dim=6,out_dim=num_actions)
        state_dict = torch.load(model_path, map_location= device)
        actuator_net.load_state_dict(state_dict)
        actuator_net.to(device)
        actuator_net.eval()

    num_envs = num_envs
    num_actions = num_actions
    print(f"use scale: {use_scale}")


    def eval_actuator_network(joint_pos, joint_pos_last, joint_pos_last_last, 
                                  joint_vel, joint_vel_last, joint_vel_last_last):

        inputs = torch.concat([joint_pos.unsqueeze(-1), joint_pos_last.unsqueeze(-1), 
                                                      joint_pos_last_last.unsqueeze(-1),
                                                      joint_vel.unsqueeze(-1), 
                                                      joint_vel_last.unsqueeze(-1),
                                                      joint_vel_last_last.unsqueeze(-1), 
                                                      ],axis=-1).view(num_envs*num_actions,6).to(device).type(torch.float32)

        if use_scale:
            scaled_inputs = scaler.transform(inputs)
            inputs = torch.tensor(scaled_inputs, dtype=torch.float32, device=device)

        torques = actuator_net(inputs.view(num_envs,num_actions, 6)).view(num_envs, num_actions)

        return torques.reshape(num_envs, num_actions)

    return eval_actuator_network

    
    


def load_data(datafile_dir, device="cpu"):

    #0) load raw data
    # checking data file path
    if os.path.isdir(datafile_dir):
        pd_data = pd.read_csv(os.path.join(datafile_dir, "actuator_data.csv"))
        print("data path:", os.path.join(datafile_dir, "actuator_data.csv"))
    else:
        if(os.path.exists(datafile_dir)):
            pd_data = pd.read_csv(os.path.join(datafile_dir))
            datafile_dir = os.path.dirname(datafile_dir)
        else:
            raise "Data file does not exist, please check the data file path!"
    columns =   list(pd_data.columns)
    print("dataset columns are:",columns)

    #1) load scaler data
    data_path = os.path.join(datafile_dir, "motor_data.pkl")
    if(os.path.exists(data_path)):
        print("data file path:",data_path)
    else:
        print(data_path)
        warnings.warn("Data file path  not exists")
    with open(data_path, 'rb') as fd:
        rawdata = pkl.load(fd)

    xs = torch.tensor(rawdata["input_data"],dtype=torch.float).to(device)
    ys = torch.tensor(rawdata["output_data"],dtype=torch.float).to(device)
    print(f"xs shape {xs.shape} and ys shape{ys.shape}")

    # load scaler, for testing and evaluation
    scaler_file = os.path.join(datafile_dir, "scaler.pkl")
    if(os.path.exists(scaler_file)):
        print("data scaler file:", scaler_file)
    else:
        print("{:} does not exist".format(scaler_file))
        exit()
    with open(scaler_file, "rb") as fd:
        scaler_dict = pkl.load(fd)
        scaler = scaler_dict["scaler"]
        use_scale = scaler_dict["use_scale"]

    return (xs,ys), scaler, pd_data, use_scale


import numpy as np
if __name__ == '__main__':
    
    # 0) hyper params
    model_path = osp.join("./app/resources","actuator.pth") # jit scripted model
    device ="cpu"
    num_envs = 1
    num_actions = 11

    # 1) load dataset and scaler
    dataset, scaler, rawdata, use_scale = load_data(os.path.dirname(model_path),device)


    # 2) load model, please specify your model path correctly, since jit script issue, load the model on cpu
    actuator_network = load_model(model_path, scaler, use_scale=use_scale, device="cpu", num_actions=num_actions)

    # 3) parameters
    joint_pos_err_last_last = torch.zeros((num_envs, num_actions), device=device)
    joint_pos_err_last = torch.zeros((num_envs, num_actions), device=device)
    joint_vel_last_last = torch.zeros((num_envs, num_actions), device=device)
    joint_vel_last = torch.zeros((num_envs, num_actions), device=device)

    actual_torques = []
    estimated_torques = []

    dof_pos_list = []
    dof_vel_list = []
    dof_pos_desired_list = []

    for idx in range(100): 
        #3) inference
        with torch.no_grad():
            
            dof_pos = np.array([rawdata["motorStatePos_"+str(i)][idx] for i in range(num_actions)]).reshape(num_envs,num_actions)
            dof_vel = np.array([rawdata["motorStateVel_"+str(i)][idx] for i in range(num_actions)]).reshape(num_envs,num_actions)
            dof_pos_desired = np.array([rawdata["motorAction_"+str(i)][idx] for i in range(num_actions)]).reshape(num_envs,num_actions)
            actual_torques.append(np.array([rawdata["motorStateCur_"+str(i)][idx] for i in range(num_actions)]).reshape(num_envs,num_actions))

            dof_pos_list.append(dof_pos)
            dof_vel_list.append(dof_vel)
            dof_pos_desired_list.append(dof_pos_desired)

            dof_pos = torch.tensor(dof_pos,device=device).view(num_envs,num_actions).view(num_envs,num_actions)
            dof_vel = torch.tensor(dof_vel,device=device).view(num_envs,num_actions).view(num_envs,num_actions)
            dof_pos_desired = torch.tensor(dof_pos_desired,device=device).view(num_envs,num_actions)

            joint_pos_err = dof_pos_desired - dof_pos  
            joint_vel = dof_vel

            torques = actuator_network(joint_pos_err, joint_pos_err_last, joint_pos_err_last_last, joint_vel, joint_vel_last, joint_vel_last_last)

            estimated_torques.append(torques)
            joint_pos_err_last_last = torch.clone(joint_pos_err_last)
            joint_pos_err_last = torch.clone(joint_pos_err)
            joint_vel_last_last = torch.clone(joint_vel_last)
            joint_vel_last = torch.clone(joint_vel)




    import matplotlib.pyplot as plt
    action_idx = 9
    actual = np.array([s[0][action_idx] for s in actual_torques])
    estimation = np.array([s[0][action_idx].cpu() for s in estimated_torques])

    dof_pos = np.array(dof_pos_list)
    dof_vel = np.array(dof_vel_list)
    dof_pos_desired = np.array(dof_pos_desired_list)
    calculation = 30 * (dof_pos_desired-dof_pos) -0.3*dof_vel
    calculation  = calculation[:,0,action_idx]

    r2 = r2_score(actual, estimation)
    cal_r2 = r2_score(actual, calculation)
    print("test r2 score:", r2, "calculation:", cal_r2)

    plt.plot(actual, label="actual",color='k')
    plt.plot(estimation, label="estimation",color='r')
    plt.plot(calculation, label="calculation",color='b')
    plt.grid()
    plt.legend()
    plt.show()

