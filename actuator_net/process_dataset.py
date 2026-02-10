from train import Train
from glob import glob
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler



class DataProcess():
    def __init__(self, 
            raw_filepath=None,
            data_start =3100, # valid of data row start
            data_end=8600, # valid of data row end
            motors=[2,3,5], # valid of data column group
            input_data_name = None,
            output_data_name = None,
            scaler="standard"
            ):
        self.raw_filepath = raw_filepath
        # checking data file path
        if os.path.isdir(raw_filepath):
            self.pd_data = pd.read_csv(
                    os.path.join(raw_filepath,"actuator_data.csv"))
            self.datafile_dir = raw_filepath
            print("data path:", os.path.join(raw_filepath,"actuator_data.csv"))
        else:
            if(os.path.exists(raw_filepath)):
                self.pd_data = pd.read_csv(os.path.join(raw_filepath))
                self.datafile_dir = os.path.dirname(raw_filepath)
            else:
                raise "Data file does not exist, please check the data file path!"
        columns =   list(self.pd_data.columns)
        print("dataset columns are:",columns)
        self.data_row_num = self.pd_data.shape[0]
        self.data_start = data_start
        self.data_end = data_end
        self.motors = motors
        
        self.input_data_name = [list(self.pd_data.columns)[2][:-2]]
        if input_data_name is not None:
            if set([key+"_0" for key in input_data_name]).issubset(set(columns)):
                self.input_data_name = input_data_name

        self.output_data_name = [list(self.pd_data.columns)[3][:-2]]
        if  output_data_name is not None:
            if set([key +"_0" for key in output_data_name]).issubset(set(columns)):
                self.output_data_name = output_data_name

        self.scaler = scaler
        print(f"input data name: {self.input_data_name}")
        print(f"output data name: {self.output_data_name}")


    def process_data(self):
        """
        Process data
        """
        #1) select dataset column
        input_data_list =[]
        output_data_list =[]
        for idx in self.motors:
            input_data_list.append(self.pd_data.loc[:,[tmp+"_"+str(idx) for tmp in self.input_data_name]])
            output_data_list.append(self.pd_data.loc[:,[tmp+"_"+str(idx) for tmp in self.output_data_name]])

        input_data = np.concatenate([value.values for value in input_data_list],axis=0)
        output_data = np.concatenate([value.values for value in output_data_list],axis=0)

        # raw data from experiment collection
        raw_dataset = np.concatenate([input_data,output_data],axis=1)

        # features and labels
        pos_error =  raw_dataset[:,2] - raw_dataset[:,0]
        pos_error = pos_error.reshape(-1,1)
        pos_last_error = np.vstack((pos_error[0,:], pos_error[:-1,:]))
        pos_last_error[0,:] = 0.0
        pos_last_last_error = np.vstack((pos_error[:2,:], pos_error[:-2,:]))
        pos_last_last_error[:2,:] = 0.0

        vel = raw_dataset[:,1].reshape(-1,1)
        vel_last = np.vstack((vel[0,:],vel[:-1,:]))
        vel_last[0,:]=0.0
        vel_last_last = np.vstack((vel[:2,:],vel[:-2,:]))
        vel_last_last[:2,:]=0.0

        label = raw_dataset[:,-1].reshape(-1,1)

        feature = np.concatenate([pos_error, pos_last_error, pos_last_last_error, vel, vel_last, vel_last_last], axis=-1)

        #2) normalizate data
        #i) normalization method
        if self.scaler=='standard':
            scaler=StandardScaler()
        if self.scaler=='minmax':
            scaler=MinMaxScaler()
        if self.scaler=='robust':
            scaler=RobustScaler()

        try:
            scaler.fit(feature)
            scaled_feature = scaler.transform(feature.astype(np.float32))
        except Exception as e:
            print(e)


        processed_data = {"input_data": feature, "output_data": label}
        print(f"input data shape: {scaled_feature.shape}")
        print(f"output data shape: {label.shape}")

        with open(os.path.join(self.datafile_dir,'motor_data.pkl'), 'wb') as f:
            pickle.dump(processed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.datafile_dir,'scaler.pkl'),'wb') as f:
            scaler_dict = {"use_scale": False,  "scaler": scaler}
            pickle.dump(scaler_dict,f) # NOTE, input_data is feature rather than scaled_feature

        print("Process ambot motor data, data file saved at: {:}".format(os.path.join(self.datafile_dir,'motor_data.pkl')))
        print("Scaler file saved at: {:}".format(os.path.join(self.datafile_dir,'scaler.pkl')))
        print("Use scale: {:}".format(scaler_dict["use_scale"]))

        """
        processed_data =[] 
        for step_idx in range(self.data_start, min(self.data_row_num,self.data_end) - self.data_start):
            processed_data.append({
            "motor_pos_target":[self.pd_data["jcm_"+str(joint_idx)][step_idx] for joint_idx in self.motors],
            "motor_pos":[self.pd_data["jointPosition_"+str(joint_idx)][step_idx] for joint_idx in self.motors],
            "motor_vel":[self.pd_data["jointVelocity_"+str(joint_idx)][step_idx] for joint_idx in self.motors],
            "motor_tor":[(self.pd_data["jointCurrent_"+str(joint_idx)][step_idx]*0.001)*0.86134+0.65971 for joint_idx in self.motors],
            #"torques":[(pd_data["jointCurrent_"+str(joint_idx)][step_idx]*0.001)*0.86134+0.65971 for joint_idx in motors],
            #"tau_est":[(pd_data["jointCurrent_"+str(joint_idx)][step_idx]*0.001)*0.86134+0.65971 for joint_idx in motors], #mA to A, to Nm
            #"tau_est":[0.0 for joint_idx in motors],
            })

        result_datas = {'motor_data':[processed_data]}

        with open(os.path.join(self.datafile_dir,'motor_data.pkl'), 'wb') as f:
            pickle.dump(result_datas, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("Process ambot motor data, data file saved at: {:}".format(os.path.join(self.datafile_dir,'motor_data.pkl')))

        """





if __name__=="__main__":
    load_pretrained_model = False
    datafile_dir = "./app/resources/"
    dp = DataProcess(datafile_dir,
                    data_start=2000,
                    data_end=50000,
                    motors=[0,1,2,3,4,5,6,7,8,9,10,11],
                    input_data_name=["motorStatePos", "motorStateVel","motorAction"],
                    output_data_name=["motorStateCur"],
                    )
    dp.process_data()
