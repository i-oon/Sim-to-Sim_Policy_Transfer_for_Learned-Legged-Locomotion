import os
import pickle as pkl
from matplotlib import pyplot as plt
import time
import imageio
import numpy as np
from tqdm import tqdm
from glob import glob
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
#from sklearn.metrics import root_mean_squared_error # only supported from scikit-learn 1.4.0 and python3.9


from torch.optim import Adam



class ActuatorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['motor_states'])

    def __getitem__(self, idx):
        return {k: v[idx] for k,v in self.data.items()}

class Act(nn.Module):
  def __init__(self, act, slope=0.05):
    super(Act, self).__init__()
    self.act = act
    self.slope = slope
    self.shift = torch.log(torch.tensor(2.0)).item()

  def forward(self, input):
    if self.act == "relu":
      return F.relu(input)
    elif self.act == "leaky_relu":
      return F.leaky_relu(input)
    elif self.act == "sp":
      return F.softplus(input, beta=1.)
    elif self.act == "leaky_sp":
      return F.softplus(input, beta=1.) - self.slope * F.relu(-input)
    elif self.act == "elu":
      return F.elu(input, alpha=1.)
    elif self.act == "leaky_elu":
      return F.elu(input, alpha=1.) - self.slope * F.relu(-input)
    elif self.act == "ssp":
      return F.softplus(input, beta=1.) - self.shift
    elif self.act == "leaky_ssp":
      return (
          F.softplus(input, beta=1.) -
          self.slope * F.relu(-input) -
          self.shift
      )
    elif self.act == "tanh":
      return torch.tanh(input)
    elif self.act == "leaky_tanh":
      return torch.tanh(input) + self.slope * input
    elif self.act == "swish":
      return torch.sigmoid(input) * input
    elif self.act == "softsign":
        return F.softsign(input)
    else:
      raise RuntimeError(f"Undefined activation called {self.act}")


class ActuatorNet(nn.Module):

    def __init__(self, in_dim, units=100, layers=4, out_dim=1,
              act='relu', layer_norm=False, act_final=False):

        super(ActuatorNet, self).__init__()
        mods = [nn.Linear(in_dim, units), Act(act)]
        for i in range(layers-1):
            mods += [nn.Linear(units, units), Act(act)]
        mods += [nn.Linear(units, out_dim)]
        if act_final:
            mods += [Act(act)]
        if layer_norm:
            mods += [nn.LayerNorm(out_dim)]
        self.model = nn.Sequential(*mods)

    def forward(self,x):
        out = self.model(x)
        return out


class EarlyStopping:

    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_acc = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, val_acc, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_acc = val_acc
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_acc = val_acc
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # save torch model state_dict
        path = os.path.join(self.save_path, 'actuator.pt')
        torch.save(model.state_dict(), path)	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

        # save torch.jit.scrpts model
        model_scripted = torch.jit.script(model)  # Export to TorchScript
        model_scripted.save(os.path.join(self.save_path,"actuator.pth"))  # Save










class Train():
    def __init__(self, 
            data_sample_freq=100,
            datafile_dir=None,
            load_pretrained_model=False,
            device="cpu",
            **kwargs
            ):
        self.data_sample_freq = data_sample_freq
        if(os.path.isfile(datafile_dir)):
            self.datafile_dir = os.path.dirname(datafile_dir)
        else:
            self.datafile_dir = datafile_dir
        self.load_pretrained_model = load_pretrained_model
        self.actuator_network_path = os.path.join(self.datafile_dir,"actuator.pth")
        self.device=device
        if("epochs" in kwargs.keys()):
            self.epochs = kwargs["epochs"]
        else:
            self.epochs = 1000

        if "display_motor_idx" in kwargs.keys():
            self.display_motor_idx = kwargs["display_motor_idx"]
        else:
            self.display_motor_idx = 0


    def train_actuator_network(self):
        """
        Train actuator model
        """
        print("model input dim: {} and output dim: {}".format( self.xs.shape, self.ys.shape))
    
    
        model = ActuatorNet(
                in_dim=self.xs.shape[1], 
                units=100, layers=4, 
                out_dim=1 if len(self.ys.shape)<2 else self.ys.shape[1], 
                act='softsign')
    
        lr = 8e-4
        opt = Adam(model.parameters(), lr=lr, eps=1e-8, weight_decay=0.0)

        early_stop = EarlyStopping(save_path=self.datafile_dir, patience=10, verbose=True)
    
        model = model.to(self.device)
        for epoch in range(self.epochs):
            epoch_loss = 0
            ct = 0
            for batch in self.train_loader:
                data = batch['motor_states'].to(self.device)

                y_pred = model(data)
    
                opt.zero_grad()
    
                y_label = batch['motor_outputs'].to(self.device)
    
                tau_est_loss = ((y_pred - y_label) ** 2).mean()
                loss = tau_est_loss
    
                loss.backward()
                opt.step()
                epoch_loss += loss.detach().cpu().numpy()
                ct += 1
            epoch_loss /= ct
    
            valid_loss = 0
            valid_r2 = 0
            ct = 0
            if epoch % 1 == 0:
                with torch.no_grad():
                    for batch in self.valid_loader:
                        data = batch['motor_states'].to(self.device)
                        y_pred = model(data)
                        y_label = batch['motor_outputs'].to(self.device)
                        valid_loss += ((y_pred - y_label) ** 2).mean()
                        #valid_loss+=root_mean_squared_error(y_label, y_pred)
                        valid_r2 += r2_score(y_label.cpu(), y_pred.cpu())
                        ct+=1
                
                valid_loss /=ct
                valid_r2 /=ct
                self.train_info = f'epoch: {epoch} | train loss: {epoch_loss:.4f} | valid loss: {valid_loss:.4f} | valid_r2: {valid_r2:.4f}'
                print(self.train_info)


            early_stop(valid_loss, valid_r2, model)
            if early_stop.early_stop:
                break

        return model



    def load_data(self):
        #1) load data
        data_path = os.path.join(self.datafile_dir, "motor_data.pkl")
        if(os.path.exists(data_path)):
            print("data file path:",data_path)
        else:
            print(data_path)
            warnings.warn("Data file path  not exists")
        with open(data_path, 'rb') as fd:
            rawdata = pkl.load(fd)

    
        self.xs = torch.tensor(rawdata["input_data"],dtype=torch.float).to(self.device)
        self.ys = torch.tensor(rawdata["output_data"],dtype=torch.float).to(self.device)

    
        # load scaler, for testing and evaluation
        scaler_file = os.path.join(self.datafile_dir, "scaler.pkl")
        if(os.path.exists(scaler_file)):
            print("data scaler file:", scaler_file)
        else:
            print("{:} does not exist".format(scaler_file))
            exit()
        with open(scaler_file,"rb") as fd:
            scaler_dict = pkl.load(fd)
            self.scaler = scaler_dict["scaler"]
            self.use_scale = scaler_dict["use_scale"]
            print(f"Use scale: {self.use_scale}")


        # make dataloader
        num_data = self.xs.shape[0]
        num_train = num_data // 5 * 4
        num_test = 500
        num_valid = num_data - num_train -num_test
    
        dataset = ActuatorDataset({"motor_states": self.xs, "motor_outputs": self.ys})
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [num_train, num_valid, num_test])
        self.train_loader = DataLoader(train_set, batch_size=128, shuffle=False)
        self.valid_loader = DataLoader(val_set, batch_size=128, shuffle=False)
        self.test_loader = DataLoader(test_set, batch_size=1, shuffle=False)


    def training_model(self):
        """
        Training model

        """
    
        # training model
        if self.load_pretrained_model:
            self.model = torch.jit.load(self.actuator_network_path).to("cpu")
        else:
            self.model = self.train_actuator_network().to(self.device)
    
    
    
    def eval_model(self):
        """
        Model evaluation

        """
    
        plot_length = 450
        model_input = []
        estimation = []
        actual = []

        with torch.no_grad():
            for batch in self.test_loader:
                x = batch["motor_states"]
                if self.load_pretrained_model:
                    x=x.to("cpu") # jit scripted model only can work on cpu
                pred = self.model(x).detach()
                model_input.append(x)
                estimation.append(pred)
                actual.append(batch["motor_outputs"].detach())


        self.time = np.arange(plot_length) / self.data_sample_freq
        self.actual = torch.tensor(actual).unsqueeze(-1).cpu().detach().numpy()[:plot_length,:]
        self.estimation = torch.tensor(estimation).unsqueeze(-1).cpu().detach().numpy()[:plot_length,:]
        self.scaled_model_input = torch.concat(model_input,dim=0).cpu().detach().numpy()[:plot_length,:]


        mae=r2_score(self.actual,self.estimation)
        print("test r2 score:", mae)



if __name__=="__main__":
    kwargs={"epochs":1000, "device":"cuda:0"}
    datafile_dir =  "./app/resources/"
    training = Train(
            data_sample_freq=50,
            datafile_dir = datafile_dir,
            load_pretrained_model = False,
            **kwargs
            )

    training.load_data()
    training.training_model()
    training.eval_model()

    import matplotlib.pyplot as plt
    plt.plot(training.time, training.actual, label="actual",color='r')
    plt.plot(training.time, training.estimation, label="estimation",color='g')
    plt.legend()
    plt.show()
