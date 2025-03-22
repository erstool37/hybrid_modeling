import os.path as osp
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Paraloader(Dataset):
    def __init__(self, dir, stats_dir, sequence_length):
        super().__init__()
        self.sequence_length = sequence_length # exists for lstmPINN
        self.ds = pd.read_csv(dir)
        self.stats = pd.read_csv(stats_dir)

        # normalize the dataset
        for item in self.ds.columns: 
            self.ds[item] = normalize(self.ds[item].values, item, train)

        time = self.ds [['time']].astype(float).values
        state = self.ds[['pressure', 'h_ref_out']].astype(float).values
        input = self.ds[['m_ref_in', 'm_ref_out', 'h_ref_in', 'm_cool', 'T_cool_in']].astype(float).values
        theta = self.ds[['z_tpsh', 'gamma', 'eps_tp', 'eps_sh']].astype(float).values
        
        self.times = torch.tensor(time, dtype=torch.float32)
        self.states = torch.tensor(state, dtype=torch.float32)
        self.inputs = torch.tensor(input, dtype=torch.float32)
        self.thetas = torch.tensor(theta, dtype=torch.float32)

    def __getitem__(self, idx):
        step_time = self.times[idx]
        step_state = self.states[idx] # current time step state
        step_input = self.inputs[idx] # current time step input
        step_theta = self.thetas[idx] # current time step theta
        pred_state = self.states[idx] # next time step state

        # Flatten tensors and concatenate them
        model_input = torch.cat((step_time, step_state, step_input, step_theta), dim=1)
        ground_truth = torch.cat((pred_state, step_theta), dim=0)

        return model_input, ground_truth
    
    def Znormalize(self, item, column, method):
        script_dir = os.path.dirname(os.path.abspath(inspect.getfile(self.__class__)))
        file_path = os.path.join(script_dir, 'dataset', f'statistics_{method}.csv')
        stats = pd.read_csv(file_path)

        mean_item = torch.tensor(stats.loc[0, column])
        std_item = torch.tensor(stats.loc[1, column])

        item_norm = (item - mean_item) / std_item
        return item_norm

    def Zunnormalize(self, item, column, method):
        script_dir = os.path.dirname(os.path.abspath(inspect.getfile(self.__class__)))
        file_path = os.path.join(script_dir, 'dataset', f'statistics_{method}.csv')
        stats = pd.read_csv(file_path)

        mean_item = torch.tensor(stats.loc[0, column])
        std_item = torch.tensor(stats.loc[1, column])

        item_un= item * std_item + mean_item
        return item_un
    
    def normalize(self, item, column, method):
        script_dir = os.path.dirname(os.path.abspath(inspect.getfile(self.__class__)))
        file_path = os.path.join(script_dir, 'dataset', f'statistics_{method}.csv')
        stats = pd.read_csv(file_path)

        max_item = torch.tensor(stats.loc[2, column])
        min_item = torch.tensor(stats.loc[3, column])

        item_norm = (item - min_item) / (max_item - min_item)
        return item_norm

    def unnormalize(self, item, column, mehthod):
        script_dir = os.path.dirname(os.path.abspath(inspect.getfile(self.__class__)))
        file_path = os.path.join(script_dir, 'dataset', f'statistics_{method}.csv')
        stats = pd.read_csv(file_path)

        max_item = torch.tensor(stats.loc[2, column])
        min_item = torch.tensor(stats.loc[3, column])

        item_un = item * (max_item - min_item) + min_item
        return item_un

    def __len__(self):
        return len(self.time)

    """ for lstmPINN
    def __getitem__(self, idx):
        step_state = self.states[idx : idx + self.sequence_length] # current time step state
        step_input = self.inputs[idx : idx + self.sequence_length] # current time step input
        step_theta = self.thetas[idx + self.sequence_length - 1] # current time step theta
        pred_state = self.states[idx + self.sequence_length] # next time step state

        # Flatten tensors and concatenate them
        model_input = torch.cat((step_state, step_input, step_theta), dim=1)
        ground_truth = torch.cat((pred_state, step_theta), dim=0)

        return model_input, ground_truth

    def __len__(self):
        return len(self.states) - self.sequence_length
    """