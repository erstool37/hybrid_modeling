import os.path as osp
import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import inspect

class Parameterloader(Dataset):
    """
    DISCARDED
    Dataloader for the PINN model, where the pretrained LSTM provides hidden paramters theta, and time is provided as an input
    """
    def __init__(self, dir, sequence_length, method):
        super().__init__()
        self.sequence_length = sequence_length # exists for lstmPINN, kept for consistency
        self.ds = pd.read_csv(dir)
        self.method = method

        # normalize the dataset
        for item in self.ds.columns: 
            self.ds[item] = self.normalize(torch.tensor(self.ds[item].values), item, self.method)

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
        step_state = self.states[idx]
        step_input = self.inputs[idx]
        step_theta = self.thetas[idx]
        pred_state = self.states[idx + 1]

        # Flatten tensors and concatenate them
        model_input = torch.cat((step_time, step_state, step_input, step_theta), dim=-1)
        ground_truth = torch.cat((pred_state), dim=0)
        
        return model_input, ground_truth

    def __len__(self):
        return len(self.times) - 1
    
