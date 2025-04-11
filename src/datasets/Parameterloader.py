import os.path as osp
import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import inspect
from utils.utils import normalize, unnormalize
import importlib

class Parameterloader(Dataset):
    """
    Dataloader for the lstmPINN model,
    """
    def __init__(self, dir, sequence_length, method, scaler):
        super().__init__()
        self.sequence_length = sequence_length # exists for lstmPINN
        self.ds = pd.read_csv(dir)
        self.method = method

        # normalize the dataset
        utils = importlib.import_module("utils")
        self.scaler = getattr(utils, scaler)

        for item in self.ds.columns: 
            self.ds[item] = self.scaler(torch.tensor(self.ds[item].values), item, self.method)

        state = self.ds[['pressure', 'h_ref_out']].astype(float).values
        input = self.ds[['m_ref_in', 'm_ref_out', 'h_ref_in', 'm_cool', 'T_cool_in']].astype(float).values
        theta = self.ds[['z_tpsh', 'gamma', 'eps_tp', 'eps_sh']].astype(float).values
        
        self.states = torch.tensor(state, dtype=torch.float32)
        self.inputs = torch.tensor(input, dtype=torch.float32)
        self.thetas = torch.tensor(theta, dtype=torch.float32)

    def __getitem__(self, idx):
        step_state = self.states[idx : idx + self.sequence_length] # current time step state
        step_input = self.inputs[idx : idx + self.sequence_length] # current time step input
        step_theta = self.thetas[idx + self.sequence_length - 1] # current time step theta
        pred_state = self.states[idx + self.sequence_length] # next time step state

        # Flatten tensors and concatenate them
        model_input = torch.cat((step_state, step_input), dim=1)
        ground_truth = torch.cat((pred_state, step_theta), dim=0)
        return model_input, ground_truth

    def __len__(self):
        return len(self.states) - self.sequence_length