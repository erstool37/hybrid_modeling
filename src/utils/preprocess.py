import os.path as osp
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class Paraloader(Dataset):
    def __init__(self, dir, sequence_length):
        file_path = glob.glob(osp.join(dir, "*.csv"))

        states = []
        inputs = []
        thetas = []
        
        for path in file_path:
            ds = pd.read_csv(path)
            state = ds[['pressure', 'h_ref_out']], 
            input = ds[['m_ref_in', 'm_ref_out', 'h_ref_in', 'm_cool', 'T_cool_in']] # rest is T_ref_in, T_ref_out, T_cool_out
            theta = ds[['z_tpsh', 'gamma', 'eps_tp', 'eps_sh']]

            states.append(state)
            inputs.append(input)
            thetas.append(theta)
        
        self.states = torch.tensor(states, dtype=torch.float32)
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.thetas = torch.tensor(thetas, dtype=torch.float32)
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.states) - self.sequence_length - 1 # reaching maximum length
    
    def __getitem__(self, idx):
        step_state = self.states[idx : idx+self.sequence_length] # current time step state
        step_input = self.inputs[idx : idx+self.sequence_length] # current time step input
        step_theta = self.thetas[idx + self.sequence_length] # current time step theta
        pred_state = self.states[idx + 1 + self.sequence_length] # next time step state
        
        model_input = torch.cat(step_state, step_input)
        ground_truth = torch.cat(pred_state, step_theta)
    
        return model_input, ground_truth

# Rename files
"""
for i in range(99):
    files = glob.glob(osp.join(optimized_dir, f'dataset_{(i+1):02d}.csv'))
    if files: 
        old_file = files[0]
        new_file = osp.join(optimized_dir, f'dataset_{(i+1):03d}.csv')
        os.rename(old_file, new_file)
    else:
        print("No such files found")
"""