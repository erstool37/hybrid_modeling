import os
import time
import torch
import pandas as pd
from torch.utils.data import IterableDataset
import os.path as osp
from utils import Xnormalizer, Unormalizer, Onormalizer  # Make sure these are implemented and imported correctly
import wandb
import glob

class Realtimeloader(IterableDataset):
    """
    Monitors directory for Simulink-generated CSVs.
    For each new file:
    - Extracts x = [pressure, h_ref_out]
    - Extracts u = [m_ref_in, m_ref_out, h_ref_in, m_cool, T_cool_in]
    - Normalizes x and u
    - Maintains a rolling buffer of seq_len steps
    - Yields concatenated (x_seq | u_seq) tensors
    """
    def __init__(self, dir, seq_len, scaler):
        super().__init__()
        self.dir = dir
        self.seq_len = seq_len
        self.poll_interval = 0.3  # seconds
        self.prefix = "result_"
        self.current_idx = 0
        self.buffer = []
        self.scaler = scaler
        self._initialize_buffer()
        self.first_call = True


    def _initialize_buffer(self):
        """Load all startup_*.csv files to fill the initial sequence buffer."""
        path = os.path.join(self.dir, "x_simul", "startup_*.csv")
        file_list = sorted(glob.glob(path))

        for file_path in file_list:
            df = pd.read_csv(file_path)
            for i in range(len(df)):
                row = df.iloc[i]
                x = torch.tensor([row["pressure"], row["h_ref_out"]], dtype=torch.float32)
                u = torch.tensor([row["m_ref_in"], row["m_ref_out"], row["h_ref_in"], row["m_cool"], row["T_cool_in"]], dtype=torch.float32)
                others = torch.tensor([row["T_ref_in"], row["T_ref_out"], row["T_cool_out"], row["z_tpsh"]], dtype=torch.float32)

                x = Xnormalizer(x, self.scaler, "optim")
                u = Unormalizer(u, self.scaler, "optim")
                others = Onormalizer(others, self.scaler, "optim")

                self.buffer.append((x, u, others))  

    def _wait_file(self, idx):
        """Polls until simulink_{idx:04d}.csv exists."""
        filename = f"x_simul/{self.prefix}{idx:04d}.csv"
        fullpath = os.path.join(self.dir, filename)
        print(f"Waiting for {fullpath}...")
        while not os.path.exists(fullpath):
            time.sleep(self.poll_interval)
        return fullpath

    def __iter__(self):
        while True:
            print(1)
            if self.first_call:
                batch = self.buffer[-self.seq_len:]
                x_seq = torch.stack([b[0] for b in batch])
                u_seq = torch.stack([b[1] for b in batch])
                others_seq = torch.stack([b[2] for b in batch])
                model_input = torch.cat((x_seq, u_seq, others_seq), dim=1).unsqueeze(0).to(torch.float32)
                self.first_call = False
                self.current_idx += 1
                yield model_input

            else:
                path = self._wait_file(self.current_idx)  # now starts from 0001
                df = pd.read_csv(path)
                row = df.iloc[0]

                x = torch.tensor([row["pressure"], row["h_ref_out"]], dtype=torch.float32)
                u = torch.tensor([row["m_ref_in"], row["m_ref_out"], row["h_ref_in"], row["m_cool"], row["T_cool_in"]], dtype=torch.float32)
                others = torch.tensor([row["T_ref_in"], row["T_ref_out"], row["T_cool_out"], row["z_tpsh"]], dtype=torch.float32)

                wandb.log({"real pressure": x[0]})
                wandb.log({"real enthalpy": x[1]})

                x = Xnormalizer(x, self.scaler, "optim")
                u = Unormalizer(u, self.scaler, "optim")
                others = Onormalizer(others, self.scaler, "optim")

                self.buffer.append((x, u, others))

                batch = self.buffer[-self.seq_len:]

                x_seq = torch.stack([b[0] for b in batch])
                u_seq = torch.stack([b[1] for b in batch])
                others_seq = torch.stack([b[2] for b in batch])

                model_input = torch.cat((x_seq, u_seq, others_seq), dim=1).unsqueeze(0).to(torch.float32)
                self.current_idx += 1
                yield model_input