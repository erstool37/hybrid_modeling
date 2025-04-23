import os
import time
import torch
import pandas as pd
from torch.utils.data import IterableDataset
import os.path as osp
from utils import Xnormalizer, Unormalizer  # Make sure these are implemented and imported correctly

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
        self.base_path = osp.join(dir, "x_simul/result_0000.csv")
        self.buffer = []
        self.scaler = scaler
        self._initialize_buffer()

    def _initialize_buffer(self):
        """Load base.csv for the first seq_len values."""
        if not os.path.exists(self.base_path):
            return
        df = pd.read_csv(self.base_path)
        for i in range(min(self.seq_len, len(df))):
            row = df.iloc[i]
            x = torch.tensor([row["pressure"], row["h_ref_out"]], dtype=torch.float32)
            u = torch.tensor([row["m_ref_in"], row["m_ref_out"], row["h_ref_in"], row["m_cool"], row["T_cool_in"]], dtype=torch.float32)
            x = Xnormalizer(x, self.scaler, "optim")
            u = Unormalizer(u, self.scaler, "optim")
            self.buffer.append((x, u))

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
            path = self._wait_file(self.current_idx)
            df = pd.read_csv(path)
            row = df.iloc[0]

            x = torch.tensor([row["pressure"], row["h_ref_out"]], dtype=torch.float32)
            u = torch.tensor([row["m_ref_in"], row["m_ref_out"], row["h_ref_in"], row["m_cool"], row["T_cool_in"]], dtype=torch.float32)
            x = Xnormalizer(x, self.scaler, "optim")
            u = Unormalizer(u, self.scaler, "optim")

            self.buffer.append((x, u))
            self.current_idx += 1

            if len(self.buffer) >= self.seq_len:
                batch = self.buffer[-self.seq_len:]
                x_seq = torch.stack([b[0] for b in batch])
                u_seq = torch.stack([b[1] for b in batch])
                model_input = torch.cat((x_seq, u_seq), dim=1).unsqueeze(0).to(torch.float32)
                yield model_input