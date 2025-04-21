import os
import json
import time
import torch
from torch.utils.data import IterableDataset
import os.path as osp
import importlib
from utils import Xnormalizer, Unormalizer

class Realtimeloader(IterableDataset):
    def __init__(self, dir, seq_len, scaler):
        self.dir = dir
        self.seq_len = seq_len
        self.poll_interval = 0.3
        self.prefix = "simulink_"
        self.current_idx = 0
        self.base_path = osp.join(dir, "x_simul/base.json")
        self.buffer = self._load_base()
        self.scaler = scaler

    def _load_base(self):
        buffer = []
        with open(self.base_path, "r") as f:
            data = json.load(f)  # list of dicts with keys "x_t", "u_t"
            for item in data[:self.seq_len]:
                x = Xnormalizer(torch.tensor(item["x_t"], dtype=torch.float32), self.scaler, "optim")
                u = Unormalizer(torch.tensor(item["u_t"], dtype=torch.float32), self.scaler, "optim")
                buffer.append((x, u))
        self.current_idx = len(buffer)
        return buffer

    def _wait_file(self, idx):
        filename = f"{self.prefix}{idx}.json"
        fullpath = os.path.join(self.dir, filename)
        while not os.path.exists(fullpath):
            time.sleep(self.poll_interval)
        return fullpath

    def __iter__(self):
        while True:
            path = self._wait_file(self.current_idx)

            with open(path, "r") as file:
                data = json.load(file)
                x = self.Xnormalizer(torch.tensor(data["x_t"], dtype=torch.float32), self.scaler, "optim")
                u = self.Unormalizer(torch.tensor(data["u_t"], dtype=torch.float32), self.scaler, "optim")
                self.buffer.append((x, u))
                self.current_idx += 1

            if len(self.buffer) >= self.seq_len:
                batch = self.buffer[-self.seq_len:]
                x_seq = torch.stack([b[0] for b in batch])
                u_seq = torch.stack([b[1] for b in batch])

                model_input = torch.cat((x_seq, u_seq), dim=1)
                yield model_input