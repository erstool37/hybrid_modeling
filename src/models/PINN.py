import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, input_size, output_size):
        super(PINN, self).__init__()
        self.fnn = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64), 
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
            )
        
    def forward(self, x):
        output = self.fnn(x) 
        return output