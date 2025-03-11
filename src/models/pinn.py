import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_size):
        super(PINN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        output, _ = self.lstm(x) # output: (batch_size, seq_len, hidden_dim)
        output = output[:, -1, :] # (batch_size, hidden_dim)
        output = self.fc(output) # (batch_size, output_size)
        return output
