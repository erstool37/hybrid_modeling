import torch
import torch.nn as nn

class HybridLSTMModel(nn.Module):
    def __init__(self, input_size=7, output_size=4, lookback=30, hidden_size=128, dense_size=32):
        super(HybridLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, dense_size)
        self.fc2 = nn.Linear(dense_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out: (B, L, H)
        x = self.sigmoid(self.fc1(lstm_out))  # (B, L, D)
        out = self.fc2(x)  # (B, L, output_dim)
        return out