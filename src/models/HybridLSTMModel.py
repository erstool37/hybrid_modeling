import torch
import torch.nn as nn

class HybridLSTMModel(nn.Module):
    def __init__(self, input_size=7, output_size=4, lookback=30, hidden_size=128, dense_size=32, dropout=0.1):
        super(HybridLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, dense_size)
        self.act = nn.Sigmoid()
        self.fc2 = nn.Linear(dense_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.dropout(lstm_out)
        x = self.act(self.fc1(x)) 
        out = self.act(self.fc2(x))
        return out