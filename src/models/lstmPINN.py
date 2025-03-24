import torch
import torch.nn as nn

class lstmPINN(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_size):
        super(lstmPINN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x):
        with torch.backends.cudnn.flags(enabled=False):
            output, _ = self.lstm(x[:,:,:7]) 
        output = output[:, -1, :] # (batch_size, hidden_dim)
        output = self.fc(output) # (batch_size, output_size)
        return output