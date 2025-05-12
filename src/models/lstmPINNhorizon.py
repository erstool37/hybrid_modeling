import torch
import torch.nn as nn

class lstmPINNhorizon(nn.Module):
    """
    Predicts hidden physical parameters and next time-step system states 
    """
    def __init__(self, input_size, hidden_dim, num_layers, output_size):
        super(lstmPINNhorizon, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        self.activation = nn.ReLU()                
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, x, hidden=None):
        output, _ = self.lstm(x[:, :, :7])
        x = self.activation(output)
        x = self.fc(x)    
        
        return x