import torch
import torch.nn as nn

class lstmPINN(nn.Module):
    """
    Predicts hidden physical parameters and next time-step system states 
    """
    def __init__(self, input_size, hidden_dim, num_layers, output_size):
        super(lstmPINN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        self.activation = nn.ReLU()                
        self.fc = nn.Linear(hidden_dim, output_size)     
        # self.fc = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, output_size)
        # )

    def forward(self, x, hidden=None):
        output, _ = self.lstm(x[:, :, :7])
        output = output[:, -1, :]
        x = self.activation(output)
        x = self.fc(x)    
        
        return x