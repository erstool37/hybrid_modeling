import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_size):
        """
        inputs = x(t), u
        outputs = x(t+1), theta
        """
        super(PINN, self).__init__()
        self.cnn
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x):
        output, _ = self.lstm(x[:,:,:7]) 
        output = output[:, -1, :] # (batch_size, hidden_dim)
        output = self.fc(output) # (batch_size, output_size)
        return output