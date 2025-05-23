import torch
import torch.nn as nn

class GRU(nn.Module):
    """
    GRU-based model to predict hidden physical parameters and next time-step system states
    """
    def __init__(self, input_size, hidden_dim, num_layers, output_size):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_dim, num_layers, batch_first=True)
        self.activation = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden=None):
        gru_out, _ = self.gru(x[:, :, :7], hidden)
        last_output = gru_out[:, -1, :]
        x = self.activation(last_output)
        x = self.fc(x)
        return x