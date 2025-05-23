import torch
import torch.nn as nn

class RNN(nn.Module):
    """
    Vanilla RNN-based model to predict hidden physical parameters and next time-step system states
    """
    def __init__(self, input_size, hidden_dim, num_layers, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_dim, num_layers, batch_first=True, nonlinearity='tanh')
        self.activation = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden=None):
        rnn_out, _ = self.rnn(x[:, :, :7], hidden)
        last_output = rnn_out[:, -1, :]
        x = self.activation(last_output)
        x = self.fc(x)
        return x