import torch
import torch.nn as nn
import torch.optim as optim

# LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_size_lstm):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size_lstm)
    
    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])
        return output

# PINN Model
class PINN(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_size_pinn):
        super(PINN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size_pinn)
        
    def forward(self, x):
        output, _ = self.lstm(x) # output: (batch_size, seq_len, hidden_dim)
        output = output[:, -1, :] # (batch_size, hidden_dim)
        output = self.fc(output) # (batch_size, output_size)
        return output

# Combined Model
class HybridModel(nn.Module):
    def __init__(self, lstm_model, pinn_model):
        super(HybridModel, self).__init__()
        self.lstm_model = lstm_model
        self.pinn_model = pinn_model
    
    def forward(self, x):
        lstm_output = self.lstm_model(x)  # Sequential prediction
        pinn_output = self.pinn_model(x)  # Apply physical constraints on LSTM output
        return lstm_output, pinn_output

