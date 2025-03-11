import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout=0.0):
        super(PINN, self).__init__()

        self.lstm = nn.LSTM(input_size=7, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout) # x(t), u(t)
        self.fc = nn.Linear(hidden_dim, 6) # x(t+1), theta(t)

    def forward(self, model_input):
        """
        model_input = [pressure, h_ref_out, m_ref_in, m_ref_out, h_ref_in, m_cool, T_cool_in]
        """
        lstm_out, _ = self.lstm(model_input)  # lstm_out.shape = (batch_size, time_steps, hidden_dim)
        next_state = self.fc(lstm_out[:, -1, :])  # Shape = (batch_size, output_dim)
        
        return next_state