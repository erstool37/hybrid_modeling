import torch
import torch.nn as nn

class MSLE(nn.Module):
    def __init__(self, w_res, w_theta, w_ode, time_step=None, descaler=None):
        super(MSLE, self).__init__()

    def forward(self, model_input, model_output, target):
        return torch.mean((torch.log1p(torch.clamp(model_output, min=0)) - torch.log1p(torch.clamp(target, min=0))) ** 2)
    
