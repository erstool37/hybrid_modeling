import torch
import torch.nn as nn
from torch.nn import functional as F

class MSE(nn.Module):
    def __init__(self, w_res, w_theta, w_ode, time_step=None, descaler=None):
        super(MSE, self).__init__()

    def forward(self, model_input, model_output, target):
        loss_res = F.mse_loss(input=model_output[:, :2], target=target[:, :2])

        return loss_res