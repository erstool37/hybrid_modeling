import torch
import torch.nn as nn
from torch.nn import functional as F

class ResLoss(nn.Module):
    def __init__(self, rate, model):
        super(ResLoss, self).__init__()

    def forward(self, model_input, model_output, ground_truth, time_step):
        # model_input and time_step are not used but are kept for compatibility with other losses
        loss_res = F.mse_loss(input=model_output[:, :2], target=ground_truth[:, :2])
        # loss_res = torch.mean((torch.log1p(torch.clamp(model_output[:,:2], min=0)) - torch.log1p(torch.clamp(ground_truth[:, :2], min=0))) ** 2)

        return loss_res