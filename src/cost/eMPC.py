import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import importlib
from utils import MAPEcalculator, MAPEtestcalculator, setseed, inference, Xdenormalizer, Udenormalizer

class eMPC(nn.Module):
    def __init__(self, T_target, descaler):
        super(eMPC, self).__init__()
        self.T_target = T_target
        self.descaler = descaler

    def forward(self, x_pred, u_now):
        p_next, h_next = Xdenormalizer(x_pred, self.descaler, "optim")
        m_ref_in_next, m_ref_out_next, h_ref_in_next, m_cool_next, T_cool_next = Udenormalizer(u_now, self.descaler, "optim")

        loss_T = F.mse_loss(self.T_target - T_cool_next)
        loss_econ = F.mse_loss(T_cool_next - self.T_target)

        loss_total = loss_T + loss_econ

        wandb.log({"loss_T": loss_T.item()})
        wandb.log({"loss_econ": loss_econ.item()})
        wandb.log({"loss_total": loss_total.item()})

        wandb.log({"pressure_pred": p_next})
        wandb.log({"enthalpy_pred": h_next})

        return loss_total