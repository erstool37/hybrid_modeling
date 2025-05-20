import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import importlib
from scipy.io import loadmat
from utils import MAPEcalculator, MAPEtestcalculator, setseed, inference, Xdenormalizer, Udenormalizer, Odenormalizer
from losses.calculator.prop_cool_evap import Coolant_Evaporator
from losses.calculator.prop_ref import Refrigerant

class eMPC(nn.Module):
    def __init__(self, T_target, descaler):
        super(eMPC, self).__init__()
        self.T_target = torch.tensor(T_target)
        self.descaler = descaler

        coeff_cool_evap_data = loadmat("src/losses/calculator/coefficients_cool_evap.mat")
        coeff_cool_evap_data = coeff_cool_evap_data["coefficients_cool_evap"]
        coeff_cool_evap_data = {field: coeff_cool_evap_data[field][0, 0] for field in coeff_cool_evap_data.dtype.names}
        self.CE = Coolant_Evaporator(coeff_cool_evap_data)

        coeff_ref_data = loadmat("src/losses/calculator/coefficients_ref.mat")
        coeff_ref_data = coeff_ref_data["coeff_ref"]
        coeff_ref_data = {field: coeff_ref_data[field][0, 0] for field in coeff_ref_data.dtype.names}
        self.ref = Refrigerant(coeff_ref_data)

    def forward(self, x_horizon, u_horizon, p_horizon, others):
        for idx in range(x_horizon.shape[1]):
            x_horizon[:, idx, :] = Xdenormalizer(x_horizon[:, idx, :], self.descaler, "optim")
            u_horizon[:, idx, :] = Udenormalizer(u_horizon[:, idx, :], self.descaler, "optim")
            p_horizon[:, idx, :] = Odenormalizer(p_horizon[:, idx, :], self.descaler, "optim")

        T_ref_in = others[:,7]
        T_ref_out = others[:,8]
        T_cool_out = others[:,9]
        T_cool_in = u_horizon[:,0,4].squeeze(1)
        m_cool_in = u_horizon[:,0,3].squeeze(1)
        m_ref_in = u_horizon[:,0,0].squeeze(1)
        h_ref_in = u_horizon[:,0,2].squeeze(1)
        
        # setpoint calculation
        p_ref_out_sp = 
        h_ref_out_sp = (self.T_target - T_cool_in) * (m_cool_in * self.CE.Cp(T_cool_out.unsqueeze(0).unsqueeze(-1))) / m_ref_in + h_ref_in

        p_sp = self.ref.Psat(T_ref_out)
        x_sp = torch.stack([p_sp, h_ref_out_sp], dim=1).to(x_horizon.device).view()

        u_sp = 

        # loss setting
        loss_x = F.mse_loss(x_horizon[:,:,:4], x_sp[:,:,:4])
        loss_u = F.mse_loss(u_horizon, u_sp)
        loss_final = F.mse_loss(x_horizon[:,:,4], x_sp[:,:,4])

        total_loss = 2 * loss_x + 2 * loss_u + 10 * loss_final

        wandb.log({"loss_x": loss_x})
        wandb.log({"loss_u": loss_u})
        wandb.log({"loss_final": loss_final})
        wandb.log({"total_loss": total_loss})

        return total_loss