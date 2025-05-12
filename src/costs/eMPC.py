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

    def forward(self, x_pred, model_input, u_now):
        p_ref_out_next, h_ref_out_next = Xdenormalizer(x_pred, self.descaler, "optim")
        m_ref_in_next, _, h_ref_in_next, m_cool_next, T_cool_in  = Udenormalizer(u_now, self.descaler, "optim")
        T_ref_in, T_ref_out, T_cool_out, z_tpsh = Odenormalizer(model_input, self.descaler, "optim")
        
        T_cool_out_pred = m_ref_in_next * (h_ref_out_next - h_ref_in_next) / (m_cool_next * self.CE.Cp(T_cool_out.unsqueeze(0).unsqueeze(-1))) + T_cool_in.detach()
        T_cool_out_pred = T_cool_out_pred.squeeze(0).squeeze(-1)
        
        loss_T = F.mse_loss(self.T_target, T_cool_out_pred)

        # h_ref_out_desired = (self.T_target - T_cool_in.detach()) * (m_cool_next.detach() * self.CE.Cp(T_cool_out.unsqueeze(0).unsqueeze(-1))) / m_ref_in_next +  h_ref_in_next
        # h_ref_out_desired = h_ref_out_desired.squeeze(0).squeeze(-1)

        # self.ref.liq_hsat(p_ref_out_next.unsqueeze(0).unsqueeze(-1)) * m_ref_in_next 
        # self.ref.vap_hsat(p_ref_out_next.unsqueeze(0).unsqueeze(-1))

        loss_pos = F.softplus(-m_cool_next.detach()) + F.softplus(-h_ref_in_next) + F.softplus(-T_cool_in)
        # loss_T = F.mse_loss(h_ref_out_next, h_ref_out_desired)
        
        loss_total = loss_T + 1000 * loss_pos

        # 1. h_ref 가 extrapolated
        # 2. 음수값
        wandb.log({"m_cool_nexst": m_cool_next})
        wandb.log({"T_cool_out_pred": T_cool_out_pred})
        wandb.log({"loss_T": loss_T.item()})
        wandb.log({"loss_total": loss_total.item()})
        wandb.log({"T_cool_out": T_cool_out.item()})
        wandb.log({"predicted_pressure": p_ref_out_next})
        wandb.log({"predicted_enthalpy": h_ref_out_next})

        return loss_total
