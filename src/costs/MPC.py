import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from utils import Xdenormalizer, Udenormalizer, Odenormalizer
from losses.calculator.prop_cool_evap import Coolant_Evaporator
from losses.calculator.prop_ref import Refrigerant

class MPC(nn.Module):
    def __init__(self, T_target, descaler):
        super(MPC, self).__init__()
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
        # Unnormalize
        _, h_ref_out_next = Xdenormalizer(x_pred, self.descaler, "optim")
        m_ref_in_optim, _, h_ref_in_optim, m_cool, T_cool_in  = Udenormalizer(u_now, self.descaler, "optim")
        T_ref_in, T_ref_out, T_cool_out, z_tpsh = Odenormalizer(model_input, self.descaler, "optim")
        
        # T_cool_out calculation
        Q_ref = m_ref_in_optim * (h_ref_out_next - h_ref_in_optim)
        C_cool = m_cool * self.CE.Cp((T_cool_out.unsqueeze(0).unsqueeze(-1)+T_cool_in.unsqueeze(0).unsqueeze(-1))/2)
        T_cool_out_pred =  -1 * Q_ref / C_cool + T_cool_in.detach()
        T_cool_out_pred = T_cool_out_pred.squeeze(0).squeeze(-1) + 1 

        # Cost function
        loss_T_pred = F.mse_loss(T_cool_out_pred, T_cool_out)
        loss_T = F.mse_loss(self.T_target, T_cool_out_pred)
        # loss_Q = -0.2 * Q_ref

        # Constaints
        loss_pos = F.softplus(-m_ref_in_optim) + F.softplus(-h_ref_in_optim)
        loss_zeta = F.softplus(z_tpsh - 1.0)
        # loss_h = F.relu(180.0 - h_ref_in_optim) + F.relu(h_ref_in_optim - 240.0)
        loss_m_cool = F.relu(0.2 - m_cool) + F.relu(m_cool - 0.8)
        loss_m = F.relu(0.01 - m_ref_in_optim) + F.relu(m_ref_in_optim - 0.05)

        loss_total = 3 * loss_T + (loss_zeta) + (loss_m + loss_m_cool)

        return loss_total, T_cool_out_pred