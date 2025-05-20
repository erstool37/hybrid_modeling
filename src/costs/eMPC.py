import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from utils import Xdenormalizer, Udenormalizer, Odenormalizer
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
        # Unnormalize
        _, h_ref_out_next = Xdenormalizer(x_pred, self.descaler, "optim")
        m_ref_in_optim, _, h_ref_in_optim, m_cool, T_cool_in  = Udenormalizer(u_now, self.descaler, "optim")
        T_ref_in, T_ref_out, T_cool_out, z_tpsh = Odenormalizer(model_input, self.descaler, "optim")
        
        # T_cool_out calculation
        Q_ref = m_ref_in_optim * (h_ref_out_next - h_ref_in_optim)
        C_cool = m_cool * self.CE.Cp((T_cool_out.unsqueeze(0).unsqueeze(-1)+T_cool_in.unsqueeze(0).unsqueeze(-1))/2)
        T_cool_out_pred =  -1 * Q_ref / C_cool + T_cool_in.detach()
        T_cool_out_pred = T_cool_out_pred.squeeze(0).squeeze(-1)

        # Cost function
        loss_T_pred = T_cool_out - T_cool_out_pred
        loss_T = F.mse_loss(self.T_target, T_cool_out_pred)
        # loss_Q = -0.2 * Q_ref

        # Constaints
        loss_pos = F.softmax(-m_ref_in_optim.detach(), dim=-1) + F.softmax(-h_ref_in_optim, dim=-1)
        loss_zeta = F.relu(1.0 - z_tpsh)
        loss_h = F.relu(180.0 - h_ref_in_optim) + F.relu(h_ref_in_optim - 240.0)
        loss_m = F.relu(0.01 - m_ref_in_optim) + F.relu(m_ref_in_optim - 0.04)

        loss_total = loss_T + 3 * loss_pos + (loss_zeta + loss_h + loss_m)

        return loss_total, T_cool_out_pred