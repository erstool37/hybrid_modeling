import torch
import torch.nn as nn
from .sys_evap_1008ver import Evaporator
from scipy.io import loadmat
# from sys_evap_mb import Evaporator_moving_boundary 
from .prop_ref import Refrigerant 
from .prop_cool_evap import Coolant_Evaporator
from statistics import mean

class PINN_Loss(nn.Module):
    def __init__(self, alpha):
        super(PINN_Loss, self).__init__()
        self.alpha = alpha

    def _Evaporator(self, x, u, p):
        # Refrigerant property functions
        coeff_ref_data = loadmat("src/losses/coefficients_ref.mat")
        coeff_ref_data = coeff_ref_data["coeff_ref"]
        coeff_ref_data = {field: coeff_ref_data[field][0, 0] for field in coeff_ref_data.dtype.names}
        Ref = Refrigerant(coeff_ref_data)
        
        # Evaporator coolant property functions
        coeff_cool_evap_data = loadmat("src/losses/coefficients_cool_evap.mat")
        coeff_cool_evap_data = coeff_cool_evap_data["coefficients_cool_evap"]
        coeff_cool_evap_data = {field: coeff_cool_evap_data[field][0, 0] for field in coeff_cool_evap_data.dtype.names}
        Cool_evap = Coolant_Evaporator(coeff_cool_evap_data)

        Evap = Evaporator(Ref, Cool_evap)
        mass, rhs = Evap._system_dynamics(x, u, p)

        return mass, rhs

    def forward(self, model_input, model_output, ground_truth, time_step):
        """
        prev_state: [pressure, h_ref_out] (batch_size, state_dim)
        input_vars: [m_ref_in, m_ref_out, h_ref_in, m_cool, T_cool_in] (batch_size, input_dim)
        target_state: [p_true, h_ref_out_true, zeta, gamma, eps_tp, eps_sh] (batch_size, state_dim)
        """
        
        p_input, h_ref_out_input = model_input[:,-1,:2].T.unsqueeze(-1) # Present time step state variables
        m_ref_in, m_ref_out, h_ref_in, m_cool, T_cool_in = model_input[:, -1, 2:].T.unsqueeze(-1) # Present time step input variables
        p_true, h_ref_out_true = ground_truth[:, :2].T.unsqueeze(-1) # True next time step state variables
        p_pred, h_ref_out_pred = model_output[:, :2].T.unsqueeze(-1) # Predicted next time step state variables
        zeta, gamma, eps_tp, eps_sh = model_output[:, 2:].T.unsqueeze(-1) # Predicted present time step hidden parameters
        balance_losses = []

        for idx in range(len(model_output)): 
            x = torch.cat((p_pred[idx], h_ref_out_pred[idx]), dim=0)
            u = torch.cat((m_ref_in[idx], m_ref_out[idx], h_ref_in[idx], m_cool[idx], T_cool_in[idx]), dim=0)
            p = torch.cat((zeta[idx], gamma[idx], eps_tp[idx], eps_sh[idx]), dim=0)

            dp_dt_mod = (p_pred[idx] - p_input[idx]) / time_step
            dh_dt_mod = (h_ref_out_pred[idx] - h_ref_out_input[idx]) / time_step
            dx_dt_mod = torch.cat((dp_dt_mod, dh_dt_mod), dim=0).unsqueeze(-1)

            mass, rhs = self._Evaporator(x, u, p) # xdot from Mass/Energy Balance Equations

            balance_loss = torch.norm(torch.matmul(mass, dx_dt_mod) - rhs, p=2, dim=0)
            balance_losses.extend(balance_loss.tolist())

        balance_loss = mean(balance_losses)
        state_loss = torch.mean((p_pred - p_true) ** 2) + torch.mean((h_ref_out_pred - h_ref_out_true) ** 2)
    
        total_loss =  state_loss + self.alpha * balance_loss 

        return total_loss
