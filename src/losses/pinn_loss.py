import torch
import torch.nn as nn
from .sys_evap_1008ver import Evaporator
from scipy.io import loadmat
# from sys_evap_mb import Evaporator_moving_boundary 
from .prop_ref import Refrigerant 
from .prop_cool_evap import Coolant_evaporator
from .sys_evap_1008ver import Evaporator as Evap

class PINN_Loss(nn.Module):
    def __init__(self, alpha):
        super(PINN_Loss, self).__init__()
        self.alpha = alpha

    def _Evaporator(self, x, u, p):
        # Refrigerant property functions
        coeff_ref_data = loadmat ("coefficients_ref.mat")
        coeff_ref_data = coeff_ref_data[ "coeff_ref"]
        coeff_ref_data = {field: coeff_ref_data[field][0, 0] for field in coeff_ref_data.dtype.names}
        Ref = Refrigerant(coeff_ref_data)
        
        # Evaporator coolant property functions
        coeff_cool_evap_data = loadmat("coefficients_cool_evap-mat")
        coeff_cool_evap_data = coeff_cool_evap_data["coefficients_cool_evap"]
        coeff_cool_evap_data = {field: coeff_cool_evap_data[field][0, 0] for field in coeff_cool_evap_data.dtype.names}
        Cool_evap = Coolant_evaporator(coeff_cool_evap_data)

        Evap = Evap(Ref, Cool_evap)
        dp_dt_bal, dh_dt_bal = Evap._system_dynamics(x, u, p)

        return dp_dt_bal, dh_dt_bal

    def forward(self, xdot_model, model_output, ground_truth):
        """
        prev_state: [pressure, h_ref_out] (batch_size, state_dim)
        input_vars: [m_ref_in, m_ref_out, h_ref_in, m_cool, T_cool_in] (batch_size, input_dim)
        target_state: [p_true, h_ref_out_true, zeta, gamma, eps_tp, eps_sh] (batch_size, state_dim)
        """

        p_true, h_ref_out_true = ground_truth[:, :2].T # True next time step state variables
        p_pred, h_ref_out_pred = model_output[:, :2].T # Predicted next time step state variables
        zeta, gamma, eps_tp, eps_sh = model_output[:, 2:].T # Predicted present time step hidden parameters

        balance_losses = []
        for idx in range(len(model_output)): # detachment is necessary for casadi does not support gpu
            x = (p_pred[idx], h_ref_out_pred[idx])
            u = model_output[idx]
            p = (zeta[idx], gamma[idx], eps_tp[idx], eps_sh[idx])
            
            dp_dt_mod, dh_dt_mod = xdot_model # xdot from torch grad calculation
            dp_dt_bal, dh_dt_bal = self._Evaporator(x, u, p) # xdot from Mass/Energy Balance Equations

            balance_loss = torch.mean((dp_dt_bal - dp_dt_mod) ** 2) + torch.mean((dh_dt_bal - dh_dt_mod) ** 2)
            balance_losses.append(balance_loss)

        state_loss = torch.mean((p_pred - p_true) ** 2) + torch.mean((h_ref_out_pred - h_ref_out_true) ** 2)
        balance_loss = torch.mean(balance_losses)

        total_loss =  state_loss + self.alpha * balance_loss 

        return total_loss
