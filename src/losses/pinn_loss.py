import torch
import torch.nn as nn
from .sys_evap_1008ver import Evaporator
from scipy.io import loadmat
from .prop_ref import Refrigerant 
from .prop_cool_evap import Coolant_Evaporator
from torch.nn import functional as F
import pandas as pd

class PINN_Loss(nn.Module):
    def __init__(self, alpha):
        super(PINN_Loss, self).__init__()
        self.alpha = alpha

    def _unnormalize(self, item, column):
        stats = pd.read_csv("dataset/statistics.csv")
        item_un = torch.tensor(stats.loc[1, column]) * item + torch.tensor(stats.loc[0, column])
        return item_un

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
        
        # balance_loss calculated
        balance_losses = []
        for idx in range(len(model_output)):
            # unnnormalize the data
            p_input_un = self._unnormalize(p_input[idx], "pressure")
            h_ref_out_input_un = self._unnormalize(h_ref_out_input[idx], "h_ref_out")
            m_ref_in_un = self._unnormalize(m_ref_in[idx], "m_ref_in")
            m_ref_out_un = self._unnormalize(m_ref_out[idx], "m_ref_out")
            h_ref_in_un = self._unnormalize(h_ref_in[idx], "h_ref_in")
            m_cool_un = self._unnormalize(m_cool[idx], "m_cool")
            T_cool_in_un = self._unnormalize(T_cool_in[idx], "T_cool_in")
            zeta_un = self._unnormalize(zeta[idx], "z_tpsh")
            gamma_un = self._unnormalize(gamma[idx], "gamma")
            eps_tp_un = self._unnormalize(eps_tp[idx], "eps_tp")
            eps_sh_un = self._unnormalize(eps_sh[idx], "eps_sh")
            p_pred_un = self._unnormalize(p_pred[idx], "pressure")
            h_ref_out_pred_un = self._unnormalize(h_ref_out_pred[idx], "h_ref_out")
            
            x = torch.cat((p_pred_un, h_ref_out_pred_un), dim=0)
            u = torch.cat((m_ref_in_un, m_ref_out_un, h_ref_in_un, m_cool_un, T_cool_in_un), dim=0)
            p = torch.cat((zeta_un, gamma_un, eps_tp_un, eps_sh_un), dim=0)

            mass, rhs = self._Evaporator(x, u, p) # mass, rhs is calculated based on correct answers.
            mass = mass.detach()
            rhs = rhs.detach()

            dp_dt_mod = (p_pred_un - p_input_un) / time_step
            dh_dt_mod = (h_ref_out_pred_un - h_ref_out_input_un) / time_step
            dx_dt_mod = torch.cat((dp_dt_mod, dh_dt_mod), dim=0).unsqueeze(-1) # (2,1)

            balance_loss = (torch.matmul(mass, dx_dt_mod) - rhs) ** 2
            balance_losses.append(balance_loss)
            
        balance_loss = torch.stack(balance_losses).mean() 
        residual_loss = F.mse_loss(input=model_output[:, :2], target=ground_truth[:, :2])

        print("loss", balance_loss, residual_loss)

        # alpha = (1-self.gamma) * alpha+ self.gamma * alpha 
        total_loss =  residual_loss + 0.000001 * balance_loss 

        return total_loss