import torch
import torch.nn as nn
from .sys_evap_1008ver import Evaporator
from scipy.io import loadmat
from .prop_ref import Refrigerant 
from .prop_cool_evap import Coolant_Evaporator
from torch.nn import functional as F
import pandas as pd
import wandb

class PINN_Loss(nn.Module):
    def __init__(self, rate, model):
        super(PINN_Loss, self).__init__()
        self.rate = rate
        self.adaptive_constant_ode = torch.tensor(1.0, dtype=torch.float32, requires_grad=False)
        
        self.adaptive_constant_res_log = []
        self.adaptive_constant_bcs_log = []
        self.model = model

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
    
    def _compute_adaptive_constant(self, loss_res, loss_bcs, model):
        max_grad_res_list = []
        mean_grad_ode_list = []

        grad_res = torch.autograd.grad(outputs=loss_res, inputs=model.parameters(), retain_graph=True, create_graph=True)
        grad_ode = torch.autograd.grad(outputs=loss_bcs, inputs=model.parameters(), retain_graph=True, create_graph=True)

        print(len(grad_res))
        print(len(grad_ode))

        for res_grad, ode_grad in zip(grad_res, grad_ode):
            max_grad_res_list.append(torch.max(torch.abs(res_grad))) # max for each layer
            mean_grad_ode_list.append(torch.mean(torch.abs(ode_grad))) # mean for each layer

        # in case of no gradients
        max_grad_res = torch.max(torch.stack(max_grad_res_list)) if len(max_grad_res_list) > 0 else torch.tensor(0.0, device=loss_res.device) # max for all layers
        mean_grad_ode = torch.mean(torch.stack(mean_grad_ode_list)) if len(mean_grad_ode_list) > 0 else torch.tensor(1.0, device=loss_res.device) # mean for all layers

        # in case of dividion by zero
        adaptive_constant_ode = max_grad_res / mean_grad_ode if mean_grad_ode > 0 else torch.tensor(1.0)
        adaptive_constant_ode = adaptive_constant_ode.detach()

        self.adaptive_constant_ode = (1 - self.rate) * self.adaptive_constant_ode + self.rate * adaptive_constant_ode
        self.adaptive_constant_bcs_log.append(self.adaptive_constant_ode.item())
        wandb.log({"adaptive_constant_ode": self.adaptive_constant_ode.item()})

    def forward(self, model_input, model_output, ground_truth, time_step):

        p_input, h_ref_out_input = model_input[:,-1,:2].T.unsqueeze(-1) # Present time step state variables
        m_ref_in, m_ref_out, h_ref_in, m_cool, T_cool_in = model_input[:, -1, 2:].T.unsqueeze(-1) # Present time step input variables
        # p_true, h_ref_out_true = ground_truth[:, :2].T.unsqueeze(-1) # True next time step state variables
        p_pred, h_ref_out_pred = model_output[:, :2].T.unsqueeze(-1) # Predicted next time step state variables
        zeta, gamma, eps_tp, eps_sh = model_output[:, 2:].T.unsqueeze(-1) # Predicted present time step hidden parameters

        # ODE based loss calculation
        p_input_un = self._unnormalize(p_input, "pressure")
        h_ref_out_input_un = self._unnormalize(h_ref_out_input, "h_ref_out")
        m_ref_in_un = self._unnormalize(m_ref_in, "m_ref_in")
        m_ref_out_un = self._unnormalize(m_ref_out, "m_ref_out")
        h_ref_in_un = self._unnormalize(h_ref_in, "h_ref_in")
        m_cool_un = self._unnormalize(m_cool, "m_cool")
        T_cool_in_un = self._unnormalize(T_cool_in, "T_cool_in")
        zeta_un = self._unnormalize(zeta, "z_tpsh")
        gamma_un = self._unnormalize(gamma, "gamma")
        eps_tp_un = self._unnormalize(eps_tp, "eps_tp")
        eps_sh_un = self._unnormalize(eps_sh, "eps_sh")
        p_pred_un = self._unnormalize(p_pred, "pressure")
        h_ref_out_pred_un = self._unnormalize(h_ref_out_pred, "h_ref_out") # (batch_size, 1)
            
        x = torch.cat((p_pred_un, h_ref_out_pred_un), dim=-1)
        u = torch.cat((m_ref_in_un, m_ref_out_un, h_ref_in_un, m_cool_un, T_cool_in_un), dim=-1)
        p = torch.cat((zeta_un, gamma_un, eps_tp_un, eps_sh_un), dim=-1)

        mass, rhs = self._Evaporator(x, u, p) 
        mass = mass.detach() # (batch_size, 2, 2)
        rhs = rhs.detach().unsqueeze(-1) # (batch_size, 2, 1)

        dp_dt_mod = (p_pred_un - p_input_un) / time_step # (batch_size, 1)
        dh_dt_mod = (h_ref_out_pred_un - h_ref_out_input_un) / time_step # (batch_size, 1)
        dx_dt_mod = torch.cat((dp_dt_mod, dh_dt_mod), dim=-1).unsqueeze(-1) # (batch_size, 2, 1)

        # loss calculation
        loss_ode = torch.bmm(mass, dx_dt_mod) - rhs
        loss_ode = F.mse_loss(loss_ode, target=torch.zeros_like(rhs))
        loss_res = F.mse_loss(input=model_output[:, :2], target=ground_truth[:, :2])
        
        self._compute_adaptive_constant(loss_ode, loss_res, self.model)

        total_loss = (loss_res + self.adaptive_constant_ode * loss_ode)

        return total_loss