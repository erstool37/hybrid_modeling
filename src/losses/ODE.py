import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy.io import loadmat
import pandas as pd
import wandb
from .calculator.sys_evap_1008ver import Evaporator
from .calculator.prop_ref import Refrigerant 
from .calculator.prop_cool_evap import Coolant_Evaporator
import importlib

class ODE(nn.Module):
    def __init__(self, w_res, w_theta, w_ode, time_step, descaler):
        super(ODE, self).__init__()
        self.time_step = time_step

        # self.x_min = torch.tensor([100., 270.], dtype=torch.float32)
        # self.x_max = torch.tensor([360., 380.], dtype=torch.float32)
        # self.scale_grad = 1. / (self.x_max - self.x_min)

        self.w_res = torch.tensor(w_res, dtype=torch.float32)
        self.w_theta = torch.tensor(w_theta, dtype=torch.float32)
        self.w_ode = torch.tensor(w_ode, dtype=torch.float32)

        utils = importlib.import_module("utils")
        self.descaler = getattr(utils, descaler)
    
    def _Evaporator(self, x, u, p):
        # Refrigerant property functions
        coeff_ref_data = loadmat("src/losses/calculator/coefficients_ref.mat")
        coeff_ref_data = coeff_ref_data["coeff_ref"]
        coeff_ref_data = {field: coeff_ref_data[field][0, 0] for field in coeff_ref_data.dtype.names}
        Ref = Refrigerant(coeff_ref_data)
        
        # Evaporator coolant property functions
        coeff_cool_evap_data = loadmat("src/losses/calculator/coefficients_cool_evap.mat")
        coeff_cool_evap_data = coeff_cool_evap_data["coefficients_cool_evap"]
        coeff_cool_evap_data = {field: coeff_cool_evap_data[field][0, 0] for field in coeff_cool_evap_data.dtype.names}
        Cool_evap = Coolant_Evaporator(coeff_cool_evap_data)

        # ODE calculation
        Evap = Evaporator(Ref, Cool_evap)
        mass, rhs = Evap._system_dynamics(x, u, p)

        return mass, rhs

    def forward(self, model_input, model_output, target):
        p_input, h_ref_out_input = model_input[:,-1,:2].T.unsqueeze(-1) # x(t)
        m_ref_in, m_ref_out, h_ref_in, m_cool, T_cool_in = model_input[:, -1, 2:].T.unsqueeze(-1) # u(t)
        p_true, h_ref_out_true = target[:, :2].T.unsqueeze(-1) # x_true(t+1)
        p_pred, h_ref_out_pred = model_output[:, :2].T.unsqueeze(-1) # x(t+1)
        zeta, gamma, eps_tp, eps_sh = target[:, 2:].T.unsqueeze(-1) # p(t)

        

        p_input_un = self.descaler(p_input, "pressure", "total")
        h_ref_out_input_un = self.descaler(h_ref_out_input, "h_ref_out", "total")
        m_ref_in_un = self.descaler(m_ref_in, "m_ref_in", "total")
        m_ref_out_un = self.descaler(m_ref_out, "m_ref_out", "total")
        h_ref_in_un = self.descaler(h_ref_in, "h_ref_in", "total")
        m_cool_un = self.descaler(m_cool, "m_cool", "total")
        T_cool_in_un = self.descaler(T_cool_in, "T_cool_in", "total")
        zeta_un = self.descaler(zeta, "z_tpsh", "total")
        gamma_un = self.descaler(gamma, "gamma", "total")
        eps_tp_un = self.descaler(eps_tp, "eps_tp", "total")
        eps_sh_un = self.descaler(eps_sh, "eps_sh", "total")
        p_pred_un = self.descaler(p_pred, "pressure", "total")
        h_ref_out_pred_un = self.descaler(h_ref_out_pred, "h_ref_out", "total")
            
        x = torch.cat((p_input_un, h_ref_out_input_un), dim=-1)
        u = torch.cat((m_ref_in_un, m_ref_out_un, h_ref_in_un, m_cool_un, T_cool_in_un), dim=-1)
        p = torch.cat((zeta_un, gamma_un, eps_tp_un, eps_sh_un), dim=-1)

        mass, rhs = self._Evaporator(x, u, p)
        mass = mass.detach()
        rhs = rhs.detach().unsqueeze(-1)

        dp_dt = (p_pred_un - p_input_un) / self.time_step
        dh_dt = (h_ref_out_pred_un - h_ref_out_input_un) / self.time_step
        dx_dt = torch.cat((dp_dt, dh_dt), dim=-1).unsqueeze(-1)

        loss_ode = torch.bmm(mass, dx_dt) - rhs
        loss_res = F.mse_loss(input=model_output[:, :2], target=target[:, :2])
        loss_theta = F.mse_loss(input=model_output[:, 2:], target=target[:, 2:])
        loss_ode = F.mse_loss(input=loss_ode, target=torch.zeros_like(rhs))

        total_loss = self.w_res * loss_res + self.w_theta * loss_theta + self.w_ode *loss_ode

        wandb.log({"loss_res": loss_res})
        wandb.log({"loss_theta": loss_theta})
        wandb.log({"loss_ode": loss_ode})
        wandb.log({"loss_res_chunk":  self.w_res * loss_res})
        wandb.log({"loss_theta_chunk": self.w_theta * loss_theta})
        wandb.log({"loss_ode_chunk": self.w_ode * loss_ode})
        
        return total_loss