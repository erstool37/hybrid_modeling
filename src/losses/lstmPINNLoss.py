import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy.io import loadmat
import pandas as pd
import wandb
from .calculator.sys_evap_1008ver import Evaporator
from .calculator.prop_ref import Refrigerant 
from .calculator.prop_cool_evap import Coolant_Evaporator
from src.utils.Paraloader import Paraloader as P

class lstmPINNLoss(nn.Module):
    def __init__(self, rate, model):
        super(lstmPINNLoss, self).__init__()
        self.rate = rate
        self.model = model
        self.x_min = torch.tensor([100., 270.], dtype=torch.float32)
        self.x_max = torch.tensor([360., 380.], dtype=torch.float32)
        self.scale_grad = 1. / (self.x_max - self.x_min)
        
        self.adaptive_constant_ode = torch.tensor(1.0, dtype=torch.float32, requires_grad=False)
        self.adaptive_constant_theta = torch.tensor(1.0, dtype=torch.float32, requires_grad=False)
        self.adaptive_constant_res_log = []
        self.adaptive_constant_ode_log = []
        self.adaptive_constant_theta_log = []

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

        Evap = Evaporator(Ref, Cool_evap)
        mass, rhs = Evap._system_dynamics(x, u, p)

        return mass, rhs
    
    def _compute_adaptive_constant(self, loss_res, loss_ode, loss_theta, model):
        if model.training:
            max_grad_res_list = []
            mean_grad_ode_list = []
            mean_grad_theta_list = []

            grad_res = torch.autograd.grad(outputs=loss_res, inputs=model.parameters(), retain_graph=True, create_graph=True)
            grad_ode = torch.autograd.grad(outputs=loss_ode, inputs=model.parameters(), retain_graph=True, create_graph=True)
            grad_theta = torch.autograd.grad(outputs=loss_theta, inputs=model.parameters(), retain_graph=True, create_graph=True)

            for res_grad, ode_grad, theta_grad in zip(grad_res, grad_ode, grad_theta):
                max_grad_res_list.append(torch.max(torch.abs(res_grad))) # max for each layer
                mean_grad_ode_list.append(torch.mean(torch.abs(ode_grad))) # mean for each layer
                mean_grad_theta_list.append(torch.mean(torch.abs(theta_grad)))

            # in case of no gradients
            max_grad_res = torch.max(torch.stack(max_grad_res_list)) if len(max_grad_res_list) > 0 else torch.tensor(0.0, device=loss_res.device) # max for all layers
            mean_grad_ode = torch.mean(torch.stack(mean_grad_ode_list)) if len(mean_grad_ode_list) > 0 else torch.tensor(1.0, device=loss_res.device) # mean for all layers
            mean_grad_theta = torch.mean(torch.stack(mean_grad_theta_list)) if (len(mean_grad_theta_list)) > 0 else torch.tensor(1.0, device=loss_res.device)

            # in case of dividion by zero
            adaptive_constant_ode = max_grad_res / mean_grad_ode if mean_grad_ode > 0 else torch.tensor(1.0)
            adaptive_constant_ode = adaptive_constant_ode.detach()

            adaptive_constant_theta = max_grad_res / mean_grad_theta if mean_grad_theta > 0 else torch.tensor(1.0)
            adaptive_constant_theta = adaptive_constant_theta.detach()

            self.adaptive_constant_ode = (1 - self.rate) * self.adaptive_constant_ode + self.rate * adaptive_constant_ode
            self.adaptive_constant_ode_log.append(self.adaptive_constant_ode.item())
            
            self.adaptive_constant_theta = (1 - self.rate) * self.adaptive_constant_theta + self.rate * adaptive_constant_theta
            self.adaptive_constant_theta_log.append(self.adaptive_constant_theta.item())

        else:
            pass

    def forward(self, model_input, model_output, ground_truth, time_step):
        # time_step is not required, but kept for train.py code compatibility
        time = model_input[:,-1,:2].T.unsqueeze(-1)
        p_input, h_ref_out_input = model_input[:,-1,:2].T.unsqueeze(-1) # Present time step state variables
        m_ref_in, m_ref_out, h_ref_in, m_cool, T_cool_in = model_input[:, -1, 2:].T.unsqueeze(-1) # Present time step input variables
        # p_true, h_ref_out_true = ground_truth[:, :2].T.unsqueeze(-1) # True next time step state variables
        p_pred, h_ref_out_pred = model_output[:, :2].T.unsqueeze(-1) # Predicted next time step state variables
        zeta, gamma, eps_tp, eps_sh = ground_truth[:, 2:].T.unsqueeze(-1) # True present time step hidden parameters

        # ODE based loss calculation
        p_input_un = P.unnormalize(p_input, "pressure", "total")
        h_ref_out_input_un = P.unnormalize(h_ref_out_input, "h_ref_out", "total")
        m_ref_in_un = P.unnormalize(m_ref_in, "m_ref_in", "total")
        m_ref_out_un = P.unnormalize(m_ref_out, "m_ref_out", "total")
        h_ref_in_un = P.unnormalize(h_ref_in, "h_ref_in", "total")
        m_cool_un = P.unnormalize(m_cool, "m_cool", "total")
        T_cool_in_un = P.unnormalize(T_cool_in, "T_cool_in", "total")
        zeta_un = P.unnormalize(zeta, "z_tpsh", "total")
        gamma_un = P.unnormalize(gamma, "gamma", "total")
        eps_tp_un = P.unnormalize(eps_tp, "eps_tp", "total")
        eps_sh_un = P.unnormalize(eps_sh, "eps_sh", "total")
        p_pred_un = P.unnormalize(p_pred, "pressure", "total")
        h_ref_out_pred_un = P.unnormalize(h_ref_out_pred, "h_ref_out", "total")
            
        x = torch.cat((p_input_un, h_ref_out_input_un), dim=-1)
        u = torch.cat((m_ref_in_un, m_ref_out_un, h_ref_in_un, m_cool_un, T_cool_in_un), dim=-1)
        p = torch.cat((zeta_un, gamma_un, eps_tp_un, eps_sh_un), dim=-1)
        x_ans = torch.cat((p_pred_un, h_ref_out_pred_un), dim=-1)
        mass, rhs = self._Evaporator(x, u, p)
        mass = mass.detach()
        rhs = rhs.detach().unsqueeze(-1)

        dp_dt = (p_pred_un - p_input_un) / 2
        dh_dt = (h_ref_out_pred_un - h_ref_out_input_un) / 2
        dx_dt = torch.cat((dp_dt, dh_dt), dim=-1).unsqueeze(-1)
        loss_ode = torch.bmm(mass, dx_dt) - rhs

        # loss calculation
        def MSLE(input, target):
            loss = torch.mean((torch.log1p(torch.clamp(input, min=0)) - torch.log1p(torch.clamp(target, min=0))) ** 2)
            return loss

        loss_res = F.mse_loss(input=model_output[:, :2], target=ground_truth[:, :2])
        loss_theta = F.mse_loss(input=model_output[:, 2:], target=ground_truth[:, 2:])
        loss_ode = F.mse_loss(input=loss_ode, target=torch.zeros_like(rhs))
        
        # self._compute_adaptive_constant(loss_res, loss_ode, loss_theta, self.model)

        # grad_res_data = torch.autograd.grad(loss_res, self.model.parameters(), retain_graph=True, create_graph=True)
        # grad_theta_data = torch.autograd.grad(loss_theta, self.model.parameters(), retain_graph=True, create_graph=True)
        # grad_ode_data = torch.autograd.grad(loss_ode, self.model.parameters(), retain_graph=True, create_graph=True)
    
        # grad_res = sum(g.norm() for g in grad_res_data)
        # grad_theta = sum(g.norm() for g in grad_theta_data)
        # grad_ode = sum(g.norm() for g in grad_ode_data)

        # constant_res = grad_res / (grad_res+ grad_theta + grad_ode)
        # constant_theta = grad_theta / (grad_res+ grad_theta + grad_ode)
        # constant_ode = grad_ode / (grad_res+ grad_theta + grad_ode)

        # total_loss = constant_res * loss_res + constant_theta * loss_theta + constant_ode * loss_ode
        # total_loss = loss_res + self.adaptive_constant_theta * loss_theta + self.adaptive_constant_ode * loss_ode
        total_loss = loss_res + 5 * loss_theta + loss_ode

        wandb.log({"loss_res": loss_res})
        wandb.log({"loss_theta":loss_theta})
        wandb.log({"loss_ode": loss_ode})
        # wandb.log({"loss_theta_chunk": self.adaptive_constant_theta * loss_theta})
        # wandb.log({"loss_ode_chunk": self.adaptive_constant_ode * loss_ode})
        wandb.log({"loss_theta_chunk": 5 * loss_ode})

        return total_loss


# for previous lstmPINN loss
        # dp_dt_mod = (p_pred_un - p_input_un) / time_step # (batch_size, 1)
        # dh_dt_mod = (h_ref_out_pred_un - h_ref_out_input_un) / time_step # (batch_size, 1)
        # dx_dt_mod = torch.cat((dp_dt_mod, dh_dt_mod), dim=-1).unsqueeze(-1) # (batch_size, 2, 1)
        # dx_dt_mod = zero_one_scale(dx_dt_mod, self.x_min, self.x_max) / self.scale_grad.unsqueeze(-1).unsqueeze(0).to(dx_dt_mod.device)# scaling match
        # print(dx_dt_mod)
        # loss_ode = torch.bmm(mass, dx_dt_mod) - rhs  