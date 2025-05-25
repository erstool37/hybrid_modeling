"""
Created on Fri Aug 30 18:51:43 2024

@author: jisung
"""

import torch
import numpy as np
from .utility import zero_one_scale, zero_one_descale
from utils import Xdenormalizer, Udenormalizer, Odenormalizer, Pdenormalizer
from torchdiffeq import odeint

class Evaporator(object):
    
    def __init__(self, Ref, Cool, *args):
        self.process_type = "continuous"
        
        if args:
            self.config = args[0]
            self.seed = self.config.seed
            self.hybrid = self.config.hybrid
            self.np_data_type = self.config.np_data_type
            self.time_interval = self.config.time_interval
            self.terminal_time = self.config.terminal_time
        else:
            # print("No user-defined configuration, using implemented setting")
            self.seed = 1999
            self.hybrid = False
            self.np_data_type = np.float64
            self.time_interval = self.np_data_type(2)
            self.terminal_time = self.np_data_type(1000)
            
        self.Ref = Ref
        self.Cool = Cool
        
        # Dimension of variables
        self.x_dim = 2 # pressure, outlet enthalpy
        self.u_dim = 5 # inlet mass, outlet mass, inlet enthalpy, utils flow rate, T
        self.y_dim = 2 
        self.p_dim = 4 # LSTM outputs
        
        # Initial value of variables
        self.x_ini = torch.tensor([250, 360], dtype=torch.float32)
        self.u_ini = torch.tensor([0.02, 0.02, 275, 0.6, -7], dtype=torch.float32)
        self.y_ini = torch.tensor([250, 360], dtype=torch.float32)
        self.p_ini = torch.tensor([0.8, 0.9, 0.2, 0.3], dtype=torch.float32)
        
        # Minimum / maximum values of variaibles
        self.x_min = torch.tensor([100., 270.], dtype=torch.float32)
        self.x_max = torch.tensor([360., 380.], dtype=torch.float32)
        
        self.u_min = torch.tensor([0.01, 0.01, 163., 0.0001, -20.], dtype=torch.float32)
        self.u_max = torch.tensor([0.05, 0.05, 365., 1.0000, +10.], dtype=torch.float32)
        
        self.y_min = torch.tensor([100., 270.], dtype=torch.float32)
        self.y_max = torch.tensor([360., 380.], dtype=torch.float32)
        
        self.p_min = torch.tensor([0., 0., 0., 0.], dtype=torch.float32)
        self.p_max = torch.tensor([1., 1., 1., 1.], dtype=torch.float32)
        
        self.scale_grad = 1. / (self.x_max - self.x_min)
        
        # Geometric parameters
        self.width = 0.1
        self.height = 0.005
        self.length = 0.5
        self.tubenum = 10
        self.V_flow = torch.tensor(self.width * self.height * self.length * self.tubenum)
        
        self.A_inner = 2 * (self.width + self.height) * self.length * self.tubenum
        self.A_outer = self.A_inner
        
        self.csa_ref = self.width * self.height * self.tubenum
        self.csa_cool = self.width * self.height * self.tubenum
        
        self.dia_hydr_ref = 4 * self.csa_ref / (2 * (self.width + self.height) * self.tubenum)
        self.dia_hydr_cool = 4 * self.csa_cool / (2 * (self.width + self.height) * self.tubenum)
        
        # Step function
        self.step_fcn = self._make_step_function()

    def go_step(self, x, u, p):
        # scaled_x = zero_one_scale(x, self.x_min, self.x_max)
        # scaled_u = zero_one_scale(u, self.u_min, self.u_max)
        # scaled_p = zero_one_scale(p, self.p_min, self.p_max)
        # scaled_up = torch.cat((scaled_u, scaled_p), dim=-1)
        # next_x = self.step_fcn(scaled_x, scaled_up)
        x_pred = self._ode_solver(x, u, p)

        return x_pred

    def get_observation(self, x):
        return x
        
    def do_reset(self):
        return self.x_ini, self.u_ini, self.y_ini, self.p_ini
    
    def sp_system_dynamics(self, x, u, p):
        pressure = x[0]
        h_ref_out = x[1].unsqueeze(-1)
        m_ref_in = u[0].unsqueeze(-1)
        m_ref_out = u[1].unsqueeze(-1)
        h_ref_in = u[2].unsqueeze(-1)
        m_cool = u[3].unsqueeze(-1)
        T_cool_in = u[4].unsqueeze(-1)
        z_tpsh = p[0].unsqueeze(-1)
        gamma = p[1].unsqueeze(-1)
        eps_tp = p[2].unsqueeze(-1)
        eps_sh = p[3].unsqueeze(-1)

        hf = self.Ref.liq_hsat(pressure)
        hg = self.Ref.vap_hsat(pressure)
        dhf_dp = self.Ref.liq_dhsatdp(pressure)
        dhg_dp = self.Ref.vap_dhsatdp(pressure)
        Df = self.Ref.liq_Dsat(pressure)
        Dg = self.Ref.vap_Dsat(pressure)
        dDf_dp = self.Ref.liq_dDsatdp(pressure)
        dDg_dp = self.Ref.vap_dDsatdp(pressure)

        h_ref_sh = (hg + h_ref_out) / 2.0
        D_ref_sh = self.Ref.vap_Dph(pressure, h_ref_sh)
        dDdp_sh = self.Ref.vap_dDdp(pressure, h_ref_sh)
        dDdh_sh = self.Ref.vap_dDdh(pressure, h_ref_sh)
        Cp_ref_sh = self.Ref.vap_Cph(pressure, h_ref_sh)

        Cp_cool = self.Cool.Cp(T_cool_in)

        Tsat = self.Ref.Tsat(pressure)
        T_ref_in_tp = Tsat
        T_ref_in_sh = Tsat

        mCp_ref_sh = m_ref_out * Cp_ref_sh
        mCp_cool = m_cool * Cp_cool
        Q_sh = eps_sh * mCp_ref_sh * (T_cool_in - T_ref_in_sh)
        T_cool_mid = T_cool_in - Q_sh / mCp_cool
        Q_tp = eps_tp * mCp_cool * (T_cool_mid - T_ref_in_tp)

        dzdt_mb_coeff = - ((Dg - D_ref_sh) + (Df - Dg) * (1 - gamma))
        dzdt_const = (m_ref_in - m_ref_out) / dzdt_mb_coeff / self.V_flow

        mass00 = z_tpsh * ((1-gamma)*((hf-hg)*dDf_dp + Df*dhf_dp) + (gamma*Dg*dhg_dp) - 1)
        mass01 = dzdt_const
        mass10 = (1-z_tpsh) * ((h_ref_out-hg)*(dDdp_sh + dDdh_sh*dhg_dp/2) + D_ref_sh*dhg_dp/2 - 1)
        mass11 = (1-z_tpsh) * ((h_ref_out-hg)*dDdh_sh + D_ref_sh) / 2
        rhs0 = m_ref_in * (h_ref_in - hg) + Q_tp
        rhs1 = m_ref_out * (hg - h_ref_out) + Q_sh    
        xdot0 = (mass11 * rhs0 - mass01 * rhs1) / (mass00 * mass11 - mass01 * mass10)
        xdot1 = (mass00 * rhs1 - mass10 * rhs0) / (mass00 * mass11 - mass01 * mass10)
        xdot = torch.stack([xdot0, xdot1], dim=-1) / self.V_flow
        xdot = xdot * self.scale_grad.to(xdot.device)

        return xdot
    
    def _inf_system_dynamics(self, x, u, p):
        pressure   = x[:, :1]    # (B,1)
        h_ref_out  = x[:, 1:2]   # (B,1)
        m_ref_in, m_ref_out, h_ref_in, m_cool, T_cool_in = [u[:, i:i+1] for i in range(5)]
        z_tpsh, gamma, eps_tp, eps_sh = [p[:, i:i+1] for i in range(4)]

        # Thermodynamic property calculations
        hf = self.Ref.liq_hsat(pressure)
        hg = self.Ref.vap_hsat(pressure)
        dhf_dp = self.Ref.liq_dhsatdp(pressure)
        dhg_dp = self.Ref.vap_dhsatdp(pressure)
        Df = self.Ref.liq_Dsat(pressure)
        Dg = self.Ref.vap_Dsat(pressure)
        dDf_dp = self.Ref.liq_dDsatdp(pressure)
        dDg_dp = self.Ref.vap_dDsatdp(pressure)

        h_ref_sh = (hg + h_ref_out) / 2.0
        D_ref_sh = self.Ref.vap_Dph(pressure, h_ref_sh)
        dDdp_sh = self.Ref.vap_dDdp(pressure, h_ref_sh)
        dDdh_sh = self.Ref.vap_dDdh(pressure, h_ref_sh)
        Cp_ref_sh = self.Ref.vap_Cph(pressure, h_ref_sh)
        Cp_cool = self.Cool.Cp(T_cool_in)
        Tsat = self.Ref.Tsat(pressure)

        # Heat exchange calculations
        T_ref_in_tp = Tsat
        T_ref_in_sh = Tsat
        mCp_ref_sh = m_ref_out * Cp_ref_sh
        mCp_cool = m_cool * Cp_cool
        Q_sh = eps_sh * mCp_ref_sh * (T_cool_in - T_ref_in_sh)
        T_cool_mid = T_cool_in - Q_sh / mCp_cool
        Q_tp = eps_tp * mCp_cool * (T_cool_mid - T_ref_in_tp)

        dzdt_mb_coeff = -((Dg - D_ref_sh) + (Df - Dg) * (1 - gamma))
        dzdt_const = (m_ref_in - m_ref_out) / dzdt_mb_coeff / self.V_flow

        mass00 = z_tpsh * ((1 - gamma) * ((hf - hg) * dDf_dp + Df * dhf_dp) + gamma * Dg * dhg_dp - 1)
        mass01 = dzdt_const
        mass10 = (1 - z_tpsh) * ((h_ref_out - hg) * (dDdp_sh + dDdh_sh * dhg_dp / 2) + D_ref_sh * dhg_dp / 2 - 1)
        mass11 = (1 - z_tpsh) * ((h_ref_out - hg) * dDdh_sh + D_ref_sh) / 2

        rhs0 = m_ref_in * (h_ref_in - hg) + Q_tp
        rhs1 = m_ref_out * (hg - h_ref_out) + Q_sh

        det = mass00 * mass11 - mass01 * mass10 + 1e-8  # avoid zero division
        xdot0 = (mass11 * rhs0 - mass01 * rhs1) / det
        xdot1 = (mass00 * rhs1 - mass10 * rhs0) / det

        xdot = torch.cat([xdot0, xdot1], dim=-1) / self.V_flow  # (B, 2)
        xdot = xdot * self.scale_grad.to(xdot.device)

        return xdot
    
    def _system_dynamics(self, x, u, p):
        pressure = x[:,0].unsqueeze(-1)
        h_ref_out = x[:,1].unsqueeze(-1)
        m_ref_in = u[:,0].unsqueeze(-1)
        m_ref_out = u[:,1].unsqueeze(-1)
        h_ref_in = u[:,2].unsqueeze(-1)
        m_cool = u[:,3].unsqueeze(-1)
        T_cool_in = u[:,4].unsqueeze(-1)
        z_tpsh = p[:,0].unsqueeze(-1)
        gamma = p[:,1].unsqueeze(-1)
        eps_tp = p[:,2].unsqueeze(-1)
        eps_sh = p[:,3].unsqueeze(-1)

        hf = self.Ref.liq_hsat(pressure)
        hg = self.Ref.vap_hsat(pressure)
        dhf_dp = self.Ref.liq_dhsatdp(pressure)
        dhg_dp = self.Ref.vap_dhsatdp(pressure)
        Df = self.Ref.liq_Dsat(pressure)
        Dg = self.Ref.vap_Dsat(pressure)
        dDf_dp = self.Ref.liq_dDsatdp(pressure)
        dDg_dp = self.Ref.vap_dDsatdp(pressure)

        h_ref_sh = (hg + h_ref_out) / 2.0
        D_ref_sh = self.Ref.vap_Dph(pressure, h_ref_sh)
        dDdp_sh = self.Ref.vap_dDdp(pressure, h_ref_sh)
        dDdh_sh = self.Ref.vap_dDdh(pressure, h_ref_sh)
        Cp_ref_sh = self.Ref.vap_Cph(pressure, h_ref_sh)

        Cp_cool = self.Cool.Cp(T_cool_in)

        Tsat = self.Ref.Tsat(pressure)
        T_ref_in_tp = Tsat
        T_ref_in_sh = Tsat

        mCp_ref_sh = m_ref_out * Cp_ref_sh
        mCp_cool = m_cool * Cp_cool
        Q_sh = eps_sh * mCp_ref_sh * (T_cool_in - T_ref_in_sh)
        T_cool_mid = T_cool_in - Q_sh / mCp_cool
        Q_tp = eps_tp * mCp_cool * (T_cool_mid - T_ref_in_tp)

        dzdt_mb_coeff = - ((Dg - D_ref_sh) + (Df - Dg) * (1 - gamma))
        dzdt_const = (m_ref_in - m_ref_out) / dzdt_mb_coeff / self.V_flow

        mass00 = z_tpsh * ((1-gamma)*((hf-hg)*dDf_dp + Df*dhf_dp) + (gamma*Dg*dhg_dp) - 1)
        mass01 = dzdt_const
        mass10 = (1-z_tpsh) * ((h_ref_out-hg)*(dDdp_sh + dDdh_sh*dhg_dp/2) + D_ref_sh*dhg_dp/2 - 1)
        mass11 = (1-z_tpsh) * ((h_ref_out-hg)*dDdh_sh + D_ref_sh) / 2
        rhs0 = m_ref_in * (h_ref_in - hg) + Q_tp
        rhs1 = m_ref_out * (hg - h_ref_out) + Q_sh    
        mass = torch.stack([torch.cat([mass00, mass01], dim=-1), torch.cat([mass10, mass11], dim=-1)], dim=1)
        rhs = torch.cat([rhs0, rhs1], dim=-1)  # shape: (batch, 2)
        # xdot0 = (mass11 * rhs0 - mass01 * rhs1) / (mass00 * mass11 - mass01 * mass10)
        # xdot1 = (mass00 * rhs1 - mass10 * rhs0) / (mass00 * mass11 - mass01 * mass10)
        # xdot = torch.stack([xdot0, xdot1], dim=-1) / self.V_flow
        mass = mass * self.V_flow.to(mass.device)

        return mass, rhs
    
    def _make_step_function(self):
        def step(x, up):
            xdot = self._system_dynamics(x, up[:self.u_dim], up[self.u_dim:])
            return x + xdot * self.time_interval
        
        return step

    def _ode_solver(self, x, u, p):
        if x.dim() == 1: x = x.unsqueeze(0)
        if u.dim() == 1: u = u.unsqueeze(0)
        if p.dim() == 1: p = p.unsqueeze(0)

        t = torch.tensor([0.0, 2.0], device=x.device, dtype=x.dtype)
        f = lambda t, y: self._inf_system_dynamics(y, u, p)
        traj = odeint(f, x, t, method='rk4')
        x_last = traj[-1].squeeze(0)
        return x_last
    
    def pCalculator(self, x_horizon, u_horizon):
        weight_path="src/weights/lstm_weights.npz"
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def lstm_forward(x_seq, W_ih, W_hh, b_ih, b_hh, h0=None, c0=None):
            T, input_size = x_seq.shape
            H = W_hh.shape[1]  # hidden size

            h = np.zeros(H) if h0 is None else h0
            c = np.zeros(H) if c0 is None else c0

            for t in range(T):
                x_t = x_seq[t]
                gates = (W_ih @ x_t + b_ih) + (W_hh @ h + b_hh)
                i, f, g, o = np.split(gates, 4)

                i = sigmoid(i)
                f = sigmoid(f)
                g = np.tanh(g)
                o = sigmoid(o)

                c = f * c + i * g
                h = o * np.tanh(c)

            return h  # final hidden state

        input_seq = np.concatenate([x_horizon, u_horizon], axis=1)  # shape: (30, 7)

        # --- Load weights ---
        weights = np.load(weight_path)
        W_ih = weights["lstm.weight_ih_l0"]
        W_hh = weights["lstm.weight_hh_l0"]
        b_ih = weights["lstm.bias_ih_l0"]
        b_hh = weights["lstm.bias_hh_l0"]
        fc1_W = weights["fc1.weight"]
        fc1_b = weights["fc1.bias"]
        fc2_W = weights["fc2.weight"]
        fc2_b = weights["fc2.bias"]

        # --- Forward pass ---
        h_final = lstm_forward(input_seq, W_ih, W_hh, b_ih, b_hh)
        h1 = sigmoid(fc1_W @ h_final + fc1_b)
        p = fc2_W @ h1 + fc2_b

        return p