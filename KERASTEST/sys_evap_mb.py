"""
Created on Mon Mar 10 11:18:31 2025

@author: jisung
"""


import numpy as np
import casadi as ca
import utility as ut


class Evaporator_moving_boundary(object):
    
    def __init__(self, Ref, Cool):
        self.process_type = "continuous"
        
        self.seed = 1999
        self.hybrid = False
        self.np_data_type = np.float64
        self.time_interval = self.np_data_type(2)
        self.terminal_time = self.np_data_type(1000)
            
        self.Ref = Ref
        self.Cool = Cool
        
        # Dimension of variables
        self.x_dim = 3
        self.u_dim = 5
        self.y_dim = 2
        
        # Index of variables
        self.x_idx = {"pressure": 0, "h_ref_out": 1, "z_tpsh": 2}
        self.u_idx = {"m_ref_in": 0, "m_ref_out": 1, "h_ref_in": 2, "m_cool": 3, "T_cool_in": 4}
        self.y_idx = {"pressure": 0, "h_ref_out": 1}
        
        # Label of variables
        self.x_label = [r"$p$ [kPa]", r"$h_{ref,out}$ [kJ/kg]", r"$z_{tp-sh}$"]
        self.u_label = [r"$\dot{m}_{ref,in}$ [kg/s]", r"$\dot{m}_{ref,out}$ [kg/s]", 
                        r"$h_{ref,in}$ [kJ/kg]", r"$\dot{m}_{cool}$ [kg/s]", 
                        r"$T_{cool,in}$ [{}^{o}C]"]
        self.y_label = [r"$p$ [kPa]", r"$h_{ref,out}$ [kJ/kg]"]
         
        # Default values of parameters
        self.theta = np.array([0.2386, 0.0045], dtype=self.np_data_type)
        # self.theta = np.array([0.5386, 0.0245], dtype=self.np_data_type)
        # self.theta = np.array([0.1426, -0.4412], dtype=self.np_data_type)
        # self.theta = np.array([0.1166, 0.1924], dtype=self.np_data_type)
        
        # Initial value of variables
        self.x_ini = np.array([250, 360, 0.], dtype=self.np_data_type)
        self.u_ini = np.array([0.02, 0.02, 275, 0.6, -7], dtype=self.np_data_type)
        self.y_ini = np.array([250, 360], dtype=self.np_data_type)
        
        # Minimum / maximum values of variaibles
        self.x_min = np.array([100., 270., 0.], dtype=self.np_data_type)
        self.x_max = np.array([360., 380., 1.], dtype=self.np_data_type)
        
        self.u_min = np.array([0.01, 0.01, 163., 0.0001, -20.], dtype=self.np_data_type)
        self.u_max = np.array([0.05, 0.05, 365., 1.0000, +10.], dtype=self.np_data_type)
        
        self.y_min = np.array([100., 270.], dtype=self.np_data_type)
        self.y_max = np.array([360., 380.], dtype=self.np_data_type)
        
        self.scale_grad = 1. / (self.x_max - self.x_min)
        
        # Infinitesimal value
        self.small_h = 1e-6
        
        # Geometric parameters
        self.width = 0.1
        self.height = 0.005
        self.length = 0.5
        self.tubenum = 10
        self.V_flow = self.width * self.height * self.length * self.tubenum
        
        self.A_inner = 2 * (self.width + self.height) * self.length * self.tubenum
        self.A_outer = self.A_inner
        
        self.csa_ref = self.width * self.height * self.tubenum
        self.csa_cool = self.width * self.height * self.tubenum
        
        self.dia_hydr_ref = 4 * self.csa_ref / (2 * (self.width + self.height) * self.tubenum)
        self.dia_hydr_cool = 4 * self.csa_cool / (2 * (self.width + self.height) * self.tubenum)
        
        # Step function
        self.step_fcn_shout = self._make_step_function("SH_out")
        self.step_fcn_tpout = self._make_step_function("TP_out")
        
        
    def go_step(self, x, u):
        pressure, h_ref_out, z_tpsh = x
        hg = self.Ref.vap_hsat(pressure)
        
        if h_ref_out >= hg:
            result = self.step_fcn_shout(x0=x, p=u)    
            
        else:
            if z_tpsh < (1 - self.small_h):
                z_tpsh = 1 - self.small_h
            x[self.x_idx["z_tpsh"]] = z_tpsh
            result = self.step_fcn_tpout(x0=x, p=u)
            
        # scaled_next_x = np.squeeze(np.array(result['xf']))
        # next_x = ut.zero_one_descale(scaled_next_x, self.x_min, self.x_max)
        x_next = np.squeeze(np.array(result['xf']))
        
        z_tpsh_next = x_next[self.x_idx["z_tpsh"]]
        if z_tpsh_next > (1 - self.small_h):
            z_tpsh_next = 1 - self.small_h
            
        elif z_tpsh_next < self.small_h:
            z_tpsh_next = self.small_h
        
        x_next[self.x_idx["z_tpsh"]] = z_tpsh_next
            
        return x_next
    
    
    def get_observation(self, x):
        y = x[:-1]
        return y
        
        
    def do_reset(self):
        pressure, h_ref_out, _ = self.x_ini
        hg = self.Ref.vap_hsat(pressure)
        
        # if h_ref_out >= hg: 
        #     z_tpsh = (hg - h_ref_in) / (h_ref_out - h_ref_in)
        # else:
        #     z_tpsh = 1 - self.small_h
        # self.x_ini = np.array([pressure, h_ref_out, float(z_tpsh)], dtype=self.np_data_type)
        
        z_ca = ca.SX.sym('z', 1)
        x_ca = ca.vcat([self.x_ini[:-1], z_ca])
        
        if h_ref_out >= hg:
            sys = self._system_dynamics(x_ca, self.u_ini, "SH_out") * self.V_flow
        else:
            sys = self._system_dynamics(x_ca, self.u_ini, "TP_out") * self.V_flow
            
        nlp = {'x': z_ca, 'f': ca.sum1(sys**2)}
        S = ca.nlpsol('S', 'ipopt', nlp)
        r = S(x0=0.5, lbx=0, ubx=1)
        
        z_tpsh_stst = float(r['x'])
        self.x_ini = np.array([pressure, h_ref_out, z_tpsh_stst],
                              dtype=self.np_data_type)
        
        self.y_ini = self.get_observation(self.x_ini)
        
        return self.x_ini, self.u_ini, self.y_ini
    
    
    def _system_dynamics(self, x, u, mode):
    
        def _system_mass():
            if mode == "SH_out":
                mass00 = z_tpsh * (Df * (hg-hf) * dgamma_dp + gamma * Dg * dhg_dp + (1-gamma) * ((hf-hg)*dDf_dp + Df*dhf_dp) - 1)
                mass01 = 0
                mass02 = (1-gamma) * Df * (hf-hg)
                mass10 = (1-z_tpsh) * ((h_ref_sh-hg) * (dDdp_sh + dDdh_sh*dhg_dp/2) + (D_ref_sh*dhg_dp/2) - 1)
                mass11 = (1-z_tpsh) * ((h_ref_sh-hg) * dDdh_sh + D_ref_sh) / 2
                mass12 = D_ref_sh * (hg - h_ref_sh)
                mass20 = z_tpsh * ((Dg-Df)*dgamma_dp + gamma*dDg_dp + (1-gamma)*dDf_dp) + (1-z_tpsh) * (dDdp_sh + dDdh_sh*dhg_dp/2)
                mass21 = (1-z_tpsh) * dDdh_sh / 2
                mass22 = gamma*Dg + (1-gamma)*Df - D_ref_sh
                
            elif mode == "TP_out":
                mass00 = (Dg - Df) * dgamma_dp + gamma * dDg_dp + (1-gamma) * dDf_dp
                mass01 = (Dg - Df) * dgamma_dh
                mass02 = 0
                mass10 = (Dg*hg - Df*hf) * dgamma_dp + gamma * (hg*dDg_dp + Dg*dhg_dp) + (1-gamma) * (hf*dDf_dp + Df*dhf_dp) - 1
                mass11 = (Dg*hg - Df*hf) * dgamma_dh
                mass12 = 0
                mass20 = 0
                mass21 = 0
                mass22 = 1
                
            mass = ca.vcat((ca.hcat((mass00, mass01, mass02)),
                            ca.hcat((mass10, mass11, mass12)),
                            ca.hcat((mass20, mass21, mass22))))
            return mass
        
        def _system_rhs():
            if mode == "SH_out":
                rhs0 = m_ref_in * (h_ref_in - hg) + Q_tp
                rhs1 = m_ref_out * (hg - h_ref_out) + Q_sh
                rhs2 = m_ref_in - m_ref_out
                
            elif mode == "TP_out":
                rhs0 = m_ref_in - m_ref_out
                rhs1 = m_ref_in * h_ref_in - m_ref_out * h_ref_out + Q_tp
                rhs2 = 0
                
            rhs = ca.vcat((rhs0, rhs1, rhs2))
            return rhs
        
        def _system_gamma(pressure, *h_ref_out):
            # Saturation properties
            hf = self.Ref.liq_hsat(pressure)
            hg = self.Ref.vap_hsat(pressure)
            Df = self.Ref.liq_Dsat(pressure)
            Dg = self.Ref.vap_Dsat(pressure)
            
            if h_ref_out:
                h_ref_out = h_ref_out[0]
            
                gamma1 = Df * (h_ref_in - hf) + Dg * (hg - h_ref_in)
                gamma2 = Df * (h_ref_out - hf) + Dg * (hg - h_ref_out)
                
                gamma_term1 = Df / ((h_ref_in - h_ref_out) * (Df- Dg)**2)
                gamma_term2 = (h_ref_in - h_ref_out) * (Df - Dg)
                gamma_term3 = Dg * (hf - hg) * np.log(gamma1/gamma2)
                
                gamma = gamma_term1 * (gamma_term2 + gamma_term3)
                
            else:
                gamma1 = Df * (h_ref_in - hf) + Dg * (hg - h_ref_in)
                gamma2 = Df * (hg - hf)
                
                gamma_term1 = Df / ((h_ref_in - hg) * (Df- Dg)**2)
                gamma_term2 = (h_ref_in - hg) * (Df - Dg)
                gamma_term3 = Dg * (hf - hg) * np.log(gamma1/gamma2)
                
                gamma = gamma_term1 * (gamma_term2 + gamma_term3)
                
            return gamma
                    
        # Variables
        pressure, h_ref_out, z_tpsh = ca.vertsplit(x)
        m_ref_in, m_ref_out, h_ref_in, m_cool, T_cool_in = ca.vertsplit(u)
        eps_tp, eps_sh = self.theta
        
        # Saturation properties
        hf = self.Ref.liq_hsat(pressure)
        hg = self.Ref.vap_hsat(pressure)
        dhf_dp = self.Ref.liq_dhsatdp(pressure)
        dhg_dp = self.Ref.vap_dhsatdp(pressure)
        
        Df = self.Ref.liq_Dsat(pressure)
        Dg = self.Ref.vap_Dsat(pressure)
        dDf_dp = self.Ref.liq_dDsatdp(pressure)
        dDg_dp = self.Ref.vap_dDsatdp(pressure)
        
        # Average properties
        h_ref_sh = (hg + h_ref_out) / 2
        D_ref_sh = self.Ref.vap_Dph(pressure, h_ref_sh)
        dDdp_sh = self.Ref.vap_dDdp(pressure, h_ref_sh)
        dDdh_sh = self.Ref.vap_dDdh(pressure, h_ref_sh)
        Cp_ref_sh = self.Ref.vap_Cph(pressure, h_ref_sh)
        
        # Coolant properties
        Cp_cool = self.Cool.Cp(T_cool_in)
    
        # Saturation & inlet refrigerant temperature
        Tsat = self.Ref.Tsat(pressure)
        
        # Mean void fraction value
        if mode == "SH_out":
            T_ref_in_tp = Tsat
            T_ref_in_sh = Tsat
            
            # Heat transfer rate for each control volume
            mCp_ref_sh = m_ref_out * Cp_ref_sh
            mCp_cool = m_cool * Cp_cool
            Q_sh = eps_sh * mCp_ref_sh * (T_cool_in - T_ref_in_sh)
            T_cool_mid = T_cool_in - Q_sh / mCp_cool
            Q_tp = eps_tp * mCp_cool * (T_cool_mid - T_ref_in_tp)
            
            gamma = _system_gamma(pressure)
            
            gamma_plus = _system_gamma(pressure + self.small_h)
            gamma_minus = _system_gamma(pressure - self.small_h)
            dgamma_dp = (gamma_plus - gamma_minus) / (2 * self.small_h)
            
        elif mode == "TP_out":
            T_ref_in_tp = Tsat
            
            # Heat transfer rate for each control volume
            mCp_cool = m_cool * Cp_cool
            Q_tp = eps_tp * mCp_cool * (T_cool_in - T_ref_in_tp)
            
            gamma = _system_gamma(pressure, h_ref_out)
            
            gamma_p_plus = _system_gamma(pressure + self.small_h, h_ref_out)
            gamma_p_minus = _system_gamma(pressure - self.small_h, h_ref_out)
            dgamma_dp = (gamma_p_plus - gamma_p_minus) / (2 * self.small_h)
            
            gamma_h_plus = _system_gamma(pressure, h_ref_out + self.small_h)
            gamma_h_minus = _system_gamma(pressure, h_ref_out - self.small_h)
            dgamma_dh = (gamma_h_plus - gamma_h_minus) / (2 * self.small_h)
        
        # Calculate xdot
        mass = _system_mass()
        rhs = _system_rhs()
        
        xdot = (ca.inv(mass) @ rhs) / self.V_flow
        # xdot = ut.zero_one_scale(xdot, self.x_min, self.x_max)
        
        return xdot
    
    
    def _make_step_function(self, mode):
        x_ca = ca.SX.sym("x", self.x_dim)
        u_ca = ca.SX.sym("u", self.u_dim)
        
        xdot = self._system_dynamics(x_ca, u_ca, mode)
        xdot = np.multiply(xdot, self.scale_grad)
        
        ode = {"x": x_ca, "p": u_ca, "ode": xdot}
        
        options = {'t0': 0, 'tf': self.time_interval}
        ode_integrator = ca.integrator("Integrator", "cvodes", ode, options)
        return ode_integrator