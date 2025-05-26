import numpy as np
import casadi as ca
from .utility import zero_one_scale, zero_one_descale

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
            print("No user-defined configuration, using implemented setting")
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
        
        # Index of variables
        self.x_idx = {"pressure": 0, "h_ref_out": 1}
        self.u_idx = {"m_ref_in": 0, "m_ref_out": 1, "h_ref_in": 2, "m_cool": 3, "T_cool_in": 4}
        self.y_idx = {"pressure": 0, "h_ref_out": 1}
        self.p_idx = {"z_tpsh": 0, "gamma": 1, "eps_tp": 2, "eps_sh": 3}
        
        # Label of variables
        self.x_label = [r"$p [kPa]$", r"$h_{ref,out} [kJ/kg]$"]
        self.u_label = [r"$\dot{m}_{ref,in} [kg/s]$", r"$\dot{m}_{ref,out} [kg/s]$", 
                        r"$h_{ref,in} [kJ/kg]$", r"$\dot{m}_{cool} [kg/s]$", 
                        r"$T_{cool,in} [{}^{o}C]$"]
        self.y_label = [r"$p [kPa]$", r"$h_{ref,out} [kJ/kg]$"]
        self.p_label = [r"$z_{tp-sh}$", r"$\gamma$", r"$\varepsilon_{tp}$", r"$\varepsilon_{sh}$"]
        
        # Initial value of variables
        self.x_ini = np.array([250, 360], dtype=self.np_data_type)
        self.u_ini = np.array([0.02, 0.02, 275, 0.6, -7], dtype=self.np_data_type)
        self.y_ini = np.array([250, 360], dtype=self.np_data_type)
        self.p_ini = np.array([0.8, 0.9, 0.2, 0.3], dtype=self.np_data_type)
        
        # Minimum / maximum values of variaibles
        self.x_min = np.array([100., 270.], dtype=self.np_data_type)
        self.x_max = np.array([360., 380.], dtype=self.np_data_type)
        
        self.u_min = np.array([0.01, 0.01, 163., 0.0001, -20.], dtype=self.np_data_type)
        self.u_max = np.array([0.05, 0.05, 365., 1.0000, +10.], dtype=self.np_data_type)
        
        self.y_min = np.array([100., 270.], dtype=self.np_data_type)
        self.y_max = np.array([360., 380.], dtype=self.np_data_type)
        
        self.p_min = np.array([0., 0., 0., 0.], dtype=self.np_data_type)
        self.p_max = np.array([1., 1., 1., 1.], dtype=self.np_data_type)
        
        self.scale_grad = 1. / (self.x_max - self.x_min)
        
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
        self.step_fcn = self._make_step_function()
        
        
    def go_step(self, x, u, p):
        scaled_x = zero_one_scale(x, self.x_min, self.x_max)
        scaled_u = zero_one_scale(u, self.u_min, self.u_max)
        scaled_p = zero_one_scale(p, self.p_min, self.p_max)
        scaled_up = np.hstack((scaled_u, scaled_p))
        
        result = self.step_fcn(x0=scaled_x, p=scaled_up)     
        scaled_next_x = np.squeeze(np.array(result['xf']))
        next_x = zero_one_descale(scaled_next_x, self.x_min, self.x_max)
        return next_x
    
    def get_observation(self, x):
        y = x
        return y
        
    
    def do_reset(self):
        return self.x_ini, self.u_ini, self.y_ini, self.p_ini
    
    
    def _system_dynamics(self, x, u, p):
    
        def _system_mass():
            mass00 = z_tpsh * ((1-gamma)*((hf-hg)*dDf_dp + Df*dhf_dp) + (gamma*Dg*dhg_dp) - 1) + dzdt_dpdt_coeff * dzdt_tp_eb_coeff
            mass01 = dzdt_dhdt_coeff * dzdt_tp_eb_coeff
            mass10 = (1-z_tpsh) * ((h_ref_out-hg) * (dDdp_sh+dDdh_sh*dhg_dp/2) + (D_ref_sh*dhg_dp/2) - 1) + dzdt_dpdt_coeff * dzdt_sh_eb_coeff
            mass11 = (1-z_tpsh) * ((h_ref_out-hg)*dDdh_sh + D_ref_sh) / 2 + dzdt_dhdt_coeff * dzdt_sh_eb_coeff
            return mass00, mass01, mass10, mass11
        
        def _system_rhs():
            rhs0 = m_ref_in * (h_ref_in - hg) + Q_tp + dzdt_const * dzdt_tp_eb_coeff
            rhs1 = m_ref_out * (hg - h_ref_out) + Q_sh + dzdt_const * dzdt_sh_eb_coeff
            return rhs0, rhs1
            
        
        # Variables
        pressure, h_ref_out = ca.vertsplit(x)
        m_ref_in, m_ref_out, h_ref_in, m_cool, T_cool_in = ca.vertsplit(u)
        z_tpsh, gamma, eps_tp, eps_sh = ca.vertsplit(p)
        
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
        T_ref_in_tp = Tsat
        T_ref_in_sh = Tsat
        
        # Heat transfer rate for each control volume
        mCp_ref_sh = m_ref_out * Cp_ref_sh
        mCp_cool = m_cool * Cp_cool
        Q_sh = eps_sh * mCp_ref_sh * (T_cool_in - T_ref_in_sh)
        T_cool_mid = T_cool_in - Q_sh / mCp_cool
        Q_tp = eps_tp * mCp_cool * (T_cool_mid - T_ref_in_tp)
        T_cool_out = T_cool_mid - Q_tp / mCp_cool
        
        # Terms regarding dz/dt
        dzdt_mb_coeff = - ((Dg - D_ref_sh) + (Df - Dg) * (1 - gamma))
        dzdt_const = (m_ref_in - m_ref_out) / dzdt_mb_coeff / self.V_flow
        dzdt_dpdt_coeff = (z_tpsh * ((1-gamma) * dDf_dp + gamma * dDg_dp) + \
                           (1-z_tpsh) * (dDdp_sh + dDdh_sh * dhg_dp / 2)) / dzdt_mb_coeff
        dzdt_dhdt_coeff = dDdh_sh * (1-z_tpsh) / 2 / dzdt_mb_coeff
        
        dzdt_tp_eb_coeff = Df * (hf - hg) * (1 - gamma)
        dzdt_sh_eb_coeff = D_ref_sh * (hg - h_ref_sh)
        
        # Calculate xdot
        mass00, mass01, mass10, mass11 = _system_mass()
        rhs0, rhs1 = _system_rhs()
        
        xdot0 = (1 / (mass00*mass11 - mass01*mass10)) * (mass11*rhs0 - mass01*rhs1)
        xdot1 = (1 / (mass00*mass11 - mass01*mass10)) * (mass00*rhs1 - mass10*rhs0)
        xdot = ca.vcat([xdot0, xdot1]) / self.V_flow
        return xdot # [dP/dt, dH/dt]    
    
    def _make_step_function(self):
        x_ca = ca.SX.sym("x", self.x_dim)
        u_ca = ca.SX.sym("u", self.u_dim)
        p_ca = ca.SX.sym("p", self.p_dim)
        
        x_d = zero_one_descale(x_ca, self.x_min, self.x_max)
        u_d = zero_one_descale(u_ca, self.u_min, self.u_max)
        p_d = zero_one_descale(p_ca, self.p_min, self.p_max)
        
        up_ca = ca.vcat([u_ca, p_ca])
        
        xdot = self._system_dynamics(x_d, u_d, p_d) # next time step (dP/dt, dH/dt)
        xdot = np.multiply(xdot, self.scale_grad)
        
        ode = {"x": x_ca, "p": up_ca, "ode": xdot}
        
        options = {'t0': 0, 'tf': self.time_interval}
        ode_integrator = ca.integrator("Integrator", "cvodes", ode, options)
        return ode_integrator

    def pCalculator(self, x_horizon, u_horizon):
        weight_path="lstm_weights.npz"
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