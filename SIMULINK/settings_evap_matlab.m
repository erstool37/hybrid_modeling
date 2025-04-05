setting_evap = struct();

% Configurations
setting_evap.config.seed = 1999;
setting_evap.config.time_interval = 2;
setting_evap.config.terminal_time = 1000;

% Indices
setting_evap.x_idx.pressure = 1;
setting_evap.x_idx.h_ref_out = 2;

setting_evap.u_idx.m_ref_in = 1;
setting_evap.u_idx.m_ref_out = 2;
setting_evap.u_idx.h_ref_in = 3;
setting_evap.u_idx.m_cool = 4;
setting_evap.u_idx.T_cool_in = 5;

setting_evap.p_idx.z_tpsh = 1;
setting_evap.p_idx.gamma = 2;
setting_evap.p_idx.eps_tp = 3;
setting_evap.p_idx.eps_sh = 4;

% Dimensions
setting_evap.x_dim = 2;
setting_evap.u_dim = 5;
setting_evap.p_dim = 4;

% Label of variables
setting_evap.x_label = ["p [kPa]", "h_{ref,out} [kJ/kg]"];
setting_evap.u_label = ["m_{ref,in} [kg/s]", "m_{ref,out} [kg/s]", "h_{ref,in} [kJ/kg]", ...
                        "m_{cool} [kg/s]", "T_{cool,in} [{}^{o}C]"];
setting_evap.p_label = ["z_{tp-sh}", "\gamma", "\varepsilon_{tp}", "\varepsilon_{sh}"];

% Minimum & maximum values of variables
setting_evap.x_min = [100; 270];
setting_evap.x_max = [360; 380];

setting_evap.u_min = [0.01; 0.01; 250.; 0.4; -10.];
setting_evap.u_max = [0.03; 0.03; 280.; 0.8; +10.];

setting_evap.p_min = [0; 0; 0; 0];
setting_evap.p_max = [1; 1; 1; 1];

% Geometric parameters
setting_evap.geom.width = 0.1;
setting_evap.geom.height = 0.005;
setting_evap.geom.length = 0.5;
setting_evap.geom.tubenum = 10;
setting_evap.geom.V_flow = setting_evap.geom.width * setting_evap.geom.height * ...
                           setting_evap.geom.length * setting_evap.geom.tubenum;

setting_evap.geom.A_inner = 2 * (setting_evap.geom.width + setting_evap.geom.height) * setting_evap.geom.length * setting_evap.geom.tubenum;
setting_evap.geom.A_outer = setting_evap.geom.A_inner;

setting_evap.geom.csa_ref = setting_evap.geom.width * setting_evap.geom.height * setting_evap.geom.tubenum;
setting_evap.geom.csa_cool = setting_evap.geom.csa_ref;

setting_evap.geom.dia_hydr_ref = 4 * setting_evap.geom.csa_ref / (setting_evap.geom.A_inner / setting_evap.geom.length);
setting_evap.geom.dia_hydr_cool = 4 * setting_evap.geom.csa_cool / (setting_evap.geom.A_inner / setting_evap.geom.length);