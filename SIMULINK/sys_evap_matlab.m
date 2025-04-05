function xdot = sys_evap_matlab(x, u, p, Ref, Cool_evap, setting_evap)

% Variables
pressure = x(setting_evap.x_idx.pressure);
h_ref_out = x(setting_evap.x_idx.h_ref_out);

m_ref_in = u(setting_evap.u_idx.m_ref_in);
m_ref_out = u(setting_evap.u_idx.m_ref_out);
h_ref_in = u(setting_evap.u_idx.h_ref_in);
m_cool = u(setting_evap.u_idx.m_cool);
T_cool_in = u(setting_evap.u_idx.T_cool_in);

z_tpsh = p(setting_evap.p_idx.z_tpsh);
gamma = p(setting_evap.p_idx.gamma);
eps_tp = p(setting_evap.p_idx.eps_tp);
eps_sh = p(setting_evap.p_idx.eps_sh);

% Saturation properties
hf = Ref.liq_hsat(pressure);
hg = Ref.vap_hsat(pressure);
dhf_dp = Ref.liq_dhsatdp(pressure);
dhg_dp = Ref.vap_dhsatdp(pressure);

Df = Ref.liq_Dsat(pressure);
Dg = Ref.vap_Dsat(pressure);
dDf_dp = Ref.liq_dDsatdp(pressure);
dDg_dp = Ref.vap_dDsatdp(pressure);

% Average properties
h_ref_sh = (hg + h_ref_out) / 2;
D_ref_sh = Ref.vap_Dph(pressure, h_ref_sh);
dDdp_sh = Ref.vap_dDdp(pressure, h_ref_sh);
dDdh_sh = Ref.vap_dDdh(pressure, h_ref_sh);
Cp_ref_sh = Ref.vap_Cph(pressure, h_ref_sh);

% Terms regarding dz/dt
dzdt_mb_coeff = - ((Dg - D_ref_sh) + (Df - Dg) * (1 - gamma));
dzdt_const = (m_ref_in - m_ref_out) / dzdt_mb_coeff / setting_evap.geom.V_flow;
dzdt_dpdt_coeff = (z_tpsh * ((1-gamma) * dDf_dp + gamma * dDg_dp) + ...
                   (1-z_tpsh) * (dDdp_sh + dDdh_sh * dhg_dp / 2)) / dzdt_mb_coeff;
dzdt_dhdt_coeff = dDdh_sh * (1-z_tpsh) / 2 / dzdt_mb_coeff;

dzdt_tp_eb_coeff = Df * (hf - hg) * (1 - gamma);
dzdt_sh_eb_coeff = D_ref_sh * (hg - h_ref_sh);

%% Heat exchange rate calculation
% Values related to the coolant
Cp_cool = Cool_evap.Cp(T_cool_in);

% mCp values
mCp_ref_sh = m_ref_out * Cp_ref_sh;
mCp_cool = m_cool * Cp_cool;

% Saturation & inlet refrigerant temperature
Tsat = Ref.Tsat(pressure);
T_ref_in_tp = Tsat;
T_ref_in_sh = Tsat;cv

% Heat transfer rate for each control volume
Q_sh = eps_sh * mCp_ref_sh * (T_cool_in - T_ref_in_sh);
T_cool_mid = T_cool_in - Q_sh / mCp_cool;
Q_tp = eps_tp * mCp_cool * (T_cool_mid - T_ref_in_tp);
T_cool_out = T_cool_mid - Q_tp / mCp_cool;

%% Mass matrix
mass00 = z_tpsh * ((1-gamma)*((hf-hg)*dDf_dp + Df*dhf_dp) + (gamma*Dg*dhg_dp) - 1) + dzdt_dpdt_coeff * dzdt_tp_eb_coeff;
mass01 = dzdt_dhdt_coeff * dzdt_tp_eb_coeff;
mass10 = (1-z_tpsh) * ((h_ref_out-hg) * (dDdp_sh+dDdh_sh*dhg_dp/2) + (D_ref_sh*dhg_dp/2) - 1) + dzdt_dpdt_coeff * dzdt_sh_eb_coeff;
mass11 = (1-z_tpsh) * ((h_ref_out-hg)*dDdh_sh + D_ref_sh) / 2 + dzdt_dhdt_coeff * dzdt_sh_eb_coeff;

%% RHS equation
rhs0 = m_ref_in * (h_ref_in - hg) + Q_tp + dzdt_const * dzdt_tp_eb_coeff;
rhs1 = m_ref_out * (hg - h_ref_out) + Q_sh + dzdt_const * dzdt_sh_eb_coeff;

%% xdot calculation
xdot0 = (1 / (mass00*mass11 - mass01*mass10)) * (mass11*rhs0 - mass01*rhs1);
xdot1 = (1 / (mass00*mass11 - mass01*mass10)) * (mass00*rhs1 - mass10*rhs0);

xdot = [xdot0; xdot1] / setting_evap.geom.V_flow;

if ~isreal(xdot)
    disp("")
end

end