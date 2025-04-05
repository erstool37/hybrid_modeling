data_dof2 = readtable("FluidProp.xlsx", "Sheet", "FluidProp");
data_liq = data_dof2(data_dof2.Phase=="Liquid", :);
data_vap = data_dof2(data_dof2.Phase=="Gas", :);
data_sat = readtable("FluidProp.xlsx", "Sheet", "FluidProp_sat");


%% Saturation properties
psat_data = data_sat.Pressure;

% Saturation temperature - Antoine equation
Tsat_data = data_sat.Temperature;

Tsat_ft = fittype("b/(a-log10(x))-c", "independent", "x", "dependent", "y");
opts = fitoptions("Method", "NonlinearLeastSquares");
opts.Display = "Off";
opts.StartPoint = [6, 1200, 300];

Tsat_cfit  = fit(psat_data, Tsat_data, Tsat_ft, opts);
Tsat_coeff = coeffvalues(Tsat_cfit);    % Antoine equation coefficients
coeff_ref.Tsat = Tsat_coeff;

% Saturation enthalpy
liq_hsat_data = data_sat.LiquidEnthalpy;
vap_hsat_data = data_sat.VaporEnthalpy;

liq_hsat_fit = fit(psat_data, liq_hsat_data, "poly9", "Normalize", "on");
vap_hsat_fit = fit(psat_data, vap_hsat_data, "poly8", "Normalize", "on");

liq_hsat_coeff = coeffvalues(liq_hsat_fit);
vap_hsat_coeff = coeffvalues(vap_hsat_fit);

coeff_ref.liq_hsat = liq_hsat_coeff;
coeff_ref.vap_hsat = vap_hsat_coeff;

liq_dhsatdp_data = differentiate(liq_hsat_fit, psat_data);
vap_dhsatdp_data = differentiate(vap_hsat_fit, psat_data);

liq_dhsatdp_fit = fit(psat_data, liq_dhsatdp_data, "poly8", "Normalize", "on");
vap_dhsatdp_fit = fit(psat_data, vap_dhsatdp_data, "poly7", "Normalize", "on");

liq_dhsatdp_coeff = coeffvalues(liq_dhsatdp_fit);
vap_dhsatdp_coeff = coeffvalues(vap_dhsatdp_fit);

coeff_ref.liq_dhsatdp = liq_dhsatdp_coeff;
coeff_ref.vap_dhsatdp = vap_dhsatdp_coeff;

% Saturation density
liq_Dsat_data = data_sat.LiquidDensity;
vap_Dsat_data = data_sat.VaporDensity;

liq_Dsat_fit = fit(psat_data, liq_Dsat_data, "poly9", "Normalize", "on");
vap_Dsat_fit = fit(psat_data, vap_Dsat_data, "poly9", "Normalize", "on");

liq_Dsat_coeff = coeffvalues(liq_Dsat_fit);
vap_Dsat_coeff = coeffvalues(vap_Dsat_fit);

coeff_ref.liq_Dsat = liq_Dsat_coeff;
coeff_ref.vap_Dsat = vap_Dsat_coeff;

liq_dDsatdp_data = differentiate(liq_Dsat_fit, psat_data);
vap_dDsatdp_data = differentiate(vap_Dsat_fit, psat_data);

liq_dDsatdp_fit = fit(psat_data, liq_dDsatdp_data, "poly8", "Normalize", "on");
vap_dDsatdp_fit = fit(psat_data, vap_dDsatdp_data, "poly8", "Normalize", "on");

liq_dDsatdp_coeff = coeffvalues(liq_dDsatdp_fit);
vap_dDsatdp_coeff = coeffvalues(vap_dDsatdp_fit);

coeff_ref.liq_dDsatdp = liq_dDsatdp_coeff;
coeff_ref.vap_dDsatdp = vap_dDsatdp_coeff;

% Saturation viscosity
liq_musat_data = data_sat.LiquidViscosity;
vap_musat_data = data_sat.VaporViscosity;

liq_musat_fit = fit(psat_data, liq_musat_data, "poly9", "Normalize", "on");
vap_musat_fit = fit(psat_data, vap_musat_data, "poly9", "Normalize", "on");

liq_musat_coeff = coeffvalues(liq_musat_fit);
vap_musat_coeff = coeffvalues(vap_musat_fit);

coeff_ref.liq_musat = liq_musat_coeff;
coeff_ref.vap_musat = vap_musat_coeff;

% Saturation thermal conductivity
liq_ksat_data = data_sat.LiquidThermCond_;
vap_ksat_data = data_sat.VaporThermCond_;

liq_ksat_fit = fit(psat_data, liq_ksat_data, "poly8", "Normalize", "on");
vap_ksat_fit = fit(psat_data, vap_ksat_data, "poly9", "Normalize", "on");

liq_ksat_coeff = coeffvalues(liq_ksat_fit);
vap_ksat_coeff = coeffvalues(vap_ksat_fit);

coeff_ref.liq_ksat = liq_ksat_coeff;
coeff_ref.vap_ksat = vap_ksat_coeff;

% Saturation Specific heat
liq_Cpsat_data = data_sat.LiquidCp;
vap_Cpsat_data = data_sat.VaporCp;

liq_Cpsat_fit = fit(psat_data, liq_Cpsat_data, "poly9", "Normalize", "on");
vap_Cpsat_fit = fit(psat_data, vap_Cpsat_data, "poly9", "Normalize", "on");

liq_Cpsat_coeff = coeffvalues(liq_Cpsat_fit);
vap_Cpsat_coeff = coeffvalues(vap_Cpsat_fit);

coeff_ref.liq_Cpsat = liq_Cpsat_coeff;
coeff_ref.vap_Cpsat = vap_Cpsat_coeff;

% Saturation Prandtl number
liq_Prsat_data = data_sat.LiquidPrandtl;
vap_Prsat_data = data_sat.VaporPrandtl;

liq_Prsat_fit = fit(psat_data, liq_Prsat_data, "poly8", "Normalize", "on");
vap_Prsat_fit = fit(psat_data, vap_Prsat_data, "poly9", "Normalize", "on");

liq_Prsat_coeff = coeffvalues(liq_Prsat_fit);
vap_Prsat_coeff = coeffvalues(vap_Prsat_fit);

coeff_ref.liq_Prsat = liq_Prsat_coeff;
coeff_ref.vap_Prsat = vap_Prsat_coeff;


%% Liquid properties
liq_p_data = data_liq.Pressure;
liq_h_data = data_liq.Enthalpy;

% Liquid density
liq_D_data = data_liq.Density;

liq_Dph_fit = fit([liq_p_data, liq_h_data], liq_D_data, "poly25", "Normalize", "on");
liq_Dph_coeff = coeffvalues(liq_Dph_fit);
coeff_ref.liq_Dph = liq_Dph_coeff;

[liq_dDdp_data, liq_dDdh_data] = differentiate(liq_Dph_fit, [liq_p_data, liq_h_data]);

liq_dDdp_fit = fit([liq_p_data, liq_h_data], liq_dDdp_data, "poly15", "Normalize", "on");
liq_dDdp_coeff = coeffvalues(liq_dDdp_fit);
coeff_ref.liq_dDdp = liq_dDdp_coeff;

liq_dDdh_fit = fit([liq_p_data, liq_h_data], liq_dDdh_data, "poly24", "Normalize", "on");
liq_dDdh_coeff = coeffvalues(liq_dDdh_fit);
coeff_ref.liq_dDdh = liq_dDdh_coeff;

% Liquid temperature
liq_T_data = data_liq.Temperature;

liq_Tph_fit = fit([liq_p_data, liq_h_data], liq_T_data, "poly24", "Normalize", "on");
liq_Tph_coeff = coeffvalues(liq_Tph_fit);
coeff_ref.liq_Tph = liq_Tph_coeff;

% Liquid viscosity
liq_mu_data = data_liq.Viscosity;

liq_muph_fit = fit([liq_p_data, liq_h_data], liq_mu_data, "poly24", "Normalize", "on");
liq_muph_coeff = coeffvalues(liq_muph_fit);
coeff_ref.liq_muph = liq_muph_coeff;

% Liquid thermal conductivity
liq_k_data = data_liq.ThermCond;

liq_kph_fit = fit([liq_p_data, liq_h_data], liq_k_data, "poly24", "Normalize", "on");
liq_kph_coeff = coeffvalues(liq_kph_fit);
coeff_ref.liq_kph = liq_kph_coeff;

% Liquid specific heat
liq_Cp_data = data_liq.Cp;

liq_Cph_fit = fit([liq_p_data, liq_h_data], liq_Cp_data, "poly24", "Normalize", "on");
liq_Cph_coeff = coeffvalues(liq_Cph_fit);
coeff_ref.liq_Cph = liq_Cph_coeff;

liq_CpT_fit = fit([liq_p_data, liq_T_data], liq_Cp_data, "poly25", "Normalize", "on");
liq_CpT_coeff = coeffvalues(liq_CpT_fit);
coeff_ref.liq_CpT = liq_CpT_coeff;

% Liquid Prandtl number
liq_Pr_data = data_liq.Prandtl;

liq_Prph_fit = fit([liq_p_data,  liq_h_data], liq_Pr_data, "poly25", "Normalize", "on");
liq_Prph_coeff = coeffvalues(liq_Prph_fit);
coeff_ref.liq_Prph = liq_Prph_coeff;


%% Vapor properties
vap_p_data = data_vap.Pressure;
vap_h_data = data_vap.Enthalpy;

% Vapor density
vap_D_data = data_vap.Density;

vap_Dph_fit = fit([vap_p_data, vap_h_data], vap_D_data, "poly24", "Normalize", "on");
vap_Dph_coeff = coeffvalues(vap_Dph_fit);
coeff_ref.vap_Dph = vap_Dph_coeff;

[vap_dDdp_data, vap_dDdh_data] = differentiate(vap_Dph_fit, [vap_p_data, vap_h_data]);

vap_dDdp_fit = fit([vap_p_data, vap_h_data], vap_dDdp_data, "poly14", "Normalize", "on");
vap_dDdp_coeff = coeffvalues(vap_dDdp_fit);
coeff_ref.vap_dDdp = vap_dDdp_coeff;

vap_dDdh_fit = fit([vap_p_data, vap_h_data], vap_dDdh_data, "poly23", "Normalize", "on");
vap_dDdh_coeff = coeffvalues(vap_dDdh_fit);
coeff_ref.vap_dDdh = vap_dDdh_coeff;

% Vapor temperature
vap_T_data = data_vap.Temperature;

vap_Tph_fit = fit([vap_p_data, vap_h_data], vap_T_data, "poly23", "Normalize", "on");
vap_Tph_coeff = coeffvalues(vap_Tph_fit);
coeff_ref.vap_Tph = vap_Tph_coeff;

% Vapor viscosity
vap_mu_data = data_vap.Viscosity;

vap_muph_fit = fit([vap_p_data, vap_h_data], vap_mu_data, "poly52", "Normalize", "on");
vap_muph_coeff = coeffvalues(vap_muph_fit);
coeff_ref.vap_muph = vap_muph_coeff;

% Vapor thermal conductivity
vap_k_data = data_vap.ThermCond;

vap_kph_fit = fit([vap_p_data, vap_h_data], vap_k_data, "poly54", "Normalize", "on");
vap_kph_coeff = coeffvalues(vap_kph_fit);
coeff_ref.vap_kph = vap_kph_coeff;

% Vapor specific heat
vap_Cp_data = data_vap.Cp;

vap_Cph_fit = fit([vap_p_data, vap_h_data], vap_Cp_data, "poly54", "Normalize", "on");
vap_Cph_coeff = coeffvalues(vap_Cph_fit);
coeff_ref.vap_Cph = vap_Cph_coeff;

vap_CpT_fit = fit([vap_p_data, vap_T_data], vap_Cp_data, "poly55", "Normalize", "on");
vap_CpT_coeff = coeffvalues(vap_CpT_fit);
coeff_ref.vap_CpT = vap_CpT_coeff;

% Vapor entropy
vap_s_data = data_vap.Entropy;

vap_sph_fit = fit([vap_p_data, vap_h_data], vap_s_data, "poly55", "Normalize", "on");
vap_sph_coeff = coeffvalues(vap_sph_fit);
coeff_ref.vap_sph = vap_sph_coeff;

vap_hps_fit = fit([vap_p_data, vap_s_data], vap_h_data, "poly55", "Normalize", "on");
vap_hps_coeff = coeffvalues(vap_hps_fit);
coeff_ref.vap_hps = vap_hps_coeff;

[vap_dsdp_data, vap_dsdh_data] = differentiate(vap_sph_fit, [vap_p_data, vap_h_data]);

vap_dsdp_fit = fit([vap_p_data, vap_h_data], vap_dsdp_data, "poly45", "Normalize", "on");
vap_dsdp_coeff = coeffvalues(vap_dsdp_fit);
coeff_ref.vap_dsdp = vap_dsdp_coeff;

vap_dsdh_fit = fit([vap_p_data, vap_h_data], vap_dsdh_data, "poly54", "Normalize", "on");
vap_dsdh_coeff = coeffvalues(vap_dsdh_fit);
coeff_ref.vap_dsdh = vap_dsdh_coeff;

[vap_dhdp_data, vap_dhds_data] = differentiate(vap_hps_fit, [vap_p_data, vap_s_data]);

vap_dhdp_fit = fit([vap_p_data, vap_s_data], vap_dhdp_data, "poly45", "Normalize", "on");
vap_dhdp_coeff = coeffvalues(vap_dhdp_fit);
coeff_ref.vap_dhdp = vap_dhdp_coeff;

vap_dhds_fit = fit([vap_p_data, vap_s_data], vap_dhds_data, "poly54", "Normalize", "on");
vap_dhds_coeff = coeffvalues(vap_dhds_fit);
coeff_ref.vap_dhds = vap_dhds_coeff;

% Vapor Prandtl number
vap_Pr_data = data_vap.Prandtl;

vap_Prph_fit = fit([vap_p_data, vap_h_data], vap_Pr_data, "poly55", "Normalize", "on");
vap_Prph_coeff = coeffvalues(vap_Prph_fit);
coeff_ref.vap_Prph = vap_Prph_coeff;


%% Saving coefficient data
save("coeff_ref.mat", "coeff_ref")