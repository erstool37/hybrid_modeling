load("coefficients_ref.mat", "coeff_ref")
Ref = struct();

%% Loaded coefficients
coeff_Tsat = coeff_ref.Tsat;

coeff_vap_hsat = coeff_ref.vap_hsat;
coeff_vap_Dsat = coeff_ref.vap_Dsat;
coeff_vap_musat = coeff_ref.vap_musat;
coeff_vap_ksat = coeff_ref.vap_ksat;
coeff_vap_Cpsat = coeff_ref.vap_Cpsat;
coeff_vap_Prsat = coeff_ref.vap_Prsat;
coeff_vap_dhsatdp = coeff_ref.vap_dhsatdp;
coeff_vap_dDsatdp = coeff_ref.vap_dDsatdp;

coeff_liq_hsat = coeff_ref.liq_hsat;
coeff_liq_Dsat = coeff_ref.liq_Dsat;
coeff_liq_musat = coeff_ref.liq_musat;
coeff_liq_ksat = coeff_ref.liq_ksat;
coeff_liq_Cpsat = coeff_ref.liq_Cpsat;
coeff_liq_Prsat = coeff_ref.liq_Prsat;
coeff_liq_dhsatdp = coeff_ref.liq_dhsatdp;
coeff_liq_dDsatdp = coeff_ref.liq_dDsatdp;

coeff_vap_sph = coeff_ref.vap_sph;
coeff_vap_hps = coeff_ref.vap_hps;
coeff_vap_Dph = coeff_ref.vap_Dph;
coeff_vap_muph = coeff_ref.vap_muph;
coeff_vap_kph = coeff_ref.vap_kph;
coeff_vap_Cph = coeff_ref.vap_Cph;
coeff_vap_Tph = coeff_ref.vap_Tph;
coeff_vap_CpT = coeff_ref.vap_CpT;
coeff_vap_Prph = coeff_ref.vap_Prph;
coeff_vap_dsdp = coeff_ref.vap_dsdp;
coeff_vap_dsdh = coeff_ref.vap_dsdh;
coeff_vap_dhdp = coeff_ref.vap_dhdp;
coeff_vap_dhds = coeff_ref.vap_dhds;
coeff_vap_dDdp = coeff_ref.vap_dDdp;
coeff_vap_dDdh = coeff_ref.vap_dDdh;

coeff_liq_Dph = coeff_ref.liq_Dph;
coeff_liq_muph = coeff_ref.liq_muph;
coeff_liq_kph = coeff_ref.liq_kph;
coeff_liq_Cph = coeff_ref.liq_Cph;
coeff_liq_Tph = coeff_ref.liq_Tph;
coeff_liq_CpT = coeff_ref.liq_CpT;
coeff_liq_Prph = coeff_ref.liq_Prph;
coeff_liq_dDdp = coeff_ref.liq_dDdp;
coeff_liq_dDdh = coeff_ref.liq_dDdh;

%% Arbitrary vectors for fitting
% Log and polynomial vectors
log_vec = @(x) [log(x), ones(length(x), 1)];
        
poly7 = @(x) [x.^7, x.^6, x.^5, x.^4, x.^3, x.^2, x, ones(length(x), 1)];
poly8 = @(x) [x.^8, x.^7, x.^6, x.^5, x.^4, x.^3, x.^2, x, ones(length(x), 1)];
poly9 = @(x) [x.^9, x.^8, x.^7, x.^6, x.^5, x.^4, x.^3, x.^2, x, ones(length(x), 1)];

poly14 = @(x, y) [ones(length(x), 1), x, y, x.*y, y.^2, x.*(y.^2), y.^3, x.*(y.^3), y.^4];
poly15 = @(x, y) [ones(length(x), 1), x, y, x.*y, y.^2, x.*(y.^2), y.^3, x.*(y.^3), y.^4, x.*(y.^4), y.^5];
poly23 = @(x, y) [ones(length(x), 1), x, y, x.^2, x.*y, y.^2, (x.^2).*y, x.*(y.^2), y.^3];
poly24 = @(x, y) [ones(length(x), 1), x, y, x.^2, x.*y, y.^2, (x.^2).*y, x.*(y.^2), y.^3, (x.^2).*(y.^2), x.*(y.^3), y.^4];
poly25 = @(x, y) [ones(length(x), 1), x, y, x.^2, x.*y, y.^2, (x.^2).*y, x.*(y.^2), y.^3, (x.^2).*(y.^2), x.*(y.^3), y.^4, (x.^2).*(y.^3), x.*(y.^4), y.^5];
poly45 = @(x, y) [ones(length(x), 1), x, y, x.^2, x.*y, y.^2, x.^3, (x.^2).*y, x.*(y.^2), y.^3, x.^4, (x.^3).*y, (x.^2).*(y.^2), x.*(y.^3), y.^4, (x.^4).*y, (x.^3).*(y.^2), (x.^2).*(y.^3), x.*(y.^4), y.^5];
poly52 = @(x, y) [ones(length(x), 1), x, y, x.^2, x.*y, y.^2, x.^3, (x.^2).*y, x.*(y.^2), x.^4, (x.^3).*y, (x.^2).*(y.^2), x.^5, (x.^4).*y, (x.^3).*(y.^2)];
poly54 = @(x, y) [ones(length(x), 1), x, y, x.^2, x.*y, y.^2, x.^3, (x.^2).*y, x.*(y.^2), y.^3, x.^4, (x.^3).*y, (x.^2).*(y.^2), x.*(y.^3), y.^4, x.^5, (x.^4).*y, (x.^3).*(y.^2), (x.^2).*(y.^3), x.*(y.^4)];
poly55 = @(x, y) [ones(length(x), 1), x, y, x.^2, x.*y, y.^2, x.^3, (x.^2).*y, x.*(y.^2), y.^3, x.^4, (x.^3).*y, (x.^2).*(y.^2), x.*(y.^3), y.^4, x.^5, (x.^4).*y, (x.^3).*(y.^2), (x.^2).*(y.^3), x.*(y.^4), y.^5];

% Normalization functions for fitting
norm_psat = @(x) (x - 1670.0) / 955.3279;
norm_Tsat = @(x) (x - 52.5643) / 33.3173;

norm_vap_p = @(x) (x - 1300.9) / 939.1761;
norm_vap_h = @(x) (x - 432.5771) / 32.0814;
norm_vap_T = @(x) (x - 83.8534) / 34.6682;
norm_vap_s = @(x) (x - 1.7544) / 0.1025;

norm_liq_p = @(x) (x - 1984.0) / 880.9612;
norm_liq_h = @(x) (x - 212.5470) / 49.7295;
norm_liq_T = @(x) (x - 7.2744) / 36.7646;

%% Saturation temperature
Ref.Tsat = @(p) coeff_Tsat(2) ./ (coeff_Tsat(1) - log10(p)) - coeff_Tsat(3);

%% Vapor saturation properties
Ref.vap_hsat = @(p) poly8(norm_psat(p)) * coeff_vap_hsat';
Ref.vap_Dsat = @(p) poly9(norm_psat(p)) * coeff_vap_Dsat';
Ref.vap_musat = @(p) poly9(norm_psat(p)) * coeff_vap_musat';
Ref.vap_ksat = @(p) poly9(norm_psat(p)) * coeff_vap_ksat';
Ref.vap_Cpsat = @(p) poly9(norm_psat(p)) * coeff_vap_Cpsat';
Ref.vap_Prsat = @(p) poly9(norm_psat(p)) * coeff_vap_Prsat';

Ref.vap_dhsatdp = @(p) poly7(norm_psat(p)) * coeff_vap_dhsatdp';
Ref.vap_dDsatdp = @(p) poly8(norm_psat(p)) * coeff_vap_dDsatdp';

%% Liquid saturation properties
Ref.liq_hsat = @(p) poly9(norm_psat(p)) * coeff_liq_hsat';
Ref.liq_Dsat = @(p) poly9(norm_psat(p)) * coeff_liq_Dsat';
Ref.liq_musat = @(p) poly9(norm_psat(p)) * coeff_liq_musat';
Ref.liq_ksat = @(p) poly8(norm_psat(p)) * coeff_liq_ksat';
Ref.liq_Cpsat = @(p) poly9(norm_psat(p)) * coeff_liq_Cpsat';
Ref.liq_Prsat = @(p) poly8(norm_psat(p)) * coeff_liq_Prsat';

Ref.liq_dhsatdp = @(p) poly8(norm_psat(p)) * coeff_liq_dhsatdp';
Ref.liq_dDsatdp = @(p) poly8(norm_psat(p)) * coeff_liq_dDsatdp';

%% Vapor region properties
Ref.vap_sph = @(p, h) poly55(norm_vap_p(p), norm_vap_h(h)) * coeff_vap_sph';
Ref.vap_hps = @(p, s) poly55(norm_vap_p(p), norm_vap_s(s)) * coeff_vap_hps';
Ref.vap_Dph = @(p, h) poly24(norm_vap_p(p), norm_vap_h(h)) * coeff_vap_Dph';
Ref.vap_muph = @(p, h) poly52(norm_vap_p(p), norm_vap_h(h)) * coeff_vap_muph';
Ref.vap_kph = @(p, h) poly54(norm_vap_p(p), norm_vap_h(h)) * coeff_vap_kph';
Ref.vap_Cph = @(p, h) poly54(norm_vap_p(p), norm_vap_h(h)) * coeff_vap_Cph';
Ref.vap_Tph = @(p, h) poly23(norm_vap_p(p), norm_vap_h(h)) * coeff_vap_Tph';
Ref.vap_CpT = @(p, T) poly55(norm_vap_p(p), norm_vap_T(T)) * coeff_vap_CpT';
Ref.vap_Prph = @(p, h) poly55(norm_vap_p(p), norm_vap_h(h)) * coeff_vap_Prph';

Ref.vap_dsdp = @(p, h) poly45(norm_vap_p(p), norm_vap_h(h)) * coeff_vap_dsdp';
Ref.vap_dsdh = @(p, h) poly54(norm_vap_p(p), norm_vap_h(h)) * coeff_vap_dsdh';
Ref.vap_dhdp = @(p, s) poly45(norm_vap_p(p), norm_vap_s(s)) * coeff_vap_dhdp';
Ref.vap_dhds = @(p, s) poly54(norm_vap_p(p), norm_vap_s(s)) * coeff_vap_dhds';
Ref.vap_dDdp = @(p, h) poly14(norm_vap_p(p), norm_vap_h(h)) * coeff_vap_dDdp';
Ref.vap_dDdh = @(p, h) poly23(norm_vap_p(p), norm_vap_h(h)) * coeff_vap_dDdh';

%% Liquid region properties
Ref.liq_Dph = @(p, h) poly25(norm_liq_p(p), norm_liq_h(h)) * coeff_liq_Dph';
Ref.liq_muph = @(p, h) poly24(norm_liq_p(p), norm_liq_h(h)) * coeff_liq_muph';
Ref.liq_kph = @(p, h) poly24(norm_liq_p(p), norm_liq_h(h)) * coeff_liq_kph';
Ref.liq_Cph = @(p, h) poly24(norm_liq_p(p), norm_liq_h(h)) * coeff_liq_Cph';
Ref.liq_Tph = @(p, h) poly24(norm_liq_p(p), norm_liq_h(h)) * coeff_liq_Tph';
Ref.liq_CpT = @(p, T) poly25(norm_liq_p(p), norm_liq_T(T)) * coeff_liq_CpT';
Ref.liq_Prph = @(p, h) poly25(norm_liq_p(p), norm_liq_h(h)) * coeff_liq_Prph';

Ref.liq_dDdp = @(p, h) poly15(norm_liq_p(p), norm_liq_h(h)) * coeff_liq_dDdp';
Ref.liq_dDdh = @(p, h) poly24(norm_liq_p(p), norm_liq_h(h)) * coeff_liq_dDdh';