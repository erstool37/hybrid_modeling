load("coefficients_cool_evap.mat", "coefficients_cool_evap")
Cool_evap = struct();

%% Loaded coefficients
coeff_Cp = coefficients_cool_evap.Cp;
coeff_D = coefficients_cool_evap.D;
coeff_mu = coefficients_cool_evap.mu;
coeff_k = coefficients_cool_evap.k;

%% Arbitrary vectors for fitting
% Log and polynomial vectors
poly1 = @(x) [x, ones(length(x), 1)];
poly2 = @(x) [x.^2, x, ones(length(x), 1)];

% Normalization functions for fitting
norm_T = @(x) (x - 45) / 48.3477;

%% Properties
Cool_evap.Cp = @(T) poly1(T) * coeff_Cp';
Cool_evap.D = @(T) poly2(T) * coeff_D';
Cool_evap.mu = @(T) coeff_mu(1) .* exp(-coeff_mu(2) .* norm_T(T));
Cool_evap.k = @(T) poly2(norm_T(T)) * coeff_k';