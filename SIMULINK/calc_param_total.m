%% Load properties & settings
prop_ref;
prop_cool_evap;
settings_evap_matlab;

clearvars -except Ref Cool_evap setting_evap

%% Settings
steplength = 10;
changes_per_data = 100;
datanum = 100;
% datanum = 1;
dt = 2;
total_change_num = changes_per_data * datanum;

%% Load 1st data
data = readtable("saved_data_evap/step_test_10s/widerange_10_total.csv");

% data = table();
% for i = 1:datanum
%     data_loaded = readtable(strcat("saved_data_evap/step_", num2str(steplength), "s/widerange_", num2str(steplength), "-", num2str(i, "%02d"), ".csv"));
%     data = [data; data_loaded];
% end
% data = data(10:end, :);

%% Empty saving matrix settings
[total_steps, num_vars] = size(data);
p_calculated = zeros(total_steps, 4);
fval_calculated = zeros(total_steps, 1);
ef_calculated = zeros(total_steps, 1);

%% For each step, calculate parameters
for step = 1:total_steps-1
    % Current step state
    pressure = data.pressure(step);
    h_ref_out = data.h_ref_out(step);

    % Current step input
    m_ref_in = data.m_ref_in(step);
    m_ref_out = data.m_ref_out(step);
    h_ref_in = data.h_ref_in(step);
    m_cool = data.m_cool(step);
    T_cool_in = data.T_cool_in(step);

    % Next step state from data
    pressure_next = data.pressure(step+1);
    h_ref_out_next = data.h_ref_out(step+1);

    % Current step phase boundary
    z_tpsh_data = data.z_tpsh(step);

    % Vector form to get into ode
    x_now = [pressure; h_ref_out];
    u_now = [m_ref_in; m_ref_out; h_ref_in; m_cool; T_cool_in];
    x_next = [pressure_next; h_ref_out_next];

    % Bounds
    p_lb = [0.0000; 0.00; -5; -5];
    p_ub = [0.9999; 0.99; 5; 5];

    % Equality constraints
    Aeq = [1, 0, 0, 0];

    % Solver options
    if z_tpsh_data == 1
        p0 = [0.9999; 0.95; 0.3; -0.8];
        beq = 0.9999;    % To give boundary data
    else
        p0 = [z_tpsh_data; 0.95; 0.5; 0.5];
        beq = z_tpsh_data;    % To give boundary data
        % continue
    end
    opts = optimoptions("fmincon", "Algorithm", "interior-point", ...
                        "MaxIterations", 200, "MaxFunctionEvaluations", 200, ...
                        "FunctionTolerance", 1e-4, "StepTolerance", 1e-4, ...
                        "Display", "none");

    % Objective function
    % obj = @(p) x_next_calculation(x_now, u_now, p, Ref, Cool_evap, setting_evap) - x_next;
    obj = @(p) sum((x_next_calculation(x_now, u_now, p, Ref, Cool_evap, setting_evap) - x_next).^2);

    % Solve
    % [p_opt, fval, ~, ef, ~] = lsqnonlin(obj, p0, p_lb, p_ub, [], [], Aeq, beq, [], opts);
    [p_opt, fval, ef, ~] = fmincon(obj, p0, [], [], Aeq, beq, p_lb, p_ub, [], opts);

    % Save
    p_calculated(step, :) = p_opt';
    fval_calculated(step) = fval;
    ef_calculated(step) = ef;

    progress = step / (total_steps - 1) * 100;
    fprintf("\rCalculating step %d of %d — %.1f%% complete", step, total_steps - 1, progress);
end

fprintf("\nCalculation complete.\n");

%% Split data
data.gamma = p_calculated(:, setting_evap.p_idx.gamma);
data.eps_tp = p_calculated(:, setting_evap.p_idx.eps_tp);
data.eps_sh = p_calculated(:, setting_evap.p_idx.eps_sh);
data.fval = fval_calculated;
data.ef = ef_calculated;

data(end, :) = [];
writetable(data, "saved_data_evap/step_test_10s/widerange_10_total_param_calc2.csv")
% for j = 1:datanum
%     smaller_data = data(changes_per_data*steplength*(j-1)/dt+1:changes_per_data*steplength*j/dt, :);
%     if j == datanum
%         smaller_data = smaller_data(1:end-1, :);
%     end
%     filename = strcat("saved_data_evap/step_", num2str(steplength), "s/widerange_", num2str(steplength), "-", num2str(j, "%02d"), "_param_calc2.csv");
%     writetable(smaller_data, filename)
% end