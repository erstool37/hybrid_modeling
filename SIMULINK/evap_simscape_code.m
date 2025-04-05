clear; clc; close all;


%% Simulation initial conditions
initial = struct();

initial.pressure = 150;
initial.quality = 0.5;
initial.p_cool = 101.325;


%% Conditions for variables
% u_min = [0.01, 0.01, 250., 0.6, +30.];
% u_max = [0.05, 0.05, 300., 1.2, +40.];
% u_min = [0.01, 0.01, 300., 0.6, +30.];
% u_max = [0.05, 0.05, 320., 1.2, +40.];

%%% Training set generation u_min & u_max
% % u_min = [0.01, 0.01, 250., 0.4, -10.];
% % u_max = [0.03, 0.03, 280., 0.8, +10.];
% u_min = [0.01, 0.01, 210., 0.4, -10.];
% u_max = [0.05, 0.05, 290., 1.2, +10.];
%%%

%%% Extrapolation dataset generation u_min & u_max
% % u_min = [0.01, 0.01, 250., 0.8, -10.];
% % u_max = [0.03, 0.03, 280., 1.0, +4.];
% % u_min = [0.03, 0.03, 200., 0.8, -10.];
% % u_max = [0.05, 0.05, 230., 1.2, +10.];
u_min = [0.01, 0.01, 200., 0.2, -13.];
u_max = [0.03, 0.03, 220., 0.5, 0.];
%%%

% u_min = [0.01, 0.01, 250., 0.4, +30.];
% u_max = [0.05, 0.05, 300., 0.6, +40.];
du_max = [0.005, 0.005, 10., 0.1, 0.5];
u_dim = length(u_min);


%% Settings for simulation
warning("off", "all");

dt = 2;
% steplengths = [10, 20, 30, 50, 70];
steplengths = [10];

%%% Training set generation settings
changes_per_data = 100;
datanum = 100;
%%%

%%% Extrapolation dataset generation settings
% changes_per_data = 50;
% datanum = 1;
%%%

total_change_num = changes_per_data * datanum;

% Random input generation
rng(1999)
random_input_scaled = rand(total_change_num, u_dim) * 2 - 1;
random_input = repmat((u_min + u_max) / 2, total_change_num, 1);
for i = 2:total_change_num
    for j = 1:u_dim
        random_input(i, j) = random_input(i-1, j) + du_max(j) * random_input_scaled(i, j);
        if random_input(i, j) < u_min(j)
            random_input(i, j) = u_min(j);
        elseif random_input(i, j) > u_max(j)
            random_input(i, j) = u_max(j);
        end
    end
end

% Make input vector
empty_vec = zeros(2*total_change_num, 1);

m_ref_in_vec = empty_vec;
m_ref_in_vec(1:2:end) = random_input(:, 1);
m_ref_in_vec(2:2:end) = random_input(:, 1);

% m_ref_out_lb = 0.98 * random_input(:, 1);
% m_ref_out_ub = 1.02 * random_input(:, 1);
% m_ref_out_vec = empty_vec;
% m_ref_out_vec(1:2:end) = m_ref_out_lb + (m_ref_out_ub - m_ref_out_lb) .* (random_input_scaled(:, 2) + 1) / 2;
% m_ref_out_vec(2:2:end) = m_ref_out_lb + (m_ref_out_ub - m_ref_out_lb) .* (random_input_scaled(:, 2) + 1) / 2;
% m_ref_out_vec(1:2) = m_ref_in_vec(1:2);
m_ref_out_vec = m_ref_in_vec;

h_ref_in_vec = empty_vec;
h_ref_in_vec(1:2:end) = random_input(:, 3);
h_ref_in_vec(2:2:end) = random_input(:, 3);

m_cool_vec = empty_vec;
m_cool_vec(1:2:end) = random_input(:, 4);
m_cool_vec(2:2:end) = random_input(:, 4);

T_cool_in_vec = empty_vec;
T_cool_in_vec(1:2:end) = random_input(:, 5);
T_cool_in_vec(2:2:end) = random_input(:, 5);

initial.T_cool_in = T_cool_in_vec(1);


%% Simulation
for i = 1:length(steplengths)
    % Define step length and terminal time
    steplength = steplengths(i);
    timestep = sort([steplength*(0:total_change_num-1), steplength*(1:total_change_num)])';
    % timestep = [0; timestep(2:end)+100];
    timestep = [0; timestep(2:end)+10];
    eTime = timestep(end);
    
    % Make timeseries input data
    m_ref_in = timeseries(m_ref_in_vec, timestep);
    m_ref_out = timeseries(m_ref_out_vec, timestep);
    h_ref_in = timeseries(h_ref_in_vec, timestep);
    m_cool = timeseries(m_cool_vec, timestep);
    T_cool_in = timeseries(T_cool_in_vec, timestep);

    % Run simulink-simscape simulation
    out = sim("evap_simscape.slx", eTime);
    
    % Define resample time by dt value
    resample_time = 0:dt:eTime;

    % Output variables
    p_ref_in_sim = out.simout_Evap.ref_in.p;
    p_ref_out_sim = out.simout_Evap.ref_out.p;
    h_ref_out_sim = out.simout_Evap.ref_out.h;

    p_ref_in_resample = resample(p_ref_in_sim, resample_time);
    p_ref_out_resample = resample(p_ref_out_sim, resample_time);
    h_ref_out_resample = resample(h_ref_out_sim, resample_time);

    p_ref_in_data = p_ref_in_resample.Data;
    p_ref_out_data = p_ref_out_resample.Data;
    h_ref_out_data = h_ref_out_resample.Data;

    % Input variables
    m_ref_in_sim = out.simout_Evap.ref_in.m;
    m_ref_out_sim = out.simout_Evap.ref_out.m;
    h_ref_in_sim = out.simout_Evap.ref_in.h;
    m_cool_in_sim = out.simout_Evap.cool_in.m;
    m_cool_out_sim = out.simout_Evap.cool_out.m;
    T_cool_in_sim = out.simout_Evap.cool_in.T;

    m_ref_in_resample = resample(m_ref_in_sim, resample_time);
    m_ref_out_resample = resample(m_ref_out_sim, resample_time);
    h_ref_in_resample = resample(h_ref_in_sim, resample_time);
    m_cool_in_resample = resample(m_cool_in_sim, resample_time);
    m_cool_out_resample = resample(m_cool_out_sim, resample_time);
    T_cool_in_resample = resample(T_cool_in_sim, resample_time);

    m_ref_in_data = m_ref_in_resample.Data;
    m_ref_out_data = m_ref_out_resample.Data;
    h_ref_in_data = h_ref_in_resample.Data;
    m_cool_in_data = m_cool_in_resample.Data;
    m_cool_out_data = m_cool_out_resample.Data;
    T_cool_in_data = T_cool_in_resample.Data;

    % Other useful variables
    T_ref_in_sim = out.simout_Evap.ref_in.T;
    T_ref_out_sim = out.simout_Evap.ref_out.T;
    T_cool_out_sim = out.simout_Evap.cool_out.T;
    z_tpsh_sim = out.simout_Evap.phase.z_tpsh;
    
    T_ref_in_resample = resample(T_ref_in_sim, resample_time);
    T_ref_out_resample = resample(T_ref_out_sim, resample_time);
    T_cool_out_resample = resample(T_cool_out_sim, resample_time);
    z_tpsh_resample = resample(z_tpsh_sim, resample_time);

    T_ref_in_data = T_ref_in_resample.Data;
    T_ref_out_data = T_ref_out_resample.Data;
    T_cool_out_data = T_cool_out_resample.Data;
    z_tpsh_data = z_tpsh_resample.Data;

    % Make table to save result
    pressure_data = (p_ref_in_data + p_ref_out_data) / 2;
    m_cool_data = (m_cool_in_data + m_cool_out_data) / 2;
    result_array = [resample_time', pressure_data, h_ref_out_data, ...
                    m_ref_in_data, m_ref_out_data, h_ref_in_data, m_cool_data, T_cool_in_data, ...
                    T_ref_in_data, T_ref_out_data, T_cool_out_data, z_tpsh_data];
    result_table = array2table(result_array, "VariableNames", ...
                               ["Time", "pressure", "h_ref_out", ...
                                "m_ref_in", "m_ref_out", "h_ref_in", "m_cool", "T_cool_in", ...
                                "T_ref_in", "T_ref_out", "T_cool_out", "z_tpsh"]);

    % Split the table into smaller table and save the table
    filefolder = strcat("saved_data_evap/step_test_", num2str(steplength), "s/");
    filename = strcat(filefolder, "widerange_", num2str(steplength), "_total.csv");
    writetable(result_table, filename);
    % for j = 1:datanum
    %     smaller_tb = result_table(changes_per_data*steplength*(j-1)/dt+1:changes_per_data*steplength*j/dt, :);
    %     filename = strcat(filefolder, "widerange_", num2str(steplength), "-", num2str(j, "%02d"), ".csv");
    %     writetable(smaller_tb, filename)
    % end
end

%%
calc_param_total;