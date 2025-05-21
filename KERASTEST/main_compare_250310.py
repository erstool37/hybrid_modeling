# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 23:39:13 2025

@author: jisung
"""

import warnings
from silence_tensorflow import silence_tensorflow
# warnings.filterwarnings(action='ignore')
# silence_tensorflow()

import time
import keras
import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.io import loadmat

from sys_evap_mb import Evaporator_moving_boundary
from sys_evap_data import Evaporator_data
from prop_ref import Refrigerant
from prop_cool_evap import Coolant_evaporator


# %%
# Refrigerant property functions
coeff_ref_data = loadmat("coefficients_ref.mat")
coeff_ref_data = coeff_ref_data["coeff_ref"]
coeff_ref_data = {field: coeff_ref_data[field][0, 0] for field in coeff_ref_data.dtype.names}
Ref = Refrigerant(coeff_ref_data)

# Evaporator coolant property functions
coeff_cool_evap_data = loadmat("coefficients_cool_evap.mat")
coeff_cool_evap_data = coeff_cool_evap_data["coefficients_cool_evap"]
coeff_cool_evap_data = {field: coeff_cool_evap_data[field][0, 0] for field in coeff_cool_evap_data.dtype.names}
Cool_evap = Coolant_evaporator(coeff_cool_evap_data)

# Heat exchanger
Evap_mb = Evaporator_moving_boundary(Ref, Cool_evap)
Evap_data = Evaporator_data(Ref, Cool_evap)


# %% Functions
def load_data(timestep, data_num):
    train_path = f"saved_data_evap/step_{timestep}s/"
    
    # Load data for training
    df = pd.DataFrame()
    # filename = train_path + f"widerange_{timestep}_total_param_calc2.csv"
    filename = train_path + f"test.csv"
    data = pd.read_csv(filename)
    df = pd.concat((df, data))
        
    # Clip useless data
    # df = df.iloc[401:, :]        
    df = df.iloc[40:, :]
    return df
    

def model_loader(fileprefix, model_type):
    if model_type == "hybrid":
        # Scalers
        X_scaler_name = fileprefix + "_X_scaler_hyb"
        y_scaler_name = fileprefix + "_y_scaler_hyb"
        X_scaler = pkl.load(open("my_model/"+X_scaler_name, "rb"))
        y_scaler = pkl.load(open("my_model/"+y_scaler_name, "rb"))
        
        # Models
        model_name = fileprefix + "_model_hyb.keras"
        model = tf.keras.models.load_model("my_model/"+model_name)
        
        # Neural network configurations
        nnConfig_name = fileprefix + "_nnconfig_hyb"
        nnConfig = pkl.load(open("my_model/"+nnConfig_name, "rb"))
    
    elif model_type == "black-box":
        # Scalers
        X_scaler_name = fileprefix + "_X_scaler_bb"
        y_scaler_name = fileprefix + "_y_scaler_bb"
        X_scaler = pkl.load(open("my_model/"+X_scaler_name, "rb"))
        y_scaler = pkl.load(open("my_model/"+y_scaler_name, "rb"))
        
        # Model
        model_name = fileprefix + "_model_bb.keras"
        model = tf.keras.models.load_model("my_model/"+model_name)
        
        # Neural network configurations
        nnConfig_name = fileprefix + "_nnconfig_bb"
        nnConfig = pkl.load(open("my_model/"+nnConfig_name, "rb"))
        
    # Settings
    setting_name = fileprefix + "_setting"
    setting = pkl.load(open("my_model/"+setting_name, "rb"))
    
    return X_scaler, y_scaler, model, nnConfig, setting


def make_data_batch(data_df, lookback, lookforward, model_type):
    data = data_df.to_numpy()
    X_df = data[:-lookforward, 1:8]
    
    if model_type == "hybrid":
        Y_df = data[lookback-lookforward:-lookforward, 11:15]
    elif model_type == "black-box":
        Y_df = data[lookback:, 1:3]
    else:
        raise ValueError("Not a valid training mode.")
        
    X_batch = np.empty(shape=(X_df.shape[0]-lookback, lookback, X_df.shape[1]))
    Y_batch = np.empty(shape=(X_df.shape[0]-lookback, lookforward, Y_df.shape[1]))
    
    for batch in range(X_df.shape[0]-lookback):
        X_batch[batch, :, :] = X_df[batch:batch+lookback, :]
        Y_batch[batch, :, :] = Y_df[batch:batch+lookforward, :]
        
    return X_batch, Y_batch


def lstm_fcn(X, hidden_dim, W, U, B, A1, b1, A2, b2):
    batch_size, lookback, _ = X.shape
    
    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    h_t = np.zeros((batch_size, hidden_dim))
    c_t = np.zeros((batch_size, hidden_dim))
    h_states = np.zeros((batch_size, lookback, hidden_dim))

    for t in range(lookback):
        X_t = X[:, t, :]
        
        gates = np.dot(X_t, W) + np.dot(h_t, U) + B
        i_t, f_t, c_tilda_t, o_t = np.split(gates, 4, axis=1)
        
        i_t = sigmoid(i_t)
        f_t = sigmoid(f_t)
        c_tilda_t = np.tanh(c_tilda_t)
        o_t = sigmoid(o_t)
        
        c_t = f_t * c_t + i_t * c_tilda_t
        h_t = o_t * np.tanh(c_t)
        
        h_states[:, t, :] = h_t
        
    X_out = sigmoid(np.dot(h_t, A1) + b1)
    y_out = sigmoid(np.dot(X_out, A2) + b2)
    
    return y_out


def plot_result(start, end, y_true, y_traj_mb, y_traj_bb, y_traj_hyb, p_true, p_traj_hyb, u_traj):
    time = np.arange(start, end+1) * 2
    
    # rcParams
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 11
    
    # Labels
    ylabels = Evap_data.y_label
    plabels = Evap_data.p_label
    ulabels = Evap_data.u_label[1: ]
    ulabels[0] = r"$\dot{m}_{ref} [kg/s]$"
    
    # Alphabetical labels
    alphabet = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    
    # Figure 1. trajectories
    fig_traj = plt.figure(figsize=(7, 12))
    gs_traj = fig_traj.add_gridspec(4, 2)
    
    for k in range(Evap_data.y_dim):
        ax = fig_traj.add_subplot(gs_traj[k, :])
        ax.plot(time, y_true[start:end+1, k], linestyle="-", linewidth=1, color="#555555")
        ax.plot(time, y_traj_mb[start:end+1, k], linestyle="-", linewidth=1, color="#60a860")
        ax.plot(time, y_traj_bb[start:end+1, k], linestyle="-", linewidth=1, color="#5461f0")
        ax.plot(time, y_traj_hyb[start:end+1, k], linestyle="-", linewidth=1, color="#ff4d4d")
        ax.set_xlim([start*2, end*2])
        ax.set_xlabel("Time (sec)", fontsize=13)
        ax.set_ylabel(ylabels[k], fontsize=13)
        ax.set_xticks([kkk for kkk in range(start*2, end*2+1, int((end-start)*2/5))])
        ax.set_title(alphabet[k], fontsize=13)
        ax.grid(linewidth=0.5)
            
    for w in range(Evap_data.u_dim-1):
        ax = fig_traj.add_subplot(gs_traj[w//2+2, w%2])
        ax.plot(time, u_traj[start:end+1, w], linestyle="-", linewidth=1, color="#555555")
        ax.set_xlim([start*2, end*2])
        ax.set_xlabel("Time (sec)", fontsize=13)
        ax.set_ylabel(ulabels[w], fontsize=13)
        ax.set_xticks([kkk for kkk in range(start*2, end*2+1, int((end-start)*2/5))])
        ax.set_title(alphabet[w+2], fontsize=13)
        ax.grid(linewidth=0.5)
    
    fig_traj.suptitle("System variables trajectory", fontsize=13)
    fig_traj.tight_layout()
    
    # Figure 2. parameters
    fig_param, axes_param = plt.subplots(2, 2, figsize=(7, 5))
    for m in range(Evap_data.p_dim):
        axes_param[m//2, m%2].plot(time-2, p_true[start-1:end, m], linestyle="-", linewidth=1, color="#555555")
        axes_param[m//2, m%2].plot(time-2, p_traj_hyb[start-1:end, m], linestyle="-", linewidth=1, color="#ff4d4d")
        axes_param[m//2, m%2].set_xlim([start*2, end*2])
        axes_param[m//2, m%2].set_xlabel("Time (sec)", fontsize=13)
        axes_param[m//2, m%2].set_ylabel(plabels[m], fontsize=13)
        axes_param[m//2, m%2].set_xticks([kkk for kkk in range(start*2, end*2+1, int((end-start)*2/5))])
        axes_param[m//2, m%2].set_title(alphabet[m], fontsize=13)
        axes_param[m//2, m%2].grid(linewidth=0.5)
        
    fig_param.suptitle("System parameters", fontsize=13)
    fig_param.tight_layout()

    def calculate_mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / (y_true))) * 100

    mape_pressure_hyb = calculate_mape(y_true[:, 0], y_traj_hyb[:-1, 0])
    mape_enthalpy_hyb = calculate_mape(y_true[:, 1], y_traj_hyb[:-1, 1])

    mape_pressure_mb = calculate_mape(y_true[:, 0], y_traj_mb[:-1, 0])
    mape_enthalpy_mb = calculate_mape(y_true[:, 1], y_traj_mb[:-1, 1])

    mape_pressure_bb = calculate_mape(y_true[:, 0], y_traj_bb[:-1, 0])
    mape_enthalpy_bb = calculate_mape(y_true[:, 1], y_traj_bb[:-1, 1])

    print(f"Hybrid MAPE - Pressure: {mape_pressure_hyb:.2f}%, Enthalpy: {mape_enthalpy_hyb:.2f}%")
    print(f"MB MAPE     - Pressure: {mape_pressure_mb:.2f}%, Enthalpy: {mape_enthalpy_mb:.2f}%")
    print(f"BB MAPE     - Pressure: {mape_pressure_bb:.2f}%, Enthalpy: {mape_enthalpy_bb:.2f}%")
    
    return fig_traj, fig_param


# %% Main
if __name__ == "__main__":
    # Load data
    data_df = load_data(10, 100)
    length = data_df.shape[0]
    
    X_hyb_min = np.min(data_df.to_numpy()[:-1, 1:8], axis=0)
    X_hyb_max = np.max(data_df.to_numpy()[:-1, 1:8], axis=0)
    y_hyb_min = np.min(data_df.to_numpy()[:-1, 11:15], axis=0)
    y_hyb_max = np.max(data_df.to_numpy()[:-1, 11:15], axis=0)
    
    X_bb_min = X_hyb_min.copy()
    X_bb_max = X_hyb_max.copy()
    y_bb_min = np.min(data_df.to_numpy()[1:, 1:3], axis=0)
    y_bb_max = np.max(data_df.to_numpy()[1:, 1:3], axis=0)
    
    X_scaler_hyb = lambda X: (X - X_hyb_min) / (X_hyb_max - X_hyb_min)
    y_scaler_hyb = lambda y: (y - y_hyb_min) / (y_hyb_max - y_hyb_min)
    X_descaler_hyb = lambda X: X_hyb_min + (X_hyb_max - X_hyb_min) * X
    y_descaler_hyb = lambda y: y_hyb_min + (y_hyb_max - y_hyb_min) * y
    
    X_scaler_bb = lambda X: (X - X_bb_min) / (X_bb_max - X_bb_min)
    y_scaler_bb = lambda y: (y - y_bb_min) / (y_bb_max - y_bb_min)
    X_descaler_bb = lambda X: X_bb_min + (X_bb_max - X_bb_min) * X
    y_descaler_bb = lambda y: y_bb_min + (y_bb_max - y_bb_min) * y
        
    # Load model
    filename_hyb = "hybrid_LSTM_final"
    filename_bb = "blackbox_LSTM_final"
    
    model_hyb = tf.keras.models.load_model(f"my_model/{filename_hyb}_model.keras")
    model_bb = tf.keras.models.load_model(f"my_model/{filename_bb}_model.keras")
    
    nnConfig_hyb = pkl.load(open(f"my_model/{filename_hyb}_nnconfig.pkl", "rb"))
    nnConfig_bb = pkl.load(open(f"my_model/{filename_bb}_nnconfig.pkl", "rb"))
    
    lookback_hyb = nnConfig_hyb["lookback"]
    lookback_bb = nnConfig_bb["lookback"]
    lookback = np.max((lookback_hyb, lookback_bb))
    
    lookforward_hyb = nnConfig_hyb["lookforward"]
    lookforward_bb = nnConfig_bb["lookforward"]
    lookforward = np.max((lookforward_hyb, lookforward_bb))
    
    hidden_hyb = nnConfig_hyb["lstm_nodes"][0]
    hidden_bb = nnConfig_bb["lstm_nodes"][0]
    
    W_hyb, U_hyb, B_hyb, A1_hyb, b1_hyb, A2_hyb, b2_hyb = model_hyb.get_weights()
    W_bb, U_bb, B_bb, A1_bb, b1_bb, A2_bb, b2_bb = model_bb.get_weights()
    
    # Set empty matrices
    y_traj_mb = np.empty(shape=(length-lookback, Evap_mb.y_dim))
    y_traj_bb = np.empty(shape=(length-lookback, Evap_data.y_dim))
    y_traj_hyb = np.empty(shape=(length-lookback, Evap_data.y_dim))
    p_traj_hyb = np.empty(shape=(length-lookback-1, Evap_data.p_dim))
    
    x_true_data = data_df.iloc[:, 1:3].to_numpy()
    x_true_mb = data_df.iloc[:, [1, 2, 11]].to_numpy()
    u_true = data_df.iloc[:, 3:8].to_numpy()
    y_true = data_df.iloc[:, 1:3].to_numpy()
    p_true = data_df.iloc[:, 11:15].to_numpy()
    
    Evap_data.x_ini = x_true_data[lookback, :]
    Evap_data.u_ini = u_true[lookback, :]
    Evap_data.y_ini = y_true[lookback, :]
    
    Evap_mb.x_ini = x_true_mb[lookback, :]
    Evap_mb.u_ini = u_true[lookback, :]
    Evap_mb.y_ini = y_true[lookback, :]
    
    y_traj_mb[0, :] = Evap_mb.y_ini
    y_traj_bb[0, :] = Evap_data.y_ini
    y_traj_hyb[0, :] = Evap_data.y_ini
    
    calc_time_mb = np.empty(shape=(length-lookback-lookforward, ))
    calc_time_bb = np.empty(shape=(length-lookback-lookforward, ))
    calc_time_hyb = np.empty(shape=(length-lookback-lookforward, ))
    
    # Calculate every step
    calc_steps = length - lookback - lookforward
    checkpoints = [int(calc_steps*(i+1)/20) for i in range(20)]
    percentages = [i for i in range(5, 101, 5)]
    
    for i in range(calc_steps):
        # Moving boundary
        x_mb = x_true_mb[i+lookback, :]
        u_mb = u_true[i+lookback, :]
        
        t_now_mb = time.time()
        # x_next_mb = Evap_mb.go_step(x_mb, u_mb)
        # y_next_mb = Evap_mb.get_observation(x_next_mb)
        calc_time_mb[i] = time.time() - t_now_mb
        
        # y_traj_mb[i+1, :] = y_next_mb
        
        # Black-box
        x_bb = x_true_data[i+(lookback-lookback_bb):i+lookback, :]
        u_bb = u_true[i+(lookback-lookback_bb):i+lookback, :]
          
        t_now_bb = time.time()
        # X_bb = np.expand_dims(X_scaler_bb.transform(np.hstack((x_bb, u_bb))), axis=0)
        # x_next_bb_scaled = model_bb.predict(X_bb, verbose=0)[:, -1, :].reshape(1, -1)
        # x_next_bb = y_scaler_bb.inverse_transform(x_next_bb_scaled).reshape(-1)
        # x_next_bb_calc_scaled = lstm_fcn(X_bb, hidden_bb, W_bb, U_bb, B_bb, A1_bb, b1_bb, A2_bb, b2_bb)
        # print(x_next_bb_scaled - x_next_bb_calc_scaled)
        # y_next_bb = Evap_data.get_observation(x_next_bb)
        X_bb = np.expand_dims(X_scaler_bb(np.hstack((x_bb, u_bb))), axis=0)
        x_next_bb_scaled = lstm_fcn(X_bb, hidden_bb, W_bb, U_bb, B_bb, A1_bb, b1_bb, A2_bb, b2_bb)
        x_next_bb = y_descaler_bb(x_next_bb_scaled)
        y_next_bb = Evap_data.get_observation(x_next_bb)
        calc_time_bb[i] = time.time() - t_now_bb
        
        y_traj_bb[i+1, :] = y_next_bb
        
        # Hybrid
        x_hyb = x_true_data[i+(lookback-lookback_hyb):i+lookback, :]
        u_hyb = u_true[i+(lookback-lookback_hyb):i+lookback, :]
        
        t_now_hyb = time.time()
        # X_hyb = np.expand_dims(X_scaler_hyb.transform(np.hstack((x_hyb, u_hyb))), axis=0)
        # theta_hyb_scaled = model_hyb.predict(X_hyb, verbose=0)[:, -1, :].reshape(1, -1)
        # theta_hyb = y_scaler_hyb.inverse_transform(theta_hyb_scaled).reshape(-1)
        # x_next_hyb = Evap_data.go_step(x_hyb[-1, :], u_hyb[-1, :], theta_hyb)
        # y_next_hyb = Evap_data.get_observation(x_next_hyb)
        X_hyb = np.expand_dims(X_scaler_hyb(np.hstack((x_hyb, u_hyb))), axis=0)
        theta_hyb_scaled = lstm_fcn(X_hyb, hidden_hyb, W_hyb, U_hyb, B_hyb, A1_hyb, b1_hyb, A2_hyb, b2_hyb)
        theta_hyb = y_descaler_hyb(theta_hyb_scaled).reshape(-1)
        
        x_next_hyb = Evap_data.go_step(x_hyb[-1, :], u_hyb[-1, :], theta_hyb)
        y_next_hyb = Evap_data.get_observation(x_next_hyb)
        calc_time_hyb[i] = time.time() - t_now_hyb
        
        y_traj_hyb[i+1, :] = y_next_hyb
        p_traj_hyb[i, :] = theta_hyb
        
        if i in checkpoints:
            percentage = percentages[checkpoints.index(i)]
            print(f"{percentage:02d}% Calculations are done...")
        
    u_true = u_true[lookback:-lookforward, :]
    y_true = y_true[lookback:-lookforward, :]    
    p_true = p_true[lookback:-lookforward, :]
    
    fig_traj, fig_param = \
        plot_result(5000, 5500, y_true, y_traj_mb, y_traj_bb, y_traj_hyb, p_true, p_traj_hyb, u_true)
    average_calc_time_mb = np.mean(calc_time_mb[calc_time_mb > 1e-6])
    average_calc_time_bb = np.mean(calc_time_bb[calc_time_bb > 1e-6])
    average_calc_time_hyb = np.mean(calc_time_hyb[calc_time_hyb > 1e-6])

    fig_traj.savefig("saved_figures_evap/Trajectory.png")
    fig_param.savefig("saved_figures_evap/Parameters.png")
    
    mape_mb = np.sum(np.abs(y_true - y_traj_mb[:-1, :]) / y_true, axis=0) * 100 / (y_true.shape[0])
    mape_hyb = np.sum(np.abs(y_true - y_traj_hyb[:-1, :]) / y_true, axis=0) * 100 / (y_true.shape[0])
    mape_bb = np.sum(np.abs(y_true - y_traj_bb[:-1, :]) / y_true, axis=0) * 100 / (y_true.shape[0])