# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 16:07:27 2025

@author: jisung
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle as pkl
import datetime
import time

from scipy.io import loadmat
from itertools import compress

from sys_evap_mb import Evaporator_moving_boundary
from sys_evap_data import Evaporator_data
from prop_ref import Refrigerant
from prop_cool_evap import Coolant_evaporator
from config_evap import EvapConfig

from data_hybrid_lstm_rnn import DataLoading, DataTraining


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

# Config
EvapConfig = EvapConfig()

# Heat exchanger
Evap_data = Evaporator_data(Ref, Cool_evap, EvapConfig)
Evap_mb = Evaporator_moving_boundary(Ref, Cool_evap)


# %% Hybrid model default settings
TRAIN_TIMESTEP = 10
TEST_TIMESTEP = 10
TRAIN_DATA_NUM = 100
TEST_DATA_NUM = 1

setting = {"TRAIN_TIMESTEP": TRAIN_TIMESTEP,
           "TEST_TIMESTEP": TEST_TIMESTEP,
           "TRAIN_DATA_NUM": TRAIN_DATA_NUM,
           "TEST_DATA_NUM": TEST_DATA_NUM}

LOOKBACK = 30
LOOKFORWARD = 1

nnConfig_hyb_lstm = {"method": "LSTM",
                     
                     "input_nodes": 7,
                     "lstm_nodes": [128],
                     "rnn_nodes": [],
                     "dense_nodes": [32],
                     "output_nodes": 4,
                     "dense_activation": "sigmoid",
                     "layer_struct": ["LSTM", "Dense"],
                     
                     "lookback": LOOKBACK,
                     "lookforward": LOOKFORWARD,
                     "batch_size": 256,
                     "test_split": 0.2,
                     "valid_split": 0.2,
                     
                     "learning_rate": 0.013,
                     "dropout": 0.1,
                     "epoch": 1000,
                     "loss": "mse",
                     "metrics": ["mae"]}

nnConfig_hyb_rnn = {"method": "RNN",
                
                    "input_nodes": 7,
                    "lstm_nodes": [],
                    "rnn_nodes": [128],
                    "dense_nodes": [32],
                    "output_nodes": 4,
                    "dense_activation": "sigmoid",
                    "layer_struct": ["LSTM", "Dense"],
                    
                    "lookback": LOOKBACK,
                    "lookforward": LOOKFORWARD,
                    "batch_size": 256,
                    "test_split": 0.2,
                    "valid_split": 0.2,
                    
                    "learning_rate": 0.013,
                    "dropout": 0.1,
                    "epoch": 1000,
                    "loss": "mse",
                    "metrics": ["mae"]}

nnConfig_bb_lstm = {"method": "LSTM",
                
                    "input_nodes": 7,
                    "lstm_nodes": [128],
                    "rnn_nodes": [],
                    "dense_nodes": [32],
                    "output_nodes": 2,
                    "dense_activation": "sigmoid",
                    "layer_struct": ["LSTM", "Dense"],
                    
                    "lookback": LOOKBACK,
                    "lookforward": LOOKFORWARD,
                    "batch_size": 256,
                    "test_split": 0.2,
                    "valid_split": 0.2,
                    
                    "learning_rate": 0.013,
                    "dropout": 0.1,
                    "epoch": 1000,
                    "loss": "mse",
                    "metrics": ["mae"]}

nnConfig_bb_rnn = {"method": "RNN",
                
                   "input_nodes": 7,
                   "lstm_nodes": [],
                   "rnn_nodes": [128],
                   "dense_nodes": [32],
                   "output_nodes": 2,
                   "dense_activation": "sigmoid",
                   "layer_struct": ["LSTM", "Dense"],
                   
                   "lookback": LOOKBACK,
                   "lookforward": LOOKFORWARD,
                   "batch_size": 256,
                   "test_split": 0.2,
                   "valid_split": 0.2,
                   
                   "learning_rate": 0.013,
                   "dropout": 0.1,
                   "epoch": 1000,
                   "loss": "mse",
                   "metrics": ["mae"]}
    

# %% Define functions
def tester_mb(X_batch, Evap, X_min, X_max, y_min, y_max):
    df_length = X_batch.shape[0] - 1
    X_traj = X_min + (X_max - X_min) * X_batch[:, -1, :]
    x_traj = X_traj[:-1, :Evap.y_dim]
    u_traj = X_traj[:-1, Evap.y_dim:]
    y_true = X_traj[1:, :Evap.y_dim]
    y_calc = np.empty(shape=(df_length, Evap.y_dim))
    
    for j in range(df_length):
        x = x_traj[j, :]
        u = u_traj[j, :]
        
        z_ = (Ref.vap_hsat(x[0]).full().reshape(-1) - u[2]) / (x[1] - u[2])
        if z_ > 0.999999:
            z = 0.999999
        elif z_ < 0.000001:
            z = 0.000001
        else:
            z = z_
        x = np.hstack((x, z))
    
        x_next_calc = Evap.go_step(x, u)
        y_next_calc = Evap.get_observation(x_next_calc)
        
        y_calc[j, :] = y_next_calc
        
    MAE_mb = np.sum(np.abs(y_calc - y_true), axis=0) / (y_true.shape[0])
    
    return y_true, y_calc, u_traj, MAE_mb
    
    
def tester_hyb(X_batch, y_batch, Evap, lookback, model_hyb, X_min, X_max, y_min, y_max):
    df_length = X_batch.shape[0] - 1
    X_true = X_min + (X_max - X_min) * X_batch[1:, -1, :]
    y_true = X_true[:, :Evap.y_dim]
    u_traj = X_true[:, Evap.y_dim:]
    p_true = y_min + (y_max - y_min) * y_batch[1:, -1, :]
    y_calc = np.empty(shape=(df_length, Evap.x_dim))
    p_calc = np.empty(shape=(df_length, Evap.p_dim))
    
    X_data_last_scaled = X_batch[:, -1, :]
    X_data_last = X_min + (X_max - X_min) * X_data_last_scaled
        
    p_pred_hyb = model_hyb.predict(X_batch)    
    for j in range(df_length):
        p_pred_hyb_vec = p_pred_hyb[j, -1, :].reshape(1, -1)
        p_pred_hyb_descaled = y_min + (y_max - y_min) * p_pred_hyb_vec
        p_pred_hyb_descaled = p_pred_hyb_descaled.reshape(-1)
        
        x = X_data_last[j, :Evap.x_dim]
        u = X_data_last[j, Evap.x_dim:]
        x_next_calc_with_p_pred_hyb = Evap.go_step(x, u, p_pred_hyb_descaled)
        
        y_calc[j, :] = x_next_calc_with_p_pred_hyb
        p_calc[j, :] = p_pred_hyb_descaled
    
    MAE_hyb = np.sum(np.abs(y_calc - y_true), axis=0) / (y_true.shape[0])
    
    return y_true, y_calc, p_true, p_calc, u_traj, MAE_hyb


def tester_bb(X_batch, y_batch, Evap, lookback, model_bb, X_min, X_max, y_min, y_max):
    df_length = X_batch.shape[0] - 1
    y_true = y_min + (y_max - y_min) * y_batch[1:, -1, :]
    X_traj = X_min + (X_max - X_min) * X_batch[1:, -1, :]
    u_traj = X_traj[:, Evap.x_dim:]
    y_calc_bb = np.empty(shape=(df_length, Evap.y_dim))
    
    x_next_pred = model_bb.predict(X_batch[:-1, :, :])
    for j in range(df_length):
        x_next_pred_vec = x_next_pred[j, -1, :].reshape(1, -1)
        x_next_pred_vec_descaled = y_min + (y_max - y_min) * x_next_pred_vec
        
        y_calc_bb[j, :] = x_next_pred_vec_descaled
    
    MAE_bb = np.sum(np.abs(y_calc_bb - y_true), axis=0) / (y_true.shape[0])
    
    return y_true, y_calc_bb, u_traj, MAE_bb


def plot(time, y_true, y_calc_dict, p_calc_dict, u_traj, plot_range):
    # rcParams
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 11
    
    # Labels
    ylabels = Evap_data.y_label
    plabels = Evap_data.p_label
    ulabels = Evap_data.u_label[1: ]
    ulabels[0] = r"$\dot{m}_{ref} [kg/s]$"
    
    y_calc_label = list(y_calc_dict.keys())
    p_calc_label = list(compress(y_calc_label, ["hybrid" in s for s in y_calc_label]))
    
    # Alphabetical labels
    alphabet = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    
    # Figure 1. trajectories
    fig_traj = plt.figure(figsize=(7, 12))
    gs_traj = fig_traj.add_gridspec(4, 2)
    
    color_label = ["#ff4d4d", "#5461f0", "#60a860", "#b054ad"]
    
    for k in range(Evap_data.y_dim):
        ax = fig_traj.add_subplot(gs_traj[k, :])
        ax.plot(time, y_true[:, k], linestyle="-", linewidth=1, color="#555555")
        
        for kk in range(len(y_calc_label)):
            ax.plot(time, y_calc_dict[y_calc_label[kk]][:, k], linestyle="-", linewidth=1, color=color_label[kk], label=y_calc_label[kk])
        
        ax.set_xlim(plot_range)
        ax.set_xlabel("Time (sec)", fontsize=13)
        ax.set_ylabel(ylabels[k], fontsize=13)
        ax.set_xticks([kkk for kkk in range(plot_range[0], plot_range[1]+1, int((plot_range[1]-plot_range[0])/5))])
        ax.set_title(alphabet[k], fontsize=13)
        ax.grid(linewidth=0.5)
        
    ax.legend()
            
    for w in range(Evap_data.u_dim-1):
        ax = fig_traj.add_subplot(gs_traj[w//2+2, w%2])
        ax.plot(time, u_traj[:, w], linestyle="-", linewidth=1, color="#555555")
        ax.set_xlim(plot_range)
        ax.set_xlabel("Time (sec)", fontsize=13)
        ax.set_ylabel(ulabels[w], fontsize=13)
        ax.set_xticks([kkk for kkk in range(plot_range[0], plot_range[1]+1, int((plot_range[1]-plot_range[0])/5))])
        ax.set_title(alphabet[w+2], fontsize=13)
        ax.grid(linewidth=0.5)
    
    fig_traj.suptitle("System variables trajectory", fontsize=13)
    fig_traj.tight_layout()
    
    # Figure 2. parameters
    fig_param, axes_param = plt.subplots(2, 2, figsize=(7, 5))
    for m in range(Evap_data.p_dim):
        axes_param[m//2, m%2].plot(time-2, p_true[:, m], linestyle="-", linewidth=1, color="#555555")

        for mm in range(len(p_calc_label)):
            axes_param[m//2, m%2].plot(time-2, p_calc_dict[p_calc_label[mm]][:, m], linestyle="-", linewidth=1, color=color_label[mm], label=p_calc_label[mm])
        
        axes_param[m//2, m%2].set_xlim(plot_range)
        axes_param[m//2, m%2].set_xlabel("Time (sec)", fontsize=13)
        axes_param[m//2, m%2].set_ylabel(plabels[m], fontsize=13)
        axes_param[m//2, m%2].set_xticks([kkk for kkk in range(plot_range[0], plot_range[1]+1, int((plot_range[1]-plot_range[0])/5))])
        axes_param[m//2, m%2].set_title(alphabet[m], fontsize=13)
        axes_param[m//2, m%2].grid(linewidth=0.5)
        
    axes_param[m//2, m%2].legend()
    fig_param.suptitle("System parameters", fontsize=13)
    fig_param.tight_layout()
    
    return fig_traj, fig_param


def model_saver(fileprefix, X_min, X_max, y_min, y_max, model, nnConfig, setting):
    # Scalers
    Xy_minmax_name = f"{fileprefix}_Xy_minmax.pkl"
    Xy_minmax = {"X_min": X_min, "X_max": X_max, "y_min": y_min, "y_max": y_max}
    pkl.dump(Xy_minmax, open(f"my_model/{Xy_minmax_name}", "wb"))
    
    # Models
    model_name = f"{fileprefix}_model.keras"
    model.save(f"my_model/{model_name}")
    
    # Neural network configurations
    nnConfig_name = f"{fileprefix}_nnconfig.pkl"
    pkl.dump(nnConfig, open(f"my_model/{nnConfig_name}", "wb"))
    
    # Settings
    setting_name = f"{fileprefix}_setting.pkl"
    pkl.dump(setting, open(f"my_model/{setting_name}", "wb"))
    
    
def model_loader(fileprefix):
    # Scalers
    Xy_minmax_name = f"{fileprefix}_Xy_minmax.pkl"
    Xy_minmax = pkl.load(open(f"my_model/{Xy_minmax_name}", "rb"))
    
    X_min = Xy_minmax["X_min"]
    X_max = Xy_minmax["X_max"]
    y_min = Xy_minmax["y_min"]
    y_max = Xy_minmax["y_max"]
    
    # Models
    model_name = f"{fileprefix}_model.keras"
    model = tf.keras.models.load_model(f"my_model/{model_name}")
    
    # Neural network configurations
    nnConfig_name = f"{fileprefix}_nnconfig.pkl"
    nnConfig = pkl.load(open(f"my_model/{nnConfig_name}", "rb"))
    
    # Settings
    setting_name = f"{fileprefix}_setting.pkl"
    setting = pkl.load(open(f"my_model/{setting_name}", "rb"))
    
    return X_min, X_max, y_min, y_max, model, nnConfig, setting

    
def fig_saver(fileprefix, figs):
    if os.path.exists("saved_figures_evap/"+fileprefix):
        pass
    else:
        os.mkdir("saved_figures_evap/"+fileprefix)
    
    for key in figs.keys():
        filename = key + ".png"
        figs[key].savefig("saved_figures_evap/"+fileprefix+"/"+filename)


# %% Main
if __name__ == "__main__":
    # Data loader & trainer
    Loader = DataLoading(Evap_data)
    
    # Setting trainer
    Trainer_hyb_lstm = DataTraining(Evap_data, nnConfig_hyb_lstm)
    Trainer_hyb_rnn = DataTraining(Evap_data, nnConfig_hyb_rnn)
    Trainer_bb_lstm = DataTraining(Evap_data, nnConfig_bb_lstm)
    Trainer_bb_rnn = DataTraining(Evap_data, nnConfig_bb_rnn)
    
    # Load training set
    train_df = Loader.load_data("train", TRAIN_TIMESTEP, TRAIN_DATA_NUM)
    
    # Training black-box model
    train_data_dict_hyb_lstm, X_df_min_hyb_lstm, X_df_max_hyb_lstm, y_df_min_hyb_lstm, y_df_max_hyb_lstm = \
        Trainer_hyb_lstm.data_preprocessing(train_df, "hybrid")
    # train_data_dict_hyb_rnn, X_df_min_hyb_rnn, X_df_max_hyb_rnn, y_df_min_hyb_rnn, y_df_max_hyb_rnn = \
    #     Trainer_hyb_rnn.data_preprocessing(train_df, "hybrid")
    train_data_dict_bb_lstm, X_df_min_bb_lstm, X_df_max_bb_lstm, y_df_min_bb_lstm, y_df_max_bb_lstm = \
        Trainer_bb_lstm.data_preprocessing(train_df, "black-box")
    # train_data_dict_bb_rnn, X_df_min_bb_rnn, X_df_max_bb_rnn, y_df_min_bb_rnn, y_df_max_bb_rnn = \
    #     Trainer_bb_rnn.data_preprocessing(train_df, "black-box")

    # Train data
    tic_hyb_lstm = time.time()
    model_hyb_lstm, history_hyb_lstm = Trainer_hyb_lstm.train(train_data_dict_hyb_lstm)
    toc_hyb_lstm = time.time() - tic_hyb_lstm
    
    # tic_hyb_rnn = time.time()
    # model_hyb_rnn, history_hyb_rnn = Trainer_hyb_rnn.train(train_data_dict_hyb_rnn)
    # toc_hyb_rnn = time.time() - tic_hyb_rnn
    
    tic_bb_lstm = time.time()
    model_bb_lstm, history_bb_lstm = Trainer_bb_lstm.train(train_data_dict_bb_lstm)
    toc_bb_lstm = time.time() - tic_bb_lstm
    
    # tic_bb_rnn = time.time()
    # model_bb_rnn, history_bb_rnn = Trainer_bb_rnn.train(train_data_dict_bb_rnn)
    # toc_bb_rnn = time.time() - tic_bb_rnn
    
    model_saver("hybrid_LSTM_final", X_df_min_hyb_lstm, X_df_max_hyb_lstm, y_df_min_hyb_lstm, y_df_max_hyb_lstm,
                model_hyb_lstm, nnConfig_hyb_lstm, setting)
    # model_saver("hybrid_RNN_final", X_df_min_hyb_rnn, X_df_max_hyb_rnn, y_df_min_hyb_rnn, y_df_max_hyb_rnn,
    #             model_hyb_rnn, nnConfig_hyb_rnn, setting)
    model_saver("blackbox_LSTM_final", X_df_min_bb_lstm, X_df_max_bb_lstm, y_df_min_bb_lstm, y_df_max_bb_lstm,
                model_bb_lstm, nnConfig_bb_lstm, setting)
    # model_saver("blackbox_RNN_final", X_df_min_bb_rnn, X_df_max_bb_rnn, y_df_min_bb_rnn, y_df_max_bb_rnn,
    #             model_bb_rnn, nnConfig_bb_rnn, setting)

    t_test_hyb_lstm = train_data_dict_hyb_lstm["t_test"][1:, :]
    # t_test_hyb_rnn = train_data_dict_hyb_rnn["t_test"][1:, :]
    t_test_bb_lstm = train_data_dict_bb_lstm["t_test"][1:, :]
    # t_test_bb_rnn = train_data_dict_bb_rnn["t_test"][1:, :]

    X_scaled_hyb_lstm, y_scaled_hyb_lstm = train_data_dict_hyb_lstm["X_test"], train_data_dict_hyb_lstm["y_test"]
    # X_scaled_hyb_rnn, y_scaled_hyb_rnn = train_data_dict_hyb_rnn["X_test"], train_data_dict_hyb_rnn["y_test"]
    X_scaled_bb_lstm, y_scaled_bb_lstm = train_data_dict_bb_lstm["X_test"], train_data_dict_bb_lstm["y_test"]
    # X_scaled_bb_rnn, y_scaled_bb_rnn = train_data_dict_bb_rnn["X_test"], train_data_dict_bb_rnn["y_test"]

    y_true, y_calc_mb, u_traj, MAE_mb = \
        tester_mb(X_scaled_bb_lstm, Evap_mb, X_df_min_bb_lstm, X_df_max_bb_lstm, y_df_min_bb_lstm, y_df_max_bb_lstm)

    _, y_calc_hyb_lstm, p_true, p_calc_hyb_lstm, _, MAE_hyb_lstm = \
        tester_hyb(X_scaled_hyb_lstm, y_scaled_hyb_lstm, Evap_data, LOOKBACK, model_hyb_lstm, 
                   X_df_min_hyb_lstm, X_df_max_hyb_lstm, y_df_min_hyb_lstm, y_df_max_hyb_lstm)
        
    # _, y_calc_hyb_rnn, _, p_calc_hyb_rnn, _, MAE_hyb_rnn = \
    #     tester_hyb(X_scaled_hyb_rnn, y_scaled_hyb_rnn, Evap_data, LOOKBACK, model_hyb_rnn,
    #                X_df_min_hyb_rnn, X_df_max_hyb_rnn, y_df_min_hyb_rnn, y_df_max_hyb_rnn)
        
    _, y_calc_bb_lstm, _, MAE_bb_lstm = \
        tester_bb(X_scaled_bb_lstm, y_scaled_bb_lstm, Evap_data, LOOKBACK, model_bb_lstm,
                  X_df_min_bb_lstm, X_df_max_bb_lstm, y_df_min_bb_lstm, y_df_max_bb_lstm)
        
    # _, y_calc_bb_rnn, _, MAE_bb_rnn = \
    #     tester_bb(X_scaled_bb_rnn, y_scaled_bb_rnn, Evap_data, LOOKBACK, model_bb_rnn,
    #               X_df_min_bb_rnn, X_df_max_bb_rnn, y_df_min_bb_rnn, y_df_max_bb_rnn)

    # y_calc_dict = {"LSTM hybrid": y_calc_hyb_lstm,
    #                "RNN hybrid": y_calc_hyb_rnn,
    #                "LSTM black-box": y_calc_bb_lstm,
    #                "RNN black-box": y_calc_bb_rnn}
    
    # p_calc_dict = {"LSTM hybrid": p_calc_hyb_lstm,
    #                "RNN hybrid": p_calc_hyb_rnn}
    
    y_calc_dict = {"LSTM hybrid": y_calc_hyb_lstm,
                   "LSTM black-box": y_calc_bb_lstm,
                   "Moving boundary": y_calc_mb}
    
    p_calc_dict = {"LSTM hybrid": p_calc_hyb_lstm}
#%%
    plot_range = [90000, 91000]
    fig_traj, fig_param = plot(t_test_hyb_lstm, y_true, y_calc_dict, p_calc_dict, u_traj, plot_range)

    fileprefix = "final_figs"    
    figs = {"Trajectory": fig_traj,
            "Parameters": fig_param}
    
    fig_saver(fileprefix, figs)