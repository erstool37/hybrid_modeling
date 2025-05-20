# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 11:27:18 2025

@author: jisung
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


class DataLoading:
    def __init__(self, HeatEx):
        self.HeatEx = HeatEx
        
        if self.HeatEx.config.name == "Evaporator":
            self.data_folder = "saved_data_evap/"
        elif self.HeatEx.name == "Condenser":
            self.data_folder = "saved_data_cond/"
        else:
            raise ValueError("Heat exchanger should be 'Evaporator' or 'Condenser.'")
        
        
    def load_data(self, train_or_test, timestep, data_num):
        if train_or_test == "train":
            train_path = self.data_folder + "step_{}s/".format(timestep)
            
            # Load data for training
            df = pd.DataFrame()
            for num in range(data_num):
                train_filename = train_path + "widerange_{}-{:02}_param_calc2.csv".format(timestep, num+1)
                train_dat = pd.read_csv(train_filename)
                df = pd.concat((df, train_dat))
                
        elif train_or_test == "test":
            test_path = self.data_folder + "step_test_{}s/".format(timestep)
            
            # Load data for test (out of operating range of training data)
            df = pd.DataFrame()
            
            for num in range(data_num):
                test_filename = test_path + "widerange_{}-{:02}_param_calc2.csv".format(timestep, num+1)
                test_dat = pd.read_csv(test_filename)
                df = pd.concat((df, test_dat))
            
        # Clip useless data
        df = df.iloc[10:, :]        
        return df


class DataTraining:
    def __init__(self, HeatEx, *args):
        self.HeatEx = HeatEx
        
        if args:
            self.nnConfig = args[0]
            self.nn_method = self.nnConfig["method"]
            
            self.input_nodes = self.nnConfig["input_nodes"]
            self.lstm_nodes = self.nnConfig["lstm_nodes"]
            self.rnn_nodes = self.nnConfig["rnn_nodes"]
            self.dense_nodes = self.nnConfig["dense_nodes"]
            self.output_nodes = self.nnConfig["output_nodes"]
            self.dense_activation = self.nnConfig["dense_activation"]
            self.layer_struct = self.nnConfig["layer_struct"]
            
            self.lookback = self.nnConfig["lookback"]
            self.lookforward = self.nnConfig["lookforward"]
            self.batch_size = self.nnConfig["batch_size"]
            self.test_split = self.nnConfig["test_split"]
            self.valid_split = self.nnConfig["valid_split"]
            
            self.lr = self.nnConfig["learning_rate"]
            self.dropout = self.nnConfig["dropout"]
            self.epoch = self.nnConfig["epoch"]
            self.loss = self.nnConfig["loss"]
            self.metrics = self.nnConfig["metrics"]
        
        else:
            self.nn_method = "LSTM"
            
            self.input_nodes = 7
            self.lstm_nodes = [64]
            self.rnn_nodes = []
            self.dense_nodes = [32]
            self.output_nodes = 4
            self.dense_activation = "sigmoid"
            self.layer_struct = ["LSTM", "Dense"]
            
            self.lookback = 20
            self.lookforward = 1
            self.batch_size = 512
            self.test_split = 0.2
            self.valid_split = 0.2
            
            self.lr = 0.005
            self.dropout = 0.1
            self.epoch = 200
            self.loss = "mse"
            self.metrics = ["mae"]
    
    
    def data_preprocessing(self, train_df, train_mode):
        # Split data into ML model input-output
        X_df = train_df.iloc[40:-self.lookforward, 1:8]    # State & input of the system
        
        if train_mode == "hybrid":
            self.output_nodes = 4
            y_df = train_df.iloc[40+self.lookback-self.lookforward:-self.lookforward, 11:15]    # Parameter calculated from the data
        elif train_mode == "black-box":
            self.output_nodes = 2
            y_df = train_df.iloc[40+self.lookback:, 1:3]    # Next step state
        else:
            raise ValueError("Not a valid training mode.")
            
        t_all = train_df.iloc[40+self.lookback:, 0].to_numpy().reshape(-1, 1)
        
        # Preprocessing by making scaler
        X_df_min = np.min(X_df.to_numpy(), axis=0)
        X_df_max = np.max(X_df.to_numpy(), axis=0)
        X_df_scaled = (X_df.to_numpy() - X_df_min) / (X_df_max - X_df_min)
        
        y_df_min = np.min(y_df.to_numpy(), axis=0)
        y_df_max = np.max(y_df.to_numpy(), axis=0)
        y_df_scaled =(y_df.to_numpy() - y_df_min) / (y_df_max - y_df_min)
        
        # Define data length
        test_start_idx = int((X_df.shape[0]-self.lookback+1) * (1-self.test_split))
        valid_start_idx = int((X_df.shape[0]-self.lookback+1) * (1-self.test_split-self.valid_split))
        
        train_data_num = valid_start_idx
        valid_data_num = test_start_idx - valid_start_idx
        test_data_num = X_df.shape[0] - self.lookback + 1 - test_start_idx
        
        # Clip data into train / valid / test sets
        t_train = t_all[:valid_start_idx+self.lookforward, :]
        X_train = X_df_scaled[:valid_start_idx+self.lookback, :]
        y_train = y_df_scaled[:valid_start_idx+self.lookforward, :]
        
        t_valid = t_all[valid_start_idx:test_start_idx+self.lookforward, :]
        X_valid = X_df_scaled[valid_start_idx:test_start_idx+self.lookback, :]
        y_valid = y_df_scaled[valid_start_idx:test_start_idx+self.lookforward, :]
        
        t_test = t_all[test_start_idx:, :]
        X_test = X_df_scaled[test_start_idx:, :]
        y_test = y_df_scaled[test_start_idx:, :]
        
        # Make training set batch
        X_train_batch = np.empty(shape=(train_data_num, self.lookback, self.input_nodes))
        y_train_batch = np.empty(shape=(train_data_num, self.lookforward, self.output_nodes))
        
        for i in range(train_data_num):
            X_train_now = X_train[i:i+self.lookback, :]
            y_train_now = y_train[i:i+self.lookforward, :]
            
            X_train_now = np.expand_dims(X_train_now, axis=0)
            y_train_now = np.expand_dims(y_train_now, axis=0)
            
            X_train_batch[i, :, :] = X_train_now
            y_train_batch[i, :, :] = y_train_now
        
        # Make validation set batch
        X_valid_batch = np.empty(shape=(valid_data_num, self.lookback, self.input_nodes))
        y_valid_batch = np.empty(shape=(valid_data_num, self.lookforward, self.output_nodes))
        
        for j in range(valid_data_num):
            X_valid_now = X_valid[j:j+self.lookback, :]
            y_valid_now = y_valid[j:j+self.lookforward, :]
            
            X_valid_now = np.expand_dims(X_valid_now, axis=0)
            y_valid_now = np.expand_dims(y_valid_now, axis=0)
            
            X_valid_batch[j, :, :] = X_valid_now
            y_valid_batch[j, :, :] = y_valid_now
            
        # Make test set batch
        X_test_batch = np.empty(shape=(test_data_num, self.lookback, self.input_nodes))
        y_test_batch = np.empty(shape=(test_data_num, self.lookforward, self.output_nodes))
        
        for k in range(test_data_num):
            X_test_now = X_test[k:k+self.lookback, :]
            y_test_now = y_test[k:k+self.lookforward, :]
            
            X_test_now = np.expand_dims(X_test_now, axis=0)
            y_test_now = np.expand_dims(y_test_now, axis=0)
            
            X_test_batch[k, :, :] = X_test_now
            y_test_batch[k, :, :] = y_test_now
            
        train_data_dict = {"t_train": t_train,
                           "t_valid": t_valid,
                           "t_test": t_test,
                           "X_train": X_train_batch,
                           "X_valid": X_valid_batch,
                           "X_test": X_test_batch,
                           "y_train": y_train_batch,
                           "y_valid": y_valid_batch,
                           "y_test": y_test_batch}
        return train_data_dict, X_df_min, X_df_max, y_df_min, y_df_max
    
    
    def train(self, data):        
        if self.nn_method == "LSTM":
            input_shape = (self.lookback, self.input_nodes)
            output_shape = (self.lookforward, self.output_nodes)
            layer_struct = self.layer_struct
            lstm_nodes = self.lstm_nodes
            dense_nodes = self.dense_nodes
            dropout = self.dropout
            dense_activation = self.dense_activation
            
            lstm_settings = {"input_shape": input_shape,
                             "output_shape": output_shape,
                             "layer_struct": layer_struct,
                             "lstm_nodes": lstm_nodes,
                             "dense_nodes": dense_nodes,
                             "dropout": dropout,
                             "dense_activation": dense_activation}
            
            lstm = LSTM_model(lstm_settings)
            model = lstm.make_model()
            
        elif self.nn_method == "RNN":
            input_shape = (self.lookback, self.input_nodes)
            output_shape = (self.lookforward, self.output_nodes)
            layer_struct = self.layer_struct
            rnn_nodes = self.rnn_nodes
            dense_nodes = self.dense_nodes
            dropout = self.dropout
            dense_activation = self.dense_activation
            
            rnn_settings = {"input_shape": input_shape,
                            "output_shape": output_shape,
                            "layer_struct": layer_struct,
                            "rnn_nodes": rnn_nodes,
                            "dense_nodes": dense_nodes,
                            "dropout": dropout,
                            "dense_activation": dense_activation}
            
            rnn = RNN_model(rnn_settings)
            model = rnn.make_model()
        
        X_train = data["X_train"]
        y_train = data["y_train"]
        
        X_test = data["X_test"]
        y_test = data["y_test"]
        
        callback = EarlyStopping(monitor='loss', patience=20)
        model.compile(optimizer=Adam(learning_rate=self.lr),
                      loss=self.loss,
                      metrics=self.metrics)
        
        if "X_valid" in data.keys():
            X_valid = data["X_valid"]
            y_valid = data["y_valid"]
            history = model.fit(X_train, y_train,
                                validation_data=(X_valid, y_valid),
                                batch_size=self.batch_size,
                                epochs=self.epoch,
                                shuffle=True,
                                callbacks=[callback])
        else:
            history = model.fit(X_train, y_train,
                                validation_split=self.valid_split,
                                batch_size=self.batch_size,
                                epochs=self.epoch,
                                shuffle=True,
                                callbacks=[callback])
            
        test_loss, test_mae = model.evaluate(X_test, y_test)
        print("Test loss: {:3f} | Test MAE: {:3f}".format(test_loss, test_mae))
        return model, history
    
    
class LSTM_model:
    def __init__(self, lstm_settings):
        self.lstm_settings = lstm_settings
        
        self.input_shape = self.lstm_settings["input_shape"]
        self.output_shape = self.lstm_settings["output_shape"]
        self.layer_struct = self.lstm_settings["layer_struct"]
        self.lstm_nodes = self.lstm_settings["lstm_nodes"]
        self.dense_nodes = self.lstm_settings["dense_nodes"]
        self.dropout = self.lstm_settings["dropout"]
        self.dense_activation = self.lstm_settings["dense_activation"]
    
    
    def make_model(self):
        model = Sequential()
        lstm_layer_idx, dense_layer_idx = 0, 0
        
        for i in range(len(self.layer_struct)):
            layer_type = self.layer_struct[i]
            
            if layer_type == 'LSTM':
                if lstm_layer_idx == 0:
                    model.add(LSTM(self.lstm_nodes[lstm_layer_idx],
                                   return_sequences=True,
                                   input_shape=self.input_shape,
                                   dropout=self.dropout))
                else:
                    model.add(LSTM(self.lstm_nodes[lstm_layer_idx],
                                   return_sequences=True,
                                   dropout=self.dropout))   
                lstm_layer_idx += 1
                
            elif layer_type == 'Dense':
                model.add(Dense(self.dense_nodes[dense_layer_idx],
                                activation=self.dense_activation))
                dense_layer_idx += 1
        
        model.add(Dense(self.output_shape[-1],
                        activation=self.dense_activation))
        
        model.summary()
        return model
    
    
class RNN_model:
    def __init__(self, rnn_settings):
        self.rnn_settings = rnn_settings
        
        self.input_shape = self.rnn_settings["input_shape"]
        self.output_shape = self.rnn_settings["output_shape"]
        self.layer_struct = self.rnn_settings["layer_struct"]
        self.rnn_nodes = self.rnn_settings["rnn_nodes"]
        self.dense_nodes = self.rnn_settings["dense_nodes"]
        self.dropout = self.rnn_settings["dropout"]
        self.dense_activation = self.rnn_settings["dense_activation"]
        
        
    def make_model(self):
        model = Sequential()
        rnn_layer_idx, dense_layer_idx = 0, 0
        
        for i in range(len(self.layer_struct)):
            layer_type = self.layer_struct[i]
            
            if layer_type == "RNN":
                if rnn_layer_idx == 0:
                    model.add(SimpleRNN(self.rnn_nodes[rnn_layer_idx],
                                        return_sequences=True,
                                        input_shape=self.input_shape,
                                        dropout=self.dropout))
                else:
                    model.add(SimpleRNN(self.rnn_nodes[rnn_layer_idx],
                                        return_sequences=True,
                                        dropout=self.dropout))
                rnn_layer_idx += 1
                    
            elif layer_type == "Dense":
                model.add(Dense(self.dense_nodes[dense_layer_idx],
                                activation=self.dense_activation))
                dense_layer_idx += 1
                
        model.add(Dense(self.output_shape[-1],
                        activation=self.dense_activation))
        
        model.summary()
        return model