import datetime
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


class EvapConfig(object):
    
    def __init__(self):
        self.seed = 1999
        self.np_data_type = np.float64
        self.name = "Evaporator"
        
        self.plot_bool = True
        self.save_bool = True
        
        # System related
        self.time_interval = self.np_data_type(2)
        self.terminal_time = self.np_data_type(1000)
        
        # Hybrid or not
        self.hybrid = False
        self.hybrid_method = None
        
        # Black-box model related
        self.HYBRID_inputnodes = 7
        self.HYBRID_outputnodes = 2
        self.HYBRID_input_idx = {"pressure": 0, "h_ref_out": 1, "m_ref_in": 2,
                                 "m_ref_out": 3, "h_ref_in": 4, "m_cool": 5, "T_cool_in": 6}
        self.HYBRID_output_idx = {"z_tpsh": 0, "gamma": 1}
        self.HYBRID_signal_min = np.array([100., 270., 0.01, 0.01, 163., 0.0001, -20.], dtype=self.np_data_type)
        self.HYBRID_signal_max = np.array([360., 380., 0.05, 0.05, 365., 1.0000, +10.], dtype=self.np_data_type)
        self.HYBRID_signal_interval = 5
        self.HYBRID_ss_transform = True
        self.HYBRID_lookback = 50
        self.HYBRID_lookforward = 1
        
        self.HYBRID_learning_rate = 0.002
        self.HYBRID_hiddennodes = 16
        self.HYBRID_epoch = 50
        self.HYBRID_batchsize = 512
        self.HYBRID_validsplit = 0.2
        self.HYBRID_testsplit = 0.2
        self.HYBRID_activation = 'sigmoid'
        self.HYBRID_optimizer = 'adam'
        
    
    @staticmethod
    def plot_data(N, dt, x, u, y, p):
        time = dt * np.arange(N+1)
        labels_u = ["m_{ref} [kg/s]", "h_{ref,in} [kJ/kg]", "m_{cool} [kg/s]", "T_{cool,in} [C]"]
        labels_y = ["pressure [kPa]", "h_{ref,out} [kJ/kg]"]
        labels_p = ["mean void fraction"]
        
        fig_u = plt.figure(figsize=(10, 8))
        fig_u.suptitle("Evaporator Input Variables")
        for j in range(5):
            ax = fig_u.add_subplot(2, 3, j+1)
            ax.set_xlabel("time [sec]")
            ax.set_ylabel(labels_u[j])
            ax.plot(time, u[j, :], "-", color="#5461f0")
        plt.tight_layout()
            
        fig_y = plt.figure(figsize=(7, 4))
        fig_y.suptitle("Evaporator Output Variables")
        for k in range(2):
            ax = fig_y.add_subplot(1, 2, k+1)
            ax.set_xlabel("time [sec]")
            ax.set_ylabel(labels_y[k])
            ax.plot(time, y[k, :], "-", color="#60a860")
        plt.tight_layout()
            
        fig_p = plt.figure(figsize=(4, 4))
        fig_p.suptitle("Evaporator Parameters")
        for m in range(1):
            ax = fig_p.add_subplot(1, 1, m+1)
            ax.set_xlabel("time [sec]")
            ax.set_ylabel(labels_p[m])
            ax.plot(time, p[k, :], "k-")
        plt.tight_layout()
    
    
    @staticmethod
    def save_data(directory, filename, N, dt, x, u, y, p):
        data_to_save = dict()
        data_to_save["horizon"] = N
        data_to_save["interval"] = dt
        data_to_save["state_trajectory"] = x
        data_to_save["input_trajectory"] = u
        data_to_save["output_trajectory"] = y
        data_to_save["parameter_trajectory"] = p
        
        now = datetime.datetime.now()
        with open("saved_data/"+filename+"_evap_"+now.strftime("%y%m%d-%H%M%S")+".pickle", "wb") as f:
            pkl.dump(data_to_save, f)