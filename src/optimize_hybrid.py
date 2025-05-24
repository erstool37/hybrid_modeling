import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
import wandb
from tqdm import tqdm
import argparse
import yaml
import importlib
import torch.optim as optim
import time
import pandas as pd
import os.path as osp
from scipy.io import loadmat
from losses.calculator.sys_evap_1008ver import Evaporator
from losses.calculator.prop_ref import Refrigerant 
from losses.calculator.prop_cool_evap import Coolant_Evaporator
from losses.casadiCalculator.sys_evap_1008ver import Evaporator as ca_Evaporator
from losses.casadiCalculator.prop_ref import Refrigerant as ca_Refrigerant
from losses.casadiCalculator.prop_cool_evap import Coolant_Evaporator as ca_Coolant_Evaporator
from utils import setseed, Xdenormalizer, Udenormalizer, Odenormalizer, setpointCalculator, Pdenormalizer
from models.HybridLSTMModel import HybridLSTMModel
from datasets.Realtimeloader import Realtimeloader

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True)
args = parser.parse_args()

with open(args.config, "r") as file:
    config = yaml.safe_load(file)

NAME = config["name"]
PROJECT = config["project"]
SCALER = config["preprocess"]["scaler"]
DESCALER = config["preprocess"]["descaler"]
RANDOM_STATE = int(config["preprocess"]["random_state"])
OPTIMIZER = config["optimizer"]["optim_class"]
SCHEDULER = config["optimizer"]["scheduler_class"]
LR = float(config["optimizer"]["lr"])
CHECKPOINT = config["directories"]["checkpoint"]
SEED = config["optimize_settings"]["seed"]
NUM = config["optimize_settings"]["num"]
TEMP = float(config["optimizer"]["temperature"])
COST = config["cost"]
OPTIM = config["optimizer"]["optim_class"]
SCHEDULER_CLASS = config["optimizer"]["scheduler_class"]
MAX_ITER = int(config["optimizer"]["max_iter"])
STEP = int(config["optimizer"]["step"])
NOISE = int(config["optimize_settings"]["noise"])
TOL = float(config["optimizer"]["tolerance"])
DATA_ROOT = config["directories"]["data_root"]
HORIZON = 5

torch.use_deterministic_algorithms(True)
torch.backends.mkldnn.deterministic = True
torch.backends.mkldnn.benchmark = False
setseed(SEED)

# wandb.init(project=PROJECT, name=NAME, reinit=True, resume="never")

# Model setup
dataset = Realtimeloader(dir="../../MATLAB/SIMULINK/optimization", seq_len=30, scaler=SCALER)
data_iter = iter(dataset)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridLSTMModel(input_size=7, output_size=4, lookback=30).to(device)
model.load_state_dict(torch.load("src/weights/hybrid_keras_LSTM.pth"))
model.eval()

optim_class = getattr(optim, OPTIM)
scheduler_class = getattr(optim.lr_scheduler, SCHEDULER_CLASS)
cost_module = importlib.import_module(f"costs.{COST}")
cost_class = getattr(cost_module, COST)

#Evaporator setup
coeff_ref_data = loadmat("src/losses/calculator/coefficients_ref.mat")
coeff_ref_data = coeff_ref_data["coeff_ref"]
coeff_ref_data = {field: coeff_ref_data[field][0, 0] for field in coeff_ref_data.dtype.names}
Ref = Refrigerant(coeff_ref_data)
ca_Ref = ca_Refrigerant(coeff_ref_data)
        
coeff_cool_evap_data = loadmat("src/losses/calculator/coefficients_cool_evap.mat")
coeff_cool_evap_data = coeff_cool_evap_data["coefficients_cool_evap"]
coeff_cool_evap_data = {field: coeff_cool_evap_data[field][0, 0] for field in coeff_cool_evap_data.dtype.names}
Cool_evap = Coolant_Evaporator(coeff_cool_evap_data)
ca_Cool_evap = ca_Coolant_Evaporator(coeff_cool_evap_data)

Evap = Evaporator(Ref, Cool_evap)
ca_Evap = ca_Evaporator(ca_Ref, ca_Cool_evap)

# Simulation start
print("Simulation start")
set_temp = TEMP
for idx in tqdm(range(NUM)):
    start_time = time.time()
    next_data = next(data_iter).to(device)
    model_input = next_data[:,:,:7]
    others = next_data[:,:,-7:]

    # set point calculation
    x = model_input[:,-1,:2].squeeze(1).squeeze(0)
    u = model_input[:,-1,2:7].squeeze(1).squeeze(0)
    temp = torch.tensor(TEMP).unsqueeze(0).unsqueeze(0)
    x_sp, u_sp = setpointCalculator(ca_Evap.go_step, x, u, model, DESCALER, Cool_evap.Cp(temp), TEMP, tol=1e-4, method="SLSQP")
    x_sp = torch.tensor(x_sp, dtype=torch.float32, device=device)
    u_sp = torch.tensor(u_sp, dtype=torch.float32, device=device)

    # optimizer, sheduler setup
    u = model_input[:,-1,2:7].unsqueeze(1)
    u_horizon = u.expand(-1, 5, -1) # [B, 5, 5]
    u_optimize = torch.stack([u_horizon[:,:,0], u_horizon[:,:,2]]).unsqueeze(1) # take m_ref_in and h_ref_in for optimization
    u_optimize.requires_grad = True
    optimizer = torch.optim.LBFGS([u_optimize], lr=LR, max_iter=100)
    scheduler = scheduler_class(optimizer, T_max=MAX_ITER, eta_min=1e-9)

    nx, nu = 2, 5
    
    Q = torch.eye(nx,  device=device, dtype=torch.float32)
    P = torch.eye(nx,  device=device, dtype=torch.float32)
    R = torch.eye(nu,  device=device, dtype=torch.float32)

    cost = cost_class(Q, R, P).to(device)

    # Optimization loop
    for step in tqdm(range(NUM)):
        # stack horizon
        x_pred_lst = []
        p_pred_lst = []
        x = model_input[:,-1,:2].squeeze(1).squeeze(0)
        x = Xdenormalizer(x, DESCALER, "optim")

        u = u.squeeze(0).squeeze(0)
        u = Udenormalizer(u, DESCALER, "optim")

        input_seq = model_input
        for idx in range(HORIZON):
            u_step = u.squeeze(0).squeeze(0)              # → (5,)
            u_step = Udenormalizer(u_step, DESCALER, "optim")  # → (5,) real units

            p_pred = model(input_seq)                     # (1, L, ?)
            p_pred = p_pred[:, -1, :].squeeze(0)          # → (num_p,)
            p_pred = Pdenormalizer(p_pred, DESCALER, "optim")
            x = Evap.go_step(x, u_step, p_pred)           # → (2,) real units
            print(x)
            # record
            x_pred_lst.append(x)                          # list of (2,) tensors
            p_pred_lst.append(p_pred.unsqueeze(0))        # list of (1, num_p) tensors
            x_norm = Xdenormalizer(x, DESCALER, "optim")  # → (2,)
            if x_norm.dim() == 1:
                x_norm = x_norm.unsqueeze(0)               # → (1,2)

            new_frame = torch.zeros((1, 1, 7),
                                    device=input_seq.device,
                                    dtype=input_seq.dtype)
            new_frame[0, 0, :2] = x_norm                   # place normalized x

            # drop the oldest timestep, append the new one
            input_seq = torch.cat([input_seq[:, 1:, :], new_frame], dim=1)
        
        x_horizon = torch.stack(x_pred_lst, dim=0) # [B, 5, 2]
        p_horizon = torch.stack(p_pred_lst, dim=1) # [B, 5, 4]

        m_cool_in_fixed = u_horizon[:, 0, 3].item()
        T_cool_in_fixed = u_horizon[:, 0, 4].item()
        
        # loss calculation
        loss = cost(x_horizon, u_horizon, p_horizon, x_sp, u_sp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if u_optimize.grad.norm() < TOL:
            break
    end_time = time.time()

    m_ref_in = u_optimize[:,-1, 1].squeeze(0).squeeze(0)
    m_ref_out = m_ref_in
    h_ref_in = u_optimize[:,-1, 1].squeeze(0).squeeze(0)
    
    full_u = torch.cat([m_ref_in, m_ref_out, h_ref_in, m_cool_in_fixed, T_cool_in_fixed], dim=0, device=device)
    T_ref_in, T_ref_out, T_cool_out, z_tpsh = Odenormalizer(next_data[0, -1, 7:11], DESCALER, "optim")
    u_now_unnorm = Udenormalizer(full_u, DESCALER, "optim")

    # Noise in T_cool_in
    if idx % int(STEP) == 0:
        u_now_unnorm[4] = (TEMP+3) + np.random.uniform(-NOISE, NOISE)
    u_optim = u_now_unnorm.cpu().numpy().tolist()
    improvement = abs(T_cool_out-TEMP)
    T_cool_out_log = T_cool_out.item()
    
    # Convergence check
    if improvement > 1e-3:
        convergence += 1
    else :
        wandb.log({"convergence step": convergence, "error": improvement}, step=idx*2)
        convergence = 0
    
    save_path = osp.join(DATA_ROOT, f"u_optim/u_optim_{(idx+1):04d}.csv")
    pd.DataFrame([u_optim], columns=["m_ref_in", "m_ref_out", "h_ref_in", "m_cool", "T_cool"]).to_csv(save_path, index=False)

    wandb.log({
        "T_cool_out desired": set_temp, "T_cool_out": T_cool_out, "T_cool_in noise": u_optim[4],
        "steps": step, "time": end_time - start_time, "total_loss": loss.item(),
        "m_ref_in_optim": u_optim[0], "h_ref_in_optim": u_optim[2],
        "T_ref_in": T_ref_in.item(), "T_ref_out": T_ref_out.item(),
        "zeta_tpsh": z_tpsh.item(),
    }, step=idx*2)

wandb.finish()