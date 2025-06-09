import torch
import numpy as np
import wandb
from tqdm import tqdm
import argparse
import yaml
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
from utils import setseed, Odenormalizer, setpointCalculator
from models.HybridLSTMModel import HybridLSTMModel
from datasets.Realtimeloader import Realtimeloader
from costs.Quad import mpc

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
LR = float(config["optimizer"]["lr"])
CHECKPOINT = config["directories"]["checkpoint"]
SEED = config["optimize_settings"]["seed"]
NUM = config["optimize_settings"]["num"]
TEMP = float(config["optimizer"]["temperature"])
STEP = int(config["optimizer"]["step"])
NOISE = int(config["optimize_settings"]["noise"])
TOL = float(config["optimizer"]["tolerance"])
DATA_ROOT = config["directories"]["data_root"]
HORIZON = 5

torch.use_deterministic_algorithms(True)
torch.backends.mkldnn.deterministic = True
torch.backends.mkldnn.benchmark = False
setseed(SEED)

wandb.init(project=PROJECT, name=NAME, reinit=True, resume="never")

# Model setup
dataset = Realtimeloader(dir="../../MATLAB/SIMULINK/optimization", seq_len=30, scaler=SCALER)
data_iter = iter(dataset)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridLSTMModel(input_size=7, output_size=4, lookback=30).to(device)
model.load_state_dict(torch.load("src/weights/hybrid_keras_LSTM.pth"))
model.eval()

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

# Optimization setup
u_min = np.array([0.005, 0.005, 250.0, 0.3, -20.0], dtype=np.float32)
u_max = np.array([0.05, 0.05, 300.0, 0.8, -10.0], dtype=np.float32)
x_min = np.array([100.0, 270.0], dtype=np.float32)
x_max = np.array([360.0, 380.0], dtype=np.float32)

Q = np.eye(2, dtype=np.float32)
R = np.eye(5, dtype=np.float32)
P = 30 * np.eye(2, dtype=np.float32)

# Simulation start
print("Simulation start")
T_cool_out_log = 0.0
convergence = 0
set_temp = TEMP
for idx in tqdm(range(NUM)):
    start_time = time.time()
    next_data = next(data_iter).to(device)
    model_input = next_data[:,:,:7]
    others = next_data[:,:,-7:]

    # set point calculation
    x = model_input[:,-1,:2].squeeze(1).squeeze(0)
    u = model_input[:,-1,2:7].squeeze(1).squeeze(0)
    others = others[:,-1,:].squeeze(1).squeeze(0)
    temp = torch.tensor(TEMP).unsqueeze(0).unsqueeze(0)
    x_sp, u_sp = setpointCalculator(ca_Evap._system_dynamics, ca_Evap.go_step, x, u, others, model, SCALER, DESCALER, Cool_evap.Cp(temp), TEMP, tol=1e-4)
    x_sp = torch.tensor(x_sp, dtype=torch.float32, device=device)
    u_sp = torch.tensor(u_sp, dtype=torch.float32, device=device)

    # optimzation
    u0_opt, x_pred = mpc(model_input, x_sp, u_sp, Evap.pCalculator, Evap.go_step, P, Q, R, SCALER, DESCALER, u_min, u_max, x_min, x_max)

    if idx % int(STEP) == 0:
        noise = (2*torch.rand(1, device=device) - 1) * NOISE
        u0_opt[4] = (TEMP + 3) + noise

    u_optim = u0_opt.cpu().numpy().tolist()
    save_path = osp.join(DATA_ROOT, f"u_optim/u_optim_{idx+1:04d}.csv")
    pd.DataFrame([u_optim], columns=["m_ref_in", "m_ref_out", "h_ref_in", "m_cool", "T_cool"]).to_csv(save_path, index=False)

    T_ref_in, T_ref_out, T_cool_out, z_tpsh = Odenormalizer(next_data[0, -1, 7:11], DESCALER, "optim")
    improvement = abs(T_cool_out - T_cool_out_log)
    T_cool_out_log = T_cool_out.item()

    if improvement > 1e-3:
        convergence += 1
    else:
        wandb.log({"convergence step": convergence, "error": improvement},
                  step=idx*2)
        convergence = 0

    # wandb logging
    wandb.log({
        "T_cool_out desired": set_temp,
        "T_cool_out": T_cool_out,
        "T_cool_in noise": u0_opt[4].item(),
        "step_time": time.time() - start_time,
        "convergence": convergence,
        "m_ref_in_optim": u0_opt[0].item(),
        "h_ref_in_optim": u0_opt[2].item(),
        "zeta_tpsh": z_tpsh.item(), 
        "predicted P" : x_pred[0].item(),
        "predicted H": x_pred[1].item(),
        "setpint P": x_sp[0].item(),
        "setpoint H": x_sp[1].item(),
        "setpoint m_ref_in": u_sp[0].item(),
        "setpoint h_ref_in": u_sp[2].item(),
    }, step=idx*2)

wandb.finish()