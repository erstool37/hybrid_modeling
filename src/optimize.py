import torch
import torch.nn as nn
import argparse
import numpy as np
import os.path as osp
import torch.optim as optim
from tqdm import tqdm
import importlib
import datetime
import yaml
import time
import wandb
import pandas as pd
from utils import  setseed, Xdenormalizer, Udenormalizer, Odenormalizer

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, default="configs/config.yaml")
args = parser.parse_args()

with open(args.config, "r") as file:
    config = yaml.safe_load(file)
cfg = config["optimization"]

NAME = cfg["name"]
PROJECT = cfg["project"]
VER = cfg["version"]
SEED = cfg["optimize_settings"]["seed"]
NUM = int(cfg["optimize_settings"]["num"])
MODEL = cfg["model"]["model_class"]
HIDDEN_DIM = int(cfg["model"]["lstm_size"])
NUM_LAYERS = int(cfg["model"]["lstm_layers"])
OUTPUT_SIZE = int(cfg["model"]["output_size"]) 
DATALOADER = cfg["model"]["dataloader"]
TIME_STEP = int(cfg["preprocess"]["time_step"])
SCALER = cfg["preprocess"]["scaler"]
DESCALER = cfg["preprocess"]["descaler"]
SEQ_LEN = int(cfg["preprocess"]["seq_len"])
CHECKPOINT = cfg["directories"]["checkpoint"]
DATA_ROOT = cfg["directories"]["data_root"]
COST = cfg["cost"]
OPTIM = cfg["optimizer"]["optim_class"]
SCHEDULER_CLASS = cfg["optimizer"]["scheduler_class"]
LR = float(cfg["optimizer"]["lr"])
MAX_ITER = int(cfg["optimizer"]["max_iter"])
TOL = float(cfg["optimizer"]["tolerance"])
TEMP= float(cfg["optimizer"]["temperature"])
STEP = int(cfg["optimizer"]["step"])
PARA_DIR = config["directories"]["data_root"]
NOISE = int(cfg["optimize_settings"]["noise"])

today = datetime.datetime.now().strftime("%m%d")
run_name = f"{NAME}_{today}_{VER}"

torch.use_deterministic_algorithms(True)
torch.backends.mkldnn.deterministic = True
torch.backends.mkldnn.benchmark = False
setseed(SEED)
wandb.init(project=PROJECT, name=run_name, reinit=True, resume="never", config= config)

# LOAD MODEL
device = "cuda" if torch.cuda.is_available() else "cpu"
data_module = importlib.import_module(f"datasets.{DATALOADER}")
model_module = importlib.import_module(f"models.{MODEL}")
cost_module = importlib.import_module(f"costs.{COST}")
cost_module = importlib.import_module(f"costs.{COST}")

data_class = getattr(data_module, DATALOADER)
model_class = getattr(model_module, MODEL)
cost_class = getattr(cost_module, COST)
optim_class = getattr(optim, OPTIM)
scheduler_class = getattr(optim.lr_scheduler, SCHEDULER_CLASS)

dataset = data_class(dir=DATA_ROOT, seq_len=SEQ_LEN, scaler=SCALER)
data_iter = iter(dataset)

model = model_class(input_size=7, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_size=6).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))

model.eval()
for param in model.parameters():
    param.requires_grad = False

# for convergence calculation
convergence = 0
T_cool_out_log = 0

# Simulation start
print("Simulation start")
set_temp = TEMP
for idx in tqdm(range(NUM)):
    full_seq = next(data_iter).to(device)
    hist = full_seq.detach()
    u_full = hist[0, -1, 2:7].clone().detach()
    h_ref_in_fixed = u_full[2].item()
    # m_cool_in_fixed = u_full[3].item()
    T_cool_in_fixed = u_full[4].item()
    u_now = torch.tensor([u_full[0].item(), u_full[2].item(), u_full[3].item()], device=u_full.device, requires_grad=True)

    cost = cost_class(T_target=set_temp, descaler=DESCALER).to(device)
    optimizer = optim_class([u_now], lr=LR)
    scheduler = scheduler_class(optimizer, T_max=MAX_ITER, eta_min=1e-9)

    # Optimization loop
    start_time = time.time()
    for step in range(1, MAX_ITER + 1):
        seq = hist.clone()
        full_u = torch.cat([u_now[0].unsqueeze(0), u_now[0].detach().unsqueeze(0), torch.tensor([h_ref_in_fixed], device=u_now.device), u_now[2].unsqueeze(0), torch.tensor([T_cool_in_fixed], device=u_now.device)])
        # full_u = torch.cat([u_now[0].unsqueeze(0), u_now[0].detach().unsqueeze(0), u_now[1].unsqueeze(0), torch.tensor([m_cool_in_fixed], device=u_now.device), torch.tensor([T_cool_in_fixed], device=u_now.device)])
        seq[0, -1, 2:7] = full_u
        x_pred = model(seq[:, :, :7]).squeeze(0)

        if step == 1:
            p_ref_out_next, h_ref_out_next = Xdenormalizer(x_pred, DESCALER, "optim")
            wandb.log({"predicted_pressure": p_ref_out_next, "predicted_enthalpy": h_ref_out_next}, step=idx*2) 
    
        optimizer.zero_grad()
        loss, T_cool_out_pred = cost(x_pred, seq[0, -1, 7:11], full_u)
        loss.backward()
        if loss < TOL:
            break
        optimizer.step()
        scheduler.step()

    end_time = time.time()
    T_ref_in, T_ref_out, T_cool_out, z_tpsh = Odenormalizer(seq[0, -1, 7:11], DESCALER, "optim")
    u_now = u_now.detach()
    # full_u = torch.cat([u_now[0].unsqueeze(0), u_now[0].unsqueeze(0), u_now[1].unsqueeze(0), torch.tensor([m_cool_in_fixed], device=u_now.device), torch.tensor([T_cool_in_fixed], device=u_now.device)])
    full_u = torch.cat([u_now[0].unsqueeze(0), u_now[0].detach().unsqueeze(0), torch.tensor([h_ref_in_fixed], device=u_now.device), u_now[2].unsqueeze(0), torch.tensor([T_cool_in_fixed], device=u_now.device)])
    u_now_unnorm = Udenormalizer(full_u, DESCALER, "optim")

    # Noise in T_cool_in
    if idx % int(STEP) == 0:
        u_now_unnorm[4] = (TEMP+3) + np.random.uniform(-NOISE, NOISE)
    u_optim = u_now_unnorm.cpu().numpy().tolist()
    improvement = abs(T_cool_out-TEMP)
    T_cool_out_log = T_cool_out.item()
    
    # Convergence check
    if improvement > 1e-2:
        convergence += 1
    else :
        wandb.log({"convergence step": convergence, "error": improvement}, step=idx*2)
        convergence = 0
    
    save_path = osp.join(DATA_ROOT, f"u_optim/u_optim_{(idx+1):04d}.csv")
    pd.DataFrame([u_optim], columns=["m_ref_in", "m_ref_out", "h_ref_in", "m_cool", "T_cool"]).to_csv(save_path, index=False)

    wandb.log({
        "T_cool_out desired": set_temp, "T_cool_out": T_cool_out, "T_cool_out_pred": T_cool_out_pred.item(), "T_cool_in noise": u_optim[4],
        "steps": step, "time": end_time - start_time, "total_loss": loss.item(),
        "m_ref_in_optim": u_optim[0], "h_ref_in_optim": u_optim[2], "m_cool": u_optim[3],
        "T_ref_in": T_ref_in.item(), "T_ref_out": T_ref_out.item(),
        "zeta_tpsh": z_tpsh.item(),
    }, step=idx*2)

wandb.finish()