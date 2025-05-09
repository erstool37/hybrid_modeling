import torch
import torch.nn as nn
import argparse
import numpy as np
import os.path as osp
import torch.optim as optim
from tqdm import tqdm
from statistics import mean
import importlib
import datetime
import yaml
import time
import json
from sklearn.model_selection import train_test_split
from utils import MAPEcalculator, MAPEtestcalculator, setseed, inference, Xdenormalizer, Udenormalizer
import wandb
import pandas as pd

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

today = datetime.datetime.now().strftime("%m%d")
run_name = f"{NAME}_{today}_{VER}"
checkpoint = f"{CHECKPOINT}{run_name}.pth"

setseed(SEED)

wandb.init(project=PROJECT, name=run_name, reinit=True, resume="never", config= config)

# LOAD MODEL
device = "cuda" if torch.cuda.is_available() else "cpu"
data_module = importlib.import_module(f"datasets.{DATALOADER}")
model_module = importlib.import_module(f"models.{MODEL}")
cost_module = importlib.import_module(f"costs.{COST}")

data_class = getattr(data_module, DATALOADER)
model_class = getattr(model_module, MODEL)
cost_class = getattr(cost_module, COST)
optim_class = getattr(optim, OPTIM)
scheduler_class = getattr(optim.lr_scheduler, SCHEDULER_CLASS)

dataset = data_class(dir=DATA_ROOT, seq_len=SEQ_LEN, scaler=SCALER)
data_iter = iter(dataset)
model = model_class(input_size=7, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_size=6).to(device)

model.eval()
for param in model.parameters():
    param.requires_grad = False

# Optimization loop
set_temp = TEMP
for idx in tqdm(range(NUM)):
    model_input = next(data_iter)
    pred = model(model_input[:,:,:7])
    x_pred = pred.squeeze(0)[:]
    u_now = model_input[0, -1:, 2:7].squeeze(0).clone().detach().requires_grad_(True)

    cost = cost_class(T_target=set_temp, descaler=DESCALER).to(device)
    optimizer = optim_class([u_now], lr=LR) # without T_cool, is actually the state variable
    scheduler = scheduler_class(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-4) # for ReduceLRonPlateau
    start_time = time.time()
    
    # Optimization loop
    print(f"Optimizing for sample {idx+1}/{NUM}")
    step = 1
    while step < MAX_ITER:
        optimizer.zero_grad()
        loss = cost(x_pred, model_input[:,-1,:].squeeze(0), u_now)
        loss.backward()
        if torch.norm(u_now.grad) < TOL:
            break
        
        scheduler.step(loss)
        scheduler.get_last_lr()[0]
        optimizer.step()
        step += 1
        wandb.log({"cost": loss.item()})
        
    u_now = u_now.detach()
    u_now[1]=u_now[0] # enforce m_ref_out = m_ref_in
    
    # LOG and SAVE
    print("Optimization finished")
    elapsed = time.time() - start_time
    u_now_unnorm = Udenormalizer(u_now, DESCALER, "optim")
    u_optim = u_now_unnorm.detach().cpu().numpy().tolist()

    if idx % STEP == 0: # change T_desired, T_cool_in to check adaptability
        set_temp = TEMP + np.random.uniform(-2, 2)
        u_optim[4]+= np.random.uniform(-5, 5)

    wandb.log({"T_cool_in desired": set_temp})
    wandb.log({"T_cool_in noise": u_optim[4]})

    save_path = osp.join(DATA_ROOT, f"u_optim/u_optim_{(idx+1):04d}.csv")
    df = pd.DataFrame([u_optim], columns=[
        "m_ref_in", "m_ref_out", "h_ref_in", "m_cool", "T_cool"
    ])
    df.to_csv(save_path, index=False)

    wandb.log({
        "steps": step, 
        "time": elapsed,
        "m_ref_in_optim": u_optim[0],
        "m_ref_out_optim": u_optim[1],
        "h_ref_in_optim": u_optim[2],
        "m_cool_optim": u_optim[3],
        "T_cool_optim": u_optim[4]
    })

wandb.finish()