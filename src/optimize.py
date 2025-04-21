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
import matlab.engine
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, default="configs/config.yaml")
args = parser.parse_args()

with open(args.config, "r") as file:
    config = yaml.safe_load(file)
cfg = config["optimization"]

NAME = cfg["name"]
PROJECT = cfg["project"]
VER = config["version"]
SEED = cfg["optimize_settings"]["seed"]
NUM = int(cfg["optimize_settings"]["num"])
VER = cfg["optimize_settings"]["version"]
MODEL = cfg["model"]["model_class"]
HIDDEN_DIM = int(cfg["model"]["lstm_size"])
NUM_LAYERS = int(cfg["model"]["lstm_layers"])
OUTPUT_SIZE = int(cfg["model"]["output_size"]) 
DATALOADER = cfg["model"]["dataloader"]
TIME_STEP = int(cfg["preprocess"]["time_step"])
SCALER = cfg["preprocess"]["scaler"]
DESCALER = cfg["preprocess"]["descaler"]
SEQ_LEN = int(cfg["preprocess"]["seq_length"])
CHECKPOINT = cfg["directories"]["checkpoint"]
DATA_ROOT = cfg["directories"]["data_root"]
BASE = cfg["directories"]["base_root"]
COST = cfg["cost"]
OPTIM = cfg["optimizer"]["optim_class"]
SCHEDULER = cfg["optimizer"]["scheduler_class"]
LR = float(cfg["optimizer"]["lr"])
MAX_ITER = int(cfg["optimizer"]["max_iter"])
TOL = float(cfg["optimizer"]["tolerance"])
TEMP= float(cfg["optimizer"]["temperature"])

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

dataset = data_class(data_root=DATA_ROOT, seq_len=SEQ_LEN, base=BASE, scaler=SCALER)
data_iter = iter(dataset)
model = model_class(input_size=7, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_size=6).to(device)
cost= cost_class(T_target=TEMP, descaler=DESCALER).to(device)

model.eval()
for param in model.parameters():
    param.requires_grad = False

# Optimization loop
for idx in tqdm(range(NUM)):
    model_input = next(data_iter)

    pred = model(model_input)
    x_pred = pred[:2]

    u_now = model_input[-1:,2:] # u_now : optimized input in the previous iter   
    optimizer = optim_class(u_now, lr=LR)
    start_time = time.time()
    optimizer.zero_grad()

    # Optimization loop
    for step in range(MAX_ITER):     
        loss = cost(x_pred, u_now)
        loss.backward()
        optimizer.step()
        wandb.log({"cost": loss.item()})

        if torch.norm(u_now.grad) < TOL:
            elapsed = time.time() - start_time
            break

    u_now_unnorm = Udenormalizer(u_now, DESCALER, "optim")
    u_optim = u_now_unnorm.detach().cpu().numpy().tolist()

    with open(osp.join(DATA_ROOT,f"u_optim/u_optim_{idx}.json", "w")) as f:
        json.dump({"u_optim": u_optim}, f)

    wandb.log({
        "steps": step+1, 
        "time": elapsed,
        "m_ref_in_optim": u_optim[0],
        "m_ref_out_optim": u_optim[1],
        "h_ref_in_optim": u_optim[2],
        "m_cool_optim": u_optim[3],
        "T_cool_optim": u_optim[4]
    })

wandb.finish()