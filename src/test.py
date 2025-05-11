import torch
import torch.nn as nn
import datetime
import wandb
import argparse
import numpy as np
import os.path as osp
import glob
import torch.optim as optim
from tqdm import tqdm
from statistics import mean
import importlib
import yaml
import json
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from utils import MAPEcalculator, MAPEtestcalculator, setseed, inference

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True, default="configs/config.yaml")
args = parser.parse_args()

with open(args.config, "r") as file:
    config = yaml.safe_load(file)
    
NAME = config["name"]
PROJECT = config["project"]
VER = config["version"]
BATCH_SIZE = int(config["train_settings"]["batch_size"])
NUM_EPOCHS = int(config["train_settings"]["num_epochs"])
NUM_WORKERS = int(config["train_settings"]["num_workers"])
SEED = int(config["train_settings"]["seed"])
PIN_MEMORY = config["train_settings"]["pin_memory"]
SCALER = config["preprocess"]["scaler"]
DESCALER = config["preprocess"]["descaler"]
TEST_SIZE = float(config["preprocess"]["test_size"])
RANDOM_STATE = int(config["preprocess"]["random_state"])
SEQ_LEN = int(config["preprocess"]["sequence_length"])
TIME_STEP = int(config["preprocess"]["time_step"])
DATALOADER = config["model"]["dataloader"]
MODEL = config["model"]["model_class"]
HIDDEN_DIM = int(config["model"]["lstm_size"])
NUM_LAYERS = int(config["model"]["lstm_layers"])
OUTPUT_SIZE = int(config["model"]["output_size"])
RATE = float(config["model"]["drop_rate"])
LOSS = config["loss"]
W_RES = float(config["model"]["w_res"])
W_THETA = float(config["model"]["w_theta"])
W_ODE = float(config["model"]["w_ode"])
OPTIMIZER = config["optimizer"]["optim_class"]
SCHEDULER = config["optimizer"]["scheduler_class"]
LR = float(config["optimizer"]["lr"])
ETA_MIN = float(config["optimizer"]["eta_min"])
W_DECAY = float(config["optimizer"]["weight_decay"])
PATIENCE = int(config["optimizer"]["patience"])
CHECKPOINT = config["directories"]["checkpoint"]
PARA_DIR = config["directories"]["data_root"]
TEST_DIR = config["directories"]["test_root"]
INF_DIR = config["directories"]["inf_root"]

setseed(SEED)

data_module = importlib.import_module(f"datasets.{DATALOADER}")
model_module = importlib.import_module(f"models.{MODEL}")
loss_module = importlib.import_module(f"losses.{LOSS}")

data_class = getattr(data_module, DATALOADER)
model_class = getattr(model_module, MODEL)
criterion_class = getattr(loss_module, LOSS)
optim_class = getattr(optim, OPTIMIZER)
scheduler_class = getattr(optim.lr_scheduler, SCHEDULER)

today = datetime.datetime.now().strftime("%m%d")
run_name = f"{NAME}_{today}_{VER}"
checkpoint = f"{CHECKPOINT}{run_name}.pth"

wandb.init(project=PROJECT, name=run_name, reinit=True, resume="never", config= config)

# DATASET
train_ds = data_class(dir = PARA_DIR, sequence_length = SEQ_LEN, method = 'train', scaler=SCALER)
val_ds = data_class(dir = PARA_DIR, sequence_length = SEQ_LEN, method = 'total', scaler=SCALER)
test_ds = data_class(dir = PARA_DIR, sequence_length = SEQ_LEN, method = 'test', scaler=SCALER)

indices = np.arange(len(train_ds))
indices2 = np.arange(len(test_ds))

train_idx, val_idx = train_test_split(indices, test_size=TEST_SIZE, shuffle=False)
dummy_idx, test_idx = train_test_split(indices2, test_size=0.1, shuffle=False)

train_ds = Subset(train_ds, train_idx)
val_ds = Subset(val_ds, val_idx)
test_ds = Subset(test_ds, test_idx)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=PIN_MEMORY)
val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=PIN_MEMORY)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=PIN_MEMORY)

# INITIALIZE
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model_class(input_size=7, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_size=6).to(device)
criterion = criterion_class(time_step=TIME_STEP, w_res=W_RES, w_theta=W_THETA, w_ode=W_ODE, descaler=DESCALER)
optimizer = optim_class(model.parameters(), lr=LR, weight_decay=W_DECAY)
scheduler = scheduler_class(optimizer, T_max=NUM_EPOCHS, eta_min=ETA_MIN)

checkpoint = "src/weights/total_test_run_0511_v2.pth"
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()

pred, targets, errors = [], [], []
hidden = None
with torch.no_grad():
    for model_input, target in tqdm(val_dl):
        model_input, target = model_input.to(device), target.to(device)
        output,_ = model(model_input, hidden)
        wandb.log({"real pressure": target[0, 0]})
        wandb.log({"pred pressure": output[0,0]}) # << 이부분이 너가 본 그래프의 왼쪽 그래프

        error = MAPEtestcalculator(output.detach(), target.detach(), DESCALER, "total")
        errors.append(error)    
        pred.append(output)
        targets.append(target)

errors = torch.cat(errors, dim=0)
pred = torch.cat(pred, dim=0)
targets = torch.cat(targets, dim=0)
keys = ["pressure", "enthalpy", "zeta"]
inference(pred, targets, errors, DESCALER, "test", INF_DIR, run_name)