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
import os

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

# for reproducibility
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.mkldnn.deterministic = True
torch.backends.mkldnn.benchmark = False
setseed(SEED)

# load modules
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
test_ds = data_class(dir = TEST_DIR, sequence_length = SEQ_LEN, method = 'test', scaler=SCALER)

indices = np.arange(len(train_ds))
indices2 = np.arange(len(test_ds))

train_idx, val_idx = train_test_split(indices, test_size=TEST_SIZE, shuffle=False)
dummy_idx, test_idx = train_test_split(indices2, test_size=0.1, shuffle=False)

train_ds = Subset(train_ds, train_idx)
val_ds = Subset(val_ds, val_idx)
test_ds = Subset(test_ds, test_idx)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=PIN_MEMORY)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=PIN_MEMORY)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=PIN_MEMORY)

# INITIALIZE
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model_class(input_size=7, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_size=6).to(device)
criterion = criterion_class(time_step=TIME_STEP, w_res=W_RES, w_theta=W_THETA, w_ode=W_ODE, descaler=DESCALER)
optimizer = optim_class(model.parameters(), lr=LR, weight_decay=W_DECAY)
scheduler = scheduler_class(optimizer, T_max=NUM_EPOCHS, eta_min=ETA_MIN)

# TRAINING
"""
wandb.watch(model, criterion, log="all", log_freq=5)
best_val_loss = float("inf")
counter = 0
for epoch in range(NUM_EPOCHS):
    model.train()
    train_losses = []
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Training ")
    for model_input, target in tqdm(train_dl): 
        model_input, target = model_input.to(device), target.to(device)

        outputs = model(model_input)
        
        train_loss = criterion(model_input, outputs, target)
        train_losses.append(train_loss.item())
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # loss print
        if (len(train_losses)) % 2 == 0:
            mean_train_loss = mean(train_losses)
            wandb.log({"train_loss": mean_train_loss})
    train_losses.clear()

    model.eval()
    val_losses = []
    with torch.no_grad():
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation")
        for model_input, target in tqdm(val_dl):
            model_input, target  = model_input.to(device), target.to(device)
            outputs = model(model_input)
            val_loss = criterion(model_input, outputs, target)
            val_losses.append(val_loss.item())
            MAPEcalculator(outputs.detach(), target.detach(), DESCALER, "total")

        mean_val_loss = mean(val_losses)
        wandb.log({"val_loss": mean_val_loss})

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} results - Train Loss: {mean_train_loss:.4f} Validation Loss: {mean_val_loss:.4f} - LR: {current_lr:.8f}")
wandb.finish()
torch.save(model.state_dict(), checkpoint)
"""

# Inference
checkpoint = "src/weights/total_test_run_0411_v2.pth"
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.lstm.flatten_parameters()
model.eval()

# Inference loop
pred, targets, errors = [], [], []
with torch.no_grad():
    for model_input, target in tqdm(val_dl):
        model_input, target = model_input.to(device), target.to(device)
        output = model(model_input)
        # for b in range(BATCH_SIZE):
        #     wandb.log({"pred test pressure": output[b, 0]})
        wandb.log({"pred test pressure": output[0, 0]})
        wandb.log({"real test pressure": target[0, 0]})

        error = MAPEtestcalculator(output.detach(), target.detach(), DESCALER, "total")
        errors.append(error)    
        pred.append(output)
        targets.append(target)

errors = torch.cat(errors, dim=0)
pred = torch.cat(pred, dim=0)
targets = torch.cat(targets, dim=0)

from sklearn.linear_model import LinearRegression
model_outputs = pred.cpu().numpy()  
true_targets = targets.cpu().numpy() 

slopes, intercepts = [], []
for i in range(model_outputs.shape[1]):
    reg = LinearRegression()
    Xi = model_outputs[:, i].reshape(-1, 1)   
    yi = true_targets[:,   i]                 
    reg.fit(Xi, yi)
    slopes.append(reg.coef_[0])
    intercepts.append(reg.intercept_)

import numpy as np
slopes = np.array(slopes)        
intercepts = np.array(intercepts)
adjusted_outputs = model_outputs * slopes + intercepts
adjusted_outputs = torch.tensor(adjusted_outputs, device=pred.device, dtype=pred.dtype)

r2_original = [r2_score(true_targets[:, i], model_outputs[:, i]) for i in range(model_outputs.shape[1])]

print("Slope:", slopes)
print("Intercept:", intercepts)

keys = ["pressure", "enthalpy", "zeta"]

inference(adjusted_outputs, targets, errors, DESCALER, "total", INF_DIR, run_name)