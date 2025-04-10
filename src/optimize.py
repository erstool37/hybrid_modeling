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

with open("configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)



# Assume trained_model is already loaded and frozen
trained_model.eval()
for param in trained_model.parameters():
    param.requires_grad = False

# Input: current state
x_t = torch.tensor([1.0], requires_grad=False)  # shape depends on your model

# Initialize u_t as a parameter to optimize
u_t = torch.tensor([0.0], requires_grad=True)

# Define optimizer
optimizer = torch.optim.Adam([u_t], lr=0.01)

# Define objective function
def cost_fn(x_pred, u_pred):
    return (x_pred - 2.0).pow(2).mean() + 0.1 * u_pred.pow(2).mean()

# Optimization loop
for step in range(100):
    optimizer.zero_grad()

    # Predict next state and control
    x_next, u_next = trained_model(x_t, u_t)  # must return tensors with gradients

    # Compute cost
    loss = cost_fn(x_next, u_next)

    # Backpropagate through model
    loss.backward()

    # Gradient step on u_t
    optimizer.step()

print(f"Optimized u_t: {u_t.detach()}")