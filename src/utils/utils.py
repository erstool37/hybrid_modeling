import pandas as pd
import torch
from importlib.resources import files
import importlib
import wandb

# Load stats
def load_stats(method):
    cs = files('utils.dataset') / f'{method}.csv'
    return pd.read_csv(cs)

# z-score normalization
def znormalize(self, item, column, method):
    stats = load_stats(method)
    mean_item = torch.tensor(stats.loc[0, column])
    std_item = torch.tensor(stats.loc[1, column])
    return (item - mean_item) / std_item

# z-score unnormalization
def zunnormalize(self, item, column, method):
    stats = load_stats(method)
    mean_item = torch.tensor(stats.loc[0, column])
    std_item = torch.tensor(stats.loc[1, column])
    return item * std_item + mean_item

# min-max normalization
def normalize(self, item, column, method):
    stats = load_stats(method)
    max_item = torch.tensor(stats.loc[2, column])
    min_item = torch.tensor(stats.loc[3, column])
    return (item - min_item) / (max_item - min_item)

# min-max unnormalization
def unnormalize(item, column, method):
    stats = load_stats(method)
    max_item = torch.tensor(stats.loc[2, column])
    min_item = torch.tensor(stats.loc[3, column])
    return item * (max_item - min_item) + min_item

# Gradient unscaling for min-max normalized gradients
def gradunscaler(column, method):
    stats = load_stats(method)
    max_item = torch.tensor(stats.loc[2, column])
    min_item = torch.tensor(stats.loc[3, column])
    return max_item - min_item

def MAPEcalculator(pred, target, descaler, method):
    utils = importlib.import_module("utils")
    descaler = getattr(utils, descaler)
    
    pred_p = descaler(pred[:,0], "p_input").unsqueeze(-1)
    pred_h = descaler(pred[:,1], "h_ref_out").unsqueeze(-1)
    pred_z = descaler(pred[:,2], "zeta").unsqueeze(-1)

    target_p = descaler(target[:,0], "p_input").unsqueeze(-1)
    target_h = descaler(target[:,1], "h_ref_out").unsqueeze(-1)
    target_z = descaler(target[:,2], "zeta").unsqueeze(-1)

    loss_mape_p = torch.mean((torch.abs(pred_p - target_p) / target_p)).unsqueeze(-1)
    loss_mape_h = torch.mean((torch.abs(pred_h - target_h) / target_h)).unsqueeze(-1)
    loss_mape_z = torch.mean((torch.abs(pred_z - target_z) / target_z)).unsqueeze(-1)

    wandb.log({f"MAPE {method} pressure %" : loss_mape_p * 100})
    wandb.log({f"MAPE {method}  %" : loss_mape_h * 100})
    wandb.log({f"MAPE {method} surfT %" : loss_mape_z * 100})