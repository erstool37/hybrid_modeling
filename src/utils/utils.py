import pandas as pd
import torch
from importlib.resources import files
import importlib
import wandb

# Load stats
def load_stats(method):
    cs = files('utils.stats') / f'{method}.csv'
    return pd.read_csv(cs)

# z-score normalization
def znormalize(item, column, method):
    stats = load_stats(method)
    mean_item = torch.tensor(stats.loc[0, column])
    std_item = torch.tensor(stats.loc[1, column])
    return (item - mean_item) / std_item

# z-score unnormalization
def zunnormalize(item, column, method):
    stats = load_stats(method)
    mean_item = torch.tensor(stats.loc[0, column])
    std_item = torch.tensor(stats.loc[1, column])
    return item * std_item + mean_item

# min-max normalization
def normalize(item, column, method):
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
    
    pred_p = descaler(pred[:,0], "pressure", method).unsqueeze(-1)
    pred_h = descaler(pred[:,1], "h_ref_out", method).unsqueeze(-1)
    pred_z = descaler(pred[:,2], "z_tpsh", method).unsqueeze(-1)

    target_p = descaler(target[:,0], "pressure", method).unsqueeze(-1)
    target_h = descaler(target[:,1], "h_ref_out", method).unsqueeze(-1)
    target_z = descaler(target[:,2], "z_tpsh", method).unsqueeze(-1)

    loss_mape_p = torch.mean((torch.abs(pred_p - target_p) / target_p)).unsqueeze(-1)
    loss_mape_h = torch.mean((torch.abs(pred_h - target_h) / target_h)).unsqueeze(-1)
    loss_mape_z = torch.mean((torch.abs(pred_z - target_z) / target_z)).unsqueeze(-1)

    wandb.log({f"MAPE {method} pressure %" : loss_mape_p * 100})
    wandb.log({f"MAPE {method} enthalpy %" : loss_mape_h * 100})
    wandb.log({f"MAPE {method} zeta %" : loss_mape_z * 100})