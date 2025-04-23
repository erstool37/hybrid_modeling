import pandas as pd
import torch
from importlib.resources import files
import importlib
import wandb
import matplotlib.pyplot as plt
import json

# Load stats
def load_stats(method):
    cs = files('utils.stats') / f'{method}.csv'
    return pd.read_csv(cs)

# z-score normalization
def znormalize(item, column, method):
    stats = load_stats(method)
    mean_item = torch.tensor(stats.loc[0, column], dtype=torch.float32)
    std_item = torch.tensor(stats.loc[1, column], dtype=torch.float32)
    return (item - mean_item) / std_item

# z-score unnormalization
def zunnormalize(item, column, method):
    stats = load_stats(method)
    mean_item = torch.tensor(stats.loc[0, column], dtype=torch.float32)
    std_item = torch.tensor(stats.loc[1, column], dtype=torch.float32)
    return item * std_item + mean_item

# min-max normalization
def normalize(item, column, method):
    stats = load_stats(method)
    max_item = torch.tensor(stats.loc[2, column], dtype=torch.float32)
    min_item = torch.tensor(stats.loc[3, column], dtype=torch.float32)
    return (item - min_item) / (max_item - min_item)

# min-max unnormalization
def unnormalize(item, column, method):
    stats = load_stats(method)
    max_item = torch.tensor(stats.loc[2, column], dtype=torch.float32)
    min_item = torch.tensor(stats.loc[3, column], dtype=torch.float32)
    return item * (max_item - min_item) + min_item

# Gradient unscaling for min-max normalized gradients
def gradunscaler(column, method):
    stats = load_stats(method)
    max_item = torch.tensor(stats.loc[2, column], dtype=torch.float32)
    min_item = torch.tensor(stats.loc[3, column], dtype=torch.float32)
    return max_item - min_item

def Xnormalizer(x, scaler, method):
    utils = importlib.import_module("utils")
    scaler = getattr(utils, scaler)
    pressure = scaler(x[0].item(), "pressure", method).unsqueeze(-1)
    h_ref_out = scaler(x[1].item(), "h_ref_out", method).unsqueeze(-1)
    x_norm = torch.cat((pressure, h_ref_out), dim=0)
    return x_norm
        
def Unormalizer(u,scaler,method):
    utils = importlib.import_module("utils")
    scaler = getattr(utils, scaler)
    m_ref_in = scaler(u[0].item(), "m_ref_in", method).unsqueeze(-1)
    m_ref_out = scaler(u[1].item(), "m_ref_out", method).unsqueeze(-1)
    h_ref_in = scaler(u[2].item(), "h_ref_in", method).unsqueeze(-1)
    m_cool = scaler(u[3].item(), "m_cool", method).unsqueeze(-1)
    T_cool = scaler(u[4].item(), "T_cool_in", method).unsqueeze(-1)
    u_norm = torch.cat((m_ref_in, m_ref_out, h_ref_in, m_cool, T_cool), dim=0)
    return u_norm

def Xdenormalizer(x, descaler, method):
    utils = importlib.import_module("utils")
    descaler = getattr(utils, descaler)
    pressure = descaler(x[0], "pressure", method).unsqueeze(-1)
    h_ref_out = descaler(x[1], "h_ref_out", method).unsqueeze(-1)
    x_denorm = torch.cat((pressure, h_ref_out), dim=0)
    return x_denorm
        
def Udenormalizer(u,descaler,method):
    utils = importlib.import_module("utils")
    descaler = getattr(utils, descaler)
    m_ref_in = descaler(u[0], "m_ref_in", method).unsqueeze(-1)
    m_ref_out = descaler(u[1], "m_ref_out", method).unsqueeze(-1)
    h_ref_in = descaler(u[2], "h_ref_in", method).unsqueeze(-1)
    m_cool = descaler(u[3], "m_cool", method).unsqueeze(-1)
    T_cool = descaler(u[4], "T_cool_in", method).unsqueeze(-1)
    u_denorm = torch.cat((m_ref_in, m_ref_out, h_ref_in, m_cool, T_cool), dim=0)
    return u_denorm

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

def MAPEtestcalculator(pred, target, descaler, method):
    utils = importlib.import_module("utils")
    descaler = getattr(utils, descaler)
    
    pred_p = descaler(pred[:,0], "pressure", method).unsqueeze(-1)
    pred_h = descaler(pred[:,1], "h_ref_out", method).unsqueeze(-1)
    pred_z = descaler(pred[:,2], "z_tpsh", method).unsqueeze(-1)

    target_p = descaler(target[:,0], "pressure", method).unsqueeze(-1)
    target_h = descaler(target[:,1], "h_ref_out", method).unsqueeze(-1)
    target_z = descaler(target[:,2], "z_tpsh", method).unsqueeze(-1)

    loss_mape_p = torch.abs(pred_p - target_p) / target_p * 100
    loss_mape_h = torch.abs(pred_h - target_h) / target_h * 100
    loss_mape_z = torch.abs(pred_z - target_z) / target_z * 100

    errors = torch.cat((loss_mape_p, loss_mape_h, loss_mape_z), dim=1)

    return errors

def inference(pred, target, errors, descaler, method, save_dir, run_name):
    utils = importlib.import_module("utils")
    descaler = getattr(utils, descaler)

    keys = ["pressure", "h_ref_out", "z_tpsh"]

    pred_p = descaler(pred[:,0], "pressure", method).unsqueeze(-1)
    pred_h = descaler(pred[:,1], "h_ref_out", method).unsqueeze(-1)
    pred_z = descaler(pred[:,2], "z_tpsh", method).unsqueeze(-1)
    
    target_p = descaler(target[:,0], "pressure", method).unsqueeze(-1)
    target_h = descaler(target[:,1], "h_ref_out", method).unsqueeze(-1)
    target_z = descaler(target[:,2], "z_tpsh", method).unsqueeze(-1)

    pred = torch.cat((pred_p, pred_h, pred_z), dim=1).transpose(0,1).cpu()
    target = torch.cat((target_p, target_h, target_z), dim=1).transpose(0,1).cpu()
    errors = errors.transpose(0,1).cpu()

    mape = {}
    for pred, target, error, key in zip(pred, target, errors, keys):
        mape[key] = f"{float(error.mean()):.2f}%"

        plt.figure(figsize=(10, 5))
        plt.plot(pred, label=f"Predicted_{key}", linestyle='-', color='r')
        plt.plot(target, label=f"True_{key}", linestyle='-', color='b')
        plt.title(f"Predicted vs True {key}")
        plt.xlabel("Time")
        plt.ylabel(f"{key}")
        plt.legend()
        plt.grid()
        plt.savefig(f'{save_dir}/{run_name}_{key}.png', dpi=300, bbox_inches='tight')
        
    with open(f"{save_dir}/{run_name}.json", "w") as f:
            json.dump(mape, f, indent=2)


def msle_loss(input, target):
    input = torch.clamp(input, min=1e-7, max=1e2)
    target = torch.clamp(target, min=1e-7, max=1e2)

    error = torch.mean((torch.log1p(input) - torch.log1p(target))**2)

    return error