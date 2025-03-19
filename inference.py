import torch
from src.models.pinn import PINN
from src.losses.pinn_loss import PINN_Loss as L
from src.utils.paraloader import Paraloader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

PARA_DIR = "dataset/dataset.csv"
STATS_DIR = "dataset/statistics_val.csv"
CHECKPOINT = "src/models/weights_pinn/PINN0320_01.pth"

SEQ_LEN=30
BATCH_SIZE = 1
NUM_WORKERS = 1
HIDDEN_DIM = 128
NUM_LAYERS = 16
SIZE = 0.05


def unnormalize(item, column):
    stats = pd.read_csv(STATS_DIR)
    item_norm = (item) * (stats.loc[1, column]) + (stats.loc[0, column])
    return item_norm

def h_satu(x):
    x = (x - 1670.0) / 955.3279
    y = np.array([x**8, x**7, x**6, x**5, x**4, x**3, x**2, x, 1.0])
    coeff = np.array([-0.870502063247313, 0.735216547740597, 3.35155930854622, -2.33678206224002, -5.32622216975478, 3.24797824375616, -4.21395627140569, 7.79742106651108, 396.065396306915])
    return np.dot(y,coeff)

# load validation set(normalized)
val_ds = Paraloader(dir = PARA_DIR, stats_dir=STATS_DIR, sequence_length = SEQ_LEN)
indices = np.arange(len(val_ds))
train_idx, val_idx = train_test_split(indices, test_size=SIZE, shuffle=False)
val_ds = Subset(val_ds, val_idx)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# load model
model = PINN(input_size=7, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_size=6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(CHECKPOINT))
model.eval()
model.cuda()

p_pred_list = []
p_true_list = []
h_pred_list = []
h_true_list = []
h_sat_list = []
zeta_pred_list = []
zeta_true_list = []

for batch in val_dl:
    model_input, ground_truth = batch
    model_input = model_input.to(device)
    outputs = model(model_input)

    outputs = outputs.detach().cpu().numpy()
    ground_truth = ground_truth.detach().cpu().numpy()
    
    p_pred = unnormalize(outputs[:,0],"pressure")
    h_pred = unnormalize(outputs[:,1],"h_ref_out")
    zeta_pred = unnormalize(outputs[:,2], "z_tpsh")

    h_sat = h_satu(p_pred.item())
    p_true = unnormalize(ground_truth[:,0],"pressure")
    h_true = unnormalize(ground_truth[:,1],"h_ref_out")
    zeta_true = unnormalize(ground_truth[:,2], "z_tpsh")

    p_pred_list.append(p_pred.item())
    p_true_list.append(p_true.item())
    h_pred_list.append(h_pred)
    h_true_list.append(h_true)
    h_sat_list.append(h_sat)
    zeta_pred_list.append(zeta_pred.item())
    zeta_true_list.append(zeta_true.item())
    
# Plot predicted vs true pressure
plt.figure(figsize=(10, 5))
plt.plot(p_pred_list, label="Predicted Pressure", linestyle='--')
plt.plot(p_true_list, label="True Pressure", linestyle='-')
plt.title("Predicted vs True Pressure")
plt.xlabel("Sample Index")
plt.ylabel("Normalized Pressure")
plt.legend()
plt.grid()
plt.savefig('predict_p.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 5))
plt.plot(h_pred_list, label="Predicted h_ref_out", linestyle='--')
plt.plot(h_true_list, label="True h_ref_out", linestyle='-')
plt.plot(h_sat_list , label="Saturation h_ref_out", linestyle='-.')
plt.title("Predicted vs True h_ref_out")
plt.xlabel("Sample Index")
plt.ylabel("Normalized h_ref_out")
plt.legend()
plt.grid()
plt.savefig('predict_h.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 5))
plt.plot(zeta_pred_list, label="Predicted zeta", linestyle='--')
plt.plot(zeta_true_list, label="True zeta", linestyle='-')
plt.title("Predicted vs True zeta")
plt.xlabel("Sample Index")
plt.ylabel("Normalized zeta")
plt.legend()
plt.grid()
plt.savefig('predict_z.png', dpi=300, bbox_inches='tight')

plt.close()