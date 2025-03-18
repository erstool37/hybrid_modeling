import torch
from src.models.pinn import PINN
from src.losses.pinn_loss import PINN_Loss as L
from src.utils.paraloader import Paraloader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PARA_DIR = "dataset/dataset.csv"
STATS_DIR = "dataset/statistics_val.csv"
SEQ_LEN=30
BATCH_SIZE = 1
NUM_WORKERS = 1
HIDDEN_DIM = 128
NUM_LAYERS = 12
SIZE = 0.05

def unnormalize(item, column):
    stats = pd.read_csv(STATS_DIR)
    item_norm = (item * stats.loc[1, column]) + stats.loc[0, column]

    return item_norm

# load validation set(normalized)
val_ds = Paraloader(dir = PARA_DIR, stats_dir=STATS_DIR, sequence_length = SEQ_LEN)
indices = np.arange(len(val_ds))
train_idx, val_idx = train_test_split(indices, test_size=SIZE, shuffle=False)
val_ds = Subset(val_ds, val_idx)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# load model
model = PINN(input_size=7, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_size=6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("src/models/PINN0318_01.pth"))
model.eval()
model.cuda()

p_pred = []
p_true = []

h_pred = []
h_true = []

for batch in val_dl:
    model_input, ground_truth = batch
    model_input = model_input.to(device)
    outputs = model(model_input)

    outputs = outputs.detach().cpu().numpy()
    ground_truth = ground_truth.detach().cpu().numpy()
    
    print("pred", unnormalize(outputs[:,0],"pressure"))
    print("true", unnormalize(ground_truth[:,0],"pressure"))

    print("h_pred", unnormalize(outputs[:,1],"h_ref_out"))
    print("h_true", unnormalize(ground_truth[:,1],"h_ref_out"))
    
    p_pred.append(unnormalize(outputs[:,0],"pressure"))
    p_true.append(unnormalize(ground_truth[:,0],"pressure"))

    h_pred.append(unnormalize(outputs[:,1],"h_ref_out"))
    h_true.append(unnormalize(ground_truth[:,1],"h_ref_out"))

# Plot predicted vs true pressure
plt.figure(figsize=(10, 5))
plt.plot(p_pred, label="Predicted Pressure", linestyle='--')
plt.plot(p_true, label="True Pressure", linestyle='-')
plt.title("Predicted vs True Pressure")
plt.xlabel("Sample Index")
plt.ylabel("Normalized Pressure")
plt.legend()
plt.grid()
plt.savefig('predict_p.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 5))
plt.plot(h_pred, label="Predicted h_ref_out", linestyle='--')
plt.plot(h_true, label="True h_ref_out", linestyle='-')
plt.title("Predicted vs True h_ref_out")
plt.xlabel("Sample Index")
plt.ylabel("Normalized h_ref_out")
plt.legend()
plt.grid()
plt.savefig('predict_h.png', dpi=300, bbox_inches='tight')

plt.close()