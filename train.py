import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from statistics import mean
from src.models.pinn import PINN
from src.losses.pinn_loss import PINN_Loss
from src.utils.paraloader import Paraloader

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
    
BATCH_SIZE = int(config["settings"]["batch_size"])
NUM_WORKERS = int(config["settings"]["num_workers"])
NUM_EPOCHS = int(config["settings"]["num_epochs"])
LR_RATE = float(config["settings"]["lr_rate"])
HIDDEN_DIM = int(config["settings"]["hidden_dim"])
NUM_LAYERS = int(config["settings"]["num_layers"])
ALPHA = float(config["settings"]["alpha"])
SEQ_LEN = int(config["settings"]["sequence_length"])

PARA_DIR = config["directories"]["para_dir"]
CHECKPOINT = config["directories"]["checkpoint"]
REF = config["directories"]["ref"]
COOL = config["directories"]["cool"]

# wandb.init(project='Heat exchanger', reinit=True, resume="never", config=config)

# load and split dataset
train_ds = Paraloader(dir = PARA_DIR, sequence_length = SEQ_LEN)
val_ds = Paraloader(dir = PARA_DIR, sequence_length = SEQ_LEN)

indices = np.arange(len(train_ds))
train_idx, val_idx = train_test_split(indices, test_size=0.2, shuffle=False, random_state=37)

train_ds = Subset(train_ds, train_idx)
val_ds = Subset(val_ds, val_idx)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Initialize model, loss, scheduler, and optimizer
model = PINN(input_size=7, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_size=6)
criterion = PINN_Loss(alpha = ALPHA)

optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training loop
num_epochs = NUM_EPOCHS

for epoch in range(num_epochs):
    model.train()
    train_losses = []
    print(f"Epoch {epoch+1}/{num_epochs} - Training ")
    for batch in train_dl: 
        model_input, ground_truth = batch
        outputs = model(model_input)
        xdot_model = 1
        # xdot_model = torch.autograd.grad(outputs=outputs[0], outputs[1], inputs=x, grad_outputs=torch.ones_like(outputs), create_graph=True, retain_graph=True)[0]
        train_loss = criterion(xdot_model, outputs, ground_truth)

        train_losses.append(train_loss.item())
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # loss print
        if (len(train_losses)) % 100 == 0:
            mean_train_loss = mean(train_losses)
            wandb.log({"train_loss": mean_train_loss})
    train_losses.clear()

    model.eval()
    val_losses = []
    with torch.no_grad():
        print(f"Epoch {epoch+1}/{num_epochs} - Validation")
    for batch in val_dl:
        model_input, ground_truth = batch
        outputs = model(model_input, ground_truth)
        train_loss = criterion(model_input, outputs, ground_truth)
    mean_val_loss = mean(val_losses)
    wandb.log({"val_loss": mean_val_loss})

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    print(f"Epoch {epoch+1}/{num_epochs} results - Train Loss: {mean_train_loss:.4f} Validation Loss: {mean_val_loss:.4f} - LR: {current_lr:.5f}")
wandb.finish()

torch.save(model.state_dict(), CHECKPOINT)