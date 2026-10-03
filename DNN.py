import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# --------------------------------------------------
# Reproducibility
# --------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

# --------------------------------------------------
# Load Data
# --------------------------------------------------
capacity = 50000.0

data = pd.read_parquet(
    "~/Saurav/MILP/Repo/Final_MLU.parquet"
).round(10)

# data = data.iloc[:400000, :]

# --------------------------------------------------
# Column Mapping
# --------------------------------------------------
LINK_START, LINK_END = 0, 74
UTIL_START, UTIL_END = 74, 148
SRC_START, SRC_END   = 148, 171
DST_START, DST_END   = 171, 194
BW_COL               = 194

# --------------------------------------------------
# Labels
# --------------------------------------------------
y = data.iloc[:, LINK_START:LINK_END].values.astype(np.float32)

# --------------------------------------------------
# Inputs
# --------------------------------------------------
X_util = data.iloc[:, UTIL_START:UTIL_END].values.astype(np.float32)
X_src  = data.iloc[:, SRC_START:SRC_END].values.astype(np.float32)
X_dst  = data.iloc[:, DST_START:DST_END].values.astype(np.float32)

X_bw = (
    data.iloc[:, BW_COL].values.astype(np.float32)
    / capacity
).reshape(-1, 1)

# --------------------------------------------------
# Final Input
# --------------------------------------------------
X = np.concatenate(
    [X_util, X_src, X_dst, X_bw],
    axis=1
)
print(X)
print("Input shape :", X.shape)
print("Label shape :", y.shape)

# --------------------------------------------------
# Train / Val / Test Split
# --------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42
)

print("Train:", len(X_train))
print(X_train)
print("Val  :", len(X_val))
print("Test :", len(X_test))

# --------------------------------------------------
# Torch Tensors
# --------------------------------------------------
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val   = torch.tensor(X_val, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_val   = torch.tensor(y_val, dtype=torch.float32)
y_test  = torch.tensor(y_test, dtype=torch.float32)
print(y_train)

# --------------------------------------------------
# DataLoaders
# --------------------------------------------------
BATCH_SIZE = 32

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=BATCH_SIZE
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=BATCH_SIZE
)

# --------------------------------------------------
# DNN Model
# --------------------------------------------------
class RoutingDNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(121, 2048),
            nn.ReLU(),

            nn.Linear(2048, 1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 74)
        )

    def forward(self, x):
        return self.net(x)

# --------------------------------------------------
# Model
# --------------------------------------------------
model = RoutingDNN().to(device)

params = sum(
    p.numel()
    for p in model.parameters()
)

print(model)
print(f"\nParameters: {params:,}")

# --------------------------------------------------
# WandB
# --------------------------------------------------
wandb.init(
    project="GNN_DNN_paper_FULL",
    mode="offline",
    name=f"DNN_{params}",
    config={
        "architecture": "DNN",
        "layers": [1024, 1024, 512],
        "parameters": params,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": BATCH_SIZE,
        "epochs": 150
    }
)

# --------------------------------------------------
# Optimizer / Loss
# --------------------------------------------------
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-5
)

loss_fn = nn.MSELoss()

# --------------------------------------------------
# Train
# --------------------------------------------------
def train_epoch(loader):

    model.train()

    total_loss = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        pred = model(x)

        loss = loss_fn(pred, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)

# --------------------------------------------------
# Evaluate
# --------------------------------------------------
def evaluate(loader):

    model.eval()

    mse = 0
    mae = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            mse += F.mse_loss(
                pred,
                y,
                reduction="sum"
            ).item()

            mae += F.l1_loss(
                pred,
                y,
                reduction="sum"
            ).item()

    n = len(loader.dataset) * 74

    return mse / n, mae / n

# --------------------------------------------------
# Training Loop
# --------------------------------------------------
EPOCHS = 150

best_val_mse = float("inf")
best_model_path = "best_dnn.pt"

for epoch in range(1, EPOCHS + 1):

    train_loss = train_epoch(train_loader)

    val_mse, val_mae = evaluate(val_loader)

    print(
        f"Epoch {epoch:03d} | "
        f"Train {train_loss:.6f} | "
        f"Val MSE {val_mse:.6f} | "
        f"Val MAE {val_mae:.6f}"
    )

    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_mse": val_mse,
        "val_mae": val_mae
    })

    if val_mse < best_val_mse:

        best_val_mse = val_mse

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mse": val_mse,
                "val_mae": val_mae,
                "parameters": params
            },
            best_model_path
        )

        print(
            f"✅ Best model saved at epoch "
            f"{epoch} with Val MSE {val_mse:.6f}"
        )

# --------------------------------------------------
# Reload Best Model
# --------------------------------------------------
checkpoint = torch.load(
    best_model_path,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

# --------------------------------------------------
# Test Evaluation
# --------------------------------------------------
test_mse, test_mae = evaluate(test_loader)

print("\n==========")
print("BEST MODEL RESULTS")
print("==========")
print("Parameters :", f"{params:,}")
print("Test MSE   :", test_mse)
print("Test MAE   :", test_mae)

wandb.log({"Train Loss": train_loss, "Val Loss": val_loss, "Val MAE": val_mae})

wandb.finish()