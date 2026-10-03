import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv
from sklearn.model_selection import train_test_split
import wandb

"""
PyTorch Geometric: GNN for Link-Level Path Prediction
-----------------------------------------------------
This script trains a GNN to predict link/path usage (Link_1 … Link_74)
in a fixed 23-node, 74-directed-edge topology, given:

  - Node features: [identity one-hot, is_source, is_dest]
  - Edge features: [current utilization ratio, requested bandwidth (normalized)]
  - Labels: link/path vector (Link_1 … Link_74)
"""
import time
start = time.time()
print("hi")
# ------------------------------
# Imports
# ------------------------------
import sys, functools
print = functools.partial(print, flush=True)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv
from sklearn.model_selection import train_test_split
import wandb
print("import done.")

# ------------------------------
# Training setup
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n=== Device Info ===")
print("Using device:", device)
if device.type == "cuda":
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("PyTorch CUDA build:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print("Current GPU:", torch.cuda.current_device())
else:
    print("Running on CPU only")

# ------------------------------
# Load raw data
# ------------------------------
V, E = 23, 74
capacity = 50000.0

data = pd.read_parquet("~/Saurav/MILP/Repo/Final_MLU.parquet").round(10)
# data = data.iloc[:4000,:]
print("\n=== Raw Data ===")
print("DataFrame shape:", data.shape)
print("First row:\n", data.iloc[0].values[:20], "...")

# Edge list
Link = pd.read_csv("/nlsasfs/home/mpplab-cdac/mitsa/Saurav/MILP/Repo/links.csv")
edges_list = list(zip(Link.iloc[:, -2], Link.iloc[:, -1]))
edge_index = torch.tensor(np.array(edges_list).T - 1, dtype=torch.long)  # (2, 74)
print("\n=== Edge Index ===")
print("Edge index shape:", edge_index.shape)
print("First 5 edges (src->dst):\n", edge_index[:, :5])

# ------------------------------
# Column mapping
# ------------------------------
LINK_START, LINK_END = 0, 74       # Link_1 … Link_74
UTIL_START, UTIL_END = 74, 148     # EdgeUtil_1 … EdgeUtil_74
SRC_START, SRC_END   = 148, 171    # Source one-hot (23)
DST_START, DST_END   = 171, 194    # Destination one-hot (23)
BW_COL               = 194         # Bandwidth

# ------------------------------
# Features and labels
# ------------------------------
# Labels: Link_1 … Link_74
y = data.iloc[:, LINK_START:LINK_END].values.astype(np.float32)  # (samples, 74)

# Edge features: EdgeUtil_1 … EdgeUtil_74
X_edges_util = data.iloc[:, UTIL_START:UTIL_END].values.reshape(-1, E, 1).astype(np.float32)

# Node features: source & destination one-hots
X_src = data.iloc[:, SRC_START:SRC_END].values.astype(np.float32)
X_dst = data.iloc[:, DST_START:DST_END].values.astype(np.float32)

# Bandwidth: normalized
X_bandwidth = (data.iloc[:, BW_COL].values.astype(np.float32) / capacity).reshape(-1, 1)

print("\n=== Parsed Data Shapes ===")
print("Labels (y - Links):", y.shape)
print("Edge utils:", X_edges_util.shape)
print("Source one-hot:", X_src.shape)
print("Dest one-hot:", X_dst.shape)
print("Bandwidth:", X_bandwidth.shape, "min/max:", X_bandwidth.min(), X_bandwidth.max())

# ------------------------------
# Train/val/test split
# ------------------------------
(
    X_edges_train, X_edges_temp,
    X_src_train, X_src_temp,
    X_dst_train, X_dst_temp,
    X_bw_train, X_bw_temp,
    y_train, y_temp
) = train_test_split(
    X_edges_util, X_src, X_dst, X_bandwidth, y, test_size=0.4, random_state=42
)

(
    X_edges_val, X_edges_test,
    X_src_val, X_src_test,
    X_dst_val, X_dst_test,
    X_bw_val, X_bw_test,
    y_val, y_test
) = train_test_split(
    X_edges_temp, X_src_temp, X_dst_temp, X_bw_temp, y_temp,
    test_size=0.5, random_state=42
)

print("\n=== Split Sizes ===")
print(f"Train: {X_edges_train.shape[0]}, Val: {X_edges_val.shape[0]}, Test: {X_edges_test.shape[0]}")

# ------------------------------
# PyG Dataset Builder
# ------------------------------
def build_dataset(X_edges, X_src, X_dst, X_bw, y, edge_index):
    dataset = []
    for i in range(len(y)):
        # 1) Edge features = [utilization, bandwidth]
        e = np.concatenate(
            [X_edges[i],
             np.repeat(X_bw[i][None, :], X_edges[i].shape[0], axis=0)], axis=-1
        )  # (74, 2)

        # 2) Node features = [identity one-hot, is_source, is_dest]
        node_identity = np.eye(V, dtype=np.float32)
        is_src = X_src[i][:, None]
        is_dst = X_dst[i][:, None]
        node_feats = np.concatenate([node_identity, is_src, is_dst], axis=1)  # (23, 25)

        # 3) Build Data object
        data_obj = Data(
            x=torch.tensor(node_feats, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=torch.tensor(e, dtype=torch.float),
            y=torch.tensor(y[i], dtype=torch.float)
        )
        dataset.append(data_obj)

        if i == 0:
            print("\n=== Debug: First Sample ===")
            print("Node features:", node_feats.shape)
            print("Edge features:", e.shape)
            print("Labels:", y[i][:5])
    return dataset

train_dataset = build_dataset(X_edges_train, X_src_train, X_dst_train, X_bw_train, y_train, edge_index)
val_dataset   = build_dataset(X_edges_val,   X_src_val,   X_dst_val,   X_bw_val,   y_val,   edge_index)
test_dataset  = build_dataset(X_edges_test,  X_src_test,  X_dst_test,  X_bw_test,  y_test,  edge_index)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32)
test_loader  = DataLoader(test_dataset, batch_size=32)

# ------------------------------
# GNN Model
# ------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import NNConv, global_mean_pool


class EdgeGNN(nn.Module):

    def __init__(
        self,
        node_in_dim,
        edge_dim,
        hidden_dim=128,
        out_dim=1
    ):
        super().__init__()

        # ---------------------------
        # Conv 1
        # ---------------------------
        nn1 = nn.Sequential(
            nn.Linear(edge_dim, 128),
            nn.ReLU(),
            nn.Linear(
                128,
                node_in_dim * hidden_dim
            )
        )

        self.conv1 = NNConv(
            node_in_dim,
            hidden_dim,
            nn1,
            aggr="mean"
        )

        # ---------------------------
        # Conv 2
        # ---------------------------
        nn2 = nn.Sequential(
            nn.Linear(edge_dim, 128),
            nn.ReLU(),
            nn.Linear(
                128,
                hidden_dim * hidden_dim
            )
        )

        self.conv2 = NNConv(
            hidden_dim,
            hidden_dim,
            nn2,
            aggr="mean"
        )

        # ---------------------------
        # Conv 3
        # ---------------------------
        nn3 = nn.Sequential(
            nn.Linear(edge_dim, 128),
            nn.ReLU(),
            nn.Linear(
                128,
                hidden_dim * hidden_dim
            )
        )

        self.conv3 = NNConv(
            hidden_dim,
            hidden_dim,
            nn3,
            aggr="mean"
        )

        # ---------------------------
        # Edge predictor
        # ---------------------------
        edge_input_dim = (
            hidden_dim +      # src
            hidden_dim +      # dst
            hidden_dim +      # global graph
            edge_dim
        )

        self.fc1 = nn.Linear(
            edge_input_dim,
            256
        )

        self.fc2 = nn.Linear(
            256,
            128
        )

        self.fc3 = nn.Linear(
            128,
            out_dim
        )

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # ---------------------------
        # Message passing
        # ---------------------------
        h1 = F.relu(
            self.conv1(
                x,
                edge_index,
                edge_attr
            )
        )

        h2 = F.relu(
            self.conv2(
                h1,
                edge_index,
                edge_attr
            )
        )

        h3 = F.relu(
            self.conv3(
                h2,
                edge_index,
                edge_attr
            )
        )

        # residual
        x = h1 + h3

        # ---------------------------
        # Global graph embedding
        # ---------------------------
        graph_emb = global_mean_pool(
            x,
            data.batch
        )

        src, dst = edge_index

        edge_batch = data.batch[src]

        global_edge_emb = graph_emb[
            edge_batch
        ]

        # ---------------------------
        # Edge representation
        # ---------------------------
        edge_emb = torch.cat(
            [
                x[src],
                x[dst],
                global_edge_emb,
                edge_attr
            ],
            dim=-1
        )

        # ---------------------------
        # Prediction head
        # ---------------------------
        edge_emb = F.relu(
            self.fc1(edge_emb)
        )

        edge_emb = F.relu(
            self.fc2(edge_emb)
        )

        out = self.fc3(edge_emb)

        return out.squeeze(-1)

# class EdgeGNN(nn.Module):
#     def __init__(self, node_in_dim, edge_dim, hidden_dim=64, out_dim=1):
#         super().__init__()
#         nn1 = nn.Sequential(
#             nn.Linear(edge_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, node_in_dim * hidden_dim)
#         )
#         self.conv1 = NNConv(node_in_dim, hidden_dim, nn1, aggr="mean")

#         nn2 = nn.Sequential(
#             nn.Linear(edge_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, hidden_dim * hidden_dim)
#         )
#         self.conv2 = NNConv(hidden_dim, hidden_dim, nn2, aggr="mean")

#         self.fc1 = nn.Linear(2 * hidden_dim + edge_dim, 128)
#         self.fc2 = nn.Linear(128, out_dim)

#     def forward(self, data):
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
#         x = F.relu(self.conv1(x, edge_index, edge_attr))
#         x = F.relu(self.conv2(x, edge_index, edge_attr))
#         src, dst = edge_index
#         edge_emb = torch.cat([x[src], x[dst], edge_attr], dim=-1)
#         edge_emb = F.relu(self.fc1(edge_emb))
#         return self.fc2(edge_emb).squeeze(-1)


# ------------------------------
# Training setup
# ------------------------------
model = EdgeGNN(node_in_dim=25, edge_dim=2, out_dim=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.MSELoss()

print("parameters")
para=sum(p.numel() for p in model.parameters())
print(sum(p.numel() for p in model.parameters()))

# ------------------------------
# Training & Eval Loops
# ------------------------------
def train_one_epoch(loader):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def eval_one_epoch(loader):
    model.eval()

    total_loss = 0
    total_mae = 0

    total_correct = 0
    total_elements = 0

    path_matches = 0
    num_graphs = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            pred = model(batch)

            total_loss += F.mse_loss(
                pred,
                batch.y,
                reduction="sum"
            ).item()
            
            total_mae += F.l1_loss(
                pred,
                batch.y,
                reduction="sum"
            ).item()
                        
            pred_bin = (torch.abs(pred) > 0.5)
            
            target_bin = (torch.abs(batch.y) > 0.5)
            
            pred_bin = pred_bin.view(-1, E)
            target_bin = target_bin.view(-1, E)
            
            path_matches += (
                (pred_bin == target_bin)
                .all(dim=1)
                .sum()
                .item()
            )
            
            num_graphs += pred_bin.shape[0]
            
            # pred_bin = (pred > 0.5)
            # target_bin = (batch.y > 0.5)

            # total_correct += (
            #     pred_bin == target_bin
            # ).sum().item()

            # total_elements += batch.y.numel()

    n = len(loader.dataset) * E

    mse = total_loss / n
    mae = total_mae / n
    path_match = path_matches / num_graphs

    return mse, mae, path_match

# ------------------------------
# WandB Init
# ------------------------------
wandb.init(
    project="Routing",
    mode="offline",
    name=f"GNN-{para}",
    config={
        "architecture": "NNConv + SD flags",
        "layers": [64, 64, 128],
        "activation": "relu",
        "optimizer": "adam",
        "loss": "mse",
        "epochs": 150,
        "batch_size": 32
    }
)
epochs = wandb.config["epochs"]

# ------------------------------
# Training loop (with best model saving)
# ------------------------------
best_val_loss = float("inf")
best_model_path = "best_edgegnn_model.pt"

for epoch in range(1, epochs + 1):

    train_loss = train_one_epoch(train_loader)

    val_loss, val_mae, val_agreement = eval_one_epoch(
        val_loader
    )

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"Val MAE: {val_mae:.6f} | "
        f"Val Agreement: {100*val_agreement:.2f}%"
    )

    wandb.log({
        "Train Loss": train_loss,
        "Val Loss": val_loss,
        "Val MAE": val_mae,
        "Val Agreement (%)": 100 * val_agreement
    })

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_agreement": val_agreement,
            "config": wandb.config,
        }, best_model_path)

        print(
            f"✅ Best model saved at epoch {epoch} "
            f"with Val Loss: {val_loss:.6f}"
        )
# ------------------------------
# Final evaluation (reload best model)
# ------------------------------

checkpoint = torch.load(
    best_model_path,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

test_loss, test_mae, test_agreement = eval_one_epoch(
    test_loader
)

print(
    f"Best Model Test Loss: {test_loss:.6f}, "
    f"Test MAE: {test_mae:.6f}, "
    f"Route Agreement: {100*test_agreement:.2f}%"
)

wandb.log({
    "Best Model Test Loss": test_loss,
    "Best Model Test MAE": test_mae,
    "Test Route Agreement (%)": 100 * test_agreement
})

wandb.finish()

print(
    f"Best model reloaded from: {best_model_path}"
)
end = time.time()

print(start - end)