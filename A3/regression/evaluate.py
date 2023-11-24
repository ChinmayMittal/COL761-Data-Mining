import sys
import torch
import numpy as np
import argparse
from models import GNNRegressor, GNN_TYPE
from dataset import GraphDataset
from utils import tocsv
from torch_geometric.loader import DataLoader


parser = argparse.ArgumentParser(description="Evaluate a model with specified paths.")

# Adding required string arguments
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to the training dataset')

# Parse the arguments
args = parser.parse_args()
model_path = args.model_path
dataset_path = args.dataset_path

print(f"Model Path: {model_path}")
print(f"Eval Dataset Path: {dataset_path}")

BATCH_SIZE = 128

X_val =  GraphDataset(dataset_path, labels=False)
val_loader = DataLoader(X_val,batch_size=BATCH_SIZE, shuffle=False)


device = torch.device('cpu')
model = GNNRegressor(128, 1, 128, GNN_TYPE.GIN, 4, True)
model.load_state_dict(torch.load(model_path))

preds = np.array([])
for i, batch in enumerate(val_loader):
    model.eval()
    # Move batch to device
    batch.to(device)
    # Forward pass
    y_pred = model(batch.x, batch.edge_index.to(torch.long), batch.edge_attr, batch.batch, batch.y.shape[0])

    y_pred = y_pred.squeeze(dim=1).detach().cpu().numpy()
    preds = np.concatenate([preds, y_pred])
    
    sys.stdout.flush()

# print(preds)
# print(preds.shape)
tocsv(preds, task="regression")
