import sys
import torch
import argparse
import numpy as np
from models import GNNRegressor, GNN_TYPE
from dataset import GraphDataset
from torch_geometric.loader import DataLoader
from evaluator import Evaluator

parser = argparse.ArgumentParser(description="Train a model with specified paths.")

# Adding required string arguments
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to the training dataset')
parser.add_argument('--val_dataset_path', type=str, required=True, help='Path to the validation dataset')

# Parse the arguments
args = parser.parse_args()
model_path = args.model_path
dataset_path = args.dataset_path
val_dataset_path = args.val_dataset_path

print(f"Model Path: {model_path}")
print(f"Training Dataset Path: {dataset_path}")
print(f"Validation Dataset Path: {val_dataset_path}")

BATCH_SIZE = 128
NUM_EPOCHS = 100

X_train = GraphDataset(dataset_path)
X_val =  GraphDataset(val_dataset_path)

train_loader = DataLoader(X_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(X_val,batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cpu')

model = GNNRegressor(128, 1, 128, GNN_TYPE.GIN, 4, True)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3, weight_decay=5e-4)
optimizer.zero_grad()
loss_fun = torch.nn.MSELoss()

best_val_mse = 100
for epoch in range(NUM_EPOCHS):
    model.train()
    model.to(device)
    total_loss = 0.0
    for i, batch in enumerate(train_loader):
        
        # Move batch to device
        batch.to(device)
       
        # Forward pass
        out = model(batch.x, batch.edge_index.to(torch.long), batch.edge_attr, batch.batch, batch.y.shape[0])
        
        # Calculate and print loss
        optimizer.zero_grad()
        loss = loss_fun(out.reshape(1, -1), batch.y.reshape(1, -1))

        # Backward pass and optimization
        loss.mean().backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss+=loss.item()


    average_loss = total_loss / len(train_loader)

     # Validation phase
    model.eval()
    with torch.no_grad():
        total_val_mse_loss = 0.0
        
        for i, batch in enumerate(val_loader):
             # Move batch to device
            batch.to(device)
            if(batch.edge_index.shape[0] == 0):
                continue
            # Forward pass
            out = model(batch.x, batch.edge_index.to(torch.long), batch.edge_attr, batch.batch, batch.y.shape[0])
            loss = loss_fun(out.reshape(1, -1), batch.y.reshape(1, -1))
            loss = loss.mean()

            total_val_mse_loss += loss.item()

        average_mse_val = total_val_mse_loss / len(val_loader)
        if average_mse_val < best_val_mse:
            best_val_mse = average_mse_val
            torch.save(model.state_dict(), model_path)
            

    # Print logs
    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {average_loss:.4f} , Val MSE: {average_mse_val:.4f}")
        sys.stdout.flush()

# Make sure to clear the computation graph after the loop
torch.cuda.empty_cache()
model = GNNRegressor(128, 1, 128, GNN_TYPE.GIN, 4, True)
model.load_state_dict(torch.load(model_path))
val_loader = DataLoader(X_val, batch_size= BATCH_SIZE, shuffle = False)
evaluator = Evaluator('dataset-1')

y_true_all = None
y_pred_all = None
for i, batch in enumerate(val_loader):
    model.eval()
    # Move batch to device
    batch.to(device)
    # Forward pass
    y_pred = model(batch.x, batch.edge_index.to(torch.long), batch.edge_attr, batch.batch, batch.y.shape[0])
    y_true = batch.y

    if i == 0:
        y_true_all = y_true.unsqueeze(1).numpy()
        y_pred_all = y_pred.detach().cpu().numpy()
    else:
        y_true_all = np.concatenate([y_true_all, y_true.unsqueeze(1).numpy()], axis=0)
        y_pred_all = np.concatenate([y_pred_all, y_pred.detach().cpu().numpy()], axis=0)
    

input_dict = {'y_true': y_true_all, 'y_pred': y_pred_all}
result = evaluator.eval(input_dict)
print(result)
sys.stdout.flush()