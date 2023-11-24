import sys
import torch
import argparse
import numpy as np
from models import GNNClassifier, GNN_TYPE
from dataset import GraphDataset
from utils import calculate_accuracy
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
evaluator = Evaluator('dataset-2')

BATCH_SIZE = 128
NUM_EPOCHS = 100

X_train = GraphDataset(dataset_path)
X_val =  GraphDataset(val_dataset_path)

train_loader = DataLoader(X_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(X_val,batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cpu')

model = GNNClassifier(128, 1, 128, GNN_TYPE.GIN, 3, True)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3, weight_decay=5e-4)
optimizer.zero_grad()

best_roc_auc = 0.0

### Training Loop
for epoch in range(NUM_EPOCHS):
    model.train()
    model.to(device)
    total_loss = 0.0
    total_roc = 0.0
    total_correct_train = 0
    total_samples_train = 0
    for i, batch in enumerate(train_loader):
        # Move batch to device
        batch.to(device)
        weights = torch.ones_like(batch.y.reshape(1,-1))
        weights[batch.y.reshape(1,-1) == 1] = 2.0
        loss_fun = torch.nn.BCELoss(weight= weights)

        out = model(batch.x, batch.edge_index.to(torch.long), batch.edge_attr, batch.batch, batch.y.shape[0])

        optimizer.zero_grad()
        loss = loss_fun(out.reshape(1, -1), batch.y.reshape(1, -1))
        
        input_dict = {'y_true': batch.y.reshape(-1, 1), 'y_pred': out.reshape(-1, 1)}
        total_roc += evaluator.eval(input_dict)['rocauc']

        # Backward pass and optimization
        loss.mean().backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss+=loss.item()
        total_correct_train += calculate_accuracy(out.reshape(1, -1), batch.y.reshape(1, -1)) * batch.y.shape[0]
        total_samples_train += batch.y.shape[0]
    # scheduler.step()
    average_loss = total_loss / len(train_loader)
    average_roc = total_roc / len(train_loader)
    accuracy_train = total_correct_train / total_samples_train

     # Validation phase
    model.eval()
    with torch.no_grad():
        total_bce = 0.0
        total_correct_val = 0
        total_samples_val = 0
        roc_auc_sum = 0.0

        for i, batch in enumerate(val_loader):
             # Move batch to device
            batch.to(device)
            # Forward pass
            loss_fun = torch.nn.BCELoss()
            if(batch.edge_index.shape[0] == 0):
                continue
            out = model(batch.x, batch.edge_index.to(torch.long), batch.edge_attr, batch.batch, batch.y.shape[0])
            
            
            loss = loss_fun(out.reshape(1, -1), batch.y.reshape(1, -1))
            loss = loss.mean()
            
            input_dict = {'y_true': batch.y.reshape(-1, 1), 'y_pred': out.reshape(-1, 1)}
            roc_auc_sum += evaluator.eval(input_dict)['rocauc']

            total_bce += loss.item()
            total_correct_val += calculate_accuracy(out.reshape(1, -1), batch.y.reshape(1, -1)) * batch.y.shape[0]
            total_samples_val += batch.y.shape[0]

        average_bce_val = total_bce / len(val_loader)
        accuracy_val = total_correct_val / total_samples_val
        roc_auc = roc_auc_sum / len(val_loader)
        
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            torch.save(model.state_dict(), model_path)
        
        
    # Print logs
    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {average_loss:.4f}, Train Accuracy: {accuracy_train:.4f}, Train-ROC: {average_roc:.4f}, Val BCE: {average_bce_val:.4f}, Val Accuracy: {accuracy_val:.4f}, Val ROC-AUC: {roc_auc:.4f}")
        sys.stdout.flush()
# Make sure to clear the computation graph after the loop
torch.cuda.empty_cache()



model = GNNClassifier(128, 1, 128, GNN_TYPE.GIN, 3, True)
model.load_state_dict(torch.load(model_path))
val_loader = DataLoader(X_val, batch_size= BATCH_SIZE, shuffle = False)

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