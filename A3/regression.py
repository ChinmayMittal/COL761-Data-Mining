import torch
from evaluate import Evaluator
from dataset import GraphDataset
from models import *
from torch_geometric.loader import DataLoader


BATCH_SIZE = 128
NUM_EPOCHS = 100

X_train = GraphDataset("./dataset/dataset_1/train")
X_val = GraphDataset("./dataset/dataset_1/valid")

train_loader = DataLoader(X_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(X_val, batch_size=len(X_val), shuffle=True)

# if torch.cuda.is_available():
#     device = torch.device('cuda')
# elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#     device = torch.device('mps')
# else:
#     device = torch.device('cpu')

device = torch.device('cpu')

model = GCNRegressor(64, 1, 128)
# model = LinearRegression(64)
# model = BaselineRegressor(64)
optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)

loss_fun = torch.nn.MSELoss()
optimizer.zero_grad()
NUM_EPOCHS = 150

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
        # print(batch.y.shape[0])
        optimizer.zero_grad()
        loss = loss_fun(out.reshape(1, -1), batch.y.reshape(1, -1))
        # print(float(loss))

        # Backward pass and optimization
        loss.mean().backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss+=loss.item()
    # scheduler.step()
    average_loss = total_loss / len(train_loader)

     # Validation phase
    model.eval()
    with torch.no_grad():
        total_bce = 0.0
        
        for i, batch in enumerate(val_loader):
             # Move batch to device
            batch.to(device)
            # Forward pass
            out = model(batch.x, batch.edge_index.to(torch.long), batch.edge_attr, batch.batch, batch.y.shape[0])
            loss = loss_fun(out.reshape(1, -1), batch.y.reshape(1, -1))
            loss = loss.mean()

            total_bce += loss.item()

        average_bce_val = total_bce / len(val_loader)

    # Print logs
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {average_loss:.4f} , Val MSE: {average_bce_val:.4f}")
# Make sure to clear the computation graph after the loop
torch.cuda.empty_cache()

evaluator = Evaluator('dataset-1')
for i, batch in enumerate(val_loader):
    model.eval()
    # Move batch to device
    batch.to(device)
    # Forward pass
    y_pred = model(batch.x, batch.edge_index.to(torch.long), batch.edge_attr, batch.batch, batch.y.shape[0])
    y_true = batch.y
    y_true = y_true.unsqueeze(1)

    input_dict = {'y_true': y_true, 'y_pred': y_pred}
    result = evaluator.eval(input_dict)
    print(result)


