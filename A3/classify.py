import torch
from models import GCNClassifier
from utils import calculate_accuracy
from dataset import GraphDataset
from torch_geometric.loader import DataLoader
from evaluate import Evaluator

BATCH_SIZE = 128
NUM_EPOCHS = 100

X_train = GraphDataset("./dataset/dataset_2/train")
X_val =  GraphDataset("./dataset/dataset_2/valid")

train_loader = DataLoader(X_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(X_val,batch_size=BATCH_SIZE, shuffle=True)


# if torch.cuda.is_available():
#     device = torch.device('cuda')
# elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#     device = torch.device('mps')
# else:
#     device = torch.device('cpu')

device = torch.device('cpu')

model = GCNClassifier(64, 1, 128)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=1e-3)
optimizer.zero_grad()

### Training Loop
for epoch in range(NUM_EPOCHS):
    model.train()
    model.to(device)
    total_loss = 0.0
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
        # print(float(loss))

        # Backward pass and optimization
        loss.mean().backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss+=loss.item()
        total_correct_train += calculate_accuracy(out.reshape(1, -1), batch.y.reshape(1, -1)) * batch.y.shape[0]
        total_samples_train += batch.y.shape[0]
    # scheduler.step()
    average_loss = total_loss / len(train_loader)
    accuracy_train = total_correct_train / total_samples_train

     # Validation phase
    model.eval()
    with torch.no_grad():
        total_bce = 0.0
        total_correct_val = 0
        total_samples_val = 0

        for i, batch in enumerate(val_loader):
             # Move batch to device
            batch.to(device)
            # Forward pass
            loss_fun = torch.nn.BCELoss()
            if(batch.edge_index.shape[0] == 0):
                continue
            out = model(batch.x, batch.edge_index.to(torch.long), batch.edge_attr, batch.batch, batch.y.shape[0])
            # print(out.shape)
            loss = loss_fun(out.reshape(1, -1), batch.y.reshape(1, -1))
            loss = loss.mean()

            total_bce += loss.item()
            total_correct_val += calculate_accuracy(out.reshape(1, -1), batch.y.reshape(1, -1)) * batch.y.shape[0]
            total_samples_val += batch.y.shape[0]

        average_bce_val = total_bce / len(val_loader)
        accuracy_val = total_correct_val / total_samples_val
    # Print logs
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {average_loss:.4f}, Train Accuracy: {accuracy_train:.4f}, Val BCE: {average_bce_val:.4f}, Val Accuracy: {accuracy_val:.4f}")
    
# Make sure to clear the computation graph after the loop
torch.cuda.empty_cache()

evaluator = Evaluator('dataset-2')
# benchmark = DataLoader(X_train_dat, batch_size= len(X_train_dat), shuffle = True)
val_loader = DataLoader(X_val, batch_size= len(X_val), shuffle = True)

for i, batch in enumerate(val_loader):
    model.eval()
    # Move batch to device
    batch.to(device)
    # Forward pass
    y_pred = model(batch.x, batch.edge_index.to(torch.long), batch.edge_attr, batch.batch, batch.y.shape[0])
    y_true = batch.y
    y_true = y_true.unsqueeze(1)
    # print(y_pred.shape)
    # print(y_true.shape)
    input_dict = {'y_true': y_true, 'y_pred': y_pred}
    result = evaluator.eval(input_dict)
    print(result)