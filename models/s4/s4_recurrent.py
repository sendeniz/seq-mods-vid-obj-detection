import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from models.s4.s4_block import S4Block 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.autograd.set_detect_anomaly(True)

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d
    
    
class S4_Recurrent(nn.Module):
    def __init__(self, input_size=1,  nclasses=10, d_model=128, dropout=0.2):
        super(S4_Recurrent, self).__init__()
        self.encoder = nn.Linear(input_size, d_model)
        self.S4Block = S4Block(d_model=d_model, dropout=dropout, transposed=False)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = dropout_fn(dropout)
        self.decoder = nn.Linear(d_model, nclasses)

    def forward(self, x, state=None):
        
        if state is None:
            state = self.S4Block.default_state(x.shape[0], device=device)
        
        x = self.encoder(x)
        x = x.transpose(-1, -2) # shape [batch, d_model, 28*28]
        z_list = []
        for t in range(x.shape[-1]):
            x_t = x[:, :, t]
            z, state = self.S4Block.step(x_t, state)
            z_list.append(z)
        
        z = torch.stack(z_list, dim=-1)
        z = self.dropout(z)
        # Residual connection
        x = z + x
        x = self.layernorm(x.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(-1, -2)
        x = x.mean(dim=1)
        # class probabilities
        x = self.decoder(x)
        return x
        
        
        
data_dir =  'data/'

train_dataset = torchvision.datasets.MNIST(root = data_dir,
                                           train = True, 
                                           transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]),
                                           download = True)

test_dataset = torchvision.datasets.MNIST(root =  data_dir,
                                          train = False, 
                                          transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]))

loss_f = nn.CrossEntropyLoss()

lr = 1e-3
num_epochs = 10
num_workers = 4
batch_size = 64
sequence_length = 784  # 28x28 pixels

# drop the last batch to ensure each batches have same size
train_loader = DataLoader(dataset = train_dataset, num_workers = num_workers,
                                            batch_size = batch_size,
                                            shuffle = True, drop_last = False)
        
test_loader = DataLoader(dataset = test_dataset, num_workers = num_workers,
                                            batch_size = batch_size,
                                            shuffle = False, drop_last = False)        

model = S4_Recurrent(input_size=1, nclasses=10, d_model = 128).to(device)
print(model)

optimizer = optim.Adam(model.parameters(), lr=lr,  weight_decay = 0.00, betas = (0.9, 0.999))

model.S4Block.setup_step()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.view(x.shape[0], -1, 1).to(device)
        y = y.to(device)
        # pass input through model
        out = model(x)
        # compute loss
        loss = loss_f(out, y)
        total_loss += loss.item()
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(out, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    
     # Evaluation Phase
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for x, y in test_loader:  # Use validation/test loader
            x = x.view(x.shape[0], -1, 1).to(device)
            y = y.to(device)

            # pass input through model
            out = model(x)

            # compute loss
            loss = loss_f(out, y)
            test_loss += loss.item()

            # accuracy
            preds = torch.argmax(out, dim=1)
            test_correct += (preds == y).sum().item()
            test_total += y.size(0)

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * test_correct / test_total
    print(f"Validation Loss: {avg_test_loss:.4f}, Validation Accuracy: {test_accuracy:.2f}%")