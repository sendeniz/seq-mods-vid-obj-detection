import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from models.s4.s4_block import S4Block 
from models.s4.s4_model import S4Model as S4

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

class S4_Model(nn.Module):
    def __init__(self, input_size=1, nclasses=10, 
                 d_model=128, dropout=0.2, n_layers=1,
                 use_state_forwarding=False):
        super(S4_Model, self).__init__()
        self.encoder = nn.Linear(input_size, d_model)
        self.use_state_forwarding = use_state_forwarding
        
        # Stack S4 blocks
        self.s4_blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_blocks.append(
                S4Block(
                    d_model=d_model,
                    dropout=dropout,
                    transposed=False,
                    bidirectional=False,
                )
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))
        
        self.decoder = nn.Linear(d_model, nclasses)

    def forward(self, x, state=None):
        # Initialize state if using state forwarding
        if self.use_state_forwarding and state is None:
            state = self.default_state(x.shape[0])
            
        x = self.encoder(x)  # (B, L, H)
        
        # Process through all S4 blocks
        for block, norm, dropout in zip(self.s4_blocks, self.norms, self.dropouts):
            y, state = block(x, state=state if self.use_state_forwarding else None)
            y = y + x  # Residual
            y = norm(y)
            y = dropout(y)
            x = y
        
        # Pool and decode
        y = x.mean(dim=1)
        output = self.decoder(y)
        
        return output, state  # Always return state for consistency

    def setup_step(self):
        for block in self.s4_blocks:
            block.setup_step()

    def default_state(self, *args, **kwargs):
        return self.s4_blocks[0].default_state(*args, **kwargs)
    

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
    
model = S4_Model(input_size=1, d_model=512, nclasses=10, n_layers=1,
                 use_state_forwarding=False).to(device)
#model = S4(d_input = 1, d_output=10, d_model=128, n_layers=1, dropout=0.2,).to(device)
print(model)

optimizer = optim.Adam(model.parameters(), lr=lr,  weight_decay = 0.00, betas = (0.9, 0.999))



for epoch in range(num_epochs):
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        # last index is input size
        x = x.view(x.shape[0], -1, 1)

        out, state = model(x)
        #out = model(x)
        loss = loss_f(out, y)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(out, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    avg_train_loss = total_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    
    # Testing phase
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.shape[0], -1, 1)
            # out, _ = model(x)
            out, state = model(x)
            loss = loss_f(out, y)
            test_loss += loss.item()
            
            preds = torch.argmax(out, dim=1)
            test_correct += (preds == y).sum().item()
            test_total += y.size(0)
    
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * test_correct / test_total
    
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}%, "
          f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.4f}%")

"""
# Example input (batch=2, seq_len=10, input_size=1)
x = torch.randn(2, 10, 1).to(device)

# Forward pass without state (training)
output, _ = model(x)
print(f"output:{output.shape}")
print(f"_: {_}" )
# Forward pass with state (e.g., for chunked processing)
state = model.default_state(2)  # batch=2
output, new_state = model(x, state=state)
print(f"output:{output.shape}")
print(f"new_state:{new_state.shape}")
"""