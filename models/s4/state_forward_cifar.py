import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from models.s4.s4_block import S4Block 
from models.s4.s4_model import S4Model as S4
from models.s4.s4d import S4D as S4DBlock

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
    def __init__(self, input_size=3, nclasses=10,  # Changed input_size to 3 for RGB
                 d_model=128, dropout=0.2, n_layers=1, bidirecitonal=False,
                 use_state_forwarding=False, ssm='s4d'):
        super(S4_Model, self).__init__()
        self.encoder = nn.Linear(input_size, d_model)
        self.use_state_forwarding = use_state_forwarding
        self.ssm = ssm # ['s4d', 's4']

        self.bidirecitonal = bidirecitonal
        # Stack S4 blocks
        self.s4_blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            
            if self.ssm == 's4d':
                self.s4_blocks.append(
                    S4DBlock(
                        d_model=d_model,
                        dropout=dropout,
                        transposed=True,
                        lr=min(0.001, 0.01),
                    )
                )
                
            elif self.ssm == 's4':
                self.s4_blocks.append(
                    S4Block(
                        d_model=d_model,
                        dropout=dropout,
                        transposed=True,
                        bidirectional=self.bidirecitonal,
                        lr=min(0.001, 0.01),
                    )
                )
            else:
                print("Set ssm flag to 's4d' or 's4'.")
                
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))
        
        self.decoder = nn.Linear(d_model, nclasses)

    def forward(self, x, state=None):
        # Initialize state if using state forwarding
        if self.use_state_forwarding and state is None:
            state = self.default_state(x.shape[0])
            
        x = self.encoder(x)  # (B, L, H)
        x = x.transpose(-1, -2)
        # Process through all S4 blocks
        for block, norm, dropout in zip(self.s4_blocks, self.norms, self.dropouts):
            z = x
            
            
            z, state = block(z, state=state if self.use_state_forwarding else None)
            
            z = dropout(z)
            x = z + x  # Residual
            x = norm(x.transpose(-1, -2)).transpose(-1, -2)
            #y = dropout(y)
            #x = y
        
        x = x.transpose(-1, -2)
        # Pool and decode
        x = x.mean(dim=1)
        output = self.decoder(x)
        
        return output, state  # Always return state for consistency

    def setup_step(self):
        for block in self.s4_blocks:
            block.setup_step()

    def default_state(self, *args, **kwargs):
        return self.s4_blocks[0].default_state(*args, **kwargs)

def setup_optimizer(model, lr=0.01, weight_decay=0.01, epochs=100):
    # Separate parameters
    all_parameters = list(model.parameters())
    
    # Regular parameters (no _optim attribute)
    params = [p for p in all_parameters if not hasattr(p, "_optim")]
    
    # Create optimizer with regular parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    
    # Add S4 parameters with special settings
    s4_params = [p for p in all_parameters if hasattr(p, "_optim")]
    for p in s4_params:
        optimizer.add_param_group({
            "params": [p],
            "lr": min(0.001, lr),  # Smaller LR for S4 params
            "weight_decay": 0.0  # Typically no weight decay for S4 params
        })
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    return optimizer, scheduler    


data_dir = 'data/'

# CIFAR-10 normalization values
transform_train = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  
])

transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_dataset = torchvision.datasets.CIFAR10(root=data_dir,
                                           train=True, 
                                           transform=transform_train,
                                           download=True)

test_dataset = torchvision.datasets.CIFAR10(root=data_dir,
                                          train=False, 
                                          transform=transform_test)

loss_f = nn.CrossEntropyLoss()

#lr = 1e-3
lr = 0.01
num_epochs = 100
num_workers = 4
batch_size = 32 * 2
sequence_length = 1024  # 32x32 pixels for CIFAR-10

train_loader = DataLoader(dataset=train_dataset, num_workers=num_workers,
                         batch_size=batch_size,
                         shuffle=True, drop_last=False)
        
test_loader = DataLoader(dataset=test_dataset, num_workers=num_workers,
                        batch_size=batch_size,
                        shuffle=False, drop_last=False)        
    
model = S4_Model(input_size=3, d_model=128, nclasses=10, n_layers=2, dropout=0.1,
                 use_state_forwarding=False, bidirecitonal=True, ssm='s4').to(device)

print(model)

#optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999))
optimizer, scheduler = setup_optimizer(model, lr=lr, weight_decay=0.01, epochs=num_epochs)

for epoch in range(num_epochs):
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        # Reshape to (batch_size, sequence_length, input_size)
        # For CIFAR-10: (B, 3, 32, 32) -> (B, 1024, 3)
        #x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, 3)
        #print("x shape:", x.shape)    

        x = x.view(x.shape[0], -1, 3)
        #print("x view shape:", x.shape)    

        out, state = model(x)
        loss = loss_f(out, y)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(out, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    scheduler.step()
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
            #x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, 3)
            x = x.view(x.shape[0], -1, 3)
            
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