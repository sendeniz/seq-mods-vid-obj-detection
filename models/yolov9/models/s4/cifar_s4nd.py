import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from src.models.sequence.modules.s4nd import S4ND
from einops import rearrange, reduce
from src.models.sequence.backbones.block import SequenceResidualBlock
from src.models.nn import Normalization
from src.tasks.decoders import NDDecoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class S4ND(nn.Module):
    def __init__(self, d_model=256, n_layers=6, dropout=0.1):
        super().__init__()

        # Encoder
        self.encoder = nn.Linear(3, d_model)
        
        self.drop = nn.Identity()

        # Backbone
        self.layers = nn.ModuleList([
            SequenceResidualBlock(
                d_input=d_model,
                prenorm=True,
                layer={
                    "_name_": "s4nd",
                    "d_state": 64,
                    "channels": 1,
                    "bidirectional": True,
                    "activation": "gelu",
                    "final_act": "glu",
                    "initializer": None,
                    "weight_norm": False,
                    "n_ssm": 1,
                    "dt_min": 0.1,
                    "dt_max": 1.0,
                    "l_max": [32, 32],
                    "dropout": dropout,
                    "tie_dropout": True,
                    "linear": False,
                    "transposed":False,
                },
                residual={"_name_": "residual"},
                norm={"_name_": "layer"},
                pool={"_name_": "avg"},
                dropout=dropout,
                tie_dropout=True,
                transposed=False,
            )
            for _ in range(n_layers)
        ])

        # Final norm
        self.norm = Normalization(d_model, _name_="layer")

        # Decoder
        self.decoder = NDDecoder(d_model, d_output=10)

    def forward(self, x):
        # x: [B, 3, 32, 32]
        x = x.permute(0, 2, 3, 1)   # [B, 32, 32, 3]
        x = self.encoder(x)         # [B, 32, 32, d_model]
        x = self.drop(x)
        #x = x.permute(0, 3, 1, 2)   # [B, d_model, 32, 32]

        for layer in self.layers:
            x, _ = layer(x)
        print("SSM out:", x.shape)
        x = self.norm(x)
        x = self.decoder(x)
        return x


lr = 1e-3
batch_size = 64 
weight_decay = 0.05 # original wd from paper 0.03 config uses 0.05 
dropout = 0.1
n_layers = 6
d_model = 256 * 2

model = S4ND(d_model=d_model, n_layers=n_layers, dropout = dropout).to(device)
print("Model:", model)

# ==== DATA LOADING ====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


# ==== LOSS & OPTIMIZER ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr , weight_decay=weight_decay)

# Scheduler: OneCycleLR with cosine annealing
num_epochs = 100
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-2,
    steps_per_epoch=len(trainloader),
    epochs=num_epochs,
    pct_start=0.3,
    anneal_strategy='cos'
)

# Mixed precision scaler
scaler = torch.cuda.amp.GradScaler()

# ==== TRAIN FUNCTION ====
def train_one_epoch():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # mixed precision forward pass
            outputs = model(inputs)
            #print(outputs.shape)

            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()   # backward pass with scaling
        scaler.step(optimizer)          # optimizer step
        scaler.update()                  # update scaling factor
        scheduler.step()                 # scheduler step per batch

        running_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ==== TEST FUNCTION ====
def evaluate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            running_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ==== MAIN LOOP ====
for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_one_epoch()
    test_loss, test_acc = evaluate()
    print(f"Epoch {epoch:03d} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
          