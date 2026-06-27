import torch
from image_classification.src.mamba import Mamba2DModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn

batch_size = 32 * 4 
epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test_data  = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=4)
test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, num_workers=4)


class SimpleMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Mamba2DModel(
            arch='small', 
            with_cls_token=False, 
            drop_path_rate=0.1, 
            drop_rate=0.1,
            final_norm=True, 
            img_size=32, 
            patch_size=1, 
            in_channels=3, 
            embed_dims=128*6,
            num_layers=12//2,
            out_type='avg_featmap', 
            is_2d=False, 
            has_reverse=True,
            has_transpose=True, 
            fused_add_norm=False, 
            constant_dim=True,
            downsample=(), 
            d_state=16,
            use_v2=False,
        )
        self.head = nn.Linear(128*6, 10)

    def forward(self, x):
        return self.head(self.backbone(x)[0])


model = SimpleMamba().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-4,
    steps_per_epoch=len(train_loader),
    epochs=epochs,
    pct_start=3 / epochs,       # 3-epoch warmup out of 50
    anneal_strategy='cos',
    div_factor=10,              # initial_lr = max_lr / 10 = 1e-4
    final_div_factor=1e3,       # final_lr  = initial_lr / 1e3 ≈ 1e-7
)

print(f"Training on {device}")

for epoch in range(epochs):
    model.train()
    correct, total = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()        # called per batch with OneCycleLR
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc  = 100. * correct / total
    current_lr = scheduler.get_last_lr()[0]

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100. * correct / total
    print(f"Epoch {epoch+1:02d}/{epochs} | LR: {current_lr:.2e} | "
          f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

print("\nDone!")