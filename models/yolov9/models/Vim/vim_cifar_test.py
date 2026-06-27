import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vim.models_mamba import VisionMamba

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

batch_size = 64
epochs = 25

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_data = torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=transform)
test_data  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=4)
test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, num_workers=4)

model = VisionMamba(
    img_size=32,
    patch_size=4,
    stride=4,
    depth=12*2,
    embed_dim=384,#192,
    d_state=16,
    channels=3,
    num_classes=10,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    if_cls_token=True,
    use_middle_cls_token=True,
    if_bidirectional=False,
    bimamba_type="v2",
    if_divide_out=True,
    if_abs_pos_embed=True,
    final_pool_type='mean',
    drop_path_rate=0.0, 
    drop_rate=0.1,
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay= 1e-8, betas=(0.9, 0.999))

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    steps_per_epoch=len(train_loader),
    epochs=epochs,
    pct_start=3 / epochs,       # 3-epoch warmup out of 50
    anneal_strategy='cos',
    div_factor=10,              # initial_lr = max_lr / 10 = 1e-4
    final_div_factor=1e3,       # final_lr  = initial_lr / 1e3 ≈ 1e-7
)

for epoch in range(1, epochs + 1):
    # Train
    model.train()
    correct, total = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    train_acc = 100. * correct / total

    # Test
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    test_acc = 100. * correct / total

    print(f"Epoch {epoch:02d}/{epochs} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

print("\nDone!")