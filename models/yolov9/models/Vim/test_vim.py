import torch
from vim.models_mamba import VisionMamba

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize a small VisionMamba model
model = VisionMamba(
    img_size=224,
    patch_size=16,
    stride=16,
    depth=24,
    embed_dim=192,       # "tiny" size
    d_state=16,
    channels=3,
    num_classes=1000,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    if_cls_token=True,
    use_middle_cls_token=True,
    bimamba_type="v2",
    if_divide_out=True,
    if_abs_pos_embed=True,
    final_pool_type='mean',
)
model.eval()
model.to(device)

# Create a fake batch: (batch_size, channels, height, width)
batch_size = 2
fake_input = torch.randn(batch_size, 3, 224, 224).to(device)

# Forward pass
with torch.no_grad():
    # Standard forward — returns logits of shape (B, num_classes)
    logits = model(fake_input)
    print("Logits shape:", logits.shape)

    # Return intermediate features instead of class logits
    features = model(fake_input, return_features=True)
    print("Features shape:", features.shape)