import torch
from image_classification.src.mamba import Mamba2DModel

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Initialize the model directly
model = Mamba2DModel(
    arch='small',              # or provide custom dict with embed_dims, num_layers, etc.
    #img_size=224,
    img_size=20,
    patch_size=1,
    #in_channels=3,
    in_channels=64, 
    #out_type='avg_featmap',
    embed_dims=256,
    out_type='featmap',
    drop_path_rate=0.1,
    drop_rate=0.1,
    with_cls_token=False,
    final_norm=True,
    fused_add_norm=False,
    d_state=16,
    is_2d=False,
    use_v2=False,
    use_nd=False,
    constant_dim=True,
    downsample=(),
    force_a2=False,
    use_mlp=False,
    has_reverse=True,   # Enable bidirectional (forward/backward)
    has_transpose=True, # Enable row/column scanning
)

# Move model to device
model = model.to(device)

# Set to evaluation mode for inference
model.eval()

# Example forward pass
#x = torch.randn(2, 3, 224, 224).to(device)  # Move input to device
x = torch.randn(2, 64, 20, 20).to(device)  # Move input to device

# Run inference
with torch.no_grad():  # Disable gradient computation for inference
    output = model(x)

print(f"Output shape: {output[0].shape}")  # Should be [2, 384] for 'avg_featmap' output type
print(f"Output device: {output[0].device}")
print("✓ Test passed successfully!")