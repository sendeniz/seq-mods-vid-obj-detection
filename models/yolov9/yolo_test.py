from models.yolo import DetectionModel
import yaml

# Initialize models
model_m = DetectionModel(
    cfg='models/detect/gelan-m.yaml',
    ch=3,  # input channels
    nc=80   # number of classes
)

model_c = DetectionModel(
    cfg='models/detect/gelan-c.yaml', 
    ch=3,  # input channels
    nc=80   # number of classes
)

print("=" * 60)
print("YOLOv9 PARAMETER COUNT VERIFICATION")
print("=" * 60)

# 1. Basic parameter counts
print("\n1. BASIC PARAMETER COUNTS:")
print("-" * 40)

# YOLOv9-m
total_params_m = sum(p.numel() for p in model_m.parameters())
trainable_params_m = sum(p.numel() for p in model_m.parameters() if p.requires_grad)
non_trainable_params_m = total_params_m - trainable_params_m

print("YOLOv9-m:")
print(f"  Total parameters: {total_params_m:,}")
print(f"  Trainable parameters: {trainable_params_m:,}") 
print(f"  Non-trainable parameters: {non_trainable_params_m:,}")

# YOLOv9-c
total_params_c = sum(p.numel() for p in model_c.parameters())
trainable_params_c = sum(p.numel() for p in model_c.parameters() if p.requires_grad)
non_trainable_params_c = total_params_c - trainable_params_c

print("\nYOLOv9-c:")
print(f"  Total parameters: {total_params_c:,}")
print(f"  Trainable parameters: {trainable_params_c:,}")
print(f"  Non-trainable parameters: {non_trainable_params_c:,}")

# 2. Verify no double-counting
print("\n2. DOUBLE-COUNTING VERIFICATION:")
print("-" * 40)

def verify_parameter_uniqueness(model, model_name):
    param_ids = set()
    total_count = 0
    has_duplicates = False
    
    for param in model.parameters():
        param_id = id(param)
        if param_id in param_ids:
            has_duplicates = True
            print(f"  ⚠️  DUPLICATE FOUND: {param_id}")
        param_ids.add(param_id)
        total_count += param.numel()
    
    if not has_duplicates:
        print(f"  ✅ {model_name}: No double-counting detected")
        print(f"     Unique parameters: {len(param_ids):,}")
        print(f"     Total count: {total_count:,}")
    else:
        print(f"  ❌ {model_name}: Double-counting detected!")
    
    return not has_duplicates

verify_parameter_uniqueness(model_m, "YOLOv9-m")
verify_parameter_uniqueness(model_c, "YOLOv9-c")

# 3. Check YAML configurations
print("\n3. MODEL CONFIGURATIONS:")
print("-" * 40)

try:
    with open('models/detect/yolov9-m.yaml', 'r') as f:
        m_config = yaml.safe_load(f)
    
    with open('models/detect/yolov9-c.yaml', 'r') as f:
        c_config = yaml.safe_load(f)

    print("YOLOv9-m Configuration:")
    print(f"  Depth multiple: {m_config.get('depth_multiple')}")
    print(f"  Width multiple: {m_config.get('width_multiple')}")
    print(f"  Number of classes: {m_config.get('nc', 'Not specified')}")
    
    print("\nYOLOv9-c Configuration:")
    print(f"  Depth multiple: {c_config.get('depth_multiple')}") 
    print(f"  Width multiple: {c_config.get('width_multiple')}")
    print(f"  Number of classes: {c_config.get('nc', 'Not specified')}")
    
except Exception as e:
    print(f"  Could not read YAML files: {e}")

# 4. Backbone vs Head breakdown
print("\n4. BACKBONE vs DETECTION HEAD BREAKDOWN:")
print("-" * 40)

def count_backbone_vs_head(model):
    backbone_params = 0
    head_params = 0
    
    for name, param in model.named_parameters():
        if any(x in name for x in ['cv2', 'cv3', 'dfl', 'detect']):
            head_params += param.numel()
        else:
            backbone_params += param.numel()
    
    return backbone_params, head_params

backbone_m, head_m = count_backbone_vs_head(model_m)
backbone_c, head_c = count_backbone_vs_head(model_c)

print("YOLOv9-m:")
print(f"  Backbone parameters: {backbone_m:,} ({backbone_m/total_params_m*100:.1f}%)")
print(f"  Head parameters: {head_m:,} ({head_m/total_params_m*100:.1f}%)")
print(f"  Total: {backbone_m + head_m:,}")

print("\nYOLOv9-c:")
print(f"  Backbone parameters: {backbone_c:,} ({backbone_c/total_params_c*100:.1f}%)")
print(f"  Head parameters: {head_c:,} ({head_c/total_params_c*100:.1f}%)") 
print(f"  Total: {backbone_c + head_c:,}")

# 5. Model structure info
print("\n5. MODEL STRUCTURE:")
print("-" * 40)

print(f"YOLOv9-m number of modules: {len(list(model_m.modules()))}")
print(f"YOLOv9-c number of modules: {len(list(model_c.modules()))}")

# Check detection head type
print(f"YOLOv9-m detection head type: {type(model_m.model[-1]).__name__}")
print(f"YOLOv9-c detection head type: {type(model_c.model[-1]).__name__}")

# 6. Compare with paper expectations
print("\n6. COMPARISON WITH PAPER:")
print("-" * 40)

print("Expected from paper:")
print("  YOLOv9-m: ~20,000,000 parameters")
print("  YOLOv9-c: ~25,000,000 parameters")

print("\nYour counts:")
print(f"  YOLOv9-m: {total_params_m:,} parameters")
print(f"  YOLOv9-c: {total_params_c:,} parameters")

print("\nDifferences:")
print(f"  YOLOv9-m: {total_params_m - 20000000:,} more than expected")
print(f"  YOLOv9-c: {total_params_c - 25000000:,} more than expected")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)