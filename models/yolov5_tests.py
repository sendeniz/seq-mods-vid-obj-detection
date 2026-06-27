import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.checkpoint import checkpoint
import math

class Conv(nn.Module):
    """Standard convolution with batch norm and SiLU activation - YOLOv5"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, activation=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU() if activation else nn.Identity()

    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck - YOLOv5"""
    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(hidden_channels, out_channels, 3, 1, groups=groups)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        if self.use_add:
            return x + self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x))


class C3(nn.Module):
    """C3 block - YOLOv5"""
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv3 = Conv(2 * hidden_channels, out_channels, 1, 1)
        
        self.m = nn.Sequential(*[
            Bottleneck(hidden_channels, hidden_channels, shortcut, groups, expansion=1.0) 
            for _ in range(n)
        ])

    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        x1 = self.conv1(x)
        x2 = self.m(self.conv2(x))
        x = torch.cat((x1, x2), dim=1)
        return self.conv3(x)


class BottleneckCSP(nn.Module):
    """CSP Bottleneck - original YOLOv5 version"""
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, 1, 1, bias=False)
        self.conv4 = Conv(2 * hidden_channels, out_channels, 1, 1)
        self.bn = nn.BatchNorm2d(2 * hidden_channels)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*[
            Bottleneck(hidden_channels, hidden_channels, shortcut, groups, expansion=1.0) 
            for _ in range(n)
        ])

    def forward(self, x):
        y1 = self.conv3(self.m(self.conv1(x)))
        y2 = self.conv2(x)
        return self.conv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (YOLOv5) version"""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(hidden_channels * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        y = torch.cat([x, y1, y2, y3], dim=1)
        return self.conv2(y)


class Focus(nn.Module):
    """Focus layer - slices input into patches"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super().__init__()
        self.conv = Conv(in_channels * 4, out_channels, kernel_size, stride)

    def forward(self, x):
        # Slice input: x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([
            x[..., ::2, ::2], 
            x[..., 1::2, ::2], 
            x[..., ::2, 1::2], 
            x[..., 1::2, 1::2]
        ], 1))


class CSPDarknet(nn.Module):
    """CSPDarknet backbone for YOLOv5 - matches official implementation"""
    def __init__(self, width_multiplier=0.5, depth_multiplier=0.33):
        super().__init__()
        
        # Apply width multiplier to standard channel sizes [64, 128, 256, 512, 1024]
        def make_divisible(x, divisor=8):
            """Make channels divisible by divisor for better performance"""
            return int(math.ceil(x / divisor) * divisor)
        
        c1 = make_divisible(64 * width_multiplier)    # 32 for YOLOv5s
        c2 = make_divisible(128 * width_multiplier)   # 64
        c3 = make_divisible(256 * width_multiplier)   # 128
        c4 = make_divisible(512 * width_multiplier)   # 256
        c5 = make_divisible(1024 * width_multiplier)  # 512
        
        # Depth: number of bottleneck repeats
        d1 = max(round(3 * depth_multiplier), 1)  # 1 for YOLOv5s
        d2 = max(round(9 * depth_multiplier), 1)  # 3 for YOLOv5s
        
        # Stem - Focus layer (P1/2)
        self.stem = Focus(3, c1, 3)
        
        # Stage 1: P2/4
        self.stage1 = nn.Sequential(
            Conv(c1, c2, 3, 2),
            BottleneckCSP(c2, c2, n=d1)
        )
        
        # Stage 2: P3/8
        self.stage2 = nn.Sequential(
            Conv(c2, c3, 3, 2),
            BottleneckCSP(c3, c3, n=d2)
        )
        
        # Stage 3: P4/16
        self.stage3 = nn.Sequential(
            Conv(c3, c4, 3, 2),
            BottleneckCSP(c4, c4, n=d2)
        )
        
        # Stage 4: P5/32
        self.stage4 = nn.Sequential(
            Conv(c4, c5, 3, 2),
            SPPF(c5, c5),  # SPP layer
            BottleneckCSP(c5, c5, n=d1, shortcut=False)
        )
        
        self.out_channels = [c3, c4, c5]  # Output channels for P3, P4, P5

    def forward(self, x):
        x = self.stem(x)     # P1/2
        x = self.stage1(x)   # P2/4
        p3 = self.stage2(x)  # P3/8 - for detection
        p4 = self.stage3(p3) # P4/16 - for detection
        p5 = self.stage4(p4) # P5/32 - for detection
        
        return p3, p4, p5


class Upsample(nn.Module):
    """Upsample with convolution - YOLOv5"""
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1, 1),
            nn.Upsample(scale_factor=scale, mode='nearest')
        )

    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        return self.upsample(x)


class Downsample(nn.Module):
    """Downsample with convolution - YOLOv5"""
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.downsample = Conv(in_channels, out_channels, 3, scale)

    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        return self.downsample(x)


class Detect(nn.Module):
    """YOLOv5 Detect head - lightweight version matching official implementation"""
    def __init__(self, nclasses=80, anchors=(), channels=()):
        super().__init__()
        self.nclasses = nclasses
        self.no = nclasses + 5  # number of outputs per anchor (x, y, w, h, obj, classes)
        self.nl = len(anchors)  # number of detection layers (3)
        self.na = len(anchors[0]) // 2  # number of anchors per layer (3)
        
        # Create a simple Conv2d head for each detection scale
        # This is MUCH lighter than separate Conv blocks
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in channels
        )

    def forward(self, x):
        """x is a list of 3 feature maps from neck [p3, p4, p5]"""
        outputs = []
        for i, conv in enumerate(self.m):
            # Apply conv to each feature map
            out = conv(x[i])
            bs, _, ny, nx = out.shape  # batch, channels, height, width
            # Reshape: (batch, anchors, grid_y, grid_x, no)
            out = out.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            outputs.append(out)
        
        return outputs


class PANet(nn.Module):
    """Path Aggregation Network - YOLOv5 neck with BottleneckCSP blocks"""
    def __init__(self, p3_channels, p4_channels, p5_channels, 
                 out_p3=128, out_p4=256, out_p5=512, depth_multiplier=0.33):
        super().__init__()
        
        # Depth for neck (number of BottleneckCSP repeats)
        d1 = max(round(3 * depth_multiplier), 1)  # 1 for YOLOv5s, 2 for YOLOv5m
        
        # Top-down path (feature pyramid)
        self.upsample_p5_p4 = Upsample(p5_channels, p4_channels)
        self.c3_p4 = BottleneckCSP(p4_channels + p4_channels, out_p4, n=d1, shortcut=False)
        
        self.upsample_p4_p3 = Upsample(out_p4, out_p3)
        self.c3_p3 = BottleneckCSP(out_p3 + p3_channels, out_p3, n=d1, shortcut=False)
        
        # Bottom-up path
        self.downsample_p3_p4 = Downsample(out_p3, out_p4)
        self.c3_p4_2 = BottleneckCSP(out_p4 + out_p4, out_p4, n=d1, shortcut=False)
        
        self.downsample_p4_p5 = Downsample(out_p4, out_p5)
        self.c3_p5 = BottleneckCSP(out_p5 + p5_channels, out_p5, n=d1, shortcut=False)

    def forward(self, p3, p4, p5):
        # Top-down path
        up_p5 = self.upsample_p5_p4(p5)
        cat_p4 = torch.cat([up_p5, p4], dim=1)
        p4_out = self.c3_p4(cat_p4)
        
        up_p4 = self.upsample_p4_p3(p4_out)
        cat_p3 = torch.cat([up_p4, p3], dim=1)
        p3_out = self.c3_p3(cat_p3)
        
        # Bottom-up path
        down_p3 = self.downsample_p3_p4(p3_out)
        cat_p4 = torch.cat([down_p3, p4_out], dim=1)
        p4_out = self.c3_p4_2(cat_p4)
        
        down_p4 = self.downsample_p4_p5(p4_out)
        cat_p5 = torch.cat([down_p4, p5], dim=1)
        p5_out = self.c3_p5(cat_p5)
        
        return p3_out, p4_out, p5_out


class YoloV5(nn.Module):
    """YOLOv5 with configurable backbone (EfficientNet or CSPDarknet)"""
    def __init__(self, nclasses=80, anchors=None, backbone='efficientnet', 
                 width_multiplier=0.5, depth_multiplier=0.33):
        super().__init__()
        self.nclasses = nclasses
        self.backbone_type = backbone
        self.width_mult = width_multiplier
        self.depth_mult = depth_multiplier
        
        # Default YOLOv5 anchors
        if anchors is None:
            anchors = [
                [10,13, 16,30, 33,23],   # P3/8
                [30,61, 62,45, 59,119],  # P4/16
                [116,90, 156,198, 373,326]  # P5/32
            ]
        self.anchors = anchors
        
        if backbone == 'efficientnet':
            self._init_efficientnet_backbone()
        elif backbone == 'cspdarknet':
            self._init_cspdarknet_backbone()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'efficientnet' or 'cspdarknet'")
        
        # Single lightweight Detect head for all scales
        self.head = Detect(nclasses=nclasses, anchors=anchors, channels=self.neck_out_channels)

    def _init_efficientnet_backbone(self):
        """Initialize EfficientNet backbone"""
        backbone = nn.Sequential(*list(models.efficientnet_v2_s(weights="DEFAULT").children())[:-2])
        
        self.backbone_scale1 = nn.Sequential(backbone[0][:4])   # /8 scale - 64 channels
        self.backbone_scale2 = nn.Sequential(backbone[0][4:6])  # /16 scale - 160 channels
        self.backbone_scale3 = nn.Sequential(backbone[0][6:])   # /32 scale - 1280 channels
        
        # Channel adaptation: 64, 160, 1280 → 128, 256, 512 (match YOLOv5s outputs)
        self.channel_adapt = nn.ModuleDict({
            'p3': Conv(64, 128, 1, 1),
            'p4': Conv(160, 256, 1, 1),
            'p5': Conv(1280, 512, 1, 1),
        })
        
        # Neck with standard channels
        self.neck = PANet(p3_channels=128, p4_channels=256, p5_channels=512,
                         out_p3=128, out_p4=256, out_p5=512, 
                         depth_multiplier=self.depth_mult)
        self.neck_out_channels = [128, 256, 512]

    def _init_cspdarknet_backbone(self):
        """Initialize CSPDarknet backbone"""
        # YOLOv5s: width=0.5, depth=0.33 (~7M params total)
        # YOLOv5m: width=0.75, depth=0.67 (~21M params total)
        self.backbone = CSPDarknet(width_multiplier=self.width_mult, 
                                   depth_multiplier=self.depth_mult)
        
        # Get output channels from backbone
        c3, c4, c5 = self.backbone.out_channels
        
        self.channel_adapt = None  # No adaptation needed for standard configs
        
        # Neck - uses same depth_multiplier as backbone
        self.neck = PANet(p3_channels=c3, p4_channels=c4, p5_channels=c5,
                         out_p3=c3, out_p4=c4, out_p5=c5,
                         depth_multiplier=self.depth_mult)
        self.neck_out_channels = [c3, c4, c5]

    def forward(self, x):
        # Backbone
        if self.backbone_type == 'efficientnet':
            p3 = checkpoint(self.backbone_scale1, x, use_reentrant=False)
            p4 = checkpoint(self.backbone_scale2, p3, use_reentrant=False)
            p5 = checkpoint(self.backbone_scale3, p4, use_reentrant=False)
            
            # Channel adaptation for EfficientNet
            p3 = self.channel_adapt['p3'](p3)
            p4 = self.channel_adapt['p4'](p4)
            p5 = self.channel_adapt['p5'](p5)
        else:  # cspdarknet
            p3, p4, p5 = self.backbone(x)
            # No channel adaptation needed - already correct sizes
        
        # Neck
        p3, p4, p5 = self.neck(p3, p4, p5)
        
        # Head - pass all three feature maps as a list
        outputs = self.head([p3, p4, p5])
        
        # Return in order: large objects (P5), medium (P4), small (P3)
        return outputs[2], outputs[1], outputs[0]


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    img_size = 640
    nclasses = 80
    
    # Test EfficientNet backbone
    print("=" * 60)
    print("Testing YOLOv5 with EfficientNet backbone")
    print("=" * 60)
    model_efficient = YoloV5(nclasses=nclasses, backbone='efficientnet')
    x = torch.randn((2, 3, img_size, img_size))
    out = model_efficient(x)
    print(f"Output shapes: {out[0].shape}, {out[1].shape}, {out[2].shape}")
    total_params, trainable_params = count_parameters(model_efficient)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test YOLOv5s (CSPDarknet with width=0.5, depth=0.33)
    print("\n" + "=" * 60)
    print("Testing YOLOv5s (CSPDarknet backbone)")
    print("width=0.5, depth=0.33 - Target: ~7.2M parameters")
    print("=" * 60)
    model_s = YoloV5(nclasses=nclasses, backbone='cspdarknet', 
                     width_multiplier=0.5, depth_multiplier=0.33)
    out = model_s(x)
    print(f"Output shapes: {out[0].shape}, {out[1].shape}, {out[2].shape}")
    total_params, trainable_params = count_parameters(model_s)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test YOLOv5m (CSPDarknet with width=0.75, depth=0.67)
    print("\n" + "=" * 60)
    print("Testing YOLOv5m (CSPDarknet backbone)")
    print("width=0.75, depth=0.67 - Target: ~21.2M parameters")
    print("=" * 60)
    model_m = YoloV5(nclasses=nclasses, backbone='cspdarknet',
                     width_multiplier=0.75, depth_multiplier=0.67)
    out = model_m(x)
    print(f"Output shapes: {out[0].shape}, {out[1].shape}, {out[2].shape}")
    total_params, trainable_params = count_parameters(model_m)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\nSuccess!")