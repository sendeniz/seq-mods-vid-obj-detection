import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.modules.upsampling import Upsample
from torch.utils.checkpoint import checkpoint
from models.s4_mod import S4_Model
from models.s4nd_model import S4ND, S4ND_v2

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
        # Note SiLU can cause instability and NaN values
        self.activation =  nn.SiLU() if activation else nn.Identity() # nn.LeakyReLU(0.1) # nn.SiLU() if activation else nn.Identity()

    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)
    # # no checkpointpointing to stabilise gradients
    #def forward(self, x):
    #        return self._forward(x)

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
        #return self._forward(x)

    def _forward(self, x):
        x1 = self.conv1(x)
        x2 = self.m(self.conv2(x))
        x = torch.cat((x1, x2), dim=1)
        return self.conv3(x)


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
    """Downsample with convolution - YOLOv5 
    """
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.downsample = Conv(in_channels, out_channels, 3, scale)

    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        return self.downsample(x)


class ScaledPrediction(nn.Module):
    """Detection head - YOLOv5"""
    def __init__(self, channels, nclasses, anchors=3, gate=None):
        super().__init__()
        self.nclasses = nclasses
        self.gate = gate
        self.anchors = anchors
        
        if self.gate == None:
            self.head = nn.Sequential(
                Conv(channels, anchors * (nclasses + 5), 1, 1, activation=False)
            )
            
        elif self.gate == 's4nd_v2':
            # S4ND as refinement
            self.conv2 = Conv(channels, anchors * (nclasses + 5), 1, 1, activation=False)
            #print(self.conv2)
            self.s4nd_head = S4ND_v2(d_model=channels, d_state=64, nclasses=nclasses, 
                                   l_max=None, channels=anchors*(nclasses+5), n_layers=2, dropout=0.1)
            #print(self.s4nd_head)
            
        elif self.gate == 's4nd_v3':
            # THIS SCENARIO DOES NOT EXIST IN YOLOV5
            # YOLOV4 used 3x3 conv then 1x1 conv for prediction or scaled prediction head
            # YoloV5 made this more efficient removed the 3x3 conv
            # replace conv 1x1 of prediction head
            #self.conv1 = Conv(channels, channels * 2, 3, 1)
            #self.s4nd_head = S4ND_v2(d_model=channels*2, d_state=64, nclasses=nclasses,
            #                       l_max=None, channels=channels*2, n_layers=4, dropout=0.1)
            raise NotImplementedError(
                "S4ND_v3 scenario does not exist in YOLOv5. " 
                "This implementation was designed for YOLOv4."
                "Please use YoloV4 or please use a different gate type or modify the architecutre."
                )
        
        elif self.gate == 's4nd_v4':
            # replace entire conv 1x1 of prediction head of yoloV5
            self.s4nd_head = S4ND_v2(d_model=channels, d_state=64, nclasses=nclasses,
                                   l_max=None, channels=channels, n_layers=6, dropout=0.1)
            #print("self.s4nd_head:",  self.s4nd_head)

    def forward(self, x, t=None, state=None):
        
        if self.gate == None:
            out = self.head(x)
            # Reshape to YOLO format: (batch, anchors, height, width, nclasses+5)
            out = out.view(x.shape[0], self.anchors, self.nclasses + 5, x.shape[2], x.shape[3])
            out = out.permute(0, 1, 3, 4, 2)
            
        elif self.gate == 's4nd_v2':
            #out = self.conv1(x)
            out = self.conv2(x)
            out = out.view(x.shape[0], self.anchors, self.nclasses + 5, x.shape[2], x.shape[3])
            out = out.permute(0, 1, 3, 4, 2)
            out = out.permute(0, 1, 4, 2, 3).reshape(out.shape[0], -1, out.shape[2], out.shape[3])
            out, state = checkpoint(self.s4nd_head, out, t, state, use_reentrant=False)
            
        elif self.gate == 's4nd_v3':
            out = self.conv1(x)
            out, state = checkpoint(self.s4nd_head, out, t, state, use_reentrant=False)
            
        elif self.gate == 's4nd_v4':
            out, state = checkpoint(self.s4nd_head, x, t, state, use_reentrant=False)

        return out


class YoloV5_EfficientNet(nn.Module):
    def __init__(self, nclasses=80, gate=None, anchors=3):
        super(YoloV5_EfficientNet, self).__init__()
        self.nclasses = nclasses
        self.gate = gate
        
        # EfficientNet backbone - remove last 2 layers (global pooling + classifier)
        # efficientnet = models.efficientnet_v2_s(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(models.efficientnet_v2_s(weights="DEFAULT").children())[:-2])
        
        # Extract feature maps at different scales
        # EfficientNet V2-S feature extraction points:
        self.backbone_scale1 = nn.Sequential(
            self.backbone[0][:4]   # features 0-3: /8 scale
        )
        self.backbone_scale2 = nn.Sequential(
            self.backbone[0][4:6]  # features 4-5: /16 scale
        )
        self.backbone_scale3 = nn.Sequential(
            self.backbone[0][6:]   # features 6-7: /32 scale → 1280 channels
        )
        
        # EfficientNet outputs: 64, 160, 1280 channels → adapt to YOLOv5: 128, 256, 512
        self.channel_adapt = nn.ModuleDict({
            'p3': Conv(64, 128, 1, 1),      # 64 → 128 channels
            'p4': Conv(160, 256, 1, 1),     # 160 → 256 channels  
            'p5': Conv(1280, 512, 1, 1),    # 1280 → 512 channels
        })
        
        # Neck - PANet with C3 blocks (YOLOv5)
        self.neck = PANet()
        
        # Head
        if gate is None or gate == "s4nd_v2":
            self.head = nn.ModuleList([
                ScaledPrediction(128, nclasses, anchors, gate=gate),   # p3 - large objects
                ScaledPrediction(256, nclasses, anchors, gate=gate),   # p4 - medium objects
                ScaledPrediction(512, nclasses, anchors, gate=gate)    # p5 - small objects
            ])
        else:
            self.head = nn.ModuleList([
                ScaledPrediction(128, nclasses, anchors, gate=gate),
                ScaledPrediction(256, nclasses, anchors, gate=gate),
                ScaledPrediction(512, nclasses, anchors, gate=gate)
            ])

    def forward(self, x, t=None, state=None):
        # Backbone 
        p3 = checkpoint(self.backbone_scale1, x, use_reentrant=False)      # /8 scale, 48 channels
        p4 = checkpoint(self.backbone_scale2, p3, use_reentrant=False)       # /16 scale, 128 channels  
        p5 = checkpoint(self.backbone_scale3, p4, use_reentrant=False)     # /32 scale, 1280 channels
        
        # Channel adaptation
        p3 = self.channel_adapt['p3'](p3)  # 64 → 128 channels
        p4 = self.channel_adapt['p4'](p4)  # 160 → 256 channels
        p5 = self.channel_adapt['p5'](p5)  # 1280 → 512 channels
        
        # Neck
        p3, p4, p5 = self.neck(p3, p4, p5)
        
        # Head
        pred_small = self.head[0](p3, t, state)   # Large objects (p3)
        pred_medium = self.head[1](p4, t, state)  # Medium objects (p4)  
        pred_large = self.head[2](p5, t, state)   # Small objects (p5)
        
        return pred_large, pred_medium, pred_small


class PANet(nn.Module):
    """Path Aggregation Network - YOLOv5 with C3 blocks"""
    def __init__(self):
        super().__init__()
        
        # Top-down path (feature pyramid)
        self.upsample_p5_p4 = Upsample(512, 256)
        self.c3_p4 = C3(512, 256)  # concat p4 (256) + up_p5 (256) = 512 → 256
        
        self.upsample_p4_p3 = Upsample(256, 128) 
        self.c3_p3 = C3(256, 128)  # concat p3 (128) + up_p4 (128) = 256 → 128
        
        # Bottom-up path
        self.downsample_p3_p4 = Downsample(128, 256)
        self.c3_p4_2 = C3(512, 256)  # concat p4 (256) + down_p3 (256) = 512 → 256
        
        self.downsample_p4_p5 = Downsample(256, 512)
        self.c3_p5 = C3(1024, 512)  # concat p5 (512) + down_p4 (512) = 1024 → 512
        
        # SPPF on p5 path
        self.sppf = SPPF(512, 512)

    def forward(self, p3, p4, p5):
        # Apply SPPF to p5 first
        p5 = self.sppf(p5)
        
        # Top-down path
        up_p5 = self.upsample_p5_p4(p5)
        cat_p4 = torch.cat([up_p5, p4], dim=1)  # 256 + 256 = 512
        p4 = self.c3_p4(cat_p4)                 # 512 → 256
        
        up_p4 = self.upsample_p4_p3(p4)
        cat_p3 = torch.cat([up_p4, p3], dim=1)  # 128 + 128 = 256
        p3 = self.c3_p3(cat_p3)                 # 256 → 128
        
        # Bottom-up path  
        down_p3 = self.downsample_p3_p4(p3)     # 128 → 256
        cat_p4 = torch.cat([down_p3, p4], dim=1) # 256 + 256 = 512
        p4 = self.c3_p4_2(cat_p4)               # 512 → 256
        
        down_p4 = self.downsample_p4_p5(p4)     # 256 → 512
        cat_p5 = torch.cat([down_p4, p5], dim=1) # 512 + 512 = 1024
        p5 = self.c3_p5(cat_p5)                 # 1024 → 512
        
        return p3, p4, p5

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

"""
if __name__ == "__main__":
    img_size = 640
    nclasses = 80
    model = YoloV5_EfficientNet(nclasses=nclasses, gate="s4nd_v2")
    x = torch.randn((2, 3, img_size, img_size))
    out = model(x)
    print(model(x)[0].shape)
    print(model(x)[1].shape)
    print(model(x)[2].shape)
    total_params, trainable_params = count_parameters(model)    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    #assert model(x)[0].shape == (2, 3, img_size//32, img_size//32, nclasses + 5)
    #assert model(x)[1].shape == (2, 3, img_size//16, img_size//16, nclasses + 5)
    #assert model(x)[2].shape == (2, 3, img_size//8, img_size//8, nclasses + 5)
    #print("Success!")
"""