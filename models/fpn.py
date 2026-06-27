import torch
import torch.nn as nn

# Feature Pyramid Networks

class Conv(nn.Module):
    """Standard YOLOV convolution block (Conv2d + BN + SiLU)"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, 
            autopad(k, p) if p is not None else 0, 
            groups=g, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def autopad(k, p=None):
    """Calculate padding to maintain same spatial dimensions"""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class SPP(nn.Module):
    """YoloV4 Spatial Pyramid Pooling """
    def __init__(self):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.MaxPool2d(5, 1, 5//2),   # 5×5
            nn.MaxPool2d(9, 1, 9//2),   # 9×9
            nn.MaxPool2d(13, 1, 13//2)  # 13×13
        ])

    def forward(self, x):
        return torch.cat([x] + [pool(x) for pool in self.pools], dim=1)

class SPPF(nn.Module):
    """YoloV5 & YoloV6 Faster Spatial Pyramid Pooling"""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)   # Channel reduction
        self.cv2 = Conv(c_ * 4, c2, 1, 1)  # Channel expansion
        self.pool = nn.MaxPool2d(k, 1, k//2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)    # 5×5
        y2 = self.pool(y1)   # 9×9
        y3 = self.pool(y2)   # 13×13
        return self.cv2(torch.cat([x, y1, y2, y3], 1))