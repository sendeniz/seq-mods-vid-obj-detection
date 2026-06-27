import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.modules.upsampling import Upsample
from torch.utils.checkpoint import checkpoint


class CnnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, act=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2  # Auto-pad to maintain spatial dims
            
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False  # YOLOv6 uses no bias when BN is present
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()  # YOLOv6 uses SiLU

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class YOLOv6Head(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        # Shared stem (RepVGG-style)
        self.stem = nn.Sequential(
            RepVGGBlock(in_channels, in_channels // 2),
            RepVGGBlock(in_channels // 2, in_channels // 2),
        )
        
        # Decoupled branches
        self.cls_convs = nn.Sequential(
            RepVGGBlock(in_channels // 2, in_channels // 4),
            RepVGGBlock(in_channels // 4, in_channels // 4),
        )
        self.reg_convs = nn.Sequential(
            RepVGGBlock(in_channels // 2, in_channels // 4),
            RepVGGBlock(in_channels // 4, in_channels // 4),
        )
        
        # Prediction layers (order matches YOLOv4 loss expectations!)
        self.obj_pred = nn.Conv2d(in_channels // 4, num_anchors * 1, kernel_size=1)  # obj score (first!)
        self.reg_pred = nn.Conv2d(in_channels // 4, num_anchors * 4, kernel_size=1)  # xywh (second)
        self.cls_pred = nn.Conv2d(in_channels // 4, num_anchors * num_classes, kernel_size=1)  # classes (last)

    def forward(self, x):
        x = self.stem(x)
        cls_feat = self.cls_convs(x)
        reg_feat = self.reg_convs(x)
        
        # Shape: (batch, num_anchors * (5 + num_classes), H, W)
        # NOTE: Order MUST be [obj, x, y, w, h, cls...] for YOLOv4 loss!
        pred = torch.cat([
            self.obj_pred(reg_feat),      # obj score (1 * num_anchors)
            self.reg_pred(reg_feat),       # xywh (4 * num_anchors)
            self.cls_pred(cls_feat),       # class (num_classes * num_anchors)
        ], dim=1)
        
        # Reshape to YOLOv4 format: (batch, num_anchors, H, W, 5 + num_classes)
        return pred.view(
            x.shape[0], 
            self.num_anchors, 
            self.num_classes + 5, 
            x.shape[2], 
            x.shape[3]
        ).permute(0, 1, 3, 4, 2)

class YOLOv6HeadDFL(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors=3, reg_max=16):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.reg_max = reg_max
        
        # Shared stem (RepVGG-style)
        self.stem = nn.Sequential(
            RepVGGBlock(in_channels, in_channels // 2),
            RepVGGBlock(in_channels // 2, in_channels // 2),
        )
        
        # Decoupled branches
        self.cls_convs = nn.Sequential(
            RepVGGBlock(in_channels // 2, in_channels // 4),
            RepVGGBlock(in_channels // 4, in_channels // 4),
        )
        self.reg_convs = nn.Sequential(
            RepVGGBlock(in_channels // 2, in_channels // 4),
            RepVGGBlock(in_channels // 4, in_channels // 4),
        )
        
        # Prediction layers - now with DFL
        self.obj_pred = nn.Conv2d(in_channels // 4, num_anchors * 1, kernel_size=1)
        self.reg_pred = nn.Conv2d(in_channels // 4, num_anchors * 4 * (reg_max + 1), kernel_size=1)  # DFL format
        self.cls_pred = nn.Conv2d(in_channels // 4, num_anchors * num_classes, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        cls_feat = self.cls_convs(x)
        reg_feat = self.reg_convs(x)
        
        # Output format: [obj, reg_dist, cls]
        pred = torch.cat([
            self.obj_pred(reg_feat),  # obj score (1 * num_anchors)
            self.reg_pred(reg_feat),  # reg_dist (4*(reg_max+1) * num_anchors)
            self.cls_pred(cls_feat),   # class (num_classes * num_anchors)
        ], dim=1)
        
        # Reshape to (batch, num_anchors, H, W, num_classes + 1 + 4*(reg_max+1))
        return pred.view(
            x.shape[0], 
            self.num_anchors, 
            self.num_classes + 1 + 4*(self.reg_max + 1), 
            x.shape[2], 
            x.shape[3]
        ).permute(0, 1, 3, 4, 2)

class SPPF(nn.Module):
    """SPPF (Spatial Pyramid Pooling - Fast) from YOLOv6"""
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        hidden_channels = in_channels // 2  # Compress channels first
        self.conv1 = CnnBlock(in_channels, hidden_channels, kernel_size=1, padding=0)
        self.conv2 = CnnBlock(hidden_channels * 4, out_channels, kernel_size=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))

# RepPanNet modules
class ConvBNReLU(nn.Module):
    '''Conv + BN + ReLU'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2  # Same padding
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, 
            padding, groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class RepVGGBlock(nn.Module):
    '''RepVGG Block from RepVGG paper'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, 
            padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        
        # Identity branch
        if in_channels == out_channels and stride == 1:
            self.identity = nn.BatchNorm2d(out_channels)
        else:
            self.identity = None

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        
        if self.identity is not None:
            out += self.identity(x)
            
        return self.act(out)

class BottleRep(nn.Module):
    '''
    Bottle Rep Block
    '''
    def __init__(self, in_channels, out_channels, basic_block=RepVGGBlock, weight=False):
        super().__init__()
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        self.weight = weight
        if weight:
            self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if self.weight:
            return self.conv2(self.conv1(x)) * self.alpha
        else:
            return self.conv2(self.conv1(x))

class RepBlock(nn.Module):
    '''
    RepBlock is a stage block with rep-style basic block
    '''
    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock, basic_block=RepVGGBlock):
        super().__init__()
        self.conv1 = block(in_channels, out_channels)
        self.block = nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None
        if block == BottleRep:
            self.conv1 = BottleRep(in_channels, out_channels, basic_block=basic_block, weight=True)
            n = n // 2
            self.block = nn.Sequential(*(BottleRep(out_channels, out_channels, basic_block=basic_block, weight=True) for _ in range(n - 1))) if n > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x

class Transpose(nn.Module):
    '''Upsample with transpose conv'''
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride,
            padding=0, output_padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class RepPANNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define your channel dimensions
        self.channels_list = {
            'scale1': 64,    # Your scale1 input channels
            'scale2': 160,   # Your scale2 input channels
            'scale3': 512,   # Your scale3 input channels
            'p4_out': 256,   # Output channels after first fusion
            'p3_out': 128,   # Output channels after second fusion
            'n3_out': 256,   # Output channels after first bottom-up fusion
            'n4_out': 512    # Output channels after second bottom-up fusion
        }
        
        # Initial feature transforms
        self.feature_transform3 = ConvBNReLU(
            in_channels=self.channels_list['scale1'],
            out_channels=self.channels_list['scale1'],
            kernel_size=1
        )

        self.feature_transform4 = ConvBNReLU(
            in_channels=self.channels_list['scale2'],
            out_channels=self.channels_list['scale2'],
            kernel_size=1
        )

        # Top-down path
        self.Rep_p4 = RepBlock(
            in_channels=self.channels_list['scale2'] + self.channels_list['scale3'],
            out_channels=self.channels_list['p4_out'],
            n=3,
            block=RepVGGBlock
        )

        self.Rep_p3 = RepBlock(
            in_channels=self.channels_list['scale1'] + self.channels_list['p4_out'],
            out_channels=self.channels_list['p3_out'],
            n=3,
            block=RepVGGBlock
        )

        # Bottom-up path
        self.Rep_n3 = RepBlock(
            in_channels=self.channels_list['p3_out'] + self.channels_list['p4_out'],
            out_channels=self.channels_list['n3_out'],
            n=3,
            block=RepVGGBlock
        )

        self.Rep_n4 = RepBlock(
            in_channels=self.channels_list['p4_out'] + self.channels_list['scale3'],
            out_channels=self.channels_list['n4_out'],
            n=3,
            block=RepVGGBlock
        )

        # Reduce and upsample/downsample layers
        self.reduce_layer0 = ConvBNReLU(
            in_channels=self.channels_list['scale3'],
            out_channels=self.channels_list['scale3'],
            kernel_size=1
        )

        self.upsample0 = Transpose(
            in_channels=self.channels_list['scale3'],
            out_channels=self.channels_list['scale3']
        )

        self.reduce_layer1 = ConvBNReLU(
            in_channels=self.channels_list['p4_out'],
            out_channels=self.channels_list['p4_out'],
            kernel_size=1
        )

        self.upsample1 = Transpose(
            in_channels=self.channels_list['p4_out'],
            out_channels=self.channels_list['p4_out']
        )

        self.downsample2 = ConvBNReLU(
            in_channels=self.channels_list['p3_out'],
            out_channels=self.channels_list['p3_out'],
            kernel_size=3,
            stride=2
        )

        self.downsample1 = ConvBNReLU(
            in_channels=self.channels_list['n3_out'],
            out_channels=self.channels_list['n3_out'],
            kernel_size=3,
            stride=2
        )

    def forward(self, scale1, scale2, scale3):
        # Apply initial feature transforms
        #print("scale1 shape:", scale1.shape)
        x2 = self.feature_transform3(scale1)
        #print("x2 shape:", x2.shape)
        #print("scale2 shape:", scale2.shape)

        x1 = self.feature_transform4(scale2)
        #print("x1 shape:", x1.shape)
        #print("scale3 shape:", scale3.shape)

        x0 = scale3
        #print("x0 shape:", x0.shape)

        # Top-down path
        fpn_out0 = self.reduce_layer0(x0)
        #print("fpn_out0 shape:", fpn_out0.shape)
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
        pan_out2 = self.Rep_p3(f_concat_layer1)

        # Bottom-up path
        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        return pan_out2, pan_out1, pan_out0

# We try to stay true as close as possible to the darknet yolov3.cfg
# we however made changes and do not count [route], [shortcut] or
# [yolo] blocks as seperate layers in the network. These are generally
# not counted as seprate layers by the darknet framework either.
class YoloV6_EfficentNet(nn.Module):
    def __init__(self, *, nclasses=80):  # , scaled_anchors):
        super(YoloV6_EfficentNet, self).__init__()
        self.nclasses = nclasses

        self.efficientnetbackbone = nn.Sequential(
            *list(models.efficientnet_v2_s(weights="DEFAULT").children())[:-2]
        )

        self.yolov4coadaptation = nn.Sequential(
            CnnBlock(
                in_channels=1280, out_channels=512, kernel_size=1, padding=0
            ),  # L1 (done)
            CnnBlock(
                # in_channels=512, out_channels=1024, kernel_size=3, padding=1
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),  # L2 (done)
            CnnBlock(
                in_channels=512, out_channels=512, kernel_size=1, padding=0
            ),  # L3 (done)
        )

        self.yolov4neck = nn.Sequential(
             SPPF(in_channels=1280, out_channels=512),
             RepPANNet(),
        )

        self.yolov4head = nn.Sequential(
            YOLOv6HeadDFL(128, nclasses),
            YOLOv6HeadDFL(256, nclasses),
            YOLOv6HeadDFL(512, nclasses),
        )

    def forward(self, x):
        # the original yolov4 backbone Darknet53 CPS returns features maps at
        # different scales, which are then further processed by the SSP and
        # PaNet. Lastly the predictions are also made at 3 different scales.
        # We adjust the backbone to accomodate an efficentnet backbone. The
        # principle however stays the same.
        backbone_scale1 = checkpoint(
            self.efficientnetbackbone[0][:4], x, use_reentrant=False
        )
        backbone_scale2 = checkpoint(
            self.efficientnetbackbone[0][4:6], backbone_scale1, use_reentrant=False
        )
        # scale 3 is out of final passed onto following parts of the architecture
        backbone_scale3 = checkpoint(
            self.efficientnetbackbone[0][6:], backbone_scale2, use_reentrant=False
        )

        #x = self.yolov4coadaptation(backbone_scale3)
        #print("Input to SPP:", x.shape)
        #ssp_out = self.yolov4neck[0](x)
        ssp_out = self.yolov4neck[0](backbone_scale3)
        #print("SPP out:", ssp_out.shape)
        panet_scale1, panet_scale2, panet_scale3 = self.yolov4neck[1](
            backbone_scale1, backbone_scale2, ssp_out
        )

        sclaed_pred1 = self.yolov4head[0](panet_scale1)
        sclaed_pred2 = self.yolov4head[1](panet_scale2)
        sclaed_pred3 = self.yolov4head[2](panet_scale3)

        return sclaed_pred3, sclaed_pred2, sclaed_pred1
"""
if __name__ == "__main__":
    img_size = 416
    nclasses = 80
    model = YoloV6_EfficentNet(nclasses=nclasses)
    x = torch.randn((2, 3, img_size, img_size))
    out = model(x)
    assert model(x)[0].shape == (2, 3, img_size//32, img_size//32, nclasses + 5)
    assert model(x)[1].shape == (2, 3, img_size//16, img_size//16, nclasses + 5)
    assert model(x)[2].shape == (2, 3, img_size//8, img_size//8, nclasses + 5)
    print("Success!")
"""