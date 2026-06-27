import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.modules.upsampling import Upsample
from torch.utils.checkpoint import checkpoint
from models.s4_mod import S4_Model
from models.s4nd_model import S4ND, S4ND_v2

class CnnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        # Leaky ReLU is activation in yolov3, but what about mish used in yolo4 ?
        # mish is primarily used in the backbone, where the activation of darknet53 is switched out with mish.
        # rest is Leaky Relu
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        return out


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2, kernel_size=1, padding=0):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            CnnBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding = padding,
            ),
            nn.Upsample(scale_factor=scale),
        )

    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        return self.upsample(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(Downsample, self).__init__()

        self.downsample = CnnBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        return self.downsample(x)


class CnnBlockNoBnActiv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        out = self.conv(x)
        return out


class ScaledPrediction(nn.Module):
    def __init__(self, channels, nclasses, padding=1, gate=None):
        super().__init__()
        self.nclasses = nclasses
        self.gate = gate
        #self.scaled_pred = nn.Sequential(
        #    CnnBlock(channels, channels * 2, kernel_size=3, padding=1),
        #    CnnBlock(channels * 2, (nclasses + 5) * 3, kernel_size=1, padding=0),
        #)
        if self.gate == None:
            self.scaled_pred1 = CnnBlock(channels, channels * 2, kernel_size=3, padding=1)
            self.scaled_pred2 = CnnBlock(channels * 2, (nclasses + 5) * 3, kernel_size=1, padding=0)
        
        elif self.gate == 's4nd_v2':
            # refine final logits 
            self.scaled_pred1 = CnnBlock(channels, channels * 2, kernel_size=3, padding=1)
            self.scaled_pred2 = CnnBlock(channels * 2, (nclasses + 5) * 3, kernel_size=1, padding=0)
            self.s4nd_head = S4ND_v2(d_model=256*2, d_state=64, nclasses=self.nclasses, l_max=None, channels= (nclasses + 5) * 3, n_layers=2, dropout=0.1)

        elif self.gate == 's4nd_v3':
            # replace conv 1x1 of prediction head
            self.scaled_pred1 = CnnBlock(channels, channels * 2, kernel_size=3, padding=1)
            self.s4nd_head = S4ND_v2(d_model=256*2, d_state=64, nclasses=self.nclasses, l_max=None, channels=channels * 2, n_layers=2, dropout=0.1)            

        elif self.gate == 's4nd_v4':
            # replace entire pred head of 3x3 conv and 1x1 conv
            self.s4nd_head = S4ND_v2(d_model=256*2, d_state=64, nclasses=self.nclasses, l_max=None, channels=channels, n_layers=2, dropout=0.1)            


    def forward(self, x, t=None, state=None):
        # Satisfy interface t and state set to None
        
        #out = self.scaled_pred[0](x)
        #out = self.scaled_pred[1](out)
        if self.gate == None:
            out = self.scaled_pred1(x)
            out = self.scaled_pred2(out)
            
            out = out.reshape(
                x.shape[0], 3, self.nclasses + 5, x.shape[2], x.shape[3]
            ).permute(0, 1, 3, 4, 2)
            
        elif self.gate == 's4nd_v2':
            # on logits
            out = self.scaled_pred1(x)
            out = self.scaled_pred2(out)
            out = out.reshape(
                x.shape[0], 3, self.nclasses + 5, x.shape[2], x.shape[3]
            ).permute(0, 1, 3, 4, 2)
            out = out.permute(0, 1, 4, 2, 3).reshape(out.shape[0], -1, out.shape[2], out.shape[3])
            out, state = checkpoint(self.s4nd_head, out, t, state, use_reentrant=False)

        
        elif self.gate == 's4nd_v3':
            out = self.scaled_pred1(x)
            out, state = checkpoint(self.s4nd_head, out, t, state, use_reentrant=False)

        elif self.gate == 's4nd_v4':
            out, state = checkpoint(self.s4nd_head, x, t, state, use_reentrant=False)

        return out


class SpatialPyramidPooling(nn.Module):
    def __init__(self):
        super().__init__()

        self.pyramid = nn.Sequential(
            nn.MaxPool2d(5, 1, 5 // 2),
            nn.MaxPool2d(9, 1, 9 // 2),
            nn.MaxPool2d(13, 1, 13 // 2),
        )

    def forward(self, x):
        features = [block(x) for block in self.pyramid]
        features = torch.cat([x] + features, dim=1)
        return features


class PathAggregationNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_transform3 = CnnBlock(
            in_channels=64, out_channels=128, kernel_size=1
        )

        self.feature_transform4 = CnnBlock(
            in_channels=160, out_channels=256, kernel_size=1
        )

        self.resample5_4 = Upsample(in_channels=512, out_channels=256)
        self.resample4_3 = Upsample(in_channels=256, out_channels=128)
        self.resample3_4 = Downsample(in_channels=128, out_channels=256)
        self.resample4_5 = Downsample(in_channels=256, out_channels=512)

        self.downstream_conv5 = nn.Sequential(
            # 2048, 512
            CnnBlock(in_channels=2048, out_channels=512, kernel_size=1),
            # 512, 1024
            CnnBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            # 1024, 512
            CnnBlock(in_channels=1024, out_channels=512, kernel_size=1),
        )

        self.downstream_conv4 = nn.Sequential(
            CnnBlock(in_channels=512, out_channels=256, kernel_size=1),
            CnnBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            CnnBlock(in_channels=512, out_channels=256, kernel_size=1),
            CnnBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            CnnBlock(in_channels=512, out_channels=256, kernel_size=1),
        )
        self.downstream_conv3 = nn.Sequential(
            CnnBlock(in_channels=256, out_channels=128, kernel_size=1),
            CnnBlock(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            CnnBlock(in_channels=256, out_channels=128, kernel_size=1),
            CnnBlock(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            CnnBlock(in_channels=256, out_channels=128, kernel_size=1),
        )

        self.upstream_conv4 = nn.Sequential(
            CnnBlock(in_channels=512, out_channels=256, kernel_size=1),
            CnnBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            CnnBlock(in_channels=512, out_channels=256, kernel_size=1),
            CnnBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            CnnBlock(in_channels=512, out_channels=256, kernel_size=1),
        )
        self.upstream_conv5 = nn.Sequential(
            CnnBlock(in_channels=1024, out_channels=512, kernel_size=1),
            CnnBlock(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            CnnBlock(in_channels=256, out_channels=512, kernel_size=1),
            CnnBlock(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            CnnBlock(in_channels=256, out_channels=512, kernel_size=1),
        )

    def forward(self, scale1, scale2, scale3):
        return checkpoint(self._forward, scale1, scale2, scale3, use_reentrant=False)

    def _forward(self, scale1, scale2, scale3):

        x1 = self.feature_transform3(scale1)
        x2 = self.feature_transform4(scale2)
        x3 = scale3

        downstream_feature5 = self.downstream_conv5(x3)
        route1 = torch.cat((x2, self.resample5_4(downstream_feature5)), dim=1)
        downstream_feature4 = self.downstream_conv4(route1)
        route2 = torch.cat((x1, self.resample4_3(downstream_feature4)), dim=1)
        downstream_feature3 = self.downstream_conv3(route2)

        route3 = torch.cat(
            (self.resample3_4(downstream_feature3), downstream_feature4), dim=1
        )
        upstream_feature4 = self.upstream_conv4(route3)
        route4 = torch.cat(
            (self.resample4_5(upstream_feature4), downstream_feature5), dim=1
        )
        upstream_feature5 = self.upstream_conv5(route4)

        return downstream_feature3, upstream_feature4, upstream_feature5


# We try to stay true as close as possible to the darknet yolov3.cfg
# we however made changes and do not count [route], [shortcut] or
# [yolo] blocks as seperate layers in the network. These are generally
# not counted as seprate layers by the darknet framework either.
class YoloV4_EfficientNet(nn.Module):
    def __init__(self, *, nclasses=80, gate=None):  # , scaled_anchors):
        super(YoloV4_EfficientNet, self).__init__()
        self.nclasses = nclasses
        self.gate = gate
        self.efficientnetbackbone = nn.Sequential(
            *list(models.efficientnet_v2_s(weights="DEFAULT").children())[:-2]
        )

        self.yolov4coadaptation = nn.Sequential(
            CnnBlock(
                in_channels=1280, out_channels=512, kernel_size=1, padding=0
            ),  # L1 (done)
            CnnBlock(
                in_channels=512, out_channels=256, kernel_size=3, padding=1
            ),  # L2 (done)
            CnnBlock(
                in_channels=256, out_channels=512, kernel_size=1, padding=0
            ),  # L3 (done)
        )

        self.yolov4neck = nn.Sequential(
            SpatialPyramidPooling(),
            PathAggregationNet(),
        )

        if gate == None or gate == "s4nd_v2":
            self.yolov4head = nn.Sequential(
                ScaledPrediction(128, nclasses),
                ScaledPrediction(256, nclasses),
                ScaledPrediction(512, nclasses),
            )
        elif gate == 's4nd_v4':
            self.yolov4head = nn.Sequential(
                ScaledPrediction(128, nclasses, gate=self.gate),
                ScaledPrediction(256, nclasses, gate=self.gate),
                ScaledPrediction(512, nclasses, gate=self.gate),
            )
            
    def forward(self, x, t=None, state=None):
        # Satisfy interface t and state set to None

        # the original yolov4 backbone Darknet53 CPS returns features maps at
        # different scales, which are then further processed by the SSP and
        # PaNet. Lastly the predictions are also made at 3 different scales.
        # We adjust the backbone to accomodate an efficientnet backbone. The
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

        x = self.yolov4coadaptation(backbone_scale3)
        ssp_out = self.yolov4neck[0](x)
        #print("SPP out:", ssp_out.shape)

        panet_scale1, panet_scale2, panet_scale3 = self.yolov4neck[1](
            backbone_scale1, backbone_scale2, ssp_out
        )

        #print("panet_scale1:", panet_scale1.shape)
        #print("panet_scale2:", panet_scale2.shape)
        #print("panet_scale3:", panet_scale3.shape)
        
        scaled_pred1 = self.yolov4head[0](panet_scale1, t=t, state=state)
        scaled_pred2 = self.yolov4head[1](panet_scale2, t=t, state=state)
        scaled_pred3 = self.yolov4head[2](panet_scale3, t=t, state=state)
        #print(self.yolov4head[0])

        #print("scaled_pred1:", scaled_pred1.shape)
        #print("scaled_pred2:", scaled_pred2.shape)
        #print("scaled_pred3:", scaled_pred3.shape)
        
        return scaled_pred3, scaled_pred2, scaled_pred1

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

if __name__ == "__main__":
    img_size = 416
    nclasses = 80
    model = YoloV4_EfficientNet(nclasses=nclasses)
    x = torch.randn((2, 3, img_size, img_size))
    out = model(x)
    total_params, trainable_params = count_parameters(model)    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    #assert model(x)[0].shape == (2, 3, img_size//32, img_size//32, nclasses + 5)
    #assert model(x)[1].shape == (2, 3, img_size//16, img_size//16, nclasses + 5)
    #assert model(x)[2].shape == (2, 3, img_size//8, img_size//8, nclasses + 5)
    #print("Success!")
