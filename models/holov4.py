import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.modules.upsampling import Upsample
from torch.utils.checkpoint import checkpoint
from models.rnn import HippoRNN_v2


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
    def __init__(self, in_channels, out_channels, stride = 2, scale=2):
        super(Downsample, self).__init__()

        self.downsample = CnnBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
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
    def __init__(
        self,
        channels,
        image_size,
        hidden_size,
        nclasses,
        maxlength,
        padding=1,
    ):
        super().__init__()
        self.nclasses = nclasses

        # channel for scale 78, 78 :-: 128 
        # channel for scale 36, 36 :-: 256
        # channel for scale 19, 19 :-: 512        
        self.scaled_pred1 = CnnBlock(channels, channels * 2, kernel_size=3, padding=1)
        self.scaled_pred2 = CnnBlock(channels * 2, channels, kernel_size=3, padding=1)
        self.scaled_pred3 = CnnBlock(channels, channels // 2, kernel_size=3, padding=1)
        self.scaled_pred4 = CnnBlock(channels // 2, channels // 4, kernel_size=3, padding=1)


        # prediction scales 1 :-: channels // channel dim, 76, 76 - channels dim denotes upscaled/downscaled channels
        # prediction scales 2 :-: channels // channel dim, 38, 38 -
        # prediction scales 3 :-: channels // channel dim, 19, 19 -

        # downsample scale 1: :-: channels, 76, 76 to channels, 38, 38
        # downsample scale 2: :-: channels, 19, 19 to channels, 19, 19
        # downsample scale 3: :-: channels, 19, 19 to channels, 9.5, 9.5
        #  downsample scale 3: :-: (NOT CLEAN - See  ScaledPredictionScale3 for scale 3)
        
        self.downsample1 = Downsample(in_channels = channels // 4, out_channels=channels // 4)
        self.upsample1 = Upsample(in_channels=channels // 2, out_channels = channels, kernel_size=1)
        # up to scale 5 input size: ((image_size // (channels // 16)) ** 2) * (channels // 8)
        # up to scale 6 input size: ((image_size // (channels // 16)) ** 2) * (channels // 16)

        # output size if we want to predict ouput classes like in yolov4 
        # output_size = ((nclasses + 5) * 3) * ((image_size * 4 // (channels // 4)) ** 2)
        # scale 79 :-: channels = 128 :-: so outputsize = 606480 non-processed prediction
        # scale 38 :-: channels = 256 :-: so  outputsize = 151620
        # scale 19 :-: channels = 512 :-: so  outputsize = 37905
        # scale 79 and 38 are particularly problematic as the linaer layers in RNNs explode
        # instead we use downsampling and skip concatinating connections

        self.scaled_hippo_rnn_pred = HippoRNN_v2(
            input_size = ((image_size // (channels // 16)) ** 2) * (channels // 16),
            hidden_size = hidden_size,
            output_size = ((image_size // (channels // 16)) ** 2) * (channels // 16),
            maxlength = maxlength,
        )

        #self.scaled_pred5 = CnnBlock(channels, channels * 2, kernel_size=3, padding=1)
        self.scaled_pred6 = CnnBlock(channels, (nclasses + 5) * 3, kernel_size=1, padding=0)
        
    def forward(self, x, t, carry):
        out = self.scaled_pred1(x)
        out = self.scaled_pred2(out)
        out = self.scaled_pred3(out)
        out = self.scaled_pred4(out)
        downscale_out = self.downsample1(out)
        
        out = torch.flatten(downscale_out, 1)
        
        out = out.unsqueeze(-1)
        
        # hippo rnn then takes entire flattened sequence channels // 8 * scale * scale
        # last dim denotes chunk of how much to take in each time-step
        out = out.expand(-1, -1, out.shape[1])

        #out, carry = self.scaled_hippo_rnn_pred(xs=out, t=t, carry=carry)
        out, carry = checkpoint(self.scaled_hippo_rnn_pred, out, t, carry, use_reentrant=False)
        # reshape flattened output to batchsize, channel // 4, scale // 2, scale // 2
        out = out.view(x.shape[0], x.shape[1] // 4, x.shape[2] // 2, x.shape[3] // 2)
        
        # concatinate rnn reshape out with downscaled backbone representation
        out = torch.cat((out, downscale_out), dim=1)

        out = self.upsample1(out)
      
        # final out 
        #out = self.scaled_pred5(out)
        out = self.scaled_pred6(out)
       
        # reshape to create predictions
        out = out.reshape( x.shape[0], 3, self.nclasses + 5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)
        #print("Final out shape:", out.shape)
        #print("box coords: preds: [..., 1:3] :", out[..., 1:3].shape)
        #print("box coords: preds: [..., 3:5] :", out[..., 3:5].shape)
        return out, carry
    
    def get_rnn_parameters(self):
        return self.scaled_hippo_rnn_pred.parameters()


class ScaledPredictionScale3(nn.Module):
    def __init__(
        self,
        channels,
        image_size,
        hidden_size,
        nclasses,
        maxlength,
        padding=1,
    ):
        super().__init__()
        self.nclasses = nclasses

        # channel for scale 78, 78 :-: 128 
        # channel for scale 36, 36 :-: 256
        # channel for scale 19, 19 :-: 512        
        self.scaled_pred1 = CnnBlock(channels, channels * 2, kernel_size=3, padding=1)
        self.scaled_pred2 = CnnBlock(channels * 2, channels, kernel_size=3, padding=1)
        self.scaled_pred3 = CnnBlock(channels, channels // 2, kernel_size=3, padding=1)
      
        # prediction scales 1 :-: channels // channel dim, 76, 76 - channels dim denotes upscaled/downscaled channels
        # prediction scales 2 :-: channels // channel dim, 38, 38 -
        # prediction scales 3 :-: channels // channel dim, 19, 19 -

        # downsample scale 1: :-: channels, 76, 76 to channels, 38, 38
        # downsample scale 2: :-: channels, 19, 19 to channels, 19, 19
        # downsample scale 3: :-: channels, 19, 19 to channels, 19, 19 
        # downsample scale 3: :-: dont downsample height, width of tensor :-: 19 not cleanly dividable by 2
        
        self.downsample1 = Downsample(in_channels = channels // 2, out_channels=channels // 2, stride = 1)
        self.downsample1 = Downsample(in_channels = channels // 2, out_channels=channels // 4, stride = 1)
        self.upsample1 = Upsample(in_channels=channels // 2, out_channels = channels, kernel_size=1, scale = 1)
      
        self.scaled_hippo_rnn_pred = HippoRNN_v2(
            input_size = ((image_size // (channels // 16)) ** 2) * (channels // 4),
            hidden_size = hidden_size,
            output_size = ((image_size // (channels // 16)) ** 2) * (channels // 4),
            maxlength = maxlength,
        )

        #self.scaled_pred4 = CnnBlock(channels, channels * 2, kernel_size=3, padding=1)
        self.scaled_pred5 = CnnBlock(channels, (nclasses + 5) * 3, kernel_size=1, padding=0)
        
    def forward(self, x, t, carry):
        out = self.scaled_pred1(x)
        out = self.scaled_pred2(out)
        out = self.scaled_pred3(out)

        downscale_out = self.downsample1(out)
        
        out = torch.flatten(downscale_out, 1)
      
        out = out.unsqueeze(-1)
        
        # last dim denotes chunk of how much model takes each time-step of flattened representation
        out = out.expand(-1, -1, out.shape[1])
        
        #out, carry = self.scaled_hippo_rnn_pred(xs=out, t=t, carry=carry)
        out, carry = checkpoint(self.scaled_hippo_rnn_pred, out, t, carry, use_reentrant=False)
        # reshape flattened output to batchsize, channel // 4, scale , scale
        out = out.view(x.shape[0], x.shape[1] // 4, x.shape[2], x.shape[3])
        
        # concatinate rnn reshape out with downscaled backbone representation
        out = torch.cat((out, downscale_out), dim=1)
        out = self.upsample1(out)
        # final out 
        #out = self.scaled_pred4(out)
        out = self.scaled_pred5(out)
        # reshape to create predictions
        out = out.reshape(x.shape[0], 3, self.nclasses + 5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)
        return out, carry
    
    def get_rnn_parameters(self):
        return self.scaled_hippo_rnn_pred.parameters()


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
            CnnBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            CnnBlock(in_channels=1024, out_channels=512, kernel_size=1),
            CnnBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            CnnBlock(in_channels=1024, out_channels=512, kernel_size=1),
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
class HoloV4_EfficentNet(nn.Module):
    def __init__(
        self,
        *,
        image_size=608,
        hidden_size=128,
        nclasses=30,
        maxlength,
    ):  # , scaled_anchors):
        super(HoloV4_EfficentNet, self).__init__()
        # self.image_size = image_size
        # self.hidden_size = hidden_size
        # self.nclasses = nclasses
        # self.maxlength = maxlength
        # self.scaled_anchors = torch.Tensor(scaled_anchors).float()
        # self.register_buffer('scaled_anchors', self.scaled_anchors)

        self.efficientnetbackbone = nn.Sequential(
            *list(models.efficientnet_v2_s(weights="DEFAULT").children())[:-2]
        )

        self.yolov4coadaptation = nn.Sequential(
            CnnBlock(
                in_channels=1280, out_channels=512, kernel_size=1, padding=0
            ),  # L1 (done)
            CnnBlock(
                in_channels=512, out_channels=1024, kernel_size=3, padding=1
            ),  # L2 (done)
            CnnBlock(
                in_channels=1024, out_channels=512, kernel_size=1, padding=0
            ),  # L3 (done)
        )

        self.yolov4neck = nn.Sequential(
            SpatialPyramidPooling(),
            PathAggregationNet(),
        )

        self.yolov4head = nn.Sequential(
            ScaledPrediction(
                128,
                image_size,
                hidden_size,
                nclasses,
                maxlength,
            ),
            ScaledPrediction(
                256,
                image_size,
                hidden_size,
                nclasses,
                maxlength,
            ),
            # the scale 19, 19 is not divisble by 2 so we treat it special
            ScaledPredictionScale3(
                512,
                image_size,
                hidden_size,
                nclasses,
                maxlength,
            ),
        )

    def forward(self, x, t, carry):
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
        # scale 3 is output of final and passed onto following parts of the architecture
        backbone_scale3 = checkpoint(
            self.efficientnetbackbone[0][6:], backbone_scale2, use_reentrant=False
        )

        x = checkpoint(self.yolov4coadaptation, backbone_scale3, use_reentrant=False)
        ssp_out = self.yolov4neck[0](x)
        
        panet_scale1, panet_scale2, panet_scale3 = checkpoint(self.yolov4neck[1],
            backbone_scale1, backbone_scale2, ssp_out, use_reentrant=False
        )

        sclaed_pred1, carry1 = checkpoint(self.yolov4head[0], panet_scale1, t=t, carry=carry[0], use_reentrant=False)
        sclaed_pred2, carry2 = checkpoint(self.yolov4head[1], panet_scale2, t=t, carry=carry[1], use_reentrant=False)
        sclaed_pred3, carry3 = checkpoint(self.yolov4head[2], panet_scale3, t=t, carry=carry[2], use_reentrant=False)

        return (sclaed_pred3, sclaed_pred2, sclaed_pred1), (carry1, carry2, carry3)
    
    def get_rnn_parameters(self):
        return list(self.yolov4head[0].get_rnn_parameters()) + \
               list(self.yolov4head[1].get_rnn_parameters()) + \
               list(self.yolov4head[2].get_rnn_parameters())

    def get_non_rnn_parameters(self):
        return list(set(self.parameters()) - set(self.get_rnn_parameters()))

"""
if __name__ == "__main__":
    from torchinfo import summary 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 608
    nclasses = 30
    batch_size = 3
    hidden_size = 1024
    
    model = HoloV4_EfficentNet(
        image_size=608, hidden_size=hidden_size, nclasses=nclasses, maxlength=1,
    ).to(device)
    input = {"x": torch.randn((batch_size, 3, img_size, img_size)).to(device), "t":0, "carry":((
                torch.zeros(batch_size, hidden_size).to(device),
                torch.zeros(batch_size, hidden_size).to(device),
            ), (
                torch.zeros(batch_size, hidden_size).to(device),
                torch.zeros(batch_size, hidden_size).to(device),
            ), (
                torch.zeros(batch_size, hidden_size).to(device),
                torch.zeros(batch_size, hidden_size).to(device),
            ))}
    summary(model=model, input_data=input)
"""
"""
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n = 5
    img_size = 608
    nclasses = 30
    x = []
    carry = ((None, None), (None, None), (None, None))
    # create n number of test data and store tensor to list
    for _ in range(n):
        tensor = torch.randn((2, 3, img_size, img_size))
        x.append(tensor)

    model = HoloV4_EfficentNet(
        image_size=608, hidden_size=4, nclasses=nclasses, maxlength=n,
    ).to(device)
    for seq_idx in range(len(x)):
        #print(f"Before Update Time step: {seq_idx} carry 1: {carry[0]}")
        #print(f"Before Update Time step: {seq_idx} carry 2: {carry[1]}")
        #print(f"Before Update Time step: {seq_idx} carry 3: {carry[2]}")
        x_t = x[seq_idx].to(device)
        out, carry = model(x=x_t, t=seq_idx, carry=carry)
        #print(f"After Update Time step: {seq_idx} carry 1: {carry[0]}")
        #print(f"After Update Time step: {seq_idx} carry 2: {carry[1]}")
        #print(f"After Update Time step: {seq_idx} carry 3: {carry[2]}")
    # assert model(x)[0].shape == (2, 3, img_size//32, img_size//32, nclasses + 5)
    # assert model(x)[1].shape == (2, 3, img_size//16, img_size//16, nclasses + 5)
    # assert model(x)[2].shape == (2, 3, img_size//8, img_size//8, nclasses + 5)
    print("Success!")
"""
