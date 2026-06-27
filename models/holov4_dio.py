import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.modules.upsampling import Upsample
from torch.utils.checkpoint import checkpoint
from models.rnn import HippoRNN_v2, UrLstmRNN_v2
from models.rnn import HippoRNN, UrLstmRNN

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
                padding=padding,
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


class ScaledRecurrentPrediction(nn.Module):
    def __init__(
        self,
        channels,
        input_size,
        hidden_size,
        output_size, 
        maxlength,
        nclasses,
        gate="urlstm",
    ):
        super().__init__()
        self.nclasses = nclasses
        self.gate = gate
        self.scaled_pred1 = CnnBlock(channels, channels * 2, kernel_size=3, padding=1)
        self.scaled_pred2 = CnnBlock(channels * 2, (nclasses + 5) * 3, kernel_size=1, padding=0)

        if gate == "hippolstm":

            """
            self.rnn_pred_segment = HippoRNN_v2(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=input_size,
                    maxlength=maxlength,
                )
            
            self.rnn_pred_y = HippoRNN_v2(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=input_size,
                    maxlength=maxlength,
                )
            
            self.rnn_pred_x = HippoRNN_v2(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=input_size,
                    maxlength=maxlength,
                )
            
            self.rnn_pred_h = HippoRNN_v2(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=input_size,
                    maxlength=maxlength,
                )
            
            self.rnn_pred_w = HippoRNN_v2(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=input_size,
                    maxlength=maxlength,
                )
            """
            
            self.rnn_pred_y = HippoRNN(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    maxlength=maxlength,
                )
            
            self.rnn_pred_x = HippoRNN(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    maxlength=maxlength,
                )
            
            self.rnn_pred_h = HippoRNN(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    maxlength=maxlength,
                )
            
            self.rnn_pred_w = HippoRNN(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    maxlength=maxlength,
                )
        
        elif gate == "urlstm":
            self.rnn_pred_segment = UrLstmRNN_v2(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=input_size,
                )
            
            self.rnn_pred_y = UrLstmRNN_v2(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=input_size,
                )
            
            self.rnn_pred_x = UrLstmRNN_v2(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=input_size,
                )
            
            self.rnn_pred_h = UrLstmRNN_v2(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=input_size,

                )
            
            self.rnn_pred_w = UrLstmRNN_v2(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=input_size,
                )
        elif gate == "linear":
            self.rnn_pred_segment_c1 = nn.Linear(input_size, input_size)
            self.rnn_pred_segment_c2 = nn.Linear(input_size, input_size)
            self.rnn_pred_segment_c3 = nn.Linear(input_size, input_size)

            self.rnn_pred_y_c1 = nn.Linear(input_size, input_size)
            self.rnn_pred_y_c2 = nn.Linear(input_size, input_size)
            self.rnn_pred_y_c3 = nn.Linear(input_size, input_size)
            
            self.rnn_pred_x_c1 = nn.Linear(input_size, input_size)
            self.rnn_pred_x_c2 = nn.Linear(input_size, input_size)
            self.rnn_pred_x_c3 = nn.Linear(input_size, input_size)
            
            self.rnn_pred_h_c1 = nn.Linear(input_size, input_size)
            self.rnn_pred_h_c2 = nn.Linear(input_size, input_size)
            self.rnn_pred_h_c3 = nn.Linear(input_size, input_size)

            self.rnn_pred_w_c1 = nn.Linear(input_size, input_size)
            self.rnn_pred_w_c2 = nn.Linear(input_size, input_size)
            self.rnn_pred_w_c3 = nn.Linear(input_size, input_size)


        elif gate == 'none':
            pass
    
    def forward(self, x):   
    #def forward(self, x, t, carry):
 
        #carry_segment = carry[0]
        #carry_x = carry[1]
        #carry_y = carry[2]
        #carry_h = carry[3]
        #carry_w = carry[4]

        out = self.scaled_pred1(x)
        out = self.scaled_pred2(out)
        out = out.reshape(
            x.shape[0], 3, self.nclasses + 5, x.shape[2], x.shape[3]
        ).permute(0, 1, 3, 4, 2)

        #print(f"Segmentation part shape:{out[...,:1].shape}")
        #print(f"xy,hw part shape:{out[...,1:3].shape}")
        #print(f"hw part shape:{out[...,3:5].shape})

        # rnn on bounding box feature maps for:
        # 1. x,y coords :-: out[...,1:3]
        # 3. h, w :-: out[...,3:5]
        # flatten from dim 1 to dont flatten batch size
        #bbox_coord = torch.flatten(out[..., 1:5], 1)
        # unsqueeze last dimension to create information chunk to be processed in rnn
        #bbox_coord = bbox_coord.unsqueeze(-1)
        # expand the last dimension to process entire flattened representation
        #bbox_coord = bbox_coord.expand(-1, -1, bbox_coord.shape[1])

        #bbox_coord, carry = checkpoint(self.rnn_pred, bbox_coord, t, carry, use_reentrant=False)

        # reshape flattened representation of rnn back to original feature size
        #bbox_coord = bbox_coord.reshape(x.shape[0], 3, x.shape[2], x.shape[3], 4)
        # replace rnn finetuned representation in the original feature out
        #out[..., 1:5] = bbox_coord

        #print(out[..., 0:1].shape)
        #print(out[..., 1:2].shape)
        #print(out[..., 2:3].shape)
        #print(out[..., 3:4].shape)
        #print(out[..., 4:5].shape)
        if self.gate == "urlstm" or self.gate == "hippolstm":
            #out_segment = torch.flatten(out[..., 0:1], 1)
            out_bbox_x = torch.flatten(out[..., 1:2], 1)
            out_bbox_y = torch.flatten(out[..., 2:3], 1)
            out_bbox_h = torch.flatten(out[..., 3:4], 1)
            out_bbox_w = torch.flatten(out[..., 4:5], 1)
            #print(f"out_seg shape : {out_segment.shape}")
            #out_segment = out_segment.view(x.shape[0], -1, 1)
            out_bbox_x = out_bbox_x.view(x.shape[0], -1, 1)
            out_bbox_y = out_bbox_y.view(x.shape[0], -1, 1)
            out_bbox_h = out_bbox_h.view(x.shape[0], -1, 1)
            out_bbox_w = out_bbox_w.view(x.shape[0], -1, 1)
            #out_segment = out_segment.unsqueeze(-1)
            #out_bbox_x = out_bbox_x.unsqueeze(-1)
            #out_bbox_y = out_bbox_y.unsqueeze(-1)
            #out_bbox_h = out_bbox_h.unsqueeze(-1)
            #out_bbox_w = out_bbox_w.unsqueeze(-1)

            #out_segment = out_segment.expand(-1, -1, out_segment.shape[1])
            #out_bbox_x = out_bbox_x.expand(-1, -1, out_bbox_x.shape[1])
            #out_bbox_y = out_bbox_y.expand(-1, -1, out_bbox_y.shape[1])
            #out_bbox_h = out_bbox_h.expand(-1, -1, out_bbox_h.shape[1])
            #out_bbox_w = out_bbox_w.expand(-1, -1, out_bbox_w.shape[1])
    
            #out_segment, carry_segment = checkpoint(self.rnn_pred_segment, out_segment, t, carry_segment, use_reentrant=False)
            #out_bbox_x, carry_x = checkpoint(self.rnn_pred_x, out_bbox_x, t, carry_x, use_reentrant=False)
            #out_bbox_y, carry_y = checkpoint(self.rnn_pred_y, out_bbox_y, t, carry_y, use_reentrant=False)
            #out_bbox_h, carry_h = checkpoint(self.rnn_pred_h, out_bbox_h, t, carry_h, use_reentrant=False)
            #out_bbox_w, carry_w = checkpoint(self.rnn_pred_w, out_bbox_w, t, carry_w, use_reentrant=False)
            #out_segment = checkpoint(self.rnn_pred_segment, out_segment,use_reentrant=False)
            #print(f"out seg shape: {out_segment.shape}")
            out_bbox_x = checkpoint(self.rnn_pred_x, out_bbox_x, use_reentrant=False)
            out_bbox_y = checkpoint(self.rnn_pred_y, out_bbox_y, use_reentrant=False)
            out_bbox_h = checkpoint(self.rnn_pred_h, out_bbox_h, use_reentrant=False)
            out_bbox_w = checkpoint(self.rnn_pred_w, out_bbox_w, use_reentrant=False)

            #print("out_segment shape:", out_segment.shape)
            #print("carry segment h_t shape:", carry_segment[0].shape)
            #print("carry segment c_t shape:", carry_segment[1].shape)


            #out_segment = out_segment.reshape(x.shape[0], 3, x.shape[2], x.shape[3], 1)
            out_bbox_x = out_bbox_x.reshape(x.shape[0], 3, x.shape[2], x.shape[3], 1)
            out_bbox_y = out_bbox_y.reshape(x.shape[0], 3, x.shape[2], x.shape[3], 1)
            out_bbox_h = out_bbox_h.reshape(x.shape[0], 3, x.shape[2], x.shape[3], 1)
            out_bbox_w = out_bbox_w.reshape(x.shape[0], 3, x.shape[2], x.shape[3], 1)

            #out[..., 0:1] = out_segment
            out[..., 1:2] = out_bbox_x
            out[..., 2:3] = out_bbox_y
            out[..., 3:4] = out_bbox_h
            out[..., 4:5] = out_bbox_w

        if self.gate == "linear":
            out_segment = out[..., 0:1]
            out_segment_c1 = torch.flatten(out_segment[:, 0, :, :, :], 1)
            out_segment_c2 = torch.flatten(out_segment[:, 1, :, :, :], 1)
            out_segment_c3 = torch.flatten(out_segment[:, 2, :, :, :], 1)
            out_bbox_x = out[..., 1:2]
            out_bbox_x_c1 = torch.flatten(out_bbox_x[:, 0, :, :, :], 1)
            out_bbox_x_c2 = torch.flatten(out_bbox_x[:, 1, :, :, :], 1)
            out_bbox_x_c3 = torch.flatten(out_bbox_x[:, 2, :, :, :], 1)
            out_bbox_y = out[..., 2:3]
            out_bbox_y_c1 = torch.flatten(out_bbox_y[:, 0, :, :, :], 1)
            out_bbox_y_c2 = torch.flatten(out_bbox_y[:, 1, :, :, :], 1)
            out_bbox_y_c3 = torch.flatten(out_bbox_y[:, 2, :, :, :], 1)
            out_bbox_h = out[..., 3:4]
            out_bbox_h_c1 = torch.flatten(out_bbox_h[:, 0, :, :, :], 1)
            out_bbox_h_c2 = torch.flatten(out_bbox_h[:, 1, :, :, :], 1)
            out_bbox_h_c3 = torch.flatten(out_bbox_h[:, 2, :, :, :], 1)
            out_bbox_w = out[..., 4:5]
            out_bbox_w_c1 = torch.flatten(out_bbox_w[:, 0, :, :, :], 1)
            out_bbox_w_c2 = torch.flatten(out_bbox_w[:, 1, :, :, :], 1)
            out_bbox_w_c3 = torch.flatten(out_bbox_w[:, 2, :, :, :], 1)

            #out_segment = torch.flatten(out[..., 0:1], 1)
            #out_bbox_x = torch.flatten(out[..., 1:2], 1)
            #out_bbox_y = torch.flatten(out[..., 2:3], 1)
            #out_bbox_h = torch.flatten(out[..., 3:4], 1)
            #out_bbox_w = torch.flatten(out[..., 4:5], 1)
            #print(f"out_seg shape : {out_segment.shape}")

            #out_segment = out_segment.unsqueeze(-1)
            #out_bbox_x = out_bbox_x.unsqueeze(-1)
            #out_bbox_y = out_bbox_y.unsqueeze(-1)
            #out_bbox_h = out_bbox_h.unsqueeze(-1)
            #out_bbox_w = out_bbox_w.unsqueeze(-1)

            #out_segment = out_segment.expand(-1, -1, out_segment.shape[1])
            #out_bbox_x = out_bbox_x.expand(-1, -1, out_bbox_x.shape[1])
            #out_bbox_y = out_bbox_y.expand(-1, -1, out_bbox_y.shape[1])
            #out_bbox_h = out_bbox_h.expand(-1, -1, out_bbox_h.shape[1])
            #out_bbox_w = out_bbox_w.expand(-1, -1, out_bbox_w.shape[1])
    
            #out_segment = checkpoint(self.rnn_pred_segment, out_segment, use_reentrant=False)
            #out_bbox_x  = checkpoint(self.rnn_pred_x, out_bbox_x, use_reentrant=False)
            #out_bbox_y = checkpoint(self.rnn_pred_y, out_bbox_y, use_reentrant=False)
            #out_bbox_h  = checkpoint(self.rnn_pred_h, out_bbox_h, use_reentrant=False)
            #out_bbox_w  = checkpoint(self.rnn_pred_w, out_bbox_w, use_reentrant=False)

            out_segment_c1 = checkpoint(self.rnn_pred_segment_c1, out_segment_c1, use_reentrant=False)
            out_segment_c2 = checkpoint(self.rnn_pred_segment_c2, out_segment_c2, use_reentrant=False)
            out_segment_c3 = checkpoint(self.rnn_pred_segment_c3, out_segment_c3, use_reentrant=False)
            
            out_bbox_x_c1  = checkpoint(self.rnn_pred_x_c1, out_bbox_x_c1, use_reentrant=False)
            out_bbox_x_c2  = checkpoint(self.rnn_pred_x_c2, out_bbox_x_c2, use_reentrant=False)
            out_bbox_x_c3  = checkpoint(self.rnn_pred_x_c3, out_bbox_x_c3, use_reentrant=False)

            out_bbox_y_c1 = checkpoint(self.rnn_pred_y_c1, out_bbox_y_c1, use_reentrant=False)
            out_bbox_y_c2 = checkpoint(self.rnn_pred_y_c2, out_bbox_y_c2, use_reentrant=False)
            out_bbox_y_c3 = checkpoint(self.rnn_pred_y_c3, out_bbox_y_c3, use_reentrant=False)

            out_bbox_h_c1  = checkpoint(self.rnn_pred_h_c1, out_bbox_h_c1, use_reentrant=False)
            out_bbox_h_c2  = checkpoint(self.rnn_pred_h_c2, out_bbox_h_c2, use_reentrant=False)
            out_bbox_h_c3  = checkpoint(self.rnn_pred_h_c3, out_bbox_h_c3, use_reentrant=False)

            out_bbox_w_c1  = checkpoint(self.rnn_pred_w_c1, out_bbox_w_c1, use_reentrant=False)
            out_bbox_w_c2  = checkpoint(self.rnn_pred_w_c2, out_bbox_w_c2, use_reentrant=False)
            out_bbox_w_c3  = checkpoint(self.rnn_pred_w_c3, out_bbox_w_c3, use_reentrant=False)

            out_segment_c1 = out_segment_c1.view(x.shape[0], x.shape[2], x.shape[3], 1)
            out_segment_c2 = out_segment_c2.view(x.shape[0], x.shape[2], x.shape[3], 1)
            out_segment_c3 = out_segment_c3.view(x.shape[0], x.shape[2], x.shape[3], 1)

            out_bbox_x_c1 = out_bbox_x_c1.view(x.shape[0], x.shape[2], x.shape[3], 1)
            out_bbox_x_c2 = out_bbox_x_c2.view(x.shape[0], x.shape[2], x.shape[3], 1)
            out_bbox_x_c3 = out_bbox_x_c3.view(x.shape[0], x.shape[2], x.shape[3], 1)

            out_bbox_y_c1 = out_bbox_y_c1.view(x.shape[0], x.shape[2], x.shape[3], 1)
            out_bbox_y_c2 = out_bbox_y_c2.view(x.shape[0], x.shape[2], x.shape[3], 1)
            out_bbox_y_c3 = out_bbox_y_c3.view(x.shape[0], x.shape[2], x.shape[3], 1)

            out_bbox_h_c1 = out_bbox_h_c1.view(x.shape[0], x.shape[2], x.shape[3], 1)
            out_bbox_h_c2 = out_bbox_h_c2.view(x.shape[0], x.shape[2], x.shape[3], 1)
            out_bbox_h_c3 = out_bbox_h_c3.view(x.shape[0], x.shape[2], x.shape[3], 1)

            out_bbox_w_c1 = out_bbox_w_c1.view(x.shape[0], x.shape[2], x.shape[3], 1)
            out_bbox_w_c2 = out_bbox_w_c2.view(x.shape[0], x.shape[2], x.shape[3], 1)
            out_bbox_w_c3 = out_bbox_w_c3.view(x.shape[0], x.shape[2], x.shape[3], 1)

            out_segment = torch.stack([out_segment_c1,  out_segment_c2,  out_segment_c3], dim=1)
            out_bbox_x = torch.stack([out_bbox_x_c1,  out_bbox_x_c2,  out_bbox_x_c3], dim=1)
            out_bbox_y = torch.stack([out_bbox_y_c1,  out_bbox_y_c2,  out_bbox_y_c3], dim=1)
            out_bbox_h = torch.stack([out_bbox_h_c1,  out_bbox_h_c2,  out_bbox_h_c3], dim=1)
            out_bbox_w = torch.stack([out_bbox_w_c1,  out_bbox_w_c2,  out_bbox_w_c3], dim=1)
            #print("Out bbox w shape:", out_bbox_w.shape)
            #print("out_segment shape:", out_segment.shape)
            #print("carry segment h_t shape:", carry_segment[0].shape)
            #print("carry segment c_t shape:", carry_segment[1].shape)


            #out_segment = out_segment.reshape(x.shape[0], 3, x.shape[2], x.shape[3], 1)
            #out_bbox_x = out_bbox_x.reshape(x.shape[0], 3, x.shape[2], x.shape[3], 1)
            #out_bbox_y = out_bbox_y.reshape(x.shape[0], 3, x.shape[2], x.shape[3], 1)
            #out_bbox_h = out_bbox_h.reshape(x.shape[0], 3, x.shape[2], x.shape[3], 1)
            #out_bbox_w = out_bbox_w.reshape(x.shape[0], 3, x.shape[2], x.shape[3], 1)

            #out[..., 0:1] = out_segment
            out[..., 1:2] = out_bbox_x
            out[..., 2:3] = out_bbox_y
            out[..., 3:4] = out_bbox_h
            out[..., 4:5] = out_bbox_w
            

        #carry = (carry_segment, carry_x, carry_y, carry_h, carry_w)

        #return out, carry
        return out#, carry


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
        # return downstream_feature3, upstream_feature4, upstream_feature5
        return upstream_feature4, upstream_feature5


# We try to stay true as close as possible to the darknet yolov3.cfg
# we however made changes and do not count [route], [shortcut] or
# [yolo] blocks as seperate layers in the network. These are generally
# not counted as seprate layers by the darknet framework either.
class HoloV4_Dio_EfficentNet(nn.Module):
    def __init__(self, *,  maxlength, nclasses=30, gate ="urlstm"):  # , scaled_anchors):
        super(HoloV4_Dio_EfficentNet, self).__init__()
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
            # scale 38, 38
            # input to rnn:
            # 1. x,y feature map :-: [batchsize, 3, 38, 38, 2]
            # 2. h,w feature map :-: [batchsize, 3, 38, 38, 2]
            # so input size :-: [batchsize, 3, 38, 38, 2] * 2 or [batchsize, 3, 38, 38, 4]
            ScaledRecurrentPrediction(
                channels=256,
                # input_size=3 * 38 * 38 * 2 * 2,
                input_size= 1,
                #hidden_size=4096,
                hidden_size=512*2,
                output_size = 3 *38 * 38 * 1,
                maxlength= (3 *38 * 38 * 1) // 3,
                nclasses=nclasses,
                gate = gate,
            ),
            # scale 19, 19
            # 1. x,y feature map :-: [batchsize, 3, 19, 19, 2]
            # 2. h,w feature map :-: [batchsize, 3, 19, 19, 2]
            # so input size :-: [batchsize, 3, 19, 19, 2] * 2 or [batchsize, 3, 19, 19, 4]
            ScaledRecurrentPrediction(
                channels=512,
                #input_size=3 * 19 * 19 * 2 * 2,
                input_size= 1,
                output_size =  3 *19 * 19 * 1,
                #hidden_size=2048,
                hidden_size=512*2,
                maxlength= (3 *19 * 19 * 1) // 1,
                nclasses=nclasses,
                gate = gate,
            ),
        )

    #def forward(self, x, t, carry):
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

        x = self.yolov4coadaptation(backbone_scale3)
        ssp_out = self.yolov4neck[0](x)

        panet_scale1, panet_scale2 = self.yolov4neck[1](
            backbone_scale1, backbone_scale2, ssp_out
        )

        #sclaled_pred1, carry1 = checkpoint(self.yolov4head[0], panet_scale1, t=t, carry=carry[0], use_reentrant=False)
        #sclaled_pred2, carry2 = checkpoint(self.yolov4head[1], panet_scale2, t=t, carry=carry[1], use_reentrant=False)
        sclaled_pred1 = checkpoint(self.yolov4head[0], panet_scale1, use_reentrant=False)
        sclaled_pred2 = checkpoint(self.yolov4head[1], panet_scale2, use_reentrant=False)
        # print(sclaed_pred3.shape)
        return (sclaled_pred2, sclaled_pred1) #, (carry1, carry2)

"""
if __name__ == "__main__":
    img_size = 608
    nclasses = 30
    model = YoloV4_Dio_EfficentNet(nclasses=nclasses)
    x = torch.randn((2, 3, img_size, img_size))
    out = model(x)
    # print(out.shape)
    print(print(out[0].shape))
    print(print(out[1].shape))
    # assert model(x)[0].shape == (2, 3, img_size//32, img_size//32, nclasses + 5)
    # assert model(x)[1].shape == (2, 3, img_size//16, img_size//16, nclasses + 5)
    # assert model(x)[2].shape == (2, 3, img_size//8, img_size//8, nclasses + 5)
    print("Success!")
"""
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n = 3
    img_size = 608
    nclasses = 30
    x = []
    #carry = ((None, None), (None, None))
    # carry one for cell state one for hidden state for each carry
    
    # create n number of test data and store tensor to list
    for _ in range(n):
        tensor = torch.randn((2, 3, img_size, img_size))
        x.append(tensor)

    model = HoloV4_Dio_EfficentNet(maxlength=n, nclasses=nclasses, gate = "hippolstm").to(device)
    for i in range(2):
        print(f"Iteration: {i}")
        #carry = (((None, None), (None, None), (None, None), (None, None), (None, None)),
        #((None, None), (None, None), (None, None), (None, None), (None, None)),
        #)
        for seq_idx in range(len(x)):
            print(f"Timestep: {seq_idx}")
            #print(f"Before Update Iteration : {i} Time step: {seq_idx} carry 1: {carry[0][0]}")
            #print(f"Before Update Iteration : {i} Time step: {seq_idx} carry 2: {carry[1][0]}")
            #print(f"Before Update Time step: {seq_idx} carry 3: {carry[2]}")
            x_t = x[seq_idx].to(device)
            # out, carry = model(x=x_t, t=seq_idx, carry=carry)
            out = model(x=x_t)
            #print(f"After Update Iteration : {i} Time step: {seq_idx} carry 1: {carry[0][0]}")
            #print(f"After Update Iteration : {i} Time step: {seq_idx} carry 2: {carry[1][0]}")

            #print(f"----After Update Time step: {seq_idx} carry 2: {carry[1]}")
            #print(f"----After Update Time step: {seq_idx} carry 2: {carry[2]}")
        #print(f"After Update Time step: {seq_idx} carry 3: {carry[2]}")
    # assert model(x)[0].shape == (2, 3, img_size//32, img_size//32, nclasses + 5)
    # assert model(x)[1].shape == (2, 3, img_size//16, img_size//16, nclasses + 5)
    # assert model(x)[2].shape == (2, 3, img_size//8, img_size//8, nclasses + 5)
    print("Success!")