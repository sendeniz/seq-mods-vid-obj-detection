import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from src.models.sequence.modules.s4nd import S4ND
from einops import rearrange, reduce


from models.s4.src.models.sequence.backbones.block import SequenceResidualBlock
from models.s4.src.models.nn import Normalization
from models.s4.src.tasks.decoders import NDDecoder, Yolo1DDecoder, YoloNDDecoder

class S4ND(nn.Module):
    def __init__(self, d_model=256, d_state=64, nclasses=30, l_max = None, channels=1, n_layers=6, dropout=0.1):
        super().__init__()

        # Encoder
        self.encoder = nn.Linear(3, d_model)
        
        self.drop = nn.Identity()

        # Backbone
        self.layers = nn.ModuleList([
            SequenceResidualBlock(
                d_input=d_model,
                prenorm=True,
                layer={
                    "_name_": "s4nd",
                    "d_state": d_state,
                    "channels": 1,
                    "bidirectional": True,
                    "activation": "gelu",
                    "final_act": "glu",
                    "initializer": None,
                    "weight_norm": False,
                    "n_ssm": 1,
                    "dt_min": 0.1,
                    "dt_max": 1.0,
                    "l_max": l_max,
                    "dropout": dropout,
                    "tie_dropout": True,
                    "linear": False,
                    "transposed":False,
                },
                residual={"_name_": "residual"},
                norm={"_name_": "layer"},
                pool={"_name_": "avg"},
                dropout=dropout,
                tie_dropout=True,
                transposed=False,
            )
            for _ in range(n_layers)
        ])

        # Final norm
        self.norm = Normalization(d_model, _name_="layer")

        # Decoder
        # self.decoder = NDDecoder(d_model, d_output=nclasses)
        self.decoder = Yolo1DDecoder(d_model=d_model, n_anchors=3, out_dim=1)

    def forward(self, x, t, state=None):
        # single final feature vec in yolo
        # x: [B, 3, 19, 19, 1]
        # reshape to B, 3, 19, 19 for S4ND
        #batch_size = x.size(0)
        #channel = x.shape[1]
        #scale = x.shape[2]
        
        x = x.squeeze(-1)
        #print("Input shape:", x.shape)
        
        # x: [B, 3, 32, 32]
        x = x.permute(0, 2, 3, 1)   # [B, 32, 32, 3]
        x = self.encoder(x)         # [B, 32, 32, d_model]
        #print("output decoder shape:", x.shape)
        x = self.drop(x)
        #x = x.permute(0, 3, 1, 2)   # [B, d_model, 32, 32]

        for layer in self.layers:
            # state is none
            x, state = layer(x)
        x = self.norm(x)
        #print("decoder input shape:" , x.shape)
        x = self.decoder(x)
        #print("decoder out shape:" , x.shape)
        
        # remove when using other model than yolo shapes wont match
        
        #x = x.reshape(batch_size, channel, scale, scale)
        #x = x.unsqueeze(-1)

        # return state for consistency
        return x, state


class S4ND_v2(nn.Module):
    def __init__(self, d_model=256, d_state=64, nc=80, l_max=None, n_layers=6, 
                 dropout=0.1, channels=3, reg_max=16, encoder_type="conv1x1"):
        super().__init__()
        
        self.nc = nc
        self.reg_max = reg_max
        self.encoder_type = encoder_type

        # Create encoder based on encoder_type
        if encoder_type == "linear":
            self.encoder = nn.Linear(channels, d_model)
        elif encoder_type == "conv1x1":
            # 1x1 convolution encoder
            # Input shape: (B, H, W, C) -> Output shape: (B, H, W, d_model)
            self.encoder = nn.Conv2d(channels, d_model, kernel_size=1, stride=1, padding=0)
        elif encoder_type == "conv3x3":
            self.encoder = nn.Conv2d(channels, d_model, kernel_size=3, stride=1, padding=1)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}. Use 'linear' or 'conv1x1' or 'conv3x3'")
        
        self.drop = nn.Identity()

        self.layers = nn.ModuleList([
            SequenceResidualBlock(
                d_input=d_model,
                prenorm=True,
                layer={
                    "_name_": "s4nd",
                    "d_state": d_state,
                    "channels": 1,
                    "bidirectional": True,
                    "activation": "gelu",
                    "final_act": "glu",
                    "initializer": None,
                    "weight_norm": False,
                    "n_ssm": 1,
                    "dt_min": 0.1,
                    "dt_max": 1.0,
                    "l_max": l_max,
                    "dropout": dropout,
                    "tie_dropout": True,
                    "linear": False,
                    "transposed": False,
                },
                residual={"_name_": "residual"},
                norm={"_name_": "layer"},
                pool={"_name_": "avg"},
                dropout=dropout,
                tie_dropout=True,
                transposed=False,
            )
            for _ in range(n_layers)
        ])

        self.norm = Normalization(d_model, _name_="layer")
        self.decoder = YoloNDDecoder(d_model=d_model, reg_max=reg_max, nc=nc, pool = 'concat')
        #print('DECODER:', self.decoder)

        self._device_initialized = False
        
    def forward(self, x, t=None, state=None):
        #print("S4ND input shape:", x.shape)
        # Store original dtype for consistency
        input_dtype = x.dtype

        # Permute for processing: (B, C, H, W) -> (B, H, W, C) for linear encoder
        # Or keep as (B, C, H, W) for conv1x1 encoder
        if self.encoder_type == "linear":
            x = x.permute(0, 2, 3, 1)
            x = self.encoder(x)
        else:  # conv1x1 or conv3x3
            x = self.encoder(x)  # (B, d_model, H, W)
            x = x.permute(0, 2, 3, 1)
        
        x = self.drop(x)
        
        for layer in self.layers:
            x, state = layer(x)
        
        x = self.norm(x)
        
        # Ensure consistent dtype before decoder
        x = x.to(input_dtype)
        
        x = self.decoder(x)
        
        return x, state

"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 1e-3
batch_size = 64 
weight_decay = 0.05 # original wd from paper 0.03 config uses 0.05 
dropout = 0.1
n_layers = 6
d_model = 256 * 2

model = S4ND(d_model=d_model, n_layers=n_layers, dropout = dropout).to(device)
print("Model:", model)
"""