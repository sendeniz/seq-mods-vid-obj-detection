import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from src.models.sequence.modules.s4nd import S4ND
from einops import rearrange, reduce
from src.models.sequence.backbones.block import SequenceResidualBlock
from src.models.nn import Normalization
from src.tasks.decoders import NDDecoder
from src.tasks.decoders import Yolo1DDecoder, YoloNDDecoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class S4ND(nn.Module):
    def __init__(self, d_model=256, n_layers=6, dropout=0.1, channels=3):
        super().__init__()

        # Encoder
        self.encoder = nn.Linear(channels, d_model)
        
        self.drop = nn.Identity()

        # Backbone
        self.layers = nn.ModuleList([
            SequenceResidualBlock(
                d_input=d_model,
                prenorm=True,
                layer={
                    "_name_": "s4nd",
                    "d_state": 64,
                    "channels": 1,
                    "bidirectional": True,
                    "activation": "gelu",
                    "final_act": "glu",
                    "initializer": None,
                    "weight_norm": False,
                    "n_ssm": 1,
                    "dt_min": 0.1,
                    "dt_max": 1.0,
                    "l_max": [19, 19],
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
        self.decoder = YoloNDDecoder(d_model)

    def forward(self, x):
        print("X input shape:", x.shape)
        x = x.permute(0, 2, 3, 1)   # [B, 19, 19, 35, 3]
        #x = x.permute(0, 2, 3, 4, 1)   # [B, 19, 19, 35, 3]
        print("X reshape input shape:", x.shape)
        x = self.encoder(x)         # [B, 19, 19, 35, d_model]
        print("X encoder out shape:", x.shape)
        x = self.drop(x)

        for layer in self.layers:
            x, _ = layer(x) # x torch.Size([4, 19, 19, 35, 512])
        print("SSM out shape:", x.shape)
        x = self.norm(x)
        x = self.decoder(x)
        print("Decoder out shape:", x.shape)
        return x


lr = 1e-3
batch_size = 64 
weight_decay = 0.05 # original wd from paper 0.03 config uses 0.05 
dropout = 0.1
n_layers = 6
d_model = 256
channels = 105 # 30 + 5 * 3 , nclasses + nboxes * anchors

model = S4ND(d_model=d_model, n_layers=n_layers, channels = channels, dropout = dropout).to(device)
print("Model:", model)

x_fake = torch.randn(4, 105, 19, 19).to(device)
y_fake = torch.randint(0, 10, (4,)).to(device)

with torch.no_grad():
    out = model(x_fake)
print("Output shape:", out.shape)

