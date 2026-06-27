import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from src.models.sequence.modules.s4nd import S4ND
from einops import rearrange, reduce

#from src.models.sequence.backbones.block import SequenceResidualBlock
#from src.models.nn import Normalization
#from src.tasks.decoders import NDDecoder

from models.s4.src.models.sequence.backbones.block import SequenceResidualBlock
from models.s4.src.models.nn import Normalization
from models.s4.src.tasks.decoders import NDDecoder


class S4ND(nn.Module):
    def __init__(self, d_model=256, n_layers=6, dropout=0.1):
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
                    "l_max": [32, 32],
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
        self.decoder = NDDecoder(d_model, d_output=10)

    def forward(self, x):
        # x: [B, 3, 32, 32]
        x = x.permute(0, 2, 3, 1)   # [B, 32, 32, 3]
        x = self.encoder(x)         # [B, 32, 32, d_model]
        x = self.drop(x)
        #x = x.permute(0, 3, 1, 2)   # [B, d_model, 32, 32]

        for layer in self.layers:
            x, state = layer(x)
        x = self.norm(x)
        x = self.decoder(x)
        #print("State shape:", state.shape)
        return x, state


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 1e-3
batch_size = 64 
weight_decay = 0.05 # original wd from paper 0.03 config uses 0.05 
dropout = 0.1
n_layers = 6
d_model = 256 * 2

model = S4ND(d_model=d_model, n_layers=n_layers, dropout = dropout).to(device)
print("Model:", model)
