import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models.s4.s4_block import S4Block as S4  # Can use full version instead of minimal S4D standalone below
from models.s4.s4d import S4D
#from models.s4.s4_block_v2 import S4Block as S4_v2
from tqdm.auto import tqdm

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d

class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=2,
        dropout=0.2,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                #S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, 1e-6))
                S4(d_model, dropout=dropout, transposed=True, lr=min(0.001, 1e-6))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
                
            # Apply S4 block: we ignore the state input and output
            z, state = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)
            # Residual connection
            x = z + x
            
            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)
        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x #, state

"""
print('==> Building model..')

batch_size = 2
d_input = 1
d_model = 4
d_output = 10 #19*19*3
n_layers = 1
l_max = 28 * 28 #19*19*3
#u = torch.randn(2, l_max, d_input)
#u = torch.rand((batch_size, 3, 19, 19))
u = torch.rand((batch_size, 28, 28))
u = torch.flatten(u, 1)
u = torch.unsqueeze(u, -1)
#print(u.shape)

model = S4Model(
    d_input=d_input,
    d_output=d_output,
    d_model=d_model,
    n_layers=n_layers,
    dropout=0.2,
    prenorm=False,
)
print(model)
"""

"""
#assert model(u).shape == (u.shape[0], model.seq_len_out, model.d_output)
#state = None
state = torch.zeros((d_model, 64))
for t in range(0, 2):
    #u = torch.rand((2, 3, 19, 19))
    #out, state =  model(u, t, state = state)
    #out, state =  model(u, state=state)
    out = model(u)
    
    #print(f"out.shape: {out.shape}")
    #print("-------Predictions Section-------")
    #print(f"out preds: {out[0][0]}")
    #print(f"state: {state}")

    #if state is not None:
        #print(f"out preds: {state[0][0]}")
    #print("out states:", state[0][0])
    #print(f"timestep : {i}")
    #print(f"state : {state}")
"""