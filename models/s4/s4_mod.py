import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
from models.s4.s4_block import S4Block 
from models.s4.s4d import S4D as S4DBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#torch.autograd.set_detect_anomaly(True)

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d

class S4_Model(nn.Module):
    def __init__(self, input_size=3, nclasses=10,  # Changed input_size to 3 for RGB
                 d_model=128, dropout=0.2, n_layers=1, bidirecitonal=False,
                 use_state_forwarding=False, ssm='s4d'):
        super(S4_Model, self).__init__()
        self.encoder = nn.Linear(input_size, d_model)
        self.use_state_forwarding = use_state_forwarding
        self.ssm = ssm # ['s4d', 's4']

        self.bidirecitonal = bidirecitonal
        # Stack S4 blocks
        self.s4_blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            
            if self.ssm == 's4d':
                self.s4_blocks.append(
                    S4DBlock(
                        d_model=d_model,
                        dropout=dropout,
                        transposed=True,
                        lr=min(0.001, 0.01),
                    )
                )
                
            elif self.ssm == 's4':
                self.s4_blocks.append(
                    S4Block(
                        d_model=d_model,
                        dropout=dropout,
                        transposed=True,
                        bidirectional=self.bidirecitonal,
                        lr=min(0.001, 0.01),
                    )
                )
            else:
                print("Set ssm flag to 's4d' or 's4'.")
                
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))
        
        self.decoder = nn.Linear(d_model, nclasses)

    def forward(self, x, t, state=None):
        # Initialize state if using state forwarding
        if self.use_state_forwarding and state is None:
            state = self.default_state(x.shape[0])
            
        x = self.encoder(x)  # (B, L, H)
        x = x.transpose(-1, -2)
        # Process through all S4 blocks
        for block, norm, dropout in zip(self.s4_blocks, self.norms, self.dropouts):
            z = x
            
            
            z, state = block(z, state=state if self.use_state_forwarding else None)
            
            z = dropout(z)
            x = z + x  # Residual
            x = norm(x.transpose(-1, -2)).transpose(-1, -2)
            #y = dropout(y)
            #x = y
        
        x = x.transpose(-1, -2)
        # Pool and decode
        x = x.mean(dim=1)
        output = self.decoder(x)
        
        return output, state  # Always return state for consistency

    def setup_step(self):
        for block in self.s4_blocks:
            block.setup_step()

    def default_state(self, *args, **kwargs):
        return self.s4_blocks[0].default_state(*args, **kwargs)