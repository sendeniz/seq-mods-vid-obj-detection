import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.nn as nn
from models.s4.s4_block import S4Block 
from models.s4.s4d import S4D as S4DBlock
#from models.s4.s4nd import S4ND as S4DNBlock

    
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
                 d_model=128, d_state=64, dropout=0.2, n_layers=1, bidirecitonal=False,
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
        
        self.bn = nn.BatchNorm2d(3)
        self.activation = nn.LeakyReLU(0.1)

        
        for _ in range(n_layers):
            
            if self.ssm == 's4d':
                self.s4_blocks.append(
                    S4DBlock(
                        d_model=d_model,
                        d_state=d_state,
                        dropout=dropout,
                        transposed=True,
                        #lr=min(0.001, 0.01),
                        #lr=min(0.0001, 0.001),
                        #lr=min(0.0005,0.001),
                        #lr=min(0.0003,0.001),
                        lr=min(0.00025,0.001),
                        #lr=min(0.0002,0.001),
                        
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
        
        # s4 uses linear decoder
        #self.decoder = nn.Linear(d_model, nclasses)
        # use 1x1 conv decoder instead
        self.decoder = nn.Conv2d(d_model, 3, kernel_size=1)

    def forward(self, x, t, state=None):
        with torch.cuda.amp.autocast(enabled=False):
            # ensure float32 for FFT requierment in torch
            x = x.to(torch.float32)   

            #print("Input dtype:", x.dtype)
            batch_size = x.size(0)
            channel = x.shape[1]
            scale = x.shape[2]
            # reshape for yolo should be removed if using model on different shape data
            # batchsize, 3, scale, scale, 1 to 32, 3, 19*19 shape
            #print("x shape:", x.shape)    
            x = x.reshape(batch_size, channel, scale*scale)
            #print("x reshape shape:", x.shape)    

            # from 32, 3, 19*19 to 32, 19*19, 3 shape
            x = x.permute(0, 2, 1)
            #print("x permute shape:", x.shape)    

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
            
            #print("x S4 Block out:", x.shape)
            #x = x.transpose(-1, -2) # [32, 512, 361] -> [32, 361, 512]
            d_model = x.shape[1]
            x = x.reshape(batch_size, d_model, scale, scale)

            # S4 standard does Pool and decode
            # pool however unsuitable for img domain problems 
            #x = x.mean(dim=1)
            #print("X shape:", x.shape)
            output = self.decoder(x)
            #print("decoder out:", output.shape)
            # remove when using other model than yolo shapes wont match
            #output = output.reshape(batch_size, channel, scale, scale)
            
            #output = self.bn(output)
            #output = self.activation(output)
            
            output = output.unsqueeze(-1)
            # print("S4 output shape:",  output.shape)
        
            return output, state  # Always return state for consistency

    def setup_step(self):
        for block in self.s4_blocks:
            block.setup_step()

    def default_state(self, *args, **kwargs):
        return self.s4_blocks[0].default_state(*args, **kwargs)