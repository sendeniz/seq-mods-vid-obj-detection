import torch
from s4_model import S4
from sashimi import ResidualBlock
import torch.nn as nn


# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d

class S4Block(nn.Module):
    def __init__(self, d_model, dropout=0.2, n_layers=2):
        super(S4Block, self).__init__()
    
        
        self.s4_layer = S4(
            d_model=d_model,
            bidirectional=False,
            dropout=dropout,
            transposed=True,
            lr=min(0.001, 1e-6),
        )
        
        self.residual_block = ResidualBlock(
            d_model=d_model,
            layer=self.s4_layer,
            dropout=dropout,
        )
        self.s4_state = None
        
            
    def forward(self, x):
        x = self.residual_block(x)
        return x

    def setup_step(self, mode="diagonal"):
        # Only call setup_step on submodules that have the method
        for module in self.children():  
            if hasattr(module, 'setup_step'):
                module.setup_step(mode=mode)

    def step(self, x, state=None):
        
        # if state is none init default state
        if state is None:
            state = self.residual_block.default_state()
        
        output, new_state = self.residual_block.step(x, state)
        self.s4_state = new_state
        return output, new_state

    
    def default_state(self):
        return self.residual_block.default_state()

# Example usage
model = S4Block(d_model=1)
print(model)

# Setup step for evaluation
model.setup_step(mode="linear")
model.eval()


# Full forward pass
#full_out, _ = model(input_seg)

# Streaming inference
s4_state = model.default_state()

# Example input
#  Yolo out at smallest scale 
# out = [32, 3, 19, 19, 35]
# out[..., 1:5] are x, y, h, w of bounding box
# out[..., 1:2] is x with shape [32, 3, 19, 19, 1]

# Mnist batchsize, 28, 28, 1

batchsize = 32
channel = 3
scale = 19
seq_len = 5

mnist_shape = (batchsize, 2, 2, 1)
x_mnist = torch.randn(mnist_shape)
x_mnist = x_mnist.reshape(batchsize, 1, 2 * 2)

#out = torch.randn(batchsize, channel, 19, 19, 35)
#print("yolo out shape:", out.shape)
#x = out[..., 1:2]
#print("x_coords shape:", x.shape)
#x = torch.flatten(x, 2)
#print("x_coords flat shape:", x.shape)
# print("x_coords flat part shape:", x_coords_flat[:, :, 1].shape)

# full forward pass
print("Input shape:", x_mnist.shape)
full_out, _ = model(x_mnist)
print(full_out.shape)
# init default state
s4_state = model.default_state()

# simulate recurrence 
for i in range(x_mnist.shape[-1]):
    x_t = x_mnist[:, :, i]
#    #print(f"x_t.shape: {x_t.shape}, state shape: {s4_state.shape}")
#    #print(f"s4 state_t-1: {s4_state[0][0]}")
#    out, s4_state = model.step(x_t, s4_state)
#    print(f"s4 out shape: {out.shape}, state shape: {s4_state.shape}")
    #print(f"s4 out: {out[0][0]}")
    #print(f"s4 state_t: {s4_state[0][0]}")



    