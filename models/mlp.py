import torch
import torch.nn as nn

class Mlp(nn.Module):
    '''
    Multilayer perceptron implementation in base torch. 
    Useage: MLP(N, [output_size]).
    Note: When using [output_size] = 1, MLP is equal to a single nn.Linear(). 
    Input:
        N (int):
            N inputs
        output_size (list):
            output size int wrapped in a list i.e., [1]

        Output:
            x (torch.tensor):
                output of the MLP.
    '''

    def __init__(self, input_size, nlayers, activ_fn = nn.ReLU):
        super(Mlp, self).__init__()
        layers =[]
        if len(nlayers) > 1:
            prev_layer_size = None
            for i, layer_size in enumerate(nlayers[:-1]):
                layer = None
                if i == 0:
                    layers.append(nn.Linear(input_size, layer_size))
                    layers.append(activ_fn())
                else:
                    layers.append(nn.Linear(prev_layer_size, layer_size))
                    layers.append(activ_fn())
                prev_layer_size = layer_size
            layers.append(nn.Linear(nlayers[-2], nlayers[-1]))
        else:
            layers.append(nn.Linear(input_size, nlayers[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x