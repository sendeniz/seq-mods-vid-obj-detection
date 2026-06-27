#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 22:52:13 2022

@author: sen
"""
import torch
from torch import nn
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
import numpy as np

class RnnCell(nn.Module):
    """
    RNN cell implementation.

    Args:
        input_size (int): Size of the input feature.
        hidden_size (int): Size of the hidden state.
        activation (str, optional): Activation function for the cell. Can be 'tanh', 'relu', or 'sigmoid'. Defaults to 'relu'.

    Note:
        Only 'tanh', 'relu', and 'sigmoid' activations are supported.

    """
    def __init__(self, input_size, hidden_size, activation = "relu"):
        super(RnnCell, self).__init__()
        self.activation = activation
        if self.activation not in ["tanh", "relu", "sigmoid"]:
            raise ValueError("Invalid nonlinearity selected for RNN. Please use tanh, relu, or sigmoid.")

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, carry):
        """
        Forward pass of the RNN cell.

        Args:
            x (torch.Tensor): Input tensor of shape [batchsize, 1].
            carry (tuple): Tuple containing the previous carry states c_t and h_t.

        Returns:
            tuple: A tuple containing the new carry states c_t and h_t.
        """

        # Carry
        h_t_1, _ = carry 
        h_t = self.i2h(x) + self.h2h(h_t_1)
        # Takes output from hidden and applies activation
        if self.activation == "tanh":
            h_t = torch.tanh(h_t)
        elif self.activation == "relu":
            h_t = torch.relu(h_t)
        elif self.activation == "sigmoid":
            h_t = torch.sigmoid(h_t)

        return (h_t, h_t)

class GruCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GruCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size * 3)
        self.h2h = nn.Linear(hidden_size , hidden_size * 3)

    def forward(self, x, carry):
        '''
        Inputs: input (torch tensor) of shape [batchsize, input_size]
                hidden state (torch tensor) of shape [batchsize, hiddensize]
        Output: output (torch tensor) of shape [batchsize, hiddensize]
        '''

        # carry
        h_t_1, _ = carry

        gates_i = self.i2h(x)
        gates_h = self.h2h(h_t_1)

        # get the gate outputs
        z_t, r_t, n_t = torch.split(gates_i + gates_h, self.hidden_size, dim=-1)
        z_t = torch.sigmoid(z_t)
        r_t = torch.sigmoid(r_t)
        n_t = torch.relu(n_t)

        h_t = ((1 - z_t) * n_t) + (z_t * h_t_1)

        return (h_t, h_t)

class LstmCell(nn.Module):
    def __init__(self, input_size, hidden_size, activation = "tanh"):
        super(LstmCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.i2h = nn.Linear(input_size, hidden_size * 4)
        self.h2h = nn.Linear(hidden_size, hidden_size * 4)

        self.activation = activation
        if self.activation not in ["tanh", "relu", "sigmoid"]:
            raise ValueError("Invalid nonlinearity selected for RNN. Please use tanh, relu, or sigmoid.")
        

    def forward(self, x, carry):
        '''
        Inputs: input (torch tensor) of shape [batchsize, input_size]
                hidden state (torch tensor) of shape [batchsize, hiddensize]
        Output: output (torch tensor) of shape [batchsize, hiddensize]
        '''

        # carry
        h_t_1, c_t_1 = carry

        gates_i = self.i2h(x)
        gates_h = self.h2h(h_t_1)

        # get the gate outputs
        input_gate, forget_gate, cell_gate, output_gate = torch.split(gates_i + gates_h, self.hidden_size, dim=-1)

        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        c_t = c_t_1 * f_t + i_t * g_t
        if self.activation == "tanh":
            h_t = o_t * torch.tanh(c_t)
        if self.activation == "relu":
            h_t = o_t * torch.relu(c_t)
        if self.activation == "sigmoid":
            h_t = o_t * torch.sigmoid(c_t)

        return (h_t, c_t)

class UrLstmCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(UrLstmCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size *4)
        self.h2h = nn.Linear(hidden_size, hidden_size *4)
        self.forget_bias = nn.Parameter(torch.Tensor(self.init_forget_bias()))

    def init_forget_bias(self):
        u = np.random.uniform(low=1/self.hidden_size, high=1-1/self.hidden_size, size=self.hidden_size)
        forget_bias = -np.log(1/u - 1)
        return forget_bias
    
    def forward(self, x, carry):
        '''
        Inputs: input (torch tensor) of shape [batchsize, input_size]
                hidden state (torch tensor) of shape [batchsize, hiddensize]
        Output: output (torch tensor) of shape [batchsize, hiddensize]
        '''

        # carry
        h_t_1, c_t_1 = carry

        gates_i = self.i2h(x)

        gates_h = self.h2h(h_t_1)

        f, r, u, o = torch.split(gates_i + gates_h, self.hidden_size, dim=-1)

        f_ = torch.sigmoid(f + self.forget_bias)
        r_ = torch.sigmoid(r - self.forget_bias)
        g = 2*r_*f_ + (1-2*r_)*f_**2

        c_t = g * c_t_1 + (1-g) * torch.tanh(u)

        h_t = torch.sigmoid(o) * torch.tanh(c_t)

        return (h_t, c_t)