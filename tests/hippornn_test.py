import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import math
from scipy import linalg as la
from scipy import signal
from scipy import special as ss
import torch
import torch.nn.init as init
from torchvision import datasets
from torchvision import transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"The Device: {device}")

class LstmCell(nn.Module):
    def __init__(self, input_size, hidden_size, activation = "tanh"):
        super(LstmCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        #self.i2h = nn.Linear(input_size, hidden_size * 4)
        #self.h2h = nn.Linear(hidden_size, hidden_size * 4)
        self.i2h = nn.Linear(input_size, hidden_size)
        #self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2h = nn.Linear(input_size, hidden_size)
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
        #print('Lstm input x.shape:',x.shape)
        #print("Lstm cell h_t_1 shape:", h_t_1.shape)
        #print("Lstm cell c_t_1 shape:", c_t_1.shape)

        gates_i = self.i2h(x)
        #gates_h = self.h2h(h_t_1)
        gates_h = self.h2h(torch.cat((x[:, :1], h_t_1), dim=-1))
        # shape: x :-: (batchsize, 1)
        # shape: h_t_1 :-: (batchsize, hiddensize)
        # shape: c_t_1 :-: (batchsize, hiddensize)
        # shape: gates_i :-: (batchsize, hiddensize * 4)
        # shape: gates_h :-: (batchsize, hiddensize * 4)
        # shape: input_gates :-: (batchsize, hiddensize)
        # shape: c_t :-: (batchsize, hiddensize)
        #input_gate, forget_gate, cell_gate, output_gate = torch.split(gates_i + gates_h, self.hidden_size, dim=-1)
        output_gate = gates_i + gates_h
        #h_t = (self.i2h(x) + self.h2h(h_t_1))
        #print('LSTM outgate shape:', output_gate.shape)
        #i_t = torch.sigmoid(input_gate)
        #f_t = torch.sigmoid(forget_gate)
        #g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)
        #c_t = c_t_1 * f_t + i_t * g_t
        c_t = c_t_1
        #print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        #print("LSTM c_t shape:", c_t.shape)
        #print("LSTM c_t:", c_t)
        #print("LSTM h_t_1:", h_t_1)
        #print("LSTM h_t:", h_t)
        #print("LSTM c_t is all zeros:", torch.all(c_t == 0).item())
        #print("LSTM h_t_1 shape:", h_t_1.shape)
        #print("LSTM h_t_1:", h_t_1)
        #print("LSTM h_t_1 is all zeros:", torch.all(h_t_1 == 1).item())
        if self.activation == "tanh":
            h_t = o_t * torch.tanh(c_t)
            #h_t = torch.tanh(h_t)
        if self.activation == "relu":
            h_t = o_t * torch.relu(c_t)
            #h_t = torch.relu(h_t)
#if self.activation == "sigmoid":
            #h_t = o_t * torch.sigmoid(c_t)
        #print("Lstm cell out h_t shape:", h_t.shape)
        #print("Lstm cell out c_t shape:", c_t.shape)
        #print("Lstm cell out c_t.unsqueeze(1) shape:", c_t.unsqueeze(1).shape)
        #print("LSTM h_t:", h_t)

        #print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

        return (h_t, c_t)

class HippoLegsCell(nn.Module):
    '''
    Hippo class utilizing legs polynomial
    '''

    def __init__(self, N, gbt_alpha = 0.5, maxlength = 1024, reconst = False):
        super(HippoLegsCell, self).__init__()
        self.N = N
        self.gbt_alpha = gbt_alpha
        self.maxlength = maxlength
        A, self._B = self.get_A_and_B(N = self.N)
        GBTA, GBTB = self.get_stacked_GBT(A = A, B = self._B)
        #self.A = nn.parameter.Parameter(torch.from_numpy(GBTA))
        #self.B = nn.parameter.Parameter(torch.from_numpy(GBTB))
        self.A = torch.from_numpy(GBTA).to(device)
        self.B = torch.from_numpy(GBTB).to(device)
        self.reconst = reconst

    def compute_A(self, n, k):
        '''
        Computes the values for the HiPPO A matrix row by column
        using the piecewise equation on p. 31 eq. 29:
                (2n+1)^{1/2} (2k+ 1)^{1/2} if n > k
        A_{nk} = n+1                       if n = k,
                 0                         if n < k
        , where n represents the row and k the columns.

        Input:
            n (int):
                nth row of a square matrix of size N
            k (int):
                kth column of a square matrix of size N

        Output:
            Values (float):
            Individual values for the elements in the A matrix.
        '''
        if n > k:
            val = np.sqrt(2 * n + 1, dtype = np.float32) * np.sqrt(2 * k + 1, dtype = np.float32)
        if n == k:
            val = n + 1
        if n < k:
            val = 0
        return val

    def compute_B(self, n):
        '''
        Computes the values for the HiPPO B matrix row by column
        using the piecewise equation on p. 31 eq. 29:
        B_{n} = (2n+1)^{1/2}

        Input:
            n (int):
                nth column of a square matrix of size N.

        Output:
            Values (float):
            Individual values for the elements in the B matrix.
            The next hidden state (aka coefficients representing the function, f(t))
        '''
        val = np.sqrt(2 * n + 1, dtype = np.float32)
        return val

    def get_A_and_B(self, N):
        '''
        Creates the HiPPO A and B matrix given the size N along a single axis of
        a square matrix.

        Input:
            N (int):
            Size N of a square matrix along a single axis.

        Output:
            A (np.ndarray)
                shape: (N,N)
                the HiPPO A matrix.
            B (np.ndarray)
                shape: (N,):
                The HiPPO B matrix.
        '''
        A = np.zeros((self.N, self.N), dtype = np.float32)
        B = np.zeros((self.N, 1), dtype = np.float32)
        for n in range(A.shape[0]):
            B[n][0] = self.compute_B(n = n)
            for k in range(A.shape[1]):
                A[n, k] = self.compute_A(n = n , k = k)
        return A  * -1, B

    def generalized_bilinear_transform(self, A, B, t, gbt_alpha):
        '''
        Performs the generalised bilinaer transform from p. 21 eq.13:
        c(t + ∆t) − ∆tαAc(t + ∆t) = (I + ∆t(1 − α)A)c(t) + ∆tBf(t)
        c(t + ∆t) = (I − ∆tαA)^{−1} (I + ∆t(1 − α)A)c(t) + ∆t(I − ∆tαA)^{−1}Bf(t).
        on the HiPPO matrix A and B, transforming them.
        Input:
            A (np.ndarray):
                shape: (N, N)
                the HiPPO A matrix
            B (np.ndarray):
                shape: (N,)
                the HiPPO B matrix
            Timestep t = 1/input length at t (int):

        Output:
            GBTA (np.array):
                shape: (N, N)
                Transformed HiPPO A matrix.

            GBTB (np.array):
                shape: (N,)
                Transformed HiPPO B matrix.
        '''
        I = np.eye(A.shape[0], dtype = np.float32)
        delta_t = 1 / t
        EQ13_p1 = I - (delta_t * gbt_alpha * A)
        EQ13_p2 = I + (delta_t * (1 - gbt_alpha) * A)
        EQA = np.linalg.solve(EQ13_p1, EQ13_p2)
        EQB =  np.linalg.solve(EQ13_p1, (delta_t * B))
        return EQA, EQB

    def get_stacked_GBT(self, A, B):
        GBTA_stacked = np.empty((self.maxlength, self.N, self.N), dtype=np.float32)
        GBTB_stacked = np.empty((self.maxlength, self.N, 1), dtype=np.float32)
        for t in range(1, self.maxlength + 1):
            GBTA, GBTB = self.generalized_bilinear_transform(A = A, B = B, t = t, gbt_alpha = self.gbt_alpha)
            GBTA_stacked[t-1] = GBTA
            GBTB_stacked[t-1] = GBTB
        return GBTA_stacked, GBTB_stacked


    def reconstruct(self, c, B):
        '''
        Input:
            c (np.ndarray): 2, 1, 32
                shape: (batchsize, 1, N_coeffs)
                coefficent matrix
            B (np.ndarray):
                shape: (N, 1)
                the discretized B matrix
        Returns:
            recon (np.ndarray):
                shape: (batchsize, maxlength, 1)
                Reconstruction matrix.
        '''
        with torch.no_grad():
            vals = np.linspace(0.0, 1.0, self.maxlength)
            # c shape from: [batchsize, 1, N_coeffs]
            # move to: [batchsize, N_coeffs, 1]
            c = torch.moveaxis(c, 1, 2).float()
            eval_mat = (self._B * np.float32(ss.eval_legendre(np.expand_dims(np.arange(self.N, dtype = np.float32), -1), 2 * vals - 1))).T
            # shape: B :-: (N, 1)
            # shape: eval_mat :-:  (maxlen, N)
            recon = (torch.tensor(eval_mat).to(device) @ c.to(device))
            # shape: recon :-: (batchsize, maxlength, 1
            return recon

    def forward(self, input, c_t = None, t = 0):
        '''
        Input:
            A (np.ndarray):
                shape: (N, N)
                the discretized A matrix
            B (np.ndarray):
                shape: (N, 1)
                the discretized B matrix
            c_t (np.ndarray): 2, 1, 32
                shape: (batch size, 1, N)
                the initial hidden state
            input (torch.tensor):
                shape: (batch size, 1 ,1)
                the input sequence
        Output:
            c (np.array):
                shape: (batch size, 1, N)
                coefficent matrix c.
        '''
        batchsize = input.shape[0]
        L = input.shape[1]
        if c_t is None:
            c_t = torch.zeros((batchsize, 1, self.N)).to(device)
            #print('Hippo init c_t shape:', c_t.shape)
        #print('Hippo c_t before updated shape:', c_t.shape)
        #print('Hippo input before updated shape:', input.shape)
        #print('Hippo input.unsqueeze(-1) before updated shape:', input.unsqueeze(-1).shape)
        #print('Hippo c_t..unsqueeze(1) before updated shape:',c_t.float().unsqueeze(1).shape)
        if t == 0:
            c_t = c_t.float().unsqueeze(1)
        #print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        #print("Hippo c_t_1:", c_t)
        c_t = F.linear(c_t.float(), self.A[t]).float() + self.B[t].squeeze(-1) * input
        #print('Hippo c_t shape:', c_t.shape)
        #print('Hippo self.A[t] shape:', self.A[t].shape)
        #print('Hippo self.B[t] shape:', self.B[t].shape)
        #print('Hippo self.B[t].squeeze(-1) shape:', self.B[t].squeeze(-1).shape)
        #print('Hippo input shape:', input.shape)
        #print("Hippo c_t shape:", c_t.shape)
        #print("Hippo c_t:", c_t)
        #print("Hippo c_t is all zeros:", torch.all(c_t == 0).item())
        #print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::")


        # shape: F.linear(torch.tensor(c_t).float(), torch.tensor(A[t])) :-: (batchsize, 1, N)
        # shape: (np.squeeze(B[t], -1) * f_t.numpy()).shape) :-: (batchsize, 1, N)
        # shape: A[t] :-: (N, N)
        # shape: torch.squeeze(B[t], -1) :-: (N, )
        # shape: f_t :-: (batchsize, 1, 1)
        # shape: c_t :-: (batchsize, 1, N)
        if self.reconst:
            # 3. Compute reconstruction r
            r =  self.reconstruct(c = c_t, B = self._B)
        else:
            r = 0
        return c_t, r


class GatedHippoCell(nn.Module):
    def __init__(
        self, input_size, hidden_size, gbt_alpha=0.5, maxlength=1024):
        super(GatedHippoCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gbt_alpha = gbt_alpha
        self.maxlength = maxlength

        self.tau = LstmCell(self.hidden_size + 1, self.hidden_size)

        #self.fc = nn.Linear(self.hidden_size, 1)
        self.fc = nn.Linear(self.hidden_size + 1, 1)
        self.hippo_t = HippoLegsCell(
            N=self.hidden_size, gbt_alpha=self.gbt_alpha, maxlength=self.maxlength
        )

    def forward(self, x, carry=(None, None), t=0):
        """
        Performs the forward pass of the cell such that it performs one time step in the recurrence

        Args:
          input:
            The input at the current time step, shape (N, sequencelen).
            example:
              sequencelen for mnist = 28 * 28
          carry:
            The hidden state and optionally a cell state in a tuple that is carried between recurrent steps, shape h_t (N, hiddensize)

        Returns:
          The carry of the cell for the given recurrence's time step
        """

        # shape: inputs :-: (N, sequencelen) sequencelen for mnist = 28 * 28
        # shape: h_t :-: (N, hiddensize)
        # shape : carry[0]  :-: h_t.shape or (N, hiddensize)
        # print('HippoGatedCell input shape', input.shape)
        """
        if carry[0] is None:
            carry = (
                torch.zeros(x.shape[0], 1, self.hidden_size).to(device),
                torch.zeros(x.shape[0], 1, self.hidden_size).to(device),
            )
            print("HippoGatedCell Init carry[0] h_t shape:", carry[0].shape)
            print("HippoGatedCell Init carry[1 c_t_1 shape:", carry[1].shape)
        """
        h_t, c_t_1 = carry
        #print("HippoGatedCell carry h_t shape:", h_t.shape)
        #print("HippoGatedCell carry c_t_1 shape:", c_t_1.shape)

        #for t in range(input.size(1)):
        #tau_x = torch.cat((x[:, t].unsqueeze(-1), h_t), dim=-1)
        #print("x.shape:", x.shape,"h_t.shape:", h_t.shape, "c_t_1.shape:", c_t_1.shape)
        #print("Steps")
        if t == 0:
            tau_x = torch.cat((x, c_t_1), dim=-1)
        if t > 0:
            tau_x = torch.cat((x, c_t_1.squeeze(1)), dim=-1)
        #print("HippoGatedCell carry carry[0] shape:", carry[0].shape)
        #print("HippoGatedCell carry carry[1] shape:", carry[0].shape)
        #print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        #print('HippoGatedCell c_t_1:', carry[1])
        #print('HippoGatedCell h_t_1:', carry[0])
        h_t, _ = self.tau(x = tau_x, carry=(carry[0], carry[1].squeeze(1)))
        #print('HippoGatedCell h_t:', h_t)
        # tau is an lstm cell:
        # shape: x :-: (batchsize, 1)
        # shape: h_t_1 :-: (batchsize, hiddensize)
        # shape: c_t_1 :-: (batchsize, hiddensize)
        # shape: gates_i :-: (batchsize, hiddensize * 4)
        # shape: gates_h :-: (batchsize, hiddensize * 4)
        # shape: input_gates :-: (batchsize, hiddensize)
        # shape: c_t :-: (batchsize, hiddensize)
        #print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        #print("Hippo cell h_t tau out shape:", h_t.shape)
        #print("Hippo cell h_t tau out all zeros:", torch.all(h_t == 0).item())
        #print("Hippo cell c_t_1 tau out shape:", c_t_1.shape)
        #print("Hippo cell c_t_1 tau out all zeros:", torch.all(c_t_1 == 0).item())
        #print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

        f_t = self.fc(torch.cat((x, h_t), dim=-1))
        # shape: h_t :-: (batchsize, hiddensize)
        # shape: h_t :-: (batchsize, hiddensize)
        #print('HippoGatedCell c_t_1 shape', c_t_1.shape)
        #print('HippoGatedCell c_t_1 unsqueeze shape', c_t_1.unsqueeze(1).shape)

        c_t, _ = self.hippo_t(f_t.unsqueeze(-1), c_t_1, t=t)
        #print('HippoGatedCell c_t:', c_t)
        #print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

        # update previous coefficents to be current coefficents
        c_t_1 = c_t
        carry = (h_t, c_t_1)
        #print('HippoGatedCell tau_x shape', tau_x.shape)
        #print('HippoGatedCell h_t shape', h_t.shape)
        #print('HippoGatedCell f_t shape', f_t.shape)
        #print('HippoGatedCell f_t unsqueeze(1) shape', f_t.unsqueeze(-1).shape)
        #print('HippoGatedCell c_t shape', c_t.shape)

        #print(f'c_t: \n{c_t}')

        return (h_t, c_t)

class HippoRNN(nn.Module):
    """
    A PyTorch Module representing a Hippocampal Recurrent Neural Network. This model combines a GatedHippoCell
    with a MLP (Multi Layer Perceptron) for output prediction.

    Attributes:
        hidden_size: The size of the hidden state.
        output_size: The size of the output.
        cell: A GatedHippoCell instance.
        mlp: A Multi Layer Perceptron used in the forward pass for output prediction.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        The __init__ method for the HippoRNN class. Initializes the GatedHippoCell and MLP.

        Args:
            input_size: The dimensionality of the input data.
            hidden_size: The size of the hidden states in the GatedHippoCell.
            output_size: The size of the output of MLP.
        """
        super(HippoRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.cell = GatedHippoCell(input_size=input_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, 10)
	
        #self.initialize_weights()

    #def initialize_weights(self):
    #    std = 1.0 / np.sqrt(self.hidden_size)
    #    for layer_name, layer in self.named_children():
    #        if layer_name in ['i2h', 'h2h']:
    #            for param_name, param in layer.named_parameters():
    #                if 'weight' in param_name:
    #                    init.uniform_(param, -std, std)
    #                elif 'bias' in param_name:
    #                    init.zeros_(param)
                    
    def forward(self, xs, carry=(None, None)):
        """
        Performs the forward pass for the entire sequence of data.

        Args:
            inputs: The input data for the whole sequence, shape (batch_size, sequence_len, input_size).
            carry: The hidden state and optionally a cell state in a tuple that is carried between recurrent steps,
            shape h_t (batch_size, hidden_size).

        Returns:
            A tuple containing the final prediction of the MLP and a tensor of all cell states in the sequence.
        """
        if carry[0] is None:
            carry = (
                torch.zeros(xs.shape[0], self.hidden_size).to(device),
                torch.zeros(xs.shape[0], self.hidden_size).to(device),
            )
            #print("Hippo RNN carrys initalised.")

        for t in range(xs.size(1)):
            #print(f'::::::::::::TimeStep:{t}:::::::::::::::::')
            # shape xs :=: (batchsize, 28*28, 1)
            # shape xs[:, t, :] :=: (batchsize, 1)
            #print("Hippo RNN xs shape:", xs.shape)
            #print("Hippo RNN xs[:, t, :] shape:", xs[:, t, :].shape)
            # Input to hippo cell should be :-: shape: f_t :-: (batchsize, 1, 1)
            # Carry to hippo cell should be :-: shape: c_t :-: (batchsize, 1, N)
            # Carry to hippo cell should be :-: shape: h_t :-: (batchsize, 1, N)
            carry = self.cell(x=xs[:, t, :], carry=carry, t=t)
           # print("Hippo RNN h_t carry shape:", carry[0].shape)
           # print("Hippo RNN h_t is all zeros:", torch.all( carry[0] == 0).item())
           # print("Hippo RNN c_t carry shape:", carry[1].shape)
            #print("Hippo RNN c_t is all zeros:", torch.all( carry[1] == 0).item())
            #print("Hippo RNN c_t:",carry[1])
            #print("Hippo RNN h_t:",carry[0])
            #rint("::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

            #print("Hippo RNN c_t shape:", carry[1].shape)
            c_t = carry[1]#.squeeze(1)
            #print("Hippo RNN out c_t squeez(1) shape:", c_t.shape)
        return self.fc(c_t)

def train (data_loader, model, optimizer, loss_f):
    """
    Input: train loader (torch loader), model (torch model), optimizer (torch optimizer)
          loss function (torch custom yolov1 loss).
    Output: loss (torch float).
    """
    loss_lst = []
    correct = 0
    total = 0
    model.train()
    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        x = x.view(batch_size, -1, 1)
        #print(x.shape)
        out = model(x)
        # print('Train out shape:', x.shape)
        del x
        # nn.cn takes raw logits and true labels y
        #print('Model out shape:', out.shape)
        #print('Model squeeze(1) shape:', out.squeeze(1).shape)
        loss_val = loss_f(out.squeeze(1), y)
        loss_lst.append(float(loss_val.item()))
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        # get class probabilities
        classprobs = F.softmax(out.squeeze(1), dim=1)
        # get class predictions
        preds = classprobs.argmax(dim=1)
        # compute accuracy
        total += y.size(0)
        #print('preds shape:', preds.shape)
        #print('y shape:', y.shape)
        correct += (preds == y).sum().item()

    # Compute average loss and accuracy
    loss_val = round(sum(loss_lst) / len(loss_lst), 4)
    accuracy = round(correct / total, 4)
    return loss_val, accuracy

batch_size = 64
weight_decay = 0

epochs = 50
nworkers = 2
lr = 1e-3  #1e-4 # 0.00001
pin_memory = True
data_dir =  'data/'

input_size = 1
hidden_size = 512
output_size = 10
train_dataset = datasets.MNIST(root=data_dir,
                               train=True,
                               transform=T.Compose([T.ToTensor(),
                                                   T.Normalize((0.5,), (0.5,))]),
                               download=True)

train_loader = DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True, num_workers = 2, drop_last = True)

model = HippoRNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
print(model)
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
loss_f = nn.CrossEntropyLoss()
for epoch in range(epochs):
    train_loss_value, train_accuracy = train(train_loader, model, optimizer, loss_f)
    print(f"Epoch:{epoch + 1}   Train[Loss:{train_loss_value}  Accuracy:{train_accuracy}]")

