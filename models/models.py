import torch
import torch.nn as nn
from common import onehot
import numpy as np
from common import modrelu
from models.expRNN.expm32 import expm32


def henaff_init(n):
    # Initialization of skew-symmetric matrix
    s = np.random.uniform(-np.pi, 0., size=int(np.floor(n / 2.)))
    return create_diag(s, n)


def create_diag(s, n):
    diag = np.zeros(n-1)
    diag[::2] = s
    A_init = np.diag(diag, k=1)
    A_init = A_init - A_init.T
    return A_init.astype(np.float32)


class RecurrentCopyModel(nn.Module):
    def __init__(self, rnn, hidden_size, onehot, n_labels, device=None):
        super(RecurrentCopyModel, self).__init__()
        self.rnn = rnn
        self.device = device
        self.hidden_size = hidden_size
        self.onehot = onehot
        self.n_labels = n_labels
        self.ol = nn.Linear(hidden_size, n_labels+1)
        nn.init.kaiming_normal_(self.ol.weight.data, nonlinearity="relu")

    def forward(self, input):
        hiddens = []
        outs = []
        hidden = None
        for i in range(input.shape[0]):
            # pass into RNN
            if self.onehot:
                inp_onehot = onehot(input[i, :], self.n_labels)
                hidden = self.rnn.forward(inp_onehot, hidden)
            else:
                hidden = self.rnn.forward(input[i, :].unsqueeze(1).float(), hidden)
            # get hid state pass into ol layer
            if isinstance(self.rnn, nn.LSTMCell):
                h, c = hidden
                out = self.ol(c)
            else:
                h = hidden
                out = self.ol(hidden)
            # retain grads for plotting
            h.retain_grad()
            out.retain_grad()
            # append to lists for return
            hiddens.append(h)
            outs.append(out)
        return torch.stack(outs, dim=1), hiddens


class MemRNN(nn.Module):
    def __init__(self, input_size, hidden_size, nonlinearity, bias=True, device=None):
        super(MemRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size

        # Add Non linearity
        if nonlinearity == 'relu':
            self.nonlinearity = nn.ReLU()
        if nonlinearity == 'modrelu':
            self.nonlinearity = modrelu(hidden_size)
        elif nonlinearity == 'tanh':
            self.nonlinearity = nn.Tanh()
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        else:
            self.nonlinearity = None

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
        # Create linear layer to act on input X and recurrent weights
        self.U = nn.Linear(input_size, hidden_size, bias=bias)
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        # memory network
        self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Va = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Parameter(torch.Tensor(1, hidden_size))

        nn.init.xavier_normal_(self.v.data)
        self.V.weight.data = torch.as_tensor(henaff_init(hidden_size))
        A = self.V.weight.data.triu(diagonal=1)
        A = A - A.t()
        self.V.weight.data = expm32(A)
        self.es = []
        self.alphas = []

    def forward(self, x, hidden, reset=False):
        if hidden is None or reset:
            if hidden is None:
                hidden = x.new_zeros(x.shape[0],
                                     self.hidden_size,
                                     requires_grad=True)
                if self.device is not None:
                    hidden = hidden.to(self.device)
            self.memory = []
            h = self.U(x) + self.V(hidden)
            self.st = h
            self.es = []
            self.alphas = []

        else:
            all_hs = torch.stack(self.memory)
            Uahs = self.Ua(all_hs)
            energy = torch.matmul(
                self.tanh(self.Va(self.st).expand_as(Uahs) + Uahs),
                self.v.unsqueeze(2)).squeeze(2)
            alphas = self.softmax(energy)
            self.es.append(energy)
            self.alphas.append(alphas)

            all_hs = torch.stack(self.memory, 0)
            ct = torch.sum(
                torch.mul(
                    alphas.unsqueeze(2).expand_as(all_hs),
                    all_hs),
                dim=0
            )
            self.st = all_hs[-1] + ct
            h = self.U(x) + self.V(self.st)

        if self.nonlinearity:
            h = self.nonlinearity(h)
        # call retain grad to keep the gradient to log later
        h.retain_grad()
        self.memory.append(h)
        return h