import torch
import torch.nn as nn
from common import modrelu
import random

class RNNEncoder(nn.Module):
    def __init__(self, inp_size, emb_size, hidden_size, n_layers, dropout):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(inp_size, emb_size)
        self.layers = []
        for i in range(n_layers):
            if i == 0:
                self.layers.append(nn.RNNCell(emb_size, hidden_size))
            else:
                self.layers.append(nn.RNNCell(hidden_size, hidden_size))

        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        hiddens = [torch.zeros((embedded.shape[1], self.hidden_size))
                   for i in range(len(self.layers))]
        outs = []
        for i in range(embedded.shape[0]):
            out = embedded[i, :, :]
            for j, layer in enumerate(self.layers):
                out = layer(out, hiddens[j])
                hiddens[j] = out
            if i == embedded.shape[0] - 1:
                outs.append(out)
        return outs

class RNNDecoder(nn.Module):
    def __init__(self, out_size, emb_size, hidden_size, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.output_dim = out_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(out_size, emb_size)
        self.layers = []
        for i in range(n_layers):
            if i == 0:
                self.layers.append(nn.RNNCell(emb_size, hidden_size))
            else:
                self.layers.append(nn.RNNCell(hidden_size, hidden_size))

        self.dropout = nn.Dropout(dropout)
        self.ol = nn.Linear(hidden_size, out_size)

    def forward(self, input, hiddens):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        out = embedded
        for j, layer in enumerate(self.layers):
            hidden = layer(out.squeeze(0), hiddens[j])
            out = hidden
            hiddens[j] = hidden
        pred = self.ol(out)
        return pred, hiddens

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert len(encoder.layers) == len(decoder.layers), \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size,
                              trg_vocab_size)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hiddens = self.encoder(src)
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hiddens = self.decoder(input, hiddens)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs



class MemRNN(nn.Module):
    def __init__(self, input_size, hidden_size, nonlinearity, bias=True, cuda=False):
        super(MemRNN, self).__init__()
        self.cudafy = cuda
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
        # Create linear layer to act on input X
        self.U = nn.Linear(input_size, hidden_size, bias=bias)
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Va = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Parameter(torch.Tensor(1,hidden_size))
        nn.init.xavier_normal_(self.v.data)
        self.es = []
        self.alphas = []

    def forward(self, x, hidden, reset=False):
        if hidden is None or reset:
            if hidden is None:
                hidden = x.new_zeros(x.shape[0],
                                     self.hidden_size,
                                     requires_grad=True)
            self.memory = []
            h = self.U(x) + self.V(hidden)
            self.st = h
            self.es = []
            self.alphas = []

        else:
            all_hs = torch.stack(self.memory)
            Uahs = self.Ua(all_hs)

            es = torch.matmul(self.tanh(self.Va(self.st).expand_as(Uahs) + Uahs), self.v.unsqueeze(2)).squeeze(2)
            alphas = self.softmax(es)
            self.alphas.append(alphas)
            self.es.append(es)
            all_hs = torch.stack(self.memory, 0)
            ct = torch.sum(
                torch.mul(
                    alphas.unsqueeze(2).expand_as(all_hs),
                    all_hs),
                dim=0
            )
            self.st = (all_hs[-1] + ct)
            h = self.U(x) + self.V(self.st)

        if self.nonlinearity:
            h = self.nonlinearity(h)
        h.retain_grad()
        self.memory.append(h)
        return h