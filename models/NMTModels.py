import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from common import modrelu


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


class BidirectionalEncoder(nn.Module):
    def __init__(self, inp_size, emb_size, enc_hid_size, dec_hid_size, dropout):
        super(BidirectionalEncoder, self).__init__()
        self.embedding = nn.Embedding(inp_size, emb_size)
        self.rnn = nn.RNN(emb_size, enc_hid_size, bidirectional=True)
        self.fc = nn.Linear(2*enc_hid_size, dec_hid_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        # deviation from paper as suggested by https://github.com/bentrevett/pytorch-seq2seq/
        # use last state for forward, first state for backward
        hidden = torch.tanh(self.fc(
            torch.cat((hidden[-2, :, :],
                       hidden[-1, :, :]), dim=1)
        ))

        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_size, dec_hid_size):
        super(Attention, self).__init__()

        self.attn = nn.Linear((enc_hid_size * 2) + dec_hid_size, dec_hid_size)
        self.v = nn.Linear(dec_hid_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src len, dec hid dim]
        attention = self.v(energy).squeeze(2)
        # attention= [batch size, src len]
        return F.softmax(attention, dim=1)


class BidirectionalDecoder(nn.Module):
    def __init__(self, out_size, emb_size, enc_hid_size, dec_hid_size, dropout, attention):
        super(BidirectionalDecoder, self).__init__()

        self.output_dim = out_size
        self.attention = attention

        self.embedding = nn.Embedding(out_size, emb_size)

        self.rnn = nn.RNN((enc_hid_size * 2) + emb_size, dec_hid_size)

        self.fc_out = nn.Linear((enc_hid_size * 2) + dec_hid_size + emb_size, out_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size]
        # hidden = [batch size, dec hid size]
        # encoder_outputs = [src len, batch size, enc hid size * 2]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb size]

        a = self.attention(hidden, encoder_outputs)

        # a = [batch size, src len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src len, enc hid size * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid size * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid size * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid size * 2) + emb size]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [seq len, batch size, dec hid size * n directions]
        # hidden = [n layers * n directions, batch size, dec hid size]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid size]
        # hidden = [1, batch size, dec hid size]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        # prediction = [batch size, output size]

        return prediction, hidden.squeeze(0)


class AttnSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)
        encoder_outputs, hidden = self.encoder(src)
        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs
