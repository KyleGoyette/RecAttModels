import torch
import torch.nn as nn
import torch.nn.functional as F

import random

##############################################################################
#                               SEQ2SEQ RNN                                  #
#                                                                            #
##############################################################################


class RNNEncoder(nn.Module):
    def __init__(self, inp_size, emb_size, hid_size, n_layers, dropout):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hid_size
        self.embedding = nn.Embedding(inp_size, emb_size)
        self.layers = []
        for i in range(n_layers):
            if i == 0:
                self.layers.append(nn.RNNCell(emb_size, hid_size))
            else:
                self.layers.append(nn.RNNCell(hid_size, hid_size))

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
    def __init__(self, out_size, emb_size, hid_size, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.output_dim = out_size
        self.hidden_size = hid_size
        self.embedding = nn.Embedding(out_size, emb_size)
        self.layers = []
        for i in range(n_layers):
            if i == 0:
                self.layers.append(nn.RNNCell(emb_size, hid_size))
            else:
                self.layers.append(nn.RNNCell(hid_size, hid_size))

        self.dropout = nn.Dropout(dropout)
        self.ol = nn.Linear(hid_size, out_size)

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

##############################################################################
#             ATTENTION SEQ2SEQ: https://arxiv.org/pdf/1409.0473)            #
#                                                                            #
##############################################################################

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


##############################################################################
#                           Transformer                                      #
#                                                                            #
##############################################################################
class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_size,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super(EncoderLayer, self).__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_size)
        self.ff_layer_norm = nn.LayerNorm(hid_size)
        self.self_attention = MultiHeadAttentionLayer(hid_size, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_size,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))

        return src


class TransformerEncoder(nn.Module):
    def __init__(self, inp_size, hid_size, n_layers, n_heads, pf_dim, dropout, max_length, device):
        super(TransformerEncoder, self).__init__()

        self.tok_embedding = nn.Embedding(inp_size, hid_size)
        self.pos_embedding = nn.Embedding(max_length, hid_size)

        self.layers = nn.ModuleList([EncoderLayer(hid_size,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([hid_size]))
        if device is not None:
            self.scale = self.scale.to(device)

    def forward(self, src, src_mask):
        # src = [batch size, src len]
        # src_mask = [batch size, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, src len]
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        # src = [batch size, src len, hid dim
        for layer in self.layers:
            src = layer(src, src_mask)

        # src = [batch size, src len, hid dim]
        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_size, n_heads, dropout, device):
        super(MultiHeadAttentionLayer, self).__init__()

        assert hid_size % n_heads == 0

        self.hid_size = hid_size
        self.n_heads = n_heads
        self.head_dim = hid_size // n_heads
        self.fc_q = nn.Linear(hid_size, hid_size)
        self.fc_k = nn.Linear(hid_size, hid_size)
        self.fc_v = nn.Linear(hid_size, hid_size)
        self.fc_o = nn.Linear(hid_size, hid_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        if device is not None:
            self.scale = self.scale.to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_size)
        x = self.fc_o(x)
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_size, pf_dim, dropout):
        super(PositionwiseFeedforwardLayer, self).__init__()

        self.fc_1 = nn.Linear(hid_size, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]
        x = self.dropout(torch.relu(self.fc_1(x)))
        # x = [batch size, seq len, pf dim]
        x = self.fc_2(x)
        # x = [batch size, seq len, hid dim]
        return x


class TransformerDecoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_size,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 max_length=100,
                 device=None):
        super(TransformerDecoder, self).__init__()

        self.tok_embedding = nn.Embedding(output_dim, hid_size)
        self.pos_embedding = nn.Embedding(max_length, hid_size)

        self.layers = nn.ModuleList([DecoderLayer(hid_size,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_size, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_size]))
        if device is not None:
            self.scale = self.scale.to(device)
        self.device = device

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1)
        if self.device is not None:
            pos = pos.to(self.device)
        # pos = [batch size, trg len]
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        # trg = [batch size, trg len, hid dim]
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]
        output = self.fc_out(trg)
        # output = [batch size, trg len, output dim]
        return output, attention


class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_size,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super(DecoderLayer, self).__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_size)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_size)
        self.ff_layer_norm = nn.LayerNorm(hid_size)
        self.self_attention = MultiHeadAttentionLayer(hid_size, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_size, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_size,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]
        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]
        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]
        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]
        return trg, attention


class TransformerSeq2Seq(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 device):
        super(TransformerSeq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # src = [batch size, src len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        if self.device is not None:
            src_mask = src_mask.to(self.device)
        # src_mask = [batch size, 1, 1, src len]
        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # trg_pad_mask = [batch size, 1, 1, trg len]
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len))).bool()
        if self.device is not None:
            trg_sub_mask = trg_sub_mask.to(self.device)
        # trg_sub_mask = [trg len, trg len]
        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]
        return trg_mask

    def forward(self, src, trg):
        # src = [batch size, src len]
        # trg = [batch size, trg len]
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]
        enc_src = self.encoder(src, src_mask)
        # enc_src = [batch size, src len, hid dim]
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]
        return output, attention
