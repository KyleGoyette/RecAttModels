# adapted from https://github.com/nke001/sparse_attentive_backtracking_release/blob/master/layers_torch.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
class Sparse_attention(nn.Module):
    def __init__(self, top_k=5):
        super(Sparse_attention, self).__init__()
        self.top_k = top_k

    def forward(self, attn_s):

        # normalize the attention weights using piece-wise Linear function
        # only top k should
        attn_plot = []
        # torch.max() returns both value and location
        # attn_s_max = torch.max(attn_s, dim = 1)[0]
        # attn_w = torch.clamp(attn_s_max, min = 0, max = attn_s_max)
        eps = 10e-8
        time_step = attn_s.size()[1]
        if time_step <= self.top_k:
            # just make everything greater than 0, and return it
            # delta = torch.min(attn_s, dim = 1)[0]
            return torch.nn.functional.softmax(attn_s, dim=1)
        else:
            # get top k and return it
            # bottom_k = attn_s.size()[1] - self.top_k
            # value of the top k elements
            # delta = torch.kthvalue(attn_s, bottm_k, dim= 1 )[0]
            delta = torch.topk(attn_s, self.top_k, dim=1)[0][:, -1] + eps
            # delta = attn_s_max - torch.topk(attn_s, self.top_k, dim= 1)[0][:,-1] + eps
            # normalize
        attn_w = attn_s - delta.repeat((1, time_step)).view(attn_s.shape)
        # relu
        attn_w = torch.clamp(attn_w, min=0)
        attn_w_sum = torch.sum(attn_w, dim=1)
        attn_w_sum = attn_w_sum + eps
        attn_w_normalize = attn_w / attn_w_sum.unsqueeze(1).repeat(1, time_step)
        assert(torch.min(attn_w_normalize) >= 0)
        return attn_w_normalize

class SAB_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, truncate_length=100,
                 block_attn_grad_past=False, print_attention_step=1, attn_every_k=1, top_k=5, device=None):
        # latest sparse attentive backprop implementation
        super(SAB_LSTM, self).__init__()
        if device is None:
            device = torch.device('cpu')
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.block_attn_grad_past = block_attn_grad_past
        self.truncate_length = truncate_length
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.attn_every_k = attn_every_k
        self.top_k = top_k
        self.tanh = torch.nn.Tanh()
        self.w_t = nn.Parameter(torch.Tensor(self.hidden_size * 2, 1).normal_(mean=0.0, std=0.01))  # nn.Parameter(torch.zeros(self.hidden_size * 2, 1))
        self.sparse_attn = Sparse_attention(top_k=self.top_k)
        self.atten_print = print_attention_step

    def print_log(self):
        model_name = '_LSTM-sparse_attn_topk_attn_in_h' + str(self.top_k) + '_truncate_length_' + str(
            self.truncate_length) + 'attn_everyk_' + str(
            self.attn_every_k)  # + '_block_MLP_gradient_' + str(self.block_attn_grad_past)
        model_log = ' LSTM Sparse attention in h........topk:' + str(self.top_k) + '....attn_everyk_' + str(
            self.attn_every_k) + '.....truncate_length:' + str(self.truncate_length)
        return (model_name, model_log)

    def forward(self, x):
        batch_size = x.size(0)
        time_size = x.size(1)
        input_size = x.size(2)
        hidden_size = self.hidden_size
        h_t = torch.zeros((batch_size, hidden_size), requires_grad=True).to(self.device)
        c_t = torch.zeros((batch_size, hidden_size), requires_grad=True).to(self.device)

        # Will eventually grow to (batch_size, time_size, hidden_size)
        # with more and more concatenations.
        h_old = h_t.view(batch_size, 1, hidden_size)
        outputs = []
        attn_all = []
        attn_w_viz = []
        h_s = []
        for i, input_t in enumerate(x.chunk(time_size, dim=1)):
            if torch.sum((torch.isnan(h_t))) > 0:
                print(f'{i} h_t')
                raise

            if torch.sum((torch.isnan(h_old))) > 0:
                print(f'{i} h_old')
                raise

            remember_size = h_old.size(1)

            input_t = input_t.contiguous().view(batch_size, input_size)
            if (i + 1) % self.truncate_length == 0:
                h_t, c_t = self.lstm1(input_t, (h_t.detach(), c_t.detach()))

            else:
                # Feed LSTM Cell
                h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t.retain_grad()
            h_s.append(h_t)
            if torch.sum((torch.isnan(h_t))) > 0:
                print(f'{i} h_t2')
                raise
            # Broadcast and concatenate current hidden state against old states

            #h_repeated = h_t.unsqueeze(1).repeat(1, remember_size, 1)
            h_repeated = h_t.unsqueeze(1).expand(h_old.shape)
            #print(h_t.shape, h_repeated.shape, h_old.shape)#, h_repeated_new.shape)
            mlp_h_attn = torch.cat((h_repeated, h_old), 2)
            if self.block_attn_grad_past:
                mlp_h_attn = mlp_h_attn.detach()

            if torch.sum((torch.isnan(mlp_h_attn))) > 0:
                print(f'{i} mlp_h_attn')
                raise

            #
            # Feed the concatenation to the MLP.
            # The tensor shapes being multiplied are
            #     mlp_h_attn.size() = (batch_size, remember_size, 2*hidden_size)
            # by
            #     self.w_t.size()   = (2*hidden_size, 1)
            # Desired result is
            #     attn_w.size()     = (batch_size, remember_size, 1)
            #
            mlp_h_attn = self.tanh(mlp_h_attn)
            if True:  # PyTorch 0.2.0
                attn_w = torch.matmul(mlp_h_attn, self.w_t)
            else:  # PyTorch 0.1.12
                mlp_h_attn = mlp_h_attn.view(batch_size * remember_size, 2 * hidden_size)
                attn_w = torch.mm(mlp_h_attn, self.w_t)
                attn_w = attn_w.view(batch_size, remember_size, 1)
            if torch.sum((torch.isnan(attn_w))) > 0:
                print(f'{i} attn_w')
                raise
            #
            # For each batch example, "select" top-k elements by sparsifying
            # attn_w.size() = (batch_size, remember_size, 1). The top k elements
            # are left non-zero and the other ones are zeroed.
            #
            attn_w = attn_w.view(batch_size, remember_size)
            attn_w = self.sparse_attn.forward(attn_w)
            assert torch.min(attn_w) >= 0, 'check 0'
            attn_w = attn_w.view(batch_size, remember_size, 1)
            assert torch.min(attn_w) >= 0, 'check 1'
            #if self.atten_print >= (time_size - i - 1):
            filler = attn_w.new_zeros((1,attn_w.shape[1]*(self.attn_every_k - 1)))
            # this line tosses 0s between all of the
            interspaced_attn = torch.cat((attn_w[0].clone().detach().t(), filler), dim=1).view(self.attn_every_k, -1).t().reshape(-1, 1)
            assert torch.min(attn_w) >= 0, 'check 2'
            #print(torch.sum(interspaced_attn), torch.min(interspaced_attn), torch.max(interspaced_attn))
            #print(interspaced_attn.shape)
            attn_w_viz.append(interspaced_attn)
            #attn_w_viz.append(torch.cat((attn_w[0, :, 0], attn_w[0, :].new_zeros(time_size - attn_w[0,:,0].shape[0])), dim=0))
            #print(time_size,attn_w_viz[-1].shape)
            if torch.sum((torch.isnan(attn_w))) > 0:
                print(f'{i} attn_w2')
                raise

            #
            # Broadcast the weights against the past remembered hidden states,
            # then compute the attention information attn_c.
            #

            attn_w = attn_w.repeat(1, 1, hidden_size)
            h_old_w = attn_w * h_old
            attn_c = torch.sum(h_old_w, 1).squeeze(1)

            if torch.sum((torch.isnan(h_old_w))) > 0:
                print(f'{i} h_old_w')
                raise
            #print(torch.max(h_old_w), torch.max(attn_w), torch.max(attn_c))
            if torch.sum((torch.isnan(attn_c))) > 0:
                print(f'{i} attn_c')
                raise

            # Feed attn_c to hidden state h_t
            h_t += attn_c

            #
            # At regular intervals, remember a hidden state.
            #
            if (i + 1) % self.attn_every_k == 0:
                h_old = torch.cat((h_old, h_t.view(batch_size, 1, hidden_size)), dim=1)

            # Record outputs
            outputs += [h_t]

            # For visualization purposes:
            attn_all += [attn_c]

        #
        # Compute return values. These should be:
        #     out        = (batch_size, time_size, num_classes)
        #     attn_w_viz = len([(remember_size)]) == time_size-100
        #
        outputs = torch.stack(outputs, 1)
        attn_all = torch.stack(attn_all, 1)
        outputs = torch.cat((outputs, attn_all), 2)
        shp = outputs.size()
        out = outputs.contiguous().view(shp[0] * shp[1], shp[2])
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)
        self.alphas = attn_w_viz

        return out, h_s