import torch
import torch.nn as nn
import wandb
import argparse
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from experiment import Experiment
from common import construct_heatmap_data
from utils import (generate_denoise_batch,
                   generate_copy_batch)

parser = argparse.ArgumentParser(description='Synthetic task runner')
# task params
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--task', type=str, default='copy',
                    choices=['copy', 'denoise'])
parser.add_argument('--iters', type=int, default=2000)
parser.add_argument('--T', type=int, default=200, help='Delay')
parser.add_argument('--n_labels', type=int, default=8, help='Number of labels')
parser.add_argument('--seq_len', type=int, default=10,
                    help='Length of sequence to copy')
parser.add_argument('--onehot', action='store_true',
                    help='onehot inputs and labels')
# model params
parser.add_argument('--model', type=str,
                    choices=['RNN', 'LSTM', 'ORNN', 'MemRNN'], default='RNN')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden units')
parser.add_argument('--nonlin', type=str, default='tanh',
                    help='Non linearity, locked to tanh for LSTM')
#optim params/data params
parser.add_argument('--opt', type=str, default='RMSProp',
                    choices=['SGD', 'RMSProp', 'Adam'])
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--lr_orth', type=float, default=None)
parser.add_argument('--alpha', type=float, default=None)
parser.add_argument('--betas', type=float, default=None, nargs="+")
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--logfreq', type=int, default=500,
                    help='frequency to log heatmaps, gradients')

def run():
    # set up hyperparameters for sweeps
    args = parser.parse_args()
    hyper_parameter_defaults = dict(
        opt='RMSProp',
        nonlin='relu',
        batch_size=12,
        learning_rate=0.0002,
        betas=(0.5, 0.999),
        alpha=0.9
    )
    # create save_dir using wandb name
    if args.name is None:
        run = wandb.init(project="rec-att-project",
                   config=hyper_parameter_defaults)
        # save run to get readable run name
        run.save()
        run.name = os.path.join(args.task, run.name)
        config = wandb.config
        config.save_dir = os.path.join('experiments', args.task, run.name)
        run.save()
    else:
        run = wandb.init(project="rec-att-project",
                   config=hyper_parameter_defaults,
                   name=args.name)
        run.name = os.path.join(args.task, run.name)
        config = wandb.config
        config.save_dir = os.path.join('experiments', args.task, args.name)
        run.save()

    if args.onehot:
        args.input_size = args.n_labels + 2
    else:
        args.input_size = 1
    loss_crit = nn.CrossEntropyLoss()

    # set up task specific configs and loss
    if args.task == 'copy':
        batch_generator = generate_copy_batch
    elif args.task == 'denoise':
        batch_generator = generate_denoise_batch

    # update config object with args
    wandb.config.update(args, allow_val_change=True)
    # create experiment object
    experiment = Experiment(config)
    model = experiment.model
    wandb.watch(model)

    if args.model in ['ORNN']:
        optimizer, orth_optimizer = experiment.optimizer
    else:
        optimizer = experiment.optimizer
        orth_optimizer = None

    accs = experiment.train_accs
    losses = experiment.train_losses
    x_const, y_const = batch_generator(delay=config.T,
                                           n_labels=config.n_labels,
                                           seq_length=config.seq_len,
                                           batch_size=1)
    for i in range(config.iters):
        s_t = time.time()
        x, y = batch_generator(delay=config.T,
                                   n_labels=config.n_labels,
                                   seq_length=config.seq_len,
                                   batch_size=config.batch_size)

        x = x.transpose(1, 0)
        model.zero_grad()
        outs, hiddens = model.forward(x)
        if i % config.logfreq == 0:
            model.zero_grad()

            labels_loss = loss_crit(outs[-config.seq_len:, :].transpose(2, 1),
                                    y[-config.seq_len:, :])
            labels_loss.backward(retain_graph=True)
            wandb.log({'label loss': labels_loss})
        model.zero_grad()
        all_loss = loss_crit(outs.transpose(2, 1), y)
        all_loss.backward()
        losses.append(all_loss.item())
        optimizer.step()
        if orth_optimizer is not None:
            orth_optimizer.step()
        preds = torch.argmax(outs[:, -config.seq_len:, :], dim=2)
        wandb.log({"predictions": wandb.Histogram(preds.detach().numpy())})
        correct = torch.sum(preds == y[:, -config.seq_len:])
        acc = correct/float(y.shape[0]*config.seq_len)
        accs.append(acc)

        if i % config.logfreq == 0:
            model.zero_grad()
            outs, hiddens = model.forward(x_const.transpose(1, 0))
            labels_loss = loss_crit(outs[-config.seq_len:, :].transpose(2, 1),
                                    y_const[-config.seq_len:, :])
            labels_loss.backward(retain_graph=True)

            grads = [h.grad.data.norm(2) for h in hiddens]
            fig = go.Figure(data=go.Scatter(x=list(range(len(grads))),
                                            y=grads))
            fig.update_layout(title='Gradient flow (update={})'.format(i),
                              xaxis=dict(title='t'),
                              yaxis=dict(title=r'$\frac{dL}{dh_t}$'))

            wandb.log({'grads': fig})
            # log heat maps for attention models
            if config.model in ['MemRNN']:
                hm = construct_heatmap_data(model.rnn.alphas)
                wandb.log({'heat map': px.imshow(hm)})

        print('Update {}, Time for Update: {} , Average Loss: {}, Accuracy: {}'
              .format(i + 1, time.time() - s_t, all_loss.item(), acc))

        wandb.log({"loss": all_loss})
        wandb.log({"accuracy": acc})



if __name__ == '__main__':
    run()