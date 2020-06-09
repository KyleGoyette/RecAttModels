import torch
import torch.nn as nn
import wandb
import argparse
import time
import os
import plotly.graph_objects as go
import numpy as np
from experiment import Experiment
from common import construct_heatmap_data, onehot
from utils import (generate_denoise_batch,
                   generate_copy_batch,
                   perform_synthtask_ablations)


parser = argparse.ArgumentParser(description='Synthetic task runner')
# task params
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--group', type=str, default=None)
parser.add_argument('--device', type=int, default=None)
parser.add_argument('--task', type=str, default='copy',
                    choices=['copy', 'denoise'])
parser.add_argument('--iters', type=int, default=10000)
parser.add_argument('--T', type=int, default=200, help='Delay')
parser.add_argument('--n_labels', type=int, default=8, help='Number of labels')
parser.add_argument('--seq_len', type=int, default=10,
                    help='Length of sequence to copy')
parser.add_argument('--onehot', action='store_true',
                    help='onehot inputs and labels')
parser.add_argument('--batch_size', type=int, default=12,
                    help='batch size')
# model params
parser.add_argument('--model', type=str,
                    choices=['RNN', 'LSTM', 'ORNN', 'MemRNN', 'SAB', 'Trans'],
                    default='RNN')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden units')
parser.add_argument('--nhead', type=int, default=2,
                    help='attention heads')
parser.add_argument('--nenc', type=int, default=2,
                    help='number of encoder layers')
parser.add_argument('--ndec', type=int, default=2,
                    help='number of decoder layers')
parser.add_argument('--nonlin', type=str, default='tanh',
                    help='Non linearity, locked to tanh for LSTM')
parser.add_argument('--nlayers', type=int, default=1,
                    help='Number of SAB layers')
#optim params/data params
parser.add_argument('--opt', type=str, default='RMSProp',
                    choices=['SGD', 'RMSProp', 'Adam'])
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--lr_orth', type=float, default=None)
parser.add_argument('--alpha', type=float, default=None)
parser.add_argument('--beta0', type=float, default=0.9)
parser.add_argument('--beta1', type=float, default=0.999)
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clipping norm value')

#SAB
parser.add_argument('--attk', type=int, default=2,
                    help='SAB attend every k steps')
parser.add_argument('--trunc', type=int, default=5,
                    help='SAB truncate backprop')
parser.add_argument('--topk', type=int, default=5,
                    help='SAB select topk memories')

# logging
parser.add_argument('--loghm', type=int, default=500,
                    help='frequency to log heatmaps')
parser.add_argument('--loghmvid', action='store_true', default=False,
                    help='log heatmaps as a video')
parser.add_argument('--loggrads', type=int, default=500,
                    help='frequency to log grads')



def run():
    # set up hyperparameters for sweeps
    args = parser.parse_args()
    if args.device is not None:
        args.device = torch.device(f'cuda:{args.device}')
    if args.group is None:
        args.group = 'main'

    hyper_parameter_defaults = dict(
        opt='RMSProp',
        nonlin='relu',
        batch_size=12,
        learning_rate=0.0002,
        beta0=0.9,
        beta1=0.999,
        alpha=0.9
    )
    # create save_dir using wandb name
    if args.name is None:
        run = wandb.init(project="RecurrentAttentiveModels",
                         config=hyper_parameter_defaults,
                         group=args.group)
        wandb.config["more"] = "custom"
        # save run to get readable run name
        run.save()
        run.name = os.path.join(args.task, run.name)
        config = wandb.config
        config.save_dir = os.path.join('experiments', args.task, run.name)
        run.save()
    else:
        run = wandb.init(project="RecurrentAttentiveModels",
                         config=hyper_parameter_defaults,
                         name=args.name,
                         group=args.group)
        wandb.config["more"] = "custom"
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
    if config.device is not None:
        x_const = x_const.to(config.device)
        y_const = y_const.to(config.device)
    hms = []
    for i in range(config.iters):
        s_t = time.time()
        x, y = batch_generator(delay=config.T,
                                   n_labels=config.n_labels,
                                   seq_length=config.seq_len,
                                   batch_size=config.batch_size)

        if config.device is not None:
            x = x.to(config.device)
            y = y.to(config.device)

        model.zero_grad()
        if config.model in ['SAB']:
            if config.onehot:
                x = onehot(x, config.n_labels)
            outs, hiddens = model.forward(x)
        else:
            x = x.transpose(1, 0)
            outs, hiddens = model.forward(x)
        all_loss = loss_crit(outs.transpose(2, 1), y)
        all_loss.backward()
        losses.append(all_loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
        optimizer.step()
        if orth_optimizer is not None:
            orth_optimizer.step()
        preds = torch.argmax(outs[:, -config.seq_len:, :], dim=2)
        wandb.log({"predictions": wandb.Histogram(preds.detach().cpu().numpy())}, step=i)
        correct = torch.sum(preds == y[:, -config.seq_len:])
        acc = correct/float(y.shape[0]*config.seq_len)
        accs.append(acc)

        # log gradients
        if i % config.loggrads == 0 or i % config.loghm == 0:
            model.zero_grad()
            if config.model in ['SAB']:
                if config.onehot:
                    x_const_onehot = onehot(x_const, config.n_labels)
                outs, hiddens = model.forward(x_const_onehot)
            else:
                outs, hiddens = model.forward(x_const.transpose(1, 0))

        # log heat maps for attention models
        if i % config.loghm == 0 and config.model in ['MemRNN', 'SAB']:
            hm = construct_heatmap_data(model.alphas).cpu().clone()
            #fig1 = xp.imshow(1-hm, color_continuous_scale=px.colors.sequential.Greys)

            fig_hm = go.Figure(go.Heatmap(z=hm,
                                          x=list(range(hm.shape[1])),
                                          y=list(range(1, hm.shape[0]+1)),
                                          showscale=False))
            fig_hm.update_layout(title="",
                                 xaxis_title="Attention step",
                                 yaxis_title="timestep")
            wandb.log({'heat map': fig_hm}, step=i)
            #wandb.log({'heat map': wandb.plots.HeatMap(x_labels = range(len(model.rnn.alphas)),
            #                                           y_labels = range(len(model.rnn.alphas)),
            #                                           matrix_values=hm)})
            if config.loghmvid:
                hms.append(hm)

        if i % config.loggrads == 0:
            labels_loss = loss_crit(outs[:, -config.seq_len:, :].transpose(2, 1),
                                    y_const[:, -config.seq_len:])
            labels_loss.backward(retain_graph=True)
            wandb.log({'label loss': labels_loss.item()}, step=i)

            grads = [h.grad.data.norm(2).clone().cpu() for h in hiddens]
            fig = go.Figure(data=go.Scatter(x=list(range(len(grads))),
                                            y=grads,
                                            name='Grads'))
            fig.update_layout(title='Gradient flow (update={})'.format(i),
                              xaxis=dict(title='t'),
                              yaxis=dict(title='$dL/dh$'))
            if config.model in ['MemRNN']:
                (grads_ablated_rec,
                 grads_ablated_attn) = perform_synthtask_ablations(
                    model,
                    x_const.transpose(1, 0),
                    y_const,
                    loss_crit,
                    config.seq_len
                )

                fig.add_trace(go.Scatter(x=list(range(len(grads_ablated_rec))),
                                         y=grads_ablated_rec,
                              name='Ablated Recurrence Grads'))

                model.zero_grad()
                fig.add_trace(go.Scatter(x=list(range(len(grads_ablated_attn))),
                                         y=grads_ablated_attn,
                              name='Ablated Attention Grads'))


            fig.show()
            wandb.log({'grads': fig}, step=i)


        print('Update {}, Time for Update: {} , Average Loss: {}, Accuracy: {}'
              .format(i + 1, time.time() - s_t, all_loss.item(), acc))

        wandb.log({"loss": all_loss.item()}, step=i)
        wandb.log({"accuracy": acc.item()}, step=i)
    if config.model in ['MemRNN', 'SAB'] and config.loghmvid:
        hms_vid = 255*torch.stack(hms, dim=0).unsqueeze(1).detach().cpu().numpy()
        wandb.log({"video": wandb.Video(hms_vid, fps=4, format="gif")})


if __name__ == '__main__':
    torch.random.manual_seed(100)
    torch.cuda.manual_seed(100)
    np.random.seed(100)
    run()
