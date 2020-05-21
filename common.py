import os
import sys
import glob
import torch
from natsort import natsorted

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def load_state(path, model, optimizer=None, scheduler=None, best=False):
    '''
    Loads state of an experiment from either the best or latest file
    :param path: path to experiment directory
    :param model: model which data will be loaded into
    :param optimizer: optimizer which will have state
    :param scheduler:
    :param best: boolean indicating if you want best or latest file
    :return: model, optimizer, scheduler and epoch
    '''
    if best:
        state = torch.load(os.path.join(path, 'best_model.pt'))
    else:
        latest_file = get_newest_model(path)
        print('Loading state from {}'.format(latest_file))
        state = torch.load(latest_file)
    model.load_state_dict(state['model_state'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler_state'])

    epoch = state['epoch']
    return model, optimizer, scheduler, epoch


def get_newest_model(path):
    files = glob.glob(os.path.join(path, '*.pt'))
    if os.path.join(path, 'best_model.pt') in files:
        files.remove(os.path.join(path, 'best_model.pt'))

    files = natsorted(files)
    return files[-1]



def load_histories(save_dir):
    x = torch.load(os.path.join(save_dir, 'exp_hist'))
    train_losses = x['train loss']
    train_metric = x['train metric']
    val_losses = x['val loss']
    val_metric = x['val metric']
    try:
        val_hist = x['val hist']
    except:
        val_hist = []
    return train_losses, train_metric, val_losses, val_metric, val_hist


def create_exp_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        pass
    with open(os.path.join(save_dir, "cmd.sh"), 'w') as f:
        f.write(' '.join(sys.argv))
    return save_dir



def load_model(path, model):
    '''
    Loads model
    :param path: path to model to load
    :param model: model which data will be loaded into
    :return: model
    '''
    state = torch.load(path)
    model.load_state_dict(state['model_state'])
    return model


def onehot(x, n_labels):
    #print(x, n_labels + 1)
    return F.one_hot(x, num_classes=n_labels+2).float()

class modrelu(nn.Module):
    def __init__(self, features):
        # For now we just support square layers
        super(modrelu, self).__init__()
        self.features = features
        self.b = nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self, inputs):
        norm = torch.abs(inputs)
        biased_norm = norm + self.b
        magnitude = nn.functional.relu(biased_norm)
        phase = torch.sign(inputs)

        return phase * magnitude


def construct_heatmap_data(alphas):
    T = len(alphas)

    return torch.stack([torch.cat((a[:, 0].clone().detach(), a.new_zeros(T - a.shape[0])),dim=0) for a in alphas], dim=0)


def train_nmt(model, iterator, optimizer, criterion, run=None):

    model.train()
    losses = []
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg)
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        if run is not None:
            run.log({'train loss': loss.item()})
        losses.append(loss.item())
    run.log({'epoch train loss': np.mean(losses)})
    return np.mean(losses)


def eval_nmt(model, iterator, criterion, run=None):

    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg)
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            if run is not None:
                run.log({'valid loss': loss.item()})
            losses.append(loss.item())
    run.log({'epoch valid loss': np.mean(losses)})
    return np.mean(losses)

