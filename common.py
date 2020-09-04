import os
import sys
import glob
import torch
from natsort import natsorted

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import matplotlib.pyplot as plt

from models.NMTModels import TransformerSeq2Seq

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
    tot_length = len(alphas)
    return torch.stack(
        [
            torch.cat(
                (a[:, 0].clone().detach(), a.new_zeros(tot_length - a.shape[0])),
                dim=0)
            for a in alphas
        ],
        dim=0)


def train_nmt(model, iterator, optimizer, criterion, config, run=None,
              src_field=None, trg_field=None):

    model.train()
    losses = []
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        if config.device is not None:
            src = src.to(config.device)
            trg = trg.to(config.device)

        optimizer.zero_grad()
        if isinstance(model, TransformerSeq2Seq):
            output, attention = model(src, trg[:, :-1])
            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)
        else:
            output = model(src, trg)
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].contiguous().view(-1)
        loss = criterion(output, trg)
        #print(loss.item())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if run is not None:
            run.log({'train loss': loss.item()})
        losses.append(loss.item())
        """
        if i % config.logfreq == 0 and run is not None:
            if isinstance(model, TransformerSeq2Seq):
                log_to_table(output, src, trg.view(src.shape[0], -1),
                             src_field, trg_field, run, 'Train')

            else:
                print(src.shape, trg.shape)
                log_to_table(output,
                             src.transpose(1, 0),
                             trg.view(-1, src.shape[1]).t(),
                             src_field, trg_field, run, 'Train')
        """
    run.log({'epoch train loss': np.mean(losses)})
    return np.mean(losses)


def eval_nmt(model, iterator, criterion, config, run=None,
             src_field=None, trg_field=None):

    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            if config.device is not None:
                src = src.to(config.device)
                trg = trg.to(config.device)
            if isinstance(model, TransformerSeq2Seq):
                output, _ = model(src, trg[:, :-1])
                output = output.contiguous().view(-1, output.shape[-1])
                trg = trg[:, 1:].contiguous().view(-1)
            else:
                output = model(src, trg)
                output = output[1:].view(-1, output.shape[-1])
                trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            if run is not None:
                run.log({'valid loss': loss.item()})
            losses.append(loss.item())
            """
            if i % config.logfreq and run is not None:
                if isinstance(model, TransformerSeq2Seq):
                    log_to_table(output, src, trg.view(src.shape[0],-1),
                                 src_field, trg_field, run, 'Val')
                else:
                    log_to_table(output, src.transpose(1, 0),
                                 trg.view(-1, src.shape[1]).t(),
                                 src_field, trg_field, run, 'Val')
            """
    run.log({'epoch valid loss': np.mean(losses)})
    return np.mean(losses)


def log_to_table(translation, src_tokens, trg_tokens, run):
    if isinstance(src_tokens, list):
        src_tokens = ' '.join(src_tokens)
    if isinstance(trg_tokens, list):
        trg_tokens = ' '.join(trg_tokens)
    if isinstance(translation, list):
        translation = ' '.join(translation)

    data = [src_tokens, trg_tokens, translation]
    run.log({
        f"Examples": wandb.Table(data=data,
                              columns=['Input', 'Ground Truth', "Predicted"])
    }, commit=False)


def visualize_attention(model):
    if isinstance(model, TransformerSeq2Seq):
        visualize_transformer_attention(model)
    else:
        visualize_memrnn_attention(model)

def visualize_transformer_attention(attention, src, translation):
    for i in range(attention.shape[1]):
        wandb.log({f'attention heatmap {i}': wandb.plots.HeatMap(
            x_labels=src,
            y_labels=translation,
            matrix_values=attention[0, i, :, :].detach().cpu().numpy())
        }, commit=False)
    summary_attn = torch.mean(attention[0, :, :, :], dim=0).detach().cpu().numpy()
    wandb.log({'attention heatmap summary':  wandb.plots.HeatMap(
        x_labels=src,
        y_labels=translation,
        matrix_values=summary_attn
    )
    }, commit=True)



def visualize_memrnn_attention(model):
    pass

def translate_sentence(sentence, src_field, trg_field, spacy_src, model, config, max_len=50):
    model.eval()

    if isinstance(sentence, str):
        tokens = [token.text.lower() for token in spacy_src(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(config.device)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(config.device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_indexes, trg_tokens[1:], attention


def convert_sentence_to_tensor(src, field):
    return torch.LongTensor([field.vocab.stoi[token] for token in src])


def convert_tensor_to_sentence(src, field):
    print(src.shape)
    return [field.vocab.itos[p.item()]
            for p in src if p != field.vocab.stoi[field.pad_token]]