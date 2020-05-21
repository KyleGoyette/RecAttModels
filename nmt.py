import torch
import torch.nn as nn
import torchtext.datasets as datasets
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator, ReversibleField
import os
import argparse
import spacy
import wandb
import revtok
from experiment import NMTExperiment
from common import train_nmt, eval_nmt
from models.NMTModels import Seq2Seq, RNNEncoder, RNNDecoder

parser = argparse.ArgumentParser(description='Neural machine translation')
# task params
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--nepochs', type=int, default=5)

# model params
parser.add_argument('--model', type=str,
                    choices=['RNN', 'LSTM', 'ORNN', 'MemRNN'],
                    default='RNN')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden units')
parser.add_argument('--nhead', type=int, default=2,
                    help='attention heads')
parser.add_argument('--nenc', type=int, default=1,
                    help='number of encoder layers')
parser.add_argument('--ndec', type=int, default=1,
                    help='number of decoder layers')
parser.add_argument('--nonlin', type=str, default='tanh',
                    help='Non linearity, locked to tanh for LSTM')
parser.add_argument('--demb', type=int, default=1024,
                    help='embedding vector size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout for embedding layers')
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
    args = parser.parse_args()
    hyper_parameter_defaults = dict(
        opt='RMSProp',
        nonlin='relu',
        batch_size=12,
        learning_rate=0.0002,
        betas=(0.5, 0.999),
        alpha=0.9
    )

    # wandb
    if args.name is None:
        run = wandb.init(project="rec-att-project",
                   config=hyper_parameter_defaults)
        wandb.config["more"] = "custom"
        # save run to get readable run name
        run.save()
        run.name = os.path.join('NMT', run.name)
        config = wandb.config
        config.save_dir = os.path.join('experiments', 'NMT', run.name)
        run.save()
    else:
        run = wandb.init(project="rec-att-project",
                   config=hyper_parameter_defaults,
                   name=args.name)
        wandb.config["more"] = "custom"
        run.name = os.path.join('NMT', run.name)
        config = wandb.config
        config.save_dir = os.path.join('experiments', 'NMT', args.name)
        run.save()

    # update config object with args
    wandb.config.update(args, allow_val_change=True)

    # set up language
    try:
        spacy_en = spacy.load('en')
    except OSError as e:
        print(e)
        print('Downloading model...')
        os.system('python -m spacy download en')
        spacy_en = spacy.load('en')
    try:
        spacy_de = spacy.load('de')
    except OSError as e:
        print(e)
        print('Downloading model...')
        os.system('python -m spacy download de')
        spacy_de = spacy.load('de')

    def tokenize_de(text):
        """
        Tokenizes German text from a string into a list of strings (tokens) and reverses it
        """

        return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

    def tokenize_en(text):
        """
        Tokenizes English text from a string into a list of strings (tokens)
        """

        return [tok.text for tok in spacy_en.tokenizer(text)]

    SRC = Field(tokenize_de,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    TRG = Field(tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    train_data, val_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                   fields=(SRC, TRG))

    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, val_data, test_data),
        batch_size=16)
    config.inp_size = len(SRC.vocab)
    config.out_size = len(TRG.vocab)

    # create experiment object
    experiment = NMTExperiment(config)
    model = experiment.model
    wandb.watch(model)
    criterion = nn.CrossEntropyLoss()
    for i in range(config.nepochs):
        train_loss = train_nmt(experiment.model, train_iterator, experiment.optimizer, criterion, run)
        val_loss = eval_nmt(model, valid_iterator, criterion, run)

        print(f'Epoch: {i} Train Loss: {train_loss} Val Loss {val_loss}')




if __name__ == '__main__':
    run()

