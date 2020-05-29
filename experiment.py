from common import (load_state,
                    load_histories,
                    create_exp_dir)
from models.models import RecurrentCopyModel, MemRNN
from models.NMTModels import (RNNDecoder, RNNEncoder, Seq2Seq,
                              BidirectionalDecoder, BidirectionalEncoder, Attention, AttnSeq2Seq,
                              TransformerEncoder, TransformerDecoder, TransformerSeq2Seq)
from models.SAB import SAB_LSTM
from models.expRNN.orthogonal import OrthogonalRNN
from models.expRNN.initialization import henaff_init_
from models.expRNN.trivializations import expm
from models.expRNN.parametrization import get_parameters

from natsort import natsorted
import os
import torch
import torch.nn as nn

class Experiment(object):

    def __init__(self, config):
        # for each item in config that is a list, select current item.

        save_dir = config.save_dir
        if not os.path.exists(save_dir):
            create_exp_dir(save_dir)

        self.save_dir = save_dir
        self.experiment_path = self.save_dir

        state_file_list = natsorted([fn for fn in os.listdir(save_dir)
                                     if fn.endswith('.pt')])

        model = self._get_model(config)
        if config.model not in ['ORNN']:
            optimizer = self._get_optimizer(config, model.parameters())
        else:
            non_orth_parameters, log_orth_parameters = get_parameters(model)
            normal_opt = self._get_optimizer(config, non_orth_parameters)
            orth_opt = self._get_optimizer(config,log_orth_parameters,orth=True)
            optimizer = (normal_opt, orth_opt)
        scheduler = self._get_scheduler(config, optimizer)

        # if path exists and some model has been saved in it, Load experiment
        if os.path.exists(save_dir) and len(state_file_list):
            # RESUME old experiment
            print('Resuming experiment from: {}'.format(save_dir))

            model, optimizer, scheduler, epoch = load_state(save_dir,
                                                            model,
                                                            optimizer,
                                                            scheduler,
                                                            best=False)
            self.epoch = epoch
            (self.train_losses, self.train_accs,
             self.val_losses, self.val_accs, self.val_hist) = load_histories(save_dir)

        # start new experiment
        else:
            (self.train_losses, self.train_accs, self.val_losses,
             self.val_accs, self.val_hist) = ([] for i in range(5))

        self.args = config
        self.model = model
        if config.device is not None:
            self.model = self.model.to(config.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        if config.cuda:
            self.model = self.model.cuda()

            if self.optimizer is not None:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

    # Adapted from https://github.com/veugene/DemiGORU
    def _get_optimizer(self, config, params, orth=False):
        name = config.opt
        kwargs = {'params'       : [p for p in params if p.requires_grad]}
        if not orth:
            if config.lr is not None:
                kwargs.update({'lr': config.lr})
        else:
            if config.lr_orth is not None:
                kwargs.update({'lr': config.lr_orth})
            # automatically set to 1/10th of normal optimizer
            elif config.lr is not None:
                kwargs.update({'lr': config.lr/10})
        if name.lower() == 'rmsprop':
            if config.alpha is not None:
                kwargs.update({'alpha': config.alpha})
            optimizer = torch.optim.RMSprop(**kwargs)
        elif name.lower() == 'adam':
            kwargs.update({'betas': (config.beta0, config.beta1)})
            optimizer = torch.optim.Adam(**kwargs)
        elif name.lower() == 'sgd':
            # add lr if none, required for sgd
            if kwargs.get('lr') is None:
                kwargs.update({'lr': 1e-4})
            optimizer = torch.optim.SGD(**kwargs)
        else:
            raise ValueError("Optimizer {} not supported.".format(name))
        return optimizer

    def _get_model(self, config):
        model_name = config.model
        if model_name == 'RNN':
            base = nn.RNNCell(input_size=config.input_size,
                              hidden_size=config.nhid,
                              nonlinearity=config.nonlin)
        elif model_name == 'ORNN':
            base = OrthogonalRNN(input_size=config.input_size,
                                 hidden_size=config.nhid,
                                 initializer_skew=henaff_init_,
                                 mode='static',
                                 param=expm)
        elif model_name == 'LSTM':
            base = nn.LSTMCell(input_size=config.input_size,
                               hidden_size=config.nhid)
        elif model_name == 'MemRNN':
            base = MemRNN(input_size=config.input_size,
                          hidden_size=config.nhid,
                          nonlinearity=config.nonlin,
                          device=config.device)
        elif model_name == 'SAB':
            model = SAB_LSTM(input_size=config.input_size,
                             hidden_size=config.nhid,
                             num_layers=config.nlayers,
                             num_classes=config.n_labels+1,
                             truncate_length=config.trunc,
                             attn_every_k=config.attk,
                             top_k=config.topk,
                             device=config.device)
            return model
        elif model_name == 'Trans':
            base = Transformer(d_model=config.input_size,
                                nhead=config.nhead,
                                num_encoder_layers=config.nenc,
                                num_decoder_layers=config.ndec,
                                dim_feedforward=config.nhid,
                                dropout=0)
        else:
            raise ValueError('Model {} not supported.'.format(model_name))

        if config.task == 'copy' or config.task == 'denoise':
            if model_name == 'Trans':
                model = base
            else:
                model = RecurrentCopyModel(base,
                                           config.nhid,
                                           config.onehot,
                                           config.n_labels,
                                           device=config.device)
        else:
            raise ValueError('Task {} not supported.'.format(config.task))
        return model

    def _get_scheduler(self, config, optimizer):
        if vars(config).get('sch_kwargs', None) is None:
            return None
        kwargs = config.sch_kwargs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               **kwargs)
        return scheduler


class NMTExperiment(Experiment):
    def _get_model(self, config):
        model_name = config.model
        if model_name == 'RNN':
            encoder = RNNEncoder(inp_size=config.inp_size,
                                 emb_size=config.demb,
                                 hid_size=config.nhid,
                                 n_layers=config.nenc,
                                 dropout=config.dropout)

            decoder = RNNDecoder(out_size=config.out_size,
                                 emb_size=config.demb,
                                 hid_size=config.nhid,
                                 n_layers=config.ndec,
                                 dropout=config.dropout)

            model = Seq2Seq(encoder=encoder, decoder=decoder)

        elif model_name == 'MemRNN':
            encoder = BidirectionalEncoder(inp_size=config.inp_size,
                                           emb_size=config.demb,
                                           enc_hid_size=config.nhid,
                                           dec_hid_size=config.nhid,
                                           dropout=config.dropout)


            attention = Attention(enc_hid_size=config.nhid,
                                  dec_hid_size=config.nhid)

            decoder = BidirectionalDecoder(out_size=config.out_size,
                                           emb_size=config.demb,
                                           enc_hid_size=config.nhid,
                                           dec_hid_size=config.nhid,
                                           dropout=config.dropout,
                                           attention=attention)

            model = AttnSeq2Seq(encoder=encoder,
                                decoder=decoder)

        elif model_name == 'Trans':
            encoder = TransformerEncoder(inp_size=config.inp_size,
                                         hid_size=config.nhid,
                                         n_layers=config.nenc,
                                         n_heads=config.nhenc,
                                         pf_dim=512,
                                         dropout=config.dropout,
                                         max_length=100)

            decoder = TransformerDecoder(output_dim=config.out_size,
                                         hid_size=config.nhid,
                                         n_layers=config.ndec,
                                         n_heads=config.nhdec,
                                         pf_dim=512,
                                         dropout=config.dropout,
                                         max_length=100)
            model = TransformerSeq2Seq(encoder=encoder,
                                       decoder=decoder,
                                       src_pad_idx=config.SRCPADIDX,
                                       trg_pad_idx=config.TRGPADIDX)

        else:
            raise ValueError('Model {} not supported.'.format(model_name))

        return model




