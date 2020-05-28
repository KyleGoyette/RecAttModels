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

    def __init__(self, args):
        # for each item in args that is a list, select current item.

        save_dir = args.save_dir
        if not os.path.exists(save_dir):
            create_exp_dir(save_dir)

        self.save_dir = save_dir
        self.experiment_path = self.save_dir

        state_file_list = natsorted([fn for fn in os.listdir(save_dir)
                                     if fn.endswith('.pt')])

        model = self._get_model(args)
        if args.model not in ['ORNN']:
            optimizer = self._get_optimizer(args, model.parameters())
        else:
            non_orth_parameters, log_orth_parameters = get_parameters(model)
            normal_opt = self._get_optimizer(args, non_orth_parameters)
            orth_opt = self._get_optimizer(args,log_orth_parameters,orth=True)
            optimizer = (normal_opt, orth_opt)
        scheduler = self._get_scheduler(args, optimizer)

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

        self.args = args
        self.model = model
        if args.device is not None:
            self.model = self.model.to(args.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        if args.cuda:
            self.model = self.model.cuda()

            if self.optimizer is not None:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

    # Adapted from https://github.com/veugene/DemiGORU
    def _get_optimizer(self, args, params, orth=False):
        name = args.opt
        kwargs = {'params'       : [p for p in params if p.requires_grad]}
        if not orth:
            if args.lr is not None:
                kwargs.update({'lr': args.lr})
        else:
            if args.lr_orth is not None:
                kwargs.update({'lr': args.lr_orth})
            # automatically set to 1/10th of normal optimizer
            elif args.lr is not None:
                kwargs.update({'lr': args.lr/10})
        if name.lower() == 'rmsprop':
            if args.betas is not None:
                kwargs.update({'alpha': args.alpha})
            optimizer = torch.optim.RMSprop(**kwargs)
        elif name.lower() == 'adam':
            if args.alpha is not None:
                kwargs.update({'betas': args.betas})
            optimizer = torch.optim.Adam(**kwargs)
        elif name.lower() == 'sgd':
            # add lr if none, required for sgd
            if kwargs.get('lr') is None:
                kwargs.update({'lr': 1e-4})
            optimizer = torch.optim.SGD(**kwargs)
        else:
            raise ValueError("Optimizer {} not supported.".format(name))
        return optimizer

    def _get_model(self, args):
        model_name = args.model
        if model_name == 'RNN':
            base = nn.RNNCell(input_size=args.input_size,
                              hidden_size=args.nhid,
                              nonlinearity=args.nonlin)
        elif model_name == 'ORNN':
            base = OrthogonalRNN(input_size=args.input_size,
                                 hidden_size=args.nhid,
                                 initializer_skew=henaff_init_,
                                 mode='static',
                                 param=expm)
        elif model_name == 'LSTM':
            base = nn.LSTMCell(input_size=args.input_size,
                               hidden_size=args.nhid)
        elif model_name == 'MemRNN':
            base = MemRNN(input_size=args.input_size,
                          hidden_size=args.nhid,
                          nonlinearity=args.nonlin,
                          device=args.device)
        elif model_name == 'SAB':
            model = SAB_LSTM(input_size=args.input_size,
                             hidden_size=args.nhid,
                             num_layers=args.nlayers,
                             num_classes=args.n_labels+1,
                             truncate_length=args.trunc,
                             attn_every_k=args.attk,
                             top_k=args.topk,
                             device=args.device)
            return model
        elif model_name == 'Trans':
            base = Transformer(d_model=args.input_size,
                                nhead=args.nhead,
                                num_encoder_layers=args.nenc,
                                num_decoder_layers=args.ndec,
                                dim_feedforward=args.nhid,
                                dropout=0)
        else:
            raise ValueError('Model {} not supported.'.format(model_name))

        if args.task == 'copy' or args.task == 'denoise':
            if model_name == 'Trans':
                model = base
            else:
                model = RecurrentCopyModel(base,
                                           args.nhid,
                                           args.onehot,
                                           args.n_labels,
                                           device=args.device)
        else:
            raise ValueError('Task {} not supported.'.format(args.task))
        return model

    def _get_scheduler(self, args, optimizer):
        if vars(args).get('sch_kwargs', None) is None:
            return None
        kwargs = args.sch_kwargs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               **kwargs)
        return scheduler


class NMTExperiment(Experiment):
    def _get_model(self, args):
        model_name = args.model
        if model_name == 'RNN':
            encoder = RNNEncoder(inp_size=args.inp_size,
                                 emb_size=args.demb,
                                 hid_size=args.nhid,
                                 n_layers=args.nenc,
                                 dropout=args.dropout)

            decoder = RNNDecoder(out_size=args.out_size,
                                 emb_size=args.demb,
                                 hid_size=args.nhid,
                                 n_layers=args.ndec,
                                 dropout=args.dropout)

            model = Seq2Seq(encoder=encoder, decoder=decoder)

        elif model_name == 'MemRNN':
            encoder = BidirectionalEncoder(inp_size=args.inp_size,
                                           emb_size=args.demb,
                                           enc_hid_size=args.nhid,
                                           dec_hid_size=args.nhid,
                                           dropout=args.dropout)


            attention = Attention(enc_hid_size=args.nhid,
                                  dec_hid_size=args.nhid)

            decoder = BidirectionalDecoder(out_size=args.out_size,
                                           emb_size=args.demb,
                                           enc_hid_size=args.nhid,
                                           dec_hid_size=args.nhid,
                                           dropout=args.dropout,
                                           attention=attention)

            model = AttnSeq2Seq(encoder=encoder,
                                decoder=decoder)

        elif model_name == 'Trans':
            encoder = TransformerEncoder(inp_size=args.inp_size,
                                         hid_size=args.nhid,
                                         n_layers=args.nenc,
                                         n_heads=args.nhenc,
                                         pf_dim=512,
                                         dropout=args.dropout,
                                         max_length=100)

            decoder = TransformerDecoder(output_dim=args.out_size,
                                         hid_size=args.nhid,
                                         n_layers=args.ndec,
                                         n_heads=args.nhdec,
                                         pf_dim=512,
                                         dropout=args.dropout,
                                         max_length=100)
            model = TransformerSeq2Seq(encoder=encoder,
                                       decoder=decoder,
                                       src_pad_idx=args.SRCPADIDX,
                                       trg_pad_idx=args.TRGPADIDX)

        else:
            raise ValueError('Model {} not supported.'.format(model_name))

        return model




