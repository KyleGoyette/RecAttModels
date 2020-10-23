import torch
import numpy as np
import json
from scipy.special import softmax
import pandas as pd

def generate_copy_batch(delay, n_labels, seq_length, batch_size):
    seq = torch.randint(low=1,
                        high=n_labels+1,
                        size=(batch_size, seq_length),
                        dtype=torch.long)
    delay_zeros = torch.zeros((batch_size, delay), dtype=torch.long)
    tailing_input_zeros = torch.zeros((batch_size, seq_length - 1),
                                      dtype=torch.long)
    label_zeros = torch.zeros((batch_size, delay + seq_length),
                              dtype=torch.long)
    markers = (n_labels+1)*torch.ones((batch_size, 1), dtype=torch.long)
    x = torch.cat((seq, delay_zeros, markers, tailing_input_zeros), dim=1)
    y = torch.cat((label_zeros, seq), dim=1)
    return x, y


def generate_denoise_batch(delay, n_labels, seq_length, batch_size):
    seq = torch.randint(low=1,
                        high=n_labels+1,
                        size=(batch_size, seq_length),
                        dtype=torch.long)
    input_zeros = torch.zeros((batch_size, delay), dtype=torch.long)
    for j in range(batch_size):
        inds = sorted(np.random.choice(delay, size=seq_length, replace=False))
        input_zeros[j, inds] = seq[j, :]
    tailing_input_zeros = torch.zeros((batch_size, seq_length - 1),
                                      dtype=torch.long)
    label_zeros = torch.zeros((batch_size, delay),
                              dtype=torch.long)
    markers = (n_labels+1)*torch.ones((batch_size, 1), dtype=torch.long)
    x = torch.cat((input_zeros, markers, tailing_input_zeros), dim=1)
    y = torch.cat((label_zeros, seq), dim=1)
    return x, y


def normal(tensor, mean=0, std=1):
    """Fills the input Tensor or Variable with values drawn from a normal distribution with the given mean and std
    Args:
        tensor: a n-dimension torch.Tensor
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.normal(w)
    """
    if isinstance(tensor, Variable):
        normal(tensor.data, mean=mean, std=std)
        return tensor
    else:
        return tensor.normal_(mean, std)


def perform_synthtask_ablations(model, x, y, loss_crit, seq_len):
    # perform recurrence ablation
    (out_ablated_rec,
     hids_ablated_rec) = model.forward(x, ablate_recurrence=True)

    labels_loss_ablated_rec = loss_crit(
        out_ablated_rec[:, -seq_len:, :].transpose(2, 1),
        y[:, -seq_len:]
    )
    model.zero_grad()
    labels_loss_ablated_rec.backward(retain_graph=True)
    grads_ablated_rec = [h.grad.data.norm(2).clone().cpu()
                         for h in hids_ablated_rec]

    # perform recurrence ablation
    (out_ablated_attn,
     hids_ablated_attn) = model.forward(x,
                                        ablate_attention=True)
    labels_loss_ablated_attn = loss_crit(
        out_ablated_attn[:, -seq_len:, :].transpose(2, 1),
        y[:, -seq_len:]
    )
    labels_loss_ablated_attn.backward(retain_graph=True)
    grads_ablated_attn = [h.grad.data.norm(2).clone().cpu()
                          for h in hids_ablated_attn]

    return grads_ablated_rec, grads_ablated_attn
