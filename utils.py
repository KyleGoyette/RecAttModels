import torch
import numpy as np
import random
import time
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


def generate_add_batch()