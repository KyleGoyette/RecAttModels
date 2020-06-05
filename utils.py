import torch
import numpy as np
import altair as alt
import altair_viewer
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


def grad_visualization(grads, attention):
    """ Produces a vega-lite plot for visualizing gradients.
    :param grads: list of length T gradients values
    :param attention: T x T matrix of attention weights (lower triangular
    :return:
    """

    select_point = alt.selection_single(
        on='mouseover', nearest=True, fields=['x2', 'y2'], empty="none"
    )

    attention_jsonlist = attention_matrix_to_jsonlist(attention, grads)
    attention_data = alt.Data(values=attention_jsonlist)
    #print(attention_data)
    attention_data_jsoned = {}
    for i in range(len(attention_jsonlist)):
        attention_data_jsoned.update({'id': i,
                                      'x2': attention_jsonlist[i]['x2']
                                      })

    lookup_data = alt.LookupData(attention_data, key='id')
    chart2 = alt.Chart(attention_data).mark_rule().encode(
        x='x1:T',
        y='y1:Q',
        x2='x2:Q',
        y2='y2:Q',
        opacity='attention strength:Q'
    ).transform_filter(
        select_point
    )

    grads_json = json.dumps([{'x2': i, 'y2': j} for i, j in enumerate(grads)])
    grads_data = alt.Data(values=grads_json)
    chart1 = alt.Chart(grads_data).mark_circle().encode(
        x='x2:T',
        y='y2:Q',
    ).add_selection(
        select_point
    )

    chart3 = alt.Chart(grads_data).mark_line().encode(
        x='x2:T',
        y='y2:Q'
    )

    (chart2 + chart1 + chart3).resolve_scale(x='shared', y='shared').show()

def attention_matrix_to_jsonlist(attention, grads):
    print(attention.shape, len(grads))
    rows = []
    for i in range(len(grads)):
        rows += create_row(attention[i, :i], grads, i)
    return rows


def create_row(attn_row, grads, t):
    row = []
    for i, val in enumerate(list(attn_row)):
        linedef = {
            'x1': t,
            'y1': grads[t],
            'x2': i,
            'y2': grads[i],
            'attention strength': attn_row[i] if i != len(attn_row) -1 else 1
        }
        row.append(linedef)
    return row

T=10
np.random.seed(1)
x = [np.random.random(1)[0] for i in range(T)]
y = softmax(np.tril(np.random.random((T, T)), 0), axis=1)
y = np.where(y > 0.15, y, 0)
y[1,0] = 1
print(y)
grad_visualization(x,y)