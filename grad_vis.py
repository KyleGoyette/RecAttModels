import numpy as np
import wandb
from scipy.special import softmax
from sklearn.naive_bayes import MultinomialNB
from wandb.sklearn import learning_curve
from math import inf

chart_limit = wandb.Table.MAX_ROWS

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


def grad_attention_viz(grads, attention, threshold=0.005):
    """
    Plots gradients, shows
    :param grads: list of gradient values length T
    :param attention: TxT matrix of attention weights
    :return:
    """

    assert len(grads) == attention.shape[0] +1


    data = []
    count = 0

    for i, grad_val in enumerate(grads):
        complete = False
        for j in range(attention.shape[1]):
            if i == len(grads) - 1:
                data.append((i, grads[i], 0, 0, 0, 0, 0))
                count += 1
            elif attention[j,i] >= threshold:
                data.append((i, grads[i], j+1, grads[j+1],  i, grads[i], attention[j, i]))
                count += 1
        if not complete:
            data.append((i, grads[i], 0, 0, 0, 0, 0))
            count += 1


            if count >= chart_limit:
                wandb.termwarn(
                    "wandb uses only the first %d datapoints to"
                    " create the plots." % wandb.Table.MAX_ROWS
                )
                break
    return wandb.Table(
            columns=['step', 'grad', 'origin_x', 'origin_y',
                     'desitination_x', 'destination_y', 'attn'],
            data=data
        )


def grad_attention_viz_name(grads, attention, name, threshold=0.005):
    """
    Plots gradients, shows
    :param grads: list of gradient values length T
    :param attention: TxT matrix of attention weights
    :return:
    """

    assert len(grads) == attention.shape[0] +1
    assert name == 'wandb/grad_bar_plot/v1' or name == 'wandb/grad_line_plot/v1'
    data = []
    count = 0

    for i, grad_val in enumerate(grads):
        complete = False
        for j in range(attention.shape[1]):
            if i == len(grads) - 1:
                data.append((i, grads[i], 0, 0, 0, 0, 0))
                count += 1
            elif attention[j,i] >= threshold:
                data.append((i, grads[i], j+1, grads[j+1],  i, grads[i], attention[j, i]))
                count += 1
        if not complete:
            data.append((i, grads[i], 0, 0, 0, 0, 0))
            count += 1


            if count >= chart_limit:
                wandb.termwarn(
                    "wandb uses only the first %d datapoints to"
                    " create the plots." % wandb.Table.MAX_ROWS
                )
                break
    return wandb.visualize(name, wandb.Table(
            columns=['step', 'grad', 'origin_x', 'origin_y',
                     'desitination_x', 'destination_y', 'attn'],
            data=data
        ))


'''

wandb.init(project='test-project')
T=5
np.random.seed(1)
x = [np.random.random(1)[0] for i in range(T)]
y = np.tril(np.ones((T-1, T-1)), 0) - np.triu(inf*np.ones((T-1, T-1)),1)
#y = np.where(y > 0.15, y, 0)
print(len(x), y)
y = softmax(y,axis=1)
print(y)
wandb.log({'grad vis': grad_attention_viz(x, y)})

def dummy_classifier(request):
    nb = MultinomialNB()
    x_train = [[1,2],[1,2],[1,2],[1,2],[2,3],[3,4],[3,4],[3,4],[3,4],[3,4],[3,4]]
    y_train = [0,0,0,0,0,1,1,1,1,1,1]
    nb.fit(x_train, y_train)
    x_test = [[4,5], [5,6]]
    y_test = [0,1]
    y_probas = nb.predict_proba(x_test)
    y_pred = nb.predict(x_test)
    return (nb, x_train, y_train, x_test, y_test, y_pred, y_probas)

(nb, x_train, y_train, x_test, y_test, y_pred, y_probas) = dummy_classifier(None)
lc_table = learning_curve(nb, x_train, y_train)


wandb.log({'roc': wandb.plots.ROC(y_test, y_probas)})

'''