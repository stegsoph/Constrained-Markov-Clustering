import numpy as np

def generate_int_labels(y):

    labels_list = np.unique(y)

    for idx, label in enumerate(labels_list):
        y[y == label] = idx
    y = y.reshape((len(y),)).astype(int)

    return y


def partition_to_labels(V):
    temp_states = np.tile(np.arange(V.shape[1]), [V.shape[0], 1])
    labels = temp_states[V.astype(bool)]

    return labels