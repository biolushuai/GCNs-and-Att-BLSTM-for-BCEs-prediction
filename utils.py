import numpy as np


def propress_data(data):
    num_complex = len(data)
    for idx_complex in range(num_complex):
        # feat1 = data[idx_complex]['ag_feat'][:, :40]
        # feat2 = data[idx_complex]['ag_feat'][:, 117:119]
        # data[idx_complex]['ag_feat'] = np.concatenate((feat1, feat2), 1)

        # Use only up to 15 neighbors during convolution
        data[idx_complex]["ag_nh_indices"] = data[idx_complex]["ag_nh_indices"][:, :15, :]
        data[idx_complex]["ag_edge_feat"] = data[idx_complex]["ag_edge_feat"][:, :15, :]


def down_sample(labels, ratio=1):
    negative_idxs = []
    positive_idxs = []
    for i, p in enumerate(labels):
        if p[-1] == 1:
            positive_idxs.append(i)
        else:
            negative_idxs.append(i)
    negative_idxs = np.array(negative_idxs)
    if np.sum(labels[:, -1] == 1) * ratio > len(negative_idxs):
        selected_idxs = negative_idxs
    else:
        selected_idxs = np.random.choice(negative_idxs, size=np.sum(labels[:, -1] == 1) * ratio, replace=False)
    all_idxs = []
    all_idxs.extend(positive_idxs)
    all_idxs.extend(selected_idxs)
    return np.array(labels[all_idxs])
