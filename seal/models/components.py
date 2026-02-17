import torch.nn as nn


def create_mlp(in_dim=None, hid_dims=[], act=nn.ReLU(), dropout=0.,
               out_dim=None, batch_norm=False,
               end_with_fc=True):
    layers = []
    if len(hid_dims) > 0:
        for hid_dim in hid_dims:
            layers.append(nn.Linear(in_dim, hid_dim))
            if batch_norm: 
                layers.append(nn.BatchNorm1d(hid_dim))
            layers.append(act)
            layers.append(nn.Dropout(dropout))
            in_dim = hid_dim
    layers.append(nn.Linear(in_dim, out_dim))
    if not end_with_fc:
        layers.append(act)
        layers.append(nn.Dropout(dropout))
    mlp = nn.Sequential(*layers)
    return mlp