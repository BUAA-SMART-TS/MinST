import torch
import torch.nn as nn


def normalize_adj_mats(adj_mats):
    mask = (adj_mats > 1e-3).float()
    adj_mats = torch.softmax(adj_mats, dim=1) * mask
    adj_mats = (1.0 / (adj_mats.sum(dim=1, keepdim=True) + 1e-8)) * adj_mats
    return adj_mats


def create_activation(activation):
    if activation == 'Sigmoid': return nn.Sigmoid()
    if activation == 'ReLU': return nn.ReLU()
    if activation == 'Tanh': return nn.Tanh()
    raise Exception('unknown activation!')
