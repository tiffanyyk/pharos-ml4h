import torch
from torch import nn as nn


def create_mlp(
    input_dim: int,
    layers_dim: list,
    dropout_ratio: float,
    batch_norm: bool = False,
    leaky_relu: bool = False,
    pre_bn: bool = False,
    special_init=False,
    special_activation=None,
):
    """Create a Multi Layer Perceptron.

    Parameters
    ----------
    input_dim: int
        Input dimension.
    layers_dim: list
        A list of layer widths.
    dropout_ratio: float
        Dropout ratio.
    batch_norm : bool
        Add batchnorm between modules.
    leaky_relu : bool
        Use leaky ReLU instead of ReLU.
    pre_bn : bool
        Add a batchnorm before the other modules.
    special_init : bool
        If true, use N(0,1) for initializing biases and Glorot initialization for
        weights.

    Returns
    -------
    torch.nn.Sequential
        A sequential container with the resulting MLP.
    """
    output_dim = layers_dim[-1]

    def create_activation():
        if special_activation and special_activation.lower() == "tanh":
            return nn.Tanh()
        if leaky_relu:
            return nn.LeakyReLU(negative_slope=0.1)
        return nn.ReLU()

    if len(layers_dim) == 1:
        res = nn.Linear(input_dim, output_dim)
        res = nn.Sequential(res, create_activation())

    else:
        layers_dim = [input_dim] + layers_dim
        layers = []
        if pre_bn and batch_norm:
            layers.append(nn.BatchNorm1d(layers_dim[0]))
        for l_i, (ld, ld_p1) in enumerate(zip(layers_dim[:-1], layers_dim[1:])):
            if l_i > 0:
                layers.append(create_activation())
                layers.append(nn.Dropout(p=dropout_ratio))
            new_linear = nn.Linear(ld, ld_p1)
            if special_init:
                nn.init.normal_(new_linear.bias, 0.0, 0.1)
                nn.init.xavier_normal_(new_linear.weight)
            layers.append(new_linear)
            if batch_norm and l_i < (len(layers_dim) - 2):
                layers.append(nn.BatchNorm1d(ld_p1))
        res = nn.Sequential(*layers)
    return res
