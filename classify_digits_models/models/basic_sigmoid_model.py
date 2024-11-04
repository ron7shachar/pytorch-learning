from typing import Iterator, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter


class BSM(nn.Module):
    def __init__(self, sizes):
        super(BSM, self).__init__()
        self.sizes = sizes
        self.biases = [torch.randn(y, 1, dtype=torch.float32, requires_grad=True) for y in sizes[1:]]
        self.weights = [torch.randn(y, x, dtype=torch.float32, requires_grad=True)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def forward(self, x):
        a = x.T
        for biases, weights in zip(self.biases, self.weights):
            a = torch.softmax(weights @ a + biases,dim=0)
        return a.T

    def parameters(self, recurse: bool = True) -> list[Tensor]:
        return self.weights + self.biases

    def loss(self,pred, target):
        """
        MSE mean((pred - target)^2)
        """
        e = pred - target
        mse = torch.mean(e @ e.T)
        return mse

