from typing import Iterator, List

import torch
import torch.nn as nn
from torch import Tensor

loss = nn.CrossEntropyLoss()
loss = nn.MSELoss()

class BSmM(nn.Module):
    def __init__(self, sizes):
        super(BSmM, self).__init__()
        self.sizes = sizes
        self.biases = [torch.randn(y, 1, dtype=torch.float32, requires_grad=True) for y in sizes[1:]]
        self.weights = [torch.randn(y, x, dtype=torch.float32, requires_grad=True)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def forward(self, x):
        a = x.T
        for biases, weights in zip(self.biases, self.weights):
            a = torch.sigmoid(weights @ a + biases)
        return a.T

    def parameters(self, recurse: bool = True) -> list[Tensor]:
        return self.weights + self.biases

    def loss(self,pred, target):
        # target = torch.argmax(target, dim=1)
        return loss(pred, target)

