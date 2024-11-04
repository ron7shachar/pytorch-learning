from logging import ERROR
from typing import Iterator, List
from loss_functions import _Loss
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter

def forward_function_w_b(layer,x):
    return layer.weights @ x + layer.biases
def forward_function_b(layer,x):
    return  x + layer.biases
def forward_function(layer,x):
    return  x
def activation_function(x):
    return x


class Model(nn.Module):
    def __init__(self, sizes : [] = None, activation_function= activation_function ,loss_function : _Loss = None,output_function = activation_function, parameters = None):
        super(Model, self).__init__()
        self.sizes = sizes
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.output_function = output_function
        self.input_forward_function = forward_function
        self.hidden_forward_function = forward_function_w_b
        self.output_forward_function = forward_function_w_b
        self.layers = []
        if parameters is None:self.set_parameters()
        self.biases = [torch.randn(y, 1, dtype=torch.float32, requires_grad=True) for y in sizes[1:]]
        self.weights = [torch.randn(y, x, dtype=torch.float32, requires_grad=True)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def set_parameters(self):
        input_layer = InputLayer(self.sizes[0],activation_function, self.input_forward_function )
        self.layers.append(input_layer)
        for size in self.sizes[1:-1]:
            input_layer = HiddenLayer(size, input_layer , self.activation_function,self.hidden_forward_function)
            self.layers.append(input_layer)
        self.layers.append(OutputLayer(self.sizes[-1], input_layer ,self.activation_function, self.output_forward_function,self.output_function))

    def forward(self, x):
        a = x.T
        for layer in self.layers:
            a = layer.forward(a)




        # for biases, weights in zip(self.biases, self.weights):
        #     a = torch.sigmoid(weights @ a + biases)

        return a.T

    def parameters(self, recurse: bool = True) -> list[Tensor]:
        parameters_ = []
        for layer in self.layers:
            parameters_ += layer.parameters()
        return parameters_
        # return self.weights + self.biases + self.layers[0].parameters()

    def loss(self,pred, target):
        return self.loss_function(pred, target)


class Layer(nn.Module):
    def __init__(self , size : int , activation_function= activation_function ,forward = forward_function):
        super(Layer, self).__init__()
        self.size = size
        self.activation_function = activation_function
        self.forward_function = forward


    def forward(self, x):
        # return self.activation_function(self.forward_function(self,x))
        a = self.activation_function(self.forward_function(self,x))
        return a


    def parameters(self) :
        return []


class InputLayer(Layer):
    def __init__(self , size : int , activation_function= activation_function,forward = forward_function_b ):
        super(InputLayer, self).__init__( size , activation_function, forward)
        # self.biases = torch.randn(size, 1, dtype=torch.float32, requires_grad=True)

    # def parameters(self):
    #     return [self.biases]


class HiddenLayer(Layer):
    def __init__(self , size : int ,input_layer : Layer, activation_function= activation_function,forward = forward_function_w_b):
        super(HiddenLayer, self).__init__( size , activation_function,forward )
        self.input_layer = input_layer
        self.biases = torch.randn(size, 1, dtype=torch.float32, requires_grad=True)
        self.weights = torch.randn(size,input_layer.size , dtype=torch.float32, requires_grad=True)
    def parameters(self):
        return [self.biases,self.weights]


class OutputLayer(HiddenLayer):
    def __init__(self, size: int, input_layer: Layer, activation_function=activation_function,
                 forward= forward_function_w_b ,output_function= activation_function):
        super(OutputLayer, self).__init__(size ,input_layer, activation_function,forward)
        self.output_function = output_function

    def forward(self, x):
        return self.output_function(super(OutputLayer, self).forward(x))



