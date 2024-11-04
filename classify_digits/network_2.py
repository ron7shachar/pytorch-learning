"""
network3.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

# Standard library
import random

# Third-party libraries
# import numpy as np
import torch
from sympy.codegen.ast import float32

##################################################################### test ########################################






















class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        # self.record = Record("Network")
        # device = torch.device("cuda")
        # self.num_layers = len(sizes)
        # self.sizes = sizes
        # self.biases = [torch.randn(y, 1).to(device) for y in sizes[1:]]
        # self.weights = [torch.randn(y, x).to(device)
        #                 for x, y in zip(sizes[:-1], sizes[1:])]

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [torch.randn(y, 1,dtype=torch.float32 ,requires_grad=True) for y in sizes[1:]]
        self.weights = [torch.randn(y, x,dtype=torch.float32,requires_grad=True)
                        for x, y in zip(sizes[:-1], sizes[1:])]



    def SGD(self, training_data, epochs, mini_batch_size, learning_rate,
            test_data=None, print_ = True):
        max_e = 0
        self.learning_rate = learning_rate
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)
            if test_data:
                evaluation = self.evaluate(test_data)
                error = sum(v[True] for v in evaluation.values())
                if error > max_e:
                    max_e = error
                    structure = {"sizes": self.sizes , "epochs" : j , "batch_size": mini_batch_size, "eta": learning_rate}
                    # my_object = {"weights": self.weights, "biases": self.biases,"error": error,"structure":structure}
                    # self.record.set_record(my_object)
                if print_:
                    print("Epoch {0}: {1} / {2}".format(
                        j,error , n_test))
            else:
                print("Epoch {0} complete".format(j))
        return evaluation

    def update_mini_batch(self, mini_batch):
        # Extract x and y from mini_batch
        x = [x_ for x_, _ in mini_batch]
        y = [y_ for _, y_ in mini_batch]


        # Convert to tensors
        x = torch.stack([x_.clone().detach() for x_ in x]).T
        y = torch.stack([y_.clone().detach() for y_ in y]).T

        # Backpropagation
        # delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        self.learn(x, y)

    def feedforward(self, x,no_grad = False):
        a=x
        for biases, weights in zip(self.biases, self.weights):
            if no_grad:
                weights.requires_grad_(False)
                biases.requires_grad_(False)
            else:
                weights.requires_grad_(True)
                biases.requires_grad_(True)
            a = torch.sigmoid(weights @ a + biases)
        return a

    def loss(self,pred, target):
        e = pred - target
        return torch.mean(e.T @ e)

    def backpropagate(self, loss):
        loss.backward()
        for biases, weights in zip(self.biases, self.weights):
            dw = weights.grad
            db = biases.grad
            with torch.no_grad():
                weights -= self.learning_rate * dw
                biases -= self.learning_rate * db
            weights.grad.zero_()
            biases.grad.zero_()

    def learn(self,x, target):
        pred = self.feedforward( x)
        loss_ = self.loss(pred, target)
        self.backpropagate( loss_)

    def evaluate(self, test_data) -> """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.""":
        test_results = [(torch.argmax(self.feedforward(x.unsqueeze(1),no_grad=True)), y)
                        for (x, y) in test_data]

        evaluation = {}
        for (x_, y) in test_results:
            x = x_.item()
            if x not in evaluation.keys():
                evaluation[x] = {True: 0, False: 0}
            evaluation[x][x == y] +=1
        return evaluation









