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
        self.biases = [torch.randn(y, 1,requires_grad=True) for y in sizes[1:]]
        self.weights = [torch.randn(y, x,requires_grad=True)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(w @ a + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, print_ = True,record = False):
        max_e = 0
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
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                evaluation = self.evaluate(test_data)
                error = sum(v[True] for v in evaluation.values())
                if error > max_e:
                    max_e = error
                    structure = {"sizes": self.sizes , "epochs" : j , "batch_size": mini_batch_size, "eta": eta}
                    # my_object = {"weights": self.weights, "biases": self.biases,"error": error,"structure":structure}
                    # self.record.set_record(my_object)
                if print_:
                    print("Epoch {0}: {1} / {2}".format(
                        j,error , n_test))
            else:
                print("Epoch {0} complete".format(j))
        return evaluation

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [torch.zeros(b.shape) for b in self.biases]
        nabla_w = [torch.zeros(w.shape) for w in self.weights]

        # Extract x and y from mini_batch
        x = [x_ for x_, _ in mini_batch]
        y = [y_ for _, y_ in mini_batch]

        # Convert to tensors
        x = torch.stack([x_.clone().detach() for x_ in x]).T
        y = torch.stack([y_.clone().detach() for y_ in y]).T

        # Backpropagation
        delta_nabla_b, delta_nabla_w = self.backprop(x, y)

        nabla_b = [dnb.sum(dim=1, keepdim=True) for dnb in delta_nabla_b]
        nabla_w = delta_nabla_w


        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]


    def backprop(self,x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [torch.zeros(b.shape) for b in self.biases]
        nabla_w = [torch.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [activation] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = w @ activation + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = delta @ activations[-2].T
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = self.weights[-l+1].T @ delta * sp
            nabla_b[-l] = delta
            nabla_w[-l] = delta @ activations[-l-1].T

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(torch.argmax(self.feedforward(x.unsqueeze(1))), y)
                        for (x, y) in test_data]

        evaluation = {}
        for (x_, y) in test_results:
            x = x_.item()
            if x not in evaluation.keys():
                evaluation[x] = {True: 0, False: 0}
            evaluation[x][x == y] +=1

        return evaluation

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+torch.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))