
# Standard library
import torch
from classify_digits_models.models.basic_sigmoid_model import BSM
from torch.utils.data import DataLoader
from classify_digits_models.evaluat import *
class Network(object):

    def __init__(self, sizes):
        """
        Initializes the network.
        :param sizes: A list defining the number of neurons in each layer of the network.
        """
        self.num_layers = len(sizes)  # Number of layers in the network
        self.sizes = sizes
        self.model = BSM(sizes)  # Create the model with the defined structure

    def fit(self, training_data, epochs, mini_batch_size, learning_rate, test_data, print_=True):
        """
        Performs training using the Stochastic Gradient Descent (SGD) algorithm.
        :param training_data: Training data, a list of (x, y) tuples where x is the input and y is the label.
        :param epochs: Number of epochs (iterations over the entire data).
        :param mini_batch_size: Size of each mini-batch.
        :param learning_rate: Learning rate.
        :param test_data: Optional, test data to evaluate the model's performance.
        :param print_: If True, prints training progress each epoch.
        :return: Returns the evaluation results on the test data after training.
        """
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)  # Set the optimizer
        if test_data: n_test = len(test_data)  # Number of examples in the test data

        for j in range(epochs):  # Loop over the number of epochs
            for mini_batch in DataLoader(training_data, mini_batch_size):
                self.learn(*mini_batch)
            accuracy = accuracy_score(self.model,test_data)
            if print_:
                print("Epoch {0}: accuracy {1} : sample size {2}".format(j+1,  accuracy , len(test_data)))
        return  accuracy


    def learn(self, x, target):
        """
        Performs a learning step for a given input-target pair.
        :param x: Input data.
        :param target: Target label.
        """
        pred = self.model(x)  # Model prediction
        loss_ = self.model.loss(pred, target)  # Compute the loss
        loss_.backward()  # Compute gradients
        self.optimizer.step()  # Update parameters
        self.optimizer.zero_grad()  # Reset gradients




