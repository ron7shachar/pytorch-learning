
# Standard library
import random
import torch
from classify_digits_models.models.basic_sigmoid_model import BSM

class Network(object):

    def __init__(self, sizes):
        """
        Initializes the network.
        :param sizes: A list defining the number of neurons in each layer of the network.
        """
        self.num_layers = len(sizes)  # Number of layers in the network
        self.sizes = sizes            # Structure of the network layers
        self.model = BSM(self.sizes)  # Create the model with the defined structure

    def fit(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None, print_=True):
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
        max_e = 0  # Variable to store the maximum error
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)  # Set the optimizer
        if test_data: n_test = len(test_data)  # Number of examples in the test data
        n = len(training_data)  # Number of examples in the training data
        training_data =  [(x,y) for x,y in zip(*test_data.get_data())]
        for j in range(epochs):  # Loop over the number of epochs
            random.shuffle(training_data)  # Shuffle the data at the start of each epoch
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)
            if test_data:
                evaluation = self.evaluate(test_data)
                error = sum(v[True] for v in evaluation.values())  # Calculate the number of incorrect answers
                if error > max_e:
                    max_e = error
                    structure = {"sizes": self.sizes, "epochs": j, "batch_size": mini_batch_size, "eta": learning_rate}
                if print_:
                    print("Epoch {0}: {1} / {2}".format(j, error, n_test))
            else:
                print("Epoch {0} complete".format(j))
        return evaluation

    def update_mini_batch(self, mini_batch):
        """
        Updates weights for a single mini-batch.
        :param mini_batch: A single mini-batch from the training data.
        """
        x, y = zip(*mini_batch)
        x = torch.stack([x_.detach().clone() for x_ in x])
        y = torch.stack([y_.detach().clone() for y_ in y])
        self.learn(x, y)

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


    def evaluate(self, test_data):
        """
        Evaluates model performance on test data.
        :param test_data: Test data consisting of (input, label) pairs.
        :return: Dictionary containing the number of correct and incorrect predictions by category.
        """
        x, y = test_data.get_data()
        pred = self.model(x)

        evaluation = { i : {True: 0, False: 0} for i in range(y.shape[1])}
        rt = (torch.argmax(pred, dim=1), torch.argmax(y, dim=1))
        for (x, y) in zip(torch.argmax(pred, dim=1), torch.argmax(y, dim=1)):
            evaluation[x.item()][x.item() == y.item()] += 1
        return evaluation








