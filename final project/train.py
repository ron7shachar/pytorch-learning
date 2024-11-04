
# Standard library
import torch
from torch.utils.data import DataLoader
from evaluat import *


class Learn():
    def __init__(self,model, epochs, mini_batch_size, learning_rate):
        self.model = model
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)  # Set the optimizer



    def __call__(self, sample):
        for j in range(self.epochs):  # Loop over the number of epochs
            dataloader = DataLoader(sample, self.mini_batch_size)
            for mini_batch in dataloader:self.step(*mini_batch)
            with torch.no_grad():
                accuracy = accuracy_score(self.model, sample)
            print("Epoch {0}: accuracy {1} : sample size {2}".format(j + 1, accuracy, len(sample)))
        return accuracy

    def step(self, x, target):
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
