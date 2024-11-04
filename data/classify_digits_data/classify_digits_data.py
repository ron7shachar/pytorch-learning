from math import floor
from os.path import split

import numpy as np
from torch import dtype

from data.classify_digits_data.mnist_loader import*
from  torch.utils.data import Dataset , DataLoader, random_split


class Classify_digits_data(Dataset):
    def __init__(self,data = None):
        if data is None:
            data = np.load("./data/classify_digits_data/mnist.npy")
            self.labels = data[0,:].reshape(-1)
        self.data = data[0:, :]
        self.x = torch.from_numpy(self.data[:, :-1]).float()

        self.y = self.data[:, -1].astype(int).tolist()
        self.y_vectorized = torch.from_numpy(np.array([vectorized_result(y) for y in self.y ])).float()



    def __getitem__(self, index) :
        return self.x[index,:], self.y_vectorized[index]


    def __len__(self):
        return self.data.shape[0]

    def vectorized_result(j):
        """Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.  This is used to convert a digit
        (0...9) into a corresponding desired output from the neural
        network."""
        e = torch.zeros(10)
        e[j] = 1.0
        return e

    def get_data(self):
        return self.x, self.y_vectorized







