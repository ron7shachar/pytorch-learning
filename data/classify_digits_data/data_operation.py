from math import floor
import numpy as np
from sympy.abc import lamda


class DataOperation():
    def __init__(self,data = None,return_data_type = lambda x:x):
        """
        get data as np.ndarray or none data for operation


        :param data:
        """
        self.data = data
        self.training_data = None
        self.test_data = None
        self.rdt = return_data_type

    def split_training_test(self,training=6.0/7, test=1/7.0,data = None):
        if data is None:
            if self.data is None:
                raise KeyError("NO training_data")
            else:
                data = self.data
        self.training_data ,self.test_data= self.split(data,[training, test],False)
        return self.rdt(self.training_data), self.rdt(self.test_data)

    def split_train_validation (self,train=5.0/6, validation=1.0/6,shuffle = True, training_data = None):
        if training_data is None:
            if self.training_data is None:
                raise KeyError("NO training_data")
            else:
                training_data = self.training_data
        self.train_data , self.validation = self.split(training_data,[train, validation],shuffle)

        return self.rdt(self.train_data), self.rdt(self.validation)




    def _pre_data(self,p,data):
        if sum(p) <=1.00001:
            return [floor(data.shape[0]*p_) for p_ in p]
        else:
            return[p_ for p_ in p]


    def split(self,data,size , shuffle = True):
        """

        :param data: np.ndarray
        :param size: list
        :param shuffle:
        :return:
        """

        # Create a sample array of shape (5000, 70)
        if shuffle:
            data = data.copy()
            np.random.shuffle(data)

        # Define the custom split sizes (e.g., 3000, 1500, and 500 rows)
        split_sizes = self._pre_data(size,data)

        # Calculate indices to split at
        split_indices = np.cumsum(split_sizes)[:-1]  # Get cumulative sums for split points

        # Split the array at the specified indices
        return np.split(data, split_indices)



