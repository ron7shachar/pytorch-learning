from math import floor
from sys import implementation


class Compose():
    def __init__(self):
        pass
    def __call__(self):
        raise KeyError("must Implement __call__ function")



from data.classify_digits_data.mnist_loader import*
from  torch.utils.data import Dataset , DataLoader, random_split


class _Classify_digits_data(Dataset):
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


class ToClassifyDigitsData(Compose):
    def __init__(self):
        super(ToClassifyDigitsData, self).__init__()

    def __call__(self, sample):
        return  _Classify_digits_data(sample)



def _pre_data(p,sample):
    if sum(p) <=1.00001:
        return [floor(sample.shape[0]*p_) for p_ in p]
    else:
        return[p_ for p_ in p]

class Split(Compose):
    def __init__(self, size, shuffle=True):
        """
                :param size: list
                :param shuffle:
                :return:
                """
        super(Split, self).__init__()
        self.size = size
        self.shuffle = shuffle
    def __call__(self, data):
        """

        :param data: np.ndarray
        :param size: list
        :param shuffle:
        :return:
        """

        # Create a sample array of shape (5000, 70)
        data = data.data
        if self.shuffle:
            data = data.copy()
            np.random.shuffle(data)

        # Define the custom split sizes (e.g., 3000, 1500, and 500 rows)
        split_sizes = _pre_data(self.size, data)

        # Calculate indices to split at
        split_indices = np.cumsum(split_sizes)[:-1]  # Get cumulative sums for split points

        # Split the array at the specified indices
        return (_Classify_digits_data(s) for s in np.split(data, split_indices))