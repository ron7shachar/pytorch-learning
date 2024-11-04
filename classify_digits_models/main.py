
from data.classify_digits_data.classify_digits_data import Classify_digits_data,data_loader

from data.classify_digits_data.mnist_loader import *
from classify_digits_models.networks.network_2 import Network
from data.classify_digits_data.data_operation import DataOperation as do
# from genetic_algorithem.genetic import *


dataset = Classify_digits_data()
data = do(dataset.data,Classify_digits_data)
_,test_data = data.split_training_test()
training_data, validation_data = data.split_train_validation()

net = Network([784,79, 10])
net.fit(training_data, 9, 21, 4.0, validation_data)


# problem = {"training":training_data, "validation":validation_data, "test":test_data}
# net = Network([784,20,10])
# population = cricher_maker(15,net,0.2,0.2,
#                   [10000,60000],[3,10],[0.5,5],[1,30])
#
# experiment = GENETIC_ALGORITHM(problem,population,150,3)




