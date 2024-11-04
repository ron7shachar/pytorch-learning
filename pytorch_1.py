from data.classify_digits_data.mnist_loader import*

# from classify_digits.network_0 import Network
# from classify_digits.network_1 import Network
from classify_digits.network_2 import Network
from genetic_algorithem.NN_cricher import cricher_maker
from genetic_algorithem.genetic import GENETIC_ALGORITHM

training_data, validation_data, test_data = load_data_wrapper()

# tensor_1 = torch.tensor([x[0] for x in training_data[0][0]], dtype=torch.float64)
# print(tensor_1)


net = Network([784,24, 10])
net.SGD(training_data, 20, 8, 3, test_data=test_data)

problem = {"training":training_data, "validation":validation_data, "test":test_data}
net = Network([784,24,10])
population = cricher_maker(15,net,0.2,0.2,
                  [1000,10000],[3,10],[0.5,5],[1,30])

experiment = GENETIC_ALGORITHM(problem,population,150,3)
