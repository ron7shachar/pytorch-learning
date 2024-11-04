import copy
import random
import numpy as np
# from classify_digits.network_2 import Network
from classify_digits_models.networks.network_1 import Network
class NN_cricher():
    def __init__(self,network,alfa,net_alfa,training_data_size,epochs,learning_rate,batch_size,
                 properties = ["training_data_size", "epochs" ,"learning_rate" ,"batch_size"]):
        self.alfa = alfa
        self.fitness = 1
        self.structure = Structure(network,net_alfa)
        self.changeable = properties
        self.properties = {
            "training_data_size" :training_data_size,
            # " max_layers" : max_layers, todo
            "epochs" : epochs,
            "learning_rate" : learning_rate,
            "batch_size" : batch_size
        }
    def performing(self,problem):
        training_data = problem["training"]
        test_data = problem["test"]
        net = Network(self.structure.sizes)
        self.test_size = len(test_data)
        # self.evaluation = net.SGD(
        self.evaluation = net.fit(
            training_data,# training_data[:min(len(training_data),self.properties["training_data_size"])],
            self.properties["epochs"],
            10,
            self.properties["learning_rate"],
            test_data=test_data,
            print_=False)
        error = self.evaluation
        # print( error/len(test_data))


    def mutate(self):
        property_name = "No chainge"
        if random.random() <  self.alfa:
            self.structure.mutate()
            property_name = random.choice(self.changeable)
            self.__change_normal(property_name)
        if property_name == "No chainge":
            return property_name
        return property_name , self.properties[property_name]
    def __change_normal(self,property_name):
        change  = random.normalvariate(
                         self.properties[property_name],
                         self.properties[property_name]/3)
        if isinstance(self.properties[property_name], int):
            self.properties[property_name] = max(1,int(round(change)))
        else:
            self.properties[property_name] = max(0.0,change)

    def reproduce(self, mate):
        e = {}
        for property in self.properties.keys():
            if random.random() < 0.5:
                e[property] = self.properties[property]
            else:
                e[property] = mate.properties[property]
        # structure = self.structure.reproduce(mate.structure)
        if self.fitness > mate.fitness: structure = self.structure
        else:structure = mate.structure
        child = NN_cricher(
            structure, # todo
            self.alfa,self.structure.alfa,

            e["training_data_size"],
            e["epochs"],
            e["learning_rate"],
            e["batch_size"],
            self.changeable)
        return child
    def update_fitness(self):
        #____________________________parameters___________________________
        #self.evaluation
        error = self.evaluation
        #         # connections = self.structure.get_weights()
        # epochs = self.properties["epochs"]
        # net_size = self.structure.sizes
        # training_data_size = self.properties["training_data_size"]

        # self.fitness = error
        self.fitness = error/(1-error)
        print(self.fitness," ", error , " ", self.structure.sizes," ",self.properties)

class Structure():
    def __init__(self ,Network,net_alfa):
        self.sizes = copy.copy(Network.sizes)
        self.hidden = self.sizes[1:-1]
        self.input = self.sizes[0]
        self.output = self.sizes[-1]
        self.alfa = net_alfa


    def get_weights(self):
        n_weights = 0
        for i ,size,in enumerate(self.sizes[:-1]):
            n_weights+=size*self.sizes[i+1]
        return n_weights
    def mutate(self):
        if random.random() < self.alfa:
            if random.random() < 0.5: self.add_layer()
            else:self.remove_layer()
        elif self.hidden:
            i, n = random.choice(list(enumerate(self.hidden)))
            layer_size = max(1, int(round(random.normalvariate(
                n,
                n / 3))))
            self.hidden[i] = layer_size
            self.sizes[i+1] = layer_size

    def remove_layer(self):
        if self.hidden:
            i = random.choice(range(len(self.hidden)))
            self.hidden.pop(i)
            self.sizes.pop(i+1)
        else:self.add_layer()


    def add_layer(self):
        m_n_l = int((sum(self.hidden)+self.output)/(len(self.hidden)+1))
        if self.hidden:
            i = random.choice(range(len(self.hidden)))
        else:
            i = 0
        self.hidden.insert(i,m_n_l)
        self.sizes.insert(i+1,m_n_l)

    def reproduce(self, structure):
        # s = [self.input]
        # if len(self.hidden) > len(structure.hidden):
        #     max_ = self.hidden
        #     min_ = structure.hidden
        # else:
        #     min_ = self.hidden
        #     max_ = structure.hidden
        #
        # for i in range(len(max_)):
        #     if i < len(min_):
        #         s.append(max(min_[i],max_[i]))
        #     else:
        #         s.append(max_[i])
        # s.append(self.output)
        # net = Network(s)
        # return Structure(net, self.alfa)
        pass



def cricher_maker(n,network,alfa,net_alfa,training_data_size,epochs,learning_rate,batch_size):
    properties = []
    population = []
    for i in range(n):
        training_data_size_ = random.choice(range(training_data_size[0], training_data_size[1],10))
        properties.append("training_data_size")
        if isinstance(epochs,int):epochs_ = epochs
        else:
            epochs_= random.choice(range(epochs[0], epochs[1]))
            properties.append("epochs")
        learning_rate_= random.choice(np.arange(learning_rate[0], learning_rate[1] , 0.01))
        properties.append("learning_rate")
        batch_size_= random.choice(range(batch_size[0], batch_size[1]))
        properties.append("batch_size")

        cricher = NN_cricher(network,alfa,net_alfa,training_data_size_,epochs_,learning_rate_,batch_size_,properties)
        population.append(cricher)

    return population