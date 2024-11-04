from record.record import Record
from data.classify_digits_data import mnist_loader
from .NN_cricher import*
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()



def random_selection(population,distinction):
    weights = [item.fitness**distinction for item in population]
    return random.choices(population,weights=weights,k=2)



def GENETIC_ALGORITHM(problem ,population,generations,distinction):
    # inputs: population, a set of individuals
    # FITNESS-FN, a function that measures the fitness of an individual

    record = Record("best_structure")
    record_ = record.get_record()
    if record_ is None:
        max_value = 0
    else:
        max_value = record_["fitness"]


    for i in range(generations):
        print(f"Generation ______  {i} ______")
        new_population=[]
        for i in range(len(population)):
            parent1,parent2 = random_selection(population,distinction)
            child = parent1.reproduce(parent2)
            child.mutate()
            child.performing(problem)
            child.update_fitness()
            new_population.append(child)
        population = new_population
        best = max(population,key=lambda cricher:cricher.fitness)
        if max_value < best.fitness:
            max_value  = best.fitness
            my_object = {"hidden" : best.structure.hidden, "properties" : best.properties,"fitness":best.fitness}
            record.set_record(my_object)