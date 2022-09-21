import numpy as np

def fitness(equation_inputs, population):
    return np.sum(population * equation_inputs, axis=1)

def selection(population, fitness, num_parents):
    parents = np.empty((num_parents, population.shape[1]))