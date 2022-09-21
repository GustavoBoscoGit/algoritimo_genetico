import numpy as np

def fitness(equation_inputs, population):
    return np.sum(population * equation_inputs, axis=1)

def selection(population, fitness, num_parents):
    parents = np.empty((num_parents, population.shape[1]))
    
    for idx in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        
        parents[idx, :] = population[max_fitness_idx, :]