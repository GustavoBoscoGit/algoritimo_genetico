import numpy as np

import ga

def main():
    equation_inputs = [4, -2, 3.5, 5, -11, -4.7]
    num_weights = 6
    
    solutions_per_population = 8
    
    population_size = (solutions_per_population, num_weights)
    
    population = np.random.uniform(low=-4.0, high=4.0, size=population_size)
    
    print("população inicial")
    print(population)
    
    num_generations = 5
    
    num_parents_crossover = 4
    
    for generation in range(num_generations):
        print(f"geração{generation}")
        
        fitness = ga.fitness(equation_inputs, population)
        print("Fitness")
        print(fitness)
        
    
if __name__ == "__main__":
    main()