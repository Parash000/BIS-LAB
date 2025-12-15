import numpy as np
import random
import copy

def initialize_wolves(search_space, num_wolves):

    dimensions = len(search_space)
    wolves = np.zeros((num_wolves, dimensions))
    for i in range(num_wolves):
        wolves[i] = np.random.uniform(search_space[:, 0], search_space[:, 1])
    return wolves

def fitness_function(solution, tasks, machines):

    makespan = np.sum(solution)
    return makespan

def correct_solution(solution, search_space):

    for i in range(len(solution)):
        if solution[i] < search_space[i, 0]:
            solution[i] = search_space[i, 0]
        if solution[i] > search_space[i, 1]:
            solution[i] = search_space[i, 1]
    return solution

def grey_wolf_optimization(search_space, num_wolves, max_iterations, tasks, machines):

    dimensions = len(search_space)
    alpha_wolf = np.zeros(dimensions)
    beta_wolf = np.zeros(dimensions)
    gamma_wolf = np.zeros(dimensions)
    

    alpha_score = float("inf")
    beta_score = float("inf")
    gamma_score = float("inf")
    
    wolves = initialize_wolves(search_space, num_wolves)

    for iteration in range(max_iterations):

        a = 2 - (iteration / max_iterations) * 2
        
        for i in range(num_wolves):

            fitness = fitness_function(wolves[i], tasks, machines)
            
            if fitness < alpha_score:
                alpha_score = fitness
                alpha_wolf = wolves[i].copy()
            elif fitness < beta_score:
                beta_score = fitness
                beta_wolf = wolves[i].copy()
            elif fitness < gamma_score:
                gamma_score = fitness
                gamma_wolf = wolves[i].copy()
                
        for i in range(num_wolves):
            for j in range(dimensions):

                r1, r2 = random.random(), random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_wolf[j] - wolves[i][j])
                X1 = alpha_wolf[j] - A1 * D_alpha
                
                r1, r2 = random.random(), random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_wolf[j] - wolves[i][j])
                X2 = beta_wolf[j] - A2 * D_beta
                
                r1, r2 = random.random(), random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_gamma = abs(C3 * gamma_wolf[j] - wolves[i][j])
                X3 = gamma_wolf[j] - A3 * D_gamma

                wolves[i][j] = (X1 + X2 + X3) / 3.0
        

            wolves[i] = correct_solution(wolves[i], search_space)

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Best Fitness (Makespan/Cost) = {alpha_score:.3f}")
            
    return alpha_wolf, alpha_score


NUM_TASKS = 10
NUM_MACHINES = 3

search_space_bounds = np.array([[0, NUM_MACHINES - 1]] * NUM_TASKS)

tasks_data = [2, 5, 1, 4, 3, 2, 5, 1, 4, 3] 
machines_data = [0, 0, 0] 

NUM_WOLVES = 50
MAX_ITERATIONS = 100

best_schedule, best_score = grey_wolf_optimization(
    search_space_bounds, 
    NUM_WOLVES, 
    MAX_ITERATIONS, 
    tasks_data, 
    machines_data
)

print("-" * 20)
print(f"Optimization finished after {MAX_ITERATIONS} iterations.")
print(f"Best solution (machine assignments): {best_schedule}")
print(f"Best fitness score (e.g., Makespan/Cost): {best_score:.3f}")
