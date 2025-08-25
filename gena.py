import random

# Objective function: maximize utility while penalizing overuse
def objective(allocation):
    utility = sum(task_value[i] * allocation[i] for i in range(len(allocation)))
    penalty = max(0, sum(allocation) - total_resource) * 100  # penalty if over budget
    return -utility + penalty  # negative because GA minimizes, we want max utility

# Initialize random population
def init_population(size, tasks):
    return [[random.uniform(0, total_resource / tasks) for _ in range(tasks)] for _ in range(size)]

# Fitness function
def fitness(pop):
    return [objective(ind) for ind in pop]

# Tournament selection (pick fitter of two random individuals)
def select(pop, fit):
    new_pop = []
    for _ in pop:
        i, j = random.sample(range(len(pop)), 2)
        if fit[i] < fit[j]:
            new_pop.append(pop[i])
        else:
            new_pop.append(pop[j])
    return new_pop

# Single-point crossover
def crossover(parents, rate):
    offspring = []
    for i in range(0, len(parents), 2):
        p1, p2 = parents[i], parents[(i + 1) % len(parents)]
        if random.random() < rate:
            pt = random.randint(1, len(p1) - 1)
            offspring.append(p1[:pt] + p2[pt:])
            offspring.append(p2[:pt] + p1[pt:])
        else:
            offspring.append(p1[:])
            offspring.append(p2[:])
    return offspring

# Mutation
def mutate(pop, rate):
    for ind in pop:
        for i in range(len(ind)):
            if random.random() < rate:
                ind[i] += random.uniform(-1, 1)
                ind[i] = max(0, ind[i])  # keep non-negative
    return pop

# Main GA
def GEA_resource_allocation():
    params = {"size": 50, "tasks": 5, "rate": 0.05, "cross": 0.7, "gens": 50}
    global total_resource, task_value
    total_resource = 100
    task_value = [10, 20, 15, 25, 30]  # utility per unit resource for each task

    pop = init_population(params["size"], params["tasks"])
    best = None
    best_score = float("inf")

    for g in range(params["gens"]):
        fit = fitness(pop)
        if min(fit) < best_score:
            best_score = min(fit)
            best = pop[fit.index(best_score)]
        pop = mutate(crossover(select(pop, fit), params["cross"]), params["rate"])

    return best, -best_score  # flip sign to show max utility


# Run it
best_alloc, best_utility = GEA_resource_allocation()
print("Optimal Allocation:", [round(x, 2) for x in best_alloc])
print("Max Utility:", round(best_utility, 2))
