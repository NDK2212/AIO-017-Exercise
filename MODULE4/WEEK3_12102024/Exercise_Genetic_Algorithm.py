import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
random.seed(0)


def load_data_from_file(file_name='MODULE4\WEEK2_05102024\Advertising copy.csv'):
    data = np.genfromtxt(file_name, dtype=None, delimiter=',', skip_header=1)
    features_x = data[:, :3]
    sales_y = data[:, 3]
    features_x = np.c_[np.ones((data.shape[0], 1)), features_x]
    return features_x, sales_y


X_features, sales_y = load_data_from_file()


def create_individual(n=4, bound=10):
    individual = []
    for _ in range(n):
        individual.append((random.random()*2-1)*bound/2)
    return individual


def compute_loss(individual):
    theta = np.array(individual)
    y_hat = X_features.dot(theta)
    loss = np.multiply((y_hat - sales_y), (y_hat - sales_y)).mean()
    return loss


def compute_fitness(individual):
    loss = compute_loss(individual)
    fitness_value = 1 / (loss + 1)
    return fitness_value


def crossover(individual1, individual2, crossover_rate=0.9):
    individual1_new = individual1.copy()
    individual2_new = individual2.copy()
    for i in range(len(individual1)):
        if random.random() > crossover_rate:
            tmp = individual1_new[i]
            individual1_new[i] = individual2_new[i]
            individual2_new[i] = tmp
    return individual1_new, individual2_new


def mutate(individual, mutation_rate=0.05, bound=10):
    # Copy cá thể để không thay đổi cá thể gốc
    individual_m = individual.copy()

    # Đột biến từng gene của cá thể
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            # Đột biến giá trị trong phạm vi hợp lý (giả sử -bound đến bound)
            mutation_value = ((random.random() * 2) - 1) * (bound * 0.1)
            individual_m[i] = individual_m[i] + mutation_value

            # Giới hạn giá trị sau khi đột biến trong khoảng hợp lý
            if individual_m[i] > bound:
                individual_m[i] = bound
            elif individual_m[i] < -bound:
                individual_m[i] = -bound

    return individual_m


def initialize_population(m):
    population = [create_individual() for _ in range(m)]
    return population


def selection(sorted_old_population, m):
    index1 = random.randint(0, m-1)
    while True:
        index2 = random.randint(0, m-1)
        if (index2 != index1):
            break
    individual_s = sorted_old_population[index1]
    if index2 > index1:
        individual_s = sorted_old_population[index2]

    return individual_s


def create_new_population(old_population, elitism=2, gen=1):
    m = len(old_population)
    sorted_population = sorted(old_population, key=compute_fitness)
    if gen % 1 == 0:
        print("Best loss:", compute_loss(
            sorted_population[m-1]), "with chromsome: ", sorted_population[m-1])
    new_population = []
    while len(new_population) < m - elitism:
        parent_1, parent_2 = selection(
            sorted_population, m), selection(sorted_population, m)
        child_1, child_2 = crossover(parent_1, parent_2)
        child1_m = mutate(child_1)
        child2_m = mutate(child_2)
        new_population.append(child1_m)
        if len(new_population) < m - elitism:
            new_population.append(child2_m)
    new_population.extend(sorted_population[m - elitism:])
    return new_population, compute_loss(sorted_population[m-1])


print(X_features)
print(sales_y)
individual = create_individual()
print(individual)
individual = [4.09, 4.82, 3.10, 4.02]
fitness_score = compute_fitness(individual)
print(fitness_score)
individual1 = [4.09, 4.82, 3.10, 4.02]
individual2 = [3.44, 2.57, -0.79, -2.41]
individual1, individual2 = crossover(individual1, individual2, 0.9)
print(individual1)
print(individual2)
before_individual = [4.09, 4.82, 3.10, 4.02]
after_individual = mutate(individual, mutation_rate=2.0)
print(before_individual == after_individual)
population = initialize_population(100)
print(len(population))
individual1 = [4.09, 4.82, 3.10, 4.02]
individual2 = [3.44, 2.57, -0.79, -2.41]
old_population = [individual1, individual2]
new_population, _ = create_new_population(old_population, elitism=2, gen=1)


def run_ga():
    n_generations = 100
    m = 600
    _, _ = load_data_from_file()
    population = initialize_population(m)
    losses_list = []
    for _ in range(n_generations):
        population, loss = create_new_population(population, elitism=2, gen=1)
        losses_list.append(loss)
    return losses_list


losses_list = run_ga()


def visualize_loss(losses_list):
    plt.plot(losses_list)
    plt.xlabel('Generations')
    plt.ylabel('losses')
    plt.show()


losses_list = run_ga()
visualize_loss(losses_list)


def visualize_predict_gt():
    # visualization of ground truth and predict value
    sorted_population = sorted(new_population, key=compute_fitness)
    print(sorted_population[-1])
    theta = np.array(sorted_population[-1])

    estimated_prices = []
    for feature in X_features:
        estimated_prices.append(feature@theta)

    _, _ = plt.subplots(figsize=(10, 6))
    plt.xlabel('Samples')
    plt.ylabel('Price')
    plt.plot(sales_y, c='green', label='Real Prices')
    plt.plot(estimated_prices, c='blue', label='Estimated Prices')
    plt.legend()
    plt.show()


visualize_predict_gt()
