import OkapiV2.Core as ok
from OkapiV2.Core import Model
from OkapiV2.Layers.Basic import FullyConnected, Dropout
from OkapiV2.Layers.Activations import ActivationLayer, PReLULayer
from OkapiV2.Layers.Convolutional import Convolutional, MaxPooling
from OkapiV2 import Activations, Datasets, Initializers
import numpy as np
import random
from operator import itemgetter

X_train, y_train, X_val, y_val, X_test, y_test = Datasets.load_mnist()

dropout_p = 0.2
num_filters = 32
filter_size = 3
pool_size = 2
num_classes = 10
pad = True
batch_size = 10000

population_size = 100
num_generations = 100
rm_p = 0.8
init_mut_p = 1e-5
init_mut_std = 1e-3
init_cross_p = 0.7

model = Model()
model.add(Convolutional(num_filters, filter_size, filter_size, pad=pad))
model.add(ActivationLayer(Activations.tanh))
model.add(MaxPooling(pool_size, pool_size))
model.add(PReLULayer())
model.add(Dropout(dropout_p))
model.add(FullyConnected(bias_initializer=Initializers.glorot_uniform))
model.add(ActivationLayer(Activations.alt_softmax))

model.compile(X_train, y_train)

X_batches, y_batches, num_batches = \
    ok.make_batches(X_train, y_train, batch_size)


def initialize(population_size):
    population = []
    for i in range(population_size):
        model.randomize_params(X_train, y_train)
        individual = {'genome': model.get_params_as_vec()}
        individual['genome'] = np.append(individual['genome'], [init_mut_std, init_mut_p, init_cross_p])
        population.append(individual)
    return population


def evaluate(population):
    for individual in population:
        model.set_params_as_vec(individual['genome'][:-3])
        individual['fitness'] = 0
        for X_batch, y_batch in zip(X_batches, y_batches):
            individual['fitness'] += model.get_test_loss(X_batch, y_batch)
        individual['fitness'] /= num_batches
        print('Fitness: %.4G' % individual['fitness'])
    return population


def remove_lowest(population, proportion=0.5):
    '''num_left = len(population) - round(proportion * len(population))
    if num_left % 2 is not 0:
        num_left += 1
    population = sorted(
        population, key=itemgetter('fitness'), reverse=True)[0:num_left]
    return population'''
    num_left = len(population) - round(proportion * len(population))
    if num_left % 2 is not 0:
        num_left += 1
    new_population = []
    while len(new_population) < num_left:
        new_population.append(selectIndividual(population))
    return population


def selectIndividual(population):
    '''max = sum([i['genome']for i in population])
    pick = random.uniform(0, max)
    current = 0
    for individual in population:
        current += individual['genome']
        if current > pick:
            return individual'''
    fits = [i['fitness'] for i in population]
    probs = 1 - (fits - min(fits)) / (max(fits) - min(fits))
    print(sum(probs))
    print(max(probs))
    print(min(probs))
    return population[np.random.multinomial(probs)]


def crossover(father, mother):
    if random.random() < 0.5:
        new_genome = father['genome']
        cross_genome = mother['genome']
    else:
        new_genome = mother['genome']
        cross_genome = father['genome']
    cross_p = new_genome[-1]
    if random.random() < cross_p:
        cross_point = random.randint(0, new_genome.shape[0])
        for i in range(cross_point, new_genome.shape[0]):
            new_genome[i] = cross_genome[i]
    return new_genome


'''def get_families(population, replace=True):
    num_families = len(population) // 2
    num_children = (population_size - len(population)) // num_families 
    last_num_children = (population_size - len(population)) % num_families

    random.shuffle(population)
    fathers = []
    mothers = []
    for i in range(num_families):
        fathers.append(population.pop())
        mothers.append(population.pop())
    if replace:
        for i in range(num_families):
            population.append(fathers[i])
            population.append(mothers[i])
    return fathers, mothers, num_children, last_num_children, population'''


def breed_best(population):
    '''fathers, mothers, num_children, last_num_children, population = get_families(population)

    for father, mother in zip(fathers, mothers):
        for child in range(num_children):
            population.append({'genome': crossover(father, mother)})
    for child in range(last_num_children):
        population.append({'genome': crossover(father, mother)})
    return population'''
    while len(population) < population_size:
        father, mother = selectIndividual(population), selectIndividual(population)
        population.append({'genome': crossover(father, mother)})
    return population


def mutate(population):
    for individual in population:
        mut_p = individual['genome'][-2]
        mut_std = individual['genome'][-3]
        rand_vals = Initializers.uniform(individual['genome'].shape, mut_std)
        for i in range(individual['genome'].shape[0]):
            if random.random() < mut_p:
                individual['genome'][i] += rand_vals[i]
    return population


print('Starting training...')
population = initialize(population_size)
print('Initialized population...')
evaluate(population)
for gen in range(num_generations):
    # population = remove_lowest(population, rm_p)
    population = breed_best(population)
    population = mutate(population)
    population = evaluate(population)

    fits = [i['fitness'] for i in population]
    best_fitness = min(fits)
    avg_fitness = sum(fits) / len(fits)
    mut_ps = [i['genome'][-2] for i in population]
    mut_stds = [i['genome'][-3] for i in population]
    cross_ps = [i['genome'][-1] for i in population]
    avg_mut_p = sum(mut_ps) / len(mut_ps)
    avg_mut_std = sum(mut_stds) / len(mut_stds)
    avg_cross_p = sum(cross_ps) / len(cross_ps)
    print('Gen: %d/%d | Best Fit: %G | Avg Fit: %G | Mut P: %.2E | Mut Std: %.2E | Cross P: %.2E' %
        (gen + 1, num_generations, best_fitness, avg_fitness, avg_mut_p, avg_mut_std, avg_cross_p))

