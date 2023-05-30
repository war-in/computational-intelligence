import time

import numpy as np
from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.operator import SBXCrossover, UniformMutation
from jmetal.operator.selection import (
    BestSolutionSelection,
    RandomSolutionSelection,
    RouletteWheelSelection,
)
from jmetal.problem.singleobjective.unconstrained import Rastrigin, Sphere
from jmetal.util.termination_criterion import StoppingByEvaluations
from matplotlib import pyplot as plt

from computational_intelligence.algorithms.genetic_algorithm import (
    PrintObjectivesObserver,
)
from computational_intelligence.algorithms.socjo_SSGA import SocioObserver, SocioSSGA


def compare_population_size():
    problem_size = 100
    problem = Rastrigin(problem_size)

    populations = list(range(20, 130, 10))

    epoch = []
    fitness = []

    for population_size in populations:
        print("Population size:", population_size)

        epoch_ = []
        fitness_ = []
        for _ in range(10):
            socio = SocioSSGA(
                problem=problem,
                population_size=population_size,
                offspring_population_size=1,
                interaction_probability=0.5,
                selection=RouletteWheelSelection(),
                mutation=UniformMutation(0.1),
                crossover=SBXCrossover(0.9),
                basic_prob=0.1,
                trust_prob=0.6,
                cost_prob=0.3,
                max_switched_genes=int(problem.number_of_variables * 0.75),
                termination_criterion=StoppingByEvaluations(10000),
            )

            observer = SocioObserver(10)
            socio.observable.register(observer)

            socio.run()

            epoch_.append(observer.epoch[0:1000])
            fitness_.append(observer.fitness[0:1000])

        epoch.append(list(np.mean(np.array(epoch_), axis=0)))
        fitness.append(list(np.mean(np.array(fitness_), axis=0)))

    plt.figure()
    plt.xlabel("Evaluations")
    plt.ylabel("Fitness")
    plt.title("Comparison of different population sizes")
    for i, data in enumerate(zip(epoch, fitness, populations)):
        plt.plot(data[0], data[1], label="Population " + str(data[2]))
    plt.legend()
    plt.show()


def compare_max_switched_genes():
    problem_size = 100
    problem = Rastrigin(problem_size)

    max_switched_genes = [
        int(problem.number_of_variables * x / 100) for x in range(10, 101, 10)
    ]

    epoch = []
    fitness = []

    for max_switched in max_switched_genes:
        print("Max switched genes", max_switched)

        epoch_ = []
        fitness_ = []
        for _ in range(10):
            socio = SocioSSGA(
                problem=problem,
                population_size=40,
                offspring_population_size=1,
                interaction_probability=0.5,
                selection=RouletteWheelSelection(),
                mutation=UniformMutation(0.3),
                crossover=SBXCrossover(0.9),
                basic_prob=0.1,
                trust_prob=0.6,
                cost_prob=0.3,
                max_switched_genes=max_switched,
                termination_criterion=StoppingByEvaluations(10000),
            )

            observer = SocioObserver(10)
            socio.observable.register(observer)

            socio.run()

            epoch_.append(observer.epoch[0:1000])
            fitness_.append(observer.fitness[0:1000])

        epoch.append(list(np.mean(np.array(epoch_), axis=0)))
        fitness.append(list(np.mean(np.array(fitness_), axis=0)))

    plt.figure()
    plt.xlabel("Evaluations")
    plt.ylabel("Fitness")
    plt.title("Comparison of different max switched genes")
    for i, data in enumerate(zip(epoch, fitness, max_switched_genes)):
        plt.plot(data[0], data[1], label="Max switched " + str(data[2]))
    plt.legend()
    plt.show()


def compare_interaction_probabilities():
    problem_size = 100
    problem = Rastrigin(problem_size)

    interaction_probabilities = [x / 100 for x in range(10, 101, 10)]

    epoch = []
    fitness = []

    for interaction_probability in interaction_probabilities:
        print("Interaction probability", interaction_probability)

        epoch_ = []
        fitness_ = []
        for _ in range(10):
            socio = SocioSSGA(
                problem=problem,
                population_size=40,
                offspring_population_size=1,
                interaction_probability=interaction_probability,
                selection=RouletteWheelSelection(),
                mutation=UniformMutation(0.3),
                crossover=SBXCrossover(0.9),
                basic_prob=0.1,
                trust_prob=0.6,
                cost_prob=0.3,
                max_switched_genes=int(problem.number_of_variables * 0.9),
                termination_criterion=StoppingByEvaluations(10000),
            )

            observer = SocioObserver(10)
            socio.observable.register(observer)

            socio.run()

            epoch_.append(observer.epoch[0:1000])
            fitness_.append(observer.fitness[0:1000])

        epoch.append(list(np.mean(np.array(epoch_), axis=0)))
        fitness.append(list(np.mean(np.array(fitness_), axis=0)))

    plt.figure()
    plt.xlabel("Evaluations")
    plt.ylabel("Fitness")
    plt.title("Comparison of different interaction probabilities")
    for i, data in enumerate(zip(epoch, fitness, interaction_probabilities)):
        plt.plot(data[0], data[1], label="Interaction probability " + str(data[2]))
    plt.legend()
    plt.show()


def compare_mutation_probabilities():
    problem_size = 100
    problem = Rastrigin(problem_size)

    mutation_probabilities = [x / 100 for x in range(10, 101, 10)]

    epoch = []
    fitness = []

    for mutation_probability in mutation_probabilities:
        print("Mutation probability", mutation_probability)

        epoch_ = []
        fitness_ = []
        for _ in range(10):
            socio = SocioSSGA(
                problem=problem,
                population_size=40,
                offspring_population_size=1,
                interaction_probability=0.5,
                selection=RouletteWheelSelection(),
                mutation=UniformMutation(mutation_probability),
                crossover=SBXCrossover(0.9),
                basic_prob=0.1,
                trust_prob=0.6,
                cost_prob=0.3,
                max_switched_genes=int(problem.number_of_variables * 0.9),
                termination_criterion=StoppingByEvaluations(10000),
            )

            observer = SocioObserver(10)
            socio.observable.register(observer)

            socio.run()

            epoch_.append(observer.epoch[0:1000])
            fitness_.append(observer.fitness[0:1000])

        epoch.append(list(np.mean(np.array(epoch_), axis=0)))
        fitness.append(list(np.mean(np.array(fitness_), axis=0)))

    plt.figure()
    plt.xlabel("Evaluations")
    plt.ylabel("Fitness")
    plt.title("Comparison of different mutation probabilities")
    for i, data in enumerate(zip(epoch, fitness, mutation_probabilities)):
        plt.plot(data[0], data[1], label="Mutation probability " + str(data[2]))
    plt.legend()
    plt.show()


def compare_trust_probabilities():
    problem_size = 100
    problem = Rastrigin(problem_size)

    trust_probabilities = [x / 100 for x in range(10, 101, 10)]

    epoch = []
    fitness = []

    for trust_probability in trust_probabilities:
        print("Trust probability", trust_probability)

        epoch_ = []
        fitness_ = []
        for _ in range(10):
            socio = SocioSSGA(
                problem=problem,
                population_size=40,
                offspring_population_size=1,
                interaction_probability=0.5,
                selection=RouletteWheelSelection(),
                mutation=UniformMutation(0.1),
                crossover=SBXCrossover(0.9),
                basic_prob=(1.0 - trust_probability) / 2,
                trust_prob=trust_probability,
                cost_prob=(1.0 - trust_probability) / 2,
                max_switched_genes=int(problem.number_of_variables * 0.9),
                termination_criterion=StoppingByEvaluations(10000),
            )

            observer = SocioObserver(10)
            socio.observable.register(observer)

            socio.run()

            epoch_.append(observer.epoch[0:1000])
            fitness_.append(observer.fitness[0:1000])

        epoch.append(list(np.mean(np.array(epoch_), axis=0)))
        fitness.append(list(np.mean(np.array(fitness_), axis=0)))

    plt.figure()
    plt.xlabel("Evaluations")
    plt.ylabel("Fitness")
    plt.title("Comparison of different trust probabilities")
    for i, data in enumerate(zip(epoch, fitness, trust_probabilities)):
        plt.plot(data[0], data[1], label="Trust probability " + str(data[2]))
    plt.legend()
    plt.show()


def compare_basic_probabilities():
    problem_size = 100
    problem = Rastrigin(problem_size)

    basic_probabilities = [x / 100 for x in range(10, 101, 10)]

    epoch = []
    fitness = []

    for basic_probability in basic_probabilities:
        print("Basic probability", basic_probability)

        epoch_ = []
        fitness_ = []
        for _ in range(10):
            socio = SocioSSGA(
                problem=problem,
                population_size=40,
                offspring_population_size=1,
                interaction_probability=0.5,
                selection=RouletteWheelSelection(),
                mutation=UniformMutation(0.1),
                crossover=SBXCrossover(0.9),
                basic_prob=basic_probability,
                trust_prob=0.3,
                cost_prob=0.7 - basic_probability,
                max_switched_genes=int(problem.number_of_variables * 0.9),
                termination_criterion=StoppingByEvaluations(10000),
            )

            observer = SocioObserver(10)
            socio.observable.register(observer)

            socio.run()

            epoch_.append(observer.epoch[0:1000])
            fitness_.append(observer.fitness[0:1000])

        epoch.append(list(np.mean(np.array(epoch_), axis=0)))
        fitness.append(list(np.mean(np.array(fitness_), axis=0)))

    plt.figure()
    plt.xlabel("Evaluations")
    plt.ylabel("Fitness")
    plt.title("Comparison of different basic probabilities")
    for i, data in enumerate(zip(epoch, fitness, basic_probabilities)):
        plt.plot(data[0], data[1], label="Basic probability " + str(data[2]))
    plt.legend()
    plt.show()


def compare_selection_algorithms():
    problem_size = 100
    problem = Rastrigin(problem_size)

    selection_algorithms = [
        RouletteWheelSelection(),
        BestSolutionSelection(),
        RandomSolutionSelection(),
    ]

    epoch = []
    fitness = []

    for selection_algorithm in selection_algorithms:
        print("Selection algorithm", selection_algorithm.get_name())

        epoch_ = []
        fitness_ = []
        for _ in range(10):
            socio = SocioSSGA(
                problem=problem,
                population_size=40,
                offspring_population_size=1,
                interaction_probability=0.5,
                selection=selection_algorithm,
                mutation=UniformMutation(0.1),
                crossover=SBXCrossover(0.9),
                basic_prob=0.4,
                trust_prob=0.3,
                cost_prob=0.3,
                max_switched_genes=int(problem.number_of_variables * 0.9),
                termination_criterion=StoppingByEvaluations(10000),
            )

            observer = SocioObserver(10)
            socio.observable.register(observer)

            socio.run()

            epoch_.append(observer.epoch[0:1000])
            fitness_.append(observer.fitness[0:1000])

        epoch.append(list(np.mean(np.array(epoch_), axis=0)))
        fitness.append(list(np.mean(np.array(fitness_), axis=0)))

    plt.figure()
    plt.xlabel("Evaluations")
    plt.ylabel("Fitness")
    plt.title("Comparison of different selection algorithms")
    for i, data in enumerate(zip(epoch, fitness, selection_algorithms)):
        plt.plot(data[0], data[1], label="Basic probability " + str(data[2].get_name()))
    plt.legend()
    plt.show()


def compare_with_common_genetic():
    problems = [
        Rastrigin(50),
        Sphere(50),
        Rastrigin(100),
        Sphere(100),
        Rastrigin(200),
        Sphere(200),
    ]

    for problem in problems:
        epoch = []
        fitness = []

        algorithms = [
            (
                SocioSSGA(
                    problem=problem,
                    population_size=40,
                    offspring_population_size=1,
                    interaction_probability=0.5,
                    selection=RouletteWheelSelection(),
                    mutation=UniformMutation(0.1),
                    crossover=SBXCrossover(0.9),
                    basic_prob=0.4,
                    trust_prob=0.3,
                    cost_prob=0.3,
                    max_switched_genes=int(problem.number_of_variables * 0.9),
                    termination_criterion=StoppingByEvaluations(10000),
                ),
                SocioObserver(10),
            ),
            (
                GeneticAlgorithm(
                    problem=problem,
                    population_size=40,
                    offspring_population_size=1,
                    selection=RouletteWheelSelection(),
                    mutation=UniformMutation(0.1),
                    crossover=SBXCrossover(0.9),
                    termination_criterion=StoppingByEvaluations(10000),
                ),
                PrintObjectivesObserver(10),
            ),
        ]

        for algorithm, observer in algorithms:
            epoch_ = []
            fitness_ = []

            for _ in range(10):
                algorithm.observable.register(observer)

                algorithm.run()

                epoch_.append(observer.epoch[0:800])
                fitness_.append(observer.fitness[0:800])

            epoch.append(list(np.mean(np.array(epoch_), axis=0)))
            fitness.append(list(np.mean(np.array(fitness_), axis=0)))

        plt.figure()
        plt.xlabel("Evaluations")
        plt.ylabel("Fitness")
        plt.title("Comparison with genetic. Problem: " + problem.get_name())
        for i, data in enumerate(zip(epoch, fitness, ["Socio", "Genetic"])):
            plt.plot(data[0], data[1], label="Algorithm " + data[2])
        plt.legend()
        plt.show()


def compare_times():
    problems = [
        Rastrigin(50),
        Sphere(50),
        Rastrigin(100),
        Sphere(100),
        Rastrigin(200),
        Sphere(200),
    ]

    for problem in problems:
        print(problem.get_name(), problem.number_of_variables)
        times = []

        algorithms = [
            SocioSSGA(
                problem=problem,
                population_size=40,
                offspring_population_size=1,
                interaction_probability=0.5,
                selection=RouletteWheelSelection(),
                mutation=UniformMutation(0.1),
                crossover=SBXCrossover(0.9),
                basic_prob=0.4,
                trust_prob=0.3,
                cost_prob=0.3,
                max_switched_genes=int(problem.number_of_variables * 0.9),
                termination_criterion=StoppingByEvaluations(10000),
            ),
            GeneticAlgorithm(
                problem=problem,
                population_size=40,
                offspring_population_size=1,
                selection=RouletteWheelSelection(),
                mutation=UniformMutation(0.1),
                crossover=SBXCrossover(0.9),
                termination_criterion=StoppingByEvaluations(10000),
            ),
        ]

        for algorithm in algorithms:
            times_ = []

            for _ in range(10):
                start = time.time()
                algorithm.run()
                end = time.time()

                times_.append(end - start)

            times.append(np.mean(np.array(times_)))

        print("Socio time:", times[0])
        print("Genetic time:", times[1])


if __name__ == "__main__":
    compare_times()
