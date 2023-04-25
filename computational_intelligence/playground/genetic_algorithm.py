import matplotlib.pyplot as plt
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.core.observer import Observer
from jmetal.core.operator import Crossover
from jmetal.core.solution import BinarySolution
from jmetal.lab.visualization import Plot
from jmetal.operator import (
    BinaryTournamentSelection,
    BitFlipMutation,
    PolynomialMutation,
    SimpleRandomMutation,
    UniformMutation,
)
from jmetal.operator.crossover import CXCrossover, PMXCrossover, SBXCrossover
from jmetal.operator.selection import BestSolutionSelection, RouletteWheelSelection
from jmetal.problem import ZDT1
from jmetal.problem.singleobjective.unconstrained import OneMax, Rastrigin, Sphere
from jmetal.util.observer import LOGGER
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

from computational_intelligence.playground.age_classes import MyRastrigin
from computational_intelligence.playground.socjo_SSGA import SocioSSGA


class PrintObjectivesObserver(Observer):
    def __init__(self, frequency: float = 1.0) -> None:
        """Show the number of evaluations, the best fitness and computing time.

        :param frequency: Display frequency."""
        self.display_frequency = frequency
        self.epoch = []
        self.fitness = []

    def update(self, *args, **kwargs):
        evaluations = kwargs["EVALUATIONS"]
        solutions = kwargs["SOLUTIONS"]

        if evaluations % self.display_frequency == 0 and solutions:
            if isinstance(solutions, list):
                fitness = sorted(solutions, key=lambda solution: solution.objectives[0])
            else:
                fitness = solutions.objectives[0]

            self.epoch.append(evaluations)
            self.fitness.append(fitness)

            LOGGER.info("Evaluations: {}. fitness: {}".format(evaluations, fitness))


if __name__ == "__main__":
    problem = MyRastrigin(50)
    evaluations = 5000
    population_size = 32

    algorithms = [
        GeneticAlgorithm(
            problem=problem,
            population_size=population_size,
            offspring_population_size=population_size // 2,
            mutation=UniformMutation(probability=0.1),
            crossover=SBXCrossover(probability=0.9),
            selection=BestSolutionSelection(),
            termination_criterion=StoppingByEvaluations(evaluations),
        ),
        SocioSSGA(
            problem=problem,
            population_size=population_size,
            offspring_population_size=1,
            interaction_probability=0.5,
            mutation_probability=0.1,
            crossover=SBXCrossover(probability=0.9),
            basic_prob=0.2,
            trust_prob=0.7,
            cost_prob=0.1,
            max_switched_genes=4,
            termination_criterion=StoppingByEvaluations(evaluations),
        ),
    ]
    fitness = []
    epoch = []

    for algorithm in algorithms:
        observer = PrintObjectivesObserver(1)
        algorithm.observable.register(observer)

        algorithm.run()

        fitness.append(observer.fitness)
        epoch.append(observer.epoch)

    plt.xlabel("Ewaluacje")
    plt.ylabel("Fitness")
    plt.title("Porównanie różnych operatorów mutacji")
    for i, data in enumerate(zip(epoch, fitness, algorithms)):
        plt.plot(data[0], data[1], label=data[2].get_name())
    plt.legend()
    plt.show()
