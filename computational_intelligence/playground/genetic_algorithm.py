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

        if (evaluations % self.display_frequency) == 0 and solutions:
            if isinstance(solutions, list):
                fitness = sorted(solutions, key=lambda solution: solution.objectives[0])
            else:
                fitness = solutions.objectives[0]

            self.epoch.append(evaluations)
            self.fitness.append(fitness)

            LOGGER.info("Evaluations: {}. fitness: {}".format(evaluations, fitness))


if __name__ == "__main__":
    problem = MyRastrigin(50)

    mutation_algorithms = [
        # PolynomialMutation(probability=0.1),
        # SimpleRandomMutation(probability=0.1),
        UniformMutation(probability=0.1),
    ]
    fitness = []

    for mutation_algorithm in mutation_algorithms:
        algorithm = SocioSSGA(
            problem=problem,
            population_size=100,
            offspring_population_size=1,
            interaction_probability=0.5,
            mutation_probability=0.05,
            crossover=SBXCrossover(probability=0.9),
            basic_prob=0.1,
            trust_prob=0.6,
            cost_prob=0.3,
            max_switched_genes=4,
            termination_criterion=StoppingByEvaluations(1500),
        )

        observer = PrintObjectivesObserver(10)
        algorithm.observable.register(observer)

        algorithm.run()

        fitness.append(observer.fitness)

    print(fitness)

    plt.xlabel("Ewaluacje")
    plt.ylabel("Fitness")
    plt.title("Porównanie różnych operatorów mutacji")
    for i, data in enumerate(zip(fitness, mutation_algorithms)):
        plt.plot(data[0], label=data[1].get_name())
    plt.legend()
    plt.show()
