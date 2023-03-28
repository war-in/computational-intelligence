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


class PrintObjectivesObserver(Observer):
    def __init__(self, frequency: float = 1.0) -> None:
        """Show the number of evaluations, best fitness and computing time.

        :param frequency: Display frequency."""
        self.display_frequency = frequency
        self.epoch = []
        self.fitness = []

    def update(self, *args, **kwargs):
        evaluations = kwargs["EVALUATIONS"]
        solutions = kwargs["SOLUTIONS"]

        if (evaluations % self.display_frequency) == 0 and solutions:
            if type(solutions) == list:
                fitness = solutions[0].objectives
            else:
                fitness = solutions.objectives

            self.epoch.append(evaluations)
            self.fitness.append(fitness)

            LOGGER.info("Evaluations: {}. fitness: {}".format(evaluations, fitness))


if __name__ == "__main__":
    problem = Rastrigin()

    algorithm = GeneticAlgorithm(
        problem=problem,
        population_size=100,
        offspring_population_size=1,
        mutation=UniformMutation(probability=0.1),
        crossover=SBXCrossover(probability=0.9),
        selection=RouletteWheelSelection(),
        termination_criterion=StoppingByEvaluations(1000),
    )

    observer = PrintObjectivesObserver(10)
    algorithm.observable.register(observer)

    algorithm.run()
    solutions = algorithm.solutions

    plt.plot(observer.fitness)
    plt.show()
