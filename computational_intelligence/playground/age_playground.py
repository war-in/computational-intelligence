import matplotlib.pyplot as plt
from age_classes import MyAlgorithm, MyRastrigin
from jmetal.core.observer import Observer
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
        self.age = []

    def update(self, *args, **kwargs):
        evaluations = kwargs["EVALUATIONS"]
        solutions = kwargs["SOLUTIONS"]
        average_age = kwargs["AVERAGE_AGE"]

        if (evaluations % self.display_frequency) == 0 and solutions:
            if type(solutions) == list:
                fitness = solutions[0].objectives
            else:
                fitness = solutions.objectives

            self.epoch.append(evaluations)
            self.fitness.append(fitness)
            self.age.append(average_age)

            LOGGER.info("Evaluations: {}. fitness: {}".format(evaluations, fitness))
            LOGGER.info("Average age: {}".format(average_age))


if __name__ == "__main__":
    offspring_sizes = [0.1, 0.2, 0.5]
    population_size = 100

    avg_age = []

    for offspring_size in offspring_sizes:
        problem = MyRastrigin()

        algorithm = MyAlgorithm(
            problem=problem,
            population_size=population_size,
            offspring_population_size=int(offspring_size * population_size),
            mutation=UniformMutation(probability=0.1),
            crossover=SBXCrossover(probability=0.9),
            selection=RouletteWheelSelection(),
            termination_criterion=StoppingByEvaluations(
                100 * int(offspring_size * population_size)
            ),
        )

        observer = PrintObjectivesObserver(10)
        algorithm.observable.register(observer)

        algorithm.run()

        avg_age.append(observer.age)

    plt.xlabel("Iteracja")
    plt.ylabel("Wiek")
    # plt.title("A test graph")

    for i, data in enumerate(zip(avg_age, offspring_sizes)):
        print(i)
        print(data)
        plt.plot(data[0], label=str(offspring_sizes[i] * population_size))

    plt.legend()
    plt.show()