import matplotlib.pyplot as plt
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.core.observer import Observer
from jmetal.operator import UniformMutation
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.selection import RouletteWheelSelection
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.observer import LOGGER
from jmetal.util.termination_criterion import StoppingByEvaluations

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

            LOGGER.info(f"Evaluations: {evaluations}. fitness: {fitness}")


if __name__ == "__main__":
    problem = Rastrigin(200)
    evaluations = 20000
    population_size = 48

    algorithms = [
        GeneticAlgorithm(
            problem=problem,
            population_size=population_size,
            offspring_population_size=population_size // 2,
            mutation=UniformMutation(probability=0.1),
            crossover=SBXCrossover(probability=0.9),
            selection=RouletteWheelSelection(),
            termination_criterion=StoppingByEvaluations(evaluations),
        ),
        SocioSSGA(
            problem=problem,
            population_size=population_size,
            offspring_population_size=1,
            interaction_probability=0.5,
            selection=RouletteWheelSelection(),
            mutation=UniformMutation(probability=0.1),
            crossover=SBXCrossover(probability=0.9),
            basic_prob=0.2,
            trust_prob=0.7,
            cost_prob=0.1,
            max_switched_genes=int(problem.number_of_variables * 0.75),
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
