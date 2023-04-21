import random
from typing import Dict, List

import matplotlib.pyplot as plt
from age_classes import MyRastrigin
from jmetal.config import store
from jmetal.core.algorithm import EvolutionaryAlgorithm, R, S
from jmetal.core.observer import Observer
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution
from jmetal.problem import Sphere
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.observer import LOGGER
from jmetal.util.termination_criterion import (
    StoppingByEvaluations,
    TerminationCriterion,
)
from numpy import interp


class SocioSSGA(EvolutionaryAlgorithm[S, R]):
    """
    Socio-cognitive steady state genetic algorithm.
    """

    ranking: Dict[S, int] = {}
    """
    Current ranking for solutions.
    ranking[solution] - "trust" for solution
    """
    MAX_TRUST: int = 100
    MIN_TRUST: int = 0
    BASIC_PROB: float  # Minimal probablity that the exchange will occur
    TRUST_PROB: float  # Weight of probability gained from trust ranking
    COST_PROB: float  # Weight of probability gained from better evaluation

    def __init__(
        self,
        problem: Problem[S],
        population_size: int,
        offspring_population_size: int,
        interaction_probability: float,
        basic_prob: float,
        trust_prob: float,
        cost_prob: float,
        termination_criterion: TerminationCriterion = store.default_termination_criteria,
        population_generator: Generator = store.default_generator,
        population_evaluator: Evaluator = store.default_evaluator,
    ):
        super().__init__(problem, population_size, offspring_population_size)

        self.BASIC_PROB = basic_prob
        self.TRUST_PROB = trust_prob
        self.COST_PROB = cost_prob

        # TODO: check if the probablities all sum up to 1.0

        self.interaction_probability = interaction_probability

        self.termination_criterion = termination_criterion

        self.population_generator = population_generator
        self.population_evaluator = population_evaluator

        self.observable.register(termination_criterion)

    def create_initial_solutions(self) -> List[S]:
        sol = [
            self.population_generator.new(self.problem)
            for _ in range(self.population_size)
        ]

        # set trust for each solution to self.MAX_TRUST // 2
        for s in sol:
            self.ranking[s] = self.MAX_TRUST // 2

        return sol

    def step(self):
        interacting_population = self.selection(self.solutions)
        mutated_population = self.mutation(interacting_population)
        self.evaluate(mutated_population)

    def selection(self, population: List[S]) -> List[S]:
        """
        Select interacting individuals.

        :param population: Entire population.
        :return: Selected solutions.
        """
        return random.sample(
            population, int(len(population) * self.interaction_probability * 2) // 2
        )

    def mutation(self, interacting_population: List[FloatSolution]) -> List[S]:
        """
        Perform interaction between two individuals (indexes [0,1], [2,3], ...).
        Change their genes.

        :param interacting_population: Even-length list of solutions.
        :return: Mutated list of solutions.
        """
        length = len(interacting_population)
        if length % 2 != 0:
            raise ValueError("List is not even-length.")

        for ind1, ind2 in zip(
            interacting_population[: length // 2], interacting_population[length // 2 :]
        ):
            trust_probablity = self.TRUST_PROB * (self.ranking[ind2] / self.MAX_TRUST)

            # TODO How to compute the Cost probablity ?
            if ind1.objectives[0] < ind2.objectives[0]:
                cost_probability = 0
            else:
                cost_probability = self.COST_PROB

            exchange_probability = self.BASIC_PROB + trust_probablity + cost_probability

            if random.uniform(0.0, 1.0) < exchange_probability:
                # TODO Implement crossover instead of simpe switching half of genes
                ind1.variables[: ind1.number_of_variables // 2] = ind2.variables[
                    : ind2.number_of_variables // 2
                ]

                # TODO Check if the solution has better evaluation due to exchange and modify the trust ranking accordingly
                # new_evaluation = self.evaluate([ind1])
                # self.ranking[ind2] += 1

            """if ind1.objectives[0] < ind2.objectives[0]:
                ind2.variables[: ind2.number_of_variables // 2] = ind1.variables[
                    : ind1.number_of_variables // 2
                ]

                self.ranking[ind1] += 1

            else:
                ind1.variables[: ind1.number_of_variables // 2] = ind2.variables[
                    : ind2.number_of_variables // 2
                ]

                self.ranking[ind2] += 1"""

        return interacting_population

    def evaluate(self, solution_list: List[S]) -> List[S]:
        """
        Evaluate solutions.

        :param solution_list: List of solutions.
        """
        return self.population_evaluator.evaluate(solution_list, self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def reproduction(self, population: List[S]) -> List[S]:
        pass

    def replacement(
        self, population: List[S], offspring_population: List[S]
    ) -> List[S]:
        pass

    def get_result(self) -> R:
        return sorted(self.solutions, key=lambda solution: solution.objectives[0])[0]

    def get_name(self) -> str:
        return "Socio-cognitive SSGA"


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
                fitness = solutions[0].objectives
            else:
                fitness = solutions.objectives

            self.epoch.append(evaluations)
            self.fitness.append(fitness)

            LOGGER.info("Evaluations: {}. fitness: {}".format(evaluations, fitness))


if __name__ == "__main__":
    problem = MyRastrigin()

    fitness = []

    socio = SocioSSGA(
        problem=problem,
        population_size=100,
        offspring_population_size=1,
        interaction_probability=1.0,
        basic_prob=0.98,
        trust_prob=0.01,
        cost_prob=0.01,
        termination_criterion=StoppingByEvaluations(1000),
    )

    observer = PrintObjectivesObserver(10)
    socio.observable.register(observer)

    socio.run()

    fitness.append(observer.fitness)

    print(socio.get_result())

    plt.plot(observer.fitness)
    plt.show()
