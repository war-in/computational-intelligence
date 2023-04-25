import random
from copy import copy
from typing import Dict, List

import matplotlib.pyplot as plt
from age_classes import MyRastrigin
from jmetal.config import store
from jmetal.core.algorithm import EvolutionaryAlgorithm, R, S
from jmetal.core.observer import Observer
from jmetal.core.operator import Crossover
from jmetal.core.problem import FloatProblem, Problem
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import SBXCrossover
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.observer import LOGGER
from jmetal.util.termination_criterion import (
    StoppingByEvaluations,
    TerminationCriterion,
)


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
    BASIC_PROB: float  # Minimal probability that the exchange will occur
    TRUST_PROB: float  # Weight of probability gained from trust ranking
    COST_PROB: float  # Weight of probability gained from better evaluation
    MAX_SWITCHED_GENES: int

    def __init__(
        self,
        problem: Problem[S],
        population_size: int,
        offspring_population_size: int,
        interaction_probability: float,
        mutation_probability: float,
        crossover: Crossover,
        basic_prob: float,
        trust_prob: float,
        cost_prob: float,
        max_switched_genes: int,
        termination_criterion: TerminationCriterion = store.default_termination_criteria,
        population_generator: Generator = store.default_generator,
        population_evaluator: Evaluator = store.default_evaluator,
    ):
        super().__init__(problem, population_size, offspring_population_size)

        self.BASIC_PROB = basic_prob
        self.TRUST_PROB = trust_prob
        self.COST_PROB = cost_prob
        self.MAX_SWITCHED_GENES = max_switched_genes

        if sum([self.COST_PROB, self.TRUST_PROB, self.BASIC_PROB]) - 1.0 > 1e-10:
            raise ValueError("All probabilities should sum up to 1.0")

        self.crossover_operator = crossover

        self.interaction_probability = interaction_probability
        self.mutation_probability = mutation_probability

        self.termination_criterion = termination_criterion

        self.population_generator = population_generator
        self.population_evaluator = population_evaluator

        self.observable.register(termination_criterion)

    def create_initial_solutions(self) -> List[S]:
        solutions = [
            self.population_generator.new(self.problem)
            for _ in range(self.population_size)
        ]

        # set trust for each solution to self.MAX_TRUST // 2
        for solution in solutions:
            self.ranking[solution] = self.MAX_TRUST // 2

        return solutions

    def step(self):
        interacting_population = self.selection(self.solutions)
        self.interaction(interacting_population)

    def selection(self, population: List[S]) -> List[S]:
        """
        Select interacting individuals.

        :param population: Entire population.
        :return: Selected solutions.
        """
        return random.sample(
            population, int(len(population) * self.interaction_probability * 2) // 2
        )

    def interaction(self, interacting_population: List[FloatSolution]):
        """
        Perform interaction between two individuals (indexes [0,1], [2,3], ...).
        Change their genes.
        Only first from the pair is changed.

        :param interacting_population: Even-length list of solutions.
        :return: List of changed solutions.
        """
        length = len(interacting_population)
        if length % 2 != 0:
            raise ValueError("List is not even-length.")

        for ind1, ind2 in zip(
            interacting_population[: length // 2], interacting_population[length // 2 :]
        ):
            trust_probability = self.TRUST_PROB * (self.ranking[ind2] / self.MAX_TRUST)

            # TODO How to compute the Cost probablity ?
            if ind1.objectives[0] < ind2.objectives[0]:
                cost_probability = 0
            else:
                cost_probability = self.COST_PROB

            exchange_probability = (
                self.BASIC_PROB + trust_probability + cost_probability
            )

            if random.uniform(0.0, 1.0) < exchange_probability:
                old_evaluation = ind1.objectives[0]
                old_variables = copy(ind1.variables)

                genes_to_switch = int(
                    (trust_probability / self.TRUST_PROB) * self.MAX_SWITCHED_GENES
                )
                for _ in range(genes_to_switch):
                    gene_to_switch = random.randint(
                        0, self.problem.number_of_variables - 1
                    )
                    ind1.variables[gene_to_switch] = ind2.variables[gene_to_switch]

                self.mutation(ind1)

                new_evaluation = self.evaluate([ind1])

                if new_evaluation[0].objectives[0] < old_evaluation:
                    self.ranking[ind2] += 1
                    self.ranking[ind1] -= 1

                else:
                    self.ranking[ind2] -= 1
                    self.ranking[ind1] += 1

                    ind1.variables = old_variables
                    self.evaluate([ind1])

    def evaluate(self, solution_list: List[S]) -> List[S]:
        """
        Evaluate solutions.

        :param solution_list: List of solutions.
        """
        return self.population_evaluator.evaluate(solution_list, self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def mutation(self, individual: S):
        MAX_MUTATED_GENES = self.problem.number_of_variables // 10

        if random.uniform(0.0, 1.0) < self.mutation_probability:
            for _ in range(MAX_MUTATED_GENES):
                gene_to_switch = random.randint(0, self.problem.number_of_variables - 1)
                individual.variables[gene_to_switch] = random.uniform(
                    self.problem.lower_bound[gene_to_switch],
                    self.problem.upper_bound[gene_to_switch],
                )

    def reproduction(self, population: List[S]) -> List[S]:
        pass

    def replacement(
        self, population: List[S], offspring_population: List[S]
    ) -> List[S]:
        pass

    def get_observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.get_result(),
            "AVERAGE_SOLUTIONS": self.get_average_objective(),
        }

    def get_average_objective(self):
        avg = 0
        for sol in self.solutions:
            avg += sol.objectives[0]

        return avg / len(self.solutions)

    def get_result(self) -> R:
        return sorted(self.solutions, key=lambda solution: solution.objectives[0])[0]

    def get_name(self) -> str:
        return "Socio-cognitive SSGA"


class PrintObjectivesObserver2(Observer):
    def __init__(self, frequency: float = 1.0) -> None:
        """Show the number of evaluations, the best fitness and computing time.

        :param frequency: Display frequency."""
        self.display_frequency = frequency
        self.epoch = []
        self.fitness = []
        self.average_fitness = []

    def update(self, *args, **kwargs):
        evaluations = kwargs["EVALUATIONS"]
        solutions = kwargs["SOLUTIONS"]
        average_solutions = kwargs["AVERAGE_SOLUTIONS"]

        if (evaluations % self.display_frequency) == 0 and solutions:
            if isinstance(solutions, list):
                fitness = solutions[0].objectives
            else:
                fitness = solutions.objectives[0]

            self.epoch.append(evaluations)
            self.fitness.append(fitness)
            self.average_fitness.append(average_solutions)

            LOGGER.info("Evaluations: {}. fitness: {}".format(evaluations, fitness))


if __name__ == "__main__":
    basic_probs = [0.2, 0.3, 0.4, 0.1]
    trust_probs = [0.7, 0.6, 0.5, 0.5]
    cost_probs = [0.1, 0.1, 0.1, 0.4]

    problem = MyRastrigin(50)

    fitness = []
    average_fitness = []

    for i, data in enumerate(zip(basic_probs, trust_probs, cost_probs)):
        socio = SocioSSGA(
            problem=problem,
            population_size=100,
            offspring_population_size=1,
            interaction_probability=0.3,
            mutation_probability=0.05,
            crossover=SBXCrossover(probability=0.9),
            basic_prob=data[0],
            trust_prob=data[1],
            cost_prob=data[2],
            max_switched_genes=4,
            termination_criterion=StoppingByEvaluations(1500),
        )

        observer = PrintObjectivesObserver2(10)
        socio.observable.register(observer)

        socio.run()

        fitness.append(observer.fitness)
        average_fitness.append(observer.average_fitness)

    for i, data in enumerate(zip(fitness, average_fitness)):
        plt.title(
            "Basic prob: "
            + str(basic_probs[i])
            + " Trust prob: "
            + str(trust_probs[i])
            + " Cost prob: "
            + str(cost_probs[i])
        )
        plt.plot(data[0], label="fitness")
        plt.plot(data[1], label="average_fitness")
        plt.legend()
        plt.show()

    for i, data in enumerate(zip(fitness, basic_probs)):
        plt.plot(data[0], label=str(data[1]))

    plt.legend()
    plt.show()

    for i, data in enumerate(zip(average_fitness, basic_probs)):
        plt.plot(data[0], label=str(data[1]))

    plt.legend()
    plt.show()
