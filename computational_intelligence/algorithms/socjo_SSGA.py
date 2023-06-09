import random
from copy import copy
from typing import Dict, List

from jmetal.config import store
from jmetal.core.algorithm import EvolutionaryAlgorithm, R, S
from jmetal.core.observer import Observer
from jmetal.core.operator import Crossover, Mutation, Selection
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.observer import LOGGER
from jmetal.util.termination_criterion import TerminationCriterion


class SocioSSGA(EvolutionaryAlgorithm[S, R]):
    """
    Socio-cognitive steady state genetic algorithm.
    """

    ranking: Dict[int, int] = {}
    """
    Current ranking for solutions.
    ranking[solution.id] - "trust" for solution
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
        selection: Selection,
        mutation: Mutation,
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
        self.selection_operator = selection
        self.mutation_operator = mutation

        self.termination_criterion = termination_criterion

        self.population_generator = population_generator
        self.population_evaluator = population_evaluator

        self.observable.register(termination_criterion)

        self.interacting_pool_size = (
            int(self.population_size * self.interaction_probability // 2) * 2
        )

    def create_initial_solutions(self) -> List[S]:
        solutions = [
            self.population_generator.new(self.problem)
            for _ in range(self.population_size)
        ]

        # set index and trust for each solution
        for index, solution in enumerate(solutions):
            solution.id = index
            self.ranking[solution.id] = self.MAX_TRUST // 2

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
        interacting_population = []

        for _ in range(self.interacting_pool_size):
            solution = self.selection_operator.execute(population)
            interacting_population.append(solution)

        return interacting_population

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

        for index in range(0, length, 2):
            ind1 = interacting_population[index]
            ind2 = interacting_population[index + 1]

            trust_probability = self.TRUST_PROB * (
                self.ranking[ind2.id] / self.MAX_TRUST
            )

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

                self.mutation_operator.execute(ind1)

                new_evaluation = self.evaluate([ind1])
                self.evaluations += 1

                if new_evaluation[0].objectives[0] < old_evaluation:
                    self.ranking[ind2.id] += 1
                    self.ranking[ind1.id] -= 1

                else:
                    self.ranking[ind2.id] -= 1
                    self.ranking[ind1.id] += 1

                    ind1.variables = old_variables
                    ind1.objectives[0] = old_evaluation

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

    def update_progress(self) -> None:
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def get_observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.get_result(),
            "AVERAGE_SOLUTIONS": self.get_average_objective(),
            "RANKING": self.ranking,
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
