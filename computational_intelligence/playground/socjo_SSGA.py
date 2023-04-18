import random
from typing import Dict, List

from jmetal.config import store
from jmetal.core.algorithm import EvolutionaryAlgorithm, R, S
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution
from jmetal.problem import Sphere
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import (
    StoppingByEvaluations,
    TerminationCriterion,
)


class SocioSSGA(EvolutionaryAlgorithm[S, R]):
    """
    Socio-cognitive steady state genetic algorithm.
    """

    ranking: Dict[S, int]
    """
    Current ranking for solutions.
    ranking[solution] - "trust" for solution
    """

    def __init__(
        self,
        problem: Problem[S],
        population_size: int,
        offspring_population_size: int,
        interaction_probability: float,
        termination_criterion: TerminationCriterion = store.default_termination_criteria,
        population_generator: Generator = store.default_generator,
        population_evaluator: Evaluator = store.default_evaluator,
    ):
        super().__init__(problem, population_size, offspring_population_size)

        self.interaction_probability = interaction_probability

        self.termination_criterion = termination_criterion

        self.population_generator = population_generator
        self.population_evaluator = population_evaluator

        self.observable.register(termination_criterion)

    def create_initial_solutions(self) -> List[S]:
        return [
            self.population_generator.new(self.problem)
            for _ in range(self.population_size)
        ]

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
            if ind1.objectives[0] < ind2.objectives[0]:
                ind2.variables[: ind2.number_of_variables // 2] = ind1.variables[
                    : ind1.number_of_variables // 2
                ]
            else:
                ind1.variables[: ind1.number_of_variables // 2] = ind2.variables[
                    : ind2.number_of_variables // 2
                ]

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


if __name__ == "__main__":
    problem = Sphere(50)
    socio = SocioSSGA(
        problem=problem,
        population_size=100,
        offspring_population_size=1,
        interaction_probability=0.3,
        termination_criterion=StoppingByEvaluations(1000),
    )
    socio.run()

    print(socio.get_result())
