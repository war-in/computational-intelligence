import math
import random
import time
from typing import List, TypeVar

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.operator import Crossover
from jmetal.core.problem import FloatProblem, Problem
from jmetal.core.solution import BinarySolution
from jmetal.operator.crossover import Crossover
from jmetal.operator.mutation import Mutation
from jmetal.operator.selection import Selection
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.observer import LOGGER
from jmetal.util.solution import FloatSolution, Solution
from jmetal.util.termination_criterion import TerminationCriterion


class MySolution(FloatSolution):
    def __init__(self, lower_bound: list, upper_bound: list, number_of_variables: int):
        super(MySolution, self).__init__(lower_bound, upper_bound, number_of_variables)
        self.age = 0  # additional parameter - age of the solution

    def copy(self) -> "FloatSolution":
        new_solution = MySolution(
            self.lower_bound, self.upper_bound, self.number_of_variables
        )
        new_solution.variables = self.variables.copy()
        new_solution.objectives = self.objectives.copy()
        new_solution.attributes = self.attributes.copy()
        return new_solution

    def __hash__(self):
        return id(self)


class MyFloatProblem(FloatProblem):
    def __init__(self):
        super().__init__()

    def create_solution(self) -> MySolution:
        new_solution = MySolution(
            self.lower_bound, self.upper_bound, self.number_of_objectives
        )
        new_solution.variables = [
            random.uniform(self.lower_bound[i] * 1.0, self.upper_bound[i] * 1.0)
            for i in range(self.number_of_variables)
        ]

        return new_solution


class MyRastrigin(MyFloatProblem):
    def __init__(self, number_of_variables: int = 10):
        super(MyRastrigin, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]

        self.lower_bound = [-5.12 for _ in range(number_of_variables)]
        self.upper_bound = [5.12 for _ in range(number_of_variables)]

        MyFloatProblem.lower_bound = self.lower_bound
        MyFloatProblem.upper_bound = self.upper_bound

    def evaluate(self, solution: MySolution) -> MySolution:
        a = 10.0
        result = a * solution.number_of_variables
        x = solution.variables

        for i in range(solution.number_of_variables):
            result += x[i] * x[i] - a * math.cos(2 * math.pi * x[i])

        solution.objectives[0] = result

        # solution.age += 1

        return solution

    def get_name(self) -> str:
        return "Rastrigin"


S = TypeVar("S", bound=Solution)


class MyAlgorithm(GeneticAlgorithm):
    def __init__(
        self,
        problem: Problem,
        population_size: int,
        offspring_population_size: int,
        mutation: Mutation,
        crossover: Crossover,
        selection: Selection,
        termination_criterion: TerminationCriterion = store.default_termination_criteria,
        population_generator: Generator = store.default_generator,
        population_evaluator: Evaluator = store.default_evaluator,
    ):
        super(MyAlgorithm, self).__init__(
            problem,
            population_size,
            offspring_population_size,
            mutation,
            crossover,
            selection,
            termination_criterion,
            population_generator,
            population_evaluator,
        )

    def get_observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.get_result(),
            "COMPUTING_TIME": time.time() - self.start_computing_time,
            "AVERAGE_AGE": self.get_avg_age(),
        }

    def reproduction(self, mating_population: List[S]) -> List[S]:
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()
        if len(mating_population) % number_of_parents_to_combine != 0:
            raise Exception("Wrong number of parents")

        offspring_population = []
        for i in range(0, self.offspring_population_size, number_of_parents_to_combine):
            parents = []
            for j in range(number_of_parents_to_combine):
                parents.append(mating_population[i + j])

            offspring = self.crossover_operator.execute(parents)

            for solution in offspring:
                solution.age = 0
                offspring_population.append(solution)
                if len(offspring_population) >= self.offspring_population_size:
                    break

        return offspring_population

    def get_avg_age(self):
        avg = 0
        for sol in self.solutions:
            avg += sol.age

        return avg / len(self.solutions)

    def replacement(
        self, population: List[S], offspring_population: List[S]
    ) -> List[S]:
        population.extend(offspring_population)

        population.sort(key=lambda s: s.objectives[0])

        return population[: self.population_size]

    def increase_age(self):
        for s in self.solutions:
            s.age += 1

    def step(self):
        mating_population = self.selection(self.solutions)
        offspring_population = self.reproduction(mating_population)
        offspring_population = self.evaluate(offspring_population)

        self.solutions = self.replacement(self.solutions, offspring_population)
        self.increase_age()
