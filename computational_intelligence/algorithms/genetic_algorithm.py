from jmetal.core.observer import Observer
from jmetal.util.observer import LOGGER


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
