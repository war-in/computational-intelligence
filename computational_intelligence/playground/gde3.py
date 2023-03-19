"""
GDE3 algorithm for ZDT1 problem
"""
from jmetal.algorithm.multiobjective.gde3 import GDE3
from jmetal.lab.visualization.plotting import Plot
from jmetal.problem import ZDT1
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == "__main__":
    problem = ZDT1()

    MAX_EVALUATIONS = 25000

    algorithm = GDE3(
        problem=problem,
        population_size=100,
        cr=0.5,
        f=0.5,
        termination_criterion=StoppingByEvaluations(MAX_EVALUATIONS),
    )

    algorithm.run()
    solutions = algorithm.get_result()

    front = get_non_dominated_solutions(solutions)

    plot_front = Plot(title="Pareto front approximation", axis_labels=["x", "y"])
    plot_front.plot(front, label="GDE3-ZDT1")
