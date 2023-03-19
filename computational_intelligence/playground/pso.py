"""
SMPSO algorithm for ZDT1 problem
"""
from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.lab.visualization import Plot
from jmetal.operator import PolynomialMutation
from jmetal.problem import ZDT1
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == "__main__":
    problem = ZDT1()
    MAX_EVALUATIONS = 25000

    algorithm = SMPSO(
        problem=problem,
        swarm_size=100,
        mutation=PolynomialMutation(
            probability=1.0 / problem.number_of_variables, distribution_index=20
        ),
        leaders=CrowdingDistanceArchive(100),
        termination_criterion=StoppingByEvaluations(max_evaluations=MAX_EVALUATIONS),
    )

    algorithm.run()

    solutions = algorithm.get_result()

    front = get_non_dominated_solutions(solutions)

    plot_front = Plot(title="Pareto front approximation", axis_labels=["x", "y"])
    plot_front.plot(front, label="SMPSO-ZDT1")
