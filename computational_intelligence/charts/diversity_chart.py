import matplotlib.pyplot as plt
import numpy as np
from jmetal.operator import SBXCrossover, UniformMutation
from jmetal.operator.selection import RouletteWheelSelection
from jmetal.problem.singleobjective.unconstrained import Rastrigin, Sphere
from jmetal.util.termination_criterion import StoppingByEvaluations

from computational_intelligence.algorithms.socjo_SSGA import SocioObserver, SocioSSGA


def main():
    problem_size = 100
    problem = Rastrigin(problem_size)

    socio = SocioSSGA(
        problem=problem,
        population_size=32,
        offspring_population_size=1,
        interaction_probability=0.5,
        selection=RouletteWheelSelection(),
        mutation=UniformMutation(0.1),
        crossover=SBXCrossover(0.9),
        basic_prob=0.1,
        trust_prob=0.6,
        cost_prob=0.3,
        max_switched_genes=int(problem.number_of_variables * 0.75),
        termination_criterion=StoppingByEvaluations(10000),
    )

    observer = SocioObserver(10)
    socio.observable.register(observer)

    socio.run()

    all_deviations = []
    for all_variables in observer.all_variables_per_evaluation:
        variables_by_index = [[]] * problem_size
        for variables in all_variables:
            for index, variable in enumerate(variables):
                variables_by_index[index].append(variable)

        deviations = [np.std(variables) for variables in variables_by_index]
        all_deviations.append(min(deviations))

    print(all_deviations)

    plt.plot(all_deviations)
    plt.show()


if __name__ == "__main__":
    main()
