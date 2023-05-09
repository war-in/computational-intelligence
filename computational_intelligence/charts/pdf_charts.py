import os
from datetime import datetime

from jmetal.operator import SBXCrossover, UniformMutation
from jmetal.operator.selection import RouletteWheelSelection
from jmetal.problem.singleobjective.unconstrained import Rastrigin, Sphere
from jmetal.util.termination_criterion import StoppingByEvaluations
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from computational_intelligence.algorithms.socjo_SSGA import (
    PrintObjectivesObserver2,
    SocioSSGA,
)


def save_to_pdf(problems, plots_per_problem, note):
    dir_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    os.mkdir(str(dir_name))

    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]

    for i, problem in enumerate(problems):
        filename = (
            str(dir_name)
            + "/"
            + problem.get_name()
            + str(problem.number_of_variables)
            + ".pdf"
        )
        pdf = PdfPages(filename)
        pdf.attach_note(note, [0, 0])
        for j in range(0, plots_per_problem):
            fig = figs[i * plots_per_problem + j]
            fig.savefig(pdf, format="pdf")

        pdf.close()


if __name__ == "__main__":
    basic_probs = [0.1, 0.2, 0.3, 0.1]
    trust_probs = [0.6, 0.7, 0.6, 0.5]
    cost_probs = [0.3, 0.1, 0.1, 0.4]

    population_size = 100
    offspring_population_size = 1
    interaction_probability = 0.5
    mutation_probability = 0.1
    crossover_probability = 0.9
    evaluations = 10000
    observer_freq = 10

    # basic_probs = [0.1]
    # trust_probs = [0.6]
    # cost_probs = [0.3]

    sizes = [50, 100, 200]
    problems = []

    for size in sizes:
        problems.append(Rastrigin(size))
        problems.append(Sphere(size))

    plots_per_problem = 2  # how many charts should be printed for each problem
    number_of_trials = 1  # number of tests per problem

    epoch = []
    fitness = []
    average_fitness = []

    test_data = (
        "Population size: "
        + str(population_size)
        + "\n"
        + "Offspring size: "
        + str(offspring_population_size)
        + "\n"
        + "Interaction probability: "
        + str(interaction_probability)
        + "\n"
        + "Mutation probability: "
        + str(mutation_probability)
        + "\n"
        + "Crossover probability: "
        + str(crossover_probability)
        + "\n"
        + "Evaluations: "
        + str(evaluations)
    )

    for problem in problems:
        epoch = []
        fitness = []
        average_fitness = []

        for i, data in enumerate(zip(basic_probs, trust_probs, cost_probs)):
            # for j in range(0, number_of_trials):

            socio = SocioSSGA(
                problem=problem,
                population_size=population_size,
                offspring_population_size=offspring_population_size,
                interaction_probability=interaction_probability,
                selection=RouletteWheelSelection(),
                mutation=UniformMutation(mutation_probability),
                crossover=SBXCrossover(crossover_probability),
                basic_prob=data[0],
                trust_prob=data[1],
                cost_prob=data[2],
                max_switched_genes=int(problem.number_of_variables * 0.75),
                termination_criterion=StoppingByEvaluations(evaluations),
            )

            observer = PrintObjectivesObserver2(observer_freq)
            socio.observable.register(observer)

            socio.run()
            print(
                "Problem: "
                + problem.get_name()
                + " "
                + str(problem.number_of_variables)
            )
            print(len(observer.epoch))

            epoch.append(observer.epoch)
            fitness.append(observer.fitness)
            average_fitness.append(observer.average_fitness)

            # epoch = [x + y for x, y in zip(epoch, observer.epoch)]
            # fitness = [x + y for x, y in zip(epoch, observer.fitness)]
            # average_fitness = [x + y for x, y in zip(epoch, observer.average_fitness)]

        plt.figure()
        plt.xlabel("Ewaluacje")
        plt.ylabel("Fitness")
        plt.title("Comparison of different probabilities (fitness vs average fitness)")
        for i, data2 in enumerate(zip(epoch, fitness, average_fitness)):
            plt.plot(data2[0], data2[1], label="fitness " + str(i))
            plt.plot(data2[0], data2[2], label="average_fitness " + str(i))
        plt.legend()
        # plt.show()

        plt.figure()
        plt.xlabel("Ewaluacje")
        plt.ylabel("Fitness")
        plt.title("Comparison of different probabilities (fitness)")
        for i, data2 in enumerate(zip(epoch, fitness, basic_probs)):
            plt.plot(data2[0], data2[1], label="Basic prob: " + str(data2[2]))
        plt.legend()
        # plt.show()

    save_to_pdf(problems, plots_per_problem, test_data)
