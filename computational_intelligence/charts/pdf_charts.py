import os
import numpy as np
from datetime import datetime
from jmetal.operator import SBXCrossover, UniformMutation
from jmetal.operator.selection import RouletteWheelSelection
from jmetal.problem.singleobjective.unconstrained import Rastrigin, Sphere
from jmetal.util.termination_criterion import StoppingByEvaluations
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from algorithms.socjo_SSGA import SocioObserver, SocioSSGA

def get_deviation(observer, problem_size):

    all_deviations = []
    for all_variables in observer.all_variables_per_evaluation:
        variables_by_index = [[]] * problem_size
        for variables in all_variables:
            for index, variable in enumerate(variables):
                variables_by_index[index].append(variable)

        deviations = [np.std(variables) for variables in variables_by_index]
        all_deviations.append(min(deviations))

    return all_deviations

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
    #basic_probs = [0.1, 0.2, 0.3, 0.1]
    #trust_probs = [0.6, 0.7, 0.6, 0.5]
    #cost_probs = [0.3, 0.1, 0.1, 0.4]

    population_size = 100
    offspring_population_size = 1
    interaction_probability = 0.5
    mutation_probability = 0.1
    crossover_probability = 0.9
    evaluations = 200
    observer_freq = 10

    basic_probs = [0.1, 0.2]
    trust_probs = [0.6, 0.7]
    cost_probs = [0.3, 0.1]

    sizes = [50, 100, 200]
    problems = []

    for size in sizes:
        problems.append(Rastrigin(size))
        problems.append(Sphere(size))

    plots_per_problem = 4  # how many charts should be printed for each problem
    number_of_trials = 2  # number of tests per problem

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
        all_deviations = []
        rankings = []

        for i, data in enumerate(zip(basic_probs, trust_probs, cost_probs)):

            trial_epoch = []
            trial_fitness = []
            trial_average_fitness = []
            trial_all_deviations = []
            trial_rankings = []

            for j in range(0, number_of_trials):

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

                observer = SocioObserver(observer_freq)
                socio.observable.register(observer)

                socio.run()
                print(
                    "Problem: "
                    + problem.get_name()
                    + " "
                    + str(problem.number_of_variables)
                )

                new_deviations = get_deviation(observer, problem.number_of_variables)

                if j == 0:
                    trial_epoch.extend(observer.epoch)
                    trial_fitness.extend(observer.fitness)
                    trial_average_fitness.extend(observer.average_fitness)
                    trial_all_deviations.extend(new_deviations)
                    trial_rankings.extend(observer.rankings)

                else:
                    min_len = min(len(trial_fitness), len(observer.fitness))
                    trial_epoch = trial_epoch[0:min_len]
                    trial_fitness = [x + y for x, y in zip(trial_fitness[0:min_len], observer.fitness[0:min_len])]
                    trial_average_fitness = [x + y for x, y in zip(trial_average_fitness[0:min_len], observer.average_fitness[0:min_len])]
                    trial_all_deviations = [x + y for x, y in zip( trial_all_deviations, new_deviations)]
                    trial_rankings = [x + y for x, y in zip( trial_rankings, observer.rankings)]

            trial_fitness = [x / number_of_trials for x in trial_fitness]
            trial_average_fitness = [x / number_of_trials for x in trial_average_fitness]
            trial_all_deviations = [ x / number_of_trials for x in trial_all_deviations]
            trial_rankings = [x / number_of_trials for x in trial_rankings]

            epoch.append(trial_epoch)
            fitness.append(trial_fitness)
            average_fitness.append(trial_average_fitness)
            all_deviations.append(trial_all_deviations)
            rankings.append(trial_rankings)

        plt.figure()
        plt.xlabel("Ewaluacje")
        plt.ylabel("Fitness")
        plt.title("Comparison of different probabilities (fitness vs average fitness)")
        for i, data2 in enumerate(zip(epoch, fitness, average_fitness)):
            plt.plot(data2[0], data2[1], label="fitness " + str(basic_probs[i]) + " " + str(trust_probs[i]) + " " + str(cost_probs[i]))
            plt.plot(data2[0], data2[2], label="average_fitness " + str(basic_probs[i]) + " " + str(trust_probs[i]) + " " + str(cost_probs[i]))
        plt.legend()
        #plt.show()

        plt.figure()
        plt.xlabel("Ewaluacje")
        plt.ylabel("Fitness")
        plt.title("Comparison of different probabilities (fitness)")
        for i, data2 in enumerate(zip(epoch, fitness, basic_probs)):
            plt.plot(data2[0], data2[1], label="Basic prob: " + str(basic_probs[i]) + " " + str(trust_probs[i]) + " " + str(cost_probs[i]))
        plt.legend()
        #plt.show()

        plt.figure()
        plt.xlabel("Ewaluacje")
        plt.ylabel("Fitness")
        plt.title("Standard Deviation")
        for i, data2 in enumerate(zip(epoch, all_deviations)):
            plt.plot(data2[0], data2[1], label="std: " + str(basic_probs[i]) + " " + str(trust_probs[i]) + " " + str(cost_probs[i]))
        plt.legend()
        #plt.show()
        
        # create labels for boxplot
        labels = []
        for i in range(0, len(basic_probs)):
            labels.append(str(basic_probs[i]) + " " + str(trust_probs[i]) + " " + str(cost_probs[i]))

        # plotting rankings
        fig = plt.figure(figsize =(10, 7))
        ax = fig.add_subplot(111)
        ax.set_yticklabels(labels)
        bp = ax.boxplot(rankings, patch_artist = True,
                    notch ='True', vert = 0)
        #plt.show()

    save_to_pdf(problems, plots_per_problem, test_data)
