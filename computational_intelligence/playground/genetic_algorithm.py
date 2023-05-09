import matplotlib.pyplot as plt
import os
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.core.observer import Observer
from jmetal.operator import UniformMutation
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.selection import RouletteWheelSelection
from jmetal.problem.singleobjective.unconstrained import Rastrigin, Sphere
from jmetal.util.observer import LOGGER
from jmetal.util.termination_criterion import StoppingByEvaluations

from socjo_SSGA import SocioSSGA


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


def save_to_pdf(problems, plots_per_problem, note):

    dir_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    os.mkdir(str(dir_name))

    fig_nums = plt.get_fignums()  
    figs = [plt.figure(n) for n in fig_nums]

    for i in range (0, len(problems)): 

        
        filename =  str(dir_name) + "/" + problems[i].get_name() + str(problems[i].number_of_variables) + ".pdf"
        p = PdfPages(filename)
        p.attach_note(note, [0,0])
        for j in range(0, plots_per_problem):

            fig = figs[i * plots_per_problem + j]
            fig.savefig(p, format='pdf') 

        
        p.close()  

if __name__ == "__main__":

    basic_probs = [0.1, 0.2, 0.3, 0.1]
    trust_probs = [0.6, 0.7, 0.6, 0.5]
    cost_probs = [0.3, 0.1, 0.1, 0.4]

    population_size = 100
    offspring_population_size = 1
    interaction_probability = 0.5
    mutation_probability = 0.1
    crossover_probability = 0.9
    evaluations = 20000
    observer_freq = 10

    #basic_probs = [0.1]
    #trust_probs = [0.6]
    #cost_probs = [0.3]

    sizes = [50, 100, 200]
    problems = []

    for size in sizes:
        problems.append(Rastrigin(size))
        problems.append(Sphere(size))

    plots_per_problem = 1 # how many charts should be printed for each problem
    number_of_trials = 1  # number of tests per problem 

    epoch = []
    fitness = []
    average_fitness = []

    test_data = ("Population size: " + str(population_size) + "\n" +
    "Offspring size: " + str(offspring_population_size) + "\n" +
    "Interaction probability: " + str(interaction_probability) + "\n" +
    "Mutation probability: " + str(mutation_probability) + "\n" +
    "Crossover probability: " + str(crossover_probability) + "\n" +
     "Evaluations: " + str(evaluations) )

    for problem in problems:

        algorithms = [
            GeneticAlgorithm(
                problem=problem,
                population_size=population_size,
                offspring_population_size=population_size // 2,
                mutation=UniformMutation(probability=0.1),
                crossover=SBXCrossover(probability=0.9),
                selection=RouletteWheelSelection(),
                termination_criterion=StoppingByEvaluations(evaluations),
            ),
            SocioSSGA(
                problem=problem,
                population_size=population_size,
                offspring_population_size=1,
                interaction_probability=0.5,
                selection=RouletteWheelSelection(),
                mutation=UniformMutation(probability=0.1),
                crossover=SBXCrossover(probability=0.9),
                basic_prob=0.2,
                trust_prob=0.7,
                cost_prob=0.1,
                max_switched_genes=int(problem.number_of_variables * 0.75),
                termination_criterion=StoppingByEvaluations(evaluations),
            ),
        ]

        epoch = []
        fitness = []
        average_fitness = []
        
        for algorithm in algorithms:


            #epoch = [0] * 5
            #fitness = [0] * 5
            #average_fitness = [0] * 5

            #for j in range(0, number_of_trials):

            observer = PrintObjectivesObserver(observer_freq)
            algorithm.observable.register(observer)

            algorithm.run()

            epoch.append(observer.epoch)
            fitness.append(observer.fitness)
            #average_fitness.append(observer.average_fitness)

        plt.figure()
        plt.xlabel("Ewaluacje")
        plt.ylabel("Fitness")
        plt.title("Porównanie różnych operatorów mutacji")
        for i, data in enumerate(zip(epoch, fitness, algorithms)):
            plt.plot(data[0], data[1], label=data[2].get_name())
        plt.legend()
        #plt.show()

    save_to_pdf(problems, plots_per_problem, test_data)
