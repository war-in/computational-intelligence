class Rastrigin(FloatProblem):

    def __init__(self, number_of_variables: int = 10):
        super(Rastrigin, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        self.lower_bound = [-5.12 for _ in range(number_of_variables)]
        self.upper_bound = [5.12 for _ in range(number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        a = 10.0
        result = a * solution.number_of_variables
        x = solution.variables

        for i in range(solution.number_of_variables):
            result += x[i] * x[i] - a * math.cos(2 * math.pi * x[i])

        solution.objectives[0] = result

        return solution

    def get_name(self) -> str:
        return 'Rastrigin'
    
class FloatProblem(Problem[FloatSolution], ABC):
    """ Class representing float problems. """

    def __init__(self):
        super(FloatProblem, self).__init__()
        self.lower_bound = []
        self.upper_bound = []

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints)
        new_solution.variables = \
            [random.uniform(self.lower_bound[i] * 1.0, self.upper_bound[i] * 1.0) for i in
             range(self.number_of_variables)]

        return new_solution

class FloatSolution(Solution[float]):
    """ Class representing float solutions """

    def __init__(self, lower_bound: List[float], upper_bound: List[float], number_of_objectives: int,
                 number_of_constraints: int = 0):
        super(FloatSolution, self).__init__(len(lower_bound), number_of_objectives, number_of_constraints)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __copy__(self):
        new_solution = FloatSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]
        new_solution.constraints = self.constraints[:]

        new_solution.attributes = self.attributes.copy()

        return new_solution
    


class MySolution(FloatSolution):
    def __init__(self, lower_bound: list, upper_bound: list, number_of_variables: int):
        super(MySolution, self).__init__(lower_bound, upper_bound, number_of_variables)
        self.age = 0
        
    def copy(self) -> 'FloatSolution':
        new_solution = MySolution(self.lower_bound, self.upper_bound, self.number_of_variables)
        new_solution.variables = self.variables.copy()
        new_solution.objectives = self.objectives.copy()
        new_solution.attributes = self.attributes.copy()
        return new_solution
    
class MyFloatProblem (FloatProblem):

    def __init__(self):
        super().__init__()

    def create_solution(self) -> MySolution:
        new_solution = MySolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives)
        new_solution.variables = \
            [random.uniform(self.lower_bound[i] * 1.0, self.upper_bound[i] * 1.0) for i in
             range(self.number_of_variables)]

        return new_solution
    
class MyRastrigin(MyFloatProblem):

    def __init__(self, number_of_variables: int = 10):
        super(MyRastrigin, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

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

        return solution
    
    def get_name(self) -> str:
        return 'Rastrigin'

S = TypeVar('S', bound=Solution)


class MyAlgorithm(GeneticAlgorithm[S, S]):
    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation[S],
                 crossover: Crossover[S, S],
                 selection: Selection[S],
                 generations: int):
        super(MyAlgorithm, self).__init__(
            problem,
            population_size,
            offspring_population_size,
            mutation,
            crossover,
            selection,
            generations
        )

    def create_initial_population(self) -> list:
        population = []
        for i in range(self.population_size):
            new_solution = MySolution(self.problem)
            population.append(new_solution)
        return population

    def evaluate_solution(self, solution: S) -> S:
        return solution.evaluate()

    def get_result(self) -> S:
        return self.solutions[0]

    def stopping_condition_is_met(self) -> bool:
        return self.current_generation >= self.max_generations
        
class MyProblem(Rastrigin):
    def __init__(self, number_of_variables: int = 10):
        super().__init__(number_of_variables)

    def evaluate(self, solution: MySolution) -> MySolution:
        attrs = vars(solution)
        #print(', '.join("%s: %s" % item for item in attrs.items()))
        x = solution.variables
        fx = sum([xi**2 - 10 * math.cos(2 * math.pi * xi) + 10 for xi in x])
        solution.objectives[0] = fx
        #solution.age += 1
        return solution
