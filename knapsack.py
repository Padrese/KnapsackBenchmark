import pyomo.environ as pyo
import linopy as lpy
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import time

def build_pyomo_knapsack_instance(nb_objects, utilities, weights, bound):
    instance = pyo.ConcreteModel()
    instance.x = pyo.Var([i+1 for i in range(nb_objects)], domain=pyo.Binary, initialize=0.0)

    def weight_constraint(instance):
        return sum(weights[i]*instance.x[i+1] for i in range(nb_objects)) <= bound

    instance.weight_constr = pyo.Constraint(expr = weight_constraint)

    def objective(instance):
        return sum(utilities[i]*instance.x[i+1] for i in range(nb_objects))

    instance.objective = pyo.Objective(expr = objective, sense=pyo.maximize)

    return instance

def build_linopy_knapsack_instance(nb_objects, utilities, weights, bound):
    instance = lpy.Model()

    for i in range(1,nb_objects+1):
        instance.add_variables(binary=True, name="x_"+str(i))

    instance.add_constraints(sum(weights[i-1]*instance.variables["x_"+str(i)] for i in range(1,nb_objects+1)) <= bound, name="capacity_constr")
    instance.add_objective(sum(utilities[i-1]*instance.variables["x_"+str(i)] for i in range(1,nb_objects+1)), sense="max")

    return instance

def instance_generator(nb_objects):
    '''
    Lower bounds on the selected random values are chosen so that some trivial instances are eliminated
    and accordingly to bound conflicts on the lower and upper bounds of the numpy randint function.
    '''
    bound = np.random.randint(2,nb_objects)
    weights = [np.random.randint(1,bound) for _ in range(nb_objects)]
    utilities = [np.random.randint(1,bound) for _ in range(nb_objects)]
    return bound, weights, utilities

def solve_instance_with_pyomo(nb_objects, weights, utilities, bound):
    begin = time.time()
    pyo_instance = build_pyomo_knapsack_instance(nb_objects=nb_objects, weights=weights, utilities=utilities, bound=bound)
    solver = pyo.SolverFactory('appsi_highs')
    solver.solve(pyo_instance)
    end = time.time()
    #pyo_instance.display()
    return end-begin

def solve_instance_with_linopy(nb_objects, weights, utilities, bound):
    begin = time.time()
    linopy_instance = build_linopy_knapsack_instance(nb_objects=nb_objects, weights=weights, utilities=utilities, bound=bound)
    linopy_instance.solve(solver_name="highs")
    end = time.time()
    #print("Solution:", linopy_instance.solution)
    return end-begin

def benchmark():
    nb_objects_array = [10,100,1000,10000]
    timer_matrix_pyomo = []
    timer_matrix_linopy = []
    for nb_objects in nb_objects_array:
        timer_array_pyomo = []
        timer_array_linopy = []
        for _ in range(10):
            bound, weights, utilities = instance_generator(nb_objects=nb_objects)
            time_pyomo = solve_instance_with_pyomo(nb_objects=nb_objects, weights=weights, utilities=utilities, bound=bound)
            time_linopy = solve_instance_with_linopy(nb_objects=nb_objects, weights=weights, utilities=utilities, bound=bound)
            timer_array_pyomo.append(time_pyomo)
            timer_array_linopy.append(time_linopy)
        timer_matrix_pyomo.append(timer_matrix_pyomo)
        timer_matrix_linopy.append(timer_array_linopy)

    timer_mean_values_pyomo = [np.mean(timer_matrix_pyomo[i]) for i in range(len(timer_matrix_pyomo))]
    timer_mean_values_linopy = [np.mean(timer_matrix_linopy[i]) for i in range(len(timer_matrix_linopy))]

    print(timer_mean_values_pyomo)
    print(timer_mean_values_linopy)

    fig, ax = plt.subplots()


def main():
    benchmark()

if __name__ == "__main__":
    main()