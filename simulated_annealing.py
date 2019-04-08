import rmsd
import pandas as pd
import numpy as np
import networkx as nx
import os
from network_tools import *
from tqdm import tqdm

TQDM_FLAG = False

"""
Better implementation of a Simulated Annealing classificator.
With 'better' I mean more generic so that it is capable to work with any
generic set of integers and any generic fitness function without having to
crack open all the code.
"""

# Simulated Annealing parameters

value_resolution = 100
min_mass = 1.
max_mass = 100.
T_0 = 0.01
T_1 = 0.00001
n_iterations = 100000

# Funcitions


def first_individual(lenght, value_resolution):
    return np.random.choice(value_resolution, lenght)


def new_individual(individual, value_resolution):
    new = (individual
           + np.random.choice([-1, 0, 1], len(individual)))
    new[new < 0] = 0
    new[new > value_resolution] = value_resolution
    return new


def simulated_annealing(n_masses,
                        fitness_function,  # Pass function
                        fitness_parameters,  # Pass parameters as list or tupla
                        value_resolution=value_resolution,
                        min_mass=min_mass,
                        max_mass=max_mass,
                        T_0=T_0,
                        T_1=T_1,
                        n_iterations=n_iterations,
                        normalized=True):
    individual = first_individual(n_masses, value_resolution)
    first_time = True

    T_list = np.logspace(np.log10(T_0),
                         np.log10(T_1),
                         n_iterations,
                         endpoint=False)
    #T_list = np.linspace(T_0,
    #                     T_1,
    #                     n_iterations,
    #                     endpoint=False)
    fitness_history = []
    for i in tqdm(range(len(T_list)), disable=TQDM_FLAG):
        # Creation and evaluation
        if first_time:
            masses = ((max_mass - min_mass) * (individual / value_resolution)
                      + min_mass)
            _, fit_score = fitness_function(masses, fitness_parameters,
                                            normalized)
            first_time = False

        candidate = new_individual(individual, value_resolution)
        masses_candidate = ((max_mass - min_mass) 
                            * (candidate / value_resolution)
                            + min_mass)
        _, fit_score_candidate = fitness_function(masses_candidate,
                                                  fitness_parameters,
                                                  normalized)
        # Annealing
        if fit_score_candidate < fit_score:
            individual = candidate
            masses = masses_candidate
            fit_score = fit_score_candidate
        else:
            if (np.random.rand()
                    < np.exp((fit_score - fit_score_candidate) / T_list[i])):
                individual = candidate
                masses = masses_candidate
                fit_score = fit_score_candidate

        fitness_history.append(fit_score)

    return masses, fitness_history
