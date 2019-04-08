import pickle
import os
import numpy as np
import pandas as pd
import networkx as nx
import itertools
import rmsd
import matplotlib.pyplot as plt
from tqdm import tqdm

import network_tools as nt
import simulated_annealing as sa
from pdb_processing import unload_all

AA_LIST = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
           "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
           "TYR", "VAL"]

THRESHOLDS = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]


def make_basic_network_from_distance_matrix(distance_matrix, threshold):
    return nx.from_numpy_array(distance_matrix <= threshold)


def make_network_from_distance_matrix(dataset, distance_matrix, threshold):
    network = nx.from_numpy_array(distance_matrix <= threshold)
    edges = list(network.edges())
    aa_edges = []
    for j, edge in enumerate(edges):
            # AA identification
            index_a = dataset.iloc[edge[0]]["aa_index"]
            index_b = dataset.iloc[edge[1]]["aa_index"]
            aa_edges.append((index_a, index_b))
    return network, aa_edges


def refresh_network_weights(network_list,
                            aa_contact_map,
                            edge_aa_list,
                            normalized=True):
    laplacian_list = []
    for i, network in enumerate(network_list):
        edge_list = list(network_list[i].edges())
        if normalized:
            laplacian = nx.normalized_laplacian_matrix(network).astype(np.float)
        else:
            laplacian = nx.laplacian_matrix(network).astype(np.float)
        for j, edge in enumerate(edge_aa_list[i]):
            # Change weigth
            laplacian[edge_list[j][0], edge_list[j][1]] = (
                aa_contact_map[edge[0], edge[1]])
            laplacian[edge_list[j][1], edge_list[j][0]] = (
                aa_contact_map[edge[0], edge[1]])
        for i in range(len(laplacian)):
            laplacian[i, i] = 0
            laplacian[i, i] = - np.sum(laplacian[i])
        laplacian_list.append(laplacian)
    return laplacian_list


def create_AA_contact_map(weight_list):
    N = len(AA_LIST)
    combo_list = (list(itertools.combinations_with_replacement(range(N), 2)))
    assert len(weight_list) == len(combo_list)
    aa_contact_map = np.zeros((N, N))
    for i in range(len(weight_list)):
        aa_contact_map[combo_list[i][0]][combo_list[i][1]] = weight_list[i]
        aa_contact_map[combo_list[i][1]][combo_list[i][0]] = weight_list[i]
    return aa_contact_map


def mask_AA_contact_map(weight_list, edge_aa_list):
    N = len(AA_LIST)
    combo_list = (list(itertools.combinations_with_replacement(range(N), 2)))
    assert len(weight_list) == len(combo_list)
    mask = np.zeros((N, N), dtype=bool)
    for edge in edge_aa_list:
        mask[edge[0], edge[1]] = True
        mask[edge[1], edge[0]] = True
    masked_AA = create_AA_contact_map(weight_list) * mask
    masked_AA[masked_AA == 0] = -1
    return masked_AA


def make_AA_statistics(weight_list, edge_aa_list):
    N = len(AA_LIST)
    data = [[[] for i in range(N)] for j in range(N)]
    for i, weigth in enumerate(weight_list):
        data[edge_aa_list[i][0]][edge_aa_list[i][1]].append(weigth)
        data[edge_aa_list[i][1]][edge_aa_list[i][0]].append(weigth)
    for i in range(N):
        for j in range(N):
            data[i][j] = np.array(data[i][j])
    average = np.array([[np.average(data[i][j])
                         for j in range(N)] for i in range(N)])
    std_dev = np.array([[np.std(data[i][j]) for j in range(N)]
                        for i in range(N)])
    np.nan_to_num(average, copy=False)
    np.nan_to_num(std_dev, copy=False)
    average[average == 0] = -1
    std_dev[average == 0] = -1
    return data, average, std_dev


def make_original_AA_dist(network, database, edge_aa_list):
    N = len(AA_LIST)
    data = [[[] for i in range(N)] for j in range(N)]
    maximum = 0.0
    minimum = np.Infinity
    for i, edge in enumerate(network.edges()):
        distance = (
            np.sqrt(
                np.square(database.iloc[edge[0]]["x"]
                          - database.iloc[edge[1]]["x"])
                + np.square(database.iloc[edge[0]]["y"] 
                            - database.iloc[edge[1]]["y"])
                + np.square(database.iloc[edge[0]]["z"]
                            - database.iloc[edge[1]]["z"])
            ))
        data[edge_aa_list[i][0]][edge_aa_list[i][1]].append(distance)
        data[edge_aa_list[i][1]][edge_aa_list[i][0]].append(distance)
        maximum = max(maximum, distance)
        minimum = min(minimum, distance)
    return data, maximum, minimum


def get_perturbed_coordinates(network, masses, target_coordinates,
    normalized=True):
    return fitness_single(masses, (network, target_coordinates), normalized)


def get_global_perturbed_coordinates(networks, edge_aa_list, aa_masses,
    target_coordinates, normalized=True):
    scores = []
    fitness_parameters = (networks, target_coordinates, edge_aa_list)
    coords, _ = fitness_all(aa_masses, fitness_parameters, normalized)
    for i, coord in enumerate(coords):
        scores.append(rmsd.kabsch_rmsd(coord,
                                       target_coordinates[i]))
    return coords, scores


def get_spectral_basic_coordinates(network, target_coordinates,
    normalized=True):
    basic_coords = nt.get_spectral_coordinates(
        (nx.normalized_laplacian_matrix(network).astype(np.float) if normalized else
            nx.laplacian_matrix(network).astype(np.float)),
        dim=3)
    return (basic_coords,
            rmsd.kabsch_rmsd(basic_coords, target_coordinates))

# Fitness Functions

def fitness_single(masses, fitness_parameters, normalized=True):
    protein_network = fitness_parameters[0]
    target_coordinates = fitness_parameters[1]
    network = nt.modify_edges_weitghts(protein_network, masses)
    if normalized:
        guess_coordinates = nt.get_spectral_coordinates(
            nx.normalized_laplacian_matrix(network).astype(np.float),
            dim=3)
    else:
        guess_coordinates = nt.get_spectral_coordinates(
            nx.laplacian_matrix(network).astype(np.float),
            dim=3)
    return (guess_coordinates,
            rmsd.kabsch_rmsd(guess_coordinates,
                             target_coordinates))


def fitness_single_correlation(masses, fitness_parameters, normalized=True):
    protein_network = fitness_parameters[0]
    target_coordinates = fitness_parameters[1]
    network = nt.modify_edges_weitghts(protein_network, masses)
    if normalized:
        guess_coordinates = nt.get_spectral_coordinates(
            nx.normalized_laplacian_matrix(network).astype(np.float),
            dim=3)
    else:
        guess_coordinates = nt.get_spectral_coordinates(
            nx.laplacian_matrix(network).astype(np.float),
            dim=3)
    guess_coordinates = rmsd.kabsch_rotate(
        guess_coordinates, target_coordinates)
    corr = (np.corrcoef(guess_coordinates[:, 0],
                        target_coordinates[:, 0])[1, 0]
            + np.corrcoef(guess_coordinates[:, 1],
                          target_coordinates[:, 1])[1, 0]
            + np.corrcoef(guess_coordinates[:, 2],
                          target_coordinates[:, 2])[1, 0])
    return (guess_coordinates, -corr)



def fitness_all(masses, fitness_parameters, normalized=True):
    protein_network_list = fitness_parameters[0]
    target_coordinates_list = fitness_parameters[1]
    edge_aa_list = fitness_parameters[2]
    aa_contact_map = create_AA_contact_map(masses)
    fitness = 0.0
    laplacian_list = refresh_network_weights(protein_network_list,
                                             aa_contact_map,
                                             edge_aa_list,
                                             normalized)
    guesses = []
    for i in range(len(laplacian_list)):
        guess_coordinates = nt.get_spectral_coordinates(
            laplacian_list[i],
            dim=3)
        fitness += rmsd.kabsch_rmsd(guess_coordinates,
                                    target_coordinates_list[i])
        guesses.append(guess_coordinates)
    return guesses, fitness
