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

THRESHOLDS = [6.0, 10.0, 12.0, 16.0, 20.0]


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
            laplacian = nx.normalized_laplacian_matrix(network).toarray().astype(np.float)
        else:
            laplacian = nx.laplacian_matrix(network).todense().astype(np.float)
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


def get_perturbed_coordinates(network, masses, target_coordinates,
    normalized=True):
    return fitness_single(masses, (network, target_coordinates), normalized)


def get_global_perturbed_coordinates(networks, edge_aa_list, aa_masses,
    target_coordinates, normalized=True):
    scores = []
    fitness_parameters = (networks, target_coordinates, edge_aa_list)
    coords, _ = fitness_all(aa_masses, fitness_parameters, normalized)
    for i, coord in enumerate(coords):
        scores.append(rmsd.kabsch_rmsd(coord.values,
                                       target_coordinates[i].values))
    return coords, scores


def get_spectral_basic_coordinates(network, target_coordinates,
    normalized=True):
    basic_coords = nt.get_spectral_coordinates(
        (nx.normalized_laplacian_matrix(network).toarray().astype(np.float) if normalized else
            nx.laplacian_matrix(network).toarray().astype(np.float)),
        dim=3)
    return (basic_coords,
            rmsd.kabsch_rmsd(basic_coords.values, target_coordinates.values))

# Fitness Functions

def fitness_single(masses, fitness_parameters, normalized=True):
    protein_network = fitness_parameters[0]
    target_coordinates = fitness_parameters[1]
    network = nt.modify_edges_weitghts(protein_network, masses)
    if normalized:
        guess_coordinates = nt.get_spectral_coordinates(
            nx.normalized_laplacian_matrix(network).toarray().astype(np.float),
            dim=3)
    else:
        guess_coordinates = nt.get_spectral_coordinates(
            nx.laplacian_matrix(network).toarray().astype(np.float),
            dim=3)
    return (guess_coordinates,
            rmsd.kabsch_rmsd(guess_coordinates.values,
                             target_coordinates.values))


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
        fitness += rmsd.kabsch_rmsd(guess_coordinates.values,
                                    target_coordinates_list[i].values)
        guesses.append(guess_coordinates)
    return guesses, fitness