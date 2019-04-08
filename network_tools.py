import random
import rmsd
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import networkx as nx
import os
from scipy.sparse.linalg import eigs
import scipy

"""
Wrapping of networkx, functions and visualization tools for working with
laplacian mass tuning.
"""

def create_mod_matrix(mass_list):
    '''
    Given a list of masses, returns the corrispective diagonal matrix for
    modulating a laplacian matrix
    '''
    return np.diagflat(mass_list)


def create_inverse_mod_matrix(mass_list):
    '''
    Given a list of masses, returns the inverse of a diagonal matrix for
    modulating a laplacian matrix
    '''
    return np.linalg.inv(create_mod_matrix(mass_list))


def create_weighted_laplacian(network, mass_list):
    '''
    Modify all the links' weight and then return the new computed laplacian
    '''
    assert len(mass_list) == len(network.edges)
    for i, edge in enumerate(network.edges):
        network.edges[edge]['weight'] = mass_list[i]
    return nx.laplacian_matrix(network).todense()


def modify_edges_weitghts(network, mass_list):
    assert len(mass_list) == len(network.edges)
    for i, edge in enumerate(network.edges):
        network.edges[edge]['weight'] = mass_list[i]
    return network


def get_spectral_coordinates(A,
                             dim=3):
    # SPECTRAL SHIFT
    max_val, _ = scipy.sparse.linalg.eigsh(A, k=1, which='LM')
    diagonal_array = np.ones((A.shape[0])) * max_val
    shift = scipy.sparse.diags(diagonal_array)
    B = A - shift
    # COMPUTATION
    vals, eigenvectors = scipy.sparse.linalg.eigsh(
        B, k=20, which='LM')
    eigenvectors = eigenvectors.transpose()
    vec1 = eigenvectors[1]
    if dim >= 2:
        vec2 = eigenvectors[2]
        if dim == 3:
            vec3 = eigenvectors[3]
        else:
            vec3 = np.zeros(len(eigenvectors[1]))
    else:
        vec2 = np.zeros(len(eigenvectors[1]))
        vec3 = np.zeros(len(eigenvectors[1]))
    vecs = np.column_stack((vec1, vec2, vec3))
    vecs -= vecs.mean(axis=0)
    vecs[:, :dim] /= np.linalg.norm(vecs[:, :dim], axis=0)
    return vecs
