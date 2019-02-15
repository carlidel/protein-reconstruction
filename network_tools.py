import random
import rmsd
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import networkx as nx
import os
from scipy.sparse.linalg import eigs

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


def get_spectral_coordinates(A=np.zeros(1),
                             M=np.zeros(1),
                             dim=3):
    '''
    Given a network laplacian, returns eigenvectors associated to the second,
    third and fourth lowest eigenvalues as (x,y,z) axis, based on the spectral
    representation.

    If a modulation matrix is given, a dot operation is performed on the
    laplacian.

    Parameters
    ----------
    laplacian : the laplacian matrix (in array matrix form)
    
    mod_matrix : mass modulation matrix
    
    dim : choose how many dimentions to consider (must be [1,3])
    '''
    if M.any():
        val, eigenvectors = eigs(A, k=dim + 1, M=M, which='SM')
    else:
        val, eigenvectors = eigs(A, k=dim + 1, which='SM')
    vec1 = eigenvectors[:, 1]
    if dim >= 2:
        vec2 = eigenvectors[:, 2]
        if dim == 3:
            vec3 = eigenvectors[:, 3]
        else:
            vec3 = np.zeros(len(eigenvectors[:,1]))
    else:
        vec2 = np.zeros(len(eigenvectors[:, 1]))
        vec3 = np.zeros(len(eigenvectors[:, 1]))
    vecs = np.column_stack((vec1, vec2, vec3))
    vecs -= vecs.mean(axis=0)
    vecs[:, :dim] /= np.linalg.norm(vecs[:, :dim], axis=0)
    coords = pd.DataFrame(vecs, columns=["x", "y", "z"], dtype=float)
    #print(coords)
    return coords
