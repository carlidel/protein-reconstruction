import numpy as np
import pandas as pd
import sys
import os
import pickle
import itertools
from tqdm import tqdm

"""
Simple .pdb parser for collecting spatial info of atoms and such. 
"""

AA_LIST = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
           "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
           "TYR", "VAL"]


def parse_pdb_file(protein_name, filepath):
    print(filepath)
    lines = open(filepath, 'r').readlines()
    atom_name = []
    residue_name = []
    residue_number = []
    x_coord = []
    y_coord = []
    z_coord = []
    element = []
    for line in lines:
        if line[0:7].strip() == "ATOM":
            atom_name.append(line[13:17].strip())
            residue_name.append(line[17:21].strip())
            residue_number.append(int(line[23:26].strip()))
            x_coord.append(float(line[31:39].strip()))
            y_coord.append(float(line[39:47].strip()))
            z_coord.append(float(line[47:55].strip()))
            element.append(line[77:79].strip())
    protein = pd.DataFrame({
        'atom_name': atom_name,
        'residue_name': residue_name,
        'residue_number': residue_number,
        'x': x_coord,
        'y': y_coord,
        'z': z_coord,
        'element': element})
    protein['aa_index'] = protein["residue_name"].apply(
        lambda x: AA_LIST.index(x))

    with open("processed_pdb/" + protein_name[:-4] + "_dataset.pkl", "wb") as f:
        pickle.dump(protein, f)
    return protein


def filter_dataset_CA(protein_name, dataset, save=True):
    """
    Filter only the CA atoms from a given dataset.
    Returns the list of datasets. 
    """
    filtered_dataset = dataset[dataset["atom_name"] == "CA"]
    if save:
        with open("processed_pdb/" + protein_name + "_CA_data.pkl", "wb") as f:
            pickle.dump(filtered_dataset, f)
    return filtered_dataset


def make_coordinate_dataset_CA(protein_name, dataset):
    """
    Returns only the coordinates of a dataset.
    """
    filtered_dataset = filter_dataset_CA(protein_name, dataset, False)
    coordinates = filtered_dataset[["x", "y", "z"]].values
    coordinates -= coordinates.mean(axis=0)
    coordinates /= np.linalg.norm(coordinates, axis=0)
    data_coords = pd.DataFrame(coordinates,
                               columns=("x", "y", "z"))
    with open("processed_pdb/" + protein_name + "_CA_coords.pkl", "wb") as f:
        pickle.dump(data_coords, f)
    return data_coords


def process_distance_matrix_CA(protein_name, dataset):
    filtered_dataset = filter_dataset_CA(protein_name, dataset, False)
    N = len(filtered_dataset)
    dist_matrix = np.zeros((N, N))
    for i, j in tqdm(itertools.combinations(range(N), 2)):
        a = np.array([filtered_dataset.iloc[i]["x"],
                      filtered_dataset.iloc[i]["y"],
                      filtered_dataset.iloc[i]["z"]])
        b = np.array([filtered_dataset.iloc[j]["x"],
                      filtered_dataset.iloc[j]["y"],
                      filtered_dataset.iloc[j]["z"]])
        dist_matrix[i][j] = np.linalg.norm(a - b)
        dist_matrix[j][i] = dist_matrix[i][j]
    with open("processed_pdb/" + protein_name + "_CA_dist.pkl", "wb") as f:
        pickle.dump(dist_matrix, f)
    return dist_matrix


def unload_all(directory="processed_pdb"):
    with open(directory + "/names.pkl", 'rb') as f:
        names = pickle.load(f)
    datasets = []
    filtered_datasets = []
    data_coords = []
    dist_matrices = []
    for name in names:
        with open("processed_pdb/" + name + "_dataset.pkl", "rb") as f:
            datasets.append(pickle.load(f))
        with open("processed_pdb/" + name + "_CA_data.pkl", "rb") as f:
            filtered_datasets.append(pickle.load(f))
        with open("processed_pdb/" + name + "_CA_coords.pkl", "rb") as f:
            data_coords.append(pickle.load(f))
        with open("processed_pdb/" + name + "_CA_dist.pkl", "rb") as f:
            dist_matrices.append(pickle.load(f))
    return names, datasets, filtered_datasets, data_coords, dist_matrices


if __name__ == "__main__":
    if not os.path.exists("processed_pdb"):
        os.makedirs("processed_pdb")
    if len(sys.argv) == 1:
        directory = "pdb_files"
    else:
        directory = sys.argv[1]
    items = os.listdir(directory)
    files = []
    names = []
    datasets = []
    for name in items:
        if name.endswith(".pdb"):
            files.append(name)
            names.append(name[: -4])

    for name in files:
        protein = parse_pdb_file(name, directory + "/" + name)
        datasets.append(protein)
    
    with open("processed_pdb" + "/names.pkl", 'wb') as f:
        pickle.dump(names, f)
    
    namefile = open("processed_pdb/names.txt", 'w')

    for i, name in enumerate(names):
        namefile.write(name + "\n")
        _ = filter_dataset_CA(name, datasets[i])
        _ = make_coordinate_dataset_CA(name, datasets[i])
        _ = process_distance_matrix_CA(name, datasets[i])
