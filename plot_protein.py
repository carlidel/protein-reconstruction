import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import protein_reconstruction as pr
import rmsd
import os

def plot_distance_statistics(distance_matrix_list,
                             n_bins,
                             title="",
                             savepath="",
                             showfig=True):
    distance_array_list = []
    for matrix in distance_matrix_list:
        distance_array_list.append(matrix.flatten())
    distance_matrix_list = np.concatenate(distance_array_list).ravel()
    plt.hist(distance_matrix_list, bins=n_bins, density=True)
    plt.xlabel("Distanza $[\\AA]$")
    plt.ylabel("Distribuzione di probabilità")
    if title != "":
        plt.title(title)
    if showfig:
        plt.show()
    if savepath != "":
        plt.savefig(savepath, dpi=300)
        plt.clf()


def plot_protein_network(coords_original,
                         distance_matrix,
                         coords_spectral_basic=pd.DataFrame(),
                         spectral_basic_flag=False,
                         coords_modified=pd.DataFrame(),
                         modified_flag=False,
                         drawing_threshold=8,
                         title="",
                         savepath="",
                         showfig=True,
                         view_thet=30,
                         view_phi=30):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Make network to be plotted (low threshold)
    plt_network = pr.make_basic_network_from_distance_matrix(
        distance_matrix, drawing_threshold)
    # Plot original coords
    ax.scatter(coords_original["x"],
               coords_original["y"],
               coords_original["z"],
               label="Originale", c="C0")
    for edge in list(plt_network.edges):
        ax.plot((coords_original.iloc[edge[0]]["x"],
                 coords_original.iloc[edge[1]]["x"]),
                (coords_original.iloc[edge[0]]["y"],
                 coords_original.iloc[edge[1]]["y"]),
                (coords_original.iloc[edge[0]]["z"],
                 coords_original.iloc[edge[1]]["z"]),
                c="grey", alpha=0.7)
    # If any, plot given coords
    if modified_flag:
        # APPLY THE RMSD (you never know...)
        coords_modified = pd.DataFrame(
            rmsd.kabsch_rotate(coords_modified.values, coords_original.values),
            columns=["x", "y", "z"])
        score_modified = rmsd.kabsch_rmsd(
            coords_modified.values, coords_original.values)
        ax.scatter(coords_modified["x"],
                   coords_modified["y"],
                   coords_modified["z"],
                   label="SD Perturbato,\nRMSD = {:.6f}".format(score_modified),
                   c="C1")
        for edge in list(plt_network.edges):
            ax.plot((coords_modified.iloc[edge[0]]["x"],
                     coords_modified.iloc[edge[1]]["x"]),
                    (coords_modified.iloc[edge[0]]["y"],
                     coords_modified.iloc[edge[1]]["y"]),
                    (coords_modified.iloc[edge[0]]["z"],
                     coords_modified.iloc[edge[1]]["z"]),
                    c="red", alpha=0.4)
    # Do you also want the spectral basic?
    if spectral_basic_flag:
        coords_spectral_basic = pd.DataFrame(
            rmsd.kabsch_rotate(coords_spectral_basic.values, coords_original.values),
            columns=["x", "y", "z"])
        score_spectral_basic = rmsd.kabsch_rmsd(
            coords_spectral_basic.values, coords_original.values)
        ax.scatter(coords_spectral_basic["x"],
                   coords_spectral_basic["y"],
                   coords_spectral_basic["z"],
                   label="SD Originale,\nRMSD = {:.6f}".format(score_spectral_basic),
                   c="C2")
        for edge in list(plt_network.edges):
            ax.plot((coords_spectral_basic.iloc[edge[0]]["x"],
                     coords_spectral_basic.iloc[edge[1]]["x"]),
                    (coords_spectral_basic.iloc[edge[0]]["y"],
                     coords_spectral_basic.iloc[edge[1]]["y"]),
                    (coords_spectral_basic.iloc[edge[0]]["z"],
                     coords_spectral_basic.iloc[edge[1]]["z"]),
                    c="green", alpha=0.4)
    ax.legend(loc='right', fontsize='small')
    ax.set_xlabel("X $[\\AA]$")
    ax.set_ylabel("Y $[\\AA]$")
    ax.set_zlabel("Z $[\\AA]$")
    if title != "":
        ax.set_title(title)
    if showfig:
        ax.view_init(view_thet, view_phi)
        plt.show()
    if savepath != "":
        ax.view_init(view_thet, view_phi)
        plt.tight_layout()
        plt.savefig(savepath, dpi=300)
        plt.clf()


def rotational_protein_movie(coords_original,
                             distance_matrix,
                             coords_spectral_basic=pd.DataFrame(),
                             spectral_basic_flag=False,
                             coords_modified=pd.DataFrame(),
                             modified_flag=False,
                             drawing_threshold=8,
                             title="",
                             filename="",
                             view_phi=30,
                             n_frames=360):
    os.system("mkdir foo")
    os.system("del \"foo\\foo*.jpg\"")
    for i, angle in enumerate(np.linspace(0, 360, n_frames, endpoint=False)):
        print(str(i) + "/" + str(n_frames))
        plot_protein_network(coords_original,
                             distance_matrix,
                             coords_spectral_basic,
                             spectral_basic_flag,
                             coords_modified,
                             modified_flag,
                             drawing_threshold,
                             title,
                             ("foo\\foo" + str(i).zfill(5) + ".jpg"),
                             False,
                             view_phi,
                             angle)
    os.system("ffmpeg -y -i \"foo\\foo%05d.jpg\" "
              + "img\\" + filename + ".mp4")
