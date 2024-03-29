import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib
import pickle
import protein_reconstruction as pr
import plot_protein as pp
import rmsd
import os
import re
import cv2
from tqdm import tqdm


def unload_names():
    with open('processed_pdb/names.pkl', 'rb') as f:
        return pickle.load(f)


def unload_proteins(names):
    proteins = {}
    for name in names:
        proteins[name] = {}
        with open("processed_pdb/" + name + "_dataset.pkl", "rb") as f:
            proteins[name]["data"] = pickle.load(f)
        with open("processed_pdb/" + name + "_CA_data.pkl", "rb") as f:
            proteins[name]["data_CA"] = pickle.load(f)
        with open("processed_pdb/" + name + "_CA_coords.pkl", "rb") as f:
            proteins[name]["coords"] = pickle.load(f)
        with open("processed_pdb/" + name + "_CA_dist.pkl", "rb") as f:
            proteins[name]["dist_matrix"] = pickle.load(f)
    return proteins


def unload_single(names):
    singles = {}
    # Threshold Cathegory
    for thresh in pr.THRESHOLDS:
        singles[thresh] = {}
        str_thr = str(thresh).replace('.', '')
        # Name Cathegory
        for name in names:
            singles[thresh][name] = {}
            # Network and aa_edges classification
            with open("results/" + str_thr + "/" + name + "_network.pkl", 'rb') as f:
                network, aa_edges = pickle.load(f)
                singles[thresh][name]["network"] = network
                singles[thresh][name]["aa_edges"] = aa_edges
            # Spectral Basic
            with open("results/" + str_thr + "/" + name + "_spectral_basic.pkl", 'rb') as f:
                sb_normed_coords, sb_normed_score, sb_not_normed_coords, sb_not_normed_score = pickle.load(
                    f)
                singles[thresh][name]["rec_basic_norm_coords"] = sb_normed_coords
                singles[thresh][name]["rec_basic_no_norm_coords"] = sb_not_normed_coords
                singles[thresh][name]["rec_basic_norm_score"] = sb_normed_score
                singles[thresh][name]["rec_basic_no_norm_score"] = sb_not_normed_score
            # Normed SA
            with open("results/" + str_thr + "/single_norm/" + name + "_single_norm.pkl", 'rb') as f:
                masses, sa_coords, sa_score, story = pickle.load(f)
                singles[thresh][name]["rec_norm_masses"] = masses
                singles[thresh][name]["rec_norm_coords"] = sa_coords
                singles[thresh][name]["rec_norm_score"] = sa_score
                singles[thresh][name]["rec_norm_story"] = story
            # Not Normed SA
            with open("results/" + str_thr + "/single_no_norm/" + name + "_single_no_norm.pkl", 'rb') as f:
                masses, sa_coords, sa_score, story = pickle.load(f)
                singles[thresh][name]["rec_no_norm_masses"] = masses
                singles[thresh][name]["rec_no_norm_coords"] = sa_coords
                singles[thresh][name]["rec_no_norm_score"] = sa_score
                singles[thresh][name]["rec_no_norm_story"] = story

    return singles


def unload_aa(names):
    aa = {}
    # Threshold Cathegory
    for thresh in pr.THRESHOLDS:
        aa[thresh] = {}
        str_thr = str(thresh).replace('.', '')
        # Name Cathegory
        for name in names:
            aa[thresh][name] = {}
            # Network and aa_edges classification
            with open("results/" + str_thr + "/" + name + "_network.pkl", 'rb') as f:
                network, aa_edges = pickle.load(f)
                aa[thresh][name]["network"] = network
                aa[thresh][name]["aa_edges"] = aa_edges
            # Spectral Basic
            with open("results/" + str_thr + "/" + name + "_spectral_basic.pkl", 'rb') as f:
                sb_normed_coords, sb_normed_score, sb_not_normed_coords, sb_not_normed_score = pickle.load(
                    f)
                aa[thresh][name]["rec_basic_norm_coords"] = sb_normed_coords
                aa[thresh][name]["rec_basic_no_norm_coords"] = sb_not_normed_coords
                aa[thresh][name]["rec_basic_norm_score"] = sb_normed_score
                aa[thresh][name]["rec_basic_no_norm_score"] = sb_not_normed_score
            # Normed SA
            with open("results/" + str_thr + "/aa_norm/" + name + "_aa_norm.pkl", 'rb') as f:
                masses, sa_coords, sa_score, story = pickle.load(f)
                aa[thresh][name]["rec_norm_masses"] = masses
                aa[thresh][name]["rec_norm_coords"] = sa_coords
                aa[thresh][name]["rec_norm_score"] = sa_score
                aa[thresh][name]["rec_norm_story"] = story
            # Not Normed SA
            with open("results/" + str_thr + "/aa_no_norm/" + name + "_aa_no_norm.pkl", 'rb') as f:
                masses, sa_coords, sa_score, story = pickle.load(f)
                aa[thresh][name]["rec_no_norm_masses"] = masses
                aa[thresh][name]["rec_no_norm_coords"] = sa_coords
                aa[thresh][name]["rec_no_norm_score"] = sa_score
                aa[thresh][name]["rec_no_norm_story"] = story

    return aa


def unload_multiple(names):
    multiple = {}
    for thresh in pr.THRESHOLDS:
        multiple[thresh] = {}
        str_thr = str(thresh).replace('.', '')

        for name in names:
            multiple[thresh][name] = {}

        for i, name in enumerate(names):
            # Spectral Basic
            with open("results/" + str_thr + "/" + name + "_spectral_basic.pkl", 'rb') as f:
                sb_normed_coords, sb_normed_score, sb_not_normed_coords, sb_not_normed_score = pickle.load(f)
                multiple[thresh][name]["rec_basic_norm_coords"] = sb_normed_coords
                multiple[thresh][name]["rec_basic_no_norm_coords"] = sb_not_normed_coords
                multiple[thresh][name]["rec_basic_norm_score"] = sb_normed_score
                multiple[thresh][name]["rec_basic_no_norm_score"] = sb_not_normed_score

        with open("results/" + str_thr + "/multi_norm/multi_norm.pkl", 'rb') as f:
            masses, sa_coords, sa_score, story = pickle.load(f)
            multiple[thresh]["norm_masses"] = masses
            multiple[thresh]["norm_score"] = sa_score
            multiple[thresh]["norm_story"] = story
            for i, name in enumerate(names):
                multiple[thresh][name]["rec_norm_coords"] = sa_coords[i]
                multiple[thresh][name]["rec_norm_score"] = (
                    rmsd.kabsch_rmsd(
                        sa_coords[i].values,
                        multiple[thresh][name]["rec_basic_norm_coords"]
                    ))
        
        with open("results/" + str_thr + "/multi_no_norm/multi_no_norm.pkl", 'rb') as f:
            masses, sa_coords, sa_score, story = pickle.load(f)
            multiple[thresh]["no_norm_masses"] = masses
            multiple[thresh]["no_norm_score"] = sa_score
            multiple[thresh]["no_norm_story"] = story
            for i, name in enumerate(names):
                multiple[thresh][name]["rec_no_norm_coords"] = sa_coords[i]
                multiple[thresh][name]["rec_no_norm_score"] = (
                    rmsd.kabsch_rmsd(
                        sa_coords[i].values,
                        multiple[thresh][name]["rec_basic_no_norm_coords"]
                    ))
        
        return multiple



def make_comparing_AA_plot(ax,
                           dist_list_ideal,
                           dist_list_reconstructed,
                           title,
                           maximum,
                           minimum):
    ax.hist(dist_list_reconstructed, range=(minimum, maximum),
            density=False, bins=12,
            lw=2, fc=(0, 0, 1, 0.5), edgecolor='blue',
            label="Reconstructed")
    ax.hist(dist_list_ideal, range=(minimum, maximum),
             density=False, bins=12,
             lw=2, fc=(1, 0, 0, 0.5), edgecolor='red',
             color='red', label="Target")
    ax.set_ylabel("# Occurences")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Distance [Normalized Units]")
    ax.legend()       
    ax.set_title(title)

######################################################################

def stack_horizontally(header, dirs):
    images = [cv2.imread(header + d) for d in dirs]
    image = images[0]
    for i in range(1, len(images)):
        image = np.concatenate((image, images[i]), axis=1)
    return image


def stack_vertically(header, dirs):
    images = [cv2.imread(header + d) for d in dirs]
    image = images[0]
    for i in range(1, len(images)):
        image = np.concatenate((image, images[i]), axis=0)
    return image


#%%
os.system("mkdir plots")
os.system("mkdir combo")
DPI = 300

names = unload_names()
proteins_dict = unload_proteins(names)
single_dict = unload_single(names)
aa_dict = unload_aa(names)

#%%
# Plot Performance Story

for thresh in single_dict:
    str_thresh = str(thresh).replace(".", "")
    for name in single_dict[thresh]:
        fig, ax = plt.subplots()
        ax.plot(single_dict[thresh][name]
                ["rec_norm_story"], label="SA evolution")
        ax.axhline(single_dict[thresh][name]["rec_basic_norm_score"],
                   color='red', label="Default spectral drawing")
        ax.set_xlabel("# of iterations")
        ax.set_ylabel("Fitness [RMSD value]")
        ax.set_title("Performance History of " + name + ", Singular Edges" +
                     "\nThresh = " + str(thresh) + ", Normalized Laplacian")
        ax.legend()
        plt.tight_layout()
        plt.savefig("plots/story_plot_singular_" + name+
                    "_norm_" + str_thresh  + ".png", dpi=DPI, pad_inches=0)

        fig, ax = plt.subplots()
        ax.plot(single_dict[thresh][name]
                ["rec_no_norm_story"], label="SA evolution")
        ax.axhline(single_dict[thresh][name]["rec_basic_no_norm_score"],
                   color='red', label="Default spectral drawing")
        ax.set_xlabel("# of iterations")
        ax.set_ylabel("Fitness [RMSD value]")
        ax.set_title("Performance History of " + name + ", Singular Edges" +
                     "\nThresh = " + str(thresh) + ", Regular Laplacian")
        ax.legend()
        plt.tight_layout()
        plt.savefig("plots/story_plot_singular_" + name +
                    "_no_norm_" + str_thresh  + ".png", dpi=DPI, pad_inches=0)

for thresh in aa_dict:
    str_thresh = str(thresh).replace(".", "")
    for name in aa_dict[thresh]:
        fig, ax = plt.subplots()
        ax.plot(aa_dict[thresh][name]
                ["rec_norm_story"], label="SA evolution")
        ax.axhline(aa_dict[thresh][name]["rec_basic_norm_score"],
                   color='red', label="Default spectral drawing")
        ax.set_xlabel("# of iterations")
        ax.set_ylabel("Fitness [RMSD value]")
        ax.set_title("Performance History of " + name + ", AA based" +
                     "\nThresh = " + str(thresh) + ", Normalized Laplacian")
        ax.legend()
        plt.tight_layout()
        plt.savefig("plots/story_plot_aa_based_" + name +
                    "_norm_" + str_thresh  + ".png", dpi=DPI, pad_inches=0)

        fig, ax = plt.subplots()
        ax.plot(aa_dict[thresh][name]
                ["rec_no_norm_story"], label="SA evolution")
        ax.axhline(aa_dict[thresh][name]["rec_basic_no_norm_score"],
                   color='red', label="Default spectral drawing")
        ax.set_xlabel("# of iterations")
        ax.set_ylabel("Fitness [RMSD value]")
        ax.set_title("Performance History of " + name + ", AA based" +
                     "\nThresh = " + str(thresh) + ", Regular Laplacian")
        ax.legend()
        plt.tight_layout()
        plt.savefig("plots/story_plot_aa_based_" + name +
                    "_no_norm_" + str_thresh  + ".png", dpi=DPI, pad_inches=0)

#%%

images = os.listdir("./plots")
images = list(filter(lambda k: "story_plot" in k, images))

for name in names:
    filter_name = list(filter(lambda k: name in k, images))

    filter_single = list(filter(lambda k: "_singular_" in k, filter_name))

    filter_single_norm = list(
        filter(lambda k: "_no_norm_" not in k, filter_single))
    filter_single_norm.sort(key=lambda k: int(
        re.search("(\d*)\.", k).group(1)))
    image_single_norm = stack_horizontally("plots/", filter_single_norm)

    filter_single_no_norm = list(
        filter(lambda k: "_no_norm_" in k, filter_single))
    filter_single_no_norm.sort(key=lambda k: int(
        re.search("(\d*)\.", k).group(1)))
    image_single_no_norm = stack_horizontally("plots/", filter_single_no_norm)

    filter_aa = list(filter(lambda k: "_aa_based_" in k, filter_name))

    filter_aa_norm = list(
        filter(lambda k: "_no_norm_" not in k, filter_aa))
    filter_aa_norm.sort(key=lambda k: int(
        re.search("(\d*)\.", k).group(1)))
    image_aa_norm = stack_horizontally("plots/", filter_aa_norm)

    filter_aa_no_norm = list(
        filter(lambda k: "_no_norm_" in k, filter_aa))
    filter_aa_no_norm.sort(key=lambda k: int(
        re.search("(\d*)\.", k).group(1)))
    image_aa_no_norm = stack_horizontally("plots/", filter_aa_no_norm)

    image = np.concatenate(
        (image_single_norm, image_single_no_norm,
         image_aa_norm, image_aa_no_norm), axis=0)
    
    cv2.imwrite("combo/story_plot_" + name + ".png", image)

#%%
# Plot Scatter Plots

for thresh in single_dict:
    str_thresh = str(thresh).replace(".", "")
    for name in single_dict[thresh]:
        # Spectral Basic Normalized
        coords = pd.DataFrame(
            rmsd.kabsch_rotate(
                single_dict[thresh][name]["rec_basic_norm_coords"].values,
                proteins_dict[name]["coords"].values),
            columns=["x", "y", "z"]
        )
        fig, (axx, axy, axz) = plt.subplots(3, 1, figsize=(8, 18))
        # X
        axx.scatter(
            proteins_dict[name]["coords"]["x"],
            coords["x"])
        points = [
            min(
                proteins_dict[name]["coords"]["x"].min(),
                coords["x"].min()
            ),
            max(
                proteins_dict[name]["coords"]["x"].max(),
                coords["x"].max()
            )]
        axx.plot(points, points, "--", alpha=0.6)
        axx.set_xlabel("Original CA Coordinates [A.U.]")
        axx.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axx.set_title("X Coordinate")
        # Y
        axy.scatter(
            proteins_dict[name]["coords"]["y"],
            coords["y"])
        points = [
            min(
                proteins_dict[name]["coords"]["y"].min(),
                coords["y"].min()
            ),
            max(
                proteins_dict[name]["coords"]["y"].max(),
                coords["y"].max()
            )]
        axy.plot(points, points, "--", alpha=0.6)
        axy.set_xlabel("Original CA Coordinates [A.U.]")
        axy.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axy.set_title("Y Coordinate")
        # Z
        axz.scatter(
            proteins_dict[name]["coords"]["z"],
            coords["z"])
        points = [
            min(
                proteins_dict[name]["coords"]["z"].min(),
                coords["z"].min()
            ),
            max(
                proteins_dict[name]["coords"]["z"].max(),
                coords["z"].max()
            )]
        axz.plot(points, points, "--", alpha=0.6)
        axz.set_xlabel("Original CA Coordinates [A.U.]")
        axz.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axz.set_title("Z Coordinate")
        fig.suptitle(name + ", Basic\nThresh = " +
                     str(thresh) + ", Normalized Laplacian")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("plots/coord_plot_basic_" + name +
                    "_norm_" + str_thresh  + ".png", dpi=DPI)

        # Spectral Basic Not Normalized
        coords = pd.DataFrame(
            rmsd.kabsch_rotate(
                single_dict[thresh][name]["rec_basic_no_norm_coords"].values,
                proteins_dict[name]["coords"].values),
            columns=["x", "y", "z"]
        )
        fig, (axx, axy, axz) = plt.subplots(3, 1, figsize=(8, 18))
        # X
        axx.scatter(
            proteins_dict[name]["coords"]["x"],
            coords["x"])
        points = [
            min(
                proteins_dict[name]["coords"]["x"].min(),
                coords["x"].min()
            ),
            max(
                proteins_dict[name]["coords"]["x"].max(),
                coords["x"].max()
            )]
        axx.plot(points, points, "--", alpha=0.6)
        axx.set_xlabel("Original CA Coordinates [A.U.]")
        axx.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axx.set_title("X Coordinate")
        # Y
        axy.scatter(
            proteins_dict[name]["coords"]["y"],
            coords["y"])
        points = [
            min(
                proteins_dict[name]["coords"]["y"].min(),
                coords["y"].min()
            ),
            max(
                proteins_dict[name]["coords"]["y"].max(),
                coords["y"].max()
            )]
        axy.plot(points, points, "--", alpha=0.6)
        axy.set_xlabel("Original CA Coordinates [A.U.]")
        axy.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axy.set_title("Y Coordinate")
        # Z
        axz.scatter(
            proteins_dict[name]["coords"]["z"],
            coords["z"])
        points = [
            min(
                proteins_dict[name]["coords"]["z"].min(),
                coords["z"].min()
            ),
            max(
                proteins_dict[name]["coords"]["z"].max(),
                coords["z"].max()
            )]
        axz.plot(points, points, "--", alpha=0.6)
        axz.set_xlabel("Original CA Coordinates [A.U.]")
        axz.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axz.set_title("Z Coordinate")
        fig.suptitle(name + ", Basic\nThresh = " +
                     str(thresh) + ", Regular Laplacian")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("plots/coord_plot_basic_" + name +
                    "_no_norm_" + str_thresh  + ".png", dpi=DPI)

        # Single Normalized
        coords = pd.DataFrame(
            rmsd.kabsch_rotate(
                single_dict[thresh][name]["rec_norm_coords"].values,
                proteins_dict[name]["coords"].values),
            columns=["x", "y", "z"]
        )
        fig, (axx, axy, axz) = plt.subplots(3, 1, figsize=(8, 18))
        # X
        axx.scatter(
            proteins_dict[name]["coords"]["x"],
            coords["x"])
        points = [
            min(
                proteins_dict[name]["coords"]["x"].min(),
                coords["x"].min()
            ),
            max(
                proteins_dict[name]["coords"]["x"].max(),
                coords["x"].max()
            )]
        axx.plot(points, points, "--", alpha=0.6)
        axx.set_xlabel("Original CA Coordinates [A.U.]")
        axx.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axx.set_title("X Coordinate")
        # Y
        axy.scatter(
            proteins_dict[name]["coords"]["y"],
            coords["y"])
        points = [
            min(
                proteins_dict[name]["coords"]["y"].min(),
                coords["y"].min()
            ),
            max(
                proteins_dict[name]["coords"]["y"].max(),
                coords["y"].max()
            )]
        axy.plot(points, points, "--", alpha=0.6)
        axy.set_xlabel("Original CA Coordinates [A.U.]")
        axy.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axy.set_title("Y Coordinate")
        # Z
        axz.scatter(
            proteins_dict[name]["coords"]["z"],
            coords["z"])
        points = [
            min(
                proteins_dict[name]["coords"]["z"].min(),
                coords["z"].min()
            ),
            max(
                proteins_dict[name]["coords"]["z"].max(),
                coords["z"].max()
            )]
        axz.plot(points, points, "--", alpha=0.6)
        axz.set_xlabel("Original CA Coordinates [A.U.]")
        axz.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axz.set_title("Z Coordinate")
        fig.suptitle(name + ", Singular Edges\nThresh = " +
                     str(thresh) + ", Normalized Laplacian")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("plots/coord_plot_singular_" + name +
                    "_norm_" + str_thresh  + ".png", dpi=DPI)

        # Single Not Normalized
        coords = pd.DataFrame(
            rmsd.kabsch_rotate(
                single_dict[thresh][name]["rec_no_norm_coords"].values,
                proteins_dict[name]["coords"].values),
            columns=["x", "y", "z"]
        )
        fig, (axx, axy, axz) = plt.subplots(3, 1, figsize=(8, 18))
        # X
        axx.scatter(
            proteins_dict[name]["coords"]["x"],
            coords["x"])
        points = [
            min(
                proteins_dict[name]["coords"]["x"].min(),
                coords["x"].min()
            ),
            max(
                proteins_dict[name]["coords"]["x"].max(),
                coords["x"].max()
            )]
        axx.plot(points, points, "--", alpha=0.6)
        axx.set_xlabel("Original CA Coordinates [A.U.]")
        axx.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axx.set_title("X Coordinate")
        # Y
        axy.scatter(
            proteins_dict[name]["coords"]["y"],
            coords["y"])
        points = [
            min(
                proteins_dict[name]["coords"]["y"].min(),
                coords["y"].min()
            ),
            max(
                proteins_dict[name]["coords"]["y"].max(),
                coords["y"].max()
            )]
        axy.plot(points, points, "--", alpha=0.6)
        axy.set_xlabel("Original CA Coordinates [A.U.]")
        axy.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axy.set_title("Y Coordinate")
        # Z
        axz.scatter(
            proteins_dict[name]["coords"]["z"],
            coords["z"])
        points = [
            min(
                proteins_dict[name]["coords"]["z"].min(),
                coords["z"].min()
            ),
            max(
                proteins_dict[name]["coords"]["z"].max(),
                coords["z"].max()
            )]
        axz.plot(points, points, "--", alpha=0.6)
        axz.set_xlabel("Original CA Coordinates [A.U.]")
        axz.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axz.set_title("Z Coordinate")
        fig.suptitle(name + ", Singular Edges\nThresh = " +
                     str(thresh) + ", Regular Laplacian")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("plots/coord_plot_singular_" + name +
                    "_no_norm_" + str_thresh  + ".png", dpi=DPI)

##############################################################################

for thresh in aa_dict:
    str_thresh = str(thresh).replace(".", "")
    for name in aa_dict[thresh]:
        # AA approach Normalized
        coords = pd.DataFrame(
            rmsd.kabsch_rotate(
                aa_dict[thresh][name]["rec_norm_coords"].values,
                proteins_dict[name]["coords"].values),
            columns=["x", "y", "z"]
        )
        fig, (axx, axy, axz) = plt.subplots(3, 1, figsize=(8, 18))
        # X
        axx.scatter(
            proteins_dict[name]["coords"]["x"],
            coords["x"])
        points = [
            min(
                proteins_dict[name]["coords"]["x"].min(),
                coords["x"].min()
            ),
            max(
                proteins_dict[name]["coords"]["x"].max(),
                coords["x"].max()
            )]
        axx.plot(points, points, "--", alpha=0.6)
        axx.set_xlabel("Original CA Coordinates [A.U.]")
        axx.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axx.set_title("X Coordinate")
        # Y
        axy.scatter(
            proteins_dict[name]["coords"]["y"],
            coords["y"])
        points = [
            min(
                proteins_dict[name]["coords"]["y"].min(),
                coords["y"].min()
            ),
            max(
                proteins_dict[name]["coords"]["y"].max(),
                coords["y"].max()
            )]
        axy.plot(points, points, "--", alpha=0.6)
        axy.set_xlabel("Original CA Coordinates [A.U.]")
        axy.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axy.set_title("Y Coordinate")
        # Z
        axz.scatter(
            proteins_dict[name]["coords"]["z"],
            coords["z"])
        points = [
            min(
                proteins_dict[name]["coords"]["z"].min(),
                coords["z"].min()
            ),
            max(
                proteins_dict[name]["coords"]["z"].max(),
                coords["z"].max()
            )]
        axz.plot(points, points, "--", alpha=0.6)
        axz.set_xlabel("Original CA Coordinates [A.U.]")
        axz.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axz.set_title("Z Coordinate")
        fig.suptitle(name + ", AA approach\nThresh = " +
                     str(thresh) + ", Normalized Laplacian")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("plots/coord_plot_aa_based_" + name +
                    "_norm_" + str_thresh  + ".png", dpi=DPI)

        # AA approach Not Normalized
        coords = pd.DataFrame(
            rmsd.kabsch_rotate(
                aa_dict[thresh][name]["rec_no_norm_coords"].values,
                proteins_dict[name]["coords"].values),
            columns=["x", "y", "z"]
        )
        fig, (axx, axy, axz) = plt.subplots(3, 1, figsize=(8, 18))
        # X
        axx.scatter(
            proteins_dict[name]["coords"]["x"],
            coords["x"])
        points = [
            min(
                proteins_dict[name]["coords"]["x"].min(),
                coords["x"].min()
            ),
            max(
                proteins_dict[name]["coords"]["x"].max(),
                coords["x"].max()
            )]
        axx.plot(points, points, "--", alpha=0.6)
        axx.set_xlabel("Original CA Coordinates [A.U.]")
        axx.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axx.set_title("X Coordinate")
        # Y
        axy.scatter(
            proteins_dict[name]["coords"]["y"],
            coords["y"])
        points = [
            min(
                proteins_dict[name]["coords"]["y"].min(),
                coords["y"].min()
            ),
            max(
                proteins_dict[name]["coords"]["y"].max(),
                coords["y"].max()
            )]
        axy.plot(points, points, "--", alpha=0.6)
        axy.set_xlabel("Original CA Coordinates [A.U.]")
        axy.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axy.set_title("Y Coordinate")
        # Z
        axz.scatter(
            proteins_dict[name]["coords"]["z"],
            coords["z"])
        points = [
            min(
                proteins_dict[name]["coords"]["z"].min(),
                coords["z"].min()
            ),
            max(
                proteins_dict[name]["coords"]["z"].max(),
                coords["z"].max()
            )]
        axz.plot(points, points, "--", alpha=0.6)
        axz.set_xlabel("Original CA Coordinates [A.U.]")
        axz.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axz.set_title("Z Coordinate")
        fig.suptitle(name + ", Singular Edges\nThresh = " +
                     str(thresh) + ", Regular Laplacian")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("plots/coord_plot_aa_based_" + name +
                    "_no_norm_" + str_thresh  + ".png", dpi=DPI)

#%%

images = os.listdir("./plots")
images = list(filter(lambda k: "coord_plot" in k, images))

for name in names:
    filter_name = list(filter(lambda k: name in k, images))

    filter_single = list(filter(lambda k: "_singular_" in k, filter_name))

    filter_single_norm = list(
        filter(lambda k: "_no_norm_" not in k, filter_single))
    filter_single_norm.sort(key=lambda k: int(
        re.search("(\d*)\.", k).group(1)))
    image_single_norm = stack_horizontally("plots/", filter_single_norm)

    filter_single_no_norm = list(
        filter(lambda k: "_no_norm_" in k, filter_single))
    filter_single_no_norm.sort(key=lambda k: int(
        re.search("(\d*)\.", k).group(1)))
    image_single_no_norm = stack_horizontally("plots/", filter_single_no_norm)

    filter_aa = list(filter(lambda k: "_aa_based_" in k, filter_name))

    filter_aa_norm = list(
        filter(lambda k: "_no_norm_" not in k, filter_aa))
    filter_aa_norm.sort(key=lambda k: int(
        re.search("(\d*)\.", k).group(1)))
    image_aa_norm = stack_horizontally("plots/", filter_aa_norm)

    filter_aa_no_norm = list(
        filter(lambda k: "_no_norm_" in k, filter_aa))
    filter_aa_no_norm.sort(key=lambda k: int(
        re.search("(\d*)\.", k).group(1)))
    image_aa_no_norm = stack_horizontally("plots/", filter_aa_no_norm)

    image = np.concatenate(
        (image_single_norm, image_single_no_norm,
         image_aa_norm, image_aa_no_norm), axis=0)

    cv2.imwrite("combo/coord_plot_" + name + ".png", image)


#%%
# 3D plot

for thresh in tqdm(single_dict):
    str_thresh = str(thresh).replace(".", "")
    for name in tqdm(single_dict[thresh]):
        pp.plot_protein_network(
            proteins_dict[name]["coords"],
            proteins_dict[name]["dist_matrix"],
            single_dict[thresh][name]["rec_basic_norm_coords"],
            True,
            title=(name + ", Basic\nThresh = " +
                   str(thresh) + ", Normalized Laplacian"),
            savepath=("plots/3D_plot_basic_" + name
                      + "_norm_" + str_thresh + ".png"),
            showfig=False)

        pp.plot_protein_network(
            proteins_dict[name]["coords"],
            proteins_dict[name]["dist_matrix"],
            single_dict[thresh][name]["rec_basic_no_norm_coords"],
            True,
            title=(name + ", Basic\nThresh = " +
                   str(thresh) + ", Regular Laplacian"),
            savepath=("plots/3D_plot_basic_" + name
                      + "_no_norm_" + str_thresh + ".png"),
            showfig=False)
        
        pp.plot_protein_network(
            proteins_dict[name]["coords"],
            proteins_dict[name]["dist_matrix"],
            single_dict[thresh][name]["rec_norm_coords"],
            True,
            title=(name + ", Singular Edges\nThresh = " +
                   str(thresh) + ", Normalized Laplacian"),
            savepath=("plots/3D_plot_singular_" + name
                      + "_norm_" + str_thresh + ".png"),
            showfig=False)

        pp.plot_protein_network(
            proteins_dict[name]["coords"],
            proteins_dict[name]["dist_matrix"],
            single_dict[thresh][name]["rec_no_norm_coords"],
            True,
            title=(name + ", Singular Edges\nThresh = " +
                   str(thresh) + ", Regular Laplacian"),
            savepath=("plots/3D_plot_singular_" + name
                      + "_no_norm_" + str_thresh + ".png"),
            showfig=False)

        pp.plot_protein_network(
            proteins_dict[name]["coords"],
            proteins_dict[name]["dist_matrix"],
            aa_dict[thresh][name]["rec_norm_coords"],
            True,
            title=(name + ", AA Approach\nThresh = " +
                   str(thresh) + ", Normalized Laplacian"),
            savepath=("plots/3D_plot_aa_based_" + name
                      + "_norm_" + str_thresh + ".png"),
            showfig=False)

        pp.plot_protein_network(
            proteins_dict[name]["coords"],
            proteins_dict[name]["dist_matrix"],
            aa_dict[thresh][name]["rec_no_norm_coords"],
            True,
            title=(name + ", AA Approach\nThresh = " +
                   str(thresh) + ", Regular Laplacian"),
            savepath=("plots/3D_plot_aa_based_" + name
                      + "_no_norm_" + str_thresh + ".png"),
            showfig=False)

#%%

images = os.listdir("./plots")
images = list(filter(lambda k: "3D_plot" in k, images))

for name in names:
    filter_name = list(filter(lambda k: name in k, images))

    filter_basic = list(filter(lambda k: "_basic_" in k, filter_name))

    filter_basic_norm = list(
        filter(lambda k: "_no_norm_" not in k, filter_basic))
    filter_basic_norm.sort(key=lambda k: int(
        re.search("(\d*)\.", k).group(1)))
    image_basic_norm = stack_horizontally("plots/", filter_basic_norm)

    filter_basic_no_norm = list(
        filter(lambda k: "_no_norm_" in k, filter_basic))
    filter_basic_no_norm.sort(key=lambda k: int(
        re.search("(\d*)\.", k).group(1)))
    image_basic_no_norm = stack_horizontally("plots/", filter_basic_no_norm)

    filter_single = list(filter(lambda k: "_singular_" in k, filter_name))

    filter_single_norm = list(
        filter(lambda k: "_no_norm_" not in k, filter_single))
    filter_single_norm.sort(key=lambda k: int(
        re.search("(\d*)\.", k).group(1)))
    image_single_norm = stack_horizontally("plots/", filter_single_norm)

    filter_single_no_norm = list(
        filter(lambda k: "_no_norm_" in k, filter_single))
    filter_single_no_norm.sort(key=lambda k: int(
        re.search("(\d*)\.", k).group(1)))
    image_single_no_norm = stack_horizontally("plots/", filter_single_no_norm)

    filter_aa = list(filter(lambda k: "_aa_based_" in k, filter_name))

    filter_aa_norm = list(
        filter(lambda k: "_no_norm_" not in k, filter_aa))
    filter_aa_norm.sort(key=lambda k: int(
        re.search("(\d*)\.", k).group(1)))
    image_aa_norm = stack_horizontally("plots/", filter_aa_norm)

    filter_aa_no_norm = list(
        filter(lambda k: "_no_norm_" in k, filter_aa))
    filter_aa_no_norm.sort(key=lambda k: int(
        re.search("(\d*)\.", k).group(1)))
    image_aa_no_norm = stack_horizontally("plots/", filter_aa_no_norm)

    image = np.concatenate(
        (image_basic_norm, image_single_norm, 
         image_aa_norm, image_basic_no_norm,
         image_single_no_norm, image_aa_no_norm), axis=0)

    cv2.imwrite("combo/3D_plot_" + name + ".png", image)

#%%
# AA stats
figsize = (8, 8)
fontsize = "x-small"
for thresh in aa_dict:
    str_thresh = str(thresh).replace(".", "")
    for name in aa_dict[thresh]:
        matrix = pr.mask_AA_contact_map(
            aa_dict[thresh][name]["rec_norm_masses"],
            aa_dict[thresh][name]["aa_edges"]
        )
        fig, ax = plt.subplots(figsize=figsize)
        masked = np.ma.masked_where(matrix == -1, matrix)
        cmap = matplotlib.cm.get_cmap('viridis')
        cmap.set_bad(color="black")
        im = ax.imshow(masked, cmap=cmap, aspect='equal',
                       vmin=1, vmax=100, origin='upper')
        ax.set_xticks(np.arange(0, len(pr.AA_LIST)))
        ax.set_yticks(np.arange(0, len(pr.AA_LIST)))
        ax.set_xticklabels(pr.AA_LIST)
        ax.set_yticklabels(pr.AA_LIST)
        #fig.colorbar(im)
        for (i, j), label in np.ndenumerate(matrix):
            if label != -1:
                ax.text(i, j, int(label), ha='center',
                        va='center', fontsize=fontsize)
        ax.set_title("AA masses of " + name + ", AA based" +
                     "\nThresh = " + str(thresh) + ", Normalized Laplacian")
        plt.tight_layout()
        plt.savefig("plots/AA_masses_plot_aa_based_" + name +
                    "_norm_" + str_thresh + ".png", dpi=DPI, pad_inches=0)

        matrix = pr.mask_AA_contact_map(
            aa_dict[thresh][name]["rec_no_norm_masses"],
            aa_dict[thresh][name]["aa_edges"]
        )
        fig, ax = plt.subplots(figsize=figsize)
        masked = np.ma.masked_where(matrix == -1, matrix)
        cmap = matplotlib.cm.get_cmap('viridis')
        cmap.set_bad(color="black")
        im = ax.imshow(masked, cmap=cmap, aspect='equal',
                       vmin=1, vmax=100, origin='upper')
        ax.set_xticks(np.arange(0, len(pr.AA_LIST)))
        ax.set_yticks(np.arange(0, len(pr.AA_LIST)))
        ax.set_xticklabels(pr.AA_LIST)
        ax.set_yticklabels(pr.AA_LIST)
        #fig.colorbar(im)
        for (i, j), label in np.ndenumerate(matrix):
            if label != -1:
              ax.text(i, j, int(label), ha='center',
                      va='center', fontsize=fontsize)
        ax.set_title("AA masses of " + name + ", AA based" +
                     "\nThresh = " + str(thresh) + ", Regular Laplacian")
        plt.tight_layout()
        plt.savefig("plots/AA_masses_plot_aa_based_" + name +
                    "_no_norm_" + str_thresh + ".png", dpi=DPI, pad_inches=0)

        data, average, std_dev = pr.make_AA_statistics(
            single_dict[thresh][name]["rec_norm_masses"],
            single_dict[thresh][name]["aa_edges"]
        )
        fig, ax = plt.subplots(figsize=figsize)
        masked = np.ma.masked_where(average == -1, average)
        cmap = matplotlib.cm.get_cmap('viridis')
        cmap.set_bad(color="black")
        im = ax.imshow(masked, cmap=cmap, aspect='equal',
                       vmin=1, vmax=100, origin='upper')
        ax.set_xticks(np.arange(0, len(pr.AA_LIST)))
        ax.set_yticks(np.arange(0, len(pr.AA_LIST)))
        ax.set_xticklabels(pr.AA_LIST)
        ax.set_yticklabels(pr.AA_LIST)
        #plt.colorbar(im)
        for (i, j), label in np.ndenumerate(average):
            if label != -1:
                string = "{0:.1f}".format(
                    label) + "\n± " + "{0:.1f}".format(std_dev[i][j])
                ax.text(i, j, string, ha='center',
                        va='center', fontsize=fontsize)
        ax.set_title("AA masses of " + name + ", Singular Edges" +
                     "\nThresh = " + str(thresh) + ", Normalized Laplacian")
        plt.tight_layout()
        plt.savefig("plots/AA_masses_plot_singular_" + name +
                    "_norm_" + str_thresh + ".png", dpi=DPI, pad_inches=0)

        data, average, std_dev = pr.make_AA_statistics(
            single_dict[thresh][name]["rec_no_norm_masses"],
            single_dict[thresh][name]["aa_edges"]
        )
        fig, ax = plt.subplots(figsize=figsize)
        masked = np.ma.masked_where(average == -1, average)
        cmap = matplotlib.cm.get_cmap('viridis')
        cmap.set_bad(color="black")
        im = ax.imshow(masked, cmap=cmap, aspect='equal',
                       vmin=1, vmax=100, origin='upper')
        ax.set_xticks(np.arange(0, len(pr.AA_LIST)))
        ax.set_yticks(np.arange(0, len(pr.AA_LIST)))
        ax.set_xticklabels(pr.AA_LIST)
        ax.set_yticklabels(pr.AA_LIST)
        #plt.colorbar(im)
        for (i, j), label in np.ndenumerate(average):
            if label != -1:
                string = "{0:.1f}".format(
                    label) + "\n± " + "{0:.1f}".format(std_dev[i][j])
                ax.text(i, j, string, ha='center',
                        va='center', fontsize=fontsize)
        ax.set_title("AA masses of " + name + ", Singular Edges" +
                     "\nThresh = " + str(thresh) + ", Regular Laplacian")
        plt.tight_layout()
        plt.savefig("plots/AA_masses_plot_singular_" + name +
                    "_no_norm_" + str_thresh + ".png", dpi=DPI, pad_inches=0)

#%%

images = os.listdir("./plots")
images = list(filter(lambda k: "AA_masses" in k, images))

for name in names:
    filter_name = list(filter(lambda k: name in k, images))

    filter_single = list(filter(lambda k: "_singular_" in k, filter_name))

    filter_single_norm = list(
        filter(lambda k: "_no_norm_" not in k, filter_single))
    filter_single_norm.sort(key=lambda k: int(
        re.search("(\d*)\.", k).group(1)))
    image_single_norm = stack_horizontally("plots/", filter_single_norm)

    filter_single_no_norm = list(
        filter(lambda k: "_no_norm_" in k, filter_single))
    filter_single_no_norm.sort(key=lambda k: int(
        re.search("(\d*)\.", k).group(1)))
    image_single_no_norm = stack_horizontally("plots/", filter_single_no_norm)

    filter_aa = list(filter(lambda k: "_aa_based_" in k, filter_name))

    filter_aa_norm = list(
        filter(lambda k: "_no_norm_" not in k, filter_aa))
    filter_aa_norm.sort(key=lambda k: int(
        re.search("(\d*)\.", k).group(1)))
    image_aa_norm = stack_horizontally("plots/", filter_aa_norm)

    filter_aa_no_norm = list(
        filter(lambda k: "_no_norm_" in k, filter_aa))
    filter_aa_no_norm.sort(key=lambda k: int(
        re.search("(\d*)\.", k).group(1)))
    image_aa_no_norm = stack_horizontally("plots/", filter_aa_no_norm)

    image = np.concatenate(
        (image_single_norm, image_single_no_norm,
         image_aa_norm, image_aa_no_norm), axis=0)

    cv2.imwrite("combo/AA_masses_" + name + ".png", image)

#%%
# Weight distribution.

for name in names:
    os.system("mkdir combo\\atlas_distributions\\" + name)
#%%

for thresh in reversed(list(single_dict)):
    str_thresh = str(thresh).replace(".", "")
    for name in single_dict[thresh]:
        aa_weights, _, _ = pr.make_AA_statistics(
            single_dict[thresh][name]["rec_norm_masses"],
            single_dict[thresh][name]["aa_edges"]
        )

        aa_distances_reconstructed, max_rec, min_rec = pr.make_original_AA_dist(
            single_dict[thresh][name]["network"],
            pd.DataFrame(
                rmsd.kabsch_rotate(
                    single_dict[thresh][name]["rec_norm_coords"].values,
                    proteins_dict[name]["coords"].values),
                columns=['x', 'y', 'z']),
            single_dict[thresh][name]["aa_edges"]
        )

        aa_distances_ideal, max_ideal, min_ideal = pr.make_original_AA_dist(
            single_dict[thresh][name]["network"],
            proteins_dict[name]["coords"],
            single_dict[thresh][name]["aa_edges"]
        )
        
        maximum = max(max_ideal, max_rec)
        minimum = min(min_ideal, min_rec)
        
        fig, axes = plt.subplots(20, 20, figsize=(10*10, 8*10))
        for i, line in enumerate(axes):
            for j, ax in enumerate(line):
                #print(i, j)
                if len(aa_distances_ideal[i][j]) != 0:
                    make_comparing_AA_plot(
                        ax,
                        aa_distances_ideal[i][j],
                        aa_distances_reconstructed[i][j],
                        pr.AA_LIST[i] + " and " + pr.AA_LIST[j] + " connections.",
                        maximum,
                        minimum
                    )
                else:
                    ax.axis('off')
        fig.suptitle("Protein: " + name + "; Thresh = " + str(thresh) + "Å; Normalized Laplacian", fontsize=50)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        print(thresh, name)
        plt.savefig("combo/atlas_distributions/" + name + "/AA_atlas_" + name + "_norm_" + str_thresh + ".png", dpi=150)
        print("SAVED!")
        plt.close('all')


#%%
# MEGA MOVIE (AT YOUR RISK AND DANGER)
os.system("mkdir movie\\foo")

for thresh in single_dict:
    str_thresh = str(thresh).replace(".", "")
    for name in single_dict[thresh]:
        pp.rotational_protein_movie(
            proteins_dict[name]["coords"],
            proteins_dict[name]["dist_matrix"],
            single_dict[thresh][name]["rec_basic_norm_coords"],
            True,
            title=(name + ", Basic\nThresh = " +
                   str(thresh) + ", Normalized Laplacian"),
            filename=("3D_plot_basic_" + name
                      + "_norm_" + str_thresh),
            )

        pp.rotational_protein_movie(
            proteins_dict[name]["coords"],
            proteins_dict[name]["dist_matrix"],
            single_dict[thresh][name]["rec_basic_no_norm_coords"],
            True,
            title=(name + ", Basic\nThresh = " +
                   str(thresh) + ", Regular Laplacian"),
            filename=("3D_plot_basic_" + name
                      + "_no_norm_" + str_thresh),
            )

        pp.rotational_protein_movie(
            proteins_dict[name]["coords"],
            proteins_dict[name]["dist_matrix"],
            single_dict[thresh][name]["rec_norm_coords"],
            True,
            title=(name + ", Singular Edges\nThresh = " +
                   str(thresh) + ", Normalized Laplacian"),
            filename=("3D_plot_singular_" + name
                      + "_norm_" + str_thresh),
            )

        pp.rotational_protein_movie(
            proteins_dict[name]["coords"],
            proteins_dict[name]["dist_matrix"],
            single_dict[thresh][name]["rec_no_norm_coords"],
            True,
            title=(name + ", Singular Edges\nThresh = " +
                   str(thresh) + ", Regular Laplacian"),
            filename=("3D_plot_singular_" + name
                      + "_no_norm_" + str_thresh),
            )

        pp.rotational_protein_movie(
            proteins_dict[name]["coords"],
            proteins_dict[name]["dist_matrix"],
            aa_dict[thresh][name]["rec_norm_coords"],
            True,
            title=(name + ", AA Approach\nThresh = " +
                   str(thresh) + ", Normalized Laplacian"),
            filename=("3D_plot_aa_based_" + name
                      + "_norm_" + str_thresh),
            )

        pp.rotational_protein_movie(
            proteins_dict[name]["coords"],
            proteins_dict[name]["dist_matrix"],
            aa_dict[thresh][name]["rec_no_norm_coords"],
            True,
            title=(name + ", AA Approach\nThresh = " +
                   str(thresh) + ", Regular Laplacian"),
            filename=("3D_plot_aa_based_" + name
                      + "_no_norm_" + str_thresh),
            )
