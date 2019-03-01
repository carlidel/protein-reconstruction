import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import protein_reconstruction as pr
import plot_protein as pp
import rmsd
import os
import re
import cv2

## TODO:::BREAK FREE FROM NETWORKX

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
        fig, (axx, axy, axz) = plt.subplots(3, 1, figsize=(8, 18))
        # X
        axx.scatter(
            proteins_dict[name]["coords"]["x"],
            single_dict[thresh][name]["rec_basic_norm_coords"]["x"])
        points = [
            min(
                proteins_dict[name]["coords"]["x"].min(),
                single_dict[thresh][name]["rec_basic_norm_coords"]["x"].min()
            ),
            max(
                proteins_dict[name]["coords"]["x"].max(),
                single_dict[thresh][name]["rec_basic_norm_coords"]["x"].max()
            )]
        axx.plot(points, points, "--", alpha=0.6)
        axx.set_xlabel("Original CA Coordinates [A.U.]")
        axx.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axx.set_title("X Coordinate")
        # Y
        axy.scatter(
            proteins_dict[name]["coords"]["y"],
            single_dict[thresh][name]["rec_basic_norm_coords"]["y"])
        points = [
            min(
                proteins_dict[name]["coords"]["y"].min(),
                single_dict[thresh][name]["rec_basic_norm_coords"]["y"].min()
            ),
            max(
                proteins_dict[name]["coords"]["y"].max(),
                single_dict[thresh][name]["rec_basic_norm_coords"]["y"].max()
            )]
        axy.plot(points, points, "--", alpha=0.6)
        axy.set_xlabel("Original CA Coordinates [A.U.]")
        axy.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axy.set_title("Y Coordinate")
        # Z
        axz.scatter(
            proteins_dict[name]["coords"]["z"],
            single_dict[thresh][name]["rec_basic_norm_coords"]["z"])
        points = [
            min(
                proteins_dict[name]["coords"]["z"].min(),
                single_dict[thresh][name]["rec_basic_norm_coords"]["z"].min()
            ),
            max(
                proteins_dict[name]["coords"]["z"].max(),
                single_dict[thresh][name]["rec_basic_norm_coords"]["z"].max()
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
        fig, (axx, axy, axz) = plt.subplots(3, 1, figsize=(8, 18))
        # X
        axx.scatter(
            proteins_dict[name]["coords"]["x"],
            single_dict[thresh][name]["rec_basic_no_norm_coords"]["x"])
        points = [
            min(
                proteins_dict[name]["coords"]["x"].min(),
                single_dict[thresh][name]["rec_basic_no_norm_coords"]["x"].min()
            ),
            max(
                proteins_dict[name]["coords"]["x"].max(),
                single_dict[thresh][name]["rec_basic_no_norm_coords"]["x"].max()
            )]
        axx.plot(points, points, "--", alpha=0.6)
        axx.set_xlabel("Original CA Coordinates [A.U.]")
        axx.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axx.set_title("X Coordinate")
        # Y
        axy.scatter(
            proteins_dict[name]["coords"]["y"],
            single_dict[thresh][name]["rec_basic_no_norm_coords"]["y"])
        points = [
            min(
                proteins_dict[name]["coords"]["y"].min(),
                single_dict[thresh][name]["rec_basic_no_norm_coords"]["y"].min()
            ),
            max(
                proteins_dict[name]["coords"]["y"].max(),
                single_dict[thresh][name]["rec_basic_no_norm_coords"]["y"].max()
            )]
        axy.plot(points, points, "--", alpha=0.6)
        axy.set_xlabel("Original CA Coordinates [A.U.]")
        axy.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axy.set_title("Y Coordinate")
        # Z
        axz.scatter(
            proteins_dict[name]["coords"]["z"],
            single_dict[thresh][name]["rec_basic_no_norm_coords"]["z"])
        points = [
            min(
                proteins_dict[name]["coords"]["z"].min(),
                single_dict[thresh][name]["rec_basic_no_norm_coords"]["z"].min()
            ),
            max(
                proteins_dict[name]["coords"]["z"].max(),
                single_dict[thresh][name]["rec_basic_no_norm_coords"]["z"].max()
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
        fig, (axx, axy, axz) = plt.subplots(3, 1, figsize=(8, 18))
        # X
        axx.scatter(
            proteins_dict[name]["coords"]["x"],
            single_dict[thresh][name]["rec_norm_coords"]["x"])
        points = [
            min(
                proteins_dict[name]["coords"]["x"].min(),
                single_dict[thresh][name]["rec_norm_coords"]["x"].min()
            ),
            max(
                proteins_dict[name]["coords"]["x"].max(),
                single_dict[thresh][name]["rec_norm_coords"]["x"].max()
            )]
        axx.plot(points, points, "--", alpha=0.6)
        axx.set_xlabel("Original CA Coordinates [A.U.]")
        axx.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axx.set_title("X Coordinate")
        # Y
        axy.scatter(
            proteins_dict[name]["coords"]["y"],
            single_dict[thresh][name]["rec_norm_coords"]["y"])
        points = [
            min(
                proteins_dict[name]["coords"]["y"].min(),
                single_dict[thresh][name]["rec_norm_coords"]["y"].min()
            ),
            max(
                proteins_dict[name]["coords"]["y"].max(),
                single_dict[thresh][name]["rec_norm_coords"]["y"].max()
            )]
        axy.plot(points, points, "--", alpha=0.6)
        axy.set_xlabel("Original CA Coordinates [A.U.]")
        axy.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axy.set_title("Y Coordinate")
        # Z
        axz.scatter(
            proteins_dict[name]["coords"]["z"],
            single_dict[thresh][name]["rec_norm_coords"]["z"])
        points = [
            min(
                proteins_dict[name]["coords"]["z"].min(),
                single_dict[thresh][name]["rec_norm_coords"]["z"].min()
            ),
            max(
                proteins_dict[name]["coords"]["z"].max(),
                single_dict[thresh][name]["rec_norm_coords"]["z"].max()
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
        fig, (axx, axy, axz) = plt.subplots(3, 1, figsize=(8, 18))
        # X
        axx.scatter(
            proteins_dict[name]["coords"]["x"],
            single_dict[thresh][name]["rec_no_norm_coords"]["x"])
        points = [
            min(
                proteins_dict[name]["coords"]["x"].min(),
                single_dict[thresh][name]["rec_no_norm_coords"]["x"].min()
            ),
            max(
                proteins_dict[name]["coords"]["x"].max(),
                single_dict[thresh][name]["rec_no_norm_coords"]["x"].max()
            )]
        axx.plot(points, points, "--", alpha=0.6)
        axx.set_xlabel("Original CA Coordinates [A.U.]")
        axx.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axx.set_title("X Coordinate")
        # Y
        axy.scatter(
            proteins_dict[name]["coords"]["y"],
            single_dict[thresh][name]["rec_no_norm_coords"]["y"])
        points = [
            min(
                proteins_dict[name]["coords"]["y"].min(),
                single_dict[thresh][name]["rec_no_norm_coords"]["y"].min()
            ),
            max(
                proteins_dict[name]["coords"]["y"].max(),
                single_dict[thresh][name]["rec_no_norm_coords"]["y"].max()
            )]
        axy.plot(points, points, "--", alpha=0.6)
        axy.set_xlabel("Original CA Coordinates [A.U.]")
        axy.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axy.set_title("Y Coordinate")
        # Z
        axz.scatter(
            proteins_dict[name]["coords"]["z"],
            single_dict[thresh][name]["rec_no_norm_coords"]["z"])
        points = [
            min(
                proteins_dict[name]["coords"]["z"].min(),
                single_dict[thresh][name]["rec_no_norm_coords"]["z"].min()
            ),
            max(
                proteins_dict[name]["coords"]["z"].max(),
                single_dict[thresh][name]["rec_no_norm_coords"]["z"].max()
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
        fig, (axx, axy, axz) = plt.subplots(3, 1, figsize=(8, 18))
        # X
        axx.scatter(
            proteins_dict[name]["coords"]["x"],
            aa_dict[thresh][name]["rec_norm_coords"]["x"])
        points = [
            min(
                proteins_dict[name]["coords"]["x"].min(),
                aa_dict[thresh][name]["rec_norm_coords"]["x"].min()
            ),
            max(
                proteins_dict[name]["coords"]["x"].max(),
                aa_dict[thresh][name]["rec_norm_coords"]["x"].max()
            )]
        axx.plot(points, points, "--", alpha=0.6)
        axx.set_xlabel("Original CA Coordinates [A.U.]")
        axx.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axx.set_title("X Coordinate")
        # Y
        axy.scatter(
            proteins_dict[name]["coords"]["y"],
            aa_dict[thresh][name]["rec_norm_coords"]["y"])
        points = [
            min(
                proteins_dict[name]["coords"]["y"].min(),
                aa_dict[thresh][name]["rec_norm_coords"]["y"].min()
            ),
            max(
                proteins_dict[name]["coords"]["y"].max(),
                aa_dict[thresh][name]["rec_norm_coords"]["y"].max()
            )]
        axy.plot(points, points, "--", alpha=0.6)
        axy.set_xlabel("Original CA Coordinates [A.U.]")
        axy.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axy.set_title("Y Coordinate")
        # Z
        axz.scatter(
            proteins_dict[name]["coords"]["z"],
            aa_dict[thresh][name]["rec_norm_coords"]["z"])
        points = [
            min(
                proteins_dict[name]["coords"]["z"].min(),
                aa_dict[thresh][name]["rec_norm_coords"]["z"].min()
            ),
            max(
                proteins_dict[name]["coords"]["z"].max(),
                aa_dict[thresh][name]["rec_norm_coords"]["z"].max()
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
        fig, (axx, axy, axz) = plt.subplots(3, 1, figsize=(8, 18))
        # X
        axx.scatter(
            proteins_dict[name]["coords"]["x"],
            aa_dict[thresh][name]["rec_no_norm_coords"]["x"])
        points = [
            min(
                proteins_dict[name]["coords"]["x"].min(),
                aa_dict[thresh][name]["rec_no_norm_coords"]["x"].min()
            ),
            max(
                proteins_dict[name]["coords"]["x"].max(),
                aa_dict[thresh][name]["rec_no_norm_coords"]["x"].max()
            )]
        axx.plot(points, points, "--", alpha=0.6)
        axx.set_xlabel("Original CA Coordinates [A.U.]")
        axx.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axx.set_title("X Coordinate")
        # Y
        axy.scatter(
            proteins_dict[name]["coords"]["y"],
            aa_dict[thresh][name]["rec_no_norm_coords"]["y"])
        points = [
            min(
                proteins_dict[name]["coords"]["y"].min(),
                aa_dict[thresh][name]["rec_no_norm_coords"]["y"].min()
            ),
            max(
                proteins_dict[name]["coords"]["y"].max(),
                aa_dict[thresh][name]["rec_no_norm_coords"]["y"].max()
            )]
        axy.plot(points, points, "--", alpha=0.6)
        axy.set_xlabel("Original CA Coordinates [A.U.]")
        axy.set_ylabel("Reconstructed CA Coordinates [A.U.]")
        axy.set_title("Y Coordinate")
        # Z
        axz.scatter(
            proteins_dict[name]["coords"]["z"],
            aa_dict[thresh][name]["rec_no_norm_coords"]["z"])
        points = [
            min(
                proteins_dict[name]["coords"]["z"].min(),
                aa_dict[thresh][name]["rec_no_norm_coords"]["z"].min()
            ),
            max(
                proteins_dict[name]["coords"]["z"].max(),
                aa_dict[thresh][name]["rec_no_norm_coords"]["z"].max()
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

for thresh in single_dict:
    str_thresh = str(thresh).replace(".", "")
    for name in single_dict[thresh]:
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

    cv2.imwrite("combo/3D_plot_" + name + ".png", image)

#%%
# AA stats

for thresh in aa_dict:
    str_thresh = str(thresh).replace(".", "")
    for name in aa_dict[thresh]:
        matrix = pr.mask_AA_contact_map(
            aa_dict[thresh][name]["rec_norm_masses"],
            aa_dict[thresh][name]["aa_edges"]
        )
        fig, ax = plt.subplots()
        ax.imshow(matrix, cmap='viridis', aspect='equal',
                  vmin=1, vmax=100, origin='upper')
        ax.set_xticks(pr.AA_LIST)
        ax.set_yticks(pr.AA_LIST)
        plt.colorbar(ax=ax)
        for (i, j), label in np.ndenumerate(matrix):
            ax.text(i, j, str(label), ha='center', va='center')
        ax.set_title("AA masses of " + name + ", AA based" +
                     "\nThresh = " + str(thresh) + ", Normalized Laplacian")
        plt.tight_layout()
        plt.savefig("plots/AA_masses_plot_aa_based_" + name +
                    "_norm_" + str_thresh + ".png", dpi=DPI, pad_inches=0)

        matrix = pr.mask_AA_contact_map(
            aa_dict[thresh][name]["rec_no_norm_masses"],
            aa_dict[thresh][name]["aa_edges"]
        )
        fig, ax = plt.subplots()
        ax.imshow(matrix, cmap='viridis', aspect='equal',
                  vmin=1, vmax=100, origin='upper')
        ax.set_xticks(pr.AA_LIST)
        ax.set_yticks(pr.AA_LIST)
        plt.colorbar(ax=ax)
        for (i, j), label in np.ndenumerate(matrix):
            ax.text(i, j, str(label), ha='center', va='center')
        ax.set_title("AA masses of " + name + ", AA based" +
                     "\nThresh = " + str(thresh) + ", Regular Laplacian")
        plt.tight_layout()
        plt.savefig("plots/AA_masses_plot_aa_based_" + name +
                    "_no_norm_" + str_thresh + ".png", dpi=DPI, pad_inches=0)

        data, average, std_dev = pr.make_AA_statistics(
            single_dict[thresh][name]["rec_norm_masses"],
            single_dict[thresh][name]["aa_edges"]
        )
        fig, ax = plt.subplots()
        ax.imshow(average,  cmap='viridis', aspect='equal',
                  vmin=1, vmax=100, origin='upper')
        ax.set_xticks(pr.AA_LIST)
        ax.set_yticks(pr.AA_LIST)
        plt.colorbar(ax=ax)
        for (i, j), label in np.ndenumerate(average):
            string = str(label) + "\n+/- " + str(std_dev[i][j])
            ax.text(i, j, label, ha='center', va='center')
        ax.set_title("AA masses of " + name + ", Singular Edges" +
                     "\nThresh = " + str(thresh) + ", Normalized Laplacian")
        plt.tight_layout()
        plt.savefig("plots/AA_masses_plot_singular_" + name +
                    "_norm_" + str_thresh + ".png", dpi=DPI, pad_inches=0)

        data, average, std_dev = pr.make_AA_statistics(
            single_dict[thresh][name]["rec_no_norm_masses"],
            single_dict[thresh][name]["aa_edges"]
        )
        fig, ax = plt.subplots()
        ax.imshow(average,  cmap='viridis', aspect='equal',
                  vmin=1, vmax=100, origin='upper')
        ax.set_xticks(pr.AA_LIST)
        ax.set_yticks(pr.AA_LIST)
        plt.colorbar(ax=ax)
        for (i, j), label in np.ndenumerate(average):
            string = str(label) + "\n+/- " + str(std_dev[i][j])
            ax.text(i, j, label, ha='center', va='center')
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