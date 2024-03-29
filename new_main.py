import os
import sys
import pickle
import itertools
from tqdm import tqdm
import protein_reconstruction as pr
import simulated_annealing as sa
import pandas as pd
import datetime

TQDM_FLAG = True

if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("No arguments recived.")
        sys.exit()

    #print("Using Netowrkx.")

    # Load Data
    names, datasets, filtered_datasets, data_coords, dist_matrices = (
        pr.unload_all())

    for threshold in tqdm(pr.THRESHOLDS, desc='Making Paths.', disable=TQDM_FLAG):
        str_thr = str(threshold).replace('.', '')
        if not os.path.exists("results/" + str_thr + "/single_norm"):
            os.makedirs("results/" + str_thr + "/single_norm")
        if not os.path.exists("results/" + str_thr + "/single_norm_v2"):
            os.makedirs("results/" + str_thr + "/single_norm_v2")

    # Big Processing (is it a good idea?)
    for i, name in tqdm(list(enumerate(names)), desc='Names to Check.', disable=TQDM_FLAG):
        if any(name in s for s in sys.argv[1:]):
            begin_time = datetime.datetime.now()
            print("Begin with {} at time: {}".format(name, begin_time))
            for threshold in tqdm(pr.THRESHOLDS, desc='Making Networks.', disable=TQDM_FLAG):
                str_thr = str(threshold).replace('.', '')
                network, aa_edges = pr.make_network_from_distance_matrix(
                    filtered_datasets[i], dist_matrices[i], threshold)

                sb_normed_coords, sb_normed_score, sb_normed_score_v2 = pr.get_spectral_basic_coordinates(
                    network, data_coords[i].values, True)
                sb_not_normed_coords, sb_not_normed_score, sb_not_normed_score_v2 = pr.get_spectral_basic_coordinates(
                    network, data_coords[i].values, False)

                sb_normed_coords = pd.DataFrame(
                    sb_normed_coords, columns=['x', 'y', 'z'])
                sb_not_normed_coords = pd.DataFrame(
                    sb_not_normed_coords, columns=['x', 'y', 'z'])

                with open("results/" + str_thr + "/" + names[i] + "_network.pkl", 'wb') as f:
                    pickle.dump((network, aa_edges), f)
                with open("results/" + str_thr + "/" + names[i] + "_spectral_basic.pkl", 'wb') as f:
                    pickle.dump((sb_normed_coords, sb_normed_score, sb_normed_score_v2,
                                 sb_not_normed_coords, sb_not_normed_score, sb_not_normed_score_v2), f)

            # Single Processing normalized
            for threshold in tqdm(pr.THRESHOLDS, desc='S.A. RMSD-Based.', disable=TQDM_FLAG):
                str_thr = str(threshold).replace('.', '')
                if not os.path.exists("results/" + str_thr + "/single_norm/" + names[i] + "_single_norm.pkl"):
                    with open("results/" + str_thr + "/" + names[i] + "_network.pkl", 'rb') as f:
                        network, aa_edges = pickle.load(f)
                    masses, story = sa.simulated_annealing(len(aa_edges),
                                                           pr.fitness_single,
                                                           (network,
                                                            data_coords[i].values),
                                                           n_iterations=10000,
                                                           normalized=True)
                    sa_coords, sa_score = pr.get_perturbed_coordinates(
                        network, masses, data_coords[i].values, normalized=True)

                    sa_coords = pd.DataFrame(
                        sa_coords, columns=['x', 'y', 'z'])

                    with open("results/" + str_thr + "/single_norm/" + names[i] + "_single_norm.pkl", 'wb') as f:
                        pickle.dump((masses, sa_coords, sa_score, story), f)
                else:
                    print("RMSD work for {}, thr {} already done. Skipping.".format(
                        names[i], threshold))

            # Single Processing normalized correlation fitness
            for threshold in tqdm(pr.THRESHOLDS, desc='S.A. Corr-Based', disable=TQDM_FLAG):
                str_thr = str(threshold).replace('.', '')
                if not os.path.exists("results/" + str_thr + "/single_norm_v2/" + names[i] + "_single_norm_v2.pkl"):
                    with open("results/" + str_thr + "/" + names[i] + "_network.pkl", 'rb') as f:
                        network, aa_edges = pickle.load(f)
                    masses, story = sa.simulated_annealing(len(aa_edges),
                                                           pr.fitness_single_correlation,
                                                           (network,
                                                            data_coords[i].values),
                                                           n_iterations=10000,
                                                           normalized=True)
                    sa_coords, sa_score = pr.get_perturbed_coordinates(
                        network, masses, data_coords[i].values, normalized=True)

                    sa_coords = pd.DataFrame(
                        sa_coords, columns=['x', 'y', 'z'])

                    with open("results/" + str_thr + "/single_norm_v2/" + names[i] + "_single_norm_v2.pkl", 'wb') as f:
                        pickle.dump((masses, sa_coords, sa_score, story), f)
                else:
                    print("Correlation work for {}, thr {} already done. Skipping.".format(
                        names[i], threshold))
            # ENDING
            end_time = datetime.datetime.now()
            print("Finished with {} at time: {}".format(name, end_time))
            difference_time = end_time - begin_time
            print("Elapsed time for {}: {}".format(name, difference_time))
