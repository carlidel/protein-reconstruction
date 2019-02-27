import os
import pickle
import itertools
from tqdm import tqdm
import protein_reconstruction as pr
import simulated_annealing as sa

if __name__ == "__main__":
    
    os.system("mkdir results")
    
    # Load Data
    names, datasets, filtered_datasets, data_coords, dist_matrices = (
        pr.unload_all())

    # Big Processing (is it a good idea?)
    for threshold in tqdm(pr.THRESHOLDS):
        str_thr = str(threshold).replace('.', '')
        os.system("mkdir results\\" + str_thr)
        for i, name in enumerate(names):
            network, aa_edges = pr.make_network_from_distance_matrix(
                filtered_datasets[i], dist_matrices[i], threshold)

            sb_normed_coords, sb_normed_score = pr.get_spectral_basic_coordinates(
                network, data_coords[i], True)
            sb_not_normed_coords, sb_not_normed_score = pr.get_spectral_basic_coordinates(
                network, data_coords[i], False)

            with open("results/" + str_thr + "/" + name + "_network.pkl",
                      'wb') as f:
                pickle.dump((network, aa_edges), f)
            with open("results/" + str_thr + "/" + name + "_spectral_basic.pkl",
                      'wb') as f:
                pickle.dump((sb_normed_coords, sb_normed_score,
                             sb_not_normed_coords, sb_not_normed_score),
                            f)

        # # Single Processing normalized
        # os.system("mkdir results\\" + str_thr + "\\single_norm")
        # for i, name in tqdm(enumerate(names)):
        #     with open("results/" + str_thr + "/" + name + "_network.pkl",
        #               'rb') as f:
        #         network, aa_edges = pickle.load(f)
        #     masses, story = sa.simulated_annealing(len(aa_edges),
        #                                            pr.fitness_single,
        #                                            (network, data_coords[i]),
        #                                            normalized=True)
        #     sa_coords, sa_score = pr.get_perturbed_coordinates(
        #         network, masses, data_coords[i], normalized=True)
        #     with open("results/" + str_thr + "/single_norm/" + name + "_single_norm.pkl", 'wb') as f:
        #         pickle.dump((masses, sa_coords, sa_score, story), f)

        # # Single Processing NOT normalized
        # os.system("mkdir results\\" + str_thr + "\\single_no_norm")
        # for i, name in tqdm(enumerate(names)):
        #     with open("results/" + str_thr + "/" + name + "_network.pkl",
        #               'rb') as f:
        #         network, aa_edges = pickle.load(f)
        #     masses, story = sa.simulated_annealing(len(aa_edges),
        #                                            pr.fitness_single,
        #                                            (network, data_coords[i]),
        #                                            normalized=False)
        #     sa_coords, sa_score = pr.get_perturbed_coordinates(
        #         network, masses, data_coords[i], normalized=False)
        #     with open("results/" + str_thr + "/single_no_norm/" + name + "_single_no_norm.pkl", 'wb') as f:
        #         pickle.dump((masses, sa_coords, sa_score, story), f)

        # AA Processing normalized
        os.system("mkdir results\\" + str_thr + "\\aa_norm")
        for i, name in tqdm(enumerate(names)):
            with open("results/" + str_thr + "/" + name + "_network.pkl",
                      'rb') as f:
                network, aa_edges = pickle.load(f)
            masses, story = sa.simulated_annealing(
                len(list(
                    itertools.combinations_with_replacement(range(20), 2))),
                pr.fitness_all,
                ([network], [data_coords[i]], [aa_edges]),
                n_iterations=100000,
                normalized=True)
            sa_coords, sa_score = pr.get_global_perturbed_coordinates(
                [network], [aa_edges], masses,
                [data_coords[i]], normalized=True)
            with open("results/" + str_thr + "/aa_norm/" + name + "_aa_norm.pkl", 'wb') as f:
                pickle.dump((masses, sa_coords[0], sa_score[0], story), f)
        
        # AA Processing not normalized
        os.system("mkdir results\\" + str_thr + "\\aa_no_norm")
        for i, name in tqdm(enumerate(names)):
            with open("results/" + str_thr + "/" + name + "_network.pkl",
                      'rb') as f:
                network, aa_edges = pickle.load(f)
            masses, story = sa.simulated_annealing(
                len(list(
                    itertools.combinations_with_replacement(range(20), 2))),
                pr.fitness_all,
                ([network], [data_coords[i]], [aa_edges]),
                n_iterations=100000,
                normalized=False)
            sa_coords, sa_score = pr.get_global_perturbed_coordinates(
                [network], [aa_edges], masses,
                [data_coords[i]], normalized=False)
            with open("results/" + str_thr + "/aa_no_norm/" + name + "_aa_no_norm.pkl", 'wb') as f:
                pickle.dump((masses, sa_coords[0], sa_score[0], story), f)