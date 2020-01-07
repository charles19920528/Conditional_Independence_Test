import numpy as np
from sklearn.cluster import KMeans
import hyperparameters as hp
import multiprocessing as mp
from functools import partial
import pickle

from naive_chisquare import chi_squared_test


def stratify_x_y_mat(x_y_mat, z_mat, cluster_number = 2):
    """
    Cluster data into {cluster_number} of clusters.
    :param x_y_mat: An n x 2 numpy array. Each row is the response of the ith observation.
    First column corresponds to x.
    :param z_mat: A 2D numpy array. Each row is a data point.
    :param cluster_number: An integer which is the number of clusters to form by the KMeans function.
    :return: z_mat_vet: A python list of length {cluster_number}. Each element is a 2D numpy array of a data cluster.
    """
    kmeans_model = KMeans(n_clusters=cluster_number, random_state=0)
    kmeans_result_vet = kmeans_model.fit(z_mat).labels_
    x_y_mat_vet = [x_y_mat[kmeans_result_vet == i, :] for i in range(cluster_number)]

    return x_y_mat_vet


def stratified_chi_squared_test(x_y_mat_vet):
    """
    Compute test statistic.

    :param x_y_mat_vet: A python list which is the output of the stratify_x_y_mat function.
    :return: A numeric which is the sum of of the chi-squared statistic on each stratum.
    """
    chi_square_statistic_vet = np.zeros(len(x_y_mat_vet))
    for iteration, x_y_mat in enumerate(x_y_mat_vet):
        chi_square_statistic_vet[iteration] = chi_squared_test(x_y_mat)[0]

    return sum(chi_square_statistic_vet)


def simulation_wrapper_stratified(simulation_index, scenario, sample_size, z_mat, cluster_number= 2):
    """
    A wrapper function for the multiprocessing Pool function. It will be passed into the partial function.
    The pool function will run iteration in parallel given a sample size and a scenario.
    This function perform Chi squared test on {simulation_index}th sample with sample size
    {sample_size} under the {scenario} hypothesis.
    The return will be used to create a dictionary.

    :param simulation_index: An integer indicating {simulation_index}th simulated.
    :param scenario: A string ('str' class) which is either "null" or "alt" indicating if the sample is simulated
    under the null or alternative hypothesis.
    :param sample_size: An integer.
    :return: A tuple (simulation_index, result_vet). result_vet is the return of the chi_squared_test function.
    """
    x_y_mat = np.loadtxt(f"./data/{scenario}/x_y_mat_{sample_size}_{simulation_index}.txt")

    x_y_mat_vet = stratify_x_y_mat(x_y_mat = x_y_mat, z_mat = z_mat, cluster_number = cluster_number)
    test_statistic = stratified_chi_squared_test(x_y_mat_vet)

    return (simulation_index, test_statistic)


#######################
# Test under the null #
#######################
stratified_chisq_result_null_dict = dict()
for sample_size in hp.sample_size_vet:

    pool = mp.Pool(processes = hp.process_number)
    simulation_index_vet = range(hp.simulation_times)
    z_mat = np.loadtxt(f"./data/z_mat/z_mat_{sample_size}.txt")
    pool_result_vet = pool.map(partial(simulation_wrapper_stratified, sample_size = sample_size, scenario = "null",
                                       z_mat = z_mat), simulation_index_vet)

    stratified_chisq_result_null_dict[sample_size] = dict(pool_result_vet)

    print(f"Null, {sample_size} finished")

with open("./results/stratified_chisq_result_null_dict.p", "wb") as fp:
    pickle.dump(stratified_chisq_result_null_dict, fp, protocol = pickle.HIGHEST_PROTOCOL)


##############################
# Test under the alternative #
##############################
stratified_chisq_result_alt_dict = dict()
for sample_size in hp.sample_size_vet:

    z_mat = np.loadtxt(f"./data/z_mat/z_mat_{sample_size}.txt")
    pool_result_vet = pool.map(partial(simulation_wrapper_stratified, sample_size = sample_size, scenario = "alt",
                                       z_mat = z_mat), simulation_index_vet)

    stratified_chisq_result_alt_dict[sample_size] = dict(pool_result_vet)

    print(f"Alt, {sample_size} finished")

with open("./results/stratified_chisq_result_alt_dict.p", "wb") as fp:
    pickle.dump(stratified_chisq_result_alt_dict, fp, protocol = pickle.HIGHEST_PROTOCOL)





"""
scenario = "null"
sample_size = 100
simulation = 10

z_mat = np.loadtxt(f"./data/{scenario}/z_mat_{sample_size}.txt")
x_y_mat = np.loadtxt(f"./data/{scenario}/x_y_mat_{sample_size}_{simulation}.txt")

kmeans_result = KMeans(n_clusters=2, random_state=0).fit(z_mat)
sum(kmeans_result.labels_)
color_vet = ["red", "blue"]
color_vet = [color_vet[i] for i in kmeans_result.labels_]
plt.scatter(z_mat[:, 0], z_mat[:, 1], color = color_vet)
"""