import numpy as np
from scipy.stats import chisquare
import hyperparameters as hp

sample_size = 1000
iteration = 0
naive_chisq_result_null_dict = dict{}

def contingency_table(x_y_mat):
    count_vet = np.unique(x_y_mat, axis=0, return_counts=True)

for sample_size in hp.sample_size_vet:
    for simulation in hp.simulation_times:
        x_y_mat = np.load(f"./data/null/x_y_mat_{sample_size}_{simulation}.txt")
        category_count = np.unique(x_y_mat, axis=0, return_counts=True)[0]


x_y_mat = np.loadtxt(f"./data/null/x_y_mat_{sample_size}_{iteration}.txt")
category_count = np.unique(x_y_mat, axis = 0, return_counts = True)
chisquare(category_count[1])


vet = np.zeros(1000)
for iteration in range(1000):
    x_y_mat = np.loadtxt(f"./data/alt/x_y_mat_{sample_size}_{iteration}.txt")
    vet[iteration] = len(np.unique(x_y_mat, axis=0))

print(np.unique(vet, return_counts=True))
