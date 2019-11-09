import numpy as np

sample_size_vet = [30, 100, 500, 1000]
dim_z = 3

np.random.seed(1)
for sample_size in sample_size_vet:
    z = np.random.normal(0, 10, (sample_size, dim_z))
    np.savetxt("./data/z_%d.txt" % sample_size, z)