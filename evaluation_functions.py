import numpy as np
import matplotlib.pyplot as plt


class Loss_Dict:
    def __init__(self, loss_one_sample_dictionary, sample_size_vet, batch_size):
        # Assume we only have four sample size.
        self.loss_dict = loss_one_sample_dictionary
        self.sample_size_vet = sample_size_vet
        self.batch_size = batch_size

    def plot_epoch_loss(self, start_epoch, end_epoch):
        start_index = start_epoch - 1
        end_index = end_epoch - 1

        fig, axs = plt.subplots(4)
        for i in range(4):
            sample_size = self.sample_size_vet[i]
            negLogLikelihood = self.loss_dict[self.sample_size_vet[i]][0, :]
            kl = self.loss_dict[self.sample_size_vet[i]][1, :]
            if sample_size // self.batch_size != 0:
                epoch_index_boolean = np.mod(np.arange(len(kl)) + 1, np.ceil(sample_size // self.batch_size)) == 0
                negLogLikelihood = negLogLikelihood[epoch_index_boolean]
                kl = kl[epoch_index_boolean]

            negLogLikelihood = negLogLikelihood[start_index: end_index]
            kl = kl[start_index: end_index]

            axs[i].plot(negLogLikelihood, label="likelihood")
            axs[i].plot(kl, label="kl")
            axs[i].legend()
            axs[i].set_title("Sample size %d" % sample_size)

    def plot_epoch_kl(self, start_epoch, end_epoch):
        start_index = start_epoch - 1
        end_index = end_epoch - 1

        fig, axs = plt.subplots(4)
        for i in range(4):
            sample_size = self.sample_size_vet[i]
            kl = self.loss_dict[self.sample_size_vet[i]][1, :]
            if sample_size // self.batch_size != 0:
                epoch_index_boolean = np.mod(np.arange(len(kl)) + 1, np.ceil(sample_size // self.batch_size)) == 0
                kl = kl[epoch_index_boolean]

            kl = kl[start_index: end_index]

            axs[i].plot(kl, label="kl")
            axs[i].legend()
            axs[i].set_title("Sample size %d" % sample_size)
