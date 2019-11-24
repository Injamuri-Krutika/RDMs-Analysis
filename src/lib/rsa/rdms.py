import numpy as np
from lib.distance_measures.gaussian_kernel import gaussian_kernal_similarity
from lib.utils.plot import plot_rdm_group


class GenerateRDMs:
    def __init__(self, config):
        self.config = config
        print(self.config)

    def rdm(self, data, roi, path):
        rdm = {}
        for distance_measure in self.config.distance_measures:
            if distance_measure == "pearson":
                rdm["pearson"] = 1 - np.corrcoef(data)

            elif distance_measure == "kernel":
                rdm["kernel"] = 1 - gaussian_kernal_similarity(data)

        plot_rdm_group(rdm, roi, self.config.distance_measures, path)
