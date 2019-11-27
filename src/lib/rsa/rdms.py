import numpy as np
from lib.distance_measures.gaussian_kernel import gaussian_kernal_similarity
from lib.distance_measures.epanechnikov_kernel import epanechnikov_similarity

from lib.utils.plot import plot_rdm_group


class GenerateRDMs:
    def __init__(self, config):
        self.config = config

    def rdm(self, data, roi, path, labels):
        rdm = {}
        for distance_measure in self.config.distance_measures:
            if distance_measure == "pearson":
                rdm["pearson"] = 1 - np.corrcoef(data)

            elif distance_measure == "kernel":
                rdm["kernel"] = 1 - gaussian_kernal_similarity(data)

            elif distance_measure == "epanechnicov":
                rdm["epanechnicov"] = 1 - epanechnikov_similarity(data)

        plot_rdm_group(rdm, roi, self.config.distance_measures, path, labels)
