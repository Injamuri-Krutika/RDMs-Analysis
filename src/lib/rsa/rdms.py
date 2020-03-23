import numpy as np
from lib.distance_measures.gaussian_kernel import gaussian_kernal_similarity
from lib.distance_measures.epanechnikov_kernel import epanechnikov_similarity

from lib.utils.plot import plot_rdm_group
from scipy import io
import os
from lib.utils.new_categories import get_ind_tuples, super_categories, get_sup_cat_ind_tuples


def get_box_plot_cat_tupleind(labels):
    start, end = 0, -1
    prev_cat = ""
    cats = labels
    cat_ind = -1
    start_end_ind = []
    for i, cat in enumerate(cats):
        if prev_cat != cat:
            if cat_ind != -1:
                start_end_ind = start_end_ind + \
                    [(start, i)]
                start = i
            cat_ind += 1
        prev_cat = cat
    start_end_ind = start_end_ind + \
        [(start, len(labels))]
    return start_end_ind


def get_box_plot_supcat_tupleind():
    return [(0, 9), (9, 32)]


class GenerateRDMs:
    def __init__(self, config):
        self.config = config

    def categorise_rdms(self, rdm, labels):

        if self.config.categorise or self.config.box_plot:
            ind_tuples = get_box_plot_cat_tupleind(
                labels) if self.config.box_plot else get_ind_tuples()
            num_of_cat = len(ind_tuples)
            new_rdm = {
                self.config.distance_measures[0]: np.zeros(
                    (num_of_cat, num_of_cat)),
                self.config.distance_measures[1]: np.zeros(
                    (num_of_cat, num_of_cat))
            }

            for distance_measure in self.config.distance_measures:
                new_labels = []
                for cat_num1 in range(num_of_cat):
                    for cat_num2 in range(num_of_cat):
                        new_rdm[distance_measure][cat_num1, cat_num2] = np.mean(
                            rdm[distance_measure][ind_tuples[cat_num1][0]:ind_tuples[cat_num1][1],
                                                  ind_tuples[cat_num2][0]:ind_tuples[cat_num2][1]])
                    new_labels = new_labels + [labels[ind_tuples[cat_num1][0]]]
            return new_rdm, new_labels

    def super_categorise_rdms(self, rdm):

        num_of_cat = 2 if self.config.box_plot else len(
            super_categories.keys())
        ind_tuples = get_box_plot_supcat_tupleind(
        ) if self.config.box_plot else get_sup_cat_ind_tuples()
        sup_cat_rdm = {
            self.config.distance_measures[0]: np.zeros(
                (num_of_cat, num_of_cat)),
            self.config.distance_measures[1]: np.zeros(
                (num_of_cat, num_of_cat))
        }
        for distance_measure in self.config.distance_measures:
            for cat_num1 in range(num_of_cat):
                for cat_num2 in range(num_of_cat):
                    sup_cat_rdm[distance_measure][cat_num1, cat_num2] = np.mean(
                        rdm[distance_measure][ind_tuples[cat_num1][0]:ind_tuples[cat_num1][1],
                                              ind_tuples[cat_num2][0]:ind_tuples[cat_num2][1]])

        return sup_cat_rdm

    def rdm(self, data, roi=None, path=None, labels=None):
        rdm = {}
        for distance_measure in self.config.distance_measures:
            if distance_measure == "pearson":
                rdm["pearson"] = 1 - np.corrcoef(data)

            elif distance_measure == "kernel":
                rdm["kernel"] = 1 - gaussian_kernal_similarity(data)

            elif distance_measure == "epanechnicov":
                rdm["epanechnicov"] = 1 - epanechnikov_similarity(data)

        io.savemat(os.path.join(path, roi+".mat"), rdm)

        if self.config.categorise or self.config.box_plot:
            new_rdm, new_labels = self.categorise_rdms(rdm, labels)
            new_path = os.path.join(path, "categorised")
            if not os.path.exists(new_path):
                os.makedirs(new_path)

            io.savemat(os.path.join(
                path, "categorised", roi+".mat"), new_rdm)
            plot_rdm_group(
                new_rdm, roi, self.config.distance_measures, new_path, new_labels)

            sup_cat_rdm = self.super_categorise_rdms(rdm)

            new_path = os.path.join(path, "super_categorised")
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            io.savemat(os.path.join(
                path, "super_categorised", roi+".mat"), sup_cat_rdm)
            labels = ["places", "faces"] if self.config.box_plot else list(
                super_categories.keys())
            plot_rdm_group(
                sup_cat_rdm, roi, self.config.distance_measures, new_path, labels)

        if path != None and len(labels) != 0:
            plot_rdm_group(
                rdm, roi, self.config.distance_measures, path, labels)
        return rdm
