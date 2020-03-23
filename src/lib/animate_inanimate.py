from lib.data.format_data import FormatData
from lib.utils.new_categories import animate_inanimate
from lib.rsa.rdms import GenerateRDMs
from lib.utils.plot import plot_rdm_group
import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class AnimateInanimate():
    def __init__(self, config):
        self.config = config

    def plot_boxplot(self, data, path, x_labels):

        fig, axs = plt.subplots(3, 1, figsize=(25, 25))
        axs[0].boxplot(data["plants"])
        axs[0].set_title('Plants Related Images')
        axs[0].set_xticklabels(x_labels)

        axs[1].boxplot(data["animals"])
        axs[1].set_title('Animals Related Images')
        axs[1].set_xticklabels(x_labels)

        axs[2].boxplot(data["inanimate"])
        axs[2].set_title('Inanimate Related Images')
        axs[2].set_xticklabels(x_labels)

        plt.savefig(path)
        plt.close()

    def categorise(self, rdm, labels, roi, path, super_categorise=False):
        if not super_categorise:
            ind_tuples = [(i, i+8) for i in range(0, 185, 8)]
        else:
            ind_tuples = [(i, i+64) for i in range(0, 129, 64)]
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
        if super_categorise:
            new_labels = ["plants", "animals", "inanimate"]
        plot_rdm_group(
            new_rdm, roi, self.config.distance_measures, path, new_labels)

    def run(self):
        self.config.subset_data = True
        new_list = list(animate_inanimate.values())
        self.config.category_ids = list(
            itertools.chain.from_iterable(new_list))
        num_cat = len(self.config.category_ids)
        num_of_images = num_cat * 8
        data = FormatData(self.config, data_type="PRE-GOD").format()

        subjs = ["1", "2", "3", "4", "5"]
        roi_names = ["V1", "V2", "V3", "V4", "LVC",
                     "HVC", "VC", "LOC", "FFA", "PPA"]
        base_path = os.path.join(self.config.result_dir,
                                 self.config.exp_id)
        avg_path = os.path.join(base_path, "AVG")
        cat_path = os.path.join(base_path, "Categorised")
        sup_cat_path = os.path.join(base_path, "Super_Categorised")

        boxplot_path = os.path.join(base_path, "boxplot.png")

        if not os.path.exists(avg_path):
            os.makedirs(avg_path)
            os.makedirs(cat_path+"/AVG")
            os.makedirs(sup_cat_path+"/AVG")

        boxplot_data = {
            "plants": [],
            "animals": [],
            "inanimate": []
        }

        for roi in tqdm(roi_names):
            roi_mean_boxplot = {
                "plants": [],
                "animals": [],
                "inanimate": []
            }

            avg_rdm = {
                "kernel": np.zeros((num_of_images, num_of_images)),
                "pearson": np.zeros((num_of_images, num_of_images))
            }
            all_subj_rdm = {
                "kernel": np.zeros((len(subjs), num_of_images, num_of_images)),
                "pearson": np.zeros((len(subjs), num_of_images, num_of_images))
            }
            for subj in subjs:
                path = os.path.join(base_path, subj)
                if not os.path.exists(path):
                    os.makedirs(path)
                    os.makedirs(os.path.join(cat_path, subj))
                    os.makedirs(os.path.join(sup_cat_path, subj))

                rdm = GenerateRDMs(self.config).rdm(
                    data[subj]["roi_data"][roi], roi, path, data[subj]["category_names"])
                for distance in self.config.distance_measures:
                    all_subj_rdm[distance][int(subj)-1] = rdm[distance]
                self.categorise(rdm,
                                data[subj]["category_names"], roi, os.path.join(cat_path, subj))
                self.categorise(rdm,
                                data[subj]["category_names"], roi, os.path.join(sup_cat_path, subj), super_categorise=True)

                i = 0
                for key in roi_mean_boxplot.keys():
                    roi_mean_boxplot[key] = roi_mean_boxplot[key] +\
                        [np.mean(data[subj]["roi_data"][roi][i:i+64, ])]
                    i += 64

            for key in roi_mean_boxplot.keys():
                roi_mean_boxplot[key] = roi_mean_boxplot[key] +\
                    [np.mean(roi_mean_boxplot[key])]
                boxplot_data[key] = boxplot_data[key] + [roi_mean_boxplot[key]]

            for distance in self.config.distance_measures:
                avg_rdm[distance] = np.mean(all_subj_rdm[distance], axis=0)
            plot_rdm_group(
                avg_rdm, roi, self.config.distance_measures, avg_path, data[subjs[0]]["category_names"])
            self.categorise(avg_rdm,
                            data[subj]["category_names"], roi, os.path.join(cat_path, "AVG"))
            self.categorise(avg_rdm,
                            data[subj]["category_names"], roi, os.path.join(sup_cat_path, "AVG"), super_categorise=True)
        print(boxplot_data)
        self.plot_boxplot(boxplot_data, boxplot_path, roi_names)
