from lib.data.format_data import FormatData
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import os


class HierarchicalClustering:
    def __init__(self, config):
        self.config = config

    def plot_dendogram(self, data, labels, subj, roi):
        path = os.path.join(self.config.result_dir,
                            self.config.exp_id, "Dendograms", subj)
        if not os.path.exists(path):
            os.makedirs(path)
        fig, ax = plt.subplots(figsize=(30, 15))
        dendrogram = sch.dendrogram(sch.linkage(
            data, method="ward"), labels=labels, leaf_font_size=12)
        plt.title('Subj:' + subj + ", Roi: " + roi)
        plt.xlabel('Categories')
        plt.ylabel('Euclidean distances')
        plt.savefig(path+"/"+roi+".png")
        plt.close()

    def run(self):
        data = FormatData(self.config, data_type="PRE-GOD").format()
        subjs = ["1", "2", "3", "4", "5"]
        roi_names = ["V1", "V2", "V3", "V4", "LVC",
                     "HVC", "VC", "LOC", "FFA", "PPA"]
        num_of_cat = 150

        category_labels = []
        for roi in roi_names:
            avg_voxels = np.zeros(
                (len(subjs), num_of_cat, data[subjs[0]]["roi_data"][roi].shape[1]))
            for subj in subjs:
                print(subj, roi, data[subj]["roi_data"][roi].shape[1])
                categorised_voxels = np.zeros(
                    (num_of_cat, data[subj]["roi_data"][roi].shape[1]))
                for i in range(8, 1201, 8):
                    roi_data = data[subj]["roi_data"][roi]
                    ind = int(i/8 - 1)
                    categorised_voxels[ind] = np.mean(
                        roi_data[i-8:i, ], axis=0)
                    category_labels = category_labels +\
                        [data[subj]["category_names"][ind]]
                # avg_voxels[int(subj)-1] = categorised_voxels
                self.plot_dendogram(categorised_voxels,
                                    category_labels, subj, roi)
            # self.plot_dendogram(avg_voxels,
            #                     category_labels, "AVG", roi)
