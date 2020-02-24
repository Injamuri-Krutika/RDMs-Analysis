import os
from lib.utils.config import Config
from lib.data.format_data import FormatData
from lib.rsa.rdms import GenerateRDMs
from lib.utils.corr import get_corr
from lib.utils.new_categories import get_new_category_list, get_new_cat_image_names
from tqdm import tqdm
import numpy as np
import sys

sys.setrecursionlimit(15000)


def analyse_RDMs(config):
    for data_type in tqdm(["GOD", "PRE-GOD"]):
        data = FormatData(config, data_type=data_type).format()
        for subj in tqdm(data.keys()):
            for roi in data[subj]["roi_names"]:
                if data_type == "GOD":
                    for stat in data[subj]["roi_data"][roi].keys():
                        path = os.path.join(
                            config.result_dir, config.exp_id, data_type, subj, stat)
                        if not os.path.exists(path):
                            os.makedirs(path)
                        GenerateRDMs(config).rdm(
                            data[subj]["roi_data"][roi][stat], roi, path, data[subj]["category_names"])

                else:

                    path = os.path.join(
                        config.result_dir, config.exp_id, data_type, subj)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    GenerateRDMs(config).rdm(
                        data[subj]["roi_data"][roi], roi, path, data[subj]["category_names"])


def find_corr(config):
    for data_type in ["GOD", "PRE-GOD"]:
        with open("../results/"+data_type+".txt", "w") as corr_file:
            data = FormatData(config, data_type=data_type,
                              ).format()
            roi_names = data["1"]["roi_names"]
            for roi in roi_names:
                for i in range(1, config.num_of_subjs+1):
                    for j in range(i, config.num_of_subjs+1):
                        rdm1 = GenerateRDMs(config).rdm(
                            data[str(i)]["roi_data"][roi])
                        rdm2 = GenerateRDMs(config).rdm(
                            data[str(j)]["roi_data"][roi])
                        corr = get_corr(
                            rdm1["pearson"], rdm2["pearson"])
                        print(f'{i}, {j} : {corr}')


def analyse_categorised_RDMs(config):
    for data_type in tqdm(["GOD", "PRE-GOD"]):
        data = FormatData(config, data_type=data_type,
                          ).format()
        for subj in tqdm(data.keys()):
            for roi in data[subj]["roi_names"]:
                new_cat_imagenames = get_new_cat_image_names()
                required_ind = []
                for i in range(len(new_cat_imagenames)):
                    required_ind = required_ind + list(np.where(np.in1d(data[subj]["image_names"], np.array(
                        new_cat_imagenames[i])) == True)[0])
                if data_type == "GOD":
                    for stat in data[subj]["roi_data"][roi].keys():

                        path = os.path.join(
                            config.result_dir, "categorised", config.exp_id, data_type, subj, stat)
                        if not os.path.exists(path):
                            os.makedirs(path)

                        GenerateRDMs(config).rdm(
                            data[subj]["roi_data"][roi]
                            [stat][required_ind], roi, path, data[subj]["category_names"][required_ind])

                else:
                    path = os.path.join(
                        config.result_dir, "categorised",  config.exp_id, data_type, subj)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    GenerateRDMs(config).rdm(
                        data[subj]["roi_data"][roi][required_ind], roi, path, data[subj]["category_names"][required_ind])


def main(config):
    if config.corr:
        find_corr(config)
    elif config.categorise:
        analyse_categorised_RDMs(config)
    else:
        analyse_RDMs(config)


if __name__ == "__main__":
    config = Config().parse()
    main(config)
