import os
from lib.utils.config import Config
from lib.data.format_data import FormatData
from lib.rsa.rdms import GenerateRDMs
from lib.utils.corr import get_corr
from tqdm import tqdm
import sys

sys.setrecursionlimit(15000)


def analyse_RDMS(Config):
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

    for data_type in tqdm(["PRE-GOD"]):
        with open("../results/"+data_type+".txt", "w") as corr_file:
            data = FormatData(config, data_type=data_type).format()
            roi_names = data["1"]["roi_names"]
            for roi in roi_names:
                for i in range(1, config.num_of_subjs+1):
                    for j in range(i, config.num_of_subjs+1):
                        corr = get_corr(
                            data[str(i)]["roi_data"][roi], data[str(j)]["roi_data"][roi])
                        print(f'{i}, {j} : {corr}')


def main(config):
    if config.corr:
        find_corr(config)
    else:
        analyse_RDMS(config)


if __name__ == "__main__":
    config = Config().parse()
    main(config)
