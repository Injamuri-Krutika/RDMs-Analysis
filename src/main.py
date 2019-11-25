import os
from lib.utils.config import Config
from lib.data.format_data import FormatData
from lib.rsa.rdms import GenerateRDMs
from tqdm import tqdm
import sys
sys.setrecursionlimit(15000)

# Initialise variables


def main(config):
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


if __name__ == "__main__":
    config = Config().parse()
    main(config)
