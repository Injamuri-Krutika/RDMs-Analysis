import os
from lib.utils.config import Config
from lib.data.format_data import FormatData
from lib.rsa.rdms import GenerateRDMs
# Initialise variables


def main(config):
    for data_type in ["GOD", "PRE-GOD"]:
        data = FormatData(config, data_type=data_type).format()
        for subj in data.keys():
            for roi in data[subj]["roi_names"]:
                if type(data[subj]["roi_data"][roi]) == dict:
                    for stat in data[subj]["roi_data"][roi].keys():
                        path = os.path.join(
                            config.result_dir, config.exp_id, data_type, subj, stat)
                        if not os.path.exists(path):
                            os.makedirs(path)
                        GenerateRDMs(config).rdm(
                            data[subj]["roi_data"][roi][stat], roi, path)


if __name__ == "__main__":
    config = Config().parse()
    main(config)
