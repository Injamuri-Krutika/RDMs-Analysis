
from lib.data.format_data import FormatData
from lib.rsa.rdms import GenerateRDMs
from lib.utils.plot import box_plot
import numpy as np

import os


class BoxPlotting:
    def __init__(self):
        pass

    def run(self, config):
        # This is only done for the PRE-GOD data_type
        # The order of subjects in the data dictionary is dict_keys(['1', '3', '2', '5', '4'])
        data = FormatData(config, data_type="PRE-GOD").format()
        place_last_ind = data["place_lastind"]
        del data["place_lastind"]
        place_data, face_data = [], []
        subjs = ["1", "2", "3", "4", "5"]
        roi_names = ["V1", "V2", "V3", "V4", "LVC",
                     "HVC", "VC", "LOC", "FFA", "PPA"]

        for roi in roi_names:
            p_data, f_data = [], []
            for subj in subjs:
                path = os.path.join(config.result_dir,
                                    config.exp_id, subj)
                if not os.path.exists(path):
                    os.makedirs(path)
                GenerateRDMs(config).rdm(
                    data[subj]["roi_data"][roi], roi, path, data[subj]["category_names"])
                p_data = p_data + \
                    [np.mean(data[subj]["roi_data"][roi][:place_last_ind])]
                f_data = f_data + \
                    [np.mean(data[subj]["roi_data"][roi][place_last_ind::])]
            p_data = p_data + [np.mean(p_data)]
            f_data = f_data + [np.mean(f_data)]
            place_data = place_data + [p_data]
            face_data = face_data + [f_data]
        path = os.path.join(config.result_dir, config.exp_id, "box_plot")
        box_plot(place_data, face_data, roi_names, path)
