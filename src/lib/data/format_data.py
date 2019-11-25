import os
from lib.utils.manage_pickle_files import load_obj
import numpy as np


class FormatData:
    def __init__(self, config, data_type="PRE-GOD"):
        self.data_type = data_type
        self.config = config
        if self.data_type == "PRE-GOD":
            self.file_name = os.path.join(config.data_dir, "")
        else:
            self.file_names = os.listdir(config.final_data_dir)

        self.image_details = {}
        with open(os.path.join(config.data_dir, "stimulus_ImageNetTraining.tsv"), "r") as imges_details_file:
            with open(os.path.join(config.data_dir, "mapped_stimulus_ImageNetTraining.tsv"), "r") as map_file:
                content = imges_details_file.read()
                rows = content.split("\n")
                rows = [row.split("\t") for row in rows]

                map_file_content = map_file.read()
                category_rows = map_file_content.split("\n")
                category_map = {
                    row.split("\t")[0]: row.split("\t")[1]
                    for row in category_rows
                }

                for row in rows:
                    self.image_details[row[1]] = {
                        "image_name": row[0],
                        "category": row[2],
                        "num": row[3],
                        "category_name": category_map[row[0].split("_")[0]].split(",")[0]}

    def format(self):
        final_data = {}

        if self.data_type == "PRE-GOD":
            god_data = load_obj(os.path.join(
                self.config.data_dir, "1200_data.pkl"))

            for subj in god_data.keys():
                final_data[subj] = {}
                final_data[subj]["roi_names"] = list(god_data["1"].keys())
                final_data[subj]["stimulus_ids"] = np.array(
                    god_data["1"]["FFA"]["labels"]).astype(str)
                final_data[subj]["category_ids"] = np.array([self.image_details[sti[0]]
                                                             ["category"] for sti in final_data[subj]["stimulus_ids"]])
                final_data[subj]["image_names"] = np.array([self.image_details[sti[0]]
                                                            ["image_name"] for sti in final_data[subj]["stimulus_ids"]])
                final_data[subj]["category_names"] = np.array([self.image_details[sti[0]]
                                                               ["category_name"] for sti in final_data[subj]["stimulus_ids"]])

                for roi in god_data[subj].keys():
                    if "roi_data" not in final_data[subj].keys():
                        final_data[subj]["roi_data"] = {}
                    final_data[subj]["roi_data"][roi] = god_data[subj][roi]["brain_data"]

        elif self.data_type == "GOD":
            files = os.listdir(os.path.join(self.config.final_data_dir))
            for file_name in files:
                subj = str(int(file_name.split(".")[0][-2:]))
                data = load_obj(os.path.join(
                    self.config.final_data_dir, file_name))
                final_data[subj] = {}
                final_data[subj]["roi_names"] = data["roi_names"]
                final_data[subj]["stimulus_ids"] = data["stimulus_id"]
                final_data[subj]["category_ids"] = data["category"]
                final_data[subj]["image_names"] = data["image_name"]
                final_data[subj]["category_names"] = np.array([self.image_details[sti]
                                                               ["category_name"] for sti in final_data[subj]["stimulus_ids"]])
                final_data[subj]["roi_data"] = data["roi_data"]

        if self.config.subset_data:
            self.get_required_labels(final_data)

        return final_data

    def get_required_labels(self,  data, label_type="category_ids"):
        for subj in data.keys():
            y = np.where(
                data[subj][label_type] == self.config.category_ids.reshape(-1, 1))
            ind = y[1]
            data[subj]["stimulus_ids"] = data[subj]["stimulus_ids"][ind]
            data[subj]["category_ids"] = data[subj]["category_ids"][ind]
            data[subj]["image_names"] = data[subj]["image_names"][ind]
            data[subj]["category_names"] = data[subj]["category_names"][ind]
            if self.data_type == "PRE-GOD":
                for roi in data[subj]["roi_data"].keys():
                    data[subj]["roi_data"][roi] = data[subj]["roi_data"][roi][ind]
            elif self.data_type == "GOD":
                for roi in data[subj]["roi_data"].keys():
                    for stat in data[subj]["roi_data"][roi]:
                        data[subj]["roi_data"][roi][stat] = data[subj]["roi_data"][roi][stat][ind]
