import os
from lib.utils.manage_pickle_files import load_obj


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
                        "category_name": category_map[row[0].split("_")[0]]}

    def format(self):
        final_data = {}

        if self.data_type == "PRE-GOD":
            god_data = load_obj(os.path.join(
                self.config.data_dir, "1200_data.pkl"))

            for subj in god_data.keys():
                final_data[subj] = {}
                final_data[subj]["roi_names"] = list(god_data["01"].keys())
                final_data[subj]["stimulus_ids"] = god_data["01"]["FFA"]["labels"]
                final_data[subj]["category_ids"] = [self.image_details[sti]
                                                    ["category"] for sti in final_data[subj]["stimulus_ids"]]
                final_data[subj]["image_names"] = [self.image_details[sti]
                                                   ["image_name"] for sti in final_data[subj]["stimulus_ids"]]
                final_data[subj]["category_names"] = [self.image_details[sti]
                                                      ["category_name"] for sti in final_data[subj]["stimulus_ids"]]

                for roi in god_data[subj].keys():
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
                final_data[subj]["category_names"] = [self.image_details[sti]
                                                      ["category_name"] for sti in final_data[subj]["stimulus_ids"]]
                final_data[subj]["roi_data"] = data["roi_data"]

        return final_data
