
from lib.data.format_data import FormatData
from lib.rsa.rdms import GenerateRDMs
from lib.utils.plot import box_plot
import numpy as np
from lib.utils.new_categories import place_categoryids, face_imagenames, homonoid_monkey_imagenames, animate_inanimate

import os
import argparse
from algonauts.src.lib.utils import utils, config, networks_factory, constants
import torch
import glob
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict
from torch.autograd import Variable
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm


class BoxPlotting:
    def __init__(self, config):
        data = FormatData(config, data_type="PRE-GOD").format()
        subj = "1"
        self.p_image_names = data[subj]["image_names"][np.where(
            data[subj]["category_ids"] == np.array(
                place_categoryids).reshape(-1, 1))[1]]
        self.f_image_names = face_imagenames + homonoid_monkey_imagenames
        self.plant_image_names = data[subj]["image_names"][np.where(
            data[subj]["category_ids"] == np.array(
                animate_inanimate["plants"]).reshape(-1, 1))[1]]
        self.animal_image_names = data[subj]["image_names"][np.where(
            data[subj]["category_ids"] == np.array(
                animate_inanimate["animals"]).reshape(-1, 1))[1]]
        self.inanimate_image_names = data[subj]["image_names"][np.where(
            data[subj]["category_ids"] == np.array(
                animate_inanimate["inanimate"]).reshape(-1, 1))[1]]
        self.config = config
        self.config.image_dir = os.path.join(
            self.config.data_dir, "images", "train_images", "allimages")

    def models_plot(self, val_list, labels, title_list, path):

        fig, axs = plt.subplots(2, 1, figsize=(15, 15))

        for i, ax in enumerate(axs):
            ax.scatter(labels, val_list[i])
            ax.set_title(title_list[i])
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylim(bottom=-5, top=5)
            for x, y in zip(labels, val_list[i]):
                label = "{:.2f}".format(y)
                ax.annotate(label,  # this is the text
                            (x, y),  # this is the point to label
                            textcoords="offset points",  # how to position the text
                            # distance from text to points (x,y)
                            xytext=(0, 10),
                            ha='center')

        plt.savefig(path+".png")
        plt.close()

    def _execute(self,  model, centre_crop, feats, image_group_list):
        for i, image_list in enumerate(image_group_list):
            image_count = 0
            for image in tqdm(image_list):
                image_path = os.path.join(self.config.image_dir, image+".jpg")
                if os.path.exists(image_path):
                    image_count += 1
                    img = Image.open(image_path)
                    img = img.convert(mode="RGB")
                    filename = image.split("/")[-1].split(".")[0]
                    input_img = Variable(centre_crop(img).unsqueeze(0))
                    if torch.cuda.is_available():
                        input_img = input_img.to(self.config.device)
                    x = model.forward(input_img)
                    for key, value in x.items():
                        if key not in feats[i].keys():
                            feats[i][key] = value.data.cpu().numpy()
                        else:
                            feats[i][key] = np.vstack(
                                (feats[i][key], value.data.cpu().numpy()))

            for key, value in feats[i].items():
                feats[i][key] = np.mean(feats[i][key])
            print(image_count)

    def execute_model(self, model, model_name):
        if torch.cuda.is_available():
            model.to(self.config.device)
        model.eval()
        centre_crop = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        feats_list = [
            {
                "feats": [OrderedDict(), OrderedDict()],
                "image_group_list":[self.p_image_names, self.f_image_names],
                "path": os.path.join(self.config.result_dir,
                                     self.config.exp_id, "places_faces")

            },
            {
                "feats": [OrderedDict(), OrderedDict(), OrderedDict()],
                "image_group_list":[self.plant_image_names, self.animal_image_names, self.inanimate_image_names],
                "path": os.path.join(self.config.result_dir,
                                     self.config.exp_id, "plants_animals_inanimate")
            },
        ]

        for feats_dict in feats_list:
            feats = feats_dict["feats"]
            self._execute(model, centre_crop, feats,
                          feats_dict["image_group_list"])
            path = feats_dict["path"]
            if not os.path.exists(path):
                os.makedirs(path)
            val_list = [feat.values() for feat in feats]
            if len(feats) == 2:
                title_list = ["Place Images", "Face Images"]
            elif len(feats) == 3:
                title_list = ["Plant Images",
                              "Animal Images", "Inanimate Images"]

            self.models_plot(val_list=val_list,
                             labels=list(feats[0].keys()),
                             title_list=title_list,
                             path=path+"/"+model_name)
            del feats

    def get_model(self, model, load_model=None):
        return utils.get_model(self.config.arch)

    def run(self):
        # This is only done for the PRE-GOD data_type
        # The order of subjects in the data dictionary is dict_keys(['1', '3', '2', '5', '4'])
        data = FormatData(self.config, data_type="PRE-GOD").format()
        place_last_ind, face_last_ind = data["place_lastind"], data["face_lastind"]
        del data["place_lastind"]
        del data["face_lastind"]
        place_data, face_data, object_data = [], [], []
        subjs = ["1", "2", "3", "4", "5"]
        roi_names = ["V1", "V2", "V3", "V4", "LVC",
                     "HVC", "VC", "LOC", "FFA", "PPA"]

        for roi in tqdm(roi_names):
            p_data, f_data, o_data = [], [], []
            for subj in subjs:
                path = os.path.join(self.config.result_dir,
                                    self.config.exp_id, subj)
                if not os.path.exists(path):
                    os.makedirs(path)
                GenerateRDMs(self.config).rdm(
                    data[subj]["roi_data"][roi], roi, path, data[subj]["category_names"])
                p_data = p_data + \
                    [np.mean(data[subj]["roi_data"][roi][:place_last_ind])]

                f_data = f_data + \
                    [np.mean(
                        data[subj]["roi_data"][roi][place_last_ind:face_last_ind])]
                o_data = o_data + \
                    [np.mean(data[subj]["roi_data"][roi][face_last_ind::])]
            p_data = p_data + [np.mean(p_data)]
            f_data = f_data + [np.mean(f_data)]
            o_data = o_data + [np.mean(o_data)]

            place_data = place_data + [p_data]
            face_data = face_data + [f_data]
            object_data = object_data + [o_data]

        path = os.path.join(self.config.result_dir,
                            self.config.exp_id, "box_plot")

        box_plot([place_data, face_data, object_data],
                 ["Scene Images", "Face Images", "Object Images"],
                 roi_names, path)

    def run_model(self):

        for arch in self.config.archs:
            print("Model running:", arch)
            self.config.arch = arch
            model_name = arch

            model = self.get_model(model_name)
            self.execute_model(model, model_name)

        print("End")
