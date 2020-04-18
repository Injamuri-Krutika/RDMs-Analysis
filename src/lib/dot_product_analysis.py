
from lib.data.format_data import FormatData
from lib.rsa.rdms import GenerateRDMs
from lib.utils.plot import box_plot
import numpy as np
from lib.utils.new_categories import place_categoryids, face_imagenames, homonoid_monkey_imagenames, animate_inanimate
from lib.distance_measures.gaussian_kernel import gaussian_kernal_similarity
from lib.distance_measures.epanechnikov_kernel import epanechnikov_similarity
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
from sklearn.metrics.pairwise import cosine_similarity
import gc
import shutil


class DotProductAnalysis:
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
        model_name = path.split("/")[-1].upper()
        # print(val_list.shape, len(labels))

        fig, axs = plt.subplots(len(val_list), 1, figsize=(15, 15))
        fig.suptitle(model_name)
        for i, ax in enumerate(axs):
            ax.plot(labels, val_list[i])
            ax.set_title(title_list[i])
            ax.set_xticklabels([])
            ax.set_ylim(bottom=-1, top=1)
            # ax.set_ylabel()
            for x, y in zip(labels, val_list[i]):
                label = "{:.2f}".format(y)
                ax.annotate(label,  # this is the text
                            (x, y),  # this is the point to label
                            # textcoords="offset points",  # how to position the text
                            # distance from text to points (x,y)
                            # xytext=(0, 10),
                            ha='center')
        plt.xticks(ticks=np.arange(1, len(labels)+1),
                   labels=labels, rotation=45, ha="right")
        plt.savefig(path+".png")
        plt.close()
        colors = ["red", "blue", "green", "cyan", "purple", "yellow"]
        fig = plt.figure(figsize=(15, 10))
        lines = []
        for i, val in enumerate(val_list):
            line = plt.plot(
                labels, val_list[i], color=colors[i], label=title_list[i])
            plt.title(model_name)
            plt.xticks(ticks=np.arange(0, len(labels)),
                       labels=labels, rotation=45, ha="right")
            plt.ylim(bottom=-1, top=1)
            lines += [line]
        plt.legend()

        plt.savefig(path+"_combined.png")
        plt.close()

    def prod(self, val):
        res = 1
        for ele in val:
            res *= ele
        return res

    def _execute(self,  model, centre_crop, image_group_list, folder_names):
        for i, image_list in enumerate(image_group_list):
            feat = OrderedDict()
            image_count = 0
            for image in tqdm(image_list):
                # print(image)
                image_path = os.path.join(self.config.image_dir, image+".jpg")
                if os.path.exists(image_path):
                    image_count += 1
                    img = Image.open(image_path)
                    img = img.convert(mode="RGB")
                    input_img = Variable(centre_crop(img).unsqueeze(0))
                    if torch.cuda.is_available():
                        input_img = input_img.to(self.config.device)
                    x = model.forward(input_img)
                    for key, value in x.items():
                        if key not in feat.keys():
                            feat[key] = value.data.cpu().numpy()
                        else:
                            feat[key] = np.vstack(
                                (feat[key], value.data.cpu().numpy()))
            path = os.path.join(self.config.result_dir,
                                self.config.exp_id, "feats")
            if not os.path.exists(path):
                os.makedirs(path)
            sio.savemat(path + "/"+folder_names[i]+".mat", feat)

    def execute_model(self, model, model_name):
        if torch.cuda.is_available():
            model.to(self.config.device)
        model.eval()
        centre_crop = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image_group_list = [self.p_image_names, self.f_image_names,
                            self.plant_image_names, self.animal_image_names, self.inanimate_image_names]

        folder_names = ["place", "face", "plant", "animal", "objects"]
        self._execute(model, centre_crop,
                      image_group_list, folder_names)
        title_list = ["Place Images", "Face Images",
                      "Plant Images", "Animal Images", "Inanimate Images"]

        for measure in ["cosine", "pearson", "gaussian", "epanechnikov", "dotproduct"]:
            val_list = []
            for folder_name in folder_names:
                path = os.path.join(self.config.result_dir,
                                    self.config.exp_id, "feats", folder_name)
                feat = sio.loadmat(path+".mat")
                mean_values_per_feat = []
                labels = []
                for key, value in feat.items():
                    if key not in ['__header__', '__version__', '__globals__']:
                        labels += [key]
                        old_shape = feat[key].shape
                        new_shape = (old_shape[0], int(
                            self.prod(list(old_shape))/old_shape[0]))

                        old_val = feat[key].reshape(new_shape)

                        if measure == "cosine":
                            mean_values_per_feat += [np.mean(cosine_similarity(
                                old_val))]
                        elif measure == "pearson":
                            mean_values_per_feat += [np.mean(np.corrcoef(
                                old_val))]
                        elif measure == "gaussian":
                            mean_values_per_feat += [np.mean(gaussian_kernal_similarity(
                                old_val))]
                        elif measure == "epanechnikov":
                            mean_values_per_feat += [np.mean(epanechnikov_similarity(
                                old_val))]

                        elif measure == "dotproduct":
                            num_of_images = old_shape[0]
                            pairwise_dotproduct = np.zeros(
                                (num_of_images, num_of_images))
                            for k in range(num_of_images):
                                for j in range(num_of_images):
                                    pairwise_dotproduct[k][j] = np.dot(
                                        old_val[k]/np.linalg.norm(old_val[k]), old_val[j]/np.linalg.norm(old_val[j]))
                            mean_values_per_feat += [
                                np.mean(pairwise_dotproduct)]
                val_list += [mean_values_per_feat]

            plot_path = os.path.join(self.config.result_dir,
                                     self.config.exp_id, measure)
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            self.models_plot(val_list=val_list,
                             labels=labels,
                             title_list=title_list,
                             path=plot_path+"/"+model_name)
        model = None
        gc.collect()

    def get_model(self, model, load_model=None):
        return utils.get_model(self.config.arch)

    def run_human(self):
        # This is only done for the PRE-GOD data_type
        # The order of subjects in the data dictionary is dict_keys(['1', '3', '2', '5', '4'])
        data = FormatData(self.config, data_type="PRE-GOD").format()
        place_last_ind, face_last_ind, plant_last_ind, animal_last_ind = \
            data["place_lastind"], data["face_lastind"], data["plant_lastind"], data["animal_lastind"]
        del data["place_lastind"], data["face_lastind"], data["plant_lastind"], data["animal_lastind"]
        place_data, face_data, plant_data, animal_data, inanimate_data = [], [], [], [], []

        subjs = ["1", "2", "3", "4", "5"]
        roi_names = ["V1", "V2", "V3", "V4", "LOC",
                     "FFA", "PPA", "LVC", "HVC", "VC"]
        title_list = ["Place Images", "Face Images",
                      "Plant Images", "Animal Images", "Inanimate Images"]
        measures = ["dotproduct", "cosine", "pearson",
                    "gaussian", "epanechnikov", "mean"]
        for measure in tqdm(measures):
            all_subj_data = []
            for subj in subjs:
                vals_for_all_rois = []
                for roi in roi_names:
                    category_data = [
                        # place data
                        data[subj]["roi_data"][roi][:place_last_ind],
                        # face data
                        data[subj]["roi_data"][roi][place_last_ind:face_last_ind],
                        # plants data
                        data[subj]["roi_data"][roi][face_last_ind:plant_last_ind],
                        # animals data
                        data[subj]["roi_data"][roi][plant_last_ind:animal_last_ind],
                        # inanimate data
                        data[subj]["roi_data"][roi][animal_last_ind::],
                    ]
                    val_per_cat = []
                    if measure == "cosine":
                        for cat_data in category_data:
                            val_per_cat += [np.mean(cosine_similarity(cat_data))]
                            # print(val_per_cat, cosine_similarity(cat_data))

                    elif measure == "pearson":
                        for cat_data in category_data:
                            val_per_cat += [np.mean(np.corrcoef(cat_data))]

                    elif measure == "gaussian":
                        for cat_data in category_data:
                            val_per_cat += [
                                np.mean(gaussian_kernal_similarity(cat_data))]

                    elif measure == "epanechnikov":
                        for cat_data in category_data:
                            val_per_cat += [np.mean(epanechnikov_similarity(cat_data))]

                    elif measure == "dotproduct":
                        for cat_data in category_data:
                            num_of_images = cat_data.shape[0]
                            pairwise_dotproduct = np.zeros(
                                (num_of_images, num_of_images))
                            for k in range(num_of_images):
                                for j in range(num_of_images):
                                    pairwise_dotproduct[k][j] = np.dot(
                                        cat_data[k]/np.linalg.norm(cat_data[k]), cat_data[j]/np.linalg.norm(cat_data[j]))
                            val_per_cat += [
                                np.mean(pairwise_dotproduct)]
                    elif measure == "mean":
                        for cat_data in category_data:
                            val_per_cat += [np.mean(cat_data)]

                    vals_for_all_rois += [val_per_cat]

                all_subj_data += [vals_for_all_rois]
                plot_path = os.path.join(self.config.result_dir,
                                         self.config.exp_id, measure)
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path)
                self.models_plot(val_list=np.transpose(vals_for_all_rois),
                                 labels=roi_names,
                                 title_list=title_list,
                                 path=plot_path+"/"+subj)
            self.models_plot(val_list=np.transpose(np.mean(np.array(all_subj_data), axis=0)),
                             labels=roi_names,
                             title_list=title_list,
                             path=plot_path+"/"+"AVG")

    def run_model(self):
        for arch in self.config.archs:
            print("Model running:", arch)
            self.config.arch = arch
            model_name = arch

            model = self.get_model(model_name)
            self.execute_model(model, model_name)
        shutil.rmtree(os.path.join(self.config.result_dir,
                                   self.config.exp_id, "feats"))
        print("End")
