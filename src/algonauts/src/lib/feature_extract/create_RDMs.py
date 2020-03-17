# This script generates RDMs from the activations of a DNN
# Input
#   --feat_dir : directory that contains the activations generated using generate_features.py
#   --save_dir : directory to save the computed RDM
#   --dist : dist used for computing RDM (e.g. 1-Pearson's R)
#   Note: If you have generated activations of your models using your own code, please replace
#      -   "get_layers_ncondns" to get num of layers, layers list and num of images, and
#      -   "get_features" functions to get activations of a particular layer (layer) for a particular image (i).
# Output
#   Model RDM for the representative layers of the DNN.
#   The output RDM is saved in two files submit_MEG.mat and submit_fMRI.mat in a subdirectory named as layer name.

import json
import glob
import os
import numpy as np
import datetime
import scipy.io as sio
import argparse
import zipfile
from tqdm import tqdm
import scipy
import torch.nn.functional as F
import torch
from lib.utils.new_categories import get_new_cat_image_names
import itertools
# from utils import zip


class CreateRDMs():
    def __init__(self, config):
        self.config = config

    def get_layers_ncondns(self, feat_dir):
        """
        to get number of representative layers in the DNN,
        and number of images(conditions).
        Input:
        feat_dir: Directory containing activations generated using generate_features.py
        Output:
        num_layers: number of layers for which activations were generated
        num_condns: number of stimulus images
        PS: This function is specific for activations generated using generate_features.py
        Write your own function in case you use different script to generate features.
        """
        activations = glob.glob(feat_dir + "/*" + ".mat")
        num_condns = len(activations)
        feat = sio.loadmat(activations[0])
        num_layers = 0
        layer_list = []
        for key in feat:
            if "__" in key:
                continue
            else:
                num_layers += 1
                layer_list.append(key)
        return num_layers, layer_list, num_condns

    def get_features(self, feat_dir, layer_id, i):
        """
        to get activations of a particular DNN layer for a particular image

        Input:
        feat_dir: Directory containing activations generated using generate_features.py
        layer_id: layer name
        i: image index

        Output:
        flattened activations

        PS: This function is specific for activations generated using generate_features.py
        Write your own function in case you use different script to generate features.
        """
        # activations = glob.glob(feat_dir + "/*.mat")

        # activations.sort()
        names = list(itertools.chain.from_iterable(get_new_cat_image_names()))
        activations = [feat_dir+"/"+name+".mat" for name in names]
        feat = sio.loadmat(activations[i])[layer_id]
        return feat.ravel()

    def create_rdm(self, save_dir, feat_dir):
        """
        Main function to create RDM from activations
        Input:
        feat_dir: Directory containing activations generated using generate_features.py
        save_dir : directory to save the computed RDM
        dist : dist used for computing RDM (e.g. 1-Pearson's R)

        Output (in submission format):
        The model RDMs for each layer are saved in
            save_dir/layer_name/submit_fMRI.mat to compare with fMRI RDMs
            save_dir/layer_name/submit_MEG.mat to compare with MEG RDMs
        """

        # get number of layers and number of conditions(images) for RDM
        num_layers, layer_list, num_condns = self.get_layers_ncondns(feat_dir)
        # print(num_layers, layer_list, num_condns)
        cwd = os.getcwd()

        # loops over layers and create RDM for each layer
        for layer in range(num_layers):
            os.chdir(cwd)
            # RDM is num_condnsxnum_condns matrix, initialized with zeros
            RDM1 = np.zeros((num_condns, num_condns))
            RDM2 = np.zeros((num_condns, num_condns))

            # save path for RDMs in  challenge submission format
            layer_id = layer_list[layer]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            RDM_filename_fmri = os.path.join(save_dir, layer_id+'.mat')
            # RDM loop
            for i in tqdm(range(num_condns)):
                for j in tqdm(range(num_condns)):
                    # get feature for image index i and j
                    feature_i = self.get_features(feat_dir, layer_id, i)
                    feature_j = self.get_features(feat_dir, layer_id, j)
                    # compute distance 1-Pearson's R
                    RDM1[i, j] = 1-np.corrcoef(feature_i, feature_j)[0][1]
            for i in tqdm(range(num_condns)):
                for j in tqdm(range(num_condns)):
                    f_i = feature_i / (feature_i*feature_i).sum()**0.5
                    f_j = feature_j / (feature_j*feature_j).sum()**0.5
                    dist = torch.dist(
                        torch.tensor(f_i.reshape(-1, 1)), torch.tensor(f_j.reshape(-1, 1)))
                    RDM2[i, j] = 1-scipy.exp(- dist**2 / 2).item()

            # saving RDMs in challenge submission format
            rdm_fmri = {}
            rdm_fmri["pearson"] = RDM1
            rdm_fmri["kernel"] = RDM2
            print(RDM_filename_fmri)
            sio.savemat(RDM_filename_fmri, rdm_fmri)

    def run(self):
        if self.config.pretrained:
            for arch in self.config.archs:
                print("Model running:", arch)
                self.config.arch = arch
                model_name = arch
                feats_dir = os.path.join(
                    self.config.feat_dir, model_name, "feats")
                for subdir, dirs, files in os.walk(feats_dir):
                    if len(dirs) == 0 and len(files) != 0:
                        save_dir = os.path.join(
                            self.config.rdms_dir, model_name, "rdms")
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        self.create_rdm(save_dir, subdir)
            return
