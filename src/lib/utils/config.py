import argparse
import os
import shutil
import numpy as np
from lib.utils.new_categories import get_new_category_list


class Config(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic experiment setting

        self.parser.add_argument(
            '--sigma_g', type=float, default=1.0, help='Bandwidth for Gaussian Kernel')

        self.parser.add_argument(
            '--sigma_ep', type=float, default=1.0, help='Bandwidth for Epanechnikov Kernel')

        self.parser.add_argument('--data_dir', default='../data/',
                                 help='GOD Dataset Directory')
        self.parser.add_argument('--result_dir', default='../results',
                                 help='GOD Dataset Directory')

        self.parser.add_argument('--exp_id', default='default')
        self.parser.add_argument('--subset_data', action="store_true",
                                 help="Analyse the RDMs only for a specified subset of data. Subset is defined by --category_ids argument.")
        self.parser.add_argument('--category_ids', default=None, type=str,
                                 help="Must be comma seperated. Values can be between 1 to 150"
                                 )
        self.parser.add_argument('--ep', action="store_true",
                                 help="It will consider epanechnicov instead of gaussian kernel."
                                 )

        self.parser.add_argument('--corr', action="store_true",
                                 help="It will calculate the correlation of RDMS between subjects for all 1200 images."
                                 )

        self.parser.add_argument('--categorise', action="store_true",
                                 help="It will create RDMs and analyse only for 2, 3, 12, 13, 22, 23, 30, 31, 143, 145, 146, 148, 149, 47, 140, 70, 40, 78, 79, 84, 85, 122, 65, 112, 58, 87, 113, 127 categories"
                                 )

    def diectory_check(self, opt):
        path = os.path.join(opt.result_dir, opt.exp_id)
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print(
                f'\'{opt.exp_id}\' Directory already exists. Do you want to REPLACE it? y/n')
            inp = input()
            if inp == "y" or inp == "Y":
                shutil.rmtree(path)
                os.makedirs(path)
            elif inp == "n" or inp == "N":
                print("Enter the value of new exp_id.\n")
                opt.exp_id = input()
                self.diectory_check(opt)
            else:
                self.diectory_check(opt)

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.final_data_dir = os.path.join(opt.data_dir, "final_data")
        opt.num_of_subjs = 5
        if opt.ep:
            opt.distance_measures = ["pearson", "epanechnicov"]
        else:
            opt.distance_measures = ["pearson", "kernel"]

        if opt.categorise:
            opt.category_ids = get_new_category_list()
            opt.subset_data = True
        elif opt.category_ids:
            opt.category_ids = np.array(opt.category_ids.split(","))

        self.diectory_check(opt)

        return opt

    def init(self, args=''):
        opt = self.parse(args)
        return opt
