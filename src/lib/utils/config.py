import argparse
import os
import shutil


class Config(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic experiment setting

        self.parser.add_argument(
            '--sigma_g', type=float, default=1.0, help='Bandwidth for Gaussian Kernel')

        self.parser.add_argument(
            '--sigma_ep', type=float, default=1.0, help='Bandwidth for Epanechnikov Kernel')

        self.parser.add_argument('--data_dir', default='/home/krutika/Data/ds001246-download/',
                                 help='GOD Dataset Directory')

        self.parser.add_argument('--exp_id', default='default')

    def diectory_check(self, opt):
        path = os.path.join("../results", opt.exp_id)
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print(
                f'\'{opt.exp_id}\' Directory already exists. Do you want to REPLACE it? Y/n')
            inp = input()
            if inp == "" or inp == "y" or inp == "Y":
                shutil.rmtree(path)
                os.makedirs(path)
            else:
                print("Enter the value of new exp_id.\n")
                opt.exp_id = input()
                self.diectory_check(opt)

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.final_data_dir = os.path.join(opt.data_dir, "final_data")
        self.diectory_check(opt)
        return opt

    def init(self, args=''):
        opt = self.parse(args)
        return opt
