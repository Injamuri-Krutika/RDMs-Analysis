import glob
import os
from tabulate import tabulate
import numpy as np
num_of_catg = 150
count_map = {}
img_dir = "../../../data/images/train_images"

for cat in range(1, num_of_catg+1):
    img_list = glob.glob(os.path.join(img_dir, str(cat), "*.jpg"))
    count_map[cat] = len(img_list)


with open("../../../data/mapped_stimulus_ImageNetTraining.tsv") as mapping_file:
    mapping_file = mapping_file.read()
    category_rows = mapping_file.split("\n")
    category_map = {}
    for i, row in enumerate(category_rows):
        # category_map has label, category name and number of images in that category
        category_map[i+1] = [row.split("\t")[0],
                             row.split("\t")[1], count_map[i+1]]

    for cat, value in category_map.items():
        if category_map[cat][2] >= 5:
            print(cat, category_map[cat])
