import itertools
import numpy as np
import glob
import os

clubed_categories = {
    "frog, toad": [2, 3],  # Animals
    "hummingbird, goose": [12, 13],  # Animals
    "dogs, greyhound": [22, 23],  # Animals
    "horse, zebra": [30, 31],  # Animals
    "tomato, watermelon, grapes": [143, 145, 146],  # Food
    "palm, trees": [148, 149],  # Trees/Greenery
    "lighthouse, windmill, gas pump": [47, 140, 70],  # Places for PPA
    "airship, helicopter, hot-air balloon": [40, 78, 79],  # Air vehicles
    "kayak, ketch, speedboat": [84, 85, 122],  # Water vehicles
    "fire engine, school bus": [65, 112],  # Land vehicles
    "keyboard, laptop": [58, 87],  # computer
    "screwdriver, syringe": [113, 127]  # similar shaped objects
}

super_categories = {
    "animals": [2, 3, 12, 13, 22, 23, 30, 31],
    "fruits": [143, 145, 146],
    "trees": [148, 149],
    "places": [47, 140, 70],
    "vehicles": [40, 78, 79, 84, 85, 122, 65, 112],
    "computer": [58, 87],
    "objects": [113, 127]
}

detailed_cat_list = {
    2: ['n01639765', 'frog, toad, toad frog, anuran, batrachian, salientian', 5],
    3: ['n01645776', 'true toad', 5],
    12: ['n01833805', 'hummingbird', 7],
    13: ['n01855672', 'goose', 7],
    22: ['n02084071', 'dog, domestic dog, Canis familiaris', 6],
    23: ['n02090827', 'greyhound', 6],
    30: ['n02374451', 'horse, Equus caballus', 5],
    31: ['n02391049', 'zebra', 7],
    40: ['n02692877', 'airship, dirigible', 5],
    47: ['n02814860', 'beacon, lighthouse, beacon light, pharos', 7],
    58: ['n03085013', 'computer keyboard, keypad', 7],
    65: ['n03345487', 'fire engine, fire truck', 6],
    70: ['n03425413', 'gas pump, gasoline pump, petrol pump, island dispenser', 6],
    78: ['n03512147', 'helicopter, chopper, whirlybird, eggbeater', 8],
    79: ['n03541923', 'hot-air balloon', 6],
    84: ['n03609235', 'kayak', 7],
    85: ['n03612010', 'ketch', 6],
    87: ['n03642806', 'laptop, laptop computer', 5],
    112: ['n04146614', 'school bus', 6],
    113: ['n04154565', 'screwdriver', 5],
    122: ['n04273569', 'speedboat', 7],
    127: ['n04376876', 'syringe', 5],
    140: ['n04587559', 'windmill', 6],
    143: ['n07734017', 'tomato', 6],
    145: ['n07756951', 'watermelon', 7],
    146: ['n07758680', 'grape', 5],
    148: ['n12582231', 'palm, palm tree', 5],
    149: ['n12596148', 'miniature fan palm, bamboo palm, fern rhapis, Rhapis excelsa', 7]
}


def get_new_category_list():
    new_list = list(clubed_categories.values())
    return list(itertools.chain.from_iterable(new_list))


def get_ind_tuples():
    start = 0
    cats = get_new_category_list()
    start_end_ind = []
    for cat in cats:
        end = start+detailed_cat_list[cat][2]
        start_end_ind = start_end_ind + \
            [(start, end)]
        start = end
    return start_end_ind


def get_sup_cat_ind_tuples():
    start = 0
    start_end_ind = []
    for cat in super_categories.keys():
        end = start+len(super_categories[cat])
        start_end_ind = start_end_ind + \
            [(start, end)]
        start = end
    return start_end_ind


def get_new_cat_image_names():  # Returns image names of the new categories under consideration
    data_dir = "../data/images/train_images"
    cat_list = get_new_category_list()
    image_name_list = []
    for cat in cat_list:
        img_list = glob.glob(os.path.join(data_dir, str(cat), "*.jpg"))
        image_names = [image_addr.split("/")[-1].split(".")[0]
                       for image_addr in img_list]
        image_names.sort()
        image_name_list = image_name_list + [image_names]
    return image_name_list


get_sup_cat_ind_tuples()
