import itertools
import numpy as np
import glob
import os

clubed_categories = {
    "frog, toad": [2, 3],  # Animals
    "hummingbird, goose": [12, 13],  # Animals
    "dogs, greyhound": [22, 23],  # Animals
    "horse, zebra": [30, 31],  # Animals
    # "tomato, watermelon, grapes": [143, 145, 146],  # Food
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
    # "fruits": [143, 145, 146],
    "trees": [148, 149],
    "places": [47, 140, 70],
    "vehicles": [40, 78, 79, 84, 85, 122, 65, 112],
    "computer": [58, 87],
    "objects": [113, 127]
}

detailed_cat_list = {
    2: ["n01639765", "frog, toad, toad frog, anuran, batrachian, salientian", 5],
    3: ["n01645776", "true toad", 5],
    12: ["n01833805", "hummingbird", 7],
    13: ["n01855672", "goose", 7],
    22: ["n02084071", "dog, domestic dog, Canis familiaris", 6],
    23: ["n02090827", "greyhound", 6],
    30: ["n02374451", "horse, Equus caballus", 5],
    31: ["n02391049", "zebra", 7],
    40: ["n02692877", "airship, dirigible", 5],
    47: ["n02814860", "beacon, lighthouse, beacon light, pharos", 7],
    58: ["n03085013", "computer keyboard, keypad", 7],
    65: ["n03345487", "fire engine, fire truck", 6],
    70: ["n03425413", "gas pump, gasoline pump, petrol pump, island dispenser", 6],
    78: ["n03512147", "helicopter, chopper, whirlybird, eggbeater", 8],
    79: ["n03541923", "hot-air balloon", 6],
    84: ["n03609235", "kayak", 7],
    85: ["n03612010", "ketch", 6],
    87: ["n03642806", "laptop, laptop computer", 5],
    112: ["n04146614", "school bus", 6],
    113: ["n04154565", "screwdriver", 5],
    122: ["n04273569", "speedboat", 7],
    127: ["n04376876", "syringe", 5],
    140: ["n04587559", "windmill", 6],
    143: ["n07734017", "tomato", 6],
    145: ["n07756951", "watermelon", 7],
    146: ["n07758680", "grape", 5],
    148: ["n12582231", "palm, palm tree", 5],
    149: ["n12596148", "miniature fan palm, bamboo palm, fern rhapis, Rhapis excelsa", 7]
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


# Place and Face images

place_categoryids = ["1", "32", "47", "99", "106", "118", "129", "132", "140"]
# All Imagenames of 35, 36, 37 categories

face_imagenames = [
    "n02472293_11152",
    "n02472293_131",
    "n02472293_13453",
    "n02472293_24267",
    "n02472293_34273",
    "n02472293_3508",
    "n02472293_46249",
    "n02472293_5718",
    "n02766534_18271",
    "n02766534_15976",
    "n02799175_9738",
    "n02799175_11275",
    "n02799175_15265",
    "n02799175_16780",
    "n02802215_26105",
    "n03085219_3690",
    "n03394916_30808",
    "n03394916_33757",
    "n03394916_36485",
    "n03394916_41444",
    "n03397947_4284",
    "n03397947_7453",
    "n03397947_10813",
    "n03494278_29023",
    "n03494278_30876",
    "n03494278_35790",
    "n03496296_14487",
    "n03543603_6861",
    "n03543603_8721",
    "n03607659_1968",
    "n03607659_5173",
    "n03609235_1655",
    "n03609235_3138",
    "n03609235_3801",
    "n03609235_10979",
    "n03721384_5136",
    "n03721384_14643",
    "n03721384_22007",
    "n03721384_36929",
    "n03743279_4728",
    "n03743279_5689",
    "n03743279_8976",
    "n03743279_11113",
    "n03743279_13788",
    "n03743279_15645",
    "n03746005_6174",
    "n03792782_16817",
    "n03924679_3671",
    "n03982430_21511",
    "n03982430_37727",
    "n04168199_9862",
    "n04168199_10150",
    "n04168199_10620",
    "n04168199_10897",
    "n04225987_6665",
    "n04225987_6877",
    "n04272054_48893",
    "n04272054_56965",
    "n04409806_7672",
    "n04409806_8542",
]

homonoid_monkey_imagenames = ["n02480855_11704", "n02480855_13432", "n02480855_14628", "n02480855_16790"
                              "n02480855_18583", "n02480855_20865", "n02480855_8638", "n02480855_9990"
                              "n02481823_12066", "n02481823_13867", "n02481823_15941", "n02481823_18421"
                              "n02481823_20882", "n02481823_23004", "n02481823_5715", "n02481823_8477"
                              ]


animate_inanimate = {
    "plants": [
        "143",
        "144",
        "145",
        "146",
        "147",
        "148",
        "149",
        "150"
    ],
    "animals": ["1",
                "2",
                "3",
                "4",
                "6",
                "13",
                "14",
                "18"
                ],
    "inanimate": [
        "81",
        "82",
        "85",
        "86",
        "87",
        "88",
        "90",
        "105"
    ]

}
