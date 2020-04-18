import os
from lib.utils.config import Config
from lib.utils.plot import plot_corr, plot_rdm_group
from lib.data.format_data import FormatData
from lib.rsa.rdms import GenerateRDMs
from lib.utils.corr import get_corr
from lib.utils.new_categories import get_new_category_list, get_new_cat_image_names, super_categories, clubed_categories, super_categories
from algonauts.src.lib.feature_extract.generate_features import GenerateFeatures
from algonauts.src.lib.feature_extract.create_RDMs import CreateRDMs
from lib.analysis_for_boxplot import BoxPlotting
from lib.hierarchical_clustering import HierarchicalClustering
from lib.animate_inanimate import AnimateInanimate
from lib.dot_product_analysis import DotProductAnalysis
from tqdm import tqdm
import numpy as np
import sys
import glob
import scipy.io as io
from lib.utils.corr import get_corr
import pandas as pd

sys.setrecursionlimit(15000)


def analyse_RDMs(config):
    for data_type in tqdm(["GOD", "PRE-GOD"]):
        data = FormatData(config, data_type=data_type).format()
        print(data[list(data.keys())[0]]["image_names"])

        base_path = os.path.join(config.result_dir, config.exp_id, data_type)
        for subj in tqdm(data.keys()):
            for roi in data[subj]["roi_names"]:
                if data_type == "GOD":
                    for stat in data[subj]["roi_data"][roi].keys():
                        path = os.path.join(base_path, subj, stat)
                        if not os.path.exists(path):
                            os.makedirs(path)
                        GenerateRDMs(config).rdm(
                            data[subj]["roi_data"][roi][stat], roi, path, data[subj]["category_names"])

                else:
                    path = os.path.join(base_path, subj)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    GenerateRDMs(config).rdm(
                        data[subj]["roi_data"][roi], roi, path, data[subj]["category_names"])

        calculate_avg_rdm_plot(data_type, base_path, config,
                               data[list(data.keys())[0]]["category_names"])


def calculate_avg_rdm_plot(data_type, base_path, config, labels):
    data = FormatData(config, data_type=data_type).format()

    avg_g, avg_pg = {}, {}
    subj_names = list(data.keys())
    subj1_data = data[subj_names[0]]
    if data_type == "PRE-GOD":
        avg_dir = os.path.join(base_path, "avg")
        if not os.path.exists(avg_dir):
            os.makedirs(avg_dir)
        for roi in subj1_data["roi_names"]:
            avg_pg = {}

            for subj in subj_names:
                path = os.path.join(base_path, subj)
                subj_roi_data = io.loadmat(path+"/"+roi+".mat")
                if avg_pg == {}:
                    avg_pg = {
                        "kernel": np.zeros((5,)+(subj_roi_data["kernel"].shape)),
                        "pearson": np.zeros((5,)+(subj_roi_data["pearson"].shape)),
                    }
                for distance_measure in config.distance_measures:
                    avg_pg[distance_measure][int(
                        subj)-1] = subj_roi_data[distance_measure]
            for distance_measure in config.distance_measures:
                avg_pg[distance_measure] = np.mean(
                    avg_pg[distance_measure], axis=0)
            plot_rdm_group(
                avg_pg, roi, config.distance_measures, avg_dir, labels)
    elif data_type == "GOD":
        for stat in subj1_data["roi_data"][subj1_data["roi_names"][0]].keys():
            avg_dir = os.path.join(base_path, "avg", stat)
            if not os.path.exists(avg_dir):
                os.makedirs(avg_dir)
            for roi in subj1_data["roi_names"]:
                avg_g = {}
                for subj in subj_names:
                    path = os.path.join(base_path, subj, stat)
                    subj_roi_data = io.loadmat(path+"/"+roi+".mat")
                    if avg_g == {}:
                        avg_g = {
                            "kernel": np.zeros((5,)+(subj_roi_data["kernel"].shape)),
                            "pearson": np.zeros((5,)+(subj_roi_data["pearson"].shape)),
                        }
                    for distance_measure in config.distance_measures:
                        avg_g[distance_measure][int(
                            subj)-1] = subj_roi_data[distance_measure]
                for distance_measure in config.distance_measures:
                    avg_g[distance_measure] = np.mean(
                        avg_g[distance_measure], axis=0)
                plot_rdm_group(
                    avg_g, roi, config.distance_measures, avg_dir, labels)


def find_corr(config):
    for data_type in ["GOD", "PRE-GOD"]:
        with open("../results/"+data_type+".txt", "w") as corr_file:
            data = FormatData(config, data_type=data_type,
                              ).format()
            roi_names = data["1"]["roi_names"]
            for roi in roi_names:
                for i in range(1, config.num_of_subjs+1):
                    for j in range(i, config.num_of_subjs+1):
                        rdm1 = GenerateRDMs(config).rdm(
                            data[str(i)]["roi_data"][roi])
                        rdm2 = GenerateRDMs(config).rdm(
                            data[str(j)]["roi_data"][roi])
                        corr = get_corr(
                            rdm1["pearson"], rdm2["pearson"])
                        print(f'{i}, {j} : {corr}')


def get_required_ind(data, subj):
    new_cat_imagenames = get_new_cat_image_names()
    required_ind = []
    for i in range(len(new_cat_imagenames)):
        required_ind = required_ind + list(np.where(np.in1d(data[subj]["image_names"], np.array(
            new_cat_imagenames[i])) == True)[0])
    return required_ind


def analyse_categorised_RDMs(config):
    for data_type in tqdm(["GOD", "PRE-GOD"]):
        data = FormatData(config, data_type=data_type,
                          ).format()
        for subj in tqdm(data.keys()):
            for roi in data[subj]["roi_names"]:
                required_ind = get_required_ind(data, subj)
                if data_type == "GOD":
                    for stat in data[subj]["roi_data"][roi].keys():
                        path = os.path.join(
                            config.result_dir, "categorised", config.exp_id, data_type, subj, stat)
                        if not os.path.exists(path):
                            os.makedirs(path)

                        GenerateRDMs(config).rdm(
                            data[subj]["roi_data"][roi]
                            [stat][required_ind], roi, path, data[subj]["category_names"][required_ind])

                else:
                    path = os.path.join(
                        config.result_dir, "categorised",  config.exp_id, data_type, subj)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    GenerateRDMs(config).rdm(
                        data[subj]["roi_data"][roi][required_ind], roi, path, data[subj]["category_names"][required_ind])


def organise(config):
    for arch in config.archs:
        path = os.path.join(
            config.result_dir, "categorised", config.exp_id, arch, "rdms")
        data = FormatData(config, data_type="GOD").format()
        subj = list(data.keys())[0]
        required_ind = get_required_ind(data, subj)
        labels = data[subj]["category_names"][required_ind]
        # if not os.path.exists(path):
        kernel_path = os.path.join(path, "kernel")
        rdm_paths = glob.glob(kernel_path+"/**/*.mat")
        for kernel_path in rdm_paths:
            rdms = {}
            pearson_path = kernel_path.replace("kernel", "pearson")
            rdms["kernel"] = io.loadmat(kernel_path)["RDM"]

            rdms["pearson"] = io.loadmat(pearson_path)["RDM"]
            layer_id = kernel_path.split("/")[-2]
            io.savemat(path+"/"+layer_id+".mat", rdms)
        rdm_paths = glob.glob(path + "/*.mat")
        cat_path = os.path.join(path, "categorised")
        sup_cat_path = os.path.join(path, "super_categorised")
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
        if not os.path.exists(sup_cat_path):
            os.makedirs(sup_cat_path)

        for rdm_path in rdm_paths:
            rdm = io.loadmat(rdm_path)
            categorised_rdm, new_labels = GenerateRDMs(
                config).categorise_rdms(rdm, labels)
            io.savemat(cat_path+"/" + rdm_path.split("/")[-1], categorised_rdm)
            sup_rdm = GenerateRDMs(config).super_categorise_rdms(rdm)
            io.savemat(sup_cat_path+"/"+rdm_path.split("/")[-1], sup_rdm)


def avg_human_rdms(config):
    num_cat = len(get_new_cat_image_names())
    num_sup_cat = len(list(super_categories.keys()))
    actual_rdms = {
        "kernel": np.zeros((5, 152, 152)),
        "pearson": np.zeros((5, 152, 152))
    }
    cat_rdms = {
        "kernel": np.zeros((5, num_cat, num_cat)),
        "pearson": np.zeros((5, num_cat, num_cat))
    }
    sup_cat_rdms = {
        "kernel": np.zeros((5, num_sup_cat, num_sup_cat)),
        "pearson": np.zeros((5, num_sup_cat, num_sup_cat))
    }
    avg_actual_rdms = {}
    avg_cat_rdms = {}
    avg_sup_cat_rdms = {}

    data_type = "PRE-GOD"
    data = FormatData(config, data_type=data_type,
                      ).format()
    subj = list(data.keys())[0]
    labels = data[list(data.keys())[0]]["category_names"]
    clubed_labels = ",".join(list(clubed_categories.keys())).split(",")
    actual_avg_path = os.path.join(config.result_dir, "categorised",
                                   config.exp_id, data_type, "avg")
    cat_avg_path = os.path.join(actual_avg_path, "categorised")
    if not os.path.exists(cat_avg_path):
        os.makedirs(cat_avg_path)

    sup_cat_avg_path = os.path.join(actual_avg_path, "super_categorised")
    if not os.path.exists(sup_cat_avg_path):
        os.makedirs(sup_cat_avg_path)

    for roi in data[subj]["roi_names"]:
        for subj in range(1, 6):
            path = os.path.join(config.result_dir, "categorised",
                                config.exp_id, data_type, str(subj))
            cat_path = os.path.join(path, "categorised")
            sup_cat_path = os.path.join(path, "super_categorised")

            for distance_measure in config.distance_measures:
                rdms = io.loadmat(path+"/"+roi+".mat")
                actual_rdms[distance_measure][subj-1] = rdms[distance_measure]
                rdms = io.loadmat(cat_path+"/"+roi+".mat")
                cat_rdms[distance_measure][subj-1] = rdms[distance_measure]
                rdms = io.loadmat(sup_cat_path+"/"+roi+".mat")
                sup_cat_rdms[distance_measure][subj-1] = rdms[distance_measure]

        for distance_measure in config.distance_measures:
            avg_actual_rdms[distance_measure] = np.mean(
                actual_rdms[distance_measure], axis=0)
            avg_cat_rdms[distance_measure] = np.mean(
                cat_rdms[distance_measure], axis=0)
            avg_sup_cat_rdms[distance_measure] = np.mean(
                sup_cat_rdms[distance_measure], axis=0)
        io.savemat(os.path.join(actual_avg_path, roi+".mat"), avg_actual_rdms)
        io.savemat(os.path.join(cat_avg_path, roi+".mat"), avg_cat_rdms)
        io.savemat(os.path.join(sup_cat_avg_path,
                                roi+".mat"), avg_sup_cat_rdms)
        plot_rdm_group(
            avg_actual_rdms, roi, config.distance_measures, actual_avg_path, labels)
        plot_rdm_group(
            avg_cat_rdms, roi, config.distance_measures, cat_avg_path, clubed_labels)
        plot_rdm_group(
            avg_sup_cat_rdms, roi, config.distance_measures, sup_cat_avg_path, list(super_categories.keys()))

    # data_type = "GOD"
    # data = FormatData(config, data_type=data_type,
    #                   ).format()
    # subj = list(data.keys())[0]
    # for roi in data[subj]["roi_names"]:
    #     for stat in data[subj]["roi_data"][roi].keys():
    #         for subj in data.keys():
    #             path = os.path.join(config.result_dir, "categorised",
    #                                 config.exp_id, data_type, str(subj), stat)
    #             avg_path = os.path.join(config.result_dir, "categorised",
    #                                     config.exp_id, data_type, "avg")
    #             actual_avg_path = os.path.join(avg_path, stat)
    #             if not os.path.exists(actual_avg_path):
    #                 os.makedirs(actual_avg_path)
    #             cat_avg_path = os.path.join(avg_path, "categorised", stat)
    #             if not os.path.exists(cat_avg_path):
    #                 os.makedirs(cat_avg_path)

    #             sup_cat_avg_path = os.path.join(
    #                 avg_path, "super_categorised", stat)
    #             if not os.path.exists(sup_cat_avg_path):
    #                 os.makedirs(sup_cat_avg_path)

    #             cat_path = os.path.join(path, "categorised")
    #             sup_cat_path = os.path.join(path, "super_categorised")

    #             for distance_measure in config.distance_measures:
    #                 rdms = io.loadmat(path+"/"+roi+".mat")
    #                 actual_rdms[distance_measure][int(subj) -
    #                                               1] = rdms[distance_measure]
    #                 rdms = io.loadmat(cat_path+"/"+roi+".mat")
    #                 cat_rdms[distance_measure][int(
    #                     subj)-1] = rdms[distance_measure]
    #                 rdms = io.loadmat(sup_cat_path+"/"+roi+".mat")
    #                 sup_cat_rdms[distance_measure][int(subj) -
    #                                                1] = rdms[distance_measure]

    #         for distance_measure in config.distance_measures:
    #             avg_actual_rdms[distance_measure] = np.mean(
    #                 actual_rdms[distance_measure], axis=0)
    #             avg_cat_rdms[distance_measure] = np.mean(
    #                 cat_rdms[distance_measure], axis=0)
    #             avg_sup_cat_rdms[distance_measure] = np.mean(
    #                 sup_cat_rdms[distance_measure], axis=0)
    #         io.savemat(os.path.join(actual_avg_path,
    #                                 roi+".mat"), avg_actual_rdms)
    #         io.savemat(os.path.join(cat_avg_path, roi+".mat"), avg_cat_rdms)
    #         io.savemat(os.path.join(sup_cat_avg_path,
    #                                 roi+".mat"), avg_sup_cat_rdms)
    #         plot_rdm_group(
    #             avg_actual_rdms, roi, config.distance_measures, actual_avg_path, labels)
    #         plot_rdm_group(
    #             avg_cat_rdms, roi, config.distance_measures, cat_avg_path,  clubed_labels)
    #         plot_rdm_group(
    #             avg_sup_cat_rdms, roi, config.distance_measures, sup_cat_avg_path,  list(super_categories.keys()))


def write_to_excel(arch, data, coloumns, sheetname):
    print(sheetname)
    writer = pd.ExcelWriter(arch+'.xlsx', engine='xlsxwriter')
    df = pd.DataFrame(data, columns=coloumns)
    df.to_excel(writer, sheet_name=sheetname, index=False)
    writer.save()
    writer.close()


def compare(config):
    g_coloumns = ["distance", "stat", "layer_id",  "roi", "corr"]
    pg_coloumns = ["distance", "layer_id", "roi", "corr"]

    data_type = "GOD"
    data = FormatData(config, data_type=data_type,
                      ).format()
    subj = list(data.keys())[0]

    for roi in data[subj]["roi_names"]:
        for stat in data[subj]["roi_data"][roi].keys():
            for subj in data.keys():
                path = os.path.join(config.result_dir, "categorised",
                                    config.exp_id, data_type, str(subj), stat)
    exp_id_path = os.path.join(config.result_dir, "categorised",
                               config.exp_id)
    for arch in config.archs:
        for val in ["", "categorised", "super_categorised"]:
            pg_data, g_data = [], []
            for distance_measure in config.distance_measures:
                model_rdms_path = os.path.join(exp_id_path, arch, "rdms", val)
                pre_god_path = os.path.join(
                    exp_id_path, "PRE-GOD",  "avg", val)
                god_rdms_path = os.path.join(exp_id_path, "GOD", "avg", val)

                rdm_mats_paths = glob.glob(model_rdms_path+"/*.mat")
                pre_god_rdm_paths = glob.glob(pre_god_path+"/*.mat")
                rdm_mats_paths.sort()
                # print("Val(PRE-GOD)", "distance",
                #       "layer_id", "stat", "roi", "corr", sep=",\t\t")
                pg_maxcorr_data = {}  # Contains the max corr for each DNN layer for PRE-GOD data
                for rdm_file_path in rdm_mats_paths:
                    layer_id = rdm_file_path.split("/")[-1].split(".")[0]
                    pg_maxcorr_data[layer_id] = {}
                    # max_corr = -999
                    # brain_layer = ""
                    # PRE-GOD
                    for pre_god_rdm in pre_god_rdm_paths:
                        roi = pre_god_rdm.split("/")[-1].split(".")[0]
                        model_rdm = io.loadmat(rdm_file_path)
                        human_rdm = io.loadmat(pre_god_rdm)
                        corr = get_corr(
                            model_rdm[distance_measure], human_rdm[distance_measure])
                        pg_data = pg_data + \
                            [[distance_measure, layer_id, roi, corr]]
                        # print(val, distance_measure, layer_id,
                        #       "N/A", roi, corr, sep=",")
                        pg_maxcorr_data[layer_id][roi] = corr

                    file_name = "_".join(["PRE-GOD", layer_id])+".png"
                    path = os.path.join(exp_id_path, arch,
                                        "graphs", distance_measure, val, "PRE-GOD")
                    if not os.path.exists(path):
                        os.makedirs(path)
                    plot_corr(pg_maxcorr_data[layer_id],
                              path+"/"+file_name, layer_id)

                # print("*"*30)
                # print("*"*30)
                # print("*"*30)
                subj = list(data.keys())[0]
                g_maxcorr_data = {}  # Contains the max corr for each DNN layer for GOD data

                # print("Val(GOD)", "distance", "layer_id",
                #       "stat", "roi", "corr", sep=",\t\t")

                for stat in ["cope", "pe", "tstat"]:
                    g_data = []
                    g_maxcorr_data[stat] = {}
                    for rdm_file_path in rdm_mats_paths:
                        layer_id = rdm_file_path.split(
                            "/")[-1].split(".")[0]
                        g_maxcorr_data[stat][layer_id] = {}

                        god_rdms_path_stat = os.path.join(god_rdms_path, stat)
                        god_rdm_paths = glob.glob(god_rdms_path_stat+"/*.mat")
                        for god_rdm in god_rdm_paths:
                            roi = god_rdm.split("/")[-1].split(".")[0]
                            model_rdm = io.loadmat(rdm_file_path)
                            human_rdm = io.loadmat(god_rdm)
                            corr = get_corr(
                                model_rdm[distance_measure], human_rdm[distance_measure])
                            g_data = g_data + \
                                [[distance_measure, stat, layer_id, roi, corr]]
                            # if corr > max_corr:
                            # max_corr = corr
                            # brain_layer = roi
                            # print(val, distance_measure, layer_id, stat, roi,
                            #       corr, sep=",")
                            g_maxcorr_data[stat][layer_id][roi] = corr
                            file_name = "_".join(
                                ["GOD", stat, layer_id])+".png"
                            path = os.path.join(
                                exp_id_path, arch, "graphs", distance_measure, val, "GOD", stat)
                            if not os.path.exists(path):
                                os.makedirs(path)
                        plot_corr(g_maxcorr_data[stat]
                                  [layer_id], path+"/"+file_name, layer_id, stat)
                    write_to_excel(arch, g_data, g_coloumns,
                                   "GOD_"+val+"_"+stat)

            write_to_excel(arch, pg_data, pg_coloumns, "PRE-GOD"+val)
            # print("*"*30)
            # print("*"*30)
            # print("*"*30)


def main(config):
    if config.corr:
        find_corr(config)

    elif config.categorise:
        analyse_categorised_RDMs(config)
        avg_human_rdms(config)
    elif config.models:
        config.pretrained = True
        config.categorise = True
        GenerateFeatures(config).run()
        CreateRDMs(config).run()
        organise(config)
        compare(config)
    elif config.box_plot:
        BoxPlotting(config).run()
    elif config.hierarchical_clustering:
        HierarchicalClustering(config).run()
    elif config.animate_inanimate:
        AnimateInanimate(config).run()
    elif config.models_box_plot:
        BoxPlotting(config).run_model()
    elif config.dot_product_analysis:
        DotProductAnalysis(config).run_model()
        DotProductAnalysis(config).run_human()

    else:
        print("Hiii")
        analyse_RDMs(config)

    print("The End")
    exit(1)


if __name__ == "__main__":
    config = Config().parse()
    main(config)
