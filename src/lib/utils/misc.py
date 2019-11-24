import numpy as np


def sim_diss_indices(rdms, n, sim_or_dissim):
    row_size, col_size = rdms.shape
    min_ind, max_ind = [], []

    new_rdm = np.copy(rdms)
    new_rdm[np.tril_indices(rdms.shape[0])] = 999
    new_rdm = new_rdm.reshape(row_size*col_size, )
    sorted_rdms_ind = np.argsort(new_rdm)
    min_10_ind = sorted_rdms_ind[row_size:row_size+n]

    new_rdm = np.copy(rdms)
    new_rdm[np.tril_indices(rdms.shape[0])] = -999
    new_rdm = new_rdm.reshape(row_size*col_size, )
    sorted_rdms_ind = np.argsort(new_rdm)
    max_10_ind = sorted_rdms_ind[:row_size*col_size-n+1:-1]

    min_ind.append([(index//row_size, index % row_size)
                    for index in min_10_ind])
    max_ind.append([(index//row_size, index % row_size)
                    for index in max_10_ind])

    if sim_or_dissim == "sim":
        return min_ind[0]
    return max_ind[0]


def get_category_img(stimulus_ids, file_name):
    category_ind, img = [], []
    with open(data_dir+file_name+".tsv", "r") as f:
        content = f.read()
        rows = content.split("\n")
        rows = np.array([row.split("\t") for row in rows])
        for stimulus_id in stimulus_ids:
            stimulus_id = str(stimulus_id[0])
            stimulus_id = stimulus_id + "0"*(6-len(stimulus_id.split(".")[-1]))
            cat_val = rows[np.where(rows[:, 1] == stimulus_id), 2][0]
            img_val = rows[np.where(rows[:, 1] == stimulus_id), 0][0]
            category_ind = np.append(category_ind, cat_val)
            img = np.append(img, img_val)

    return category_ind, img
