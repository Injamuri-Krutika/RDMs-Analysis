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
