import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import numpy as np
from lib.utils.corr import get_corr
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform


def plot_rdm_group(rdms, roi, distance_measures, path, labels):
    subplot_2 = [121, 122]
#     subplot_3 = [131, 132, 133]
    fig = plt.figure(figsize=(16, 12))

    distance = distance_measures[0]
    ax1 = fig.add_subplot(subplot_2[0])
    im1 = ax1.imshow(rdms[distance],
                     interpolation='None', cmap='bwr',)
    ax1.set_title('RDM, %s, distance: %s,' % (roi, distance))
    ax1.set_yticks(np.arange(len(labels)))
    ax1.set_yticklabels(labels)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')

    distance = distance_measures[1]
    ax2 = fig.add_subplot(subplot_2[1])
    im2 = ax2.imshow(rdms[distance],
                     interpolation='None', cmap='bwr',)
    ax2.set_title('RDM, %s, distance: %s,' % (roi, distance))
    ax2.set_yticks(np.arange(len(labels)))
    ax2.set_yticklabels(labels)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')
    corr = get_corr(rdms[distance_measures[0]], rdms[distance_measures[1]])
    plt.title(
        f'Correlation between two RDMs: {corr}', x=-25, y=-0.2, fontsize=20)
    plt.savefig(os.path.join(path, roi+".png"))
    plt.close()
