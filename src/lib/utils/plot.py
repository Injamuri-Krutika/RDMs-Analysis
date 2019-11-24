import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os


def plot_rdm_group(rdms, roi, distance_measures, path):
    print(distance_measures)
    subplot_2 = [121, 122]
    subplot_3 = [131, 132, 133]
    fig = plt.figure(figsize=(16, 12))
    for distance in distance_measures:
        subplot_nums = subplot_2 if len(distance_measures) == 2 else subplot_3
        for i, subplot_num in enumerate(subplot_nums):
            distance = distance_measures[i]
            ax = fig.add_subplot(subplot_num)
            print("distance :", distance)
            im = ax.imshow(rdms[distance],
                           interpolation='None', cmap='bwr',)
            ax.set_title('RDM, %s, distance: %s,' % (roi, distance))

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
    print(path+"/"+roi+".png")
    plt.savefig(os.path.join(path, roi+".png"))
