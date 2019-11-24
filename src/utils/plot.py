import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_rdm(rdms, roi):

    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(121)
    im1 = ax1.imshow(rdms["pearson"][roi], interpolation='None', cmap='bwr',)
    ax1.set_title('RDM, sorted, %s, distance: %s,' % (roi, "pearson"))

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')

    ax2 = fig.add_subplot(122)
    im2 = ax2.imshow(rdms["kernel"][roi], interpolation='None', cmap='bwr',)
    ax2.set_title('RDM, sorted, %s, distance: %s,' % (roi, "kernel"))

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')
