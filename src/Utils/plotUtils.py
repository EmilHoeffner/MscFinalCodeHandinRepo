import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html
def scatter_hist(x, y, title, x_label, y_label, save_path, binwidth = 0.05):

    print("Plotting Scatter Hist Begin...")

    fig, axs = plt.subplot_mosaic([['histx', '.'],
                               ['scatter', 'histy']],
                              figsize=(6, 6),
                              width_ratios=(4, 1), height_ratios=(1, 4),
                              layout='constrained')
    ax = axs["scatter"]
    ax_histx = axs["histx"]
    ax_histy = axs["histy"]
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # Title and labels:
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # the scatter plot:
    ax.scatter(x, y)

    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(0, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

    fig.savefig(save_path)

    plt.close(fig)

    print("Plotting Scatter Hist End")

# https://python-graph-gallery.com/83-basic-2d-histograms-with-matplotlib/
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist2d.html
# https://stackoverflow.com/questions/17201172/a-logarithmic-colorbar-in-matplotlib-scatter-plot
# https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.SymLogNorm.html#matplotlib.colors.SymLogNorm
# https://matplotlib.org/stable/gallery/images_contours_and_fields/colormap_normalizations.html
# https://saturncloud.io/blog/how-to-change-the-font-size-of-colorbars-in-matplotlib-a-guide/
# https://stackoverflow.com/questions/15305737/python-matplotlib-decrease-size-of-colorbar-labels
def scatter_density_2d(x, y, title, x_label, y_label, save_path, log_color_bar = False):
    plt.title(title, fontsize = 18)
    plt.xlabel(x_label, fontsize = 18)
    plt.ylabel(y_label, fontsize = 18)

    slack_x = max(np.abs(x)) * 0.1
    slack_y = max(np.abs(y)) * 0.1

    plt.hist2d(x, y, bins=(10, 10), cmap=plt.cm.Greys_r, 
               range = [[min(x) - slack_x, max(x) + slack_x], [min(y) - slack_y, max(y) + slack_y]],
               #norm=matplotlib.colors.SymLogNorm(linthresh=0.01), 
               norm = matplotlib.colors.PowerNorm(gamma=0.5))
    
    cbar = plt.colorbar()
    cbar.set_label("Frequency", size=18)
    cbar.ax.tick_params(labelsize=18)

    plt.scatter(x, y)
        
    #plt.xlim((min(x), max(x)))
    #plt.ylim((min(y), max(y)))
    plt.tick_params("x", labelsize = 18)
    plt.tick_params("y", labelsize = 18)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()