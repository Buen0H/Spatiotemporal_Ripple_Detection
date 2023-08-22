'''
Artists
'''
import matplotlib.pyplot as plt

from numpy import median
def GI_plot(img, dpi=100, title=None, a=2, cmap='gray', scaleLim=None):
    ''' Artist to draw spatiotemporal map of data
    INPUTS
    img     - (N, T) array to plot via imshow
    dpi     - dots per inch; quality of figure output
    title   - figure title
    a       - scaling factor of color axis
    cmap    - colormap
    scaleLim- optional control of color axis
    OUTPUTS
    none
    '''
    fig = plt.figure(figsize=(15, 5), dpi=dpi)
    if title:
        plt.title(title, fontsize=30)
    if not scaleLim:
        mu, sigma = median(img, axis=0).mean(), img.std()
        vmin, vmax = mu-a*sigma, mu+a*sigma
    else:
        vmin, vmax = scaleLim
    img = plt.imshow(img, cmap=cmap,vmin=vmin, vmax=vmax, aspect='auto')
    cbar = fig.colorbar(img)
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Space', fontsize=20)
    plt.show()
    plt.close()


from numpy import median
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from algorithm import detectClusteredRipples
def GI_plot_pretty(X, title=None, a=1.5, dpi=100):
    ''' Artist to draw spatiotemporal map with outputs of the clustered ripple algorithm.
    INPUTS
    X     - (N, T) array to plot via imshow
    title   - figure title
    a       - threshold for peak detection
    dpi     - dots per inch; quality of figure output

    OUTPUTS
    none
    '''
    fig = plt.figure(figsize=(25, 10), dpi=dpi)
    gs = fig.add_gridspec(2, 2, height_ratios=(1, 4), width_ratios=(40, 1),
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.05, hspace=0.05)
    if title:
        fig.suptitle(title, fontsize=30)

    # Draw image
    ax_img = fig.add_subplot(gs[1, 0])
    mu, sigma = median(X, axis=0).mean(), X.std(); c=2  ### scaling
    vmin, vmax = mu-c*sigma, mu+c*sigma
    img = plt.imshow(X, cmap='gray',vmin=vmin, vmax=vmax, aspect='auto')
    ax_img.set_xlabel('Time', fontsize=20)
    ax_img.set_ylabel('Space', fontsize=20)

    # Draw colorbar
    cbax = fig.add_subplot(gs[1, 1])
    cbar = fig.colorbar(img, cax=cbax, )

    # Plot line
    X_t, X_t_filt, peaks, properties = detectClusteredRipples(X, a)
    ax_line = fig.add_subplot(gs[0, 0])
    ax_line.plot(X_t, label='energy over time', linewidth=1)
    ax_line.plot(X_t_filt, linewidth=2)
    ax_line.plot(peaks, X_t_filt[peaks], 'k*', markersize=10, label='peak')
    ax_line.hlines(y=[X_t.mean(), X_t.mean()+a*X_t.std()], xmin=0, xmax=X.shape[1],
                   colors='gray', linewidths=1, linestyles='dashed')
    ax_line.hlines(y=properties['width_heights'], xmin=properties['left_ips'], xmax=properties['right_ips'],
           colors='green', linewidths=5, label='peak width')
    ax_line.set_xticks([])
    ax_line.set_yticks([])
    ax_line.set_xlim(0, len(X_t))
    ax_line.legend(loc='upper left', bbox_to_anchor=(1., 1.))

    # Add rectangles from the peaks found
    clustered_ripples = [Rectangle(xy=(x0, 0), width=x1-x0, height=X.shape[0])
                            for x0, x1 in zip(properties['left_ips'], properties['right_ips'])]
    pc = PatchCollection(clustered_ripples, facecolor='g', alpha=0.2)
    ax_img.add_collection(pc)

    plt.show()
    plt.close()
