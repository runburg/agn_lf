#!/usr/bin/env python3
"""
File: utils.py
Author: Jack Runburg
Email: jack.runburg@gmail.com
Github: https://github.com/runburg
Description: Helpful funcs
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.io import fits


def colorbar_for_subplot(fig, axs, cmap, image):
    """Place a colorbar by each plot.

    Helper function to place a color bar with nice spacing by the plot.


    Inputs:
        - fig: figure with the relevant axes
        - axs: axs to place the colorbar next to
        - cmap: color map for the colorbar
        - image: image for the colorbar

    Returns:
        - the colorbar object
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(axs)
    # Create the axis for the colorbar 
    cax = divider.append_axes('right', size='5%', pad=0.05)

    # Create the colorbar
    cbar = fig.colorbar(image, cax=cax)
    # cm.ScalarMappable(norm=None, cmap=cmap),

    return cbar


def trim_axes(axes, N):
    """Trim the axes list to proper length.

    Helper function if too many subplots are present.


    Input:
        - axes: list of axes of the subplots
        - N: the number of subplots to keep

    Return:
        - list of retained axes
    """
    if N > 1:
        axes = axes.flat
        for ax in axes[N:]:
            ax.remove()
        return axes[:N]

    return [axes]


def configure_plot_params(fontsize=20, spacing=0.2):
    """Make plots look the way I want."""
    mpl.rc('font', **{'family': 'serif', 'serif': ['cmr10'], 'weight': 'light'})
    mpl.rc('text', usetex=True)
    mpl.rc('text.latex', preamble=r'\usepackage{amsmath, graphicx}')
    mpl.rcParams['mathtext.fontset'] = 'cm'
    # mpl.rcParams['font.serif'] = 'Computer Modern Roman'
    mpl.rcParams["font.family"] = "serif"
    # mpl.rcParams['font.size'] = fontsize
    # mpl.rcParams['mathtext.fontset'] = 'cm'
    # mpl.rcParams['axes.labelsize'] = 'xx-large'
    # mpl.rcParams['axes.titlesize'] = 'xx-large'

    mpl.rcParams['figure.subplot.wspace'] = spacing
    mpl.rcParams['figure.subplot.hspace'] = spacing

    # mpl.rcParams['xtick.labelsize'] = 'xx-large'
    mpl.rcParams['xtick.major.size'] = 7.5
    mpl.rcParams['xtick.major.width'] = 1
    mpl.rcParams['xtick.minor.size'] = 3.75
    mpl.rcParams['xtick.minor.width'] = 0.5

    # mpl.rcParams['ytick.labelsize'] = 'xx-large'
    mpl.rcParams['ytick.major.size'] = 7.5
    mpl.rcParams['ytick.major.width'] = 1
    mpl.rcParams['ytick.minor.size'] = 3.75
    mpl.rcParams['ytick.minor.width'] = 0.5


def plot_setup(rows, cols, d=0, buffer=(0.4, 0.4), set_global_params=False):
    """Set mpl parameters for beautification.

    Make matplotlib pretty again!


    Input:
        - rows: number of rows of subplots
        - cols: number of columns of subplots
        - d: number of total subplots needed
        - buffer: tuple of white space around each subplot

    Returns:
        - figure object, list of subplot axes
    """
    # setup plot
    plt.close('all')

    if set_global_params is True:
        configure_plot_params(spacing=0)

    # Create figure
    figsize = (5 * cols + buffer[0], 3.5 * rows + buffer[1])
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize, sharex=True, sharey=True)

    # Trim unnecessary axes
    if d != 0:
        axs = trim_axes(axs, d)

    return fig, axs


def load_wcs(path, which_hdu=0):
    """Load and return wcs and hdu."""
    from astropy.wcs import WCS

    hdu = fits.open(path)[which_hdu]
    wcs = WCS(hdu.header)
    print(wcs)

    return wcs, hdu
