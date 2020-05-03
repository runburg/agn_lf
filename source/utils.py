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


def plot_setup(rows, cols, d=0, buffer=(0.4, 0.4)):
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

    # Format label sizes
    mpl.rcParams['axes.labelsize'] = 'xx-large'
    mpl.rcParams['ytick.labelsize'] = 'x-large'
    mpl.rcParams['xtick.labelsize'] = 'x-large'

    # Choose font
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Create figure
    figsize = (5 * cols + buffer[0], 3.5 * rows + buffer[1])
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize, sharex=True, sharey=True)

    # Trim unnecessary axes
    if d != 0:
        axs = trim_axes(axs, d)

    return fig, axs



