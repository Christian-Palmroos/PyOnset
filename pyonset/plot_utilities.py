
"""
This file contains functions and constants for plotting procedures for PyOnset.

"""


__author__ = "Christian Palmroos"

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText

# For consistent figure sizes
STANDARD_FIGSIZE = (21,9)
VDA_FIGSIZE = (16,9)

TITLE_FONTSIZE = 30
AXLABEL_FONTSIZE = 26
TICK_LABELSIZE = 22
TXTBOX_SIZE = 23
LEGEND_SIZE = 24

# For consistent color usage
COLOR_SCHEME = {
    "median" : "red",
    "mode" : "navy",
    "mean" : "darkorange",
    "1-sigma" : "red",
    "2-sigma" : "blue"
}

BACKGROUND_ALPHA = 0.15 # used for the background shading when plotting


def set_fig_ylimits(ax:plt.Axes, ylim:list=None, flux_series:pd.Series=None):
    """
    Sets the vertical axis limits of the figure, given an Axes and a tuple or list of y-values.
    """

    # In case not otherwise specified, set the lower limit to half of the smallest plotted value,
    # and the higher limit to 1.5 times the highest value
    if ylim is None:
        try:
            ylim = [np.nanmin(flux_series[flux_series > 0]) * 0.5,
                    np.nanmax(flux_series) * 1.5]
        except ValueError:
            # This error is caused by a zero-size array to which operation minimum has no identity
            # -> can't manually set reasonable limits. Let plt dynamically handle it.
            return ax.get_ylim()

        # If the lower y-boundary is some ridiculously small number, adjust the y-axis a little
        if np.log10(ylim[1])-np.log10(ylim[0]) > 10:
            ylim[0] = 1e-4

    ylim = ax.set_ylim(ylim)

    return ylim


def set_standard_ticks(ax):
    """
    Handles tickmarks, their sizes etc...
    """
    ticklen = 11
    tickw = 2.8

    ax.tick_params(which="major", length=ticklen, width=tickw, labelsize=TICK_LABELSIZE)
    ax.tick_params(which="minor", length=ticklen-3, width=tickw-0.6)


def set_legend(ax: plt.Axes, legend_loc: str, fontsize:int):

    # Legend placement:
    if legend_loc=="out":
        # loc=3 means that the legend handle is "lower left"
        legend_handle, legend_bbox = 3, (1.0, 0.01)
    elif legend_loc=="in":
        # loc=4 means that the legend handle is "lower right"
        legend_handle, legend_bbox = 4, (1.0, 0.01)
        fontsize = fontsize - 2 # Having legend in makes everything bigger -> decrease font a bit
    else:
        raise ValueError(f"Argument legend_loc has to be either 'in' or 'out', not {legend_loc}")

    # Sets the legend
    ax.legend(loc=legend_handle, bbox_to_anchor=legend_bbox, fontsize=fontsize)

