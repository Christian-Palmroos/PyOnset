
"""
This file contains the BootstrapWindow class for PyOnset.

"""


__author__ = "Christian Palmroos"

import numpy as np
import pandas as pd

import matplotlib.colors as cl
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText

from .plot_utilities import BACKGROUND_ALPHA, TXTBOX_SIZE
from .calc_utilities import k_parameter, k_legacy

# We recommend to have at least this many data points in the background for good statistics
MIN_RECOMMENDED_POINTS = 100

KCONTOUR_FIGSIZE = (12,10)
KCONTOUR_TITLESIZE = 24

DEFAULT_KCONTOUR_CMAP = "seismic"
KCONTOUR_COLORS_AMOUNT = 61


class BootstrapWindow:

    def __init__(self, start, end, n_shifts=0, bootstraps=1):
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        self.n_shifts = n_shifts
        self.bootstraps = bootstraps

        self.attrs_dict = {
                        "start" : self.start,
                        "end" : self.end,
                        "n_shifts" : self.n_shifts,
                        "bootstraps": self.bootstraps
                          }

        self.init_state = self.save_state(return_state=True)

        self.background_selection = None
        self.max_recommended_reso: int = None
        self.mu = np.nan
        self.sigma = np.nan

    def __repr__(self):
        return self.attrs()

    def __len__(self):
        return int((self.end - self.start).total_seconds()//60)

    def attrs(self, key=None):

        if key is None:
            return str(self.attrs_dict)
        else:
            return self.attrs_dict[key]

    def apply_background_selection(self, flux_series:pd.Series):
        self.background_selection = flux_series.loc[(flux_series.index >= self.start) & (flux_series.index < self.end)]
        self.set_background_params()

    def set_background_params(self):
        self.mu = self.background_selection.mean()
        self.sigma = self.background_selection.std()

    def draw_background(self, ax):
        """
        Draws the window on a plot, given the axes of a plot.

        Parameters:
        -----------
        ax : {plt.Axes}
                The axes of the figure.
        """
        ax.axvspan(xmin=self.start, xmax=self.end,
                        color="#e41a1c", label="Background", alpha=BACKGROUND_ALPHA)
    
        # Textbox indicating onset time
        blabel = AnchoredText(f"Background:\n{self.start} - {self.end}", prop=dict(size=TXTBOX_SIZE), frameon=True, loc="upper left")
        blabel.patch.set_boxstyle("round, pad=0., rounding_size=0.2")
        blabel.patch.set_linewidth(2.0)
        ax.add_artist(blabel)

    def poisson_comparison_figure(self, num_of_bins:int=None, seed:int=1) -> plt.Figure:
        """
        Produces a histogram displaying the observed intensity in the background
        and overlapping corresponding Poisson-distributed histogram.
        Allows for visual inspection of how closely the distribution of background
        intensities resemble a poisson-distributed selection.

        Parameters:
        -----------
        num_of_bins : {int} Number of bins for the histogram. Optional.
        seed : {int} The seeding number for the random number generator.
        """

        # The selection of the VALUES from the background. 
        # type(background_selection) == pd.Series
        bg_sel = self.background_selection.values

        # Init the random number generator and produce 
        rng = np.random.default_rng(seed=seed)
        pois = rng.poisson(lam=bg_sel.mean(), size=len(bg_sel))

        # Try to find a reasonable number of bins if not given a definite amount
        if num_of_bins is None:
            num_of_bins = (max(bg_sel.astype(int)) - min(bg_sel.astype(int)))//4
        bins = np.linspace(min(bg_sel.astype(int)),max(bg_sel.astype(int)), num_of_bins)

        # Init the figure
        fig = plt.figure(layout="constrained")
        ax = fig.add_subplot()

        # Draw the two histograms: first for observed and the second for theoretical
        # values
        obs = ax.hist(bg_sel.astype(int), bins=bins, color="tab:blue", label="observed")
        exp = ax.hist(pois, bins=bins, color="darkorange", alpha=0.6, label="theoretical")

        ax.set_title("Background intensity distribution:\nobserved + theoretical")
        ax.legend()

        plt.close()

        return fig

    def fast_poisson_test(self, tolerance=3e-1) -> bool:
        """
        Does a preliminary fast poisson test: checks the relation between the mean and variance of
        the background selection. For a Poisson distribution, the mean and the variance are equal.

        parameters:
        ------------
        tolerance : {float} The relative tolerance accepted 
        """

        bg_sel = self.background_selection

        mean = bg_sel.mean()
        var = bg_sel.var()

        # Catch discrepancy here and issue a warning
        if not np.isclose(mean, var, rtol=tolerance):
            print(f"Fast Poisson test warning: ratio of background var/mu = {(var/mean):.3e}.")
            print("Ratio > 1 may lead to late detection of the event.")
            return False

        return True

    def k_contour(self, sigma_multiplier:int, cmap:str=None, fig:plt.Figure=None, ax:plt.Axes=None,
                  k_model:str=None):
        """
        Draws a k-contour plot as a function of background 

        Parameters:
        -----------
        sigma_multiplier : {int}
        cmap : {str} Name of the colormap
        fig : {plt.Figure}
        ax : {plt.Axes}
        k_model : {Callable,str} The model that calculates k. Either a function or
                    the string 'legacy'.
        """

        KCONTOUR_LOW_LIMIT = 1e-1
        KCONTOUR_HIGH_LIMIT = 1e1

        KCONTOUR_OOM_PM = 2

        def order_of_magnitude(num):
            return np.floor(np.log10(num))

        mu = self.background_selection.mean()
        sigma = self.background_selection.std()

        # Initialize axis variables (mu & sigma)
        # Check also that mu,sigma>0, because otherwise they don't have an order of magnitude
        mu_oom = order_of_magnitude(num=mu) if mu>0 else 0
        sigma_oom = order_of_magnitude(num=sigma) if sigma>0 else 0

        # Generate the mus and sigmas with the datatype longdouble, because in extreme cases
        # calculation precision starts to be a problem with regular floating point numbers.
        mus = np.logspace(mu_oom-KCONTOUR_OOM_PM, mu_oom+KCONTOUR_OOM_PM, num=1500, dtype=np.longdouble)
        sigmas = np.logspace(sigma_oom-KCONTOUR_OOM_PM, sigma_oom+KCONTOUR_OOM_PM, num=1500, dtype=np.longdouble)

        # The meshgrid to use in plotting
        xx, yy = np.meshgrid(mus, sigmas)

        # By default we use the k_parameter that is defined as a part of this software. If
        # user inputs the word 'legacy', then the old SEPpy version of the k-parameter is used.
        # Otherwise the user inputs a callable function for the k-parameter with identical 
        # signature to the standard k-parameter.
        if k_model is None:
            k_model = k_parameter
        if k_model=="legacy":
            k_model = k_legacy

        ks = k_model(mu=xx, sigma=yy, sigma_multiplier=sigma_multiplier)
        user_k = k_model(mu=mu, sigma=sigma, sigma_multiplier=sigma_multiplier)

        if cmap is None:
            cmap = DEFAULT_KCONTOUR_CMAP

        colormap = plt.get_cmap(cmap, KCONTOUR_COLORS_AMOUNT)

        # Initialize the figure (maybe)
        if ax is None:
            fig, ax = plt.subplots(figsize=KCONTOUR_FIGSIZE)
            ax_provided = False
        else:
            ax_provided = True

        ax.set_title(fr"k as a function of background $\mu$ and $\sigma$ (n={sigma_multiplier})", fontsize=KCONTOUR_TITLESIZE)

        ax.set_yscale("log")
        ax.set_xscale("log")

        # Plotting
        ks_lognorm = cl.LogNorm(vmin=KCONTOUR_LOW_LIMIT, vmax=KCONTOUR_HIGH_LIMIT)
        kesh = ax.pcolormesh(mus, sigmas, ks, shading="nearest", cmap=colormap,
                            norm = ks_lognorm)

        # The colorbar for k values
        cb = fig.colorbar(kesh, ax=ax)
        cb.set_label('k', rotation=360, fontsize=22)

        cb.ax.axhline(user_k, color="darkorange", lw=2.5)

        # Lines for mu=sigma and mu=sigma^2
        ax.plot(mus, mus, ls="--", color="black", zorder=3, label=r"$\mu = \sigma$")
        ax.plot(mus, sigmas**2, lw=1.0, color="black", zorder=4, label=r"$\mu = \sigma^2$")

        # Plotting user mu and sigma
        ax.scatter(mu, sigma, s=125, color='k')
        ax.scatter(mu, sigma, s=85, color="orange",
                label=fr"$\mu={mu:.3e}$"+"\n"+fr"$\sigma={sigma:.3e}$"+"\n"+f"k={user_k:.2f}")

        ax.set_ylabel(r"$\sigma$", fontsize=24)
        ax.set_xlabel(r"$\mu$", fontsize=24)

        ax.set_ylim([sigmas.min(), sigmas.max()])
        ax.set_xlim([mus.min(), mus.max()])
        
        ax.legend(loc=2)

        if ax_provided:
            return cb

        return (ax, fig)

    def print_max_recommended_reso(self):
        """
        Prints out the maximum recommended resolution that the time series should be averaged to in order to still have
        at least {MIN_RECOMMENDED_POINTS} data points inside the background window.
        """

        minutes_in_background = len(self)

        # We recommend a maximum reso such that there are at least {MIN_RECOMMENDED_POINTS} data points to pick from
        self.max_recommended_reso = int(minutes_in_background/MIN_RECOMMENDED_POINTS)
        print(f"Your chosen background is {minutes_in_background} minutes long. To preserve the minimum of {MIN_RECOMMENDED_POINTS} data points to choose from,\nit is recommended that you either limit averaging up to {self.max_recommended_reso} minutes or enlarge the background window.")


    def move_window(self, hours=0, minutes=0):
        """
        Moves the BootstrapWindow by a given amount of hours and minutes.
        """
        if self.n_shifts > 0:
            self.start = self.start + pd.Timedelta(hours=hours, minutes=minutes)
            self.end = self.end +pd.Timedelta(hours=hours, minutes=minutes)
            self.n_shifts = self.n_shifts - 1

    def save_state(self, return_state=False):
        """
        Saves the state in which the window object is at the time of running this method.
        Mainly used to save / load the initialization state

        Parameters:
        -----------
        return_state : {bool}, default False
                    if True, return the dictionary that contains the state of class attributes. This is used to save
                    the initialization state of the object to a class attribute.
        """

        self.saved_state = {
            "start" : self.start,
            "end" : self.end,
            "n_shifts" : self.n_shifts,
            "bootstraps" : self.bootstraps
            }

        if return_state:
            return self.saved_state

    def load_state(self, which="previous"):
        """
        Loads the by default the previously saved state, i.e., sets the object's attributes back to what they were at the previous save point.
        """

        if which == "previous":

            self.start = self.saved_state["start"]
            self.end = self.saved_state["end"]
            self.n_shifts = self.saved_state["n_shifts"]
            self.bootstraps = self.saved_state["bootstraps"]

        elif which == "init":

            self.start = self.init_state["start"]
            self.end = self.init_state["end"]
            self.n_shifts = self.init_state["n_shifts"]
            self.bootstraps = self.init_state["bootstraps"]

        else:

            raise Exception(f"Can only load either 'previous' or 'init' state! The input was {which}")



def subwindows(Window, window_placement="equal", fraction:float = 0.2):
    """
    A function that turns a BootstrapWindow object into a list of smaller BootstrapWindows according to n_shifts.
    Each new smaller BootstrapWindow will still be contained within the original Window.
    Returned windows fill always have n_shifts set to 0.

    Parameters:
    -----------
    Window : BootstrapWindow object
            Defines the boundaries within the small new windows will be created, and the amount of them
    window_placement : str
            Defines the type of subwindow distribution. Either 'random' or 'equal'
    fraction : float
            The size fraction of the input window that the subwindows will be. Defaults to 0.2, i.e., 20 %
    Returns:
    -----------
    windowlist : list(BootstrapWindow)
            A list of <n_shifts> new Bootstrapwindows.
    """

    # This is the amount of samples that are picked from the big window, use the same for the little windows
    bootstraps = Window.bootstraps

    # We need to know that temporal range of the big window, because the little windows have to be some reasonable
    # fraction of the big window
    big_windowrange_in_secs = (Window.end - Window.start).seconds

    if window_placement == "random":

        # Small window will reach for fraction times the big window's range
        small_window_len = int(fraction * big_windowrange_in_secs)

        # Pre-calculate random starting points for the little windows, remembering that they cannot start later
        # than (1 - fraction) of the big window, in order for the big window to completely contain all the little windows
        window_startpoints = (1.0 - fraction) * big_windowrange_in_secs * np.random.random((Window.n_shifts,))

    elif window_placement == "equal":

        # Small window will in the equal case be 1/n_shifts the size of the big windows, so as to fit equal-sized
        # windows inside the big window
        fraction = 1/Window.n_shifts
        small_window_len = int(fraction * big_windowrange_in_secs)

        # Generate starting points for the windows, we already know everything else about them
        window_startpoints = [sec for sec in np.linspace(start=0, stop=big_windowrange_in_secs-small_window_len, num=Window.n_shifts)]

    else:
        raise Exception(f"The window_placement must be either 'random' or 'equal'. The input was: {window_placement}")


    windowlist = []
    for sec in window_startpoints:

        start = Window.start + pd.Timedelta(seconds=int(sec))
        end = start + pd.Timedelta(seconds=small_window_len)

        # New bootstrapwindows is initialized with random start, end that's at a set distance from the start,
        # n_shifts always 0 and the amount of bootstrapped samples is equal to the original amount
        new_window = BootstrapWindow(start=start, end=end, n_shifts=0, bootstraps=bootstraps)

        windowlist.append(new_window)

    return windowlist