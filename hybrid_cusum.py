
import datetime
import glob
import os
import pickle
import warnings

import astropy.units as u
import ipywidgets as widgets
import matplotlib.colors as cl
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from astropy import constants as const
# from matplotlib.ticker import LogLocator, ScalarFormatter
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.dates import DateFormatter
from matplotlib.offsetbox import AnchoredText
from sunpy.util.net import download_file

from sunpy import __version__

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from IPython.display import Markdown

import seppy.tools # this is imported first to avert ImportError in the data loaders that are imported after
import seppy.util as util
from seppy.loader.psp import (calc_av_en_flux_PSP_EPIHI,
                              calc_av_en_flux_PSP_EPILO, psp_isois_load)
from seppy.loader.soho import calc_av_en_flux_ERNE, soho_load
from seppy.loader.solo import epd_load
from seppy.loader.stereo import calc_av_en_flux_HET as calc_av_en_flux_ST_HET
from seppy.loader.stereo import calc_av_en_flux_SEPT, stereo_load
from seppy.loader.wind import wind3dp_load
from seppy.tools import Event

"""
A library that holds the Onset, BackgroundWindow and OnsetStatsArray classes.

@Author: Christian Palmroos <chospa@utu.fi>

@Updated: 2023-11-16

Known problems/bugs:
    > Does not work with SolO/STEP due to electron and proton channels not defined in all_channels() -method
"""


__author__ = "Christian Palmroos"
__email__ = "chospa@utu.fi"

# Some useful global constants
CURRENT_PATH = os.getcwd()
STANDARD_FIGSIZE = (21,9)
VDA_FIGSIZE = (16,9)
C_SQUARED = const.c.value*const.c.value

# We recommend to have at least this many data points in the background for good statistics
MIN_RECOMMENDED_POINTS = 100

# A new class to inherit everything from serpentine Event, and to extend its scope of functionality
class Onset(Event):

    def __init__(self, start_date, end_date, spacecraft, sensor, species, data_level, data_path, viewing=None, radio_spacecraft=None, threshold=None):
        super().__init__(start_date, end_date, spacecraft, sensor,
                 species, data_level, data_path, viewing, radio_spacecraft,
                 threshold)

        # Everytime an onset is found any way, the last used channel should be updated
        self.last_used_channel = np.NaN

        # The background window is stored to this attribute when cusum_onset() is called with a BootStrapWindow input
        self.background = np.NaN

        # This list is for holding multiple background windows if such were to be used
        self.list_of_bootstrap_windows = []
        self.window_colors = ["blue", "orange", "green", "purple", "navy", "maroon"]

        # This will turn true once the extensive statistics analysis is run
        self.mean_of_medians_onset_acquired = False

        # This is a check to make sure viewing is correctly set for single-aperture instruments
        if self.viewing:
            self.check_viewing()

        self.all_channels = {
            "sta_let_e" : np.arange(1, dtype=int),
            "sta_sept_e" : np.arange(2,17, dtype=int),
            "sta_het_e" : np.arange(3, dtype=int),
            "sta_let_p" : np.arange(1, dtype=int),
            "sta_sept_p" : np.arange(2,32, dtype=int),
            "sta_het_p" : np.arange(11, dtype=int),
            
            "stb_let_e" : np.arange(1, dtype=int),
            "stb_sept_e" : np.arange(2,17, dtype=int),
            "stb_het_e" : np.arange(3, dtype=int),
            "stb_let_p" : np.arange(1, dtype=int),
            "stb_sept_p" : np.arange(2,32, dtype=int),
            "stb_het_p" : np.arange(11, dtype=int),

            "solo_step_e" : np.arange(1, dtype=int),
            "solo_ept_e" : np.arange(34, dtype=int),
            "solo_het_e" : np.arange(4, dtype=int),
            "solo_step_p" : np.arange(1, dtype=int),
            "solo_ept_p" : np.arange(64, dtype=int),
            "solo_het_p" : np.arange(36, dtype=int),

            "soho_ephin_e" : (150, 300, 1300, 3000),
            "soho_erne_p" : np.arange(10, dtype=int),

            "wind_3dp_e" : np.arange(7, dtype=int),
            "wind_3dp_p" : np.arange(9, dtype=int),

            "psp_isois-epihi_e" : np.arange(19, dtype=int),
            "psp_isois-epihi_p" : np.arange(15, dtype=int),

            "psp_isois-epilo_e" : np.arange(12, dtype=int),
        }

        # This dictionary holds the median onset time and weighted confidence intervals derived by CUSUM-bootstrap 
        # hybrid method. In the dictionary each channel identifier is paired with a list of [mode, median, 1st_minus, 1st_plus, 2nd_minus, 2nd_plus]
        self.onset_statistics = {}

        # This is a dictionary that holds the information of each instrument's minimum time resolution. That value
        # will be used in the case that cusum-bootstrap yields uncertainty smaller than that.
        self.minimum_cadences = {
            "bepicolombo_sixs-p" : pd.Timedelta("8s"),
            "psp_isois-epihi" : pd.Timedelta("1min"),
            "psp_isois-epilo" : pd.Timedelta("1min"),
            "soho_erne" : pd.Timedelta("1min"),
            "soho_ephin" : pd.Timedelta("1min"),
            "sta_sept" : pd.Timedelta("1min"),
            "sta_het" : pd.Timedelta("1min"),
            "stb_sept" : pd.Timedelta("1min"),
            "stb_het" : pd.Timedelta("1min"),
            "solo_step" : pd.Timedelta("1s"),
            "solo_ept" : pd.Timedelta("1s"),
            "solo_het" : pd.Timedelta("1s"),
            "wind_3dp" : pd.Timedelta("12s")
        }

    def check_viewing(self, returns=False):
        """
        A method that checks at initialization that a viewing direction wasn't erroneously appointed
        for an instrument that doesn't have viewing directions.
        Sets self.viewing to None for STA/STB HET and all SOHO instruments (ERNE and EPHIN)

        returns : {bool}
                    Switch to return the viewing
        """

        if self.sensor.lower() in ("het", "erne", "ephin"):

            if self.spacecraft.lower() in ("sta", "stb", "soho"):
                self.viewing = None

        if returns:
            return self.viewing

    def __repr__(self):
        return str(f"({self.spacecraft},{self.sensor},{self.species})")

    def get_all_channels(self):
        """
        Returns a range(first,last+1) of all the channel identifiers for any unique instrument+species pair.
        """
        return self.all_channels[f"{self.spacecraft}_{self.sensor}_{self.species}"]

    def get_minimum_cadence(self):
        return self.minimum_cadences[f"{self.spacecraft}_{self.sensor}"]

    def get_time_resolution_str(self, resample):
        # Choose resample as the averaging string if it exists
        if resample:
            time_reso_str = f"{resample} averaging" 
        # If not, check if the native cadence is less than 60. If yes, address the cadence in seconds
        elif self.get_minimum_cadence().seconds<60:
            time_reso_str = f"{self.get_minimum_cadence().seconds} s data"
        # If at least 60 seconds, address the cadence in minutes
        else:
            time_reso_str = f"{int(self.get_minimum_cadence().seconds/60)} min data"

        return time_reso_str

    def add_bootstrap_window(self, window):
        """
        A little helper method to add bootstrapping windows to the class list
        """
        self.list_of_bootstrap_windows.append(window)

    def update_onset_statistics(self, stats):
        """
        Adds the weighted median and confidence intervals of a specific channel to onset_statistics dictionary
        """

        self.onset_statistics[self.last_used_channel] = stats

    def input_nat_onset_stats(self, key):
        """
        This method can be used to e.g., fill in SolO/EPT channel 0 stats, which is unavailable in latest data.
        It may also be used to erase clearly erroneous onset time data from the object.
        """
        nan_stats = [pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT]
        self.onset_statistics[key] = nan_stats

    def cusum_onset(self, channels, background_range, viewing=None, resample=None, cusum_minutes=30, sigma=2, title=None, save=False, savepath=None, 
                    yscale='log', ylim=None, erase=None, xlim=None, show_stats=True, diagnostics=False, plot=True):
        """
        Does a Poisson-CUSUM-method-based onset analysis for given OnsetAnalysis object.
        Based on an earlier version by: Eleanna Asvestari <eleanna.asvestari@helsinki.fi>

        Parameters
        ----------
        channel: int or a list of 2 ints
                Index of the channel inside the DataFrame that holds the data.
        background_range : tuple/list or {BootstrapWindow}
                The starting and ending points of what is considered background. 
        resample: str
                Pandas-compatible time string, e.g., '1min' or '30s'
        cusum_minutes: int
                The amount of minutes the intensity should stay above threshold until onset is identified. Corresponds to the amount
                of consecutive "out-of-control" alarms one should get before finding the onset.
        sigma: int, default 2
                How many standard deviations is the mu_d variable of the CUSUM function?
        title: str, default None
                A custom title for your plot.
        save: bool
                Wether or not to save the figure in local "plots" directory. Only works if plot=True.
        savepath: str, default None
                The path to which to save the plot, if not the default "plots".
        yscale: str
                The scale of the y-axis, either 'log' or 'linear'.
        ylim: 2-tuple, default None
                Defines the lower and upper bounds for the plots vertical axis.
        erase: 2-tuple, default None
                If there are spikes in the background one wishes to omit, set the threshold value and ending point for ignoring those
                values here. Example: [1000, '2022-05-10 12:00'] discards all values 1000 or above until 2022-05-10 noon.
        xlim: 2-tuple (str, str), default None
                Pandas-compatible dtateime strings. Defines the start and end for the horizontal axis of the plot.
        show_stats : bool, default True
                Switch to show the mean of medians onset and confidence intervals. Also shows the earliest possible onset (first nonzero CUSUM value)
        diagnostics: bool, default False
                Diagnostic tool allows for plotting the CUSUM function, and printing many 'behind the curtains' variables for the user to see.
        plot: bool, default True
                Switch to produce a plot. 

        Returns:
        ---------
        onset_stats: list
                A list of onset statistics. [background_mean, background_mean+sigma*std, k_round, normed_values, CUSUM_function, Timestamp_of_Onset]
        flux_series: Pandas Series
                The series used for producing the plot and running the onset analysis.
        """

        spacecraft = self.spacecraft.upper()

        # Set the viewing even in the case it was not input by a user
        viewing = viewing if viewing else self.viewing

        # Logic of this check:
        # If viewing was set (meaning viewing evaluates to True) and Onset.check_viewing() returns None 
        # (which evaluates to False), then viewing must be invalid.
        if viewing and not self.check_viewing(returns=True):
            raise ValueError("Invalid viewing direction!")

        color_dict = {
            'onset_time': '#e41a1c',
            'bg_mean': '#e41a1c',
            'flux_peak': '#1a1682',
            'bg': '#de8585'
        }

        # This check was initially transforming the 'channels' integer to a tuple of len==1, but that
        # raised a ValueError with solo/ept. However, a list of len==1 is somehow okay.
        # Also save the channel to a class attribute for later use
        if isinstance(channels, (int, np.int64)):
            self.last_used_channel = int(channels)
            channels = [int(channels)]

        if isinstance(background_range,(list,tuple)):
            background_start, background_end = pd.to_datetime(background_range[0]), pd.to_datetime(background_range[1])
        
        if isinstance(background_range, BootstrapWindow):
            background_start, background_end = background_range.start, background_range.end
            self.background = background_range

        flux_series, en_channel_string = self.choose_flux_series(channels=channels, viewing=viewing)

        # Glitches from the data should really be erased BEFORE resampling data
        if erase is not None:
            flux_series, glitches = erase_glitches(flux_series, erase[0], erase[1])
        else:
            glitches = None

        # Resample data if requested
        if resample is not None:
            flux_series = util.resample_df(flux_series, resample)

        # Here just to make sure, check that the flux series is cut correctly
        if len(flux_series) == 0:
            raise Exception("Intensity data selection is empty! Check that plot limits are correct.")

        # The series is indexed by time
        time = flux_series.index

        # Saves the intensity time series to a class attribute, handy if debugging or such is needed
        self.flux_series = flux_series

        # Onset is not yet found
        onset_found = False

        # Background stats are a tuple of (mean, standard_deviation)
        background_stats = calculate_mean_and_std(background_start, background_end, flux_series)

        # Also check that the background parameters are calculated using a reasonable background_window
        if background_stats[0] == 0 or np.isnan(background_stats[0]):
            background_warning = f"WARNING: background mean is {background_stats[0]}! If this was unintended, it may have been caused by an incorrect choice of background window."
            warnings.warn(message=background_warning)

        # Calculate the length of cusum_window in terms of data points from the time resolution
        time_reso = get_time_reso(flux_series)
        cusum_window = calculate_cusum_window(time_reso, cusum_minutes)

        # The function finds the onset and returns a list of stats related to the onset
        # onset_stats = [ma, md, k_round, h, norm_channel, cusum, onset_time]
        onset_stats = onset_determination_v2(background_stats, flux_series, cusum_window, background_end, sigma)
        
        # If the timestamp of onset is not NaT, then onset was found
        if not isinstance(onset_stats[-1],pd._libs.tslibs.nattype.NaTType):
            onset_found = True

        # Prints out useful information if diagnostics is enabled
        if diagnostics:
            print(f"Cusum_window, {cusum_minutes} minutes = {cusum_window} datapoints")
            print(f"onset time: {onset_stats[-1]}")
            print(f"mu and sigma of background intensity: \n{np.round(background_stats[0],2)}, {np.round(background_stats[1],2)}")

        # --Only plotting related code from this line onward ->
        
        # Before starting the plot, save the original rcParam options and update to new ones
        original_rcparams = self.save_and_update_rcparams("onset_tool")

        fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

        # Setting the x-axis limits
        if xlim is None:
            xlim = [time[0], time[-1]]
        else:

            # Check that xlim makes sense
            if xlim[0] == xlim[1]:
                raise Exception("xlim[0] and xlim[1] cannot be the same time!")

            xlim[0], xlim[1] = pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1])

        # Just make sure that the user did not mess up their x-axis limits and the background for the cusum function
        if (background_range is not None) and (xlim is not None):
            # Check if background is separated from plot range by over a day, issue a warning if so, but don't
            if (background_start < xlim[0] - datetime.timedelta(days=1) and background_start < xlim[1] - datetime.timedelta(days=1)) or \
               (background_end > xlim[0] + datetime.timedelta(days=1) and background_end > xlim[1] + datetime.timedelta(days=1)):
                background_warning = "NOTICE that your background_range is separated from plot_range by over a day.\nIf this was intentional you may ignore this warning."
                warnings.warn(message=background_warning)

        # Figure limits and scale
        flux_in_plot = flux_series.loc[(flux_series.index < xlim[-1])&(flux_series.index >= xlim[0])]
        ylim = set_fig_ylimits(ax=ax, flux_series=flux_in_plot, ylim=ylim)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_yscale(yscale)

        # The measurements. kw where dictates where the step happens -> "mid" for at the middle of the bin
        ax.step(time, flux_series.values, c="C0", where="mid")

        if erase is not None:
            ax.scatter(glitches.index, glitches.values, s=3, c='maroon')
            if diagnostics:
                print("omitted values:")
                print(glitches)

        # These are for bughunting
        if diagnostics:
            ax.step(time, onset_stats[-3], color="darkgreen", label=r"$I_{z-score}$")
            ax.step(time, onset_stats[-2], color="maroon", label="CUSUM")
            ax.axhline(y=onset_stats[2], ls="--", color='k', label='k')
            ax.axhline(y=onset_stats[3], ls="-.", color='k', label='h')

        # Onset time
        if onset_found:
            ax.axvline(x=onset_stats[-1], linewidth=1.5, color=color_dict["onset_time"], linestyle= '-', 
                        label="onset time")

            # Textbox indicating onset time
            plabel = AnchoredText(f" {str(onset_stats[-1])[:19]} ", prop=dict(size=13), frameon=True, loc=(4) )
            plabel.patch.set_boxstyle("round, pad=0., rounding_size=0.2")
            plabel.patch.set_linewidth(2.0)
            ax.add_artist(plabel)

        # Background mean
        ax.axhline(y=onset_stats[0], linewidth=2, color=color_dict["bg_mean"], linestyle= "--", 
                    label=r'$\mu_{a}$ for background intensity')

        # Background mean + n*std
        ax.axhline(y=onset_stats[1], linewidth=2, color=color_dict["bg_mean"], linestyle= ':',
                    label=r'$\mu_{d}$ for background intensity')

        # Background shaded area
        if isinstance(background_range, BootstrapWindow):
            background_range.draw_background(ax=ax)
        else:
            ax.axvspan(xmin=background_start, xmax=background_end,
                        color=color_dict["bg"], label="Background", alpha=0.15)

        # Plot the earliest possible onset time, mean of medians and confidence intervals if they're found
        if show_stats:

            # Mean of median onsets accompanied by their confidence intervals
            if self.mean_of_medians_onset_acquired:

                ax.axvline(x=self.mean_of_medians_onset, linewidth=1.5, color="magenta", linestyle='-',
                        label="mean of medians onset")

                ax.axvspan(xmin=self.conf1_low, xmax=self.conf1_high, color="red", alpha=0.3, label=r"1~\sigma")
                ax.axvspan(xmin=self.conf2_low, xmax=self.conf2_high, color="blue", alpha=0.3, label=r"2~\sigma")


        ax.set_xlabel("Time", fontsize=20)
        ax.set_ylabel(f"Intensity [1/(cm^2 sr s MeV)]", fontsize=20)

        # Tickmarks, their size etc...
        ax.tick_params(which='major', length=6, width=2, labelsize=17)
        ax.tick_params(which='minor', length=5, width=1.4)

        # Date tick locator and formatter
        ax.xaxis_date()
        #ax.xaxis.set_major_locator(ticker.AutoLocator()) # ticker.MaxNLocator(9)
        utc_dt_format1 = DateFormatter('%H:%M \n%Y-%m-%d')
        ax.xaxis.set_major_formatter(utc_dt_format1)

        # Setting the title
        if title is None:

            # Choosing the particle species identifier for the title
            if self.species in ["electron", 'e']:
                s_identifier = 'electrons'
            if self.species in ["proton", 'p', 'H']:
                s_identifier = 'protons'
            if self.species in ["ion", 'i']:
                s_identifier = 'ions'

            time_reso_str = self.get_time_resolution_str(resample=resample)

            if viewing:
                ax.set_title(f"{spacecraft}/{self.sensor.upper()}({viewing}) {en_channel_string} {s_identifier}\n{time_reso_str}")
            else:
                ax.set_title(f"{spacecraft}/{self.sensor.upper()} {en_channel_string} {s_identifier}\n{time_reso_str}")

        else:
            ax.set_title(title)

        if diagnostics:
            ax.legend(loc="best")

        # Attach the figure to class attribute even if not saving the figure
        self.fig, self.ax = fig, ax

        if save:
            if savepath is None:
                savepath = CURRENT_PATH

            if spacecraft.lower() in ["bepicolombo", "bepi"]:
                plt.savefig(f"{savepath}{os.sep}{self.spacecraft}{self.sensor}_side{viewing}_{self.species}_{channels}_onset.png", transparent=False,
                        facecolor='white', bbox_inches='tight')
            elif viewing != "" and viewing is not None:
                plt.savefig(f"{savepath}{os.sep}{self.spacecraft}{self.sensor}_{viewing.lower()}_{self.species}_{channels}_onset.png", transparent=False,
                        facecolor='white', bbox_inches='tight')
            else:
                plt.savefig(f"{savepath}{os.sep}{self.spacecraft}{self.sensor}_{self.species}_{channels[:]}_onset.png", transparent=False,
                        facecolor='white', bbox_inches='tight')

        if plot:
            plt.show()
        else:
            plt.close()

        return onset_stats, flux_series


    def plot_all_channels(self, viewing:str=None, resample:str=None, omit:list=None, cmap:str="twilight_shifted", xlim=None, title:str=None,
                          save=False, savepath=None):
        """
        Plots by default all the energy channels of the currently chosen instrument in the same time series plot.
        
        Parameters:
        -----------
        viewing : {str}, optional
                    The viewing direction of the instrument. If not provided then use the currently 
                    active viewing direction from self.viewing.
        resample : {str}, optional
                    The time-averaging for the time series. Use a pandas-compatible time string, e.g., '1min' or '600s'
        omit : {list, tuple}, optional
                    A list or a tuple of the channels to forgo plotting.
        cmap : {str}, optional
                    The colormap used to plot the channels.
        xlim : {tuple or list with __len__ == 2}, optional
                    The x-axis limits of the figure.
        title : {str}, optional
                    A custom title for the figure.
        save : {bool}, optional
                    Boolean save switch, default False.
        savepath : {str}, optional
                    Either relative or absolute path leading to the directory where the plot should be saved to.
        
        Returns:
        -----------
        fig, ax : The figure and the axis objects.
        """

        # Choose viewing
        viewing = viewing if viewing else self.viewing

        # Collect all channels to a list
        all_channels = self.get_all_channels()

        # Choose which channels to plot, by default plot everything
        plotted_channels = all_channels if not omit else list(set(all_channels) - set(omit))

        # Get a colormap for the different channels.
        colors = plt.get_cmap(cmap, len(plotted_channels))

        # Initializing the figure and axes
        fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

        # Before starting the plot, save the original rcParam options and update to new ones
        # original_rcparams = self.save_and_update_rcparams("onset_tool")
        # This seemed to sometimes get figure visual parameters messed up, such as fontsize of the title and figure size

        # Plotting loop
        for i, channel in enumerate(plotted_channels):

            # Getting a time series from the object's dataframe
            intensity, en_channel_string = self.choose_flux_series(channels=[int(channel)], viewing=viewing)

            if isinstance(en_channel_string,(list, np.ndarray)):
                en_channel_string = en_channel_string[0]

            # If time-averaging, do it here before plotting
            if resample:
                intensity = util.resample_df(intensity, resample)

            # Plotting a channel. kw where=="mid" to ensure that step happens in the middle of the time bin
            ax.step(intensity.index, intensity.values, where="mid", label=en_channel_string, color=colors(i))


        # Setting the x-axis limits
        if not xlim:
            xlim = [intensity.index[0], intensity.index[-1]]
        else:

            # Check that xlim makes sense
            if xlim[0] == xlim[1]:
                raise Exception("xlim[0] and xlim[1] cannot be the same time!")

            xlim[0], xlim[1] = pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1])
        
        ax.set_xlim(xlim)

        ax.set_yscale("log")

        ax.set_xlabel("Time", fontsize=20)
        ax.set_ylabel(f"Intensity [1/(cm^2 sr s MeV)]", fontsize=20)

        # Tickmarks, their size etc...
        ax.tick_params(which='major', length=6, width=2, labelsize=17)
        ax.tick_params(which='minor', length=5, width=1.4)

        # Date tick locator and formatter
        ax.xaxis_date()
        #ax.xaxis.set_major_locator(ticker.AutoLocator()) # ticker.MaxNLocator(9)
        utc_dt_format1 = DateFormatter('%H:%M \n%Y-%m-%d')
        ax.xaxis.set_major_formatter(utc_dt_format1)

        ax.legend(loc=3, bbox_to_anchor=(1.0, 0.01), prop={'size': 16})

        # Setting the title
        if title is None:

            # Choosing the particle species identifier for the title
            if self.species in ["electron", 'e']:
                s_identifier = 'electrons'
            if self.species in ["proton", 'p', 'H']:
                s_identifier = 'protons'
            if self.species in ["ion", 'i']:
                s_identifier = 'ions'

            if viewing:
                ax.set_title(f"{self.spacecraft.upper()}/{self.sensor.upper()} {s_identifier}\n"
                        f"{resample} averaging, viewing: "
                        f"{viewing.upper()}")
            else:
                ax.set_title(f"{self.spacecraft.upper()}/{self.sensor.upper()} {s_identifier}\n"
                        f"{resample} averaging")
        
        else:
            ax.set_title(title)

        if save:
            if savepath is None:
                savepath = CURRENT_PATH

            if self.spacecraft.lower() in ['bepicolombo', 'bepi']:
                fig.savefig(f"{savepath}{os.sep}{self.spacecraft}_{self.sensor}_side{viewing}_{self.species}_all_channels.png", transparent=False,
                        facecolor='white', bbox_inches='tight')
            elif viewing != "" and viewing:
                fig.savefig(f"{savepath}{os.sep}{self.spacecraft}_{self.sensor}_{viewing.lower()}_{self.species}_all_channels.png", transparent=False,
                        facecolor='white', bbox_inches='tight')
            else:
                fig.savefig(f"{savepath}{os.sep}{self.spacecraft}_{self.sensor}_{self.species}_all_channels.png", transparent=False,
                        facecolor='white', bbox_inches='tight')

        return fig, ax
        


    def statistic_onset(self, channels, Window, viewing:str, resample:str=None, erase:tuple=None, sample_size:float=None, cusum_minutes:int=None, 
                        small_windows:str=None, offset_origins:bool=True, detrend=True, sigma_multiplier=2):
        """
        This method looks at a particular averaging window with length <windowlen>, and draws from it
        points <n_bstraps> times. From these <n_bstraps> different distributions of measurements, it 
        calculates the mean and standard deviation for each distribution. Using the acquired distribution of
        different means and standard deviations, it produces a distribution of expected onset times. Finally
        the method moves the averaging window forward if windows has n_shifts>1.
        
        The procedure described above is repeated <n_windows> times, yielding <n_windows>*<n_bstraps> onset times.
        
        Parameters:
        ------------
        channels : int or list
                    Channel or channels that the onset is searched from
        Window: BootstrapWindow object
                    A BootstrapWindow object that holds information about averaging window parameters
        viewing : str
                    The viewing direction of the instrument
        resample : str
                    A pandas-compatible time string representing the averaging of data
        erase : tuple(float, str)
                    The maximum acceptable intensity value and the timestamp at which larger values are accepted
        sample_size : {float}, optional
                    The proportion of the ensemble of data points inside a window that are drawn for the bootstrapping method
        cusum_minutes : {int}, optional 
                    An integer stating the amount of minutes that threshold exceeding intensity must consistently be observed 
                    before identifying an onset. If not provided, use a set of 2,4,8 and 16 times the current data resolution
        small_windows : str, default None
                    Choose either 'random' or 'equal' placement of the subwindows. Randomly placed windows can overlap,
                    while equally placed windows start where the previous ended.
        offset_origins : {bool}, default True
                    Boolean switch to control if onset times can be found on a continuous scale or only at intervals allowed by the
                    resolution of the data. By default origin is being offset, i.e., onset can be found at 1-minute resolution
                    regardless of the integration time.
        detrend : bool, default True
                    Controls if the onset times are 'de-trended', i.e., shifted forward in time by 0.5*time_resolution - 30s. Does not 
                    apply to 1min data.
        sigma_multiplier : {int, float} default 2
                    The multiplier for the $\mu_{d}$ variable in the CUSUM method.

        Returns:
        -----------
        mean_onset: a timestamp indicating the mean of all onset times found
        std_onset: the standard deviation of onset times
        most_common_val: the most common onset time found
        percent_mode: the prevalence of the most common onset time (fraction of all onset times)
        onset_list: a list containing all onset times in the order they were calculated by this function
        """
        # Bootstrapping:
        # https://towardsdatascience.com/calculating-confidence-interval-with-bootstrapping-872c657c058d

        # Channels are considered a list in the data loader, that's why this check is done here
        if isinstance(channels,int):
            self.last_used_channel = channels
            channels = [channels]

        # Do not search for onset earlier than this point
        big_window_end = Window.end

        # Before going to pick the correct flux_series, check that viewing is reasonable.
        if viewing and not self.check_viewing(returns=True):
            raise ValueError("Invalid viewing direction!")

        # Choose the right intensity time series according to channel and viewing direction.
        # Also remember which channel was examined most recently.
        flux_series, self.recently_examined_channel = self.choose_flux_series(channels, viewing)

        # By default there will be a list containing timeseries indices of varying origins of offset
        if offset_origins and resample:

            # So far we will only allow origins to be offset if the data is a multiple of minutes resolution
            if resample[-3:] == "min":

                # This integer is one larger than the offset will ever go
                data_res = int(resample[:-3])

                # There will be a set amount of varying offsets with linear intervals to the time indices of flux_series.
                # offsets are plain integers
                offsets = np.arange(data_res)

                # Using the number of offsets, compile a list of copies of flux_series
                list_of_series = [flux_series if i==0 else flux_series.copy() for i in offsets]

                # Produces a list of indices of varying offsets. The first one in the list has offset=0, which are 
                # just the original indices. These indices will replace the indices of the series in the list.
                list_of_indices = [flux_series.index + to_offset(f"{offset}min") for offset in offsets]

                # These are a random numbers to access any of the indices in the list of indices.
                # Calculated beforehand to save computation time.
                random_choices = np.random.randint(data_res, size=Window.bootstraps)
            
            else:
                list_of_series = [flux_series]
                list_of_indices = [flux_series.index]
                random_choices = np.zeros(Window.bootstraps, dtype=int)

        # If we don't offset origin of timeseries, then just init list_of_series, list_of_indices and random_choises
        # with default values for a single unaltered flux_series
        else:

            list_of_series = [flux_series]
            list_of_indices = [flux_series.index]
            random_choices = np.zeros(Window.bootstraps, dtype=int)


        # Replace the indices of the copies of the series. Each new series will have its indices shifted
        # forward by one minute (in the current version).
        # After the possible shifting of indices, first erase possible erroneous data, only after 
        # resample it if need be.
        for i, series in enumerate(list_of_series):

            # Shifting of indices forward
            series.index = list_of_indices[i]

            # Erasing erroneous values
            if erase:
                series, _ = erase_glitches(series, erase[0], erase[1])

            # Resampling of data to a differrent resolution
            if resample:
                series = util.resample_df(series, resample)

            # Finally placing the series back to the list at the correct position
            list_of_series[i] = series


        # These apply regardless of which series we choose, as they all have the same resolution and hence cusum_window
        if not resample:
            time_reso = f"{int(self.get_minimum_cadence().seconds/60)} min" if self.get_minimum_cadence().seconds > 59 else f"{int(self.get_minimum_cadence().seconds)} s"
        else:
            time_reso = get_time_reso(list_of_series[0])

        # If cusum_window was not explicitly stated, use a set of multiples of the time resolution as 
        # a set of cusum_windows
        if cusum_minutes:
            cusum_windows = [calculate_cusum_window(time_reso, cusum_minutes)]
        else:
            cusum_window_resolution_multipliers = (2,4,8,16,32)
            if time_reso[-3:]=="min":
                cusum_minutes_list = [c*int(time_reso[:-3]) for c in cusum_window_resolution_multipliers]
            else:
                # For now just go with a fixed number of minutes if resolution is less than a minute
                cusum_minutes_list = [c for c in cusum_window_resolution_multipliers]
                # cusum_minutes_list = [int(c*int(time_reso[:-1])/60) for c in cusum_window_resolution_multipliers]
            cusum_windows = [calculate_cusum_window(time_reso, cusum_minutes) for cusum_minutes in cusum_minutes_list]

        # Initialize necessary lists to collect data about onset time and the corresponding background parameters
        onset_list = []
        mus_list = []
        sigmas_list = []

        # Small windows will initialize a set of randomly placed smaller windows inside the chosen (input) 
        # BootstrapWindow object. The small windows will all have n_shifts=0, and bootstraps equal to the
        # amount the original Window had.
        if small_windows:

            # Before doing anything, check that small_windows is the type it is meant to be
            if small_windows not in ("random", "equal"):
                raise Exception(f"small_windows = {small_windows} is not a valid input! Choose either 'random' or 'equal'.")

            # First check that there are shifts in the Window, otherwise it makes no sense to even
            # call subwindows()
            if Window.n_shifts < 1:
                raise Exception(f"The number of small windows can't be {Window.n_shifts}")

            # Produce a list of little windows to loop over and apply bootstrapping
            windowlist = subwindows(Window=Window, window_placement=small_windows)

            # Set the class-attached list to an empty list, in order to stop new windows accumulating there every time
            # this method is run. This is essentially flushing the list clean.
            self.list_of_bootstrap_windows = []
            self.subwindows_plotted = False

        # If no small windows, just loop over one window
        else:
            windowlist = [Window]

        # Controls if a warning is printed in the event the background window is less than 100 data points. Hard coded to False for now
        prints_warning = False

        # Nested loops :(
        # But in reality either the middle or the outer loop is only run once
        for window in windowlist:

            # Shifts here is either the times one big background window is shifted towards the onset of the event,
            # or it is the amount of small little windows inside the larger background window
            shifts = window.n_shifts if window.n_shifts > 0 else 0

            for _ in range(shifts+1):

                # Init arrays to collect used background parameters
                mus, sigmas = np.array([]), np.array([])

                # Loop through the mus and sigmas of the current window, and find onsets 
                for j in range(window.bootstraps):

                    # Pick a number from random_choices to access a copy of flux_series in the list_of_series
                    # In the case we don't offset any of the series, there will be only one instance of flux_series in the
                    # list, and random_choices will be an array of only zeroes
                    chosen_series = list_of_series[random_choices[j]]

                    # Calculate background parameters (mean and std), and append them to their respective arrays
                    mu, sigma = sample_mean_and_std(window.start, window.end, chosen_series, sample_size=sample_size, prints_warning=prints_warning)
                    mus = np.append(mus, mu)
                    sigmas = np.append(sigmas,sigma)

                    # Choose the cusum_window to use for this iteration of the run
                    cusum_window = cusum_windows[j%len(cusum_windows)]

                    # Find an onset and save it into onset_series. Use the chosen series from the list of series
                    onset_i = onset_determination_v2([mu, sigma], chosen_series, cusum_window, big_window_end, sigma_multiplier=sigma_multiplier)
                    onset_list.append(onset_i[-1])


                # Append the arrays to the list of all bootstrapped background params
                mus_list.append(mus)
                sigmas_list.append(sigmas)

                # Finally move the averaging window forward one hour before repeating, or in the case of small_windows,
                # add the window to the class list of windows 
                if not small_windows:
                    window.move_window(hours=1)
                else:
                    self.add_bootstrap_window(window=window)

        # After exiting the loop, return the background window to its initialized state to not cause unintended effects later
        Window.load_state(which="init")

        # Sort the list of onsets to chronological order
        onset_list.sort()

        # If de-trending is to be done, it must be done here. De-trending can only be applied if the data was resampled
        if detrend and resample:
            rs_final_index = -3 if resample[-3:] == "min" else -1
            dt_shift = f"{int(30*int(resample[:rs_final_index]))+30}s"
            onset_list = detrend_onsets(onset_list, shift=dt_shift)

        # onset_series needs to be converted to DatetimeIndex, so that the elements' mean can be calculated
        onset_indices = pd.DatetimeIndex(onset_list)
        mean_onset = onset_indices.mean()
        std_onset = np.std(onset_indices)

        # unique_vals() returns a list of the unique appearances of the onsets and their respective counts
        unique_onsets, counts = unique_vals(onset_indices)

        # First calculate the relative abundances of each unique onset
        percent_modes = np.array([count/np.nansum(counts) for count in counts])

        # The final entry of counts is the number of NaTs in the input onset_indices. Hence, it doesn't make sense
        # to keep the redundant percentage aboard. Omit the last entry to avoid errors later. No information is 
        # lost here in doing this.
        percent_modes = percent_modes[:-1]

        # Sort the percentages and unique onsets in ascending order
        percent_indices = percent_modes.argsort() # numpy array.argsort() returns a list of indices of values in ascending order

        # If absolutely no onset was found, then percent_indices will be an empty list
        if len(percent_indices) == 0:

            mean_onset = pd.NaT
            median_onset = pd.NaT
            std_onset = pd.NaT
            most_common_val = (pd.NaT, np.nan)
            onset_list = [pd.NaT]
            unique_onsets = [pd.NaT]
            confidence_intervals = [(pd.NaT, pd.NaT), (pd.NaT, pd.NaT)]

        # Here the normal case where onsets are found is treated
        else:

            unique_onsets = unique_onsets[percent_indices]
            percent_modes = percent_modes[percent_indices]

            # Unique onsets hold tuples of onset-percentage pairs in ascending order (= least likely onset first and most likely last)
            unique_onsets = np.array([(onset, percent_modes[i]) for i, onset in enumerate(unique_onsets)])

            # The most "popular" onset time and its mode are the last entries in the arrays
            most_common_val = (unique_onsets[-1,0], unique_onsets[-1,1])

            # The median onset is the middlemost onset in the distribution
            median_onset =  onset_list[len(onset_list)//2]

            # Also calculate 1-sigma and 2-sigma confidence intervals for the onset distribution
            confidence_intervals = self.get_distribution_percentiles(onset_list=onset_list, percentiles=[(15.89,84.1), (2.3,97.7)])

        # Attach the statistics to class attributes for easier handling. The first entry of list_of_series is the 
        # original flux_series without shifting any indices
        self.flux_series = list_of_series[0]
        bootstrap_onset_statistics = {
                                    "mean_onset" : mean_onset,
                                    "median_onset" : median_onset,
                                    "std_onset" : std_onset,
                                    "most_likely_onset" : most_common_val,
                                    "onset_list" : onset_list,
                                    "unique_onsets" : unique_onsets,
                                    "1-sigma_confidence_interval" : confidence_intervals[0],
                                    "2-sigma_confidence_interval" : confidence_intervals[1]
                                    }

        mu_and_sigma_distributions = {
                                    "mus_list" : mus_list,
                                    "sigmas_list" : sigmas_list
                                    }

        self.bootstrap_onset_statistics = bootstrap_onset_statistics
        self.mu_and_sigma_distributions = mu_and_sigma_distributions

        return bootstrap_onset_statistics, mu_and_sigma_distributions


    def show_onset_distribution(self, show_background=True, ylim=None, xlim=None):
        """
        Displays all the unique onsets found with statistic_onset() -method in a single plot. The mode onset, that is
        the most common out of the distribution, will be shown as a solid line. All other unique onset times are shown
        in a dashed line, with the shade of their color indicating their relative frequency in the distribution.

        Note that statistic_onset() has to be run before this method. Otherwise KeyError will be raised.

        show_background : {bool}, optional
                            Boolean switch to show the background used in the plot.
        """

        rcParams["font.size"] = 20

        flux_series = self.flux_series
        most_likely_onset = self.bootstrap_onset_statistics["most_likely_onset"]
        onsets = self.bootstrap_onset_statistics["unique_onsets"]

        # Create a colormap for different shades of red for the onsets. Create the map with the amount of onsets+2, so that
        # the first and the final color are left unused. They are basically white and a very very dark red.
        cmap = plt.get_cmap("Reds", len(onsets)+2)

        # Creating the figure and axes
        fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

        # x- and y-axis settings
        ylim = set_fig_ylimits(ax=ax, flux_series=flux_series, ylim=ylim)

        ax.set_yscale("log")
        ax.set_ylabel("Intensity [1/(cm^2 sr s MeV)]")

        if not xlim:
            xlim = (flux_series.index[0], flux_series.index[-1])
        ax.set_xlim(xlim)
        ax.xaxis_date()
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        utc_dt_format1 = DateFormatter('%H:%M \n%m-%d')
        ax.xaxis.set_major_formatter(utc_dt_format1)


        # Plots the intensity data. Keyword 'where' dictates where the step happens. Set 'mid' for the middle of the bin instead of the start.
        ax.step(flux_series.index, flux_series.values, where="mid")

        # Draws the background on the plot
        if show_background:
            self.background.draw_background(ax=ax)

        # Marking the onsets as vertical lines of varying shades of red
        for i, onset in enumerate(onsets):

            # Not the most likely onset
            if onset[0] != most_likely_onset[0]:
                linestyle = (5, (10, 3)) if onset[1] > 0.2 else (0, (5, 5))
                ax.axvline(x=onset[0], c=cmap(i+1), ls=linestyle, label=f"{str(onset[0])[11:19]} ({np.round(onset[1]*100,2):.2f} %)")

            # The most likely onset
            # I'm accessing the i+1th color value in the map because I never want the first value (that is white in the standard Reds colormap)
            else:
                ax.axvline(x=onset[0], c=cmap(i+1), label=f"{str(onset[0])[11:19]} ({np.round(onset[1]*100,2):.2f} %)") 

        # Legend and title for the figure
        ax.legend(loc=10, bbox_to_anchor=(1.0, 0.95), prop={'size': 12})
        ax.set_title("Onset distribution")

        plt.show()


    def show_onset_statistics(self, percentiles=[(15.89,84.1),(2.3,97.7)], show_background=True, xlim=None, ylim=None):
        """
        Shows the median, mode, mean and confidence intervals for the distribution of onsets got from statistic_onset()

        Parameters:
        -----------
        percentiles : {tuple, list}, optional
                    Choose which kind of percentiles of the distribution to display
        show_bacground : {bool}, optional
                    Boolean switch to show the background on the plot.
        """

        # Take the 1-std and 2-std limits from a normal distribution (default)
        sigma1, sigma2 = percentiles[0], percentiles[1]

        # Collect the 1-sigma and 2-sigma confidence intervals out of the onset distribution
        confidence_intervals = self.get_distribution_percentiles(onset_list = self.bootstrap_onset_statistics["onset_list"], percentiles=[sigma1,sigma2])

        # Collect the median, mode and mean onsets from the distribution
        onset_median = self.bootstrap_onset_statistics["median_onset"]
        onset_mode = self.bootstrap_onset_statistics["most_likely_onset"][0]
        onset_mean = self.bootstrap_onset_statistics["mean_onset"]

        # This is just for plotting with the latest resolution used
        flux = self.flux_series

        # Plot commands and settings:
        fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

        ax.step(flux.index, flux.values, where="mid")

        # Draw the background
        if show_background:
            self.background.draw_background(ax=ax)

        # Vertical lines for the median, mode and mean of the distributions
        ax.axvline(onset_median, c="red", label="median")
        ax.axvline(onset_mode, c="blue", label="mode")
        ax.axvline(onset_mean, c="purple", label="mean")

        # 1-sigma uncertainty shading
        ax.axvspan(xmin=confidence_intervals[0][0], xmax=confidence_intervals[0][1], color="red", alpha=0.3)

        #2-sigma uncertainty shading
        ax.axvspan(xmin=confidence_intervals[1][0], xmax=confidence_intervals[1][1], color="blue", alpha=0.3)

        # Figure settings
        ax.set_yscale("log")
        ax.set_ylabel("Intensity")
        ax.set_xlabel("Time")

        ylim = set_fig_ylimits(ax=ax, flux_series=flux, ylim=ylim)

        # x-axis settings
        if not xlim:
            xlim = (flux.index[0], flux.index[-1])
        ax.set_xlim(xlim)
        ax.xaxis_date()
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        utc_dt_format1 = DateFormatter('%H:%M \n%m-%d')
        ax.xaxis.set_major_formatter(utc_dt_format1)

        ax.legend()
        plt.show()


    def choose_flux_series(self, channels, viewing) -> tuple:
        """
        Method to choose the correct Series from a dataframe.

        Parameters:
        -----------
        channels : list
        viewing : str

        Returns:
        -----------
        flux_series : pd.Series
        en_channel_string : dict
        """

        self.choose_data(viewing)
        
        if (self.spacecraft[:2].lower() == 'st' and self.sensor == 'sept') \
                or (self.spacecraft.lower() == 'psp' and self.sensor.startswith('isois')) \
                or (self.spacecraft.lower() == 'solo' and self.sensor == 'ept') \
                or (self.spacecraft.lower() == 'solo' and self.sensor == 'het') \
                or (self.spacecraft.lower() == 'wind' and self.sensor == '3dp') \
                or (self.spacecraft.lower() == 'bepi'):
            self.viewing_used = viewing
            self.choose_data(viewing)
        elif (self.spacecraft[:2].lower() == 'st' and self.sensor == 'het'):
            self.viewing_used = ''
        elif (self.spacecraft.lower() == 'soho' and self.sensor == 'erne'):
            self.viewing_used = ''
        elif (self.spacecraft.lower() == 'soho' and self.sensor == 'ephin'):
            self.viewing_used = ''

        if self.spacecraft == 'solo':

            if self.sensor == 'het':

                if self.species in ['p', 'i']:

                    df_flux, en_channel_string =\
                        self.calc_av_en_flux_HET(self.current_df_i,
                                                 self.current_energies,
                                                 channels)
                elif self.species == 'e':

                    df_flux, en_channel_string =\
                        self.calc_av_en_flux_HET(self.current_df_e,
                                                 self.current_energies,
                                                 channels)

            elif self.sensor == 'ept':

                if self.species in ['p', 'i']:

                    df_flux, en_channel_string =\
                        self.calc_av_en_flux_EPT(self.current_df_i,
                                                 self.current_energies,
                                                 channels)
                elif self.species == 'e':

                    df_flux, en_channel_string =\
                        self.calc_av_en_flux_EPT(self.current_df_e,
                                                 self.current_energies,
                                                 channels)

            else:
                invalid_sensor_msg = "Invalid sensor!"
                raise Exception(invalid_sensor_msg)

        if self.spacecraft[:2] == 'st':

            # Super ugly implementation, but easiest to just wrap both sept and het calculators
            # in try block. KeyError is caused by an invalid channel choice.
            try:

                if self.sensor == 'het':

                    if self.species in ['p', 'i']:

                        df_flux, en_channel_string =\
                            calc_av_en_flux_ST_HET(self.current_df_i,
                                                   self.current_energies['channels_dict_df_p'],
                                                   channels,
                                                   species='p')
                    elif self.species == 'e':

                        df_flux, en_channel_string =\
                            calc_av_en_flux_ST_HET(self.current_df_e,
                                                   self.current_energies['channels_dict_df_e'],
                                                   channels,
                                                   species='e')

                elif self.sensor == 'sept':

                    if self.species in ['p', 'i']:

                        df_flux, en_channel_string =\
                            calc_av_en_flux_SEPT(self.current_df_i,
                                                 self.current_i_energies,
                                                 channels)
                    elif self.species == 'e':

                        df_flux, en_channel_string =\
                            calc_av_en_flux_SEPT(self.current_df_e,
                                                 self.current_e_energies,
                                                 channels)

            except KeyError:
                raise Exception(f"{channels} is an invalid channel or a combination of channels!")

        if self.spacecraft == 'soho':

            # A KeyError here is caused by invalid channel
            try:

                if self.sensor == 'erne':

                    if self.species in ['p', 'i']:

                        df_flux, en_channel_string =\
                            calc_av_en_flux_ERNE(self.current_df_i,
                                                 self.current_energies['channels_dict_df_p'],
                                                 channels,
                                                 species='p',
                                                 sensor='HET')

                if self.sensor == 'ephin':
                    # convert single-element "channels" list to integer
                    if type(channels) == list:
                        if len(channels) == 1:
                            channels = channels[0]
                        else:
                            print("No multi-channel support for SOHO/EPHIN included yet! Select only one single channel.")
                    if self.species == 'e':
                        df_flux = self.current_df_e[f'E{channels}']
                        en_channel_string = self.current_energies[f'E{channels}']

            except KeyError:
                raise Exception(f"{channels} is an invalid channel or a combination of channels!")

        if self.spacecraft == 'wind':
            if self.sensor == '3dp':
                # convert single-element "channels" list to integer
                if type(channels) == list:
                    if len(channels) == 1:
                        channels = channels[0]
                    else:
                        print("No multi-channel support for Wind/3DP included yet! Select only one single channel.")
                if self.species in ['p', 'i']:
                    if viewing != "omnidirectional":
                        df_flux = self.current_df_i.filter(like=f"FLUX_E{channels}_P{self.viewing[-1]}")
                    else:
                        df_flux = self.current_df_i.filter(like=f'FLUX_{channels}')
                    # extract pd.Series for further use:
                    df_flux = df_flux[df_flux.columns[0]]
                    # change flux units from '#/cm2-ster-eV-sec' to '#/cm2-ster-MeV-sec'
                    df_flux = df_flux*1e6
                    en_channel_string = self.current_i_energies['channels_dict_df']['Bins_Text'][f'ENERGY_{channels}']
                elif self.species == 'e':
                    if viewing != "omnidirectional":
                        df_flux = self.current_df_e[f"FLUX_E{channels}_P{self.viewing[-1]}"]
                        #df_flux = self.current_df_e.filter(like=f'FLUX_E{channels}')
                    else:
                        df_flux = self.current_df_e[f"FLUX_{channels}"]
                        #df_flux = self.current_df_e.filter(like=f'FLUX_{channels}')
                    # extract pd.Series for further use:
                    #df_flux = df_flux[df_flux.columns[0]]
                    # change flux units from '#/cm2-ster-eV-sec' to '#/cm2-ster-MeV-sec'
                    df_flux = df_flux*1e6
                    en_channel_string = self.current_e_energies['channels_dict_df']['Bins_Text'][f'ENERGY_{channels}']

        if self.spacecraft.lower() == 'bepi':
            if type(channels) == list:
                if len(channels) == 1:
                    # convert single-element "channels" list to integer
                    channels = channels[0]
                    if self.species == 'e':
                        df_flux = self.current_df_e[f'E{channels}']
                        en_channel_string = self.current_energies['Energy_Bin_str'][f'E{channels}']
                    if self.species in ['p', 'i']:
                        df_flux = self.current_df_i[f'P{channels}']
                        en_channel_string = self.current_energies['Energy_Bin_str'][f'P{channels}']
                else:
                    if self.species == 'e':
                        df_flux, en_channel_string = calc_av_en_flux_sixs(self.current_df_e, channels, self.species)
                    if self.species in ['p', 'i']:
                        df_flux, en_channel_string = calc_av_en_flux_sixs(self.current_df_i, channels, self.species)

        if self.spacecraft.lower() == 'psp':
            if self.sensor.lower() == 'isois-epihi':
                if self.species in ['p', 'i']:
                    # We're using here only the HET instrument of EPIHI (and not LET1 or LET2)
                    df_flux, en_channel_string =\
                        calc_av_en_flux_PSP_EPIHI(df=self.current_df_i,
                                                  energies=self.current_i_energies,
                                                  en_channel=channels,
                                                  species='p',
                                                  instrument='het',
                                                  viewing=viewing.upper())
                if self.species == 'e':
                    # We're using here only the HET instrument of EPIHI (and not LET1 or LET2)
                    df_flux, en_channel_string =\
                        calc_av_en_flux_PSP_EPIHI(df=self.current_df_e,
                                                  energies=self.current_e_energies,
                                                  en_channel=channels,
                                                  species='e',
                                                  instrument='het',
                                                  viewing=viewing.upper())
            if self.sensor.lower() == 'isois-epilo':
                if self.species == 'e':
                    # We're using here only the F channel of EPILO (and not E or G)
                    df_flux, en_channel_string =\
                        calc_av_en_flux_PSP_EPILO(df=self.current_df_e,
                                                  en_dict=self.current_e_energies,
                                                  en_channel=channels,
                                                  species='e',
                                                  mode='pe',
                                                  chan='F',
                                                  viewing=viewing)

        if self.spacecraft == 'solo':
            flux_series = df_flux #[channels]
        if self.spacecraft[:2].lower() == 'st':
            flux_series = df_flux  # [channel]'
        if self.spacecraft.lower() == 'soho':
            flux_series = df_flux  # [channel]
        if self.spacecraft.lower() == 'wind':
            flux_series = df_flux  # [channel]
        if self.spacecraft.lower() == 'psp':
            flux_series = df_flux #[channels]
        if self.spacecraft.lower() == 'bepi':
            flux_series = df_flux  # [channel]


        # Before returning, make sure that the type is pandas series, and not a 1-dimensional dataframe
        if not isinstance(flux_series, pd.core.series.Series):
            flux_series = flux_series.squeeze()

        return flux_series, en_channel_string


    def plot_subwindows(self):
        """
        Shows the subwindows used in the bootstrapping process
        """

        try:
            subwindows_plotted = self.subwindows_plotted
        except AttributeError:
            print("The onset object currently has no subwindows!")
            return 0

        fig, ax = self.fig, self.ax

        if not self.subwindows_plotted:
            for i, window in enumerate(self.list_of_bootstrap_windows):

                # Color the window
                ax.axvspan(xmin=window.start, xmax=window.end,
                            color=self.window_colors[i], alpha=0.25)

                # Draw the window boundaries with thin black lines
                ax.axvline(x=window.start, c='k', lw=1)
                ax.axvline(x=window.end, c='k', lw=1)

            # Keep track of wether subwindows have been plotted, to avoid over-plotting
            self.subwindows_plotted = True

        #plt.show()
        return fig


    def plot_cdf(self, prints=False) -> None:
        """
        Plots the Cumulatice Distribution Function (CDF) for a distribution of onset times.

        Parameters:
        -----------
        prints : bool, default False
                If True, will also print out the median onset time and the confidence intervals.
        """

        # Remember that onset times are sorted according to their probability in ascending order
        onset_timestamps = np.array([pair[0] for pair in self.bootstrap_onset_statistics["unique_onsets"]])
        probabilities = np.array([pair[1] for pair in self.bootstrap_onset_statistics["unique_onsets"]])

        # Here let's sort the onsets and probabilities in temporal order:
        onset_time_indices = onset_timestamps.argsort() # numpy array.argsort() returns a list of indices of values in ascending order

        onset_timestamps = pd.to_datetime(onset_timestamps[onset_time_indices])
        probabilities = probabilities[onset_time_indices]

        # Effectively calculate the cumulative sum of probabilities
        cumulative_probabilities = [np.nansum(probabilities[:i+1]) for i in range(len(probabilities))]
        
        # Get the median, mode, mean and confidence intervals (1sigma and 2sigma of normal distribution) of the onset distribution
        median_onset, _, _, confidence_intervals_all = self.get_distribution_statistics(onset_statistics=self.bootstrap_onset_statistics,
                                                                                    percentiles=[(15.89,84.1), (2.3,97.7)])

        # Flatten the sigma lists for plotting. Look at this list comprehension :D
        confidence_intervals_all = [timestamp for sublist in confidence_intervals_all for timestamp in sublist]

        # Init the figure
        fig, ax = plt.subplots(figsize=(13,7))

        rcParams["font.size"] = 20

        # Settings for axes
        ax.xaxis_date()
        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title("Cumulative Distribution Function")
        hour_minute_format = DateFormatter("%H:%M")
        ax.xaxis.set_major_formatter(hour_minute_format)

        # Plot command(s)
        ax.step(onset_timestamps, cumulative_probabilities, zorder=3, c='k', where="post")

        colors = ("red", "red", "blue", "blue")
        # Horizontal dashed lines for 1-sigma and 2-sigma limits
        for i, num in enumerate((0.159, 0.841, 0.023, 0.977)):
            label = "15.9-84.1 percentiles" if num==0.841 else ("2.3-97.7 percentiles" if num==0.977 else None)
            ax.axhline(y=num, ls="--", c=colors[i], zorder=1, alpha=0.8, label=label)

        # Vertical dashed lines for 1-sigma and 2-sigma limits
        for i, time in enumerate(confidence_intervals_all):
            ax.axvline(x=time, ls="--", c=colors[i], zorder=1, alpha=0.8)

        ax.axvline(x=median_onset, ls="--", c="maroon", zorder=2, alpha=0.8, label="median onset (50th percentile)")
        ax.axhline(y=0.50, ls="--", c="maroon", zorder=1, alpha=0.8)
        
        ax.legend(loc=10, bbox_to_anchor=(1.27, 0.9))

        if prints:
            print(f"The median onset time is: {str(median_onset.time())[:8]}")
            # print(f"The np.percentile()-calculated median onset time (50th percentile) is: {str(median_onset.time())[:8]}")
            display(Markdown(r"Confidence interval for 1- $\sigma$:"))
            print(f"{str(confidence_intervals_all[0].time())[:8]} - {str(confidence_intervals_all[1].time())[:8]}")
            print(" ")
            display(Markdown(r"Confidence interval for 2- $\sigma$:"))
            print(f"{str(confidence_intervals_all[2].time())[:8]} - {str(confidence_intervals_all[3].time())[:8]}")

        plt.show()


    def scatter_histogram(self, x:str="mean", xbinwidth:int=1, ybinwidth:str="1min") -> None:
        """
        A method to plot a scatter and histogram plots of either background mean or background std
        vs. onset time.

        Parameters:
        -----------
        x : str, default 'mean'
            Either 'mean' or 'std'
        xbinwidth : int
            The width of x-axis bins
        ybinwidth: str
            Pandas-compatible time string for the width of onset distribution.
        """

        x_axes_choice = {
            "mean" : self.mu_and_sigma_distributions["mus_list"],
            "std" : self.mu_and_sigma_distributions["sigmas_list"]
        }

        xdata = x_axes_choice[x]
        ydata = self.bootstrap_onset_statistics["onset_list"]

        title = r"Background $\mu$ vs onset time" if x=="mean" else r"Background $\sigma$ vs onset time"
        xlabel = r"Sample mean intensity" if x=="mean" else r"Sample standard deviation"
        ylabel = "Onset time [D HH:MM]"

        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.005

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.2]
        rect_histy = [left + width + spacing, bottom, 0.2, height]

        rcParams["font.size"] = 20

        def scatter_hist(x, y, ax, ax_histx, ax_histy) -> None:

            # No labels for the histograms
            ax_histx.tick_params(axis="x", labelbottom=False)
            ax_histy.tick_params(axis="y", labelleft=False)

            # The scatter plot:
            for i, arr in enumerate(x):
                element = len(arr)
                ax.scatter(arr, y[i*element:element*(1+i)], color=self.window_colors[i], s=10)

            ylims = (np.min(y) - pd.Timedelta(minutes=2), np.max(y) + pd.Timedelta(minutes=2))
            ax.set_ylim(ylims)
            ax.grid(True)

            # Now determine limits by hand:
            binwidth = xbinwidth
            xymax = int(np.max(x))
            xmin = int(np.min(x))

            xbins = np.arange(xmin, xymax, binwidth)
            for i, arr in enumerate(x):
                ax_histx.hist(arr, bins=xbins, color=self.window_colors[i], alpha=0.6)

            half_bin = pd.Timedelta(seconds=60)
            ybins = pd.date_range(start=ylims[0]+half_bin, end=ylims[1]+half_bin, freq=ybinwidth).tolist()

            onset_frequencies = np.ones_like(y)/len(y)

            ax_histy.hist(y, bins=ybins, edgecolor="black", orientation='horizontal', weights=onset_frequencies)

            max_freq = np.nanmax([pair[1] for pair in self.bootstrap_onset_statistics["unique_onsets"]])
            ax_histy.set_xticks([np.round(max_freq,2)/4, np.round(max_freq,2)/2, np.round(max_freq,2)])


        # Start with a square-shaped Figure
        fig = plt.figure(figsize=(12, 12))

        # These axes are for the scatter plot in the middle
        ax = fig.add_axes(rect_scatter)
        ax.yaxis_date()
        
        # These axes are for the histograms
        ax_histx = fig.add_axes(rect_histx, sharex=ax)
        ax_histy = fig.add_axes(rect_histy, sharey=ax)

        # Use the function defined above
        scatter_hist(xdata, ydata, ax, ax_histx, ax_histy)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        fig.suptitle(title, fontsize=20)

        # plt.savefig("histogram_bg_vars")
        plt.show()


    def VDA(self, onset_times=np.array([]), Onset=None, energy:str='gmean', selection=None, 
            yerrs=None, reference:str="mode", title=None, ylim=None, plot=True, guess=None, save=False,
            savepath=None):
        """
        Performs Velocity Dispersion Analysis.

        Parameters:
        -----------
        onset_times : array-like, default empty numpy array
                    List of onset times. If None, automatically acquired with cusum_onset()
        Onset : self@Onset
                    The second onset object if joint-VDA
        energy: str, default 'gmean'
                    Mode of assessing nominal channel energy. 'gmean' or 'upper'
        selection: slice(start,stop) or a pair of slices, default: None
                    Selection of the channels one wishes to include in VDA calculation. If a pair of slices are delivered,
                    consider each slice to apply individually to each instrument respectively.
        yerrs : {array-like}, optional
                    Use provided y-errors instead of the ones in the object's memory.
        reference : {str}, either 'median' or 'mode' (default)
                    Which one to consider as the onset times extracted with the statistical method.
        title : {str}, default None
                    Custom title. If None, generate standard title.
        ylim : tuple/list with len==2, default None
                    the limits of y-axis.
        plot: {bool}, default True
                    Switch to produce plot of the analysis.
        guess : {array-like} of len()==2, default None
                    The initial guess values for the ODR fit. First one's the slope, which is a float, and the second is the
                    intersection point, which is a pandas-compatible datetime string.
        save: {bool}, default False
                    Switch to save the plotted figure. Only works if plot=True.

        Returns:
        ---------
        axes: list[list]
                    A list of all the relevant information to reproduce the analysis and plot.
        """

        from scipy.stats import t as studentt

        import numpy.ma as ma

        spacecraft = self.spacecraft.lower()
        instrument = self.sensor.lower()
        species = self.species.lower()

        if species in ("electron", "ele", 'e'):
            species_title = "electrons"
            m_species = const.m_e.value
        if species in ("ion", 'i', 'h'):
            species_title = "ions"
            m_species = const.m_p.value
        if species in ("protons", "proton", 'p'):
            species_title = "protons"
            m_species = const.m_p.value

        # E=mc^2, a fundamental property of an object with mass
        mass_energy = m_species*C_SQUARED # ~511 keV for electrons

        # Get the energies of each energy channel, to calculate the mean energy of particles and ultimately
        # to get the dimensionless speeds of the particles (v/c = beta).
        # This method returns lower and higher energy bounds in electron volts
        e_min, e_max = self.get_channel_energy_values()

        # SOHO /EPHIN really has only 4 active channels, but 5th one still exists, causing an erroneous amount
        # of channel nominal energies and hence wrong sized mask (5 instead of 4). For now we just discard
        # the final entries fo these lists.
        if spacecraft=="soho" and instrument=="ephin" and len(e_min)==5:
            e_min, e_max = e_min[:-1], e_max[:-1]

        # Get the nominal channel energies, which are by default the geometric means of channel boundaries.
        nominal_energies = calc_chnl_nominal_energy(e_min, e_max, mode=energy)

        # Check here if onset_times were given as an input. If not, use median/mode onset times and yerrs.
        onset_times = np.array(onset_times) if len(onset_times) != 2 else onset_times[0]
        if len(onset_times)==0:

            # Initialize the list of channels according to the sc and instrument used
            channels = self.get_all_channels()

            for ch in channels:
                # The median onset is found as the first entry of any list within the dictionary 'onset_statistics'. If that,
                # however, does not exist for a particular channel, then insert NaT to be masked away later in the method.
                try:
                    onset_times = np.append(onset_times,self.onset_statistics[ch][0])
                except KeyError as e:
                    print(e)
                    onset_times = np.append(onset_times,pd.NaT)

        # Here check if two lists of onset times are being performed VDA upon
        if Onset:

            # First get the energies and nominal energies for the second instrument
            e_min1, e_max1 = Onset.get_channel_energy_values()

            # SOHO /EPHIN really has only 4 active channels, but 5th one still exists, causing an erroneous amount
            # of channel nominal energies and hence wrong sized mask (5 instead of 4). For now we just discard
            # the final entries fo these lists.
            if Onset.spacecraft=="soho" and Onset.sensor=="ephin" and len(e_min1)==5:
                e_min1, e_max1 = e_min1[:-1], e_max1[:-1]

            nominal_energies1 = calc_chnl_nominal_energy(e_min1, e_max1, mode=energy)

             # Check here if onset_times were given as an input. If not, use median/mode onset times and yerrs.
            if len(onset_times) == 2 and isinstance(onset_times[0], (list,tuple, np.ndarray)):

                onset_times1 = pd.to_datetime(onset_times[1])
                onset_times = pd.to_datetime(onset_times[0])

            else:

                # initialize the list of channels according to the sc and instrument used
                channels1 = Onset.get_all_channels()

                # If there was no input for onset times, init an empty array and filll it up with values from the object
                onset_times1 = np.array([])

                for ch in channels1:

                    # The mode onset is found as the first entry of any list within the dictionary 'onset_statistics'. If that,
                    # however, does not exist for a particular channel, then insert NaT to be masked away later in the method.
                    try:
                        onset_times1 = np.append(onset_times1,Onset.onset_statistics[ch][0])
                    except KeyError as e:
                        print(e)
                        onset_times1 = np.append(onset_times1,pd.NaT)


            onset_times_all = np.concatenate((onset_times, onset_times1))


            # Calculate the inverse betas corresponding to the nominal channel energies
            inverse_beta = calculate_inverse_betas(channel_energies=nominal_energies, mass_energy=mass_energy)
            inverse_beta1 = calculate_inverse_betas(channel_energies=nominal_energies1, mass_energy=mass_energy)

            inverse_beta_all = np.concatenate((inverse_beta, inverse_beta1))

            # This is for the fitting function. 8.33 min = light travel time/au
            inverse_beta_corrected = np.array([8.33*60*b for b in inverse_beta_all]) #multiply by 60 -> minutes to seconds

            # Second values of the onset times
            date_in_sec = datetime_to_sec(onset_times=onset_times)
            date_in_sec1 = datetime_to_sec(onset_times=onset_times1)

            date_in_sec_all = np.concatenate((date_in_sec, date_in_sec1))

            # Error bars in x direction:
            x_errors_lower, x_errors_upper,  x_errors = get_x_errors(e_min=e_min, e_max=e_max, inverse_betas=inverse_beta, mass_energy=mass_energy)
            x_errors_lower1, x_errors_upper1,  x_errors1 = get_x_errors(e_min=e_min1, e_max=e_max1, inverse_betas=inverse_beta1, mass_energy=mass_energy)

            # Arrays to hold all the lower and upper energy bounds. These might be redundant
            x_errors_lower_all = np.concatenate((x_errors_lower, x_errors_lower1))
            x_errors_upper_all = np.concatenate((x_errors_upper, x_errors_upper1))

            x_errors_all = np.concatenate((x_errors, x_errors1))

            # Get all the y-errors there are in this object's database
            if not isinstance(yerrs, (list,np.ndarray)):

                plus_errs, minus_errs = np.array([]), np.array([])
                plus_errs1, minus_errs1 = np.array([]), np.array([])

                # Loop through all possible, channels, even those that not necessarily show an onset
                # Remember:  self.onset_statistics : {channel_id : [mode, median, 1st_sigma_minus, 1st_sigma_plus, 2nd_sigma_minus, 2nd_sigma_plus]}
                if reference == "mode":
                    ref_idx = 0
                elif reference == "median":
                    ref_idx = 1
                else:
                    raise ValueError(f"Argument {reference} is an invalid input for the variable 'reference'. Acceptable input values are are 'mode' and 'median'.")

                for ch in channels:

                    try: 
                        minus_err = self.onset_statistics[ch][ref_idx] - self.onset_statistics[ch][4] # the difference of a timestamp and a timedelta is a timedelta
                        plus_err = self.onset_statistics[ch][5] - self.onset_statistics[ch][ref_idx] 
                    except KeyError as e:
                        print(f"KeyError in channel {e}. Missing onset?")
                        # It's irrelevant what timedelta we insert here; the corresponding data point does not exist
                        plus_err = pd.Timedelta(seconds=1)
                        minus_err = pd.Timedelta(seconds=1)

                    plus_errs = np.append(plus_errs, plus_err) if plus_err >= self.minimum_cadences[f"{spacecraft}_{instrument}"]/2 else np.append(plus_errs, pd.Timedelta(self.minimum_cadences[f"{spacecraft}_{instrument}"])/2)
                    minus_errs = np.append(minus_errs, minus_err) if minus_err >= self.minimum_cadences[f"{spacecraft}_{instrument}"]/2 else np.append(minus_errs, pd.Timedelta(self.minimum_cadences[f"{spacecraft}_{instrument}"])/2)

                for ch in channels1:

                    try: 
                        minus_err1 = Onset.onset_statistics[ch][ref_idx] - Onset.onset_statistics[ch][4] # the difference of a timestamp and a timedelta is a timedelta
                        plus_err1 = Onset.onset_statistics[ch][5] - Onset.onset_statistics[ch][ref_idx] 
                    except KeyError as e:
                        print(f"KeyError in channel {e}. Missing onset?")
                        # It's irrelevant what timedelta we insert here; the corresponding data point does not exist
                        plus_err1 = pd.Timedelta(seconds=1)
                        minus_err1 = pd.Timedelta(seconds=1)

                    plus_errs1 = np.append(plus_errs1, plus_err1) if plus_err1 >= Onset.minimum_cadences[f"{Onset.spacecraft}_{Onset.sensor}"]/2 else np.append(plus_errs1, pd.Timedelta(Onset.minimum_cadences[f"{Onset.spacecraft}_{Onset.sensor}"])/2)
                    minus_errs1 = np.append(minus_errs1, minus_err1) if minus_err1 >= Onset.minimum_cadences[f"{Onset.spacecraft}_{Onset.sensor}"]/2 else np.append(minus_errs1, pd.Timedelta(Onset.minimum_cadences[f"{Onset.spacecraft}_{Onset.sensor}"])/2)

                plus_errs_all = np.append(plus_errs, plus_errs1)
                minus_errs_all = np.append(minus_errs, minus_errs1)

                # Convert errors in time to seconds
                plus_errs_secs = [err.seconds for err in plus_errs]
                minus_errs_secs = [err.seconds for err in minus_errs]
                plus_errs_secs1 = [err.seconds for err in plus_errs1]
                minus_errs_secs1 = [err.seconds for err in minus_errs1]

                plus_errs_secs_all = np.append(plus_errs_secs, plus_errs_secs1)
                minus_errs_secs_all = np.append(minus_errs_secs, minus_errs_secs1)

                # Uneven errorbars need to be shape (2,N), where first row contains the lower errors, the second row contains the upper errors.
                y_errors_all = np.array([minus_errs_secs_all, plus_errs_secs_all])

            else:
                # The y-directional error, which is the temporal uncertainty
                try:

                    reso_str = get_time_reso(self.flux_series)
                    reso_str1 = get_time_reso(Onset.flux_series)

                    time_error = [pd.Timedelta(reso_str) for _ in range(len(x_errors))]
                    time_error1 = [pd.Timedelta(reso_str1) for _ in range(len(x_errors1))]

                except ValueError:

                    # get_time_errors() attempts to manually extract the time errors of each data point
                    time_error = get_time_errors(onset_times=onset_times, spacecraft=spacecraft)
                    time_error1 = get_time_errors(onset_times=onset_times1, spacecraft=Onset.spacecraft.lower())

                    try:

                        _ = str(time_error[0].seconds)

                    # Indexerror is caused by all NaTs -> no point in doing VDA
                    except IndexError:
                        return None


                # These are for the fitting function
                y_errors = [err.seconds for err in time_error]
                y_errors1 = [err.seconds for err in time_error1]

                y_errors_all = np.concatenate((y_errors, y_errors1))

            # Numpy masks work so that True values get masked as invalid, while False remains unaltered
            mask = np.isnan(date_in_sec_all)

            # These can be used as maskedarray even when no invalid values are in the onset times
            date_in_sec_all = ma.array(date_in_sec_all, mask=mask)
            onset_times_all = ma.array(onset_times_all, mask=mask)
            inverse_beta_all = ma.array(inverse_beta_all, mask=mask)
            inverse_beta_corrected = ma.array(inverse_beta_corrected, mask=mask)
            x_errors_all = ma.array(x_errors_all, mask=mask)

            if yerrs:
                y_errors_plot = ma.array([minus_errs, plus_errs], mask=[np.isnan(date_in_sec),np.isnan(date_in_sec)])
                y_errors1_plot = ma.array([minus_errs1, plus_errs1], mask=[np.isnan(date_in_sec1),np.isnan(date_in_sec1)])
                y_errors_all_plot = ma.array([minus_errs_all, plus_errs_all], mask=[mask,mask])
                y_errors_all_secs = ma.array([minus_errs_secs_all, plus_errs_secs_all], mask=[mask,mask])

            else:
                y_errors_plot = ma.array(time_error, mask=np.isnan(date_in_sec))
                y_errors1_plot = ma.array(time_error1, mask=np.isnan(date_in_sec1))
                y_errors_all_plot = ma.array(y_errors_all, mask=[mask,mask])
                y_errors_all_secs = ma.array(y_errors_all, mask=mask)

            # After masking NaNs and Nats away, slice
            # which datapoints to consider for the fit
            if selection is not None and isinstance(selection[0],(slice,list)):

                if isinstance(selection[0],slice):
                    selection1 = selection[1]
                    selection = selection[0]

                    # The joint selection is a little tricky, as it requires concatenating two selections
                    bool_mask = [True if (i >= selection.start and i < selection.stop) else False for i in range(len(onset_times))]

                    # Check selection1 type to treat accordingly
                    if isinstance(selection1, slice):
                        bool_mask1 = [True if (i >= selection1.start and i < selection1.stop) else False for i in range(len(onset_times1))]
                    else:
                        bool_mask1 = selection1

                    # Finally combine selections
                    selection_all = np.append(bool_mask, bool_mask1)

                # In this case selection is a boolean [TRUE TRUE FALSE ...] list
                else:

                    selection1 = selection[1]
                    selection = selection[0]

                    # Check selection1 type to treat accordingly
                    if isinstance(selection1, slice):
                        selection1 = [True if (i >= selection1.start and i < selection1.stop) else False for i in range(len(onset_times1))]

                    # Finally combine selections
                    selection_all = np.append(selection,selection1) 

                plot_omitted = True

            else:
                selection = slice(0,len(onset_times))
                selection1 = slice(0,len(onset_times1))
                selection_all = slice(0,len(onset_times_all))
                plot_omitted = False

            # These are only used for the fit -> slice them to fit selection
            inverse_beta_corrected = inverse_beta_corrected[selection_all]
            date_in_sec_all = date_in_sec_all[selection_all]

        # Only one instrument:
        else:

            onset_times_all = pd.to_datetime(onset_times)

            # Calculate the inverse betas corresponding to the nominal channel energies
            inverse_beta = calculate_inverse_betas(channel_energies=nominal_energies, mass_energy=mass_energy)

            # This is for the fitting function. 8.33 min = light travel time/au , coeff = 8.33*60
            inverse_beta_corrected = np.array([b*8.33*60 for b in inverse_beta]) #multiply by 60 -> minutes to seconds

            # Get the second values for onset times for the fit
            date_in_sec = datetime_to_sec(onset_times=onset_times)

            # Error bars in x direction:
            x_errors_lower, x_errors_upper,  x_errors_all = get_x_errors(e_min=e_min, e_max=e_max, inverse_betas=inverse_beta, mass_energy=mass_energy)

            # Get all the y-errors there are in this object's database
            if not isinstance(yerrs, (list, np.ndarray)):

                plus_errs, minus_errs = np.array([]), np.array([])
                # Loop through all possible, channels, even those that not necessarily show an onset
                # self.onset_statistics : {channel_id : [mode, median, 1st_sigma_minus, 1st_sigma_plus, 2nd_sigma_minus, 2nd_sigma_plus]}
                if reference == "mode":
                    ref_idx = 0
                elif reference == "median":
                    ref_idx = 1
                else:
                    raise ValueError(f"Argument {reference} is an invalid input for the variable 'reference'. Acceptable input values are are 'mode' and 'median'.")

                for ch in channels:

                    try: 
                        minus_err = self.onset_statistics[ch][ref_idx] - self.onset_statistics[ch][4] # the difference of a timestamp and a timedelta is a timedelta
                        plus_err = self.onset_statistics[ch][5] - self.onset_statistics[ch][ref_idx] 
                    except KeyError as e:
                        plus_err = pd.Timedelta(seconds=1)
                        minus_err = pd.Timedelta(seconds=1)

                    plus_errs = np.append(plus_errs, plus_err) if plus_err >= self.minimum_cadences[f"{spacecraft}_{instrument}"]/2 else np.append(plus_errs, pd.Timedelta(self.minimum_cadences[f"{spacecraft}_{instrument}"])/2)
                    minus_errs = np.append(minus_errs, minus_err) if minus_err >= self.minimum_cadences[f"{spacecraft}_{instrument}"]/2 else np.append(minus_errs, pd.Timedelta(self.minimum_cadences[f"{spacecraft}_{instrument}"])/2)

                # Convert errors in time to seconds
                plus_errs_secs = [err.seconds for err in plus_errs]
                minus_errs_secs = [err.seconds for err in minus_errs]

                # Uneven errorbars need to be shape (2,N), where first row contains the lower errors, the second row contains the upper errors.
                y_errors_all = np.array([minus_errs, plus_errs])

            else:
                # User gave y-errors as input

                # 2 arrays, so asymmetric plus and minus errors
                if len(yerrs)==2:

                    plus_errs, minus_errs = yerrs[0], yerrs[1]

                    y_errors_all = np.array([plus_errs,minus_errs])

                    # Convert errors in time to seconds
                    plus_errs_secs = [err.seconds for err in plus_errs]
                    minus_errs_secs = [err.seconds for err in minus_errs]

                    y_errors_all_secs = np.array([plus_errs_secs,minus_errs_secs])

                # 1 array -> symmetric errors
                else:
                    y_errors = yerrs
                    y_errors_all_secs = [err.seconds for err in y_errors]


            # Numpy masks work so that True values get masked as invalid, while False remains unaltered
            mask = np.isnan(date_in_sec)

            # These can be used as maskedarray even when no invalid values are in the onset times
            date_in_sec = ma.array(date_in_sec, mask=mask)
            onset_times = ma.array(onset_times, mask=mask)
            inverse_beta = ma.array(inverse_beta, mask=mask)
            inverse_beta_corrected = ma.array(inverse_beta_corrected, mask=mask)
            x_errors_all = ma.array(x_errors_all, mask=mask)

            if yerrs:
                y_errors_plot = ma.array([minus_errs, plus_errs], mask=[mask,mask])
                y_errors_all_secs = ma.array([minus_errs_secs, plus_errs_secs], mask=[mask,mask])

            else:
                y_errors_all_secs = ma.array(y_errors_all, mask=mask)
                y_errors_plot = ma.array(time_error, mask=mask)

            # Just to have a shared alias for all the y-errors, also in the case there are two instruments
            y_errors_all_plot = y_errors_plot

            # After masking NaNs and Nats away, slice
            # which datapoints to consider for the fit
            if selection is not None and isinstance(selection,(slice,list)):

                # User may give empty slice, e.g., slice(0,0) as an input. Handle it as if selection was None.
                if isinstance(selection,slice):
                    if selection.start==selection.stop:
                        selection_all = ~mask
                        selection = selection_all
                        plot_omitted = False
                
                selection_all = selection
                plot_omitted = True

            else:
                selection_all = ~mask
                selection = selection_all
                plot_omitted = False
                

            # These are only used for the fit -> slice them to fit selection
            inverse_beta_corrected = inverse_beta_corrected[selection_all]
            date_in_sec_all = date_in_sec[selection_all]

            # Common name to take into account single instrument and two-instrument code blocks
            inverse_beta_all = inverse_beta


        # Here onward we do the fit, and find the slope and intersection of this fit ->

        # Here happens the fit and errors ->
        # x_errors and y_errors are fed in here with selections because we want to plot them for all data points, but only
        # consider the selection of them in the fit
        # The .compressed() -method here ensures that only the valid values of the masked arrays are fed into the function. If this is not
        # done, then all of the values that the function yields will be nan.
        odr_output = seek_fit_and_errors(x=inverse_beta_corrected.compressed(), y=date_in_sec_all.compressed(), 
                                         xerr=x_errors_all.compressed(), yerr=y_errors_all_secs.compressed(), guess=guess)
        
        # This would print out the output
        # odr_output.pprint()
        
        slope, constant = odr_output.beta[0], odr_output.beta[1]
        
        dof = len(inverse_beta_corrected) - len(odr_output.beta) # degrees of freedom (number of data points minus number of free parameters)
        t_val = studentt.interval(0.95, dof)[1] # 95 % confidence interval
        errors = t_val * odr_output.sd_beta
        slope_uncertainty = errors[0]

        try:
            t_inj_uncertainty = pd.Timedelta(seconds=int(errors[1]))

        # Valueerror is caused by nan
        except ValueError:
            t_inj_uncertainty = np.nan

        # Residual variance
        residual_variance = odr_output.res_var

        # Reason for stopping the regression:
        stopreason = odr_output.stopreason

        odr_fit_sec = inverse_beta_corrected*slope
        odr_fit = np.array([datetime.datetime.utcfromtimestamp(sec + constant) for sec in odr_fit_sec.compressed()])
        
        # Release time is the timestamp where the line intersects y-axis
        release_time = datetime.datetime.utcfromtimestamp(constant)

        # Precision of t_inj display:
        precision = 5 if spacecraft != "solo" else 8
        rel_time_str =str(release_time.time())[:precision]

        for i in range(len(onset_times)):
            try:
                date_of_event = onset_times[i].date()

            # This error is caused by .date() method being called by a 'MaskedConstant'. I did not investigate further
            # what the root of the error exactly is.
            except AttributeError:
                continue
            if not isinstance(date_of_event, pd._libs.tslibs.nattype.NaTType):
                break

        # Declare axes, the things plotted and returned by this function
        # First observed datapoints, then fit and its slope+constant, lastly y-and x errors

        output = {
                "inverse_beta" : inverse_beta_all,
                "x_errors" : x_errors_all,
                "onset_times" : onset_times_all,
                "y_errors" : y_errors_all_plot,
                "path_length" : slope,
                "t_inj" : release_time,
                "path_length_uncertainty" : slope_uncertainty,
                "t_inj_uncertainty" : t_inj_uncertainty,
                "residual_variance" : residual_variance,  # The unexplained variance of the data according to the model
                "stopreason" : stopreason
                 }

        # Only plotting commands from here onward->
        if plot:

            plt.rcParams['axes.linewidth'] = 2.5
            plt.rcParams['font.size'] = 16

            # The colorblind style is apparently dependent of the configuration of stars and planets,
            # because sometimes 'seaborn-v0_8-colorblind works and sometimes it doesn't. So let's
            # try/except here to find a style that actually works.
            try:
                plt.style.use("seaborn-v0_8-colorblind")
            except OSError:
                plt.style.use("seaborn-colorblind")

            fig, ax = plt.subplots(figsize=VDA_FIGSIZE)

            ax.grid(visible=True, axis="both")

            # About matplotlib.Axes.errorbar:
            # shape(2, N): Separate - and + values for each bar. First row contains the lower errors, the second row contains the upper errors.
            # The reason xerr seems to be wrong way is that 'upper' refers to the upper ENERGY boundary, which corresponds to the LOWER 1/beta boundary
            if Onset:
                ax.errorbar(inverse_beta1, onset_times1, yerr=y_errors1_plot, xerr=[x_errors_upper1, x_errors_lower1], 
                        fmt='o', elinewidth=1.5, capsize=4.5, zorder=1, label=Onset.sensor.upper())

            # The reason xerr seems to be wrong way is that 'upper' refers to the upper ENERGY boundary, which corresponds to the LOWER 1/beta boundary
            ax.errorbar(inverse_beta, onset_times, yerr=y_errors_plot, xerr=[x_errors_upper, x_errors_lower], 
                        fmt='o', elinewidth=1.5, capsize=4.5, zorder=1, label="onset times" if not Onset else self.sensor.upper())

            # Omitted datapoints, paint all points white and then those not omitted blue (+ red) again
            if plot_omitted:
                ax.scatter(inverse_beta_all, onset_times_all, c="white", s=10, zorder=2)
                ax.scatter(inverse_beta[selection], onset_times[selection], s=11, zorder=3)

                if Onset:
                    ax.scatter(inverse_beta1[selection1], onset_times1[selection1], s=11, zorder=3)


            # The odr fit
            # Here we need to first take the selection of i_beta_all and ONLY after that take the compressed form, which is the set of valid values
            ax.plot(inverse_beta_all[selection_all].compressed(), odr_fit, ls='--',
                label=f"L: {np.round(slope,3):.3f} $\pm$ {np.round(slope_uncertainty,3):.3f} AU\nt_inj: {rel_time_str} $\pm$ {str(t_inj_uncertainty)[7:]}")

            ax.set_xlabel(r"1/$\beta$", fontsize = 20)

            # Format the y-axis. For that make a selection to exclude NaTs from the set of onset times that define 
            # the vertical axis boundaries.
            nat_onsets = pd.isnull(onset_times_all)
            not_nats = np.array(onset_times_all)[~nat_onsets]

            # We have to catch on the above line all non-NaT onset times, because numpy nanmin() and nanmax() don't recognize them
            if len(not_nats) > 0:
                if np.nanmax(not_nats)-np.nanmin(not_nats) > pd.Timedelta(minutes=10):
                    y_axis_time_format = DateFormatter("%H:%M")
                    ax.set_ylabel("Time (HH:MM)", fontsize = 20)
                else:
                    y_axis_time_format = DateFormatter("%H:%M:%S")
                    ax.set_ylabel("Time (HH:MM:SS)", fontsize = 20)
            ax.yaxis.set_major_formatter(y_axis_time_format)

            if ylim:
                ax.set_ylim(pd.to_datetime(ylim[0]),pd.to_datetime(ylim[1]))

            ax.legend(loc=4)

            # Title for the figure
            if title is None:
                if Onset:
                    # This is the default for joint VDA, two instruments of the same spacecraft
                    if self.spacecraft == Onset.spacecraft:
                        if self.viewing:
                            ax.set_title(f"VDA, {spacecraft.upper()} / {instrument.upper()}({self.viewing}) + {Onset.sensor.upper()} {species_title}, {date_of_event}")
                        else:
                            ax.set_title(f"VDA, {spacecraft.upper()} / {instrument.upper()} + {Onset.sensor.upper()} {species_title}, {date_of_event}")

                    else:
                        # In this case these are two different spacecraft
                        if self.viewing and Onset.viewing:
                            ax.set_title(f"VDA, {spacecraft.upper()}/{instrument.upper()}({self.viewing}) + {Onset.spacecraft.upper()}/{Onset.sensor.upper()}({Onset.viewing})\n{species_title}, {date_of_event}")
                        elif self.viewing:
                            ax.set_title(f"VDA, {spacecraft.upper()}/{instrument.upper()}({self.viewing}) + {Onset.spacecraft.upper()}/{Onset.sensor.upper()}\n{species_title}, {date_of_event}")
                        elif Onset.viewing:
                            ax.set_title(f"VDA, {spacecraft.upper()}/{instrument.upper()} + {Onset.spacecraft.upper()}/{Onset.sensor.upper()}({Onset.viewing})\n{species_title}, {date_of_event}")
                        else:
                            ax.set_title(f"VDA, {spacecraft.upper()}/{instrument.upper()} + {Onset.spacecraft.upper()}/{Onset.sensor.upper()} {species_title}, {date_of_event}")

                else:
                    # Single spacecraft, single instrument
                    if self.viewing:
                        ax.set_title(f"VDA, {spacecraft.upper()} {instrument.upper()}({self.viewing}) {species_title}, {date_of_event}")
                    else:
                        ax.set_title(f"VDA, {spacecraft.upper()} {instrument.upper()} {species_title}, {date_of_event}")
            else:
                ax.set_title(title)

            # Saving the figure in "/plots"
            if save:
                if not savepath:
                    savepath = CURRENT_PATH
                if Onset:
                    savestr = f"{savepath}{os.sep}VDA_{spacecraft}_{instrument}({self.viewing})+{Onset.sensor}_{species_title}_{date_of_event}.png" if self.viewing else f"{savepath}{os.sep}VDA_{spacecraft}_{instrument}+{Onset.sensor}_{species_title}_{date_of_event}.png"
                else:
                    savestr = f"{savepath}{os.sep}VDA_{spacecraft}_{instrument}_{self.viewing}_{species_title}_{date_of_event}.png" if self.viewing else f"{savepath}{os.sep}VDA_{spacecraft}_{instrument}_{species_title}_{date_of_event}.png"
                plt.savefig(savestr, transparent=False,
                        facecolor="white", bbox_inches="tight")

        # Add the figure and axes of it to the return
        output["fig"] = fig
        output["axes"] = ax

        return output


    def automatic_onset_stats(self, channels, background, viewing, erase, cusum_minutes:int=None, sample_size:float=0.5, 
                              small_windows=None, stop=None, weights=None, limit_computation_time=True, sigma=2, 
                              detrend:bool=True, prints:bool=False,
                              limit_averaging:str=None, fail_avg_stop:int=None):
        """
        Automates the uncertainty estimation for a single channel provided by Poisson-CUSUM-bootstrap hybrid method

        Parameters:
        ----------
        channels : {int}
                    Channel number.
        background : {BootstrapWindow}
                    The background that precedes the onset of the event.
        viewing : {str}
                    Viewing direction of the instrument.
        erase : {tuple(float, str)}
                    Acts a a filter that discards all values above the first entry BEFORE the timestamp defined by the second entry.
        cusum_minutes : {int}, optional
                    The amount of time in minutes that we demand on threshold-exceeding intensity.
        sample_size : {float} default 0.5
                    The fraction of the datapoints that the method will randomly pick.
        small_windows : {str} default None
                     Choose either 'random' or 'equal' to partition the big background window into smaller ones. 'equal' small windows
                     are equally placed inside the background, while 'random' placement allows the windows to overlap.
        stop : {str} default None
                    A pandas-compatible time string that decides on how long to continue averaging data while making onset 
                    distributions. If None, data will be averaged up to the 1-sigma uncertainty of whatever the native data resolution
                    produces.
        weights : {str} default None
                    Choose weights for calculating the mean uncertainty. 'int_time' will weight uncertainties by their individual
                    integrating times, while 'uncertainty' will weight small uncertainties more and large less.
        limit_computation_time : {bool}, default True
                    If enabled, skips all integration times above 10 minutes that are not multiples of 5. 
        sigma : {int, float} default 2
                    The multiplier for the $\mu_{d}$ variable in the CUSUM method.
        detrend : {bool}, default True
                    Switch to apply a shift on all but the native data distributions such that the onset times are shifted backwards in
                    time by half of the data time resolution.
        prints : {bool}, optional
                    Switch to print information about which channels is being analyzed and what's its 1-sigma uncertainty.
        limit_averaging : {str}, optional
                    Pandas-compatible time string. Limits the averaging to a certain time. Leave to None to not limit averaging. 
        fail_avg_stop : {int}, optional
                    If absolutely no onset is found in the native time resolution, how far should the method average the data to
                    try and find onset times? Default is up to 5 minutes.

        Returns:
        ----------
        stats_arr : {OnsetStatsArray}
        """

        def dt_str_to_int(stop):
            """
            Converts stop condition string to a integer.
            """

            if isinstance(stop, str):
                    if stop[-3:] == "min":
                        split_str = "min"
                        divisor = 1
                    elif stop[-1:] == 's':
                        split_str = 's'
                        divisor = 60
                    stop_int = int(stop.split(split_str)[0])//divisor
            else:
                raise ValueError("Time string has to pandas-compatible time string, e.g., '15min' or '60s'.")
            
            return stop_int

        if not isinstance(channels,int):
            if not isinstance(channels,list):
                channels = int(channels)

        # Get the integer numbers that represent stop and/or limit_averaging in minutes
        if stop:
            stop_int = dt_str_to_int(stop)
        if limit_averaging:
            limit_averaging_int = dt_str_to_int(limit_averaging)

        # SolO/EPT first channel does not provide proper data as of late
        if self.spacecraft=="solo" and self.sensor=="ept" and channels==0:
        #    self.input_nan_onset_stats(channels)
            return  None

        # Wind/3DP first electron channel and the first two proton channels don't provide proper data
        if self.spacecraft=="wind" and self.species=='e' and channels==0:
            return  None
        if self.spacecraft=="wind" and self.species=='p' and channels in (0,1):
            return  None

        # The first round of onset statistics is acquired by time resolution close to or exactly the native resolution 
        if self.spacecraft == "wind":
            first_resample = "1 min"
        elif self.spacecraft == "solo" and limit_computation_time:
            first_resample = "1 min"
        else:
            first_resample = None

        # SOHO/EPHIN E300 is deactivated from 2017 onward -> there will be no reasonable onset there
        if self.spacecraft=="soho" and self.sensor=="ephin" and channels==300 and self.flux_series.index[0].year >= 2017:
            print("Channel deactivated because of failure mode D.")
            return None

        # Run statistic_onset() once to get the confidence intervals for the bare not resampled data
        first_run_stats, _ = self.statistic_onset(channels=channels, Window=background, viewing=viewing, 
                                            sample_size=sample_size, resample=first_resample, erase=erase, small_windows=small_windows,
                                            cusum_minutes=cusum_minutes, detrend=False, sigma_multiplier=sigma)

        # For the first iteration initialize the OnsetStatsArray object, which can plot the integration time plot
        # This step has to be done after running statistic_onset() the first time, because otherwise "self.bootstrap_onset_statistics"
        # does not exist yet
        stats_arr = OnsetStatsArray(self)

        # Integer number of first run uncertainty in minutes
        first_run_uncertainty_mins = (first_run_stats["1-sigma_confidence_interval"][1] - first_run_stats["1-sigma_confidence_interval"][0]).seconds // 60
        first_run_uncertainty = first_run_stats["1-sigma_confidence_interval"][1] - first_run_stats["1-sigma_confidence_interval"][0]

        if prints:
            print(f"1-sigma uncertainty for the self time with native data resolution: {first_run_uncertainty}")

        # Could be that the native resolution identifies no onset at all, in this case handle it
        if not isinstance(first_run_uncertainty, pd._libs.tslibs.nattype.NaTType):

            # If stop is not defined, then average up to predefined (default or first_run_uncertainty_mins) time
            if not stop:

                # Most of the high-energy particle instruments have a time resolution of 1 min, so don't do averaging for them
                # if uncertainty is something like 1 min 07 sec
                if first_run_uncertainty_mins < 2 and self.spacecraft not in ("solo", "wind"):

                    stats_arr.calculate_weighted_uncertainty("int_time")
                    return stats_arr

                else:
                    # SolO instruments and Wind/3DP have high cadence (< 1 min), so start integrating from 1 minute measurements
                    start_idx = 1 if self.spacecraft in ("solo", "wind") and not limit_computation_time else 2

                    # Initialize integration times only up to the amount of minutes that the first run had uncertainty
                    int_times = np.array([i for i in range(start_idx,first_run_uncertainty_mins+1)], dtype=int)

                    if prints:
                        if limit_averaging:
                            upto_averaging_display = limit_averaging_int if limit_averaging_int < first_run_uncertainty_mins else first_run_uncertainty_mins
                        else:
                            upto_averaging_display = first_run_uncertainty_mins
                        print(f"Averaging up to {upto_averaging_display} minutes") if upto_averaging_display > 0 else print("Not averaging.")

            else:

                if stop_int > 0:
                    if prints:
                        print(f"Averaging up to {stop_int} minutes")
                    else:
                        pass

                if self.spacecraft in ("solo", "wind"):
                    int_times = np.array([i for i in range(1,stop_int+1)])
                else:
                    int_times = np.array([i for i in range(2,stop_int+1)])

        # Go here if no onset found at all
        else:

            if self.spacecraft in ("solo", "wind") and not limit_computation_time:
                try_avg_start = 1
            else:
                try_avg_start = 2
            try_avg_stop = 5 if not isinstance(fail_avg_stop,int) else fail_avg_stop

            # Try up to {try_avg_stop} minutes averaging (default 5), if still no onset -> give up
            for i in range(try_avg_start,try_avg_stop+1):

                next_run_stats, _ = self.statistic_onset(channels=channels, Window=background, viewing=viewing, 
                                            sample_size=sample_size, resample=f"{i}min", erase=erase, small_windows=small_windows,
                                            cusum_minutes=cusum_minutes, sigma_multiplier=sigma, detrend=True)
                next_run_uncertainty = next_run_stats["1-sigma_confidence_interval"][1] - next_run_stats["1-sigma_confidence_interval"][0]
                next_run_uncertainty_mins = (next_run_stats["1-sigma_confidence_interval"][1] - next_run_stats["1-sigma_confidence_interval"][0]).seconds // 60

                if not isinstance(next_run_uncertainty, pd._libs.tslibs.nattype.NaTType):
                    if prints:
                        print(f"No onset found in the native data resolution. 1-sigma uncertainty with {i} min resolution: {next_run_uncertainty}")

                    # Here check if it makes sense to average "from i minutes to <uncertainty> minutes or up to "stop" minutes
                    if stop:

                        if i < stop_int:
                            int_times = np.array([j for j in range(i,stop_int+1)])
                            if prints:
                                print(f"Averaging up to {stop} minutes")
                        else:
                            if prints:
                                print(f"Stop condition set to {stop_int}, which is less than {i} min. Using only {i} minutes averaged data.")
                            int_times = np.array([j for j in range(i,i+1)])

                    elif i < next_run_uncertainty_mins:
                        int_times = np.array([j for j in range(i,next_run_uncertainty_mins+1)])
                        if prints:
                            if limit_averaging:
                                upto_averaging_display = limit_averaging_int if limit_averaging_int < next_run_uncertainty_mins else next_run_uncertainty_mins
                            else:
                                upto_averaging_display = next_run_uncertainty_mins

                            print(f"Averaging from {i} minutes up to {upto_averaging_display} minutes")

                    else:
                        if prints:
                            print(f"1-sigma uncertainty less than current time-averaging. Terminating.")

                        stats_arr.add(self)
                        stats_arr.calculate_weighted_uncertainty(weights)
                        return stats_arr

                    break

                else:
                    if i==5:
                        if prints:
                            print(f"No onsets found with {i} min time averaging. Terminating.")
                    else:
                        pass

            # If we tried everything and still no onset -> NaT and exit
            if i==5:
                stats_arr.calculate_weighted_uncertainty("int_time")
                return stats_arr

        # Here if int_times gets too long, coarsen it up a little from 10 minutes onward
        # Logic of this selection: preserve an element of int_times if it's at most 10 OR if it's divisible by 5
        if limit_computation_time:
            int_times = int_times[np.where((int_times <= 10) | (int_times%5==0))]

        # If the user set's some upper limit to the averaging, apply that limit here
        if isinstance(limit_averaging,str):
            int_times = int_times[np.where(int_times <= limit_averaging_int)]

        # Finally convert int_times (minutes) to pandas-compatible time strs
        int_times = np.array([f"{i}min" for i in int_times])

        # Loop through int_times as far as the first run uncertainty reaches
        for _, resample in enumerate(int_times):

            if int(resample[:-3]) > 10:
                cusum_minutes = int(resample[:-3])*3

            _, _ = self.statistic_onset(channels=channels, Window=background, viewing=viewing, 
                                            sample_size=sample_size, resample=str(resample), erase=erase, small_windows=small_windows,
                                            cusum_minutes=cusum_minutes, sigma_multiplier=sigma, detrend=detrend)

            stats_arr.add(self)

        # Calculate the weighted medians and confidence intervals. This method automatically updates the onset object's
        # dictionary of uncertainties as well.
        stats_arr.calculate_weighted_uncertainty(weights)

        return stats_arr


    def channelwise_onset_statistics(self, background, viewing, channels=None, erase:(tuple,list)=None, cusum_minutes:int=30, sample_size:float=0.50, 
                                     weights:str="uncertainty", detrend=True, limit_computation_time=True, average_to=None, prints=False, 
                                     limit_averaging=None, fail_avg_stop:int=None):
        """
        Wrapper method for automatic_onset_stats(), that completes full onset and uncertainty analysis for a single channel.
        Does a complete onset uncertainty analysis on, by default all, the energy channels for the given instrument.

        Parameters:
        -----------
        channels : {str or array-like}, optional
                    A tuple, list or Range of channel ID's to run the method on. Leave to None or set to 'all' to run for all channels.
        background : {BootstrapWindow}
                    The common pre-event background used for the energy channels. 
        viewing : {str}
                    The viewing direction if the instrument.
        erase : {tuple(float, str)}, optional
                    loepreuinm impasum
        cusum_minutes : {int, float}, optional
                    The amount of MINUTES that the method will demand for continuous threshold-axceeding intensity before identifying
                    an onset. 
        sample_size : {float}, optional
                    The fraction of the data points inside the background window that will be considered for each of the bootstrapped
                    runs of the method.
        weights : {str}, optional
                    Either 'uncertainty' to use the width of 2-sigma intervals as the basis for weighting timestamps, or 'int_time' to
                    use the integration time as a basis for the weighting.
        detrend : {bool}, optional
                    Switch to apply detrending on the onset time distributions.
        limit_computation_time : {bool}, optional
                    lorem pipsum
        average_upto : {str}, optional
                    Tells the method to explicitly average every channel up to a specific time resolution, disregarding the
                    recommendation got from 1-sigma width of the native data. If both 'average_to' and 'limit_averaging' are
                    given as an input, 'limit_averaging' will take precedence over 'average_to'.
        prints : {bool}
                    Switch to print when a new channel is being analyzed and for how far it will be time-averaged to.
        limit_averaging : {str}, optional
                    Pandas compatible time string to limit the averaging time to a certain time, e.g., '60 min'
        fail_avg_stop : {int}, optional
                    If absolutely no onset is found in the native time resolution, how far should the method average the data to
                    try and find onset times? Default is up to 5 minutes.

        Returns:
        ----------
        uncertainty_stats_by_channel : {np.ndarray(OnsetStatsArray)}
                    lorem ipsum+
        """

        # Initialize the array which will store all the uncertaintystats objects
        uncertainty_stats_by_channel = np.array([])

        # Check which channels to run uncertainty stats on
        if not channels or channels=="all":
            all_channels = self.get_all_channels()
        elif isinstance(channels, (tuple, list, range)):
            all_channels = channels
        else:
            raise TypeError(f"{type(channels)} is and incorrect type of argument 'channels'! It should be None, str=='all', tuple, list or range.")

        # Include a check on the length of the background in relation to the absolute number of data points taken by a 
        # random sample. 
        bg_window_length = background.end - background.start
        minutes_in_background = int(bg_window_length.seconds/60)

        # We recommend a maximum reso such that there are at least {MIN_RECOMMENDED_POINTS} data points to pick from
        max_reso = int(minutes_in_background/MIN_RECOMMENDED_POINTS)
        print(f"Your chosen background is {minutes_in_background} minutes long. To have at least {MIN_RECOMMENDED_POINTS} data points to choose from,\nit is recommended that you either limit averaging up to {max_reso} minutes or enlarge the background window.")

        # Loop through all the channels and save the onset statistics to objects that will be stored in the array initialized in the start
        for channel in all_channels:

            if prints:
                print(f"Channel {channel}:")

            # automatic_onset_stats() -method runs statistic_onset() for all different data integration times for a single channel
            onset_uncertainty_stats = self.automatic_onset_stats(channels=channel, background=background, viewing=viewing, erase=erase,
                                                                stop=average_to, cusum_minutes=cusum_minutes, sample_size=sample_size, 
                                                                weights=weights, detrend=detrend, limit_computation_time=limit_computation_time,
                                                                prints=prints, limit_averaging=limit_averaging, fail_avg_stop=fail_avg_stop)

            # Add statistics to the array that holds statistics related to each individual channel
            uncertainty_stats_by_channel = np.append(uncertainty_stats_by_channel, onset_uncertainty_stats)

        return uncertainty_stats_by_channel


    def find_first_peak(self, channels, window, xlim, viewing, resample, plot=True):
        """
        Finds the first maximum in time series data.
        
        Parameters:
        -----------
        window : {tuple(datetime str)}
        
        Returns:
        ----------
        first_maximum_peak_time : {pd.datetime}
        """

        flux_series, en_channel_string = self.choose_flux_series(channels=channels, viewing=viewing)

        # Resample data if requested
        if resample is not None:
            flux_series = util.resample_df(flux_series, resample)

        # Cut flux_series according to given window
        window = pd.to_datetime(window)
        cut_series = flux_series.loc[(flux_series.index > window[0]) & (flux_series.index < window[1])]

        # The series is indexed by time
        time = cut_series.index

        first_maximum_peak_val = cut_series.values[0]
        first_maximum_peak_time = time[0]
        for i, val in enumerate(cut_series.values):

            if val > first_maximum_peak_val:
                first_maximum_peak_val = val
                first_maximum_peak_time = time[i]

        # Plotting commands ->
        fig, ax = plt.subplots(figsize=(21,9))

        # Setting the x-axis limits
        if xlim is None:
            xlim = [flux_series.index[0], flux_series.index[-1]]
        else:
            # Check that xlim makes sense
            if xlim[0] == xlim[1]:
                raise Exception("xlim[0] and xlim[1] cannot be the same time!")

            xlim[0], xlim[1] = pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1])
            ax.set_xlim(xlim)

        ax.set_yscale("log")

        # The measurements. kw where dictates where the step happens -> "mid" for at the middle of the bin
        ax.step(flux_series.index, flux_series.values, c="C0", where="mid")

        # Window shaded area
        ax.axvspan(xmin=window[0], xmax=window[1],
                    color="lightblue", label="Window", alpha=0.3)

        # Peak time, value and a textbox displaying both
        ax.axvline(x=first_maximum_peak_time, linewidth=1.2, color="navy", linestyle= '-', 
                    label="First maximum peak")

        # Textbox settings:
        plabel = AnchoredText(f"First maximum peak time+value\n{first_maximum_peak_time}\n{first_maximum_peak_val}", prop=dict(size=13), frameon=True, loc=(4) )
        plabel.patch.set_boxstyle("round, pad=0., rounding_size=0.2")
        plabel.patch.set_linewidth(2.0)
        ax.add_artist(plabel)

        # Labels for axes
        ax.set_xlabel("Time", fontsize=20)
        ax.set_ylabel(f"Intensity [1/(cm^2 sr s MeV)]", fontsize=20)

        # Tickmarks, their size etc...
        ax.tick_params(which='major', length=6, width=2, labelsize=17)
        ax.tick_params(which='minor', length=5, width=1.4)

        # Date tick locator and formatter
        ax.xaxis_date()
        utc_dt_format1 = DateFormatter('%H:%M \n%Y-%m-%d')
        ax.xaxis.set_major_formatter(utc_dt_format1)

        # Setting the title
        if self.species in ["electron", 'e']:
            s_identifier = 'electrons'
        if self.species in ["proton", 'p', 'H']:
            s_identifier = 'protons'
        if self.species in ["ion", 'i']:
            s_identifier = 'ions'

        if (viewing != '' and viewing is not None):

            plt.title(f"{self.spacecraft}/{self.sensor.upper()} {en_channel_string} {s_identifier}\n"
                    f"{resample} averaging, viewing: "
                    f"{viewing.upper()}")

        else:

            plt.title(f"{self.spacecraft}/{self.sensor.upper()} {en_channel_string} {s_identifier}\n"
                    f"{resample} averaging")

        if plot:
            plt.show()
        else:
            plt.close()

        return first_maximum_peak_time


    def get_distribution_statistics(self, onset_statistics:dict, percentiles:tuple = [(15.89,84.10), (2.30, 97.7)]):
        """
        Returns the median, mode, mean and confidence intervals of an onset distribution.
        
        Parameters:
        -----------
        onset_statistics : dict
                        A bootstrap_onset_statistics dictionary produced by the Onset class
        percentiles : tuple or a list of tuples
                        A tuple of percentiles, or a list containing the pairs of percentiles.
        
        Returns:
        -----------
        median : pandas datetime
                The median of the onset distribution
        mode : pandas datetime
                The mode of the onset distribution
        mean : pandas datetime
                The mean of the onset distribution
        confidence_intervals : tuple or a list of tuples
                Contains the percentiles or a list of percentiles if input was a list
        """

        onsets = pd.to_datetime(onset_statistics["onset_list"])

        # Calculate median with numpy.median
        median =  onsets[len(onsets)//2]

        # The most frequently appearing onset is already found
        mode = onset_statistics["unique_onsets"][-1,0]

        # Mean onset is already calculated
        mean = onset_statistics["mean_onset"]

        # Find the percentiles that were given
        confidence_intervals = self.get_distribution_percentiles(onsets, percentiles)

        return median, mode, mean, confidence_intervals


    def get_distribution_percentiles(self, onset_list, percentiles:tuple):
        """
        Returns the confidence intervals of an onset distribution. Includes a check to make sure
        that the uncertainty intervals are not smaller than the data time resolution.
        
        Parameters:
        -----------
        percentiles : tuple or a list of tuples
                        A tuple of percentiles, or a list containing the pairs of percentiles.
        
        Returns:
        -----------
        confidence_intervals : tuple or a list of tuples
                Contains the percentiles or a list of percentiles if input was a list
        """

        # Find the percentiles that were given
        if isinstance(percentiles[0],(tuple,list)):
            confidence_intervals = []
            for pair in percentiles:
                conf_interval = pd.to_datetime(np.percentile(onset_list,pair))
                confidence_intervals.append(conf_interval)
        else:
            confidence_intervals = pd.to_datetime(np.percentile(onset_list,percentiles))

        # Finally check that the percentiles make sense (in that they are not less than the native data resolution of the instrument)
        time_reso = self.get_minimum_cadence()
        confidence_intervals = check_confidence_intervals(confidence_intervals, time_reso=time_reso)

        return confidence_intervals


# The class that holds background window length+position and bootstrapping parameters
class BootstrapWindow:

    def __init__(self, start, end, n_shifts=0, bootstraps=0):
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

    def __repr__(self):
        return self.attrs()


    def attrs(self, key=None):

        if key is None:
            return str(self.attrs_dict)
        else:
            return self.attrs_dict[key]

    def draw_background(self, ax):
        """
        Draws the window on a plot, given the axes of a plot.

        Parameters:
        -----------
        ax : {plt.Axes}
                The axes of the figure.
        """
        BACKGROUND_ALPHA = 0.15

        ax.axvspan(xmin=self.start, xmax=self.end,
                        color="#e41a1c", label="Background", alpha=BACKGROUND_ALPHA)


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



class OnsetStatsArray:
    """
    Contains statistics and uncertainty of a single onset at a particular energy.
    """

    def __init__(self, onset_object):

        # Initialize the list of statistics (NOT A LIST OF ONSET OBJECTS)
        # self.statistics always points to the most recent dictionary in the archive of dictionaries
        self.statistics = onset_object.bootstrap_onset_statistics.copy()
        self.archive = [self.statistics]

        # Save the individual time series for later plotting purposes
        self.list_of_series = [onset_object.flux_series]

        # Save the onset object attributes
        self.spacecraft = onset_object.spacecraft
        self.sensor = onset_object.sensor
        self.species = onset_object.species
        self.channel_str = onset_object.recently_examined_channel

        # Remember to which onset object this statistics array is linked to
        self.linked_object = onset_object

        # Init integration times as an empty list and immediately calculate the first integration time
        self.integration_times = []
        self.calculate_integration_time(onset_object=onset_object)


    def __repr__(self):
        return str(f"Dictionaries in archive: {len(self.archive)}")

    def __len__(self):
        return len(self.archive)

    def attrs(self):
        print("OnsetStatsArray.statistics: the most recently added dictionary in the archive of dictionaries")
        print("OnsetStatsArray.archive: a list containing all the added dictionaries that each contain onset statistics")
        print("OnsetStatsArray.integration_times: a list containing integration times <float> in minutes corresponding to the statistics added to the archive")


    def add(self, onset_object):
        """
        Adds the bootstrap statistics of an onset object into the 'archive' of OnsetStatsArray. In practicality the archive is just a 
        list of dictionaries.
        """

        # Check that these are all the same sc / instrument / particle species
        if len(self.archive) > 1:
            if onset_object.spacecraft.lower() != self.spacecraft.lower():
                raise Exception("Only one spacecraft per OnsetStatsArray permitted!")
            if onset_object.sensor.lower() != self.sensor.lower():
                raise Exception("Only one sensor per OnsetStatsArray permitted!")
            if onset_object.species.lower() != self.species.lower():
                raise Exception("Only one particle species per OnsetStatsArray permitted!")

        # Assert statistics as the latest stats dictionary
        self.statistics = onset_object.bootstrap_onset_statistics.copy()

        # Add the latest statistics dictionary to the list of dictionaries
        self.archive.append(self.statistics)

        # Calculate the most recent integration time and add it to the list of integration times
        self.calculate_integration_time(onset_object=onset_object)

        # Add the latest flux_series to the list of series for plotting purposes
        self.list_of_series.append(onset_object.flux_series)


    def drop(self, index:int=-1):
        """
        Drops by default the latest dictionary from the archive, unless given an index, 
        in which case the onset corresponding to the index is dropped. Also gets rid of 
        the chosen flux_series from the list of series.
        """
        _ = self.archive.pop(index)
        _ = self.integration_times.pop(index)
        _ = self.list_of_series.pop(index)

    def set_mean_of_median_onsets(self, mean_of_medians_onset):

        self.mean_of_medians_onset = mean_of_medians_onset

        self.linked_object.mean_of_medians_onset_acquired = True
        self.linked_object.mean_of_medians_onset = mean_of_medians_onset

    def set_w_median_and_confidence_intervals(self, mode, median, conf1_low, conf1_high, conf2_low, conf2_high):
        """
        Sets the weighted median and confidence intervals for onset time 
        """

        # Save the median and class attributes to the linked object's database:
        self.linked_object.update_onset_statistics([mode, median, conf1_low, conf1_high, conf2_low, conf2_high])

    def calculate_integration_time(self, onset_object) -> None:
        """
        Calculates the integration time (i.e. time resolution of data) of the given onset object and adds it to the list of integration times.
        Saves the integration times to a class attribute integration_times, which is a list of integration times in minutes as floats.
        """
        delta_sec = (onset_object.flux_series.index[2] - onset_object.flux_series.index[1]).seconds
        delta_min = delta_sec/60
        int_time = f"{delta_min} min" if delta_sec > 59 else f"{delta_sec} s"
        self.integration_times.append(int_time) if str(delta_min)[:7] != "0.98333" else self.integration_times.append("1 min")


    def onset_time_histogram(self, binwidth="1min", xlims=None, ylims=None, save=False, savepath=None):
        """
        A method to display the probability density histogram for the distribution of onset times collected
        to the object.
        """
       
        stats = self.statistics

        # Plotting 
        fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

        # If xlims not manually defined, let them be \pm 5 minutes from the first and last onset of the distribution
        if not xlims:
            xlims = (np.nanmin(stats["onset_list"]) - pd.Timedelta(minutes=2), np.nanmax(stats["onset_list"]) + pd.Timedelta(minutes=2))
        ax.set_xlim(xlims)

        #show percentage on y-axis
        yvalues = [m/10 for m in range(0,11)]
        ax.set_yticks(yvalues)
        #ax.yaxis.set_major_locator(mticker.FixedLocator(yvalues)) #this is to fix the yticks at place, recommended to do before altering labels
        ax.set_yticklabels(['{}'.format(np.round(x, 1)) for x in yvalues])


        # Bins for the x axis 
        half_bin = pd.Timedelta(seconds=30)
        bins = pd.date_range(start=xlims[0]+half_bin, end=xlims[1]+half_bin, freq=binwidth).tolist()

        onset_frequencies = np.ones_like(stats["onset_list"])/len(stats["onset_list"])

        # Plotting the histogram rectangles
        bar_heights, bins, patches = ax.hist(stats["onset_list"], bins=bins, color="lightblue", edgecolor="black", weights=onset_frequencies, zorder=2)

        if not ylims:
            ylims = (0, np.nanmax(bar_heights)+0.02)
        ax.set_ylim(ylims)

        # Mean, mode and median onset times as vertical solid lines
        ax.axvline(stats["mean_onset"], linewidth=2.0, color="purple", label=f"mean {str(stats['mean_onset'].time())[:8]}", zorder=3)
        ax.axvline(stats["most_likely_onset"][0], linewidth=2.0, color="blue", label=f"mode {str(stats['most_likely_onset'][0].time())[:8]}", zorder=3)
        ax.axvline(stats["median_onset"], linewidth=2.0, color="red", label=f"median {str(stats['median_onset'].time())[:8]}", zorder=3)

        # 1 -and 2-sigma intervals as red and blue dashed lines
        ax.axvspan(xmin=stats["2-sigma_confidence_interval"][0], xmax=stats["2-sigma_confidence_interval"][1], color="blue", alpha=0.15, label="2-sigma", zorder=1)
        ax.axvspan(xmin=stats["1-sigma_confidence_interval"][0], xmax=stats["1-sigma_confidence_interval"][1], color="red", alpha=0.15, label="1-sigma", zorder=1)

        ax.set_xlabel(f"Time\n{stats['mean_onset'].strftime('%Y-%m-%d')}")
        ax.set_ylabel("PD")
        ax.grid(True)

        ax.set_title(f"Probability density for {self.linked_object.background.bootstraps} onset times")
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))

        plt.legend(loc=1, bbox_to_anchor=(1.0, 1.0), fancybox=True, ncol=3, fontsize = 14)

        if save:
            if savepath is None:
                plt.savefig("onset_times_histogram.png", transparent=False, facecolor='white', bbox_inches='tight')
            else:
                plt.savefig(f"{savepath}{os.sep}onset_times_histogram.png", transparent=False, facecolor='white', bbox_inches='tight')

        plt.show()

    def integration_time_plot(self, title=None, ylims=None, save=False, savepath:str=None) -> None:
        """
        Plots the median, mean, mode and confidence intervals for an array of Onset objects as a function
        of integration time (basically time resolution of the data, may be resampled)

        Parameters:
        -----------
        title : {str}, default None
                Title of the figure, generate automatically if None
        ylims : {array-like}, len==2, optional
                The lower and upper limit of the y-axis as datetimes or pandas-compatible datetime strings,
                e.g., '2010-12-31 14:00'.
        save : {bool}, default False
                Saves the figure
        savepath : {str}, optional
                The directory path or subdirectory to save the figure.
        """

        # Collect the stats and different time resolutions
        means = []
        medians = []
        modes = []
        confidence_interval_1sigma = []
        confidence_interval_2sigma = []
        for statistics in self.archive:

            means.append(statistics["mean_onset"])
            medians.append(statistics["median_onset"])
            modes.append(statistics["most_likely_onset"][0])
            confidence_interval_1sigma.append(statistics["1-sigma_confidence_interval"])
            confidence_interval_2sigma.append(statistics["2-sigma_confidence_interval"])

        # This is done to preserve the confidence_intervals as class attributes that can be used later
        self.confidence_intervals_1sigma = np.array(confidence_interval_1sigma)
        self.confidence_intervals_2sigma = np.array(confidence_interval_2sigma)

        # Lower and higher boundaries of the confidence intervals of the distributions, in order of appearance
        conf1_lows = [pair[0] for pair in confidence_interval_1sigma]
        conf1_highs = [pair[1] for pair in confidence_interval_1sigma]

        conf2_lows = [pair[0] for pair in confidence_interval_2sigma]
        conf2_highs = [pair[1] for pair in confidence_interval_2sigma]

        # For plotting the mean of medians and modes lines (pandas series allows for computing the mean of datetimes)
        median_series = pd.Series(medians)
        mode_series = pd.Series(modes)
        mean_of_medians = median_series.mean()
        mean_of_modes = mode_series.mean()

        # Make the figure longer if there are a lot of ticks on the x-axis
        fig_len = 13 if len(self.archive) < 30 else 19

        # Initializing the figure
        fig, ax = plt.subplots(figsize=(fig_len,7))

        # These integration time strings are transformed to floating point numbers in units of minutes
        xaxis_int_times = [pd.Timedelta(td).seconds/60 for td in self.integration_times]

        ax.set_xlim(0, xaxis_int_times[-1]+1)

        if ylims:
            ax.set_ylim(pd.to_datetime(ylims[0]), pd.to_datetime(ylims[1]))

        # We only want to have integer tickmarks
        ax.set_xticks(range(0,int(xaxis_int_times[-1]+1)))

        ax.scatter(xaxis_int_times, means, s=115, label="mean", zorder=2, color="orange", marker=".")
        ax.scatter(xaxis_int_times, medians, s=65, label="median", zorder=2, color="red", marker="^")
        ax.scatter(xaxis_int_times, modes, s=65, label="mode", zorder=2, color="navy", marker="p")

        ax.axhline(y=mean_of_medians, color="red", label=f"Mean of median onsets: ({str(mean_of_medians.time())[:8]})")
        ax.axhline(y=mean_of_modes, color="navy", label=f"Mean of mode onsets: ({str(mean_of_modes.time())[:8]})")

        ax.fill_between(xaxis_int_times, y1=conf1_lows, y2=conf1_highs, facecolor="red", alpha=0.3, zorder=1)
        ax.fill_between(xaxis_int_times, y1=conf2_lows, y2=conf2_highs, facecolor="blue", alpha=0.3, zorder=1)

        ax.set_xlabel("Data integration time [min]")
        ax.set_ylabel(f"{means[0].date()}\nTime [HH:MM]")

        if not title:
            particle_str = "electrons" if self.species=='e' else "protons" if self.species=='p' else "ions"
            ax.set_title(f"{self.spacecraft}/{self.sensor} ({self.channel_str}) {particle_str}, data integration time vs. onset distribution stats")
        else:
            ax.set_title(title)

        hour_minute_format = DateFormatter("%H:%M")
        ax.yaxis.set_major_formatter(hour_minute_format)

        ax.grid()
        ax.legend(loc=3, bbox_to_anchor=(1.0, 0.01), prop={'size': 12})

        if save:
            if not savepath:
                savepath = CURRENT_PATH
            plt.savefig(f"{savepath}{os.sep}int_time_vs_onset_distribution_stats_{self.spacecraft}_{self.sensor}_{particle_str}.png", transparent=False,
                        facecolor='white', bbox_inches='tight')

        plt.show()


    def show_onset_distribution(self, index:int=0, xlim=None, show_background=True, save=False, savepath:str=None, legend_loc="out") -> None:
        """
        Displays all the unique onsets found with statistic_onset() -method in a single plot. The mode onset, that is
        the most common out of the distribution, will be shown as a solid line. All other unique onset times are shown
        in a dashed line, with the shade of their color indicating their relative frequency in the distribution.

        Note that statistic_onset() has to be run before this method. Otherwise KeyError will be raised.

        This is in all practicality just a dublicate method of Onset class' method with the same name. This exists just
        as an alternative to save computation time and use more memory instead.

        Parameters:
        -----------
        index : int, default 0
                Choose which distribution from the integration time plot to show
        xlim : {tuple or list with __len__==2}, optional
                The limits of x-axis as pandas-compatible datetime strings, e.g., '2021-09-25 12:00'
        show_background : {bool}
                Boolean switch to show the background on the plot.
        save : {bool}, default False
                Boolean save switch.
        savepath : {str}, optional
                The directory path or subdirectory to save the figure.
        legend_loc : {str}, optional
                Either 'in' or 'out' to control wether the legend is inside or outside the figure borders.
        """

        rcParams["font.size"] = 20

        flux_series = self.list_of_series[index]
        most_likely_onset = self.archive[index]["most_likely_onset"]
        onsets = self.archive[index]["unique_onsets"]

        # Create a colormap for different shades of red for the onsets. Create the map with the amount of onsets+2, so that
        # the first and the final color are left unused. They are basically white and a very very dark red.
        cmap = plt.get_cmap("Reds", len(onsets)+2)

        # Creating the figure and axes
        fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

        # x-axis settings:
        if not xlim:
            xlim = (most_likely_onset[0]-pd.Timedelta(hours=3), most_likely_onset[0]+pd.Timedelta(hours=2))
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1]))

        ax.xaxis_date()
        utc_dt_format1 = DateFormatter("%H:%M")
        ax.xaxis.set_major_formatter(utc_dt_format1)

        # This is what we draw and check y-axis limits against to
        flux_in_plot = flux_series.loc[(flux_series.index > xlim[0]) & (flux_series.index < xlim[1])]

        # y-axis: 
        ylim = set_fig_ylimits(ax=ax, ylim=None, flux_series=flux_in_plot)
        ax.set_ylim(ylim)

        ax.set_yscale("log")
        ax.set_ylabel("Intensity [1/(cm^2 sr s MeV)]")

        # The intensity data, kw where dictates where the step happens. "mid" for the middle of the bin
        ax.step(flux_in_plot.index, flux_in_plot.values, where="mid")

        # Get the Onset object that was used to create this OnsetStatsArray, and call its BootstrapWindow to draw the background
        if show_background:
            self.linked_object.background.draw_background(ax=ax)

        # Marking the onsets as vertical lines of varying shades of red
        for i, onset in enumerate(onsets):

            # Not the most likely onset
            if onset[0] != most_likely_onset[0]:
                linestyle = (5, (10, 3)) if onset[1] > 0.2 else (0, (5, 5))
                ax.axvline(x=onset[0], c=cmap(i+1), ls=linestyle, label=f"{str(onset[0])[11:19]} ({np.round(onset[1]*100,2):.2f} %)")

            # The most likely onset
            # I'm accessing the i+1th color value in the map because I never want the first value (that is white in the standard Reds colormap)
            else:
                ax.axvline(x=onset[0], c="red", label=f"{str(onset[0])[11:19]} ({np.round(onset[1]*100,2):.2f} %)") 

        # Legend and title for the figure.
        if legend_loc=="out":
            # loc=3 means that the legend handle is "lower left"
            legend_handle, legend_bbox = 3, (1.0, 0.01)
        elif legend_loc=="in":
            # loc=4 means that the legend handle is "lower left"
            legend_handle, legend_bbox = 4, (1.0, 0.01)
        else:
            raise ValueError(f"Argument legend_loc has to be either 'in' or 'out', not {legend_loc}")
        ax.legend(loc=legend_handle, bbox_to_anchor=legend_bbox, prop={"size": 12})
        int_time_str = f"{self.integration_times[index]} integration time" if index != 0 else f"{int(self.linked_object.get_minimum_cadence().seconds/60)} min data" if self.linked_object.get_minimum_cadence().seconds>59 else f"{self.linked_object.get_minimum_cadence().seconds} s data"
        ax.set_title(f"{most_likely_onset[0].date()}\nOnset distribution ({int_time_str})")

        if save:
            if not savepath:
                savepath = CURRENT_PATH
            particle_str = "electrons" if self.species=='e' else "protons" if self.species=='p' else "ions"
            plt.savefig(f"{savepath}{os.sep}onset_distribution_{self.spacecraft}_{self.sensor}_{particle_str}.png", transparent=False,
                        facecolor="white", bbox_inches="tight")

        plt.show()


    def show_onset_statistics(self, index:int=0, percentiles=[(15.89,84.1),(2.3,97.7)], xlim=None, show_background=True, save=False, savepath:str=None) -> None:
        """
        Shows the median, mode, mean and confidence intervals for the distribution of onsets got from statistic_onset().

        This is in all practicality just a dublicate method of Onset class' method with the same name. This exists just
        as an alternative to save computation time and use more memory instead.

        Parameters:
        -----------
        index : int, default 0
                Choose which distribution from the integration time plot to show
        percentiles : list of tuples, default [(15.89,84.1),(2.3,97.7)]
                Shows the percentiles of the distribution as shading over the plot
        xlim : {tuple, list} of len()==2
                The left and right limits for the x-axis as pandas-compatible time strings.
        show_background : {bool}, optional
                Boolean switch to show the used background window on the plot
        save : {bool}, default False
                Boolean save switch.
        savepath : {str}, optional
                The directory path or subdirectory to save the figure to.
        """

        # Take the 1-std and 2-std limits from a normal distribution (default)
        sigma1, sigma2 = percentiles[0], percentiles[1]

        # Collect the 1-sigma and 2-sigma confidence intervals out of the onset distribution
        confidence_intervals = self.linked_object.get_distribution_percentiles(onset_list = self.archive[index]["onset_list"], percentiles=[sigma1,sigma2])

        # Collect the median, mode and mean onsets from the distribution
        onset_median = self.archive[index]["median_onset"]
        onset_mode = self.archive[index]["most_likely_onset"][0]
        onset_mean = self.archive[index]["mean_onset"]

        # This is just for plotting the time series with the chosen resolution
        flux_series = self.list_of_series[index]

        # Plot commands and settings:
        fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

        ax.step(flux_series.index, flux_series.values, where="mid")

        # Get the Onset object that was used to create this OnsetStatsArray, and call its BootstrapWindow to draw the background
        if show_background:
            self.linked_object.background.draw_background(ax=ax)

        # Vertical lines for the median, mode and mean of the distributions
        ax.axvline(onset_median, c="red", label="median")
        ax.axvline(onset_mode, c="blue", label="mode")
        ax.axvline(onset_mean, c="darkorange", label="mean")

        # 1-sigma uncertainty shading
        ax.axvspan(xmin=confidence_intervals[0][0], xmax=confidence_intervals[0][1], color="red", alpha=0.3, label=r"1-$\sigma$ confidence")

        #2-sigma uncertainty shading
        ax.axvspan(xmin=confidence_intervals[1][0], xmax=confidence_intervals[1][1], color="blue", alpha=0.3, label=r"2-$\sigma$ confidence")


        # x-axis settings:
        if not xlim:
            xlim = (onset_median-pd.Timedelta(hours=3), onset_median+pd.Timedelta(hours=2))
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1]))

        ax.xaxis_date()
        utc_dt_format1 = DateFormatter("%H:%M")
        ax.xaxis.set_major_formatter(utc_dt_format1)

        # y-axis: 
        flux_in_plot = flux_series.loc[(flux_series.index > xlim[0]) & (flux_series.index < xlim[1])]
        ylim = set_fig_ylimits(ax=ax, ylim=None, flux_series=flux_in_plot)

        # If the lower y-boundary is some ridiculously small number, adjust the y-axis a little
        if np.log10(ylim[1])-np.log10(ylim[0]) > 10:
            ylim[0] = 1e-4

        ax.set_ylim(ylim)
        ax.set_yscale("log")
        ax.set_ylabel("Intensity [1/(cm^2 sr s MeV)]")

        ax.legend()
        int_time_str = f"{self.integration_times[index]} integration time" if index != 0 else f"{int(self.linked_object.get_minimum_cadence().seconds/60)} min data" if self.linked_object.get_minimum_cadence().seconds>59 else f"{self.linked_object.get_minimum_cadence().seconds} s data"
        ax.set_title(f"{onset_median.date()}\nOnset statistics ({int_time_str})")

        if save:
            if not savepath:
                savepath = CURRENT_PATH
            particle_str = "electrons" if self.species=='e' else "protons" if self.species=='p' else "ions"
            plt.savefig(f"{savepath}{os.sep}onset_statistics_{self.spacecraft}_{self.sensor}_{particle_str}.png", transparent=False,
                        facecolor='white', bbox_inches='tight')

        plt.show()


    def plot_cdf(self, index:int=0, prints:bool=False) -> None:
        """
        Plots the Cumulatice Distribution Function (CDF) for a distribution of onset times.

        This is in all practicality just a dublicate method of Onset class' method with the same name. This exists just
        as an alternative to save computation time and use more memory instead.

        Parameters:
        -----------
        index : int, default 0
                Choose which resampled distribution to plot
        prints : bool, default False
                If True, will also print out the median onset time and the confidence intervals.
        """

        # Remember that onset times are sorted according to their probability in ascending order
        onset_timestamps = np.array([pair[0] for pair in self.archive[index]["unique_onsets"]])
        probabilities = np.array([pair[1] for pair in self.archive[index]["unique_onsets"]])

        # Here let's sort the onsets and probabilities in temporal order:
        onset_time_indices = onset_timestamps.argsort() # numpy array.argsort() returns a list of indices of values in ascending order

        onset_timestamps = pd.to_datetime(onset_timestamps[onset_time_indices])
        probabilities = probabilities[onset_time_indices]

        # Effectively calculate the cumulative sum of probabilities
        cumulative_probabilities = [np.nansum(probabilities[:i+1]) for i in range(len(probabilities))]
        
        # Get the median, mode, mean and confidence intervals (1sigma and 2sigma of normal distribution) of the onset distribution
        median_onset, _, _, confidence_intervals_all = self.get_distribution_statistics(onset_statistics=self.archive[index], 
                                                                                    percentiles=[(15.89,84.10), (2.30, 97.7)])
        
        # Flatten the sigma lists for plotting. Look at this list comprehension :D
        confidence_intervals_all = [timestamp for sublist in confidence_intervals_all for timestamp in sublist]

        # Init the figure
        fig, ax = plt.subplots(figsize=(13,7))

        rcParams["font.size"] = 20

        int_time_str = f"{self.integration_times[index]} integration time" if index != 0 else f"{int(self.linked_object.get_minimum_cadence().seconds/60)} min data" if self.linked_object.get_minimum_cadence().seconds>59 else f"{self.linked_object.get_minimum_cadence().seconds} s data"

        # Settings for axes
        ax.xaxis_date()
        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title(f"Cumulative Distribution Function, ({int_time_str})")
        hour_minute_format = DateFormatter("%H:%M")
        ax.xaxis.set_major_formatter(hour_minute_format)

        # Plot command(s)
        ax.step(onset_timestamps, cumulative_probabilities, zorder=3, c='k', where="post")

        colors = ("red", "red", "blue", "blue")
        # Horizontal dashed lines for 1-sigma and 2-sigma limits
        for i, num in enumerate((0.159, 0.841, 0.023, 0.977)):
            label = "15.9-84.1 percentiles" if num==0.841 else ("2.3-97.7 percentiles" if num==0.977 else None)
            ax.axhline(y=num, ls="--", c=colors[i], zorder=1, alpha=0.8, label=label)

        # Vertical dashed lines for 1-sigma and 2-sigma limits
        for i, time in enumerate(confidence_intervals_all):
            ax.axvline(x=time, ls="--", c=colors[i], zorder=1, alpha=0.8)

        ax.axvline(x=median_onset, ls="--", c="maroon", zorder=2, alpha=0.8, label="median onset (50th percentile)")
        ax.axhline(y=0.50, ls="--", c="maroon", zorder=1, alpha=0.8)

        ax.legend(loc=10, bbox_to_anchor=(1.27, 0.88))

        if prints:
            print(f"The median onset time is: {str(median_onset.time())[:8]}")
            # print(f"The np.percentile()-calculated median onset time (50th percentile) is: {str(median_onset.time())[:8]}")
            display(Markdown(r"Confidence interval for 1- $\sigma$:"))
            print(f"{str(confidence_intervals_all[0].time())[:8]} - {str(confidence_intervals_all[1].time())[:8]}")
            print(" ")
            display(Markdown(r"Confidence interval for 2- $\sigma$:"))
            print(f"{str(confidence_intervals_all[2].time())[:8]} - {str(confidence_intervals_all[3].time())[:8]}")

        plt.show()


    def calculate_weighted_uncertainty(self, weight_type:str="uncertainty", returns=False):
        """
        Calculates the weighted confidence intervals based on the confidence intervals of the varying
        integration times. Weighting is done so that the confidence intervals are in a way normalized to 
        to the confidence intervals acquired from the native data resolution.

        Parameters:
        -----------
        weight_type : {str} either 'int_time' or 'uncertainty' (default)
                    Defines the logic at which uncertainties are weighted.

        returns : {bool} default False
                    Switch for this method to also return the weights.
        """

        if weight_type not in ("uncertainty", "int_time"):
            raise ValueError(f"Argument {weight_type} is not a valid value for weight_type! It has to be either 'uncertainty' or 'int_time'.")

        # Asserting the weights so that w_0 = integration time of final distribution and w_{-1} = 1. 
        weights = [w for w in range(len(self.archive), 0, -1)]

        # Collect the confidence intervals and median to their respective arrays
        sigma1_low_bounds = np.array([stats["1-sigma_confidence_interval"][0] for stats in self.archive])
        sigma2_low_bounds = np.array([stats["2-sigma_confidence_interval"][0] for stats in self.archive])
        modes = np.array([stats["most_likely_onset"][0] for stats in self.archive])
        medians = np.array([stats["median_onset"] for stats in self.archive])
        sigma1_upper_bounds = np.array([stats["1-sigma_confidence_interval"][1] for stats in self.archive])
        sigma2_upper_bounds = np.array([stats["2-sigma_confidence_interval"][1] for stats in self.archive])

        if weight_type == "uncertainty":

            # Instead weighting by the width of 2-sigma uncertainty intervals?
            sigma2_widths = np.array([(sigma2_upper_bounds[i] - low_bound) for i, low_bound in enumerate(sigma2_low_bounds)])

            # Get the indices that would sort uncertainities in ASCENDING order, and then flip the order of the indices
            # and add 1, to give the most weight on the smallest timedelta and a weight of 1 to the largest delta
            sorted_indices = np.argsort(sigma2_widths)
            weights = np.flip(sorted_indices+1)

        # Weighting the confidence intervals and the median
        self.w_sigma1_low_bound = weight_timestamp(weights=weights, timestamps=sigma1_low_bounds)
        self.w_sigma2_low_bound = weight_timestamp(weights=weights, timestamps=sigma2_low_bounds)
        self.w_mode = weight_timestamp(weights=weights, timestamps=modes)
        self.w_median = weight_timestamp(weights=weights, timestamps=medians)
        self.w_sigma1_upper_bound = weight_timestamp(weights=weights, timestamps=sigma1_upper_bounds)
        self.w_sigma2_upper_bound = weight_timestamp(weights=weights, timestamps=sigma2_upper_bounds)

        # Propagate the infromation of the weighted median and confidence intervals to the linked object too
        self.set_w_median_and_confidence_intervals( self.w_mode,
                                                    self.w_median,
                                                    self.w_sigma1_low_bound, self.w_sigma1_upper_bound,
                                                    self.w_sigma2_low_bound, self.w_sigma2_upper_bound)

        if returns:
            return self.w_sigma1_low_bound, self.w_sigma2_low_bound, self.w_median, self.w_sigma1_upper_bound, self.w_sigma2_upper_bound


def check_confidence_intervals(confidence_intervals, time_reso):
    """
    Small helper function to make sure that the confidence intervals are at least as large as the time resolution
    of the time series data.
    """

    center_of_intervals = datetime_mean(np.append(confidence_intervals[0],confidence_intervals[1]))

    new_intervals = []
    for interval in confidence_intervals:

        if interval[1] - interval[0] < pd.Timedelta(time_reso):
            new_interval0, new_interval1 = center_of_intervals - pd.Timedelta(time_reso)/2, center_of_intervals + pd.Timedelta(time_reso)/2
            new_intervals.append((new_interval0,new_interval1))
        else:
            new_intervals.append((interval[0], interval[1]))

    return new_intervals


def weight_timestamp(weights, timestamps):
    """
    Calculates weighted timestamp from a list of weights and timestamps.

    Parameters:
    -----------
    weight : {list of ints}
    timestamp : {list of pd.datetimes}
    """

    for i, ts in enumerate(timestamps):

        # init the series that will do the averaging calculation for the timestamps
        if i == 0:
            weight_times_timestamps = pd.Series(data=[timestamps[i]]*weights[i])

        else:
            new_series = pd.Series(data=[ts]*weights[i])

            weight_times_timestamps = pd.concat([weight_times_timestamps, new_series], ignore_index=True)
        
    # At this point weight_times_timestamps is a series of n*ts[0] + (n-1)*ts[1] + (n-2)*ts[2] + ... + 1*ts[-1]
    # A weighted average of the timestamps is just a mean of this series
    weighted_timestamp = weight_times_timestamps.mean()

    # Return the concatenation of the lower and upper bounds
    return weighted_timestamp


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


def estimate_uncertainty(data_resolution, onset_std):
    """
    Calculates the estimated uncertainty of an onset time based on the resolution of the time series data,
    and the standard deviation of the onset distribution.
    
    Parameters:
    ----------
    data_resolution : str
                        Pandas-compatible time string representing the time resolution of the time series data
    onset_std : pandas._libs.tslibs.timedeltas.Timedelta
                        A pandas-timedelta object that represents the standard deviation of the onset distribution
    Returns:
    ---------
    true_uncertainty : pandas._libs.tslibs.timedeltas.Timedelta
                        The sum of data resolution and the std of onset distribution.
    """

    if data_resolution[-1:] == 's':
        reso_minutes = 0
        reso_seconds = int(data_resolution[:-1])
    if data_resolution[-3:] == "min":
        reso_minutes = int(data_resolution[:-3])
        reso_seconds = 0

    # The uncertainty has to be at least the time resolution of the data
    minimum_uncertainty = pd.Timedelta(minutes=reso_minutes, seconds=reso_seconds)

    # The true uncertainty in this function is defined as the sum of time resolution 
    # and the standard deviation of the onset distribution
    true_uncertainty = minimum_uncertainty + onset_std
    
    # Another approach could maybe be that the true uncertainty is the max of time resolution and onset_std
    # true_uncertainty = np.nanmax(minimum_uncertainty, onset_std)

    return true_uncertainty


def get_x_errors(e_min, e_max, inverse_betas, mass_energy:float):
        """
        Calculates the width of energy channels in terms of the inverse speed (beta = v/c)

        Parameters:
        -----------
        e_min : array-like
                    The lower boundaries of energy channels in terms of eVs
        e_max : array-like
                    The higher boundaries of energy channels in terms of eVs
        inverse_betas : array-like
                    The inverse betas corresponding to channel nominal energies
        mass_energy : float
                    E = mc^2, the energy related to the mass of the particle

        Returns:
        -----------
        x_errors_lower : tuple
                    The lower boundaries of energy channels in terms of inverse beta
        x_errors_upper : tuple
                    The higher boundaries of energy channels in terms of inverse beta
        x_errors : tuple
                    The widths of energy channels in terms of 1/beta
        """

        # Error bars in x direction:
        # First turn channel energy boundaries to Joules
        e_min_joule = [((e*u.eV).to(u.J)).value for e in e_min]
        e_max_joule = [((e*u.eV).to(u.J)).value for e in e_max]

        # Calculate beta from channel energy boundaries
        beta_lower = [np.sqrt(1-(( e_j/mass_energy + 1)**(-2))) for e_j in e_min_joule]
        beta_upper = [np.sqrt(1-(( e_j/mass_energy + 1)**(-2))) for e_j in e_max_joule]

        # The Inverse of beta
        beta_inv_lower = [1/x for x in beta_lower]
        beta_inv_upper = [1/x for x in beta_upper]

        # Calculate the difference between boundary and actual value
        x_errors_lower = [x-inverse_betas[indx] for indx, x in enumerate(beta_inv_lower)]
        x_errors_upper = [inverse_betas[indx]-x for indx, x in enumerate(beta_inv_upper)]

        x_errors = [x-x_errors_upper[i] for i, x in enumerate(x_errors_lower)]

        return x_errors_lower, x_errors_upper, x_errors


def get_time_errors(onset_times, spacecraft):
    """
    Gets the errors in y-direction (time).

    Parameters:
    -----------
    onset_times : array-like
                All the onset times
    spacecraft : str
                The spacecraft identifier string.
    
    Returns:
    -----------
    time_error : list(pandas.Timedelta)
                A list of time errors as pandas-Timedelta objects
    """

    # Error in the y-direction (time):
    time_error = []

    for ts in onset_times:

        try:

            if isinstance(ts.freqstr,str):

                # freqstr could be negative, check!
                if ts.freqstr[0] == '-':
                    freqstr = ts.freqstr[1:]
                else:
                    freqstr = ts.freqstr
                time_error.append(pd.Timedelta(freqstr))

            else:
                if spacecraft in ["stereo-a", "stereo a", "stereo-b", "stereo b", "sta", "stb"]:
                    time_error.append(pd.Timedelta('1T'))
                else:
                    time_error.append(pd.Timedelta('1S'))

        except AttributeError:

            if spacecraft in ["stereo-a", "stereo a", "stereo-b", "stereo b", "sta", "stb"]:
                time_error.append(pd.Timedelta('1T'))
            else:
                time_error.append(pd.Timedelta('1S'))

    return time_error


def calculate_inverse_betas(channel_energies, mass_energy):
    """
    Calculates the inverse unitless speeds (1/beta = c/v) of particles given channel energies.

    Parameters:
    -----------
    channel_energies : array-like
                The nominal energies of energy channels.
    
    Returns:
    -----------
    inverse_betas : list
                The inverse betas corresponding to the nominal energies of the channels.
    """

    # Transform channel energies in eV to Joules
    e_Joule = [((En*u.eV).to(u.J)).value for En in channel_energies]

    # Beta, the unitless speed (v/c)
    beta = [np.sqrt(1-(( e_j/mass_energy + 1)**(-2))) for e_j in e_Joule]

    # Invert the beta values
    inverse_betas = np.array([1/b for b in beta])

    return inverse_betas


def tsa(t0:datetime.datetime, L:float, v:float):
    """
    Calculates the time shifted injection time assuming a path length L and particle speed v

    Parameters:
    -----------
    t0 : datetime.datetime
            The onset time of the event
    L : float
            The path length in AU
    v : float
            particle speed in m/s
    """

    path_in_meters = L * u.au.to(u.m)
    v_in_ms = v * u.m / u.s
    
    path_delta_sec = path_in_meters / v_in_ms

    t_inj = t0 - pd.Timedelta(seconds=path_delta_sec.value)

    return t_inj


def datetime_to_sec(onset_times):
    """
    Returns the the timestamps of onset times as seconds since the linux Epoch (first of January, 1970)

    Parameters:
    -----------
    onset_times : {array_like}
    """

    # date_in_sec is the list that goes to the fitting function
    dates_in_sec = []

    # Calculate the difference between an onset time and the start of the epoch in seconds
    for date in onset_times:
        dates_in_sec.append((date.to_pydatetime() - datetime.datetime(1970,1,1,0,0,0)).total_seconds())

    return dates_in_sec


def sample_mean_and_std(start, end, flux_series, sample_size=None, prints_warning=True):
    """
    Calculates the mean and standard deviation of the background period by
    taking a random sample of the background window of size sample_size * background_window.

    Parameters:
    -----------
    start: datetime
            Starting point of the calculation
    end: datetime
            Ending point of the calculation
    flux_series: pandas Series
            An array of observed values, indexed by datetime
    sample_size: float, default None
            A fraction of all the points to be considered in the calculation
            of mean and std. The sample is a random sample.
    prints_warning: bool, default True
            The function prints a warning if the sample size is small (data points < 100)

    Returns:
    --------
    (mean, std): tuple
            mean is the mean of the background period, std is its standard deviation
    """

    # Background contains all the points that could be considered for the calculation
    # of mean and std
    background = flux_series.loc[(flux_series.index >= start) & (flux_series.index < end)]

    # If sample_size is not provided, assume that all data points are picked
    if not sample_size:
        print("sample_size was not provided, assuming a 100 % sample.")
        sample_size = int(len(background))

    # Sample size is a fraction of the amount of background window data points
    else:
        if sample_size < 0.0 or sample_size > 1.0:
            raise Exception(f"The sample size must be between 0 and 1, not {sample_size}!")
        else:
            sample_size = int(sample_size * len(background))
            if len(background) < 100 and prints_warning:
                print(f"Warning, random sample is only {sample_size} data points!")
        

    # A random sample of all the background data points.
    sample = np.random.choice(background, replace=False, size=sample_size)

    # Calculate mean and std of the chosen sample of data points
    mean = np.nanmean(sample)
    std = np.nanstd(sample)

    return (mean, std)


def detrend_onsets(timestamps, shift):
    """
    Takes as an input an array of datetimes, and shifts them forward in time by the given amount.
    
    Parameters:
    ---------
    timestamps : array-like
                A list or an array containing the timestamps to be shifted
    shift : str
                Pandas-compatible timedelta string to shift the timestamps, e.g., '30s' or '1min'
    
    Returns:
    --------
    new_timestamps : array-like
                The detrended timestamps
    """

    # Init a new array for the timestamps to ensure that vectorization works
    timestamps_arr = np.asarray(timestamps, dtype=object)
    new_timestamps = timestamps_arr + pd.Timedelta(shift)

    return new_timestamps


def datetime_mean(arr):
    """
    Returns the mean of an array of datetime values
    """
            
    arr1 = pd.Series(arr)
    
    return arr1.mean()


def find_earliest_possible_onset(flux_series, cusum_function, onset_time):
    """
    Finds the earliest possible onset time by backtracking the cusum_function starting from the onset time and finding the
    first nonzero occurence of the cusum function.

    Parameters:
    -----------
    flux_series : pandas series
                    The series from which the onset was found. Intensity measurements indexed by datetimes.
    cusum_function : array-like
                    The values of the cusum function that is used to identify the onset time in the aforementioned time series.
    onset_time : datetime
                    The true onset time found from the time series with the cusum method.
    Returns:
    -----------
    earliest_possible_onset : datetime
                    The earliest possible onset time, identified by the first nonzero occurence of the cusum function 
                    backtracking from the true onset time.
    """
    # First to find the index of the onset, use numpy.where()
    datetimes = flux_series.index

    index_of_onset = np.where(datetimes == onset_time)[0][0]

    # Before looping througfh the cusum_function, flip it to make sure we are advancing in the past, not towards the future
    flipped_cusum_function = np.flip(cusum_function)

    # Init the earliest possible onset time
    earliest_possible_onset = onset_time

    # We'll go though the cusum_function, backtracking towards the past starting from the onset time
    for i in range(index_of_onset, len(flipped_cusum_function)):

        # The earliest possible onset time is the first nonzero occurence of cusum function PRIOR to the true onset
        if flipped_cusum_function[i] == 0:
            earliest_possible_onset = datetimes[i-1]
            break

    return earliest_possible_onset




# ==============================================================================================

def get_time_reso(series):
    """
    Returns the time resolution of the input series.

    Parameters:
    -----------
    series: Pandas Series

    Returns:
    ----------
    resolution: str
            Pandas-compatible freqstr
    """

    for i in range(len(series)):
        try:
            resolution = (series.index[i+1] - series.index[i]).seconds
            break
        except AttributeError:
            continue

    # STEREO data often looks like 59s resolution
    if resolution==59:
        return "1min"

    if resolution%60==0:
        return f'{int(resolution/60)}min'
    else:
        return f"{resolution}s"

#===========================================================================================

def calc_chnl_nominal_energy(e_min, e_max, mode='gmean'):
    """
    Calculates the nominal energies of each channel, given the lower and upper bound of
    the channel energies. Channel energy should be given in eVs.

    Parameters:
    -----------
    e_min, e_max: np.arrays
            Contains lower and hugher energy bounds for each channel respectively in eV
    mode: str, default='gmean'
        The mode for calculating nominal channel energy. Use 'gmean' for geometric mean, and
        'upper' for the upper energy bound.
    
    Returns:
    --------
    energies: np.array 
            Contains the nominal channel energy for each channel
    """

    # geometric mean
    if mode=='gmean':
        energies = np.sqrt(np.multiply(e_min,e_max))
        return energies
    
    # upper bound of energy channels
    elif mode=='upper':
        energies = e_max
        return energies
    
    else:
        raise Exception("Choose either 'gmean' or 'upper' as the nominal energy.")

#===========================================================================================

def path_length_calculator(distance, solar_wind = 400):
    """
    Calculates the length of the nominal parker spiral arm.

    Parameters:
    ------------
    distance : float, int
                heliospheric distance of the object in AU
    solar_wind = float, int
                solar wind speed in km/s, default = 400

    Returns:
    -----------
    path_length : float
                the path length in AU assuming the nominal Parker spiral
    """
    from sunpy.sun import constants as sunc

    # init the relevant constants
    OMEGA = sunc.sidereal_rotation_rate.value * (u.deg/u.day).to(u.rad/u.s) #rad/s, the equatorial sidereal rotation period of the Sun

    solar_wind_au = solar_wind * u.km.to(u.au)

    # These constants appears many times in the equation
    a = solar_wind_au/OMEGA
    psi = distance/a

    sqrt_term = np.sqrt(psi*psi + 1)

    # Calculate the length of the spiral arc
    path_length = 0.5 * a * ( psi*sqrt_term + np.log(psi + sqrt_term) ) # np.log() is the natural logarithm

    return path_length

#===========================================================================================

def flux2series(flux, dates, cadence=None):
    """
    Converts an array of observed particle flux + timestamps into a pandas series
    with the desired cadence.

    Parameters:
    -----------
    flux: an array of observed particle fluxes
    dates: an array of corresponding dates/times
    cadence: str - desired spacing between the series elements e.g. '1s' or '5min'

    Returns:
    ----------
    flux_series: Pandas Series object indexed by the resampled cadence
    """

    # from pandas.tseries.frequencies import to_offset

    # set up the series object
    flux_series = pd.Series(flux, index=dates)
    
    # if no cadence given, then just return the series with the original
    # time resolution
    if cadence is not None:
        try:
            flux_series = flux_series.resample(cadence, origin='start').mean()
            flux_series.index = flux_series.index + pd.tseries.frequencies.to_offset(pd.Timedelta(cadence)/2)
        except ValueError:
            raise Warning(f"Your 'resample' option of [{cadence}] doesn't seem to be a proper Pandas frequency!")

    return flux_series

# ======================================================================

def calculate_mean_and_std(start, end, flux_series):
    """
    This function calculates the mean and standard deviation of the background period
    which is used in the onset analysis. 

    Parameters:
    -----------
    start: datetime
            Starting point of the calculation
    end: datetime
            Ending point of the calculation
    flux_series: pandas Series
            An array of observed values, indexed by datetime

    Returns:
    --------
    (mean, std): tuple
            mean is the mean of the background period, std is its standard deviation
    """

    background = flux_series.loc[(flux_series.index >= start) & (flux_series.index < end)]

    mean = np.nanmean(background)
    std = np.nanstd(background)

    return (mean, std)

# =======================================================================

def erase_glitches(series, glitch_threshold, time_barr):
    """
    Every value above glitch_threshold before time_barr will be replace with nan.
    Also return the erased values in a series called "glitches", in case we need them for something, e.g. plotting.

    Parameters:
    -------------
    series : Pandas Series
                A series of intensity values indexed by pandas-compatible datetime-objects.
    glitch_threshold : float
                The maximum number above which in the right timeframe the values of the series are considered glitches.
    time_barr : datetime
                The time at which glitch_threshold stop applying.
    
    Returns:
    -------------
    series2 : Pandas Series
                The new series, with glitches omitted.
    glitches : Pandas Series
                All the found glitches collected to their own series, indexed by datetime.
    """

    if isinstance(time_barr,str):
        time_barr = pd.to_datetime(time_barr)

    # logically: if a value is BOTH before time_barr AND larger than glitch_threshold,
    # then the statement is false, and the corresponding value will be replaced with nan.
    glitches = series.loc[(series.values >= glitch_threshold) & (series.index < time_barr)]
    series2 = series.where((series.values < glitch_threshold) | (series.index >= time_barr))

    return series2, glitches

# ============================================================================

def onset_determination_v2(ma_sigma, flux_series, cusum_window, avg_end, sigma_multiplier : int = 2) -> list :
    """
    Calculates the CUSUM function to find an onset time from the given time series data.

    Parameters:
    -----------
    ma_sigma : tuple(float, float)

    flux_series : pandas Series

    cusum_window : int

    avg_end : pandas datetime

    sigma_multiplier : float, int

    Returns:
    --------
    list(ma, md, k_round, h, norm_channel, cusum, onset_time)
    """

    # assert date and the starting index of the averaging process
    date = flux_series.index

    # First cut the part of the series that's before the ending of the averaging window. Then subtract that
    # from the size of the original series to get the numeral index that corresponds to the ending of the 
    # averaging window. starting_index is then the numeral index of the first datapoint that belongs outside the averaging window
    cut_series = flux_series.loc[flux_series.index > avg_end]
    start_index = flux_series.size - cut_series.size

    ma = ma_sigma[0]
    sigma = ma_sigma[1]

    md = ma + sigma_multiplier*sigma

    # k may get really big if sigma is large in comparison to mean
    try:
        k = (md-ma)/(np.log(md)-np.log(ma))
        k_round = round(k/sigma)

    except (ValueError, OverflowError) as error:
        # the first ValueError I encountered was due to ma=md=2.0 -> k = "0/0"
        # OverflowError is due to k = inf
        # print(error)
        k_round = 1

    # choose h, the variable dictating the "hastiness" of onset alert
    if k <= 1.0:
        h = 1
    else:
        h = 2

    alert = 0
    cusum = np.zeros(len(flux_series))
    norm_channel = np.zeros(len(flux_series))
    
    # set the onset as default to be NaT (Not a daTe)
    onset_time = pd.NaT

    # start at the index where averaging window ends
    for i in range(start_index,len(cusum)):

        # normalize the observed flux
        norm_channel[i] = (flux_series[i]-ma)/sigma

        # calculate the value for ith cusum entry
        cusum[i] = max(0, norm_channel[i] - k_round + cusum[i-1])

        # check if cusum[i] is above threshold h, if it is -> increment alert
        if cusum[i]>h:
            alert=alert+1
        else:
            alert=0

        # cusum_window(default:30) subsequent increments to alert means that the onset was found
        if alert == cusum_window:
            onset_time = date[i - alert]
            break

    # ma = mu_a = background average
    # md = mu_d = background average + 2*sigma
    # k_round = integer value of k, that is the reference value to poisson cumulative sum
    # h = 1 or 2,describes the hastiness of onset alert
    # onset_time = the time of the onset

    return [ma, md, k_round, h, norm_channel, cusum, onset_time]


#==============================================================================================

def bootstrap_mus_and_sigmas(flux_series, window_start, window_end, n_bstraps, sample_size=None):
    """
    The function looks at flux_series through a time window, and extracts n_bstraps samples
    from it. The size of a sample size <sample_size> must not be greater than the amount of data points
    inside the time window. It then calculates the mean and standard deviation of those samples and returns
    an array of means and an array of stds.
    """ 
    
    # make a selection including only the measurements in the window
    window = flux_series.loc[(flux_series.index>=window_start) & (flux_series.index<window_end)]

    if sample_size is None:
        #default sample size is set to 50 % of total datapoints in a window
        sample_size = int(len(window)/2)
    
        # Print out a warning if the amount of datapoints in a window is less than 100
        # This is done ONLY IF no sample size is provided
        if len(window)<100:
            print(f"Warning! The ensemble of background values is {len(window)}<100. \nRaising sample size to 75 %.")
            sample_size = int(3*len(window)/4)

    # if sample_size is not None, check that it's a reasonable value
    else:
        if sample_size < 0.0 or sample_size > 1.0:
            raise Exception(f"The sample size must be between 0 and 1, not {sample_size}!")
        else:
            sample_size = int(sample_size * len(window))
            if len(window) < 100:
                print(f"Warning, random sample is only {sample_size} data points!")

    # initialize arrays
    mean_arr = np.zeros(n_bstraps)
    std_arr = np.zeros(n_bstraps)

    # repeat n_bstraps times
    for i in range(n_bstraps):

        # look inside the window and randomly pick sample_size amount of measurements there
        sample = np.random.choice(window, replace=False, size=sample_size)
        
        # then calculate the mean and standard deviation of that sample
        mean_arr[i] = np.nanmean(sample)
        std_arr[i] = np.nanstd(sample)


    return mean_arr, std_arr

#================================================================================

def unique_vals(onset_list, prints=False):
    '''
    Returns all the unique values from a list of onset times, and the corresponding counts. Counts also include
    the amount of NaT values at the end of the array, which is why len(uniques) == len(counts)-1.

    Parameters:
    -----------
    onset_list : list
                A list of onset times
    prints : bool, default False
                Switch to print out the relative abundances of each onset in the list
    
    Returns:
    -----------
    uniques : np.array
                An array of the UNIQUE appearances of the onset times in the input list
    counts : np.array
                An array of the number of appearances of each onset in the input list + the amount of NaTs.
    '''
    
    # Convert to pandas datetimes to ensure functionality with NaTs
    onsets = pd.to_datetime(onset_list)
    
    # Construct a mask, which is True on NaT values
    natmask = pd.isnull(onsets)
    
    # Also keep track of the amount of NaTs
    nat_amount = len(np.where(natmask)[0])
    
    # Use the NOT mask to get all the values which are NOT NaTs
    masked_onsets = onsets[~natmask]
    
    # Sort the times in chronological order
    masked_onsets = np.sort(masked_onsets)
    
    # Returns unique values from the masked array, and an array corresponding to the 
    # mount of each unique instance
    uniques, counts = np.unique(masked_onsets, return_counts=True)
    
    # ...Again convert to pandas datetimes
    uniques = pd.to_datetime(uniques)
    
    # Add the amount of NaTs to the end of counts
    counts = np.append(counts, nat_amount)

    count_sum = sum(counts)
    
    if prints:
        for i, uni_val in enumerate(uniques):
            print(f"{str(uni_val)[:-4]},  n={counts[i]}  ({np.round(100*counts[i]/count_sum, 2)} %)")

        print(f"NaTs: {counts[-1]}  ({np.round(100*counts[-1]/count_sum, 2)} %)")
        print("#----------------------------------")
    
    # Check that not ALL onsets are NaTs (this is evident if counts only holds one number)
    if len(counts) == 1:
        print(f" function unique_vals() output: {onsets[0]}, {counts}")
        return [onsets[0]], counts
    else:
        return uniques, counts

#================================================================================================================

def calc_chnl_nominal_energy(e_min, e_max, mode='gmean'):
    '''
    Calculates the nominal energies of each channel, given the lower and upper bound of
    the channel energies. Channel energy should be given in eVs.
    
    Returns:
        energies: np.array containing the nominal channel energy for each channel
    '''

    #geometric mean
    if mode=='gmean':
        energies = np.sqrt(np.multiply(e_min,e_max))
        return energies
    
    #upper bound of energy channels
    elif mode=='upper':
        energies = e_max
        return energies
    
    else:
        raise Exception("Choose either 'gmean' or 'upper' as the nominal energy.")

#==============================================================================

def sc_distance(mag_data):

    from sunpy.coordinates import frames, get_horizons_coord

    pos = get_horizons_coord("Solar Orbiter", mag_data.index, "id")  # (lon, lat, radius) in (deg, deg, AU)
    pos = pos.transform_to(frames.HeliographicCarrington)
    dist = np.round(pos.radius.value, 2)

    return dist

#==============================================================================

def seek_fit_and_errors(x,y,xerr,yerr, guess=None):
    '''
    Seeks a fit and its errors by ODR (Orthogonal Distance Regression) 
    https://www.tutorialspoint.com/scipy/scipy_odr.htm

    Parameters
    -----------
    x, y :  arrays
            values for the x and y axes
    xerr, yerr : array
            errors of x and y, respectively
    guess : {array-like, list, tuple} with len()==2, default None
            The first guess values for the ODR fit.

    Returns:
    ----------
    out : {ODR.regression}
            An object containing information of the fit and it's errors
    '''

    from scipy.odr import ODR, Model, RealData

    # Define a polynomial of rank 1 to fit the data with.
    def linear_func(p, x):
        slope, const = p
        return slope*x + const

    # Create a model for fitting.
    linear_model = Model(linear_func)

    # Create a RealData object using our initiated data from above.
    data = RealData(x, y, xerr, yerr)

    if not guess:
        slope0, const0 = 1.0, y[0]
    else:
        slope0, const0 = guess[0], datetime_to_sec([pd.to_datetime(guess[1])])[0]

    # Set up ODR with the model and data.
    odr = ODR(data, linear_model, beta0=[slope0, const0])

    # Run the regression.
    out = odr.run()

    # The built-in pprint method can be used to show the results
    # out.pprint()
    
    return out


def calculate_cusum_window(time_reso, window_minutes:int=30) -> int:
    """
    Calculates the cusum window in terms of datapoints.
    
    Parameters:
    -----------
    time_reso: str or None
                A pandas-compatible time resolution string. Examples: '20s' or '45min'
    window_minutes: int, default 30
                The amount of minutes of data that a cusum window corresponds to.
    
    Returns:
    ----------
    cusum_window: int
                Cusum window in terms of datapoints.
    """

    if time_reso[-3:] == "min":
        datapoint_multiplier = 1
        reso_value = float(time_reso[:-3])
    elif time_reso[-1] == 's':
        datapoint_multiplier = 60
        reso_value = int(time_reso[:-1])
    else:
        raise Exception("Time resolution format not recognized. Use either 'min' or 's'.")

    cusum_window = (window_minutes*datapoint_multiplier)/reso_value
    
    return int(cusum_window)


def set_fig_ylimits(ax:plt.Axes, ylim:(list,tuple)=None, flux_series:pd.Series=None):
    """
    Sets the vertical axis limits of the figure, given an Axes and a tuple or list of y-values.
    """

    # In case not otherwise specified, set the lower limit to half of the smallest plotted value,
    # and the higher limit to 1.5 times the highest value
    if ylim is None:
        ylim = [np.nanmin(flux_series[flux_series > 0]) * 0.5,
                np.nanmax(flux_series) * 1.5]

        # If the lower y-boundary is some ridiculously small number, adjust the y-axis a little
        if np.log10(ylim[1])-np.log10(ylim[0]) > 10:
            ylim[0] = 1e-4

    ylim = ax.set_ylim(ylim)

    return ylim


def _isnotebook():
    # https://stackoverflow.com/a/39662359/2336056
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False

# ============================================================================================================================
# this will be ran when importing
# set Jupyter notebook cells to 100% screen size:
if _isnotebook():
    from IPython.core.display import HTML, display
    display(HTML(data="""<style> div#notebook-container { width: 99%; } div#menubar-container { width: 85%; } div#maintoolbar-container { width: 99%; } </style>"""))
