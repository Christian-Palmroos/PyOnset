
"""
This file contains the OnsetStatsArray class for PyOnset.

"""


__author__ = "Christian Palmroos"

import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter

from IPython.display import Markdown, display

from .plot_utilities import COLOR_SCHEME, STANDARD_FIGSIZE, LEGEND_SIZE, TITLE_FONTSIZE, \
                            AXLABEL_FONTSIZE, TICK_LABELSIZE, \
                            set_standard_ticks, set_legend, set_fig_ylimits

from .datetime_utilities import get_figdate, weight_timestamp

CURRENT_PATH = os.getcwd()


class OnsetStatsArray:
    """
    Contains statistics and uncertainty of a single onset at a particular energy channel.
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
        self.channel_str = onset_object.recently_examined_channel_str
        self.channel_id = onset_object.current_channel_id

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
        if len(self.archive) >= 1:
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


    def onset_time_histogram(self, integration_time_index=0, binwidth="1 min", xlim=None, ylims=None, legend_loc=1,
                            save=False, savepath=None, grid=True):
        """
        A method to display the probability density histogram for the distribution of onset times collected
        to the object.

        Parameters:
        -----------
        integration_time_index : {int}, default 0
                                    Chooses the integration time, where 0 represents the native data time resolution.
        binwidth : {str}, default '1 min'
                                    Sets the width of time bins on the x-axis.
        xlims : {tuple}
        ylims : {tuple}
        legend_loc : {int}, default 1
                                    Sets the location of the legend.
        save : {bool}, default False
                                    Boolean save switch.
        savepath : {str}, optional
                                    Path to the directory to save the figure.
        grid : {bool}, default True
                                    Boolean switch to apply gridlines.
        """

        stats = self.archive[integration_time_index]

        # Plotting 
        fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

        # If xlims not manually defined, let them be \pm 2 minutes from the first and last onset of the distribution
        if not xlim:
            # There might be NaTs among the onset list, so here choose only the valid ones to avert 
            # TypeError when taking np.nanmin() of the list
            valid_onset_list = np.array(stats["onset_list"])[~pd.isnull(stats["onset_list"])]
            xlim = (np.nanmin(valid_onset_list) - pd.Timedelta(minutes=2), np.nanmax(valid_onset_list) + pd.Timedelta(minutes=2))
        ax.set_xlim(xlim)

        # Show percentage on y-axis
        yvalues = [m/10 for m in range(0,11)]
        ax.set_yticks(yvalues)
        ax.set_yticklabels(['{}'.format(np.round(x, 1)) for x in yvalues])


        # Bins for the x axis 
        half_bin = pd.Timedelta(seconds=30)
        bins = pd.date_range(start=xlim[0]+half_bin, end=xlim[1]+half_bin, freq=binwidth).tolist()

        onset_frequencies = np.ones_like(stats["onset_list"])/len(stats["onset_list"])

        # Plotting the histogram rectangles
        bar_heights, bins, patches = ax.hist(stats["onset_list"], bins=bins, color="lightblue", edgecolor="black", weights=onset_frequencies, zorder=2)

        if not ylims:
            ylims = (0, np.nanmax(bar_heights)+0.02)
        ax.set_ylim(ylims)
    
        set_standard_ticks(ax=ax)

        # Mean, mode and median onset times as vertical solid lines
        ax.axvline(stats["mean_onset"], linewidth=2.0, color=COLOR_SCHEME["mean"], label=f"mean {str(stats['mean_onset'].time())[:8]}", zorder=3)
        ax.axvline(stats["most_likely_onset"][0], linewidth=2.0, color=COLOR_SCHEME["mode"], label=f"mode {str(stats['most_likely_onset'][0].time())[:8]}", zorder=3)
        ax.axvline(stats["median_onset"], linewidth=2.0, color=COLOR_SCHEME["median"], label=f"median {str(stats['median_onset'].time())[:8]}", zorder=3)

        # 1 -and 2-sigma intervals as red and blue dashed lines
        ax.axvspan(xmin=stats["2-sigma_confidence_interval"][0], xmax=stats["2-sigma_confidence_interval"][1], color=COLOR_SCHEME["2-sigma"], alpha=0.15, label="~95 % confidence", zorder=1)
        ax.axvspan(xmin=stats["1-sigma_confidence_interval"][0], xmax=stats["1-sigma_confidence_interval"][1], color=COLOR_SCHEME["1-sigma"], alpha=0.15, label="~68 % confidence", zorder=1)

        ax.set_xlabel(f"Time ({stats['mean_onset'].strftime('%Y-%m-%d')})", fontsize=AXLABEL_FONTSIZE)
        ax.set_ylabel("PD", fontsize=AXLABEL_FONTSIZE)

        ax.grid(visible=grid, axis="both")

        integration_time_str = f"{self.integration_times[integration_time_index]} integration time" if pd.Timedelta(self.integration_times[integration_time_index]) != self.linked_object.native_resolution else f"{self.integration_times[integration_time_index]} data"

        ax.set_title(f"Probability density for {self.linked_object.background.bootstraps} onset times\n{integration_time_str}", fontsize=TITLE_FONTSIZE)
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))

        ax.legend(loc=legend_loc, bbox_to_anchor=(1.0, 1.0), fancybox=True, ncol=3, fontsize = LEGEND_SIZE)
        #set_legend(ax=ax, legend_loc="in", fontsize=LEGEND_SIZE)

        if save:
            if savepath is None:
                plt.savefig("onset_times_histogram.png", transparent=False, facecolor="white", bbox_inches="tight")
            else:
                plt.savefig(f"{savepath}{os.sep}onset_times_histogram.png", transparent=False, facecolor="white", bbox_inches="tight")

        plt.show()

    def integration_time_plot(self, title=None, ylims=None, save=False, savepath:str=None, grid=True) -> None:
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
        grid : {bool}, default True
                Boolean switch to turn on gridlines.
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

        # these are the real weighted mode and median
        mean_of_modes = self.linked_object.onset_statistics[self.channel_id][0]
        mean_of_medians = self.linked_object.onset_statistics[self.channel_id][1]

        figdate = get_figdate(modes)

        # Initializing the figure
        fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

        # These integration time strings are transformed to floating point numbers in units of minutes
        xaxis_int_times = [pd.Timedelta(td).seconds/60 for td in self.integration_times]

        ax.set_xlim(0, xaxis_int_times[-1]+1)

        if ylims:
            ax.set_ylim(pd.to_datetime(ylims[0]), pd.to_datetime(ylims[1]))

        # We only want to have integer tickmarks
        ax.set_xticks(range(0,int(xaxis_int_times[-1]+1)))

        ax.scatter(xaxis_int_times, means, s=165, label="mean", zorder=2, color=COLOR_SCHEME["mean"], marker=".")
        ax.scatter(xaxis_int_times, medians, s=115, label="median", zorder=2, color=COLOR_SCHEME["median"], marker="^")
        ax.scatter(xaxis_int_times, modes, s=115, label="mode", zorder=2, color=COLOR_SCHEME["mode"], marker="p")

        ax.axhline(y=mean_of_medians, color=COLOR_SCHEME["median"], lw=2, label=f"Mean of medians:\n{str(mean_of_medians.time())[:8]}")
        ax.axhline(y=mean_of_modes, color=COLOR_SCHEME["mode"], lw=2, label=f"Mean of modes:\n{str(mean_of_modes.time())[:8]}")

        ax.fill_between(xaxis_int_times, y1=conf1_lows, y2=conf1_highs, facecolor=COLOR_SCHEME["1-sigma"], alpha=0.3, zorder=1)
        ax.fill_between(xaxis_int_times, y1=conf2_lows, y2=conf2_highs, facecolor=COLOR_SCHEME["2-sigma"], alpha=0.3, zorder=1)

        # For some reason all the labelsizes appear smaller in the plots created with this method, which is 
        # why I add some extra size to them.

        ax.set_xlabel("Data integration time [min]", fontsize=AXLABEL_FONTSIZE+6)
        ax.set_ylabel(f"{figdate}\nTime [HH:MM]", fontsize=AXLABEL_FONTSIZE+6)

        if not title:
            # particle_str = "electrons" if self.species=='e' else "protons" if self.species=='p' else "ions"
            ax.set_title(f"{self.spacecraft.upper()} / {self.sensor.upper()} ({self.channel_str}) {self.linked_object.s_identifier}\ndata integration time vs. onset distribution stats", 
                         fontsize=TITLE_FONTSIZE+9)
        else:
            ax.set_title(title, fontsize=TITLE_FONTSIZE+9)

        set_standard_ticks(ax=ax)
        # More tick settings (nonstandard)
        ax.tick_params(which="both", labelsize=TICK_LABELSIZE+9)
        hour_minute_format = DateFormatter("%H:%M")
        ax.yaxis.set_major_formatter(hour_minute_format)

        ax.grid(visible=grid, axis="both")

        set_legend(ax=ax, legend_loc="out", fontsize=LEGEND_SIZE+4)
        # ax.legend(loc=3, bbox_to_anchor=(1.0, 0.01), prop={'size': 24})

        if save:
            if not savepath:
                savepath = CURRENT_PATH
            plt.savefig(f"{savepath}{os.sep}int_time_vs_onset_distribution_stats_{self.spacecraft}_{self.sensor}_{self.linked_object.s_identifier}.png", transparent=False,
                        facecolor='white', bbox_inches='tight')

        plt.show()


    def show_onset_distribution(self, integration_time_index:int=0, xlim=None, show_background=True, 
                                save=False, savepath:str=None, legend_loc="out") -> None:
        """
        Displays all the unique onsets found with statistic_onset() -method in a single plot. The mode onset, that is
        the most common out of the distribution, will be shown as a solid line. All other unique onset times are shown
        in a dashed line, with the shade of their color indicating their relative frequency in the distribution.

        Note that statistic_onset() has to be run before this method. Otherwise KeyError will be raised.

        This is in all practicality just a dublicate method of Onset class' method with the same name. This exists just
        as an alternative to save computation time and use more memory instead.

        Parameters:
        -----------
        integration_time_index : int, default 0
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

        # rcParams["font.size"] = 20

        flux_series = self.list_of_series[integration_time_index]
        most_likely_onset = self.archive[integration_time_index]["most_likely_onset"]
        onsets = self.archive[integration_time_index]["unique_onsets"]

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

        set_standard_ticks(ax=ax)
        ax.xaxis_date()
        utc_dt_format1 = DateFormatter("%H:%M")
        ax.xaxis.set_major_formatter(utc_dt_format1)

        # This is what we draw and check y-axis limits against to
        flux_in_plot = flux_series.loc[(flux_series.index > xlim[0]) & (flux_series.index < xlim[1])]

        # Gets the date of the event
        figdate = get_figdate(flux_in_plot.index)

        # y-axis: 
        ylim = set_fig_ylimits(ax=ax, ylim=None, flux_series=flux_in_plot)
        ax.set_ylim(ylim)

        ax.set_yscale("log")
        ax.set_ylabel(self.linked_object.unit, fontsize=AXLABEL_FONTSIZE)
        ax.set_xlabel(f"Time ({figdate})", fontsize=AXLABEL_FONTSIZE)

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

        set_legend(ax=ax, legend_loc=legend_loc, fontsize=LEGEND_SIZE)

        int_time_str = f"{self.integration_times[integration_time_index]} integration time" if pd.Timedelta(self.integration_times[integration_time_index]) != self.linked_object.get_minimum_cadence() else f"{self.integration_times[integration_time_index]} data"
        # int_time_str = f"{self.integration_times[index]} integration time" if index != 0 else f"{int(self.linked_object.get_minimum_cadence().seconds/60)} min data" if self.linked_object.get_minimum_cadence().seconds>59 else f"{self.linked_object.get_minimum_cadence().seconds} s data"
        ax.set_title(f"{self.spacecraft.upper()}/{self.sensor.upper()} ({self.channel_str}) {self.linked_object.s_identifier}\nOnset distribution ({int_time_str})", fontsize=TITLE_FONTSIZE)

        if save:
            if not savepath:
                savepath = CURRENT_PATH
            plt.savefig(f"{savepath}{os.sep}onset_distribution_{self.spacecraft}_{self.sensor}_{self.linked_object.s_identifier}.png", transparent=False,
                        facecolor="white", bbox_inches="tight")

        plt.show()


    def show_onset_statistics(self, integration_time_index:int=0, xlim=None, show_background=True, 
                              save=False, savepath:str=None, legend_loc="out") -> None:
        """
        Shows the median, mode, mean and confidence intervals for the distribution of onsets got from statistic_onset().

        This is in all practicality just a dublicate method of Onset class' method with the same name. This exists just
        as an alternative to save computation time and use more memory instead.

        Parameters:
        -----------
        integration_time_index : int, default 0
                Choose which distribution from the integration time plot to show
        xlim : {tuple, list} of len()==2
                The left and right limits for the x-axis as pandas-compatible time strings.
        show_background : {bool}, optional
                Boolean switch to show the used background window on the plot
        save : {bool}, default False
                Boolean save switch.
        savepath : {str}, optional
                The directory path or subdirectory to save the figure to.
        """

        # Collect the median, mode and mean onsets from the distribution
        onset_median = self.archive[integration_time_index]["median_onset"]
        onset_mode = self.archive[integration_time_index]["most_likely_onset"][0]
        onset_mean = self.archive[integration_time_index]["mean_onset"]
        confidence_intervals1 = self.archive[integration_time_index]["1-sigma_confidence_interval"]
        confidence_intervals2 = self.archive[integration_time_index]["2-sigma_confidence_interval"]

        # This is just for plotting the time series with the chosen resolution
        flux_series = self.list_of_series[integration_time_index]

        # Plot commands and settings:
        fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

        ax.step(flux_series.index, flux_series.values, where="mid")

        # Get the Onset object that was used to create this OnsetStatsArray, and call its BootstrapWindow to draw the background
        if show_background:
            self.linked_object.background.draw_background(ax=ax)

        # Vertical lines for the median, mode and mean of the distributions
        ax.axvline(onset_median, c=COLOR_SCHEME["median"], label="median")
        ax.axvline(onset_mode, c=COLOR_SCHEME["mode"], label="mode")
        ax.axvline(onset_mean, c=COLOR_SCHEME["mean"], label="mean")

        # 1-sigma uncertainty shading
        ax.axvspan(xmin=confidence_intervals1[0], xmax=confidence_intervals1[1], color=COLOR_SCHEME["1-sigma"], alpha=0.3, label="~68 % confidence")

        #2-sigma uncertainty shading
        ax.axvspan(xmin=confidence_intervals2[0], xmax=confidence_intervals2[1], color=COLOR_SCHEME["2-sigma"], alpha=0.3, label="~95 % confidence")


        # x-axis settings:
        if not xlim:
            xlim = (onset_median-pd.Timedelta(hours=3), onset_median+pd.Timedelta(hours=2))
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1]))

        set_standard_ticks(ax=ax)
        ax.xaxis_date()
        utc_dt_format1 = DateFormatter("%H:%M")
        ax.xaxis.set_major_formatter(utc_dt_format1)

        # y-axis: 
        flux_in_plot = flux_series.loc[(flux_series.index > xlim[0]) & (flux_series.index < xlim[1])]
        ylim = set_fig_ylimits(ax=ax, ylim=None, flux_series=flux_in_plot)

        # Date of the event
        #figdate = onset_mean.date().strftime("%Y-%m-%d")
        figdate = get_figdate(dt_array=flux_in_plot.index)

        # If the lower y-boundary is some ridiculously small number, adjust the y-axis a little
        if np.log10(ylim[1])-np.log10(ylim[0]) > 10:
            ylim[0] = 1e-4

        ax.set_ylim(ylim)
        ax.set_yscale("log")
        ax.set_ylabel(self.linked_object.unit, fontsize=AXLABEL_FONTSIZE)
        ax.set_xlabel(f"Time ({figdate})", fontsize=AXLABEL_FONTSIZE)

        set_legend(ax=ax, legend_loc=legend_loc, fontsize=LEGEND_SIZE)

        # particle_str = "electrons" if self.species=='e' else "protons" if self.species=='p' else "ions"
        int_time_str = f"{self.integration_times[integration_time_index]} integration time" if pd.Timedelta(self.integration_times[integration_time_index]) != self.linked_object.get_minimum_cadence() else f"{self.integration_times[integration_time_index]} data"
        # int_time_str = f"{self.integration_times[index]} integration time" if index != 0 else f"{int(self.linked_object.get_minimum_cadence().seconds/60)} min data" if self.linked_object.get_minimum_cadence().seconds>59 else f"{self.linked_object.get_minimum_cadence().seconds} s data"
        ax.set_title(f"{self.spacecraft.upper()}/{self.sensor.upper()} ({self.channel_str}) {self.linked_object.s_identifier}\nOnset statistics ({int_time_str})", fontsize=TITLE_FONTSIZE)

        if save:
            if not savepath:
                savepath = CURRENT_PATH
            plt.savefig(f"{savepath}{os.sep}onset_statistics_{self.spacecraft}_{self.sensor}_{self.linked_object.s_identifier}.png", transparent=False,
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
        fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

        # rcParams["font.size"] = 20

        int_time_str = f"{self.integration_times[index]} integration time" if pd.Timedelta(self.integration_times[index]) != self.linked_object.get_minimum_cadence() else f"{self.integration_times[index]} data"
        # int_time_str = f"{self.integration_times[index]} integration time" if index != 0 else f"{int(self.linked_object.get_minimum_cadence().seconds/60)} min data" if self.linked_object.get_minimum_cadence().seconds>59 else f"{self.linked_object.get_minimum_cadence().seconds} s data"

        # Settings for axes
        ax.xaxis_date()
        ax.set_xlabel("Time", fontsize=AXLABEL_FONTSIZE)
        ax.set_ylabel("Cumulative Probability", fontsize=AXLABEL_FONTSIZE)
        ax.set_title(f"Cumulative Distribution Function, ({int_time_str})", fontsize=TITLE_FONTSIZE)
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
            display(Markdown("~68 % confidence interval"))
            print(f"{str(confidence_intervals_all[0].time())[:8]} - {str(confidence_intervals_all[1].time())[:8]}")
            print(" ")
            display(Markdown("~95 % confidence interval"))
            print(f"{str(confidence_intervals_all[2].time())[:8]} - {str(confidence_intervals_all[3].time())[:8]}")

        plt.show()


    def calculate_weighted_uncertainty(self, weight_type:str="inverse_variance", returns=False):
        """
        Calculates the weighted confidence intervals based on the confidence intervals of the varying
        integration times. Weighting is done so that the confidence intervals are in a way normalized to 
        to the confidence intervals acquired from the native data resolution.

        Parameters:
        -----------
        weight_type : {str} either 'int_time', 'uncertainty' or 'inverse_variance' (default)
                    Defines the logic at which uncertainties are weighted.

        returns : {bool} default False
                    Switch for this method to also return the weights.
        """

        if weight_type not in ("uncertainty", "int_time", "inverse_variance"):
            raise ValueError(f"Argument {weight_type} is not a valid value for weight_type! It has to be either 'uncertainty', 'int_time' or 'inverse_variance'.")

        # Asserting the weights so that w_0 = integration time of final distribution and w_{-1} = 1. 
        int_weights = [w for w in range(len(self.archive), 0, -1)]

        # Collect the confidence intervals and median to their respective arrays
        sigma1_low_bounds = np.array([stats["1-sigma_confidence_interval"][0] for stats in self.archive])
        sigma2_low_bounds = np.array([stats["2-sigma_confidence_interval"][0] for stats in self.archive])
        modes = np.array([stats["most_likely_onset"][0] for stats in self.archive])
        medians = np.array([stats["median_onset"] for stats in self.archive])
        sigma1_upper_bounds = np.array([stats["1-sigma_confidence_interval"][1] for stats in self.archive])
        sigma2_upper_bounds = np.array([stats["2-sigma_confidence_interval"][1] for stats in self.archive])

        if weight_type == "inverse_variance":

            # Inverse-variance weighting:
            # https://en.wikipedia.org/wiki/Inverse-variance_weighting
            weights = self.inverse_variance_weights()

        elif weight_type == "uncertainty":

            # Instead weighting by the width of 2-sigma uncertainty intervals
            sigma2_widths = np.array([(sigma2_upper_bounds[i] - low_bound) for i, low_bound in enumerate(sigma2_low_bounds)])

            # Convert the widths to minutes as floating point numbers
            sigma2_widths = np.array([width.seconds/60 for width in sigma2_widths])

            # The weights here are the inverse's of the widths of the 2-sigma uncertainty intervals
            weights = np.array([1/width for width in sigma2_widths])

            if len(weights)==0:
                weights = np.array(int_weights)

        # Using integration time as weighting
        else:
            weights = np.array(int_weights)

        # Weighting the confidence intervals and the median
        self.w_sigma1_low_bound = weight_timestamp(weights=weights, timestamps=sigma1_low_bounds)
        self.w_sigma2_low_bound = weight_timestamp(weights=weights, timestamps=sigma2_low_bounds)
        self.w_mode = weight_timestamp(weights=weights, timestamps=modes)
        self.w_median = weight_timestamp(weights=weights, timestamps=medians)
        self.w_sigma1_upper_bound = weight_timestamp(weights=weights, timestamps=sigma1_upper_bounds)
        self.w_sigma2_upper_bound = weight_timestamp(weights=weights, timestamps=sigma2_upper_bounds)

        # Introduce a check to move the boundaries away from mode and median if they are too close
        self.check_weighted_timestamps()

        # Propagate the infromation of the weighted median and confidence intervals to the linked object too
        self.set_w_median_and_confidence_intervals( self.w_mode,
                                                    self.w_median,
                                                    self.w_sigma1_low_bound, self.w_sigma1_upper_bound,
                                                    self.w_sigma2_low_bound, self.w_sigma2_upper_bound)

        if returns:
            return self.w_mode, self.w_median, self.w_sigma1_low_bound, self.w_sigma1_upper_bound, self.w_sigma2_low_bound, self.w_sigma2_upper_bound

    def check_weighted_timestamps(self):
        """
        Checks that the intervals boundaries are separated from the mode and the median by at least half
        of the finest cadence used (minimum integration time). Moves them if not.
        """

        # We will not allow for a separation of less than the smalles integration time used
        #min_separation = self.linked_object.get_minimum_cadence()/2
        min_separation = pd.Timedelta(self.integration_times[0])/2

        if self.w_mode - self.w_sigma1_low_bound < min_separation:
            self.w_sigma1_low_bound = self.w_mode - min_separation
        if self.w_mode - self.w_sigma2_low_bound < min_separation:
            self.w_sigma2_low_bound = self.w_mode - min_separation
        
        if self.w_sigma1_upper_bound - self.w_mode < min_separation:
            self.w_sigma1_upper_bound = self.w_mode + min_separation
        if self.w_sigma2_upper_bound - self.w_mode < min_separation:
            self.w_sigma2_upper_bound = self.w_mode + min_separation


    def inverse_variance_weights(self):
        """
        Calculates the inverse variance weights for onset distributions.
        
        Uses the inverse variance IN MINUTES as weights.
        
        Returns: weights : {np.ndarray} array of shape==(len(archive),)
        """

        # This is the variance of a distribution with 1000 onset times wide 1 minute equally spaced.
        # Experimentally found with: pd.date_range("2024-01-01 00:00:00", "2024-01-01 00:01:00", 1000)
        VAR_OF_MINUTE_WIDE_DISTRIBUTION = 0.08350016683180035

        weights = np.zeros([len(self.archive)], dtype=float)

        # Loop through each archive individually
        for i, stats in enumerate(self.archive):

            # Fetch onset list
            onsets = pd.DatetimeIndex(stats["onset_list"])

            # Convert to nanoseconds, because variance can NOT be calculated for
            # datetime type objects
            onsets_as_nanoseconds = onsets.astype(np.int64)

            # NaTs equal to negative nanosecond values. Replace with nans as they are 
            # not considered for calculation anyway
            onsets_as_nanoseconds = np.where(onsets_as_nanoseconds<0, np.nan, onsets_as_nanoseconds)

            # Calculating the variance of these numbers is in essence the variance
            # of the onset times (in units of nanoseconds squared)
            variance_in_nanoseconds = np.nanvar(onsets_as_nanoseconds)

            # Multiply with 10^-18 to convert nanoseconds squared to seconds squared
            variance_in_seconds = variance_in_nanoseconds * 1e-18

            # Dividing by 3600 = 60*60 converts seconds squared to minutes squared
            variance_in_minutes = variance_in_seconds / 3600.

            # This check is here because sometimes numpy calculates the variance
            # for a delta function to be an extremely small number of the order of
            # 10^-17. I suspect it is due to calculating precision as in the case of
            # a delta function the variance is 0. Anyway replace that here with 
            # the variance of an equi-spaced distribution with width 1 minute times the time reso.
            weights[i] = variance_in_minutes if variance_in_minutes > 1e-10 else VAR_OF_MINUTE_WIDE_DISTRIBUTION * (i+1)

        # Return the reciprocal (1 / weights) 
        return np.reciprocal(weights)

