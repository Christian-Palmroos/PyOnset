
A software package to determine SEP event onset times and associated uncertainties.
The method is described in detail in Palmroos et al., 2024 (submitted).

The full functionality of PyOnset is encompassed within three classes, which along with their respective methods are described here.

$\\frac{1}{\pi}$

Onset
=====

The ``Onset`` class is the main tool of the Pyonset package. It inherits all the functionality of the ``Event`` class from the SEPpy module (Palmroos et al., 2022), offering full backwards-compatibility. Perhaps the most notable feature of ``Onset`` is its ability to automatically download SEP data for all the SEPpy-supported missions (Solar Orbiter, Parker Solar Probe, STEREO-A&B, SOHO and Wind), and accept user-defined data for other missions in the form of Pandas DataFrames. 

Attributes:
----------

*  ``onset_statistics``
    Contains the channel-respective onset times and related uncertainties in a dictionary. The statistics are produced by the         ``onset_statistics_per_channel()``-method.

Methods:
-------

*  ``plot_all_channels()``
    Creates a quick-look time-series plot of all the channels included in the data of the object. 

*  ``cusum_onset()``
    Finds the onset time of an event for the given timeframe and background using the Poisson-CUSUM method. Creates a figure of       the time-series, showing the chosen background, background parameters ($\mu$ and $\mu_{d}$) and the onset time.

*  ``onset_statistics_per_channel()``
    Automatically finds the onset time with the respective uncertainty for all of the chosen energy channels. The onset times and     uncertainties therein will be saved to the class attribute ``onset_statistics``, and used by the ``VDA()``-method. 

*  ``set_custom_channel_energies()``
    In case of custom input data, the low and high boundaries of the energy channels must be manually given to the class.             Otherwise e.g., ``VDA()`` method can not run.

*  ``VDA()``
    Performs velocity dispersion analysis to the onset times found by ``onset_statistics_per_channel()``-method.


BootstrapWindow
===============

The ``BootstrapWindow`` class defines what is considered the pre-event background period, how many bootstrap runs are ran for each onset time, and if the window is shifted during analysis or not.

Attributes:
----------

*  ``start``
    A pandas-compatible datetime string that defines the starting point of the pre-event background, e.g., "2024-12-31 00:00".

*  ``end``
    Defines the ending point of the background window. See ``start``.

*  ``bootstraps``
    The amount of bootstrap runs for an onset time assuming constant integration time. For ample statistics, a value of e.g.,         1000 is recommended.

*  ``n_shifts``
    The number of times the background is shifted forwards. by default this value is 0, and it should only be explicitly given as     an input if the user knows what they're doing. Shifting the ``BootstrapWindow`` forward resets the number of bootstrap runs,          effectively multiplying the TOTAL number of bootstrap runs by ``n_shifts``+1. 

Methods:
-------

*  ``print_max_recommended_reso()``
    Prints out the maximum recommended resolution that the time series should be averaged to in order to preserve the minimum         recommended amount of data points (100) inside the background.


OnsetStatsArray
===============

The ``OnsetStatsArray`` class is mainly used to store statistics related to all the found onset times within the ``Onset`` class. It also acts as a cogwheel in the machinery that calculates the confidence intervals related to the onset times, due to its notable feature of containing copies of the time-averaged intensity time-series that are integral part of the Poisson-CUSUM Bootstrap hybrid method. in practice, the usage of this class is not relevant other than for data visualization purposes. For each energy channel one ``OnsetStatsArray`` is created when ``onset_statistics_per_channel()`` is called.

Attributes:
----------

*  ``statistics``
    Contains, in a dictionary, the mean and standard devations of all the bootstrapped runs for a single integration time. Also       contains all the found onset times and the mode, i.e., most common onset time for the bootstrap runs.

*  ``archive``
    A list containing all the ``statistic`` dictionaries for all the integration times. 

*  ``linked_object``
    Each ``OnsetStatsArray`` is created for a specific ``Onset`` object. Only statistics of the ``linked_object`` may be added to     the ``archive``.

Methods:
-------

*  ``onset_time_histogram()``
    Displays the probability density histogram of all found onset times for the given integration time. The integration times         appear in the order they were created, i.e., the native cadence corresponds to index=0, etc. Also displays the corresponding      distributions ~68% and ~95% as red and blue shadings, respoectively on overlaid on the histogram.

*  ``integration_time_plot()``
    Shows the mean, mode and median onset times as a function of integration time. Also Displays the ~68% and ~95% confidence         intervals as red and blue shading, respectively.

*  ``show_onset_distribution()``
    For a given integration time (given by index, see ``onset_time_histogram()``) plots the distribution of unique onset times        found by the method on top of the intensity time series.

*  ``show_onset_statistics()``
    For a given integration time (given by index, see ``onset_time_histogram()``) plots the ~68% and ~95% confidence intervals on     top of the intensity time series as red and blue shadings, respectively.
