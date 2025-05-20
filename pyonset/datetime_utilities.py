
"""
This file contains functions for datetime management in PyOnset.

"""


__author__ = "Christian Palmroos"

import datetime

import numpy as np
import pandas as pd

def weight_timestamp(weights:list, timestamps:list):
    """
    Calculates weighted timestamp from a list of weights and timestamps.

    Parameters:
    -----------
    weights : {list of weights}
    timestamp : {list of pd.datetimes}
    """

    # It could happen that there is only one timestamp and that it's a NaT. In this case numpy will falsely
    # average the NaT to 1970-01-01 00:00, so check it here to avert this.
    if len(timestamps)==1 and isinstance(timestamps[0], pd._libs.tslibs.nattype.NaTType):
        return pd.NaT

    # First make sure timestamps come numpy datetime64 format (in nanoseconds)
    timestamps = np.array([t.asm8 for t in timestamps])

    # Mask the weights and timestamps that contain nans/NaTs with these indices
    mask = ~np.isnan(timestamps)

    # Calculate weighted_avg. Use np.ndarray.view() to convert datetime values to floating point numbers so that
    # math can be done with them. The timestamp that this yields will also be a floating point number
    wavg_ns = np.average(timestamps[mask].view(dtype="float64"), weights=weights[mask])

    # Convert the float to a numpy timestamp
    wavg_timestamp = np.array(wavg_ns).view(dtype="datetime64[ns]")

    # Return the weighted average timestamp (final conversion to pandas datetime is perhaps not necessary, but let's be sure)
    return pd.to_datetime(wavg_timestamp)


def check_confidence_intervals(confidence_intervals, time_reso):
    """
    Small helper function to make sure that the confidence intervals are at least as large as the time resolution
    of the time series data.
    """

    # Initialize minimum inner boundaries (no outer boundary can be inside these)
    min_low_boundary = confidence_intervals[0][0]
    min_high_boundary = confidence_intervals[0][1]

    new_intervals = []
    for i, interval in enumerate(confidence_intervals):

        if interval[1] - interval[0] < pd.Timedelta(time_reso):

            # Calculate the center of this interval
            center_of_interval = datetime_mean(np.array([interval[0], interval[1]]))

            # Push the boundaries of the interval further from their center by half of the time resolution
            new_interval0, new_interval1 = center_of_interval - pd.Timedelta(time_reso)/2, center_of_interval + pd.Timedelta(time_reso)/2

            # Check that inner boundaries are never outside outer boundaries
            # Also check that 1sigma boundary is never outside 2sigma boundaries
            if i==0:
                min_low_boundary = new_interval0
                min_high_boundary = new_interval1

            if new_interval0 > min_low_boundary:
                new_interval0 = min_low_boundary
            
            if new_interval1 < min_high_boundary:
                new_interval1 = min_high_boundary

        # Even if the interval length is larger than the time resolution, some outer boundary may be left inside
        # an inner interval
        else:

            # Here check that outer interval boundaries are not inside inner interval boundaries
            if interval[0] > min_low_boundary:
                new_interval0 = min_low_boundary
            else:
                new_interval0 = interval[0]
            
            if interval[1] < min_high_boundary:
                new_interval1 = min_high_boundary
            else:
                new_interval1 = interval[1]

        new_intervals.append((new_interval0,new_interval1))

    return new_intervals


def datetime_mean(arr):
    """
    Returns the mean of an array of datetime values
    """
            
    arr1 = pd.Series(arr)
    
    return arr1.mean()


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


def get_time_reso(series: pd.Series) -> str:
    """
    Returns the time resolution of the input series.

    Parameters:
    -----------
    series: Pandas Series

    Returns:
    ----------
    resolution: {str}
            Pandas-compatible freqstr
    """

    if series.index.freq is None:
        
        # Better approach:
        index_diffs = series.index.diff()
        
        # There might be unregular time differences, pick the most
        # appearing one -> mode.
        diffs, counts = np.unique(index_diffs, return_counts=True)
        mode_dt = pd.Timedelta(diffs[np.argmax(counts)])

        # Round up to the nearest second, because otherwise e.g., STEREO / SEPT data
        # that may have cadence of '59.961614005' seconds is interpreted to have nanosecond
        # precision.
        mode_dt = mode_dt.round(freq='s')

        # If less than 1 minute, express in seconds
        divisor = 60 if mode_dt.resolution_string == "min" else 1
        
        return f"{mode_dt.seconds//divisor} {mode_dt.resolution_string}"

    
    else:
        freq_str = series.index.freq.freqstr
        return freq_str if freq_str!="min" else f"1 {freq_str}"


def calculate_cusum_window(time_reso, window_minutes:int=30) -> int:
    """
    Calculates the cusum window in terms of datapoints.
    Cusum window means the the amount of time that is demanded of the cusum function
    to stay above the hastiness threshold before the onset of the event is 
    identified.
    
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

    if isinstance(time_reso, (pd._libs.tslibs.offsets.Second, pd._libs.tslibs.offsets.Minute, pd._libs.tslibs.offsets.Hour)):
        time_reso = time_reso.freqstr

    if time_reso[-3:] == "min":
        datapoint_multiplier = 1
        reso_value = float(time_reso[:-3]) if len(time_reso) > 3 else 1
    elif time_reso[-1] == 'T':
        datapoint_multiplier = 1
        reso_value = int(time_reso[:-1]) if len(time_reso) > 1 else 1
    elif time_reso[-1] == 'H':
        datapoint_multiplier = 1/60
        reso_value = float(time_reso[:-1]) if len(time_reso) > 1 else 1
    elif time_reso[-1] in ('S', 's'):
        datapoint_multiplier = 60
        reso_value = int(time_reso[:-1]) if len(time_reso) > 1 else 1
    else:
        raise Exception(f"Time resolution format ({time_reso}) not recognized. Use either 'min' or 's'.")

    cusum_window = (window_minutes*datapoint_multiplier)/reso_value
    
    return int(cusum_window)


def datetime_nanmedian(timestamps):
    """
    Finds the median datetime, ignoring NaTs.

    timestamps : {array-like}
                    An array or list of pandas-compatible timestamps.
    """

    if not isinstance(timestamps,np.ndarray):
        timestamps = np.array(timestamps)

    # Mask away NaTs. pd.isnull() also applies on NaTs
    timestamps_masked = timestamps[~pd.isnull(timestamps)]

    # Sort the array
    timestamps_masked.sort()
    
    # Pick the middlemost value to return
    return timestamps_masked[len(timestamps_masked)//2]


def get_figdate(dt_array):
    """
    Gets the string representation as YYYY-MM-DD from a list of datetimes, ignoring NaTs.
    """

    figdate = pd.NaT

    for date in dt_array:

        if not pd.isnull(date):
            figdate = date
            break

    try:
        date = figdate.date().strftime("%Y-%m-%d")

    # An attributeerror is cause by figdate being numpy.datetime64 -object. handle
    # it appropriately
    except AttributeError:
        date = np.datetime_as_string(figdate, unit='D')

    return date


def find_biggest_nonzero_unit(timedelta):
    """
    Finds the biggest unit of time that describes the timedelta.
    Available units of time include a day, an hour, a minutes and a second.
    """
    
    days, hours, minutes, seconds, _, _, _ = timedelta.components
    
    if days != 0:
        return 'D'
    if hours != 0:
        return 'H'
    if minutes != 0:
        return "min"
    return 's'
