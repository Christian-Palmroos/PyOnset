
"""
This file contains some calculation utilities for PyOnset.

"""


__author__ = "Christian Palmroos"

import numpy as np
import pandas as pd


def k_parameter(mu, sigma, n_sigma):

    md = mu + (n_sigma*sigma)

    nominator = md-mu
    denominator = np.log(md) - np.log(mu)

    return nominator/(denominator*sigma)


def z_score(series:pd.Series, mu:float, sigma:float):
    """
    Standardizes the given series such that its mean is 0 and standard deviation is 1.

    Parameters:
    -----------
    series : {pd.Series}
    mu : {float}
    sigma : {float}

    Returns:
    x_score_series : {pd.Series} The z-standardized version of the input series.
    """
    standard_values = (series.values - mu) / sigma
    return pd.Series(standard_values, index=series.index)


