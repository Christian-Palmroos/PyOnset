
"""
This file contains some calculation utilities for PyOnset.

"""


__author__ = "Christian Palmroos"

import numpy as np
import pandas as pd


def k_parameter(mu:float, sigma:float, sigma_multiplier:float) -> float:

    md = mu + (sigma_multiplier*sigma)

    nominator = md-mu
    denominator = np.log(md) - np.log(mu)

    try:
        k = nominator/(denominator*sigma)
    except (ValueError, OverflowError):
        k = 1 if mu > 0 else 0
    return k


def experimental_k_param(mu:float, sigma:float, sigma_multiplier:float) -> float:

    nominator = sigma_multiplier

    denominator = np.log( 1 + mu/(sigma_multiplier*sigma) )

    try:
        k = nominator/denominator
    except (ValueError, OverflowError):
        k = 1 if mu > 0 else 0
    return k


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


