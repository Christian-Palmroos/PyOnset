
"""
This file contains some calculation utilities for PyOnset.

"""


__author__ = "Christian Palmroos"

import numpy as np
import pandas as pd


def k_parameter(mu:float, sigma:float, sigma_multiplier:int|float) -> float:
    """
    The standard version of k for the z-standardized intensity CUSUM.

    Parameters:
    -----------
    mu : {float, np.ndarray} 
                    The mean of the background.
    sigma : {float, np.ndarray} 
                    The standard deviation of the background.
    sigma_multiplier : {int,float} 
                    The multiplier for mu_{d} != 0.

    Returns:
    --------
    k_param : {float, np.ndarray} Type depends on the input type.
                    A valid k_parameter value (k >= 0).
    """
    if sigma_multiplier == 0:
        raise ValueError("sigma_multiplier may not be 0!")

    # Let's not divide by zero.
    # Only do this check if mu and sigma are singular values, numpy will take
    # care of the cases with arrays.
    if not isinstance(mu, (list, np.ndarray)):
        if mu==0 or sigma==0:
            return 0

    nominator = sigma_multiplier
    denominator = np.log(1 + (sigma_multiplier*sigma)/mu)

    k_param = (nominator/denominator) - (mu/sigma)

    if not isinstance(k_param, (int, float, np.int64, np.float64, np.longdouble)):
        return k_param

    return k_param if k_param >= 0 else 0


def k_classic(mu:float, sigma:float, sigma_multiplier:float) -> float:
    """
    The classical k-parameters as defined in the classical Poisson-CUSUM
    """
    if sigma_multiplier == 0:
        raise ValueError("sigma_multiplier may not be 0!")

    # Let's not divide by zero
    if not isinstance(mu, (list, np.ndarray)):
        if mu==0:
            return 0

    nominator = sigma_multiplier * sigma
    denominator = np.log(1 + nominator/mu)

    return np.round(nominator/denominator)


def k_legacy(mu:float, sigma:float, sigma_multiplier:float) -> float:
    """
    The old standard k-parameter for SEPpy.
    """
    if sigma_multiplier == 0:
        raise ValueError("sigma_multiplier may not be 0!")

    # Let's not divide by zero
    if not isinstance(mu, (list, np.ndarray)):
        if mu==0:
            return 0

    nominator = sigma_multiplier
    denominator = np.log(1 + (sigma_multiplier*sigma)/mu)

    # In legacy SEPpy, k is rounded to the nearest integer
    return np.round(nominator/denominator)


def experimental_k_param(mu:float, sigma:float, sigma_multiplier:float) -> float:
    """
    k-parameter but the argument of logarithm is flipped.
    """
    if sigma_multiplier == 0:
        raise ValueError("sigma_multiplier may not be 0!")

    # Let's not divide by zero
    if isinstance(mu, float) and isinstance(sigma, float):
        if mu==0 or sigma==0:
            return 0

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
    z_score_series : {pd.Series} The z-standardized version of the input series.
    """

    # z-score makes no sense with 0/0
    if mu!=0 and sigma!=0:
        standard_values = (series.values - mu) / sigma
    else:
        print("Warning! Z-score calculation impossible due to background mean = background standard deviation = 0.\nFalling back to scaled intensity.")

        # Find a value that makes sure that the intensity rise goes over unity instantly
        try:
            scale_factor = np.reciprocal(np.nanmin(series.values[series>0]))
        except ValueError as e:
            print(e)
            print("This is caused by the total lack of nonzero values in the data selection. Terminating without action.")
            return series

        standard_values = series.values * scale_factor

    return pd.Series(standard_values, index=series.index)


def sigma_norm(series:pd.Series, sigma:float) -> pd.Series:
    """
    Normalizes intensity to background (bg) standard deviation (std).

    Parameters:
    ----------
    series : {pd.Series} Time series representation of intensity.
    sigma : {float} The bg std.

    Returns:
    --------
    standard_series : {pd.Series} Intensity normalized to bg std.
    """

    if sigma!=0:
        standard_values = series.values/sigma
    else:
        standard_values = series.values

    return pd.Series(standard_values, index=series.index)