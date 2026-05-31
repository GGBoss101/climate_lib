# import modules
import numpy as np
import pandas as pd
import xarray as xr
import scipy as sp
import dask
from scipy import stats
import metpy
import metpy.calc as mpcalc
from metpy.units import units
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from sklearn.linear_model import LinearRegression
# from windspharm.xarray import VectorWind

# import Ngl
import geocat.comp

import warnings
warnings.filterwarnings("ignore", message=".*multiple fill values.*")
warnings.filterwarnings("ignore", message="Interpolation point out of data bounds encountered")

def linear_trend(da):
    """
    Compute linear regression statistics along the time dimension
    of an Xarray DataArray.

    Args:
        da (DataArray): Input data with a "time" dimension.

    Returns:
        slope (DataArray): Linear trend slope.
        intercept (DataArray): Regression intercept.
        rvalue (DataArray): Correlation coefficient.
        pvalue (DataArray): Statistical p-value.
        stderr (DataArray): Standard error of the slope.
    """

    # Create numerical time coordinates for regression
    time_coords = np.arange(len(da["time"]))

    # Wrapper around scipy linear regression
    def _linregress(y):

        return stats.linregress(
            time_coords,
            y,
        )[:5]

    # Apply regression along the time dimension
    results = xr.apply_ufunc(
        _linregress,
        da,
        vectorize=True,
        input_core_dims=[["time"]],
        output_core_dims=[[], [], [], [], []],
        output_dtypes=[
            np.float64,
            np.float64,
            np.float64,
            np.float64,
            np.float64,
        ],
        dask="parallelized",
    )

    # Extract regression outputs
    slope = results[0].rename("slope")
    intercept = results[1].rename("intercept")
    rvalue = results[2].rename("rvalue")
    pvalue = results[3].rename("pvalue")
    stderr = results[4].rename("stderr")

    return (
        slope,
        intercept,
        rvalue,
        pvalue,
        stderr,
    )