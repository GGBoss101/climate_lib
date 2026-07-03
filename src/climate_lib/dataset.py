"""
This module contains functions for manipulating climate datasets.
"""


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

from climate_lib.utils import *
from climate_lib.constants import *

import warnings
warnings.filterwarnings("ignore", message=".*multiple fill values.*")
warnings.filterwarnings("ignore", message="Interpolation point out of data bounds encountered")


def make_ds(
    da_temp,
    name,
    longname,
    title,
    pos_dir,
    units,
    full_units,
):
    """Convert a DataArray into a Dataset with metadata.

    Args:
        da_temp (xarray.DataArray): Input data array.
        name (str): Variable name for the output dataset.
        longname (str): Descriptive variable name.
        title (str): Dataset title.
        pos_dir (str): Positive direction of the variable.
        units (str): Variable units.
        full_units (str): Expanded description of the units.

    Returns:
        xarray.Dataset: Dataset containing the input data with updated metadata.
    """

    # Assign metadata to the DataArray.
    da_temp.attrs["units"] = units
    da_temp.attrs["longname"] = longname
    da_temp.attrs["positive_dir"] = pos_dir
    da_temp.attrs["title"] = title
    da_temp.attrs["full_units"] = full_units

    # Convert the DataArray into a Dataset.
    ds_new = da_temp.to_dataset(name=name)

    return ds_new

def ds_add_prec_rain_snow(ds):
    """
    Compute total precipitation and add it to the dataset. Sums convective and large-scale precipitation (full-sky and clear-sky) components into a single rainfall variable.
    
    Args:
        ds (xarray.Dataset): Dataset containing precipitation components: PRECC, PRECL, PRECSC, PRECSL.
    
    Returns:
        ds (xarray.Dataset): Input dataset with an added ``rain`` variable representing total precipitation.

    Note:
        These are all rates that are being used/calculated.
    """
    # Validation check
    required_precip_vars = ['PRECC', 'PRECL', 'PRECSC', 'PRECSL']
    for var in required_precip_vars:
        if var not in ds:
            raise ValueError(f"Dataset must contain {var} for the calculations.")

    # Total precipitation = convective + large-scale precipitation
    ds['PRECT'] = ds['PRECC'] + ds['PRECL']

    # Total snowfall = convective + large-scale snowfall
    ds['snow'] = ds['PRECSC'] + ds['PRECSL']

    # Total rainfall = total precipitation - total snowfall
    ds['rain'] = ds['PRECT'] - ds['snow']

    # Copy the metadata attributes
    ds['PRECT'].attrs = ds['PRECC'].attrs.copy()
    ds['snow'].attrs = ds['PRECSC'].attrs.copy()
    ds['rain'].attrs = ds['PRECC'].attrs.copy()

    # Update the 'long_name' so the plot titles are accurate
    ds['PRECT'].attrs['long_name'] = 'Total Precipitation Rate'
    ds['snow'].attrs['long_name'] = 'Total Snowfall Rate (Water Equivalent)'
    ds['rain'].attrs['long_name'] = 'Total Rainfall Rate'

    return ds

def ds_add_alb(ds):
    """
    Add surface and top-of-atmosphere albedo variables to a dataset.

    Computes full-sky and clear-sky albedo at both the surface and the
    top of the atmosphere (TOA) using the ``albedo`` helper function, and
    stores the results as new variables in the dataset.

    Args:
        ds (xarray.Dataset): Dataset containing the radiation variables required for albedo calculations: ``FSDS``, ``FSNS``, ``FSDSC``, ``FSNSC``, ``SOLIN``, ``FSNT``, and ``FSNTC``.

    Returns:
        ds (xarray.Dataset): The input dataset with the following additional variables:

            * ``albedo_sfc_fullsky``: Surface albedo under all-sky conditions.
            * ``albedo_sfc_clearsky``: Surface albedo under clear-sky conditions.
            * ``albedo_toa_fullsky``: TOA albedo under all-sky conditions.
            * ``albedo_toa_clearsky``: TOA albedo under clear-sky conditions.
    """
    # Validation check
    vars = ['FSDS', 'FSNS', 'FSDSC', 'FSNSC', 'SOLIN', 'FSNT', 'FSNTC']
    for var in vars:
        if var not in ds:
            raise ValueError(f"Dataset must contain {var} for albedo calculations.")
        
    # Compute surface albedo using all-sky shortwave fluxes.
    ds['albedo_sfc_fullsky'] = albedo(ds['FSDS'], ds['FSNS'])

    # Compute surface albedo using clear-sky shortwave fluxes.
    ds['albedo_sfc_clearsky'] = albedo(ds['FSDSC'], ds['FSNSC'])

    # Compute TOA albedo using all-sky shortwave fluxes.
    ds['albedo_toa_fullsky'] = albedo(ds['SOLIN'], ds['FSNT'])

    # Compute TOA albedo using clear-sky shortwave fluxes.
    ds['albedo_toa_clearsky'] = albedo(ds['SOLIN'], ds['FSNTC'])

    return ds