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
from climate_lib.compute import *

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
        ds (xarray.Dataset): Input dataset with an added ``PRECT``, ``snow``, and ``rain`` variable representing total precipitation, snow, and rainfall rates.

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

def add_albedo_variables(ds, pairs):
    """Add one or more albedo variables to a dataset from downwelling/net SW pairs.
 
    Args:
        ds (xarray.Dataset): Input dataset containing the shortwave flux variables.
        pairs (dict): Mapping of ``{output_var_name: (swdn_var_name, swnet_var_name)}``, e.g. ``{"albedo_sfc_fullsky": ("FSDS", "FSNS")}``.
 
    Returns:
        ds (xarray.Dataset): Dataset with the requested albedo variables added.
    """
    for out_name, (swdn_var, swnet_var) in pairs.items():
        ds[out_name] = albedo(ds[swdn_var], ds[swnet_var])
    return ds
 
 
def sum_variables(ds, components, out_name):
    """Sum several variables in a dataset into a single new variable.
 
    Useful for things like combining convective/large-scale, liquid/solid
    precipitation components into one "total precipitation" variable.
 
    Args:
        ds (xarray.Dataset): Input dataset containing the component variables.
        components (list of str): Names of variables in ``ds`` to sum together.
        out_name (str): Name of the new summed variable to add to ``ds``.
 
    Returns:
        ds (xarray.Dataset): Dataset with the new summed variable added.
    """
    total = ds[components[0]]
    for var in components[1:]:
        total = total + ds[var]
    ds[out_name] = total
    return ds
 
 
def unstack_time_to_year_month(ds, variables=None, year_dim="year", month_dim="month"):
    """Reshape a monthly time-series dataset from a single 'time' dimension into (year, month).
 
    Builds a year/month MultiIndex from the dataset's decoded time
    coordinate and unstacks each requested variable individually before
    remerging, since ``unstack`` only works cleanly one DataArray at a time
    for this pattern.
 
    Args:
        ds (xarray.Dataset): Input dataset with a decoded 'time' dimension covering whole years of monthly data.
        variables (list of str or None): Variables to reshape; if None, reshapes every data variable in ``ds``.
        year_dim (str): Name to give the new year dimension (default: "year").
        month_dim (str): Name to give the new month dimension (default: "month").
 
    Returns:
        ds_out (xarray.Dataset): Dataset with 'time' replaced by separate (year, month) dimensions.
    """
    years = ds.groupby("time.year").mean("time").year
    months = ds.groupby("time.month").mean("time").month
 
    midx = pd.MultiIndex.from_product([years.values, months.values], names=(year_dim, month_dim))
 
    var_list = variables if variables is not None else list(ds.data_vars)
 
    da_list = []
    for var in var_list:
        da_temp = ds[var].copy()
        da_temp = da_temp.assign_coords({"time": midx})
        da_list.append(da_temp.unstack().to_dataset(name=var))
 
    ds_out = xr.merge(da_list)
    return ds_out
 
 
def annotate_dataarray(da, name, longname, title, pos_dir, units, full_units):
    """Attach standard metadata attributes to a DataArray and wrap it in a named Dataset.
 
    Args:
        da (xarray.DataArray): Field to annotate and export.
        name (str): Variable name to use in the output Dataset.
        longname (str): Human-readable long description of the variable.
        title (str): Short plot-friendly title for the variable.
        pos_dir (str): Sign convention description, e.g. "up" or "down".
        units (str): Units string, e.g. "W/m2".
        full_units (str): Fuller description of the quantity and its units, e.g. "Delta TOA SW".
 
    Returns:
        ds_new (xarray.Dataset): Single-variable Dataset named ``name`` with the given attributes attached.
    """
    da = da.copy()
    da.attrs["units"] = units
    da.attrs["longname"] = longname
    da.attrs["positive_dir"] = pos_dir
    da.attrs["title"] = title
    da.attrs["full_units"] = full_units
    return da.to_dataset(name=name)
 
 
def write_netcdf(ds, filename, overwrite=True, make_dirs=True):
    """Write a Dataset to a netCDF file, optionally overwriting and creating parent directories.
 
    Args:
        ds (xarray.Dataset): Dataset to write out.
        filename (str): Destination file path.
        overwrite (bool): If True, delete any existing file at ``filename`` before writing (default: True).
        make_dirs (bool): If True, create the parent directory of ``filename`` if it doesn't exist (default: True).
 
    Returns:
        filename (str): The path the dataset was written to.
    """
    if make_dirs:
        parent = os.path.dirname(filename)
        if parent:
            os.makedirs(parent, exist_ok=True)
    if overwrite and os.path.exists(filename):
        os.remove(filename)
    ds.to_netcdf(path=filename, mode="w")
    return filename