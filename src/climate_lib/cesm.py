"""
This module contains functions concerning CESM datasets.
"""

import sys

import numpy as np
import netCDF4 as nc
import numpy.matlib
import datetime
import xarray as xr
from scipy import interpolate
from numpy import ma
from scipy import stats
import scipy.io as sio
import pickle as pickle
from sklearn import linear_model
import numpy.ma as ma

import scipy as sp
import pandas as pd

import time
import cftime

from copy import copy 

import matplotlib as mpl
import matplotlib.pyplot as plt

import os
import sys
import cftime

import glob

import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

import dask
import dask.bag as db

from scipy import interpolate

from climate_lib.utils import *
from climate_lib.constants import *


def modify_cesm(ds_old, var_list, zeros = None, global_mean = None, simple_zonal_mean = None, mirror_zonal_mean = None):
    """
    Create copy of an input file to calculate the idealized equivalent with several methods by erasing the geographical signature
    of a variable, using global mean and/or simple zonal mean and/or mirror_zonal_mean and/or zeros values.

    Args:
        ds_old (xarray): original dataset that will be copied to make changes
        var_list (list of str): list of variables that need to be changed
        
        zeros (bool): allows the function to create a copy of the dataset with 0 values at all longitudes and latitudes if True
        global_mean (bool): allows the function to create a copy with a global mean if True
        simple_zonal_mean (bool): allows the function to create a copy with values averaged by longitude if True
        mirror_zonal_mean (bool): allows the function to create a copy with values averaged by longitude AND mirrored by the equator if True

    Returns:
        ds_zeros (xarray): new dataset with 0 values at all longitudes and latitudes
        ds_mean_new (xarray): new dataset with global mean
        ds_zonal_new (xarray): new dataset with zonal mean
        ds_mirror_zonal_new (xarray): new dataset with mirrored zonal mean
    """   
    # Creation of new Datasets
    ds_mirror_zonal_new = ds_old.copy()
    ds_zeros = ds_old.copy()
    ds_mean_new = ds_old.copy()
    ds_zonal_new = ds_old.copy()
    
    # Do global mean area weighted average
    weights = np.cos(np.deg2rad(ds_old.lat))
    weights.name = "weights"

    if zeros :
        for var in var_list :
            # Set all the values to zeros
            ds_zeros[var] = ds_old[var]*0.0

    
    if global_mean :        
        weighted_mean = {}
        for var in var_list :
            da_weighted = ds_old[var].weighted(weights)
            weighted_mean[var] = da_weighted.mean(("lon", "lat"))
    
            # Set the calculated mean to the new dataset
            ds_mean_new[var] = ds_old[var]*0.0 + weighted_mean[var]#.values

    
    if simple_zonal_mean :        
        zonal_mean = {}
        for var in var_list :
            # Calcul of global mean over longitude and latitude
            zonal_mean[var] = ds_old[var].mean('lon')
                
            # Set the calculated zonal mean to the new dataset
            ds_zonal_new[var] = ds_old[var]*0.0 + zonal_mean[var]#.values

    
    if mirror_zonal_mean :

        zonal_mean = {}
        for var in var_list :
            # Calcul of global mean over longitude and latitude
            zonal_mean[var] = ds_old[var].mean('lon')
        
        mirror_zonal_mean = {}
        for var in var_list :
            mirror_zonal_mean[var] = (zonal_mean[var] + zonal_mean[var].isel(lat=slice(None, None, -1)).values) / 2

            # Set the calculated zonal mean to the new dataset
            ds_mirror_zonal_new[var] = ds_old[var]*0.0 + mirror_zonal_mean[var]#.values


    return ds_zeros, ds_mean_new, ds_zonal_new, ds_mirror_zonal_new

 
def align_and_filter_time(ds, day_shift=15, year_range=None, time_var="time"):
    """Shift CESM monthly-average timestamps and optionally drop years outside a range.
 
    CESM monthly time coordinates are stamped at the end of the averaging
    period, so a small backward shift (commonly 15 days) is needed to line
    the timestamp up with the month the data actually represents. Decoded
    cftime objects are produced from the raw numeric time values so this
    should be called on a dataset opened with ``decode_times=False``.
 
    Args:
        ds (xarray.Dataset): Input dataset with a raw (non-decoded) numeric time coordinate.
        day_shift (float): Number of days to subtract from the raw time values before decoding (default: 15).
        year_range (range or None): If given, keep only timesteps whose decoded year falls in this range (optional).
        time_var (str): Name of the time coordinate/variable (default: "time").
 
    Returns:
        ds (xarray.Dataset): Dataset with decoded, shifted time coordinates, optionally filtered to ``year_range``.
    """
    time_new = cftime.num2date(
        ds[time_var].values[:] - day_shift,
        units=ds[time_var].units,
        calendar=ds[time_var].calendar,
        only_use_cftime_datetimes=True,
    )
    ds = ds.assign_coords({time_var: time_new})
    if year_range is not None:
        ds = ds.where(ds[time_var].dt.year.isin(year_range), drop=True)
    return ds
 
 
def build_cesm_filelist(archive_path, run_name, variables, component="atm",
                         stream="cam.h0", proc_subdir="atm/proc/tseries/month_1"):
    """Glob together per-variable CESM time-series output files for one run.
 
    CESM/CAM post-processed time series are typically stored one file per
    variable, named like ``<run_name>.<stream>.<VAR>.<dates>.nc``. This
    walks a list of variable names and collects all matching files, sorted,
    into a single flat list suitable for ``xarray.open_mfdataset``.
 
    Args:
        archive_path (str): Root archive directory containing the run's output.
        run_name (str): CESM case/run name, used both as a subdirectory and filename prefix.
        variables (list of str): Variable names to search for.
        component (str): Model component subdirectory, e.g. "atm" or "lnd" (default: "atm"); informational, use ``proc_subdir`` for the actual path.
        stream (str): Output stream identifier used in filenames, e.g. "cam.h0" or "clm2.h0" (default: "cam.h0").
        proc_subdir (str): Path (relative to ``archive_path/run_name/``) to the post-processed time-series directory (default: "atm/proc/tseries/month_1").
 
    Returns:
        filelist (list of str): Sorted list of file paths matching all requested variables.
    """
    folder = os.path.join(archive_path, run_name, proc_subdir)
    filelist = []
    for var in variables:
        pattern = os.path.join(folder, f"{run_name}.{stream}.{var}.*.nc")
        files = sorted(glob.glob(pattern))
        filelist.extend(files)
    return filelist