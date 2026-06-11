import sys

# netcdf/numpy/xarray
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

from copy import copy 

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt

# OS interaction
import os
import sys
import cftime

import glob

import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

import glob
import dask
import dask.bag as db

from scipy import interpolate


def modify_cesm(ds_old, var_list, zeros = None, global_mean = None, simple_zonal_mean = None, mirror_zonal_mean = None):
    '''
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
    '''    
    # Creation of new Datasets
    ds_mirror_zonal_new = ds_old.copy()
    ds_zeros = ds_old.copy()
    ds_mean_new = ds_old.copy()
    ds_zonal_new = ds_old.copy()
    
    # do global mean area weighted average
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