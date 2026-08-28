"""
This module contains helper functions for climateLib, can also be imported/used for other purposes.
"""

# import modules
from climate_lib.compute import humidsat
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

from climate_lib.constants import *

import warnings
warnings.filterwarnings("ignore", message=".*multiple fill values.*")
warnings.filterwarnings("ignore", message="Interpolation point out of data bounds encountered")

def vertically_interpolate(varname, ds_ctrl, pnew, PS, hyam_ctrl, hybm_ctrl, P0pa_ctrl, interpolate_method = 'linear'):
    """
    Vertically interpolate a variable from hybrid sigma-pressure coordinates to specified pressure levels.

    Args:
        varname (str): Name of the variable to interpolate (e.g., 'T', 'U', 'V').
        ds_ctrl (xarray.Dataset): Original dataset containing the variable in hybrid coordinates.
        pnew (array-like): New pressure levels to interpolate to (in hPa).
        PS (xarray.DataArray): Surface pressure variable from the dataset.
        hyam_ctrl (xarray.DataArray): Hybrid A coefficients for the control dataset.
        hybm_ctrl (xarray.DataArray): Hybrid B coefficients for the control dataset.
        P0pa_ctrl (xarray.DataArray): Reference pressure level (usually 100000 Pa) for the control dataset.
        
    Returns:
        var_ctrl_out (xarray.DataArray): The variable interpolated to the new pressure levels, with dimensions (time, lev, lat, lon).
    """
    var_ctrl = ds_ctrl[varname]  # (time:30, lev:30, lat:192, lon:288)
    # Vertical interpolation
    var_ctrl_tmp = np.ones((var_ctrl.shape[0], len(pnew), var_ctrl.shape[2], var_ctrl.shape[3]))*float('nan')
    for i in range(ds_ctrl['time'].size):
        # try with geocat instead of ngl
       var_ctrl_tmp[i,:,:,:] = geocat.comp.interpolation.interp_hybrid_to_pressure(var_ctrl[i], 
                                    PS[i], hyam_ctrl[i], hybm_ctrl[i], p0=P0pa_ctrl[i],
                                    new_levels=pnew, 
                                    lev_dim='lev', 
                                    method=interpolate_method, extrapolate=False, variable=None, t_bot=None, phi_sfc=None)
    
    var_ctrl_out = xr.DataArray(data=var_ctrl_tmp, 
                          dims=['time','lev','lat','lon'], 
                          coords={'time':var_ctrl.time, 'lev':pnew, 'lat':var_ctrl.lat, 'lon':var_ctrl.lon}, 
                          attrs={'long_name': var_ctrl.long_name, 'units': var_ctrl.units})

    return var_ctrl_out

def albedo(swdn, swnet):
    """
    Compute surface albedo from downward and net shortwave radiation.

    Args:
        swdn (xarray.DataArray or numpy.ndarray): Downward shortwave radiation at the surface.
        swnet (xarray.DataArray or numpy.ndarray): Net shortwave radiation at the surface.

    Returns:
        alpha(xarray.DataArray or numpy.ndarray): Surface albedo (same type and shape as input arrays).
    """
    swup = swdn - swnet
    alpha = swup / swdn
    return alpha

def mask_above_tropopause(da, p, p_tropopause):
    """Zero out (mask) values above a given tropopause pressure surface.
 
    Sets values to NaN wherever the pressure ``p`` is above (i.e. less than)
    the local tropopause pressure, so stratospheric temperature/moisture
    changes don't contaminate a tropospheric kernel convolution.
 
    Args:
        da (xarray.DataArray): Field to mask (e.g. a temperature or moisture change), with a 'lev' dimension.
        p (xarray.DataArray): Pressure field broadcastable against ``da``, in the same units as ``p_tropopause``.
        p_tropopause (xarray.DataArray): Tropopause pressure field (e.g. from ``kernels.estimate_diagnostic_tropopause``, broadcast to match ``p``).
 
    Returns:
        da_masked (xarray.DataArray): ``da`` with values above the tropopause set to NaN.
    """
    return da.where(p >= p_tropopause)
 
 
def logq_kernel_normalization(t0, dt, p):
    """Compute d(log qsat)/dT, used to normalize a moisture kernel onto a log-q basis.
 
    Normalizing the water-vapour radiative kernel by the change in
    saturation specific humidity expected for a 1 K warming at constant
    relative humidity ("log-q kernel") makes the moisture feedback
    decomposition insensitive to the model's climatological humidity field.
 
    Args:
        t0 (xarray.DataArray): Baseline (control) temperature in Kelvin.
        dt (xarray.DataArray): Temperature change (case minus control) in Kelvin.
        p (xarray.DataArray): Pressure in hPa, broadcastable against ``t0``.
 
    Returns:
        dlogqdt (xarray.DataArray): d(log qsat)/dT, the moisture kernel normalization factor.
    """
    _, qsat1, _ = humidsat(t0, p)
    _, qsat2, _ = humidsat(t0 + dt, p)
    dlogqdt = (np.log(qsat2) - np.log(qsat1)) / dt
    return dlogqdt