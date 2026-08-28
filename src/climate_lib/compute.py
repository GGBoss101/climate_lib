"""
This module contains functions for computing various atmospheric diagnostics.
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

#===============================
# Vertical integration functions
#===============================

def vert_integral_native(da, var = None, g = g_earth, pressure_factor = 100.0):
    """Vertically integrate a Dataset using Xarray's native calculus.

    Args:
        da (xarray.Dataset): The input atmospheric data.
        var (str): The specific variable name to extract from ``da`` (optional).
        g (float): Gravity acceleration constant in m/s^2 (default: 9.80665).
        pressure_factor (float): Multiplier to convert pressure units to Pascals (default: 100.0 for hPa -> Pa).

    Returns:
        integral (xarray.Dataset): Pressure-coordinate vertical integral with dimensions (time, lat, lon).
    """
    if var is not None:
        if var not in da:
            raise ValueError(f"Variable '{var}' not found in the dataset.")
        da = da[var]

    if "lev" not in da.dims:
        raise ValueError(
            "Variable must have 'lev' (level) dimension for vertical"
            " integration."
        )

    # Scale the vertical coordinate values to Pascals (e.g., hPa -> Pa)
    da_scaled = da.assign_coords(lev = da.lev * pressure_factor)

    # Perform the mathematical integration along the vertical axis
    raw_integral = da_scaled.integrate(dim="lev")

    # Divide the column by gravity to get mass-weighted units
    integral = raw_integral / g

    return integral


def vert_integral(ds, var = None, g = g_earth, pressure_factor = 100.0):
    """Vertically integrate a variable in pressure coordinates.

    Args:
        ds (xarray.Dataset): Input dataset containing the variable and pressure interface coordinates.
        var (str): Variable name to vertically integrate.
        g (float): Gravitational acceleration (m s^-2).
        pressure_factor (float): Factor used to convert pressure units to Pascals (e.g., 100.0 for hPa → Pa).

    Returns:
        integral (xarray.DataArray): Vertically integrated field.
    """
    if var is None:
        raise ValueError("A variable name must be provided.")

    if var not in ds:
        raise ValueError(f"Variable '{var}' not found in dataset.")

    da = ds[var]

    if "lev" not in da.dims:
        raise ValueError(f"Variable '{var}' must contain a 'lev' dimension.")

    if "ilev" not in ds.coords:
        raise ValueError("Dataset must contain 'ilev' interface coordinates.")

    # Compute layer pressure thicknesses cleanly without stale metadata
    dp = ds["ilev"].diff(dim="ilev").drop_vars("ilev").rename({"ilev": "lev"})
    dp = dp.assign_coords({"lev": da.lev}) * pressure_factor

    # Handle vertical grid orientation safely
    dp = abs(dp)

    # Compute mass-weighted vertical integral.
    integral = (da * dp).sum(dim="lev") / g

    return integral


def vert_int_hybrid(ds, var, ds_hybrid, g = g_earth, P0 = P0pa):
    """Computes the vertical integration of a variable on hybrid coordinates.

    This function calculates the mass-weighted column integration of an atmospheric variable (such as specific humidity to get total column water vapor) using hybrid sigma-pressure coordinate coefficients.

    Args:
        ds (xarray.Dataset): Input dataset containing the target variable.
        var (str): Name of the variable inside `ds` to vertically integrate.
        ds_hybrid (xarray.Dataset): Dataset containing the surface pressure ('PS') and the hybrid interface coefficients ('hyai', 'hybi').
        g (float): Gravitational acceleration in m/s^2 (default: 9.80665).
        P0 (float): Reference pressure in Pascals for the hybrid coordinate system (default: 100000.0 Pa).

    Returns:
        total_int (xarray.DataArray): The vertically integrated field with the vertical dimension ('lev') integrated out absolutely.
    """

    # Validate inputs
    if var is None:
        raise ValueError("A variable name must be provided for vertical integration.")
    if var not in ds:
        raise ValueError(f"Variable '{var}' not found in the input dataset.")

    # Get data and coefficients
    q = ds[var]
    PS = ds_hybrid["PS"]
    hyai = ds_hybrid["hyai"]
    hybi = ds_hybrid["hybi"]

    # Interface pressures: P = A*P0 + B*PS
    P_i = hyai * P0 + hybi * PS

    # Layer thickness: dp = P(k+1) - P(k)
    dp = P_i.diff(dim="ilev")

    # Match coordinates to layer centers
    dp = dp.drop_vars("ilev").rename({"ilev": "lev"})
    dp = dp.assign_coords({"lev": q.coords["lev"]})

    # Ensure positive thickness
    dp = abs(dp)

    # Hydrostatic column mass weight
    layer_int = (q * dp) / g

    # Sum over vertical dimension
    total_int = layer_int.sum(dim="lev", skipna=True)

    return total_int




#=====================
# Divergence functions
#=====================

def div_uqvq_manual(ds, g = g_earth, R = R_earth):
    """Compute vertically integrated moisture flux divergence.

    Args:
        ds (xarray.Dataset): Dataset containing U, V, Q, PS, hyai, hybi, and optionally P0.
        g (float): Gravitational acceleration (m s^-2).
        R (float): Planetary radius (m).

    Returns:
        xarray.DataArray: Vertically integrated moisture divergence (kg m^-2 s^-1).
    """

    # Validate required inputs
    required_vars = ["PS", "U", "V", "Q", "hyai", "hybi"]
    for var in required_vars:
        if var not in ds:
            raise ValueError(
                f"Dataset must contain variable '{var}'."
            )

    if "lev" not in ds.coords:
        raise ValueError(
            "Dataset must contain 'lev' coordinates."
        )

    # Extract variables
    U = ds["U"]
    V = ds["V"]
    Q = ds["Q"]
    PS = ds["PS"]

    hyai = ds["hyai"]
    hybi = ds["hybi"]

    P0 = ds.get("P0", P0pa)

    # Interface pressures (Pa)
    p_int = hyai * P0 + hybi * PS

    # Layer pressure thicknesses
    dp = (
        p_int.diff(dim="ilev")
        .drop_vars("ilev")
        .rename({"ilev": "lev"})
        .assign_coords({"lev": ds.lev})
    )

    # Protect against reversed level ordering
    dp = abs(dp)

    # Vertically integrated moisture fluxes
    uq_int = ((U * Q) * dp).sum(dim="lev") / g
    vq_int = ((V * Q) * dp).sum(dim="lev") / g

    # Spherical geometry factors
    lat_rad = np.deg2rad(ds.lat)
    coslat = np.cos(lat_rad)

    deg_per_rad = 180.0 / np.pi

    # Partial flux derivatives in spherical coordinates
    du_dx = (
        (uq_int * coslat)
        .differentiate("lon")
        * deg_per_rad
        / (R * coslat)
    )

    dv_dy = (
        vq_int
        .differentiate("lat")
        * deg_per_rad
        / R
    )

    # Moisture flux divergence
    qdiv = du_dx + dv_dy

    qdiv.name = "QDIV"
    qdiv.attrs = {
        "long_name": "Vertically integrated water vapor divergence",
        "units": "kg m-2 s-1",
    }

    return qdiv


def dominguez_uqdiv(ds,
    dPmb = 25,
    g = g_earth,
    R = R_earth,
    pressure_factor = 100.0,
    interpolate_method = 'linear'):
    """
    Compute vertically integrated moisture flux divergence
    using pressure-level interpolation.

    Args:
        ds (xarray.Dataset): CESM atmospheric dataset
        dPmb (float): Pressure spacing in hPa (default 25)
        g (float): Gravity (m/s²) (default 9.81)
        R (float): Planet radius (m) (default 6.37122e6 = Earth radius)
        pressure_factor (float): Factor to convert pressure levels to Pascals (default 100.0)
        interpolate_method (str): Method for vertical interpolation ('linear' or 'log', default 'linear')

    Returns:
        ctrl_q_div (dict(str: xarray.Dataset)): Moisture flux divergence (mm/day) and its components (kg m-1 s-1) in a Dataset.
            - 'VIMF_x': Vertically integrated moisture flux x-component (kg m-1 s-1)
            - 'VIMF_y': Vertically integrated moisture flux y-component (kg m-1 s-1)
            - 'q_div': Vertically integrated moisture flux divergence (mm/day)
    """
    # Validation checks
    required_vars = ['U', 'V', 'Q', 'PS', 'hyam', 'hybm', 'P0']
    for var in required_vars:
        if var not in ds:
            raise ValueError(f"Dataset must contain variable '{var}' for this calculation.")
    ds_ctrl = ds

    # Calculate moisture flux divergence (qv vi -> q_div -> diff -> tavg)
    U_ctrl = ds_ctrl.U
    V_ctrl = ds_ctrl.V
    Q_ctrl = ds_ctrl.Q

    PS = ds.PS # (time, lat, lon)

    # Constants
    lat = ds_ctrl.lat
    lon = ds_ctrl.lon

    r = R * np.cos(lat * np.pi / 180.)

    # Interpolate to pressure coordinates
    dPmb = 25
    pnew = (np.arange(100, 1000 + dPmb, dPmb))

    # Convert units to Pascals
    dP = dPmb * pressure_factor
    pnew = pnew * pressure_factor

    # Extract the desired variables
    hyam_ctrl = ds_ctrl.hyam  # (time:30, lev:30)
    hybm_ctrl = ds_ctrl.hybm  # (time:30, lev:30)
    psrf_ctrl = ds_ctrl.PS  # (time:30, lat:192, lon:288)
    P0pa_ctrl = ds_ctrl.P0  # (time:30)

    U_ctrl = vertically_interpolate('U', ds_ctrl, pnew, PS, hyam_ctrl, hybm_ctrl, P0pa_ctrl, interpolate_method)
    V_ctrl = vertically_interpolate('V', ds_ctrl, pnew, PS, hyam_ctrl, hybm_ctrl, P0pa_ctrl, interpolate_method)
    Q_ctrl = vertically_interpolate('Q', ds_ctrl, pnew, PS, hyam_ctrl, hybm_ctrl, P0pa_ctrl, interpolate_method)
    
    # Vertically integrated qv
    VIMF_ctrl_x = np.nansum(U_ctrl * Q_ctrl * dP, axis = 1) / g
    VIMF_ctrl_y = np.nansum(V_ctrl * Q_ctrl * dP, axis = 1) / g

    # get the cosine stuff in
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    cosphi = np.cos(lat_rad.values)

    Z_ctrl_tmp = VIMF_ctrl_x*float('nan')
    
    for i in range(VIMF_ctrl_x.shape[0]):
        U_ctrl_tmp = VIMF_ctrl_x[i, :, :]
        V_ctrl_tmp = VIMF_ctrl_y[i, :, :]

        # ∂(Fx cosφ)/∂λ
        term1 = np.gradient(U_ctrl_tmp * cosphi[:,None], lon_rad, axis=1) / (R * cosphi[:,None])
    
        # ∂Fy/∂φ
        term2 = np.gradient(V_ctrl_tmp, lat_rad, axis=0) / R 

        Z_ctrl_tmp[i, :, :] = (term1 + term2) * 86400.

    varname = 'Q'

    var_ctrl = ds_ctrl[varname]  # (time:30, lev:30, lat:192, lon:288)

    VIMF_ctrl_x_da = xr.Dataset(data=VIMF_ctrl_x, 
                                dims=['time','lat','lon'], 
                                coords={'time':var_ctrl['time'], 'lat':var_ctrl['lat'], 'lon':var_ctrl['lon']}, 
                                attrs={'long_name': 'Vertically integrated moisture flux x-component', 'units': 'kg m-1 s-1'})
    
    VIMF_ctrl_y_da = xr.Dataset(data=VIMF_ctrl_y, 
                                dims=['time','lat','lon'], 
                                coords={'time':var_ctrl['time'], 'lat':var_ctrl['lat'], 'lon':var_ctrl['lon']}, 
                                attrs={'long_name': 'Vertically integrated moisture flux y-component', 'units': 'kg m-1 s-1'})

    q_div_ctrl_da = xr.Dataset(data=Z_ctrl_tmp, 
                                dims=['time','lat','lon'], 
                                coords={'time':var_ctrl['time'], 'lat':var_ctrl['lat'], 'lon':var_ctrl['lon']}, 
                                attrs={'long_name': 'Vertically integrated moisture flux divergence', 'units': 'mm/day'})

    ctrl_q_div = xr.Dataset({'VIMF_x': VIMF_ctrl_x_da, 'VIMF_y': VIMF_ctrl_y_da, 'q_div': q_div_ctrl_da})

    return ctrl_q_div

#============================
# Linear regression functions
#============================

def linear_trend(da):
    """
    Compute linear regression statistics along the ``time`` dimension
    of an Xarray Dataset.

    Args:
        da (xarray.Dataset): Input data with a ``time`` dimension.

    Returns:
        tuple:
            - slope (xarray.Dataset): Linear trend slope.
            - intercept (xarray.Dataset): Regression intercept.
            - rvalue (xarray.Dataset): Correlation coefficient.
            - pvalue (xarray.Dataset): Statistical p-value.
            - stderr (xarray.Dataset): Standard error of the slope.
    """

    # Validation checks
    if "time" not in da.dims:
        raise ValueError("Input Dataset must have a 'time' dimension for linear regression.")

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

#=====================
# General Calculations
#=====================

def multi_apply_along_axis(func1d, axis, arrs, *args, **kwargs):
    """
    Apply a function to multiple arrays along a given axis.

    Extends ``numpy.apply_along_axis`` to support functions that operate on multiple 1D arrays simultaneously.

    Args:
        func1d (callable): Function that accepts one 1D slice from each array in ``arrs``.
        axis (int): Axis along which to apply the function.
        arrs (sequence of numpy.ndarray): Input arrays with compatible shapes for concatenation along ``axis``.
        *args: Additional positional arguments passed to ``func1d``.
        **kwargs: Additional keyword arguments passed to ``func1d``.

    Returns:
        numpy.ndarray: Result of applying ``func1d`` along the specified axis of the input arrays.
    """


    # Combine input arrays so they can be processed by apply_along_axis.
    carrs = np.concatenate(arrs, axis)

    # Record split locations needed to recover the original arrays.
    offsets = []
    start = 0
    for i in range(len(arrs) - 1):
        start += arrs[i].shape[axis]
        offsets.append(start)

    # Split each slice back into the original arrays before applying func1d.
    def helperfunc(a, *args, **kwargs):
        arrs = np.split(a, offsets)
        return func1d(*[*arrs, *args], **kwargs)

    # Apply the function along the requested axis.
    return np.apply_along_axis(helperfunc, axis, carrs, *args, **kwargs)

def means(ds_case, mask, var):
    """
    Calculates the global, ocean and land mean of a given variable for a given configuration.

    Args:
        ds_case (xarray): dictionary containing lat and lon dimensions.
        mask (list): representation of continental distribution, with value 1 if land and value 0 if ocean.
        var (str): name of the variable that needs to be averaged.

    Returns:
        gm (float OR array): global mean (can be an array if ds_case contains others dimensions).
        lm (float OR array): land mean (can be an array if ds_case contains others dimensions).
        om (float OR array): ocean mean (can be an array if ds_case contains others dimensions).
    """

    # Collects the datas to plot the surface temperature
    mapdata = (ds_case[var].mean('year'))

    # Weights the latitude, renames it to "weights", then prints the value
    weights = np.cos(np.deg2rad(ds_case[var].lat))
    weights.name = "weights"

    # Finds the weighted global mean based on latitude and longitude
    gm = mapdata.weighted(weights).mean(('lat', 'lon')).values

    # Defines a land timeseries where the land mask is present
    land = mapdata.where(mask == 1)

    # Defines an ocean timeseries where the land mask is not present (i.e. just ocean/water)
    ocean = mapdata.where(mask != 1)

    # Mean Land
    lm = land.weighted(weights).mean(('lat', 'lon')).values

    # Mean Ocean
    om = ocean.weighted(weights).mean(('lat', 'lon')).values

    return gm, lm, om

def inferred_heat_transport( energy_in, lat_deg ):
    '''
    Returns the inferred heat transport (in PW) by integrating the net energy imbalance from pole to pole.

    Args:
        energy_in (xarray): net radiative flux that enters the atmosphere (longitude-averaged)
        lat_deg (xarray): latitude values of the dictionary

    Returns:
        heat_transport (xarray): total northward heat transport (longitude-averaged)
    '''
    # Convert latitude from degrees to radians
    lat_rad = np.deg2rad(lat_deg)

    # Radius of Earth in metres
    earth_radius = 6.371e6

    # Latitude weighting based on the area of each latitude band
    latitude_weights = np.cos(lat_rad)

    # Calculate the global, area-weighted mean energy imbalance
    weighted_energy = energy_in * latitude_weights
    global_mean_energy = weighted_energy.sum() / latitude_weights.sum()

    # Remove the global mean so that the total energy imbalance
    # integrates to zero from pole to pole
    corrected_energy = energy_in - global_mean_energy

    # Weight the corrected energy imbalance by latitude
    weighted_corrected_energy = latitude_weights * corrected_energy

    # Integrate the energy imbalance from the South Pole northward
    cumulative_energy = integrate.cumtrapz(
        weighted_corrected_energy,
        x=lat_rad,
        initial=0.
    )

    # Convert the integrated energy into northward heat transport
    # and convert from watts to petawatts
    heat_transport = (
        1E-15
        * 2
        * np.pi
        * earth_radius**2
        * cumulative_energy
    )

    return heat_transport

def humidsat(t, p):
    """Compute saturation humidity quantities from temperature and pressure.

    Calculates saturation vapor pressure, saturation specific humidity,
    and saturation mixing ratio using the formulations of Buck (1981).
    Saturation vapor pressure is computed over liquid water for
    temperatures above 0°C, over ice for temperatures below -23°C,
    and interpolated between these limits.

    Args:
        t (xarray.Dataset): Air temperature in Kelvin.
        p (xarray.Dataset): Air pressure in hPa.

    Returns:
        esat (xarray.Dataset): Saturation vapor pressure (hPa).
        qsat (xarray.Dataset): Saturation specific humidity (kg kg^-1).
        rsat (xarray.Dataset): Saturation mixing ratio (kg kg^-1).
    """
    # Convert temperature from Kelvin to Celsius.
    tc = t - 273.16

    # Temperature thresholds (°C) for ice, water, and interpolation.
    tice = -23
    t0 = 0

    # Gas constants (J kg^-1 K^-1).
    Rd = 287.04
    Rv = 461.5
    epsilon = Rd / Rv

    # Saturation vapor pressure over liquid water (Buck, 1981).
    ewat = (
        (1.0007 + (3.46e-6 * p))
        * 6.1121
        * np.exp(17.502 * tc / (240.97 + tc))
    )

    # Saturation vapor pressure over ice (Buck, 1981).
    eice = (
        (1.0003 + (4.18e-6 * p))
        * 6.1115
        * np.exp(22.452 * tc / (272.55 + tc))
    )

    # Quadratic interpolation between ice and water formulations.
    eint = eice + (ewat - eice) * ((tc - tice) / (t0 - tice))**2

    # Use ice values below -23°C, water values above 0°C, and interpolated values in between.
    esat = eice.where(tc < tice, eint)
    esat = ewat.where(tc > t0, esat)

    # Convert saturation vapor pressure to mixing ratio and specific humidity.
    rsat = epsilon * esat / (p - esat)
    qsat = epsilon * esat / (p - esat * (1 - epsilon))

    return esat, qsat, rsat

def meridional_streamfunction(ds, lat_name = 'lat', 
                              lev_name = 'lev', 
                              v_name = 'V', 
                              a = R_earth, 
                              g = g_earth, 
                              pressure_factor = 100.0):
    '''
    Calculates the meridional stream function from a standard CESM Dataset.

    Args:
        ds (xarray.Dataset): dictionary containing lon, lat and lev dimensions (time is optional) and meridional wind variable.
        lat_name (str): name of the latitude dimension of the Dataset ('lat' by default).
        lev_name (str): name of the pressure level dimension of the Dataset ('lev' by default).
        v_name (str): name of the meridional wind variable ('V' by default).
        a (float): radius of the Earth in meters (default: 6.371e6).
        g (float): acceleration due to gravity in m/s^2 (default: 9.80665).
        pressure_factor (float): factor to convert pressure levels to Pascals (default: 100.0 for hPa → Pa).

    Return:
        psi (xarray.DataArray): containing 'lev', 'lat' and 'time' (depends on ds) dimension and values of the meridional stream function.
    '''

    # Zonal mean for V
    V_zonal = ds[v_name].mean('lon')  # (year, lev, lat)

    # Verification if pressure levels are ordered
    lev = ds[lev_name]
    lev_Pa = lev * pressure_factor  # hPa → Pa

    # dp: pressure thickness of each layer
    # New dict with dp values (same dimensions than lev)
    lev_vals = lev_Pa.values

    # create a table with the same shape as lev_vals
    dp_vals = np.zeros_like(lev_vals)

    # calculate deltas
    dp_vals[1:-1] = (lev_vals[:-2] - lev_vals[2:]) / 2.0  # center-diff
    dp_vals[0] = lev_vals[0] - lev_vals[1] # top boundary
    dp_vals[-1] = lev_vals[-2] - lev_vals[-1] # surface boundary

    # Build DataArray with same lev coords as V_zonal after reordering
    dp = xr.DataArray(dp_vals, coords={lev_name: lev.values}, dims=[lev_name])

    # Cosinus of latitude
    cos_lat = np.cos(np.deg2rad(ds[lat_name]))

    # Integration
    integrand = V_zonal * dp
    psi = integrand.cumsum(dim=lev_name) * (2 * np.pi * a / g) * cos_lat

    return psi  # (year, lev, lat)


def albedo(swdn, swnet):
    """Compute albedo from downwelling and net shortwave radiation.
 
    Args:
        swdn (xarray.DataArray): Downwelling shortwave radiation.
        swnet (xarray.DataArray): Net (down minus up) shortwave radiation.
 
    Returns:
        alpha (xarray.DataArray): Albedo, computed as upwelling/downwelling shortwave.
    """
    swup = swdn - swnet
    alpha = swup / swdn
    return alpha
 
 
def multi_apply_along_axis(func1d, axis, arrs, *args, **kwargs):
    """Apply a function that takes multiple 1-D arrays along a shared axis of N-D arrays.
 
    Given a function ``func1d(A, B, C, ..., *args, **kwargs)`` that acts on
    multiple one-dimensional arrays, apply that function to the N-dimensional
    arrays listed in ``arrs`` along axis ``axis``.
 
    If the arrays in ``arrs`` are one-dimensional this is equivalent to
    ``func1d(*arrs, *args, **kwargs)``. If there is only one array in
    ``arrs`` this is equivalent to
    ``numpy.apply_along_axis(func1d, axis, arrs[0], *args, **kwargs)``.
    All arrays in ``arrs`` must have compatible dimensions to be able to run
    ``numpy.concatenate(arrs, axis)``.
 
    Source: https://climate-cms.org/2019/07/29/multi-apply-along-axis.html
 
    Args:
        func1d (callable): Function operating on ``len(arrs)`` 1-D arrays, with signature ``f(*arrs, *args, **kwargs)``.
        axis (int): Axis of all arrays in ``arrs`` to apply the function along.
        arrs (iterable of numpy.ndarray): Arrays to operate on, concatenated internally along ``axis``.
        *args: Passed to ``func1d`` after the array arguments.
        **kwargs: Passed to ``func1d`` as keyword arguments.
 
    Returns:
        result (numpy.ndarray): Output of applying ``func1d`` along ``axis`` across all input arrays.
    """
    carrs = np.concatenate(arrs, axis)
 
    offsets = []
    start = 0
    for i in range(len(arrs) - 1):
        start += arrs[i].shape[axis]
        offsets.append(start)
 
    def helperfunc(a, *args, **kwargs):
        split_arrs = np.split(a, offsets)
        return func1d(*[*split_arrs, *args], **kwargs)
 
    return np.apply_along_axis(helperfunc, axis, carrs, *args, **kwargs)
 
 
def humidsat(t, p):
    """Compute saturation vapor pressure, specific humidity, and mixing ratio.
 
    Uses the modified Tetens-like formulae of Buck (1981, J. Appl. Meteorol.)
    for vapor pressure over liquid water at temperatures above 0 C, over ice
    at temperatures below -23 C, and a quadratic polynomial interpolation for
    intermediate temperatures.
 
    Args:
        t (xarray.DataArray or numpy.ndarray): Temperature in Kelvin.
        p (xarray.DataArray or numpy.ndarray): Pressure in hPa.
 
    Returns:
        esat (xarray.DataArray or numpy.ndarray): Saturation vapor pressure (hPa).
        qsat (xarray.DataArray or numpy.ndarray): Saturation specific humidity (kg/kg).
        rsat (xarray.DataArray or numpy.ndarray): Saturation mixing ratio (kg/kg).
    """
    tc = t - 273.16
    tice = -23
    t0 = 0
    Rd = 287.04
    Rv = 461.5
    epsilon = Rd / Rv
 
    ewat = (1.0007 + (3.46e-6 * p)) * 6.1121 * np.exp(17.502 * tc / (240.97 + tc))
    eice = (1.0003 + (4.18e-6 * p)) * 6.1115 * np.exp(22.452 * tc / (272.55 + tc))
    eint = eice + (ewat - eice) * ((tc - tice) / (t0 - tice)) ** 2
 
    esat = eice.where(tc < tice, eint)
    esat = ewat.where(tc > t0, esat)
 
    rsat = epsilon * esat / (p - esat)
    qsat = epsilon * esat / (p - esat * (1 - epsilon))
    return esat, qsat, rsat
 
 
def vert_integral_layer_sum(ds, var, ilev_var="ilev", g=g_earth, pressure_factor=100.0):
    """Vertically integrate a variable by summing over layer thickness (mass-weighted).
 
    Uses layer interface pressures (``ilev``) to compute per-layer thickness
    ``dp`` and sums ``var * dp / g`` over the vertical dimension. This is
    dimension-order agnostic (unlike a manual roll-axis implementation) and
    works for both 3-D (lev, lat, lon) and 4-D (time, lev, lat, lon) inputs.
 
    Args:
        ds (xarray.Dataset): Input dataset containing ``var`` and the interface-level coordinate ``ilev_var``.
        var (str): Name of the variable in ``ds`` to integrate; must have a ``lev`` dimension.
        ilev_var (str): Name of the layer-interface pressure coordinate (default: "ilev").
        g (float): Gravitational acceleration in m/s^2 (default: 9.80665).
        pressure_factor (float): Multiplier to convert ``ilev_var`` units to Pascals (default: 100.0 for hPa -> Pa).
 
    Returns:
        integral (xarray.DataArray): Column integral of ``var``, with the ``lev`` dimension removed.
    """
    if var not in ds:
        raise ValueError(f"Variable '{var}' not found in the dataset.")
    if "lev" not in ds[var].dims:
        raise ValueError("Variable must have a 'lev' dimension for vertical integration.")
 
    p_interfaces = ds[ilev_var] * pressure_factor
    dp = p_interfaces.diff(ilev_var)
    dp = dp.rename({ilev_var: "lev"}).assign_coords(lev=ds["lev"].values)
 
    integral = (ds[var] * dp).sum("lev") / g
    return integral
 
 
def gaussian_area_weights(gw, nlon):
    """Build a normalized 2-D (lat, lon) area-weighting matrix from 1-D Gaussian latitude weights.
 
    Args:
        gw (numpy.ndarray or xarray.DataArray): 1-D Gaussian latitude weights, length nlat.
        nlon (int): Number of longitude points to tile the weights across.
 
    Returns:
        weight (numpy.ndarray): 2-D (lat, lon) array of area weights, normalized to sum to 1.
    """
    weight = np.transpose(np.tile(gw, [nlon, 1]))
    weight = weight / np.nansum(weight)
    return weight
 