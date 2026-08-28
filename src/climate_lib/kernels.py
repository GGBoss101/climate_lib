"""
Utilities for radiative kernel analysis of climate model output.

Provides functions for estimating tropopause pressure, loading CAM5
radiative kernels, decomposing radiative flux changes into physical
components, and evaluating kernel reconstruction residuals.
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
import os

import geocat.comp

from climate_lib.utils import *
from climate_lib.constants import *


def estimate_diagnostic_tropopause(lat, p_eq=300.0, p_pole=150.0):
    """Estimate a crude climatological tropopause pressure as a function of latitude.

    Follows the common diagnostic approximation of ~100 hPa in the tropics,
    lowering with the cosine of latitude to ~300 hPa at the poles (the
    default parameters give 300 hPa at the poles and 150 hPa at the equator;
    adjust ``p_eq``/``p_pole`` to match the convention in use).

    Args:
        lat (xarray.DataArray): Latitude coordinate, in degrees.
        p_eq (float): Tropopause pressure offset used in ``p_eq - p_pole * cos(lat)`` (default: 300.0).
        p_pole (float): Amplitude of the cosine(latitude) term (default: 150.0).

    Returns:
        p_tropopause_zonalmean (xarray.DataArray): Zonal-mean estimated tropopause pressure (hPa), indexed by latitude.
    """
    x = np.cos(lat * np.pi / 180)
    p_tropopause_zonalmean = p_eq - p_pole * x
    return p_tropopause_zonalmean


def load_cam5_kernels(kernel_dir, target_lev=None, time_dim="time", month_dim="month"):
    """Load the standard CAM5 radiative kernel files into a dictionary and rename time->month.

    Expects the standard CAM5 kernel filenames (``alb.kernel.nc``,
    ``q.kernel.nc``, ``t.kernel.nc``, ``ts.kernel.nc``, ``PS.nc``) in
    ``kernel_dir``. The vertically-resolved 'q' and 't' kernels are
    interpolated onto ``target_lev`` if given (e.g. the model's own level
    coordinate), so they can be multiplied directly against model fields.

    Args:
        kernel_dir (str): Directory containing the CAM5 kernel netCDF files.
        target_lev (xarray.DataArray or None): Vertical level coordinate to interpolate the 'q' and 't' kernels onto (optional).
        time_dim (str): Name of the kernel files' native monthly time dimension (default: "time").
        month_dim (str): Name to rename ``time_dim`` to (default: "month").

    Returns:
        ker_dict (dict): Dictionary with keys 'alb', 'q', 'T', 'Ts', 'PS', each an xarray.Dataset with the time dimension renamed to ``month_dim``.
    """
    ker_dict = {}
    ker_dict["alb"] = xr.open_dataset(os.path.join(kernel_dir, "alb.kernel.nc"),
                                       decode_times=False, chunks={time_dim: 1})
    ker_dict["Ts"] = xr.open_dataset(os.path.join(kernel_dir, "ts.kernel.nc"),
                                      decode_times=False, chunks={time_dim: 1})
    ker_dict["PS"] = xr.open_dataset(os.path.join(kernel_dir, "PS.nc"),
                                      decode_times=False, chunks={time_dim: 1})

    q_kernel = xr.open_dataset(os.path.join(kernel_dir, "q.kernel.nc"),
                                decode_times=False, chunks={time_dim: 1, "lev": 1})
    t_kernel = xr.open_dataset(os.path.join(kernel_dir, "t.kernel.nc"),
                                decode_times=False, chunks={time_dim: 1, "lev": 1})
    if target_lev is not None:
        q_kernel = q_kernel.interp(lev=target_lev)
        t_kernel = t_kernel.interp(lev=target_lev)
    ker_dict["q"] = q_kernel
    ker_dict["T"] = t_kernel

    for key in ker_dict:
        ker_dict[key] = ker_dict[key].rename({time_dim: month_dim})

    return ker_dict


def decompose_flux_change(ds_ctrl, ds_ctrl_vert, ds_case, ds_case_vert, ker_dict,
                           p, p_tropopause, dalb, flux_sw, flux_lw, alb_scale=100.0):
    """Decompose the change in a TOA or surface radiative flux into kernel-attributed components.

    Implements the radiative kernel technique (Soden et al. 2008; Shell et
    al. 2008): the modelled change in a shortwave/longwave flux pair is
    attributed to surface albedo, surface temperature, atmospheric
    temperature, and water vapour changes by convolving each field's change
    with its corresponding pre-computed radiative kernel, with clouds taken
    as the residual between the modelled flux change and the sum of the
    other kernel-predicted terms.

    Calling this once per flux pair (e.g. ``("FSNT", "FLNT")`` for full-sky
    TOA, ``("FSNTC", "FLNTC")`` for clear-sky TOA, ``("FSNS", "FLNS")`` for
    full-sky surface, ``("FSNSC", "FLNSC")`` for clear-sky surface) replaces
    duplicating this ~150-line calculation once per flux type.

    Args:
        ds_ctrl (xarray.Dataset): Control-run surface/TOA flux dataset (unstacked to year x month), must contain 'TS', ``flux_sw``, ``flux_lw``.
        ds_ctrl_vert (xarray.Dataset): Control-run vertical profile dataset (unstacked to year x month), must contain 'T', 'Q' with a 'lev' dimension.
        ds_case (xarray.Dataset): Perturbed-run counterpart to ``ds_ctrl``.
        ds_case_vert (xarray.Dataset): Perturbed-run counterpart to ``ds_ctrl_vert``.
        ker_dict (dict): Radiative kernel dictionary keyed by kernel type ('alb', 'Ts', 'T', 'q'), each a Dataset containing variables named after flux fields (e.g. ``ker_dict['T']['FLNT']``). See ``load_cam5_kernels``.
        p (xarray.DataArray): Full atmospheric pressure field (hPa), broadcastable against the vertical datasets.
        p_tropopause (xarray.DataArray): Tropopause pressure field, broadcastable against ``p`` (see ``estimate_diagnostic_tropopause``).
        dalb (xarray.DataArray): Surface albedo change (case minus control) to convolve with the albedo kernel.
        flux_sw (str): Name of the shortwave flux variable to decompose, e.g. "FSNT", "FSNTC", "FSNS", or "FSNSC".
        flux_lw (str): Name of the longwave flux variable to decompose, e.g. "FLNT", "FLNTC", "FLNS", or "FLNSC".
        alb_scale (float): Multiplier applied to the albedo kernel convolution, e.g. to convert a fractional albedo change to percent (default: 100.0).

    Returns:
        breakdown (dict): Dictionary of xarray.DataArray with keys 'dSW_total', 'dSW_dq', 'dSW_dclouds', 'dSW_alb_tot', 'dLW_total', 'dLW_dq', 'dLW_dclouds', 'dLW_dT', 'dLW_dTs'.
    """
    dTs = ds_case["TS"] - ds_ctrl["TS"]
    dT = ds_case_vert["T"] - ds_ctrl_vert["T"]

    # --- Albedo SW effect ---
    dalb_clean = xr.where(np.isnan(dalb), 0.0, dalb)
    dFSNT_alb_t = ker_dict["alb"][flux_sw] * dalb_clean * alb_scale

    # --- Surface temperature LW effect ---
    dLW_Ts = ker_dict["Ts"][flux_lw] * dTs

    # --- Atmospheric temperature LW effect (masked to troposphere) ---
    dT_trop = mask_above_tropopause(dT, p, p_tropopause)
    dLW_T = (ker_dict["T"][flux_lw] * dT_trop).sum("lev")

    # --- Water vapour SW/LW effect (log-q normalized) ---
    T0 = ds_ctrl_vert["T"]
    dlogqdt = logq_kernel_normalization(T0, dT, p)

    logq_LW_kernel = ker_dict["q"][flux_lw] / dlogqdt
    logq_SW_kernel = ker_dict["q"][flux_sw] / dlogqdt

    dlogq = np.log(ds_case_vert["Q"]) - np.log(ds_ctrl_vert["Q"])

    dLW_logq = (logq_LW_kernel * dlogq).sum("lev")
    dSW_logq = (logq_SW_kernel * dlogq).sum("lev")

    # --- Cloud effect (residual) ---
    dFSNT_model = ds_case[flux_sw] - ds_ctrl[flux_sw]
    dFSNT_clouds = dFSNT_model - (dFSNT_alb_t + dSW_logq)

    dFLNT_model = ds_case[flux_lw] - ds_ctrl[flux_lw]
    dLW_clouds = dFLNT_model - (dLW_logq + dLW_Ts + dLW_T)

    return {
        "dSW_total": dFSNT_model,
        "dSW_dq": dSW_logq,
        "dSW_dclouds": dFSNT_clouds,
        "dSW_alb_tot": dFSNT_alb_t,
        "dLW_total": dFLNT_model,
        "dLW_dq": dLW_logq,
        "dLW_dclouds": dLW_clouds,
        "dLW_dT": dLW_T,
        "dLW_dTs": dLW_Ts,
    }


def sum_flux_components(breakdown, plus_vars, minus_vars=None):
    """Linearly combine kernel-decomposed flux components into a reconstructed total.

    Useful for reassembling a kernel-predicted total flux change (e.g. SW
    components added, LW components subtracted for a TOA net-down
    convention) to compare against the modelled total for a linearity check.

    Args:
        breakdown (dict or xarray.Dataset): Mapping of component name to xarray.DataArray (e.g. the output of ``decompose_flux_change``, or a Dataset of the same).
        plus_vars (list of str): Component names to add.
        minus_vars (list of str or None): Component names to subtract (optional).

    Returns:
        total (xarray.DataArray): Linear combination of the requested components.
    """
    total = breakdown[plus_vars[0]]
    for var in plus_vars[1:]:
        total = total + breakdown[var]
    if minus_vars:
        for var in minus_vars:
            total = total - breakdown[var]
    return total


def kernel_residual(kernel_sum, model_total, as_percent=True):
    """Compute the residual between a kernel-reconstructed flux total and the modelled total.

    A common linearity check for the kernel technique (cf. Vial et al.
    2013): the kernel-predicted sum of components should closely match the
    directly modelled flux change; a large residual suggests a missing term
    or non-linearity.

    Args:
        kernel_sum (xarray.DataArray): Kernel-reconstructed total flux change (e.g. from ``sum_flux_components``).
        model_total (xarray.DataArray): Directly modelled flux change to compare against.
        as_percent (bool): If True, express the residual as a percentage of ``model_total`` (default: True).

    Returns:
        residual (xarray.DataArray): ``kernel_sum - model_total``, or that difference as a percent of ``model_total`` if ``as_percent`` is True.
    """
    diff = kernel_sum - model_total
    if as_percent:
        return 100.0 * diff / model_total
    return diff