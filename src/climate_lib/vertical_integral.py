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


def vert_integral(ds, var = None, g = 9.80665, pressure_factor = 100.0):
    """Vertically integrate a DataArray in pressure coordinates.

    Args:
        ds (Dataset/DataArray): The input atmospheric data.
        var (str, optional): The specific variable name to extract from ds.
        g (float): Gravity acceleration constant in m/s^2 (default: 9.80665).
        pressure_factor (float): Multiplier to convert pressure units to Pascals
        (default: 100.0 for hPa -> Pa).

    Returns:
        integral (DataArray): Vertically integrated field (time, lat, lon).
    """

    # Select the specific variable if a variable was passed
    if var is not None:
        da = ds[var]
    else:
        da = ds

    # Calculate layer thicknesses from interface levels (ilev)
    interface_diffs = ds.ilev.diff(dim="ilev")

    # Convert to a DataArray for thickness using the mid-level (lev) coordinates
    dp = interface_diffs.rename({"ilev": "lev"}).assign_coords(lev=da.lev)

    # Scale the thicknesses to Pascals (e.g., hPa -> Pa)
    dp = dp * pressure_factor

    # Multiply the data by the layer thickness
    weighted_data = da * dp

    # Sum along the vertical axis and divide by gravity
    integral = weighted_data.sum(dim = "lev") / g

    return integral

def vert_integral_optimized(da, var = None, g = 9.80665, pressure_factor = 100.0):
    """Vertically integrate a DataArray using Xarray's native calculus.

    Args:
        da (Dataset/DataArray): The input atmospheric data.
        var (str, optional): The specific variable name to extract from da.
        g (float): Gravity acceleration constant in m/s^2 (default: 9.80665).
        pressure_factor (float): Multiplier to convert pressure units to Pascals (default: 100.0 for hPa -> Pa).

    Returns:
        integral (DataArray): Pressure-coordinate vertical integral
    with dimensions (time, lat, lon).
    """

    # Select the specific variable if a Dataset was passed
    if var is not None:
        da = da[var]

    # Scale the vertical coordinate values to Pascals (e.g., hPa -> Pa)
    da_scaled = da.assign_coords(lev = da.lev * pressure_factor)

    # Perform the mathematical integration along the vertical axis
    raw_integral = da_scaled.integrate(dim = "lev")

    # Divide the column by gravity to get mass-weighted units
    integral = raw_integral / g

    return integral

def vert_integral_hybrid(ds, var, ds_hybrid, g = 9.80665):
    """Vertically integrate a variable across hybrid sigma-pressure coordinates.

    Args:
        ds (Dataset): Input dataset containing the variable to integrate.
        var (str): String name of the variable to extract (e.g., 'Q').
        ds_hybrid (Dataset/DataArray): Source dataset for the hybrid coefficients.
        g (float): Gravitational acceleration in m/s^2 (default: 9.80665).

    Returns:
        total_int (DataArray): Vertically integrated result (e.g., kg/m² for Q).
    """

    # Dynamically read reference surface pressure safely
    # Falls back to 100000.0 Pa if P0 is missing from the dataset attributes/variables
    P0 = ds.get("P0", ds_hybrid.get("P0", 100000.0))

    # Extract necessary variables
    q = ds[var]
    PS = ds_hybrid.PS
    hyai = ds_hybrid.hyai
    hybi = ds_hybrid.hybi

    # Compute interface pressures (in Pa)
    # CESM formula: P_interfaces = A(k)*P0 + B(k)*PS
    P_i = hyai * P0 + hybi * PS

    # Compute pressure thickness (dp) natively using .diff()
    # .data preserves underlying Dask chunks and strips indexing mismatched names
    dp_raw = P_i.diff(dim = "ilev").data

    # Build dp directly using the target data's dimensions and coordinates
    # This replaces the slow .copy() and .values assignment entirely
    dp = xr.DataArray(dp_raw, dims = q.dims, coords = q.coords)

    # Compute mass-weighted column layer values and sum over vertical axis
    # Xarray handles broadcasting automatically; no manual transposing needed
    layer_int = (q * dp) / g
    total_int = layer_int.sum(dim = "lev", skipna = True)

    return total_int