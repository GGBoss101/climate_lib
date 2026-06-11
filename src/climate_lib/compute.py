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

#===============================
# Vertical integration functions
#===============================
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

def div_uqvq_manual(ds, g = 9.81, Re = 6.371e6):
    """Compute vertically integrated moisture flux divergence from CESM data.

    Args:
        ds (Dataset): Input CESM dataset containing variables U, V, Q, PS, and hybrid coordinate coefficients.
        g (float): Gravitational acceleration in m/s^2 (default: 9.81).
        Re (float): Earth radius in meters (default: 6.371e6).

    Returns:
        qdiv (DataArray): Vertically integrated moisture divergence (kg m-2 s-1).
    """

    hyai = ds.hyai
    hybi = ds.hybi
    # Default to 100000 Pa if P0 is missing from the dataset
    P0 = ds.get("P0", 100000.0)

    #(time, lat, lon)
    PS = ds.PS
    #(time, lev, lat, lon)
    U = ds.U
    V = ds.V
    Q = ds.Q

    # Compute pressure at interfaces (Pa)
    # CESM formula: p = A * P0 + B * PS
    p_int = hyai * P0 + hybi * PS

    # Compute pressure thickness Δp between interfaces
    dp_raw = p_int.diff(dim="ilev")
    
    # Swap the coordinate dimension name from 'ilev' to 'lev' 
    dp = dp_raw.rename({"ilev": "lev"}).assign_coords(lev=ds.lev)

    # Moisture fluxes and vertical integration
    # Combines flux calculations and integration into streamlined memory passes.
    UQ = U * Q
    VQ = V * Q
    uq_int = (UQ * dp).sum(dim = "lev") / g
    vq_int = (VQ * dp).sum(dim = "lev") / g

    # Compute horizontal divergence in spherical coordinates
    # Convert latitude to radians to scale grid cell sizes by latitude circles
    lat_rad = np.deg2rad(ds.lat)
    coslat = np.cos(lat_rad)

    # Convert coordinates from degrees to radians for calculus metrics
    deg_per_rad = 180.0 / np.pi

    # Longitudinal gradient component (scaled by parallel convergence near poles)
    du_dx = (
        (uq_int * coslat).differentiate("lon")
        * deg_per_rad
        / (Re * coslat)
    )

    # Latitudinal gradient component
    dv_dy = vq_int.differentiate("lat") * deg_per_rad / Re

    # Combined horizontal moisture divergence
    qdiv = du_dx + dv_dy

    # Attach Metadata (Modifies attributes directly on the generated array)
    qdiv.name = "QDIV"
    qdiv.attrs = {
        "long_name": "Vertically integrated water vapor divergence",
        "units": "kg m-2 s-1",
    }

    return qdiv

#=====================
# Divergence functions
#=====================

def dominguez_uqdiv(ds,
    dPmb = 25,
    g = 9.81,
    R = 6.37122e6,
    pressure_factor = 100.0,):
    """
    Compute vertically integrated moisture flux divergence
    using pressure-level interpolation.

    Args:
        ds (Dataset): CESM atmospheric dataset
        dPmb (float): Pressure spacing in hPa (default 25)
        g (float): Gravity (m/s²) (default 9.81)
        R (float): Planet radius (m) (default 6.37122e6 = Earth radius)
        pressure_factor (float): Factor to convert pressure levels to Pascals (default 100.0)

    Returns:
        qdiv (DataArray): Moisture flux divergence (mm/day)
    """
    ds_ctrl = ds
    # Calculate moisture flux divergence (qv vi -> q_div -> diff -> tavg)


    U_ctrl = ds_ctrl.U
    V_ctrl = ds_ctrl.V
    Q_ctrl = ds_ctrl.Q


    PS = ds.PS # (time, lat, lon)

    # constants
    lat = ds_ctrl.lat
    lon = ds_ctrl.lon

    r = R * np.cos(lat * np.pi / 180.)

    # Interpolate to pressure coordinates
    dPmb = 25
    pnew = (np.arange(100, 1000 + dPmb, dPmb))

    # Convert units to Pascals
    dP = dPmb * pressure_factor
    pnew = pnew * pressure_factor

    # Do the interpolation.
    intyp = 1                              # 1=linear, 2=log, 3=log-log
    kxtrp = False                          # True=extrapolate (when the output pressure level is outside of the range of psrf)

    # Extract the desired variables
    hyam_ctrl = ds_ctrl.hyam  # (time:30, lev:30)
    hybm_ctrl = ds_ctrl.hybm  # (time:30, lev:30)
    psrf_ctrl = ds_ctrl.PS  # (time:30, lat:192, lon:288)
    P0pa_ctrl = ds_ctrl.P0  # (time:30)

    U_ctrl = vertically_interpolate('U', ds_ctrl, pnew, PS, hyam_ctrl, hybm_ctrl, P0pa_ctrl)
    V_ctrl = vertically_interpolate('V', ds_ctrl, pnew, PS, hyam_ctrl, hybm_ctrl, P0pa_ctrl)
    Q_ctrl = vertically_interpolate('Q', ds_ctrl, pnew, PS, hyam_ctrl, hybm_ctrl, P0pa_ctrl)

    ctrl_plev = xr.Dataset({'U': U_ctrl, 'V': V_ctrl, 'Q': Q_ctrl})

    
    
    # vertically integrated qv
    VIMF_ctrl_x = np.nansum(U_ctrl * Q_ctrl * dP, axis = 1) / g
    VIMF_ctrl_y = np.nansum(V_ctrl * Q_ctrl * dP, axis = 1) / g


    # get the cosine stuff in
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    dlat = np.gradient(lat_rad)
    dlon = np.gradient(lon_rad)

    # cosphi = np.cos(lat_rad)
    cosphi = np.cos(lat_rad.values)

    # q_div
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


    VIMF_ctrl_x_da = xr.DataArray(data=VIMF_ctrl_x, 
                                dims=['time','lat','lon'], 
                                coords={'time':var_ctrl['time'], 'lat':var_ctrl['lat'], 'lon':var_ctrl['lon']}, 
                                attrs={'long_name': 'Vertically integrated moisture flux x-component', 'units': 'kg m-1 s-1'})
    
    VIMF_ctrl_y_da = xr.DataArray(data=VIMF_ctrl_y, 
                                dims=['time','lat','lon'], 
                                coords={'time':var_ctrl['time'], 'lat':var_ctrl['lat'], 'lon':var_ctrl['lon']}, 
                                attrs={'long_name': 'Vertically integrated moisture flux y-component', 'units': 'kg m-1 s-1'})

    q_div_ctrl_da = xr.DataArray(data=Z_ctrl_tmp, 
                                dims=['time','lat','lon'], 
                                coords={'time':var_ctrl['time'], 'lat':var_ctrl['lat'], 'lon':var_ctrl['lon']}, 
                                attrs={'long_name': 'Vertically integrated moisture flux divergence', 'units': 'mm/day'})

    ctrl_q_div = xr.Dataset({'VIMF_x': VIMF_ctrl_x_da, 'VIMF_y': VIMF_ctrl_y_da, 'q_div': q_div_ctrl_da})

    return q_div_ctrl_da

#============================
# Linear regression functions
#============================

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