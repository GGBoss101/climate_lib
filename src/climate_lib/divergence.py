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

def vertically_interpolate(varname, ds_ctrl, pnew, PS, hyam_ctrl, hybm_ctrl, P0pa_ctrl):

    var_ctrl = ds_ctrl[varname]  # (time:30, lev:30, lat:192, lon:288)
    # Vertical interpolation
    var_ctrl_tmp = np.ones((var_ctrl.shape[0], len(pnew), var_ctrl.shape[2], var_ctrl.shape[3]))*float('nan')
    for i in range(ds_ctrl['time'].size):
        # try with geocat instead of ngl
       var_ctrl_tmp[i,:,:,:] = geocat.comp.interpolation.interp_hybrid_to_pressure(var_ctrl[i], 
                                    PS[i], hyam_ctrl[i], hybm_ctrl[i], p0=P0pa_ctrl[i],
                                    new_levels=pnew, 
                                    lev_dim='lev', 
                                    method='linear', extrapolate=False, variable=None, t_bot=None, phi_sfc=None)
    
    var_ctrl_out = xr.DataArray(data=var_ctrl_tmp, 
                          dims=['time','lev','lat','lon'], 
                          coords={'time':var_ctrl.time, 'lev':pnew, 'lat':var_ctrl.lat, 'lon':var_ctrl.lon}, 
                          attrs={'long_name': var_ctrl.long_name, 'units': var_ctrl.units})

    return var_ctrl_out

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