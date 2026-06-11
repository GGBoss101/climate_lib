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

# Helper functions for climateLib, can also be imported/used for other purposes.

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