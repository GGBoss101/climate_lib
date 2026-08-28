"""
This module contains functions for plotting climate data, including maps and other visualizations.
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

def close_map(fig=None, ax=None):
    """
    Fully closes a Matplotlib/Cartopy figure and clears memory. If ``ax`` and/or ``fig`` passed, it will clear both individually, otherwise, it will close all figures.

    Args:
        fig: matplotlib Figure object (preferred)
        ax: optional axis (not required, included for flexibility)
    """

    if ax is not None:
        ax.clear()  # removes plot elements from axis

    if fig is not None:
        plt.close(fig)  # fully destroys the figure window
    else:
        plt.close("all")  # fallback: closes everything

def NA_map(mapdata, cmap, clim, units="", title=""):
    """
    Plot a North America map from a 2D Xarray DataArray.

    Args:
        mapdata (DataArray): 2D data with latitude and longitude coordinates. Expected dimensions are typically (lat, lon).
        cmap (str or Colormap): Matplotlib colormap name/object.
        clim (tuple/list): Color limits as (vmin, vmax).
        units (str): Label for the colorbar units.
        title (str): Figure title.

    Returns:
        fig (Figure): Matplotlib figure object.
        ax (GeoAxes): Cartopy map axis object.
    """

    # Create figure and map projection

    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(12, 12),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    # Add geographic features

    ax.coastlines(linewidth=0.5)

    ax.add_feature(
        cfeature.BORDERS,
        linestyle=":",
        linewidth=0.5,
    )

    ax.add_feature(
        cfeature.STATES,
        linewidth=0.5,
    )

    # Restrict map view to North America

    ax.set_extent(
        [220, 280, 0, 60],
        crs=ccrs.PlateCarree(),
    )

    # Plot the data

    # pcolormesh draws colored grid cells.

    cs = ax.pcolormesh(
        mapdata.lon,
        mapdata.lat,
        mapdata,
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
        transform=ccrs.PlateCarree(),
        shading="auto",
    )

    # Add title

    ax.set_title(title, fontsize=15)

    # Create colorbar

    cbar = fig.colorbar(
        cs,
        ax=ax,
        orientation="horizontal",
        pad=0.02,
        shrink=0.9,
    )

    cbar.ax.tick_params(labelsize=14)

    cbar.set_label(
        units,
        fontsize=15,
    )

    # Display the figure

    plt.show()

    return fig, ax

def zonal_mean_spread(da, avg_dims=("month", "year"), lon_dim="lon", spread_dim="year", spread_stat="std"):
    """Compute a zonal (longitude) mean profile and a spread band across another dimension.
 
    Averages ``da`` over ``avg_dims`` and ``lon_dim`` to get a zonal-mean
    latitude profile, and separately computes the spread (e.g. inter-annual
    standard deviation) across ``spread_dim`` after averaging over
    longitude, for use as a shaded uncertainty band in latitude-profile plots.
 
    Args:
        da (xarray.DataArray): Field with at least latitude, longitude, and ``spread_dim`` dimensions.
        avg_dims (tuple of str): Dimensions to average over for the central estimate (default: ("month", "year")).
        lon_dim (str): Name of the longitude dimension (default: "lon").
        spread_dim (str): Dimension across which to compute the spread statistic, e.g. "year" (default: "year").
        spread_stat (str): Spread statistic to compute; one of "std" or "sem" (default: "std").
 
    Returns:
        mean_profile (xarray.DataArray): Latitude profile averaged over ``avg_dims`` and ``lon_dim``.
        spread_profile (xarray.DataArray): Latitude profile of the requested spread statistic, computed over ``spread_dim`` after averaging over ``lon_dim``.
    """
    mean_profile = da.mean(list(avg_dims))
    if lon_dim not in avg_dims:
        mean_profile = mean_profile.mean(lon_dim)
 
    lon_avgd = da.mean(lon_dim)
    remaining_avg_dims = [d for d in avg_dims if d != spread_dim]
    if remaining_avg_dims:
        lon_avgd = lon_avgd.mean(remaining_avg_dims)
 
    if spread_stat == "std":
        spread_profile = lon_avgd.std(spread_dim)
    elif spread_stat == "sem":
        spread_profile = lon_avgd.std(spread_dim) / np.sqrt(lon_avgd.sizes[spread_dim])
    else:
        raise ValueError(f"Unknown spread_stat '{spread_stat}'; expected 'std' or 'sem'.")
 
    return mean_profile, spread_profile