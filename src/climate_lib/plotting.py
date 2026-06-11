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

import warnings
warnings.filterwarnings("ignore", message=".*multiple fill values.*")
warnings.filterwarnings("ignore", message="Interpolation point out of data bounds encountered")

def close_map(fig=None, ax=None):
    """Fully closes a Matplotlib/Cartopy figure and clears memory. If ax and/or fig passed, it will clear both individually, otherwise, it will close all figures.

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
    """Plot a North America map from a 2D Xarray DataArray.

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