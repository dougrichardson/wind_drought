import xarray as xr
import numpy as np

def get_Aus_boundary():
    """
    Returns a list of bounding coordinates for Australia.
    """
    return [110, 155, -10, -45]

def windspeed(u, v):
    """
    Compute windspeed from u and v
    
    u: array of zonal wind
    v: array of meridional wind
    """
    return np.sqrt(u ** 2 + v ** 2)