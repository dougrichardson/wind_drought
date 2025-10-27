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

def sel_month(ds, month):
    """
    Return array for specified month
    
    ds: dataset to select from
    month: int or list of int between 1 and 12, default is None
    """
    if month is None:
        return ds
    elif isinstance(month, int):
        if 1 <= month <= 12:
            return ds.isel(time=ds.time.dt.month == month)
        else:
            raise ValueError("Incorrect month specified.")
    elif isinstance(month, list):
        if all((isinstance(i, int)) & (1 <= i <= 12) for i in month):
            return ds.isel(time=ds.time.dt.month.isin(month))
        else:
            raise ValueError("Incorrect month specified.")