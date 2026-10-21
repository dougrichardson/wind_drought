# Wind speed/capacity factor/drought event functions
import xarray as xr
import xclim as xc
import numpy as np

def capacity_factor(W):
    """
    Computes capacity factor from wind speed data.
    
    W: dataarray of wind speed (m/s)
    """
    W_0 = 3.5 # cut-in speed (m/s)
    W_r = 13 # rated speed
    W_1 = 25 # cut-out speed (m/s)
    
    # Cubic
    c_f = (W ** 3 - W_0 ** 3) / (W_r ** 3 - W_0 ** 3)
    c_f = c_f.where(W >= W_0, 0) # Set values below cut-in to zero
    c_f = c_f.where(W < W_r, 1) # Set values above rated speed to 1
    c_f = c_f.where(W < W_1, 0) # Set values above cut-off to zero
    c_f = c_f.where(W.notnull(), np.nan) # Ensure NaNs are retained
    
    return c_f

def calc_wdd_timeseries(timeseries, pctl):
    """Calculate wind drought durations (WDD) for a single time series"""
    
    calm_days = timeseries < pctl
    drought_events = xc.indices.run_length.find_events(calm_days, window=1)
    drought_events = drought_events.assign_coords(
        time=('event', drought_events['event_start'].data)
    ).drop_vars('event_start')
    drought_events = drought_events.swap_dims({'event': 'time'}).dropna('time')

    return drought_events
    
def calc_wddx_timeseries(timeseries, pctl):
    """Calculate the annual max wind drought duration (WDDx) for a single timeseries"""

    # calm_days = timeseries < pctl10
    # drought_events = xc.indices.run_length.find_events(calm_days, window=1)
    # drought_events = drought_events.assign_coords(
    #     time=('event', drought_events['event_start'].data)
    # ).drop_vars('event_start')
    # drought_events = drought_events.swap_dims({'event': 'time'}).dropna('time')
    
    drought_events = calc_wdd_timeseries(timeseries, pctl)
    wddx_da = drought_events['event_length'].resample(time='1YS').max()
    wddx_times = drought_events['event_length'].resample(time='1YS').map(xr.DataArray.idxmax, dim='time', keep_attrs=True)
    wddx_da['time'] = wddx_times
    wddx_ds = wddx_da.to_dataset()

    return wddx_ds