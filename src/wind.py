# Wind speed/capacity factor/drought event functions
import xarray as xr
import xclim as xc
import numpy as np
import pandas as pd

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

    return drought_events['event_length']
    
def calc_wddx_timeseries(timeseries, pctl):
    """Calculate the annual max wind drought duration (WDDx) for a single timeseries"""
    
    drought_events = calc_wdd_timeseries(timeseries, pctl)
    wddx_da = drought_events.resample(time='1YS').max()
    wddx_times = drought_events.resample(time='1YS').map(xr.DataArray.idxmax, dim='time', keep_attrs=True)
    wddx_da['time'] = wddx_times
    wddx_ds = wddx_da.to_dataset()

    return wddx_ds

def get_droughts(da):
    """Return dataframe of drought duration, start date and end date"""
    
    years = da.time.dt.year.values
    df = pd.DataFrame(index=years)
    df['WDD'] = da.values.astype('int32')
    df['start_date'] = pd.to_datetime(da.time.dt.strftime('%Y-%m-%d'))
    df['end_date'] = pd.to_datetime(da.time.dt.strftime('%Y-%m-%d')) + pd.to_timedelta(df['WDD'] - 1, unit='D')
    return df.set_index('start_date')

def split_droughts(df, wdd_col="WDD", q1=0.95, q2=0.99):
    """Return dict of top events and other events, as defined by quantiles"""
    q_low, q_high = df[wdd_col].quantile([q1, q2])
    
    print("n events:", len(df))
    print("low WDD thresh:", q_low, "high WDD thresh:", q_high)

    top = df[df[wdd_col] >= q_high].sort_values(wdd_col, ascending=False)
    other = df[
        (df[wdd_col] >= q_low) & (df[wdd_col] < q_high)
    ].sort_values(wdd_col, ascending=False)

    print("n top events:", len(top), "n other events:", len(other))
    print()

    return {"top": top, "other": other}