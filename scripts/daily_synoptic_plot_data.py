#!/usr/bin/env python
# coding: utf-8

# =================================
# Compute data for synoptic evolution figures
# =================================

import xarray as xr
import pandas as pd
from pathlib import Path

import src.wind as wind
import src.utils as utils
import src.read as read
import src.regions as regions
import src.meteorology as meteo

def iter_leaves(obj, key_path=()):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from iter_leaves(value, key_path + (str(key),))
    else:
        yield key_path, obj

def event_slice(da, event_start, event_duration, days_before, days_after):
    """ Return da around event """
    time_slice = pd.date_range(
        pd.to_datetime(event_start) - pd.DateOffset(days=days_before),
        pd.to_datetime(event_start) + pd.DateOffset(days=event_duration + days_after - 1)
    ) + pd.Timedelta(hours=12)
    return da.sel(time=time_slice)

def barra_var_for_month(var_name, var_path, year, month):
    """
    BARRA-R2 data for a given month

    var_name: str, name of variable
    var_path: str, path to variable
    year: str, year to open
    month: str, month to open
    """
    def _barraR2_preprocess(ds):
        """ Return smaller region """
        ds = ds.sel(
            lon=slice(REGION[0], REGION[1]),
            lat=slice(REGION[3], REGION[2])
        )
        return ds
    REGION = regions.get_Aus_boundary()
        
    var_fname = var_name + '_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_day_'
    return xr.open_mfdataset(
        var_path + var_fname + year + month + '-' + year + month + '.nc',
        preprocess=_barraR2_preprocess
    )[var_name]

def barra_var_for_event(event_start, event_duration, days_before, days_after, var_name, var_path):
    """
    BARRA-R2 data during an event

    event_start: str, start day of event in format yyyy-mm-dd
    event_duration: int or float, duration of event
    days_before: int, days before event
    days_after: int, days after event
    var_name: str, name of variable
    var_path: str, path to variable
    """
    year, month, day = event_start.split('-')
    next_month = (int(month)+1) % 12
    next_month = f"{next_month:02}"

    # Data for start month, and next month, in case event lasts into next month
    var_month = barra_var_for_month(var_name, var_path, year, month)
    var_month_next = barra_var_for_month(var_name, var_path, year, next_month)
    var_month = xr.concat([var_month, var_month_next], dim="time")

    time_slice = pd.date_range(
        pd.to_datetime(event_start) - pd.DateOffset(days=days_before),
        pd.to_datetime(event_start) + pd.DateOffset(days=event_duration + days_after - 1)
    ) + pd.Timedelta(hours=12)

    return var_month.sel(time=time_slice)

def write_sfc_plot_data(event_df, path, event_id, days_before=0, days_after=2):
    """Return array of event data to plot"""
    start = event_df.iloc[event_id].name.strftime("%Y-%m-%d")
    duration, end = event_df[['WDD', 'end_date']].iloc[event_id]

    # # sfcWind speed and direction
    # wind_speed = event_slice(wind_speed_barra, start, duration, days_before, days_after)
    # wind_direction = event_slice(wind_dir_barra, start, duration, days_before, days_after)

    # surface u and v wind
    uas = barra_var_for_event(start, duration, days_before, days_after, 'uas', uas_barra_path)
    vas = barra_var_for_event(start, duration, days_before, days_after, 'vas', vas_barra_path)

    # MSLP
    mslp = barra_var_for_event(start, duration, days_before, days_after, 'psl', psl_barra_path)

    # temperature anomalies
    tas = event_slice(tas_barra, start, duration, days_before, days_after)

    # cloud cover
    clt = barra_var_for_event(start, duration, days_before, days_after, 'clt', clt_barra_path)

    # ds = xr.merge([wind_speed, wind_direction, mslp, tas, clt], compat="no_conflicts")
    ds = xr.merge([uas, vas, mslp, tas, clt], compat="no_conflicts")

    path += "/sfc_barra/"
    out_path = Path(path + "wdd_composite_" + str(event_id) + "_" + start + ".nc")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    ds.to_netcdf(out_path)

def write_pl_plot_data(event_df, path, event_id, days_before=0, days_after=2):
    """Return array of event data to plot"""
    start = event_df.iloc[event_id].name.strftime("%Y-%m-%d")
    duration, end = event_df[['WDD', 'end_date']].iloc[event_id]

    # 850 hPa temperature anomalies
    t850 = event_slice(t850_era["t_anom"], start, duration, days_before, days_after)
    t850.name = 't850_anom'
    
    # 850 hPa u and v anomalies
    u850 = event_slice(uv850_era["u"], start, duration, days_before, days_after)
    u850.name = 'u850'
    v850 = event_slice(uv850_era["v"], start, duration, days_before, days_after)
    v850.name = 'v850'

    # 500 hPa geopotential height
    z500 = event_slice(z500_era["z"], start, duration, days_before, days_after)
    z500.name = 'z500'
    z500_anom = event_slice(z500_era["z_anom"], start, duration, days_before, days_after)
    z500_anom.name = 'z500_anom'

    # 300 hPa u
    u300 = event_slice(u300_era["u"], start, duration, days_before, days_after)
    u300.name = 'u300'

    ds = xr.merge([t850, u850, v850, z500, z500_anom, u300], compat="no_conflicts")

    path += "/pl_era/"
    out_path = Path(path + "wdd_composite_" + str(event_id) + "_" + start + ".nc")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    ds.to_netcdf(out_path)

OUTPUT_ROOT = Path("/g/data/ng72/dr6273/work/projects/wind_drought/plotting/wind_drought_composites")

if __name__ == "__main__":

    # Load WDD data - for now just look at BARRA-R2, capacity factor, SC30 weighted results
    wdd_dict = {}
    for dataset in ["BARRA-R2"]:
        wdd_dict[dataset] = {}
    
        for variable in ["capacity_factor"]:
            wdd_dict[dataset][variable] = {}
    
            for region in ["NEM", "SE", "SWIS", "NWIS"]:
                
                if region in ["NEM", "SE"]:
                    for scenario in ["SC30"]:
                        wdd_dict[dataset][variable][region+"-"+scenario] = {}
    
                        for averaging in ["weighted"]:
                            da = read.load_wind(dataset, variable, region, scenario, averaging)[variable]
                            pctl = utils.sel_month(da, [5, 6, 7]).quantile(0.1, "time")
                            wdd = wind.calc_wdd_timeseries(da, pctl)
                            
                            wdd_dict[dataset][variable][region+"-"+scenario][averaging] = wdd
    
                else:
                    da = read.load_wind(dataset, variable, region)[variable]
                    pctl = utils.sel_month(da, [5, 6, 7]).quantile(0.1, "time")
                    wdd = wind.calc_wdd_timeseries(da, pctl)
                    
                    wdd_dict[dataset][variable][region] = wdd

    # Compute drought events
    events_dict = utils.map_leaves(
        wdd_dict,
        lambda value: wind.get_droughts(value)
    )
    
    # Obtain subset of longest droughts (>99th percentile) (discarding other droughts (95-99th percentile))
    subset_dict = utils.map_leaves(events_dict, wind.split_droughts)

    # Meteorological data
    # BARRA-R2
    barra_diagnostics_path = "/g/data/ng72/dr6273/work/projects/wind_drought/data/BARRA-R2/"

    # tas
    tas_barra = xr.open_zarr(
        barra_diagnostics_path + "tas_dayofyear_anoms_1979-2025.zarr",
        consolidated=False
    )["tas"].drop_vars(["height", "dayofyear", "crs"])

    # MSLP
    psl_barra_path = '/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/day/psl/latest/'

    # Cloud cover
    clt_barra_path = '/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/day/clt/latest/'

    # u and v surface
    uas_barra_path = '/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/day/uas/latest/'
    vas_barra_path = '/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/day/vas/latest/'

    # Wind speed and direction
    wind_barra = xr.open_zarr(
        barra_diagnostics_path + "wind_sfc_speed_direction_1979-2025.zarr",
        consolidated=False
    ).drop_vars(["height", "crs"])

    wind_speed_barra = wind_barra["wind_speed"]
    wind_dir_barra = wind_barra["wind_direction"]

    # ERA5
    era_diagnostics_path = "/g/data/ng72/dr6273/work/projects/wind_drought/data/ERA5/"

    # t850
    t850_era = xr.open_zarr(
        era_diagnostics_path + "t850_dayofyear_anoms_1979-2024.zarr",
        consolidated=False
    ).drop_vars(["dayofyear", "level"])
    t850_era["time"] = t850_era["time"] + pd.Timedelta(hours=1) # Add 1 hour to match BARRA times

    # u and v at 850 hPa
    uv850_era = xr.open_zarr(
        era_diagnostics_path + "uv850_dayofyear_anoms_1979-2024.zarr",
        consolidated=False
    ).drop_vars(["level"])
    uv850_era["time"] = uv850_era["time"] + pd.Timedelta(hours=1)

    # z500
    z500_era = xr.open_zarr(
        era_diagnostics_path + "z500_dayofyear_anoms_1979-2024.zarr",
        consolidated=False
    ).drop_vars(["dayofyear", "level"])
    z500_era["time"] = z500_era["time"] + pd.Timedelta(hours=1)
    # Convert to geopotential height
    g0 = 9.80665 # m / s^2
    z500_era = z500_era / g0

    # u300
    u300_era = xr.open_zarr(
        era_diagnostics_path + "u300_dayofyear_anoms_1979-2024.zarr",
        consolidated=False
    ).drop_vars(["dayofyear", "level"])
    u300_era["time"] = u300_era["time"] + pd.Timedelta(hours=1)

    # Loop through dictionary and write composite data for each event
    for key_path, leaf in iter_leaves(subset_dict):    
        df = (
            leaf.rename_axis("start_date")
            .reset_index()
            .sort_values(
                ["WDD", "start_date"],
                ascending=[False, True],
            )
            .set_index("start_date")
        )
    
        leaf_output_dir = OUTPUT_ROOT.joinpath(*key_path)
        leaf_output_dir.mkdir(parents=True, exist_ok=True)

        for event_id, start_date in enumerate(df.index):
            if start_date.year < 2025:
                write_sfc_plot_data(df, str(leaf_output_dir), event_id)
                write_pl_plot_data(df, str(leaf_output_dir), event_id)