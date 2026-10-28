#!/usr/bin/env python
# coding: utf-8

# =================================
# Compute daily mean ERA5 100m capacity factor
# =================================

import xarray as xr
import numpy as np
import argparse

import src.regions as regions
import src.read as read
import src.wind as wind

def era_preprocess(ds):
    """
    Preprocess function for open_mfdataset.
    Selects Australian region and renames coords.
    """
    ds = ds.sel(
        longitude=slice(REGION[0], REGION[1]),
        latitude=slice(REGION[2], REGION[3])
    )
    ds = ds.rename(
        {'longitude': 'lon',
         'latitude': 'lat'}
    )
    return ds

ERA_PATH = "/g/data/rt52/era5/single-levels/reanalysis/" # hourly instantaneous
WRITE_PATH = "/g/data/ng72/dr6273/work/projects/wind_drought/data/ERA5/"
REGION = regions.get_Aus_boundary()
# YEARS = range(1979, 2026)

if __name__ == "__main__":
    from dask_setup import setup_dask_client 

    parser = argparse.ArgumentParser()
    parser.add_argument("first_year", type=int)
    parser.add_argument("last_year", type=int)
    args = parser.parse_args()

    YEARS = range(args.first_year, args.last_year + 1)
    
    client, cluster, dask_tmp = setup_dask_client(mode="interactive", workload_type="cpu")   # heavy compute

    u100_files = read.era_get_files(ERA_PATH, '100u', start_year=YEARS[0], end_year=YEARS[-1])
    v100_files = read.era_get_files(ERA_PATH, '100v', start_year=YEARS[0], end_year=YEARS[-1])

    u100 = read.open_files(u100_files, preprocess=era_preprocess)
    v100 = read.open_files(v100_files, preprocess=era_preprocess)
    
    u100_chunked = u100.chunk({"time": -1, "lat": 25, "lon": 3})
    v100_chunked = v100.chunk({"time": -1, "lat": 25, "lon": 3})
    
    u100_chunked = u100_chunked["u100"]
    v100_chunked = v100_chunked["v100"]
    
    wind_speed = np.sqrt(u100_chunked ** 2 + v100_chunked ** 2)
    capacity_factor = wind.capacity_factor(wind_speed)

    capacity_factor = capacity_factor.coarsen(time=24).mean() # Hourly to daily mean
    capacity_factor["time"] = capacity_factor["time"].dt.floor("D")
    
    capacity_factor = capacity_factor.chunk({"time": -1, "lat": -1, "lon": 16})
    
    capacity_factor = capacity_factor.to_dataset(name="capacity_factor")
    
    capacity_factor_encoding = read.specify_encoding(capacity_factor)
    
    read.write_zarr(
        capacity_factor,
        WRITE_PATH,
        'era5_capacity_factor_100m_'+str(YEARS[0]) + '-' + str(YEARS[-1]) + '.zarr',
        capacity_factor_encoding
    )