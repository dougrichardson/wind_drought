#!/usr/bin/env python
# coding: utf-8

# =================================
# Compute ERA5 surface diagnostics
# =================================
import argparse
from pathlib import Path
import glob
from zarr.codecs import BloscCodec

import xarray as xr
import numpy as np

import src.regions as regions
import src.read as read
import src.utils as utils

def preprocess(ds):
    """ Return smaller region """
    ds = ds.sel(
        longitude=slice(REGION[0], REGION[1]),
        latitude=slice(REGION[2], REGION[3]),
    )
    return ds

ERA5_PATH = "/g/data/rt52/era5/single-levels/reanalysis/" # hourly instantaneous
ERA5_WRITE_PATH = "/g/data/ng72/dr6273/work/projects/wind_drought/data/ERA5/"
REGION = regions.get_Aus_boundary()

if __name__ == "__main__":
    from dask_setup import setup_dask_client
    client, cluster, dask_tmp = setup_dask_client(mode="interactive", workload_type="cpu")   # heavy compute

    parser = argparse.ArgumentParser()
    parser.add_argument("first_year", type=int)
    parser.add_argument("last_year", type=int)
    parser.add_argument("variable", type=str)
    parser.add_argument("first_year_clim", type=int)
    parser.add_argument("last_year_clim", type=int)

    args = parser.parse_args()
    
    first_year = args.first_year
    last_year = args.last_year
    variable = args.variable
    first_year_clim = args.first_year_clim
    last_year_clim = args.last_year_clim
    
    YEARS = range(first_year, last_year + 1)
    start = YEARS[0]
    end = YEARS[-1]

    clim_slice = slice(str(first_year_clim), str(last_year_clim))

    chunks = {"time": -1, "latitude": 25, "longitude": 3}

    files = read.era_get_files(ERA5_PATH, variable, start_year=start, end_year=end)

    ds = read.open_files(files, preprocess=preprocess)

    if variable == "2t": # Rename some variables due to annoying ERA5 inconsistency
        variable = "t2m"
    elif variable == "100u":
        variable = "u100"
    elif variable == "100v":
        variable = "v100"
    else:
        pass

    ds_chunked = ds[variable].chunk(chunks)

    # Daily mean
    ds_chunked = ds_chunked.coarsen(time=24).mean() # Hourly to daily mean
    ds_chunked["time"] = ds_chunked["time"].dt.floor("D")

    # Anomalies
    ds_anoms = utils.doy_anom(ds_chunked, clim=clim_slice)
    
    ds_anoms = ds_anoms.chunk({"time": -1, "latitude": -1, "longitude": 8})
    ds = ds_chunked.chunk({"time": -1, "latitude": -1, "longitude": 8})
    
    ds = ds.to_dataset(name=variable)
    ds[variable+"_anom"] = ds_anoms

    encoding = read.specify_encoding(ds)
    
    read.write_zarr(
        ds,
        ERA5_WRITE_PATH,
        variable+"_daily_dayofyear_anoms_"+str(YEARS[0]) + '-' + str(YEARS[-1])+".zarr",
        encoding
    )
