#!/usr/bin/env python
# coding: utf-8

# =================================
# Compute ERA5 diagnostics
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

def l850_preprocess(ds):
    """ Return 850 hPa level and smaller region """
    ds = ds.sel(
        lon=slice(REGION[0], REGION[1]),
        lat=slice(REGION[3], REGION[2]),
        level=850
    )
    return ds

def l500_preprocess(ds):
    """ Return 500 hPa level and smaller region """
    ds = ds.sel(
        lon=slice(REGION[0], REGION[1]),
        lat=slice(REGION[3], REGION[2]),
        level=500
    )
    return ds

def l300_preprocess(ds):
    """ Return 300 hPa level and smaller region """
    ds = ds.sel(
        lon=slice(REGION[0], REGION[1]),
        lat=slice(REGION[3], REGION[2]),
        level=300
    )
    return ds

ERA5_PATH = "/g/data/if69/ls3248/data/ERA5/daily/"
ERA5_WRITE_PATH = "/g/data/ng72/dr6273/work/projects/wind_drought/data/ERA5/"
REGION = regions.get_Aus_boundary()

if __name__ == "__main__":
    from dask_setup import setup_dask_client
    client, cluster, dask_tmp = setup_dask_client(mode="interactive", workload_type="cpu")   # heavy compute

    parser = argparse.ArgumentParser()
    parser.add_argument("first_year", type=int)
    parser.add_argument("last_year", type=int)
    parser.add_argument("variable", type=str)
    parser.add_argument("level", type=int)
    parser.add_argument("first_year_clim", type=int)
    parser.add_argument("last_year_clim", type=int)

    args = parser.parse_args()
    
    first_year = args.first_year
    last_year = args.last_year
    variable = args.variable
    level = args.level
    first_year_clim = args.first_year_clim
    last_year_clim = args.last_year_clim
    
    YEARS = range(first_year, last_year + 1)
    start = YEARS[0]
    end = YEARS[-1]

    clim_slice = slice(str(first_year_clim), str(last_year_clim))

    chunks = {"time": -1, "lat": -1, "lon": 30}

    files = read.era_1deg_get_files(ERA5_PATH, variable, start_year=start, end_year=end)
    
    if level == 850:
        ds = read.open_files(files, preprocess=l850_preprocess).drop_vars("time_bnds")
    elif level == 500:
        ds = read.open_files(files, preprocess=l500_preprocess).drop_vars("time_bnds")
    elif level == 300:
        ds = read.open_files(files, preprocess=l300_preprocess).drop_vars("time_bnds")
    else:
        raise ValueError("Level not implemented")

    ds_chunked = ds[variable].chunk(chunks)
    ds_anoms = utils.doy_anom(ds_chunked, clim=clim_slice)
    ds_anoms = ds_anoms.chunk(chunks)
    
    ds = ds_chunked.to_dataset(name=variable)
    ds[variable+"_anom"] = ds_anoms

    encoding = read.specify_encoding(ds)
    read.write_zarr(
        ds,
        ERA5_WRITE_PATH,
        variable+str(level)+"_dayofyear_anoms_"+str(YEARS[0]) + '-' + str(YEARS[-1])+".zarr",
        encoding
    )
