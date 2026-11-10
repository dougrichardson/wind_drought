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

ERA5_PATH = "/g/data/su28/ERA5/daily/"
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
    args = parser.parse_args()

    YEARS = range(args.first_year, args.last_year + 1) 
    start = YEARS[0]
    end = YEARS[-1]

    chunks = {"time": -1, "lat": -1, "lon": 30}

    files = read.era_get_files(ERA5_PATH, variable, start=start, end=end)

    if level == 850:
        ds = read.open_files(files, preprocess=l850_preprocess).drop_vars("time_bnds")
    elif level == 500:
        ds = read.open_files(files, preprocess=l500_preprocess).drop_vars("time_bnds")
    elif level == 300:
        ds = read.open_files(files, preprocess=l300_preprocess).drop_vars("time_bnds")
    else:
        raise ValueError("Level not implemented")

    ds_chunked = ds[variable].chunk(chunks)
    ds_anoms = utils.doy_anom(ds_anoms)
    ds_anoms = ds_anoms.chunk(chunks)
    
    ds = ds_chunked.to_dataset(name=variable)
    ds[variable+"_anom"] = ds_anoms

    encoding = read.specify_encoding(ds)
    read.write_zarr(
        ds,
        ERA5_WRITE_PATH,
        variable+level+"_dayofyear_anoms_"+str(YEARS[0]) + '-' + str(YEARS[-1])+".zarr",
        encoding
    )
