#!/usr/bin/env python
# coding: utf-8

# =================================
# Compute BARRA-R2 diagnostics
# =================================
import argparse
from pathlib import Path
import glob
from zarr.codecs import BloscCodec
import xarray as xr
import numpy as np

import src.read as read
import src.utils as utils

def barraR2_preprocess(ds):
    """ Return smaller region """
    ds = ds.sel(
        lon=slice(REGION[0], REGION[1]),
        lat=slice(REGION[3], REGION[2])
    )
    return ds

BARRA_R2_PATH = "/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/day/"
BARRA_R2_WRITE_PATH = "/g/data/ng72/dr6273/work/projects/wind_drought/data/BARRA-R2/"
REGION = regions.get_Aus_boundary()

if __name__ == "__main__":
    from dask_setup import setup_dask_client
    client, cluster, dask_tmp = setup_dask_client(mode="interactive", workload_type="cpu")   # heavy compute

    parser = argparse.ArgumentParser()
    parser.add_argument("first_year", type=int)
    parser.add_argument("last_year", type=int)
    parser.add_argument("variable", type=str)
    args = parser.parse_args()

    YEARS = range(args.first_year, args.last_year + 1) 
    start = YEARS[0] * 100 + 1
    end = YEARS[-1] * 100 + 12

    chunks = {"time": -1, "lat": -1, "lon": 5}

    if variable in ["tas", "rsds"]:

        files = read.barra_get_files(BARRA_R2_PATH, variable, start=start, end=end)
    
        if variable == "tas":
            ds = read.open_barra(files).drop_vars("time_bnds")
        elif variable == "rsds":
            ds = read.open_barra(files).drop_vars("time_bnds").drop_dims("bnds")
        else:
            raise ValueError("specified variable not valid")
    
        ds_chunked = ds.chunk(chunks)
        ds_anoms = utils.doy_anom(ds_chunked)
        ds_anoms = ds_anoms.chunk(chunks)
        encoding = read.specify_encoding(ds_anoms)
    
        read.write_zarr(
            ds_anoms,
            BARRA_R2_WRITE_PATH,
            variable+"_dayofyear_anoms_"+str(YEARS[0]) + '-' + str(YEARS[-1])+".zarr",
            encoding,
        )

    elif variable == "wind_sfc":

        uas_files = read.barra_get_files(BARRA_R2_PATH, "uas", start=start, end=end)
        vas_files = read.barra_get_files(BARRA_R2_PATH, "vas", start=start, end=end)

        uas = read.open_barra(uas_files).drop_vars('time_bnds')
        vas = read.open_barra(vas_files).drop_vars('time_bnds')

        uas_chunked = uas.chunk(chunks)["uas"]
        vas_chunked = vas.chunk(chunks)["vas"]

        speed = np.sqrt(uas_chunked ** 2 + vas_chunked ** 2)
        direction = np.mod(180 + (180 / np.pi) * np.arctan2(uas_chunked, vas_chunked), 360)
        direction = direction.rename("wind_direction").drop_attrs()

        wind = speed.to_dataset(name="wind_speed")
        wind["wind_direction"] = wind_dir

        # Function used on other variables not working here so change the encoding script
        wind_encoding = {}
        
        for name, da in wind.data_vars.items():
            wind_encoding[name] = {
                "chunks": tuple(
                    chunk[0] for chunk in da.data.chunks
                ),
                "dtype": da.dtype,
                "compressors": BloscCodec(
                    cname="zstd",
                    clevel=6,
                    shuffle="bitshuffle",
                ),
            }
            
        read.write_zarr(
            wind,
            BARRA_R2_WRITE_PATH,
            "wind_sfc_speed_direction_"+str(YEARS[0]) + '-' + str(YEARS[-1])+".zarr",
            wind_encoding
        )
        
    else:
        raise ValueError("specified variable not valid")