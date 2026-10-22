#!/usr/bin/env python
# coding: utf-8

# =================================
# Compute daily mean ERA5 100m wind speed
# =================================

import xarray as xr
import numpy as np

import src.regions as regions
import src.read as read

def era_preprocess(ds):
    """
    Preprocess function for open_mfdataset.
    Selects Australian region, computes daily mean and renames coords.
    """
    ds = ds.sel(
        longitude=slice(REGION[0], REGION[1]),
        latitude=slice(REGION[2], REGION[3])
    )
    ds = ds.rename(
        {'longitude': 'lon',
         'latitude': 'lat'}
    )
    ds = ds.coarsen(time=24).mean()
    ds["time"] = ds["time"].dt.floor("D")
    return ds

# def get_files(path, var, start_year=1979, end_year=2025):
#     """Return files for years from start_year to end_year, inclusive."""
#     files = (
#         Path(path) / var
#     ).glob("*/*.nc")

#     return sorted(
#         f for f in files
#         if f.parent.name.isdigit()
#         and start_year <= int(f.parent.name) <= end_year
#     )

# def open_era(files):
#     """
#     Load and preprocess hourly data for a given year
#     """
#     # Open all hours in the year (~33 GB)
#     ds = xr.open_mfdataset(
#         files,
#         preprocess=era_preprocess,
#         chunks='auto',
#         parallel=True
#     )
#     return ds

# def specify_encoding(ds):
#     codec = BloscCodec(cname="zstd", clevel=6, shuffle="bitshuffle")

#     encoding = {}
#     for name, da in ds.data_vars.items():
#         chunks = tuple(da.chunksizes[dim][0] for dim in da.dims)
#         encoding[name] = {
#             "chunks": chunks,
#             "compressors": codec,
#             "dtype": "float64",
#         }

#     return encoding

# def write_zarr(ds, path, filename, encoding):
#     """ Write to zarr """
#     ds.to_zarr(
#         path + filename,
#         mode='w',
#         encoding=encoding,
#         zarr_format=3,
#         consolidated=False
#     )

ERA_PATH = "/g/data/rt52/era5/single-levels/reanalysis/" # hourly instantaneous
WRITE_PATH = "/g/data/ng72/dr6273/work/projects/wind_drought/data/ERA5/"
REGION = regions.get_Aus_boundary()
YEARS = range(1979, 2026)

if __name__ == "__main__":
    from dask_setup import setup_dask_client 
    
    client, cluster, dask_tmp = setup_dask_client(mode="interactive", workload_type="cpu")   # heavy compute

    u100_files = read.era_get_files(ERA_PATH, '100u', start_year=YEARS[0], end_year=YEARS[-1])
    v100_files = read.era_get_files(ERA_PATH, '100v', start_year=YEARS[0], end_year=YEARS[-1])

    u100 = read.open_files(u100_files, preprocess=era_preprocess)
    v100 = read.open_files(v100_files, preprocess=era_preprocess)
    
    u100_chunked = u100.chunk({"time": -1, "lat": -1, "lon": 16})
    v100_chunked = v100.chunk({"time": -1, "lat": -1, "lon": 16})
    
    u100_chunked = u100_chunked["u100"]
    v100_chunked = v100_chunked["v100"]
    
    wind_speed = np.sqrt(u100_chunked ** 2 + v100_chunked ** 2)
    wind_speed = wind_speed.chunk({"time": -1, "lat": -1, "lon": 16})
    
    wind = wind_speed.to_dataset(name="wind_speed")
    
    wind_encoding = read.specify_encoding(wind)
    
    read.write_zarr(
        wind,
        WRITE_PATH,
        'era5_wind_speed_100m_'+str(YEARS[0]) + '-' + str(YEARS[-1]) + '.zarr',
        wind_encoding
    )