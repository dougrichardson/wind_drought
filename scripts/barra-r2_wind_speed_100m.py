
#!/usr/bin/env python
# coding: utf-8

# =================================
# Compute daily mean BARRA-R2 100m wind speed
# =================================

import xarray as xr
import numpy as np
import argparse

import src.regions as regions
import src.read as read

def barraR2_preprocess(ds):
    """
    Preprocess function for open_mfdataset.
    Selects Australian region, computes daily mean and renames coords.
    """
    ds = ds.sel(
        lon=slice(REGION[0], REGION[1]),
        lat=slice(REGION[3], REGION[2])
    )
    return ds

BARRA_R2_PATH = "/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/day/" # daily mean
WRITE_PATH = "/g/data/ng72/dr6273/work/projects/wind_drought/data/BARRA-R2/"
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

    start = YEARS[0] * 100 + 1
    end = YEARS[-1] * 100 + 12
    
    u100_files = read.barra_get_files(BARRA_R2_PATH, 'ua100m', start=start, end=end)
    v100_files = read.barra_get_files(BARRA_R2_PATH, 'va100m', start=start, end=end)

    u100 = read.open_files(u100_files, preprocess=barraR2_preprocess).drop_vars('time_bnds')
    v100 = read.open_files(v100_files, preprocess=barraR2_preprocess).drop_vars('time_bnds')
    
    u100_chunked = u100.chunk({"time": -1, "lat": -1, "lon": 5})
    v100_chunked = v100.chunk({"time": -1, "lat": -1, "lon": 5})
    
    u100_chunked = u100_chunked["ua100m"]
    v100_chunked = v100_chunked["va100m"]
    
    wind_speed = np.sqrt(u100_chunked ** 2 + v100_chunked ** 2)
    wind_speed = wind_speed.chunk({"time": -1, "lat": -1, "lon": 5})
    
    wind = wind_speed.to_dataset(name="wind_speed")
    
    wind_encoding = read.specify_encoding(wind)
    
    read.write_zarr(
        wind,
        WRITE_PATH,
        'barra-r2_wind_speed_100m_'+str(YEARS[0]) + '-' + str(YEARS[-1]) + '.zarr',
        wind_encoding
    )