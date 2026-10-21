
#!/usr/bin/env python
# coding: utf-8

# =================================
# Compute daily mean BARRA-R2 100m capacity factor
# =================================

import xarray as xr
import numpy as np

import src.regions as regions
import src.read as read
import src.wind as wind

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

BARRA_R2_PATH = "/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/"
WRITE_PATH = "/g/data/ng72/dr6273/work/projects/wind_drought/data/BARRA-R2/"
REGION = regions.get_Aus_boundary()
YEARS = range(1979, 2026)

if __name__ == "__main__":
    from dask_setup import setup_dask_client 
    
    client, cluster, dask_tmp = setup_dask_client(mode="interactive", workload_type="cpu")   # heavy compute

    chunks = {"time": 24 * 7 * 2, "lat": -1, "lon": -1}

    for i, year in enumerate(YEARS[:]):
    
        start = year * 100 + 1
        end = year * 100 + 12
    
        u100_files = read.barra_get_files(BARRA_R2_PATH, 'ua100m', start=start, end=end)
        v100_files = read.barra_get_files(BARRA_R2_PATH, 'va100m', start=start, end=end)
    
        u100 = read.open_files(u100_files, preprocess=barraR2_preprocess)
        v100 = read.open_files(v100_files, preprocess=barraR2_preprocess)
        
        u100_chunked = u100.chunk(chunks)
        v100_chunked = v100.chunk(chunks)
        
        u100_chunked = u100_chunked["ua100m"]
        v100_chunked = v100_chunked["va100m"]
        
        capacity_factor = wind.capacity_factor(np.hypot(u100_chunked, v100_chunked))
    
        capacity_factor = capacity_factor.coarsen(time=24).mean() # Hourly to daily mean
        capacity_factor["time"] = capacity_factor["time"].dt.floor("D")
        
        capacity_factor = capacity_factor.chunk({"time": -1, "lat": -1, "lon": -1})
        
        capacity_factor = capacity_factor.to_dataset(name="capacity_factor")
        
        capacity_factor_encoding = read.specify_encoding(capacity_factor)
        
        if i == 0:
            read.write_zarr(
                capacity_factor,
                WRITE_PATH,
                'barra-r2_capacity_factor_100m_'+str(YEARS[0]) + '-' + str(YEARS[-1]) + '.zarr',
                capacity_factor_encoding,
            )
        else:
            read.write_zarr(
                capacity_factor,
                WRITE_PATH,
                'barra-r2_capacity_factor_100m_'+str(YEARS[0]) + '-' + str(YEARS[-1]) + '.zarr',
                None,
                mode="a",
                append_dim="time"
            )