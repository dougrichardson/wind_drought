#!/usr/bin/env python
# coding: utf-8

# =================================
# Generate masks for grids
# Takes about 15 minutes on a normalbw queue, 4 CPUs, 18GB memory
# =================================
import numpy as np
import geopandas
import xarray as xr
import regionmask

def prepare_regions(gdf, name_col):
    """Prepare a GeoDataFrame and regionmask Regions object."""
    gdf = gdf.to_crs("EPSG:4326").copy()
    gdf["_region_id"] = np.arange(len(gdf))

    regions = regionmask.from_geopandas(
        gdf,
        numbers="_region_id",
        names=name_col,
    )
    return gdf, regions

def align_orog_to_wind(wind_da, orog, lat_name, lon_name):
    land = orog.squeeze(drop=True)

    # Rename orography coordinates to match the wind data.
    rename = {}
    for target, aliases in (
        (lat_name, ("lat", "latitude")),
        (lon_name, ("lon", "longitude")),
    ):
        if target not in land.coords:
            source = next(n for n in aliases if n in land.coords)
            rename[source] = target
    if rename:
        land = land.rename(rename)

    # Convert longitude convention if one grid uses -180..180
    # and the other uses 0..360.
    wind_lon = wind_da[lon_name]
    land_lon = land[lon_name]

    if float(land_lon.min()) < 0 and float(wind_lon.min()) >= 0:
        land = land.assign_coords({lon_name: land_lon % 360})
    elif float(land_lon.max()) > 180 and float(wind_lon.max()) <= 180:
        land = land.assign_coords({lon_name: ((land_lon + 180) % 360) - 180})

    # Interpolate the mask onto the wind grid; nearest preserves its
    # land/ocean classification.
    land = land.sortby(lat_name).sortby(lon_name)
    return land.interp(
        {
            lat_name: wind_da[lat_name],
            lon_name: wind_da[lon_name],
        },
        method="nearest",
    )

def regional_wind_means(wind_da, regions, orog=None):
    """Return an area-weighted time series for each polygon."""
    lat_name = next(n for n in ("lat", "latitude") if n in wind_da.coords)
    lon_name = next(n for n in ("lon", "longitude") if n in wind_da.coords)

    if orog is not None:
        land = align_orog_to_wind(wind_da, orog, lat_name, lon_name)
        wind_da = wind_da.where(land > 0)

    fractions = regions.mask_3D_frac_approx(
        wind_da[lon_name],
        wind_da[lat_name],
        drop=False,
    )

    # Approximate cell-area weights for a regular lon/lat grid.
    weights = fractions * np.cos(np.deg2rad(wind_da[lat_name]))

    spatial_dims = tuple(dict.fromkeys(
        wind_da[lat_name].dims + wind_da[lon_name].dims
    ))

    return wind_da.weighted(weights).mean(spatial_dims)

WIND_PATH = "/g/data/ng72/dr6273/work/projects/wind_drought/data/"

if __name__ == "__main__":

    for variable in ["wind_speed", "capacity_factor"]:

        for dataset in ["ERA5", "BARRA-R2"]:
            wind_fp = dataset + "/" + dataset.lower() + "_" + variable + "_100m_1979-2025.zarr/"
            
            # Open wind data
            wind_da = xr.open_zarr(
                WIND_PATH + wind_fp,
                consolidated=False
            )[variable]

            # ====== NEM ========
            # GeoDataFrame for NEM
            _nem_gdf = geopandas.read_file(
                "data/resource_with_indicative_REZ_boundaries_2026_gis_data.shp",
            ).rename(columns={"index": "REZ"})
        
            for scenario in ["SC30_wind", "SC40_wind", "SC50_wind"]:
                # exclude REZs with no capacity
                nem_gdf = _nem_gdf.loc[_nem_gdf[scenario].gt(0)].copy()

                # Full NEM and NEM minus QLD & TAS
                for nem_grid in ["NEM", "SE"]: 
                    if nem_grid == "SE":
                        # nem_gdf = nem_gdf.rename(columns={"index": "REZ"}).copy()
                        nem_gdf = nem_gdf.loc[
                            nem_gdf[scenario].gt(0)
                            & ~nem_gdf["REZ"].str.startswith(("Q", "T"))
                        ].copy()
        
                    # Prepare REZs
                    nem_gdf, nem_regions = prepare_regions(nem_gdf, name_col="REZ")
            
                    # REZ mean wind
                    nem_by_rez = regional_wind_means(wind_da, nem_regions, orog=None)
            
                    # Grid mean wind
                    nem_equal_mean = nem_by_rez.mean("region", skipna=True)
            
                    # Capacity mean wind
                    capacity = xr.DataArray(
                        nem_gdf.sort_values("_region_id")[scenario].to_numpy(),
                        dims="region",
                        coords={"region": nem_by_rez["region"]},
                    )
                    nem_capacity_mean = nem_by_rez.weighted(capacity).mean("region")
            
                    # Write
                    fy = str(nem_by_rez.time.dt.year[0].values)
                    ly = str(nem_by_rez.time.dt.year[-1].values)
                    fp = dataset.lower() + "_" + variable + "_100m_" + nem_grid + "-" + scenario.split("_")[0] + "_" + fy + "-" + ly
                    
                    nem_by_rez.to_netcdf(
                        WIND_PATH + dataset + "/power_grid/" + fp + "_by_rez.nc"
                    )
                    nem_equal_mean.to_netcdf(
                        WIND_PATH + dataset + "/power_grid/" + fp + "_equal.nc"
                    )
                    nem_capacity_mean.to_netcdf(
                        WIND_PATH + dataset + "/power_grid/" + fp + "_weighted.nc"
                    )

            # ====== SWIS & NWIS ========
            # Because of the buffer applied to the transmission lines in the shapefile, we need to use
            #  orography to remove ocean cells
            if dataset == "ERA5":
                orog = xr.open_dataset(
                    "/g/data/rt52/era5/single-levels/monthly-averaged/lsm/2025/lsm_era5_moda_sfc_20250101-20250131.nc"
                )["lsm"]
            elif dataset == "BARRA-R2":
                orog = xr.open_dataset(
                    "/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/fx/orog/latest/orog_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1.nc"
                )["orog"]
            else:
                raise ValueError("Incorrect orography dataset specified, currently accepts 'ERA5' or 'BARRA-R2'.")

            # Load GeoDataFrames
            swis_gdf = geopandas.read_file(
                "data/SWIS_transmission_region.shp"
            )
            nwis_gdf = geopandas.read_file(
                "data/NWIS_transmission_region.shp"
            )

            # Prepare regions
            swis_gdf, swis_regions = prepare_regions(swis_gdf, name_col="region")
            nwis_gdf, nwis_regions = prepare_regions(nwis_gdf, name_col="region")

            # Regional mean
            swis_mean = regional_wind_means(wind_da, swis_regions, orog=orog).isel(
                region=0, drop=True
            )
            nwis_mean = regional_wind_means(wind_da, nwis_regions, orog=orog).isel(
                region=0, drop=True
            )

            # Write
            swis_fp = dataset.lower() + "_" + variable + "_100m_SWIS_" + fy + "-" + ly
            swis_mean.to_netcdf(
                WIND_PATH + dataset + "/power_grid/" + swis_fp + "_equal.nc"
            )

            nwis_fp = dataset.lower() + "_" + variable + "_100m_NWIS_" + fy + "-" + ly
            nwis_mean.to_netcdf(
                WIND_PATH + dataset + "/power_grid/" + nwis_fp + "_equal.nc"
            )