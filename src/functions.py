import xarray as xr
import numpy as np
import regionmask

def get_Aus_boundary():
    """
    Returns a list of bounding coordinates for Australia.
    """
    return [110, 155, -10, -45]

def windspeed(u, v):
    """
    Compute windspeed from u and v
    
    u: array of zonal wind
    v: array of meridional wind
    """
    return np.sqrt(u ** 2 + v ** 2)

def sel_month(ds, month):
    """
    Return array for specified month
    
    ds: dataset to select from
    month: int or list of int between 1 and 12, default is None
    """
    if month is None:
        return ds
    elif isinstance(month, int):
        if 1 <= month <= 12:
            return ds.isel(time=ds.time.dt.month == month)
        else:
            raise ValueError("Incorrect month specified.")
    elif isinstance(month, list):
        if all((isinstance(i, int)) & (1 <= i <= 12) for i in month):
            return ds.isel(time=ds.time.dt.month.isin(month))
        else:
            raise ValueError("Incorrect month specified.")

def create_mask(gpd_df, template_ds, boundary=None, lon_name='lon', lat_name='lat'):
    """
    Create mask from shapefiles and a template xarray dataArray or dataset.
    """
    mask = regionmask.mask_3D_geopandas(
        gpd_df,
        template_ds[lon_name],
        template_ds[lat_name]
    )
    
    if lon_name != 'lon':
        mask = mask.rename({lon_name: 'lon'})
    if lat_name != 'lat':
        mask = mask.rename({lat_name: 'lat'})
        
    if isinstance(boundary, list):
        mask = mask.sel(
            lon=slice(boundary[0], boundary[1]),
            lat=slice(boundary[2], boundary[3])
        )
        
    return mask

def load_reanalysis_wind_speed(fp):
    """
    Return dictionary of ERA5 and BARRA-R2 wind speeds for different scenarios
    """
    ws_dict = {}
    for data_name in ["ERA5", "BARRA-R2"]:
        if data_name == "ERA5":
            var_name = "ws10m"
        else:
            var_name = "wss"
        
        for grid in ["NEM", "SWIS", "NWIS"]:
            
            if grid == "NEM":
                
                for scenario in ["wind_2025", "wind_2030", "wind_2040", "wind_2050"]:
                    
                    for subgrid in ["NEM", "SE"]:
                        
                        for unweight in [True, False]:
                            
                            fname = "grid_mean_"+var_name+"_"+data_name+"_"+grid+"_"+scenario+"_"+subgrid+"_"
                            key = data_name+"_"+grid+"_"+subgrid+"_"+scenario+"_"
                            
                            if unweight:
                                fname = fname+"unweighted"
                                key = key+"unweighted"
                            else:
                                fname = fname+"weighted"
                                key = key+"weighted"
                            
                            ws_dict[key] = xr.open_dataarray(fp+fname+".nc")
            else:
                fname = "grid_mean_"+var_name+"_"+data_name+"_"+grid
                key = data_name+"_"+grid
                
                ws_dict[key] = xr.open_dataarray(fp+fname+".nc")
    return ws_dict