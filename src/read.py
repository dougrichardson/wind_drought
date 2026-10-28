# Functions to read and process data
from pathlib import Path
import glob
from zarr.codecs import BloscCodec
import xarray as xr

def era_get_files(path, var, start_year=1979, end_year=2025):
    """Return files for years from start_year to end_year, inclusive."""
    files = (
        Path(path) / var
    ).glob("*/*.nc")

    return sorted(
        f for f in files
        if f.parent.name.isdigit()
        and start_year <= int(f.parent.name) <= end_year
    )

def barra_get_files(path, var, start=197901, end=202512):
    """ Return files between start and end month """
    files = sorted(Path(path).glob(
        var + "/latest/*.nc"
    ))
    
    files = [
        f for f in files
        if start <= int(f.stem[-13:-7]) <= end
    ]
    
    return files

def open_files(files, preprocess):
    """
    Load and preprocess hourly data for a given year
    """
    # Open all hours in the year 
    ds = xr.open_mfdataset(
        files,
        preprocess=preprocess,
        chunks='auto',
        parallel=True
    )
    return ds

def specify_encoding(ds):
    codec = BloscCodec(cname="zstd", clevel=6, shuffle="bitshuffle")

    encoding = {}
    for name, da in ds.data_vars.items():
        chunks = tuple(da.chunksizes[dim][0] for dim in da.dims)
        encoding[name] = {
            "chunks": chunks,
            "compressors": codec,
            "dtype": "float32",
        }

    return encoding

def write_zarr(ds, path, filename, encoding, mode='w', append_dim=None):
    """ Write to zarr """
    ds.to_zarr(
        path + filename,
        mode=mode,
        append_dim=append_dim,
        encoding=encoding,
        zarr_format=3,
        consolidated=False
    )

def load_wind(dataset, variable, region, scenario=None, averaging="equal"):
    """ Return da of wind speed data for given criteria """
    wind_path = "/g/data/ng72/dr6273/work/projects/wind_drought/data/"+dataset+"/power_grid/"
    directory_path = Path(wind_path)

    # Filter by variable
    matching_files = [f.name for f in directory_path.iterdir() if f.is_file() and variable in f.name]

    # Filter by region
    matching_files = [f for f in matching_files if region in f]

    # Filter by scenario
    if scenario is not None:
        matching_files = [f for f in matching_files if scenario in f]

    # Filter by averaging method
    matching_files = [f for f in matching_files if averaging in f]
    
    if len(matching_files) > 1:
        raise ValueError("Need to filter more files")
        
    return xr.open_dataset(wind_path + matching_files[0])