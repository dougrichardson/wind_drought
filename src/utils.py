import xarray as xr

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

def map_leaves(obj, operation):
    """Apply operation to each value of a dictionary and return in same structure as input"""
    def _map(node):
        if isinstance(node, dict):
            return {key: _map(value) for key, value in node.items()}
        return operation(node)

    return _map(obj)