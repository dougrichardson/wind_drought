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

def map_leaves_pair(obj1, obj2, operation):
    """Apply operation to each value of a dictionary, with another dictionary as the second argument,
        and return in same structure as input"""
    if isinstance(obj1, dict) and isinstance(obj2, dict):
        if obj1.keys() != obj2.keys():
            raise ValueError("The dictionaries must have matching keys")

        return {
            key: map_leaves_pair(obj1[key], obj2[key], operation)
            for key in obj1
        }

    if isinstance(obj1, dict) or isinstance(obj2, dict):
        raise TypeError("The dictionaries must have matching nested structures")

    return operation(obj1, obj2)