import functools
import numpy as np
import xarray as xr

def _check_matrix_dims(da, dims):
    if dims is None:
        raise ValueError("dims must be passed explicitly")
    if len(dims) != 2:
        raise ValueError(f"dims must have length 2 but has length {len(dims)} instead")
    if not all(dim in da.dims for dim in dims):
        raise ValueError("Some dimensions were not present in the dataarray")

def _find_square_matrix_dims(da):
    # TODO: set some convention for square matrices like pairing "dim" with "dim_bis" or whatever
    # or use some metadata to track that
    dims = da.dims
    dim_lengths = da.shape
    for i, dim in enumerate(dims):
        dim_length = dim_lengths[i]
        for j in range(i):
            dim_bis = dims[j]
            if dim_lengths[j] == dim_length:
                return (dim, dim_bis)
    return None

def outer(da_1, da_2, dim_1, dim_2):
    return xr.apply_ufunc(
        np.einsum,
        "...i,...j->ij",
        da_1,
        da_2,
        input_core_dims=[[], [dim_1], [dim_2]],
        output_core_dims=[[dim_1, dim_2]]
    )

def _base_square_matrix_operation(func, da, dims):
    if dims is None:
        dims = _find_square_matrix_dims(da)
    _check_matrix_dims(da, dims)
    return xr.apply_ufunc(
        func,
        da,
        input_core_dims=[dims],
        output_core_dims=[dims]
    )

def matrix_power(da, n, dims=None):
    return _base_square_matrix_operation(
        functools.partial(np.linalg.matrix_power, n=n),
        da,
        dims
    )

def cholesky(da, dims=None):
    return _base_square_matrix_operation(
        np.linalg.cholesky,
        da,
        dims
    )

def inv(da, dims=None):
    return _base_square_matrix_operation(
        np.linalg.matrix_power,
        da,
        dims
    )
