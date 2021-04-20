import numpy as np
import xarray as xr
import numba

from .numpy_linalg import _find_square_matrix_dims, _check_matrix_dims

# qr doesn't work on (..., M, N) shapes, so we use numba for that
@numba.guvectorize(
    [
        "void(float64[:], float64[:], float64[:])",
        "void(float32[:], float32[:], float32[:])",
        "void(int64[:], int64[:], int64[:])",
        "void(int32[:], int32[:], int32[:])",
    ],
    "(m,n)->(m,n),(n,n)",
    cache=True,
    target="parallelize"
)
def qr_ufunc_mgtn(a, q, r):
    q, r = np.linalg.qr(a)

@numba.guvectorize(
    [
        "void(float64[:], float64[:], float64[:])",
        "void(float32[:], float32[:], float32[:])",
        "void(int64[:], int64[:], int64[:])",
        "void(int32[:], int32[:], int32[:])",
    ],
    "(m,n)->(m,m),(m,n)",
    cache=True,
    target="parallelize"
)
def qr_ufunc_ngtm(a, q, r):
    q, r = np.linalg.qr(a)

def qr(da, dims):
    if dims is None:
        dims = _find_square_matrix_dims(da)
    _check_matrix_dims(da, dims)
    dim_lengths = {
        dim: length
        for length, dim in zip(da.shape, da.dims)
        if dim in dims
    }
    if dim_lengths[dims[0]] >= dim_lengths[dims[1]]:
        return xr.apply_ufunc(
            qr_ufunc_mgtn,
            da,
            input_core_dims=[dims],
            output_core_dims=[dims, [dims[1], dims[1]]]
        )
    return xr.apply_ufunc(
        qr_ufunc_ngtm,
        da,
        input_core_dims=[dims],
        output_core_dims=[[dims[0], dims[0]], dims]
    )

