import numpy as np
import xarray as xr
import numba


@numba.guvectorize(
    [
        "void(float64[:], float64[:], float64[:])",
        "void(float32[:], float32[:], float32[:])",
        "void(int64[:], int64[:], int64[:])",
        "void(int32[:], int32[:], int32[:])",
    ],
    "(n),(m)->(m)",
    cache=True,
    target="parallelize"
)
def hist_ufunc(data, bin_edges, res):
    m = len(bin_edges)
    res[:] = 0
    aux, _ = np.histogram(data, bins=bin_edges)
    for i in numba.prange(m-1):
        res[i] = aux[i]

def histogram(da, dims, bin_edges=None):
    if bin_edges is None:
        bin_edges = np.histogram_bin_edges(da.values.flatten())
    elif isinstance(bin_edges, (str, int)):
        bin_edges = np.histogram_bin_edges(da.values.flatten(), bins=bin_edges)
    histograms = xr.apply_ufunc(
        hist_ufunc.stack(__hist__=dims),
        da.stack(),
        bin_edges,
        input_core_dims=[["__hist__"], []],
        output_core_dims=[["bin"]],
        kwargs={"axis": -1}
    )
    histograms = histograms.isel({"bin": slice(stop=-1)}).assign_bins(
        left_edges=("bin", bin_edges[:-1]),
        right_edges=("bin", bin_edges[1:])
    )
    return histograms

@numba.guvectorize(
    [
        "void(int64[:], int64[:], int64[:])",
        "void(int32[:], int32[:], int32[:])",
    ],
    "(m),(n)->(n)",
    cache=True
)
def bincount_ufunc(x, dummy, res):
    m = len(dummy)
    n = len(x)
    for j in range(m):
        res[j] = 0
    for i in range(n):
        res[x[i]-1] += 1

def bincount(da, dims, minlength=None):
    if minlength is None:
        minlength = da.max().item()
    bins = np.arange(minlength)
    bincounts = xr.apply_ufunc(
        bincount_ufunc,
        da.stack(__hist__=dims),
        bins,
        input_core_dims=[["__hist__"], []],
        output_core_dims=[["bin"]],
        kwargs={"axis": -1}
    )
    bincounts = bincounts.assign_coords({"bin": ("bin", bins)})
    return bincounts
