import numpy as np
import numba
import scipy
import xarray as xr
from xarray_einstats.einops import raw_rearrange
from xarray_einstats import stats

def _backtransform_ranks(da, c=3 / 8):  # pylint: disable=invalid-name
    """Backtransformation of ranks.

    Parameters
    ----------
    da : xr.DataArray
        Ranks array. It must have dimensions named `chain` and `draw`
    c : float
        Fractional offset. Defaults to c = 3/8 as recommended by Blom (1958).

    Returns
    -------
    xr.DataArray

    References
    ----------
    Blom, G. (1958). Statistical Estimates and Transformed Beta-Variables. Wiley; New York.
    """
    size = len(da.chain) * len(da.draw)
    return (da - c) / (size - 2 * c + 1)


def _z_scale(da, **kwargs):
    """Calculate z_scale.

    Parameters
    ----------
    da : xr.DataArray
        Input array. It must have dimensions named `chain` and `draw`

    Returns
    -------
    xr.DataArray
    """
    kwargs = kwargs.copy()
    if kwargs.get("dask", None) == "allowed":
        # scipy doesn't support dask properly, so even if we use dask="allowed" or dask directly
        # for everything else we need to force "parallelized" mode here
        kwargs["dask"] = "parallelized"
    rank = stats.rankdata(da, dims=("chain", "draw"), method="average", **kwargs)
    rank = _backtransform_ranks(rank)
    norm_dist = stats.XrContinuousRV(scipy.stats.norm)
    return norm_dist.ppf(rank, apply_kwargs=kwargs)


def _split_chains(da, **kwargs):
    """Split and stack chains."""
    half = len(da.draw) // 2
    if kwargs.get("dask", None) == "parallelized":
        # here we force dask="allowed" because einops doesn't play well with parallelized mode
        kwargs["dask"] = "allowed"
    if kwargs.get("dask", None) == "allowed":
        from xarray_einstats.einops import DaskBackend

    return raw_rearrange(
        da,
        "(d1 d2)=draw -> (chain d1)=c2 d2",
        d2=half,
        **kwargs
    ).rename(d2="draw", c2="chain")


def _z_fold(da, **kwargs):
    """Fold and z-scale values."""
    da = abs(da - da.median(("chain", "draw")))
    da = _z_scale(da, **kwargs)
    return da


def _rhat(da, **kwargs):
    """Compute the rhat for an n-D DataArray."""
    num_samples = len(da.draw)

    # Calculate chain mean
    chain_mean = da.mean("draw")
    # Calculate chain variance
    chain_var = da.var("draw", ddof=1)
    # Calculate between-chain variance
    between_chain_variance = num_samples * chain_mean.var("chain", ddof=1)
    # Calculate within-chain variance
    within_chain_variance = chain_var.mean("chain")
    # Estimate of marginal posterior variance
    rhat_value = np.sqrt(
        (between_chain_variance / within_chain_variance + num_samples - 1) / (num_samples)
    )
    return rhat_value


def _rhat_rank(da, **kwargs):
    """Compute the rank normalized rhat for an n-D DataArray.

    Computation follows https://arxiv.org/abs/1903.08008
    """
    split_da = _split_chains(da, **kwargs)
    rhat_bulk = _rhat(_z_scale(split_da, **kwargs))

    rhat_tail = _rhat(_z_fold(split_da, **kwargs))

    rhat_rank = xr.where(rhat_bulk > rhat_tail, rhat_bulk, rhat_tail)
    return rhat_rank

def rhat_einstats(ds, method="rank", **kwargs):
    func_map = {
        "identity": _rhat,
        "rank": _rhat_rank
    }
    if method not in func_map:
        raise ValueError("method not recognized")
    rhat_func = func_map[method]
    if isinstance(ds, xr.Dataset):
        return ds.map(rhat_func, **kwargs)
    if isinstance(ds, xr.DataArray):
        return rhat_func(ds, **kwargs)

@numba.guvectorize(
    [
        "void(float64[:,:], float64, float64[:])",
    ],
    "(n,m),()->()",
    cache=True,
    target="parallel",
    nopython=True
)
def geyer(acov, chain_mean_term, tau_hat):
    tau_hat[:] = 0
    n_draw = acov.shape[1]
    mean_var = np.mean(acov[:, 0]) * n_draw / (n_draw - 1.0)
    var_plus = mean_var * (n_draw - 1.0) / n_draw + chain_mean_term


    rho_hat_t = np.zeros(n_draw)
    rho_hat_even = 1.0
    rho_hat_t[0] = rho_hat_even
    rho_hat_odd = 1.0 - (mean_var - np.mean(acov[:, 1])) / var_plus
    rho_hat_t[1] = rho_hat_odd

    # Geyer's initial positive sequence
    t = 1
    while t < (n_draw - 3) and (rho_hat_even + rho_hat_odd) > 0.0:
        rho_hat_even = 1.0 - (mean_var - np.mean(acov[:, t + 1])) / var_plus
        rho_hat_odd = 1.0 - (mean_var - np.mean(acov[:, t + 2])) / var_plus
        if (rho_hat_even + rho_hat_odd) >= 0:
            rho_hat_t[t + 1] = rho_hat_even
            rho_hat_t[t + 2] = rho_hat_odd
        t += 2

    max_t = t - 2
    # improve estimation
    if rho_hat_even > 0:
        rho_hat_t[max_t + 1] = rho_hat_even
    # Geyer's initial monotone sequence
    t = 1
    while t <= max_t - 2:
        if (rho_hat_t[t + 1] + rho_hat_t[t + 2]) > (rho_hat_t[t - 1] + rho_hat_t[t]):
            rho_hat_t[t + 1] = (rho_hat_t[t - 1] + rho_hat_t[t]) / 2.0
            rho_hat_t[t + 2] = rho_hat_t[t + 1]
        t += 2

    tau_hat += -1.0 + 2.0 * np.sum(rho_hat_t[: max_t + 1]) + np.sum(rho_hat_t[max_t + 1 : max_t + 2])
