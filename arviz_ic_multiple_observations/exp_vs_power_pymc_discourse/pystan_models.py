import arviz as az
import xarray as xr

from generate_data import generate_data
from utils import StanModel_cache

n = 200
Years_indiv, Mean_RT_comp_Indiv, Mean_RT_incomp_Indiv = generate_data(7, n)
dims = {"y_obs_comp": ["subject"], "y_obs_incomp": ["subject"]}
log_lik_dict = {"y_obs_comp": "log_lik_comp", "y_obs_incomp": "log_lik_comp"}
data = {
    "n": n,
    "y_obs_comp": Mean_RT_comp_Indiv,
    "y_obs_incomp": Mean_RT_incomp_Indiv,
    "age": Years_indiv,
    "mean_rt_c": Mean_RT_comp_Indiv.mean(),
    "mean_rt_i": Mean_RT_incomp_Indiv.mean(),
}

exp_code = """
data {
    int<lower=0> n;
    real y_obs_comp[n];
    real y_obs_incomp[n];
    int<lower=0> age[n];
    real mean_rt_c;
    real mean_rt_i;
}

parameters {
    real b;
    real<lower=0> sigma;
    real<lower=0> a_c;
    real<lower=0> a_i;
    real g_c;
    real g_i;
}

transformed parameters {
    real mu_c[n];
    real mu_i[n];
    for (j in 1:n) {
        mu_c[j] = a_c*exp(-b*age[j]) + g_c;
        mu_i[j] = a_i*exp(-b*age[j]) + g_i;
    }
}

model {
    a_c ~ cauchy(0, 5);
    a_i ~ cauchy(0, 5);
    b ~ normal(1, 1);
    g_c ~ normal(mean_rt_c, .5);
    g_i ~ normal(mean_rt_i, .5);
    sigma ~ normal(0, .2);
    y_obs_comp ~ normal(mu_c, sigma);
    y_obs_incomp ~ normal(mu_i, sigma);
}

generated quantities {
    real log_lik_comp[n];
    real log_lik_incomp[n];

    for (j in 1:n) {
        log_lik_comp[j] = normal_lpdf(y_obs_comp[j] | mu_c[j], sigma);
        log_lik_incomp[j] = normal_lpdf(y_obs_incomp[j] | mu_i[j], sigma);
    }
}
"""

stan_model = StanModel_cache(model_code=exp_code)
fit = stan_model.sampling(data=data, iter=4000, control={"adapt_delta" : 0.9})

idata_exp = az.from_pystan(fit, dims=dims, log_likelihood=log_lik_dict)

log_lik_exp = idata_exp.log_likelihood

print("\n\nLeave one *observation* out cross validation (whole model)")
condition_dim = xr.DataArray(["compatible", "incompatible"], name="condition")
idata_exp.sample_stats["log_likelihood"] = xr.concat((log_lik_exp.y_obs_comp, log_lik_exp.y_obs_incomp), dim=condition_dim)
print(az.loo(idata_exp), "\n")

print("\n\nLeave one *subject* out cross validation (whole model)")
idata_exp.sample_stats["log_likelihood"] = log_lik_exp.to_array().sum("variable")
print(az.loo(idata_exp), "\n")

print("\n\nLeave one observation out cross validation (y_obs_comp only)")
idata_exp.sample_stats["log_likelihood"] = log_lik_exp.y_obs_comp
print(az.loo(idata_exp), "\n")

print("\n\nLeave one observation out cross validation (y_obs_incomp only)")
idata_exp.sample_stats["log_likelihood"] = log_lik_exp.y_obs_incomp
print(az.loo(idata_exp), "\n")
