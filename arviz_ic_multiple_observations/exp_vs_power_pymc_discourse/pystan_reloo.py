import arviz as az
import numpy as np

import sys
from io import StringIO

# setup the environment
backup = sys.stdout

# ####
output = StringIO()
sys.stdout = output

from generate_data import generate_data
from utils import StanModel_cache

n = 100
Years_indiv, Mean_RT_comp_Indiv, Mean_RT_incomp_Indiv = generate_data(7, n)
dims = {"y_obs_comp": ["subject"], "y_obs_incomp": ["subject"]}
log_lik_dict = ["log_lik", "log_lik_ex"]
data = {
    "n": n,
    "y_obs_comp": Mean_RT_comp_Indiv,
    "y_obs_incomp": Mean_RT_incomp_Indiv,
    "age": Years_indiv,
    "mean_rt_c": Mean_RT_comp_Indiv.mean(),
    "mean_rt_i": Mean_RT_incomp_Indiv.mean(),
    "n_ex": 0,
    "age_ex": np.array([], dtype=int),
    "y_obs_comp_ex": [],
    "y_obs_incomp_ex": [],
}

loo_subject_code = """
data {
    int<lower=0> n;
    real y_obs_comp[n];
    real y_obs_incomp[n];
    int<lower=0> age[n];
    real mean_rt_c;
    real mean_rt_i;
    // excluded data
    int<lower=0> n_ex;
    real y_obs_comp_ex[n_ex];
    real y_obs_incomp_ex[n_ex];
    int<lower=0> age_ex[n_ex];
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
    real mu_c_ex[n_ex];
    real mu_i_ex[n_ex];

    for (j in 1:n) {
        mu_c[j] = a_c*exp(-b*age[j]) + g_c;
        mu_i[j] = a_i*exp(-b*age[j]) + g_i;
    }
    for (i in 1:n_ex) {
        mu_c_ex[i] = a_c*exp(-b*age_ex[i]) + g_c;
        mu_i_ex[i] = a_i*exp(-b*age_ex[i]) + g_i;
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
    real log_lik[n];
    real log_lik_ex[n_ex];

    for (j in 1:n) {
        log_lik[j] = normal_lpdf(y_obs_comp[j] | mu_c[j], sigma) + normal_lpdf(y_obs_incomp[j] | mu_i[j], sigma);
    }
    for (i in 1:n_ex) {
        log_lik_ex[i] = normal_lpdf(y_obs_comp_ex[i] | mu_c_ex[i], sigma) + normal_lpdf(y_obs_incomp_ex[i] | mu_i_ex[i], sigma);
    }
}
"""

stan_model = StanModel_cache(model_code=loo_subject_code)
fit_kwargs = dict(iter=4000, control={"adapt_delta" : 0.9})
fit = stan_model.sampling(data=data, **fit_kwargs)

idata_kwargs = dict(
    observed_data=["y_obs_comp", "y_obs_incomp"],
    constant_data=["age"],
    dims=dims,
    log_likelihood=log_lik_dict
)
idata_exp = az.from_pystan(fit, **idata_kwargs)

class ExpWrapper(az.PyStanSamplingWrapper):

    def sel_observations(self, idx):
        age = self.idata_orig.constant_data.age.values
        y_c = self.idata_orig.observed_data.y_obs_comp.values
        y_i = self.idata_orig.observed_data.y_obs_incomp.values
        mask = np.full_like(age, True, dtype=bool)
        mask[idx] = False
        n_obs = np.sum(mask)
        n_ex = np.sum(~mask)
        observations = {
            "n": n_obs,
            "age": age[mask],
            "y_obs_comp": y_c[mask],
            "y_obs_incomp": y_i[mask],
            "mean_rt_c": y_c[mask].mean(),
            "mean_rt_i": y_i[mask].mean(),
            "n_ex": n_ex,
            "age_ex": age[~mask],
            "y_obs_comp_ex": y_c[~mask],
            "y_obs_incomp_ex": y_i[~mask]
        }
        return observations, "log_lik_ex"

sys.stdout = backup
print("\n\n(PSIS) Leave one *subject* out cross validation (whole model)")
idata_exp.sample_stats["log_likelihood"] = idata_exp.log_likelihood.log_lik
loo_psis = az.loo(idata_exp, pointwise=True)
print(loo_psis, "\n")

print("\n\n(exact) Leave one *subject* out cross validation (whole model)")
sys.stdout = output
loo_psis.pareto_k[:] = 1.2
exp_wrapper = ExpWrapper(
    stan_model,
    idata_orig=idata_exp,
    sample_kwargs=fit_kwargs,
    idata_kwargs=idata_kwargs
)
loo_exact = az.reloo(exp_wrapper, loo_orig=loo_psis)
sys.stdout = backup
print(loo_exact, "\n")
