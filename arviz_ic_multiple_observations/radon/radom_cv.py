import arviz as az

idata = az.load_arviz_data("radon")
log_lik = idata.sample_stats[["log_likelihood"]]

loo_obs = az.loo(idata, pointwise=True)
print(loo_obs)

idata.sample_stats = log_lik.groupby("observed_county").sum()
loo_county = az.loo(idata, pointwise=True)
print(loo_county)
