# src.model.copula.py
# copyright 2024 Oreum OÜ
"""Basic models in family copula"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from oreum_core import model_pymc as mt

__all__ = ["ModelA0"]  # , "ModelA1", "ModelA2"]


class ModelA0(mt.BasePYMCModel):
    """Demo naive architecture: priors, marginal likelihoods, but no copula
    Uses:
      + nD shape handling throughout for better readability
    Core components used from oreum_core.model.BasePYMCModel
    As used by 001_MRE_ModelA0.ipynb
    """

    name = "mdla0"
    version = "1.3.0"

    def __init__(self, obsd: dict, dfx_creatord: dict, *args, **kwargs):
        """Expects `obsd` dict as {m0: obs_m0, m1: obs_m1}, each obs arranged as:
            y ~ x, aka pd.concat((dfx_en, dfx_exs), axis=1)
        Also expects `dfx_creatord` dict as {m0: dfx_creator, ...}
        """
        super().__init__(*args, **kwargs)
        self.obs_nm = kwargs.pop("obs_nm", "obs")
        self.rng = np.random.default_rng(seed=self.rsd)
        self.sample_kws.update(
            dict(target_accept=0.80)  # set higher to avoid divergences
        )

        # set obs, coords, and do data validity checks
        self.obs_m0 = obsd["m0"].copy()
        self.obs_m1 = obsd["m1"].copy()
        assert len(self.obs_m0) == len(self.obs_m1)
        self.ft_en_m0 = dfx_creatord["m0"].ft_en
        self.ft_en_m1 = dfx_creatord["m1"].ft_en
        self.factor_map_m0 = dfx_creatord["m0"].factor_map
        self.factor_map_m1 = dfx_creatord["m0"].factor_map
        # setup coords with dict expansion and additional structural names
        self.coords = dict(
            x0_nm=self.obs_m0.columns.drop(self.ft_en_m0).values,
            x1_nm=self.obs_m1.columns.drop(self.ft_en_m1).values,
            # x0_nm={k: list(d.keys()) for k, d in factor_map_m0.items()},
            # x1_nm={k: list(d.keys()) for k, d in factor_map_m1.items()},
            s_nm=["s0", "s1"],
            m_nm=["m0", "m1"],
            mhat_nm=["m0hat", "m1hat"],
        )
        self.calc_potential_loglike = False

    def _build(self):
        """Builds and returns the model. Also sets self.model"""
        self.coords.update(dict(oid=self.obs_m0.index.values))  # (i, )
        self.n = len(self.obs_m0)
        m_r = pd.concat(
            [self.obs_m0[self.ft_en_m0], self.obs_m1[self.ft_en_m1]], axis=1
        )
        y_r = m_r.product(axis=1)
        x0_r = self.obs_m0.drop(self.ft_en_m0, axis=1)
        x1_r = self.obs_m1.drop(self.ft_en_m1, axis=1)

        with pm.Model(coords=self.coords) as self.model:
            # 0. Create (Mutable)Data containers for obs (Y, X)
            m = pm.Data("m", m_r, dims=("oid", "m_nm"))
            _ = pm.Data("y", y_r, dims="oid")  # convenience for post-processing
            x0 = pm.Data("x0", x0_r, dims=("oid", "x0_nm"))
            x1 = pm.Data("x1", x1_r, dims=("oid", "x1_nm"))

            # 1. Define Observed Y marginal lognormal dists, regressing on F(X)
            b_m0 = pm.Normal("beta_m0", mu=0.0, sigma=1.0, dims="x0_nm")
            b_m1 = pm.Normal("beta_m1", mu=0.0, sigma=1.0, dims="x1_nm")
            mu = pt.stack([pt.dot(x0, b_m0.T), pt.dot(x1, b_m1.T)], axis=1)
            s = pm.InverseGamma("sigma", alpha=5.0, beta=4.0, dims="s_nm")

            # 2. Evidence observed marginals
            mhat = pm.LogNormal(
                "mhat", mu=mu, sigma=s, observed=m, dims=("oid", "mhat_nm")
            )

            # 3. Create joint product for evaluation
            # Note yhat gets into posterior but we must ignore it, since it
            # has the direct values of observed m, and is not made from mhat
            # We can get yhat from sample_posterior_predictive
            _ = pm.Deterministic("yhat", pm.math.prod(mhat, axis=1), dims="oid")

        self.rvs_marg = ["sigma", "beta_m0", "beta_m1"]
        self.rvs_ppc = ["mhat"]
        self.rvs_det = ["yhat"]

        return self.model


# class ModelA1(mt.BasePYMCModel):
#     """Demo partial architecture: priors, marginal likelihoods, copula
#     (without Jacobian adjustment)
#     Uses:
#       + Potentials to overcome ObservedRVs in likelihood
#       + nD shape handling throughout for better readability
#     Core components used from oreum_core.model.BasePYMCModel
#     As used by 101_Demo_ModelA1.ipynb.
#     """

#     version = "1.0.0"
#     name = "mdla1"

#     def __init__(self, obs_m0: pd.DataFrame, obs_m1: pd.DataFrame, *args, **kwargs):
#         """Expects 2 dfx dataframes for observations per marginal (obs_m0, obs_m1),
#         each marginal arranged as: y ~ x, aka pd.concat((dfx_en, dfx_exs), axis=1)
#         """
#         super().__init__(*args, **kwargs)
#         self.sample_kws["target_accept"] = 0.80  # set > 0.8 to tame divergences
#         self.sample_kws["tune"] = 2000  # tune 2x more than default

#         # data validity checks and set unchanging coords
#         assert len(obs_m0) == len(obs_m1)
#         self.obs_m0 = obs_m0.copy()
#         self.obs_m1 = obs_m1.copy()
#         self.coords = dict(
#             x0_nm=obs_m0.columns.drop(["m0"]).values,
#             x1_nm=obs_m1.columns.drop(["m1"]).values,
#             s_nm=["s0", "s1"],
#             y_nm=["y0", "y1"],
#             u_nm=["u0", "u1"],
#             c_nm=["c0", "c1"],
#             yhat_nm=["yhat0", "yhat1"],
#             ppc_yhat_nm=["ppc_yhat0", "ppc_yhat1"],
#         )
#         self.calc_potential_loglike = True
#         self.rvs_potential_loglike = ["pot_c", "pot_yhat"]

#     def _build(self):
#         """Builds and returns the model. Also sets self.model"""
#         self.coords_m = dict(oid=self.obs_m0.index.values)
#         self.n = len(self.obs_m0)
#         y_r = pd.concat([self.obs_m0["m0"], self.obs_m1["m1"]], axis=1)
#         x0_r = self.obs_m0.drop("m0", axis=1)
#         x1_r = self.obs_m1.drop("m1", axis=1)

#         with pm.Model(coords=self.coords, coords_mutable=self.coords_m) as self.model:
#             # 0. Create MutableData containers for obs (Y, X)
#             y = pm.MutableData("y", y_r, dims=("oid", "y_nm"))
#             x0 = pm.MutableData("x0", x0_r, dims=("oid", "x0_nm"))
#             x1 = pm.MutableData("x1", x1_r, dims=("oid", "x1_nm"))

#             # 1. Define Observed Y marginal lognormal dists, regressing on F(X)
#             m0_b = pm.Normal("m0_b", mu=0.0, sigma=1.0, dims="x0_nm")
#             m1_b = pm.Normal("m1_b", mu=0.0, sigma=1.0, dims="x1_nm")
#             m_mu = pt.stack([pt.dot(m0_b, x0.T), pt.dot(m1_b, x1.T)], axis=1)
#             m_s = pm.InverseGamma("m_s", alpha=5.0, beta=4.0, dims="s_nm")
#             m_d = pm.LogNormal.dist(mu=m_mu, sigma=m_s, shape=(self.n, 2))

#             # 2. Evidence Observed Y against marginal PDFs
#             # NOTE this lets us easily use compute_log_likelihood post sample
#             _ = pm.Potential("pot_yhat", pm.logp(m_d, y), dims=("oid", "yhat_nm"))

#             # 3. Transform Observed Y to Uniform through marginal CDFs
#             u = pm.Deterministic("u", pt.exp(pm.logcdf(m_d, y)), dims=("oid", "u_nm"))

#             # 4. Transform Uniformed Y to an Empirical Copula via MvN InvCDF
#             #    to be later evidenced against our latent MvN copula RVs
#             c = pm.Deterministic("c", mt.normal_icdf(u), dims=("oid", "c_nm"))

#             # 5. Create Latent Copula using 2D MvN w/ estimated cov = chol.dot(chol.T)
#             sd = pm.InverseGamma.dist(alpha=5.0, beta=4.0)
#             chol, corr_, stds_ = pm.LKJCholeskyCov("lkjcc", n=2, eta=2.0, sd_dist=sd)
#             c_rv = pm.MvNormal.dist(mu=pt.zeros(2), chol=chol, shape=(self.n, 2))

#             # 6. Evidence Transformed Y against Latent Copula using Potential
#             #    because TypeError: Variables that depend on other nodes cannot
#             #    be used for observed data (c)
#             _ = pm.Potential("pot_c", pm.logp(c_rv, c), dims=("oid", "c_nm"))

#             # 8. Forward estimation to still provide sample_prior_predictive and
#             #   sample_posterior_predictive overcome: UserWarning: The effect of
#             #   Potentials on other parameters is ignored during prior predictive
#             #   sampling.
#             #   Same general process as synthetic.create_copula.CopulaBuilder:
#             #   1. Transform latent MvN copula `cop_d` (w/ est. covariance `chol`)
#             #      through a standard Normal CDF to get to Uniform [0, 1]
#             #   2. Transform Uniformed marginals to Observed via their iCDFs with
#             #      their estimated parameters
#             #   NOTE:
#             #   1. Using pm.logcdf(cop_u_rv, cop_d) as a convenience, then needs pt.exp()
#             #   2. Using mt.lognormal_icdf() to avoid CLIP issues

#             normal_rv = pm.Normal.dist(mu=0.0, sigma=1.0, shape=(self.n, 2))
#             u_rv = pt.exp(pm.logcdf(normal_rv, c_rv))
#             _ = pm.Deterministic(
#                 "ppc_yhat",
#                 mt.lognormal_icdf(x=u_rv, mu=m_mu, sigma=m_s),
#                 dims=("oid", "yhat_nm"),
#             )

#         self.rvs_lkjcc = ["lkjcc", "lkjcc_stds"]
#         self.rvs_corr = ["lkjcc_corr"]
#         self.rvs_marg = ["m0_b", "m1_b", "m_s"]
#         self.rvs_unobs = self.rvs_lkjcc + self.rvs_corr + self.rvs_marg
#         self.rvs_det = ["u", "c"]
#         self.rvs_pot = ["pot_c", "pot_yhat"]
#         self.rvs_ppc = ["ppc_yhat"]

#         return self.model


# class ModelA2(mt.BasePYMCModel):
#     """Demo full architecture: priors, marginal likelihoods, copula, log_jcd
#     Uses:
#       + Potentials to overcome ObservedRVs in likelihood
#       + Jacobian adjustment for transformed observations
#       + nD shape handling throughout for better readability
#     Core components used from oreum_core.model.BasePYMCModel
#     As used by 102_Demo_ModelA2.ipynb.
#     """

#     version = "1.0.1"
#     name = "mdla2"

#     def __init__(self, obs_m0: pd.DataFrame, obs_m1: pd.DataFrame, *args, **kwargs):
#         """Expects 2 dfx dataframes for observations per marginal (obs_m0, obs_m1),
#         each marginal arranged as: y ~ x, aka pd.concat((dfx_en, dfx_exs), axis=1)
#         """
#         super().__init__(*args, **kwargs)
#         self.sample_kws["target_accept"] = 0.80  # set > 0.8 to tame divergences
#         self.sample_kws["tune"] = 2000

#         # data validity checks and set unchanging coords
#         assert len(obs_m0) == len(obs_m1)
#         self.obs_m0 = obs_m0.copy()
#         self.obs_m1 = obs_m1.copy()
#         self.coords = dict(
#             x0_nm=obs_m0.columns.drop(["m0"]).values,
#             x1_nm=obs_m1.columns.drop(["m1"]).values,
#             s_nm=["s0", "s1"],
#             y_nm=["y0", "y1"],
#             u_nm=["u0", "u1"],
#             c_nm=["c0", "c1"],
#             yhat_nm=["yhat0", "yhat1"],
#             ppc_yhat_nm=["ppc_yhat0", "ppc_yhat1"],
#         )
#         self.calc_potential_loglike = True
#         self.rvs_potential_loglike = ["pot_c", "pot_yhat"]

#     def _build(self):
#         """Builds and returns the model. Also sets self.model"""
#         self.coords_m = dict(oid=self.obs_m0.index.values)
#         self.n = len(self.obs_m0)
#         y_r = pd.concat([self.obs_m0["m0"], self.obs_m1["m1"]], axis=1)
#         x0_r = self.obs_m0.drop("m0", axis=1)
#         x1_r = self.obs_m1.drop("m1", axis=1)

#         with pm.Model(coords=self.coords, coords_mutable=self.coords_m) as self.model:
#             # 0. Create MutableData containers for obs (Y, X)
#             y = pm.MutableData("y", y_r, dims=("oid", "y_nm"))
#             x0 = pm.MutableData("x0", x0_r, dims=("oid", "x0_nm"))
#             x1 = pm.MutableData("x1", x1_r, dims=("oid", "x1_nm"))

#             # 1. Define Observed Y marginal lognormal dists, regressing on F(X)
#             m0_b = pm.Normal("m0_b", mu=0.0, sigma=1.0, dims="x0_nm")
#             m1_b = pm.Normal("m1_b", mu=0.0, sigma=1.0, dims="x1_nm")
#             m_mu = pt.stack([pt.dot(m0_b, x0.T), pt.dot(m1_b, x1.T)], axis=1)
#             m_s = pm.InverseGamma("m_s", alpha=5.0, beta=4.0, dims="s_nm")
#             m_d = pm.LogNormal.dist(mu=m_mu, sigma=m_s, shape=(self.n, 2))

#             # 2. Evidence Observed Y against marginal PDFs
#             # NOTE this lets us easily use compute_log_likelihood post sample
#             _ = pm.Potential("pot_yhat", pm.logp(m_d, y), dims=("oid", "yhat_nm"))

#             # 3. Transform Observed Y to Uniform through marginal CDFs
#             u = pm.Deterministic("u", pt.exp(pm.logcdf(m_d, y)), dims=("oid", "u_nm"))

#             # 4. Transform Uniformed Y to an Empirical Copula via MvN InvCDF
#             #    to be later evidenced against our latent MvN copula RVs
#             c = pm.Deterministic("c", mt.normal_icdf(u), dims=("oid", "c_nm"))

#             # 5. Create Latent Copula using 2D MvN w/ estimated cov = chol.dot(chol.T)
#             sd = pm.InverseGamma.dist(alpha=5.0, beta=4.0)
#             chol, corr_, stds_ = pm.LKJCholeskyCov("lkjcc", n=2, eta=2.0, sd_dist=sd)
#             c_rv = pm.MvNormal.dist(mu=pt.zeros(2), chol=chol, shape=(self.n, 2))

#             # 6. Evidence Transformed Y against Latent Copula using Potential
#             #    because TypeError: Variables that depend on other nodes cannot
#             #    be used for observed data (c)
#             _ = pm.Potential("pot_c", pm.logp(c_rv, c), dims=("oid", "c_nm"))

#             # 7. Jacobian adjustment to reduce variance in the estimated
#             #    marginals due change in volume due to transformation.
#             _ = pm.Potential(
#                 "pot_jcd_c",
#                 mt.get_log_jcd_scan(c, y, upstream_rvs=[m_mu, m_s]),
#                 dims="oid",
#             )

#             # 8. Forward estimation to still provide sample_prior_predictive and
#             #   sample_posterior_predictive overcome: UserWarning: The effect of
#             #   Potentials on other parameters is ignored during prior predictive
#             #   sampling.
#             #   Same general process as synthetic.create_copula.CopulaBuilder:
#             #   1. Transform latent MvN copula `cop_d` (w/ est. covariance `chol`)
#             #      through a standard Normal CDF to get to Uniform [0, 1]
#             #   2. Transform Uniformed marginals to Observed via their iCDFs with
#             #      their estimated parameters
#             #   NOTE:
#             #   1. Using pm.logcdf(cop_u_rv, cop_d) as a convenience, then needs pt.exp()
#             #   2. Using mt.lognormal_icdf() to avoid CLIP issues

#             normal_rv = pm.Normal.dist(mu=0.0, sigma=1.0, shape=(self.n, 2))
#             u_rv = pt.exp(pm.logcdf(normal_rv, c_rv))
#             _ = pm.Deterministic(
#                 "ppc_yhat",
#                 mt.lognormal_icdf(x=u_rv, mu=m_mu, sigma=m_s),
#                 dims=("oid", "yhat_nm"),
#             )

#         self.rvs_lkjcc = ["lkjcc", "lkjcc_stds"]
#         self.rvs_corr = ["lkjcc_corr"]
#         self.rvs_marg = ["m0_b", "m1_b", "m_s"]
#         self.rvs_unobs = self.rvs_lkjcc + self.rvs_corr + self.rvs_marg
#         self.rvs_det = ["u", "c"]
#         self.rvs_pot = ["pot_c", "pot_yhat", "pot_jcd_c"]
#         self.rvs_ppc = ["ppc_yhat"]

#         return self.model
