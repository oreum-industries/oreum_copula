# src.model.copula.py
# copyright 2024 Oreum Industries
"""Basic models in Copula family"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from oreum_core import model_pymc as mt

__all__ = ["ModelA0", "ModelA1", "ModelA2"]


class ModelA0(mt.BasePYMCModel):
    """Basic naive architecture: priors, marginal likelihoods, but no copula
    Core components used from oreum_core.model.BasePYMCModel
    As used by 100_ModelA0.ipynb
    """

    name = "mdla0"
    version = "1.3.0"

    def __init__(
        self,
        obs_m0: pd.DataFrame,
        obs_m1: pd.DataFrame,
        dfx_creatord: dict,
        *args,
        **kwargs,
    ):
        """Expects 2 dataframes for obs, each arranged as:
            y ~ x, aka pd.concat((dfx_en, dfx_exs), axis=1)
        Also expects `dfx_creatord` dict as {m0: dfx_creator, ...}
        """
        super().__init__(*args, **kwargs)
        self.obs_nm = kwargs.pop("obs_nm", "obs")
        self.rng = np.random.default_rng(seed=self.rsd)
        self.sample_kws.update(dict(target_accept=0.80))

        # set obs, coords, and do data validity checks
        self.obs_m0 = obs_m0.copy()
        self.obs_m1 = obs_m1.copy()
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

            # 1. Define latent params for observed Marginals, regressing on F(X)
            b_m0 = pm.Normal("beta_m0", mu=0.0, sigma=1.0, dims="x0_nm")
            b_m1 = pm.Normal("beta_m1", mu=0.0, sigma=1.0, dims="x1_nm")
            mu = pt.stack([pt.dot(x0, b_m0.T), pt.dot(x1, b_m1.T)], axis=1)
            s = pm.InverseGamma("sigma", alpha=5.0, beta=4.0, dims="s_nm")

            # 2. Define marginals and Evidence against observed
            mhat = pm.LogNormal(
                "mhat", mu=mu, sigma=s, observed=m, dims=("oid", "mhat_nm")
            )

            # 3. Post-process for forward / PPC estimation: create joint marginal
            # product yhat.
            # Sidenote pymc also puts yhat into posterior but we must ignore it,
            # since it has the direct values of observed m, and is not made from
            # mhat. We can get yhat from sample_posterior_predictive
            _ = pm.Deterministic("yhat", pm.math.prod(mhat, axis=1), dims="oid")

        self.rvs_marg = ["sigma", "beta_m0", "beta_m1"]
        self.rvs_ppc = ["mhat"]
        self.rvs_det = ["yhat"]

        return self.model


class ModelA1(mt.BasePYMCModel):
    """Partial architecture (extends ModelA0)
    New:
      + Gaussian copula (but without Jacobian adjustment)
      + Potentials to overcome ObservedRVs in likelihood
    Core components used from oreum_core.model.BasePYMCModel
    As used by 101_ModelA1.ipynb
    """

    name = "mdla1"
    version = "1.1.0"

    def __init__(
        self,
        obs_m0: pd.DataFrame,
        obs_m1: pd.DataFrame,
        dfx_creatord: dict,
        *args,
        **kwargs,
    ):
        """Expects 2 dataframes for obs, each arranged as:
            y ~ x, aka pd.concat((dfx_en, dfx_exs), axis=1)
        Also expects `dfx_creatord` dict as {m0: dfx_creator, ...}
        """
        super().__init__(*args, **kwargs)
        self.obs_nm = kwargs.pop("obs_nm", "obs")
        self.rng = np.random.default_rng(seed=self.rsd)
        self.sample_kws.update(dict(target_accept=0.80))

        # set obs, coords, and do data validity checks
        self.obs_m0 = obs_m0.copy()
        self.obs_m1 = obs_m1.copy()
        assert len(self.obs_m0) == len(self.obs_m1)
        self.ft_en_m0 = dfx_creatord["m0"].ft_en
        self.ft_en_m1 = dfx_creatord["m1"].ft_en
        self.factor_map_m0 = dfx_creatord["m0"].factor_map
        self.factor_map_m1 = dfx_creatord["m0"].factor_map
        # setup coords with dict expansion and additional structural names
        self.coords = dict(
            x0_nm=self.obs_m0.columns.drop(self.ft_en_m0).values,
            x1_nm=self.obs_m1.columns.drop(self.ft_en_m1).values,
            s_nm=["s0", "s1"],
            m_nm=["m0", "m1"],
            mhat_nm=["m0hat", "m1hat"],
            u_nm=["u0", "u1"],
            c_nm=["c0", "c1"],
            chat_nm=["c0hat", "c1hat"],
        )
        self.calc_loglike_of_potential = True
        self.rvs_potential_loglike = ["pot_chat", "pot_mhat"]

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

            # 1. Define latent params for observed Marginals, regressing on F(X)
            b_m0 = pm.Normal("beta_m0", mu=0.0, sigma=1.0, dims="x0_nm")
            b_m1 = pm.Normal("beta_m1", mu=0.0, sigma=1.0, dims="x1_nm")
            mu = pt.stack([pt.dot(x0, b_m0.T), pt.dot(x1, b_m1.T)], axis=1)
            s = pm.InverseGamma("sigma", alpha=5.0, beta=4.0, dims="s_nm")

            # 2. Define marginals and Evidence against observed
            # Here we need to establish a LogNormal dist that we subsequently
            # use to transform the observations. So we have to "evidence" the
            # created dist object using a Potential (and minimise logp).
            # Requires calc_loglike_of_potential = True
            m_d = pm.LogNormal.dist(mu=mu, sigma=s, shape=(self.n, 2))
            _ = pm.Potential("pot_mhat", pm.logp(m_d, m), dims=("oid", "mhat_nm"))

            # 3. Transformation path pt1: Observed -> Uniform via Marginal CDF
            u = pm.Deterministic("u", pt.exp(pm.logcdf(m_d, m)), dims=("oid", "u_nm"))

            # 4. Transformation path pt2: Uniform -> Normal via Normal InvCDF
            c = pm.Deterministic("c", mt.normal_icdf(u), dims=("oid", "c_nm"))
            # y_cop_u_rv = pm.Normal.dist(mu=0., sigma=1.)
            # c = pm.Deterministic("c", pm.icdf(y_cop_u_rv, u), dims=("oid", "c_nm"))

            # 5. Create Latent Copula dist using a 2D MvNormal
            # with estimated cov = chol.dot(chol.T)
            sd = pm.InverseGamma.dist(alpha=5.0, beta=4.0)
            chol, corr_, sdev_ = pm.LKJCholeskyCov("lkjcc", n=2, eta=2, sd_dist=sd)
            c_d = pm.MvNormal.dist(mu=pt.zeros(2), chol=chol, shape=(self.n, 2))

            # 6. Evidence transformed C against Latent Copula using Potential
            #  because pymc TypeError: Variables that depend on other nodes
            #  cannot be used for observed data (c)
            # NOTE: MVNormal drops the last dim, becoming 1D !
            # see explanations in https://github.com/pymc-devs/pymc/issues/7602
            _ = pm.Potential("pot_chat", pm.logp(c_d, c, warn_rvs=True), dims="oid")

            # 7. Post-process for forward / PPC estimation: create marginals mhat
            # and joint product yhat.
            # This allows `sample_prior_predictive`, `sample_posterior_predictive`,
            # and overcomes issue: "UserWarning: The effect of Potentials on
            # other parameters is ignored during prior predictive sampling."
            # We follow the forward data-generating process as
            # `synthetic.create_copula.CopulaBuilder`:
            #  1. Transform latent MvN copula `c_d` (w/ est. covariance `chol`)
            #    through a standard Normal CDF to get to Uniform [0, 1]
            #  2. Transform Uniformed marginals to Observed via their iCDFs with
            #    their estimated parameters
            #  NOTE:
            #  1. Using pm.logcdf(u_d, c_d) for convenience, then needs pt.exp()
            #  2. Using mt.lognormal_icdf() to avoid CLIP issues
            normal_d = pm.Normal.dist(mu=0.0, sigma=1.0, shape=(self.n, 2))
            u_d = pt.exp(pm.logcdf(normal_d, c_d))
            mhat = pm.Deterministic(
                "mhat",
                mt.lognormal_icdf(x=u_d, mu=mu, sigma=s),
                dims=("oid", "mhat_nm"),
            )

            # Create joint marginal product yhat.
            # Sidenote pymc also puts yhat into posterior but we must ignore it,
            # since it has the direct values of observed m, and is not made from
            # mhat. We can get yhat from sample_posterior_predictive
            _ = pm.Deterministic("yhat", pm.math.prod(mhat, axis=1), dims="oid")

        self.rvs_marg = ["sigma", "beta_m0", "beta_m1"]
        self.rvs_lkjcc = ["lkjcc"]
        self.rvs_unobs = self.rvs_marg + self.rvs_lkjcc
        self.rvs_pot = ["pot_chat", "pot_mhat"]
        self.rvs_det = ["u", "c", "lkjcc_stds", "lkjcc_corr", "yhat"]
        self.rvs_ppc = ["mhat"]

        return self.model


class ModelA2(mt.BasePYMCModel):
    """Full architecture (extends ModelA1)
    New:
      + Jacobian Adjustment on transformed observations (see step #7)
    Core components used from oreum_core.model.BasePYMCModel
    As used by 102_ModelA2.ipynb.
    """

    name = "mdla2"
    version = "1.2.0"

    def __init__(
        self,
        obs_m0: pd.DataFrame,
        obs_m1: pd.DataFrame,
        dfx_creatord: dict,
        *args,
        **kwargs,
    ):
        """Expects 2 dataframes for obs, each arranged as:
            y ~ x, aka pd.concat((dfx_en, dfx_exs), axis=1)
        Also expects `dfx_creatord` dict as {m0: dfx_creator, ...}
        """
        super().__init__(*args, **kwargs)
        self.obs_nm = kwargs.pop("obs_nm", "obs")
        self.rng = np.random.default_rng(seed=self.rsd)
        self.sample_kws.update(dict(target_accept=0.80))

        # set obs, coords, and do data validity checks
        assert len(obs_m0) == len(obs_m1)
        self.obs_m0 = obs_m0.copy()
        self.obs_m1 = obs_m1.copy()
        self.ft_en_m0 = dfx_creatord["m0"].ft_en
        self.ft_en_m1 = dfx_creatord["m1"].ft_en
        self.factor_map_m0 = dfx_creatord["m0"].factor_map
        self.factor_map_m1 = dfx_creatord["m0"].factor_map
        # setup coords with dict expansion and additional structural names
        self.coords = dict(
            x0_nm=self.obs_m0.columns.drop(self.ft_en_m0).values,
            x1_nm=self.obs_m1.columns.drop(self.ft_en_m1).values,
            s_nm=["s0", "s1"],
            m_nm=["m0", "m1"],
            mhat_nm=["m0hat", "m1hat"],
            u_nm=["u0", "u1"],
            c_nm=["c0", "c1"],
            chat_nm=["c0hat", "c1hat"],
        )
        self.calc_loglike_of_potential = True
        self.rvs_potential_loglike = ["pot_chat", "pot_mhat"]

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

            # 1. Define latent params for observed Marginals, regressing on F(X)
            b_m0 = pm.Normal("beta_m0", mu=0.0, sigma=1.0, dims="x0_nm")
            b_m1 = pm.Normal("beta_m1", mu=0.0, sigma=1.0, dims="x1_nm")
            mu = pt.stack([pt.dot(x0, b_m0.T), pt.dot(x1, b_m1.T)], axis=1)
            s = pm.InverseGamma("sigma", alpha=5.0, beta=4.0, dims="s_nm")

            # 2. Define marginals and Evidence against observed
            # Here we need to establish a LogNormal dist that we subsequently
            # use to transform the observations. So we have to "evidence" the
            # created dist object using a Potential (and minimise logp).
            # Requires calc_loglike_of_potential = True
            m_d = pm.LogNormal.dist(mu=mu, sigma=s, shape=(self.n, 2))
            _ = pm.Potential("pot_mhat", pm.logp(m_d, m), dims=("oid", "mhat_nm"))

            # 3. Transformation path pt1: Observed -> Uniform via Marginal CDF
            u = pm.Deterministic("u", pt.exp(pm.logcdf(m_d, m)), dims=("oid", "u_nm"))

            # 4. Transformation path pt2: Uniform -> Normal via Normal InvCDF
            c = pm.Deterministic("c", mt.normal_icdf(u), dims=("oid", "c_nm"))
            # y_cop_u_rv = pm.Normal.dist(mu=0., sigma=1.)
            # c = pm.Deterministic("c", pm.icdf(y_cop_u_rv, u), dims=("oid", "c_nm"))

            # 5. Create Latent Copula dist using a 2D MvNormal
            # with estimated cov = chol.dot(chol.T)
            sd = pm.InverseGamma.dist(alpha=5.0, beta=4.0)
            chol, corr_, sdev_ = pm.LKJCholeskyCov("lkjcc", n=2, eta=2, sd_dist=sd)
            c_d = pm.MvNormal.dist(mu=pt.zeros(2), chol=chol, shape=(self.n, 2))

            # 6. Evidence transformed C against Latent Copula using Potential
            #  because pymc TypeError: Variables that depend on other nodes
            #  cannot be used for observed data (c)
            # NOTE: MVNormal drops the last dim, becoming 1D !
            # see explanations in https://github.com/pymc-devs/pymc/issues/7602
            _ = pm.Potential("pot_chat", pm.logp(c_d, c, warn_rvs=True), dims="oid")

            # 7. Jacobian adjustment to reduce variance in C due to change in
            #  volume due to transformation `M -> U -> C`
            _ = pm.Potential(
                "pot_jcd_c", mt.get_log_jcd_scan(c, m, upstream_rvs=[mu, s]), dims="oid"
            )

            # 8. Post-process for forward / PPC estimation: create marginals mhat
            # and joint product mhat.
            # This allows `sample_prior_predictive`, `sample_posterior_predictive`,
            # and overcomes issue: "UserWarning: The effect of Potentials on
            # other parameters is ignored during prior predictive sampling."
            # We follow the forward data-generating process as
            # `synthetic.create_copula.CopulaBuilder`:
            #  1. Transform latent MvN copula `c_d` (w/ est. covariance `chol`)
            #    through a standard Normal CDF to get to Uniform [0, 1]
            #  2. Transform Uniformed marginals to Observed via their iCDFs with
            #    their estimated parameters
            #  NOTE:
            #  1. Using pm.logcdf(u_d, c_d) for convenience, then needs pt.exp()
            #  2. Using mt.lognormal_icdf() to avoid CLIP issues
            normal_d = pm.Normal.dist(mu=0.0, sigma=1.0, shape=(self.n, 2))
            u_d = pt.exp(pm.logcdf(normal_d, c_d))
            mhat = pm.Deterministic(
                "mhat",
                mt.lognormal_icdf(x=u_d, mu=mu, sigma=s),
                dims=("oid", "mhat_nm"),
            )
            _ = pm.Deterministic("yhat", pm.math.prod(mhat, axis=1), dims="oid")

        self.rvs_marg = ["sigma", "beta_m0", "beta_m1"]
        self.rvs_lkjcc = ["lkjcc"]
        self.rvs_unobs = self.rvs_marg + self.rvs_lkjcc
        self.rvs_pot = ["pot_chat", "pot_mhat", "pot_jcd_c"]
        self.rvs_det = ["u", "c", "yhat", "lkjcc_stds", "lkjcc_corr"]
        self.rvs_ppc = ["mhat"]

        return self.model
