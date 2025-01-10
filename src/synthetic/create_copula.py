# src.synthetic.create_copula.py
"""Create synthetic copula dataset"""
# copyright 2023 Oreum Industries

import numpy as np
import pandas as pd
from scipy import stats


class CopulaBuilder:
    """Create synthetic copula dataset using a "forward-pass":
    1. Start at copula (`c0, c1`) ->
    2. Transform to uniform (`u0, u1`) ->
    3. Transform to marginals (`m0, m1`)
    4. Also for comparison, create marginals (`m0x`, `m1x`) without copula
    5. Create simple index for oid
    NOTE:
        + Currently copula is MvN
        + Currently marginals are both LogNormal
        + We store the parameter values for the marginals
        + We will attempt to recover these later using a copula model
    """

    version = "0.3.0"
    rsd = 42
    rng = np.random.default_rng(seed=rsd)

    def __init__(self, c_r=-0.7):
        """Set some defaults"""
        self.ref_vals = dict(
            c_r=c_r,
            c_cov=np.array([[1.0, c_r], [c_r, 1.0]]),
            m0_kind="lognorm",
            m1_kind="lognorm",
        )
        self.c_dist = stats.multivariate_normal(
            mean=np.zeros(2), cov=self.ref_vals["c_cov"], seed=self.rng
        )

        self.cx_dist = stats.norm(
            loc=np.zeros(2),
            scale=np.ones(2),  # seed=self.rng
        )

    def create(
        self, nobs: int = 200, m0_params: dict = None, m1_params: dict = None
    ) -> tuple[pd.DataFrame, tuple]:
        """Create observed marginals using a 'forward-pass'
        NOTE: Only MvN Copula, LogNormal marginals currently supported
        Pass lognormal mu which will be used as scale = np.exp(mu)
        """
        if m0_params is None:
            m0_params = {"mu": 0.2, "sigma": 0.5}
        if m1_params is None:
            m1_params = {"mu": 2.0, "sigma": 1.0}

        self.ref_vals.update({"m0_params": m0_params, "m1_params": m1_params})

        # 1. Generate latent copula from MvN w/ known covariance
        #  + Lazily choose to set off-diagonal covariance to 1:
        #    i.e. use a correlation matrix, not a covariance matrix
        #  + We will still use a proper covariance matrix in the model estimation,
        #    but the off-diagonal fits should get very close to 1
        #  + Set a high correlation for ease of viewing
        df = pd.DataFrame(self.c_dist.rvs(nobs), columns=["c0", "c1"])

        # 2. Transform copula marginals to Uniform [0, 1] pass through Normal CDF
        df = pd.concat(
            [df, pd.DataFrame(stats.norm.cdf(df[["c0", "c1"]]), columns=["u0", "u1"])],
            axis=1,
        )

        # 3. Transform Uniformed marginals to Observed marginals pass through
        #    their Inverse CDFs aka Percent Point Function (PPF) aka Quantile Function
        self.m0_dist = stats.lognorm(
            scale=np.exp(m0_params["mu"]), s=m0_params["sigma"]
        )
        self.m1_dist = stats.lognorm(
            scale=np.exp(m1_params["mu"]), s=m1_params["sigma"]
        )
        df["m0"] = self.m0_dist.ppf(df["u0"])
        df["m1"] = self.m1_dist.ppf(df["u1"])

        # 4. Also create uncorrelated obs using the same workflow (no copula)
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    self.cx_dist.rvs(size=(nobs, 2), random_state=42),
                    columns=["c0x", "c1x"],
                ),
            ],
            axis=1,
        )
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    stats.norm.cdf(df[["c0x", "c1x"]]), columns=["u0x", "u1x"]
                ),
            ],
            axis=1,
        )
        df["m0x"] = self.m0_dist.ppf(df["u0x"])
        df["m1x"] = self.m1_dist.ppf(df["u1x"])

        # 5. Create index oid
        df["oid"] = [f"i{str(i).zfill(3)}" for i in range(len(df))]
        df.set_index("oid", inplace=True)

        return df
