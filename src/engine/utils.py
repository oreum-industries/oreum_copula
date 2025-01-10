# src.engine.utils.py
# copyright 2024 Oreum Industries
"""Assorted display and plotting utilities for the project"""

import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from oreum_core import curate
from oreum_core import model_pymc as mt

__all__ = ["ProjectDFXCreator", "display_rvs"]

sns.set_theme(
    style="darkgrid",
    palette="muted",
    context="notebook",
    rc={"figure.dpi": 72, "savefig.dpi": 144, "figure.figsize": (12, 4)},
)


class ProjectDFXCreator:
    """Convenience class to process the main dataset that's used
    throughout this project into `dfx` for model input. Saves us having to
    redeclare it in each notebook.
    """

    def __init__(self, kind="m0"):
        # NOTE: The death event is an endog feature, but is recorded in our data
        # as a bool so for convenience we will patsy transform the death event
        # as if it were part of the linear model, then remove it from fts_ex and
        # put into fts_en, and rearrange dfx accordingly also
        self.fml_tfmr = "1"
        self.dfcmb = None
        self.dfscale = None
        self.factor_map = None
        if kind == "m0":
            self.ft_en = ["m0"]
        else:  ## m1
            self.ft_en = ["m1"]

    def get_dfx(self, df: pd.DataFrame, in_sample: bool = True) -> pd.DataFrame:
        """Reshape, transform & standardize df to create dfx for model input"""

        # 1. Create dfcmb (reshaped) from df
        if in_sample:
            reshaper = curate.DatasetReshaper()
            self.dfcmb = reshaper.create_dfcmb(df)
        else:
            assert self.dfcmb is not None, "run in_sample = True first"

        # 2. create Transformer based on dfcmb
        tfmr = curate.Transformer()
        _ = tfmr.fit_transform(self.fml_tfmr, self.dfcmb, propagate_nans=True)
        self.factor_map = tfmr.factor_map

        # 3. Transform df according to dfcmb
        df_ex = tfmr.transform(df, propagate_nans=True)

        # 4. Standardize
        stdr = curate.Standardizer(tfmr)
        if in_sample:
            df_exs = stdr.fit_standardize(df_ex)
            self.dfscale, _ = stdr.get_scale()  # unused in this process, because no dfb
        else:
            stdr.set_scale(self.dfscale)
            df_exs = stdr.standardize(df_ex)

        df_en = df[self.ft_en].copy()
        kws_mrg = dict(how="inner", left_index=True, right_index=True)
        dfx = pd.merge(df_en, df_exs, **kws_mrg).rename(
            columns=lambda x: tfmr.snl.clean_patsy(x)
        )

        fts_ex = [v for v in dfx.columns.values if v not in self.ft_en]
        dfx = dfx.reindex(columns=self.ft_en + fts_ex)

        return dfx


def display_rvs(mdl: mt.BasePYMCModel):
    """Convenience to display RVS and values"""
    _ = display(f"RVs for {mdl.mdl_id}")
    if mdl.model is not None:
        _ = [display(Markdown(s)) for s in mt.print_rvs(mdl)]
