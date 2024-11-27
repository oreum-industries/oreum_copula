# src.engine.plots.py
"""Common plots used during model evaluation"""

import pandas as pd
import seaborn as sns
from matplotlib import figure

from . import app_logger

__all__ = ["plot_points"]


sns.set(
    style="darkgrid",
    palette="muted",
    context="notebook",
    rc={"savefig.dpi": 300, "figure.figsize": (12, 6)},
)

log = app_logger.get_logger(__name__)


def plot_points(
    dfm: pd.DataFrame, x: str, y: str, nobs: int, **kwargs
) -> figure.Figure:
    """Convenience pointplot for ppc predictions"""

    gd = sns.catplot(
        x="elosshat",
        y="pol_id",
        data=dfm,
        kind="point",
        join=False,
        height=0.2 * nobs,
        aspect=2,
    )

    txtadd = kwargs.pop("txtadd", None)
    _ = gd.fig.suptitle(txtadd, y=1, fontsize=14)
    _ = gd.fig.tight_layout(pad=0.95)
    return gd.fig
